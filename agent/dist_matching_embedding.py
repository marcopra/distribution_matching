from collections import OrderedDict
import os
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from dm_env import StepType, specs

import utils
from distribution_matching import DistributionVisualizer


class Encoder(nn.Module):
    def __init__(self, obs_shape, hidden_dim, feature_dim):
        super(Encoder, self).__init__()
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim

        self.fc = nn.Sequential(
            nn.Linear(obs_shape[0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs.view(obs.shape[0], -1)
        h = self.fc(obs)
        return h

class DistributionMatcher:
    """
    Policy optimizer for matching target state distributions using mirror descent.
    
    Args:
        env: gym.Env instance
        gamma: Discount factor for occupancy measure
        eta: Learning rate for mirror descent
        gradient_type: Type of KL gradient ('reverse' or 'forward')
    """
    
    def __init__(self, 
                 n_states: int,
                 n_actions: int, 
                 nu0: np.ndarray,
                 nu_target: np.ndarray,
                 batch_size: int,
                 gamma: float = 0.9, 
                 eta: float = 0.1, 
                 alpha = 0.1, 
                 gradient_type: str = 'reverse',
                 device: str = "cpu"):
        self.gamma = gamma
        self.eta = eta
        self.nu0 = nu0
        self.nu_target = nu_target
        self.gradient_type = gradient_type
        self.alpha = alpha
        self.uniform_policy_operator = np.ones((n_states * n_actions, n_states)) / n_actions
        self.policy_operator = self.uniform_policy_operator.copy()
        
        self.n_states = n_states
        self.n_actions = n_actions

        self.lava=False
        self.kl_history = []

        # sample states from nu0
        obs_from_nu0 = []
        for _ in range(batch_size):
            state = np.random.choice(self.n_states, p=nu_target.flatten())
            obs_from_nu0.append(state)
        obs_from_nu0 = np.array(obs_from_nu0).reshape(-1, 1)  # [batch_size, obs_dim]
        self.obs_from_nu0 = torch.tensor(obs_from_nu0).float().to(device)

    
    
    @staticmethod
    def M_pi_operator(P: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Compute M_π = T ∘ Π_π operator.
        Maps state distribution to next state distribution under policy.
        """
        return T @ P

    def append_loss_history(self, loss_value: float):
        self.kl_history.append(loss_value) # TODO is a loss history, it is needed just for visualizer that it is in another file.

    def compute_discounted_occupancy(self, nu0: np.ndarray, P: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Compute discounted occupancy measure: ν_π = (1-γ)(I - γM_π)^{-1} ν_0
        
        Args:
            nu0: Initial state distribution
            P: Policy operator (uses self.policy_operator if None)
        
        Returns:
            Discounted occupancy measure
        """
        # T = T.weight.detach().cpu().numpy()  # [n_states, n_states * n_actions] 
        I = np.eye(self.n_states)
        M = self.M_pi_operator(P, T)
        return (1 - self.gamma) * np.linalg.solve(I - self.gamma * M, nu0)

    def kl_divergence(self, p: np.ndarray, q: np.ndarray, P: np.ndarray) -> float:
        """
        Compute KL(p||q) with numerical stability.
        Args:
            p: Target distribution
            q: Current distribution
            P: Current policy operator
        Returns:
            KL divergence value
        """

        if self.alpha > 0:
            second_term = -np.sum(P * np.nan_to_num(np.log(P/self.uniform_policy_operator), nan=0.0))
        else:
            second_term = 0.0

        return (1-self.alpha) * np.sum(np.nan_to_num(p * np.log(p / q), nan=0.0)) + self.alpha*second_term

    def compute_gradient(self, nu0: np.ndarray, nu_target: np.ndarray, P: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Compute gradient of KL divergence w.r.t. policy.
        
        Args:
            nu0: Initial state distribution
            nu_target: Target state distribution
        
        Returns:
            Gradient of shape (n_states * n_actions, n_states)
        """
        raise NotImplementedError("This method is deprecated, use compute_new_gradient instead.")   
        I = np.eye(self.n_states)
        M = self.M_pi_operator(P, T)
        second_term = np.nan_to_num(1 + P/self.uniform_policy_operator, nan=0.0) if self.alpha > 0 else 0.0  
        
        if self.gradient_type == 'forward':
            r = nu0 / ((I - self.gamma * M) @ nu_target)
            gradient =(1-self.alpha) * (self.gamma * T.T @ r @ nu_target.T)/(1-self.gamma)- self.alpha * second_term
            
        elif self.gradient_type == 'reverse':
            nu_pi_over_nu_target = np.clip(
                (1-self.gamma)*np.linalg.solve((I - self.gamma * M), nu0) / nu_target, 
                a_min=1e-10, a_max=None
            )
            log_nu_pi_over_nu_target = np.log(nu_pi_over_nu_target)
            
            gradient = self.gamma * np.linalg.solve(I - self.gamma * M, T).T
            gradient = gradient @ (np.ones_like(log_nu_pi_over_nu_target) + log_nu_pi_over_nu_target)
            
            gradient = (1-self.alpha) *(1 - self.gamma) * gradient @ np.linalg.solve(I - self.gamma * M, nu0).T - self.alpha * second_term
        elif self.gradient_type == 'MMD':
            assert self.alpha == 0.0, "MMD gradient not compatible with policy regularization"
            gradient = 2*self.gamma * (1- self.gamma)* np.linalg.solve(I - self.gamma * M, self.T_operator).T @ ((1 - self.gamma) * np.linalg.solve(I - self.gamma * M, nu0) - nu_target)@np.ones_like(nu0.T) 
        else:
            raise ValueError(f"Unknown gradient type: {self.gradient_type}")
        
        return gradient

    def compute_new_gradient(self, P: np.ndarray, T: np.ndarray, nu0: np.ndarray) -> np.ndarray:
        """
        Compute gradient of KL divergence w.r.t. policy.
        
        Args:
            nu0: Initial state distribution
            nu_target: Target state distribution
        
        Returns:
            Gradient of shape (n_states * n_actions, n_states)
        """

        M = self.M_pi_operator(P, T)
        
        I = np.eye(self.n_states)
        gradient = 2*self.gamma * (1- self.gamma)* np.linalg.solve(I - self.gamma * M, T).T @ ((1 - self.gamma) * np.linalg.solve(I - self.gamma * M, nu0))@np.ones_like(nu0.T) 
        
        return gradient

    
    def mirror_descent_update(self, P: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Apply mirror descent update: π_{t+1}(a|s) ∝ π_t(a|s) * exp(-η ∇_{a,s} f)
        
        Args:
            P: Current policy operator
            gradient: Policy gradient
        
        Returns:
            Updated policy operator
        """
        policy_3d = P.reshape((self.n_states, self.n_actions, self.n_states))
        new_policy_3d = np.zeros_like(policy_3d)
        gradient_3d = gradient.reshape((self.n_states, self.n_actions, self.n_states))
        
        
        # Normalize the block-diagonal elements (policy probabilities per state)
        for s in range(self.n_states):
            new_policy_3d[s, :, s] = policy_3d[s, :, s] * np.exp(-self.eta * gradient_3d[s, :, s])
            policy_s_actions = new_policy_3d[s, :, s]
            new_policy_3d[s, :, s] = policy_s_actions / (policy_s_actions.sum() + 1e-10)
        
        return new_policy_3d.reshape((self.n_states * self.n_actions, self.n_states))

    # TODO: remove this in the future, are now useful for debugging and visualizing plot in an already existing visualizer class
    def update_internal_policy(self, new_policy_operator: np.ndarray):
        """
        Update internal policy operator.
        
        Args:
            new_policy_operator: New policy operator to set
        """
        self.policy_operator = new_policy_operator

    def get_policy_per_state(self, uniform_policy: bool = False) -> np.ndarray:
        """
        Extract policy probabilities π(a|s) for each state.
        Args:
            uniform_policy: Whether to return uniform random policy instead of learned policy
        
        Returns:
            Array of shape (n_states, n_actions) with policy probabilities
        """
        if uniform_policy:
            policy_matrix = self.uniform_policy_operator.reshape((self.n_states, self.n_actions, self.n_states))
        else:
            policy_matrix = self.policy_operator.reshape((self.n_states, self.n_actions, self.n_states))
        policy_per_state = np.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            policy_per_state[s, :] = policy_matrix[s, :, s]

        return policy_per_state

class DistMatchingEmbeddingAgent:
    def __init__(self,
                 name,
                 obs_type,
                 obs_shape,
                 action_shape,
                 lr_actor, #eta
                 discount, # gamma
                 alpha, # second term regularization in the loss
                 batch_size,
                 nstep,
                 use_tb,
                 use_wandb,
                 lr_T,
                 lr_encoder,
                 update_every_steps,
                 num_expl_steps,
                 T_learning_steps,
                 gradient_type,
                 internal_dataset_type: str = "unique", 
                 starting_policy: str = "uniform",
                 device: str = "cpu",
                 linear_actor: bool = False, # TODO not used yet
                 ):


        self.n_states = obs_shape[0]
        self.n_actions = action_shape[0]
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        self.action_dim = action_shape[0]
        self.lr_actor = lr_actor
        self.discount = discount
        self.alpha = alpha
        self.lr_T = lr_T
        self.T_learning_steps = T_learning_steps
        self.batch_size = batch_size
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.device = device
        self.gradient_type = gradient_type  # could be 'forward' but it is not working practically (it is a theorethical problem not a bug)
        self.internal_dataset_type = internal_dataset_type
        if self.gradient_type != 'MMD':
            raise ValueError(f"Only MMD gradient is supported currently, got: {self.gradient_type}")
        self.num_expl_steps = num_expl_steps

        self.initial_distribution = None
        self.T_operator = None
        self.internal_dataset = None

        if starting_policy == "uniform":
            self.policy_operator = self._create_uniform_policy()
        elif starting_policy == "random":
            self.policy_operator = self._create_random_policy()
        else:
            raise ValueError(f"Unknown starting policy: {starting_policy}")
        
        self.encoder = Encoder((1,), hidden_dim=128, feature_dim=self.n_states).to(self.device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr_encoder)

        self.nu_target = (np.ones(self.n_states) / self.n_states).reshape(-1,1) # Homogeneus target distribution


        self.policy_matrix = self._from_operator_to_policy(self.policy_operator)
        self.training = False

        
    
    def train(self, training=True):
        self.training = training
    
    def insert_env(self, env):
        self.env = env.unwrapped

    def init_meta(self):
        # TODO future reminder: Remove this when using non-custom envs. we access some env methods that are not available in usual gym envs
        if self.internal_dataset is None:
          
            self.T_operator = self._create_transition_matrix()
            
            self.distribution_matcher = DistributionMatcher(
                n_states=self.n_states,
                n_actions=self.n_actions,
                nu0=self.initial_distribution,
                nu_target=self.nu_target,
                batch_size=self.batch_size,
                gamma=self.discount,
                eta=self.lr_actor,
                alpha=self.alpha,
                gradient_type=self.gradient_type,
                device=self.device,
            )
            self.visualizer = DistributionVisualizer(self.env, self.distribution_matcher)

            # Initialize as empty tensors
            self.internal_dataset = {
                'observation': torch.empty((0, ), device=self.device), # save just the int
                'action': torch.empty((0,), dtype=torch.long, device=self.device), # save just the int
                'next_observation': torch.empty((0, ), device=self.device), # save just the int
                'alpha': torch.empty((0, ), device=self.device) # save just the int
            }
            self._unique_obs_action = set()
        return OrderedDict()

    def get_meta_specs(self):
        return tuple()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        
        # Store transition in internal dataset
        if self.internal_dataset_type == "unique":
            # ** Store only unique states-action pairs ** # OK obnly for discrete envs
            if time_step.step_type == StepType.FIRST:
                self._prev_obs = time_step.observation
            elif time_step.step_type == StepType.MID:
                obs_action_pair = (np.argmax(self._prev_obs), int(time_step.action))
                if obs_action_pair not in self._unique_obs_action:
                    self._unique_obs_action.add(obs_action_pair)
                    # Convert to tensors and concatenate
                    obs_idx = np.argmax(self._prev_obs)
                    self.internal_dataset['observation'] = torch.cat([
                        self.internal_dataset['observation'],
                        torch.tensor([obs_idx], device=self.device).unsqueeze(0)
                    ], dim=0)
                    self.internal_dataset['action'] = torch.cat([
                        self.internal_dataset['action'],
                        torch.tensor([time_step.action], dtype=torch.long, device=self.device)
                    ], dim=0)
                    next_obs_idx = np.argmax(time_step.observation)
                    self.internal_dataset['next_observation'] = torch.cat([
                        self.internal_dataset['next_observation'],
                        torch.tensor([next_obs_idx], device=self.device).unsqueeze(0)
                    ], dim=0)
                    # first time we have a unique pair, alpha is 1.0 else is 0
                    if len(self._unique_obs_action) == 1:
                        self.internal_dataset['alpha'] = torch.cat([
                            self.internal_dataset['alpha'],
                            torch.tensor([1.0], device=self.device).unsqueeze(0)
                        ], dim=0)
                    else:
                        self.internal_dataset['alpha'] = torch.cat([
                            self.internal_dataset['alpha'],
                            torch.tensor([0.0], device=self.device).unsqueeze(0)
                        ], dim=0)
                self._prev_obs = time_step.observation

            elif time_step.step_type == StepType.LAST:
                obs_action_pair = (np.argmax(self._prev_obs), int(time_step.action))
                if obs_action_pair not in self._unique_obs_action:
                    self._unique_obs_action.add(obs_action_pair)
                    # Convert to tensors and concatenate
                    obs_idx = np.argmax(self._prev_obs)
                    self.internal_dataset['observation'] = torch.cat([
                        self.internal_dataset['observation'],
                        torch.tensor([obs_idx], device=self.device).unsqueeze(0)
                    ], dim=0)
                    self.internal_dataset['action'] = torch.cat([
                        self.internal_dataset['action'],
                        torch.tensor([time_step.action], dtype=torch.long, device=self.device)
                    ], dim=0)
                    next_obs_idx = np.argmax(time_step.observation)
                    self.internal_dataset['next_observation'] = torch.cat([
                        self.internal_dataset['next_observation'],
                        torch.tensor([next_obs_idx], device=self.device).unsqueeze(0)
                    ], dim=0)
            else:
                raise ValueError("Unknown step type")
            
            if len(self._unique_obs_action) == self.n_states * self.n_actions and "enc_obs_action" not in self.internal_dataset:
                # Precompute encoded state-action pairs for faster training
                obs_tensor = self.internal_dataset['observation'].unsqueeze(-1).float()  # [num_unique, 1]
                action_tensor = self.internal_dataset['action'].unsqueeze(-1).float()  # [num_unique, 1]
                self.internal_dataset['enc_obs_action'] = self._enc_obs_action(obs_tensor.to(self.device), action_tensor.to(self.device))  # Store for faster access
                self.internal_dataset['enc_next_obs'] = self.encoder(self.internal_dataset['next_observation'].unsqueeze(-1).float().to(self.device))  # [num_unique, feature_dim]
                self.internal_dataset['enc_obs'] = self.encoder(self.internal_dataset['observation'].unsqueeze(-1).float().to(self.device))  # [num_unique, feature_dim]

            assert self.internal_dataset['observation'].shape[0] == self.internal_dataset['action'].shape[0] == self.internal_dataset['next_observation'].shape[0], \
                f"len(obs): {self.internal_dataset['observation'].shape[0]}, len(action): {self.internal_dataset['action'].shape[0]}, len(next_obs): {self.internal_dataset['next_observation'].shape[0]}"
            assert len(self._unique_obs_action) <= self.n_states * self.n_actions, \
                f"len unique obs-action pairs: {len(self._unique_obs_action)}, n_states * n_actions: {self.n_states * self.n_actions}" 

        elif self.internal_dataset_type == "all":
            
            # *** Store all dataset ***
            if time_step.step_type == StepType.FIRST:  # first step
                obs_idx = np.argmax(time_step.observation)
                self.internal_dataset['observation'] = torch.cat([
                    self.internal_dataset['observation'],
                    torch.tensor([obs_idx], device=self.device).unsqueeze(0)
                ], dim=0)

                # first time we have a unique pair, alpha is 1.0 else is 0
                if self.internal_dataset['observation'].shape[0] == 1:
                    self.internal_dataset['alpha'] = torch.cat([
                        self.internal_dataset['alpha'],
                        torch.tensor([1.0], device=self.device).unsqueeze(0)
                    ], dim=0)
                else:
                    self.internal_dataset['alpha'] = torch.cat([
                        self.internal_dataset['alpha'],
                        torch.tensor([0.0], device=self.device).unsqueeze(0)
                    ], dim=0)
            elif time_step.step_type == StepType.MID:  # mid step
                obs_idx = np.argmax(self._prev_obs)
                self.internal_dataset['observation'] = torch.cat([
                    self.internal_dataset['observation'],
                    torch.tensor([obs_idx], device=self.device).unsqueeze(0)
                ], dim=0)
                self.internal_dataset['action'] = torch.cat([
                    self.internal_dataset['action'],
                    torch.tensor([time_step.action], dtype=torch.long, device=self.device)
                ], dim=0)
                next_obs_idx = np.argmax(time_step.observation)
                self.internal_dataset['next_observation'] = torch.cat([
                    self.internal_dataset['next_observation'],
                    torch.tensor([next_obs_idx], device=self.device).unsqueeze(0)
                ], dim=0)
                self.internal_dataset['alpha'] = torch.cat([
                        self.internal_dataset['alpha'],
                        torch.tensor([0.0], device=self.device).unsqueeze(0)
                    ], dim=0)
            elif time_step.step_type == StepType.LAST:  # last step
                self.internal_dataset['action'] = torch.cat([
                    self.internal_dataset['action'],
                    torch.tensor([time_step.action], dtype=torch.long, device=self.device)
                ], dim=0)
                next_obs_idx = np.argmax(time_step.observation)
                self.internal_dataset['next_observation'] = torch.cat([
                    self.internal_dataset['next_observation'],
                    torch.tensor([next_obs_idx], device=self.device).unsqueeze(0)
                ], dim=0)
                assert self.internal_dataset['observation'].shape[0] == self.internal_dataset['action'].shape[0] == self.internal_dataset['next_observation'].shape[0], f"len(obs): {self.internal_dataset['observation'].shape[0]}, len(action): {self.internal_dataset['action'].shape[0]}, len(next_obs): {self.internal_dataset['next_observation'].shape[0]}"
            else:
                raise ValueError("Unknown step type")
        
        else:
            raise ValueError(f"Unknown internal_dataset_type: {self.internal_dataset_type}")

        
        return meta
    
    def _enc_obs_action(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        obs_enc = self.encoder(obs) # [batch_size, feature_dim]
        action_onehot = F.one_hot(action.long(), num_classes=self.n_actions).float().reshape(-1, self.n_actions) # [batch_size, n_actions]

        obs_enc_action = torch.einsum('be,ba->bea', obs_enc, action_onehot).reshape(obs_enc.shape[0], -1)

        return obs_enc_action

    def _create_uniform_policy(self) -> np.ndarray:
        """Create uniform random policy operator."""
        P = np.zeros((self.n_states * self.n_actions, self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                row = s * self.n_actions + a
                P[row, s] = 1.0 / self.n_actions
                
        return P
    
    def _create_random_policy(self) -> np.ndarray:
        """Create random random policy operator."""
        P = np.zeros((self.n_states * self.n_actions, self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                row = s * self.n_actions + a
                P[row, s] = np.random.rand()
            P[:, s] /= P[:, s].sum()
        return P

    def _create_transition_matrix(self) -> np.ndarray:
        """
        Get the transition matrix T for all state-action pairs.
        
        Returns:
            T: Transition matrix of shape (n_states, n_states * n_actions)
               T[s', s*n_actions + a] = 1 if action a in state s leads to s'
        """
        self.best_T_learnt = False
        T = nn.Linear(self.n_states * self.n_actions, self.n_states, bias=False).to(self.device)
        self.T_optimizer = torch.optim.Adam(T.parameters(), lr=self.lr_T)
        # T = np.zeros((self.n_states, self.n_states * self.n_actions))
        return T
    

    def _from_operator_to_policy(self, policy_operator: np.ndarray = None) -> np.ndarray:
        """Convert policy operator to policy matrix."""
        if policy_operator is None:
            policy_operator = self.policy_operator
        policy_matrix = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                row = s * self.n_actions + a
                policy_matrix[s, a] = policy_operator[row, s]
        return policy_matrix
            

    def load_policy_operator(self, path: str):
        """Load policy operator from file."""
        self.policy_operator = np.load(path)
        self.policy_matrix = self._from_operator_to_policy(self.policy_operator)
        print(f"Loaded policy operator from: {path}")
    
    
    def act(self, obs, meta, step, eval_mode):
        if step < self.num_expl_steps:
            return np.random.randint(self.n_actions)
        
        if "enc_obs" not in self.internal_dataset:
            phi_all_next_obs = self.encoder(self.internal_dataset['next_observation'].unsqueeze(-1).float().to(self.device)).detach().cpu().numpy()  # [num_unique, feature_dim]
            psi_all_obs_action = self._enc_obs_action(
                self.internal_dataset['observation'].unsqueeze(-1).float().to(self.device), 
                self.internal_dataset['action'].unsqueeze(-1).float().to(self.device)
            ).detach().cpu().numpy()  # [num_unique, feature_dim * n_actions]
            phi_all_obs = self.encoder(self.internal_dataset['observation'].unsqueeze(-1).float().to(self.device)).detach().cpu().numpy()  # [num_unique, feature_dim]
        else:
            phi_all_next_obs = self.internal_dataset['enc_next_obs'].detach().cpu().numpy()  # [num_unique, feature_dim]
            psi_all_obs_action = self.internal_dataset['enc_obs_action'].detach().cpu().numpy()  # [num_unique, feature_dim * n_actions]
            phi_all_obs = self.internal_dataset['enc_obs'].detach().cpu().numpy()  # [num_unique, feature_dim]

        # if type(self.T_operator) is nn.Linear:
        #     if self.best_T_learnt is False:
        #         self.update_closed_form_transition_matrix()
        #         self.update_initial_distribution()

        I_n = np.eye(phi_all_next_obs.shape[0])
        E = F.one_hot(self.internal_dataset['action'].long(), num_classes=self.n_actions).float().detach().cpu().numpy()  # [num_unique, n_actions]
        M = psi_all_obs_action @ self.policy_operator @ phi_all_next_obs.T  # [num_unique, num_unique]
        BM = np.linalg.solve(psi_all_obs_action @ psi_all_obs_action.T + 1e-6 * I_n, M)  # [num_unique, num_unique]

        coeff = np.diag(np.linalg.solve(I_n - self.discount * BM, self.internal_dataset['alpha'].detach().cpu().numpy()).flatten())@E  # [num_unique, n_actions]

        phi_current_obs = self.encoder(torch.tensor([np.argmax(obs)], dtype=torch.float32).unsqueeze(-1).to(self.device)).detach().cpu().numpy()  # [1, feature_dim]
        kernel_values = phi_current_obs @ phi_all_obs.T  # [1, num_unique]
        action_scores = kernel_values @ coeff  # [1, n_actions]
        action_probs = F.softmax(torch.tensor(action_scores).float(), dim=-1).numpy()
        if np.sum(action_probs) == 0.0 or np.isnan(np.sum(action_probs)):
            raise ValueError("action_probs sum to zero or NaN")
        return np.random.choice(self.n_actions, p=action_probs.flatten())

       
    def update_closed_form_transition_matrix(self):
        if "enc_obs_action" not in self.internal_dataset:
            psi_obs_action = self._enc_obs_action(
                self.internal_dataset['observation'].unsqueeze(-1).float().to(self.device), 
                self.internal_dataset['action'].unsqueeze(-1).float().to(self.device)
            )  # [num_unique, feature_dim * n_actions]
            phi_next_s = self.encoder(self.internal_dataset['next_observation'].unsqueeze(-1).float().to(self.device))  # [num_unique, feature_dim]
        else:
            self.best_T_learnt = True*(self.internal_dataset_type == "unique") # this is just because of the unique dataset not being filled yet
            
            psi_obs_action = self.internal_dataset['enc_obs_action']  # [num_unique, feature_dim * n_actions]
            phi_next_obs = self.internal_dataset['enc_next_obs']  # [num_unique, feature_dim]
        
        self.T_operator = phi_next_obs.T @ torch.linalg.solve(
            psi_obs_action @ psi_obs_action.T + 1e-6 * torch.eye(psi_obs_action.shape[0], device=self.device),
            psi_obs_action
        )  # [feature_dim, feature_dim * n_actions]
       
        # Transion matrix to np.ndarray
        self.T_operator = self.T_operator.detach().cpu().numpy()

        return 
    
    def update_transition_matrix(self, obs, action, next_obs):
        metris = dict()

        T_loss = F.mse_loss(
            self.T_operator(
                self._enc_obs_action(torch.argmax(obs, dim=-1).unsqueeze(-1).float().to(self.device), action.unsqueeze(-1).float().to(self.device))
            ),
            self.encoder(torch.argmax(next_obs, dim=-1).unsqueeze(-1).float().to(self.device))
            )
        self.encoder_optimizer.zero_grad()
        self.T_optimizer.zero_grad()
        T_loss.backward()
        self.encoder_optimizer.step()
        self.T_optimizer.step()
        print(f"T_loss: {T_loss.item()}")
        if self.use_tb or self.use_wandb:
            metris['T_loss'] = T_loss.item()
        return metris

    def update_initial_distribution(self):
        phi_obs = self.encoder(self.internal_dataset['next_observation'].unsqueeze(-1).float().to(self.device)).detach().cpu().numpy()  # [num_unique, feature_dim]
        self.initial_distribution = phi_obs.T@self.internal_dataset['alpha'].to(self.device).detach().cpu().numpy()


    def update_actor(self):
        metrics = dict()
        if self.best_T_learnt is False:
            self.update_closed_form_transition_matrix()
            self.update_initial_distribution()

        if self.gradient_type == 'MMD':
            nu_pi = self.distribution_matcher.compute_discounted_occupancy(self.initial_distribution, self.policy_operator, self.T_operator)
            # actor loss is the norm of nu_pi with the target = 0
            actor_loss = np.linalg.norm(nu_pi) 
        else:
            raise ValueError(f"Unknown gradient type: {self.gradient_type}")
        
        gradient = self.distribution_matcher.compute_new_gradient(self.policy_operator, self.T_operator, self.initial_distribution)
        self.policy_operator = self.distribution_matcher.mirror_descent_update(self.policy_operator, gradient)
        
        self.policy_matrix = self._from_operator_to_policy(self.policy_operator)

        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss
        
        self.distribution_matcher.append_loss_history(actor_loss)

        return metrics

    def aug_and_encode(self, obs):
        pass
        # obs = self.aug(obs)
        # return self.encoder(obs)

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # # augment and encode
        # obs = self.aug_and_encode(obs)
        # with torch.no_grad():
        #     next_obs = self.aug_and_encode(next_obs)

        if self.use_tb or self.use_wandb:
            metrics['batch_reward'] = reward.mean().item()
        
        if step < self.num_expl_steps + self.T_learning_steps:
            # update transition matrix
            metrics.update(self.update_transition_matrix(obs, action, next_obs))
            return metrics
        # update actor
        metrics.update(self.update_actor())
        self.distribution_matcher.update_internal_policy(self.policy_operator)
        
        if step%6000 == 0:
            print(f"Last KL divergence: {self.distribution_matcher.kl_history[-1]}")
            self.visualizer.plot_results(
                self.initial_distribution, 
                self.nu_target, 
                self.distribution_matcher.compute_discounted_occupancy(self.initial_distribution, self.policy_operator, self.T_operator), 
                False,
                os.getcwd()+f"/plot_{step}")
            print(f"os.getcwd(): {os.getcwd()}")
        return metrics

