from collections import OrderedDict
import os
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from distribution_matching import DistributionVisualizer

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
                 gamma: float = 0.9, 
                 eta: float = 0.1, 
                 alpha = 0.1, 
                 gradient_type: str = 'reverse'):
        self.gamma = gamma
        self.eta = eta
        self.nu0 = nu0
        self.gradient_type = gradient_type
        self.alpha = alpha
        self.uniform_policy_operator = np.ones((n_states * n_actions, n_states)) / n_actions
        self.policy_operator = self.uniform_policy_operator.copy()
        
        self.n_states = n_states
        self.n_actions = n_actions

        self.lava=False
        self.kl_history = []

    
    
    @staticmethod
    def M_pi_operator(P: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Compute M_π = T ∘ Π_π operator.
        Maps state distribution to next state distribution under policy.
        """
        return T @ P

    def append_kl_history(self, kl_value: float):
        self.kl_history.append(kl_value)

    def compute_discounted_occupancy(self, nu0: np.ndarray, P: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Compute discounted occupancy measure: ν_π = (1-γ)(I - γM_π)^{-1} ν_0
        
        Args:
            nu0: Initial state distribution
            P: Policy operator (uses self.policy_operator if None)
        
        Returns:
            Discounted occupancy measure
        """
            
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

    def compute_gradient_kl(self, nu0: np.ndarray, nu_target: np.ndarray, P: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Compute gradient of KL divergence w.r.t. policy.
        
        Args:
            nu0: Initial state distribution
            nu_target: Target state distribution
        
        Returns:
            Gradient of shape (n_states * n_actions, n_states)
        """
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
        else:
            raise ValueError(f"Unknown gradient type: {self.gradient_type}")
        
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

class DistMatchingAgent:
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
                 learn_T,
                 lr_T,
                 update_every_steps,
                 num_expl_steps,
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
        self.learn_T = learn_T
        self.lr_T = lr_T
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.device = device
        self.gradient_type = 'reverse'  # could be 'forward' but it is not working practically (it is a theorethical problem not a bug)
        self.num_expl_steps = num_expl_steps
        visited_states = set()

        self.initial_distribution = None

        if starting_policy == "uniform":
            self.policy_operator = self._create_uniform_policy()
        elif starting_policy == "random":
            self.policy_operator = self._create_random_policy()
        else:
            raise ValueError(f"Unknown starting policy: {starting_policy}")
        

        self.nu_target = (np.ones(self.n_states) / self.n_states).reshape(-1,1) # Homogeneus target distribution


        self.policy_matrix = self._from_operator_to_policy(self.policy_operator)
        self.training = False

        
    
    def train(self, training=True):
        self.training = training
    
    def insert_env(self, env):
        self.env = env.unwrapped

    def init_meta(self):
        # TODO future reminder: Remove this when using non-custom envs. we access some env methods that are not available in usual gym envs
        if self.initial_distribution is None:
            self.initial_distribution = self._create_initial_distribution()
            if self.learn_T:
                self.T_operator = np.zeros((self.n_states, self.n_states * self.n_actions))
            else:
                # ATTENTION: not all gym envs give free access to the simulator
                self.T_operator = self._create_transition_matrix()
            self.distribution_matcher = DistributionMatcher(
                n_states=self.n_states,
                n_actions=self.n_actions,
                nu0=self.initial_distribution,
                gamma=self.discount,
                eta=self.lr_actor,
                alpha=self.alpha,
                gradient_type=self.gradient_type
            )
            self.visualizer = DistributionVisualizer(self.env, self.distribution_matcher)
        return OrderedDict()

    def get_meta_specs(self):
        return tuple()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        return meta
    
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
        
        T = np.zeros((self.n_states, self.n_states * self.n_actions))
        
        for s_idx in range(self.n_states):
            cell = self.env.idx_to_state[s_idx]
            for action in range(self.n_actions):
                next_cell = self.env.step_from(cell, action)
                next_idx = self.env.state_to_idx[next_cell]
                col = s_idx * self.n_actions + action
                T[next_idx, col] = 1.0
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

    def _create_initial_distribution(self, mode: str = 'top_left_cell') -> np.ndarray: # TODO be careful with this when using non-custom envs
        """
        Create initial state distribution.
        
        Args:
            env: gym.Env instance
            mode: Distribution mode ('top_left_cell', 'uniform', 'dirac_delta')
                - 'top_left_cell': Single cell at top-left (0, 0)
                - 'uniform': Uniform distribution over all states
                - 'dirac_delta': Dirac delta at the environment's start position (must be set)
        
        Returns:
            Initial distribution as column vector
        """
        nu0 = np.zeros(self.env.n_states)
        
        if mode == 'top_left_cell':
            # Single cell at top-left (0, 0)
            if (0, 0) in self.env.state_to_idx:
                nu0[self.env.state_to_idx[(0, 0)]] = 1.0
            else:
                raise ValueError("Cell (0, 0) is not a valid cell in this environment")
        
        elif mode == 'uniform':
            # Uniform over all states
            nu0 = np.ones(self.env.n_states) / self.env.n_states
        
        elif mode == 'dirac_delta':
            # Dirac delta at the environment's start position
            if self.env.start_position is None:
                raise ValueError(
                    "mode='dirac_delta' requires start_position to be set in the environment. "
                    "Please specify start_position in the config or use reset(options={'start_position': ...})"
                )
            if self.env.start_position not in self.env.state_to_idx:
                raise ValueError(f"Start position {self.env.start_position} is not a valid cell")
            nu0[self.env.state_to_idx[self.env.start_position]] = 1.0
        
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Choose from 'top_left_cell', 'uniform', 'dirac_delta'"
            )
        
        return nu0.reshape((-1, 1))


    def act(self, obs, meta, step, eval_mode):
        if step < self.num_expl_steps:
            return np.random.randint(self.n_actions)
        # One hot to state
        state = np.argmax(obs)
        action_probs = self.policy_matrix[state]
        action_probs = action_probs / np.sum(action_probs)  # Normalize probabilities, it could be slightly off due to numerical issues
        if eval_mode:
            return np.argmax(action_probs)
        return np.random.choice(self.n_actions, p=action_probs)
       
    def update_transition_matrix(self, obs, action, next_obs):
        metrics = dict()
        loss = 0.0
        for s, a, s_next in zip(obs, action, next_obs):
            # Convert tensors to integers
            state = int(torch.argmax(s).item())
            next_state = int(torch.argmax(s_next).item())

            col = state * self.n_actions + a
            loss += (1.0 - self.T_operator[next_state, col])**2
            self.T_operator[next_state, col] = 1.0 # += self.lr_T * (1.0 - self.T_operator[next_state, col])
        
        metrics['T_loss'] = loss / obs.shape[0]
            
        return metrics
        # for s_idx in range(self.n_states):
        #     cell = self.env.idx_to_state[s_idx]
        #     for action in range(self.n_actions):
        #         next_cell = self.env.step_from(cell, action)
        #         next_idx = self.env.state_to_idx[next_cell]
        #         col = s_idx * self.n_actions + action
        #         T[next_idx, col] = 1.0
        # return T

    def update_actor(self):
        metrics = dict()
        if self.gradient_type == 'reverse':
                nu_pi = self.distribution_matcher.compute_discounted_occupancy(self.initial_distribution, self.policy_operator, self.T_operator)
                actor_loss = self.distribution_matcher.kl_divergence(nu_pi, self.nu_target, self.policy_operator)
        else:  # forward
            I = np.eye(self.n_states)
            M = self.distribution_matcher.M_pi_operator(self.policy_operator, self.T_operator)
            actor_loss = self.distribution_matcher.kl_divergence(self.initial_distribution, (I - self.discount * M)/(1-self.discount) @ self.nu_target, self.policy_operator)

        gradient = self.distribution_matcher.compute_gradient_kl(self.initial_distribution, self.nu_target, self.policy_operator, self.T_operator)
        self.policy_operator = self.distribution_matcher.mirror_descent_update(self.policy_operator, gradient)
        
        self.policy_matrix = self._from_operator_to_policy(self.policy_operator)

        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss
        
        self.distribution_matcher.append_kl_history(actor_loss)

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

        if self.learn_T:
            # update transition matrix
            metrics.update(self.update_transition_matrix( obs, action, next_obs))

        # update actor
        metrics.update(self.update_actor())
        self.distribution_matcher.update_internal_policy(self.policy_operator)
        
        if (step%4300 == 0 or step%4600 == 0):
            print(f"Last KL divergence: {self.distribution_matcher.kl_history[-1]}")
            self.visualizer.plot_results(
                self.initial_distribution, 
                self.nu_target, 
                self.distribution_matcher.compute_discounted_occupancy(self.initial_distribution, self.policy_operator, self.T_operator), 
                False,
                os.getcwd()+f"/plot_{step}")
            print(f"os.getcwd(): {os.getcwd()}")
        return metrics

     