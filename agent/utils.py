import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from utils import ColorPrint
from typing import Tuple, Optional, Dict
from dm_env import StepType, specs
from copy import deepcopy

import numpy as np
from time import time
import os
import matplotlib
matplotlib.use('Agg')  # Backend non-interattivo per salvare senza display
import matplotlib.pyplot as plt

# torch.set_default_tensor_type(torch.FloatTensor)
float_type = torch.float32

class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class ActorDiscrete(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim, linear= False):
        super().__init__()

        if linear:
            self.trunk = nn.Identity(gradient=False)
            self.policy = nn.Linear(obs_dim, action_dim)
            self.apply(utils.weight_init)
            ColorPrint.yellow("Using linear actor!")
            return

        feature_dim = feature_dim if obs_type == 'pixels' else hidden_dim

        self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        policy_layers = []
        policy_layers += [
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True)
        ]
        # add additional hidden layer for pixels
        if obs_type == 'pixels':
            policy_layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
        policy_layers += [nn.Linear(hidden_dim, action_dim)]

        self.policy = nn.Sequential(*policy_layers)
        

        self.apply(utils.weight_init)

    def forward(self, obs):
        h = self.trunk(obs)

        logits = self.policy(h)
       
        return F.softmax(logits, dim=1)

    def _logits(self, obs):
        # Helper to get logits without softmax
        h = self.trunk(obs)
        return self.policy(h)

    def get_log_p(self, states, actions):
        """
        states:  (T, obs_dim) or (batch, obs_dim)
        actions: (T,) or (batch,) float with action indices
        returns: (T,) log-probabilities log pi(a_t | s_t)
        """
        logits = self._logits(states)                      # (T, K)
        log_probs = F.log_softmax(logits, dim=-1)          # (T, K)
        # convert actions to int64 for gather
        actions = actions.long()             # (T, 1)
        # Gather the log-prob of the taken action at each step
        log_p = log_probs.gather(dim=1, index=actions)     # (T, 1)
        return log_p.squeeze(-1)                           # (T,)

class KernelActorDiscrete(nn.Module):
    """
    Kernel-based actor that computes: π(a|s) = softmax(-η · (H^T · C_actions ⊙ E + C_bias))
    where:
    - H = [φ(s); 0] @ Φ_dataset^T  (augmented kernel similarities)
    - C_actions: gradient coefficients for state-action pairs [dataset_dim, n_actions]
    - C_bias: bias term for each action (last row of original gradient_coeff)
    - E: action one-hot encoding matrix [dataset_dim, n_actions]
    
    This architecture allows loading pretrained kernel weights and
    optionally finetuning them with RL algorithms.
    """
    
    def __init__(self, obs_type, input_dim, dataset_dim, action_dim, eta, trainable=True):
        """
        Args:
            obs_type: Type of observation ('states' or 'pixels')
            input_dim: Dimension of input features (d)
            dataset_dim: Number of dataset examples (n)
            action_dim: Number of actions
            eta: Scalar scaling factor (learning rate)
            trainable: If True, allows weights to be updated during finetuning
        """
        super().__init__()
        
        # Layer 1: Kernel layer computes H = [φ(x); 0] @ Φ_dataset^T
        # We need input_dim+1 to account for the augmented zero
        self.kernel_layer = nn.Linear(input_dim + 1, dataset_dim, bias=False, dtype=float_type)
        
        # Layer 2: Action-specific gradient coefficients
        # Shape: [dataset_dim, n_actions] corresponding to C[:-1] ⊙ E in original formulation
        self.action_coeffs = nn.Linear(dataset_dim, action_dim, bias=False, dtype=float_type)
        
        # Bias term: corresponds to C[-1] in original formulation
        # This is added uniformly to all actions
        self.bias_coeff = nn.Parameter(torch.zeros(action_dim, dtype=float_type))
        
        self.eta = nn.Parameter(torch.tensor(eta, dtype=float_type), requires_grad=trainable)
        self.softmax = nn.Softmax(dim=1)
        
        # Control whether weights are trainable
        if not trainable:
            for param in self.parameters():
                param.requires_grad = False
        
        self.apply(utils.weight_init)

    def initialize_from_pretrained(self, phi_dataset, gradient_coeff, eta, E=None):
        """
        Initialize weights from pretrained kernel policy.
        
        Args:
            phi_dataset: [num_unique, feature_dim+1] - augmented dataset feature matrix
            gradient_coeff: [num_unique+1, 1] - learned coefficients (last element is bias)
            eta: scalar - learning rate / temperature
            E: [num_unique, n_actions] - action one-hot encoding matrix (optional)
        """
        # 1. Initialize kernel layer: W = Φ_dataset (augmented with zeros)
        self.kernel_layer.weight.data.copy_(phi_dataset)
        
        # 2. Split gradient_coeff into action coeffs and bias
        # gradient_coeff shape: [num_unique+1, 1]
        # C[:-1] are action-specific coefficients, C[-1] is the bias
        action_grad = gradient_coeff[:-1].squeeze(-1)  # [num_unique]
        bias_grad = gradient_coeff[-1].item()  # scalar
        
        # 3. Initialize action_coeffs layer
        # We need to account for element-wise multiplication with E
        # Original: H @ (C[:-1] ⊙ E) where ⊙ is element-wise product
        # If E is provided, we can pre-compute C[:-1] ⊙ E
        
        # E shape: [num_unique, n_actions]
        # C[:-1] shape: [num_unique]
        # Broadcasting: C[:-1].unsqueeze(1) * E → [num_unique, n_actions]
        weighted_E = action_grad.unsqueeze(1) * E  # [num_unique, n_actions]
        # action_coeffs.weight shape: [n_actions, num_unique]
        # We want: logits = H @ weighted_E = H @ W^T, so W^T = weighted_E
        self.action_coeffs.weight.data.copy_(weighted_E.T)
    
        
        # 4. Initialize bias term
        self.bias_coeff.data.fill_(bias_grad)
        
        # 5. Set eta
        self.eta.data.copy_(torch.tensor(eta))
        print("all dtypes:", self.kernel_layer.weight.dtype, self.action_coeffs.weight.dtype, self.bias_coeff.dtype, self.eta.dtype)
        print(f"Kernel actor initialized from pretrained weights:")
        print(f"  - Kernel layer: {self.kernel_layer.weight.shape}")
        print(f"  - Action coeffs: {self.action_coeffs.weight.shape}")
        print(f"  - Bias: {self.bias_coeff.shape}")
        print(f"  - Eta: {self.eta.item()}")

    def forward(self, phi_x):
        """
        Forward pass matching dist_matching_embedding_augmented.py structure:
        
        1. Augment φ(x) with zero: [φ(x); 0]
        2. Compute kernel similarities: H = [φ(x); 0] @ Φ_dataset^T
        3. Apply gradient coefficients: 
           - action_logits = H @ (C[:-1] ⊙ E)  [via action_coeffs layer]
           - bias_logits = 1 * C[-1]             [via bias_coeff parameter]
        4. Combine: logits = action_logits + bias_logits
        5. Apply softmax: π(a|s) = softmax(-η * logits)
        
        Args:
            phi_x: [batch_size, feature_dim] - encoded observations
            
        Returns:
            probs: [batch_size, n_actions] - action probabilities
        """
        batch_size = phi_x.shape[0]
        
        # Step 1: Augment φ(x) con zero nell'ultima dimensione
        # Original: enc_obs_augmented = torch.cat([enc_obs, torch.zeros((1, 1))], dim=1)
        phi_x_aug = torch.cat([phi_x, torch.zeros(batch_size, 1, device=phi_x.device)], dim=1)
        phi_x_aug = phi_x_aug.to(dtype=float_type)
        # Shape: [batch_size, feature_dim + 1]
        
        # Step 2: Calcola le similarità del kernel H = [φ(x); 0] @ Φ_dataset^T
        # kernel_layer computes: H = phi_x_aug @ kernel_layer.weight^T
        h = self.kernel_layer(phi_x_aug)
        # Shape: [batch_size, dataset_dim]
        
        # Step 3a: Applica i coefficienti del gradiente specifici per azione
        # Original: H @ (self.gradient_coeff[:-1] * self.E)
        # action_coeffs.weight già contiene (C[:-1] ⊙ E)^T
        action_logits = self.action_coeffs(h)
        # Shape: [batch_size, n_actions]
        
        # Step 3b: Aggiungi il termine di bias (corrisponde a C[-1] nella formulazione originale)
        # Original: + torch.ones(1, self.E.shape[1]) * self.gradient_coeff[-1]
        bias_logits = self.bias_coeff.unsqueeze(0).expand(batch_size, -1)
        # Shape: [batch_size, n_actions]
        
        # Step 4: Combina i logit delle azioni e il bias
        logits = action_logits + bias_logits
        
        # Step 5: Scala per -eta e applica softmax
        # Original: torch.softmax(-self.lr_actor * (...), dim=1)
        probs = self.softmax(-self.eta * logits)
        
        return probs

    def _logits(self, phi_x):
        """
        Ottieni i logit senza softmax (utile per alcuni algoritmi RL).
        Segue lo stesso calcolo di forward() ma restituisce i logit grezzi.
        """
        batch_size = phi_x.shape[0]
        
        # Augmenta con zero
        phi_x_aug = torch.cat([phi_x, torch.zeros(batch_size, 1, device=phi_x.device)], dim=1)
        
        # Similarità del kernel
        h = self.kernel_layer(phi_x_aug)
        
        # Logit specifici per azione + bias
        action_logits = self.action_coeffs(h)
        bias_logits = self.bias_coeff.unsqueeze(0).expand(batch_size, -1)
        logits = action_logits + bias_logits
        
        # Scala per -eta (senza softmax)
        return -self.eta * logits

    def get_log_p(self, phi_x, actions):
        """
        Compute log probabilities for given actions.
        
        Args:
            phi_x: [T, feature_dim] - encoded states
            actions: [T] - action indices
            
        Returns:
            log_p: [T] - log probabilities
        """
        logits = self._logits(phi_x)  # [T, n_actions]
        log_probs = F.log_softmax(logits, dim=-1)  # [T, n_actions]
        
        # Gather log-prob of taken actions
        actions = actions.long().unsqueeze(1)  # [T, 1]
        log_p = log_probs.gather(dim=1, index=actions)  # [T, 1]
        
        return log_p.squeeze(-1)  # [T]
    
class CriticDiscrete(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        self.obs_type = obs_type

        if obs_type == 'pixels':
            # for pixels actions will be added after trunk
            self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
                                       nn.LayerNorm(feature_dim), nn.Tanh())
            trunk_dim = feature_dim
        else:
            # for states actions come in the beginning
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.Tanh())
            trunk_dim = hidden_dim

        def make_q():
            q_layers = []
            q_layers += [
                nn.Linear(trunk_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
            if obs_type == 'pixels':
                q_layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True)
                ]
            q_layers += [nn.Linear(hidden_dim, action_dim)]
            return nn.Sequential(*q_layers)

        self.Q1 = make_q()
        self.Q2 = make_q()

        self.apply(utils.weight_init)

    def forward(self, obs):
        inpt = obs
        h = self.trunk(inpt)

        q1 = self.Q1(h)
        q2 = self.Q2(h)

        return q1, q2
    


class Actor(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        feature_dim = feature_dim if obs_type == 'pixels' else hidden_dim

        self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        policy_layers = []
        policy_layers += [
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True)
        ]
        # add additional hidden layer for pixels
        if obs_type == 'pixels':
            policy_layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
        policy_layers += [nn.Linear(hidden_dim, action_dim)]

        self.policy = nn.Sequential(*policy_layers)

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        self.obs_type = obs_type

        if obs_type == 'pixels':
            # for pixels actions will be added after trunk
            self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
                                       nn.LayerNorm(feature_dim), nn.Tanh())
            trunk_dim = feature_dim + action_dim
        else:
            # for states actions come in the beginning
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.Tanh())
            trunk_dim = hidden_dim

        def make_q():
            q_layers = []
            q_layers += [
                nn.Linear(trunk_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
            if obs_type == 'pixels':
                q_layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True)
                ]
            q_layers += [nn.Linear(hidden_dim, 1)]
            return nn.Sequential(*q_layers)

        self.Q1 = make_q()
        self.Q2 = make_q()

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        inpt = obs if self.obs_type == 'pixels' else torch.cat([obs, action],
                                                               dim=-1)
        h = self.trunk(inpt)
        h = torch.cat([h, action], dim=-1) if self.obs_type == 'pixels' else h

        q1 = self.Q1(h)
        q2 = self.Q2(h)

        return q1, q2


# # ****From MEPOL ****

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F  # Correzione qui



# from collections import OrderedDict
# torch.set_default_tensor_type(torch.FloatTensor)
# float_type = torch.float32

# int_type = torch.int64
# eps = 1e-7

# class GaussianPolicy(nn.Module):
#     """
#     Gaussian Policy with state-independent diagonal covariance matrix
#     """

#     def __init__(self, hidden_sizes, num_features, action_dim, log_std_init=-0.5, activation=nn.ReLU):
#         super().__init__()

#         self.activation = activation

#         layers = []
#         layers.extend((nn.Linear(num_features, hidden_sizes[0]), self.activation()))
#         for i in range(len(hidden_sizes) - 1):
#             layers.extend((nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), self.activation()))

#         self.net = nn.Sequential(*layers)

#         self.mean = nn.Linear(hidden_sizes[-1], action_dim)
#         self.log_std = nn.Parameter(log_std_init * torch.ones(action_dim, dtype=float_type))

#         # Constants
#         self.log_of_two_pi = torch.tensor(np.log(2*np.pi), dtype=float_type)

#         self.initialize_weights()

#     def initialize_weights(self):
#         nn.init.xavier_uniform_(self.mean.weight)

#         for l in self.net:
#             if isinstance(l, nn.Linear):
#                 nn.init.xavier_uniform_(l.weight)

#     def get_log_p(self, states, actions):
#         mean, _ = self(states)
#         return torch.sum(
#             -0.5 * (
#                 self.log_of_two_pi
#                 + 2*self.log_std
#                 + ((actions - mean)**2 / (torch.exp(self.log_std) + eps)**2)
#             ), dim=1
#         )

#     def forward(self, x, deterministic=False):
#         mean = self.mean(self.net(x))

#         if deterministic:
#             output = mean
#         else:
#             output = mean + torch.randn(mean.size(), dtype=float_type) * torch.exp(self.log_std)

#         return mean, output


#     def predict(self, s, deterministic=False):
#         with torch.no_grad():
#             s = torch.tensor(s, dtype=float_type).unsqueeze(0)
#             return self(s, deterministic=deterministic)[1][0]


# def train_supervised(env, policy, train_steps=100, batch_size=5000):
#     optimizer = torch.optim.Adam(policy.parameters(), lr=0.00025)
#     dict_like_obs = True if type(env.observation_space.sample()) is OrderedDict else False

#     for _ in range(train_steps):
#         optimizer.zero_grad()

#         if dict_like_obs:
#             states = torch.tensor([env.observation_space.sample()['observation'] for _ in range(5000)], dtype=float_type)
#         else:
#             states = torch.tensor([env.observation_space.sample()[:env.num_features] for _ in range(5000)], dtype=float_type)

#         actions = policy(states)[0]
#         loss = torch.mean((actions - torch.zeros_like(actions, dtype=float_type)) ** 2)

#         loss.backward()
#         optimizer.step()

#     return policy

# ============================================================================
# Internal Dataset Management
# ============================================================================
class InternalDataset:
    """Manages the agent's internal experience buffer."""
    
    def __init__(self, dataset_type: str, n_states: int, n_actions: int, gamma: float, n_subsamples: int, device: str = 'cpu'):
        self.dataset_type = dataset_type
        self.n_states = n_states
        self.n_actions = n_actions
        self.expected_size = n_states * n_actions
        assert dataset_type in ("unique", "all"), "dataset_type must be 'unique' or 'all'"
        self.n_subsamples = n_subsamples
        self.gamma = gamma
        self.device = torch.device(device)
        self.reset()
    
    def reset(self):
        utils.ColorPrint.yellow("Resetting internal dataset.")
        self.data = {
            'observation': torch.empty((0, ), device=self.device, dtype=torch.double),
            'action': torch.empty((0,), device=self.device, dtype=torch.long),
            'next_observation': torch.empty((0, ), device=self.device, dtype=torch.double),
            'alpha': torch.empty((0, ), device=self.device, dtype=torch.double)
        }
        self._trajectory_idx = np.array([], dtype=np.int32)
        self._unique_pairs: set = set()
        self._prev_obs: Optional[np.ndarray] = None
        
        self._traj_boundaries: Dict[int, Tuple[int, int]] = {}
        self._current_dataset_idx = 0
        self._trajectory_active = False
        
    
    @property
    def observation(self) -> Dict[str, torch.Tensor]:
        if hasattr(self, '_sampled_data'):
            return self._sampled_data['observation']
        return self.data['observation']
    
    @property
    def action(self) -> Dict[str, torch.Tensor]:
        if hasattr(self, '_sampled_data'):
            return self._sampled_data['action']
        return self.data['action']
    
    @property
    def next_observation(self) -> Dict[str, torch.Tensor]:
        if hasattr(self, '_sampled_data'):
            return self._sampled_data['next_observation']
        return self.data['next_observation']
    
    @property
    def alpha(self) -> Dict[str, torch.Tensor]:
        if hasattr(self, '_sampled_data'):
            return self._sampled_data['alpha']
        return self.data['alpha']
    
    @property
    def size(self) -> int:
        if self.n_subsamples is None:
            if hasattr(self, '_sampled_data'):
                return len(self._sampled_data['next_observation'])
            return len(self.data['next_observation'])
        return self.n_subsamples+1 # dummy transition
    
    @property
    def data_size(self) -> int:
        """Get the size of the internal dataset."""
        return len(self.data['next_observation'])
    
    def add_pairs(self, state, action):
        pair = (np.argmax(state), action)
        self._unique_pairs.add(pair)
        return
    
    @property
    def is_complete(self) -> bool:
        """Check if we have all unique state-action pairs."""
        return self.dataset_type == "unique" and len(self._unique_pairs) == self.expected_size
    
    def add_transition(
        self, 
        time_step
    ):
        """Add a transition to the dataset."""
        if self.dataset_type == "unique":
            self._add_unique(time_step)
        else:
            self._add_all(time_step)
        
        
    def _add_unique(self, time_step):
        """Add only unique (s,a) pairs."""
   
        if time_step.step_type == StepType.FIRST:
            self._prev_obs = time_step.observation
            self._current_dataset_idx += 1
            self._trajectory_active = True
            return
        # Skipping adding data if no active trajectory (e.g., after reset mid-trajectory)
        if not self._trajectory_active:
            return
        
        elif time_step.step_type in  (StepType.MID, StepType.LAST):
            pair = (np.argmax(self._prev_obs), time_step.action)
            
            if pair not in self._unique_pairs:
                self._unique_pairs.add(pair)
                # Convert to tensors and add to dataset
                self.data['observation'] = torch.cat([
                    self.data['observation'],
                    torch.tensor(self._prev_obs, device=self.device, dtype=torch.double).unsqueeze(0)
                ], dim=0)
                self.data['action'] = torch.cat([
                    self.data['action'],
                    torch.tensor([time_step.action], device=self.device, dtype=torch.long).unsqueeze(0)
                ], dim=0)
                self.data['next_observation'] = torch.cat([
                    self.data['next_observation'],
                    torch.tensor(time_step.observation, device=self.device, dtype=torch.double).unsqueeze(0)
                ], dim=0)
                # First unique pair gets alpha=1.0, others get alpha=0.0
                if len(self._unique_pairs) == 1:
                    self.data['alpha'] = torch.cat([
                        self.data['alpha'],
                        torch.tensor([1.0], device=self.device, dtype=torch.double).unsqueeze(0)
                    ], dim=0)
                else:
                    self.data['alpha'] = torch.cat([
                        self.data['alpha'],
                        torch.tensor([0.0], device=self.device, dtype=torch.double).unsqueeze(0)
                    ], dim=0)
                
                # Add trajectory index
                self._trajectory_idx = np.append(self._trajectory_idx, self._current_dataset_idx)
                
                # Update trajectory boundaries
                current_idx = len(self._trajectory_idx) - 1
                if self._current_dataset_idx not in self._traj_boundaries:
                    self._traj_boundaries[self._current_dataset_idx] = (current_idx, current_idx)
                else:
                    start_idx = self._traj_boundaries[self._current_dataset_idx][0]
                    self._traj_boundaries[self._current_dataset_idx] = (start_idx, current_idx)
            
            self._prev_obs = time_step.observation
            # Mark trajectory as inactive if LAST step
            if time_step.step_type == StepType.LAST:
                self._trajectory_active = False
        else:
            raise ValueError("Unknown step type")
    
    def _add_all(self, time_step):
        """Add all transitions."""
        if time_step.step_type == StepType.FIRST:
            self._current_dataset_idx += 1
            self._trajectory_active = True
            current_idx = len(self._trajectory_idx)
            self._traj_boundaries[self._current_dataset_idx] = (current_idx, current_idx)
            
            self.data['observation'] = torch.cat([
                self.data['observation'],
                torch.tensor(time_step.observation, device=self.device, dtype=torch.double).unsqueeze(0)
            ], dim=0)

            if self.data['observation'].shape[0] == 1:
                self.data['alpha'] = torch.cat([
                    self.data['alpha'],
                    torch.tensor([1.0], device=self.device, dtype=torch.double).unsqueeze(0)
                ], dim=0)
            else:
                self.data['alpha'] = torch.cat([
                    self.data['alpha'],
                    torch.tensor([0.0], device=self.device, dtype=torch.double).unsqueeze(0)
                ], dim=0)
            
            self._trajectory_idx = np.append(self._trajectory_idx, self._current_dataset_idx)

        elif time_step.step_type == StepType.MID:
            # Skip adding data if no active trajectory (e.g., after reset mid-trajectory)
            if not self._trajectory_active:
                return
                
            self.data['observation'] = torch.cat([
                self.data['observation'],
                torch.tensor(time_step.observation, device=self.device, dtype=torch.double).unsqueeze(0)
            ], dim=0)
            self.data['action'] = torch.cat([
                self.data['action'],
                torch.tensor([time_step.action], device=self.device, dtype=torch.long)
            ], dim=0)
            self.data['next_observation'] = torch.cat([
                self.data['next_observation'],
                torch.tensor(time_step.observation, device=self.device, dtype=torch.double).unsqueeze(0)
            ], dim=0)
            self.data['alpha'] = torch.cat([
                    self.data['alpha'],
                    torch.tensor([0.0], device=self.device, dtype=torch.double).unsqueeze(0)
                ], dim=0)
            
            self._trajectory_idx = np.append(self._trajectory_idx, self._current_dataset_idx)
            start_idx = self._traj_boundaries[self._current_dataset_idx][0]
            self._traj_boundaries[self._current_dataset_idx] = (start_idx, len(self._trajectory_idx) - 1)
            
        elif time_step.step_type == StepType.LAST:
            # Skip adding data if no active trajectory
            if not self._trajectory_active:
                return
                
            # For LAST step, we only add action and next_observation
            # Do NOT add to _trajectory_idx since there's no new observation to sample from
            self.data['action'] = torch.cat([
                self.data['action'],
                torch.tensor([time_step.action], device=self.device, dtype=torch.long)
            ], dim=0)
            self.data['next_observation'] = torch.cat([
                self.data['next_observation'],
                torch.tensor(time_step.observation, device=self.device, dtype=torch.double).unsqueeze(0)
            ], dim=0)
            # Don't update trajectory boundaries for LAST step
            # The last valid index remains from the previous MID step
            self._trajectory_active = False
        else:
            raise ValueError("Unknown step type")
    
    def add_dummy_transition(self):
        """Add a dummy transition to ensure at least one datapoint."""
        
        if not hasattr(self, '_dummy_transition') or self._dummy_first_state is False:
            self._dummy_first_state = True
            # For image observations, compare using first channel or flattened comparison
            if len(self.obs_shape) > 1:  # Image observations
                # Use first observation as dummy
                indices = torch.tensor([0])
            else:
                eye_tensor = torch.eye(self.n_states, device=self.device, dtype=torch.double)
                indices = torch.where(torch.all(self.data['next_observation'] == eye_tensor[0].unsqueeze(0), axis=1))[0]
                if indices.shape[0] == 0:
                    indices = torch.where(torch.all(self.data['next_observation'] == eye_tensor[1].unsqueeze(0), axis=1))[0]
                    self._dummy_first_state = False
            
            self._dummy_transition = {
                'observation': self.data['observation'][indices[0]].unsqueeze(0),
                'action': self.data['action'][indices[0]].unsqueeze(0),
                'next_observation': self.data['next_observation'][indices[0]].unsqueeze(0),
                'alpha': self.data['alpha'][indices[0]].unsqueeze(0)
            }
        
        self._sampled_data['observation'] = torch.cat([self._dummy_transition['observation'], self._sampled_data['observation']], dim=0)
        self._sampled_data['action'] = torch.cat([self._dummy_transition['action'], self._sampled_data['action']], dim=0)
        self._sampled_data['next_observation'] = torch.cat([self._dummy_transition['next_observation'], self._sampled_data['next_observation']], dim=0)
        self._sampled_data['alpha'] = torch.cat([self._dummy_transition['alpha'], self._sampled_data['alpha']], dim=0)
            
           
    
    def get_data(self) -> Dict[str, torch.Tensor]:
        """Retrieve the internal dataset, optionally with subsampling."""
  
        if self.n_subsamples is None or len(self._trajectory_idx) == 0:
            self._sampled_data = deepcopy(self.data)
            # self.reset()
            return self._sampled_data

        if self.n_subsamples >= len(self._trajectory_idx):
            utils.ColorPrint.yellow(f"Requested subsample size {self.n_subsamples} exceeds dataset size {len(self._trajectory_idx)}. Returning full dataset.")
            self.add_dummy_transition()
            return self.data

        utils.ColorPrint.green(f"Subsampling {self.n_subsamples} datapoints from dataset of size {len(self._trajectory_idx)}.")
        
        # Get unique trajectory IDs (excluding last incomplete trajectory if any)
        unique_traj_ids = list(self._traj_boundaries.keys())[:-1] if len(self._traj_boundaries) > 1 else list(self._traj_boundaries.keys())
        
        if len(unique_traj_ids) == 0:
            return self.data
        # Sample random trajectories for each datapoint (vectorized)
        sampled_traj_ids = np.random.choice(unique_traj_ids, size=self.n_subsamples)
        
        # For each sampled trajectory, sample a random time t
        sampled_indices = np.empty(self.n_subsamples, dtype=np.int32)
    
        print(f"Total dataset size: {len(self.data['observation'])}")
        for i, traj_id in enumerate(sampled_traj_ids):
            start_idx, end_idx = self._traj_boundaries[traj_id]
            traj_len = end_idx - start_idx + 1

            t = 1000000
            # Sample time t within trajectory bounds
            while t >= traj_len:
                n = np.random.random()
                t = np.rint(np.log(1 - n) / np.log(self.gamma) + 1)
            # t = min(int(t), traj_len - 1)
            
            # Get absolute index
            sampled_indices[i] = start_idx + t
            assert sampled_indices[i] < len(self.data['observation']), "Sampled index exceeds trajectory bounds"
        
        # Return subsampled data using advanced indexing (no loop)
        self._sampled_data =  {
            'observation': self.data['observation'][sampled_indices],
            'action': self.data['action'][sampled_indices],
            'next_observation': self.data['next_observation'][sampled_indices],
            'alpha': self.data['alpha'][sampled_indices]
        }
        print("subsample lenght ", len(self._sampled_data['observation']))
        self.add_dummy_transition()
        # reset the dataset
        self.reset()


        return self._sampled_data

class InternalDatasetFIFO:
    """
    FIFO-based internal dataset that maintains only the last N sampling periods.
    Each call to get_data() marks the end of a sampling period and retrieves
    data from the last N periods.
    """
    
    def __init__(self, dataset_type: str, n_states: int, n_actions: int, 
                 gamma: float, window_size: int, n_subsamples: int, 
                 subsampling_strategy: str, dynamic_horizon: bool = False,
                 obs_shape: tuple = None,
                 device: str = 'cpu', data_type=torch.double, first_state = None, second_state = None):
        """
        Args:
            dataset_type: "unique" or "all"
            n_states: Number of states
            n_actions: Number of actions
            gamma: Discount factor for geometric sampling
            window_size: Number of sampling periods to keep in memory
            n_subsamples: Number of samples to return per period (None = all)
            subsampling_strategy: Strategy for subsampling ("random" or "eder")
            dynamic_horizon: Whether to use a dynamic horizon
            obs_shape: Shape of observations (e.g., (84, 84, 3) for images)
            device: Torch device
        """
        self.dataset_type = dataset_type
        self.n_states = n_states
        self.n_actions = n_actions
        self.expected_size = n_states * n_actions
        assert dataset_type in ("unique", "all"), "dataset_type must be 'unique' or 'all'"
        self.n_subsamples = n_subsamples
        self.gamma = gamma
        self.window_size = window_size
        self.device = torch.device(device)
        self.dynamic_horizon = dynamic_horizon
        self.obs_shape = obs_shape if obs_shape is not None else (n_states,)
        self.data_type = data_type
        self.first_state = first_state
        self.second_state = second_state
        
        # FIFO queue: list of sampling periods, each period is a dict of tensors
        self._periods_queue = []
        self._current_period_data = None
        self._current_period_idx = 0
        self._last_period_size = 0  # Track size of last added period
        self.max_log_det = -np.inf
        self.subsampling_strategy = subsampling_strategy
        
        # Track horizons for dynamic horizon mode
        self._horizon_history = []
        self._plot_counter = 0
        
        # Cache for dummy transition (first complete sample)
        self._dummy_cache = None
        
        self.reset()
    
    def reset(self):
        """Reset the FIFO dataset."""
        utils.ColorPrint.yellow("Resetting FIFO internal dataset.")
        self._periods_queue = []
        self._current_period_idx = 0
        self._start_new_period()
    
    def _start_new_period(self):
        """Initialize a new sampling period."""
        self._current_period_data = {
            'observation': torch.empty((0, *self.obs_shape), device=self.device, dtype=self.data_type),
            'action': torch.empty((0,), device=self.device, dtype=torch.long),
            'next_observation': torch.empty((0, *self.obs_shape), device=self.device, dtype=self.data_type),
            'proprio_observation': torch.empty((0, 0), device=self.device, dtype=self.data_type),  # Will be resized on first add
            'alpha': torch.empty((0,), device=self.device, dtype=self.data_type),
        }
        self._trajectory_idx = np.array([], dtype=np.int32)
        self._unique_pairs = set()
        self._prev_obs = None
        self._prev_proprio = None
        self._traj_boundaries = {}
        self._current_dataset_idx = 0
        self._trajectory_active = False
    
    @property
    def data(self) -> Dict[str, torch.Tensor]:
        """
        Property for compatibility with existing code that accesses dataset.data.
        Returns aggregated data from all periods in the FIFO window plus current period.
        """
        if hasattr(self, '_sampled_data'):
            return self._sampled_data
        
        # Aggregate data from all periods in queue
        aggregated = self._aggregate_periods()
        
        
        # Add current period data
        if len(self._current_period_data['next_observation']) > 0:
            return {
                'observation': torch.cat([aggregated['observation'], self._current_period_data['observation']], dim=0),
                'action': torch.cat([aggregated['action'], self._current_period_data['action']], dim=0),
                'next_observation': torch.cat([aggregated['next_observation'], self._current_period_data['next_observation']], dim=0),
                'proprio_observation': torch.cat([aggregated['proprio_observation'], self._current_period_data['proprio_observation']], dim=0) if aggregated['proprio_observation'].shape[0] > 0 else self._current_period_data['proprio_observation'],
                'alpha': torch.cat([aggregated['alpha'], self._current_period_data['alpha']], dim=0)
            }
        else:
            return aggregated
    
    @property
    def current_data_size(self) -> int:
        """Size of current period data (number of transitions)."""
        return self.current_period_data_size
    
    @property
    def last_size(self) -> int:
        """Size of the last period that was added to the queue."""
        return self._last_period_size
    
    @property
    def size(self) -> int:
        """Total number of transitions across all periods in window (excluding current period)."""
        total = sum(len(period['data']['next_observation']) for period in self._periods_queue)
        return total
    
    @property
    def current_period_data_size(self) -> int:
        """Size of current period data."""
        if self._current_period_data is None:
            return 0
        return len(self._current_period_data['next_observation'])
    
    def add_pairs(self, state, action):
        """Track unique state-action pairs (for compatibility with ideal mode)."""
        pair = (np.argmax(state), action)
        self._unique_pairs.add(pair)
        return
    
    @property
    def is_complete(self) -> bool:
        """Check if current period has all unique state-action pairs."""
        return self.dataset_type == "unique" and len(self._unique_pairs) == self.expected_size
    
    @property
    def greater_equal_target_horizon(self) -> bool:
        """Check if current traj exceeds expected horizon size."""
        if not hasattr(self, 'current_target_horizon') or  len(self._traj_boundaries) == 0: #self._current_dataset_idx > len(self._traj_boundaries) or
            return False
        return (self._traj_boundaries[self._current_dataset_idx][1] - self._traj_boundaries[self._current_dataset_idx][0]) >= self.current_target_horizon
    
    @property
    def reset_episode(self) -> bool:
        """Check if current traj exceeds expected horizon size."""
        if not hasattr(self, 'current_target_horizon') or  len(self._traj_boundaries) == 0: #self._current_dataset_idx > len(self._traj_boundaries) or
            return False
        if (self._traj_boundaries[self._current_dataset_idx][1] - self._traj_boundaries[self._current_dataset_idx][0]) >= self.current_target_horizon+1:
            utils.ColorPrint.red("Resetting due to exceeding target horizon")
        return (self._traj_boundaries[self._current_dataset_idx][1] - self._traj_boundaries[self._current_dataset_idx][0]) >= self.current_target_horizon+1
    
    def add_transition(self, time_step):
        """Add a transition to the current sampling period."""
        if self.dataset_type == "unique":
            self._add_unique(time_step)
        else:
            if self.dynamic_horizon== True:
                self._add_dynamic_horizon(time_step)
            else:
                self._add_all(time_step)
    
    def _add_unique(self, time_step):
        """Add only unique (s,a) pairs to current period."""
        if time_step.step_type == StepType.FIRST:
            self._prev_obs = time_step.observation
            self._prev_proprio = getattr(time_step, 'proprio_observation', None)
            self._current_dataset_idx += 1
            self._trajectory_active = True
            return
        
        if not self._trajectory_active:
            return
        
        if time_step.step_type in (StepType.MID, StepType.LAST):
            pair = (np.argmax(self._prev_obs), time_step.action)
            
            if pair not in self._unique_pairs:
                self._unique_pairs.add(pair)
                
                self._current_period_data['observation'] = torch.cat([
                    self._current_period_data['observation'],
                    torch.tensor(self._prev_obs, device=self.device, dtype=self.data_type).unsqueeze(0)
                ], dim=0)
                self._current_period_data['action'] = torch.cat([
                    self._current_period_data['action'],
                    torch.tensor([time_step.action], device=self.device, dtype=torch.long)
                ], dim=0)
                self._current_period_data['next_observation'] = torch.cat([
                    self._current_period_data['next_observation'],
                    torch.tensor(time_step.observation, device=self.device, dtype=self.data_type).unsqueeze(0)
                ], dim=0)
                
                # Add proprio observation if available
                if self._prev_proprio is not None:
                    proprio_tensor = torch.tensor(self._prev_proprio, device=self.device, dtype=self.data_type).unsqueeze(0)
                    if self._current_period_data['proprio_observation'].shape[0] == 0:
                        # Initialize with correct shape
                        self._current_period_data['proprio_observation'] = proprio_tensor
                    else:
                        self._current_period_data['proprio_observation'] = torch.cat([
                            self._current_period_data['proprio_observation'],
                            proprio_tensor
                        ], dim=0)
                
                alpha_val = 1.0 if len(self._unique_pairs) == 1 else 0.0
                self._current_period_data['alpha'] = torch.cat([
                    self._current_period_data['alpha'],
                    torch.tensor([alpha_val], device=self.device, dtype=self.data_type)
                ], dim=0)
                
                self._trajectory_idx = np.append(self._trajectory_idx, self._current_dataset_idx)
                
                current_idx = len(self._trajectory_idx) - 1
                if self._current_dataset_idx not in self._traj_boundaries:
                    self._traj_boundaries[self._current_dataset_idx] = (current_idx, current_idx)
                else:
                    start_idx = self._traj_boundaries[self._current_dataset_idx][0]
                    self._traj_boundaries[self._current_dataset_idx] = (start_idx, current_idx)
                
                # Cache first complete transition for dummy
                self._cache_first_transition()
            
            self._prev_obs = time_step.observation
            self._prev_proprio = getattr(time_step, 'proprio_observation', None)
            if time_step.step_type == StepType.LAST:
                self._trajectory_active = False
    
    def _add_all(self, time_step):
        """Add all transitions to current period."""
        if time_step.step_type == StepType.FIRST:
            self._current_dataset_idx += 1
            self._trajectory_active = True
            current_idx = len(self._trajectory_idx)
            self._traj_boundaries[self._current_dataset_idx] = (current_idx, current_idx)
            
            self._current_period_data['observation'] = torch.cat([
                self._current_period_data['observation'],
                torch.tensor(time_step.observation, device=self.device, dtype=self.data_type).unsqueeze(0)
            ], dim=0)
            
            # Add proprio observation if available
            proprio_obs = getattr(time_step, 'proprio_observation', None)
            if proprio_obs is not None:
                proprio_tensor = torch.tensor(proprio_obs, device=self.device, dtype=self.data_type).unsqueeze(0)
                if self._current_period_data['proprio_observation'].shape[0] == 0:
                    self._current_period_data['proprio_observation'] = proprio_tensor
                else:
                    self._current_period_data['proprio_observation'] = torch.cat([
                        self._current_period_data['proprio_observation'],
                        proprio_tensor
                    ], dim=0)
            
            alpha_val = 1.0 if self._current_period_data['observation'].shape[0] == 1 else 0.0
            self._current_period_data['alpha'] = torch.cat([
                self._current_period_data['alpha'],
                torch.tensor([alpha_val], device=self.device, dtype=self.data_type)
            ], dim=0)
            
            self._trajectory_idx = np.append(self._trajectory_idx, self._current_dataset_idx)
        
        elif time_step.step_type == StepType.MID:
            if not self._trajectory_active:
                return
            
            self._current_period_data['observation'] = torch.cat([
                self._current_period_data['observation'],
                torch.tensor(time_step.observation, device=self.device, dtype=self.data_type).unsqueeze(0)
            ], dim=0)
            self._current_period_data['action'] = torch.cat([
                self._current_period_data['action'],
                torch.tensor([time_step.action], device=self.device, dtype=torch.long)
            ], dim=0)
            self._current_period_data['next_observation'] = torch.cat([
                self._current_period_data['next_observation'],
                torch.tensor(time_step.observation, device=self.device, dtype=self.data_type).unsqueeze(0)
            ], dim=0)
            
            # Add proprio observation if available
            proprio_obs = getattr(time_step, 'proprio_observation', None)
            if proprio_obs is not None:
                proprio_tensor = torch.tensor(proprio_obs, device=self.device, dtype=self.data_type).unsqueeze(0)
                if self._current_period_data['proprio_observation'].shape[0] == 0:
                    self._current_period_data['proprio_observation'] = proprio_tensor
                else:
                    self._current_period_data['proprio_observation'] = torch.cat([
                        self._current_period_data['proprio_observation'],
                        proprio_tensor
                    ], dim=0)
            
            self._current_period_data['alpha'] = torch.cat([
                self._current_period_data['alpha'],
                torch.tensor([0.0], device=self.device, dtype=self.data_type)
            ], dim=0)
            
            self._trajectory_idx = np.append(self._trajectory_idx, self._current_dataset_idx)
            start_idx = self._traj_boundaries[self._current_dataset_idx][0]
            self._traj_boundaries[self._current_dataset_idx] = (start_idx, len(self._trajectory_idx) - 1)
            
            # Cache first complete transition for dummy
            self._cache_first_transition()
  
        elif time_step.step_type == StepType.LAST:
            if not self._trajectory_active:
                return
            
            self._current_period_data['action'] = torch.cat([
                self._current_period_data['action'],
                torch.tensor([time_step.action], device=self.device, dtype=torch.long)
            ], dim=0)
            self._current_period_data['next_observation'] = torch.cat([
                self._current_period_data['next_observation'],
                torch.tensor(time_step.observation, device=self.device, dtype=self.data_type).unsqueeze(0)
            ], dim=0)
            self._trajectory_active = False
    
    def _add_dynamic_horizon(self, time_step):
        """Add all transitions to current period."""
        if time_step.step_type == StepType.FIRST:
            # Horizon Computation

            prob = np.random.rand()
            horizon = np.log(1 - prob) / np.log(self.gamma) - 1
            self.current_target_horizon = int(np.round(horizon))
            
            # Track horizon for plotting
            if self.dynamic_horizon:
                self._horizon_history.append(self.current_target_horizon)
            
            ColorPrint.green(f"New trajectory with target horizon: {self.current_target_horizon}")
            # --------------------------------
            
            self._current_dataset_idx += 1
            self._trajectory_active = True
            current_idx = len(self._trajectory_idx)
            self._traj_boundaries[self._current_dataset_idx] = (current_idx, current_idx)
            
            self._current_period_data['observation'] = torch.cat([
                self._current_period_data['observation'],
                torch.tensor(time_step.observation, device=self.device, dtype=self.data_type).unsqueeze(0)
            ], dim=0)
            
            # Add proprio observation if available
            proprio_obs = getattr(time_step, 'proprio_observation', None)
            if proprio_obs is not None:
                proprio_tensor = torch.tensor(proprio_obs, device=self.device, dtype=self.data_type).unsqueeze(0)
                if self._current_period_data['proprio_observation'].shape[0] == 0:
                    self._current_period_data['proprio_observation'] = proprio_tensor
                else:
                    self._current_period_data['proprio_observation'] = torch.cat([
                        self._current_period_data['proprio_observation'],
                        proprio_tensor
                    ], dim=0)
            
            alpha_val = 1.0 if self._current_period_data['observation'].shape[0] == 1 else 0.0
            self._current_period_data['alpha'] = torch.cat([
                self._current_period_data['alpha'],
                torch.tensor([alpha_val], device=self.device, dtype=self.data_type)
            ], dim=0)
            
            self._trajectory_idx = np.append(self._trajectory_idx, self._current_dataset_idx)
        
        elif time_step.step_type == StepType.MID:
            if not self._trajectory_active:
                return
            
            if not self.reset_episode:
                self._current_period_data['observation'] = torch.cat([
                    self._current_period_data['observation'],
                    torch.tensor(time_step.observation, device=self.device, dtype=self.data_type).unsqueeze(0)
                ], dim=0)
                
                # Add proprio observation if available
                proprio_obs = getattr(time_step, 'proprio_observation', None)
                if proprio_obs is not None:
                    proprio_tensor = torch.tensor(proprio_obs, device=self.device, dtype=self.data_type).unsqueeze(0)
                    if self._current_period_data['proprio_observation'].shape[0] == 0:
                        self._current_period_data['proprio_observation'] = proprio_tensor
                    else:
                        self._current_period_data['proprio_observation'] = torch.cat([
                            self._current_period_data['proprio_observation'],
                            proprio_tensor
                        ], dim=0)
                
                self._current_period_data['alpha'] = torch.cat([
                    self._current_period_data['alpha'],
                    torch.tensor([0.0], device=self.device, dtype=self.data_type)
                ], dim=0)
            else:
                self._trajectory_active = False
            
            self._current_period_data['action'] = torch.cat([
                self._current_period_data['action'],
                torch.tensor([time_step.action], device=self.device, dtype=torch.long)
            ], dim=0)
            self._current_period_data['next_observation'] = torch.cat([
                self._current_period_data['next_observation'],
                torch.tensor(time_step.observation, device=self.device, dtype=self.data_type).unsqueeze(0)
            ], dim=0)
            
            self._trajectory_idx = np.append(self._trajectory_idx, self._current_dataset_idx)
            start_idx = self._traj_boundaries[self._current_dataset_idx][0]
            self._traj_boundaries[self._current_dataset_idx] = (start_idx, len(self._trajectory_idx) - 1)
            
            # Cache first complete transition for dummy
            self._cache_first_transition()
  
        elif time_step.step_type == StepType.LAST:
            if not self._trajectory_active:
                return
    
            self._current_period_data['action'] = torch.cat([
                self._current_period_data['action'],
                torch.tensor([time_step.action], device=self.device, dtype=torch.long)
            ], dim=0)
            self._current_period_data['next_observation'] = torch.cat([
                self._current_period_data['next_observation'],
                torch.tensor(time_step.observation, device=self.device, dtype=self.data_type).unsqueeze(0)
            ], dim=0)
            self._trajectory_active = False
    
    def _cache_first_transition(self):
        """Cache the first complete transition for use as dummy transition."""
        if self._dummy_cache is not None:
            return  # Already cached
        
        # Check if we have at least one complete transition
        if (len(self._current_period_data['observation']) > 0 and
            len(self._current_period_data['action']) > 0 and
            len(self._current_period_data['next_observation']) > 0):
            
            indices = torch.where(torch.all(self._current_period_data['next_observation'] == self.first_state, dim=1))[0]
            if indices.shape[0] == 0:
                return
            # indices = indices[0]

            self._dummy_cache = {
                'observation': self._current_period_data['observation'][indices:indices+1].clone(),
                'action': self._current_period_data['action'][indices:indices+1].clone(),
                'next_observation': self._current_period_data['next_observation'][indices:indices+1].clone(),
                'alpha': self._current_period_data['alpha'][indices:indices+1].clone()
            }
            
            # Add proprio if available
            if self._current_period_data['proprio_observation'].shape[0] > 0:
                self._dummy_cache['proprio_observation'] = self._current_period_data['proprio_observation'][indices:indices+1].clone()
            else:
                self._dummy_cache['proprio_observation'] = torch.empty((1, 0), device=self.device, dtype=self.data_type)
            
            utils.ColorPrint.green("Cached first transition for dummy use")
    
    def add_dummy_transition(self):
        """Add a dummy transition using the cached first sample."""
        if self._dummy_cache is None:
            utils.ColorPrint.yellow("No dummy cache available, skipping dummy transition")
            return
        
        self._current_period_data['observation'] = torch.cat([
            self._dummy_cache['observation'], 
            self._current_period_data['observation']
        ], dim=0)
        self._current_period_data['action'] = torch.cat([
            self._dummy_cache['action'], 
            self._current_period_data['action']
        ], dim=0)
        self._current_period_data['next_observation'] = torch.cat([
            self._dummy_cache['next_observation'], 
            self._current_period_data['next_observation']
        ], dim=0)
        self._current_period_data['alpha'] = torch.cat([
            self._dummy_cache['alpha'], 
            self._current_period_data['alpha']
        ], dim=0)
        
        # Add proprio if available
        if self._dummy_cache['proprio_observation'].shape[0] > 0:
            if 'proprio_observation' not in self._current_period_data or self._current_period_data['proprio_observation'].shape[0] == 0:
                self._current_period_data['proprio_observation'] = self._dummy_cache['proprio_observation']
            else:
                self._current_period_data['proprio_observation'] = torch.cat([
                    self._dummy_cache['proprio_observation'],
                    self._current_period_data['proprio_observation']
                ], dim=0)
    
    def _aggregate_periods(self) -> Dict[str, torch.Tensor]:
        """Concatenate data from all periods in the window."""
        if len(self._periods_queue) == 0:
            return {
                'observation': torch.empty((0, *self.obs_shape), device=self.device, dtype=self.data_type),
                'action': torch.empty((0,), device=self.device, dtype=torch.long),
                'next_observation': torch.empty((0, *self.obs_shape), device=self.device, dtype=self.data_type),
                'proprio_observation': torch.empty((0, 0), device=self.device, dtype=self.data_type),
                'alpha': torch.empty((0,), device=self.device, dtype=self.data_type)
            }
        
        all_obs = []
        all_actions = []
        all_next_obs = []
        all_proprio = []
        all_alpha = []
        
        for period in self._periods_queue:
            data = period['data']
            if len(data['next_observation']) > 0:
                all_obs.append(data['observation'])
                all_actions.append(data['action'])
                all_next_obs.append(data['next_observation'])
                if data['proprio_observation'].shape[0] > 0:
                    all_proprio.append(data['proprio_observation'])
                all_alpha.append(data['alpha'])
        
        if len(all_obs) == 0:
            return {
                'observation': torch.empty((0, *self.obs_shape), device=self.device, dtype=self.data_type),
                'action': torch.empty((0,), device=self.device, dtype=torch.long),
                'next_observation': torch.empty((0, *self.obs_shape), device=self.device, dtype=self.data_type),
                'proprio_observation': torch.empty((0, 0), device=self.device, dtype=self.data_type),
                'alpha': torch.empty((0,), device=self.device, dtype=self.data_type)
            }
        
        return {
            'observation': torch.cat(all_obs, dim=0),
            'action': torch.cat(all_actions, dim=0),
            'next_observation': torch.cat(all_next_obs, dim=0),
            'proprio_observation': torch.cat(all_proprio, dim=0) if len(all_proprio) > 0 else torch.empty((0, 0), device=self.device, dtype=self.data_type),
            'alpha': torch.cat(all_alpha, dim=0)
        }
    
    def get_data(self, unique=False) -> Dict[str, torch.Tensor]:
        """
        End current sampling period, clean incomplete trajectories, add to FIFO queue, 
        maintain window size, and return aggregated data from last N periods.
        
        Returns:
            Dictionary with concatenated data from all periods in window
        """
        # Plot horizon histogram before resetting
        if self.dynamic_horizon:
            self._plot_horizon_histogram()
        else:
            # Clean incomplete trajectories from current period
            self._clean_incomplete_trajectories()
        
        self.add_dummy_transition()
        # Store current period data with metadata
        period_entry = {
            'data': deepcopy(self._current_period_data),
            'trajectory_idx': self._trajectory_idx.copy(),
            'traj_boundaries': deepcopy(self._traj_boundaries),
            'period_idx': self._current_period_idx
        }
        
        # Track the size of this period
        self._last_period_size = len(self._current_period_data['next_observation'])
        
        # Add to queue
        self._periods_queue.append(period_entry)
        utils.ColorPrint.green(f"Completed sampling period {self._current_period_idx} with {self._last_period_size} transitions.")
        
        # Maintain FIFO: remove oldest if exceeds window size
        if len(self._periods_queue) > self.window_size:
            removed = self._periods_queue.pop(0)
            utils.ColorPrint.yellow(f"Removed oldest period {removed['period_idx']} from FIFO queue.")
        
        # Aggregate data from all periods in window
        aggregated_data = self._aggregate_periods()
        
        # Start new period
        self._current_period_idx += 1
        self._start_new_period()
        
        # Reset horizon history after plotting
        if self.dynamic_horizon:
            self._horizon_history = []
        
        # Filter for unique state-action pairs if requested
        if unique and len(aggregated_data['next_observation']) > 0:
            aggregated_data = self._filter_unique_state_action_pairs(aggregated_data)
        
        # Apply subsampling if needed
        if self.n_subsamples is not None and len(aggregated_data['next_observation']) > 0:
            assert unique is False, "Subsampling with unique state-action pairs is not supported."
            if self.subsampling_strategy == "random":
                aggregated_data = self._subsample_data(aggregated_data)
            elif self.subsampling_strategy == "eder":
                aggregated_data = self._eder_subsampling(aggregated_data)
        
        self._sampled_data = aggregated_data
        
        return aggregated_data

    
    def _filter_unique_state_action_pairs(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Filter data to keep only unique state-action pairs.
        For duplicate (s,a) pairs, keeps the first occurrence.
        
        Args:
            data: Dictionary with observation, action, next_observation, alpha tensors
            
        Returns:
            Filtered dictionary with only unique (s,a) pairs
        """
        if len(data['observation']) == 0:
            return data
        
        # Convert observations to state indices (assuming one-hot encoding)
        state_indices = torch.argmax(data['observation'], dim=1).cpu().numpy()
        action_indices = data['action'].cpu().numpy()
        
        # Track unique (state, action) pairs and their first occurrence indices
        seen_pairs = set()
        unique_indices = []
        
        for idx in range(len(state_indices)):
            pair = (int(state_indices[idx]), int(action_indices[idx]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique_indices.append(idx)
        
        utils.ColorPrint.green(f"Filtered {len(state_indices)} transitions to {len(unique_indices)} unique state-action pairs.")
        
        # Return filtered data
        return {
            'observation': data['observation'][unique_indices],
            'action': data['action'][unique_indices],
            'next_observation': data['next_observation'][unique_indices],
            'alpha': data['alpha'][unique_indices],
            'proprio_observation': data['proprio_observation'][unique_indices] if 'proprio_observation' in data else torch.empty((0, 0), device=self.device, dtype=self.data_type)
        }
    
    def _clean_incomplete_trajectories(self):
        """Remove data from incomplete trajectories in the current period."""
        if len(self._traj_boundaries) == 0:
            return
        
        # Find incomplete trajectories (where trajectory is still active)
        incomplete_traj_ids = []
        for traj_id in self._traj_boundaries.keys():
            # Check if this trajectory is still active (not properly terminated)
            if traj_id == self._current_dataset_idx and self._trajectory_active:
                incomplete_traj_ids.append(traj_id)
        
        if len(incomplete_traj_ids) == 0:
            return
        
        utils.ColorPrint.yellow(f"Cleaning {len(incomplete_traj_ids)} incomplete trajectory/trajectories from current period.")
        
        # Find indices to keep (all indices not belonging to incomplete trajectories)
        indices_to_keep = []
        for idx, traj_id in enumerate(self._trajectory_idx):
            if traj_id not in incomplete_traj_ids:
                indices_to_keep.append(idx)
        
        if len(indices_to_keep) == 0:
            # All data was from incomplete trajectories, reset current period
            utils.ColorPrint.yellow("All data in current period was from incomplete trajectories. Resetting period.")
            self._start_new_period()
            return
        
        # Filter data to keep only complete trajectories
        self._current_period_data['observation'] = self._current_period_data['observation'][indices_to_keep]
        self._current_period_data['alpha'] = self._current_period_data['alpha'][indices_to_keep]
        
        # For action and next_observation, we need to handle the offset
        # These arrays may have one less element than observation
        action_indices = [i for i in indices_to_keep if i < len(self._current_period_data['action'])]
        self._current_period_data['action'] = self._current_period_data['action'][action_indices]
        self._current_period_data['next_observation'] = self._current_period_data['next_observation'][action_indices]
        
        # Update trajectory_idx
        self._trajectory_idx = self._trajectory_idx[indices_to_keep]
        
        # Remove incomplete trajectories from boundaries
        for traj_id in incomplete_traj_ids:
            del self._traj_boundaries[traj_id]
        
        # Rebuild trajectory boundaries with new indices
        new_boundaries = {}
        for traj_id in self._traj_boundaries.keys():
            # Find min and max indices for this trajectory in the filtered data
            traj_indices = [i for i, tid in enumerate(self._trajectory_idx) if tid == traj_id]
            if len(traj_indices) > 0:
                new_boundaries[traj_id] = (min(traj_indices), max(traj_indices))
        
        self._traj_boundaries = new_boundaries
        utils.ColorPrint.green(f"Cleaned period now has {len(self._current_period_data['observation'])} observations.")
    
    def _subsample_data(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Subsample data using geometric distribution based on gamma."""
        total_size = len(data['next_observation'])
        
        if self.n_subsamples >= total_size:
            utils.ColorPrint.yellow(f"Requested subsample size {self.n_subsamples} >= total size {total_size}. Returning full data.")
            return data
        
        utils.ColorPrint.green(f"Subsampling {self.n_subsamples} from {total_size} transitions.")
        
        # Random subsampling (can be enhanced with trajectory-aware sampling)
        indices = np.random.choice(total_size, size=self.n_subsamples, replace=False)
        
        return {
            'observation': data['observation'][indices],
            'action': data['action'][indices],
            'next_observation': data['next_observation'][indices],
            'alpha': data['alpha'][indices],
            'proprio_observation': data['proprio_observation'][indices] if 'proprio_observation' in data else torch.emp
        }

    def spd_logdet_cholesky(self, K, jitter=1e-6):
        # K: (..., n, n), symmetric PSD/SPD kernel submatrix
        # K = 0.5 * (K + K.transpose(-1, -2))
        n = self.n_subsamples
        I = torch.eye(n, device=K.device, dtype=K.dtype)

        L, info = torch.linalg.cholesky_ex(K + jitter * I, upper=False, check_errors=False)
        if torch.any(info != 0):
            raise RuntimeError("Cholesky failed; increase jitter or check kernel definiteness.")
        d = L.diagonal(dim1=-2, dim2=-1)
        return 2.0 * d.log().sum(dim=-1)

    def _eder_subsampling(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Subsample data using EDER method (not implemented)."""
        
        total_size = len(data['next_observation'])
        
        
        if self.n_subsamples >= total_size:
            utils.ColorPrint.yellow(f"Requested subsample size {self.n_subsamples} >= total size {total_size}. Returning full data.")
            return data
        
        utils.ColorPrint.green(f"Subsampling {self.n_subsamples} from {total_size} transitions.")
        
        starting_search_time = time()

        tmp_max = -np.inf

        for i in range(100):
            # Random subsampling (can be enhanced with trajectory-aware sampling)
            indices = np.random.choice(total_size, size=self.n_subsamples, replace=False)
            
            sampled_data = {
                'observation': data['observation'][indices],
                'action': data['action'][indices],
                'next_observation': data['next_observation'][indices],
                'proprio_observation': data['proprio_observation'][indices] if 'proprio_observation' in data else torch.empty((self.n_subsamples, 0), device=self.device, dtype=self.data_type),
                'alpha': data['alpha'][indices]
            }

            # Encoding state-action pairs as unique integers
            action_onehot = F.one_hot(sampled_data['action'].long(), self.n_actions).reshape(-1, self.n_actions)  # [B, |A|]
            
            # Outer product: [B, d] ⊗ [B, |A|] -> [B, d*|A|]
            encoded_sa = torch.einsum('bd,ba->bda', sampled_data['observation'], action_onehot).reshape(self.n_subsamples, -1)

            kernel_sa = encoded_sa@encoded_sa.T # [B, B]

            log_det =self.spd_logdet_cholesky(kernel_sa)

            # if np.random.rand() <= np.exp(log_det - self.max_log_det):
            # if log_det > self.max_log_det:

            #     if self.max_log_det == -np.inf:
            #         ColorPrint.green(f"EDER subsampling with log-det: {log_det.item():.4f}  in {time() - starting_search_time:.2f}s accepted after {i+1} attempts")
            #     else:
            #         ColorPrint.green(f"EDER subsampling with log-det: {log_det.item():.4f} (max: {self.max_log_det:.4f}) in {time() - starting_search_time:.2f}s accepted after {i+1} attempts")
            #     if log_det > self.max_log_det:
            #         self.max_log_det = log_det.item()
            #     return sampled_data
            
            if log_det > tmp_max:
                tmp_max = log_det.item()
                best_sampled_data = sampled_data


            if log_det > self.max_log_det:
                self.max_log_det = log_det.item()
        ColorPrint.red(f"EDER subsampling failed to find better subset; returning best sampled subset, with log-det: {tmp_max:.4f} (max: {self.max_log_det:.4f}) in {time() - starting_search_time:.2f}s")
        return best_sampled_data
    
    def _plot_horizon_histogram(self):
        """Plot histogram of dynamic horizons and save to file."""
        if not self.dynamic_horizon or len(self._horizon_history) == 0:
            return
        
        save_dir = os.path.join(os.getcwd(), "horizon_plots")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"horizon_histogram_period_{self._plot_counter}.png")
        
        # Get range of horizons
        min_horizon = min(self._horizon_history)
        max_horizon = max(self._horizon_history)
        
        # Create bins that include all integer values from min to max
        bins = np.arange(min_horizon - 0.5, max_horizon + 1.5, 1)
        
        plt.figure(figsize=(10, 6))
        plt.hist(self._horizon_history, bins=bins, edgecolor='black', alpha=0.7, align='mid')
        
        # Set x-axis ticks to show each integer horizon value
        plt.xticks(range(min_horizon, max_horizon + 1))
        
        plt.xlabel('Target Horizon')
        plt.ylabel('Frequency')
        plt.title(f'Dynamic Horizon Distribution - Period {self._plot_counter}\n'
                  f'Total trajectories: {len(self._horizon_history)}, '
                  f'Mean: {np.mean(self._horizon_history):.2f}, '
                  f'Std: {np.std(self._horizon_history):.2f}')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        ColorPrint.green(f"Saved horizon histogram to {save_path}")
        
        # Increment counter for next plot
        self._plot_counter += 1

