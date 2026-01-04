import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from utils import ColorPrint
from typing import Tuple, Optional, Dict
from dm_env import StepType, specs
from copy import deepcopy

import numpy as np

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
    Kernel-based actor that computes: π(a|s) = softmax(-η · H^T · C)
    where H = φ(s) @ φ_dataset^T
    
    This architecture allows loading pretrained kernel weights and
    optionally finetuning them with RL algorithms.
    """
    
    def __init__(self, obs_type, input_dim, dataset_dim, action_dim, eta, trainable=True):
        """
        Args:
            obs_type: Type of observation ('states' or 'pixels')
            dataset_dim: Number of dataset examples (n)
            action_dim: Number of actions
            feature_dim: Dimension of feature space (d)
            eta: Scalar scaling factor (learning rate)
            trainable: If True, allows weights to be updated during finetuning
        """
        super().__init__()
        
        # Layer 1: Kernel layer computes H = φ(x) @ φ_dataset^T
        # Linear layer computes y = x @ W^T
        # We want H = φ(x) @ φ_dataset^T
        # Therefore W = φ_dataset
        self.kernel_layer = nn.Linear(input_dim, dataset_dim, bias=False)
        
        # Layer 2: Computes logits = H @ C
        # We want logits = H @ C
        # Linear computes z = H @ W^T, so W = C^T
        self.grad_coefficient = nn.Linear(dataset_dim, action_dim, bias=False)
        
        self.eta = nn.Parameter(torch.tensor(eta), requires_grad=trainable)
        self.softmax = nn.Softmax(dim=1)
        
       
        # Control whether weights are trainable
        if not trainable:
            for param in self.parameters():
                param.requires_grad = False
        
        self.apply(utils.weight_init)

    def initialize_from_pretrained(self, phi_dataset, gradient_coeff, eta):
        """
        Initialize weights from pretrained kernel policy.
        
        Args:
            phi_dataset: [num_unique, feature_dim] - dataset feature matrix
            gradient_coeff: [num_unique, n_actions] - learned coefficients
            eta: scalar - learning rate / temperature
        """
        # kernel_layer.weight shape: [dataset_dim, feature_dim]
        # We want: H = φ(x) @ φ_dataset^T = φ(x) @ W^T
        # So W = φ_dataset
        self.kernel_layer.weight.data.copy_(phi_dataset)
        
        # grad_coefficient.weight shape: [n_actions, dataset_dim]
        # We want: logits = H @ C = H @ W^T
        # So W^T = C, hence W = C^T
        self.grad_coefficient.weight.data.copy_(gradient_coeff.T)
        
        # Set eta
        self.eta.data.copy_(torch.tensor(eta))
        
        print(f"Kernel actor initialized from pretrained weights:")
        print(f"  - Kernel layer: {self.kernel_layer.weight.shape}")
        print(f"  - Grad coeff: {self.grad_coefficient.weight.shape}")
        print(f"  - Eta: {self.eta.item()}")

    def forward(self, phi_x):
        """
        Forward pass: π(a|s) = softmax(-η · φ(s)^T φ_dataset^T C)
        
        Args:
            phi_x: [batch_size, feature_dim] - encoded observations
            
        Returns:
            probs: [batch_size, n_actions] - action probabilities
        """
        # 1. Compute kernel similarities: H = φ(x) @ φ_dataset^T
        # Shape: [batch_size, dataset_dim]
        h = self.kernel_layer(phi_x)
        
        # 2. Apply gradient coefficients: logits = H @ C
        # Shape: [batch_size, n_actions]
        logits = self.grad_coefficient(h)
        
        # 3. Scale by -eta and apply softmax
        # Note: negative sign because we minimize in the original formulation
        probs = self.softmax(-self.eta * logits)
        
        return probs

    def _logits(self, phi_x):
        """Get logits without softmax (useful for some RL algorithms)."""
        h = self.kernel_layer(phi_x)
        logits = self.grad_coefficient(h)
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
        self._current_traj_idx = 0
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
            self._current_traj_idx += 1
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
                self._trajectory_idx = np.append(self._trajectory_idx, self._current_traj_idx)
                
                # Update trajectory boundaries
                current_idx = len(self._trajectory_idx) - 1
                if self._current_traj_idx not in self._traj_boundaries:
                    self._traj_boundaries[self._current_traj_idx] = (current_idx, current_idx)
                else:
                    start_idx = self._traj_boundaries[self._current_traj_idx][0]
                    self._traj_boundaries[self._current_traj_idx] = (start_idx, current_idx)
            
            self._prev_obs = time_step.observation
            # Mark trajectory as inactive if LAST step
            if time_step.step_type == StepType.LAST:
                self._trajectory_active = False
        else:
            raise ValueError("Unknown step type")
    
    def _add_all(self, time_step):
        """Add all transitions."""
        if time_step.step_type == StepType.FIRST:
            self._current_traj_idx += 1
            self._trajectory_active = True
            current_idx = len(self._trajectory_idx)
            self._traj_boundaries[self._current_traj_idx] = (current_idx, current_idx)
            
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
            
            self._trajectory_idx = np.append(self._trajectory_idx, self._current_traj_idx)

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
            
            self._trajectory_idx = np.append(self._trajectory_idx, self._current_traj_idx)
            start_idx = self._traj_boundaries[self._current_traj_idx][0]
            self._traj_boundaries[self._current_traj_idx] = (start_idx, len(self._trajectory_idx) - 1)
            
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
            eye_tensor = torch.eye(self.n_states, device=self.device, dtype=torch.double)
            indices = torch.where(torch.all(self.data['next_observation'] == eye_tensor[0].unsqueeze(0), axis=1))[0]
            if indices.shape[0] == 0:
                indices = torch.where(torch.all(self.data['next_observation'] == eye_tensor[1].unsqueeze(0), axis=1))[0]
                self._dummy_first_state = False
            # TODO at the moment using second state for alpha not the real first, change this in the future
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