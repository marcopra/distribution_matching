import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from utils import ColorPrint
from contextlib import contextmanager
from typing import Tuple, Optional, Dict
from dm_env import StepType, specs
from copy import deepcopy

import numpy as np
from time import time
import os
import matplotlib
matplotlib.use('Agg')  # Backend non-interattivo per salvare senza display
import matplotlib.pyplot as plt
from PIL import Image

# torch.set_default_tensor_type(torch.FloatTensor)
float_type = torch.float32


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        # self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())
        
        # compute representation dimension after conv layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            dummy_output = self.convnet(dummy_input)
            self.repr_dim = dummy_output.view(1, -1).shape[1]

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
    
    SUPPORTED_KERNELS = ("inner_product", "gaussian")

    def __init__(
        self,
        obs_type,
        input_dim,
        dataset_dim,
        action_dim,
        eta,
        trainable=True,
        kernel_type="inner_product",
        kernel_bandwidth=None,
    ):
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
        self.obs_type = obs_type
        self.input_dim = int(input_dim)
        self.dataset_dim = int(dataset_dim)
        self.action_dim = int(action_dim)
        self.kernel_type = str(kernel_type or "inner_product").strip().lower()
        if self.kernel_type not in self.SUPPORTED_KERNELS:
            raise ValueError(
                f"Unsupported kernel_type={self.kernel_type!r}; "
                f"expected one of {self.SUPPORTED_KERNELS}"
            )
        
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
        if self.kernel_type == "gaussian":
            if kernel_bandwidth is None or float(kernel_bandwidth) <= 0.0:
                raise ValueError("Gaussian kernel actor requires a positive kernel_bandwidth")
            self.log_bandwidth = nn.Parameter(
                torch.tensor(np.log(float(kernel_bandwidth)), dtype=float_type),
                requires_grad=trainable,
            )
        else:
            self.register_parameter("log_bandwidth", None)
        self.softmax = nn.Softmax(dim=1)
        
        # Control whether weights are trainable
        if not trainable:
            for param in self.parameters():
                param.requires_grad = False
        
        self.apply(utils.weight_init)

    @property
    def kernel_bandwidth(self):
        if self.log_bandwidth is None:
            return None
        return torch.exp(self.log_bandwidth)

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
        expected_phi_shape = (self.dataset_dim, self.input_dim + 1)
        if tuple(phi_dataset.shape) != expected_phi_shape:
            raise ValueError(
                f"phi_dataset shape {tuple(phi_dataset.shape)} does not match "
                f"actor shape {expected_phi_shape}"
            )
        if E is None or tuple(E.shape) != (self.dataset_dim, self.action_dim):
            raise ValueError(
                f"E shape {None if E is None else tuple(E.shape)} does not match "
                f"{(self.dataset_dim, self.action_dim)}"
            )
        if gradient_coeff.shape[0] != self.dataset_dim + 1:
            raise ValueError(
                f"gradient_coeff has {gradient_coeff.shape[0]} rows; "
                f"expected {self.dataset_dim + 1}"
            )
        self.kernel_layer.weight.data.copy_(
            phi_dataset.to(device=self.kernel_layer.weight.device, dtype=self.kernel_layer.weight.dtype)
        )
        
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
        self.action_coeffs.weight.data.copy_(
            weighted_E.T.to(
                device=self.action_coeffs.weight.device,
                dtype=self.action_coeffs.weight.dtype,
            )
        )
    
        
        # 4. Initialize bias term
        self.bias_coeff.data.fill_(bias_grad)
        
        # 5. Set eta
        self.eta.data.fill_(float(eta))
        print("all dtypes:", self.kernel_layer.weight.dtype, self.action_coeffs.weight.dtype, self.bias_coeff.dtype, self.eta.dtype)
        print(f"Kernel actor initialized from pretrained weights:")
        print(f"  - Kernel layer: {self.kernel_layer.weight.shape}")
        print(f"  - Action coeffs: {self.action_coeffs.weight.shape}")
        print(f"  - Bias: {self.bias_coeff.shape}")
        print(f"  - Eta: {self.eta.item()}")
        print(f"  - Kernel: {self.kernel_type}")
        if self.kernel_bandwidth is not None:
            print(f"  - Bandwidth: {self.kernel_bandwidth.item()}")

    def _kernel_features(self, phi_x_aug):
        landmarks = self.kernel_layer.weight
        if self.kernel_type == "inner_product":
            return F.linear(phi_x_aug, landmarks)

        x_sq = torch.sum(phi_x_aug * phi_x_aug, dim=1, keepdim=True)
        y_sq = torch.sum(landmarks * landmarks, dim=1).unsqueeze(0)
        squared_distance = torch.clamp(
            x_sq + y_sq - 2.0 * (phi_x_aug @ landmarks.T),
            min=0.0,
        )
        bandwidth = self.kernel_bandwidth.clamp_min(1e-12)
        return torch.exp(-squared_distance / (2.0 * bandwidth * bandwidth))

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
        h = self._kernel_features(phi_x_aug)
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
        phi_x_aug = phi_x_aug.to(dtype=float_type)
        h = self._kernel_features(phi_x_aug)
        
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
