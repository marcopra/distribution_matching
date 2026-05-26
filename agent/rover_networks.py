from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class Encoder(nn.Module):
    def __init__(self, obs_shape, hidden_dim, feature_dim):
        super(Encoder, self).__init__()
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.repr_dim = feature_dim
        self.temperature = 0.05


        self.fc =  nn.Sequential(
            nn.Linear(obs_shape[0], hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim, bias=False),
            nn.ReLU()
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs.view(obs.shape[0], -1)
        obs = obs.to(dtype=torch.get_default_dtype())
        h = self.fc(obs)
        h = F.normalize(h, p=1, dim=-1)
        return h
    
    def encode_and_project(self, obs):
        return self.forward(obs)

class CNNEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, mode='l2'):
        super().__init__()

        assert len(obs_shape) == 3
        assert mode in ['l1', 'l2'], "Mode must be 'l1' or 'l2'"
        self.mode = mode

        self.conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.ReLU(), 
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(), 
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(), 
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU()
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.repr_dim = 32 * 7 * 7  # 1,568 features

      
        
        self.projector = nn.Sequential(
            nn.Linear(self.repr_dim, feature_dim),  # Project to 256 dimensions
            nn.LayerNorm(feature_dim),
            # nn.Tanh()
            nn.ReLU(inplace=True)
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.
        obs = obs.to(dtype=torch.get_default_dtype())
        h = self.conv(obs)
        h = self.adaptive_pool(h)
        h = h.view(h.shape[0], -1)
        return h

    def encode_and_project(self, obs):
        h = self.forward(obs)
        z = self.projector(h)
        if self.mode == 'l2':   
            z =F.normalize(z, p=2, dim=-1)
        elif self.mode == 'l1':
            z =F.normalize(z, p=1, dim=-1)
        return z
    
class ProjectSA(nn.Module):
    """ Projects state-action embeddings to state embeddings. """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, linear=False):
        super().__init__()
        if linear:
            self.project_sa = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.project_sa= nn.Sequential(
                nn.Linear(input_dim, hidden_dim, bias=False),
                # nn.ReLU(inplace=True),
                nn.SiLU(inplace=True),
                nn.Linear(hidden_dim, output_dim, bias=False),
                # nn.Linear(input_dim, output_dim, bias=False)

            )
    
    def forward(self, encoded_state_action: torch.Tensor) -> torch.Tensor:
        return self.project_sa(encoded_state_action)
