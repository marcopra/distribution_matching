from collections import OrderedDict
import copy
from ctypes.wintypes import PSIZE
from dis import disco, show_code
import os
import hydra
import numpy as np
import numpy.ma as ma
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Tuple, Optional, Dict
from dm_env import StepType, specs
from scipy.special import softmax
from scipy.linalg import cho_factor, cho_solve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import utils
from distribution_matching import DistributionVisualizer
from PIL import Image
from sklearn.manifold import TSNE
import seaborn as sns
import logging
from agent.utils_debug_visualization import build_debug_visualizer_suite
# set logging level to info
import logging


def _resolve_torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    dtype = str(dtype).lower()
    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "32": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
        "double": torch.float64,
        "64": torch.float64,
    }
    if dtype not in dtype_map:
        raise ValueError("compute_dtype must be one of: float32, fp32, float64, fp64, double")
    return dtype_map[dtype]


torch.set_default_dtype(_resolve_torch_dtype(os.environ.get("ROVER_COMPUTE_DTYPE", "float32")))

logger = logging.getLogger("myapp")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)

logger.addHandler(handler)
# ============================================================================
# Neural Network Components
# ============================================================================
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


class EncodedTransitionFIFO:
    """FIFO storage for actor-ready encoded transitions.

    The first transition is pinned separately so actor batches can always place
    it at index 0 without duplicating it in the random sample.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("encoded FIFO capacity must be positive")
        self.capacity = int(capacity)
        self._data = None
        self._ids = None
        self._first = None
        self._first_id = None

    def __len__(self):
        size = 0 if self._ids is None else int(self._ids.numel())
        return size + (1 if self._first is not None else 0)

    @property
    def has_first(self):
        return self._first is not None

    @property
    def data_count(self):
        return 0 if self._ids is None else int(self._ids.numel())

    @staticmethod
    def _index(encoded, index):
        return {key: value[index] for key, value in encoded.items()}

    @staticmethod
    def _cat(encoded_batches):
        keys = encoded_batches[0].keys()
        return {
            key: torch.cat([batch[key] for batch in encoded_batches], dim=0)
            for key in keys
        }

    def add(self, transition_ids, encoded):
        transition_ids = torch.as_tensor(transition_ids, dtype=torch.long, device='cpu')
        encoded = {
            key: value.detach().to('cpu')
            for key, value in encoded.items()
        }

        first_mask = transition_ids == 0
        if first_mask.any():
            first_idx = int(torch.nonzero(first_mask, as_tuple=False)[0].item())
            self._first = self._index(encoded, slice(first_idx, first_idx + 1))
            self._first_id = int(transition_ids[first_idx].item())

        if self._first_id is None:
            keep_mask = torch.ones_like(transition_ids, dtype=torch.bool)
        else:
            keep_mask = transition_ids != self._first_id
        if not keep_mask.any():
            return

        new_ids = transition_ids[keep_mask]
        new_data = self._index(encoded, keep_mask)
        if self._data is None:
            self._ids = new_ids
            self._data = new_data
        else:
            self._ids = torch.cat([self._ids, new_ids], dim=0)
            self._data = self._cat([self._data, new_data])

        overflow = int(self._ids.numel()) - max(0, self.capacity - (1 if self._first is not None else 0))
        if overflow > 0:
            self._ids = self._ids[overflow:]
            self._data = self._index(self._data, slice(overflow, None))

    def sample(self, size, device, include_first=True):
        if size <= 0:
            raise ValueError("sample size must be positive")
        if len(self) == 0:
            raise RuntimeError("Encoded actor FIFO is empty")

        batches = []
        remaining = int(size)
        if include_first and self._first is not None:
            batches.append(self._first)
            remaining -= 1

        data_size = 0 if self._ids is None else int(self._ids.numel())
        if remaining > 0 and data_size > 0:
            take = min(remaining, data_size)
            indices = torch.randperm(data_size)[:take]
            batches.append(self._index(self._data, indices))

        if not batches:
            raise RuntimeError("Encoded actor FIFO does not contain a first transition yet")

        sampled = batches[0] if len(batches) == 1 else self._cat(batches)
        return {
            key: value.to(device)
            for key, value in sampled.items()
        }

    def all(self, device, include_first=True):
        if len(self) == 0:
            raise RuntimeError("Encoded actor FIFO is empty")

        batches = []
        if include_first and self._first is not None:
            batches.append(self._first)
        if self._data is not None and self.data_count > 0:
            batches.append(self._data)

        if not batches:
            raise RuntimeError("Encoded actor FIFO does not contain a first transition yet")

        encoded = batches[0] if len(batches) == 1 else self._cat(batches)
        return {
            key: value.to(device)
            for key, value in encoded.items()
        }


# ============================================================================
# Distribution Matching Mathematics
# ============================================================================
class DistributionMatcher:
    """Handles mathematical operations for distribution matching via PMD."""

    def __init__(self, 
                 lambda_reg: float,
                 gamma: float = 0.9, 
                 device: str = "cpu"):
        
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.device = device    

    def _regularized_solve(
            self,
            A: torch.Tensor,
            B: torch.Tensor,
            jitter_scale: float = 1e-10,
        ) -> torch.Tensor:
        """Solve AX=B robustly when A is singular/ill-conditioned."""
        eye = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        
        A  = A + jitter_scale* eye

        X = torch.linalg.solve(A, B)
        
        return X 
    
    def _regularized_solve_memory_efficient(
            self,
            A: torch.Tensor,
            B: torch.Tensor,
            jitter_scale: float = 1e-10,
        ) -> torch.Tensor:
        """Solve AX=B robustly when A is singular/ill-conditioned."""
        

        idx = torch.arange(A.shape[0], device=A.device)
        A[idx, idx] += jitter_scale

        X = torch.linalg.solve(A, B)
        
        return X 
            
    def compute_nu_pi(
            self, 
            phi_all_next_obs: torch.Tensor, 
            psi_all_obs_action: torch.Tensor,
            K: torch.Tensor,
            M: torch.Tensor,
            alpha: torch.Tensor,
            sink_norm: float
        ) -> torch.Tensor:
        """Compute discounted occupancy: ν = (1-γ)Φᵀ(I - γBM)⁻¹α."""
       
        N = K.shape[0]
       
        # α̃ augmented to be [α; 1]
        tilde_alpha = torch.ones((alpha.shape[0] + 1, 1), device=alpha.device, dtype=alpha.dtype)
        tilde_alpha[:-1] = alpha

        # ** COMPUTATION STEP **
        # Compute Cholesky decomposition and solve: B̃M̃ = Ã⁻¹M̃
        A = K + self.lambda_reg * torch.eye(N, device=self.device)
        L = torch.linalg.cholesky(A)
        BM = torch.cholesky_solve(M, L)
        
        # M̃ augmented to be [M 0; 0 1]
        tilde_BM = torch.zeros(BM.shape[0] + 1, BM.shape[1] + 1, device=BM.device, dtype=BM.dtype)
        tilde_BM[:-1, :-1] = BM
        tilde_BM[-1, -1] = 1.0

        inv_term = torch.linalg.solve( torch.eye(N+1, device=self.device) - self.gamma * tilde_BM, tilde_alpha)
        
        sink_state = torch.zeros((phi_all_next_obs.shape[1],1), device=self.device, dtype=phi_all_next_obs.dtype)
        sink_state[-1] = sink_norm

        # Computing Ψ̃ and Φ̃ are now of shape [N+1, d*|A| + 2] and [N+1, d + 2] respectively
        upper_left = phi_all_next_obs.T - sink_state@torch.ones((1, psi_all_obs_action.shape[1]), device=psi_all_obs_action.device, dtype=psi_all_obs_action.dtype)@psi_all_obs_action.T
        tilde_phi_all_next_obs_transposed = torch.zeros((phi_all_next_obs.shape[1]+1, phi_all_next_obs.shape[0]+1), device=phi_all_next_obs.device, dtype=phi_all_next_obs.dtype)
        tilde_phi_all_next_obs_transposed[:upper_left.shape[0], :upper_left.shape[1]] = upper_left
        tilde_phi_all_next_obs_transposed[:sink_state.shape[0], -1:] = sink_state
        # tilde_phi_all_next_obs_transposed[-1, -1] = 1.0 # TODO patch 0.1

        occupancy = (1 - self.gamma) *  tilde_phi_all_next_obs_transposed @ inv_term
        # print(f"Occupancy sum: {occupancy.sum().item()} and occupancy of sink state: {occupancy[-1].item()}")
        return occupancy

    def pseudo_inversa_svd(self, A, tol=1e-12):
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        
        # Inverti solo i valori singolari non nulli
        S_inv = torch.where(S > tol, 1.0 / S, torch.zeros_like(S))
        
        S_inv_mat = torch.diag(S_inv)
        
        A_pinv = Vh.transpose(-2, -1) @ S_inv_mat @ U.transpose(-2, -1)
        return A_pinv


    def compute_nu_pi_nystrom(
            self, 
            phi_sub_next_obs: torch.Tensor, 
            psi_sub_obs_action: torch.Tensor,
            psi_all_obs_action: torch.Tensor,
            M: torch.Tensor,
            alpha: torch.Tensor,
            sink_norm: float
        ) -> torch.Tensor:
        """Compute discounted occupancy: ν = (1-γ)Φᵀ(I - γBM)⁻¹α."""
       
        N = psi_all_obs_action.shape[0]
        subsamples = psi_sub_obs_action.shape[0]
       
        # α̃ augmented to be [α; 1]
        tilde_alpha = torch.ones((alpha.shape[0] + 1, 1), device=alpha.device, dtype=alpha.dtype)
        tilde_alpha[:-1] = alpha

        K_nm = psi_all_obs_action @ psi_sub_obs_action.T # [n, m]
        K_mm = psi_sub_obs_action @ psi_sub_obs_action.T # [m, m]
        A_nystrom = K_nm.T@K_nm + self.lambda_reg * N* K_mm # [m, m]

        BM = self._regularized_solve(A_nystrom, K_nm.T@M ) # [m, n]
       
        # ** COMPUTATION STEP **
        # M̃ augmented to be [M 0; 0 1]
        tilde_BM = torch.zeros(BM.shape[0] + 1, BM.shape[1] + 1, device=BM.device, dtype=BM.dtype)
        tilde_BM[:-1, :-1] = BM
        tilde_BM[-1, -1] = 1.0

        inv_term = torch.linalg.solve( torch.eye(subsamples+1, device=self.device) - self.gamma * tilde_BM, tilde_alpha)
        
        sink_state = torch.zeros((phi_sub_next_obs.shape[1],1), device=self.device, dtype=phi_sub_next_obs.dtype)
        sink_state[-1] = sink_norm

        # Computing Ψ̃ and Φ̃ are now of shape [N+1, d*|A| + 2] and [N+1, d + 2] respectively
        upper_left = phi_sub_next_obs.T - sink_state@torch.ones((1, psi_sub_obs_action.shape[1]), device=psi_sub_obs_action.device, dtype=psi_sub_obs_action.dtype)@psi_sub_obs_action.T
        tilde_phi_sub_next_obs_transposed = torch.zeros((phi_sub_next_obs.shape[1]+1, phi_sub_next_obs.shape[0]+1), device=phi_sub_next_obs.device, dtype=phi_sub_next_obs.dtype)
        tilde_phi_sub_next_obs_transposed[:upper_left.shape[0], :upper_left.shape[1]] = upper_left
        tilde_phi_sub_next_obs_transposed[:sink_state.shape[0], -1:] = sink_state
        # tilde_phi_sub_next_obs_transposed[-1, -1] = 1.0 # TODO patch 0.1

        occupancy = (1 - self.gamma) *  tilde_phi_sub_next_obs_transposed @ inv_term
        # print(f"Occupancy sum: {occupancy.sum().item()} and occupancy of sink state: {occupancy[-1].item()}")
        return occupancy
    
    def compute_gradient_coefficient(
            self, 
            M: torch.Tensor, 
            phi_all_next_obs:torch.Tensor, 
            psi_all_obs_action:torch.Tensor, 
            alpha:torch.Tensor,
            sink_norm: float
        ) -> torch.Tensor:
        """Compute gradient coefficient for policy update."""
        # Identity matrix
        I_n_plus1 = torch.eye(psi_all_obs_action.shape[0], device=self.device)

        sink_state = torch.zeros((phi_all_next_obs.shape[1],1), device=self.device, dtype=phi_all_next_obs.dtype)
        sink_state[-1] = sink_norm

        # Computing Ψ̃ and Φ̃ are now of shape [N+1, d*|A| + 2] and [N+1, d + 2] respectively
        upper_left = phi_all_next_obs.T - sink_state@torch.ones((1, psi_all_obs_action.shape[1]), device=psi_all_obs_action.device, dtype=psi_all_obs_action.dtype)@psi_all_obs_action.T
        tilde_phi_all_next_obs_transposed = torch.zeros((phi_all_next_obs.shape[1]+1, phi_all_next_obs.shape[0]+1), device=phi_all_next_obs.device, dtype=phi_all_next_obs.dtype)
        tilde_phi_all_next_obs_transposed[:upper_left.shape[0], :upper_left.shape[1]] = upper_left
        assert sink_state.shape[0] == upper_left.shape[0], "Sink state and upper left matrix row size mismatch"
        tilde_phi_all_next_obs_transposed[:sink_state.shape[0], -1:] = sink_state
        tilde_phi_all_next_obs = tilde_phi_all_next_obs_transposed.T
        assert torch.all(tilde_phi_all_next_obs_transposed[:sink_state.shape[0], -1:] == sink_state), "Last column of tilde_phi_all_next_obs should be sink_state"

        # Ã augmented to be [A 0; 0 1]
        # Symmetric positive definite matrix A = ψψᵀ + λI
        A = psi_all_obs_action @ psi_all_obs_action.T + self.lambda_reg * I_n_plus1
        tilde_A = torch.zeros(A.shape[0] + 1, A.shape[1] + 1, device=A.device, dtype=A.dtype)
        tilde_A[:-1, :-1] = A
        tilde_A[-1, -1] = 1.0

        # M̃ augmented to be [M 0; 0 1]
        tilde_M = torch.zeros(M.shape[0] + 1, M.shape[1] + 1, device=M.device, dtype=M.dtype)
        tilde_M[:-1, :-1] = M
        tilde_M[-1, -1] = 1.0

        # α̃ augmented to be [α; 1]
        tilde_alpha = torch.ones((alpha.shape[0] + 1, 1), device=alpha.device, dtype=alpha.dtype)
        tilde_alpha[:-1] = alpha

        # ** COMPUTATION STEP **
        # Compute Cholesky decomposition and solve: BM = A⁻¹M
        L = torch.linalg.cholesky(A)
        BM = torch.cholesky_solve(M, L)
        tilde_B_tilde_M = torch.zeros(BM.shape[0] + 1, BM.shape[1] + 1, device=BM.device, dtype=BM.dtype)
        tilde_B_tilde_M[:-1, :-1] = BM
        tilde_B_tilde_M[-1, -1] = 1.0

        # gradient = 2 γ (1 - γ)² Ã⁻ᵀ (I - γ Ã⁻¹M̃)⁻ᵀΦ̃Φ̃ᵀ(I - γ Ã⁻¹M̃)⁻¹ α̃ 
        # Using the precomputed terms and solves:
        # (I - γ Ã⁻¹M̃)⁻ᵀΦ̃ = [Φ̃ᵀ(I - γ Ã⁻¹M̃)⁻¹]ᵀ
        I_n_plus1 = torch.eye(tilde_B_tilde_M.shape[0], device=tilde_B_tilde_M.device, dtype=tilde_B_tilde_M.dtype)
        symmetric_term = torch.linalg.solve((I_n_plus1 - self.gamma * tilde_B_tilde_M).T, tilde_phi_all_next_obs)

        # Left term: Ã⁻ᵀ(I - γB̃M̃)⁻ᵀΦ̃
        # Solve Ãᵀ x = left_term_without_b using Cholesky
        L_T = torch.linalg.cholesky(tilde_A.T)
        left_term = torch.cholesky_solve(symmetric_term, L_T)

        
        # Right term: Φ̃ᵀ(I - γB̃M̃)⁻¹ α̃
        right_term = symmetric_term.T @ tilde_alpha
        gradient = 2 * self.gamma * ((1 - self.gamma) ** 2) * left_term @ right_term
      
        return gradient
    
    def compute_gradient_coefficient_nystrom(
            self, 
            M: torch.Tensor, 
            phi_all_next_obs:torch.Tensor, 
            phi_sub_next_obs:torch.Tensor,
            psi_all_obs_action:torch.Tensor, 
            psi_sub_obs_action:torch.Tensor,
            alpha:torch.Tensor,
            sink_norm: float
        ) -> torch.Tensor:
        """Compute gradient coefficient for policy update."""
        # Identity matrix
        I_n_plus1 = torch.eye(psi_all_obs_action.shape[0], device=self.device)
        N = psi_all_obs_action.shape[0]

        sink_state = torch.zeros((phi_all_next_obs.shape[1],1), device=self.device, dtype=phi_all_next_obs.dtype)
        sink_state[-1] = sink_norm

        # Computing Ψ̃ and Φ̃ are now of shape [N+1, d*|A| + 2] and [N+1, d + 2] respectively
        tilde_phi_sub_next_obs_transposed = torch.zeros((phi_sub_next_obs.shape[1]+1, phi_sub_next_obs.shape[0]+1), device=phi_sub_next_obs.device, dtype=phi_sub_next_obs.dtype)
        upper_left_sub = phi_sub_next_obs.T - sink_state@torch.ones((1, psi_sub_obs_action.shape[1]), device=psi_sub_obs_action.device, dtype=psi_sub_obs_action.dtype)@psi_sub_obs_action.T
        tilde_phi_sub_next_obs_transposed[:upper_left_sub.shape[0], :upper_left_sub.shape[1]] = upper_left_sub

        assert sink_state.shape[0] == upper_left_sub.shape[0], "Sink state and upper left matrix row size mismatch"

        tilde_phi_sub_next_obs_transposed[:sink_state.shape[0], -1:] = sink_state
        tilde_phi_sub_next_obs = tilde_phi_sub_next_obs_transposed.T

        # Ã augmented to be [A 0; 0 1]
        # Symmetric positive definite matrix A = ψψᵀ + λI
        K_nm = psi_all_obs_action @ psi_sub_obs_action.T # [n, m]
        K_mm = psi_sub_obs_action @ psi_sub_obs_action.T # [m, m]
        A_nystrom = K_nm.T@K_nm + self.lambda_reg * N * K_mm# [m, m]
        B = self._regularized_solve(A_nystrom,K_nm.T) # [m, n]
        tilde_B = torch.zeros(B.shape[0] + 1, B.shape[1] + 1, device=B.device, dtype=B.dtype)
        tilde_B[:-1, :-1] = B
        tilde_B[-1, -1] = 1.0

        # M̃ augmented to be [M 0; 0 1]
        tilde_M = torch.zeros(M.shape[0] + 1, M.shape[1] + 1, device=M.device, dtype=M.dtype)
        tilde_M[:-1, :-1] = M
        tilde_M[-1, -1] = 1.0

        # α̃ augmented to be [α; 1]
        tilde_alpha = torch.ones((alpha.shape[0] + 1, 1), device=alpha.device, dtype=alpha.dtype)
        tilde_alpha[:-1] = alpha

        # ** COMPUTATION STEP **
        # gradient = 2 γ (1 - γ)² Ã⁻ᵀ (I - γ Ã⁻¹M̃)⁻ᵀΦ̃Φ̃ᵀ(I - γ Ã⁻¹M̃)⁻¹ α̃ 
        # Using the precomputed terms and solves:
        # (I - γ Ã⁻¹M̃)⁻ᵀΦ̃ = [Φ̃ᵀ(I - γ Ã⁻¹M̃)⁻¹]ᵀ
        tilde_B_tilde_M = tilde_B @ tilde_M
        I_n_plus1 = torch.eye(tilde_B_tilde_M.shape[0], device=tilde_B_tilde_M.device, dtype=tilde_B_tilde_M.dtype)
        symmetric_term = torch.linalg.solve((I_n_plus1 - self.gamma * tilde_B_tilde_M).T, tilde_phi_sub_next_obs)

        # Left term: Ã⁻ᵀ(I - γB̃M̃)⁻ᵀΦ̃
        left_term = tilde_B.T @ symmetric_term

        
        # Right term: Φ̃ᵀ(I - γB̃M̃)⁻¹ α̃
        right_term = symmetric_term.T @ tilde_alpha
        gradient = 2 * self.gamma * ((1 - self.gamma) ** 2) * left_term @ right_term
      
        return gradient

    def compute_nu_pi_nystrom_memory_efficient(
            self, 
            phi_all_obs: torch.Tensor,
            phi_sub_next_obs: torch.Tensor, 
            psi_sub_obs_action: torch.Tensor,
            psi_all_obs_action: torch.Tensor,
            H: torch.Tensor,
            pi: torch.Tensor,
            E: torch.Tensor,
            alpha: torch.Tensor,
            sink_norm: float
        ) -> torch.Tensor:
        """Compute discounted occupancy: ν = (1-γ)Φᵀ(I - γBM)⁻¹α."""
        N = psi_all_obs_action.shape[0]

        m = psi_sub_obs_action.shape[0]

        d = phi_sub_next_obs.shape[1]

        # α̃ = [α; 1], but avoid torch.ones
        alpha_tilde = torch.empty(
            (alpha.shape[0] + 1, 1),
            device=alpha.device,
            dtype=alpha.dtype,
        )
        alpha_tilde[:-1] = alpha
        alpha_tilde[-1] = 1.0

        # Nyström matrices
        K_nm = psi_all_obs_action @ psi_sub_obs_action.T
        K_mm = psi_sub_obs_action @ psi_sub_obs_action.T
        A_nystrom = K_nm.T @ K_nm
        A_nystrom.add_(K_mm, alpha=self.lambda_reg * N)

        # H = phi_all_obs @ phi_sub_next_obs.T # [n, m] 
        M = H*(E@pi.T) # [n, m]

        BM = self._regularized_solve_memory_efficient(A_nystrom,K_nm.T @ M)

        # release big temporaries earlier
        del K_nm, K_mm, A_nystrom

        # Build S = I - gamma * tilde_BM directly
        # tilde_BM = [BM 0]
        #            [0  1]

        S = torch.empty(
            (BM.shape[0] + 1, BM.shape[1] + 1),
            device=BM.device,
            dtype=BM.dtype,

        )

        S[:-1, :-1] = BM
        S[:-1, :-1].mul_(-self.gamma)
        idx = torch.arange(BM.shape[0], device=BM.device)
        S[idx, idx] += 1.0
        S[:-1, -1] = 0.0
        S[-1, :-1] = 0.0
        S[-1, -1] = 1.0 - self.gamma

        inv_term = torch.linalg.solve(S,alpha_tilde)

        del S, alpha_tilde, BM

        # Build Φ̃ᵀ directly
        tilde_phi_T = torch.zeros(
            (d + 1, m + 1),
            device=phi_sub_next_obs.device,
            dtype=phi_sub_next_obs.dtype,
        )

        tilde_phi_T[:d, :m] = phi_sub_next_obs.T
        # sink_state is zero except at index d-1, so only this row changes
        tilde_phi_T[d - 1, :m] -= sink_norm * psi_sub_obs_action.sum(dim=1)
        tilde_phi_T[d - 1, m] = sink_norm
        occupancy = tilde_phi_T @ inv_term
        occupancy.mul_(1 - self.gamma)

        return occupancy


    
    def compute_gradient_coefficient_nystrom_memory_efficient(
            self, 
            phi_all_obs: torch.Tensor,
            phi_all_next_obs:torch.Tensor, 
            phi_sub_next_obs:torch.Tensor,
            psi_all_obs_action:torch.Tensor, 
            psi_sub_obs_action:torch.Tensor,
            H: torch.Tensor,
            pi: torch.Tensor,
            E: torch.Tensor,
            alpha:torch.Tensor,
            sink_norm: float
        ) -> torch.Tensor:
        """Compute gradient coefficient for policy update."""
        # Identity matrix
        # I_n_plus1 = torch.eye(psi_all_obs_action.shape[0], device=self.device)
        N = psi_all_obs_action.shape[0]
        m = phi_sub_next_obs.shape[0]
        d = phi_sub_next_obs.shape[1]

        # Build Phi-tilde directly, without upper_left_sub temporary
        tilde_phi_sub_next_obs_T = torch.zeros(

            (d + 1, m + 1),

            device=phi_sub_next_obs.device,

            dtype=phi_sub_next_obs.dtype,

        )
        tilde_phi_sub_next_obs_T[:d, :m] = phi_sub_next_obs.T
        tilde_phi_sub_next_obs_T[d - 1, :m] -= sink_norm * psi_sub_obs_action.sum(dim=1)
        tilde_phi_sub_next_obs_T[d - 1, m] = sink_norm
        tilde_phi_sub_next_obs = tilde_phi_sub_next_obs_T.T

        sink_state = torch.zeros((phi_all_next_obs.shape[1],1), device=self.device, dtype=phi_all_next_obs.dtype)
        sink_state[-1] = sink_norm

        K_nm = psi_all_obs_action @ psi_sub_obs_action.T # [n, m]
        K_mm = psi_sub_obs_action @ psi_sub_obs_action.T # [m, m]
        A_nystrom = K_nm.T @ K_nm
        A_nystrom.add_(K_mm, alpha=self.lambda_reg * N)  # In-place addition for memory efficiency

        B = self._regularized_solve_memory_efficient(
            A_nystrom,
            K_nm.T,
        ) # [m, n]
        # H = self._phi_all_obs @ self._phi_sub_next.T
        # H = phi_all_obs @ phi_sub_next_obs.T # [n, m] 
        M = H*(E@pi.T) # [n, m]
        BM = B @ M

        # Build S = I - gamma * tilde_B * tilde_M directly
        S = torch.empty(

            (BM.shape[0] + 1, BM.shape[1] + 1),

            device=BM.device,

            dtype=BM.dtype,

        )

        S[:-1, :-1] = BM
        S[:-1, :-1].mul_(-self.gamma)
        idx = torch.arange(BM.shape[0], device=BM.device)
        S[idx, idx] += 1.0
        S[:-1, -1] = 0.0
        S[-1, :-1] = 0.0
        S[-1, -1] = 1.0 - self.gamma

        symmetric_term = torch.linalg.solve(S.T,tilde_phi_sub_next_obs)   

        # left_term = tilde_B.T @ symmetric_term, without tilde_B
        left_term = torch.empty(
            (B.shape[1] + 1, symmetric_term.shape[1]),
            device=symmetric_term.device,
            dtype=symmetric_term.dtype,
        )
        left_term[:-1] = B.T @ symmetric_term[:-1]
        left_term[-1:] = symmetric_term[-1:]

        # right_term = symmetric_term.T @ tilde_alpha, without tilde_alpha

        right_term = symmetric_term[:-1].T @ alpha + symmetric_term[-1:].T

        gradient = left_term @ right_term

        gradient.mul_(2 * self.gamma * ((1 - self.gamma) ** 2))

        return gradient
           
# ============================================================================
# Distribution Visualizer
# ============================================================================
# ============================================================================
# Exploration Metrics Visualizer
# ============================================================================
from pathlib import Path
from collections import deque, Counter, defaultdict
from typing import Optional, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from scipy.spatial.distance import pdist, squareform

class FixedRandomEncoder(nn.Module):
    """Fixed random encoder for stable state hashing (witness network)."""
    
    def __init__(self, obs_shape, obs_type='pixels', hash_dim=128):
        super().__init__()
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        
        if obs_type == 'pixels':
            assert len(obs_shape) == 3, "Expected image observations [C, H, W]"
            
            # CNN for pixel observations
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
            repr_dim = 32 * 7 * 7
            
        else:  # obs_type == 'states' (one-hot, continuous, or learned embeddings)
            # Simple MLP for state vectors
            input_dim = obs_shape[0] if len(obs_shape) == 1 else np.prod(obs_shape)
            hidden_dim = max(128, input_dim * 2)
            
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 256),
                nn.ReLU()
            )
            repr_dim = 256
        
        # Random projection matrix for SimHash
        self.register_buffer(
            'projection_matrix',
            torch.randn(hash_dim, repr_dim) / np.sqrt(repr_dim)
        )
        
        # Initialize with Kaiming (preserves distances)
        self.apply(self._init_weights)
        
        # FREEZE all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        self.eval()  # Always in eval mode
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, obs):
        """
        Args:
            obs: [B, C, H, W] images OR [B, state_dim] state vectors
        Returns:
            features: [B, repr_dim] continuous features
        """
        with torch.no_grad():
            if self.obs_type == 'pixels':
                obs = obs.to(dtype=torch.get_default_dtype()) / 255.0
                h = self.conv(obs)
                h = self.adaptive_pool(h)
                h = h.reshape(h.size(0), -1)
            else:
                # Flatten state to [B, state_dim]
                obs = obs.to(dtype=torch.get_default_dtype())
                h = obs.reshape(obs.size(0), -1)
                h = self.mlp(h)
            return h
    
    def compute_hash(self, obs):
        """
        Args:
            obs: [B, ...] observations (any shape)
        Returns:
            hash_codes: [B] string hashes (for uniqueness on high-dimensional spaces like Atari)
        """
        with torch.no_grad():
            features = self.forward(obs)  # [B, repr_dim]
            projections = features @ self.projection_matrix.T  # [B, hash_dim]
            
            # Binary hash: sign of each projection
            binary_code = (projections > 0).long()  # [B, hash_dim]
            
            # *** FIX: Convert to string hash instead of int64 ***
            # This avoids collisions on high-dimensional spaces like Atari
            hash_codes = []
            for i in range(binary_code.shape[0]):
                # Convert each binary vector to a string (e.g., "10110101...")
                hash_str = ''.join(binary_code[i].cpu().numpy().astype(str))
                hash_codes.append(hash_str)
            
            return np.array(hash_codes, dtype=object)


class EmpiricalOccupancyTracker:
    """Track state visitation distribution over a moving window."""
    
    def __init__(self, window_size: int = 100000):
        self.window = deque(maxlen=window_size)
        self.window_size = window_size
    
    def add(self, state_hashes: np.ndarray):
        """Add batch of state hashes."""
        self.window.extend(state_hashes.tolist())
    
    def get_counts(self) -> Counter:
        """Get visit counts for each state."""
        return Counter(self.window)
    
    def get_unique_count(self) -> int:
        """Number of unique states visited."""
        return len(set(self.window))
    
    def compute_gini(self) -> float:
        """Gini coefficient: 0=uniform, 1=all mass on one state."""
        if len(self.window) == 0:
            return 0.0
        
        counts = np.array(sorted(self.get_counts().values()))
        n = len(counts)
        
        if n == 0:
            return 0.0
        
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * counts)) / (n * np.sum(counts)) - (n + 1) / n
        return gini
    
    def compute_entropy(self) -> float:
        """Shannon entropy of state distribution."""
        if len(self.window) == 0:
            return 0.0
        
        counts = np.array(list(self.get_counts().values()))
        probs = counts / counts.sum()
        return -np.sum(probs * np.log(probs + 1e-10))
    
class ExplorationVisualizer:
    """Comprehensive exploration metrics tracking and visualization."""
    
    def __init__(
        self,
        obs_shape: Tuple,  # Can be (C, H, W) for images or (state_dim,) for states
        obs_type: str,  # 'pixels' or 'states'
        feature_dim: int,
        hash_dim: int = 128,
        k_neighbors: int = 5,
        occupancy_window: int = 100000,
        save_dir: str = './exploration_plots',
        device: str = 'cpu'
    ):
        self.obs_shape = obs_shape
        self.obs_type = obs_type
        self.feature_dim = feature_dim
        self.k = k_neighbors
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Fixed random encoder for stable hashing (works for both pixels and states)
        self.random_encoder = FixedRandomEncoder(obs_shape, obs_type, hash_dim).to(device)
        
        # Occupancy tracker
        self.occupancy = EmpiricalOccupancyTracker(occupancy_window)
        
        # Metrics history: {metric_name: [(step, value), ...]}
        self.history = defaultdict(list)
        
        print(f"ExplorationVisualizer initialized:")
        print(f"  - Observation type: {obs_type}")
        print(f"  - Observation shape: {obs_shape}")
        print(f"  - Fixed random encoder: {sum(p.numel() for p in self.random_encoder.parameters())} params (frozen)")
        print(f"  - Hash dimension: {hash_dim} bits")
        print(f"  - Occupancy window: {occupancy_window} states")
    
    def update(
        self, 
        obs_batch: torch.Tensor,
        z_batch: torch.Tensor,
        step: int
    ) -> Dict[str, float]:
        """
        Update metrics with new batch.
        
        Args:
            obs_batch: [B, ...] raw observations (pixels OR state vectors)
            z_batch: [B, feature_dim] learned embeddings (for geometry metrics)
            step: current training step
        
        Returns:
            metrics: dict of computed metrics
        """
        metrics = {}
        
        # 1. Compute state hashes (fixed random encoder - works for both pixels and states)
        with torch.no_grad():
            state_hashes = self.random_encoder.compute_hash(obs_batch)
        
        self.occupancy.add(state_hashes)
        
        # 2. State coverage
        unique_states = self.occupancy.get_unique_count()
        self.history['unique_states'].append((step, unique_states))
        metrics['exploration/unique_states'] = unique_states
        
        # 3. Gini coefficient (uniformity of visits)
        gini = self.occupancy.compute_gini()
        self.history['gini'].append((step, gini))
        metrics['exploration/gini'] = gini
        
        # 4. Shannon entropy of state distribution
        entropy = self.occupancy.compute_entropy()
        self.history['entropy'].append((step, entropy))
        metrics['exploration/entropy'] = entropy
        
        # 5. k-NN distance (particle entropy on LEARNED embeddings)
        z_np = z_batch.detach().cpu().numpy()
        knn_dist = self._compute_knn_distance(z_np)
        self.history['knn_entropy'].append((step, knn_dist))
        metrics['exploration/knn_log_distance'] = knn_dist
        
        # 6. Uniformity loss (on learned embeddings)
        uniformity = self._compute_uniformity(z_np)
        self.history['uniformity'].append((step, uniformity))
        metrics['exploration/uniformity'] = uniformity
        
        return metrics
    
    def _compute_knn_distance(self, z: np.ndarray) -> float:
        """
        Kozachenko-Leonenko entropy estimator via k-NN distances.
        Higher = more spread out = better exploration.
        """
        if len(z) < self.k + 1:
            return 0.0
        
        # Subsample for efficiency
        if len(z) > 2000:
            idx = np.random.choice(len(z), 2000, replace=False)
            z = z[idx]
        
       
        
        dists = squareform(pdist(z, metric='euclidean'))
        np.fill_diagonal(dists, np.inf)
        
        # k-th nearest neighbor distance for each point
        knn_dists = np.partition(dists, self.k, axis=1)[:, self.k]
        
        # Average log-distance (entropy proxy)
        avg_log_knn = np.mean(np.log(knn_dists + 1e-8))
        
        return avg_log_knn
    
    def _compute_uniformity(self, z: np.ndarray, t: float = 2.0) -> float:
        """
        Uniformity loss from Wang & Isola (2020).
        Lower = more uniform on hypersphere.
        """
        if len(z) < 2:
            return 0.0
        
        # Subsample for efficiency
        if len(z) > 1000:
            idx = np.random.choice(len(z), 1000, replace=False)
            z = z[idx]
        
        # Normalize to unit hypersphere
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
        
        # Pairwise squared distances
        sq_dists = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=2)
        
        # Uniformity = log average of exp(-t * dist^2)
        uniformity = np.log(np.mean(np.exp(-t * sq_dists)) + 1e-8)
        
        return uniformity
    
    def plot_all(self, step: int, param_text: str = ""):
        """Generate comprehensive visualization of all metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Exploration Metrics (Step {step})', fontsize=16)
        
        # Plot 1: Cumulative unique states
        self._plot_metric(
            axes[0, 0],
            'unique_states',
            'State Coverage (Fixed Random Hash)',
            'Unique States Visited',
            color='tab:blue'
        )
        
        # Plot 2: Gini coefficient
        ax = axes[0, 1]
        self._plot_metric(
            ax,
            'gini',
            'Visit Distribution Inequality',
            'Gini Coefficient',
            color='tab:orange'
        )
        ax.axhline(0, color='green', linestyle='--', linewidth=1, label='Perfect Uniform', alpha=0.7)
        ax.legend()
        
        # Plot 3: Shannon entropy
        self._plot_metric(
            axes[0, 2],
            'entropy',
            'State Distribution Entropy',
            'Shannon Entropy (nats)',
            color='tab:green'
        )
        
        # Plot 4: k-NN distance (particle entropy)
        self._plot_metric(
            axes[1, 0],
            'knn_entropy',
            'Particle Entropy (Learned Embeddings)',
            'Log k-NN Distance',
            color='tab:red'
        )
        
        # Plot 5: Uniformity
        self._plot_metric(
            axes[1, 1],
            'uniformity',
            'Latent Space Uniformity',
            'Uniformity Loss',
            color='tab:purple'
        )
        
        # Plot 6: Lorenz curve (visit distribution)
        ax = axes[1, 2]
        self._plot_lorenz_curve(ax)
        
        # Add text to plot with hyperparameters
        if param_text:
           fig.text(0.02, 0.98, param_text, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        save_path = self.save_dir / f'exploration_metrics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved exploration metrics plot to {save_path}")
    
    def _plot_metric(self, ax, key: str, title: str, ylabel: str, color: str = 'tab:blue'):
        """Helper to plot a single metric timeseries."""
        if key not in self.history or len(self.history[key]) == 0:
            ax.text(0.5, 0.5, 'No data yet', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        
        steps, values = zip(*self.history[key])
        ax.plot(steps, values, color=color, linewidth=2, alpha=0.8)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    def _plot_lorenz_curve(self, ax):
        """Plot Lorenz curve of state visitation distribution."""
        counts = self.occupancy.get_counts()
        
        if len(counts) == 0:
            ax.text(0.5, 0.5, 'No data yet', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Visit Distribution (Lorenz Curve)')
            return
        
        # Sort counts ascending
        sorted_counts = np.array(sorted(counts.values()))
        cumsum_counts = np.cumsum(sorted_counts)
        
        # Normalize to [0, 1]
        x = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
        y = cumsum_counts / cumsum_counts[-1]
        
        # Plot
        ax.plot([0, 1], [0, 1], 'g--', linewidth=1, label='Perfect Uniform', alpha=0.7)
        ax.plot(x, y, 'b-', linewidth=2, label='Actual Distribution')
        ax.fill_between(x, x, y, alpha=0.2)
        
        ax.set_xlabel('Cumulative % of States (sorted)')
        ax.set_ylabel('Cumulative % of Visits')
        ax.set_title('Visit Distribution (Lorenz Curve)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_tsne(
        self, 
        z_batch: torch.Tensor, 
        step: int, 
        max_points: int = 3000,
        method: str = 'tsne'  # 'tsne' or 'umap'
    ):
        """
        2D visualization of learned embedding space.
        
        Args:
            z_batch: [B, feature_dim] learned embeddings
            step: current step
            max_points: subsample if batch too large
            method: 'tsne' or 'umap'
        """
        z = z_batch.detach().cpu().numpy()
        
        if len(z) < 50:
            print(f"Skipping {method} plot: need at least 50 points, got {len(z)}")
            return
        
        # Subsample
        if len(z) > max_points:
            idx = np.random.choice(len(z), max_points, replace=False)
            z = z[idx]
        
        # Dimensionality reduction
        if method == 'tsne':
            from sklearn.manifold import TSNE
            z_2d = TSNE(n_components=2, perplexity=min(30, len(z) // 2), random_state=42).fit_transform(z)
            title = f't-SNE Latent Space (Step {step})'
        elif method == 'umap':
            try:
                reducer = umap.UMAP(n_components=2, random_state=42)
                z_2d = reducer.fit_transform(z)
                title = f'UMAP Latent Space (Step {step})'
            except ImportError:
                print("UMAP not installed, falling back to t-SNE")
                z_2d = TSNE(n_components=2, perplexity=min(30, len(z) // 2), random_state=42).fit_transform(z)
                title = f't-SNE Latent Space (Step {step})'
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 10))
        scatter = ax.scatter(
            z_2d[:, 0], 
            z_2d[:, 1], 
            c=np.arange(len(z_2d)),  # Color by order (temporal)
            cmap='viridis',
            alpha=0.6,
            s=20
        )
        
        plt.colorbar(scatter, ax=ax, label='Temporal Order')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(f'{method.upper()} 1')
        ax.set_ylabel(f'{method.upper()} 2')
        ax.grid(True, alpha=0.3)
        
        save_path = self.save_dir / f'{method}_{step}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {method.upper()} plot to {save_path}")
    
    def get_summary(self) -> Dict[str, float]:
        """Get latest values of all metrics."""
        summary = {}
        for key, values in self.history.items():
            if len(values) > 0:
                summary[key] = values[-1][1]
        return summary

# ============================================================================
# Gridworld-Specific Visualizer (Adapted for v2)
# ============================================================================
class DiscreteStateVisualizationAdapter:
    """Small adapter that turns different discrete env APIs into one plotting surface."""

    def __init__(self, env):
        self.env = self._find_discrete_env(env)
        self.n_states = self.env.n_states
        self.dead_state = getattr(self.env, "DEAD_STATE", None)
        self.state_plot_cells = {}
        self.plot_cells = []
        seen_plot_cells = set()

        for state_idx in range(self.n_states):
            state = self.env.idx_to_state[state_idx]
            plot_cell = self._state_to_plot_cell(state)
            if plot_cell is None:
                continue
            self.state_plot_cells[state_idx] = plot_cell
            if plot_cell not in seen_plot_cells:
                seen_plot_cells.add(plot_cell)
                self.plot_cells.append(plot_cell)

        if not self.plot_cells:
            raise ValueError("No plottable states found for discrete visualization")

        self.min_x = min(cell[0] for cell in self.plot_cells)
        self.min_y = min(cell[1] for cell in self.plot_cells)
        self.max_x = max(cell[0] for cell in self.plot_cells)
        self.max_y = max(cell[1] for cell in self.plot_cells)
        self.grid_width = self.max_x - self.min_x + 1
        self.grid_height = self.max_y - self.min_y + 1
        self.plot_cell_to_idx = {cell: idx for idx, cell in enumerate(self.plot_cells)}

    def _find_discrete_env(self, env):
        current = env
        while current is not None:
            if hasattr(current, "n_states") and hasattr(current, "idx_to_state") and hasattr(current, "state_to_idx"):
                return current

            if hasattr(current, "env"):
                current = current.env
            elif hasattr(current, "unwrapped") and current.unwrapped is not current:
                current = current.unwrapped
            else:
                break

        raise AttributeError(
            "Could not find a discrete environment interface. "
            "Expected attributes 'n_states', 'idx_to_state', and 'state_to_idx'."
        )

    def _state_to_plot_cell(self, state):
        if self.dead_state is not None and state == self.dead_state:
            return None
        if isinstance(state, np.ndarray):
            state = tuple(state.tolist())
        if isinstance(state, (tuple, list)) and len(state) >= 2:
            return (int(state[0]), int(state[1]))
        return None

    def values_to_grid(self, values: np.ndarray, reduce: str = "sum") -> np.ndarray:
        grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        counts = np.zeros_like(grid)

        for state_idx, value in enumerate(values):
            plot_cell = self.state_plot_cells.get(state_idx)
            if plot_cell is None:
                continue
            x = plot_cell[0] - self.min_x
            y = plot_cell[1] - self.min_y
            grid[y, x] += value
            counts[y, x] += 1

        if reduce == "mean":
            np.divide(grid, np.maximum(counts, 1.0), out=grid)
        elif reduce != "sum":
            raise ValueError(f"Unsupported reduction: {reduce}")
        return grid

    def aggregate_policy_per_cell(self, policy_per_state: np.ndarray) -> np.ndarray:
        policy_per_cell = np.zeros((len(self.plot_cells), policy_per_state.shape[1]), dtype=np.float32)
        counts = np.zeros(len(self.plot_cells), dtype=np.float32)

        for state_idx, probs in enumerate(policy_per_state):
            plot_cell = self.state_plot_cells.get(state_idx)
            if plot_cell is None:
                continue
            cell_idx = self.plot_cell_to_idx[plot_cell]
            policy_per_cell[cell_idx] += probs
            counts[cell_idx] += 1

        policy_per_cell /= np.maximum(counts[:, None], 1.0)
        return policy_per_cell

    def iter_plot_cells(self):
        for plot_cell in self.plot_cells:
            yield plot_cell, self.plot_cell_to_idx[plot_cell]

    def state_label(self, state_idx: int) -> str:
        return str(self.env.idx_to_state[state_idx])

    def state_components(self, state_idx: int):
        state = self.env.idx_to_state[state_idx]
        if isinstance(state, np.ndarray):
            state = tuple(state.tolist())
        if not isinstance(state, (tuple, list)):
            return None, None, ()

        plot_cell = self._state_to_plot_cell(state)
        orientation = int(state[2]) if len(state) >= 3 else None
        extras = tuple(state[3:]) if len(state) > 3 else ()
        return plot_cell, orientation, extras

    def is_orientation_augmented(self) -> bool:
        orientations = []
        for state_idx in range(self.n_states):
            _, orientation, _ = self.state_components(state_idx)
            if orientation is not None:
                orientations.append(orientation)
        return len(set(orientations)) > 1


class EmbeddingDistributionVisualizerV2:
    """Visualizer for embedding-based distribution matching results (adapted for v2)."""
    def __init__(self, agent):
        """
        Initialize visualizer with agent reference.
        
        Args:
            agent: DistMatchingEmbeddingAgentv2 instance
        """
        self.agent = agent
        self.state_adapter = DiscreteStateVisualizationAdapter(agent.env)
        self.env = self.state_adapter.env
        self.n_states = self.state_adapter.n_states
        self.n_actions = agent.n_actions
        self.all_state_ids_one_hot = torch.eye(self.n_states, device=self.agent.device)
        self.min_x = self.state_adapter.min_x
        self.min_y = self.state_adapter.min_y
        self.grid_width = self.state_adapter.grid_width
        self.grid_height = self.state_adapter.grid_height
        self.is_minigrid_style = self.state_adapter.is_orientation_augmented() and self.n_actions == 7
        
        # Action symbols and colors - support both 4 and 8 actions
        if self.n_actions == 4:
            self.action_symbols = ['↑', '↓', '←', '→']  # 0=up, 1=down, 2=left, 3=right
            self.action_names = ['Up', 'Down', 'Left', 'Right']
            self.action_colors = ['#D81B60', '#1E88E5', '#43A047', '#FB8C00']
        elif self.n_actions == 8:
            self.action_symbols = ['→', '↘', '↓', '↙', '←', '↖', '↑', '↗']
            self.action_names = ['Right', 'Down-Right', 'Down', 'Down-Left', 'Left', 'Up-Left', 'Up', 'Up-Right']
            self.action_colors = [
                '#FB8C00',  # 0: right
                '#E53935',  # 1: down-right
                '#1E88E5',  # 2: down
                '#00ACC1',  # 3: down-left
                '#43A047',  # 4: left
                '#7CB342',  # 5: up-left
                '#D81B60',  # 6: up
                '#8E24AA',  # 7: up-right
            ]
        elif self.n_actions == 2:
            self.action_symbols = ['→', '↓']
            self.action_names = ['Right', 'Down']
            self.action_colors = ['#D81B60', '#1E88E5']
        elif self.n_actions == 7:
            self.action_symbols = ['↺', '↻', '↑', 'P', 'D', 'T', '✓']
            self.action_names = [
                '0 left: Turn left',
                '1 right: Turn right',
                '2 forward: Move forward',
                '3 pickup: Pick up an object',
                '4 drop: Unused',
                '5 toggle: Toggle/activate an object',
                '6 done: Unused',
            ]
            self.action_colors = [
                '#D81B60',
                '#1E88E5',
                '#FB8C00',
                '#43A047',
                '#E53935',
                '#8E24AA',
                '#6D4C41',
            ]
        else:
            self.action_symbols = [str(i) for i in range(self.n_actions)]
            self.action_names = [f'Action {i}' for i in range(self.n_actions)]
            self.action_colors = plt.cm.tab20(np.linspace(0, 1, self.n_actions))
        
        # Pre-render all state observations if using pixel observations
        if self.agent.obs_type == 'pixels':
            print("Pre-rendering all state images for correlation matrix...")
            self._prerendered_states = []
            
            render_resolution = getattr(self.agent.wrapped_env, 'render_resolution', 224)
            frame_stack = self.agent.obs_shape[0] // self.agent.image_channels
            
            for s_idx in range(self.n_states):
                if s_idx % 10 == 0:
                    print(f"  Rendering state {s_idx}/{self.n_states}...")
                
                image = self.env.render_from_position(self.env.idx_to_state[s_idx], show_goal=False)
                image = self._prepare_rendered_state_image(image, render_resolution)
                
                # Convert HWC to CHW and stack frames
                image_chw = image.transpose(2, 0, 1).copy()
                stacked_image = np.tile(image_chw, (frame_stack, 1, 1))
                
                self._prerendered_states.append(stacked_image)
            
            # Stack into tensor [n_states, C, H, W]
            self._prerendered_states = torch.from_numpy(
                np.stack(self._prerendered_states)
            ).float().to(self.agent.device)
            
            print(f"✓ Pre-rendered {self.n_states} states with shape {self._prerendered_states.shape}")
        else:
            self._prerendered_states = None

    def _prepare_rendered_state_image(self, image: np.ndarray, render_resolution: int) -> np.ndarray:
        image = np.asarray(image, dtype=np.uint8)

        if self.agent.grayscale:
            if image.ndim == 3 and image.shape[2] == 1:
                image = image[..., 0]
            elif image.ndim == 3:
                image = np.asarray(Image.fromarray(image).convert('L'))
            elif image.ndim != 2:
                raise ValueError(f"Expected grayscale image to be 2D or HWC, got shape {image.shape}")
        elif image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)

        if image.shape[:2] != (render_resolution, render_resolution):
            image = np.asarray(
                Image.fromarray(image).resize(
                    (render_resolution, render_resolution),
                    Image.LANCZOS,
                )
            )

        if self.agent.grayscale:
            if image.ndim == 2:
                image = image[..., None]
        elif image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)

        if image.ndim != 3 or image.shape[2] != self.agent.image_channels:
            raise ValueError(
                f"Expected image shape [H, W, {self.agent.image_channels}], got {image.shape}"
            )

        return image

    def _orientation_label(self, orientation: int) -> str:
        mapping = {
            0: "dir=0 (right)",
            1: "dir=1 (down)",
            2: "dir=2 (left)",
            3: "dir=3 (up)",
        }
        return mapping.get(int(orientation), f"dir={orientation}")

    def _format_extra_value(self, value) -> str:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bool):
            return "yes" if value else "no"
        if value is None:
            return "none"
        return str(value)

    def _extra_state_label(self, extras: tuple) -> str:
        if not extras:
            return "base state"

        if len(extras) == 1:
            # Generic label: many MiniGrid variants add a carried-object indicator here.
            return f"extra={self._format_extra_value(extras[0])}"

        parts = [self._format_extra_value(value) for value in extras]
        return "extras=" + ", ".join(parts)

    def _build_minigrid_policy_panels(self, policy_per_state: np.ndarray):
        panel_map = {}
        orientations = set()
        extras_set = set()

        for state_idx in range(self.n_states):
            plot_cell, orientation, extras = self.state_adapter.state_components(state_idx)
            if plot_cell is None or orientation is None:
                continue
            orientations.add(int(orientation))
            extras_set.add(tuple(extras))
            panel_map.setdefault((tuple(extras), int(orientation)), {})[plot_cell] = policy_per_state[state_idx]

        return panel_map, sorted(extras_set), sorted(orientations)
        
    def _state_dist_to_grid(self, nu: np.ndarray) -> np.ndarray:
        """Convert state distribution vector to 2D grid."""
        return self.state_adapter.values_to_grid(nu, reduce="sum")

    def _actor_alpha_features_for_visualization(self):
        """Return the alpha support that matches the active actor update mode."""
        if (
            getattr(self.agent, 'subsamples', None) is not None
            and hasattr(self.agent, '_sub_alpha')
            and self.agent._sub_alpha is not None
            and hasattr(self.agent, '_phi_sub_next')
            and self.agent._phi_sub_next is not None
        ):
            # Nyström updates optimize against the subsample support, so alpha
            # must be interpreted on the same support for visual diagnostics.
            return self.agent._phi_sub_next, self.agent._sub_alpha

        if (
            hasattr(self.agent, '_alpha')
            and self.agent._alpha is not None
            and hasattr(self.agent, '_phi_all_next')
            and self.agent._phi_all_next is not None
        ):
            return self.agent._phi_all_next, self.agent._alpha

        return None, None
    
    def _compute_initial_distribution(self) -> np.ndarray:
        """Compute initial distribution on the active alpha support."""
        with torch.no_grad():
            if self.agent.obs_type == 'pixels':
                # Use pre-rendered images
                enc_all_states = self.agent.aug_and_encode(self._prerendered_states, project=True).detach() #.cpu()
            else:
                # Use one-hot encodings
                enc_all_states = self.agent.encoder(self.all_state_ids_one_hot)
            
            phi_next, alpha = self._actor_alpha_features_for_visualization()
            if alpha is not None:
                
                # Add augmented dimension to encoded states
                zero_col = torch.zeros(*enc_all_states.shape[:-1], 1, device=enc_all_states.device)
                enc_all_states_aug = torch.cat([enc_all_states, zero_col], dim=-1) #.cpu()
                
                kernel = enc_all_states_aug @ phi_next.T
                nu_init = kernel @ alpha
            else:
                nu_init = torch.ones(self.n_states, 1) / self.n_states
        return nu_init.flatten().cpu().numpy()
    
    
    def render_observation_from_state(self, state_idx: int) -> np.ndarray:
        """
        Render observation from a state index.
        
        For pixel observations: renders image from position and stacks frames
        For state observations: returns one-hot encoding
        
        Args:
            state_idx: State index
            
        Returns:
            Observation in the format expected by the agent
        """
        if self.agent.obs_type == 'pixels':
            # Get render resolution and frame stack
            render_resolution = getattr(self.agent.wrapped_env, 'render_resolution', 224)
            frame_stack = self.agent.obs_shape[0] // self.agent.image_channels
            
            # Get position from state index
            image = self.env.render_from_position(self.env.idx_to_state[state_idx], show_goal=False)
            image = self._prepare_rendered_state_image(image, render_resolution)
            
            # Convert HWC to CHW format [C, H, W]
            image_chw = image.transpose(2, 0, 1).copy()
            
            # Stack the frame multiple times to match frame_stack
            # The agent expects [C*frame_stack, H, W]
            stacked_image = np.tile(image_chw, (frame_stack, 1, 1))
            
            return stacked_image
        else:
            # For state observations, return one-hot encoding
            obs_onehot = np.eye(self.n_states, dtype=np.float32)[state_idx]
            return obs_onehot

    def _get_policy_per_state(self) -> np.ndarray:
        """Extract policy probabilities for each state."""
        policy_per_state = np.zeros((self.n_states, self.n_actions))
        
        for s_idx in range(self.n_states):
            # Get observation for this state (handles both pixels and states)
            obs = self.render_observation_from_state(s_idx)
            policy_per_state[s_idx] = self.agent.compute_action_probs(obs)
        
        return policy_per_state
    
    def _compute_state_correlation_matrix(self) -> np.ndarray:
        """Compute correlation matrix between encoded states."""
        with torch.no_grad():
            if self.agent.obs_type == 'pixels':
                # Use pre-rendered images
                enc_all_states = self.agent.encoder(self._prerendered_states).detach().cpu()
            else:
                # Use one-hot encodings
                all_states = self.all_state_ids_one_hot.to(self.agent.device)
                enc_all_states = self.agent.encoder(all_states).detach().cpu()
            
            # Normalize embeddings
            enc_norm = F.normalize(enc_all_states, p=2, dim=1)
            
            # Compute cosine similarity matrix
            correlation_matrix = enc_norm @ enc_norm.T
            
        return correlation_matrix.numpy()
    
    def _compute_state_to_states_correlation(self) -> np.ndarray:
        """Compute average correlation of each state with all others."""
        correlation_matrix = self._compute_state_correlation_matrix()
        
        # Set diagonal to 0 (we don't want self-correlation)
        np.fill_diagonal(correlation_matrix, 0)
        
        # Average absolute correlation for each state
        state_orthogonality_deviation = np.mean(np.abs(correlation_matrix), axis=1)
        
        return state_orthogonality_deviation
    
    def plot_embeddings_2d(self, save_path: str, use_tsne: bool = False, project=False):
        """Plot 2D projection of state embeddings using PCA or t-SNE."""
        with torch.no_grad():
            if self.agent.obs_type == 'pixels':
                observations = self._prerendered_states
            else:
                observations = self.all_state_ids_one_hot

            if project:
                embeddings = self.agent.encoder.encode_and_project(observations).detach().cpu().numpy()
            else:
                embeddings = self.agent.encoder(observations).detach().cpu().numpy()
        
        # Dimensionality reduction
        if use_tsne:
            reducer = TSNE(n_components=2, random_state=42)
            method_name = 't-SNE'
        else:
            reducer = PCA(n_components=2)
            method_name = 'PCA'
        
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Color code by state ID or grid position
        colors = plt.cm.viridis(np.linspace(0, 1, len(embeddings)))
        
        for idx, embedding_2d in enumerate(embeddings_2d):
            ax.scatter(embedding_2d[0], embedding_2d[1], c=[colors[idx]], s=100, alpha=0.7)
            ax.text(
                embedding_2d[0],
                embedding_2d[1],
                self.state_adapter.state_label(idx),
                fontsize=8,
                ha='center',
                va='center'
            )
        
        obs_type_str = "Image" if self.agent.obs_type == 'pixels' else "State"
        ax.set_xlabel(f'{method_name} Component 1')
        ax.set_ylabel(f'{method_name} Component 2')
        ax.set_title(f'{obs_type_str} Embeddings Visualization ({method_name})')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Embeddings visualization saved to: {save_path}")
        plt.close(fig)
    
    def plot_results(self, step: int, save_path: str = None):
        """Create comprehensive visualization of learning progress."""
        figsize = (28, 15)
        fig = plt.figure(figsize=figsize)

        # Add parameter text with dataset novelty info
        param_text = (
            f"Step: {step}\n"
            f"γ = {self.agent.discount}\n"
            f"η = {self.agent.lr_actor}\n"
            f"λ = {self.agent.lambda_reg}\n"
            f"sink notm = {utils.schedule(self.agent.sink_schedule, step):.6f}\n"
            f"PMD steps = {self.agent.pmd_steps}\n"
            
        )
        fig.text(0.02, 0.98, param_text, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        gs = fig.add_gridspec(3, 6, hspace=0.35, wspace=0.4, height_ratios=[1.0, 1.2, 1.2])
        
        # Top row: initial distribution, policy arrows, correlation matrix
        ax_init = fig.add_subplot(gs[0, 0])
        ax_policy = fig.add_subplot(gs[0, 1:3])
        ax_corr = fig.add_subplot(gs[0, 3:6])
        
        # Lower rows: give the policy bar chart most of the vertical space
        ax_sample_occ = fig.add_subplot(gs[1, 0])
        ax_state_corr = fig.add_subplot(gs[2, 0])
        ax_policy_bars = fig.add_subplot(gs[1:, 1:5])
        
        # Compute distributions
        nu_init = self._compute_initial_distribution()
        policy_per_state = self._get_policy_per_state()
        
        # Plot distributions
        self._plot_distribution(ax_init, nu_init, 'Initial Distribution')
        
        if self.is_minigrid_style:
            self._plot_minigrid_policy_summary(
                ax_policy,
                step,
                'Policy summary is saved separately.\n'
                'Main plot omits per-cell policy aggregation because MiniGrid states\n'
                'depend on orientation and may depend on additional discrete factors.'
            )
            self._plot_minigrid_policy_summary(
                ax_policy_bars,
                step,
                'See the separate MiniGrid policy debug image for orientation-conditioned\n'
                'action probabilities. Batch occupancy remains meaningful here.'
            )
        else:
            # Plot policy arrows with grid cells
            self._plot_policy_arrows(ax_policy, policy_per_state)
            ax_policy.set_title(f'Policy (Step {step})', fontsize=12, fontweight='bold')
            
            # Plot policy bars per cell
            self._plot_policy_bars_per_cell(ax_policy_bars, policy_per_state)
        
        # Plot correlation matrix
        correlation_matrix = self._compute_state_correlation_matrix()
        self._plot_state_correlations(ax_corr, correlation_matrix)
        
        # Plot sample occupancy (NOT NORMALIZED)
        self._plot_sample_occupancy(ax_sample_occ, title=f'Batch State Occupancy (Step {step})', normalize=False)
        
        # Plot state-to-states correlation
        state_corrs = self._compute_state_to_states_correlation()
        self._plot_state_to_states_correlation(ax_state_corr, state_corrs)
        
        plt.suptitle(f'Distribution Matching Progress (Step {step})', fontsize=16, y=0.995, fontweight='bold')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Gridworld visualization saved to: {save_path}")
            if self.is_minigrid_style:
                self._save_minigrid_policy_debug_plot(step, policy_per_state, save_path)
            if self.agent.subsamples is not None:
                self._save_nystrom_subsample_plot(step, save_path)
        
        plt.close(fig)

    def _action_legend_elements(self):
        return [
            Patch(facecolor=self.action_colors[i], edgecolor='black', label=self.action_names[i])
            for i in range(self.n_actions)
        ]

    def _plot_policy_bars_per_cell(self, ax, policy_per_state):
        """Plot policy bars inside each grid cell, similar to action probabilities grid."""
        ax.set_xlim(self.min_x - 0.5, self.min_x + self.grid_width - 0.5)
        ax.set_ylim(self.min_y - 0.5, self.min_y + self.grid_height - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Invert Y axis so (0,0) is top-left
        ax.set_title('Policy Action Probabilities per Cell', fontsize=13, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.25, linewidth=0.6)
        
        # Draw environment structure
        if hasattr(self.env, 'walkable_areas'):
            for area in self.env.walkable_areas:
                rect = Rectangle((area[0], area[1]), area[2], area[3],
                            fill=False, edgecolor='gray', linewidth=1, linestyle='--', alpha=0.5)
                ax.add_patch(rect)
        
        # Cell size and bar parameters
        cell_size = 1.0
        inner_padding = 0.08
        usable_width = cell_size - (2 * inner_padding)
        bar_spacing = usable_width / self.n_actions
        bar_width = bar_spacing * 0.9
        max_bar_height = cell_size * 0.92
        
        # MiniGrid has multiple heading-specific states per cell, so we average them
        # to obtain one robust per-cell debugging view.
        policy_per_cell = self.state_adapter.aggregate_policy_per_cell(policy_per_state)

        for (x, y), cell_idx in self.state_adapter.iter_plot_cells():
            
            # Draw cell background
            rect = Rectangle(
                (x - cell_size/2, y - cell_size/2), 
                cell_size, cell_size,
                linewidth=1.8,
                edgecolor='black',
                facecolor='#F3F4F6',
                alpha=0.95
            )
            ax.add_patch(rect)
            
            # Get action probabilities
            probs = policy_per_cell[cell_idx]
            
            # Draw bars for each action
            start_x = x - cell_size/2 + inner_padding + bar_width / 2
            
            for a_idx in range(self.n_actions):
                bar_x = start_x + a_idx * bar_spacing
                bar_height = probs[a_idx] * max_bar_height
                
                # Bars start from bottom of cell (y + cell_size/2) and grow upward
                bar_y = y + cell_size/2 - bar_height - 0.04
                
                bar_rect = Rectangle(
                    (bar_x - bar_width/2, bar_y),
                    bar_width, 
                    bar_height,
                    facecolor=self.action_colors[a_idx],
                    edgecolor='black', 
                    linewidth=0.8
                )
                ax.add_patch(bar_rect)
        
        # Set proper ticks
        ax.set_xticks(np.arange(self.min_x, self.min_x + self.grid_width))
        ax.set_yticks(np.arange(self.min_y, self.min_y + self.grid_height))
        
        # Add legend
        ax.legend(
            handles=self._action_legend_elements(),
            loc='upper left',
            bbox_to_anchor=(1.01, 1.0),
            title='Action Mapping',
            fontsize=9,
            title_fontsize=10,
            frameon=True
        )

    def _plot_minigrid_policy_summary(self, ax, step: int, text: str):
        ax.axis('off')
        ax.set_title(f'MiniGrid Policy Summary (Step {step})', fontsize=12, fontweight='bold')
        ax.text(
            0.5,
            0.5,
            text,
            ha='center',
            va='center',
            fontsize=12,
            linespacing=1.4,
            bbox=dict(boxstyle='round', facecolor='#F3F4F6', edgecolor='black', alpha=0.95),
            transform=ax.transAxes,
        )

    def _plot_policy_bars_for_state_subset(self, ax, subset_probs, title: str):
        ax.set_xlim(self.min_x - 0.5, self.min_x + self.grid_width - 0.5)
        ax.set_ylim(self.min_y - 0.5, self.min_y + self.grid_height - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(np.arange(self.min_x, self.min_x + self.grid_width))
        ax.set_yticks(np.arange(self.min_y, self.min_y + self.grid_height))
        ax.grid(True, alpha=0.2, linewidth=0.5)

        cell_size = 1.0
        inner_padding = 0.08
        usable_width = cell_size - (2 * inner_padding)
        bar_spacing = usable_width / self.n_actions
        bar_width = bar_spacing * 0.9
        max_bar_height = cell_size * 0.92

        for plot_cell in self.state_adapter.plot_cells:
            x, y = plot_cell
            rect = Rectangle(
                (x - cell_size / 2, y - cell_size / 2),
                cell_size,
                cell_size,
                linewidth=1.3,
                edgecolor='black',
                facecolor='#F3F4F6',
                alpha=0.95
            )
            ax.add_patch(rect)

            probs = subset_probs.get(plot_cell)
            if probs is None:
                continue

            start_x = x - cell_size / 2 + inner_padding + bar_width / 2
            for a_idx in range(self.n_actions):
                bar_x = start_x + a_idx * bar_spacing
                bar_height = probs[a_idx] * max_bar_height
                bar_y = y + cell_size / 2 - bar_height - 0.04
                bar_rect = Rectangle(
                    (bar_x - bar_width / 2, bar_y),
                    bar_width,
                    bar_height,
                    facecolor=self.action_colors[a_idx],
                    edgecolor='black',
                    linewidth=0.6
                )
                ax.add_patch(bar_rect)

    def _save_minigrid_policy_debug_plot(self, step: int, policy_per_state: np.ndarray, save_path: str):
        panel_map, extras_values, orientations = self._build_minigrid_policy_panels(policy_per_state)
        if not orientations:
            return

        n_rows = max(1, len(extras_values))
        n_cols = len(orientations)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5.5 * n_cols + 4, 4.8 * n_rows),
            squeeze=False
        )

        for row_idx, extras in enumerate(extras_values):
            for col_idx, orientation in enumerate(orientations):
                ax = axes[row_idx][col_idx]
                subset_probs = panel_map.get((extras, orientation), {})
                title = f"{self._orientation_label(orientation)}\n{self._extra_state_label(extras)}"
                self._plot_policy_bars_for_state_subset(ax, subset_probs, title)
                if row_idx == n_rows - 1:
                    ax.set_xlabel('X')
                if col_idx == 0:
                    ax.set_ylabel('Y')

        legend_ax = axes[0][-1]
        legend_ax.legend(
            handles=self._action_legend_elements(),
            loc='upper left',
            bbox_to_anchor=(1.02, 1.0),
            title='Action Mapping',
            fontsize=9,
            title_fontsize=10,
            frameon=True
        )

        fig.suptitle(
            f'MiniGrid Policy Debug Panels (Step {step})\n'
            'Panels are split by agent orientation and extra discrete state factors when present.',
            fontsize=15,
            y=0.995
        )
        plt.tight_layout()

        root, ext = os.path.splitext(save_path)
        panel_save_path = f"{root}_minigrid_policy{ext}"
        plt.savefig(panel_save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"MiniGrid policy debug visualization saved to: {panel_save_path}")

    def _compute_batch_and_subsample_state_counts(self):
        """Infer state counts for the latest actor batch and Nyström subsample."""
        def _accumulate_state_counts(batch_embeddings, all_state_embeddings):
            counts = np.zeros(self.n_states, dtype=np.float32)
            for batch_emb in batch_embeddings:
                similarities = F.cosine_similarity(
                    batch_emb.unsqueeze(0),
                    all_state_embeddings,
                    dim=1
                )
                closest_state = torch.argmax(similarities).item()
                counts[closest_state] += 1
            return counts

        # Use the cached features from the last actor update
        if not hasattr(self.agent, '_phi_all_next') or self.agent._phi_all_next is None:
            return None, None
        
        # We need to infer which states are in the batch
        # Since we have embeddings, we can compare them to known state embeddings
        with torch.no_grad():
            if self.agent.obs_type == 'pixels':
                # Use pre-rendered images
                enc_all_states = self.agent.aug_and_encode(self._prerendered_states, project=True).detach().cpu()
            else:
                # Use one-hot encodings
                all_states = self.all_state_ids_one_hot.to(self.agent.device)
                enc_all_states = self.agent.encoder(all_states).detach().cpu()

            # Always show occupancy for the full actor batch.
            all_batch_embeddings = self.agent._phi_all_next[:, :-1].detach().cpu()
            state_counts = _accumulate_state_counts(all_batch_embeddings, enc_all_states)

            subsample_counts = None
            if self.agent.subsamples is not None and hasattr(self.agent, '_phi_sub_next'):
                subsample_embeddings = self.agent._phi_sub_next[:, :-1].detach().cpu()
                subsample_counts = _accumulate_state_counts(subsample_embeddings, enc_all_states)
        return state_counts, subsample_counts

    def _plot_sample_occupancy(self, ax, title='Batch State Occupancy', normalize=True):
        """Plot state occupancy from the current batch.
        
        Args:
            ax: matplotlib axis
            title: plot title
            normalize: if True, normalize counts to probabilities; if False, show raw counts
        """
        state_counts, _ = self._compute_batch_and_subsample_state_counts()
        if state_counts is None:
            ax.text(0.5, 0.5, 'No batch data available yet',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(title, fontsize=12, fontweight='bold')
            return
        
        # Normalize or keep raw counts
        if normalize and state_counts.sum() > 0:
            state_dist = state_counts / state_counts.sum()
            colorbar_label = 'Probability'
        else:
            state_dist = state_counts
            colorbar_label = 'Count'
        
        # Plot on grid
        grid = self._state_dist_to_grid(state_dist)
        
        im = ax.imshow(grid, cmap='YlGnBu', interpolation='nearest')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xticks(np.arange(self.grid_width))
        ax.set_yticks(np.arange(self.grid_height))
        ax.grid(True, which='both', color='white', linewidth=0.5, alpha=0.35)
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)

    def _plot_nystrom_subsamples(self, ax, title='Nyström Subsample Occupancy'):
        """Plot the Nyström subsample counts on their own grid."""
        _, subsample_counts = self._compute_batch_and_subsample_state_counts()
        if subsample_counts is None or subsample_counts.sum() <= 0:
            ax.text(
                0.5,
                0.5,
                'No Nyström subsample data available yet',
                ha='center',
                va='center',
                transform=ax.transAxes,
                fontsize=12
            )
            ax.set_title(title, fontsize=12, fontweight='bold')
            return False

        subsample_grid = self._state_dist_to_grid(subsample_counts)
        im = ax.imshow(subsample_grid, cmap='Oranges', interpolation='nearest')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xticks(np.arange(self.grid_width))
        ax.set_yticks(np.arange(self.grid_height))
        ax.grid(True, which='both', color='white', linewidth=0.5, alpha=0.35)

        for y, x in np.argwhere(subsample_grid > 0):
            ax.text(
                x,
                y,
                f'{int(subsample_grid[y, x])}',
                ha='center',
                va='center',
                fontsize=9,
                fontweight='bold',
                color='black'
            )

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Count')
        return True

    def _compute_nystrom_subsample_state_counts_by_action(self):
        """Infer Nyström subsample source-state counts, split by sampled action."""
        if (
            self.agent.subsamples is None
            or not hasattr(self.agent, '_phi_sub_obs')
            or self.agent._phi_sub_obs is None
            or not hasattr(self.agent, '_sub_actions')
            or self.agent._sub_actions is None
        ):
            return None

        counts_by_action = np.zeros((self.n_actions, self.n_states), dtype=np.float32)

        with torch.no_grad():
            if self.agent.obs_type == 'pixels':
                enc_all_states = self.agent.aug_and_encode(self._prerendered_states, project=True).detach().cpu()
            else:
                all_states = self.all_state_ids_one_hot.to(self.agent.device)
                enc_all_states = self.agent.encoder(all_states).detach().cpu()

            subsample_embeddings = self.agent._phi_sub_obs[:, :-1].detach().cpu()
            sub_actions = self.agent._sub_actions.detach().cpu().long().reshape(-1)
            usable_count = min(subsample_embeddings.shape[0], sub_actions.shape[0])

            for batch_emb, action_idx in zip(subsample_embeddings[:usable_count], sub_actions[:usable_count]):
                action_idx = int(action_idx.item())
                if action_idx < 0 or action_idx >= self.n_actions:
                    continue
                similarities = F.cosine_similarity(
                    batch_emb.unsqueeze(0),
                    enc_all_states,
                    dim=1
                )
                closest_state = torch.argmax(similarities).item()
                counts_by_action[action_idx, closest_state] += 1

        return counts_by_action

    def _plot_nystrom_subsamples_by_action(self, fig, axes, step: int):
        """Plot one Nyström subsample state heatmap per action."""
        counts_by_action = self._compute_nystrom_subsample_state_counts_by_action()
        flat_axes = np.asarray(axes).reshape(-1)

        if counts_by_action is None or counts_by_action.sum() <= 0:
            for ax in flat_axes:
                ax.axis('off')
            flat_axes[0].text(
                0.5,
                0.5,
                'No Nyström subsample data available yet',
                ha='center',
                va='center',
                transform=flat_axes[0].transAxes,
                fontsize=12
            )
            flat_axes[0].set_title(f'Nyström Subsamples by Action (Step {step})', fontsize=12, fontweight='bold')
            return False

        grids_by_action = [
            self._state_dist_to_grid(counts_by_action[action_idx])
            for action_idx in range(self.n_actions)
        ]
        max_count = max(float(grid.max()) for grid in grids_by_action)
        vmax = max(max_count, 1.0)
        last_im = None

        for action_idx, ax in enumerate(flat_axes):
            if action_idx >= self.n_actions:
                ax.axis('off')
                continue

            grid = grids_by_action[action_idx]
            last_im = ax.imshow(
                grid,
                cmap='Oranges',
                interpolation='nearest',
                vmin=0,
                vmax=vmax
            )
            ax.set_title(
                f'{self.action_names[action_idx]} ({action_idx})',
                fontsize=11,
                fontweight='bold'
            )
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_xticks(np.arange(self.grid_width))
            ax.set_yticks(np.arange(self.grid_height))
            ax.grid(True, which='both', color='white', linewidth=0.5, alpha=0.35)

            for y, x in np.argwhere(grid > 0):
                ax.text(
                    x,
                    y,
                    f'{int(grid[y, x])}',
                    ha='center',
                    va='center',
                    fontsize=8,
                    fontweight='bold',
                    color='black'
                )

        fig.colorbar(last_im, ax=flat_axes[:self.n_actions].tolist(), fraction=0.025, pad=0.02, label='Count')
        fig.suptitle(f'Nyström Subsample State Occupancy by Action (Step {step})', fontsize=14, fontweight='bold')
        return True

    def _save_nystrom_subsample_plot(self, step: int, save_path: str):
        """Save a dedicated Nyström subsample occupancy figure next to the main plot."""
        n_cols = min(self.n_actions, 4)
        n_rows = int(np.ceil(self.n_actions / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.4 * n_cols, 4.1 * n_rows),
            squeeze=False
        )
        plotted = self._plot_nystrom_subsamples_by_action(fig, axes, step)
        if not plotted:
            plt.close(fig)
            return

        plt.tight_layout()
        root, ext = os.path.splitext(save_path)
        subsample_save_path = f"{root}_nystrom_subsamples{ext}"
        plt.savefig(subsample_save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Nyström subsample visualization saved to: {subsample_save_path}")

    def _plot_distribution(self, ax, nu, title):
        """Plot state distribution on grid WITHOUT text annotations."""
        grid = self._state_dist_to_grid(nu)
        
        im = ax.imshow(grid, cmap='YlOrRd', interpolation='nearest')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _plot_policy_arrows(self, ax, policy_per_state):
        """Plot policy as arrows on grid WITH cell boundaries."""
        # Create background grid
        grid = np.zeros((self.grid_height, self.grid_width))
        ax.imshow(grid, cmap='gray', alpha=0.05, interpolation='nearest')

        # MiniGrid has multiple heading-specific states per cell, so we average them
        # to obtain one robust per-cell debugging view.
        policy_per_cell = self.state_adapter.aggregate_policy_per_cell(policy_per_state)

        for (cell_x, cell_y), cell_idx in self.state_adapter.iter_plot_cells():
            x, y = cell_x - self.min_x, cell_y - self.min_y
            
            # Draw rectangle around each cell
            rect = Rectangle(
                (x - 0.5, y - 0.5), 
                1, 1,
                linewidth=1.8,
                edgecolor='black',
                facecolor='#F3F4F6',
                alpha=0.95
            )
            ax.add_patch(rect)
            
            # Draw arrow for most likely action
            probs = policy_per_cell[cell_idx]
            max_action = np.argmax(probs)
            
            ax.text(x, y, self.action_symbols[max_action],
                ha='center', va='center',
                fontsize=24, color=self.action_colors[max_action],
                weight='bold', alpha=min(0.9, probs[max_action] + 0.3))
        
        ax.set_xlim(-0.5, self.grid_width - 0.5)
        ax.set_ylim(self.grid_height - 0.5, -0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xticks(np.arange(self.grid_width))
        ax.set_yticks(np.arange(self.grid_height))
        ax.grid(True, which='both', color='black', linewidth=0.5, alpha=0.3)
        ax.legend(
            handles=self._action_legend_elements(),
            loc='upper left',
            bbox_to_anchor=(1.01, 1.0),
            title='Action Mapping',
            fontsize=9,
            title_fontsize=10,
            frameon=True
        )



    def _plot_state_correlations(self, ax, correlation_matrix):
        """Plot correlation matrix heatmap WITHOUT text annotations."""
        im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
        ax.set_title('State Embedding Correlations', fontsize=12, fontweight='bold')
        ax.set_xlabel('State Index')
        ax.set_ylabel('State Index')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _plot_state_to_states_correlation(self, ax, state_correlations):
        """Plot per-state average correlation WITHOUT text annotations."""
        grid = self.state_adapter.values_to_grid(state_correlations, reduce="mean")
        
        im = ax.imshow(grid, cmap='RdYlGn_r', interpolation='nearest', vmin=0, vmax=1)
        ax.set_title('State Orthogonality Deviation\n(Lower = More Orthogonal)', fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Avg |Correlation|')

            
# ============================================================================
# Main Agent
# ============================================================================
class RoverAgent:
    def __init__(self,
                 name,
                 obs_type,
                 obs_shape,
                 grayscale,
                 action_shape,
                 lr_actor,
                 discount,
                 lambda_reg,
                 batch_size,
                 batch_size_actor,
                 subsamples,
                 nstep,
                 use_tb,
                 use_wandb,
                 lr_T,
                 lr_encoder,
                 curl,
                 embedding_sum_loss,
                 hidden_dim,
                 feature_dim,
                 update_every_steps,
                 update_actor_every_steps,
                 pmd_steps,
                 num_expl_steps,
                 T_init_steps,
                 total_train_steps,
                 sink_schedule,
                 epsilon_schedule,
                 mode,
                 reward,
                 embeddings = True,
                 linear_projection = False,
                 pmd_eta_mode: str = "none",
                 pmd_best_iterate: bool = True,
                 pmd_grad_clip_norm: float = 0.0,
                 pmd_adagrad_eps: float = 1e-8,
                 pmd_eta_min: float = 1e-8,
                 pmd_eta_max: float = 1e3,
                 pmd_backtrack_factor: float = 0.5,
                 pmd_backtrack_max_trials: int = 8,
                 compute_dtype: str = "float32",
                 encoded_fifo_capacity: Optional[int] = None,
                 encoded_fifo_encode_batch_size: int = 4096,
                 encoded_fifo_cuda_oom_splits: int = 4,
                 device: str = "cpu",
                 ):

        self.compute_dtype = _resolve_torch_dtype(compute_dtype)
        torch.set_default_dtype(self.compute_dtype)

        self.n_states = obs_shape[0]
        self.n_actions = action_shape[0]
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        self.grayscale = grayscale
        self.feature_dim = feature_dim if feature_dim is not None else self.n_states
        self.action_dim = action_shape[0]
        self.latent_a_dim = int(self.action_dim * 1.25) + 1 # From TACO
        self.lr_actor = lr_actor
        self.discount = discount
        self.lr_T = lr_T
        self.T_init_steps = T_init_steps
        self.batch_size = batch_size
        self.batch_size_actor = batch_size_actor
        # assert batch_size_actor >= batch_size, "Actor update batch size must be greater than or equal to encoder update batch size"
        self.update_every_steps = update_every_steps
        self.update_actor_every_steps = update_actor_every_steps
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.device = device
        self.pmd_steps = pmd_steps
        self.embeddings = embeddings
        self.curl = curl
        if curl:
            utils.ColorPrint.red("CURL is enabled, but stromgly suggested to not use it.\nAll the paper results are without CURL, and it may cause poor performance. Use with caution.")

        self.embedding_sum_loss = embedding_sum_loss
        self.reward = reward
        self.pmd_eta_mode = pmd_eta_mode.lower()
        assert self.pmd_eta_mode in ["none", "adagrad", "backtracking", "adadiff"], "pmd_eta_mode must be one of ['none', 'adagrad', 'backtracking', 'adadiff']"
        self.pmd_best_iterate = pmd_best_iterate
        self.pmd_grad_clip_norm = pmd_grad_clip_norm
        self.pmd_adagrad_eps = pmd_adagrad_eps
        self.pmd_eta_min = pmd_eta_min
        self.pmd_eta_max = pmd_eta_max
        self.pmd_backtrack_factor = pmd_backtrack_factor
        self.pmd_backtrack_max_trials = pmd_backtrack_max_trials

        self.mode = mode
        assert self.mode in ['l1', 'l2'], "Mode must be 'l1' or 'l2'"

        self.first_save = False
        self.sink_schedule = sink_schedule
        self.epsilon_schedule = epsilon_schedule
        self.gradient_coeff = None

        self.num_expl_steps = num_expl_steps
        self.lambda_reg = lambda_reg
        self.image_channels = 1 if self.grayscale else 3
        self.subsamples = subsamples
        min_fifo_capacity = max(
            int(self.batch_size_actor),
            int(self.subsamples) if self.subsamples is not None else 0,
            1,
        )
        if encoded_fifo_capacity is None:
            encoded_fifo_capacity = min_fifo_capacity
        self.encoded_fifo_capacity = int(encoded_fifo_capacity)
        if self.encoded_fifo_capacity < min_fifo_capacity:
            utils.ColorPrint.yellow(
                f"encoded_fifo_capacity={self.encoded_fifo_capacity} is smaller than "
                f"the actor sample size; raising it to {min_fifo_capacity}."
            )
            self.encoded_fifo_capacity = min_fifo_capacity
        self.encoded_fifo_encode_batch_size = int(encoded_fifo_encode_batch_size)
        self.encoded_fifo_cuda_oom_splits = int(encoded_fifo_cuda_oom_splits)
        self._encoded_actor_fifo = EncodedTransitionFIFO(self.encoded_fifo_capacity)
        self._encoded_fifo_replay_marker = None
        
        # Track unique state-action pairs from previous dataset
        self._previous_unique_pairs = set()
        self._previous_unique_next_states = set()
        self._dataset_novelty_stats = {
            'total_current': 0,
            'new_pairs': 0,
            'old_pairs': 0,
            'new_percentage': 0.0,
            'total_previous': 0,
            'new_next_states': 0,
            'old_next_states': 0,
            'next_states_new_percentage': 0.0,
            'total_previous_next_states': 0
        }
        
        if obs_type == 'pixels':
            if self.curl:
                self.aug = utils.RandomShiftsAug(pad=4)
            else:
                self.aug = nn.Identity()
            assert embeddings, "Pixel observations require embeddings to be True"
            self.encoder = CNNEncoder(
                obs_shape,
                feature_dim,
                mode=mode
            ).to(self.device)
            
            self.obs_dim = self.feature_dim
        else:
            # Components
            self.aug = nn.Identity()
            if embeddings == False:
                self.encoder = nn.Identity()
                self.feature_dim = obs_shape[0]
                utils.ColorPrint.yellow("WARNING: Using identity encoder for state observations")
            else:
                self.encoder = Encoder(
                        obs_shape, 
                        hidden_dim, 
                        self.feature_dim,
                    ).to(self.device)
            self.obs_dim = self.feature_dim
       

        self.project_sa = ProjectSA(
            self.obs_dim * self.n_actions,
            hidden_dim,
            self.obs_dim,
            linear=linear_projection
        ).to(self.device)

        
        self.policy_encoder = copy.deepcopy(self.encoder).to(self.device)
        self._freeze_module(self.policy_encoder)
        self._policy_is_synced = True
        
        self.distribution_matcher = DistributionMatcher(
            gamma=self.discount,
            lambda_reg=self.lambda_reg,
            device=self.device  
        )
        
        self.W = None #nn.Parameter(torch.rand(feature_dim, feature_dim).to(self.device))
       
        if self.reward:
            self.reward = nn.Sequential(
                nn.Linear(self.obs_dim * self.n_actions, hidden_dim), 
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1)
            ).to(self.device)
        
        # parameter list:
        parameters = list(self.encoder.parameters()) 
        if self.W is not None:
            parameters+= [self.W]
        else:
            self.W = nn.Identity()
        if self.reward:
            parameters += list(self.reward.parameters())
        
        # Optimizers
        if embeddings:
            self.encoder_optimizer = torch.optim.AdamW(
                parameters,
                lr=lr_encoder,
                weight_decay=1e-5,   # try 1e-6 to 1e-4
                betas=(0.9, 0.999),
                eps=1e-8
            )

            # decay to 10% of initial LR over total training steps
            self.encoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.encoder_optimizer,
                T_max=total_train_steps,   # pass from config
                eta_min=lr_encoder * 0.1)
                        
        else:
            self.encoder_optimizer = None
        self.transition_optimizer = torch.optim.Adam(
            self.project_sa.parameters(),
            lr=lr_T
        )
        
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.training = False

        self.current_action_probs = []
        self.action_probs_history = []  # List of [step, mean_action_probs_array]
        self.policy_deviation_history = []  # List of [step, deviation_value]

    
        self.debug_visualizer = build_debug_visualizer_suite(
            agent=self,
            exploration_visualizer_cls=ExplorationVisualizer,
            gridworld_visualizer_cls=EmbeddingDistributionVisualizerV2,
        )
        self.visualizer = self.debug_visualizer.exploration_visualizer
        self.gridworld_visualizer = None
        self.env = None
        self.wrapped_env = None
        self._discrete_env = None

        # Gradient norm tracking by reward
        self.max_samples_per_reward = 150
        self.gradient_samples = {
            '+1': [],  # List of (step, gradient_norm)
            '-1': [],  # List of (step, gradient_norm)
            '0': []    # List of (step, gradient_norm)
        }
        self.gradient_norm_history = {
            '+1': [],  # List of (step, mean_norm, std_norm)
            '-1': [],
            '0': []
        }
        self.current_eta = 0.0
        self._adagrad_accum = None

        self.subsampled = None

    def _find_discrete_env(self, env):
        current = env
        while current is not None:
            if hasattr(current, "n_states") and hasattr(current, "idx_to_state") and hasattr(current, "state_to_idx"):
                return current

            if hasattr(current, "env"):
                current = current.env
            elif hasattr(current, "unwrapped") and current.unwrapped is not current:
                current = current.unwrapped
            else:
                break
        return env.unwrapped
    
    def insert_env(self, env):
        """
        Insert environment reference for gridworld-specific visualizations.
        Call this from pretrain.py after agent creation.
        """       
        self.wrapped_env = env
        self.env = self._find_discrete_env(env)
        
        try:
            self.gridworld_visualizer = self.debug_visualizer.attach_env(env)
            if self.gridworld_visualizer is not None:
                print("✓ Domain-specific debug visualizer initialized")
        except Exception as e:
            print(f"⚠ Could not initialize domain-specific debug visualizer: {e}")
            self.gridworld_visualizer = None


    
    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.project_sa.train(training)
        self.policy_encoder.eval()

    def init_meta(self):
        return OrderedDict()

    def get_meta_specs(self):
        return tuple()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        return meta
    
    def _encode_state_action(
        self, 
        encoded_obs: torch.Tensor, 
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Encode (s,a) pairs as ψ(s,a) = φ(s) ⊗ e_a."""
        action_onehot = F.one_hot(actions.long(), self.n_actions).reshape(-1, self.n_actions)  # [B, |A|]
        
        # Outer product: [B, d] ⊗ [B, |A|] -> [B, d*|A|]
        encoded_sa = torch.einsum('bd,ba->bda', encoded_obs, action_onehot)
        return encoded_sa.reshape(encoded_obs.shape[0], -1)

    def _freeze_module(self, module: nn.Module) -> None:
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)

    def _sync_policy_encoder(self) -> None:
        self.policy_encoder.load_state_dict(self.encoder.state_dict())
        self._freeze_module(self.policy_encoder)
        self._policy_is_synced = True

    def ready_for_snapshot(self) -> bool:
        return self._policy_is_synced

    def _encode_with_module(self, module: nn.Module, obs: torch.Tensor, project: bool = False) -> torch.Tensor:
        obs = self.aug(obs)
        if not self.embeddings:
            return module(obs)
        if project:
            return module.encode_and_project(obs)
        return module(obs)

    def _policy_logits_from_H(self, H: torch.Tensor, coeff: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute policy logits for a given kernel matrix H and PMD coefficient vector."""
        coeff = self.gradient_coeff if coeff is None else coeff
        if coeff is None:
            return torch.zeros(H.shape[0], self.n_actions, device=H.device, dtype=H.dtype)
        sink_bias = torch.ones(H.shape[0], self.E.shape[1], device=H.device, dtype=H.dtype) * coeff[-1]
        return H @ (coeff[:-1] * self.E) + sink_bias

    def _policy_from_H(self, H: torch.Tensor, coeff: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Closed-form PMD policy from logits."""
        logits = self._policy_logits_from_H(H, coeff=coeff)
        return torch.softmax(-logits, dim=1, dtype=logits.dtype)
    
    
    def compute_action_probs(self, obs: np.ndarray) -> np.ndarray:
        """Compute π(·|s) for given observation."""
        with torch.no_grad():
            # Handle different observation types
            if self.obs_type == 'pixels':
                # obs should already be an image [C, H, W]
                if obs.ndim == 2:
                    raise ValueError(
                        "For pixel observations, compute_action_probs expects an image [C, H, W], "
                        f"but got shape {obs.shape}. Use render_observation_from_state() first."
                    )
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)  # [1, C, H, W]
            else:
                # State observations: [x, y] -> [1, 2]
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)  # [1, obs_dim]
            
            enc_obs = self._encode_with_module(self.policy_encoder, obs_tensor, project=True)
    
            if self.gradient_coeff is None:
                return np.ones(self.n_actions) / self.n_actions
            
            # Add a zero to enc_obs to account for the extra row in H
            enc_obs_augmented = torch.cat([enc_obs, torch.zeros((1, 1), device=enc_obs.device)], dim=1)  # [1, feature_dim + 1]
            H = enc_obs_augmented @ self._phi_all_obs.T  # [1, num_unique]
            probs = self._policy_from_H(H)

            
            if torch.sum(probs) == 0.0 or torch.isnan(torch.sum(probs)):
                utils.ColorPrint.red(f"Warning: action_probs sum to zero or NaN. Returning uniform distribution. Check training stability and learning rates.{torch.sum(probs)}, {probs}")
                probs = torch.ones_like(probs) / self.n_actions
                # raise ValueError(f"action_probs sum to zero or NaN", torch.sum(probs), probs)
            logger.debug(f"Action probabilities: {probs.cpu().numpy().flatten()}")
            return probs.cpu().numpy().flatten()

    
    def act(self, obs, meta, step, eval_mode):
        if step < self.num_expl_steps or np.random.rand() < utils.schedule(self.epsilon_schedule, step):
            return np.random.randint(self.n_actions)

        # Compute action probabilities
        action_probs = self.compute_action_probs(obs)
        self.current_action_probs.append(action_probs)  # Store for visualization
        # print(f"Step {step}: Action probabilities: {action_probs}")
        # Sample action
        return np.random.choice(self.n_actions, p=action_probs)

    
    def _is_T_sufficiently_initialized(self, step: int) -> bool:
        """Check if transition learning phase is complete."""
        return step >= self.num_expl_steps + self.T_init_steps 
       
    def update_encoders(self, obs, action, next_obs, reward):
        metrics = dict()
        
        # Encode
        obs_en = self.aug_and_encode(obs, project=True)
        with torch.no_grad():
            next_obs_en = self.aug_and_encode(next_obs, project=True)

        encoded_state_action = self._encode_state_action(obs_en, action)
        
        # Predict next state
        projected_sa = self.project_sa(encoded_state_action)  
        
        # Normalize embeddings L2
        if self.mode == 'l1':
            norm_next_obs_en = F.normalize(next_obs_en, p=2, dim=1, eps=1e-10)
        elif self.mode == 'l2':
            norm_next_obs_en = next_obs_en
        norm_projected_sa = F.normalize(projected_sa, p=2, dim=1, eps=1e-10)

        # Compute loss
        # 1. Contrastive loss: 
        # Wz = torch.matmul(self.W, norm_next_obs_en.T)  # [feature_dim, B]
        logits = torch.matmul(norm_projected_sa, norm_next_obs_en.T)  # [B, B]
        logits = logits - torch.max(logits, 1)[0][:, None]  # For numerical stability
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        contrastive_loss = self.cross_entropy_loss(logits, labels)
        
        z_anchor = self.aug_and_encode(obs, project=True)
        with torch.no_grad():
            z_pos = self.aug_and_encode(obs, project=True)

        ### Compute CURL loss
        if self.curl:
            # Normalize embeddings L2
            if self.mode == 'l1':
                z_anchor = F.normalize(z_anchor, p=2, dim=1, eps=1e-10)
                z_pos = F.normalize(z_pos, p=2, dim=1, eps=1e-10)
            # Wz = torch.matmul(self.W, z_pos.T)  # [feature_dim, B]
            logits = torch.matmul(z_anchor, z_pos.T)  # [B, B]
            logits = logits - torch.max(logits, 1)[0][:, None]  # For numerical stability
            labels = torch.arange(logits.shape[0]).long().to(self.device)
            curl_loss = self.cross_entropy_loss(logits, labels)
        else:
            curl_loss = torch.tensor(0.0, device=self.device)

        if self.reward:
            reward_pred = self.reward(encoded_state_action)
            reward_loss = F.mse_loss(reward_pred, reward.to(self.device))
        else:
            reward_loss = torch.tensor(0.0, device=self.device)
        metrics['reward_loss'] = reward_loss.item()


        if self.embedding_sum_loss>0:
            # Sum of embeddings loss = 1
            sum_next_obs_en = torch.sum(next_obs_en, dim=1)  # [B]
            embedding_sum_loss = self.embedding_sum_loss * torch.mean((sum_next_obs_en - 1.0) ** 2)
        else:
            embedding_sum_loss = torch.tensor(0.0, device=self.device)

        loss =  contrastive_loss + curl_loss + embedding_sum_loss+reward_loss
        
        # max_grad_norm = 1.0
        # Optimize
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad()      
            # torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_grad_norm)
        # torch.nn.utils.clip_grad_norm_(self.project_sa.parameters(), max_grad_norm)
        self.transition_optimizer.zero_grad()
        loss.backward()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.step()
            self._policy_is_synced = False
        self.transition_optimizer.step()
        self.encoder_scheduler.step()

        # Print losses
        logger.debug(f"Transition Model Losses: Contrastive={contrastive_loss.item():.4f}, CURL={curl_loss.item():.4f}, Embedding Sum={embedding_sum_loss.item():.4f}, Reward={reward_loss.item():.4f}, Total={loss.item():.4f}")
        if self.use_tb or self.use_wandb:
            metrics['transition_loss'] = loss.item()
            metrics['contrastive_loss'] = contrastive_loss.item()
            metrics['curl_loss'] = curl_loss.item()
        return metrics
    

    def update_actor(self, obs=None, action=None, next_obs=None, step=None, rewards=None, encoded_full=None):
        """Update policy using Projected Mirror Descent."""
        metrics = dict()

        if encoded_full is None:
            self._sync_policy_encoder()
            self._cache_features(obs, action, next_obs, encoder=self.policy_encoder)
        else:
            self._cache_encoded_features(encoded_full)

        self.gradient_coeff = torch.zeros((self._phi_all_obs.shape[0]+1, 1), device=self.device)  # [z_x + 1, 1]
        prev_gradient_coeff = self.gradient_coeff.clone()
        self.H = self._phi_all_obs @ self._phi_all_next.T # [n, n]
        self.K = self._psi_all @ self._psi_all.T  # [n, n]
        base_eta = float(utils.schedule(self.lr_actor, step))
        base_eta = float(np.clip(base_eta, self.pmd_eta_min, self.pmd_eta_max))
        self.current_eta = base_eta

        sink_norm = utils.schedule(self.sink_schedule, step)
        self.pi = self._policy_from_H(self.H.T, coeff=self.gradient_coeff)  # [z_x+1, n_actions]

        M = self.H*(self.E@self.pi.T) 

        nu_pi = self.distribution_matcher.compute_nu_pi(
                phi_all_next_obs = self._phi_all_next,
                psi_all_obs_action= self._psi_all,
                K= self.K,
                M = M,
                alpha=self._alpha,
                sink_norm=sink_norm 
        )
        actor_loss = torch.linalg.norm(nu_pi)**2
        print(f"Actor loss (squared norm of occupancy measure): {actor_loss}")
        best_loss = actor_loss
        best_pi = self.pi.clone()
        best_coeff = self.gradient_coeff.clone()

        self._adagrad_accum = 0.0

        for iteration in range(self.pmd_steps):
            grad_update = self.distribution_matcher.compute_gradient_coefficient(
                M, 
                phi_all_next_obs = self._phi_all_next, 
                psi_all_obs_action = self._psi_all, 
                alpha = self._alpha,
                sink_norm=sink_norm
            ) 

            # Track gradient norms by reward (only on final iteration)
            if iteration == self.pmd_steps - 1 and rewards is not None:
                self._track_gradient_norms(grad_update, rewards, step)

            if self.pmd_grad_clip_norm > 0:
                grad_norm = torch.linalg.norm(grad_update)
                if grad_norm > self.pmd_grad_clip_norm:
                    grad_update = grad_update * (self.pmd_grad_clip_norm / (grad_norm + 1e-12))

            if self.pmd_eta_mode == "adagrad":
                # Infinite norm for mirror descent
                grad_norm_sq = float(torch.max(grad_update * grad_update).item())
                self._adagrad_accum += grad_norm_sq
                eta_t = base_eta / np.sqrt(self._adagrad_accum + self.pmd_adagrad_eps)
                eta_t = float(np.clip(eta_t, self.pmd_eta_min, self.pmd_eta_max))
            elif self.pmd_eta_mode == "adadiff":
        
                # Infinite norm for mirror descent
                grad_norm_sq = float(torch.max(grad_update * grad_update - prev_gradient_coeff*prev_gradient_coeff).item())
                self._adagrad_accum += grad_norm_sq
                eta_t = base_eta / np.sqrt(self._adagrad_accum + self.pmd_adagrad_eps)
                eta_t = float(np.clip(eta_t, self.pmd_eta_min, self.pmd_eta_max))
            else:
                eta_t = base_eta

            candidate_coeff = self.gradient_coeff + eta_t * grad_update
            candidate_pi = self._policy_from_H(self.H.T, coeff=candidate_coeff)
            candidate_M = self.H * (self.E @ candidate_pi.T)
            candidate_nu = self.distribution_matcher.compute_nu_pi(
                phi_all_next_obs=self._phi_all_next,
                psi_all_obs_action=self._psi_all,
                K=self.K,
                M=candidate_M,
                alpha=self._alpha,
                sink_norm=sink_norm
            )
            candidate_loss = torch.linalg.norm(candidate_nu) ** 2

            if self.pmd_eta_mode == "backtracking":
                trial_eta = eta_t
                trial = 0
                while candidate_loss > actor_loss and trial < self.pmd_backtrack_max_trials:
                    trial_eta *= self.pmd_backtrack_factor
                    trial_eta = float(np.clip(trial_eta, self.pmd_eta_min, self.pmd_eta_max))
                    candidate_coeff = self.gradient_coeff + trial_eta * grad_update
                    candidate_pi = self._policy_from_H(self.H.T, coeff=candidate_coeff)
                    candidate_M = self.H * (self.E @ candidate_pi.T)
                    candidate_nu = self.distribution_matcher.compute_nu_pi(
                        phi_all_next_obs=self._phi_all_next,
                        psi_all_obs_action=self._psi_all,
                        K=self.K,
                        M=candidate_M,
                        alpha=self._alpha,
                        sink_norm=sink_norm
                    )
                    candidate_loss = torch.linalg.norm(candidate_nu) ** 2
                    trial += 1
                eta_t = trial_eta

            self.current_eta = eta_t
            self.gradient_coeff = candidate_coeff
            prev_gradient_coeff = grad_update.clone()
            self.pi = candidate_pi
            M = candidate_M
            actor_loss = candidate_loss

            if actor_loss < best_loss:
                best_loss = actor_loss
                best_pi = self.pi.clone()
                best_coeff = self.gradient_coeff.clone()

            if iteration % 10 == 0 or iteration == self.pmd_steps - 1:
                print(f"  PMD Iteration {iteration}, Actor loss: {actor_loss}, eta: {self.current_eta:.6g}")

        if self.pmd_best_iterate:
            self.pi = best_pi
            self.gradient_coeff = best_coeff
            actor_loss = best_loss
            

        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss
            metrics['actor_eta'] = float(self.current_eta)
            metrics['actor_best_loss'] = float(best_loss)
            metrics['sink_norm'] = float(sink_norm)
   
        return metrics

    def update_actor_nystrom(self,
                             full_obs,
                             full_action,
                             full_next_obs,
                             step,
                             rewards=None,
                             sub_obs=None,
                             sub_action=None,
                             sub_next_obs=None,
                             sub_rewards=None,
                             encoded_full=None,
                             encoded_sub=None):
        """Update policy using Projected Mirror Descent and Nystrom Approximation."""
        metrics = dict()
        if encoded_full is not None or encoded_sub is not None:
            if encoded_full is None or encoded_sub is None:
                raise ValueError("Nyström actor update requires both encoded_full and encoded_sub.")
            self._cache_encoded_features(encoded_full, encoded_sub=encoded_sub)
        elif sub_obs is None or sub_action is None or sub_next_obs is None:
            raise ValueError("Nyström actor update requires subsampled observations, actions, and next observations.")
        else:
            self._sync_policy_encoder()
            self._cache_features(
                full_obs,
                full_action,
                full_next_obs,
                encoder=self.policy_encoder,
                sub_obs=sub_obs,
                sub_action=sub_action,
                sub_next_obs=sub_next_obs,
            )

        utils.ColorPrint.blue(f"Starting Nyström PMD actor update with {self._phi_all_obs.shape[0]} total samples and {self._phi_sub_next.shape[0]} subsampled points.")
        self.gradient_coeff = torch.zeros((self._phi_all_obs.shape[0]+1, 1), device=self.device)  # [z_x + 1, 1]
        prev_gradient_coeff = self.gradient_coeff.clone()
        sub_H = self._phi_all_obs @ self._phi_sub_next.T # [n, m]
        base_eta = float(utils.schedule(self.lr_actor, step))
        base_eta = float(np.clip(base_eta, self.pmd_eta_min, self.pmd_eta_max))
        self.current_eta = base_eta

        sink_norm = utils.schedule(self.sink_schedule, step)
        self.pi = self._policy_from_H(sub_H.T, coeff=self.gradient_coeff)  # [z_x+1, n_actions]

        # M = self.H*(self.E@self.pi.T) # [n, ]

        nu_pi = self.distribution_matcher.compute_nu_pi_nystrom_memory_efficient(
                    phi_all_obs=self._phi_all_obs,
                    phi_sub_next_obs = self._phi_sub_next,
                    psi_sub_obs_action = self._psi_sub,
                    psi_all_obs_action = self._psi_all,
                    H = sub_H,
                    pi = self.pi,
                    E = self.E,
                    alpha=self._sub_alpha,
                    sink_norm=sink_norm 
                )
        actor_loss = torch.linalg.norm(nu_pi)**2
        print(f"Actor loss (squared norm of occupancy measure): {actor_loss}")
        best_loss = actor_loss
        best_pi = self.pi.clone()
        best_coeff = self.gradient_coeff.clone()

        self._adagrad_accum = 0.0

        for iteration in range(self.pmd_steps):
            grad_update = self.distribution_matcher.compute_gradient_coefficient_nystrom_memory_efficient(
                phi_all_obs=self._phi_all_obs,
                phi_all_next_obs = self._phi_all_next,
                phi_sub_next_obs = self._phi_sub_next,
                psi_all_obs_action = self._psi_all,
                psi_sub_obs_action = self._psi_sub,
                H = sub_H,
                pi=self.pi,
                E=self.E,
                alpha = self._sub_alpha,
                sink_norm=sink_norm
            )           

            # Track gradient norms by reward (only on final iteration)
            # if iteration == self.pmd_steps - 1 and sub_rewards is not None:
            #     self._track_gradient_norms(grad_update, sub_rewards, step)

            if self.pmd_grad_clip_norm > 0:
                grad_norm = torch.linalg.norm(grad_update)
                if grad_norm > self.pmd_grad_clip_norm:
                    grad_update = grad_update * (self.pmd_grad_clip_norm / (grad_norm + 1e-12))

            if self.pmd_eta_mode == "adagrad":
                # Infinite norm for mirror descent
                grad_norm_sq = float(torch.max(grad_update * grad_update).item())
                self._adagrad_accum += grad_norm_sq
                eta_t = base_eta / np.sqrt(self._adagrad_accum + self.pmd_adagrad_eps)
                eta_t = float(np.clip(eta_t, self.pmd_eta_min, self.pmd_eta_max))
            elif self.pmd_eta_mode == "adadiff":
        
                # Infinite norm for mirror descent
                grad_norm_sq = float(torch.max(grad_update * grad_update - prev_gradient_coeff*prev_gradient_coeff).item())
                self._adagrad_accum += grad_norm_sq
                eta_t = base_eta / np.sqrt(self._adagrad_accum + self.pmd_adagrad_eps)
                eta_t = float(np.clip(eta_t, self.pmd_eta_min, self.pmd_eta_max))
            else:
                eta_t = base_eta

            candidate_coeff = self.gradient_coeff + eta_t * grad_update
            candidate_pi = self._policy_from_H(sub_H.T, coeff=candidate_coeff)
            # candidate_M = sub_H * (self.E @ candidate_pi.T)
            candidate_nu = self.distribution_matcher.compute_nu_pi_nystrom_memory_efficient(
                    phi_all_obs=self._phi_all_obs,
                    phi_sub_next_obs = self._phi_sub_next,
                    psi_sub_obs_action = self._psi_sub,
                    psi_all_obs_action = self._psi_all,
                    H = sub_H,
                    pi = candidate_pi,
                    E = self.E,
                    alpha=self._sub_alpha,
                    sink_norm=sink_norm 
                )
            
            candidate_loss = torch.linalg.norm(candidate_nu) ** 2

            if self.pmd_eta_mode == "backtracking":
                trial_eta = eta_t
                trial = 0
                while candidate_loss > actor_loss and trial < self.pmd_backtrack_max_trials:
                    trial_eta *= self.pmd_backtrack_factor
                    trial_eta = float(np.clip(trial_eta, self.pmd_eta_min, self.pmd_eta_max))
                    candidate_coeff = self.gradient_coeff + trial_eta * grad_update
                    candidate_pi = self._policy_from_H(sub_H.T, coeff=candidate_coeff)
                    # candidate_M = sub_H * (self.E @ candidate_pi.T)
                    candidate_nu = self.distribution_matcher.compute_nu_pi_nystrom_memory_efficient(
                            phi_all_obs=self._phi_all_obs,
                            phi_sub_next_obs = self._phi_sub_next,
                            psi_sub_obs_action = self._psi_sub,
                            psi_all_obs_action = self._psi_all,
                            H = sub_H,
                            pi = candidate_pi,
                            E = self.E,
                            alpha=self._sub_alpha,
                            sink_norm=sink_norm 
                        )                    
                    candidate_loss = torch.linalg.norm(candidate_nu) ** 2
                    trial += 1
                eta_t = trial_eta

            self.current_eta = eta_t
            self.gradient_coeff = candidate_coeff
            prev_gradient_coeff = grad_update.clone()
            self.pi = candidate_pi
            actor_loss = candidate_loss

            if actor_loss < best_loss:
                best_loss = actor_loss
                best_pi = self.pi.clone()
                best_coeff = self.gradient_coeff.clone()

            if iteration % 10 == 0 or iteration == self.pmd_steps - 1:
                print(f"  PMD Iteration {iteration}, Actor loss: {actor_loss}, eta: {self.current_eta:.6g}")

        if self.pmd_best_iterate:
            self.pi = best_pi
            self.gradient_coeff = best_coeff
            actor_loss = best_loss
            

        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss
            metrics['actor_eta'] = float(self.current_eta)
            metrics['actor_best_loss'] = float(best_loss)
            metrics['sink_norm'] = float(sink_norm)
   
        return metrics


    def _track_gradient_norms(self, gradient, rewards, step):
        """
        Track gradient norms for samples with different reward values.
        
        Args:
            gradient: [batch_size+1, 1] gradient tensor
            rewards: [batch_size] reward tensor
            step: current training step
        """
        with torch.no_grad():
            # Compute per-sample gradient norm (excluding the last augmented dimension)
            grad_per_sample = gradient[:-1]  # [batch_size, 1]
            
            # Group by reward value
            for reward_val, reward_key in [(1.0, '+1'), (-1.0, '-1'), (0.0, '0')]:
                mask = (rewards == reward_val)
                if mask.sum() > 0:
                    # Get gradients for this reward type
                    grads_for_reward = grad_per_sample[mask]
                    
                    print(f"shapes for reward {reward_key}: grads {grads_for_reward.shape}, rewards {rewards[mask].shape}")
                    # Compute norms
                    norms = torch.norm(grads_for_reward.reshape(grads_for_reward.shape[0], -1), dim=1).cpu().numpy()
                    
                    # Store individual samples (up to max)
                    for norm_val in norms:
                        if len(self.gradient_samples[reward_key]) < self.max_samples_per_reward:
                            self.gradient_samples[reward_key].append((step, float(norm_val)))
                        else:
                            # Replace oldest sample
                            self.gradient_samples[reward_key].pop(0)
                            self.gradient_samples[reward_key].append((step, float(norm_val)))
            
            # Compute statistics for this step
            for reward_key in ['+1', '-1', '0']:
                # Get norms from current samples at this step
                current_norms = [norm for s, norm in self.gradient_samples[reward_key] if s == step]
                if len(current_norms) > 0:
                    mean_norm = np.mean(current_norms)
                    std_norm = np.std(current_norms) if len(current_norms) > 1 else 0.0
                    self.gradient_norm_history[reward_key].append((step, mean_norm, std_norm))
                    
                    print(f"  Reward {reward_key}: {len(current_norms)} samples, "
                          f"mean_norm={mean_norm:.6f}, std={std_norm:.6f}")

    
    def _cache_features(self, obs, action, next_obs, encoder=None, sub_obs=None, sub_action=None, sub_next_obs=None):
        """Pre-compute and cache dataset features."""
        encoder = self.encoder if encoder is None else encoder
       
        with torch.no_grad():
    
            self._phi_all_obs = self._encode_with_module(encoder, obs, project=True)
            self._phi_all_next = self._encode_with_module(encoder, next_obs, project=True)

            action = action #.cpu()
            self._psi_all = self._encode_state_action(self._phi_all_obs, action) #.cpu()
            self._all_actions = action.long().reshape(-1).detach().cpu()
           
            self._alpha = torch.zeros((self._phi_all_next.shape[0], 1), device=self.device)  # [n, 1]
    
            self._alpha[0] = 1.0  # set alpha to 1.0 for the first state
            self.E = F.one_hot(
                action, 
                self.n_actions,
            ).reshape(-1, self.n_actions).to(dtype=self.compute_dtype, device=self.device)

            # ** AUGMENTATION STEP **
            # ψ and Φ are augmented with an additional zero dimension
            zeros_col = torch.zeros(*self._psi_all.shape[:-1], 1, device=self._psi_all.device)
            self._psi_all = torch.cat([self._psi_all, zeros_col], dim=-1)

            zero_col = torch.zeros(*self._phi_all_next.shape[:-1], 1, device=self._phi_all_next.device)
            self._phi_all_next = torch.cat([self._phi_all_next, zero_col], dim=-1)

            zero_col = torch.zeros(*self._phi_all_obs.shape[:-1], 1, device=self._phi_all_obs.device)
            self._phi_all_obs = torch.cat([self._phi_all_obs, zero_col], dim=-1)

            if sub_obs is not None and sub_next_obs is not None and sub_action is not None:
                self._phi_sub_obs = self._encode_with_module(encoder, sub_obs, project=True)
                self._phi_sub_next = self._encode_with_module(encoder, sub_next_obs, project=True)
                self._sub_actions = sub_action.long().reshape(-1).detach().cpu()

                self._psi_sub = self._encode_state_action(self._phi_sub_obs, sub_action)

                zeros_col_sub_next = torch.zeros(*self._phi_sub_next.shape[:-1], 1, device=self._phi_sub_next.device)
                self._phi_sub_next = torch.cat([self._phi_sub_next, zeros_col_sub_next], dim=-1)

                zero_col_sub_obs = torch.zeros(*self._phi_sub_obs.shape[:-1], 1, device=self._phi_sub_obs.device)
                self._phi_sub_obs = torch.cat([self._phi_sub_obs, zero_col_sub_obs], dim=-1)

                zero_col_sub_psi = torch.zeros(*self._psi_sub.shape[:-1], 1, device=self._psi_sub.device)
                self._psi_sub = torch.cat([self._psi_sub, zero_col_sub_psi], dim=-1)

                self._sub_alpha = torch.zeros((self._phi_sub_next.shape[0], 1), device=self.device)  # [m, 1]
                self._sub_alpha[0] = 1.0  # set alpha to 1.0 for the first state

            print(f"dimensions after augmentation: psi_all {self._psi_all.shape}, phi_all_next {self._phi_all_next.shape}, phi_all_obs {self._phi_all_obs.shape}")

    def _append_zero_feature_column(self, tensor):
        zeros_col = torch.zeros(*tensor.shape[:-1], 1, device=tensor.device, dtype=tensor.dtype)
        return torch.cat([tensor, zeros_col], dim=-1)

    def _cache_encoded_features(self, encoded_full, encoded_sub=None):
        with torch.no_grad():
            self._phi_all_obs = self._append_zero_feature_column(encoded_full["phi_obs"])
            self._phi_all_next = self._append_zero_feature_column(encoded_full["phi_next"])
            self._psi_all = self._append_zero_feature_column(encoded_full["psi"])

            self._alpha = torch.zeros((self._phi_all_next.shape[0], 1), device=self.device, dtype=self._phi_all_next.dtype)
            self._alpha[0] = 1.0

            self.E = encoded_full["E"].to(dtype=self.compute_dtype, device=self.device)
            self._all_actions = torch.argmax(encoded_full["E"], dim=1).long().detach().cpu()

            if encoded_sub is not None:
                self._phi_sub_obs = self._append_zero_feature_column(encoded_sub["phi_obs"])
                self._phi_sub_next = self._append_zero_feature_column(encoded_sub["phi_next"])
                self._psi_sub = self._append_zero_feature_column(encoded_sub["psi"])
                self._sub_actions = torch.argmax(encoded_sub["E"], dim=1).long().detach().cpu()

                self._sub_alpha = torch.zeros((self._phi_sub_next.shape[0], 1), device=self.device, dtype=self._phi_sub_next.dtype)
                self._sub_alpha[0] = 1.0

            print(f"dimensions after augmentation: psi_all {self._psi_all.shape}, phi_all_next {self._phi_all_next.shape}, phi_all_obs {self._phi_all_obs.shape}")

    def _make_actor_batch(self, obs, action, next_obs, reward):
        return (
            obs,
            action,
            next_obs,
            reward.reshape(obs.shape[0], -1),
        )

    def _slice_actor_batch(self, actor_batch, index):
        return tuple(field[index] for field in actor_batch)

    def _concat_actor_batches(self, actor_batches, max_samples):
        if not actor_batches:
            raise RuntimeError("No replay samples available for actor update")
        actor_batch = tuple(
            torch.cat([batch[field_idx] for batch in actor_batches], dim=0)
            for field_idx in range(len(actor_batches[0]))
        )
        return self._slice_actor_batch(actor_batch, slice(0, max_samples))

    def _load_first_actor_transition(self, replay_buffer=None, fallback_actor_batch=None):
        if replay_buffer is not None and hasattr(replay_buffer, "get_first_transition"):
            first_batch = replay_buffer.get_first_transition()
            first_obs, first_action, first_reward, _, first_next_obs = utils.to_torch(
                first_batch[:5],
                self.device,
            )
            return self._make_actor_batch(first_obs, first_action, first_next_obs, first_reward)

        if fallback_actor_batch is None:
            return None
        return self._slice_actor_batch(fallback_actor_batch, slice(0, 1))

    def _replace_first_actor_transition(self, actor_batch, first_actor_transition):
        if first_actor_transition is None:
            return actor_batch

        actor_batch = tuple(field.clone() for field in actor_batch)
        for field_idx, first_field in enumerate(first_actor_transition):
            actor_batch[field_idx][:1] = first_field.to(actor_batch[field_idx].device)
        return actor_batch

    def _is_cuda_oom(self, error):
        return (
            isinstance(error, RuntimeError)
            and "out of memory" in str(error).lower()
            and "cuda" in str(error).lower()
        )

    def _encoded_batch_size(self, encoded):
        return next(iter(encoded.values())).shape[0]

    def _concat_encoded_batches(self, encoded_batches):
        return {
            key: torch.cat([batch[key] for batch in encoded_batches], dim=0)
            for key in encoded_batches[0].keys()
        }

    def _slice_raw_transition_batch(self, transitions, index):
        return tuple(field[index] for field in transitions)

    def _encode_actor_transition_batch(self, transitions):
        obs, action, reward, _, next_obs = utils.to_torch(transitions[:5], self.device)
        reward = reward.reshape(obs.shape[0], -1)
        with torch.no_grad():
            phi_obs = self._encode_with_module(self.policy_encoder, obs, project=True)
            phi_next = self._encode_with_module(self.policy_encoder, next_obs, project=True)
            psi = self._encode_state_action(phi_obs, action)
            action_onehot = F.one_hot(
                action.long(),
                self.n_actions,
            ).reshape(-1, self.n_actions).to(dtype=self.compute_dtype, device=self.device)
        return {
            "phi_obs": phi_obs,
            "phi_next": phi_next,
            "psi": psi,
            "E": action_onehot,
            "reward": reward,
        }

    def _encode_actor_transition_batch_with_retries(self, transitions, splits_left=None):
        splits_left = self.encoded_fifo_cuda_oom_splits if splits_left is None else splits_left
        batch_size = transitions[0].shape[0]
        try:
            return self._encode_actor_transition_batch(transitions)
        except RuntimeError as error:
            if not self._is_cuda_oom(error) or splits_left <= 0 or batch_size <= 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            midpoint = batch_size // 2
            left = self._slice_raw_transition_batch(transitions, slice(0, midpoint))
            right = self._slice_raw_transition_batch(transitions, slice(midpoint, None))
            encoded_left = self._encode_actor_transition_batch_with_retries(left, splits_left - 1)
            encoded_right = self._encode_actor_transition_batch_with_retries(right, splits_left - 1)
            return self._concat_encoded_batches([encoded_left, encoded_right])

    def _insert_first_transition_if_available(self, replay_buffer):
        if self._encoded_actor_fifo.has_first:
            return
        if replay_buffer is None or not hasattr(replay_buffer, "get_first_transition"):
            return
        try:
            first_transition = replay_buffer.get_first_transition()
        except RuntimeError:
            return
        encoded = self._encode_actor_transition_batch_with_retries(first_transition)
        self._encoded_actor_fifo.add(np.array([0], dtype=np.int64), encoded)

    def _update_encoded_actor_fifo(self, replay_buffer):
        if replay_buffer is None or not hasattr(replay_buffer, "get_new_transitions_since"):
            return False

        self._sync_policy_encoder()
        inserted = 0
        encode_batch_size = max(1, self.encoded_fifo_encode_batch_size)

        while True:
            transition_ids, transitions = replay_buffer.get_new_transitions_since(
                self._encoded_fifo_replay_marker,
                limit=encode_batch_size,
            )
            if transition_ids is None:
                break

            # Encode only the new replay-buffer transitions, then immediately
            # acknowledge them so raw pending data can be released by storage.
            encoded = self._encode_actor_transition_batch_with_retries(transitions)
            self._encoded_actor_fifo.add(transition_ids, encoded)
            self._encoded_fifo_replay_marker = int(transition_ids[-1])
            if hasattr(replay_buffer, "mark_transitions_encoded"):
                replay_buffer.mark_transitions_encoded(self._encoded_fifo_replay_marker)
            inserted += int(len(transition_ids))

        self._insert_first_transition_if_available(replay_buffer)
        return inserted > 0 or len(self._encoded_actor_fifo) > 0

    def _sample_encoded_actor_data(self, size, include_first):
        encoded = self._encoded_actor_fifo.sample(
            int(size),
            self.device,
            include_first=include_first,
        )
        return encoded, encoded.get("reward")

    def _all_encoded_actor_data(self, include_first=True):
        encoded = self._encoded_actor_fifo.all(
            self.device,
            include_first=include_first,
        )
        return encoded, encoded.get("reward")

    def _load_actor_batch_from_replay_iter(
            self,
            replay_iter,
            obs,
            action,
            next_obs,
            reward,
            max_samples,
            replay_buffer=None):
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")

        actor_batches = [self._make_actor_batch(obs, action, next_obs, reward)]
        collected = obs.shape[0]

        while collected < max_samples:
            batch = next(replay_iter)
            obs_b, action_b, reward_b, _, next_obs_b = utils.to_torch(batch, self.device)
            actor_batch = self._make_actor_batch(obs_b, action_b, next_obs_b, reward_b)
            actor_batches.append(actor_batch)
            collected += obs_b.shape[0]

        actor_batch = self._concat_actor_batches(actor_batches, max_samples)
        first_actor_transition = self._load_first_actor_transition(
            replay_buffer=replay_buffer,
            fallback_actor_batch=actor_batch,
        )
        return self._replace_first_actor_transition(actor_batch, first_actor_transition)

    def _load_actor_subsample_from_replay_iter(
            self,
            replay_iter,
            max_samples,
            replay_buffer=None,
            fallback_actor_batch=None):
        if max_samples <= 0:
            raise ValueError("subsamples must be positive when provided")

        first_actor_transition = self._load_first_actor_transition(
            replay_buffer=replay_buffer,
            fallback_actor_batch=fallback_actor_batch,
        )
        if first_actor_transition is None:
            raise RuntimeError("Could not build a subsample with the first transition in position 0")
        if max_samples == 1:
            return first_actor_transition

        actor_batches = []
        collected = 0
        remaining_samples = max_samples - 1

        while collected < remaining_samples:
            batch = next(replay_iter)
            obs_b, action_b, reward_b, _, next_obs_b = utils.to_torch(batch, self.device)
            actor_batch = self._make_actor_batch(obs_b, action_b, next_obs_b, reward_b)

            if actor_batch[0].shape[0] > 1:
                actor_batch = self._slice_actor_batch(actor_batch, slice(1, None))
            if actor_batch[0].shape[0] == 0:
                continue
            actor_batches.append(actor_batch)
            collected += actor_batch[0].shape[0]

        sampled_actor_batch = self._concat_actor_batches(actor_batches, remaining_samples)
        return self._concat_actor_batches(
            [first_actor_transition, sampled_actor_batch],
            max_samples,
        )

    def _get_actor_update_data(self, replay_iter, obs, action, next_obs, reward, replay_buffer=None):
        if self._update_encoded_actor_fifo(replay_buffer):
            if self.subsamples is None:
                encoded_full, rewards = self._sample_encoded_actor_data(
                    self.batch_size_actor,
                    include_first=True,
                )
                return encoded_full, rewards, None, None

            if self.subsamples <= 0:
                raise ValueError("subsamples must be positive when provided")

            # Nyström uses the FIFO as the full support and samples only the
            # landmark/subsample set. The FIFO capacity controls the maximum
            # retained support size; batch_size_actor is only for non-Nyström.
            encoded_full, rewards = self._all_encoded_actor_data(include_first=True)
            encoded_sub, sub_rewards = self._sample_encoded_actor_data(
                int(self.subsamples),
                include_first=True,
            )
            return encoded_full, rewards, encoded_sub, sub_rewards

        full_actor_batch = self._load_actor_batch_from_replay_iter(
            replay_iter,
            obs,
            action,
            next_obs,
            reward,
            max_samples=self.batch_size_actor,
            replay_buffer=replay_buffer,
        )

        if self.subsamples is None:
            return (*full_actor_batch, None, None, None, None)

        if self.subsamples <= 0:
            raise ValueError("subsamples must be positive when provided")

        if self.subsamples >= full_actor_batch[0].shape[0]:
            subsampled_actor_batch = full_actor_batch
        else:
            subsampled_actor_batch = self._load_actor_subsample_from_replay_iter(
                replay_iter,
                max_samples=int(self.subsamples),
                replay_buffer=replay_buffer,
                fallback_actor_batch=full_actor_batch,
            )

        return (*full_actor_batch, *subsampled_actor_batch)

    def _build_debug_visualizer_batch(self, obs, max_observations=2000):
        if obs is None:
            return None, None

        max_observations = min(max_observations, obs.shape[0])
        visualizer_obs = obs[:max_observations]
        with torch.no_grad():
            visualizer_z = self._encode_with_module(
                self.policy_encoder,
                visualizer_obs,
                project=True,
            )
        return visualizer_obs, visualizer_z

    def _compute_mean_action_probs_deviation(self, action_probs: np.ndarray) -> float:
        """
        Compute mean deviation of action probabilities from uniform distribution.
        
        Args:
            action_probs: [batch_size, n_actions] action probabilities from current policy
            
        Returns:
            Mean absolute deviation from uniform (1/n_actions)
        """
        uniform_prob = 1.0 / self.n_actions
        # Average over batch, then compute mean absolute deviation across actions
        mean_probs = np.mean(action_probs, axis=0)  # [n_actions]
        deviation = np.mean(np.abs(mean_probs - uniform_prob))
        return deviation
    
    def plot_gradient_norm_by_reward(self, save_dir: str = './gradient_plots'):
        """
        Plot gradient norms over time, separated by reward value.
        
        Args:
            save_dir: Directory to save the plot
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Check if we have data
        has_data = any(len(self.gradient_norm_history[key]) > 0 for key in self.gradient_norm_history)
        if not has_data:
            print("No gradient norm history to plot yet")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Gradient Norms by Reward Type', fontsize=16, fontweight='bold')
        
        colors = {'+1': 'green', '-1': 'red', '0': 'blue'}
        labels = {'+1': 'Reward +1', '-1': 'Reward -1', '0': 'Reward 0'}
        
        # Plot 1: Mean gradient norms over time
        ax1 = axes[0, 0]
        for reward_key in ['+1', '-1', '0']:
            if len(self.gradient_norm_history[reward_key]) > 0:
                steps, means, stds = zip(*self.gradient_norm_history[reward_key])
                ax1.plot(steps, means, color=colors[reward_key], linewidth=2, 
                        label=labels[reward_key], alpha=0.8)
                means_arr = np.array(means)
                stds_arr = np.array(stds)
                ax1.fill_between(steps, means_arr - stds_arr, means_arr + stds_arr, 
                                color=colors[reward_key], alpha=0.2)
        
        ax1.set_xlabel('Training Steps', fontsize=11)
        ax1.set_ylabel('Mean Gradient Norm', fontsize=11)
        ax1.set_title('Mean Gradient Norms Over Time', fontsize=12, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Current distribution of gradient norms (latest samples)
        ax2 = axes[0, 1]
        current_samples = {k: [norm for _, norm in v[-50:]] for k, v in self.gradient_samples.items() if len(v) > 0}
        
        if any(len(samples) > 0 for samples in current_samples.values()):
            positions = []
            data_to_plot = []
            tick_labels = []
            box_colors = []
            
            for i, (reward_key, samples) in enumerate(current_samples.items()):
                if len(samples) > 0:
                    positions.append(i)
                    data_to_plot.append(samples)
                    tick_labels.append(labels[reward_key])
                    box_colors.append(colors[reward_key])
            
            bp = ax2.boxplot(data_to_plot, positions=positions, patch_artist=True,
                           widths=0.6, showfliers=True)
            
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            ax2.set_xticks(positions)
            ax2.set_xticklabels(tick_labels)
            ax2.set_ylabel('Gradient Norm', fontsize=11)
            ax2.set_title('Current Gradient Norm Distribution\n(Last 50 samples per reward)', 
                         fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
        else:
            ax2.text(0.5, 0.5, 'Not enough data yet', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
        
        # Plot 3: Sample counts over time
        ax3 = axes[1, 0]
        for reward_key in ['+1', '-1', '0']:
            if len(self.gradient_norm_history[reward_key]) > 0:
                steps, _, _ = zip(*self.gradient_norm_history[reward_key])
                # Count cumulative samples at each step
                cumulative_counts = []
                for step in steps:
                    count = len([s for s, _ in self.gradient_samples[reward_key] if s <= step])
                    cumulative_counts.append(count)
                
                ax3.plot(steps, cumulative_counts, color=colors[reward_key], 
                        linewidth=2, label=labels[reward_key], alpha=0.8)
        
        ax3.set_xlabel('Training Steps', fontsize=11)
        ax3.set_ylabel('Cumulative Sample Count', fontsize=11)
        ax3.set_title('Sample Collection Progress', fontsize=12, fontweight='bold')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(self.max_samples_per_reward, color='black', linestyle='--', 
                   linewidth=1, alpha=0.5, label=f'Max ({self.max_samples_per_reward})')
        
        # Plot 4: Ratio comparison
        ax4 = axes[1, 1]
        if len(self.gradient_norm_history['+1']) > 0 and len(self.gradient_norm_history['0']) > 0:
            # Get aligned steps
            steps_pos = [s for s, _, _ in self.gradient_norm_history['+1']]
            steps_zero = [s for s, _, _ in self.gradient_norm_history['0']]
            steps_neg = [s for s, _, _ in self.gradient_norm_history['-1']]
            
            common_steps = sorted(set(steps_pos) & set(steps_zero))
            
            if len(common_steps) > 0:
                ratios_pos_zero = []
                for step in common_steps:
                    mean_pos = [m for s, m, _ in self.gradient_norm_history['+1'] if s == step][0]
                    mean_zero = [m for s, m, _ in self.gradient_norm_history['0'] if s == step][0]
                    if mean_zero > 1e-10:
                        ratios_pos_zero.append(mean_pos / mean_zero)
                    else:
                        ratios_pos_zero.append(np.nan)
                
                ax4.plot(common_steps, ratios_pos_zero, color='purple', linewidth=2, 
                        label='||∇(r=+1)|| / ||∇(r=0)||', alpha=0.8)
                ax4.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            
            # Add negative reward ratio if available
            common_steps_neg = sorted(set(steps_neg) & set(steps_zero))
            if len(common_steps_neg) > 0:
                ratios_neg_zero = []
                for step in common_steps_neg:
                    mean_neg = [m for s, m, _ in self.gradient_norm_history['-1'] if s == step][0]
                    mean_zero = [m for s, m, _ in self.gradient_norm_history['0'] if s == step][0]
                    if mean_zero > 1e-10:
                        ratios_neg_zero.append(mean_neg / mean_zero)
                    else:
                        ratios_neg_zero.append(np.nan)
                
                ax4.plot(common_steps_neg, ratios_neg_zero, color='orange', linewidth=2,
                        label='||∇(r=-1)|| / ||∇(r=0)||', alpha=0.8)
            
            ax4.set_xlabel('Training Steps', fontsize=11)
            ax4.set_ylabel('Gradient Norm Ratio', fontsize=11)
            ax4.set_title('Relative Gradient Magnitudes', fontsize=12, fontweight='bold')
            ax4.legend(loc='best')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Not enough data for comparison', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=12)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'gradient_norms_by_reward.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Gradient norm plot saved to: {save_path}")
        
        # Print summary statistics
        print("\n=== Gradient Norm Summary ===")
        for reward_key in ['+1', '-1', '0']:
            if len(self.gradient_samples[reward_key]) > 0:
                norms = [norm for _, norm in self.gradient_samples[reward_key]]
                print(f"Reward {reward_key:>2}: n={len(norms):3d}, "
                      f"mean={np.mean(norms):.6f}, std={np.std(norms):.6f}, "
                      f"min={np.min(norms):.6f}, max={np.max(norms):.6f}")


    def plot_policy_deviation_history(self, save_dir: str = './policy_plots'):
        """
        Plot cumulative history of policy deviation from uniform distribution.
        
        Args:
            save_dir: Directory to save the plot
        """
        if len(self.policy_deviation_history) == 0:
            print("No policy deviation history to plot yet")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        steps, deviations = zip(*self.policy_deviation_history)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(steps, deviations, color='blue', linewidth=2, alpha=0.8, label='Policy Deviation')
        ax.axhline(0, color='green', linestyle='--', linewidth=1.5, label='Uniform Policy', alpha=0.7)
        
        # Theoretical maximum (when policy is deterministic on one action)
        max_deviation = (self.n_actions - 1) / self.n_actions
        ax.axhline(max_deviation, color='red', linestyle='--', linewidth=1.5, 
                   label=f'Deterministic Policy ({max_deviation:.3f})', alpha=0.7)
        
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Mean |P(a) - 1/|A||', fontsize=12)
        ax.set_title('Policy Concentration Over Time\n(Deviation from Uniform Distribution)', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add current value annotation
        if len(deviations) > 0:
            current_val = deviations[-1]
            ax.text(0.02, 0.98, f'Current: {current_val:.4f}', 
                    transform=ax.transAxes, fontsize=11,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'policy_deviation_history.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Policy deviation plot saved to: {save_path}")

    def aug_and_encode(self, obs, project=False):
        obs = self.aug(obs)
        if not self.embeddings:
            return self.encoder(obs)
        if project:
            return self.encoder.encode_and_project(obs)
        else:
            return self.encoder(obs)

    def _sample_batch(self, replay_iter):
        batch = next(replay_iter)
        return utils.to_torch(batch, self.device)

    def update(self, replay_iter, step, replay_buffer=None):
        metrics = dict()

        if step % self.update_every_steps != 0 and self._is_T_sufficiently_initialized(step) is True:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        if self.use_tb or self.use_wandb:
            metrics['batch_reward'] = reward.mean().item()
        if self.embeddings:
            metrics.update(self.update_encoders(obs, action, next_obs, reward))

        # If T is not sufficiently initialized, skip actor update
        if self._is_T_sufficiently_initialized(step) is False:   
            metrics['actor_loss'] = 100.0  # dummy value
            return metrics
        
        # In ideal mode, we can update actor immediately
        if  step % self.update_actor_every_steps == 0 or step == self.num_expl_steps + self.T_init_steps: # or self.ideal:  
            actor_update_data = self._get_actor_update_data(
                replay_iter,
                obs,
                action,
                next_obs,
                reward,
                replay_buffer=replay_buffer,
            )
            if isinstance(actor_update_data[0], dict):
                encoded_full, all_reward_actor, encoded_sub, subsampled_reward_actor = actor_update_data
                utils.ColorPrint.red(
                    f"samples for actor update: full {self._encoded_batch_size(encoded_full)}, "
                    f"subsampled {self._encoded_batch_size(encoded_sub) if encoded_sub is not None else 'N/A'}"
                )
            else:
                (
                    all_obs_actor,
                    all_action_actor,
                    all_next_obs_actor,
                    all_reward_actor,
                    subsampled_obs_actor,
                    subsampled_action_actor,
                    subsampled_next_obs_actor,
                    subsampled_reward_actor,
                ) = actor_update_data
                encoded_full = None
                encoded_sub = None
                utils.ColorPrint.red(
                    f"samples for actor update: full {all_obs_actor.shape[0]}, "
                    f"subsampled {subsampled_obs_actor.shape[0] if subsampled_obs_actor is not None else 'N/A'}"
                )

            if self.subsamples is None and encoded_full is not None:
                metrics.update(
                    self.update_actor(
                        step=step,
                        rewards=all_reward_actor,
                        encoded_full=encoded_full,
                    )
                )
            elif self.subsamples is None:
                metrics.update(
                    self.update_actor(
                        all_obs_actor,
                        all_action_actor,
                        all_next_obs_actor,
                        step=step,
                        rewards=all_reward_actor,
                    )
                )
            elif encoded_full is not None:
                metrics.update(
                    self.update_actor_nystrom(
                        None,
                        None,
                        None,
                        step=step,
                        rewards=all_reward_actor,
                        sub_rewards=subsampled_reward_actor,
                        encoded_full=encoded_full,
                        encoded_sub=encoded_sub,
                    )
                )
            else:
                metrics.update(
                    self.update_actor_nystrom(
                        all_obs_actor,
                        all_action_actor,
                        all_next_obs_actor,
                        step=step,
                        rewards=all_reward_actor,
                        sub_obs=subsampled_obs_actor,
                        sub_action=subsampled_action_actor,
                        sub_next_obs=subsampled_next_obs_actor,
                        sub_rewards=subsampled_reward_actor,
                    )
                )


        if self.debug_visualizer is not None:
            param_text = (
                f"Step: {step}\n"
                f"γ = {self.discount}\n"
                f"η = {self.current_eta}\n"
                f"λ = {self.lambda_reg}\n"
                f"sink norm = {utils.schedule(self.sink_schedule, step):.6f}\n"
                f"PMD steps = {self.pmd_steps}\n"
                f"subsamples = {self.subsamples if self.subsamples is not None else 'all'}\n"
            )

            if  step % self.update_actor_every_steps == 0: #step % 10000 == 0 
                visualizer_obs, visualizer_z = self._build_debug_visualizer_batch(obs)
                metrics.update(
                    self.debug_visualizer.save(
                        step=step,
                        obs_batch=visualizer_obs,
                        z_batch=visualizer_z,
                        param_text=param_text,
                    )
                )
            
                with torch.no_grad():
                
                    if len(self.current_action_probs) == 0:
                        return metrics
                    current_action_probs = np.array(self.current_action_probs)  # [num_recorded, n_actions]
                    # Compute mean deviation from uniform
                    mean_deviation = self._compute_mean_action_probs_deviation(current_action_probs)
                    
                    # Store in history
                    self.policy_deviation_history.append((step, mean_deviation))
                    
                    # Also store the mean action probabilities
                    mean_probs = np.mean(current_action_probs, axis=0)
                    self.action_probs_history.append((step, mean_probs))
                    
                    # Log to metrics
                    metrics['policy_deviation_from_uniform'] = mean_deviation
                    print(f"Policy deviation from uniform: {mean_deviation:.4f} (0=uniform, {(self.n_actions-1)/self.n_actions:.3f}=deterministic)")
                    self.current_action_probs = []  # Clear after processing
                self.plot_policy_deviation_history(save_dir=os.path.join(os.getcwd(), 'policy_plots'))
                self.plot_gradient_norm_by_reward(save_dir=os.path.join(os.getcwd(), 'gradient_plots'))

        return metrics
