from collections import OrderedDict
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
torch.set_default_dtype(torch.float32)
from agent.utils import InternalDatasetFIFO
from PIL import Image
from sklearn.manifold import TSNE
import logging
# set logging level to info
import logging

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
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.project_sa= nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            # nn.ReLU(inplace=True),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=False),
            # nn.Linear(input_dim, output_dim, bias=False)

        )
    
    def forward(self, encoded_state_action: torch.Tensor) -> torch.Tensor:
        return self.project_sa(encoded_state_action)


    

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

    def compute_BM(
            self,
            M: torch.Tensor,  # [N, N] forward operator
            psi: torch.Tensor  # [N, d*|A|] state-action features
        ) -> torch.Tensor:
        """Compute BM = (ψψᵀ + λI)⁻¹M."""
        N = psi.shape[0]
        identity = torch.eye(N, device=self.device)
        gram_matrix = psi @ psi.T + self.lambda_reg * identity

        
        L = torch.linalg.cholesky(gram_matrix)
        BM = torch.cholesky_solve(M, L)
        return BM
    
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
                obs = obs.float() / 255.0
                h = self.conv(obs)
                h = self.adaptive_pool(h)
                h = h.reshape(h.size(0), -1)
            else:
                # Flatten state to [B, state_dim]
                obs = obs.float()
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
        save_path = self.save_dir / f'exploration_metrics_{step}.png'
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
class EmbeddingDistributionVisualizerV2:
    """Visualizer for embedding-based distribution matching results (adapted for v2)."""
    def __init__(self, agent):
        """
        Initialize visualizer with agent reference.
        
        Args:
            agent: DistMatchingEmbeddingAgentv2 instance
        """
        self.agent = agent
        self.env = agent.env
        self.n_states = self.env.n_states
        self.n_actions = agent.n_actions

        # Get grid dimensions
        valid_cells = [cell for cell in self.env.cells if cell != self.env.DEAD_STATE]
        min_x = min(cell[0] for cell in valid_cells)
        min_y = min(cell[1] for cell in valid_cells)
        max_x = max(cell[0] for cell in valid_cells)
        max_y = max(cell[1] for cell in valid_cells)

        valid_ids = [self.env.state_to_idx[cell] for cell in valid_cells]
        print(f"self.n_states: {self.n_states}, len(valid_ids): {len(valid_ids)}")
        self.all_state_ids_one_hot = torch.eye(self.n_states)[valid_ids].to(self.agent.device)

        if hasattr(self.env, 'lava') and self.env.lava:
            min_x = min(min_x, -1)
            min_y = min(min_y, -1)
        
        self.min_x = min_x
        self.min_y = min_y
        self.grid_width = max_x - min_x + 1
        self.grid_height = max_y - min_y + 1
        
        # Action symbols and colors - support both 4 and 8 actions
        if self.n_actions == 4:
            self.action_symbols = ['↑', '↓', '←', '→']  # 0=up, 1=down, 2=left, 3=right
            self.action_names = ['Up', 'Down', 'Left', 'Right']
            self.action_colors = ['red', 'blue', 'green', 'orange']
        elif self.n_actions == 8:
            self.action_symbols = ['→', '↘', '↓', '↙', '←', '↖', '↑', '↗']
            self.action_names = ['Right', 'Down-Right', 'Down', 'Down-Left', 'Left', 'Up-Left', 'Up', 'Up-Right']
            self.action_colors = [
                'orange',      # 0: right
                'salmon',   # 1: down-right
                'blue',     # 2: down
                'cyan',     # 3: down-left
                'green',    # 4: left
                'lime',     # 5: up-left
                'red',   # 6: up
                'gold'      # 7: up-right
            ]
        elif self.n_actions == 2:
            self.action_symbols = ['→', '↓']
            self.action_names = ['Right', 'Down']
            self.action_colors = ['red', 'blue']
        else:
            self.action_symbols = [str(i) for i in range(self.n_actions)]
            self.action_names = [f'Action {i}' for i in range(self.n_actions)]
            self.action_colors = plt.cm.tab20(np.linspace(0, 1, self.n_actions))
        
        # Pre-render all state observations if using pixel observations
        if self.agent.obs_type == 'pixels':
            print("Pre-rendering all state images for correlation matrix...")
            self._prerendered_states = []
            
            render_resolution = getattr(self.agent.wrapped_env, 'render_resolution', 224)
            frame_stack = self.agent.obs_shape[0] // 3
            
            for s_idx in range(self.n_states):
                if s_idx % 10 == 0:
                    print(f"  Rendering state {s_idx}/{self.n_states}...")
                
                position = self.env.idx_to_state[s_idx]
                image = self.env.render_from_position(position)
                
                # Resize if needed
                if image.shape[:2] != (render_resolution, render_resolution):
                    pil_img = Image.fromarray(image.astype(np.uint8))
                    pil_img_resized = pil_img.resize(
                        (render_resolution, render_resolution), 
                        Image.LANCZOS
                    )
                    image = np.array(pil_img_resized)
                
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
        
    def _state_dist_to_grid(self, nu: np.ndarray) -> np.ndarray:
        """Convert state distribution vector to 2D grid."""
        grid = np.zeros((self.grid_height, self.grid_width))
        
        for s_idx in range(self.n_states):
            cell = self.env.idx_to_state[s_idx]
            x, y = cell[0] - self.min_x, cell[1] - self.min_y
            grid[y, x] = nu[s_idx]
        
        return grid
    
    def _compute_initial_distribution(self) -> np.ndarray:
        """Compute initial distribution using φ(unique_states) @ alpha."""
        with torch.no_grad():
            if self.agent.obs_type == 'pixels':
                # Use pre-rendered images
                enc_all_states = self.agent.aug_and_encode(self._prerendered_states, project=True).detach() #.cpu()
            else:
                # Use one-hot encodings
                all_states = self.all_state_ids_one_hot.to(self.agent.device)
                enc_all_states = self.agent.encoder(all_states)
            
            if hasattr(self.agent, '_alpha') and self.agent._alpha is not None:
                alpha = self.agent._alpha
                phi_all_next = self.agent._phi_all_next
                
                # Add augmented dimension to encoded states
                zero_col = torch.zeros(*enc_all_states.shape[:-1], 1, device=enc_all_states.device)
                enc_all_states_aug = torch.cat([enc_all_states, zero_col], dim=-1) #.cpu()
                
                print(f"device of enc_all_states_aug: {enc_all_states_aug.device}, phi_all_next: {phi_all_next.device}, alpha: {alpha.device}")
                kernel = enc_all_states_aug @ phi_all_next.T
                nu_init = kernel @ alpha
            else:
                nu_init = torch.ones(self.n_states, 1) / self.n_states
        return nu_init.flatten().cpu().numpy()
    
    def _compute_current_distribution(self) -> np.ndarray:
        """Compute current occupancy distribution for all states."""
        if self.agent.gradient_coeff is None:
            return np.ones(self.n_states) / self.n_states
        
        nu_current = torch.zeros(self.n_states)
        if self.agent.obs_type == 'pixels':
            # Use pre-rendered images
            enc_all_states = self.agent.aug_and_encode(self._prerendered_states, project=True).detach()
        else:
            # Use one-hot encodings
            all_states = self.all_state_ids_one_hot.to(self.agent.device)
            enc_all_states = self.agent.encoder(all_states).detach()
        
        with torch.no_grad():
            # Add augmented dimension
            zero_col = torch.zeros(*enc_all_states.shape[:-1], 1, device=enc_all_states.device)
            enc_all_states_aug = torch.cat([enc_all_states, zero_col], dim=-1)
            
            H = enc_all_states_aug @ self.agent._phi_all_obs.T
            pi_all = self.agent._policy_from_H(H)
            
            M = H * (self.agent.E @ pi_all.T)
            
            alpha_all = torch.zeros(self.n_states, 1)
            alpha_all[0] = 1.0
            
            nu_current = self.agent.distribution_matcher.compute_nu_pi(
                phi_all_next_obs=self.agent._phi_all_next,
                psi_all_obs_action=self.agent._psi_all,
                K=self.agent.K,
                M=M,
                alpha=alpha_all,
                sink_norm=utils.schedule(self.agent.sink_schedule, 0)
            )
        
        # Normalize
        nu_current = nu_current[:-1].flatten()  # Remove sink state
        nu_current = nu_current / (nu_current.sum() + 1e-10)
        return nu_current.numpy()
    
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
            frame_stack = self.agent.obs_shape[0] // 3  # Assuming RGB (3 channels)
            
            # Get position from state index
            position = self.env.idx_to_state[state_idx]
            
            # Render image from position [H, W, C]
            image = self.env.render_from_position(position)
            
            # Auto-resize if needed
            if image.shape[:2] != (render_resolution, render_resolution):
                # Convert to PIL Image, resize, convert back
                pil_img = Image.fromarray(image.astype(np.uint8))
                pil_img_resized = pil_img.resize(
                    (render_resolution, render_resolution), 
                    Image.LANCZOS
                )
                image = np.array(pil_img_resized)
            
            # Verify channels
            if image.shape[2] != 3:
                raise ValueError(f"Expected 3 channels (RGB), got {image.shape[2]}")
            
            # Convert HWC to CHW format [C, H, W]
            image_chw = image.transpose(2, 0, 1).copy()
            
            # Stack the frame multiple times to match frame_stack
            # The agent expects [C*frame_stack, H, W]
            stacked_image = np.tile(image_chw, (frame_stack, 1, 1))
            
            return stacked_image
        else:
            # For state observations, return one-hot encoding
            obs_onehot = np.eye(self.n_states)[state_idx]
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
            all_states = self.all_state_ids_one_hot.to(self.agent.device)
            if project:
                embeddings = self.agent.encoder.encode_and_project(all_states).detach().cpu().numpy()
            else:
                embeddings = self.agent.encoder(all_states).detach().cpu().numpy()
        
        valid_cells = [cell for cell in self.env.cells if cell != self.env.DEAD_STATE]
        
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
        
        for idx, (cell, embedding_2d) in enumerate(zip(valid_cells, embeddings_2d)):
            ax.scatter(embedding_2d[0], embedding_2d[1], c=[colors[idx]], s=100, alpha=0.7)
            ax.text(embedding_2d[0], embedding_2d[1], f"{cell}", fontsize=8, ha='center', va='center')
        
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
        figsize = (22, 10)
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
        
        gs = fig.add_gridspec(2, 5, hspace=0.35, wspace=0.35)
        
        # Row 1: Initial dist, Policy arrows (bigger), Policy bars (bigger), Correlation matrix
        ax_init = fig.add_subplot(gs[0, 0])
        ax_policy = fig.add_subplot(gs[0, 1:3])  # Span 2 columns
        ax_policy_bars = fig.add_subplot(gs[1, 1:3])  # Span 2 columns
        ax_corr = fig.add_subplot(gs[0, 3:5])  # Span 2 columns
        
        # Row 2: Sample occupancy, State correlations (smaller)
        ax_sample_occ = fig.add_subplot(gs[1, 0])
        ax_state_corr = fig.add_subplot(gs[1, 3:5])  # Span 2 columns
        
        # Compute distributions
        nu_init = self._compute_initial_distribution()
        policy_per_state = self._get_policy_per_state()
        
        # Plot distributions
        self._plot_distribution(ax_init, nu_init, 'Initial Distribution')
        
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
        
        plt.close(fig)

    def _plot_policy_bars_per_cell(self, ax, policy_per_state):
        """Plot policy bars inside each grid cell, similar to action probabilities grid."""
        ax.set_xlim(self.min_x - 0.5, self.min_x + self.grid_width - 0.5)
        ax.set_ylim(self.min_y - 0.5, self.min_y + self.grid_height - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Invert Y axis so (0,0) is top-left
        ax.set_title('Policy Action Probabilities per Cell', fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
        
        # Draw environment structure
        if hasattr(self.env, 'walkable_areas'):
            for area in self.env.walkable_areas:
                rect = Rectangle((area[0], area[1]), area[2], area[3],
                            fill=False, edgecolor='gray', linewidth=1, linestyle='--', alpha=0.5)
                ax.add_patch(rect)
        
        # Cell size and bar parameters
        cell_size = 1.0
        bar_width = cell_size / (self.n_actions + 1)
        bar_spacing = cell_size / self.n_actions
        max_bar_height = cell_size * 0.8
        
        # For each state, draw bars inside the cell
        for s_idx in range(self.n_states):
            cell = self.env.idx_to_state[s_idx]
            x, y = cell[0], cell[1]
            
            # Draw cell background
            rect = Rectangle(
                (x - cell_size/2, y - cell_size/2), 
                cell_size, cell_size,
                linewidth=1.5,
                edgecolor='black',
                facecolor='lightgray',
                alpha=0.3
            )
            ax.add_patch(rect)
            
            # Get action probabilities
            probs = policy_per_state[s_idx]
            
            # Draw bars for each action
            start_x = x - cell_size/2 + bar_width/2
            
            for a_idx in range(self.n_actions):
                bar_x = start_x + a_idx * bar_spacing
                bar_height = probs[a_idx] * max_bar_height
                
                # Bars start from bottom of cell (y + cell_size/2) and grow upward
                bar_y = y + cell_size/2 - bar_height - 0.1
                
                bar_rect = Rectangle(
                    (bar_x - bar_width/2, bar_y),
                    bar_width, 
                    bar_height,
                    facecolor=self.action_colors[a_idx],
                    edgecolor='black', 
                    linewidth=0.5
                )
                ax.add_patch(bar_rect)
        
        # Set proper ticks
        ax.set_xticks(np.arange(self.min_x, self.min_x + self.grid_width))
        ax.set_yticks(np.arange(self.min_y, self.min_y + self.grid_height))
        
        # Add legend
        legend_elements = [Patch(facecolor=self.action_colors[i], label=self.action_names[i])
                        for i in range(self.n_actions)]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    def _plot_sample_occupancy(self, ax, title='Batch State Occupancy', normalize=True):
        """Plot state occupancy from the current batch.
        
        Args:
            ax: matplotlib axis
            title: plot title
            normalize: if True, normalize counts to probabilities; if False, show raw counts
        """
        # Use the cached features from the last actor update
        if not hasattr(self.agent, '_phi_all_next') or self.agent._phi_all_next is None:
            ax.text(0.5, 0.5, 'No batch data available yet',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(title, fontsize=12, fontweight='bold')
            return
        
        # Count state visits in the current batch
        state_counts = np.zeros(self.n_states)
        
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
                
            # Get batch embeddings (remove augmented dimension)
            batch_embeddings = self.agent._phi_all_next[:, :-1].detach().cpu()
            
            # Find closest state for each batch sample
            for batch_emb in batch_embeddings:
                # Compute cosine similarity
                similarities = F.cosine_similarity(
                    batch_emb.unsqueeze(0),
                    enc_all_states,
                    dim=1
                )
                closest_state = torch.argmax(similarities).item()
                state_counts[closest_state] += 1
        
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
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)

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
        
        # Draw cell boundaries
        for s_idx in range(self.n_states):
            cell = self.env.idx_to_state[s_idx]
            x, y = cell[0] - self.min_x, cell[1] - self.min_y
            
            # Draw rectangle around each cell
            rect = Rectangle(
                (x - 0.5, y - 0.5), 
                1, 1,
                linewidth=1.5,
                edgecolor='black',
                facecolor='lightgray',
                alpha=0.3
            )
            ax.add_patch(rect)
            
            # Draw arrow for most likely action
            probs = policy_per_state[s_idx]
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



    def _plot_state_correlations(self, ax, correlation_matrix):
        """Plot correlation matrix heatmap WITHOUT text annotations."""
        im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
        ax.set_title('State Embedding Correlations', fontsize=12, fontweight='bold')
        ax.set_xlabel('State Index')
        ax.set_ylabel('State Index')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _plot_state_to_states_correlation(self, ax, state_correlations):
        """Plot per-state average correlation WITHOUT text annotations."""
        grid = np.zeros((self.grid_height, self.grid_width))
        
        for s_idx in range(self.n_states):
            cell = self.env.idx_to_state[s_idx]
            x, y = cell[0] - self.min_x, cell[1] - self.min_y
            grid[y, x] = state_correlations[s_idx]
        
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
                 action_shape,
                 lr_actor,
                 discount,
                 lambda_reg,
                 batch_size,
                 batch_size_actor,
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
                 sink_schedule,
                 epsilon_schedule,
                 mode,
                 reward,
                 embeddings = True,
                 pmd_eta_mode: str = "none",
                 pmd_best_iterate: bool = True,
                 pmd_grad_clip_norm: float = 0.0,
                 pmd_adagrad_eps: float = 1e-8,
                 pmd_eta_min: float = 1e-8,
                 pmd_eta_max: float = 1e3,
                 pmd_backtrack_factor: float = 0.5,
                 pmd_backtrack_max_trials: int = 8,
                 device: str = "cpu",
                 ):

        self.n_states = obs_shape[0]
        self.n_actions = action_shape[0]
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim if feature_dim is not None else self.n_states
        self.action_dim = action_shape[0]
        self.latent_a_dim = int(self.action_dim * 1.25) + 1 # From TACO
        self.lr_actor = lr_actor
        self.discount = discount
        self.lr_T = lr_T
        self.T_init_steps = T_init_steps
        self.batch_size = batch_size
        self.batch_size_actor = batch_size_actor
        assert batch_size_actor >= batch_size, "Actor update batch size must be greater than or equal to encoder update batch size"
        self.update_every_steps = update_every_steps
        self.update_actor_every_steps = update_actor_every_steps
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.device = device
        self.pmd_steps = pmd_steps
        self.embeddings = embeddings
        self.curl = curl
        self.embedding_sum_loss = embedding_sum_loss
        self.reward = reward
        self.pmd_eta_mode = pmd_eta_mode.lower()
        assert self.pmd_eta_mode in ["none", "adagrad", "backtracking"], "pmd_eta_mode must be one of ['none', 'adagrad', 'backtracking']"
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
            self.obs_dim
        ).to(self.device)
        
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
            self.encoder_optimizer = torch.optim.Adam(
            parameters,
            lr=lr_encoder
            )
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

    
        self.visualizer = ExplorationVisualizer(
            obs_shape=obs_shape,
            obs_type=obs_type,  # Pass obs_type here!
            feature_dim=self.feature_dim,
            hash_dim=1024,
            k_neighbors=5,
            occupancy_window=self.update_actor_every_steps*3,
            save_dir=os.path.join("exploration_plots", os.getcwd()),
            device=self.device
        )

        # Gridworld-specific visualizer (initialized later via insert_env)
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
    
    def insert_env(self, env):
        """
        Insert environment reference for gridworld-specific visualizations.
        Call this from pretrain.py after agent creation.
        """       
        self.wrapped_env = env
        self.env = env.unwrapped  # Get the base environment
        
        # Initialize gridworld visualizer
        try:
            self.gridworld_visualizer = EmbeddingDistributionVisualizerV2(self)
            print("✓ Gridworld-specific visualizer initialized")
        except Exception as e:
            print(f"⚠ Could not initialize gridworld visualizer: {e}")
            self.gridworld_visualizer = None


    
    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.project_sa.train(training)

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
        return torch.softmax(-logits, dim=1, dtype=torch.float32)
    
    
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
            
            enc_obs = self.aug_and_encode(obs_tensor, project=True) #.cpu()  # [1, feature_dim]
    
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
            norm_projected_sa = F.normalize(projected_sa, p=2, dim=1, eps=1e-10)
        elif self.mode == 'l2':
            norm_next_obs_en = next_obs_en
            norm_projected_sa = projected_sa

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
        
        # Optimize
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad()
        self.transition_optimizer.zero_grad()
        loss.backward()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.step()
        self.transition_optimizer.step()

        # Print losses
        logger.debug(f"Transition Model Losses: Contrastive={contrastive_loss.item():.4f}, CURL={curl_loss.item():.4f}, Embedding Sum={embedding_sum_loss.item():.4f}, Reward={reward_loss.item():.4f}, Total={loss.item():.4f}")
        if self.use_tb or self.use_wandb:
            metrics['transition_loss'] = loss.item()
        return metrics

    def update_actor(self, obs, action, next_obs, step, rewards=None):
        """Update policy using Projected Mirror Descent."""
        metrics = dict()

        # Compute features augmented
        self._cache_features(obs, action, next_obs)

        self.gradient_coeff = torch.zeros((self._phi_all_obs.shape[0]+1, 1), device=self.device)  # [z_x + 1, 1]
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

        if self._adagrad_accum is None:
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
                grad_norm_sq = float(torch.sum(grad_update * grad_update).item())
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

    
    def _cache_features(self, obs, action, next_obs):
        """Pre-compute and cache dataset features."""
       
        with torch.no_grad():
    
            self._phi_all_obs = self.aug_and_encode(obs, project=True) #.cpu()
            self._phi_all_next = self.aug_and_encode(next_obs, project=True) #.cpu()

            action = action #.cpu()
            self._psi_all = self._encode_state_action(self._phi_all_obs, action) #.cpu()
           
            self._alpha = torch.zeros((self._phi_all_next.shape[0], 1), device=self.device)  # [n, 1]
    
            self._alpha[0] = 1.0  # set alpha to 1.0 for the first state
            self.E = F.one_hot(
                action, 
                self.n_actions,
            ).reshape(-1, self.n_actions).to(torch.float32).to(self.device)

            # ** AUGMENTATION STEP **
            # ψ and Φ are augmented with an additional zero dimension
            zeros_col = torch.zeros(*self._psi_all.shape[:-1], 1, device=self._psi_all.device)
            self._psi_all = torch.cat([self._psi_all, zeros_col], dim=-1)

            zero_col = torch.zeros(*self._phi_all_next.shape[:-1], 1, device=self._phi_all_next.device)
            self._phi_all_next = torch.cat([self._phi_all_next, zero_col], dim=-1)

            zero_col = torch.zeros(*self._phi_all_obs.shape[:-1], 1, device=self._phi_all_obs.device)
            self._phi_all_obs = torch.cat([self._phi_all_obs, zero_col], dim=-1)

            print(f"dimensions after augmentation: psi_all {self._psi_all.shape}, phi_all_next {self._phi_all_next.shape}, phi_all_obs {self._phi_all_obs.shape}")

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

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        if self.use_tb or self.use_wandb:
            metrics['batch_reward'] = reward.mean().item()
           
        metrics.update(self.update_encoders(obs, action, next_obs, reward))

        # If T is not sufficiently initialized, skip actor update
        if self._is_T_sufficiently_initialized(step) is False:   
            metrics['actor_loss'] = 100.0  # dummy value
            return metrics
        
        # In ideal mode, we can update actor immediately
        if  step % self.update_actor_every_steps == 0 or step == self.num_expl_steps + self.T_init_steps: # or self.ideal:  

            num_batches_needed = self.batch_size_actor // self.batch_size
            
            obs_list = [obs]
            action_list = [action]
            next_obs_list = [next_obs]
            reward_list = [reward]
            for _ in range(num_batches_needed - 1):
                batch = next(replay_iter)
                obs_b, action_b, reward_b, _, next_obs_b = utils.to_torch(batch, self.device)
                obs_list.append(obs_b)
                action_list.append(action_b)
                next_obs_list.append(next_obs_b)
                reward_list.append(reward_b.reshape(-1, 1))  # Ensure reward has shape [B, 1]
            

            # Concatena tutti i batch
            obs_actor = torch.cat(obs_list, dim=0)
            action_actor = torch.cat(action_list, dim=0)
            next_obs_actor = torch.cat(next_obs_list, dim=0)
            reward_actor = torch.cat(reward_list, dim=0)

            # update actor (now with rewards)
            metrics.update(self.update_actor(obs_actor, action_actor, next_obs_actor, step, rewards=reward_actor))


            # === UPDATE VISUALIZER ===
            if self.visualizer is not None:
                # Update metrics with current batch
                vis_metrics = self.visualizer.update(
                    obs_batch=obs_actor,  # Raw pixels for hashing
                    z_batch=self._phi_all_obs[:, :-1],  # Learned embeddings (remove augmented dimension)
                    step=step
                )
                metrics.update(vis_metrics)
                
                #add text
                  
                param_text = (
                    f"Step: {step}\n"
                    f"γ = {self.discount}\n"
                    f"η = {self.current_eta}\n"
                    f"λ = {self.lambda_reg}\n"
                    f"sink norm = {utils.schedule(self.sink_schedule, step):.6f}\n"
                    f"PMD steps = {self.pmd_steps}\n"
                    
                )
                        # Generate plots periodically
                self.visualizer.plot_all(step, param_text=param_text)
                    
                # Generate t-SNE less frequently (expensive)
                if step % (self.update_actor_every_steps * 3) == 0:
                    try:
                        self.visualizer.plot_tsne(
                            self._phi_all_obs[:, :-1],  # Remove augmented dim
                            step,
                            method='tsne'
                        )
                    except Exception as e:
                        print(f"⚠ Could not generate t-SNE plot at step {step}: {e}")

                try:
                    save_path = f"gridworld_plots/step_{step}.png"
                    os.makedirs("gridworld_plots", exist_ok=True)
                    self.gridworld_visualizer.plot_results(step, save_path)
                    print(f"✓ Gridworld plot saved: {save_path}")
                except Exception as e:
                    print(f"⚠ Could not generate gridworld plot at step {step}: {e}")
        
        
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
