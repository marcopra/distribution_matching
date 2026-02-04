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
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3

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
            # nn.LayerNorm(feature_dim),
            nn.ReLU()
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.
        h = self.conv(obs)
        h = h.view(h.shape[0], -1)
        return h

    def encode_and_project(self, obs):
        h = self.forward(obs)
        z = self.projector(h)
        z =F.normalize(z, p=1, dim=-1)
        return z

class TransitionModel(nn.Module):
    """Learnable transition dynamics T: (s,a) -> s'."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.T = nn.Linear(input_dim, output_dim, bias=False)
    
    def forward(self, encoded_state_action: torch.Tensor) -> torch.Tensor:
        return self.T(encoded_state_action)
    
    def compute_closed_form(
        self, 
        psi: torch.Tensor,  # [N, d*|A|] state-action features
        phi_next: torch.Tensor,  # [N, d] next state features
        lambda_reg: float
    ) -> torch.Tensor:
        """Compute closed-form solution: T = φ'ᵀ(ψψᵀ + λI)⁻¹ψ."""
        N = psi.shape[0]
        identity = torch.eye(N, device=psi.device)
        gram_matrix = psi @ psi.T + lambda_reg * identity
        
        T_optimal = phi_next.T @ torch.linalg.solve(gram_matrix, psi)
        return T_optimal

    

# ============================================================================
# Distribution Matching Mathematics
# ============================================================================
class DistributionMatcher:
    """Handles mathematical operations for distribution matching via PMD."""

    def __init__(self, 
                 lambda_reg: float,
                 gamma: float = 0.9, 
                 eta: float = 0.1, 
                 device: str = "cpu"):
        
        self.gamma = gamma
        self.eta = eta
        self.lambda_reg = lambda_reg
        self.device = device    
            
    def compute_nu_pi(
            self, 
            phi_all_next_obs: torch.Tensor, 
            psi_all_obs_action: torch.Tensor,
            K: torch.Tensor,
            M: torch.Tensor,
            alpha: torch.Tensor,
            epsilon: float
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
        sink_state[-1] = epsilon

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
            epsilon: float
        ) -> torch.Tensor:
        """Compute gradient coefficient for policy update."""
        # Identity matrix
        I_n_plus1 = torch.eye(psi_all_obs_action.shape[0], device=self.device)

        sink_state = torch.zeros((phi_all_next_obs.shape[1],1), device=self.device, dtype=phi_all_next_obs.dtype)
        sink_state[-1] = epsilon

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


class FixedRandomEncoder(nn.Module):
    """Fixed random CNN for stable state hashing (witness network)."""
    
    def __init__(self, obs_shape, hash_dim=128):
        super().__init__()
        assert len(obs_shape) == 3, "Expected image observations [C, H, W]"
        
        # Same architecture as trained encoder but FROZEN
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
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, obs):
        """
        Args:
            obs: [B, C, H, W] uint8 images
        Returns:
            features: [B, repr_dim] continuous features
        """
        with torch.no_grad():
            obs = obs.float() / 255.0
            h = self.conv(obs)
            h = self.adaptive_pool(h)
            h = h.reshape(h.size(0), -1)
            return h
    
    def compute_hash(self, obs):
        """
        Args:
            obs: [B, C, H, W] images
        Returns:
            hash_codes: [B] int64 unique state IDs
        """
        with torch.no_grad():
            features = self.forward(obs)  # [B, repr_dim]
            projections = features @ self.projection_matrix.T  # [B, hash_dim]
            
            # Binary hash: sign of each projection
            binary_code = (projections > 0).long()  # [B, hash_dim]
            
            # Convert binary to unique integer (like a base-2 number)
            # Use only first 63 bits to avoid overflow with int64
            powers_of_2 = 2 ** torch.arange(63, device=obs.device, dtype=torch.long)
            hash_codes = (binary_code[:, :63] * powers_of_2).sum(dim=1)
            
            return hash_codes


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
        obs_shape: Tuple[int, int, int],
        feature_dim: int,
        hash_dim: int = 128,
        k_neighbors: int = 5,
        occupancy_window: int = 100000,
        save_dir: str = './exploration_plots',
        device: str = 'cpu'
    ):
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.k = k_neighbors
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Fixed random encoder for stable hashing
        self.random_encoder = FixedRandomEncoder(obs_shape, hash_dim).to(device)
        
        # Occupancy tracker
        self.occupancy = EmpiricalOccupancyTracker(occupancy_window)
        
        # Metrics history: {metric_name: [(step, value), ...]}
        self.history = defaultdict(list)
        
        print(f"ExplorationVisualizer initialized:")
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
            obs_batch: [B, C, H, W] raw pixel observations (for hashing)
            z_batch: [B, feature_dim] learned embeddings (for geometry metrics)
            step: current training step
        
        Returns:
            metrics: dict of computed metrics
        """
        metrics = {}
        
        # 1. Compute state hashes (fixed random encoder)
        with torch.no_grad():
            state_hashes = self.random_encoder.compute_hash(obs_batch).cpu().numpy()
        
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
        
        from scipy.spatial.distance import pdist, squareform
        
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
    
    def plot_all(self, step: int):
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
                from sklearn.manifold import TSNE
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
# Main Agent
# ============================================================================
class DistMatchingEmbeddingAgentv2:
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
                 hidden_dim,
                 feature_dim,
                 update_every_steps,
                 update_actor_every_steps,
                 pmd_steps,
                 num_expl_steps,
                 T_init_steps,
                 sink_schedule,
                 epsilon_schedule,
                 embeddings = True,
                 device: str = "cpu",
                 ):

        self.n_states = obs_shape[0]
        self.n_actions = action_shape[0]
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim if feature_dim is not None else self.n_states
        self.action_dim = action_shape[0]
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
            self.aug = nn.Identity() #utils.RandomShiftsAug(pad=4)
            assert embeddings, "Pixel observations require embeddings to be True"
            self.encoder = CNNEncoder(
                obs_shape,
                feature_dim
            ).to(self.device)
            
            self.obs_dim = self.feature_dim
        else:
            # Components
            self.aug = nn.Identity()
            if embeddings == False:
                self.encoder = nn.Identity()
                self.feature_dim = obs_shape[0]
            else:
                self.encoder = Encoder(
                        obs_shape, 
                        hidden_dim, 
                        self.feature_dim,
                    ).to(self.device)
            self.obs_dim = self.feature_dim
       
        self.transition_model = TransitionModel(
            self.obs_dim * self.n_actions,
            self.obs_dim
        ).to(self.device)
        
        self.distribution_matcher = DistributionMatcher(
            gamma=self.discount,
            eta=self.lr_actor,
            lambda_reg=self.lambda_reg,
            device='cpu' #self.device At the moment forcing computatiosn on cpu, to save gpu memory
        )
        
       
        # Optimizers
        if embeddings:
            self.encoder_optimizer = torch.optim.Adam(
                self.encoder.parameters(), 
                lr=lr_encoder
            )
        else:
            self.encoder_optimizer = None
        self.transition_optimizer = torch.optim.Adam(
            self.transition_model.parameters(),
            lr=lr_T
        )
        
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.training = False

        self.visualizer = None

        if obs_type == 'pixels':
            self.visualizer = ExplorationVisualizer(
                obs_shape=obs_shape,
                feature_dim=self.feature_dim,
                hash_dim=128,
                k_neighbors=5,
                occupancy_window=100000,
                save_dir= os.path.join("exploration_plots", os.getcwd()),
                device=self.device
    )


    
    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.transition_model.train(training)

    def init_meta(self):
        return OrderedDict()

    def get_meta_specs(self):
        return tuple()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        if self.ideal:
            # In ideal mode, we do not update the dataset during training
            return meta
        self.dataset.add_transition(time_step)
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
            
            enc_obs = self.aug_and_encode(obs_tensor, project=True).cpu()  # [1, feature_dim]
    
            if self.gradient_coeff is None:
                return np.ones(self.n_actions) / self.n_actions
            
            # Add a zero to enc_obs to account for the extra row in H
            enc_obs_augmented = torch.cat([enc_obs, torch.zeros((1, 1))], dim=1)  # [1, feature_dim + 1]
            H = enc_obs_augmented @ self._phi_all_obs.T  # [1, num_unique]

            probs = torch.softmax(-self.lr_actor * (H@(self.gradient_coeff[:-1]*self.E)+ torch.ones(1, self.E.shape[1])*self.gradient_coeff[-1]), dim=1)  # [1, n_actions]

            
            if torch.sum(probs) == 0.0 or torch.isnan(torch.sum(probs)):
                raise ValueError("action_probs sum to zero or NaN")
            return probs.numpy().flatten()

    
    def act(self, obs, meta, step, eval_mode):
        if step < self.num_expl_steps or np.random.rand() < utils.schedule(self.epsilon_schedule, step):
            return np.random.randint(self.n_actions)

        # Compute action probabilities
        action_probs = self.compute_action_probs(obs)
        
        # Sample action
        return np.random.choice(self.n_actions, p=action_probs)

    
    def _is_T_sufficiently_initialized(self, step: int) -> bool:
        """Check if transition learning phase is complete."""
        return step >= self.num_expl_steps + self.T_init_steps 
       
    def update_encoders(self, obs, action, next_obs):
        metrics = dict()

        # Minibatch size for encoder update
        idxs = np.random.randint(1, obs.shape[0], size=self.batch_size_encoder) # skipping the first index which is the start of episode
        obs = obs[idxs]
        action = action[idxs]
        next_obs = next_obs[idxs]
        
        # Encode
        obs_en = self.aug_and_encode(obs, project=True)
        with torch.no_grad():
            next_obs_en = self.aug_and_encode(next_obs, project=True)

        encoded_state_action = self._encode_state_action(obs_en, action)
        
        # Predict next state
        predicted_next = self.transition_model(encoded_state_action)
        
        # Compute loss
        # 1. Contrastive loss: 
        logits = predicted_next/torch.norm(predicted_next, p=2, dim=1, keepdim=True) @ (next_obs_en/torch.norm(next_obs_en, p=2, dim=1, keepdim=True)).T  # [B, B]
        logits = logits - torch.max(logits, 1)[0][:, None]  # For numerical stability
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        contrastive_loss = self.cross_entropy_loss(logits, labels)
        
        # 4. Loss embeddings must sum to 1
        embedding_sum_loss = torch.abs(torch.sum(self.aug_and_encode(obs, project=True), dim=-1) - 1).sum()
        beta = 0.1 
        # 5. \phi(s) and \phi(s') must be close in L2 norm
        l2_loss = torch.norm(obs_en - next_obs_en, p=2, dim=1).mean()

        loss =  contrastive_loss #+ beta*embedding_sum_loss + 0.01*l2_loss
        
        # Optimize
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad()
        self.transition_optimizer.zero_grad()
        loss.backward()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.step()
        self.transition_optimizer.step()

        # Print losses
        print(f"Transition Model Losses: Contrastive={contrastive_loss.item():.4f}, EmbeddingSum={embedding_sum_loss.item():.4f}, L2={l2_loss.item():.4f}, Total={loss.item():.4f}")
        if self.use_tb or self.use_wandb:
            metrics['transition_loss'] = loss.item()
        return metrics

    def update_actor(self, obs, action, next_obs, step):
        """Update policy using Projected Mirror Descent."""
        metrics = dict()

        # Compute features augmented
        self._cache_features(obs, action, next_obs)

        self.gradient_coeff = torch.zeros((self._phi_all_obs.shape[0]+1, 1))
        self.H = self._phi_all_obs @ self._phi_all_next.T # [n, n]
        self.unique_states = torch.eye(self.n_states)
        self.K = self._psi_all @ self._psi_all.T  # [n, n]

        epsilon = utils.schedule(self.sink_schedule, step)
        self.pi = torch.softmax(-self.lr_actor * (self.H.T@(self.gradient_coeff[:-1]*self.E)+ torch.ones(self._phi_all_next.shape[0], self.E.shape[1])*self.gradient_coeff[-1]), dim=1, dtype=torch.float32)  # [z_x+1, n_actions]

        M = self.H*(self.E@self.pi.T) 

        nu_pi = self.distribution_matcher.compute_nu_pi(
                phi_all_next_obs = self._phi_all_next,
                psi_all_obs_action= self._psi_all,
                K= self.K,
                M = M,
                alpha=self._alpha,
                epsilon=epsilon 
        )
        actor_loss = torch.linalg.norm(nu_pi)**2
        print(f"Actor loss (squared norm of occupancy measure): {actor_loss}")

        for iteration in range(self.pmd_steps):
            self.gradient_coeff += self.distribution_matcher.compute_gradient_coefficient(
                M, 
                phi_all_next_obs = self._phi_all_next, 
                psi_all_obs_action = self._psi_all, 
                alpha = self._alpha,
                epsilon=epsilon
            ) 
            
         
            self.pi = torch.softmax(-self.lr_actor * (self.H.T@(self.gradient_coeff[:-1]*self.E)+ torch.ones(self._phi_all_next.shape[0], self.E.shape[1])*self.gradient_coeff[-1]), dim=1)  # [z_x+1, n_actions]

            M = self.H*(self.E@self.pi.T) # [num_unique, num_unique]


            if iteration % 10 == 0 or iteration == self.pmd_steps - 1:
                nu_pi = self.distribution_matcher.compute_nu_pi(
                        phi_all_next_obs = self._phi_all_next,
                        psi_all_obs_action= self._psi_all,
                        K= self.K,
                        M = M,
                        alpha=self._alpha,
                        epsilon=epsilon
                )
                actor_loss = torch.linalg.norm(nu_pi)**2
                print(f"  PMD Iteration {iteration}, Actor loss: {actor_loss}")
            

        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss
   
        return metrics
    
    def _cache_features(self, obs, action, next_obs):
        """Pre-compute and cache dataset features."""
       
        with torch.no_grad():
    
            self._phi_all_obs = self.aug_and_encode(obs, project=True).cpu()
            self._phi_all_next = self.aug_and_encode(next_obs, project=True).cpu()

            self._psi_all = self._encode_state_action(self._phi_all_obs, action).cpu()
           
            self._alpha = torch.zeros((self._phi_all_next.shape[0], 1))
    
            self._alpha[0] = 1.0  # set alpha to 1.0 for the first state
            self.E = F.one_hot(
                action, 
                self.n_actions,
            ).reshape(-1, self.n_actions).to(torch.float32)  

            # ** AUGMENTATION STEP **
            # ψ and Φ are augmented with an additional zero dimension
            zeros_col = torch.zeros(*self._psi_all.shape[:-1], 1, device=self._psi_all.device)
            self._psi_all = torch.cat([self._psi_all, zeros_col], dim=-1)

            zero_col = torch.zeros(*self._phi_all_next.shape[:-1], 1, device=self._phi_all_next.device)
            self._phi_all_next = torch.cat([self._phi_all_next, zero_col], dim=-1)

            zero_col = torch.zeros(*self._phi_all_obs.shape[:-1], 1, device=self._phi_all_obs.device)
            self._phi_all_obs = torch.cat([self._phi_all_obs, zero_col], dim=-1)

            print(f"dimensions after augmentation: psi_all {self._psi_all.shape}, phi_all_next {self._phi_all_next.shape}, phi_all_obs {self._phi_all_obs.shape}")
            optimal_T = self.transition_model.compute_closed_form(self._psi_all, self._phi_all_next, self.lambda_reg)
            print(f"==== Optimal T error {F.mse_loss(optimal_T @ self._psi_all.T, self._phi_all_next.T).item()} ====")


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
           
        metrics.update(self.update_encoders(obs, action, next_obs))

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
            
            # Accumula batch aggiuntivi
            for _ in range(num_batches_needed - 1):
                batch = next(replay_iter)
                obs_b, action_b, _, _, next_obs_b = utils.to_torch(batch, self.device)
                obs_list.append(obs_b)
                action_list.append(action_b)
                next_obs_list.append(next_obs_b)
            
            # Concatena tutti i batch
            obs_actor = torch.cat(obs_list, dim=0)
            action_actor = torch.cat(action_list, dim=0)
            next_obs_actor = torch.cat(next_obs_list, dim=0)

            # update actor
            metrics.update(self.update_actor(obs_actor, action_actor, next_obs_actor, step))

            # === UPDATE VISUALIZER ===
            if self.visualizer is not None:
                # Update metrics with current batch
                vis_metrics = self.visualizer.update(
                    obs_batch=obs_actor,  # Raw pixels for hashing
                    z_batch=self._phi_all_obs[:, :-1],  # Learned embeddings (remove augmented dimension)
                    step=step
                )
                metrics.update(vis_metrics)
                
                # Generate plots periodically
                self.visualizer.plot_all(step)
                    
                # Generate t-SNE less frequently (expensive)
                if step % (self.update_actor_every_steps * 50) == 0:
                    self.visualizer.plot_tsne(
                        self._phi_all_obs[:, :-1],  # Remove augmented dim
                        step,
                        method='tsne'
                    )
    
    
        return metrics

