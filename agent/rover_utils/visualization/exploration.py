from __future__ import annotations

from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import umap


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
            idx = np.round(np.linspace(0, len(z) - 1, max_points)).astype(np.int64)
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
            c=np.arange(len(z_2d)),  # Color by the order supplied to t-SNE.
            cmap='viridis',
            alpha=0.6,
            s=20
        )
        
        plt.colorbar(scatter, ax=ax, label='Input / positional order')
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
