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
torch.set_default_dtype(torch.double)
from agent.utils import InternalDatasetFIFO

SINK_STATE_VALUE = 1

# ============================================================================
# Neural Network Components
# ============================================================================
class Encoder(nn.Module):
    def __init__(self, obs_shape, hidden_dim, feature_dim):
        super(Encoder, self).__init__()
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.repr_dim = feature_dim
        self.temperature = 0.1

        # self.fc = nn.Identity()
        # self.fc = nn.Linear(obs_shape[0], feature_dim, bias=False)
        self.fc =  nn.Sequential(
            nn.Linear(obs_shape[0], hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim, bias=False),
            # nn.LayerNorm(feature_dim),
        )

        # nn.init.eye_(self.fc[0].weight)
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs.view(obs.shape[0], -1)
        h = self.fc(obs)
        h = F.normalize(h, p=1, dim=-1)
        # h = F.softmax(h/self.temperature, dim=-1)

        return h

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
            
    def compute_discounted_occupancy(
            self, 
            phi_curr_states:torch.Tensor, 
            phi_all_next_obs:torch.Tensor, 
            BM: torch.Tensor,
            alpha: torch.Tensor
        ) -> torch.Tensor:
        """Compute discounted occupancy: ν = (1-γ)(I - γBM)⁻¹α."""
        raise NotImplementedError("Use compute_nu_pi")
        N = BM.shape[0]
        identity = torch.eye(N, device=self.device)
        
        kernel = phi_curr_states @ phi_all_next_obs.T  # [B, N]
        inv_term = torch.linalg.solve(identity - self.gamma * BM, alpha)
        
        occupancy = (1 - self.gamma) * kernel @ inv_term
        return occupancy

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
        sink_state[-1] = SINK_STATE_VALUE*epsilon

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
        sink_state[-1] = SINK_STATE_VALUE*epsilon

        # Computing Ψ̃ and Φ̃ are now of shape [N+1, d*|A| + 2] and [N+1, d + 2] respectively
        upper_left = phi_all_next_obs.T - sink_state@torch.ones((1, psi_all_obs_action.shape[1]), device=psi_all_obs_action.device, dtype=psi_all_obs_action.dtype)@psi_all_obs_action.T
        tilde_phi_all_next_obs_transposed = torch.zeros((phi_all_next_obs.shape[1]+1, phi_all_next_obs.shape[0]+1), device=phi_all_next_obs.device, dtype=phi_all_next_obs.dtype)
        tilde_phi_all_next_obs_transposed[:upper_left.shape[0], :upper_left.shape[1]] = upper_left
        assert sink_state.shape[0] == upper_left.shape[0], "Sink state and upper left matrix row size mismatch"
        tilde_phi_all_next_obs_transposed[:sink_state.shape[0], -1:] = sink_state
        # tilde_phi_all_next_obs_transposed[-1, -1] = 1.0
        tilde_phi_all_next_obs = tilde_phi_all_next_obs_transposed.T
        assert torch.all(tilde_phi_all_next_obs_transposed[:sink_state.shape[0], -1:] == sink_state), "Last column of tilde_phi_all_next_obs should be sink_state"

        # tilde_psi_all_obs_action = torch.zeros((psi_all_obs_action.shape[0]+1, psi_all_obs_action.shape[1]+1), device=psi_all_obs_action.device, dtype=psi_all_obs_action.dtype)
        # tilde_psi_all_obs_action[:-1, :-1] = psi_all_obs_action
        # tilde_psi_all_obs_action[-1, :-1] = 1.0
        # assert torch.all(tilde_psi_all_obs_action[-1, :-1] == torch.ones((1, tilde_psi_all_obs_action.shape[1]-1), device=tilde_psi_all_obs_action.device, dtype=tilde_psi_all_obs_action.dtype)), "Last row of tilde_psi_all_obs_action should be 1s"

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
class EmbeddingDistributionVisualizer:
    """Visualizer for embedding-based distribution matching results."""
    
    def __init__(self, agent, n_trajectories: int = 50, traj_length: int = 50):
        """
        Initialize visualizer with agent reference.
        
        Args:
            agent: DistMatchingEmbeddingAgent instance
            n_trajectories: Number of trajectories to sample
            traj_length: Length of each trajectory
        """
        self.agent = agent
        self.env = agent.env
        self.n_trajectories = n_trajectories
        self.traj_length = traj_length
        
        # Assert that we're not in unique mode for continuous spaces
        assert agent.data_type != "unique", "Cannot use 'unique' data_type for continuous state spaces"
        
        # Get environment bounds from observation space
        self.obs_low = self.env.observation_space.low
        self.obs_high = self.env.observation_space.high
        self.n_actions = self.agent.n_actions
        
        print(f"Visualizer bounds: x=[{self.obs_low[0]:.2f}, {self.obs_high[0]:.2f}], "
              f"y=[{self.obs_low[1]:.2f}, {self.obs_high[1]:.2f}]")
        
        # Action symbols and colors
        self.action_symbols = {0: '↑', 1: '↓', 2: '←', 3: '→', 4: '↖', 5: '↗', 6: '↙', 7: '↘'}
        self.action_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        self.action_names = ['up', 'down', 'left', 'right', 'up-left', 'up-right', 'down-left', 'down-right']
    
    def _sample_trajectories(self):
        """Sample trajectories from current policy."""
        trajectories = []
        action_sequences = []
        
        for traj_idx in range(self.n_trajectories):
            obs, _ = self.env.reset()
            trajectory = [obs.copy()]
            actions = []
            
            if traj_idx == 0:
                print(f"Starting trajectory from initial observation: {obs}")
            
            for i in range(self.traj_length):
                action = self.agent.act(obs, None, self.agent.num_expl_steps + 1, eval_mode=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                
                if traj_idx == 0 and i < 5:
                    print(f"  Step {i}: action={action} -> obs={obs}")
                # else:
                #     exit()
                
                trajectory.append(obs.copy())
                actions.append(action)
                
                if terminated or truncated:
                    break
            
            trajectories.append(np.array(trajectory))
            action_sequences.append(actions)
        
        return trajectories, action_sequences
    
    def plot_embeddings_2d(self, save_path: str, use_tsne: bool = False):
        """
        Plot 2D projection of state embeddings using PCA or t-SNE.
        Samples states from trajectories.
        
        Args:
            save_path: Path to save the figure
            use_tsne: If True, use t-SNE; otherwise use PCA
        """
        # Sample trajectories to get states
        trajectories, _ = self._sample_trajectories()
        
        # Collect all observations
        all_obs = []
        for traj in trajectories:
            all_obs.extend(traj)
        all_obs = np.array(all_obs)
        
        with torch.no_grad():
            obs_tensor = torch.from_numpy(all_obs).double().to(self.agent.device)
            embeddings = self.agent.encoder(obs_tensor).cpu().numpy()
        
        # Dimensionality reduction
        if use_tsne:
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_obs)-1))
            method_name = "t-SNE"
        else:
            reducer = PCA(n_components=2)
            method_name = "PCA"
        
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Color by trajectory
        colors = plt.cm.viridis(np.linspace(0, 1, self.n_trajectories))
        
        idx = 0
        for traj_idx, traj in enumerate(trajectories):
            traj_len = len(traj)
            traj_embeddings = embeddings_2d[idx:idx+traj_len]
            
            # Plot trajectory as connected line
            ax.scatter(traj_embeddings[:, 0], traj_embeddings[:, 1], 
                   c=colors[traj_idx], alpha=0.6, linewidth=1.5, marker="1", s=100)
            # Mark start and end
            ax.scatter(traj_embeddings[0, 0], traj_embeddings[0, 1],
                      c=[colors[traj_idx]], s=100, marker='o', edgecolors='black', linewidth=2, label=f'Start {traj_idx}' if traj_idx < 3 else '')
            ax.scatter(traj_embeddings[-1, 0], traj_embeddings[-1, 1],
                      c=[colors[traj_idx]], s=100, marker='*', edgecolors='black', linewidth=2)
            
            idx += traj_len
        
        ax.set_xlabel(f'{method_name} Component 1')
        ax.set_ylabel(f'{method_name} Component 2')
        ax.set_title(f'State Embeddings from {self.n_trajectories} Trajectories ({method_name})')
        ax.grid(True, alpha=0.3)
        if self.n_trajectories <= 3:
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Embeddings visualization saved to: {save_path}")
        plt.close(fig)
    
    def plot_results(self, step: int, save_path: str = None):
        """
        Create visualization of trajectories and their embeddings.
        
        Args:
            step: Current training step
            save_path: Path to save figure (optional)
        """
        trajectories, action_sequences = self._sample_trajectories()
        
        fig = plt.figure(figsize=(20, 10))
        
        # Add parameter text
        param_text = (
            f"Step: {step}\n"
            f"γ = {self.agent.discount}\n"
            f"η = {self.agent.lr_actor}\n"
            f"λ = {self.agent.lambda_reg}\n"
            f"PMD steps = {self.agent.pmd_steps}\n"
            f"Trajectories: {self.n_trajectories}"
        )
        fig.text(0.02, 0.98, param_text, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Create subplot grid
        ax_trajectories = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
        ax_dataset = plt.subplot2grid((2, 3), (0, 2), colspan=1)
        ax_embeddings = plt.subplot2grid((2, 3), (1, 2), colspan=1)
        
        # Plot trajectories in observation space
        self._plot_trajectories(ax_trajectories, trajectories)
        
        # Plot dataset occupancy
        self._plot_dataset_occupancy(ax_dataset)
        
        # Plot embedding distribution
        self._plot_embedding_distribution(ax_embeddings, trajectories)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nVisualization saved to: {save_path}")
            plt.close(fig)
    
    def _plot_trajectories(self, ax, trajectories):
        """Plot sampled trajectories in observation space."""
        ax.set_xlim(self.obs_low[0], self.obs_high[0])
        ax.set_ylim(self.obs_low[1], self.obs_high[1])
        ax.set_aspect('equal')
        ax.set_title(f'{self.n_trajectories} Sampled Trajectories')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
        
        colors = plt.cm.viridis(np.linspace(0, 1, self.n_trajectories))
        
        for idx, traj in enumerate(trajectories):
            # Plot trajectory
            ax.scatter(traj[:, 0], traj[:, 1], c=colors[idx], alpha=0.6, linewidth=1.5)
            
            # Mark start (circle) and end (star)
            ax.scatter(traj[0, 0], traj[0, 1], c=[colors[idx]], s=80, marker='o', 
                      edgecolors='black', linewidth=1.5, zorder=10)
            ax.scatter(traj[-1, 0], traj[-1, 1], c=[colors[idx]], s=120, marker='*',
                      edgecolors='black', linewidth=1.5, zorder=10)
        
        # Draw environment structure if available
        if hasattr(self.env, 'walkable_areas'):
            for area in self.env.walkable_areas:
                rect = Rectangle((area[0], area[1]), area[2], area[3],
                               fill=False, edgecolor='gray', linewidth=1, linestyle='--', alpha=0.5)
                ax.add_patch(rect)
    
    def _plot_dataset_occupancy(self, ax, title='Dataset Observation Distribution'):
        """Plot 2D histogram of observations in the dataset."""
        dataset_dict = self.agent.dataset._sampled_data if hasattr(self.agent.dataset, '_sampled_data') else self.agent.dataset.data
        observations = dataset_dict['observation']
        
        if observations.shape[0] == 0:
            ax.text(0.5, 0.5, 'Dataset vuoto', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(title)
            return
        
        obs_np = observations.cpu().numpy()
        
        # Create 2D histogram
        ax.hist2d(obs_np[:, 0], obs_np[:, 1], bins=30, cmap='Blues', 
                 range=[[self.obs_low[0], self.obs_high[0]], 
                        [self.obs_low[1], self.obs_high[1]]])
        
        ax.set_xlim(self.obs_low[0], self.obs_high[0])
        ax.set_ylim(self.obs_low[1], self.obs_high[1])
        ax.set_title(f'{title}\n(Total: {observations.shape[0]} samples)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
    
    def _plot_embedding_distribution(self, ax, trajectories):
        """Plot distribution of embeddings using PCA projection."""
        # Collect all observations
        all_obs = []
        for traj in trajectories:
            all_obs.extend(traj)
        all_obs = np.array(all_obs)
        
        with torch.no_grad():
            obs_tensor = torch.from_numpy(all_obs).double().to(self.agent.device)
            embeddings = self.agent.encoder(obs_tensor).cpu().numpy()
        
        # PCA to 2D
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # Plot as scatter
        colors = plt.cm.viridis(np.linspace(0, 1, self.n_trajectories))
        
        idx = 0
        for traj_idx, traj in enumerate(trajectories):
            traj_len = len(traj)
            traj_embeddings = embeddings_2d[idx:idx+traj_len]
            ax.scatter(traj_embeddings[:, 0], traj_embeddings[:, 1],
                      c=[colors[traj_idx]], s=20, alpha=0.6)
            idx += traj_len
        
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_title('Embedding Distribution (PCA)')
        ax.grid(True, alpha=0.3)

        
# ============================================================================
# Main Agent
# ============================================================================
class DistMatchingEmbeddingAgent:
    def __init__(self,
                 name,
                 obs_type,
                 obs_shape,
                 action_shape,
                 lr_actor,
                 discount,
                 lambda_reg,
                 batch_size,
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
                 window_size,
                 unique_window,
                 n_subsamples,
                 subsampling_strategy,
                 data_type: str = "unique",
                 device: str = "cpu",
                 linear_actor: bool = False,
                 ideal: bool = False
                 ):

        assert data_type != "unique", "Cannot use 'unique' data_type for continuous state spaces"
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
        self.update_every_steps = update_every_steps
        self.update_actor_every_steps = update_actor_every_steps
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.device = device
        self.data_type = data_type
        self.pmd_steps = pmd_steps
        self.ideal = ideal
        self.unique_window = unique_window

        self.sink_schedule = sink_schedule
        self.epsilon_schedule = epsilon_schedule
        self.subsampling_strategy = subsampling_strategy
        assert subsampling_strategy in ['random', 'eder'], "Subsampling strategy must be either 'random' or 'eder'"

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
        
        # Components
        self.encoder = Encoder(
            obs_shape, 
            hidden_dim, 
            self.feature_dim
        ).to(self.device)
        
        self.transition_model = TransitionModel(
            self.feature_dim * self.n_actions,
            self.feature_dim
        ).to(self.device)
        
        self.distribution_matcher = DistributionMatcher(
            gamma=self.discount,
            eta=self.lr_actor,
            lambda_reg=self.lambda_reg,
            device=self.device #self.device At the moment forcing computatiosn on cpu, to save gpu memory
        )
        
        self.dataset = InternalDatasetFIFO(
            dataset_type=self.data_type, 
            n_states=self.n_states, 
            n_actions=self.n_actions, 
            gamma=self.discount, 
            window_size=window_size, 
            n_subsamples=n_subsamples,
            subsampling_strategy=subsampling_strategy
        )
        
       
        # Optimizers
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(), 
            lr=lr_encoder
        )
        self.transition_optimizer = torch.optim.Adam(
            self.transition_model.parameters(),
            lr=lr_T
        )
        
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.training = False
        # Initialize visualizer (will be properly set after env is inserted)
        self.visualizer = None
        self._ideal_dataset_filled = False
    
    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.transition_model.train(training)
    
    def insert_env(self, env):
        self.env = env.unwrapped
        timestep = env.reset()
        self.first_state = torch.tensor(timestep.observation, device=self.device, dtype=torch.double)
        self.second_states = []
        for action in range(self.n_actions):
            next_timestep = env.step(action)
            self.second_states.append(torch.tensor(next_timestep.observation, device=self.device, dtype=torch.double))
            env.reset()
            
        # For continuous spaces, we don't use second_state indexing
        
        # Initialize visualizer now that we have the environment
        self.visualizer = EmbeddingDistributionVisualizer(self, n_trajectories=30, traj_length=100)
        
        # Ideal mode not supported for continuous spaces
        if self.ideal:
            raise NotImplementedError("Ideal mode not supported for continuous state spaces")
    
    def _find_discrete_env(self, env):
        """
        Find the discrete environment interface by traversing the wrapper chain.
        Returns the first environment that has 'cells' attribute (either native discrete
        or DiscretizedContinuousEnv wrapper).
        """
        current = env
        while current is not None:
            # Check if this level has the discrete interface (cells, state_to_idx, etc.)
            if hasattr(current, 'cells') and hasattr(current, 'state_to_idx'):
                return current
            
            # Move to next wrapper level
            if hasattr(current, 'env'):
                current = current.env
            elif hasattr(current, 'unwrapped') and current.unwrapped is not current:
                current = current.unwrapped
            else:
                break
        
        raise AttributeError(
            "Could not find discrete environment interface. "
            "Ensure the environment has 'cells' and 'state_to_idx' attributes "
            "(either native discrete env or wrapped with DiscretizedContinuousEnv)."
        )

    def _populate_ideal_dataset(self):
        """Not supported for continuous spaces."""
        raise NotImplementedError("Ideal dataset population not supported for continuous state spaces")
    
    
    def init_meta(self):
        return OrderedDict()

    def get_meta_specs(self):
        return tuple()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        self.dataset.add_transition(time_step)
        return meta
    
    def _encode_state_action(
        self, 
        encoded_obs: torch.Tensor, 
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Encode (s,a) pairs as ψ(s,a) = φ(s) ⊗ e_a."""
        action_onehot = F.one_hot(actions.long(), self.n_actions).double().reshape(-1, self.n_actions)  # [B, |A|]
        
        # print dimensions for debugging
        print(f"encoded_obs shape: {encoded_obs.shape}, action_onehot shape: {action_onehot.shape}")
        # Outer product: [B, d] ⊗ [B, |A|] -> [B, d*|A|]
        encoded_sa = torch.einsum('bd,ba->bda', encoded_obs, action_onehot)
        return encoded_sa.reshape(encoded_obs.shape[0], -1)
    
    
    def compute_action_probs(self, obs: np.ndarray) -> np.ndarray:
        """Compute π(·|s) for given observation."""
        with torch.no_grad():
            obs_tensor = torch.tensor(obs).unsqueeze(0).double().to(self.device)  # [1, obs_dim]
            enc_obs = self.encoder(obs_tensor).cpu()  # [1, feature_dim]
    
            if self.gradient_coeff is None:
                return np.ones(self.n_actions) / self.n_actions
            
            # Add a zero to enc_obs to account for the extra row in H
            enc_obs_augmented = torch.cat([enc_obs, torch.zeros((1, 1), dtype=torch.double)], dim=1)  # [1, feature_dim + 1]
            H = enc_obs_augmented @ self._phi_all_obs.T  # [1, num_unique]

            probs = torch.softmax(-self.lr_actor * (H@(self.gradient_coeff[:-1]*self.E)+ torch.ones(1, self.E.shape[1])*self.gradient_coeff[-1]), dim=1, dtype=torch.double)  # [1, n_actions]

            
            if torch.sum(probs) == 0.0 or torch.isnan(torch.sum(probs)):
                raise ValueError("action_probs sum to zero or NaN")
            return probs.numpy().flatten()

    
    def act(self, obs, meta, step, eval_mode):
        if step < self.num_expl_steps or np.random.rand() < utils.schedule(self.epsilon_schedule, step):
            return np.random.randint(self.n_actions)
        
        if self.dataset.greater_equal_target_horizon:
            utils.ColorPrint.blue("Acting randomly because dataset has reached target horizon")
            return np.random.randint(self.n_actions)

        # Compute action probabilities
        action_probs = self.compute_action_probs(obs)
        
        # Sample action
        return np.random.choice(self.n_actions, p=action_probs)
    

    
    def _is_T_sufficiently_initialized(self, step: int) -> bool:
        """Check if transition learning phase is complete."""
    
        return step >= self.num_expl_steps + self.T_init_steps 
       
    def update_transition_matrix(self, obs, action, next_obs):
        metrics = dict()

        if self.ideal:
            # Use the full ideal dataset
            obs = self.dataset.data['observation'].to(self.device)
            action = self.dataset.data['action'].to(self.device)
            next_obs = self.dataset.data['next_observation'].to(self.device)
            assert obs.shape[0] == action.shape[0] == next_obs.shape[0], f"Ideal dataset tensors have mismatched sizes, received obs: {obs.shape}, action: {action.shape}, next_obs: {next_obs.shape}"
        
        # Encode
        encoded_obs = self.encoder(obs.double())
        with torch.no_grad():
            encoded_next = self.encoder(next_obs.double())
        encoded_state_action = self._encode_state_action(encoded_obs, action)
        
        # Predict next state
        predicted_next = self.transition_model(encoded_state_action)
        
        # Compute loss
        # 1. Contrastive loss:
        logits = predicted_next @ encoded_next.T  # [B, B]
        logits = logits - torch.max(logits, 1)[0][:, None]  # For numerical stability
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        contrastive_loss = self.cross_entropy_loss(logits, labels)

        # 2. Compute Feature Correlation Matrix (Z x Z)
        #    Shape: [Feature_Dim, Feature_Dim]
        #    We want this to be close to Identity (uncorrelated features)
        # remove duplicates from encoded_obs
        unique_obs = torch.unique(obs, dim=0)
        encoded_obs = self.encoder(unique_obs.double())
        cov_matrix = (encoded_obs.T @ encoded_obs) / (encoded_obs.shape[0] - 1)

        # 3. Decorrelation Loss (Push off-diagonals to 0)
        #    This forces each dimension to represent independent information, 
        #    preventing "feature collapse" where all dimensions encode the same thing.
        off_diagonal_mask = ~torch.eye(cov_matrix.shape[0], dtype=torch.bool, device=self.device)
        decorrelation_loss = torch.abs(cov_matrix[off_diagonal_mask]).sum()
        
        # 4. Loss embeddings must sum to 1
        embedding_sum_loss = torch.abs(torch.sum(encoded_obs, dim=-1) - 1).sum()

        loss =  contrastive_loss + 0*decorrelation_loss + embedding_sum_loss
        
        # Optimize
        self.encoder_optimizer.zero_grad()
        self.transition_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()
        self.transition_optimizer.step()

        print(f"transition_loss: {loss.item()}, deco_loss: {decorrelation_loss.item()}, contrastive_loss: {contrastive_loss.item()}")
        if self.use_tb or self.use_wandb:
            metrics['transition_loss'] = loss.item()
        return metrics

    def update_actor(self, obs, step):
        """Update policy using Projected Mirror Descent."""
        metrics = dict()
        # Compute features for internal dataset
        if not hasattr(self, '_features_cached'):
            self._cache_features()
            # if self.gradient_coeff is None or (self.gradient_coeff is not None and self.gradient_coeff.shape[0] != self.dataset.size):
            self.gradient_coeff = torch.zeros((self._phi_all_obs.shape[0]+1, 1), device=self.device, dtype=torch.double)  # [n+1, 1]
            self.H = self._phi_all_obs @ self._phi_all_next.T # [n, n]
            self.unique_states = torch.eye(self.n_states, device=self.device).double()
            self.K = self._psi_all @ self._psi_all.T  # [n, n]

        epsilon = utils.schedule(self.sink_schedule, step)
        self.pi = torch.softmax(-self.lr_actor * (self.H.T@(self.gradient_coeff[:-1]*self.E)+ torch.ones(self._phi_all_next.shape[0], self.E.shape[1], device=self.device)*self.gradient_coeff[-1]), dim=1, dtype=torch.double)  # [z_x+1, n_actions]
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
        # print("Gradient coeff norm:", torch.linalg.norm(self.gradient_coeff))
        # print("Policy matrix sample (first 5 states):", self.pi[:5, :])

        for iteration in range(self.pmd_steps):
            self.gradient_coeff += self.distribution_matcher.compute_gradient_coefficient(
                M, 
                phi_all_next_obs = self._phi_all_next, 
                psi_all_obs_action = self._psi_all, 
                alpha = self._alpha,
                epsilon=epsilon
            ) 
            
            # print("Gradient last term:", self.gradient_coeff[-1].item())
            
            self.pi = torch.softmax(-self.lr_actor * (self.H.T@(self.gradient_coeff[:-1]*self.E)+ torch.ones(self._phi_all_next.shape[0], self.E.shape[1], device=self.device)*self.gradient_coeff[-1]), dim=1, dtype=torch.double)  # [z_x+1, n_actions]

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

    
    def _cache_features(self):
        """Pre-compute and cache dataset features."""
        tensors = self.dataset.get_data()
        print("Caching features from dataset of size:", len(tensors['observation']))
        with torch.no_grad():
            obs = tensors['observation'][:len(tensors['next_observation'])].to(self.device, dtype=torch.double)
            actions = tensors['action'].to(self.device)
            next_obs = tensors['next_observation'].to(self.device, dtype=torch.double)
            
            self._phi_all_obs = self.encoder(obs)
            self._phi_all_next = self.encoder(next_obs)
         
            # Transfer first_state to GPU for comparison
            first_state_gpu = self.first_state.to(self.device)
            indices = torch.where(torch.all(next_obs == first_state_gpu, dim=1))[0]
            if indices.shape[0] == 0:
                for second_state in self.second_states:
                    second_state_gpu = second_state.to(self.device)
                    indices = torch.where(torch.all(next_obs == second_state_gpu, dim=1))[0]
                    print("DEBUG: looking for second_state, found indices:", indices)
                    if indices.shape[0] > 0:
                        break
            
            self._psi_all = self._encode_state_action(self._phi_all_obs, actions)
           
            self._alpha = torch.zeros((self._phi_all_next.shape[0], 1), device=self.device, dtype=torch.double)
            print("DEBUG: setting alpha for index", indices[0].item(), len(self._alpha))
            self._alpha[indices[0]] = 1.0
            self.E = F.one_hot(
                actions.long(), 
                self.n_actions
            ).to(self.device, dtype=torch.double).reshape(-1, self.n_actions)

            # ** AUGMENTATION STEP **
            # ψ and Φ are augmented with an additional zero dimension
            zeros_col = torch.zeros(*self._psi_all.shape[:-1], 1, device=self._psi_all.device, dtype=self._psi_all.dtype)
            self._psi_all = torch.cat([self._psi_all, zeros_col], dim=-1)

            zero_col = torch.zeros(*self._phi_all_next.shape[:-1], 1, device=self._phi_all_next.device, dtype=self._phi_all_next.dtype)
            self._phi_all_next = torch.cat([self._phi_all_next, zero_col], dim=-1)

            zero_col = torch.zeros(*self._phi_all_obs.shape[:-1], 1, device=self._phi_all_obs.device, dtype=self._phi_all_obs.dtype)
            self._phi_all_obs = torch.cat([self._phi_all_obs, zero_col], dim=-1)

            print(f"all shapes: phi_all_obs: {self._phi_all_obs.shape}, phi_all_next: {self._phi_all_next.shape}, psi_all: {self._psi_all.shape}, alpha: {self._alpha.shape}, E: {self.E.shape}")
            optimal_T = self.transition_model.compute_closed_form(self._psi_all, self._phi_all_next, self.lambda_reg)
            print(f"==== Optimal T error {F.mse_loss(optimal_T @ self._psi_all.T, self._phi_all_next.T).item()} ====")
        # print(optimal_T)
        # print(torch.sum(optimal_T, dim=0))
        if not self.dataset.is_complete:
            return
        print("=================================Features cached=================================")

        self._features_cached = True
        

    def aug_and_encode(self, obs):
        pass
        # obs = self.aug(obs)
        # return self.encoder(obs)

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        if not self.ideal:
            batch = next(replay_iter)
            obs, action, reward, discount, next_obs = utils.to_torch(
                batch, self.device)
        else:
            # In ideal mode, we don't need batch from replay_iter
            # We'll use the full dataset in update_transition_matrix
            obs = action = next_obs = reward = discount = None

        if self.use_tb or self.use_wandb:
            if not self.ideal:
                metrics['batch_reward'] = reward.mean().item()
            else:
                metrics['batch_reward'] = 0.0  # placeholder
        
        metrics.update(self.update_transition_matrix(obs, action, next_obs))

        print(f"last dataset size: {self.dataset.last_size}")
        print(f"current dataset size: {self.dataset.size}, current periodo size {self.dataset.current_data_size}")
        # If T is not sufficiently initialized, skip actor update
        if self._is_T_sufficiently_initialized(step) is False:   
            metrics['actor_loss'] = 100.0  # dummy value
            return metrics
        

        # In ideal mode, we can update actor immediately
        if  step % self.update_actor_every_steps == 0 or step == self.num_expl_steps + self.T_init_steps: # or self.ideal:  
            # update actor
            if not self.ideal:
                metrics.update(self.update_actor(obs, step))
            else:
                # Pass dummy obs, not used in update_actor
                metrics.update(self.update_actor(None, step))
        
            if self.visualizer is not None:
                save_path = os.path.join(os.getcwd(), f"plot_step_{step}.png")
                self.visualizer.plot_results(step, save_path=save_path)
                save_path = os.path.join(os.getcwd(), f"features_step_{step}.png")
                self.visualizer.plot_embeddings_2d(save_path=save_path, use_tsne=True)
                print(f"Visualization saved to: {save_path}")
        return metrics

