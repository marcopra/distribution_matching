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
        self.temperature = 0.05


        self.fc =  nn.Sequential(
            nn.Linear(obs_shape[0], hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim, bias=False),
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs.view(obs.shape[0], -1)
        h = self.fc(obs)
        h = F.normalize(h, p=1, dim=-1)


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
    
    def __init__(self, agent):
        """
        Initialize visualizer with agent reference.
        
        Args:
            agent: DistMatchingEmbeddingAgent instance
        """
        self.agent = agent
        self.env = agent.env
        self.n_states = agent.n_states
        self.n_actions = agent.n_actions

        # Get grid dimensions
        valid_cells = [cell for cell in self.env.cells if cell != self.env.DEAD_STATE]
        min_x = min(cell[0] for cell in valid_cells)
        min_y = min(cell[1] for cell in valid_cells)
        max_x = max(cell[0] for cell in valid_cells)
        max_y = max(cell[1] for cell in valid_cells)

        valid_ids = [self.env.state_to_idx[cell] for cell in valid_cells]
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
            self.action_symbols = {0: '↑', 1: '↓', 2: '←', 3: '→'}
            self.action_colors = ['red', 'blue', 'green', 'orange']
            self.action_names = ['up', 'down', 'left', 'right']
        elif self.n_actions == 8:
            self.action_symbols = {
                0: '↑', 1: '↓', 2: '←', 3: '→',
                4: '↖', 5: '↗', 6: '↙', 7: '↘'
            }
            self.action_colors = [
                'red', 'blue', 'green', 'orange',
                'purple', 'cyan', 'magenta', 'brown'
            ]
            self.action_names = [
                'up', 'down', 'left', 'right',
                'up-left', 'up-right', 'down-left', 'down-right'
            ]
        else:
            # Fallback for other action counts
            self.action_symbols = {i: str(i) for i in range(self.n_actions)}
            self.action_colors = plt.cm.tab10(np.linspace(0, 1, self.n_actions)).tolist()
            self.action_names = [f'action_{i}' for i in range(self.n_actions)]
    
    def _state_dist_to_grid(self, nu: np.ndarray) -> np.ndarray:
        """Convert state distribution vector to 2D grid."""
        grid = np.zeros((self.grid_height, self.grid_width))
        
        for s_idx in range(self.n_states):
            cell = self.env.idx_to_state[s_idx]
            grid_x = cell[0] - self.min_x
            grid_y = cell[1] - self.min_y
            grid[grid_y, grid_x] = nu[s_idx]
        
        return grid
    
    def _compute_initial_distribution(self) -> np.ndarray:
        """Compute initial distribution using φ(unique_states) @ alpha."""
        with torch.no_grad():
            # One hot encoding of all states
            one_hot_unique_states = torch.eye(self.n_states).to(self.agent.device)
            
            phi_states = self.agent.encoder(one_hot_unique_states).cpu()
            # augment phi_states to account for extra row in phi_all_obs
            zero_col = torch.zeros((phi_states.shape[0], 1), device=phi_states.device, dtype=phi_states.dtype)
            phi_states = torch.cat([phi_states, zero_col], dim=-1)  # [n_states, feature_dim + 1]
            

            # Compute distribution: φᵀ @ alpha
            H = phi_states @ self.agent._phi_all_obs.cpu().T

            
            nu_init = H @ self.agent._alpha
        return nu_init.flatten().numpy()
    
    def _compute_current_distribution(self) -> np.ndarray:
        """Compute current occupancy distribution for all states."""
        if self.agent.gradient_coeff is None:
            return np.ones(self.n_states) / self.n_states
        
        nu_current = torch.zeros(self.n_states)
        all_states = self.all_state_ids_one_hot.to(self.agent.device)
        enc_all_states = self.agent.encoder(all_states).detach().cpu()  # [n_states, feature_dim]
        
        with torch.no_grad():
            for s_idx in range(self.n_states):
                
                enc_curr_obs = enc_all_states[s_idx].unsqueeze(0)  # [1, feature_dim]
                
                
                BM = self.agent.distribution_matcher.compute_BM(
                    self.agent._phi_all_obs @ self.agent._phi_all_obs.T * (self.agent.pi @ self.agent.E.T),
                    self.agent._psi_all
                )
                
                occupancy = self.agent.distribution_matcher.compute_discounted_occupancy(
                    phi_curr_states=enc_curr_obs,
                    phi_all_next_obs=self.agent._phi_all_next,
                    BM=BM,
                    alpha=self.agent.dataset.alpha
                )
                
                nu_current[s_idx] = occupancy.cpu().numpy().item()
        
        # Normalize
        nu_current = nu_current / (nu_current.sum() + 1e-10)
        return nu_current
    
    def _get_policy_per_state(self) -> np.ndarray:
        """Extract policy probabilities for each state."""
        policy_per_state = np.zeros((self.n_states, self.n_actions))
        
        all_states = self.all_state_ids_one_hot.to(self.agent.device)
        
        for s_idx in range(self.n_states):
            policy_per_state[s_idx] = self.agent.compute_action_probs(all_states[s_idx].unsqueeze(0))
        
        return policy_per_state
    
    def _compute_state_correlation_matrix(self) -> np.ndarray:
        """Compute correlation matrix between encoded unique states."""
        with torch.no_grad():
            # Get embeddings for all valid states
            valid_cells = [cell for cell in self.env.cells if cell != self.env.DEAD_STATE]
            valid_ids = [self.env.state_to_idx[cell] for cell in valid_cells]
            
            one_hot_states = torch.eye(self.n_states)[valid_ids].to(self.agent.device)
            embeddings = self.agent.encoder(one_hot_states).cpu()  # [n_states, feature_dim]
            normalized_embeddings = F.normalize(embeddings, p=2, dim=-1)
            
            # Compute correlation matrix: Φ @ Φᵀ
            correlation_matrix = normalized_embeddings @ normalized_embeddings.T
            
        return correlation_matrix.numpy()
    
    def _compute_state_to_states_correlation(self) -> np.ndarray:
        """
        Compute for each state how much it deviates from orthogonality with other states.
        
        For orthonormal embeddings we want:
        - φ(s) · φ(s') ≈ 0 for s ≠ s' (orthogonal)
        - ||φ(s)|| = 1 (normalized, enforced by encoder)
        
        Returns:
            Array of shape [n_states] with average absolute correlation for each state.
            Values close to 0.0 indicate good orthogonality.
            Values close to 1.0 indicate poor orthogonality (states are aligned).
        """
        with torch.no_grad():
            # Get embeddings for all states
            one_hot_states = torch.eye(self.n_states).to(self.agent.device)
            embeddings = self.agent.encoder(one_hot_states).cpu()  # [n_states, feature_dim]
            normalized_embeddings = F.normalize(embeddings, p=2, dim=-1)
            # Compute Gram matrix (correlation matrix): Φ @ Φᵀ
            # For normalized embeddings: G[i,j] = cos(angle between φ(i) and φ(j))
            gram_matrix = normalized_embeddings @ normalized_embeddings.T  # [n_states, n_states]
            
            # For each state, compute average deviation from orthogonality
            # We want off-diagonal elements to be close to 0
            state_orthogonality_deviation = torch.zeros(self.n_states)
            
            for i in range(self.n_states):
                # Get correlations with all other states (exclude self)
                correlations = gram_matrix[i].clone()
                correlations[i] = 0.0  # Exclude self-correlation
                
                # Average absolute correlation (deviation from orthogonality)
                # |cos(θ)| where θ is angle between embeddings
                # 0.0 = orthogonal (good), 1.0 = aligned (bad)
                state_orthogonality_deviation[i] = torch.abs(correlations).sum()#.mean()
            
        return state_orthogonality_deviation.numpy()
    
    def plot_embeddings_2d(self, save_path: str, use_tsne: bool = False):
        """
        Plot 2D projection of state embeddings using PCA or t-SNE.
        
        Args:
            save_path: Path to save the figure
            use_tsne: If True, use t-SNE; otherwise use PCA
        """
        with torch.no_grad():
            # Get embeddings for all valid states
            valid_cells = [cell for cell in self.env.cells if cell != self.env.DEAD_STATE]
            valid_ids = [self.env.state_to_idx[cell] for cell in valid_cells]
            
            one_hot_states = torch.eye(self.n_states)[valid_ids].to(self.agent.device)
            embeddings = self.agent.encoder(one_hot_states).cpu().numpy()
        
        # Dimensionality reduction
        if use_tsne:
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(valid_ids)-1))
            method_name = "t-SNE"
        else:
            reducer = PCA(n_components=2)
            method_name = "PCA"
        
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Color code by state ID or grid position
        colors = plt.cm.viridis(np.linspace(0, 1, len(valid_ids)))
        
        for idx, (cell, embedding_2d) in enumerate(zip(valid_cells, embeddings_2d)):
            ax.scatter(embedding_2d[0], embedding_2d[1], 
                      c=[colors[idx]], s=100, edgecolors='black', linewidth=1.5)
            ax.annotate(f"{cell}", (embedding_2d[0], embedding_2d[1]), 
                       fontsize=8, ha='center', va='center', color='green', fontweight='bold')
        
        ax.set_xlabel(f'{method_name} Component 1')
        ax.set_ylabel(f'{method_name} Component 2')
        ax.set_title(f'State Embeddings Visualization ({method_name})')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Embeddings visualization saved to: {save_path}")
        plt.close(fig)
    
    def plot_results(self, step: int, save_path: str = None):
        """
        Create comprehensive visualization of learning progress.
        
        Args:
            step: Current training step
            save_path: Path to save figure (optional)
        """
        # Adjust figure size based on number of actions
        n_action_cols = min(self.n_actions, 8)
        fig = plt.figure(figsize=(5 * max(5, n_action_cols), 12))
        
        # Compute current epsilon value
        epsilon = utils.schedule(self.agent.sink_schedule, step)
        
        # Get dataset novelty statistics
        novelty = self.agent._dataset_novelty_stats
        
        # Add parameter text with dataset novelty info
        param_text = (
            f"Step: {step}\n"
            f"γ = {self.agent.discount}\n"
            f"η = {self.agent.lr_actor}\n"
            f"λ = {self.agent.lambda_reg}\n"
            f"ε = {epsilon:.6f}\n"
            f"PMD steps = {self.agent.pmd_steps}\n"
            f"Actions = {self.n_actions}\n"
            f"\nDataset Novelty:\n"
            f"Current size: {novelty['total_current']}\n"
            f"New (s,a): {novelty['new_pairs']} ({novelty['new_percentage']:.1f}%)\n"
            f"Old (s,a): {novelty['old_pairs']}\n"
            f"Prev unique: {novelty['total_previous']}\n"
            f"New s': {novelty['new_next_states']} ({novelty['next_states_new_percentage']:.1f}%)\n"
            f"Old s': {novelty['old_next_states']}\n"
            f"Prev unique s': {novelty['total_previous_next_states']}"
        )
        fig.text(0.02, 0.98, param_text, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
          # Create subplot grid: 2x3 layout for the 6 plots
        ax_correlation = plt.subplot2grid((2, 3), (0, 0), colspan=1)
        ax_state_corr = plt.subplot2grid((2, 3), (0, 1), colspan=1)  # New plot
        ax_dataset = plt.subplot2grid((2, 3), (0, 2), colspan=1)
        ax_arrows = plt.subplot2grid((2, 3), (1, 0), colspan=1)
        ax_bars = plt.subplot2grid((2, 3), (1, 1), colspan=1)
        ax_placeholder = plt.subplot2grid((2, 3), (1, 2), colspan=1)  # Placeholder for future use
        ax_placeholder.axis('off')  # Hide for now
        
        # Compute and plot correlation matrix
        correlation_matrix = self._compute_state_correlation_matrix()
        self._plot_state_correlations(ax_correlation, correlation_matrix)
        
        # Compute and plot state-to-states correlation
        state_correlations = self._compute_state_to_states_correlation()
        self._plot_state_to_states_correlation(ax_state_corr, state_correlations)
        
        # Plot dataset occupancy
        self._plot_dataset_occupancy(ax_dataset)
        
        # Plot policy
        policy_per_state = self._get_policy_per_state()
        self._plot_policy_arrows(ax_arrows, policy_per_state)
        self._plot_policy_bars(ax_bars, policy_per_state)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nVisualization saved to: {save_path}")
            plt.close(fig)
    
    def _plot_state_correlations(self, ax, correlation_matrix):
        """Plot correlation matrix heatmap for encoded states."""
        im = ax.imshow(correlation_matrix, cmap='RdBu_r', interpolation='nearest', 
                      vmin=-1, vmax=1, aspect='auto')
        ax.set_title('State Embedding Correlations\n(Φ @ Φᵀ)')
        ax.set_xlabel('State Index')
        ax.set_ylabel('State Index')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation', rotation=270, labelpad=15)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    def _plot_state_to_states_correlation(self, ax, state_correlations):
        """
        Plot heatmap showing orthogonality quality of each state's embedding.
        
        The visualization shows how much each state deviates from being orthogonal
        to all other states. Lower values (lighter colors) indicate better orthogonality.
        
        Color interpretation:
        - Light yellow: embeddings are nearly orthogonal to others (GOOD)
        - Dark red: embeddings are aligned with others (BAD - representation collapse)
        """
        # Convert to grid matching environment structure
        normalized_state_correlations = state_correlations #/ max(state_correlations + 1e-10)
        grid = self._state_dist_to_grid(normalized_state_correlations)
        
        # Plot with YlOrRd colormap: yellow (good orthogonality) to red (poor orthogonality)
        im = ax.imshow(grid, cmap='YlOrRd', interpolation='nearest', vmin=0)
        ax.set_title('Orthogonality Quality\n(avg |⟨φ(s), φ(s\')⟩| for s≠s\')\nLower = Better')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # Add grid
        ax.set_xticks(np.arange(-0.5, self.grid_width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid_height, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Highlight dead state if lava enabled
        if hasattr(self.env, 'lava') and self.env.lava:
            dead_grid_x = -1 - self.min_x
            dead_grid_y = -1 - self.min_y
            dead_rect = Rectangle((dead_grid_x - 0.5, dead_grid_y - 0.5), 1, 1,
                                 fill=False, edgecolor='#CF1020', linewidth=3)
            ax.add_patch(dead_rect)
        
        # Add colorbar with clear labels
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Deviation from Orthogonality', 
                      rotation=270, labelpad=20)
        

    def plot_results_old(self, step: int, save_path: str = None):
        """
        Create comprehensive visualization of learning progress.
        
        Args:
            step: Current training step
            save_path: Path to save figure (optional)
        """
        fig = plt.figure(figsize=(24, 12))
        
        # Add parameter text
        param_text = (
            f"Step: {step}\n"
            f"γ = {self.agent.discount}\n"
            f"η = {self.agent.lr_actor}\n"
            f"λ = {self.agent.lambda_reg}\n"
            f"PMD steps = {self.agent.pmd_steps}"
        )
        fig.text(0.02, 0.98, param_text, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Create subplot grid (now 2x5 to include dataset occupancy)
        ax_init = plt.subplot2grid((2, 5), (0, 0), colspan=1)
        ax_current = plt.subplot2grid((2, 5), (0, 1), colspan=1)
        ax_dataset = plt.subplot2grid((2, 5), (0, 2), colspan=1)  # New subplot
        ax_arrows = plt.subplot2grid((2, 5), (0, 3), colspan=1)
        ax_bars = plt.subplot2grid((2, 5), (0, 4), colspan=1)
        
        # Second row: action heatmaps
        ax_actions = [plt.subplot2grid((2, 5), (1, i), colspan=1) for i in range(4)]
        
        # Compute distributions
        nu_init = self._compute_initial_distribution()
        nu_current = np.zeros(self.n_states) # TODO remove
        
        # Plot distributions
        self._plot_distribution(ax_init, nu_init, 'Initial Distribution')
        self._plot_distribution(ax_current, nu_current, 'Current Occupancy')
        self._plot_dataset_occupancy(ax_dataset)  # New plot
        
        # Plot policy
        policy_per_state = self._get_policy_per_state()
        self._plot_policy_arrows(ax_arrows, policy_per_state)
        self._plot_policy_bars(ax_bars, policy_per_state)
        
        # Plot action heatmaps
        self._plot_action_heatmaps(ax_actions, policy_per_state)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nVisualization saved to: {save_path}")
            plt.close(fig)
    
    def _plot_distribution(self, ax, nu, title):
        """Plot state distribution heatmap."""
        grid = self._state_dist_to_grid(nu)
        
        im = ax.imshow(grid, cmap='YlOrRd', interpolation='nearest', vmin=0)
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        ax.set_xticks(np.arange(-0.5, self.grid_width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid_height, 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Highlight dead state if lava enabled
        if hasattr(self.env, 'lava') and self.env.lava:
            dead_grid_x = -1 - self.min_x
            dead_grid_y = -1 - self.min_y
            dead_rect = Rectangle((dead_grid_x - 0.5, dead_grid_y - 0.5), 1, 1,
                                 fill=False, edgecolor='#CF1020', linewidth=3)
            ax.add_patch(dead_rect)
        
        plt.colorbar(im, ax=ax)
    
    def _plot_policy_arrows(self, ax, policy_per_state):
        """Plot policy as arrows showing most probable actions."""
        ax.set_xlim(-0.5, self.grid_width - 0.5)
        ax.set_ylim(self.grid_height - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.set_title('Policy Actions')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
        
        for s_idx in range(self.n_states):
            cell = self.env.idx_to_state[s_idx]
            x = cell[0] - self.min_x
            y = cell[1] - self.min_y
            
            max_prob = np.max(policy_per_state[s_idx])
            max_actions = np.where(np.isclose(policy_per_state[s_idx], max_prob, atol=1e-6))[0]
            arrow_text = ''.join([self.action_symbols[a] for a in max_actions])
            
            if cell == self.env.DEAD_STATE:
                rect = Rectangle((x - 0.4, y - 0.4), 0.8, 0.8,
                               facecolor='#CF1020', edgecolor='black', linewidth=0.5, alpha=0.3)
            else:
                rect = Rectangle((x - 0.4, y - 0.4), 0.8, 0.8,
                               facecolor='lightgray', edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            ax.text(x, y, arrow_text, ha='center', va='center',
                   fontsize=12, fontweight='bold')
    
    def _plot_policy_bars(self, ax, policy_per_state):
        """Plot policy as mini bar charts."""
        ax.set_xlim(-0.5, self.grid_width - 0.5)
        ax.set_ylim(self.grid_height - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.set_title('Policy Probabilities')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
        
        for s_idx in range(self.n_states):
            cell = self.env.idx_to_state[s_idx]
            x = cell[0] - self.min_x
            y = cell[1] - self.min_y
            
            if cell == self.env.DEAD_STATE:
                rect = Rectangle((x - 0.4, y - 0.4), 0.8, 0.8,
                               facecolor='#CF1020', edgecolor='black', linewidth=0.5, alpha=0.3)
            else:
                rect = Rectangle((x - 0.4, y - 0.4), 0.8, 0.8,
                               facecolor='lightgray', edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            
            probs = policy_per_state[s_idx]
            bar_width = 0.15
            bar_spacing = 0.2
            start_x = x - 1.5 * bar_spacing
            max_bar_height = 0.7
            
            for a_idx in range(self.n_actions):
                bar_x = start_x + a_idx * bar_spacing
                bar_height = probs[a_idx] * max_bar_height
                
                bar_rect = Rectangle((bar_x - bar_width/2, y + 0.35 - bar_height),
                                    bar_width, bar_height,
                                    facecolor=self.action_colors[a_idx],
                                    edgecolor='black', linewidth=0.3)
                ax.add_patch(bar_rect)
        
        legend_elements = [Patch(facecolor=self.action_colors[i], 
                                label=self.action_names[i])
                          for i in range(self.n_actions)]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    def _plot_action_heatmaps(self, axes_list, policy_per_state):
        """Plot heatmaps for each action's probability distribution."""
        # Metodo rimosso - non più necessario
        pass
    
    def _plot_dataset_occupancy(self, ax, title='Dataset State Occupancy'):
        """Plot heatmap of state occupancy in the internal dataset."""
        # Get dataset
        dataset_dict = self.agent.dataset._sampled_data if hasattr(self.agent.dataset, '_sampled_data') else self.agent.dataset.data
        observations = dataset_dict['observation']
        
        if observations.shape[0] == 0:
            ax.text(0.5, 0.5, 'Dataset vuoto', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(title)
            return
        
        # Convert one-hot observations to state indices
        state_indices = torch.argmax(observations, dim=1).numpy()
        
        # Count occurrences of each state
        state_counts = np.zeros(self.n_states)
        for s_idx in state_indices:
            state_counts[s_idx] += 1
        
        # Normalize to get occupancy distribution
        total = state_counts.sum()
        if total > 0:
            state_occupancy = state_counts #/ total
        else:
            state_occupancy = state_counts
        
        # Convert to grid
        grid = self._state_dist_to_grid(state_occupancy)
        
        # Plot
        im = ax.imshow(grid, cmap='Blues', interpolation='nearest', vmin=0)
        ax.set_title(f'{title}\n(Total: {int(total)} samples)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        ax.set_xticks(np.arange(-0.5, self.grid_width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid_height, 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Highlight dead state if lava enabled
        if hasattr(self.env, 'lava') and self.env.lava:
            dead_grid_x = -1 - self.min_x
            dead_grid_y = -1 - self.min_y
            dead_rect = Rectangle((dead_grid_x - 0.5, dead_grid_y - 0.5), 1, 1,
                                 fill=False, edgecolor='#CF1020', linewidth=3)
            ax.add_patch(dead_rect)
        
        plt.colorbar(im, ax=ax)
        
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
            device='cpu' #self.device At the moment forcing computatiosn on cpu, to save gpu memory
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
        # Find the discretized environment (could be the env itself or a wrapper)
        self.wrapped_env = env  # Keep reference to full wrapped env
        
        # Find the discrete environment interface (either native or discretized wrapper)
        self._discrete_env = self._find_discrete_env(env)
        
        # For compatibility, also keep unwrapped reference
        self.env = self._discrete_env
        
        timestep = env.reset()
        self.first_state = torch.tensor(timestep.observation).double()
      
        self.second_state = torch.eye(self.n_states)[1]
        
        # Initialize visualizer now that we have the environment
        self.visualizer = EmbeddingDistributionVisualizer(self)

        # If ideal mode, pre-populate the dataset
        if self.ideal and not self._ideal_dataset_filled:
            self._populate_ideal_dataset()
    
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
        """Pre-populate the dataset with all state-action pairs in ideal mode."""
        print("=== Populating ideal dataset with all state-action pairs ===")
        
        # Create all state-action combinations
        for state_idx in range(self.n_states):
            state = self.env.idx_to_state[state_idx]
            
            # Skip dead state if it exists
            if self.env.lava and state == self.env.DEAD_STATE:
                continue
            
            for action in range(self.n_actions):
                # Track unique pairs
                self.dataset.add_pairs(
                    torch.eye(self.n_states)[state_idx].numpy(),
                    action
                )
                
                # Get next state from environment dynamics
                next_state = self.env.step_from(state, action)
                next_state_idx = self.env.state_to_idx[next_state]
                
                # Create one-hot encodings
                obs_onehot = torch.zeros(self.n_states)
                obs_onehot[state_idx] = 1.0
                
                next_obs_onehot = torch.zeros(self.n_states)
                next_obs_onehot[next_state_idx] = 1.0
                
                # Add to current period data (not directly to data property)
                self.dataset._current_period_data['observation'] = torch.cat([
                    self.dataset._current_period_data['observation'],
                    obs_onehot.unsqueeze(0)
                ], dim=0)
                
                self.dataset._current_period_data['action'] = torch.cat([
                    self.dataset._current_period_data['action'],
                    torch.tensor([action], dtype=torch.long)
                ], dim=0)
                
                self.dataset._current_period_data['next_observation'] = torch.cat([
                    self.dataset._current_period_data['next_observation'],
                    next_obs_onehot.unsqueeze(0)
                ], dim=0)
                
                # Set alpha: 1.0 for the first entry (start state), 0.0 for others
                if self.dataset._current_period_data['alpha'].shape[0] == 0:
                    alpha_val = 1.0
                else:
                    alpha_val = 0.0
                
                self.dataset._current_period_data['alpha'] = torch.cat([
                    self.dataset._current_period_data['alpha'],
                    torch.tensor([alpha_val])
                ], dim=0)
        
        self._ideal_dataset_filled = True
        assert self.dataset.current_period_data_size == self.n_states * self.n_actions - (1 if self.env.lava else 0), \
            "Ideal dataset size does not match expected number of state-action pairs."
        print(f"Ideal dataset populated with {self.dataset.current_period_data_size} state-action pairs")
        print(f"Expected size: {self.n_states * self.n_actions}")
    
    
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
        logits = predicted_next/torch.norm(predicted_next, p=2, dim=1, keepdim=True) @ (encoded_next/torch.norm(encoded_next, p=2, dim=1, keepdim=True)).T  # [B, B]
        logits = logits - torch.max(logits, 1)[0][:, None]  # For numerical stability
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        contrastive_loss = self.cross_entropy_loss(logits, labels)
        
        # 4. Loss embeddings must sum to 1
        embedding_sum_loss = torch.abs(torch.sum(encoded_obs, dim=-1) - 1).sum()
        beta = 1 
        # 5. \phi(s) and \phi(s') must be close in L2 norm
        l2_loss = torch.norm(encoded_obs - encoded_next, p=2, dim=1).mean()

        loss =  1000*contrastive_loss + beta*embedding_sum_loss + 1*l2_loss
        
        # Optimize
        self.encoder_optimizer.zero_grad()
        self.transition_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()
        self.transition_optimizer.step()
        # Print losses
        print(f"Embeddings L2 norm mean : {torch.norm(encoded_obs, p=2, dim=1).mean().item():.4f}")
        print(f"Transition Model Losses: Contrastive={contrastive_loss.item():.4f}, EmbeddingSum={embedding_sum_loss.item():.4f}, L2={l2_loss.item():.4f}, Total={loss.item():.4f}")
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
            self.gradient_coeff = torch.zeros((self._phi_all_obs.shape[0]+1, 1))
            self.H = self._phi_all_obs @ self._phi_all_next.T # [n, n]
            self.unique_states = torch.eye(self.n_states).double()
            self.K = self._psi_all @ self._psi_all.T  # [n, n]

        epsilon = utils.schedule(self.sink_schedule, step)
        self.pi = torch.softmax(-self.lr_actor * (self.H.T@(self.gradient_coeff[:-1]*self.E)+ torch.ones(self._phi_all_next.shape[0], self.E.shape[1])*self.gradient_coeff[-1]), dim=1, dtype=torch.double)  # [z_x+1, n_actions]
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
        # print("Policy matrix sample (first 5 states):", self.pi[:5, :])

        for iteration in range(self.pmd_steps):
            self.gradient_coeff += self.distribution_matcher.compute_gradient_coefficient(
                M, 
                phi_all_next_obs = self._phi_all_next, 
                psi_all_obs_action = self._psi_all, 
                alpha = self._alpha,
                epsilon=epsilon
            ) 
            
            # print("Gradient coeff norm:", torch.linalg.norm(self.gradient_coeff))
            # print("Gradient last term:", self.gradient_coeff[-1].item())
            
            self.pi = torch.softmax(-self.lr_actor * (self.H.T@(self.gradient_coeff[:-1]*self.E)+ torch.ones(self._phi_all_next.shape[0], self.E.shape[1])*self.gradient_coeff[-1]), dim=1, dtype=torch.double)  # [z_x+1, n_actions]

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
    
    def _compute_dataset_novelty(self, tensors):
        """Compute how many pairs in current dataset are new vs. in previous unique set."""
        
        if len(tensors['observation']) == 0:
            self._dataset_novelty_stats = {
                'total_current': 0,
                'new_pairs': 0,
                'old_pairs': 0,
                'new_percentage': 0.0,
                'total_previous': len(self._previous_unique_pairs),
                'new_next_states': 0,
                'old_next_states': 0,
                'next_states_new_percentage': 0.0,
                'total_previous_next_states': len(self._previous_unique_next_states)
            }
            return
        
        # Get all state-action pairs from current dataset
        obs = tensors['observation']
        actions = tensors['action']
        next_obs = tensors['next_observation']
        assert len(obs) == len(actions) == len(next_obs), f"Mismatched tensor lengths in dataset novelty computation, got {len(obs)}, {len(actions)}, {len(next_obs)}"
        
        new_count = 0
        old_count = 0
        new_next_states_count = 0
        old_next_states_count = 0
        
        for i in range(len(obs)):
            # Convert observation to state index
            state_idx = torch.argmax(obs[i]).item()
            action_idx = actions[i].item()
            next_state_idx = torch.argmax(next_obs[i]).item()
            
            pair = (state_idx, action_idx)
            
            # Check (s,a) pair novelty
            if pair in self._previous_unique_pairs:
                old_count += 1
            else:
                new_count += 1
            
            # Check s' novelty
            if next_state_idx in self._previous_unique_next_states:
                old_next_states_count += 1
            else:
                new_next_states_count += 1
        
        total_current = len(obs)
        
        self._dataset_novelty_stats = {
            'total_current': total_current,
            'new_pairs': new_count,
            'old_pairs': old_count,
            'new_percentage': (new_count / total_current * 100) if total_current > 0 else 0.0,
            'total_previous': len(self._previous_unique_pairs),
            'new_next_states': new_next_states_count,
            'old_next_states': old_next_states_count,
            'next_states_new_percentage': (new_next_states_count / total_current * 100) if total_current > 0 else 0.0,
            'total_previous_next_states': len(self._previous_unique_next_states)
        }
    
    def _update_previous_unique_pairs(self, tensors):
        """Update the set of unique pairs from current dataset for next comparison."""
        
        if len(tensors['observation']) == 0:
            return
        
        current_unique_pairs = set()
        current_unique_next_states = set()
        obs = tensors['observation']
        actions = tensors['action']
        next_obs = tensors['next_observation']
        
        for i in range(len(obs)):
            state_idx = torch.argmax(obs[i]).item()
            action_idx = actions[i].item()
            next_state_idx = torch.argmax(next_obs[i]).item()
            
            current_unique_pairs.add((state_idx, action_idx))
            current_unique_next_states.add(next_state_idx)
        
        # Update previous unique pairs and next states for next iteration
        self._previous_unique_pairs = current_unique_pairs
        self._previous_unique_next_states = current_unique_next_states
    
    def _cache_features(self):
        """Pre-compute and cache dataset features."""
        # Compute dataset novelty before caching (compares current with previous)
        
        tensors = self.dataset.get_data(unique=self.unique_window)  # Set to True if you want unique (s,a) pairs
        
        self._compute_dataset_novelty(tensors)
        
        print("Caching features from dataset of size:", len(tensors['observation']))
        with torch.no_grad():
            obs = tensors['observation'][:len(tensors['next_observation'])].double()
            actions = tensors['action']
            next_obs = tensors['next_observation'].double()
            self._phi_all_obs = self.encoder(obs.to(self.device)).cpu()
            self._phi_all_next = self.encoder(next_obs.to(self.device)).cpu()
         
            # find first row in self._phi_all_next that is equal to first_state
            indices = torch.where(torch.all(next_obs == self.first_state, dim=1))[0]
            if indices.shape[0] == 0:
                indices = torch.where(torch.all(next_obs == self.second_state, dim=1))[0]
            print("Found indices for second state:", indices)

            self._psi_all = self._encode_state_action(self._phi_all_obs, actions).cpu()
           
            self._alpha = torch.zeros((self._phi_all_next.shape[0], 1)).double()
            # print("DEBUG: setting alpha for index", indices[0].item(), len(self._alpha))
            self._alpha[indices[0]] = 1.0  # set alpha to 1.0 for the first state
            self.E = F.one_hot(
                actions, 
                self.n_actions
            ).double().reshape(-1, self.n_actions)

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

        # After caching, update the previous unique pairs for next comparison
        self._update_previous_unique_pairs(tensors)

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

