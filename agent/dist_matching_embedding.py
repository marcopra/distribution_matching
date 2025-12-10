from collections import OrderedDict
import os
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Tuple, Optional, Dict
from dm_env import StepType, specs
from scipy.special import softmax
from scipy.linalg import cho_factor, cho_solve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import utils
from distribution_matching import DistributionVisualizer
torch.set_default_dtype(torch.double)


# ============================================================================
# Neural Network Components
# ============================================================================
class Encoder(nn.Module):
    def __init__(self, obs_shape, hidden_dim, feature_dim):
        super(Encoder, self).__init__()
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.repr_dim = feature_dim

        # self.fc = nn.Identity()
        self.fc =  nn.Sequential(
            nn.Linear(obs_shape[0], hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs.view(obs.shape[0], -1)
        h = self.fc(obs)
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
# Internal Dataset Management
# ============================================================================
class InternalDataset:
    """Manages the agent's internal experience buffer."""
    
    def __init__(self, dataset_type: str, n_states: int, n_actions: int):
        self.dataset_type = dataset_type
        self.n_states = n_states
        self.n_actions = n_actions
        self.expected_size = n_states * n_actions
        assert dataset_type in ("unique", "all"), "dataset_type must be 'unique' or 'all'"
        
        # Initialize as empty tensors
        self.data = {
            'observation': torch.empty((0, )), # save just the int
            'action': torch.empty((0,), dtype=torch.long), # save just the int
            'next_observation': torch.empty((0, )), # save just the int
            'alpha': torch.empty((0, )) # save just the int
        }
        self._unique_pairs: set = set()
        self._prev_obs: Optional[np.ndarray] = None
    
    @property
    def observation(self) -> Dict[str, torch.Tensor]:
        return self.data['observation']
    
    @property
    def action(self) -> Dict[str, torch.Tensor]:
        return self.data['action']
    
    @property
    def next_observation(self) -> Dict[str, torch.Tensor]:
        return self.data['next_observation']
    
    @property
    def alpha(self) -> Dict[str, torch.Tensor]:
        return self.data['alpha']
    
    @property
    def size(self) -> int:
        return len(self.data['observation'])
    
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
            return
        
        elif time_step.step_type in  (StepType.MID, StepType.LAST):
            pair = (np.argmax(self._prev_obs), time_step.action)
            
            if pair not in self._unique_pairs:
                self._unique_pairs.add(pair)
                # Convert to tensors and concatenate
                self.data['observation'] = torch.cat([
                    self.data['observation'],
                    torch.tensor(self._prev_obs).unsqueeze(0)
                ], dim=0)
                self.data['action'] = torch.cat([
                    self.data['action'],
                    torch.tensor([time_step.action], dtype=torch.long).unsqueeze(0)
                ], dim=0)
                self.data['next_observation'] = torch.cat([
                    self.data['next_observation'],
                    torch.tensor(time_step.observation).unsqueeze(0)
                ], dim=0)
                # first time we have a unique pair, alpha is 1.0 else is 0
                if len(self._unique_pairs) == 1:
                    self.data['alpha'] = torch.cat([
                        self.data['alpha'],
                        torch.tensor([1.0]).unsqueeze(0)
                    ], dim=0)
                else:
                    self.data['alpha'] = torch.cat([ # TODO alpha is one when next obs is the upper right cell
                        self.data['alpha'],
                        torch.tensor([0.0]).unsqueeze(0)
                    ], dim=0)
            self._prev_obs = time_step.observation
        else:
                raise ValueError("Unknown step type")
        
    
    def _add_all(self, time_step):
        """Add all transitions."""
        # Implementation for storing all transitions
        # *** Store all dataset ***
        if time_step.step_type == StepType.FIRST:  # first step
            self.internal_dataset['observation'] = torch.cat([
                self.data['observation'],
                torch.tensor(time_step.observation).unsqueeze(0)
            ], dim=0)

            # first time we have a unique pair, alpha is 1.0 else is 0
            if self.data['observation'].shape[0] == 1:
                self.data['alpha'] = torch.cat([
                    self.data['alpha'],
                    torch.tensor([1.0]).unsqueeze(0)
                ], dim=0)
            else:
                self.data['alpha'] = torch.cat([
                    self.data['alpha'],
                    torch.tensor([0.0]).unsqueeze(0)
                ], dim=0)

        elif time_step.step_type == StepType.MID:  # mid step
            self.data['observation'] = torch.cat([
                self.data['observation'],
                torch.tensor(time_step.observation).unsqueeze(0)
            ], dim=0)
            self.data['action'] = torch.cat([
                self.data['action'],
                torch.tensor([time_step.action], dtype=torch.long)
            ], dim=0)
            self.data['next_observation'] = torch.cat([
                self.data['next_observation'],
                torch.tensor(time_step.observation).unsqueeze(0)
            ], dim=0)
            self.data['alpha'] = torch.cat([
                    self.data['alpha'],
                    torch.tensor([0.0]).unsqueeze(0)
                ], dim=0)
        elif time_step.step_type == StepType.LAST:  # last step
            self.data['action'] = torch.cat([
                self.data['action'],
                torch.tensor([time_step.action], dtype=torch.long)
            ], dim=0)
            self.data['next_observation'] = torch.cat([
                self.data['next_observation'],
                torch.tensor(time_step.observation).unsqueeze(0)
            ], dim=0)
            assert self.data['observation'].shape[0] == self.data['action'].shape[0] == self.data['next_observation'].shape[0], f"len(obs): {self.data['observation'].shape[0]}, len(action): {self.data['action'].shape[0]}, len(next_obs): {self.data['next_observation'].shape[0]}"
        else:
            raise ValueError("Unknown step type")
        
    def get_data(self) -> Dict[str, torch.Tensor]:
        """Retrieve the internal dataset."""
        return self.data
    

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
            phi_all_next_obs:torch.Tensor, 
            BM: torch.Tensor,
            alpha: torch.Tensor
        ) -> torch.Tensor:
        """Compute discounted occupancy: ν = (1-γ)Φᵀ(I - γBM)⁻¹α."""
        N = BM.shape[0]
        identity = torch.eye(N, device=self.device)
        

        inv_term = torch.linalg.solve(identity - self.gamma * BM, alpha)
        
        occupancy = (1 - self.gamma) *  phi_all_next_obs.T @ inv_term
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

        # s,v,d= torch.linalg.svd(gram_matrix)
        # print(f"SVD of gram matrix:  {v}, V shape {v.shape}, D shape {d.shape}")
        
        L = torch.linalg.cholesky(gram_matrix)
        BM = torch.cholesky_solve(M, L)
        return BM
    
    def compute_gradient_coefficient(
            self, 
            M: torch.Tensor, 
            K: torch.Tensor,
            phi_all_next_obs:torch.Tensor, 
            psi_all_obs_action:torch.Tensor, 
            alpha:torch.Tensor
        ) -> torch.Tensor:
        """Compute gradient coefficient for policy update."""
        # Identity matrix
        I_n = torch.eye(psi_all_obs_action.shape[0], device=self.device)

        # Symmetric positive definite matrix A = ψψᵀ + λI
        A = psi_all_obs_action @ psi_all_obs_action.T + self.lambda_reg * I_n

        # Compute Cholesky decomposition and solve: BM = A⁻¹M
        L = torch.linalg.cholesky(A)
        BM = torch.cholesky_solve(M, L)

        # gradient = 2 γ (1 - γ)² A⁻ᵀ (I - γ A⁻¹M)⁻ᵀΦΦᵀ(I - γ A⁻¹M)⁻¹ α
        # Using the precomputed terms and solves:
        # (I - γ A⁻¹M)⁻ᵀΦ = [Φᵀ(I - γ A⁻¹M)⁻¹]ᵀ
        symmetric_term = torch.linalg.solve((I_n - self.gamma * BM).T, phi_all_next_obs)

        # Left term: A⁻ᵀ(I - γBM)⁻ᵀΦ
        # Solve A^T x = left_term_without_b using Cholesky
        L_T = torch.linalg.cholesky(A.T)
        left_term = torch.cholesky_solve(symmetric_term, L_T)

        # Right term: Φᵀ(I - γBM)⁻¹α
        right_term = symmetric_term.T @ alpha

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
        
        # Action symbols and colors
        self.action_symbols = {0: '↑', 1: '↓', 2: '←', 3: '→'}
        self.action_colors = ['red', 'blue', 'green', 'orange']
        self.action_names = ['up', 'down', 'left', 'right']
    
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
            

            # Compute distribution: φᵀ @ alpha
            H = phi_states @ self.agent._phi_all_obs.cpu().T
            alpha = self.agent.dataset.alpha
            nu_init = H @ alpha
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
    
    def plot_results(self, step: int, save_path: str = None):
        """
        Create comprehensive visualization of learning progress.
        
        Args:
            step: Current training step
            save_path: Path to save figure (optional)
        """
        fig = plt.figure(figsize=(20, 12))
        
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
        
        # Create subplot grid
        ax_init = plt.subplot2grid((2, 4), (0, 0), colspan=1)
        ax_current = plt.subplot2grid((2, 4), (0, 1), colspan=1)
        ax_arrows = plt.subplot2grid((2, 4), (0, 2), colspan=1)
        ax_bars = plt.subplot2grid((2, 4), (0, 3), colspan=1)
        
        # Second row: action heatmaps
        ax_actions = [plt.subplot2grid((2, 4), (1, i), colspan=1) for i in range(4)]
        
        # Compute distributions
        nu_init = self._compute_initial_distribution()
        nu_current = np.zeros(self.n_states) # self._compute_current_distribution()
        
        # Plot distributions
        self._plot_distribution(ax_init, nu_init, 'Initial Distribution')
        self._plot_distribution(ax_current, nu_current, 'Current Occupancy')
        
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
        for a_idx, ax in enumerate(axes_list):
            grid_action = np.zeros((self.grid_height, self.grid_width))
            
            for s_idx in range(self.n_states):
                cell = self.env.idx_to_state[s_idx]
                grid_x = cell[0] - self.min_x
                grid_y = cell[1] - self.min_y
                grid_action[grid_y, grid_x] = policy_per_state[s_idx, a_idx]
            
            im = ax.imshow(grid_action, cmap='YlOrRd', interpolation='nearest',
                          vmin=0, vmax=1)
            ax.set_title(f'π({self.action_names[a_idx]}|s)')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_xticks(np.arange(-0.5, self.grid_width, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, self.grid_height, 1), minor=True)
            ax.grid(which='minor', color='white', linestyle='-',
                   linewidth=0.5, alpha=0.5)
            
            if hasattr(self.env, 'lava') and self.env.lava:
                dead_grid_x = -1 - self.min_x
                dead_grid_y = -1 - self.min_y
                dead_rect = Rectangle((dead_grid_x - 0.5, dead_grid_y - 0.5), 1, 1,
                                     fill=False, edgecolor='#CF1020', linewidth=2)
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
                 T_learning_steps,
                 internal_dataset_type: str = "unique",
                 device: str = "cpu",
                 linear_actor: bool = False
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
        self.T_learning_steps = T_learning_steps
        self.batch_size = batch_size
        self.update_every_steps = update_every_steps
        self.update_actor_every_steps = update_actor_every_steps
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.device = device
        self.internal_dataset_type = internal_dataset_type
        self.pmd_steps = pmd_steps

        self.gradient_coeff = None

        self.num_expl_steps = num_expl_steps
        self.lambda_reg = lambda_reg
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
        
        self.dataset = InternalDataset(self.internal_dataset_type, self.n_states, self.n_actions)
        
        # Optimizers
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(), 
            lr=lr_encoder
        )
        self.transition_optimizer = torch.optim.Adam(
            self.transition_model.parameters(),
            lr=lr_T
        )
        

        self.training = False
        # Initialize visualizer (will be properly set after env is inserted)
        self.visualizer = None
    
    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.transition_model.train(training)
    
    def insert_env(self, env):
        self.env = env.unwrapped # This is needed just for visualization. # TODO remove when using non-custom envs
        # Initialize visualizer now that we have the environment
        self.visualizer = EmbeddingDistributionVisualizer(self)

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
            
            H = enc_obs @ self._phi_all_obs.T  # [1, num_unique]
            logits = -self.lr_actor * H @ self.gradient_coeff  # [1, n_actions]
            probs = torch.softmax(logits, dim=1) # [1, n_actions]
            if torch.sum(probs) == 0.0 or torch.isnan(torch.sum(probs)):
                raise ValueError("action_probs sum to zero or NaN")
            return probs.numpy().flatten()

    
    def act(self, obs, meta, step, eval_mode):
        if step < self.num_expl_steps:
            return np.random.randint(self.n_actions)
        
        # Compute action probabilities
        action_probs = self.compute_action_probs(obs)
        
        # Sample action
        return np.random.choice(self.n_actions, p=action_probs)
    

    
    def _is_T_sufficiently_initialized(self, step: int) -> bool:
        """Check if transition learning phase is complete."""
        # Could add convergence criteria here
        return step >= self.num_expl_steps + self.T_learning_steps  # Example threshold
    
    def update_transition_matrix(self, obs, action, next_obs):
        metrics = dict()

        # Encode
        encoded_obs = self.encoder(obs.double())
        encoded_next = self.encoder(next_obs.double())
        encoded_state_action = self._encode_state_action(encoded_obs, action)
        
        # Predict next state
        predicted_next = self.transition_model(encoded_state_action)
        
        # Compute loss
        loss = F.mse_loss(predicted_next, encoded_next)
        
        # Optimize
        self.encoder_optimizer.zero_grad()
        self.transition_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()
        self.transition_optimizer.step()

        print(f"transition_loss: {loss.item()}")
        if self.use_tb or self.use_wandb:
            metrics['transition_loss'] = loss.item()
        return metrics



    def update_actor(self, obs):
        """Update policy using Projected Mirror Descent."""
        metrics = dict()
        # Compute features for internal dataset
        if not hasattr(self, '_features_cached'):
            self._cache_features()
            self.gradient_coeff = torch.zeros((self.dataset.observation.shape[0], self.n_actions))
            self.H = self._phi_all_obs @ self._phi_all_next.T  # [num_unique, num_unique]
            self.K = self._phi_all_next @ self._phi_all_next.T  # [num_unique, num_unique]
            self.unique_states = torch.eye(self.n_states).double()

        self.pi = torch.softmax(-self.lr_actor * self.H.T@self.gradient_coeff, dim=1, dtype=torch.double)  # [num_unique, n_actions]

        M = self.H*(self.E@self.pi.T) # [num_unique, num_unique]

        nu_pi = self.distribution_matcher.compute_nu_pi(
                phi_all_next_obs = self._phi_all_next,
                BM = self.distribution_matcher.compute_BM(M, self._psi_all),
                alpha=self._alpha
        )
        actor_loss = torch.linalg.norm(nu_pi)**2
        print(f"Actor loss (squared norm of occupancy measure): {actor_loss}")
        # print("Gradient coeff norm:", torch.linalg.norm(self.gradient_coeff))
        # print("Policy matrix sample (first 5 states):", self.pi[:5, :])

        for iteration in range(self.pmd_steps):
            self.gradient_coeff += self.distribution_matcher.compute_gradient_coefficient(
                M, 
                self.K,
                phi_all_next_obs = self._phi_all_next, 
                psi_all_obs_action = self._psi_all, 
                alpha = self._alpha) * self.E
            
            # print(self.lr_actor * self.H.T@self.gradient_coeff)
            self.pi = torch.softmax(-self.lr_actor * self.H.T@self.gradient_coeff, dim=1, dtype=torch.double)

            M = self.H*(self.E@self.pi.T) # [num_unique, num_unique]

            if iteration % 10 == 0 or iteration == self.pmd_steps - 1:
                nu_pi = self.distribution_matcher.compute_nu_pi(
                        phi_all_next_obs = self._phi_all_next,
                        BM = self.distribution_matcher.compute_BM(M, self._psi_all),
                        alpha=self._alpha
                )
                actor_loss = torch.linalg.norm(nu_pi)**2
                print(f"  PMD Iteration {iteration}, Actor loss: {actor_loss}")
            

        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss
   
        return metrics
    
    def _cache_features(self):
        """Pre-compute and cache dataset features."""
        tensors = self.dataset.get_data()

        with torch.no_grad():
            obs = tensors['observation'].double().unsqueeze(-1)
            actions = tensors['action']
            next_obs = tensors['next_observation'].double().unsqueeze(-1)
            
            self._phi_all_obs = self.encoder(obs.to(self.device)).cpu()
            self._phi_all_next = self.encoder(next_obs.to(self.device)).cpu()
            first_state = self._phi_all_obs[0]
            # find first row in self._phi_all_next that is equal to first_state
            indices = torch.where(torch.all(self._phi_all_next == first_state, dim=1))[0]

            
            self._psi_all = self._encode_state_action(self._phi_all_obs, actions).cpu()
           
            self._alpha = torch.zeros((self._phi_all_next.shape[0], 1)).double()
            self._alpha[indices[0]] = 1.0  # set alpha to 1.0 for the first state
            self.E = F.one_hot(
                actions, 
                self.n_actions
            ).double().reshape(-1, self.n_actions)

            # optimal_T = self.transition_model.compute_closed_form(self._psi_all, self._phi_all_next, self.lambda_reg)
            # print(f"==== Optimal T error {F.mse_loss(optimal_T @ self._psi_all.T, self._phi_all_next.T).item()} ====")
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

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # # augment and encode
        # obs = self.aug_and_encode(obs)
        # with torch.no_grad():
        #     next_obs = self.aug_and_encode(next_obs)

        if self.use_tb or self.use_wandb:
            metrics['batch_reward'] = reward.mean().item()
        
        # update transition matrix
        # if dataset is not yet complete, we always update T
        # if dataset is complete, we update T only during the initial learning phase
        if not self.dataset.is_complete or not self._is_T_sufficiently_initialized(step):
            metrics.update(self.update_transition_matrix(obs, action, next_obs))

        print(self.dataset.is_complete, "dataset complete")
        # If T is not sufficiently initialized, skip actor update
        if self._is_T_sufficiently_initialized(step) is False:   
            metrics['actor_loss'] = 100.0  # dummy value
            return metrics
        
        if step % self.update_actor_every_steps == 0:     
            # update actor
            metrics.update(self.update_actor(obs))
        
            if self.visualizer is not None:
                save_path = os.path.join(os.getcwd(), f"plot_step_{step}.png")
                self.visualizer.plot_results(step, save_path=save_path)
                print(f"Visualization saved to: {save_path}")
        return metrics

