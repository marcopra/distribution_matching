'''
python pretrain.py agent=dist_matching_embedding_augmented_continuous env.render_mode=rgb_array save_video=false save_train_video=false num_train_frames=300000 use_wandb=false agent.T_init_steps=100 configs/env=continuous_four_rooms num_seed_frames=2000 agent.update_actor_every_steps=1600 agent.window_size=1000 agent.epsilon_schedule=0.3 "agent.sink_schedule='linear(0.0, 0.1, 1000000)'"  agent.lr_actor=0.1 agent.pmd_steps=200 obs_type=pixels agent.T_init_steps=200 agent.ideal=true  agent.lambda_reg=1e-8  wandb_project="distribution_matching" agent.feature_dim=1000 
'''
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
import math

SINK_STATE_VALUE = 1

# ============================================================================
# Neural Network Components
# ============================================================================
class Encoder(nn.Module):
    def __init__(self, obs_shape, hidden_dim, feature_dim, embedding_type):
        super(Encoder, self).__init__()
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.repr_dim = feature_dim
        assert embedding_type in ['rff', 'cl'], "Unsupported embedding type"
        self.embedding_type = embedding_type
        if self.embedding_type == 'rff':
            self.linear = nn.Linear(
                obs_shape[0],
                feature_dim,
                bias=True
            )

            # --- inizializzazione random features ---
            nn.init.normal_(self.linear.weight, mean=0.0, std=1.0)
            nn.init.uniform_(self.linear.bias, 0.0, 2 * math.pi)

            # congela i parametri
            for p in self.parameters():
                p.requires_grad = False

        elif self.embedding_type == 'cl':
            self.fc =  nn.Sequential(
                nn.Linear(obs_shape[0], hidden_dim, bias=False),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.ReLU(),
                nn.Linear(hidden_dim, feature_dim, bias=False),
            )

            self.apply(utils.weight_init)

    def forward(self, obs):
        if self.embedding_type == 'rff':
            # Random Fourier Features
            h = self.linear(obs)
            h = torch.cos(h)

            # opzionale ma spesso utile
            h = F.normalize(h, p=2, dim=-1)
        elif self.embedding_type == 'cl':
            obs = obs.view(obs.shape[0], -1)
            h = self.fc(obs)
            h = F.normalize(h, p=1, dim=-1)
        return h

class CNNEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3

        self.conv = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                  nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                  nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                  nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                  nn.ReLU())

        self.repr_dim = 32 * 35 * 35

        self.projector = nn.Sequential(
            nn.Linear(self.repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh()
            # nn.ReLU( )
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.
        h = self.conv(obs)
        h = h.view(h.shape[0], -1)
        # h = F.softmax(h/0.1, dim=-1)
        return h

    def encode_and_project(self, obs):
        h = self.forward(obs)
        z = self.projector(h)
        z = F.normalize(z, p=1, dim=-1)
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
        tilde_alpha = torch.ones((alpha.shape[0] + 1, 1), dtype=alpha.dtype)
        tilde_alpha[:-1] = alpha

        # ** COMPUTATION STEP **
        # Compute Cholesky decomposition and solve: B̃M̃ = Ã⁻¹M̃
        A = K + self.lambda_reg * torch.eye(N)
        L = torch.linalg.cholesky(A)
        BM = torch.cholesky_solve(M, L)
        
        # M̃ augmented to be [M 0; 0 1]
        tilde_BM = torch.zeros(BM.shape[0] + 1, BM.shape[1] + 1, dtype=BM.dtype)
        tilde_BM[:-1, :-1] = BM
        tilde_BM[-1, -1] = 1.0

        inv_term = torch.linalg.solve( torch.eye(N+1) - self.gamma * tilde_BM, tilde_alpha)
        
        sink_state = torch.zeros((phi_all_next_obs.shape[1],1))
        sink_state[-1] = SINK_STATE_VALUE*epsilon

        # Computing Ψ̃ and Φ̃ are now of shape [N+1, d*|A| + 2] and [N+1, d + 2] respectively
        upper_left = phi_all_next_obs.T - sink_state@torch.ones((1, psi_all_obs_action.shape[1]), dtype=psi_all_obs_action.dtype)@psi_all_obs_action.T
        tilde_phi_all_next_obs_transposed = torch.zeros((phi_all_next_obs.shape[1]+1, phi_all_next_obs.shape[0]+1), dtype=phi_all_next_obs.dtype)
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
        I_n_plus1 = torch.eye(psi_all_obs_action.shape[0])

        sink_state = torch.zeros((phi_all_next_obs.shape[1],1), dtype=phi_all_next_obs.dtype)
        sink_state[-1] = SINK_STATE_VALUE*epsilon

        # Computing Ψ̃ and Φ̃ are now of shape [N+1, d*|A| + 2] and [N+1, d + 2] respectively
        upper_left = phi_all_next_obs.T - sink_state@torch.ones((1, psi_all_obs_action.shape[1]), dtype=psi_all_obs_action.dtype)@psi_all_obs_action.T
        tilde_phi_all_next_obs_transposed = torch.zeros((phi_all_next_obs.shape[1]+1, phi_all_next_obs.shape[0]+1), dtype=phi_all_next_obs.dtype)
        tilde_phi_all_next_obs_transposed[:upper_left.shape[0], :upper_left.shape[1]] = upper_left
        assert sink_state.shape[0] == upper_left.shape[0], "Sink state and upper left matrix row size mismatch"
        tilde_phi_all_next_obs_transposed[:sink_state.shape[0], -1:] = sink_state
        # tilde_phi_all_next_obs_transposed[-1, -1] = 1.0
        tilde_phi_all_next_obs = tilde_phi_all_next_obs_transposed.T
        assert torch.all(tilde_phi_all_next_obs_transposed[:sink_state.shape[0], -1:] == sink_state), "Last column of tilde_phi_all_next_obs should be sink_state"
 
        # Ã augmented to be [A 0; 0 1]
        # Symmetric positive definite matrix A = ψψᵀ + λI
        A = psi_all_obs_action @ psi_all_obs_action.T + self.lambda_reg * I_n_plus1
        tilde_A = torch.zeros(A.shape[0] + 1, A.shape[1] + 1, dtype=A.dtype)
        tilde_A[:-1, :-1] = A
        tilde_A[-1, -1] = 1.0

        # M̃ augmented to be [M 0; 0 1]
        tilde_M = torch.zeros(M.shape[0] + 1, M.shape[1] + 1, dtype=M.dtype)
        tilde_M[:-1, :-1] = M
        tilde_M[-1, -1] = 1.0

        # α̃ augmented to be [α; 1]
        tilde_alpha = torch.ones((alpha.shape[0] + 1, 1), dtype=alpha.dtype)
        tilde_alpha[:-1] = alpha

        # ** COMPUTATION STEP **
        # Compute Cholesky decomposition and solve: BM = A⁻¹M
        L = torch.linalg.cholesky(A)
        BM = torch.cholesky_solve(M, L)
        tilde_B_tilde_M = torch.zeros(BM.shape[0] + 1, BM.shape[1] + 1, dtype=BM.dtype)
        tilde_B_tilde_M[:-1, :-1] = BM
        tilde_B_tilde_M[-1, -1] = 1.0

        # gradient = 2 γ (1 - γ)² Ã⁻ᵀ (I - γ Ã⁻¹M̃)⁻ᵀΦ̃Φ̃ᵀ(I - γ Ã⁻¹M̃)⁻¹ α̃ 
        # Using the precomputed terms and solves:
        # (I - γ Ã⁻¹M̃)⁻ᵀΦ̃ = [Φ̃ᵀ(I - γ Ã⁻¹M̃)⁻¹]ᵀ
        I_n_plus1 = torch.eye(tilde_B_tilde_M.shape[0], dtype=tilde_B_tilde_M.dtype)
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
    """Visualizer for embedding-based distribution matching results.
    
    Works with both proprioceptive (state) and pixel observations by:
    - Using raw states directly for proprioceptive observations
    - Using proprio_observation from pixel observations for 2D trajectory plots
    - Using actual pixel observations for embedding visualization
    """
    
    def __init__(self, agent, n_trajectories: int = 50, traj_length: int = 50):
        """
        Initialize visualizer with agent reference.
        
        Args:
            agent: DistMatchingEmbeddingAgent instance
            n_trajectories: Number of trajectories to sample
            traj_length: Length of each trajectory
        """
        self.agent = agent
        # Store both wrapped and unwrapped environment
        self.env_unwrapped = agent.env  # This is the unwrapped env
        self.env_wrapped = agent.wrapped_env if hasattr(agent, 'wrapped_env') else None
        
        self.n_trajectories = n_trajectories
        self.traj_length = traj_length
     
        # Detect observation type
        self.obs_type = agent.obs_type
        self.is_pixel_obs = (self.obs_type == 'pixels')
        
        # Get action information
        self.n_actions = self.agent.n_actions
        self.action_names = ['up', 'down', 'left', 'right', 'up-left', 'up-right', 'down-left', 'down-right'][:self.n_actions]
        self.action_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray'][:self.n_actions]
        
        # Get environment bounds (works for both pixel and state obs)
        self.obs_low = self.env_unwrapped.observation_space.low
        self.obs_high = self.env_unwrapped.observation_space.high
        print(f"Visualizer bounds: x=[{self.obs_low[0]:.2f}, {self.obs_high[0]:.2f}], "
              f"y=[{self.obs_low[1]:.2f}, {self.obs_high[1]:.2f}]")
        
        print(f"Visualizer initialized for {self.obs_type} observations")
        
        # Pre-sample grid states and render observations
        self._sample_and_cache_grid_states()
    
    def _sample_and_cache_grid_states(self):
        """Sample grid states once and cache both 2D positions and rendered observations."""
        # Sample 2D states
        self.grid_states_2d = self.agent._sample_grid_positions()
        
        # Render observations for each state
        self.grid_observations = []
        print(f"Pre-rendering {len(self.grid_states_2d)} grid observations...")
        
        for idx, state_2d in enumerate(self.grid_states_2d):
            obs = self.agent.render_observation_from_state(state_2d)
            self.grid_observations.append(obs)
            
            if (idx + 1) % 50 == 0:
                print(f"  Rendered {idx + 1}/{len(self.grid_states_2d)} observations")
        
        self.grid_observations = np.array(self.grid_observations)
        print(f"Grid observations cached: shape={self.grid_observations.shape}")
    
    def _get_state_and_obs_from_timestep(self, time_step) -> Tuple[np.ndarray, np.ndarray]:
        """Extract both 2D state (for plotting) and observation (for embedding).
        
        Returns:
            state_2d: [2,] array for trajectory plotting
            observation: Full observation for embedding (image or state)
        """
        # Handle both ExtendedTimeStep and standard gym tuple
        if hasattr(time_step, 'proprio_observation'):
            # ExtendedTimeStep from wrapped env
            state_2d = time_step.proprio_observation[:2]  # First 2 dims are x,y position
            observation = time_step.observation  # Full observation (image or state)
        else:
            # Standard gym tuple (obs, info) or just obs
            if isinstance(time_step, tuple):
                obs = time_step[0]
            else:
                obs = time_step
            
            # For unwrapped envs, observation is the state
            state_2d = obs[:2] if len(obs) >= 2 else obs
            observation = obs
        
        return state_2d, observation
    
    def _sample_trajectories(self):
        """Sample trajectories from current policy.
        
        Returns:
            trajectories: List of 2D state arrays [traj_length, 2] for plotting
            observations: List of observation arrays for embeddings
            action_sequences: List of action sequences
        """
        trajectories = []
        observations = []
        action_sequences = []
        
        # Use wrapped env if available, otherwise unwrapped
        env_to_use = self.env_wrapped if self.env_wrapped is not None else self.env_unwrapped
        
        for _ in range(self.n_trajectories):
            time_step = env_to_use.reset()
            state_2d, obs = self._get_state_and_obs_from_timestep(time_step)
            trajectory = [state_2d.copy()]
            obs_list = [obs.copy()]
            actions = []
            
            for _ in range(self.traj_length):
                # Get observation for agent.act()
                if hasattr(time_step, 'observation'):
                    obs_for_agent = time_step.observation
                else:
                    obs_for_agent = time_step[0] if isinstance(time_step, tuple) else time_step
                
                # Get action from agent
                action = self.agent.act(
                    obs_for_agent, 
                    None, 
                    self.agent.num_expl_steps + 1, 
                    eval_mode=True
                )
                
                # Step environment
                time_step = env_to_use.step(action)
                state_2d, obs = self._get_state_and_obs_from_timestep(time_step)
                
                trajectory.append(state_2d.copy())
                obs_list.append(obs.copy())
                actions.append(action)
                
                # Check termination
                if hasattr(time_step, 'last'):
                    done = time_step.last()
                else:
                    # Standard gym format: (obs, reward, terminated, truncated, info)
                    done = time_step[2] or time_step[3] if len(time_step) >= 4 else False
                
                if done:
                    break
            
            trajectories.append(np.array(trajectory))
            observations.append(obs_list)
            action_sequences.append(actions)
        
        return trajectories, observations, action_sequences
    
    
    def _plot_trajectories(self, ax, trajectories):
        """Plot trajectories in 2D state space using proprio_observation."""
        if len(trajectories) == 0 or trajectories[0].shape[-1] != 2:
            ax.text(0.5, 0.5, 'Trajectory visualization\nrequires 2D position data', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        ax.set_xlim(self.obs_low[0], self.obs_high[0])
        ax.set_ylim(self.obs_low[1], self.obs_high[1])
        ax.set_aspect('equal')
        ax.set_title(f'{self.n_trajectories} Sampled Trajectories')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
        
        colors = plt.cm.viridis(np.linspace(0, 1, self.n_trajectories))
        
        for idx, traj in enumerate(trajectories):
            ax.scatter(traj[:, 0], traj[:, 1], c=colors[idx], alpha=0.6, s=20)
            ax.scatter(traj[0, 0], traj[0, 1], c=[colors[idx]], s=80, marker='o', 
                      edgecolors='black', linewidth=1.5, zorder=10)
            ax.scatter(traj[-1, 0], traj[-1, 1], c=[colors[idx]], s=120, marker='*',
                      edgecolors='black', linewidth=1.5, zorder=10)
        
        # Draw environment structure
        if hasattr(self.env_unwrapped, 'walkable_areas'):
            for area in self.env_unwrapped.walkable_areas:
                rect = Rectangle((area[0], area[1]), area[2], area[3],
                               fill=False, edgecolor='gray', linewidth=1, linestyle='--', alpha=0.5)
                ax.add_patch(rect)
    
    def _plot_dataset_occupancy(self, ax):
        """Plot 2D histogram of dataset observations."""
        dataset_dict = (self.agent.dataset._sampled_data 
                       if hasattr(self.agent.dataset, '_sampled_data') 
                       else self.agent.dataset.data)
        observations = dataset_dict['observation']
        
        if observations.shape[0] == 0:
            ax.text(0.5, 0.5, 'Empty Dataset', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Dataset Observation Distribution')
            return
        
        # For pixel observations, extract 2D positions from dataset
        if self.is_pixel_obs:
            # The dataset should store proprio_observation separately
            # For now, we can't visualize pixel dataset without position info
            ax.text(0.5, 0.5, 'Dataset visualization\nfor pixel obs\ncoming soon', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        obs_np = observations.cpu().numpy()
        
        if obs_np.shape[-1] < 2:
            ax.text(0.5, 0.5, f'Cannot visualize\n{obs_np.shape[-1]}D observations', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Use first 2 dimensions for plotting
        obs_2d = obs_np[:, :2]
        
        # Smart binning
        x_range = self.obs_high[0] - self.obs_low[0]
        y_range = self.obs_high[1] - self.obs_low[1]
        move_delta = getattr(self.env_unwrapped, 'move_delta', 0.3)
        
        if self.n_actions == 4:
            n_bins_x = max(10, int(np.ceil(x_range / move_delta)) + 1)
            n_bins_y = max(10, int(np.ceil(y_range / move_delta)) + 1)
        else:
            diagonal_delta = move_delta / np.sqrt(2)
            n_bins_x = max(15, int(np.ceil(x_range / diagonal_delta)) + 1)
            n_bins_y = max(15, int(np.ceil(y_range / diagonal_delta)) + 1)
        
        n_bins_x = min(n_bins_x, 50)
        n_bins_y = min(n_bins_y, 50)
        
        x_bins = np.linspace(self.obs_low[0], self.obs_high[0], n_bins_x)
        y_bins = np.linspace(self.obs_low[1], self.obs_high[1], n_bins_y)
        
        h, xedges, yedges = np.histogram2d(obs_2d[:, 0], obs_2d[:, 1], bins=[x_bins, y_bins])
        
        im = ax.imshow(h.T, origin='lower', cmap='Blues', 
                      extent=[self.obs_low[0], self.obs_high[0], 
                             self.obs_low[1], self.obs_high[1]],
                      aspect='auto', interpolation='nearest')
        
        ax.set_xlim(self.obs_low[0], self.obs_high[0])
        ax.set_ylim(self.obs_low[1], self.obs_high[1])
        ax.set_title(f'Dataset Distribution\n({observations.shape[0]} samples)\n'
                    f'Bins: {n_bins_x}x{n_bins_y}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(im, ax=ax, label='Count')
        
        if hasattr(self.env_unwrapped, 'walkable_areas'):
            for area in self.env_unwrapped.walkable_areas:
                rect = Rectangle((area[0], area[1]), area[2], area[3],
                               fill=False, edgecolor='red', linewidth=1.5, 
                               linestyle='--', alpha=0.7)
                ax.add_patch(rect)
    
    def _plot_action_probabilities_grid(self, ax, n_samples: int = 16):
        """Plot action probabilities for sampled states."""
        # Use pre-cached grid states instead of sampling new ones
        sampled_states = self.grid_states_2d
        sampled_observations = self.grid_observations
        
        ax.set_xlim(self.obs_low[0], self.obs_high[0])
        ax.set_ylim(self.obs_low[1], self.obs_high[1])
        ax.set_aspect('equal')
        ax.set_title(f'Action Probabilities\n({len(sampled_states)} states)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
        
        if hasattr(self.env_unwrapped, 'walkable_areas'):
            for area in self.env_unwrapped.walkable_areas:
                rect = Rectangle((area[0], area[1]), area[2], area[3],
                               fill=False, edgecolor='gray', linewidth=1, linestyle='--', alpha=0.5)
                ax.add_patch(rect)
        
        square_size = 0.3
        
        # Debug: save first rendered image
        debug_saved = False
        
        for idx, (state, observation) in enumerate(zip(sampled_states, sampled_observations)):
            x, y = state[:2]  # Use only first 2 dimensions
            
            # Background square
            rect = Rectangle((x - square_size/2, y - square_size/2), square_size, square_size,
                           facecolor='lightgray', edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            
            # Use pre-rendered observation instead of rendering again
            # observation is already available from the loop
            
            # Compute action probabilities using the pre-rendered observation
            action_probs = self.agent.compute_action_probs(observation)
            
            # Draw bars
            bar_width = square_size / (self.n_actions + 1)
            bar_spacing = square_size / self.n_actions
            start_x = x - square_size/2 + bar_width/2
            max_bar_height = square_size * 0.8
            
            for a_idx in range(self.n_actions):
                bar_x = start_x + a_idx * bar_spacing
                bar_height = action_probs[a_idx] * max_bar_height
                
                bar_rect = Rectangle((bar_x - bar_width/2, y - square_size/2 + 0.1),
                                    bar_width, bar_height,
                                    facecolor=self.action_colors[a_idx],
                                    edgecolor='black', linewidth=0.3)
                ax.add_patch(bar_rect)
        
        # Legend
        legend_elements = [Patch(facecolor=self.action_colors[i], label=self.action_names[i])
                          for i in range(self.n_actions)]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    
    def plot_embeddings_2d(self, save_path: str, use_tsne: bool = False):
        """Plot 2D projection of observation embeddings using PCA or t-SNE.
        
        For pixel observations: Uses actual images
        For state observations: Uses state vectors
        """
        trajectories, observations, _ = self._sample_trajectories()
        
        # Flatten list of observations from all trajectories
        all_obs = []
        for obs_list in observations:
            all_obs.extend(obs_list)
        
        print(f"Collected {len(all_obs)} observations from {len(trajectories)} trajectories")
        
        # Add grid observations to the mix
        all_obs.extend(self.grid_observations)
        n_traj_obs = len(all_obs) - len(self.grid_observations)
        
        print(f"Added {len(self.grid_observations)} grid observations (total: {len(all_obs)})")
        
        # Encode observations using agent's encoder
        with torch.no_grad():
            # Convert observations to tensor
            if self.is_pixel_obs:
                # Stack images: [N, C, H, W]
                obs_tensor = torch.stack([torch.from_numpy(obs) for obs in all_obs]).float()
                obs_tensor = obs_tensor.to(self.agent.device)
            else:
                # Stack states: [N, state_dim]
                obs_tensor = torch.from_numpy(np.array(all_obs)).float()
                obs_tensor = obs_tensor.to(self.agent.device)
            
            # Get embeddings
            embeddings = self.agent.encoder(obs_tensor).cpu().numpy()
            print(f"Embeddings shape: {embeddings.shape}")
        
        # Dimensionality reduction
        if use_tsne:
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_obs)-1))
            method_name = "t-SNE"
        else:
            reducer = PCA(n_components=2)
            method_name = "PCA"
        
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Split embeddings: trajectories vs grid
        traj_embeddings_2d = embeddings_2d[:n_traj_obs]
        grid_embeddings_2d = embeddings_2d[n_traj_obs:]
        
        # Split trajectory embeddings by trajectory
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        cmap = plt.cm.viridis
        idx = 0
        
        for traj_idx, obs_list in enumerate(observations):
            traj_len = len(obs_list)
            traj_emb = traj_embeddings_2d[idx:idx+traj_len]
            colors = cmap(np.linspace(0.3, 0.9, traj_len))
            
            # Plot middle points
            if traj_len > 2:
                ax.scatter(traj_emb[1:-1, 0], traj_emb[1:-1, 1],
                          c=colors[1:-1], alpha=0.6, s=30, marker='o')
            
            # Mark start (star) and end (triangle)
            ax.scatter(traj_emb[0, 0], traj_emb[0, 1],
                      c=[colors[0]], s=200, marker='*', edgecolors='black', linewidth=2,
                      label='Start' if traj_idx == 0 else '', zorder=10)
            ax.scatter(traj_emb[-1, 0], traj_emb[-1, 1],
                      c=[colors[-1]], s=120, marker='^', edgecolors='black', linewidth=2,
                      label='End' if traj_idx == 0 else '', zorder=10)
            
            idx += traj_len
        
        # Plot grid samples as red crosses
        ax.scatter(grid_embeddings_2d[:, 0], grid_embeddings_2d[:, 1],
                  c='red', s=100, marker='x', linewidths=2,
                  label=f'Grid samples ({len(self.grid_observations)})', zorder=11)
        
        ax.set_xlabel(f'{method_name} Component 1')
        ax.set_ylabel(f'{method_name} Component 2')
        obs_type_name = "Image" if self.is_pixel_obs else "State"
        ax.set_title(f'{obs_type_name} Observation Embeddings\n'
                    f'({self.n_trajectories} trajectories, {n_traj_obs} traj obs, {len(self.grid_observations)} grid samples)\n'
                    f'Light→Dark: Start→End')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Embedding visualization saved to: {save_path}")
        plt.close(fig)
    
    def plot_results(self, step: int, save_path: str = None):
        """Create comprehensive visualization."""
        trajectories, observations, _ = self._sample_trajectories()
        epsilon = utils.schedule(self.agent.sink_schedule, step)
        
        param_text = (
            f"Step: {step}\n"
            f"γ = {self.agent.discount}\n"
            f"η = {self.agent.lr_actor}\n"
            f"λ = {self.agent.lambda_reg}\n"
            f"PMD steps = {self.agent.pmd_steps}\n"
            f"Obs type: {self.obs_type}\n"
            f"ε = {epsilon:.6f}\n"
        )
        
        fig = plt.figure(figsize=(24, 12))
        fig.text(0.02, 0.98, param_text, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax_trajectories = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
        ax_dataset = plt.subplot2grid((2, 3), (0, 2))
        ax_action_grid = plt.subplot2grid((2, 3), (1, 2))
        
        self._plot_trajectories(ax_trajectories, trajectories)
        self._plot_dataset_occupancy(ax_dataset)
        
        # Action probability grid works for both pixel and state obs (uses 2D states)
        if len(trajectories) > 0 and trajectories[0].shape[-1] == 2:
            self._plot_action_probabilities_grid(ax_action_grid, n_samples=150)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nVisualization saved to: {save_path}")
            plt.close(fig)

        
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
                 embedding_type,
                 curl,
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
        self.embedding_type = embedding_type
        self.curl = curl


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
        
        if obs_type == 'pixels':
            assert embedding_type == 'cl', "For pixel observations, embedding_type must be 'cl'"
            self.aug = nn.Identity() #utils.RandomShiftsAug(pad=4)
            self.encoder = CNNEncoder(
                obs_shape,
                feature_dim
            ).to(self.device)
            
            self.obs_dim = self.feature_dim
        else:
            # Components
            self.aug = nn.Identity()
            self.encoder = Encoder(
                obs_shape, 
                hidden_dim, 
                self.feature_dim,
                embedding_type=self.embedding_type
            ).to(self.device)

            self.obs_dim = self.feature_dim
        
        self.transition_model = TransitionModel(
            self.obs_dim * self.n_actions,
            self.obs_dim,
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
            subsampling_strategy=subsampling_strategy,
            obs_shape=obs_shape,
            data_type=torch.float32,
            device='cpu' # Force dataset on CPU to save GPU memory
        )
        
       
        # Optimizers
        if self.embedding_type == 'cl':
            self.encoder_optimizer = torch.optim.Adam(
                self.encoder.parameters(), 
                lr=lr_encoder
            )
            self.transition_optimizer = torch.optim.Adam(
                self.transition_model.parameters(),
                lr=lr_T
            )
        else:
            self.encoder_optimizer = None
            self.transition_optimizer = None
        
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
        # Store both wrapped and unwrapped versions
        self.wrapped_env = env  # The wrapped version with ExtendedTimeStep
        self.env = env.unwrapped  # The unwrapped base environment
        
        # Store rendering resolution from wrapped env
        self.render_resolution = None
        if hasattr(env, 'render_resolution'):
            self.render_resolution = env.render_resolution
        else:
            # Try to get it from the resize wrapper
            current = env
            while current is not None:
                if hasattr(current, 'resolution'):
                    self.render_resolution = current.resolution
                    break
                current = getattr(current, 'env', None)
        
        if self.render_resolution is None and self.obs_type == 'pixels':
            raise ValueError("Could not determine render resolution from environment")
        
        print(f"Agent using render resolution: {self.render_resolution}")
        
        timestep = env.reset()
        # Handle both ExtendedTimeStep and tuple
        if hasattr(timestep, 'observation'):
            self.first_state = torch.tensor(timestep.observation)
        else:
            obs = timestep[0] if isinstance(timestep, tuple) else timestep
            self.first_state = torch.tensor(obs)
        
        self.second_states = []
        for action in range(self.n_actions):
            next_timestep = env.step(action)
            # Handle both ExtendedTimeStep and tuple
            if hasattr(next_timestep, 'observation'):
                self.second_states.append(torch.tensor(next_timestep.observation))
            else:
                obs = next_timestep[0] if isinstance(next_timestep, tuple) else next_timestep
                self.second_states.append(torch.tensor(obs))
            env.reset()
        
        # Initialize visualizer now that we have the environment
        self.visualizer = EmbeddingDistributionVisualizer(self, n_trajectories=5, traj_length=300)
        
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
        """
        Pre-populate the dataset with all reachable state-action pairs in ideal mode.
        
        For pixel observations: renders images from sampled positions
        For state observations: uses position directly
        """
        print("=== Populating ideal dataset with all reachable state-action pairs ===")
        
        # Get discretized positions from walkable areas
        sampled_positions = self._sample_grid_positions()
        
        print(f"Sampled {len(sampled_positions)} grid positions from walkable areas")
        
        # Get the dataset device to ensure consistency
        dataset_device = self.dataset.device
        
        # For pixel observations, we need to render images
        if self.obs_type == 'pixels':
            print("Rendering images for ideal dataset (this may take a while)...")
        
        # For each position, try all actions
        for pos_idx, position in enumerate(sampled_positions):
            if pos_idx % 100 == 0:
                print(f"Processing position {pos_idx}/{len(sampled_positions)}...", end='\r')
            
            for action in range(self.n_actions):
                # Use step_from_position to get next state without changing env state
                next_position, reward, terminated, truncated, info = self.env.step_from_position(
                    position, action
                )
                
                if self.obs_type == 'pixels':
                    # Render observations for current and next position
                    obs_rendered = self.render_observation_from_state(position)
                    next_obs_rendered = self.render_observation_from_state(next_position)
                    
                    # Convert to tensors
                    obs_tensor = torch.from_numpy(obs_rendered).to(dataset_device)
                    next_obs_tensor = torch.from_numpy(next_obs_rendered).to(dataset_device)
                else:
                    # For state observations, use positions directly
                    obs_tensor = torch.tensor(position, device=dataset_device, dtype=torch.float32)
                    next_obs_tensor = torch.tensor(next_position, device=dataset_device, dtype=torch.float32)
                
                # Add to current period data
                self.dataset._current_period_data['observation'] = torch.cat([
                    self.dataset._current_period_data['observation'],
                    obs_tensor.unsqueeze(0)
                ], dim=0)
                
                self.dataset._current_period_data['action'] = torch.cat([
                    self.dataset._current_period_data['action'],
                    torch.tensor([action], device=dataset_device, dtype=torch.long)
                ], dim=0)
                
                self.dataset._current_period_data['next_observation'] = torch.cat([
                    self.dataset._current_period_data['next_observation'],
                    next_obs_tensor.unsqueeze(0)
                ], dim=0)
                
                # Set alpha: 1.0 for the first entry (start state), 0.0 for others
                if self.dataset._current_period_data['alpha'].shape[0] == 0:
                    alpha_val = 1.0
                else:
                    alpha_val = 0.0
                
                self.dataset._current_period_data['alpha'] = torch.cat([
                    self.dataset._current_period_data['alpha'],
                    torch.tensor([alpha_val], device=dataset_device, dtype=torch.float32)
                ], dim=0)
        
        print()  # New line after progress
        self._ideal_dataset_filled = True
        dataset_size = len(self.dataset._current_period_data['observation'])
        expected_size = len(sampled_positions) * self.n_actions
        
        print(f"Ideal dataset populated with {dataset_size} state-action pairs")
        print(f"Expected size: {expected_size} (positions: {len(sampled_positions)}, actions: {self.n_actions})")
        
        if self.obs_type == 'pixels':
            obs_shape = self.dataset._current_period_data['observation'][0].shape
            print(f"Observation shape in dataset: {obs_shape}")
        
        if dataset_size != expected_size:
            utils.ColorPrint.yellow(
                f"Warning: Dataset size {dataset_size} differs from expected {expected_size}. "
                "Some transitions may have been filtered or duplicated."
            )
    
    def _sample_grid_positions(self) -> np.ndarray:
        """
        Sample positions on a regular grid within walkable areas.
        Grid spacing is based on move_delta to capture all reachable discrete positions.
        
        Returns:
            np.ndarray: Array of positions [n_positions, 2]
        """
        if not hasattr(self.env, 'walkable_areas'):
            raise ValueError(
                "Environment must have 'walkable_areas' attribute for ideal mode. "
                "Make sure you're using a continuous room environment."
            )
        
        positions = []
        grid_spacing = self.env.move_delta
        
        # Add margin for agent radius to ensure positions are valid
        margin = self.env.agent_radius + 0.05
        
        for area in self.env.walkable_areas:
            ax, ay, aw, ah = area
            
            # Create grid within this area
            x = ax + margin
            while x <= ax + aw - margin:
                y = ay + margin
                while y <= ay + ah - margin:
                    # Verify position is valid with agent radius
                    if self.env._is_position_valid_with_radius(x, y):
                        positions.append([x, y])
                    y += grid_spacing
                x += grid_spacing
            
        if len(positions) == 0:
            raise ValueError(
                "No valid grid positions found. Check walkable_areas and agent_radius."
            )
        
        print(f"Generated grid with {len(positions)} positions (spacing={grid_spacing:.3f})")
        return np.array(positions, dtype=np.float32)

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
        """Compute π(·|s) for given observation.
        
        Args:
            obs: For state observations: [x, y] position
                 For pixel observations: [C, H, W] image
        """
        with torch.no_grad():
            # Handle different observation types
            if self.obs_type == 'pixels':
                # obs should already be an image [C, H, W]
                if obs.ndim == 2:
                    # If it's a 2D position [x, y], we need to render it
                    # This happens in _plot_action_probabilities_grid
                    raise ValueError(
                        "For pixel observations, compute_action_probs expects an image [C, H, W], "
                        f"but got shape {obs.shape}. Use render_observation_from_state() first."
                    )
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)  # [1, C, H, W]
            else:
                # State observations: [x, y] -> [1, 2]
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
            
            enc_obs = self.aug_and_encode(obs_tensor, project=True).cpu()
    
            if self.gradient_coeff is None:
                return np.ones(self.n_actions) / self.n_actions
            
            # Add a zero to enc_obs to account for the extra row in H
            enc_obs_augmented = torch.cat([enc_obs, torch.zeros((1, 1))], dim=1)  # [1, feature_dim + 1]
            H = enc_obs_augmented @ self._phi_all_obs.T  # [1, num_unique]

            probs = torch.softmax(-self.lr_actor * (H@(self.gradient_coeff[:-1]*self.E)+ torch.ones(1, self.E.shape[1])*self.gradient_coeff[-1]), dim=1)  # [1, n_actions]
            
            if torch.sum(probs) == 0.0 or torch.isnan(torch.sum(probs)):
                raise ValueError("action_probs sum to zero or NaN")
            return probs.numpy().flatten()
    
    def render_observation_from_state(self, state_2d: np.ndarray) -> np.ndarray:
        """
        Render observation from a 2D state position.
        
        For pixel observations: renders image from position and stacks frames
        For state observations: returns state as-is
        
        Args:
            state_2d: [x, y] position
            
        Returns:
            Observation in the format expected by the agent
        """
        if self.obs_type == 'pixels':
            from PIL import Image
            
            # Render image from position [H, W, C]
            image = self.env.render_from_position(state_2d)
            
            # Auto-resize if needed
            if image.shape[:2] != (self.render_resolution, self.render_resolution):
                print(f"Warning: Rendered image shape {image.shape[:2]} doesn't match expected "
                      f"({self.render_resolution}, {self.render_resolution}). Resizing...")
                
                # Convert to PIL Image, resize, convert back
                pil_img = Image.fromarray(image.astype(np.uint8))
                pil_img_resized = pil_img.resize(
                    (self.render_resolution, self.render_resolution), 
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
            frame_stack = self.obs_shape[0] // 3  # Assuming RGB (3 channels)
            stacked_image = np.tile(image_chw, (frame_stack, 1, 1))
            
            # Final shape verification
            expected_final_shape = (self.obs_shape[0], self.render_resolution, self.render_resolution)
            if stacked_image.shape != expected_final_shape:
                print(f"Warning: Final stacked image shape {stacked_image.shape} doesn't match "
                      f"expected {expected_final_shape}")
            
            return stacked_image
        else:
            # For state observations, return as-is
            return state_2d
    
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
        if self.embedding_type != 'cl':
            return metrics  # No transition model to update
        
        if self.ideal:
            # Use the full ideal dataset
            obs = self.dataset.data['observation'].to(self.device)
            action = self.dataset.data['action'].to(self.device)
            next_obs = self.dataset.data['next_observation'].to(self.device)
            assert obs.shape[0] == action.shape[0] == next_obs.shape[0], f"Ideal dataset tensors have mismatched sizes, received obs: {obs.shape}, action: {action.shape}, next_obs: {next_obs.shape}"
        
        obs_en = self.aug_and_encode(obs, project=True)
        with torch.no_grad():
            next_obs_en = self.aug_and_encode(next_obs, project=True)

        encoded_state_action = self._encode_state_action(obs_en, action)
        
        # Predict next state
        predicted_next = self.transition_model(encoded_state_action)
        
        # Compute loss
        # 1. Contrastive loss:
        logits = predicted_next/torch.norm(predicted_next, p=2, dim=1, keepdim=True) @ (next_obs_en/torch.norm(next_obs_en, p=2, dim=1, keepdim=True)).T   # [B, B]
        logits = logits - torch.max(logits, 1)[0][:, None]  # For numerical stability
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        contrastive_loss = self.cross_entropy_loss(logits, labels)

        # 2. Loss embeddings must sum to 1
        embedding_sum_loss = torch.abs(torch.sum(self.aug_and_encode(obs, project=False), dim=-1) - 1).sum()
        beta = 1  
        # 3. \phi(s) and \phi(s') must be close in L2 norm
        l2_loss = torch.norm(obs_en - next_obs_en, p=2, dim=1).mean()

        # 4. augmentation loss (optional)
        
        with torch.no_grad():
            obs_pos = self.aug_and_encode(obs, project=True)
        ### Compute CURL loss
        if self.curl:
            logits = obs_en/torch.norm(obs_en, p=2, dim=1, keepdim=True) @ (obs_pos/torch.norm(obs_pos, p=2, dim=1, keepdim=True)).T  # [B, B]
            logits = logits - torch.max(logits, 1)[0][:, None]
            labels = torch.arange(logits.shape[0]).long().to(self.device)
            curl_loss = self.cross_entropy_loss(logits, labels)
        else:
            curl_loss = torch.tensor(0.)

        loss = contrastive_loss + 0.001*embedding_sum_loss + l2_loss # beta*embedding_sum_loss + l2_loss  + curl_loss     
        
        # Optimize
        self.encoder_optimizer.zero_grad()
        self.transition_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()
        self.transition_optimizer.step()

        # Print losses
        
        print(f"Transition Model Losses: Contrastive={contrastive_loss.item():.4f}, EmbeddingSum={embedding_sum_loss.item():.4f}, L2={l2_loss.item():.4f}, Total={loss.item():.4f}")
        if self.use_tb or self.use_wandb:
            metrics['contrastive_loss'] = contrastive_loss.item()
            metrics['embedding_sum_loss'] = embedding_sum_loss.item()
            metrics['l2_loss'] = l2_loss.item()
            metrics['curl_loss'] = curl_loss.item()
            metrics['total_transition_loss'] = loss.item()

        return metrics

    def update_actor(self, step):
        """Update policy using Projected Mirror Descent."""
        metrics = dict()
        # Compute features for internal dataset
        if not hasattr(self, '_features_cached'):
            self._cache_features()
            # if self.gradient_coeff is None or (self.gradient_coeff is not None and self.gradient_coeff.shape[0] != self.dataset.size):
            self.gradient_coeff = torch.zeros((self._phi_all_obs.shape[0]+1, 1))  # [n+1, 1]
            self.H = self._phi_all_obs @ self._phi_all_next.T # [n, n]
            self.unique_states = torch.eye(self.n_states)
            self.K = self._psi_all @ self._psi_all.T  # [n, n]

        epsilon = utils.schedule(self.sink_schedule, step)
        self.pi = torch.softmax(-self.lr_actor * (self.H.T@(self.gradient_coeff[:-1]*self.E)+ torch.ones(self._phi_all_next.shape[0], self.E.shape[1])*self.gradient_coeff[-1]), dim=1)  # [z_x+1, n_actions]
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

    
    def _cache_features(self):
        """Pre-compute and cache dataset features."""
        tensors = self.dataset.get_data()
        print("Caching features from dataset of size:", len(tensors['observation']))
        with torch.no_grad():
            obs = tensors['observation'][:len(tensors['next_observation'])]
            actions = tensors['action']
            next_obs = tensors['next_observation']
            
            print(f"Encoding {obs.shape} observations and next observations on device {self.device}...")
            self._phi_all_obs = self.aug_and_encode(obs.to(self.device), project=True).cpu().float()
            self._phi_all_next = self.aug_and_encode(next_obs.to(self.device), project=True).cpu().float()
         
            # Transfer first_state to GPU for comparison
            indices = torch.where(torch.all(next_obs.cpu() == self.first_state, dim=1))[0]
            if indices.shape[0] == 0:
                for second_state in self.second_states:
                    indices = torch.where(torch.all(next_obs.cpu() == second_state, dim=1))[0]
                    print("DEBUG: looking for second_state, found indices:", indices)
                    if indices.shape[0] > 0:
                        break
            
            self._psi_all = self._encode_state_action(self._phi_all_obs, actions)
           
            self._alpha = torch.zeros((self._phi_all_next.shape[0], 1))
            print("DEBUG: setting alpha for index", indices[0].item(), len(self._alpha))
            self._alpha[indices[0]] = 1.0
            self.E = F.one_hot(
                actions.long(), 
                self.n_actions
            ).to(torch.float32).reshape(-1, self.n_actions)

            # ** AUGMENTATION STEP **
            # ψ and Φ are augmented with an additional zero dimension
            zeros_col = torch.zeros(*self._psi_all.shape[:-1], 1)
            self._psi_all = torch.cat([self._psi_all, zeros_col], dim=-1)

            zero_col = torch.zeros(*self._phi_all_next.shape[:-1], 1)
            self._phi_all_next = torch.cat([self._phi_all_next, zero_col], dim=-1)

            zero_col = torch.zeros(*self._phi_all_obs.shape[:-1], 1)
            self._phi_all_obs = torch.cat([self._phi_all_obs, zero_col], dim=-1)

        if not self.dataset.is_complete:
            return
        print("=================================Features cached=================================")

        self._features_cached = True
        

    def aug_and_encode(self, obs, project=False):
        obs = self.aug(obs)
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
            if not self.ideal:
                metrics['batch_reward'] = reward.mean().item()
            else:
                metrics['batch_reward'] = 0.0  # placeholder
        

        metrics.update(self.update_transition_matrix(obs, action, next_obs))

        # If T is not sufficiently initialized, skip actor update
        if self._is_T_sufficiently_initialized(step) is False:   
            metrics['actor_loss'] = 100.0  # dummy value
            return metrics
        

        # In ideal mode, we can update actor immediately
        if  step % self.update_actor_every_steps == 0 or step == self.num_expl_steps + self.T_init_steps: # or self.ideal:  
            # update actor
        
            # Pass dummy obs, not used in update_actor
            metrics.update(self.update_actor(step))
        
            if self.visualizer is not None:
                save_path = os.path.join(os.getcwd(), f"plot_step_{step}.png")
                self.visualizer.plot_results(step, save_path=save_path)
                save_path = os.path.join(os.getcwd(), f"features_step_{step}.png")
                self.visualizer.plot_embeddings_2d(save_path=save_path, use_tsne=True)
                print(f"Visualization saved to: {save_path}")
        return metrics

