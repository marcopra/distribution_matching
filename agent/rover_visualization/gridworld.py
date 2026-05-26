from __future__ import annotations

import os
from textwrap import shorten

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch, Rectangle
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.nn.functional as F


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
                
                if hasattr(self.agent, "_kernel"):
                    kernel = self.agent._kernel(enc_all_states_aug, phi_next)
                else:
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
            if getattr(self.agent, 'subsamples', None) is not None:
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
            if getattr(self.agent, 'subsamples', None) is not None and hasattr(self.agent, '_phi_sub_next'):
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
            getattr(self.agent, 'subsamples', None) is None
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

            
