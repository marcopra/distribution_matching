"""
Script to visualize sampled trajectories with discounted occupancy heatmap.

This script reads trajectories saved by the agent and creates visualizations
showing the discounted state occupancy measure with both heatmap and histogram.
python scripts/visualize_trajectories.py
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import glob


def load_trajectories(folder_path):
    """
    Load all trajectories from a folder.
    
    Args:
        folder_path: Path to folder containing trajectory files
        
    Returns:
        trajectories: List of trajectory dictionaries
        metadata: Metadata dictionary
    """
    # Load metadata
    metadata_file = os.path.join(folder_path, "metadata.npz")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    metadata = dict(np.load(metadata_file))
    
    # Load all trajectory files
    traj_files = sorted(glob.glob(os.path.join(folder_path, "trajectory_*.npz")))
    trajectories = []
    
    for traj_file in traj_files:
        data = np.load(traj_file)
        trajectory = {
            'states': data['states'],
            'positions': data['positions'],
            'actions': data['actions'],
            'rewards': data['rewards']
        }
        trajectories.append(trajectory)
    
    print(f"Loaded {len(trajectories)} trajectories from {folder_path}")
    return trajectories, metadata


def compute_discounted_occupancy(trajectories, gamma, n_states=None):
    """
    Compute discounted state occupancy from trajectories.
    
    For each state visited at time t in a trajectory, add γ^t / (1-γ) to its count.
    
    Args:
        trajectories: List of trajectory dictionaries
        gamma: Discount factor
        n_states: Number of states (if None, inferred from trajectories)
        
    Returns:
        occupancy: Array of discounted occupancy counts per state
    """
    if n_states is None:
        # Infer from max state index
        all_states = []
        for traj in trajectories:
            all_states.extend(traj['states'])
        n_states = max(all_states) + 1
    
    occupancy = np.zeros(n_states)
    normalization =  (1.0 - gamma)
    
    for traj in trajectories:
        states = traj['states']
        for t, state_idx in enumerate(states):
            discount_factor = (gamma ** t) * normalization
            occupancy[state_idx] += discount_factor
    
    return occupancy


def positions_to_grid_occupancy(trajectories, gamma, min_x, max_x, min_y, max_y):
    """
    Convert trajectories to grid-based discounted occupancy.
    
    Args:
        trajectories: List of trajectory dictionaries
        gamma: Discount factor
        min_x, max_x, min_y, max_y: Grid bounds
        
    Returns:
        grid: 2D array of discounted occupancy
    """
    grid_width = max_x - min_x + 1
    grid_height = max_y - min_y + 1
    grid = np.zeros((grid_height, grid_width))
    normalization = 1.0 / (1.0 - gamma)
    
    for traj in trajectories:
        positions = traj['positions']
        for t, pos in enumerate(positions):
            x, y = pos
            grid_x = x - min_x
            grid_y = y - min_y
            discount_factor = 1# (gamma ** t) * normalization
            grid[grid_y, grid_x] += discount_factor
    
    return grid

def plot_corridor_occupancy(grid, min_x, max_x, min_y, max_y, save_path, metadata):
    """
    Create visualization with heatmap and histogram of corridor occupancy.
    
    Args:
        grid: 2D array of discounted occupancy
        min_x, max_x, min_y, max_y: Corridor bounds
        save_path: Path to save the figure
        metadata: Metadata dictionary
    """
    grid_height, grid_width = grid.shape
    
    # Adjust figure size: wider for long corridors, shorter height
    fig_width = max(12, grid_width * 0.4)  # Scale with corridor length
    fig_height = 6  # Reduced height
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create grid spec for layout: histogram on top, heatmap below
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1], hspace=0.0)
    
    ax_hist = fig.add_subplot(gs[0])
    ax_heatmap = fig.add_subplot(gs[1])
    
    # For corridor: sum over height dimension to get 1D occupancy
    occupancy_1d = grid.sum(axis=0)
    
    # Find min and max positions where data exists (non-zero occupancy)
    nonzero_positions = np.where(occupancy_1d > 0)[0]
    if len(nonzero_positions) > 0:
        data_min_pos = nonzero_positions[0] + 0.5  # Grid position (centered)
        data_max_pos = nonzero_positions[-1] + 0.5  # Grid position (centered)
    else:
        data_min_pos = None
        data_max_pos = None
    
    # Normalize the 1D occupancy
    if occupancy_1d.sum() > 0:
        occupancy_1d_normalized = occupancy_1d / occupancy_1d.sum()
    else:
        occupancy_1d_normalized = occupancy_1d
    
    # Use centered positions for bars to match heatmap pixels
    x_positions = np.arange(grid_width) + 0.5
    
    # Plot histogram with normalized values - bars centered on pixels
    ax_hist.bar(x_positions, occupancy_1d_normalized, color='steelblue', 
                edgecolor='black', linewidth=0.5, width=1.0, align='center')
    ax_hist.set_ylabel('Normalized\nOccupancy', fontsize=10)
    
    # Add vertical lines for min and max data positions
    if data_min_pos is not None and data_max_pos is not None:
        ax_hist.axvline(x=data_min_pos, color='darkred', linestyle='-', linewidth=2, 
                       label=f'Min pos (x={int(data_min_pos - 0.5 + min_x)})')
        ax_hist.axvline(x=data_max_pos, color='darkred', linestyle='-', linewidth=2,
                       label=f'Max pos (x={int(data_max_pos - 0.5 + min_x)})')
        ax_hist.legend(loc='upper right', fontsize=8)
    
    # Extract metadata values safely
    training_step = int(metadata.get('training_step', 0))
    gamma = float(metadata.get('gamma', 0.9))
    n_trajectories = int(metadata.get('n_trajectories', 0))
    
    ax_hist.set_title(
        f'Corridor State Occupancy (Step {training_step}, '
        f'γ={gamma:.3f}, {n_trajectories} trajectories)',
        fontsize=12, fontweight='bold', pad=10
    )
    ax_hist.set_xlim(0, grid_width)
    ax_hist.grid(True, alpha=0.3, axis='y')
    ax_hist.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    
    # Normalize the grid for heatmap
    if grid.sum() > 0:
        grid_normalized = grid / grid.sum()
    else:
        grid_normalized = grid
    
    # Plot heatmap with Blues colormap (NO COLORBAR)
    im = ax_heatmap.imshow(grid_normalized, cmap='Blues', interpolation='nearest', 
                           aspect='auto', vmin=0, vmax=grid_normalized.max(),
                           extent=[0, grid_width, grid_height, 0])
    ax_heatmap.set_xlabel('x position', fontsize=10)
    ax_heatmap.set_ylabel('y', fontsize=10)
    
    # Set the same xlim as histogram for perfect alignment
    ax_heatmap.set_xlim(0, grid_width)
    
    # Set x-ticks to match actual positions
    # Place ticks at pixel centers (0.5, 1.5, 2.5, ...)
    if grid_width > 20:
        tick_step = max(1, grid_width // 20)
        tick_positions = np.arange(0, grid_width, tick_step) + 0.5
        tick_labels = np.arange(min_x, max_x + 1, tick_step)
    else:
        tick_positions = np.arange(grid_width) + 0.5
        tick_labels = np.arange(min_x, max_x + 1)
    
    ax_heatmap.set_xticks(tick_positions)
    ax_heatmap.set_xticklabels(tick_labels)
    
    # Set y-ticks
    ax_heatmap.set_yticks(np.arange(grid_height) + 0.5)
    ax_heatmap.set_yticklabels(np.arange(min_y, max_y + 1))
    
    # Add subtle grid at pixel boundaries
    ax_heatmap.set_xticks(np.arange(0, grid_width + 1, 1), minor=True)
    ax_heatmap.set_yticks(np.arange(0, grid_height + 1, 1), minor=True)
    ax_heatmap.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
    
    # # Add statistics text (includes max value info instead of colorbar)
    # stats_text = (
    #     f'Max: {grid_normalized.max():.4f}\n'
    #     f'Mean: {grid_normalized.mean():.4f}\n'
    #     f'Total: {grid.sum():.2f}'
    # )
    # ax_heatmap.text(
    #     0.02, 0.98, stats_text,
    #     transform=ax_heatmap.transAxes,
    #     fontsize=9,
    #     verticalalignment='top',
    #     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    # )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {save_path}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Visualize sampled trajectories')
    parser.add_argument('folder', type=str, help='Path to folder containing trajectories')
    parser.add_argument('--output', type=str, default=None, 
                       help='Output path for visualization (default: folder/occupancy_plot.png)')
    args = parser.parse_args()
    
    # Load trajectories
    trajectories, metadata = load_trajectories(args.folder)
    
    # Get corridor dimensions from metadata
    min_x = int(metadata.get('min_x', 0))
    max_x = int(metadata.get('max_x', 0))
    min_y = int(metadata.get('min_y', 0))
    max_y = int(metadata.get('max_y', 0))
    
    print(f"Corridor dimensions: x=[{min_x}, {max_x}], y=[{min_y}, {max_y}]")
    
    # Compute discounted occupancy grid
    gamma = float(metadata['gamma'])
    grid = positions_to_grid_occupancy(trajectories, gamma, min_x, max_x, min_y, max_y)
    
    print(f"Total discounted occupancy: {grid.sum():.2f}")
    max_pos = np.unravel_index(grid.argmax(), grid.shape)
    print(f"Max occupancy: {grid.max():.4f} at grid position {max_pos}")
    print(f"Corresponding to actual position: x={max_pos[1] + min_x}, y={max_pos[0] + min_y}")
    
    # Plot
    if args.output is None:
        output_path = os.path.join(args.folder, "occupancy_plot.png")
    else:
        output_path = args.output
    
    plot_corridor_occupancy(grid, min_x, max_x, min_y, max_y, output_path, metadata)


if __name__ == "__main__":
    main()