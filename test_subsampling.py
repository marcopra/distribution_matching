"""
Test Subsampling Method - Evaluate the uniformity of subsampling from internal dataset.

This script loads a pretrained agent, collects N transitions, and tests the subsampling
method to verify that data is sampled homogeneously across the state space.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import gymnasium as gym
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import os
import torch
import wandb
from PIL import Image
import env.rooms
from collections import Counter
from agent.dist_matching_embedding import InternalDataset
from dm_env import StepType


class SubsamplingTester:
    """Test and visualize subsampling behavior."""
    
    def __init__(self, env: gym.Env, agent, gamma: float, n_subsamples: int):
        self.env = env
        self.agent = agent
        self.n_states = env.unwrapped.n_states
        self.n_actions = env.action_space.n
        
        # Create our own instance of InternalDataset for testing
        self.test_dataset = InternalDataset(
            dataset_type="all",  # Collect all transitions
            n_states=self.n_states,
            n_actions=self.n_actions,
            gamma=gamma,
            n_subsamples=n_subsamples
        )
        
        # Get grid dimensions for visualization
        valid_cells = [cell for cell in env.unwrapped.cells if cell != env.unwrapped.DEAD_STATE]
        min_x = min(cell[0] for cell in valid_cells)
        min_y = min(cell[1] for cell in valid_cells)
        max_x = max(cell[0] for cell in valid_cells)
        max_y = max(cell[1] for cell in valid_cells)
        
        self.min_x = min_x
        self.min_y = min_y
        self.grid_width = max_x - min_x + 1
        self.grid_height = max_y - min_y + 1
    
    def collect_data(self, n_steps: int, seed: int = None):
        """Collect N transitions using the agent's policy."""
        if seed is not None:
            np.random.seed(seed)
            self.env.reset(seed=seed)
        
        print(f"Collecting {n_steps} transitions...")
        
        # Reset our test dataset
        self.test_dataset.reset()
        
        obs, info = self.env.reset()
        obs_onehot = np.zeros(self.n_states, dtype=np.float32)
        obs_onehot[obs] = 1.0
        
        steps = 0
        episodes = 0
        
        # Create a dummy time_step for FIRST
        class TimeStep:
            def __init__(self, step_type, observation, action=None):
                self.step_type = step_type
                self.observation = observation
                self.action = action
        
        # Add FIRST transition
        time_step = TimeStep(StepType.FIRST, obs_onehot)
        self.test_dataset.add_transition(time_step)
        
        while steps < n_steps:
            # Sample action from agent
            action = self.agent.act(obs_onehot, {}, step=1000000, eval_mode=True)
            
            # Execute action
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            next_obs_onehot = np.zeros(self.n_states, dtype=np.float32)
            next_obs_onehot[next_obs] = 1.0
            
            # Add transition to our test dataset
            if terminated or truncated:
                time_step = TimeStep(StepType.LAST, next_obs_onehot, action)
                self.test_dataset.add_transition(time_step)
                
                # Reset environment
                obs, info = self.env.reset()
                obs_onehot = np.zeros(self.n_states, dtype=np.float32)
                obs_onehot[obs] = 1.0
                
                time_step = TimeStep(StepType.FIRST, obs_onehot)
                self.test_dataset.add_transition(time_step)
                
                episodes += 1
            else:
                time_step = TimeStep(StepType.MID, next_obs_onehot, action)
                self.test_dataset.add_transition(time_step)
                
                obs = next_obs
                obs_onehot = next_obs_onehot
            
            steps += 1
            
            if steps % 1000 == 0:
                print(f"  Collected {steps}/{n_steps} transitions ({episodes} episodes)")
        
        dataset_size = len(self.test_dataset.data['observation'])
        print(f"Data collection complete: {dataset_size} transitions in test dataset")
        print(f"Number of trajectory indices: {len(self.test_dataset._trajectory_idx)}")
            
        return dataset_size
    
    def test_subsampling(self, n_trials: int = 100):
        """
        Test subsampling method multiple times and collect statistics.
        
        Args:
            n_trials: Number of subsampling trials
            
        Returns:
            Dictionary with subsampling statistics
        """
        n_subsamples = self.test_dataset.n_subsamples
        print(f"\nTesting subsampling with {n_subsamples} samples over {n_trials} trials...")
        
        # First, analyze the full dataset ONCE before any subsampling
        print("\nAnalyzing full dataset distribution (this will be constant across trials)...")
        full_dataset_obs = self.test_dataset.data['next_observation']
        full_state_indices = torch.argmax(full_dataset_obs, dim=1).numpy()
        
        full_state_counts = np.zeros(self.n_states)
        for s_idx in full_state_indices:
            full_state_counts[s_idx] += 1
        
        full_dataset_size = len(full_state_indices)
        full_state_probs = full_state_counts / full_dataset_size
        
        print(f"Full dataset size: {full_dataset_size} transitions")
        print(f"States with data: {np.sum(full_state_counts > 0)}/{self.n_states}")
        print(f"Mean visits per state: {np.mean(full_state_counts):.2f} ± {np.std(full_state_counts):.2f}")
        print(f"Min/Max visits: {np.min(full_state_counts):.0f} / {np.max(full_state_counts):.0f}")
        
        all_state_counts = []
        
        # Save the original data before subsampling (do this ONCE)
        original_data = {
            'observation': self.test_dataset.data['observation'].clone(),
            'action': self.test_dataset.data['action'].clone(),
            'next_observation': self.test_dataset.data['next_observation'].clone(),
            'alpha': self.test_dataset.data['alpha'].clone()
        }
        original_trajectory_idx = self.test_dataset._trajectory_idx.copy()
        original_traj_boundaries = self.test_dataset._traj_boundaries.copy()
        
        # Now run multiple trials of subsampling
        for trial in range(n_trials):
            # Restore original data before each subsample
            self.test_dataset.data = {
                'observation': original_data['observation'].clone(),
                'action': original_data['action'].clone(),
                'next_observation': original_data['next_observation'].clone(),
                'alpha': original_data['alpha'].clone()
            }
            self.test_dataset._trajectory_idx = original_trajectory_idx.copy()
            self.test_dataset._traj_boundaries = original_traj_boundaries.copy()
            
            # Get subsampled data
            subsampled = self.test_dataset.get_data()
            
            # Convert observations to state indices (skip dummy transition at index 0)
            state_indices = torch.argmax(subsampled['observation'][1:], dim=1).numpy()
            
            # Count state occurrences
            state_counts = np.zeros(self.n_states)
            for s_idx in state_indices:
                state_counts[s_idx] += 1
            
            all_state_counts.append(state_counts)
            
            if (trial + 1) % 10 == 0:
                print(f"  Completed {trial + 1}/{n_trials} trials")
        
        all_state_counts = np.array(all_state_counts)
        
        # Compute statistics for subsampled data
        mean_counts = np.mean(all_state_counts, axis=0)
        std_counts = np.std(all_state_counts, axis=0)
        
        # Normalize to get probabilities
        mean_probs = mean_counts / n_subsamples
        std_probs = std_counts / n_subsamples
        
        # Compute uniformity metrics
        expected_uniform_prob = 1.0 / self.n_states
        uniformity_score = 1.0 - np.mean(np.abs(mean_probs - expected_uniform_prob))
        
        stats = {
            'n_trials': n_trials,
            'n_subsamples': n_subsamples,
            'mean_counts': mean_counts,
            'std_counts': std_counts,
            'mean_probs': mean_probs,
            'std_probs': std_probs,
            'uniformity_score': uniformity_score,
            'all_counts': all_state_counts,
            # Add full dataset statistics (computed ONCE, not averaged)
            'full_dataset_size': full_dataset_size,
            'full_state_counts': full_state_counts,
            'full_state_probs': full_state_probs
        }
        
        print(f"\nUniformity score: {uniformity_score:.4f} (1.0 = perfect uniform)")
        
        return stats
    
    def plot_subsampling_heatmap(self, stats: dict, save_path: str):
        """Plot heatmap of subsampled state distribution."""
        mean_counts = stats['mean_counts']
        std_counts = stats['std_counts']
        full_counts = stats['full_state_counts']
        
        # Convert to grid
        mean_grid = self._counts_to_grid(mean_counts)
        std_grid = self._counts_to_grid(std_counts)
        full_grid = self._counts_to_grid(full_counts)
        
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
        
        # Plot 1: Full dataset distribution
        im1 = ax1.imshow(full_grid, cmap='Purples', interpolation='nearest')
        ax1.set_title(f'Full Dataset State Visit Counts\n(Total: {stats["full_dataset_size"]} transitions)')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_xticks(np.arange(-0.5, self.grid_width, 1), minor=True)
        ax1.set_yticks(np.arange(-0.5, self.grid_height, 1), minor=True)
        ax1.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Add count values to cells
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                count = full_grid[i, j]
                if count > 0:
                    ax1.text(j, i, f'{count:.0f}', ha='center', va='center', 
                            color='black' if count < np.max(full_grid)/2 else 'white',
                            fontsize=8)
        
        plt.colorbar(im1, ax=ax1, label='Visit Count')
        
        # Plot 2: Mean subsampled counts
        im2 = ax2.imshow(mean_grid, cmap='YlOrRd', interpolation='nearest')
        ax2.set_title(f'Mean Subsampled State Visit Counts\n(n={stats["n_subsamples"]} samples, {stats["n_trials"]} trials)')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_xticks(np.arange(-0.5, self.grid_width, 1), minor=True)
        ax2.set_yticks(np.arange(-0.5, self.grid_height, 1), minor=True)
        ax2.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Add count values to cells
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                count = mean_grid[i, j]
                if count > 0:
                    ax2.text(j, i, f'{count:.1f}', ha='center', va='center', 
                            color='black' if count < np.max(mean_grid)/2 else 'white',
                            fontsize=8)
        
        plt.colorbar(im2, ax=ax2, label='Mean Count')
        
        # Plot 3: Std counts
        im3 = ax3.imshow(std_grid, cmap='Blues', interpolation='nearest')
        ax3.set_title(f'Standard Deviation of Subsampled Counts')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_xticks(np.arange(-0.5, self.grid_width, 1), minor=True)
        ax3.set_yticks(np.arange(-0.5, self.grid_height, 1), minor=True)
        ax3.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Add std values to cells
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                std = std_grid[i, j]
                if std > 0:
                    ax3.text(j, i, f'{std:.1f}', ha='center', va='center',
                            color='black' if std < np.max(std_grid)/2 else 'white',
                            fontsize=8)
        
        plt.colorbar(im3, ax=ax3, label='Std Count')
        
        # Add uniformity score as text
        fig.text(0.5, 0.02, f'Uniformity Score: {stats["uniformity_score"]:.4f}', 
                ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
        plt.close(fig)
    
    def _counts_to_grid(self, counts: np.ndarray) -> np.ndarray:
        """Convert state counts vector to 2D grid."""
        grid = np.zeros((self.grid_height, self.grid_width))
        
        for s_idx in range(self.n_states):
            cell = self.env.unwrapped.idx_to_state[s_idx]
            grid_x = cell[0] - self.min_x
            grid_y = cell[1] - self.min_y
            grid[grid_y, grid_x] = counts[s_idx]
        
        return grid


@hydra.main(version_base=None, config_path="configs", config_name="config_test_subsampling")
def main(cfg: DictConfig):
    """Main execution function."""
    
    # Set random seed
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    # Get Hydra output directory
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"\nHydra output directory: {output_dir}\n")
    
    # Create environment
    env_kwargs = {
        'max_steps': cfg.env.max_steps,
        'render_mode': cfg.env.render_mode,
        'show_coordinates': cfg.env.show_coordinates,
        'goal_position': tuple(cfg.env.goal_position) if cfg.env.goal_position else None,
        'start_position': tuple(cfg.env.start_position) if cfg.env.start_position else None,
    }
    
    # Add environment-specific parameters
    if "SingleRoom" in cfg.env.name:
        env_kwargs['room_size'] = cfg.env.room_size
    elif "TwoRooms" in cfg.env.name:
        env_kwargs['room_size'] = cfg.env.room_size
        env_kwargs['corridor_length'] = cfg.env.corridor_length
        env_kwargs['corridor_y'] = cfg.env.corridor_y
    elif "FourRooms" in cfg.env.name:
        env_kwargs['room_size'] = cfg.env.room_size
        env_kwargs['corridor_length'] = cfg.env.corridor_length
        env_kwargs['corridor_positions'] = {
            'horizontal': cfg.env.corridor_positions.horizontal,
            'vertical': cfg.env.corridor_positions.vertical
        }
    
    env = gym.make(cfg.env.name, **env_kwargs)
    env.reset(seed=cfg.seed)
    print(f"Environment {cfg.env.name} created with {env.unwrapped.n_states} states\n")
    
    # Save environment visualization
    frame = env.render()
    img = Image.fromarray(frame)
    img.save(os.path.join(output_dir, f"{cfg.env.name}.png"))
    
    # Load pretrained agent
    print(f"Loading pretrained agent from: {cfg.policy_path}")
    with Path(cfg.policy_path).open('rb') as f:
        payload = torch.load(f, weights_only=False)
    
    if 'agent' in payload:
        agent = payload['agent']
        print("Agent loaded successfully")
    else:
        print(f"Available keys in payload: {list(payload.keys())}")
        raise ValueError("Could not find 'agent' in loaded checkpoint")
    
    # Set agent's environment
    agent.insert_env(env)
    agent.train(False)  # Set to eval mode
    
    # Initialize wandb if enabled
    if cfg.wandb.use_wandb:
        wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            project=cfg.wandb.wandb_project,
            name=cfg.wandb.wandb_run_name,
            tags=cfg.wandb.wandb_tag.split('_') if cfg.wandb.wandb_tag and cfg.wandb.wandb_tag != "none" else None,
            sync_tensorboard=True,
            mode='online'
        )
    
    # Create tester with its own dataset instance
    tester = SubsamplingTester(
        env=env, 
        agent=agent, 
        gamma=cfg.gamma,
        n_subsamples=cfg.n_subsamples
    )
    
    # Collect data
    print(f"\n{'='*60}")
    print("DATA COLLECTION PHASE")
    print(f"{'='*60}\n")
    
    dataset_size = tester.collect_data(cfg.n_collection_steps, seed=cfg.seed)
    
    # Test subsampling
    print(f"\n{'='*60}")
    print("SUBSAMPLING TEST PHASE")
    print(f"{'='*60}\n")
    
    stats = tester.test_subsampling(n_trials=cfg.n_trials)
    
    # Plot results
    save_path = os.path.join(output_dir, 'subsampling_heatmap.png')
    tester.plot_subsampling_heatmap(stats, save_path)
    
    # Save statistics
    np.save(os.path.join(output_dir, 'subsampling_stats.npy'), stats, allow_pickle=True)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUBSAMPLING TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Environment: {cfg.env.name} ({env.unwrapped.n_states} states)")
    print(f"\nFull Dataset Statistics:")
    print(f"  Total size: {dataset_size} transitions")
    print(f"  States covered: {np.sum(stats['full_state_counts'] > 0)}/{env.unwrapped.n_states}")
    print(f"  Mean visits per state: {np.mean(stats['full_state_counts']):.2f} ± {np.std(stats['full_state_counts']):.2f}")
    print(f"  Min/Max visits: {np.min(stats['full_state_counts']):.0f} / {np.max(stats['full_state_counts']):.0f}")
    print(f"\nSubsampling Statistics:")
    print(f"  Subsample size: {cfg.n_subsamples}")
    print(f"  Number of trials: {cfg.n_trials}")
    print(f"  Gamma (geometric sampling): {cfg.gamma}")
    print(f"  Uniformity score: {stats['uniformity_score']:.4f}")
    print(f"  Mean visit count: {np.mean(stats['mean_counts']):.2f} ± {np.std(stats['mean_counts']):.2f}")
    print(f"  Min/Max mean count: {np.min(stats['mean_counts']):.2f} / {np.max(stats['mean_counts']):.2f}")
    print(f"{'='*60}\n")
    
    # Log to wandb
    if cfg.wandb.use_wandb:
        wandb.log({
            'full_dataset_size': dataset_size,
            'full_dataset_coverage': np.sum(stats['full_state_counts'] > 0) / env.unwrapped.n_states,
            'full_dataset_mean_visits': np.mean(stats['full_state_counts']),
            'full_dataset_std_visits': np.std(stats['full_state_counts']),
            'uniformity_score': stats['uniformity_score'],
            'mean_visit_count': np.mean(stats['mean_counts']),
            'std_visit_count': np.std(stats['mean_counts']),
            'min_visit_count': np.min(stats['mean_counts']),
            'max_visit_count': np.max(stats['mean_counts'])
        })
        wandb.finish()
    
    env.close()


if __name__ == "__main__":
    main()
