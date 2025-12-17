"""
Test State Coverage Speed - Evaluate how fast a policy reaches all states.

This script tests how quickly a given policy (pretrained or uniform) can visit
all states in a tabular environment. Useful for evaluating exploration efficiency.
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
import seaborn as sns


class StateCoverageTracker:
    """Track state coverage during policy rollouts."""
    
    def __init__(self, n_states: int, evaluate_every: int):
        self.n_states = n_states
        self.evaluate_every = evaluate_every
        self.visited_states = set()
        self.coverage_history = []
        self.timestep_history = []
        self.state_visit_counts = np.zeros(n_states, dtype=np.int64)
        self._timestep = 0
        self._next_eval = evaluate_every
    
    def add_state(self, state: int):
        """Add a visited state and record coverage at evaluation intervals."""
        self.visited_states.add(int(state))
        self.state_visit_counts[int(state)] += 1
        self._timestep += 1
        
        # Record coverage at evaluation intervals
        if self._timestep >= self._next_eval:
            coverage = len(self.visited_states) / self.n_states
            self.coverage_history.append(coverage)
            self.timestep_history.append(self._timestep)
            self._next_eval += self.evaluate_every
            
            # Once we reach full coverage, keep monitoring but coverage should stay at 1.0
            if coverage >= 1.0 and len(self.coverage_history) > 1:
                if self.coverage_history[-2] >= 1.0:
                    # Already reached full coverage, verify we don't go backwards
                    assert coverage >= 1.0, "Coverage decreased after reaching 100%!"
    
    @property
    def coverage(self) -> float:
        """Current coverage ratio [0, 1]."""
        return len(self.visited_states) / self.n_states
    
    @property
    def is_complete(self) -> bool:
        """Check if all states have been visited."""
        return len(self.visited_states) == self.n_states
    
    def reset(self):
        """Reset the tracker."""
        self.visited_states.clear()
        self.coverage_history.clear()
        self.timestep_history.clear()
        self.state_visit_counts = np.zeros(self.n_states, dtype=np.int64)
        self._timestep = 0
        self._next_eval = self.evaluate_every


class PolicyEvaluator:
    """Evaluate state coverage for a given policy."""
    
    def __init__(self, env: gym.Env, agent=None, uniform_policy: bool = False):
        self.env = env
        self.agent = agent
        self.uniform_policy = uniform_policy
        self.n_states = env.unwrapped.n_states
        self.n_actions = env.action_space.n
    
    def sample_action(self, obs: np.ndarray) -> int:
        """Sample action from policy or uniform distribution."""
        if self.uniform_policy or self.agent is None:
            return self.env.action_space.sample()
        else:
            # Use agent's policy
            obs_onehot = np.zeros(self.n_states, dtype=np.float32)
            obs_onehot[obs] = 1.0
            return self.agent.act(obs_onehot, {}, step=1000000, eval_mode=True)
    
    def run_coverage_test(self, max_steps: int, evaluate_every: int, seed: int = None) -> StateCoverageTracker:
        """
        Run a coverage test for a given number of steps.
        
        Args:
            max_steps: Maximum number of steps to run
            evaluate_every: Evaluate coverage every N steps
            seed: Random seed for reproducibility
            
        Returns:
            StateCoverageTracker with coverage history
        """
        if seed is not None:
            np.random.seed(seed)
            self.env.reset(seed=seed)
        
        tracker = StateCoverageTracker(self.n_states, evaluate_every)
        obs, info = self.env.reset()
        tracker.add_state(obs)
        
        steps = 0
        while steps < max_steps:
            action = self.sample_action(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            tracker.add_state(next_obs)
            
            obs = next_obs
            steps += 1
            
            # Reset environment if episode ends
            if terminated or truncated:
                obs, info = self.env.reset()
                tracker.add_state(obs)
        
        # Add final evaluation if we haven't evaluated at the last step
        if tracker._timestep > tracker.timestep_history[-1] if tracker.timestep_history else True:
            coverage = len(tracker.visited_states) / tracker.n_states
            tracker.coverage_history.append(coverage)
            tracker.timestep_history.append(tracker._timestep)
        
        return tracker


def interpolate_coverage(timesteps: np.ndarray, coverage: np.ndarray, 
                         target_timesteps: np.ndarray) -> np.ndarray:
    """
    Interpolate coverage values to common timestep grid.
    
    Args:
        timesteps: Original timestep array
        coverage: Original coverage array
        target_timesteps: Target timestep grid
        
    Returns:
        Interpolated coverage values
    """
    return np.interp(target_timesteps, timesteps, coverage)


def create_state_distribution_heatmap(state_distributions: list, 
                                     env: gym.Env, 
                                     output_path: str,
                                     policy_type: str):
    """
    Create a heatmap showing the average state visitation counts (not normalized).
    
    Args:
        state_distributions: List of state visit count arrays from all runs
        env: The environment instance
        output_path: Path to save the heatmap
        policy_type: Type of policy used
    """
    # Calculate average hit count across runs (NOT normalized)
    avg_hits = np.mean(state_distributions, axis=0)
    
    # Get grid dimensions from environment
    # Handle environments that may not have grid_height/grid_width attributes
    if hasattr(env.unwrapped, 'grid_height') and hasattr(env.unwrapped, 'grid_width'):
        grid_height = env.unwrapped.grid_height
        grid_width = env.unwrapped.grid_width
    else:
        # Calculate grid dimensions from cells
        # Handle both dict and list types for cells
        cells_data = env.unwrapped.cells
        
        if isinstance(cells_data, dict):
            cells = list(cells_data.keys())
        elif isinstance(cells_data, list):
            cells = cells_data
        else:
            raise ValueError(f"Unsupported cells type: {type(cells_data)}")
        
        if not cells:
            raise ValueError("Environment has no cells defined")
        
        x_coords = [pos[0] for pos in cells]
        y_coords = [pos[1] for pos in cells]
        grid_width = max(x_coords) + 1
        grid_height = max(y_coords) + 1
    
    # Create a full grid with NaN for non-navigable cells
    distribution_grid = np.full((grid_height, grid_width), np.nan)
    
    # Map state indices back to (x, y) positions
    # Handle both dict and list types for cells
    cells_data = env.unwrapped.cells
    if isinstance(cells_data, dict):
        state_to_pos = {state: pos for pos, state in cells_data.items()}
    elif isinstance(cells_data, list):
        state_to_pos = {idx: pos for idx, pos in enumerate(cells_data)}
    else:
        raise ValueError(f"Unsupported cells type: {type(cells_data)}")
    
    # Fill in the average hits for navigable cells only
    for state_idx, avg_count in enumerate(avg_hits):
        if state_idx in state_to_pos:
            x, y = state_to_pos[state_idx]
            distribution_grid[y, x] = avg_count
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(10, grid_width * 0.8), max(8, grid_height * 0.8)))
    
    # Create heatmap with NaN values shown as gray
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color='lightgray', alpha=0.3)
    
    im = ax.imshow(distribution_grid, cmap=cmap, aspect='equal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Visit Count', rotation=270, labelpad=20, fontsize=11)
    
    # Set tick positions at cell centers for labels
    ax.set_xticks(np.arange(grid_width))
    ax.set_yticks(np.arange(grid_height))
    ax.set_xticklabels(np.arange(grid_width))
    ax.set_yticklabels(np.arange(grid_height))
    
    # Add grid lines between cells (at -0.5, 0.5, 1.5, etc.)
    ax.set_xticks(np.arange(-0.5, grid_width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_height, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
    ax.tick_params(which='minor', length=0)  # Hide minor tick marks
    
    # Add value annotations only for navigable cells
    max_val = np.nanmax(distribution_grid)
    for i in range(grid_height):
        for j in range(grid_width):
            value = distribution_grid[i, j]
            if not np.isnan(value):
                # Use white text for dark cells, black for light cells
                text_color = 'white' if value > max_val * 0.5 else 'black'
                # Format as integer if close to integer, otherwise with 1 decimal
                if abs(value - round(value)) < 0.1:
                    text = f'{value:.0f}'
                else:
                    text = f'{value:.1f}'
                ax.text(j, i, text, 
                       ha='center', va='center', 
                       color=text_color, fontsize=8)
    
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title(f'Average State Visit Counts ({policy_type} Policy)', 
                fontsize=14, pad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


@hydra.main(version_base=None, config_path="configs", config_name="config_reach_full_coverage")
def main(cfg: DictConfig):
    """Main execution function."""
    
    # Set random seed
    np.random.seed(cfg.seed)
    
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
    print(f"Environment {cfg.env.name}")
    env = gym.make(cfg.env.name, **env_kwargs)
    env.reset(seed=cfg.seed)
    print(f"Environment {cfg.env.name} created with {env.unwrapped.n_states} states\n")
    
    # Save environment visualization
    frame = env.render()
    img = Image.fromarray(frame)
    img.save(os.path.join(output_dir, f"{cfg.env.name}.png"))
    
    # Load agent if policy path is provided
    agent = None
    use_uniform = cfg.policy_path is None or cfg.policy_path == "none"
    
    if not use_uniform:
        print(f"Loading pretrained policy from: {cfg.policy_path}")
        with Path(cfg.policy_path).open('rb') as f:
            payload = torch.load(f, weights_only=False)
        
        # Extract agent from payload
        if 'agent' in payload:
            agent = payload['agent']
            print("Agent loaded successfully")
        else:
            print(f"Available keys in payload: {list(payload.keys())}")
            raise ValueError("Could not find 'agent' in loaded checkpoint")
    else:
        print("Using uniform random policy")
    
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
    
    # Storage for all runs
    all_runs_timesteps = []
    all_runs_coverage = []
    all_runs_state_distributions = []
    
    print(f"\n{'='*60}")
    print(f"RUNNING {cfg.n_runs} COVERAGE TESTS")
    print(f"Max steps per run: {cfg.max_steps}")
    print(f"Evaluate every: {cfg.evaluate_every} steps")
    print(f"Policy type: {'Uniform Random' if use_uniform else 'Pretrained'}")
    print(f"{'='*60}\n")
    
    # Run multiple coverage tests
    for run_idx in range(cfg.n_runs):
        print(f"Run {run_idx + 1}/{cfg.n_runs}...", end=" ")
        
        run_seed = cfg.seed + run_idx * 1000
        evaluator = PolicyEvaluator(env, agent=agent, uniform_policy=use_uniform)
        tracker = evaluator.run_coverage_test(
            max_steps=cfg.max_steps, 
            evaluate_every=cfg.evaluate_every,
            seed=run_seed
        )
        
        all_runs_timesteps.append(np.array(tracker.timestep_history))
        all_runs_coverage.append(np.array(tracker.coverage_history))
        all_runs_state_distributions.append(tracker.state_visit_counts.copy())
        
        # Find when full coverage was reached
        full_coverage_idx = np.where(np.array(tracker.coverage_history) >= 1.0)[0]
        if len(full_coverage_idx) > 0:
            completion_step = tracker.timestep_history[full_coverage_idx[0]]
            status = f"✓ Complete at step {completion_step}"
        else:
            completion_step = None
            status = f"✗ Incomplete ({tracker.coverage:.2%})"
        
        print(status)
        
        # Log to wandb
        if cfg.wandb.use_wandb:
            wandb.log({
                'run_idx': run_idx,
                'final_coverage': tracker.coverage,
                'completion_steps': completion_step,
                'is_complete': tracker.is_complete
            })
    
    # Find common timestep grid for interpolation
    max_timestep = max(ts[-1] for ts in all_runs_timesteps)
    # Use evaluation intervals as the common grid
    common_timesteps = np.arange(cfg.evaluate_every, max_timestep + cfg.evaluate_every, cfg.evaluate_every)
    
    # Interpolate all runs to common grid
    interpolated_coverage = []
    for timesteps, coverage in zip(all_runs_timesteps, all_runs_coverage):
        interp_cov = interpolate_coverage(timesteps, coverage, common_timesteps)
        interpolated_coverage.append(interp_cov)
    
    coverage_array = np.array(interpolated_coverage)
    
    # Compute statistics
    median_coverage = np.median(coverage_array, axis=0)
    ci_lower = np.percentile(coverage_array, 2.5, axis=0)
    ci_upper = np.percentile(coverage_array, 97.5, axis=0)
    
    # Compute completion statistics
    completed_runs = np.sum(coverage_array[:, -1] >= 1.0)
    
    # Find median steps to completion
    completion_steps = []
    for cov_history, ts_history in zip(all_runs_coverage, all_runs_timesteps):
        full_cov_idx = np.where(cov_history >= 1.0)[0]
        if len(full_cov_idx) > 0:
            completion_steps.append(ts_history[full_cov_idx[0]])
    
    # Plot coverage over time
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot individual runs with low alpha
    for run_cov in coverage_array:
        ax.plot(common_timesteps, run_cov, alpha=0.2, color='gray', linewidth=0.5)
    
    # Plot median and confidence interval
    ax.plot(common_timesteps, median_coverage, linewidth=2.5, color='blue', label='Median', zorder=10)
    ax.fill_between(
        common_timesteps,
        ci_lower,
        ci_upper,
        alpha=0.3,
        color='blue',
        label=f'95% CI (n={cfg.n_runs} runs)',
        zorder=5
    )
    
    # Add horizontal line at full coverage
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, 
               label='Full Coverage', alpha=0.7)
    
    # Add vertical line at median completion time if available
    if len(completion_steps) > 0:
        median_completion = np.median(completion_steps)
        ax.axvline(x=median_completion, color='red', linestyle=':', linewidth=2, 
                   label=f'Median Completion: {median_completion:.0f} steps', alpha=0.7)
    
    ax.set_xlabel('Timesteps', fontsize=12)
    ax.set_ylabel('State Coverage', fontsize=12)
    policy_type = 'Uniform Random' if use_uniform else 'Pretrained'
    ax.set_title(f'State Coverage vs Timesteps ({policy_type} Policy)', fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'state_coverage_plot.png'), dpi=150)
    print(f"\nCoverage plot saved to: {os.path.join(output_dir, 'state_coverage_plot.png')}")
    plt.close(fig)
    
    # Create state distribution heatmap
    heatmap_path = os.path.join(output_dir, 'state_distribution_heatmap.png')
    create_state_distribution_heatmap(
        all_runs_state_distributions,
        env,
        heatmap_path,
        policy_type
    )
    print(f"State distribution heatmap saved to: {heatmap_path}")
    
    # Save statistics
    stats = {
        'n_runs': cfg.n_runs,
        'n_states': env.unwrapped.n_states,
        'max_steps': cfg.max_steps,
        'evaluate_every': cfg.evaluate_every,
        'policy_type': policy_type,
        'common_timesteps': common_timesteps,
        'median_coverage': median_coverage,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'all_runs_coverage': coverage_array,
        'all_runs_timesteps': [np.array(ts) for ts in all_runs_timesteps],
        'all_runs_state_distributions': all_runs_state_distributions,
        'completion_steps': completion_steps,
        'completed_runs': completed_runs
    }
    np.save(os.path.join(output_dir, 'coverage_stats.npy'), stats, allow_pickle=True)
    
    # Print summary
    print(f"\n{'='*60}")
    print("COVERAGE TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Environment: {cfg.env.name} ({env.unwrapped.n_states} states)")
    print(f"Policy type: {policy_type}")
    print(f"Runs completed: {cfg.n_runs}")
    print(f"Runs reaching full coverage: {completed_runs}/{cfg.n_runs} ({completed_runs/cfg.n_runs:.1%})")
    
    if len(completion_steps) > 0:
        print(f"Median steps to full coverage: {np.median(completion_steps):.0f}")
        print(f"Mean steps to full coverage: {np.mean(completion_steps):.0f} ± {np.std(completion_steps):.0f}")
        print(f"Min/Max steps: {np.min(completion_steps):.0f} / {np.max(completion_steps):.0f}")
    
    print(f"Final median coverage: {median_coverage[-1]:.2%}")
    print(f"{'='*60}\n")
    
    # Log final statistics to wandb
    if cfg.wandb.use_wandb:
        wandb.log({
            'final_median_coverage': median_coverage[-1],
            'completed_runs_ratio': completed_runs / cfg.n_runs,
            'median_steps_to_completion': np.median(completion_steps) if len(completion_steps) > 0 else None
        })
        wandb.finish()
    
    env.close()


if __name__ == "__main__":
    main()
