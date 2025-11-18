"""
Distribution Matching with Mirror Descent on Two Rooms Environment.

This script demonstrates how to learn a policy that matches a target state 
distribution using mirror descent optimization in a custom Gymnasium environment.
"""
import agent
import utils
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import gymnasium as gym
import hydra
from omegaconf import DictConfig
from PIL import Image
import env
import os
import jax
import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf

class NaivePowr:
    """
    Policy optimizer for maximizing the Q function using mirror descent.
    
    Args:
        env: gym.Env instance
        gamma: Discount factor for occupancy measure
        eta: Learning rate for mirror descent
        gradient_type: Type of KL gradient ('reverse' or 'forward')
    """
    
    def __init__(self, env: gym.Env , gamma: float = 0.9, eta: float = 0.1):
        self.env = env.unwrapped
        self.gamma = gamma
        self.eta = eta
        
        self.n_states = env.unwrapped.n_states
        self.n_actions = env.action_space.n
        self.visited_states = set()
        
        self.dataset = {
            'states': np.array([]),
            'actions': np.array([]),
            'rewards': np.array([]),
            'next_states': np.array([])
        }

        # Get transition matrix from environment
        self.T_operator = self._create_transition_matrix()
        self.R_vector = np.zeros((self.n_states * self.n_actions, 1))
        self.R_tilde = np.zeros((self.n_states, 1))
        
        # Initialize uniform random policy
        self.uniform_policy_operator = self._create_uniform_policy() 
        self.initial_policy_operator = self._create_uniform_policy().copy() 
        self.policy_operator = self.initial_policy_operator.copy()  

        self.agent = None

        self.reference_R_vector = self._create_R_vector()
        self.reference_R_tilde = self._create_R_tilde_vector()
        
        
    def _create_uniform_policy(self) -> np.ndarray:
        """Create uniform random policy operator."""
        P = np.zeros((self.n_states * self.n_actions, self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                row = s * self.n_actions + a
                P[row, s] = 1.0 / self.n_actions
                
        return P
    
    def _create_random_policy(self) -> np.ndarray:
        """Create random random policy operator."""
        P = np.zeros((self.n_states * self.n_actions, self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                row = s * self.n_actions + a
                P[row, s] = np.random.rand()
            P[:, s] /= P[:, s].sum()
        return P

    def _create_transition_matrix(self) -> np.ndarray:
        """
        Get the transition matrix T for all state-action pairs.
        
        Returns:
            T: Transition matrix of shape (n_states, n_states * n_actions)
               T[s', s*n_actions + a] = 1 if action a in state s leads to s'
        """
        n_actions = self.env.action_space.n
        T = np.zeros((self.n_states, self.n_states * n_actions))
        
        for s_idx in range(self.n_states):
            cell = self.env.idx_to_state[s_idx]
            for action in range(n_actions):
                next_cell = self.env.unwrapped.step_from(cell, action)
                next_idx = self.env.state_to_idx[next_cell]
                col = s_idx * n_actions + action
                T[next_idx, col] = 1.0
        
        return T

    def _create_R_vector(self) -> np.ndarray:
        """Create reward vector R for all state-action pairs."""
        R = np.zeros((self.n_states*self.n_actions))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_cell = self.env.unwrapped.step_from(self.env.idx_to_state[s], a)
                if next_cell == self.env.goal_position:
                    R[s * self.n_actions + a] = 0.0  # Reward for reaching goal
                else:
                    R[s * self.n_actions + a] = -1.0  # Initialize rewards to -1
        return R

    def _create_R_tilde_vector(self) -> np.ndarray:
        """Create reward vector R for all state-action pairs."""
        R = np.zeros((self.n_states))
        for s in range(self.n_states):
            cell = self.env.idx_to_state[s]
            if cell == self.env.goal_position:
                R[s] = 0.0  # Reward for reaching goal
            else:
                R[s] = -1.0  # Initialize rewards to -1
        return R
    

    def mirror_descent_update_cumulative(self) -> np.ndarray:
        initial_policy_3d = self.initial_policy_operator.reshape((self.n_states, self.n_actions, self.n_states))
        new_policy_3d = np.zeros_like(initial_policy_3d)
        I = np.eye(self.n_states * self.n_actions)
        Q_estimated = np.linalg.solve(I - self.gamma * self.T_operator.T @ self.policy_operator.T , self.T_operator.T@self.R_tilde)
        Q_estimated = Q_estimated.reshape((self.n_states, self.n_actions))
        self.comulative_Qs += Q_estimated
       
                
        for s in range(self.n_states):
            new_policy_3d[s, :, s] = np.log(initial_policy_3d[s, :, s]) + self.eta * self.comulative_Qs[s, :]
            # try:
            #     new_policy_3d[s, :, s] = jax.nn.softmax(new_policy_3d[s, :, s], axis = 0) # Problems with jax and cuda and torch together, GPU 10800ti too old!!
            # except:
            new_policy_3d[s, :, s] = F.softmax(torch.tensor(new_policy_3d[s, :, s]), dim = 0).numpy()
                
        new_policy_3d = new_policy_3d.reshape((self.n_states * self.n_actions, self.n_states))
    
        return new_policy_3d
      
    
    def save_policy(self, filepath: str) -> None:
        """
        Save the learned policy operator to a file.
        
        Args:
            filepath: Path where to save the policy (will add .npy extension)
        """
        np.save(filepath, self.policy_operator)
        print(f"Policy operator saved to: {filepath}.npy")
    
    def load_policy(self, filepath: str, key: str = None) -> None:
        """
        Load a policy operator from a file.
        
        Args:
            filepath: Path to the saved policy file
        """
        if filepath.endswith('.pt'):
            if self.agent is None:
                with Path(filepath).open('rb') as f:
                    payload = torch.load(f, weights_only=False)
                print(f" Keys in loaded payload: {list(payload.keys())}")
                self.agent = payload[key]
            self.policy_operator = self.initial_policy_operator.copy()  
        else:
            self.policy_operator = np.load(filepath)
        

    def save_transition_matrix(self, filepath: str) -> None:
        """
        Save the transition matrix to a file.
        
        Args:
            filepath: Path where to save the transition matrix (will add .npy extension)
        """
        np.save(filepath, self.T_operator)
        print(f"Transition matrix saved to: {filepath}.npy")
    
    def save_training_data(self, save_dir: str, verbose: bool = True) -> None:
        """
        Save all training artifacts (policy, transition matrix, histories).
        
        Args:
            save_dir: Directory where to save the data
            verbose: Whether to print save messages
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save policy operator
        np.save(os.path.join(save_dir, "policy_operator.npy"), self.policy_operator)
        if verbose:
            print(f"Policy operator saved to: {os.path.join(save_dir, 'policy_operator.npy')}")
        
        # Save transition matrix
        np.save(os.path.join(save_dir, "transition_matrix.npy"), self.T_operator)
        if verbose:
            print(f"Transition matrix saved to: {os.path.join(save_dir, 'transition_matrix.npy')}")
        
        # Save uniform policy for reference
        np.save(os.path.join(save_dir, "uniform_policy_operator.npy"), self.uniform_policy_operator)
        if verbose:
            print(f"Uniform policy operator saved to: {os.path.join(save_dir, 'uniform_policy_operator.npy')}")
        
        # Save metadata
        metadata = {
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'gamma': self.gamma,
            'eta': self.eta
        }
        np.save(os.path.join(save_dir, "metadata.npy"), metadata)
        if verbose:
            print(f"Metadata saved to: {os.path.join(save_dir, 'metadata.npy')}")
            print(f"\n✓ All training data saved to: {save_dir}")
    
    @staticmethod
    def load_training_data(load_dir: str) -> dict:
        """
        Load all training artifacts from a directory.
        
        Args:
            load_dir: Directory from where to load the data
            
        Returns:
            Dictionary containing all loaded data
        """
        data = {}
        data['policy_operator'] = np.load(os.path.join(load_dir, "policy_operator.npy"))
        data['transition_matrix'] = np.load(os.path.join(load_dir, "transition_matrix.npy"))
        data['uniform_policy_operator'] = np.load(os.path.join(load_dir, "uniform_policy_operator.npy"))
        
        if os.path.exists(os.path.join(load_dir, "policy_distance_history.npy")):
            data['policy_distance_history'] = np.load(os.path.join(load_dir, "policy_distance_history.npy"))
        
        data['metadata'] = np.load(os.path.join(load_dir, "metadata.npy"), allow_pickle=True).item()
        
        print(f"✓ Training data loaded from: {load_dir}")
        return data
    
    def collect_dataset(self, n_timesteps: int = 1000) -> None:
        initial_size = len(self.dataset['states'])
        target_size = initial_size + n_timesteps
        
        # Dataset collection
        obs, info = self.env.reset()
        done = False

        while len(self.dataset['states']) < target_size:
            
            action = self.sample_action(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Store transition in dataset
            self.dataset['states'] = np.append(self.dataset['states'], obs)
            self.dataset['actions'] = np.append(self.dataset['actions'], action)
            self.dataset['rewards'] = np.append(self.dataset['rewards'], reward)
            self.dataset['next_states'] = np.append(self.dataset['next_states'], next_obs)

            obs = next_obs
            if done:
                obs, info = self.env.reset()
                done = False

    def train(self, n_pmd_iter: int = 10) -> None:
        # Learning the reward vector
        
        # Convert goal_position to index once
        goal_idx = self.env.state_to_idx[self.env.goal_position]

        for s,a,s_next in zip(self.dataset['states'], self.dataset['actions'], self.dataset['next_states']):
            self.visited_states.add(int(s))
            if int(s_next) != goal_idx:
                row = int(s) * self.n_actions + int(a)
                self.R_vector[row] = -1.0
                self.R_tilde[int(s)] = -1.0
            
        # if len(self.visited_states) > self.n_states - 5:
        #     print(f"R: {self.R_vector}")
        #     print(f"R_tilde: {self.R_tilde}")   
        #     exit()
            # else:
            #     print(f"Reached goal at state {s_next}, no penalty assigned.")
        
        # print(f"reward vector error: {np.linalg.norm(self.R_vector - self.reference_R_vector)}")


        self.comulative_Qs = np.zeros((self.n_states, self.n_actions))
        # PMD iterations
        for _ in range(n_pmd_iter):
            # Compute gradient and update policy
            self.policy_operator = self.mirror_descent_update_cumulative()
   
    
    def eval(self, n_episodes):
        total_rewards = []

        for episode in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.sample_action(obs, eval =True)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                obs = next_obs

            total_rewards.append(episode_reward)
        return total_rewards


    def get_policy_per_state(self, uniform_policy: bool = False) -> np.ndarray:
        """
        Extract policy probabilities π(a|s) for each state.
        Args:
            uniform_policy: Whether to return uniform random policy instead of learned policy
        
        Returns:
            Array of shape (n_states, n_actions) with policy probabilities
        """
        if uniform_policy:
            policy_matrix = self.uniform_policy_operator.reshape((self.n_states, self.n_actions, self.n_states))
            raise NotImplementedError("Uniform policy per state extraction not implemented yet.")
        else:
            policy_matrix = self.policy_operator.reshape((self.n_states, self.n_actions, self.n_states))
        policy_per_state = np.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            policy_per_state[s, :] = policy_matrix[s, :, s]

        return policy_per_state

    def sample_action(self, state: int, uniform_policy: bool = False, eval: bool = False) -> int:
        """
        Sample an action from the policy for a given state.
        
        Args:
            state: Current state index
            uniform_policy: Whether to use a uniform random policy instead of learned policy
        
        Returns:
            Sampled action index
        """
        if self.agent is not None:
            obs  = np.zeros(self.n_states, dtype=np.float32)
            obs[state] = 1.0
            action = self.agent.act(obs, {}, 1000000, False)
        
            # Return numpy action
            return action
        else:
            policy_per_state = self.get_policy_per_state(uniform_policy=uniform_policy) 
            action_probs = policy_per_state[state]

            # Check stochasticity
            prob_sum = np.sum(action_probs)
            if not np.isclose(prob_sum, 1.0, atol=1e-8):
                print(f"⚠️  State {state}: probabilities sum to {prob_sum:.10f} (deviation: {abs(prob_sum - 1.0):.2e})")
                print(f"   Probabilities: {action_probs}")
            # if eval:
            #     # In evaluation mode, choose the action with highest probability
            #     return np.argmax(action_probs)
            
            action_probs = action_probs / np.sum(action_probs)  # Normalize probabilities, it could be slightly off due to numerical issues
            return np.random.choice(self.n_actions, p=action_probs)
    
    def stochasticity_check(self, operator = None) -> None:
        """Check and report policy stochasticity statistics."""
        if operator is None:
            operator = self.policy_operator
        
        
        reference = np.ones((self.n_states))

        if len(operator.shape) == 1 or (len(operator.shape) == 2 and 1 == operator.shape[1]):
            summed_operator = operator.sum(axis=0)
            reference = np.ones((1))
        elif len(operator.shape) == 2:
            if operator.shape[0] == self.n_states:
                summed_operator = operator.sum(axis=1)
            elif operator.shape[1] == self.n_states:
                summed_operator = operator.sum(axis=0)
        else:
            raise ValueError("Operator shape is incompatible for stochasticity check.")
        
    
        avg_deviation = np.mean(np.abs(summed_operator - reference.flatten()))
        total_deviation = np.sum(np.abs(summed_operator - reference.flatten())) 
        max_deviation = np.max(np.abs(summed_operator - reference.flatten()))
        max_deviation_state = np.argmax(np.abs(summed_operator - reference.flatten()))
        print(f"\n{'='*60}")
        print("POLICY STOCHASTICITY CHECK")
        print(f"{'='*60}")
        print(f"Total sum deviation from 1.0: {total_deviation:.10e}")
        print(f"Average deviation per state: {avg_deviation:.10e}")
        print(f"Max deviation: {max_deviation:.10e} (state {max_deviation_state})")
        
        if total_deviation > 1e-6:
            print(f"\n⚠️  WARNING: Policy significantly deviates from stochastic!")
        else:
            print(f"\n✓ Policy is stochastic (within tolerance)")
        print(f"{'='*60}\n")
    
    def reset_policy(self, p_path: str = None) -> None:
        """Reset the policy to a random initialization or load a pretrained policy."""
        if p_path:
            self.load_policy(p_path)
        else:
            self.policy_operator = self.initial_policy_operator.copy()
    
    def rollout(self, start_state: int, horizon: int, uniform_policy: bool = False, seed: int = None) -> tuple:
        """
        Execute a rollout following the learned policy.
        
        Args:
            start_state: Starting state index
            horizon: Maximum number of steps
            uniform_policy: Whether to use a uniform random policy instead of learned policy
            seed: Random seed for reproducibility
        
        Returns:
            trajectory: List of (state, action) tuples
            states: List of visited states
            actions: List of taken actions
        """
        if seed is not None:
            np.random.seed(seed)
        
        trajectory = []
        states = [start_state]
        actions = []
        
        current_state = start_state
        
        for step in range(horizon):
            # Sample action from policy
            action = self.sample_action(current_state, uniform_policy=uniform_policy)
            actions.append(action)
            
            # Get next state from environment
            current_cell = self.env.idx_to_state[current_state]
            next_cell = self.env.unwrapped.step_from(current_cell, action)
            next_state = self.env.state_to_idx(next_cell)
            
            trajectory.append((current_state, action))
            states.append(next_state)
            current_state = next_state
        
        return trajectory, states, actions
    
    def visualize_policy_bars(self, save_path: str) -> None:
        """
        Visualize policy as bar charts for each state.
        
        Args:
            save_path: Path to save the visualization
        """
        # Get grid dimensions
        max_x = max(cell[0] for cell in self.env.cells)
        max_y = max(cell[1] for cell in self.env.cells)
        min_x = min(cell[0] for cell in self.env.cells)
        min_y = min(cell[1] for cell in self.env.cells)
        grid_width = max_x - min_x + 1
        grid_height = max_y - min_y + 1
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(-0.5, grid_width - 0.5)
        ax.set_ylim(grid_height - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.set_title('Policy Probabilities per State')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
        
        policy_per_state = self.get_policy_per_state()
        action_colors = ['red', 'blue', 'green', 'orange']
        action_names = ['up', 'down', 'left', 'right']
        
        for s_idx in range(self.n_states):
            x, y = self.env.idx_to_state[s_idx]
            x_plot, y_plot = x - min_x, y - min_y
            
            # Draw background
            rect = Rectangle((x_plot - 0.4, y_plot - 0.4), 0.8, 0.8,
                           facecolor='lightgray', edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            
            # Draw mini bar chart
            probs = policy_per_state[s_idx]
            bar_width = 0.15
            bar_spacing = 0.2
            start_x = x_plot - 1.5 * bar_spacing
            max_bar_height = 0.7
            
            for a_idx in range(self.n_actions):
                bar_x = start_x + a_idx * bar_spacing
                bar_height = probs[a_idx] * max_bar_height
                
                bar_rect = Rectangle((bar_x - bar_width/2, y_plot + 0.35 - bar_height),
                                    bar_width, bar_height,
                                    facecolor=action_colors[a_idx],
                                    edgecolor='black', linewidth=0.3)
                ax.add_patch(bar_rect)
        
        # Add legend
        legend_elements = [Patch(facecolor=action_colors[i], label=action_names[i])
                          for i in range(self.n_actions)]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def visualize_dataset_heatmap(self, save_path: str) -> None:
        """
        Visualize dataset state visitation as heatmap.
        
        Args:
            save_path: Path to save the heatmap
        """
        # Get grid dimensions
        max_x = max(cell[0] for cell in self.env.cells)
        max_y = max(cell[1] for cell in self.env.cells)
        min_x = min(cell[0] for cell in self.env.cells)
        min_y = min(cell[1] for cell in self.env.cells)
        grid_width = max_x - min_x + 1
        grid_height = max_y - min_y + 1
        
        # Count state visitations
        state_counts = np.zeros(self.n_states)
        for state in self.dataset['states']:
            state_counts[int(state)] += 1
        
        # Create grid
        grid = np.zeros((grid_height, grid_width))
        for s_idx in range(self.n_states):
            x, y = self.env.idx_to_state[s_idx]
            grid[y - min_y, x - min_x] = state_counts[s_idx]
        
        # Mask zero values to show background color
        masked_grid = np.ma.masked_where(grid == 0, grid)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        # Set light gray background
        ax.set_facecolor("#B2B2B2")
        # Plot only non-zero values
        im = ax.imshow(masked_grid, cmap='YlOrRd', interpolation='nearest')
        ax.set_title(f'Dataset State Visitation (n={len(self.dataset["states"])})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xticks(np.arange(-0.5, grid_width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_height, 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.colorbar(im, ax=ax, label='Visit Count')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def visualize_eval_trajectory(self, save_path: str, n_steps: int = 100, seed: int = None) -> None:
        """
        Visualize a single evaluation trajectory.
        
        Args:
            save_path: Path to save the visualization
            n_steps: Number of steps in trajectory
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Run evaluation episode
        obs, info = self.env.reset()
        trajectory_states = [obs]
        done = False
        steps = 0
        
        while not done and steps < n_steps:
            action = self.sample_action(obs, eval =True)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            trajectory_states.append(next_obs)
            obs = next_obs
            steps += 1
        
        # Get grid dimensions
        max_x = max(cell[0] for cell in self.env.cells)
        max_y = max(cell[1] for cell in self.env.cells)
        min_x = min(cell[0] for cell in self.env.cells)
        min_y = min(cell[1] for cell in self.env.cells)
        grid_width = max_x - min_x + 1
        grid_height = max_y - min_y + 1
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(-0.5, grid_width - 0.5)
        ax.set_ylim(grid_height - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.set_title(f'Evaluation Trajectory ({steps} steps)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
        
        # Draw all valid cells
        for s_idx in range(self.n_states):
            x, y = self.env.idx_to_state[s_idx]
            x_plot, y_plot = x - min_x, y - min_y
            rect = Rectangle((x_plot - 0.4, y_plot - 0.4), 0.8, 0.8,
                           facecolor='lightgray', edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
        
        # Draw trajectory path
        path_x = []
        path_y = []
        for state in trajectory_states:
            x, y = self.env.idx_to_state[int(state)]
            path_x.append(x - min_x)
            path_y.append(y - min_y)
        
        # Plot path with arrows
        for i in range(len(path_x) - 1):
            ax.annotate('', xy=(path_x[i+1], path_y[i+1]), 
                       xytext=(path_x[i], path_y[i]),
                       arrowprops=dict(arrowstyle='->', color='blue', lw=1.5, alpha=0.6))
        
        # Mark start and end
        ax.plot(path_x[0], path_y[0], 'go', markersize=15, label='Start', zorder=10)
        ax.plot(path_x[-1], path_y[-1], 'r*', markersize=20, label='End', zorder=10)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def save_trajectories(self, save_dir: str, n_episodes: int = 5, max_steps: int = 100) -> None:
        """
        Save sample trajectories from the current policy.
        
        Args:
            save_dir: Directory where to save the trajectory files
            n_episodes: Number of episodes to simulate
            max_steps: Maximum number of steps per episode
        """
        os.makedirs(save_dir, exist_ok=True)
        
        for episode in range(n_episodes):
            # Generate trajectory
            trajectory, states, actions = self.rollout(
                start_state=self.env.state_to_idx[self.env.start_position],
                horizon=max_steps,
                uniform_policy=False,
                seed=episode
            )
            
            # Convert to numpy array for saving
            states_array = np.array(states)
            actions_array = np.array(actions)
            
            # Save state-action pairs as separate files
            np.savetxt(os.path.join(save_dir, f"trajectory_{episode+1}_states.txt"), states_array, fmt='%d')
            np.savetxt(os.path.join(save_dir, f"trajectory_{episode+1}_actions.txt"), actions_array, fmt='%d')
            
            print(f"Saved trajectory {episode+1} to {save_dir}")
    


@hydra.main(version_base=None, config_path="configs", config_name="config_powr")
def main(cfg: DictConfig):
    """Main execution function."""

    # Set random seed
    np.random.seed(cfg.experiment.seed)
    
    # Close any existing figures from previous runs
    plt.close('all')
    
    # Get Hydra output directory
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"\nHydra output directory: {output_dir}\n")
    
    # Initialize wandb
    if cfg.wandb.use_wandb:
        if cfg.wandb.wandb_id is not None and cfg.wandb.wandb_id != "none":
            wandb.init(
                id=cfg.wandb.wandb_id,
                resume='must',
                project=cfg.wandb.wandb_project,
                name=cfg.wandb.wandb_run_name,
                tags=cfg.wandb.wandb_tag.split('_') if cfg.wandb.wandb_tag and cfg.wandb.wandb_tag != "none" else None,
                sync_tensorboard=True,
                mode='online')
        else:
            wandb.init(
                config=OmegaConf.to_container(cfg, resolve=True),
                project=cfg.wandb.wandb_project,
                name=cfg.wandb.wandb_run_name,
                tags=cfg.wandb.wandb_tag.split('_') if cfg.wandb.wandb_tag and cfg.wandb.wandb_tag != "none" else None,
                sync_tensorboard=True,
                mode='online')

    
    # Create environment using gym.make with config parameters
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
    
    # Create environment using gym.make
    env = gym.make(cfg.env.name, **env_kwargs)
    env.reset(seed=cfg.experiment.seed)
    print(f"Environment {cfg.env.name} created.")   
    frame = env.render()
    img = Image.fromarray(frame)
    img.save(os.path.join(output_dir, f"{cfg.env.name}.png"))

    print(f"Environment: {cfg.env.name}")
    print(f"Start position: {env.unwrapped.start_position}")
    print(f"Environment created with {env.unwrapped.n_states} states\n")

    # Get parameters from config
    n_runs = cfg.experiment.n_runs
    eta = cfg.experiment.eta
    pmd_iter_updates = cfg.experiment.pmd_iter_updates
    gamma = cfg.experiment.gamma
    eval_episodes = cfg.experiment.eval_episodes
    timestep_interval = cfg.experiment.timestep_interval
    timesteps = cfg.experiment.timesteps
    
    p_path = cfg.experiment.p_path
    

    # Storage for all runs
    all_runs_rewards = []  # List of lists: [run][checkpoint]
    all_runs_data_len = []  # List of lists: [run][checkpoint]
    
    for run_idx in range(n_runs):
        print(f"\n{'='*60}")
        print(f"RUN {run_idx + 1}/{n_runs}")
        print(f"{'='*60}\n")
        
        # Set seed for this run
        run_seed = cfg.experiment.seed + run_idx * 1000
        np.random.seed(run_seed)
        env.reset(seed=run_seed)
        
        # Initialize NaivePowr agent
        agent = NaivePowr(env, gamma=gamma, eta=eta)
            
        if p_path:
            agent.load_policy(p_path, key = 'agent') # TODO from cfg
        agent.visualize_policy_bars(os.path.join(output_dir, "policy_bars.png"))
        eval_rewards = []
        data_len = []
        checkpoint_idx = 0
        
        while True:
            agent.collect_dataset(n_timesteps=timestep_interval)
            # Create checkpoint directory
            checkpoint_dir = os.path.join(output_dir, f"run_{run_idx:02d}", f"checkpoint_{checkpoint_idx:04d}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            agent.visualize_policy_bars(os.path.join(checkpoint_dir, "starting_policy_bars.png"))

            agent.train(n_pmd_iter=pmd_iter_updates)
            eval_reward = agent.eval(n_episodes=eval_episodes)
            
            
            
            # Save visualizations for this checkpoint
            agent.visualize_policy_bars(os.path.join(checkpoint_dir, "policy_bars.png"))
            agent.visualize_dataset_heatmap(os.path.join(checkpoint_dir, "dataset_heatmap.png"))
            agent.visualize_eval_trajectory(
                os.path.join(checkpoint_dir, "eval_trajectory.png"),
                n_steps=100,
                seed=run_seed + checkpoint_idx
            )
            
            agent.reset_policy(p_path)
            
            eval_rewards.append(np.mean(eval_reward))
            data_len.append(len(agent.dataset['states']))
            
            # Log to wandb
            if cfg.wandb.use_wandb:
                wandb.log({
                    f'eval_reward_mean': np.mean(eval_reward),
                    f'eval_reward_std': np.std(eval_reward),
                    f'dataset_length': len(agent.dataset['states']),
                    f'checkpoint': checkpoint_idx
                })
            
            print(f"Dataset size: {len(agent.dataset['states'])}, "
                  f"Eval reward: {np.mean(eval_reward):.2f} ± {np.std(eval_reward):.2f}")
            
            checkpoint_idx += 1
            
            if len(agent.dataset['states']) >= timesteps:
                break
        
        # Store results for this run
        all_runs_rewards.append(eval_rewards)
        all_runs_data_len.append(data_len)
        
        # Save final model for this run
        run_dir = os.path.join(output_dir, f"run_{run_idx:02d}", "final_model")
        agent.save_training_data(run_dir, verbose=False)
    
    # Compute statistics across runs
    # Find minimum length (in case runs have different checkpoint counts)
    min_checkpoints = min(len(rewards) for rewards in all_runs_rewards)
    
    # Truncate all runs to same length and convert to numpy arrays
    rewards_array = np.array([rewards[:min_checkpoints] for rewards in all_runs_rewards])
    data_len_array = np.array([data_len[:min_checkpoints] for data_len in all_runs_data_len])
    
    # Compute median and 95% confidence interval across runs
    median_rewards = np.median(rewards_array, axis=0)
    ci_lower = np.percentile(rewards_array, 2.5, axis=0)
    ci_upper = np.percentile(rewards_array, 97.5, axis=0)
    mean_data_len = np.mean(data_len_array, axis=0)
    
    # Plot evaluation rewards with confidence interval
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(mean_data_len, median_rewards, marker='o', linewidth=2, label='Median')
    ax.fill_between(
        mean_data_len,
        ci_lower,
        ci_upper,
        alpha=0.3,
        label=f'95% CI (n={n_runs} runs)'
    )
    ax.set_xlabel('Dataset Size (timesteps)', fontsize=12)
    ax.set_ylabel('Average Eval Reward', fontsize=12)
    ax.set_title(f'Evaluation Reward vs Dataset Size ({n_runs} runs)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eval_rewards_with_error.png'), dpi=150)
    print(f"Saved evaluation rewards plot to: {os.path.join(output_dir, 'eval_rewards_with_error.png')}")
    
    # Log final plot to wandb
    if cfg.wandb.use_wandb:
        wandb.log({
            'aggregated/eval_rewards_plot': wandb.Image(fig),
            'aggregated/final_median_reward': median_rewards[-1],
            'aggregated/final_ci_lower': ci_lower[-1],
            'aggregated/final_ci_upper': ci_upper[-1],
            'aggregated/final_dataset_size': mean_data_len[-1]
        })
    
    plt.close(fig)
    
    # Save aggregated statistics
    stats = {
        'n_runs': n_runs,
        'median_rewards': median_rewards,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'mean_data_len': mean_data_len,
        'all_runs_rewards': rewards_array,
        'all_runs_data_len': data_len_array
    }
    np.save(os.path.join(output_dir, 'aggregated_stats.npy'), stats)
    
    print(f"\n{'='*60}")
    print("FINAL STATISTICS")
    print(f"{'='*60}")
    print(f"Runs completed: {n_runs}")
    print(f"Final median reward: {median_rewards[-1]:.2f} (95% CI: [{ci_lower[-1]:.2f}, {ci_upper[-1]:.2f}])")
    print(f"Final dataset size: {mean_data_len[-1]:.0f}")
    print(f"{'='*60}\n")
    
    # Finish wandb run
    if cfg.wandb.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()