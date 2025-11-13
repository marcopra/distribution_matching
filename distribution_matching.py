"""
Distribution Matching with Mirror Descent on Two Rooms Environment.

This script demonstrates how to learn a policy that matches a target state 
distribution using mirror descent optimization in a custom Gymnasium environment.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import gymnasium as gym
import hydra
from omegaconf import DictConfig
from env import TwoRoomsEnv, SingleRoomEnv
import os

class DistributionMatcher:
    """
    Policy optimizer for matching target state distributions using mirror descent.
    
    Args:
        env: gym.Env instance
        gamma: Discount factor for occupancy measure
        eta: Learning rate for mirror descent
        gradient_type: Type of KL gradient ('reverse' or 'forward')
    """
    
    def __init__(self, env: gym.Env , gamma: float = 0.9, 
                 eta: float = 0.1, alpha = 0.1, gradient_type: str = 'reverse'):
        self.env = env
        self.gamma = gamma
        self.eta = eta
        self.gradient_type = gradient_type
        self.alpha = alpha
        
        self.n_states = env.n_states
        self.n_actions = env.action_space.n
        
        # Get transition matrix from environment
        self.T_operator = self._create_transition_matrix()
        
        # Initialize uniform random policy
        self.uniform_policy_operator = self._create_uniform_policy() 
        self.policy_operator = self._create_random_policy() 
        
        # History tracking
        self.kl_history = []
        
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
                next_cell = self.env._step_from(cell, action)
                next_idx = self.env.state_to_idx[next_cell]
                col = s_idx * n_actions + action
                T[next_idx, col] = 1.0
                # if next_cell == (3, 3) or cell == (3, 3) or next_cell == (3, 2) or cell == (3, 2) or next_cell == (2, 3) or cell == (2, 3) or next_cell == (1, 3) or cell == (1, 3) or next_cell == (3, 1) or cell == (3, 1) or next_cell == (2, 2) or cell == (2, 2):  # Example of special handling for terminal state
                #     T[next_idx, col] = 0.0 # No transitions from terminal state
        
        return T
    
    def M_pi_operator(self, P: np.ndarray) -> np.ndarray:
        """
        Compute M_π = T ∘ Π_π operator.
        Maps state distribution to next state distribution under policy.
        """
        return self.T_operator @ P
    
    def compute_discounted_occupancy(self, nu0: np.ndarray, P: np.ndarray = None) -> np.ndarray:
        """
        Compute discounted occupancy measure: ν_π = (1-γ)(I - γM_π)^{-1} ν_0
        
        Args:
            nu0: Initial state distribution
            P: Policy operator (uses self.policy_operator if None)
        
        Returns:
            Discounted occupancy measure
        """
        if P is None:
            P = self.policy_operator
            
        I = np.eye(self.n_states)
        M = self.M_pi_operator(P)
        return (1 - self.gamma) * np.linalg.solve(I - self.gamma * M, nu0)
    
    def kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute KL(p||q) with numerical stability."""

        if self.alpha > 0:
            second_term = -np.sum(self.policy_operator * np.nan_to_num(np.log(self.policy_operator/self.uniform_policy_operator), nan=0.0))
        else:
            second_term = 0.0

        return (1-self.alpha) * np.sum(np.nan_to_num(p * np.log(p / q), nan=0.0)) + self.alpha*second_term

    def compute_gradient_kl(self, nu0: np.ndarray, nu_target: np.ndarray) -> np.ndarray:
        """
        Compute gradient of KL divergence w.r.t. policy.
        
        Args:
            nu0: Initial state distribution
            nu_target: Target state distribution
        
        Returns:
            Gradient of shape (n_states * n_actions, n_states)
        """
        I = np.eye(self.n_states)
        M = self.M_pi_operator(self.policy_operator)
        second_term = np.nan_to_num(1 + self.policy_operator/self.uniform_policy_operator, nan=0.0) if self.alpha > 0 else 0.0  
        
        if self.gradient_type == 'forward':
            r = nu0 / ((I - self.gamma * M) @ nu_target)
            gradient =(1-self.alpha) * (self.gamma * self.T_operator.T @ r @ nu_target.T)/(1-self.gamma)- self.alpha * second_term
            
        elif self.gradient_type == 'reverse':
            nu_pi_over_nu_target = np.clip(
                (1-self.gamma)*np.linalg.solve((I - self.gamma * M), nu0) / nu_target, 
                a_min=1e-10, a_max=None
            )
            log_nu_pi_over_nu_target = np.log(nu_pi_over_nu_target)
            
            gradient = self.gamma * np.linalg.solve(I - self.gamma * M, self.T_operator).T
            gradient = gradient @ (np.ones_like(log_nu_pi_over_nu_target) + log_nu_pi_over_nu_target)
            
            gradient = (1-self.alpha) *(1 - self.gamma) * gradient @ np.linalg.solve(I - self.gamma * M, nu0).T - self.alpha * second_term
        else:
            raise ValueError(f"Unknown gradient type: {self.gradient_type}")
        
        return gradient
    
    def mirror_descent_update(self, gradient: np.ndarray) -> np.ndarray:
        """
        Apply mirror descent update: π_{t+1}(a|s) ∝ π_t(a|s) * exp(-η ∇_{a,s} f)
        
        Args:
            gradient: Policy gradient
        
        Returns:
            Updated policy operator
        """
        policy_3d = self.policy_operator.reshape((self.n_states, self.n_actions, self.n_states))
        new_policy_3d = np.zeros_like(policy_3d)
        gradient_3d = gradient.reshape((self.n_states, self.n_actions, self.n_states))
        
        
        # Normalize the block-diagonal elements (policy probabilities per state)
        for s in range(self.n_states):
            new_policy_3d[s, :, s] = policy_3d[s, :, s] * np.exp(-self.eta * gradient_3d[s, :, s])
            policy_s_actions = new_policy_3d[s, :, s]
            new_policy_3d[s, :, s] = policy_s_actions / (policy_s_actions.sum() + 1e-10)
        
        return new_policy_3d.reshape((self.n_states * self.n_actions, self.n_states))

    # def mirror_descent_update(self, gradient: np.ndarray) -> np.ndarray:
    #     """
    #     Apply mirror descent update: π_{t+1}(a|s) ∝ π_t(a|s) * exp(-η ∇_{a,s} f)
        
    #     Args:
    #         gradient: Policy gradient
        
    #     Returns:
    #         Updated policy operator
    #     """
    #     # Reshape the policy and gradient to 3D
    #     policy_3d = self.policy_operator.reshape((self.n_states, self.n_actions, self.n_states))
    #     gradient_3d = gradient.reshape((self.n_states, self.n_actions, self.n_states))
        
    #     # Apply the mirror descent update for all states at once
    #     exp_grad = np.exp(-self.eta * gradient_3d)
    #     new_policy_3d = policy_3d * exp_grad
        
    #     # Normalize the new policies for each state (along actions dimension)
    #     new_policy_3d /= np.sum(new_policy_3d, axis=1, keepdims=True) + 1e-10
        
    #     return new_policy_3d.reshape((self.n_states * self.n_actions, self.n_states))
    
    def save_policy(self, filepath: str) -> None:
        """
        Save the learned policy operator to a file.
        
        Args:
            filepath: Path where to save the policy (will add .npy extension)
        """
        np.save(filepath, self.policy_operator)
        print(f"Policy operator saved to: {filepath}.npy")
    
    def load_policy(self, filepath: str) -> None:
        """
        Load a policy operator from a file.
        
        Args:
            filepath: Path to the saved policy file
        """
        self.policy_operator = np.load(filepath)
        print(f"Policy operator loaded from: {filepath}")
    
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
            'eta': self.eta,
            'alpha': self.alpha,
            'gradient_type': self.gradient_type,
            'n_updates': len(self.kl_history)
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
    
    def optimize(self, nu0: np.ndarray, nu_target: np.ndarray, 
                 n_updates: int = 1000, verbose: bool = True, 
                 print_every: int = None, save_path_prefix: str = None) -> None:
        """
        Run mirror descent optimization to match target distribution.
        
        Args:
            nu0: Initial state distribution
            nu_target: Target state distribution
            n_updates: Number of optimization iterations
            verbose: Whether to print progress
            print_every: Save and print results every n iterations (None = only at end)
            save_path_prefix: Prefix for saving intermediate visualizations (e.g., '/path/to/results')
        """
        if verbose:
            print("Starting optimization...")
            print("\n" + "="*60)
            print("OPTIMIZATION PARAMETERS")
            print("="*60)
            print(f"Discount factor γ: {self.gamma}")
            print(f"Learning rate η: {self.eta}")
            print(f"Number of updates: {n_updates}")
            print(f"Gradient type: {self.gradient_type}")
            if print_every:
                print(f"Saving results every: {print_every} iterations")
            print("="*60 + "\n")
        
        self.kl_history = []
        self.policy_distance_history = []  # Track L2 distance from uniform policy
        
        for iteration in range(n_updates):
            # Compute KL divergence
            if self.gradient_type == 'reverse':
                nu_pi = self.compute_discounted_occupancy(nu0)
                kl = self.kl_divergence(nu_pi, nu_target)
            else:  # forward
                I = np.eye(self.n_states)
                M = self.M_pi_operator(self.policy_operator)
                kl = self.kl_divergence(nu0, (I - self.gamma * M)/(1-self.gamma) @ nu_target)
            
            self.kl_history.append(kl)
            
            # Compute L2 distance from uniform policy
            policy_distance = np.linalg.norm(self.policy_operator - self.uniform_policy_operator)
            self.policy_distance_history.append(policy_distance)
            
            # Compute gradient and update policy
            gradient = self.compute_gradient_kl(nu0, nu_target)
            self.policy_operator = self.mirror_descent_update(gradient)
            
            # Print progress
            if verbose and iteration % 100 == 0:
                print(f"Iter {iteration:5d}: KL = {kl:.10f}, Policy L2 = {policy_distance:.6f}")
            
            # Save intermediate results
            if print_every and (iteration + 1) % print_every == 0:
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"CHECKPOINT AT ITERATION {iteration + 1}")
                    print(f"{'='*60}")
                    print(f"KL Divergence: {kl:.10f}")
                    print(f"Policy L2 Distance: {policy_distance:.6f}")
                    print(f"{'='*60}\n")
                
                if save_path_prefix:
                    # Compute current distribution
                    nu_current = self.compute_discounted_occupancy(nu0)
                    
                    # Create visualizer and save image
                    visualizer = DistributionVisualizer(self.env, self)
                    save_path = f"{save_path_prefix}_iter{iteration+1:06d}.png"
                    visualizer.plot_results(nu0, nu_target, nu_current, save_path=save_path)
                    
                    # Save training data at checkpoint (overwriting files with same names)
                    # Extract directory from save_path_prefix
                    checkpoint_dir = os.path.dirname(save_path_prefix)
                    data_dir = os.path.join(checkpoint_dir, "trained_model")
                    self.save_training_data(data_dir, verbose=False)
                    
                    if verbose:
                        print(f"Saved visualization to: {save_path}")
                        print(f"Saved training data to: {data_dir}\n")
        
        if verbose:
            print(f"\nOptimization complete!")
            print(f"Final KL: {self.kl_history[-1]:.6f}")
            print(f"Final Policy L2 distance: {self.policy_distance_history[-1]:.6f}")

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
        else:
            policy_matrix = self.policy_operator.reshape((self.n_states, self.n_actions, self.n_states))
        policy_per_state = np.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            policy_per_state[s, :] = policy_matrix[s, :, s]

        return policy_per_state

    def sample_action(self, state: int, uniform_policy: bool = False) -> int:
        """
        Sample an action from the policy for a given state.
        
        Args:
            state: Current state index
            uniform_policy: Whether to use a uniform random policy instead of learned policy
        
        Returns:
            Sampled action index
        """
        policy_per_state = self.get_policy_per_state(uniform_policy=uniform_policy) 
        action_probs = policy_per_state[state]

        # Check stochasticity
        prob_sum = np.sum(action_probs)
        if not np.isclose(prob_sum, 1.0, atol=1e-8):
            print(f"⚠️  State {state}: probabilities sum to {prob_sum:.10f} (deviation: {abs(prob_sum - 1.0):.2e})")
            print(f"   Probabilities: {action_probs}")

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
            next_cell = self.env._step_from(current_cell, action)
            next_state = self.env.state_to_idx[next_cell]
            
            trajectory.append((current_state, action))
            states.append(next_state)
            current_state = next_state
        
        return trajectory, states, actions


class DistributionVisualizer:
    """Visualizer for distribution matching results."""
    
    def __init__(self, env: gym.Env , matcher: DistributionMatcher):
        self.env = env
        self.matcher = matcher
        
        # Grid dimensions for visualization
        self.grid_width = max(cell[0] for cell in env.cells) + 1
        self.grid_height = max(cell[1] for cell in env.cells) + 1
        
        # Action symbols and colors
        self.action_symbols = {0: '↑', 1: '↓', 2: '←', 3: '→'}
        self.action_colors = ['red', 'blue', 'green', 'orange']
        self.action_names = ['up', 'down', 'left', 'right']
    
    def state_dist_to_grid(self, nu: np.ndarray) -> np.ndarray:
        """Convert state distribution vector to 2D grid."""
        grid = np.zeros((self.grid_height, self.grid_width))
        for s_idx in range(self.env.n_states):
            x, y = self.env.idx_to_state[s_idx]
            grid[y, x] = nu[s_idx]
        return grid
    
    def plot_results(self, nu0: np.ndarray, nu_target: np.ndarray, 
                    nu_final: np.ndarray, uniform_policy: bool = False, save_path: str = None):
        """
        Create comprehensive visualization of optimization results.
        
        Args:
            nu0: Initial state distribution
            nu_target: Target state distribution
            nu_final: Final learned distribution
            save_path: Path to save figure (optional)
        """
        fig = self._create_figure_layout()
        self._add_parameter_text(fig)
        
        # Get subplots
        axes = self._get_axes(fig)
        
        # Plot distributions
        self._plot_distribution(axes['init'], nu0, 'Initial Distribution ν₀')
        self._plot_distribution(axes['target'], nu_target, 'Target Distribution ν*')
        self._plot_distribution(axes['final'], nu_final, 'Final Discounted Occupancy')
        
        # Plot policy visualizations
        policy_per_state = self.matcher.get_policy_per_state(uniform_policy=uniform_policy)
        self._plot_policy_arrows(axes['arrows'], policy_per_state)
        self._plot_policy_bars(axes['bars'], policy_per_state)
        
        # Plot KL history and policy distance
        self._plot_kl_history(axes['kl'])
        self._plot_policy_distance(axes['policy_dist'])
        
        # Plot action heatmaps
        self._plot_action_heatmaps(axes['actions'], policy_per_state)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nVisualization saved to: {save_path}")
            plt.close(fig)  # Close figure after saving to prevent display issues
       
    
    def _create_figure_layout(self):
        """Create figure with custom grid layout."""
        return plt.figure(figsize=(20, 15))
    
    def _add_parameter_text(self, fig):
        """Add parameter information to figure."""
        param_text = (f"Parameters:\n"
                     f"γ = {self.matcher.gamma}\n"
                     f"η = {self.matcher.eta}\n"
                     f"Updates = {len(self.matcher.kl_history)}\n"
                     f"Gradient = {self.matcher.gradient_type}\n"
                     f"Alpha = {self.matcher.alpha}")
        fig.text(0.02, 0.98, param_text, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _get_axes(self, fig):
        """Create subplot grid and return dictionary of axes."""
        axes = {}
        # First row: 3 columns
        axes['init'] = plt.subplot2grid((3, 4), (0, 0), colspan=1)
        axes['target'] = plt.subplot2grid((3, 4), (0, 1), colspan=1)
        axes['final'] = plt.subplot2grid((3, 4), (0, 2), colspan=1)
        
        # Second row: 4 columns
        axes['arrows'] = plt.subplot2grid((3, 4), (1, 0), colspan=1)
        axes['bars'] = plt.subplot2grid((3, 4), (1, 1), colspan=1)
        axes['kl'] = plt.subplot2grid((3, 4), (1, 2), colspan=1)
        axes['policy_dist'] = plt.subplot2grid((3, 4), (1, 3), colspan=1)
        
        # Third row: 4 columns for action heatmaps
        axes['actions'] = [
            plt.subplot2grid((3, 4), (2, i), colspan=1)
            for i in range(4)
        ]
        
        return axes
    
    def _plot_distribution(self, ax, nu, title):
        """Plot state distribution heatmap."""
        grid = self.state_dist_to_grid(nu)
        im = ax.imshow(grid, cmap='YlOrRd', interpolation='nearest', vmin=0)#, vmin=0.2, vmax=0.26)
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xticks(np.arange(-0.5, self.grid_width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid_height, 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.colorbar(im, ax=ax)
    
    def _plot_policy_arrows(self, ax, policy_per_state):
        """Plot policy as arrows showing most probable actions."""
        ax.set_xlim(-0.5, self.grid_width - 0.5)
        ax.set_ylim(self.grid_height - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.set_title('Policy Actions (arrows = most probable)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
        
        for s_idx in range(self.env.n_states):
            x, y = self.env.idx_to_state[s_idx]
            
            # Find most probable actions
            max_prob = np.max(policy_per_state[s_idx])
            max_actions = np.where(np.isclose(policy_per_state[s_idx], max_prob, atol=1e-6))[0]
            arrow_text = ''.join([self.action_symbols[a] for a in max_actions])
            
            # Draw cell and arrow
            rect = Rectangle((x - 0.4, y - 0.4), 0.8, 0.8,
                           facecolor='lightgray', edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            ax.text(x, y, arrow_text, ha='center', va='center',
                   fontsize=12, fontweight='bold')
    
    def _plot_policy_bars(self, ax, policy_per_state):
        """Plot policy as mini bar charts in each cell."""
        ax.set_xlim(-0.5, self.grid_width - 0.5)
        ax.set_ylim(self.grid_height - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.set_title('Policy Probabilities per State')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
        
        for s_idx in range(self.env.n_states):
            x, y = self.env.idx_to_state[s_idx]
            
            # Draw background
            rect = Rectangle((x - 0.4, y - 0.4), 0.8, 0.8,
                           facecolor='lightgray', edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            
            # Draw mini bar chart
            probs = policy_per_state[s_idx]
            bar_width = 0.15
            bar_spacing = 0.2
            start_x = x - 1.5 * bar_spacing
            max_bar_height = 0.7
            
            for a_idx in range(self.matcher.n_actions):
                bar_x = start_x + a_idx * bar_spacing
                bar_height = probs[a_idx] * max_bar_height
                
                bar_rect = Rectangle((bar_x - bar_width/2, y + 0.35 - bar_height),
                                    bar_width, bar_height,
                                    facecolor=self.action_colors[a_idx],
                                    edgecolor='black', linewidth=0.3)
                ax.add_patch(bar_rect)
        
        # Add legend
        legend_elements = [Patch(facecolor=self.action_colors[i], 
                                label=self.action_names[i])
                          for i in range(self.matcher.n_actions)]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    def _plot_kl_history(self, ax):
        """Plot KL divergence over iterations."""
        ax.clear()  # Clear any existing data on this axis
        ax.plot(self.matcher.kl_history)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('KL Divergence')
        ax.set_title('KL Divergence vs Iteration')
        ax.grid(True)
    
    def _plot_action_heatmaps(self, axes_list, policy_per_state):
        """Plot heatmaps for each action's probability distribution."""
        for a_idx, ax in enumerate(axes_list):
            grid_action = np.zeros((self.grid_height, self.grid_width))
            for s_idx in range(self.env.n_states):
                x, y = self.env.idx_to_state[s_idx]
                grid_action[y, x] = policy_per_state[s_idx, a_idx]
            
            im = ax.imshow(grid_action, cmap='YlOrRd', interpolation='nearest',
                          vmin=0, vmax=1)
            ax.set_title(f'π({self.action_names[a_idx]}|s)')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_xticks(np.arange(-0.5, self.grid_width, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, self.grid_height, 1), minor=True)
            ax.grid(which='minor', color='white', linestyle='-',
                   linewidth=0.5, alpha=0.5)
            plt.colorbar(im, ax=ax)
    
    def _plot_policy_distance(self, ax):
        """Plot L2 distance between current policy and uniform policy over iterations."""
        ax.clear()  # Clear any existing data on this axis
        if hasattr(self.matcher, 'policy_distance_history'):
            ax.plot(self.matcher.policy_distance_history, color='purple')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('L2 Distance')
            ax.set_title('Policy L2 Distance from Uniform')
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, 'No policy distance data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    def plot_trajectory(self, states: list, actions: list, 
                       start_state: int, save_path: str = None):
        """
        Visualize agent trajectory on the grid.
        
        Args:
            states: List of visited states
            actions: List of taken actions
            start_state: Starting state index
            save_path: Path to save figure (optional)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Trajectory on grid
        self._plot_trajectory_on_grid(ax1, states, start_state)
        
        # Plot 2: Action frequency histogram
        self._plot_action_histogram(ax2, actions)
        
        # Add statistics text
        unique_states = len(set(states))
        coverage = unique_states / self.env.n_states * 100
        stats_text = (f"Unique states visited: {unique_states}/{self.env.n_states}\n"
                     f"Coverage: {coverage:.1f}%")
        fig.text(0.5, 0.02, stats_text, fontsize=12, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nTrajectory visualization saved to: {save_path}")
            plt.close(fig)  # Close figure after saving
        else:
            plt.show()

    def _plot_trajectory_on_grid(self, ax, states, start_state):
        """Plot the trajectory as a path on the grid."""
        ax.set_xlim(-0.5, self.grid_width - 0.5)
        ax.set_ylim(self.grid_height - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.set_title(f'Agent Trajectory ({len(states)-1} steps)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
        
        # Draw all valid cells
        for s_idx in range(self.env.n_states):
            x, y = self.env.idx_to_state[s_idx]
            rect = Rectangle((x - 0.4, y - 0.4), 0.8, 0.8,
                           facecolor='lightgray', edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
        
        # Create visitation heatmap
        visit_counts = np.zeros(self.env.n_states)
        for state in states:
            visit_counts[state] += 1
        
        # Normalize and plot as colors
        max_visits = max(visit_counts)
        for s_idx in range(self.env.n_states):
            if visit_counts[s_idx] > 0:
                x, y = self.env.idx_to_state[s_idx]
                intensity = visit_counts[s_idx] / max_visits
                rect = Rectangle((x - 0.4, y - 0.4), 0.8, 0.8,
                               facecolor=plt.cm.YlOrRd(intensity),
                               edgecolor='black', linewidth=0.5, alpha=0.7)
                ax.add_patch(rect)
                # Add visit count text
                if visit_counts[s_idx] > 0:
                    ax.text(x, y, f'{int(visit_counts[s_idx])}', 
                           ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Draw trajectory path
        path_x = []
        path_y = []
        for state in states:
            x, y = self.env.idx_to_state[state]
            path_x.append(x)
            path_y.append(y)
        
        # Plot path with arrows
        for i in range(len(path_x) - 1):
            ax.annotate('', xy=(path_x[i+1], path_y[i+1]), 
                       xytext=(path_x[i], path_y[i]),
                       arrowprops=dict(arrowstyle='->', color='blue', 
                                     lw=1.5, alpha=0.6))
        
        # Mark start and end
        start_x, start_y = self.env.idx_to_state[start_state]
        end_x, end_y = path_x[-1], path_y[-1]
        
        ax.plot(start_x, start_y, 'go', markersize=15, label='Start', zorder=10)
        ax.plot(end_x, end_y, 'r*', markersize=20, label='End', zorder=10)
        ax.legend(loc='upper right')
    
    def _plot_action_histogram(self, ax, actions):
        """Plot histogram of actions taken."""
        action_counts = np.bincount(actions, minlength=self.matcher.n_actions)
        
        bars = ax.bar(range(self.matcher.n_actions), action_counts, 
                     color=self.action_colors, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Action')
        ax.set_ylabel('Frequency')
        ax.set_title('Action Distribution in Trajectory')
        ax.set_xticks(range(self.matcher.n_actions))
        ax.set_xticklabels(self.action_names)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, count in zip(bars, action_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}', ha='center', va='bottom')
    
    def plot_multiple_trajectories(self, trajectories_data: list, save_path: str = None):
        """
        Plot multiple trajectories for comparison.
        
        Args:
            trajectories_data: List of (states, actions, label) tuples
            save_path: Path to save figure (optional)
        """
        n_trajectories = len(trajectories_data)
        fig, axes = plt.subplots(1, n_trajectories, figsize=(8*n_trajectories, 6))
        
        if n_trajectories == 1:
            axes = [axes]
        
        for idx, (states, actions, label) in enumerate(trajectories_data):
            self._plot_single_trajectory_on_grid(axes[idx], states, label)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nMultiple trajectories visualization saved to: {save_path}")
            plt.close(fig)  # Close figure after saving
        else:
            plt.show()

    def _plot_single_trajectory_on_grid(self, ax, states, label):
        """Plot a single trajectory on a grid."""
        ax.set_xlim(-0.5, self.grid_width - 0.5)
        ax.set_ylim(self.grid_height - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.set_title(f'{label} ({len(states)-1} steps)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
        
        # Draw all valid cells
        for s_idx in range(self.env.n_states):
            x, y = self.env.idx_to_state[s_idx]
            rect = Rectangle((x - 0.4, y - 0.4), 0.8, 0.8,
                           facecolor='lightgray', edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
        
        # Draw trajectory
        path_x = []
        path_y = []
        for state in states:
            x, y = self.env.idx_to_state[state]
            path_x.append(x)
            path_y.append(y)
        
        # Plot path
        ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.6)
        ax.plot(path_x[0], path_y[0], 'go', markersize=12, label='Start')
        ax.plot(path_x[-1], path_y[-1], 'r*', markersize=15, label='End')
        ax.legend()


def create_initial_distribution(env: gym.Env , mode: str = 'uniform') -> np.ndarray:
    """
    Create initial state distribution.
    
    Args:
        env: gym.Env instance
        mode: Distribution mode ('top_left_cell', 'uniform', 'dirac_delta')
            - 'top_left_cell': Single cell at top-left (0, 0)
            - 'uniform': Uniform distribution over all states
            - 'dirac_delta': Dirac delta at the environment's start position (must be set)
    
    Returns:
        Initial distribution as column vector
    """
    nu0 = np.zeros(env.n_states)
    
    if mode == 'top_left_cell':
        # Single cell at top-left (0, 0)
        if (0, 0) in env.state_to_idx:
            nu0[env.state_to_idx[(0, 0)]] = 1.0
        else:
            raise ValueError("Cell (0, 0) is not a valid cell in this environment")
    
    elif mode == 'uniform':
        # Uniform over all states
        nu0 = np.ones(env.n_states) / env.n_states
    
    elif mode == 'dirac_delta':
        # Dirac delta at the environment's start position
        if env.start_position is None:
            raise ValueError(
                "mode='dirac_delta' requires start_position to be set in the environment. "
                "Please specify start_position in the config or use reset(options={'start_position': ...})"
            )
        if env.start_position not in env.state_to_idx:
            raise ValueError(f"Start position {env.start_position} is not a valid cell")
        nu0[env.state_to_idx[env.start_position]] = 1.0
    
    else:
        raise ValueError(
            f"Unknown mode: {mode}. Choose from 'top_left_cell', 'uniform', 'dirac_delta'"
        )
    
    return nu0.reshape((-1, 1))

def compute_target_distribution(env: gym.Env , gamma: float) -> np.ndarray:
    """
    Compute target distribution (uniform over all states).
    
    Args:
        env: gym.Env instance
    
    Returns:
        Target distribution as column vector
    """

    nu_target = np.zeros(env.n_states)
    for i in range(1000):
        for s in range(env.n_states):
            (x, y) = env.idx_to_state[s]

            if (x, y) == (0,0):
                nu_target[s] += gamma ** (0+i)
            elif (x, y) == (1,0):
                nu_target[s] += gamma ** (3+i)
            elif (x, y) == (1,1):
                nu_target[s] += gamma ** (2+i)
            elif (x, y) == (0,1):
                nu_target[s] += gamma ** (1+i)
            else:
                raise ValueError(f"State {(x,y)} not recognized in target distribution computation.")

    # Normalize
    nu_target = nu_target / np.sum(nu_target)
    return nu_target.reshape((-1, 1))
    

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main execution function."""
    # Set random seed
    np.random.seed(cfg.experiment.seed)
    
    # Close any existing figures from previous runs
    plt.close('all')
    
    # Get Hydra output directory
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"\nHydra output directory: {output_dir}\n")
    
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
    print(f"Environment: {cfg.env.name}")
    print(f"Start position: {env.unwrapped.start_position}")
    print(f"Environment created with {env.unwrapped.n_states} states\n")

    # Get parameters from config
    eta = cfg.experiment.eta
    gradient_type = cfg.experiment.gradient_type
    n_updates = cfg.experiment.n_updates
    print_every = cfg.experiment.print_every
    n_rollouts = cfg.experiment.n_rollouts
    horizon = cfg.experiment.horizon
    gamma = cfg.experiment.gamma
    initial_mode = cfg.experiment.initial_mode
    alpha = cfg.experiment.alpha
    
 
    # Create initial and target distributions
    # Call create_initial_distribution AFTER environment reset so start_position is set
    nu0 = create_initial_distribution(env.unwrapped, mode=initial_mode)
    nu_target = np.ones(env.unwrapped.n_states) / env.unwrapped.n_states  # Uniform target
    nu_target = nu_target.reshape((-1, 1))
    # nu_target = compute_target_distribution(env.unwrapped, gamma)
    
    # Create matcher and optimize
    matcher = DistributionMatcher(
        env=env.unwrapped,
        gamma=gamma,
        eta=eta,
        alpha=alpha,
        gradient_type=gradient_type
    )

    # Compute final distribution with uniform policy
    nu_uniform = matcher.compute_discounted_occupancy(nu0, matcher.uniform_policy_operator)
    
    # Visualize results with uniform policy
    visualizer = DistributionVisualizer(env.unwrapped, matcher)
    visualizer.plot_results(
        nu0, nu_target, nu_uniform, uniform_policy=True,
        save_path=os.path.join(output_dir, "results_uniform.png")
    )
    
    # ==================== UNIFORM POLICY ROLLOUTS ====================
    print("\n" + "="*60)
    print("UNIFORM POLICY ROLLOUTS")
    print("="*60)
    
    print(f"\nGenerating {n_rollouts} rollouts with uniform policy (horizon={horizon})")
    
    # Flatten nu0 for probability sampling
    nu0_probs = nu0.flatten()
    nu0_probs = nu0_probs / nu0_probs.sum()  # Ensure it sums to 1
    uniform_rollouts = []
    for i in range(n_rollouts):
        # Sample initial state from nu0 distribution
        np.random.seed(cfg.experiment.seed + i)
        start_state = np.random.choice(env.unwrapped.n_states, p=nu0_probs)

        _, states_i, actions_i = matcher.rollout(start_state, horizon, uniform_policy=True, seed=cfg.experiment.seed+i)
        unique_states = len(set(states_i))
        coverage = unique_states / env.unwrapped.n_states * 100
        uniform_rollouts.append((states_i, actions_i, start_state))
        print(f"  Rollout {i+1}: start_state={start_state}, {unique_states}/{env.unwrapped.n_states} states ({coverage:.1f}% coverage)")
    
    # Plot only first rollout for uniform policy
    print(f"\nSaving visualization for first uniform policy rollout...")
    visualizer.plot_trajectory(
        uniform_rollouts[0][0], uniform_rollouts[0][1], uniform_rollouts[0][2],
        save_path=os.path.join(output_dir, "trajectory_uniform.png")
    )
    
    # ==================== POLICY OPTIMIZATION ====================
    # Optimize policy
    print("\n" + "="*60)
    print("POLICY OPTIMIZATION")
    print("="*60)
    
    # Update save_path_prefix to use output_dir
    results_prefix = os.path.join(output_dir, "optimization")
    matcher.optimize(
        nu0, nu_target, 
        n_updates=n_updates, 
        verbose=True,
        print_every=print_every,
        save_path_prefix=results_prefix
    )
    
    # Compute final distribution
    nu_final = matcher.compute_discounted_occupancy(nu0)
    
    # Visualize optimization results
    visualizer.plot_results(
        nu0, nu_target, nu_final,
        save_path=os.path.join(output_dir, "results_optimized.png")
    )
    
    # ==================== OPTIMIZED POLICY ROLLOUTS ====================
    print("\n" + "="*60)
    print("OPTIMIZED POLICY ROLLOUTS")
    print("="*60)
    
    print(f"\nGenerating {n_rollouts} rollouts with optimized policy (horizon={horizon})")
    
    optimized_rollouts = []
    for i in range(n_rollouts):
        # Sample initial state from nu0 distribution
        np.random.seed(cfg.experiment.seed + i)
        start_state = np.random.choice(env.unwrapped.n_states, p=nu0_probs)
        
        _, states_i, actions_i = matcher.rollout(start_state, horizon, seed=cfg.experiment.seed+i)
        unique_states = len(set(states_i))
        coverage = unique_states / env.unwrapped.n_states * 100
        optimized_rollouts.append((states_i, actions_i, start_state))
        print(f"  Rollout {i+1}: start_state={start_state}, {unique_states}/{env.unwrapped.n_states} states ({coverage:.1f}% coverage)")
    
    matcher.stochasticity_check()
    # Plot only first rollout for optimized policy
    print(f"\nSaving visualization for first optimized policy rollout...")
    visualizer.plot_trajectory(
        optimized_rollouts[0][0], optimized_rollouts[0][1], optimized_rollouts[0][2],
        save_path=os.path.join(output_dir, "trajectory_optimized.png")
    )
    
    # ==================== SAVE TRAINING DATA ====================
    print("\n" + "="*60)
    print("SAVING TRAINING DATA")
    print("="*60)
    
    # Save all training artifacts (will overwrite intermediate checkpoints)
    trained_model_dir = os.path.join(output_dir, "trained_model")
    matcher.save_training_data(trained_model_dir)
    
    # Save initial and target distributions
    np.save(os.path.join(trained_model_dir, "nu0.npy"), nu0)
    print(f"Initial distribution saved to: {os.path.join(trained_model_dir, 'nu0.npy')}")
    np.save(os.path.join(trained_model_dir, "nu_target.npy"), nu_target)
    print(f"Target distribution saved to: {os.path.join(trained_model_dir, 'nu_target.npy')}")
    np.save(os.path.join(trained_model_dir, "nu_final.npy"), nu_final)
    print(f"Final distribution saved to: {os.path.join(trained_model_dir, 'nu_final.npy')}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Number of states: {env.unwrapped.n_states}")
    print(f"Number of actions: {env.unwrapped.action_space.n}")
    print(f"Discount factor γ: {matcher.gamma}")
    print(f"Learning rate η: {matcher.eta}")
    print(f"Gradient type: {matcher.gradient_type}")
    print(f"\nInitial KL: {matcher.kl_history[0]:.6f}")
    print(f"Final KL: {matcher.kl_history[-1]:.6f}")
    print(f"KL reduction: {matcher.kl_history[0] - matcher.kl_history[-1]:.6f}")
    print(f"\nTarget distribution (uniform): mean={nu_target.mean():.6f}, "
          f"std={nu_target.std():.6f}")
    print(f"Final distribution: mean={nu_final.mean():.6f}, std={nu_final.std():.6f}")
    print(f"Distribution L2 distance: {np.linalg.norm(nu_final - nu_target):.6f}")
    
    # Rollout statistics
    uniform_avg_unique = np.mean([len(set(r[0])) for r in uniform_rollouts])
    optimized_avg_unique = np.mean([len(set(r[0])) for r in optimized_rollouts])
    print(f"\nRollout Statistics (horizon={horizon}, n_rollouts={n_rollouts}):")
    print(f"  Uniform Policy:")
    print(f"    Average unique states: {uniform_avg_unique:.1f}/{env.unwrapped.n_states}")
    print(f"    Average coverage: {uniform_avg_unique/env.unwrapped.n_states*100:.1f}%")
    print(f"  Optimized Policy:")
    print(f"    Average unique states: {optimized_avg_unique:.1f}/{env.unwrapped.n_states}")
    print(f"    Average coverage: {optimized_avg_unique/env.unwrapped.n_states*100:.1f}%")
    print(f"    L2 policy distance improvement: {np.linalg.norm(matcher.uniform_policy_operator) - np.linalg.norm(matcher.policy_distance_history[-1]):.6f}")
    print("="*60)
    
    # Clean up all figures at the end
    plt.close('all')
    env.close()


if __name__ == "__main__":
    main()
