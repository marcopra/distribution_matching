"""
Example script showing how to load and use a trained policy.
"""

import numpy as np
import gymnasium as gym
import env.rooms
from distribution_matching import DistributionMatcher, DistributionVisualizer, create_initial_distribution
import sys
import os

def load_and_use_policy(model_dir: str):
    """
    Load a trained policy and demonstrate its usage.
    
    Args:
        model_dir: Directory containing the trained model files
    """
    print(f"Loading training data from: {model_dir}")
    
    # Load all training data
    data = DistributionMatcher.load_training_data(model_dir)
    
    # Print metadata
    print("\n" + "="*60)
    print("LOADED MODEL METADATA")
    print("="*60)
    for key, value in data['metadata'].items():
        print(f"{key}: {value}")
    print("="*60 + "\n")
    
    # Access the loaded components
    policy_operator = data['policy_operator']
    transition_matrix = data['transition_matrix']
    uniform_policy = data['uniform_policy_operator']
    
    print(f"Policy operator shape: {policy_operator.shape}")
    print(f"Transition matrix shape: {transition_matrix.shape}")
    
    # Recreate environment with same parameters as training
    # Note: You may need to adjust these based on your actual environment setup
    env = gym.make("FourRooms-v0")
    env.reset()
    
    # Recreate matcher with loaded policy
    matcher = DistributionMatcher(
        env=env.unwrapped,
        gamma=data['metadata']['gamma'],
        eta=data['metadata']['eta'],
        alpha=data['metadata']['alpha'],
        gradient_type=data['metadata']['gradient_type']
    )
    
    # Set the loaded policy
    matcher.policy_operator = policy_operator
    matcher.uniform_policy_operator = uniform_policy
    matcher.T_operator = transition_matrix
    
    # Load distributions if available
    nu0 = np.load(os.path.join(model_dir, "nu0.npy")) if os.path.exists(os.path.join(model_dir, "nu0.npy")) else create_initial_distribution(env.unwrapped, mode='top_left_cell')
    nu_target = np.load(os.path.join(model_dir, "nu_target.npy")) if os.path.exists(os.path.join(model_dir, "nu_target.npy")) else np.ones(env.unwrapped.n_states).reshape(-1, 1) / env.unwrapped.n_states
    nu_final = np.load(os.path.join(model_dir, "nu_final.npy")) if os.path.exists(os.path.join(model_dir, "nu_final.npy")) else matcher.compute_discounted_occupancy(nu0)
    
    # Create visualizer
    visualizer = DistributionVisualizer(env.unwrapped, matcher)
    
    # Plot results with loaded policy
    print("\nGenerating visualization plots...")
    output_dir = os.path.dirname(model_dir)
    
    # Plot optimized policy results
    visualizer.plot_results(
        nu0, nu_target, nu_final,
        save_path=os.path.join(output_dir, "loaded_policy_results.png")
    )
    print(f"Saved loaded policy visualization to: {os.path.join(output_dir, 'loaded_policy_results.png')}")
    
    # Plot uniform policy for comparison
    nu_uniform = matcher.compute_discounted_occupancy(nu0, matcher.uniform_policy_operator)
    visualizer.plot_results(
        nu0, nu_target, nu_uniform, uniform_policy=True,
        save_path=os.path.join(output_dir, "loaded_uniform_results.png")
    )
    print(f"Saved uniform policy visualization to: {os.path.join(output_dir, 'loaded_uniform_results.png')}")
    
    # Run a sample rollout with loaded policy
    print("\n" + "="*60)
    print("SAMPLE ROLLOUT WITH LOADED POLICY")
    print("="*60)
    
    nu0_probs = nu0.flatten()
    nu0_probs = nu0_probs / nu0_probs.sum()
    start_state = np.random.choice(env.unwrapped.n_states, p=nu0_probs)
    
    _, states, actions = matcher.rollout(start_state, horizon=100, seed=42)
    unique_states = len(set(states))
    coverage = unique_states / env.unwrapped.n_states * 100
    
    print(f"Start state: {start_state}")
    print(f"Unique states visited: {unique_states}/{env.unwrapped.n_states} ({coverage:.1f}% coverage)")
    
    visualizer.plot_trajectory(
        states, actions, start_state,
        save_path=os.path.join(output_dir, "loaded_policy_trajectory.png")
    )
    print(f"Saved trajectory visualization to: {os.path.join(output_dir, 'loaded_policy_trajectory.png')}")
    
    # Check policy stochasticity
    matcher.stochasticity_check()
    
    # Print distribution statistics
    print("\n" + "="*60)
    print("DISTRIBUTION STATISTICS")
    print("="*60)
    kl = matcher.kl_divergence(nu_final.flatten(), nu_target.flatten())
    print(f"KL divergence: {kl:.6f}")
    print(f"L2 distance from target: {np.linalg.norm(nu_final - nu_target):.6f}")
    print(f"Target distribution: mean={nu_target.mean():.6f}, std={nu_target.std():.6f}")
    print(f"Final distribution: mean={nu_final.mean():.6f}, std={nu_final.std():.6f}")
    print("="*60 + "\n")
    
    env.close()
    return data

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load_trained_policy.py <model_directory>")
        print("Example: python load_trained_policy.py outputs/2024-01-01/12-00-00/trained_model")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    data = load_and_use_policy(model_dir)
