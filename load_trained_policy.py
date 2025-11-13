"""
Example script showing how to load and use a trained policy.
"""

import numpy as np
import gymnasium as gym
from distribution_matching import DistributionMatcher
import sys

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
    
    
    return data

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load_trained_policy.py <model_directory>")
        print("Example: python load_trained_policy.py outputs/2024-01-01/12-00-00/trained_model")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    data = load_and_use_policy(model_dir)
