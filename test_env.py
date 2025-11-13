"""
Test script for custom environments with Hydra configuration.
"""

import os
import hydra
from omegaconf import DictConfig
import gymnasium as gym
import numpy as np
from PIL import Image
import env  # Import to register environments


@hydra.main(version_base=None, config_path="configs", config_name="config_test")
def test_environments(cfg: DictConfig):
    """Test the custom environments."""
    
    # Create save directory
    os.makedirs(cfg.test.save_dir, exist_ok=True)
    
    print(f"Testing environment: {cfg.env.name}")
    print(f"Configuration: {cfg.env}")
    print("-" * 50)
    
    # Create environment based on config
    env_kwargs = {
        'max_steps': cfg.env.max_steps,
        'render_mode': cfg.env.render_mode,
        'show_coordinates': cfg.env.get('show_coordinates', False),
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
    
    # Create environment
    test_env = gym.make(cfg.env.name, **env_kwargs)
    
    # Run test episodes
    np.random.seed(cfg.test.random_seed)
    
    for episode in range(cfg.test.num_episodes):
        print(f"\nEpisode {episode + 1}/{cfg.test.num_episodes}")
        
        obs, info = test_env.reset(seed=cfg.test.random_seed + episode)
        print(f"  Initial state: {info['agent_position']}, State index: {obs}")
        print(f"  Goal position: {test_env.unwrapped.goal_position}")
        print(f"  Start position: {test_env.unwrapped.start_position}")
        
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0
        
        # Save initial frame
        if cfg.env.render_mode == "rgb_array":
            frame = test_env.render()
            img = Image.fromarray(frame)
            img.save(os.path.join(cfg.test.save_dir, f"{cfg.env.name}_ep{episode+1}_step{step_count:03d}.png"))
        
        # Run episode
        while not (terminated or truncated):
            action = test_env.action_space.sample()
            obs, reward, terminated, truncated, info = test_env.step(action)
            total_reward += reward
            step_count += 1
            
            # Save frame every 5 steps or at termination
            if cfg.env.render_mode == "rgb_array" and (step_count % 5 == 0 or terminated):
                frame = test_env.render()
                img = Image.fromarray(frame)
                img.save(os.path.join(cfg.test.save_dir, f"{cfg.env.name}_ep{episode+1}_step{step_count:03d}.png"))
        
        print(f"  Finished in {step_count} steps")
        print(f"  Total reward: {total_reward:.4f}")
        print(f"  Goal reached: {terminated}")
        print(f"  Final position: {info['agent_position']}")
    
    test_env.close()
    print(f"\nTest completed! Results saved to {cfg.test.save_dir}")
    
    # Print environment statistics
    print(f"\nEnvironment Statistics:")
    print(f"  Total states: {test_env.unwrapped.n_states}")
    print(f"  Goal randomization: {'Yes' if env_kwargs['goal_position'] is None else 'No'}")
    print(f"  Start randomization: {'Yes' if env_kwargs['start_position'] is None else 'No'}")


if __name__ == "__main__":
    test_environments()
