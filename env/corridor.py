"""
Corridor Environment - A horizontal corridor with S-shaped curves.

This module provides an environment with a horizontal corridor that 
can have multiple curves, creating an S-shaped or snake-like path.
"""

from typing import Optional, Tuple
import numpy as np
import gymnasium as gym
from env.rooms import BaseRoomEnv


class CorridorEnv(BaseRoomEnv):
    """
    A gridworld environment with a horizontal corridor and S-shaped curves.
    
    The environment consists of horizontal segments connected by vertical 
    segments (curves), creating an S-shaped or snake-like path.
    
    Layout example (length=5, height=3, num_curves=2):
    
      ───────────────┐
       Segment 0     │
                     │ 
                     │ Curve 0
                     │
    ┌────────────────┘
    │    Segment 1
    │             
    │              Curve 1
    └────────────────┐  
        Segment 2    │
                     │
    
    The agent starts at one end and must navigate to the goal at the other end.
    
    Args:
        length: Length of each horizontal segment (default: 5)
        height: Height of each vertical curve segment (default: 3)
        num_curves: Number of curves/turns in the corridor (default: 2)
        corridor_width: Width of the corridor (default: 1)
        goal_position: Goal position as (x, y) tuple or state index (optional, random if None)
        start_position: Start position as (x, y) tuple or state index (optional, random if None)
        max_steps: Maximum steps per episode
        render_mode: Rendering mode ("human", "ansi", "rgb_array")
        show_coordinates: Whether to show coordinates in rendering
        lava: Whether walls are lava (agent dies when hitting them)
        two_actions: Force two-action mode (left/right only). Auto-enabled when num_curves=0 and height=1
    """
    
    def __init__(
        self,
        length: int = 5,
        height: int = 3,
        num_curves: int = 2,
        corridor_width: int = 1,
        goal_position: Optional[Tuple[int, int]] = None,
        start_position: Optional[Tuple[int, int]] = None,
        max_steps: int = 300,
        render_mode: Optional[str] = None,
        show_coordinates: bool = False,
        lava: bool = False,
        two_actions: Optional[bool] = None
    ):
        if length < 2:
            raise ValueError("length must be at least 2")
        if height < 1:
            raise ValueError("height must be at least 1")
        if num_curves < 0:
            raise ValueError("num_curves must be non-negative")
        if corridor_width < 1:
            raise ValueError("corridor_width must be at least 1")
        
        self.length = length
        self.height = height
        self.num_curves = num_curves
        self.corridor_width = corridor_width
        
        # Auto-enable two_actions mode for straight horizontal corridor
        if two_actions is None:
            self.two_actions = (num_curves == 0 and height == 1)
        else:
            self.two_actions = two_actions
            
        # Validate two_actions compatibility
        if self.two_actions and (num_curves > 0 or height > 1):
            raise ValueError("two_actions mode requires num_curves=0 and height=1")
        
        # Calculate total segments (horizontal segments = num_curves + 1)
        self.num_segments = num_curves + 1
        
        super().__init__(
            goal_position=goal_position,
            start_position=start_position,
            max_steps=max_steps,
            render_mode=render_mode,
            show_coordinates=show_coordinates,
            lava=lava
        )
        
        # Override action space if in two_actions mode
        if self.two_actions:
            self.action_space = gym.spaces.Discrete(2)
            # 0=left, 1=right
            self._action_to_direction = {
                0: np.array([-1, 0]),  # left
                1: np.array([1, 0]),   # right
            }
    
    def _build_cells(self):
        """Build cells for the S-shaped corridor."""
        # Track current position for building
        current_y = 0
        
        for segment_idx in range(self.num_segments):
            # Determine direction: even segments go left-to-right, odd go right-to-left
            # But we'll build all segments left-to-right and just offset them
            goes_right = (segment_idx % 2 == 0)
            
            # Calculate x offset for this segment
            # Odd segments are offset to the right by (length - corridor_width)
            if goes_right:
                x_start = 0
            else:
                x_start = 0  # All segments start at same x, but are at different y levels
            
            # Build horizontal segment
            for x in range(x_start, x_start + self.length):
                for w in range(self.corridor_width):
                    self._add_cell((x, current_y + w))
            
            # Build vertical curve (if not the last segment)
            if segment_idx < self.num_curves:
                # Determine which end the curve is on
                if goes_right:
                    # Curve is on the right end
                    curve_x = x_start + self.length - 1
                else:
                    # Curve is on the left end
                    curve_x = x_start
                
                # Build vertical segment for the curve
                for h in range(1, self.height + 1):
                    for w in range(self.corridor_width):
                        self._add_cell((curve_x + w, current_y + self.corridor_width - 1 + h))
                
                # Move to next horizontal level
                current_y += self.corridor_width + self.height - 1
    
    def _get_default_goal(self) -> Tuple[int, int]:
        """Default goal: end of the last segment."""
        # Calculate the position at the end of the corridor
        last_segment_idx = self.num_segments - 1
        goes_right = (last_segment_idx % 2 == 0)
        
        # Calculate y position of last segment
        y_pos = last_segment_idx * (self.corridor_width + self.height - 1)
        
        if goes_right:
            # End is on the right
            x_pos = self.length - 1
        else:
            # End is on the left
            x_pos = 0
        
        return (x_pos, y_pos)
    
    def get_segment_info(self) -> list[dict]:
        """Return information about each segment of the corridor."""
        segments = []
        current_y = 0
        
        for segment_idx in range(self.num_segments):
            goes_right = (segment_idx % 2 == 0)
            
            segment_info = {
                'index': segment_idx,
                'y_start': current_y,
                'y_end': current_y + self.corridor_width - 1,
                'x_start': 0,
                'x_end': self.length - 1,
                'direction': 'right' if goes_right else 'left'
            }
            segments.append(segment_info)
            
            if segment_idx < self.num_curves:
                current_y += self.corridor_width + self.height - 1
        
        return segments
    
    def get_curve_positions(self) -> list[Tuple[int, int]]:
        """Return the center position of each curve."""
        positions = []
        current_y = 0
        
        for segment_idx in range(self.num_curves):
            goes_right = (segment_idx % 2 == 0)
            
            if goes_right:
                curve_x = self.length - 1
            else:
                curve_x = 0
            
            # Center of the curve
            curve_center_y = current_y + self.corridor_width + self.height // 2
            positions.append((curve_x, curve_center_y))
            
            current_y += self.corridor_width + self.height - 1
        
        return positions


# Register the environment
gym.register(
    id="Corridor-v0",
    entry_point="env.corridor:CorridorEnv",
    max_episode_steps=300,
)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("Testing CorridorEnv - S-Shaped Corridor")
    print("=" * 60)
    
    # Test 0: Two-actions mode (straight horizontal corridor)
    print("\n[Test 0] Two-actions mode (straight corridor, left/right only)")
    print("-" * 60)
    env = CorridorEnv(
        length=10,
        height=1,
        num_curves=0,
        corridor_width=1,
        render_mode="human",
        show_coordinates=True
    )
    
    obs, info = env.reset(seed=42)
    print(f"Action space: {env.action_space}")
    print(f"Action space size: {env.action_space.n}")
    print(f"Initial state: {info['agent_position']}")
    print(f"Goal state: {env.goal_position}")
    print(f"Total states: {env.n_states}")
    print("\nInitial render:")
    env.render()
    
    # Test movement with only left/right
    print("\nTesting left/right movement:")
    print("Taking 5 right actions (action=1):")
    for i in range(5):
        obs, reward, terminated, truncated, info = env.step(1)  # right
        print(f"  Step {i+1}: position = {info['agent_position']}")
    
    print("\nTaking 3 left actions (action=0):")
    for i in range(3):
        obs, reward, terminated, truncated, info = env.step(0)  # left
        print(f"  Step {i+1}: position = {info['agent_position']}")
    
    env.render()
    env.close()
    
    # Test 1: Simple corridor with no curves (straight line)
    print("\n[Test 1] Straight corridor (no curves)")
    print("-" * 60)
    env = CorridorEnv(
        length=10,
        height=3,
        num_curves=0,
        corridor_width=1,
        render_mode="human",
        show_coordinates=True
    )
    
    obs, info = env.reset(seed=42)
    print(f"Initial state: {info['agent_position']}")
    print(f"Goal state: {env.goal_position}")
    print(f"Total states: {env.n_states}")
    print("\nInitial render:")
    env.render()
    env.close()
    
    # Test 2: Single curve (L-shaped)
    print("\n[Test 2] Single curve (L-shaped)")
    print("-" * 60)
    env = CorridorEnv(
        length=5,
        height=3,
        num_curves=1,
        corridor_width=1,
        render_mode="human",
        show_coordinates=True
    )
    
    obs, info = env.reset(seed=42)
    print(f"Initial state: {info['agent_position']}")
    print(f"Goal state: {env.goal_position}")
    print(f"Total states: {env.n_states}")
    print(f"Segment info: {env.get_segment_info()}")
    print("\nInitial render:")
    env.render()
    env.close()
    
    # Test 3: Double curve (S-shaped)
    print("\n[Test 3] Double curve (S-shaped)")
    print("-" * 60)
    env = CorridorEnv(
        length=6,
        height=2,
        num_curves=2,
        corridor_width=1,
        render_mode="human",
        show_coordinates=True
    )
    
    obs, info = env.reset(seed=42)
    print(f"Initial state: {info['agent_position']}")
    print(f"Goal state: {env.goal_position}")
    print(f"Total states: {env.n_states}")
    print(f"Curve positions: {env.get_curve_positions()}")
    print("\nInitial render:")
    env.render()
    env.close()
    
    # Test 4: Multiple curves (snake-like)
    print("\n[Test 4] Multiple curves (snake-like)")
    print("-" * 60)
    env = CorridorEnv(
        length=8,
        height=2,
        num_curves=4,
        corridor_width=1,
        render_mode="human",
        show_coordinates=True
    )
    
    obs, info = env.reset(seed=42)
    print(f"Initial state: {info['agent_position']}")
    print(f"Goal state: {env.goal_position}")
    print(f"Total states: {env.n_states}")
    print("\nInitial render:")
    env.render()
    
    # Take some steps
    print("\nTaking 20 random steps:")
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            print(f"Goal reached at step {step + 1}!")
            break
        if truncated:
            print(f"Truncated at step {step + 1}")
            break
    
    env.render()
    env.close()
    
    # Test 5: Wider corridor
    print("\n[Test 5] Wider corridor (width=2)")
    print("-" * 60)
    env = CorridorEnv(
        length=5,
        height=3,
        num_curves=2,
        corridor_width=2,
        render_mode="human",
        show_coordinates=True
    )
    
    obs, info = env.reset(seed=42)
    print(f"Initial state: {info['agent_position']}")
    print(f"Goal state: {env.goal_position}")
    print(f"Total states: {env.n_states}")
    print("\nInitial render (wider corridor):")
    env.render()
    env.close()
    
    # Test 6: RGB rendering with matplotlib
    print("\n[Test 6] RGB rendering with matplotlib")
    print("-" * 60)
    env = CorridorEnv(
        length=6,
        height=2,
        num_curves=3,
        corridor_width=1,
        render_mode="rgb_array",
        show_coordinates=True
    )
    
    obs, info = env.reset(seed=789)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Initial state
    rgb_array = env.render()
    axes[0].imshow(rgb_array)
    axes[0].set_title(f"Initial State\nAgent: {info['agent_position']}")
    axes[0].axis('off')
    
    # After some steps
    for _ in range(15):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    rgb_array = env.render()
    axes[1].imshow(rgb_array)
    axes[1].set_title(f"After Steps\nAgent: {info['agent_position']}")
    axes[1].axis('off')
    
    # Different configuration
    obs, info = env.reset(seed=999)
    rgb_array = env.render()
    axes[2].imshow(rgb_array)
    axes[2].set_title(f"New Episode\nAgent: {info['agent_position']}")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('corridor_test.png', dpi=150, bbox_inches='tight')
    print("Saved RGB rendering to 'corridor_test.png'")
    plt.show()
    
    env.close()
    
    # Test 7: With lava
    print("\n[Test 7] Corridor with lava")
    print("-" * 60)
    env = CorridorEnv(
        length=5,
        height=2,
        num_curves=2,
        corridor_width=1,
        render_mode="rgb_array",
        show_coordinates=True,
        lava=True
    )
    
    obs, info = env.reset(seed=111)
    print(f"Initial state: {info['agent_position']}")
    print(f"Goal state: {env.goal_position}")
    print("Lava mode enabled - walls are deadly!")
    
    img = env.render()
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title("Corridor with Lava (red background)")
    plt.axis('off')
    plt.savefig('corridor_lava_test.png', dpi=150, bbox_inches='tight')
    print("Saved lava rendering to 'corridor_lava_test.png'")
    
    env.close()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("Key features demonstrated:")
    print("  - Straight corridor (no curves)")
    print("  - L-shaped corridor (1 curve)")
    print("  - S-shaped corridor (2 curves)")
    print("  - Snake-like corridor (multiple curves)")
    print("  - Configurable corridor width")
    print("  - Configurable segment length and curve height")
    print("  - Lava mode support")
    print("=" * 60)
