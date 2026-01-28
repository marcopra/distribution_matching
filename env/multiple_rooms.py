"""
Multiple Rooms Environment - A corridor with multiple rooms attached.

This module provides an environment with a main horizontal corridor
and multiple rooms attached below it via single-cell connectors.
"""

from typing import Optional, Tuple, Dict
import numpy as np
import gymnasium as gym
try:
    from env.rooms import BaseRoomEnv
except:
    from rooms import BaseRoomEnv

class MultipleRoomsEnv(BaseRoomEnv):
    """
    A gridworld environment with a horizontal corridor and multiple isolated rooms.
    
    The environment consists of:
    - A main horizontal corridor (width calculated from number of rooms, height configurable)
    - Multiple square rooms below the corridor (NOT connected to each other)
    - Each room connected to the corridor by a vertical connector of configurable length
    - Walls (1 cell wide) separate the rooms from each other
    
    Layout example (3 rooms, corridor_height=1, connector_length=1):
    ┌─────────────────────┐
    │     Corridor        │
    └─┬───┬─┬───┬─┬───┬───┘
      │   │ │   │ │   │
    ┌─┴─┐ │ ┌─┴─┐ │ ┌─┴─┐
    │ 0 │ │ │ 1 │ │ │ 2 │
    └───┘ │ └───┘ │ └───┘
          W       W
    
    To go from room 0 to room 2, the agent must:
    1. Go up through connector to main corridor
    2. Move along the corridor
    3. Go down through another connector to room 2
    
    Args:
        num_rooms: Number of rooms below the corridor (default: 3)
        room_size: Size of each square room (default: 5)
        corridor_height: Height of the main corridor in cells (default: 1)
        connector_length: Length of vertical connectors between corridor and rooms (default: 1)
        connector_position: X-coordinate within each room where connector attaches (default: middle)
        goal_position: Goal position as (x, y) tuple or state index (optional, random if None)
        start_position: Start position as (x, y) tuple or state index (optional, random if None)
        max_steps: Maximum steps per episode
        render_mode: Rendering mode ("human", "ansi", "rgb_array")
        show_coordinates: Whether to show coordinates in rendering
        lava: Whether walls are lava (agent dies when hitting them)
    """
    
    def __init__(
        self,
        num_rooms: int,
        room_size: int,
        corridor_height: int,
        connector_length: int,
        connector_position: Optional[int],
        goal_position: Optional[Tuple[int, int]] = None,
        start_position: Optional[Tuple[int, int]] = None,
        max_steps: int = 300,
        render_mode: Optional[str] = None,
        show_coordinates: bool = False,
        lava: bool = False, 
        dense_reward: bool = False,
    ):
        if num_rooms < 1:
            raise ValueError("num_rooms must be at least 1")
        if room_size < 2:
            raise ValueError("room_size must be at least 2")
        if corridor_height < 1:
            raise ValueError("corridor_height must be at least 1")
        if connector_length < 1:
            raise ValueError("connector_length must be at least 1")
        
        self.num_rooms = num_rooms
        self.room_size = room_size
        self.corridor_height = corridor_height
        self.connector_length = connector_length
        self.wall_width = 1  # Width of walls between rooms
        
        # Default connector position is in the middle of the room
        if connector_position is None:
            self.connector_position = room_size // 2
        else:
            if connector_position >= room_size:
                raise ValueError(f"connector_position ({connector_position}) must be less than room_size ({room_size})")
            self.connector_position = connector_position
        
        # Calculate corridor length: room_size per room + wall_width between rooms
        # For n rooms, we have (n-1) walls between them
        self.corridor_length = num_rooms * room_size + (num_rooms - 1) * self.wall_width
        
        super().__init__(
            goal_position=goal_position,
            start_position=start_position,
            max_steps=max_steps,
            render_mode=render_mode,
            show_coordinates=show_coordinates,
            lava=lava,
            dense_reward=dense_reward,
        )
    
    def _build_cells(self):
        """Build cells for corridor and multiple isolated rooms with walls between them."""
        # Main horizontal corridor (at the top)
        for x in range(self.corridor_length):
            for y in range(self.corridor_height):
                self._add_cell((x, y))
        
        # For each room
        for room_idx in range(self.num_rooms):
            # Calculate room's starting x position accounting for walls
            # Each room starts at: room_idx * (room_size + wall_width)
            room_start_x = room_idx * (self.room_size + self.wall_width)
            
            # Connector x position (at connector_position within the room)
            connector_x = room_start_x + self.connector_position
            
            # Vertical connector from corridor to room (configurable length)
            for i in range(self.connector_length):
                connector_y = self.corridor_height + i
                self._add_cell((connector_x, connector_y))
            
            # Room cells (below the connector)
            # Rooms are NOT connected to each other horizontally - walls in between
            room_start_y = self.corridor_height + self.connector_length
            for x in range(room_start_x, room_start_x + self.room_size):
                for y in range(room_start_y, room_start_y + self.room_size):
                    self._add_cell((x, y))
    
    def _get_default_goal(self) -> Tuple[int, int]:
        """Default goal: center of the last room."""
        last_room_idx = self.num_rooms - 1
        room_start_x = last_room_idx * (self.room_size + self.wall_width)
        room_center_x = room_start_x + self.room_size // 2
        room_start_y = self.corridor_height + self.connector_length
        room_center_y = room_start_y + self.room_size // 2
        return (room_center_x, room_center_y)
    
    def get_room_centers(self) -> list[Tuple[int, int]]:
        """Return the center position of each room."""
        centers = []
        room_start_y = self.corridor_height + self.connector_length
        for room_idx in range(self.num_rooms):
            room_start_x = room_idx * (self.room_size + self.wall_width)
            center_x = room_start_x + self.room_size // 2
            center_y = room_start_y + self.room_size // 2
            centers.append((center_x, center_y))
        return centers
    
    def get_connector_positions(self) -> list[Tuple[int, int]]:
        """Return the position where each connector meets the main corridor."""
        positions = []
        for room_idx in range(self.num_rooms):
            room_start_x = room_idx * (self.room_size + self.wall_width)
            connector_x = room_start_x + self.connector_position
            connector_y = self.corridor_height - 1  # Last cell of corridor before connector
            positions.append((connector_x, connector_y))
        return positions


# Register the environment
gym.register(
    id="MultipleRooms-v0",
    entry_point="env.multiple_rooms:MultipleRoomsEnv",
    max_episode_steps=300,
)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from rooms import BaseRoomEnv

    
    print("=" * 60)
    print("Testing MultipleRoomsEnv - Isolated Rooms")
    print("=" * 60)
    
    # Test 1: Basic environment with 3 rooms
    print("\n[Test 1] 3 rooms, corridor_height=1, connector_length=1")
    print("-" * 60)
    env = MultipleRoomsEnv(
        num_rooms=3,
        room_size=5,
        corridor_height=1,
        connector_position=2,
        connector_length=1,
        render_mode="human",
        show_coordinates=True, goal_position=(1,0), start_position=(0,0)
    )
    
    obs, info = env.reset(seed=42)
    print(f"Initial state: {info['agent_position']}")
    print(f"Goal state: {env.goal_position}")
    print(f"Total states: {env.n_states}")
    print(f"Room centers: {env.get_room_centers()}")
    print(f"Connector positions: {env.get_connector_positions()}")
    print("\nInitial render:")
    env.render()
    
    # Take a few random steps
    print("\nTaking 10 random steps:")
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step + 1}: action={action}, position={info['agent_position']}, reward={reward:.2f}")
        if terminated or truncated:
            break
    
    env.render()
    env.close()
    
    # Test 2: Longer connectors
    print("\n[Test 2] 4 rooms, corridor_height=2, connector_length=3")
    print("-" * 60)
    env = MultipleRoomsEnv(
        num_rooms=4,
        room_size=4,
        corridor_height=2,
        connector_position=2,
        connector_length=3,
        render_mode="human",
        show_coordinates=True
    )
    
    obs, info = env.reset(seed=123)
    print(f"Initial state: {info['agent_position']}")
    print(f"Goal state: {env.goal_position}")
    print(f"Total states: {env.n_states}")
    print("\nInitial render (note the longer connectors):")
    env.render()
    env.close()
    
    # Test 3: Test room isolation - agent in room 0 trying to reach room 2
    print("\n[Test 3] Testing room isolation")
    print("-" * 60)
    env = MultipleRoomsEnv(
        num_rooms=3,
        room_size=4,
        corridor_height=1,
        connector_position=2,
        connector_length=2,
        render_mode="human",
        show_coordinates=True
    )
    
    # Start in center of room 0
    room_centers = env.get_room_centers()
    start_pos = room_centers[0]
    goal_pos = room_centers[2]
    
    obs, info = env.reset(options={"start_position": start_pos, "goal_position": goal_pos})
    print(f"Starting in room 0: {info['agent_position']}")
    print(f"Goal in room 2: {env.goal_position}")
    print("\nTo reach room 2, the agent must go up to corridor, move right, then down")
    env.render()
    
    # Manually navigate: up through connector, right along corridor, down to room 2
    print("\nNavigating from room 0 to room 2:")
    actions_sequence = [0] * 3 + [3] * 8 + [1] * 3  # up, right, down
    for i, action in enumerate(actions_sequence):
        obs, reward, terminated, truncated, info = env.step(action)
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        print(f"Step {i+1}: {action_names[action]} -> {info['agent_position']}")
        if terminated:
            print(f"Goal reached! Total steps: {i+1}")
            break
        if truncated:
            break
    
    env.render()
    env.close()
    
    # Test 4: RGB rendering with matplotlib
    print("\n[Test 4] RGB rendering with matplotlib")
    print("-" * 60)
    env = MultipleRoomsEnv(
        num_rooms=5,
        room_size=2,
        corridor_height=1,
        connector_position=1,
        connector_length=2,
        render_mode="rgb_array",
        show_coordinates=True,
    )
    
    obs, info = env.reset(seed=789)
    
    # Create figure with multiple states
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Initial state
    rgb_array = env.render()
    axes[0].imshow(rgb_array)
    axes[0].set_title(f"Initial State\nAgent: {info['agent_position']}\n5 Isolated Rooms")
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
    
    # Reset to show different configuration
    obs, info = env.reset(seed=999)
    rgb_array = env.render()
    axes[2].imshow(rgb_array)
    axes[2].set_title(f"New Episode\nAgent: {info['agent_position']}")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('multiple_rooms_test.png', dpi=150, bbox_inches='tight')
    print("Saved RGB rendering to 'multiple_rooms_test.png'")
    plt.show()
    
    env.close()
    
    # Test 5: Different connector positions
    print("\n[Test 5] Custom connector positions")
    print("-" * 60)
    env = MultipleRoomsEnv(
        num_rooms=3,
        room_size=5,
        corridor_height=1,
        connector_length=1,
        connector_position=1,  # Connector at position 1 instead of middle
        render_mode="human",
        show_coordinates=True
    )
    
    obs, info = env.reset(seed=555)
    print(f"Connector position: {env.connector_position} (left side of rooms)")
    print(f"Connector positions: {env.get_connector_positions()}")
    env.render()
    env.close()
    
    # Test 6: Test with lava
    print("\n[Test 6] 3 rooms with lava enabled")
    print("-" * 60)
    env = MultipleRoomsEnv(
        num_rooms=3,
        room_size=4,
        corridor_height=1,
        connector_position=2,
        connector_length=1,
        render_mode="rgb_array",
        show_coordinates=True,
        lava=True
    )
    
    obs, info = env.reset(seed=111)
    print(f"Initial state: {info['agent_position']}")
    print(f"Goal state: {env.goal_position}")
    print("\nInitial render (walls are lava - red background):")
    img = env.render()
    # save image
    print("\nNote: Rooms are isolated. Agent cannot move directly between rooms.")
    print("      Must use the main corridor to navigate between rooms.")
    env.close()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("Key features demonstrated:")
    print("  - Rooms are isolated (walls between them)")
    print("  - Must use main corridor to move between rooms")
    print("  - Configurable connector length")
    print("  - Configurable corridor height")
    print("  - Configurable connector position within rooms")
    print("=" * 60)
