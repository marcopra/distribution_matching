"""
Two Rooms Environment - A Gymnasium custom environment.

This environment consists of two rectangular rooms connected by a corridor.
The agent can navigate between the rooms using discrete actions (up, down, left, right).
"""

from typing import Optional, Tuple, Dict
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SingleRoomEnv(gym.Env):
    """
    A simple gridworld environment with a single square room.
    
    The environment consists of a single square room of size room_size x room_size.
    
    Args:
        room_size: Size of the square room (default: 5)
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}
    
    def __init__(
        self, 
        room_size: int = 5,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        # Environment parameters
        self.room_size = room_size
        self.render_mode = render_mode
        
        # Build valid cells (states)
        self.cells = []
        self.state_to_idx = {}
        self.idx_to_state = {}
        
        # Single room
        for x in range(room_size):
            for y in range(room_size):
                idx = len(self.cells)
                self.cells.append((x, y))
                self.state_to_idx[(x, y)] = idx
                self.idx_to_state[idx] = (x, y)
        
        self.n_states = len(self.cells)
        
        # Define observation space: discrete state index
        self.observation_space = spaces.Discrete(self.n_states)
        
        # Define action space: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)
        
        # Action mapping
        self._action_to_direction = {
            0: np.array([0, -1]),  # up
            1: np.array([0, 1]),   # down
            2: np.array([-1, 0]),  # left
            3: np.array([1, 0]),   # right
        }
        
        # Current state
        self._agent_location = None
    
    def _is_valid_cell(self, cell: Tuple[int, int]) -> bool:
        """Check if a cell is valid in the environment."""
        return cell in self.state_to_idx
    
    def _step_from(self, cell: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        Compute next cell from current cell and action.
        If the move would go outside valid cells, stay in place.
        """
        direction = self._action_to_direction[action]
        next_cell = (cell[0] + direction[0], cell[1] + direction[1])
        
        if self._is_valid_cell(next_cell):
            return next_cell
        else:
            return cell  # Stay in place if hitting a wall
    
    def _get_obs(self) -> int:
        """Get current observation (state index)."""
        return self.state_to_idx[self._agent_location]
    
    def _get_info(self) -> Dict:
        """Get auxiliary information."""
        return {
            "agent_position": self._agent_location,
            "state_index": self.state_to_idx[self._agent_location]
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[int, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (can include 'start_state' key)
        
        Returns:
            observation: Initial state index
            info: Auxiliary information
        """
        super().reset(seed=seed)
        
        # Allow specifying start state via options
        if options is not None and "start_state" in options:
            start_idx = options["start_state"]
            if start_idx < 0 or start_idx >= self.n_states:
                raise ValueError(f"start_state must be in [0, {self.n_states-1}]")
            self._agent_location = self.idx_to_state[start_idx]
        else:
            # Random start position
            start_idx = self.np_random.integers(0, self.n_states)
            self._agent_location = self.idx_to_state[start_idx]
        
        self.current_s0 = self.idx_to_state[start_idx]
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self.render()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0-3)
        
        Returns:
            observation: New state index
            reward: Reward for this step (-1 per step, encouraging efficiency)
            terminated: Whether episode ended (always False for this env)
            truncated: Whether episode was truncated (handled by wrapper)
            info: Auxiliary information
        """
        # Move the agent
        self._agent_location = self._step_from(self._agent_location, action)
        
        # Simple reward structure: -1 per step
        reward = -1.0
        
        terminated = False
        truncated = False
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "ansi" or self.render_mode == "human":
            return self._render_ansi()
    
    def _render_ansi(self) -> Optional[str]:
        """Render the environment as ASCII art."""
        # Create grid
        grid = []
        for y in range(self.room_size):
            row = []
            for x in range(self.room_size):
                if (x, y) == self._agent_location:
                    row.append('A')  # Agent
                else:
                    row.append('.')  # Valid cell
            grid.append(' '.join(row))
        
        output = '\n'.join(grid)
        
        if self.render_mode == "human":
            print(output)
            print()
        
        return output
    
    def close(self):
        """Clean up resources."""
        pass

class SingleRoomSzeroEnv(gym.Env):
    """
    A simple gridworld environment with a single square room.
    
    The environment consists of a single square room of size room_size x room_size.
    
    Args:
        room_size: Size of the square room (default: 5)
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}
    
    def __init__(
        self, 
        room_size: int = 5,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        # Environment parameters
        self.room_size = room_size
        self.render_mode = render_mode
        
        # Build valid cells (states)
        self.cells = []
        self.state_to_idx = {}
        self.idx_to_state = {}
        
        # Single room
        for x in range(room_size):
            for y in range(room_size):
                idx = len(self.cells)
                self.cells.append((x, y))
                self.state_to_idx[(x, y)] = idx
                self.idx_to_state[idx] = (x, y)
        
        self.n_states = len(self.cells)**2
        self.n_cells = len(self.cells)
        self.reset_probability = 0.15 #1/self.n_cells
        
        # Define observation space: discrete state index
        self.observation_space = spaces.Discrete(self.n_states)
        
        # Define action space: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)
        
        # Action mapping
        self._action_to_direction = {
            0: np.array([0, -1]),  # up
            1: np.array([0, 1]),   # down
            2: np.array([-1, 0]),  # left
            3: np.array([1, 0]),   # right
        }
        
        # Current state
        self._agent_location = None
        self.current_s0 = None
    
    def _is_valid_cell(self, cell: Tuple[int, int]) -> bool:
        """Check if a cell is valid in the environment."""
        return cell in self.state_to_idx
    
    def _step_from(self, cell: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        Compute next cell from current cell and action.
        If the move would go outside valid cells, stay in place.
        """
        direction = self._action_to_direction[action]
        next_cell = (cell[0] + direction[0], cell[1] + direction[1])
        
        if self._is_valid_cell(next_cell):
            cell = next_cell
        if np.random.rand() < self.reset_probability:
            self.current_s0 = cell
    
        return (cell, self.current_s0)
    
    def _step_single(self, cell: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        Compute next cell from current cell and action.
        If the move would go outside valid cells, stay in place.
        """
        
        direction = self._action_to_direction[action]
        next_cell = (cell[0] + direction[0], cell[1] + direction[1])
        
        if self._is_valid_cell(next_cell):
            cell = next_cell
    
        return cell
    
    def _get_obs(self) -> int:
        """Get current observation (state index)."""
        return self.state_to_idx[self._agent_location]+ self.state_to_idx[self.current_s0]*self.n_cells
    
    def _get_info(self) -> Dict:
        """Get auxiliary information."""
        return {
            "agent_position": self._agent_location,
            "state_index": self.state_to_idx[self._agent_location]
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[int, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (can include 'start_state' key)
        
        Returns:
            observation: Initial state index
            info: Auxiliary information
        """
        super().reset(seed=seed)
        
        # Allow specifying start state via options
        if options is not None and "start_state" in options:
            start_idx = options["start_state"]
            if start_idx < 0 or start_idx >= self.n_states:
                raise ValueError(f"start_state must be in [0, {self.n_states-1}]")
            self._agent_location = self.idx_to_state[start_idx]
            self.current_s0 = self.idx_to_state[start_idx]
        else:
            # Random start position
            start_idx = self.np_random.integers(0, self.n_cells)
            self._agent_location = self.idx_to_state[start_idx]
            self.current_s0 = self.idx_to_state[start_idx]
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self.render()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0-3)
        
        Returns:
            observation: New state index
            reward: Reward for this step (-1 per step, encouraging efficiency)
            terminated: Whether episode ended (always False for this env)
            truncated: Whether episode was truncated (handled by wrapper)
            info: Auxiliary information
        """
        # Move the agent
        self._agent_location, self.current_s0 = self._step_from(self._agent_location, action)
        
        # Simple reward structure: -1 per step
        reward = -1.0
        
        terminated = False
        truncated = False
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "ansi" or self.render_mode == "human":
            return self._render_ansi()
    
    def _render_ansi(self) -> Optional[str]:
        """Render the environment as ASCII art."""
        # Create grid
        grid = []
        for y in range(self.room_size):
            row = []
            for x in range(self.room_size):
                if (x, y) == self._agent_location:
                    row.append('A')  # Agent
                else:
                    row.append('.')  # Valid cell
            grid.append(' '.join(row))
        
        output = '\n'.join(grid)
        
        if self.render_mode == "human":
            print(output)
            print()
        
        return output
    
    def close(self):
        """Clean up resources."""
        pass

class TwoRoomsEnv(gym.Env):
    """
    A gridworld environment with two rooms connected by a corridor.
    
    The environment consists of:
    - Left room: (0, 0) to (room_size-1, room_size-1)
    - Corridor: at position (room_size, corridor_y) with length corridor_length
    - Right room: (room_size + corridor_length, 0) to (2*room_size + corridor_length - 1, room_size - 1)
    
    Args:
        room_size: Size of each square room (default: 5)
        corridor_length: Length of the corridor connecting the rooms (default: 1)
        corridor_y: Y-coordinate of the corridor (default: 2, must be < room_size)
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}
    
    def __init__(
        self, 
        room_size: int = 5,
        corridor_length: int = 1,
        corridor_y: int = 2,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        # Environment parameters
        self.room_size = room_size
        self.corridor_length = corridor_length
        self.corridor_y = corridor_y
        self.render_mode = render_mode
        
        # Validate corridor position
        if corridor_y >= room_size:
            raise ValueError(f"corridor_y ({corridor_y}) must be less than room_size ({room_size})")
        
        # Build valid cells (states)
        self.cells = []
        self.state_to_idx = {}
        self.idx_to_state = {}
        
        # Left room
        for x in range(room_size):
            for y in range(room_size):
                idx = len(self.cells)
                self.cells.append((x, y))
                self.state_to_idx[(x, y)] = idx
                self.idx_to_state[idx] = (x, y)
        
        # Corridor
        for i in range(corridor_length):
            x = room_size + i
            y = corridor_y
            idx = len(self.cells)
            self.cells.append((x, y))
            self.state_to_idx[(x, y)] = idx
            self.idx_to_state[idx] = (x, y)
        
        # Right room
        start_x = room_size + corridor_length
        for x in range(start_x, start_x + room_size):
            for y in range(room_size):
                idx = len(self.cells)
                self.cells.append((x, y))
                self.state_to_idx[(x, y)] = idx
                self.idx_to_state[idx] = (x, y)
        
        self.n_states = len(self.cells)
        
        # Define observation space: discrete state index
        self.observation_space = spaces.Discrete(self.n_states)
        
        # Define action space: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)
        
        # Action mapping
        self._action_to_direction = {
            0: np.array([0, -1]),  # up
            1: np.array([0, 1]),   # down
            2: np.array([-1, 0]),  # left
            3: np.array([1, 0]),   # right
        }
        
        # Current state
        self._agent_location = None
        
    def _is_valid_cell(self, cell: Tuple[int, int]) -> bool:
        """Check if a cell is valid in the environment."""
        return cell in self.state_to_idx
    
    def _step_from(self, cell: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        Compute next cell from current cell and action.
        If the move would go outside valid cells, stay in place.
        """
        direction = self._action_to_direction[action]
        next_cell = (cell[0] + direction[0], cell[1] + direction[1])
        
        if self._is_valid_cell(next_cell):
            return next_cell
        else:
            return cell  # Stay in place if hitting a wall
    
    def _get_obs(self) -> int:
        """Get current observation (state index)."""
        return self.state_to_idx[self._agent_location]
    
    def _get_info(self) -> Dict:
        """Get auxiliary information."""
        return {
            "agent_position": self._agent_location,
            "state_index": self.state_to_idx[self._agent_location]
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[int, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (can include 'start_state' key)
        
        Returns:
            observation: Initial state index
            info: Auxiliary information
        """
        super().reset(seed=seed)
        
        # Allow specifying start state via options
        if options is not None and "start_state" in options:
            start_idx = options["start_state"]
            if start_idx < 0 or start_idx >= self.n_states:
                raise ValueError(f"start_state must be in [0, {self.n_states-1}]")
            self._agent_location = self.idx_to_state[start_idx]
        else:
            # Random start position
            start_idx = self.np_random.integers(0, self.n_states)
            self._agent_location = self.idx_to_state[start_idx]
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self.render()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0-3)
        
        Returns:
            observation: New state index
            reward: Reward for this step (-1 per step, encouraging efficiency)
            terminated: Whether episode ended (always False for this env)
            truncated: Whether episode was truncated (handled by wrapper)
            info: Auxiliary information
        """
        # Move the agent
        self._agent_location = self._step_from(self._agent_location, action)
        
        # Simple reward structure: -1 per step (encourages finding shortest paths)
        reward = -1.0
        
        # This environment doesn't have terminal states by default
        # (could be modified to have a goal state)
        terminated = False
        truncated = False
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "ansi" or self.render_mode == "human":
            return self._render_ansi()
    
    def _render_ansi(self) -> Optional[str]:
        """Render the environment as ASCII art."""
        # Calculate grid dimensions
        max_x = max(cell[0] for cell in self.cells)
        max_y = max(cell[1] for cell in self.cells)
        
        # Create grid
        grid = []
        for y in range(max_y + 1):
            row = []
            for x in range(max_x + 1):
                if (x, y) == self._agent_location:
                    row.append('A')  # Agent
                elif (x, y) in self.state_to_idx:
                    row.append('.')  # Valid cell
                else:
                    row.append(' ')  # Wall/invalid
            grid.append(' '.join(row))
        
        output = '\n'.join(grid)
        
        if self.render_mode == "human":
            print(output)
            print()
        
        return output

    def __getattribute__(self, name):
        return super().__getattribute__(name)
    
    def close(self):
        """Clean up resources."""
        pass


class TwoRoomsSzeroEnv(gym.Env):
    """
    A gridworld environment with two rooms connected by a corridor.
    
    The environment consists of:
    - Left room: (0, 0) to (room_size-1, room_size-1)
    - Corridor: at position (room_size, corridor_y) with length corridor_length
    - Right room: (room_size + corridor_length, 0) to (2*room_size + corridor_length - 1, room_size - 1)
    
    Args:
        room_size: Size of each square room (default: 5)
        corridor_length: Length of the corridor connecting the rooms (default: 1)
        corridor_y: Y-coordinate of the corridor (default: 2, must be < room_size)
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}
    
    def __init__(
        self, 
        room_size: int = 5,
        corridor_length: int = 1,
        corridor_y: int = 2,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        # Environment parameters
        self.room_size = room_size
        self.corridor_length = corridor_length
        self.corridor_y = corridor_y
        self.render_mode = render_mode
        
        
        # Validate corridor position
        if corridor_y >= room_size:
            raise ValueError(f"corridor_y ({corridor_y}) must be less than room_size ({room_size})")
        
        # Build valid cells (states)
        self.cells = []
        self.state_to_idx = {}
        self.idx_to_state = {}
        
        # Left room
        for x in range(room_size):
            for y in range(room_size):
                idx = len(self.cells)
                self.cells.append((x, y))
                self.state_to_idx[(x, y)] = idx
                self.idx_to_state[idx] = (x, y)
        
        # Corridor
        for i in range(corridor_length):
            x = room_size + i
            y = corridor_y
            idx = len(self.cells)
            self.cells.append((x, y))
            self.state_to_idx[(x, y)] = idx
            self.idx_to_state[idx] = (x, y)
        
        # Right room
        start_x = room_size + corridor_length
        for x in range(start_x, start_x + room_size):
            for y in range(room_size):
                idx = len(self.cells)
                self.cells.append((x, y))
                self.state_to_idx[(x, y)] = idx
                self.idx_to_state[idx] = (x, y)
        
        self.n_states = len(self.cells)**2
        self.n_cells = len(self.cells)
        self.reset_probability = 1/self.n_cells
        # Define observation space: discrete state index
        self.observation_space = spaces.Discrete(self.n_states)
        
        # Define action space: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)
        
        # Action mapping
        self._action_to_direction = {
            0: np.array([0, -1]),  # up
            1: np.array([0, 1]),   # down
            2: np.array([-1, 0]),  # left
            3: np.array([1, 0]),   # right
        }
        
        # Current state
        self._agent_location = None
        self.current_s0 = None
        
    def _is_valid_cell(self, cell: Tuple[int, int]) -> bool:
        """Check if a cell is valid in the environment."""
        return cell in self.state_to_idx
    
    def _step_from(self, cell: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        Compute next cell from current cell and action.
        If the move would go outside valid cells, stay in place.
        """
        
        direction = self._action_to_direction[action]
        next_cell = (cell[0] + direction[0], cell[1] + direction[1])
        
        if self._is_valid_cell(next_cell):
            cell = next_cell
        if np.random.rand() < self.reset_probability:
            print("Resetting s0--------------")
            self.current_s0 = cell
    
        return (cell, self.current_s0)
    
    def _step_single(self, cell: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        Compute next cell from current cell and action.
        If the move would go outside valid cells, stay in place.
        """
        
        direction = self._action_to_direction[action]
        next_cell = (cell[0] + direction[0], cell[1] + direction[1])
        
        if self._is_valid_cell(next_cell):
            cell = next_cell
    
        return cell  # Stay in place if hitting a wall
    
    def _get_obs(self) -> int:
        """Get current observation (state index)."""
        return self.state_to_idx[self._agent_location]+ self.state_to_idx[self.current_s0]*self.n_cells
    
    def _get_info(self) -> Dict:
        """Get auxiliary information."""
        return {
            "agent_position": self._agent_location,
            "state_index": self.state_to_idx[self._agent_location]
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[int, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (can include 'start_state' key)
        
        Returns:
            observation: Initial state index
            info: Auxiliary information
        """
        super().reset(seed=seed)
        
        # Allow specifying start state via options
        if options is not None and "start_state" in options:
            start_idx = options["start_state"]
            if start_idx < 0 or start_idx >= self.n_states:
                raise ValueError(f"start_state must be in [0, {self.n_states-1}]")
            self._agent_location = self.idx_to_state[start_idx]
            self.current_s0 = self.idx_to_state[start_idx]
        else:
            # Random start position
            start_idx = self.np_random.integers(0, self.n_cells)
            self._agent_location = self.idx_to_state[start_idx]
            self.current_s0 = self.idx_to_state[start_idx]
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self.render()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0-3)
        
        Returns:
            observation: New state index
            reward: Reward for this step (-1 per step, encouraging efficiency)
            terminated: Whether episode ended (always False for this env)
            truncated: Whether episode was truncated (handled by wrapper)
            info: Auxiliary information
        """
        # Move the agent
        self._agent_location, self.current_s0 = self._step_from(self._agent_location, action)
        
        # Simple reward structure: -1 per step
        reward = -1.0
        
        terminated = False
        truncated = False
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "ansi" or self.render_mode == "human":
            return self._render_ansi()
    
    def _render_ansi(self) -> Optional[str]:
        """Render the environment as ASCII art."""
        # Calculate grid dimensions
        max_x = max(cell[0] for cell in self.cells)
        max_y = max(cell[1] for cell in self.cells)
        
        # Create grid
        grid = []
        for y in range(max_y + 1):
            row = []
            for x in range(max_x + 1):
                if (x, y) == self._agent_location:
                    row.append('A')  # Agent
                elif (x, y) in self.state_to_idx:
                    row.append('.')  # Valid cell
                else:
                    row.append(' ')  # Wall/invalid
            grid.append(' '.join(row))
        
        output = '\n'.join(grid)
        
        if self.render_mode == "human":
            print(output)
            print()
        
        return output

    def __getattribute__(self, name):
        return super().__getattribute__(name)
    
    def close(self):
        """Clean up resources."""
        pass


# Register the environment
gym.register(
    id="SingleRoom-v0",
    entry_point="env:SingleRoomEnv",
    max_episode_steps=300,
)

gym.register(
    id="TwoRooms-v0",
    entry_point="env:TwoRoomsEnv",
    max_episode_steps=300,
)

gym.register(
    id="TwoRoomsSzero-v0",
    entry_point="env:TwoRoomsSzeroEnv",
    max_episode_steps=300,
)

