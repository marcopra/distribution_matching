"""
Room Environments - A Gymnasium custom environment with abstract base class.

This module provides a base class for room-based gridworld environments
and specific implementations for single, two, and four room layouts.
"""

from typing import Optional, Tuple, Dict, List, Set
from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image, ImageDraw, ImageFont


class BaseRoomEnv(gym.Env, ABC):
    """
    Abstract base class for room-based gridworld environments.
    
    This class provides common functionality for all room environments including:
    - State management
    - Action handling
    - Reward calculation
    - Rendering (both ASCII and RGB)
    
    Subclasses must implement _build_cells() to define the environment layout.
    """
    
    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 4}
    
    DEAD_STATE = (-1, -1)  # Special coordinate for dead state
    
    def __init__(
        self,
        goal_position: Optional[Tuple[int, int]] = None,
        start_position: Optional[Tuple[int, int]] = None,
        max_steps: int = 300,
        render_mode: Optional[str] = None,
        show_coordinates: bool = False,
        lava: bool = False,
        dense_reward: bool = False
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.show_coordinates = show_coordinates
        self.lava = lava
        self._step_count = 0
        self.dense_reward = dense_reward
        
        # Build the environment layout (implemented by subclasses)
        self.cells = []
        self.state_to_idx = {}
        self.idx_to_state = {}
        self._build_cells()
        
        # Add dead state if lava is enabled
        if self.lava:
            self._add_cell(self.DEAD_STATE)
        
        self.n_states = len(self.cells)
        
        # Set goal and start positions
        self._goal_position_param = goal_position  # Store original parameter
        self._start_position_param = start_position  # Store original parameter
        self.goal_position = None
        self.start_position = None  # This will be set during reset
        
        # Define spaces
        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(4)
        
        # Action mapping: 0=up, 1=down, 2=left, 3=right
        self._action_to_direction = {
            0: np.array([0, -1]),  # up
            1: np.array([0, 1]),   # down
            2: np.array([-1, 0]),  # left
            3: np.array([1, 0]),   # right
        }
        
        self._agent_location = None
    
    @abstractmethod
    def _build_cells(self):
        """
        Build the valid cells for the environment.
        Must populate self.cells, self.state_to_idx, and self.idx_to_state.
        """
        pass
    
    @abstractmethod
    def _get_default_goal(self) -> Tuple[int, int]:
        """Return the default goal position for this environment."""
        pass
    
    def _add_cell(self, cell: Tuple[int, int]):
        """Add a cell to the environment."""
        if cell not in self.state_to_idx:
            idx = len(self.cells)
            self.cells.append(cell)
            self.state_to_idx[cell] = idx
            self.idx_to_state[idx] = cell
    
    def _set_goal_position(self, goal_position: Optional[Tuple[int, int]]) -> Tuple[int, int]:
        """Set and validate goal position."""
        if goal_position is not None:
            if isinstance(goal_position, int):
                return self.idx_to_state[goal_position]
            else:
                # convert list to tuple if needed
                goal_position = tuple(goal_position)
                assert isinstance(goal_position, tuple), "Goal position must be a tuple (x, y) or an int index"
                if goal_position not in self.state_to_idx:
                    raise ValueError(f"Goal position {goal_position} is not a valid cell, valid cells: {self.cells}")
                return goal_position
        else:
            return self._get_default_goal()
    
    def _set_start_position(self, start_position: Optional[Tuple[int, int]]) -> Tuple[int, int]:
        """Set and validate start position."""
        if start_position is not None:
            if isinstance(start_position, int):
                return self.idx_to_state[start_position]
            else:
                start_position = tuple(start_position)
                if start_position not in self.state_to_idx:
                    raise ValueError(f"Start position {start_position} is not a valid cell, valid cells: {self.cells}")
                return start_position
        else:
            # Random position
            start_idx = self.np_random.integers(0, self.n_states)
            return self.idx_to_state[start_idx]
    
    def _is_valid_cell(self, cell: Tuple[int, int]) -> bool:
        """Check if a cell is valid in the environment."""
        return cell in self.state_to_idx
    
    def step_from(self, cell: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        Compute next cell from current cell and action.
        If the move would go outside valid cells, stay in place (or go to dead state if lava).
        """
        # If already in dead state, stay there
        if self.lava and cell == self.DEAD_STATE:
            return self.DEAD_STATE
        
        direction = self._action_to_direction[action]
        next_cell = (cell[0] + direction[0], cell[1] + direction[1])
        
        if self._is_valid_cell(next_cell) and next_cell != self.DEAD_STATE:
            return next_cell
        else:
            # Hit a wall/lava
            if self.lava:
                return self.DEAD_STATE
            else:
                return cell  # Stay in place if hitting a wall
    
    def render_from_position(self, position: Tuple[int, int]) -> np.ndarray:
        """
        Render the environment from a specific agent position without modifying state.
        
        Args:
            position: (x, y) tuple representing agent position
            
        Returns:
            RGB image array of shape (H, W, 3)
        """
        # Save current agent location
        original_location = self._agent_location
        
        # Temporarily set agent to desired position
        self._agent_location = position
        
        # Render the image
        img = self._render_rgb()
        
        # Restore original agent location
        self._agent_location = original_location
        
        return img
    
    def _get_obs(self) -> int:
        """Get current observation (state index)."""
        return self.state_to_idx[self._agent_location]
    
    def _get_info(self) -> Dict:
        """Get auxiliary information."""
        return {
            "agent_position": self._agent_location,
            "state_index": self.state_to_idx[self._agent_location],
            "step_count": self._step_count
        }
    
    def __getattribute__(self, name):
        return super().__getattribute__(name)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[int, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self._step_count = 0
        
        # Set goal position (random if not specified, or from options, or from constructor)
        if options is not None and "goal_position" in options:
            goal_pos = options["goal_position"]
            if isinstance(goal_pos, int):
                self.goal_position = self.idx_to_state[goal_pos]
            else:
                self.goal_position = goal_pos
        elif self._goal_position_param is not None:
            self.goal_position = self._set_goal_position(self._goal_position_param)
        else:
            # Random goal position (excluding dead state)
            valid_states = [i for i in range(self.n_states) 
                          if self.idx_to_state[i] != self.DEAD_STATE]
            goal_idx = self.np_random.choice(valid_states)
            self.goal_position = self.idx_to_state[goal_idx]
        
        # Set start position (random if not specified, or from options, or from constructor)
        if options is not None and "start_state" in options:
            # Backward compatibility with "start_state"
            start_idx = options["start_state"]
            if start_idx < 0 or start_idx >= self.n_states:
                raise ValueError(f"start_state must be in [0, {self.n_states-1}]")
            self._agent_location = self.idx_to_state[start_idx]
            self.start_position = self._agent_location  # Update start_position
        elif options is not None and "start_position" in options:
            start_pos = options["start_position"]
            if isinstance(start_pos, int):
                self._agent_location = self.idx_to_state[start_pos]
            else:
                self._agent_location = start_pos
            self.start_position = self._agent_location  # Update start_position
        elif self._start_position_param is not None:
            self._agent_location = self._set_start_position(self._start_position_param)
            self.start_position = self._agent_location  # Update start_position
        else:
            # Random start position (excluding dead state)
            valid_states = [i for i in range(self.n_states) 
                          if self.idx_to_state[i] != self.DEAD_STATE]
            start_idx = self.np_random.choice(valid_states)
            self._agent_location = self.idx_to_state[start_idx]
            self.start_position = self._agent_location  # Update start_position
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self.render()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self._step_count += 1
        
        # Move the agent
        self._agent_location = self.step_from(self._agent_location, action)
        
        # Check if goal is reached
        terminated = self._agent_location == self.goal_position
        
        # Truncate if in dead state (lava) or max steps reached
        in_dead_state = self.lava and self._agent_location == self.DEAD_STATE
        truncated = in_dead_state or self._step_count >= self.max_steps

        # Reward calculation
        if self.dense_reward:
            # Dense reward: negative Manhattan distance to goal
            if terminated:
                reward = 0.0
            elif in_dead_state:
                reward = -1.0  # Penalty for lava
            else:
                # Manhattan distance to goal
                distance = abs(self._agent_location[0] - self.goal_position[0]) + \
                        abs(self._agent_location[1] - self.goal_position[1])
                reward = -distance
        else:
            # Reward: 1 - 0.9 * (step_count / max_steps) for success, 0 for failure
            if terminated:
                reward = 0 #1.0 - 0.9 * (self._step_count / self.max_steps)
                # terminated =  False  
            else:
                reward = - 1.0
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None
        elif self.render_mode == "ansi" or self.render_mode == "human":
            return self._render_ansi()
        elif self.render_mode == "rgb_array":
            return self._render_rgb()
    
    def _render_ansi(self) -> Optional[str]:
        """Render the environment as ASCII art."""
        # Don't include dead state in rendering bounds
        valid_cells = [cell for cell in self.cells if cell != self.DEAD_STATE]
        
        max_x = max(cell[0] for cell in valid_cells)
        max_y = max(cell[1] for cell in valid_cells)
        min_x = min(cell[0] for cell in valid_cells)
        min_y = min(cell[1] for cell in valid_cells)
        
        grid = []
        for y in range(min_y, max_y + 1):
            row = []
            for x in range(min_x, max_x + 1):
                if self.lava and self._agent_location == self.DEAD_STATE and (x, y) not in self.state_to_idx:
                    # Show lava
                    row.append('L')
                elif (x, y) == self._agent_location and self._agent_location != self.DEAD_STATE:
                    row.append('A')
                elif (x, y) == self.goal_position:
                    row.append('G')
                elif (x, y) in self.state_to_idx:
                    row.append('.')
                else:
                    row.append('L' if self.lava else '#')
            grid.append(' '.join(row))
        
        output = '\n'.join(grid)
        
        if self.render_mode == "human":
            print(output)
            print()
        
        return output
    
    def _render_rgb(self) -> np.ndarray:
        """Render the environment as RGB image."""
        cell_size = 64
        cell_padding = 2  # Padding to create gray lines between cells
        
        # Calculate grid dimensions (excluding dead state)
        valid_cells = [cell for cell in self.cells if cell != self.DEAD_STATE]
        
        max_x = max(cell[0] for cell in valid_cells)
        max_y = max(cell[1] for cell in valid_cells)
        min_x = min(cell[0] for cell in valid_cells)
        min_y = min(cell[1] for cell in valid_cells)
        
        grid_width = max_x - min_x + 1
        grid_height = max_y - min_y + 1
        
        # Add coordinate axis space if needed (white background for coordinates)
        coord_space = cell_size if self.show_coordinates else 0
        
        # Image dimensions with border cells around the environment
        # Adding coord_space on all sides to make it symmetric
        img_width = (grid_width + 2) * cell_size + coord_space * 2
        img_height = (grid_height + 2) * cell_size + coord_space * 2
        
        # Create image with background (gray for walls, lava color for lava)
        bg_color = (207, 16, 32) if self.lava else (128, 128, 128)  # #CF1020 for lava
        img = Image.new('RGB', (img_width, img_height), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Draw white background for coordinate area
        if self.show_coordinates:
            # Top white bar
            draw.rectangle(
                [0, 0, img_width, coord_space],
                fill=(255, 255, 255)
            )
            # Left white bar
            draw.rectangle(
                [0, 0, coord_space, img_height],
                fill=(255, 255, 255)
            )
            # Right white bar
            draw.rectangle(
                [img_width - coord_space, 0, img_width, img_height],
                fill=(255, 255, 255)
            )
            # Bottom white bar
            draw.rectangle(
                [0, img_height - coord_space, img_width, img_height],
                fill=(255, 255, 255)
            )
        
        # Try to load font for coordinates only
        try:
            coord_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            try:
                coord_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 20)
            except:
                coord_font = ImageFont.load_default()
        
        # Draw coordinate labels if enabled
        if self.show_coordinates:
            # X-axis labels (top and bottom)
            for x in range(min_x, max_x + 1):
                px = (x - min_x + 1) * cell_size + cell_size // 2 + coord_space
                # Top
                py_top = coord_space // 2
                text = str(x)
                bbox = draw.textbbox((0, 0), text, font=coord_font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                draw.text((px - text_width // 2, py_top - text_height // 2), text, 
                         fill=(0, 0, 0), font=coord_font)
                # Bottom
                py_bottom = img_height - coord_space // 2
                draw.text((px - text_width // 2, py_bottom - text_height // 2), text, 
                         fill=(0, 0, 0), font=coord_font)
            
            # Y-axis labels (left and right)
            for y in range(min_y, max_y + 1):
                py = (y - min_y + 1) * cell_size + cell_size // 2 + coord_space
                text = str(y)
                bbox = draw.textbbox((0, 0), text, font=coord_font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                # Left
                px_left = coord_space // 2
                draw.text((px_left - text_width // 2, py - text_height // 2), text, 
                         fill=(0, 0, 0), font=coord_font)
                # Right
                px_right = img_width - coord_space // 2
                draw.text((px_right - text_width // 2, py - text_height // 2), text, 
                         fill=(0, 0, 0), font=coord_font)
        
        # Draw border cells (gray or lava, around the environment)
        for x in range(-1, grid_width + 1):
            for y in range(-1, grid_height + 1):
                if x == -1 or x == grid_width or y == -1 or y == grid_height:
                    px = (x + 1) * cell_size + coord_space
                    py = (y + 1) * cell_size + coord_space
                    # Border cells remain background color (gray or lava)
                    draw.rectangle(
                        [px, py, px + cell_size, py + cell_size],
                        fill=bg_color
                    )
        
        # Draw cells
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                # Position in image (offset by border and coordinates)
                px = (x - min_x + 1) * cell_size + coord_space
                py = (y - min_y + 1) * cell_size + coord_space
                
                if (x, y) in self.state_to_idx and (x, y) != self.DEAD_STATE:
                    # Valid cell - black background with padding
                    draw.rectangle(
                        [px + cell_padding, py + cell_padding, 
                         px + cell_size - cell_padding, py + cell_size - cell_padding],
                        fill=(0, 0, 0)
                    )
                    
                    # Check if this is the goal or agent position
                    is_goal = (x, y) == self.goal_position
                    is_agent = (x, y) == self._agent_location and self._agent_location != self.DEAD_STATE
                    
                    center_x = px + cell_size // 2
                    center_y = py + cell_size // 2
                    
                    if is_goal:
                        # Green star for goal (using polygon to draw a star shape)
                        star_outer_radius = cell_size // 3
                        star_inner_radius = cell_size // 6
                        star_points = []
                        for i in range(10):
                            angle = (i * 36 - 90) * np.pi / 180  # 36 degrees between points, start at top
                            radius = star_outer_radius if i % 2 == 0 else star_inner_radius
                            x_point = center_x + radius * np.cos(angle)
                            y_point = center_y + radius * np.sin(angle)
                            star_points.append((x_point, y_point))
                        draw.polygon(star_points, fill=(0, 255, 0))
                    
                    if is_agent:
                        # Red square for agent
                        square_size = cell_size // 3
                        draw.rectangle(
                            [center_x - square_size, center_y - square_size,
                             center_x + square_size, center_y + square_size],
                            fill=(255, 0, 0)
                        )
                else:
                    # Wall/Lava - background color (creates seamless walls/lava)
                    draw.rectangle(
                        [px, py, px + cell_size, py + cell_size],
                        fill=bg_color
                    )
        
        return np.array(img)
    
    def close(self):
        """Clean up resources."""
        pass


class SingleRoomEnv(BaseRoomEnv):
    """
    A simple gridworld environment with a single square room.
    
    Args:
        room_size: Size of the square room (default: 5)
        goal_position: Goal position as (x, y) tuple or state index (optional, random if None)
        start_position: Start position as (x, y) tuple or state index (optional, random if None)
    """
    
    def __init__(
        self,
        room_size: int = 5,
        goal_position: Optional[Tuple[int, int]] = None,
        start_position: Optional[Tuple[int, int]] = None,
        max_steps: int = 300,
        render_mode: Optional[str] = None,
        show_coordinates: bool = False,
        lava: bool = False
    ):
        self.room_size = room_size
        super().__init__(
            goal_position=goal_position,
            start_position=start_position,
            max_steps=max_steps,
            render_mode=render_mode,
            show_coordinates=show_coordinates,
            lava=lava
        )
    
    def _build_cells(self):
        """Build cells for single room."""
        for x in range(self.room_size):
            for y in range(self.room_size):
                self._add_cell((x, y))
    
    def _get_default_goal(self) -> Tuple[int, int]:
        """Default goal: bottom-right corner."""
        return (self.room_size - 1, self.room_size - 1)


class TwoRoomsEnv(BaseRoomEnv):
    """
    A gridworld environment with two rooms connected by a corridor.
    
    Args:
        room_size: Size of each square room (default: 5)
        corridor_length: Length of the corridor connecting the rooms (default: 1)
        corridor_y: Y-coordinate of the corridor (default: 2)
        goal_position: Goal position as (x, y) tuple or state index (optional, random if None)
        start_position: Start position as (x, y) tuple or state index (optional, random if None)
    """
    
    def __init__(
        self,
        room_size: int = 5,
        corridor_length: int = 1,
        corridor_y: int = 2,
        goal_position: Optional[Tuple[int, int]] = None,
        start_position: Optional[Tuple[int, int]] = None,
        max_steps: int = 300,
        render_mode: Optional[str] = None,
        show_coordinates: bool = False,
        lava: bool = False
    ):
        self.room_size = room_size
        self.corridor_length = corridor_length
        self.corridor_y = corridor_y
        
        if corridor_y >= room_size:
            raise ValueError(f"corridor_y ({corridor_y}) must be less than room_size ({room_size})")
        
        super().__init__(
            goal_position=goal_position,
            start_position=start_position,
            max_steps=max_steps,
            render_mode=render_mode,
            show_coordinates=show_coordinates,
            lava=lava
        )
    
    def _build_cells(self):
        """Build cells for two rooms with corridor."""
        # Left room
        for x in range(self.room_size):
            for y in range(self.room_size):
                self._add_cell((x, y))
        
        # Corridor
        for i in range(self.corridor_length):
            x = self.room_size + i
            y = self.corridor_y
            self._add_cell((x, y))
        
        # Right room
        start_x = self.room_size + self.corridor_length
        for x in range(start_x, start_x + self.room_size):
            for y in range(self.room_size):
                self._add_cell((x, y))
    
    def _get_default_goal(self) -> Tuple[int, int]:
        """Default goal: bottom-right corner of right room."""
        start_x = self.room_size + self.corridor_length
        return (start_x + self.room_size - 1, self.room_size - 1)


class FourRoomsEnv(BaseRoomEnv):
    """
    A gridworld environment with four rooms arranged in a 2x2 grid,
    connected by corridors in a circular fashion.
    
    Layout:
    ┌─────┬─────┐
    │  0  │  1  │
    ├─────┼─────┤
    │  3  │  2  │
    └─────┴─────┘
    
    Corridors connect: 0↔1, 1↔2, 2↔3, 3↔0
    
    Args:
        room_size: Size of each square room (default: 5)
        corridor_length: Length of the corridor (default: 1)
        corridor_positions: Dict with 'horizontal' and 'vertical' corridor positions
        goal_position: Goal position as (x, y) tuple or state index (optional, random if None)
        start_position: Start position as (x, y) tuple or state index (optional, random if None)
    """
    
    def __init__(
        self,
        room_size: int = 5,
        corridor_length: int = 1,
        corridor_positions: Optional[Dict[str, int]] = None,
        goal_position: Optional[Tuple[int, int]] = None,
        start_position: Optional[Tuple[int, int]] = None,
        max_steps: int = 300,
        render_mode: Optional[str] = None,
        show_coordinates: bool = False,
        lava: bool = False
    ):
        self.room_size = room_size
        self.corridor_length = corridor_length
        
        # Default corridor positions (middle of each wall)
        if corridor_positions is None:
            corridor_positions = {
                'horizontal': room_size // 2,
                'vertical': room_size // 2
            }
        self.corridor_positions = corridor_positions
        
        super().__init__(
            goal_position=goal_position,
            start_position=start_position,
            max_steps=max_steps,
            render_mode=render_mode,
            show_coordinates=show_coordinates,
            lava=lava
        )
    
    def _build_cells(self):
        """Build cells for four rooms with circular corridors."""
        # Room 0 (top-left)
        for x in range(self.room_size):
            for y in range(self.room_size):
                self._add_cell((x, y))
        
        # Corridor 0→1 (horizontal, connects room 0 to room 1)
        y_pos = self.corridor_positions['horizontal']
        for i in range(self.corridor_length):
            x = self.room_size + i
            self._add_cell((x, y_pos))
        
        # Room 1 (top-right)
        start_x = self.room_size + self.corridor_length
        for x in range(start_x, start_x + self.room_size):
            for y in range(self.room_size):
                self._add_cell((x, y))
        
        # Corridor 1→2 (vertical, connects room 1 to room 2)
        x_pos = start_x + self.corridor_positions['vertical']
        for i in range(self.corridor_length):
            y = self.room_size + i
            self._add_cell((x_pos, y))
        
        # Room 2 (bottom-right)
        start_y = self.room_size + self.corridor_length
        for x in range(start_x, start_x + self.room_size):
            for y in range(start_y, start_y + self.room_size):
                self._add_cell((x, y))
        
        # Corridor 2→3 (horizontal, connects room 2 to room 3)
        y_pos = start_y + self.corridor_positions['horizontal']
        for i in range(self.corridor_length):
            x = self.room_size + self.corridor_length - 1 - i
            self._add_cell((x, y_pos))
        
        # Room 3 (bottom-left)
        for x in range(self.room_size):
            for y in range(start_y, start_y + self.room_size):
                self._add_cell((x, y))
        
        # Corridor 3→0 (vertical, connects room 3 to room 0)
        x_pos = self.corridor_positions['vertical']
        for i in range(self.corridor_length):
            y = self.room_size + i
            self._add_cell((x_pos, y))
    
    def _get_default_goal(self) -> Tuple[int, int]:
        """Default goal: center of room 2 (bottom-right)."""
        start_x = self.room_size + self.corridor_length
        start_y = self.room_size + self.corridor_length
        return (start_x + self.room_size // 2, start_y + self.room_size // 2)


# Register the environments
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
    id="FourRooms-v0",
    entry_point="env:FourRoomsEnv",
    max_episode_steps=300,
)

