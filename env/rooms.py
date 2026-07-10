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
from dm_env import specs


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
        self._render_background = None
        self._render_agent_sprite = None
        self._render_agent_mask = None
        self._render_goal_sprite = None
        self._render_goal_mask = None
        self._render_cell_origins = {}
        self._render_layout = None
        self._coord_font = None
        self._render_cache_ready = False
        self._ensure_render_cache()
    
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
    
    def render_from_position(
        self,
        position: Tuple[int, int],
        show_goal: bool = False,
    ) -> np.ndarray:
        """
        Render the environment from a specific agent position without modifying state.
        
        Args:
            position: (x, y) tuple representing agent position
            show_goal: Whether to render the goal marker
            
        Returns:
            RGB image array of shape (H, W, 3)
        """
        # Save current agent location
        original_location = self._agent_location
        
        # Temporarily set agent to desired position
        self._agent_location = position
        
        # Render the image
        img = self._render_rgb(show_goal=show_goal)
        
        # Restore original agent location
        self._agent_location = original_location
        
        return img
    
    def _get_obs(self) -> int:
        """Get current observation (state index)."""
        return self.state_to_idx[self._agent_location]

    def reward_spec(self):
        return specs.Array(shape=(1,), dtype=np.float32, name='reward')

    def discount_spec(self):
        return specs.Array(shape=(1,), dtype=np.float32, name='discount')

    def compute_reward_from_observation(self, observation: int) -> float:
        """Compute the reward associated with a stored observation under the current goal."""
        observation = np.asarray(observation)
        if observation.ndim == 0 or observation.size == 1:
            state_idx = int(observation.item())
        else:
            state_idx = int(np.argmax(observation))

        cell = self.idx_to_state[state_idx]
        terminated = cell == self.goal_position
        in_dead_state = self.lava and cell == self.DEAD_STATE

        if self.dense_reward:
            if terminated:
                return 0.0
            if in_dead_state:
                return -1.0
            distance = abs(cell[0] - self.goal_position[0]) + abs(cell[1] - self.goal_position[1])
            return -distance

        return 0.0 if terminated else -1.0
    
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
    
    def render(self, show_goal: bool = True):
        """Render the environment."""
        if self.render_mode is None:
            return None
        elif self.render_mode == "ansi" or self.render_mode == "human":
            return self._render_ansi(show_goal=show_goal)
        elif self.render_mode == "rgb_array":
            return self._render_rgb(show_goal=show_goal)

    def render_observation(self) -> np.ndarray:
        """Render the pixel observation consumed by the agent."""
        return self._render_rgb(show_goal=False)

    def render_image_observation(self) -> np.ndarray:
        """Render an auxiliary image observation with the goal visible."""
        return self._render_rgb(show_goal=True)

    def _get_render_layout(self) -> Dict[str, object]:
        """Compute and cache static rendering geometry."""
        if self._render_layout is not None:
            return self._render_layout

        cell_size = 64
        cell_padding = 2

        valid_cells = [cell for cell in self.cells if cell != self.DEAD_STATE]

        max_x = max(cell[0] for cell in valid_cells)
        max_y = max(cell[1] for cell in valid_cells)
        min_x = min(cell[0] for cell in valid_cells)
        min_y = min(cell[1] for cell in valid_cells)

        grid_width = max_x - min_x + 1
        grid_height = max_y - min_y + 1
        coord_space = cell_size if self.show_coordinates else 0

        img_width = (grid_width + 2) * cell_size + coord_space * 2
        img_height = (grid_height + 2) * cell_size + coord_space * 2
        bg_color = (207, 16, 32) if self.lava else (128, 128, 128)

        cell_origins = {}
        for cell in valid_cells:
            px = (cell[0] - min_x + 1) * cell_size + coord_space
            py = (cell[1] - min_y + 1) * cell_size + coord_space
            cell_origins[cell] = (px, py)

        self._render_layout = {
            "cell_size": cell_size,
            "cell_padding": cell_padding,
            "valid_cells": valid_cells,
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "grid_width": grid_width,
            "grid_height": grid_height,
            "coord_space": coord_space,
            "img_width": img_width,
            "img_height": img_height,
            "bg_color": bg_color,
            "cell_origins": cell_origins,
        }
        return self._render_layout

    def _get_coord_font(self):
        """Load the coordinate font once and reuse it across renders."""
        if self._coord_font is not None:
            return self._coord_font

        try:
            self._coord_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except Exception:
            try:
                self._coord_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 20)
            except Exception:
                self._coord_font = ImageFont.load_default()

        return self._coord_font

    def _build_background_image(self) -> np.ndarray:
        """Pre-render the static maze canvas without goal or agent markers."""
        layout = self._get_render_layout()
        cell_size = layout["cell_size"]
        cell_padding = layout["cell_padding"]
        coord_space = layout["coord_space"]
        img_width = layout["img_width"]
        img_height = layout["img_height"]
        bg_color = layout["bg_color"]
        min_x = layout["min_x"]
        max_x = layout["max_x"]
        min_y = layout["min_y"]
        max_y = layout["max_y"]
        grid_width = layout["grid_width"]
        grid_height = layout["grid_height"]

        img = Image.new("RGB", (img_width, img_height), color=bg_color)
        draw = ImageDraw.Draw(img)

        if self.show_coordinates:
            draw.rectangle([0, 0, img_width, coord_space], fill=(255, 255, 255))
            draw.rectangle([0, 0, coord_space, img_height], fill=(255, 255, 255))
            draw.rectangle([img_width - coord_space, 0, img_width, img_height], fill=(255, 255, 255))
            draw.rectangle([0, img_height - coord_space, img_width, img_height], fill=(255, 255, 255))

            coord_font = self._get_coord_font()
            for x in range(min_x, max_x + 1):
                px = (x - min_x + 1) * cell_size + cell_size // 2 + coord_space
                text = str(x)
                bbox = draw.textbbox((0, 0), text, font=coord_font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                py_top = coord_space // 2
                py_bottom = img_height - coord_space // 2
                draw.text((px - text_width // 2, py_top - text_height // 2), text, fill=(0, 0, 0), font=coord_font)
                draw.text((px - text_width // 2, py_bottom - text_height // 2), text, fill=(0, 0, 0), font=coord_font)

            for y in range(min_y, max_y + 1):
                py = (y - min_y + 1) * cell_size + cell_size // 2 + coord_space
                text = str(y)
                bbox = draw.textbbox((0, 0), text, font=coord_font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                px_left = coord_space // 2
                px_right = img_width - coord_space // 2
                draw.text((px_left - text_width // 2, py - text_height // 2), text, fill=(0, 0, 0), font=coord_font)
                draw.text((px_right - text_width // 2, py - text_height // 2), text, fill=(0, 0, 0), font=coord_font)

        for x in range(-1, grid_width + 1):
            for y in range(-1, grid_height + 1):
                if x == -1 or x == grid_width or y == -1 or y == grid_height:
                    px = (x + 1) * cell_size + coord_space
                    py = (y + 1) * cell_size + coord_space
                    draw.rectangle([px, py, px + cell_size, py + cell_size], fill=bg_color)

        for (x, y), (px, py) in layout["cell_origins"].items():
            if (x, y) in self.state_to_idx and (x, y) != self.DEAD_STATE:
                draw.rectangle(
                    [
                        px + cell_padding,
                        py + cell_padding,
                        px + cell_size - cell_padding,
                        py + cell_size - cell_padding,
                    ],
                    fill=(0, 0, 0),
                )

        return np.array(img, copy=True)

    def _build_goal_sprite(self) -> Tuple[np.ndarray, np.ndarray]:
        """Pre-render the goal marker once as a transparent sprite."""
        cell_size = self._get_render_layout()["cell_size"]
        sprite = Image.new("RGBA", (cell_size, cell_size), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(sprite)

        center = cell_size // 2
        star_outer_radius = cell_size // 3
        star_inner_radius = cell_size // 6
        star_points = []
        for i in range(10):
            angle = (i * 36 - 90) * np.pi / 180
            radius = star_outer_radius if i % 2 == 0 else star_inner_radius
            x_point = center + radius * np.cos(angle)
            y_point = center + radius * np.sin(angle)
            star_points.append((x_point, y_point))
        draw.polygon(star_points, fill=(0, 255, 0, 255))

        sprite_arr = np.array(sprite)
        return sprite_arr[..., :3], sprite_arr[..., 3] > 0

    def _build_agent_sprite(self) -> Tuple[np.ndarray, np.ndarray]:
        """Pre-render the agent marker once as a transparent sprite."""
        cell_size = self._get_render_layout()["cell_size"]
        sprite = Image.new("RGBA", (cell_size, cell_size), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(sprite)

        center = cell_size // 2
        square_size = cell_size // 3
        draw.rectangle(
            [
                center - square_size,
                center - square_size,
                center + square_size,
                center + square_size,
            ],
            fill=(255, 0, 0, 255),
        )

        sprite_arr = np.array(sprite)
        return sprite_arr[..., :3], sprite_arr[..., 3] > 0

    def _ensure_render_cache(self):
        """Build and retain the static render assets the first time they are needed."""
        if self._render_cache_ready:
            return

        layout = self._get_render_layout()
        self._render_background = self._build_background_image()
        self._render_cell_origins = layout["cell_origins"]
        self._render_goal_sprite, self._render_goal_mask = self._build_goal_sprite()
        self._render_agent_sprite, self._render_agent_mask = self._build_agent_sprite()
        self._render_cache_ready = True

    def _apply_sprite(
        self,
        frame: np.ndarray,
        position: Optional[Tuple[int, int]],
        sprite_rgb: np.ndarray,
        sprite_mask: np.ndarray,
    ) -> None:
        """Stamp a cached sprite into the pre-rendered maze image."""
        if position not in self._render_cell_origins:
            return

        px, py = self._render_cell_origins[position]
        cell_size = self._get_render_layout()["cell_size"]
        region = frame[py:py + cell_size, px:px + cell_size]
        region[sprite_mask] = sprite_rgb[sprite_mask]
    
    def _render_ansi(self, show_goal: bool = True) -> Optional[str]:
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
                elif show_goal and (x, y) == self.goal_position:
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
    
    def _render_rgb(self, show_goal: bool = True) -> np.ndarray:
        """Render the environment as RGB image."""
        self._ensure_render_cache()

        frame = self._render_background.copy()
        if show_goal and self.goal_position != self.DEAD_STATE:
            self._apply_sprite(frame, self.goal_position, self._render_goal_sprite, self._render_goal_mask)
        if self._agent_location != self.DEAD_STATE:
            self._apply_sprite(frame, self._agent_location, self._render_agent_sprite, self._render_agent_mask)

        return frame
    
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
