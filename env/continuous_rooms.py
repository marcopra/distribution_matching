"""
Continuous state space versions of room environments.
The agent moves in continuous 2D space with discrete actions (8 directions).
Goal is reached when agent is within a threshold distance.
Uses Pygame for modern rendering.
"""
import pygame
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register


# Colors (RGB)
COLORS = {
    'background': (245, 245, 250),
    'wall': (70, 80, 110),
    'floor': (255, 255, 255),
    'agent': (220, 80, 80),
    'agent_border': (180, 50, 50),
    'goal': (80, 200, 120),
    'goal_border': (50, 160, 90),
    'text': (60, 60, 60),
}


class ContinuousRoomEnv(gym.Env, ABC):
    """
    Abstract base class for continuous room environments.
    """
    
    ACTIONS = {
        0: (0, 1),    # Up
        1: (0, -1),   # Down
        2: (-1, 0),   # Left
        3: (1, 0),    # Right
        4: (-1, 1),   # Up-Left
        5: (1, 1),    # Up-Right
        6: (-1, -1),  # Down-Left
        7: (1, -1),   # Down-Right
    }
    
    ACTION_NAMES = {
        0: "Up", 1: "Down", 2: "Left", 3: "Right",
        4: "Up-Left", 5: "Up-Right", 6: "Down-Left", 7: "Down-Right"
    }
    
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 10}
    
    def __init__(
        self,
        move_delta: float = 0.3,
        goal_threshold: float = 0.4,
        max_steps: int = 200,
        render_mode: Optional[str] = None,
        show_coordinates: bool = False,
        goal_position: Optional[Tuple[float, float]] = None,
        start_position: Optional[Tuple[float, float]] = None,
        render_resolution: int = 512,
        wall_thickness: float = 0.3,
        agent_radius: float = 0.15,
        num_actions: int = 8,
        dense_reward: bool = False,
        wall_penalty: float = 0.0,
    ):
        super().__init__()
        
        # Validate num_actions
        if num_actions not in [2, 4, 8]:
            raise ValueError(f"num_actions must be 2, 4, or 8, got {num_actions}")
        
        self.num_actions = num_actions
        self.move_delta = move_delta
        self.goal_threshold = goal_threshold
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.show_coordinates = show_coordinates
        self.render_resolution = render_resolution  # Store resolution
        self.wall_thickness = wall_thickness
        self.agent_radius = agent_radius
        self.dense_reward = dense_reward
        self.wall_penalty = wall_penalty
        
        self._fixed_goal_position = goal_position
        self._fixed_start_position = start_position
        
        self.width: float = 0.0
        self.height: float = 0.0
        self.walkable_areas: List[Tuple[float, float, float, float]] = []
        
        self._build_environment()
        
        # Calculate actual walkable bounds (excluding walls)
        min_x = min(area[0] for area in self.walkable_areas)
        min_y = min(area[1] for area in self.walkable_areas)
        max_x = max(area[0] + area[2] for area in self.walkable_areas)
        max_y = max(area[1] + area[3] for area in self.walkable_areas)
        
        self.observation_space = spaces.Box(
            low=np.array([min_x, min_y], dtype=np.float32),
            high=np.array([max_x, max_y], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.num_actions)
        
        self.position: np.ndarray = np.zeros(2, dtype=np.float32)
        self.goal: np.ndarray = np.zeros(2, dtype=np.float32)
        self.steps: int = 0
        self.wall_collision: bool = False
        
        # Validate fixed positions if provided
        if self._fixed_start_position is not None:
            self._validate_position(self._fixed_start_position, "start")
        if self._fixed_goal_position is not None:
            self._validate_position(self._fixed_goal_position, "goal")
        
        self._pygame_initialized = False
        self._screen = None
        self._clock = None

    def _init_pygame(self):
        if self._pygame_initialized:
            return
        
        pygame.init()
        self._pygame_initialized = True
        if self.render_mode == "human":
            self._screen = pygame.display.set_mode(
                (self.render_resolution, self.render_resolution)
            )
            pygame.display.set_caption("Continuous Room Environment")
        self._clock = pygame.time.Clock()

    def _validate_position(self, position: Tuple[float, float], name: str):
        """Validate that a position (with agent radius) is fully inside walkable areas."""
        x, y = position
        if not self._is_position_valid_with_radius(x, y):
            raise ValueError(
                f"{name.capitalize()} position ({x:.2f}, {y:.2f}) with agent radius "
                f"{self.agent_radius:.2f} is not fully inside any walkable area. "
                f"Walkable areas: {self.walkable_areas}"
            )
    
    @abstractmethod
    def _build_environment(self) -> None:
        pass
    
    def _point_in_rect(self, x: float, y: float, rect: Tuple[float, float, float, float]) -> bool:
        rx, ry, rw, rh = rect
        return rx <= x <= rx + rw and ry <= y <= ry + rh
    
    def _circle_in_rect(self, x: float, y: float, radius: float, rect: Tuple[float, float, float, float]) -> bool:
        """Check if a circle (agent) is fully inside a rectangle."""
        rx, ry, rw, rh = rect
        return (rx + radius <= x <= rx + rw - radius and 
                ry + radius <= y <= ry + rh - radius)
    
    def _is_walkable(self, x: float, y: float) -> bool:
        """Check if position is walkable considering agent radius."""
        for area in self.walkable_areas:
            if self._circle_in_rect(x, y, self.agent_radius, area):
                return True
        return False
    
    def _is_position_valid_with_radius(self, x: float, y: float) -> bool:
        """
        Check if the agent (as a circle) can be at position (x, y).
        The entire circle must be inside at least one walkable area,
        OR the circle can span multiple connected walkable areas.
        """
        # Check if center + radius fits in any single walkable area
        for area in self.walkable_areas:
            if self._circle_in_rect(x, y, self.agent_radius, area):
                return True
        
        # Check if all edges of the agent circle are in walkable areas
        # Sample points around the circle perimeter
        num_samples = 8
        for i in range(num_samples):
            angle = 2 * np.pi * i / num_samples
            edge_x = x + self.agent_radius * np.cos(angle)
            edge_y = y + self.agent_radius * np.sin(angle)
            
            # Check if this edge point is in any walkable area
            edge_in_walkable = False
            for area in self.walkable_areas:
                if self._point_in_rect(edge_x, edge_y, area):
                    edge_in_walkable = True
                    break
            
            if not edge_in_walkable:
                return False
        
        # Also check the center
        center_in_walkable = False
        for area in self.walkable_areas:
            if self._point_in_rect(x, y, area):
                center_in_walkable = True
                break
        
        return center_in_walkable
    
    def _move(self, action: int) -> Tuple[np.ndarray, bool]:
        """
        Move the agent based on action.
        Returns: (new_position, wall_collision)
        """
        # Validate action is within available actions
        if action >= self.num_actions:
            raise ValueError(f"Action {action} is invalid for num_actions={self.num_actions}")
        
        direction = np.array(self.ACTIONS[action], dtype=np.float32)
        if action >= 4:
            direction = direction / np.sqrt(2)
        
        new_position = self.position + direction * self.move_delta
        wall_collision = False
        
        # Check if new position is valid considering agent radius
        if self._is_position_valid_with_radius(new_position[0], new_position[1]):
            # Check path for collisions (sample intermediate points)
            collision_free = True
            for t in np.linspace(0, 1, 5):
                mid = self.position + t * (new_position - self.position)
                if not self._is_position_valid_with_radius(mid[0], mid[1]):
                    collision_free = False
                    break
            
            if collision_free:
                return new_position, wall_collision
        
        # Movement was blocked - mark collision
        wall_collision = True
        
        # If direct movement fails, try sliding along walls
        # Try moving only in X direction
        x_only = np.array([new_position[0], self.position[1]], dtype=np.float32)
        if self._is_position_valid_with_radius(x_only[0], x_only[1]):
            return x_only, wall_collision
        
        # Try moving only in Y direction
        y_only = np.array([self.position[0], new_position[1]], dtype=np.float32)
        if self._is_position_valid_with_radius(y_only[0], y_only[1]):
            return y_only, wall_collision
        
        # If all attempts fail, stay in current position
        return self.position.copy(), wall_collision
    
    def _distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.position - self.goal))
    
    def _is_goal_reached(self) -> bool:
        return self._distance_to_goal() <= self.goal_threshold
    
    def _get_valid_positions(self) -> List[Tuple[float, float]]:
        """Sample valid positions from walkable areas, considering agent radius."""
        positions = []
        step = self.move_delta * 0.5
        
        for area in self.walkable_areas:
            ax, ay, aw, ah = area
            # Margin includes agent radius to ensure the full circle fits
            margin = max(self.move_delta * 0.5, self.agent_radius)
            x = ax + margin
            while x <= ax + aw - margin:
                y = ay + margin
                while y <= ay + ah - margin:
                    positions.append((x, y))
                    y += step
                x += step
        
        return positions
    
    def _sample_valid_position(self) -> np.ndarray:
        positions = self._get_valid_positions()
        if not positions:
            raise ValueError("No valid positions")
        idx = self.np_random.integers(len(positions))
        return np.array(positions[idx], dtype=np.float32)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        if self._fixed_start_position is not None:
            self.position = np.array(self._fixed_start_position, dtype=np.float32)
        else:
            self.position = self._sample_valid_position()
        
        # Validate position is within observation space bounds
        assert self.observation_space.contains(self.position), \
            f"Start position {self.position} is outside observation space bounds"
        
        if self._fixed_goal_position is not None:
            self.goal = np.array(self._fixed_goal_position, dtype=np.float32)
        else:
            self.goal = self._sample_valid_position()
            attempts = 0
            while np.linalg.norm(self.goal - self.position) < self.goal_threshold * 2 and attempts < 100:
                self.goal = self._sample_valid_position()
                attempts += 1
        
        # Validate goal is within observation space bounds
        assert self.observation_space.contains(self.goal), \
            f"Goal position {self.goal} is outside observation space bounds"
        
        self.steps = 0
        
        info = {
            "position": self.position.copy(),
            "goal": self.goal.copy(),
            "distance_to_goal": self._distance_to_goal()
        }
        return self.position.astype(np.float32).copy(), info
    
    def step(self, action: int):
        self.steps += 1
        self.position, self.wall_collision = self._move(action)
        
        terminated = self._is_goal_reached()
        truncated = self.steps >= self.max_steps
        
        # Calculate reward
        if self.dense_reward:
            # terminated = False  # Disable termination for dense reward
            # otherwise, the maximum reward for the episode is always staying close to the target without
            # actually reaching it.

            # Dense reward: exponential negative Euclidean distance to goal
            distance = self._distance_to_goal()
            reward = np.exp(-distance)
            reward += 1000 if terminated else 0.0
            # Apply wall penalty if collision occurred
            if self.wall_collision and self.wall_penalty > 0.0:
                reward -= self.wall_penalty
        else:
            # Sparse reward: 1.0 if goal reached, 0.0 otherwise
            reward = 0.0 if terminated else -1.0
        
        
        info = {
            "position": self.position.copy(),
            "goal": self.goal.copy(),
            "distance_to_goal": self._distance_to_goal(),
            "success": terminated,
            "wall_collision": self.wall_collision
        }
        return self.position.astype(np.float32).copy(), reward, terminated, truncated, info
    
    def step_from_position(self, position: np.ndarray, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Simulate a step from an arbitrary position without changing environment state.
        Useful for ideal dataset population.
        
        Args:
            position: Starting position [x, y]
            action: Action to take
            
        Returns:
            Tuple of (next_position, reward, terminated, truncated, info)
        """
        # Save current state
        original_position = self.position.copy()
        original_steps = self.steps
        
        # Set to desired position
        self.position = position.astype(np.float32)
        
        # Execute step
        new_position, wall_collision = self._move(action)
        
        # Compute reward and termination (simplified for ideal case)
        terminated = self._is_goal_reached()
        truncated = False
        
        if self.dense_reward:
            distance = self._distance_to_goal()
            reward = np.exp(-distance)
            if terminated:
                reward += 1000.0
            if wall_collision and self.wall_penalty > 0.0:
                reward -= self.wall_penalty
        else:
            reward = 0.0 if terminated else -1.0
        
        info = {
            "position": new_position.copy(),
            "goal": self.goal.copy(),
            "distance_to_goal": float(np.linalg.norm(new_position - self.goal)),
            "success": terminated,
            "wall_collision": wall_collision
        }
        
        # Restore original state
        self.position = original_position
        self.steps = original_steps
        
        return new_position, reward, terminated, truncated, info
    
    def render_from_position(
        self,
        position: np.ndarray,
        goal: Optional[np.ndarray] = None,
        show_goal: bool = False,
    ) -> np.ndarray:
        """
        Render the environment from a specific position without changing state.
        
        Args:
            position: Agent position [x, y]
            goal: Optional goal position [x, y]. If None, uses current goal.
            
        Returns:
            RGB array of the rendered image [H, W, C]
        """
        if self.render_mode != "rgb_array":
            raise ValueError("render_from_position requires render_mode='rgb_array'")
        
        # Save current state
        original_position = self.position.copy()
        original_goal = self.goal.copy() if goal is not None else None
        
        # Set temporary state
        self.position = position.astype(np.float32)
        if goal is not None:
            self.goal = goal.astype(np.float32)
        
        # Render
        image = self.render(show_goal=show_goal)
        
        # Restore original state
        self.position = original_position
        if original_goal is not None:
            self.goal = original_goal
        
        return image
    
    def render(self, show_goal: bool = False) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None
        
        self._init_pygame()
        
        
        surface = pygame.Surface((self.render_resolution, self.render_resolution))
        surface.fill(COLORS['wall'])
        
        scale = self.render_resolution / max(self.width, self.height)
        offset_x = (self.render_resolution - self.width * scale) / 2
        offset_y = (self.render_resolution - self.height * scale) / 2
        
        def to_screen(x, y):
            return (
                int(offset_x + x * scale),
                int(self.render_resolution - offset_y - y * scale)
            )
        
        def to_screen_rect(rx, ry, rw, rh):
            sx, sy = to_screen(rx, ry + rh)
            return pygame.Rect(sx, sy, int(rw * scale), int(rh * scale))
        
        # Draw walkable areas (floor)
        for area in self.walkable_areas:
            rect = to_screen_rect(*area)
            pygame.draw.rect(surface, COLORS['floor'], rect)
        
        # Draw goal only for auxiliary/debug images. The agent-facing pixel
        # observation stays goal-hidden, matching the PointMaze debug setup.
        if show_goal:
            goal_screen = to_screen(self.goal[0], self.goal[1])
            goal_radius = max(1, int(self.goal_threshold * scale))
            pygame.draw.circle(surface, COLORS['goal'], goal_screen, goal_radius)
            pygame.draw.circle(surface, COLORS['goal_border'], goal_screen, goal_radius, 3)
        
        # Draw agent (red) - using actual agent_radius for visual consistency
        agent_screen = to_screen(self.position[0], self.position[1])
        agent_radius_pixels = int(self.agent_radius * scale)
        pygame.draw.circle(surface, COLORS['agent'], agent_screen, agent_radius_pixels)
        pygame.draw.circle(surface, COLORS['agent_border'], agent_screen, agent_radius_pixels, 2)
        
        if self.show_coordinates:
            font = pygame.font.SysFont('Arial', 14)
            pos_text = font.render(f"Pos: ({self.position[0]:.1f}, {self.position[1]:.1f})", True, COLORS['text'])
            goal_text = font.render(f"Goal: ({self.goal[0]:.1f}, {self.goal[1]:.1f})", True, COLORS['text'])
            surface.blit(pos_text, (10, 10))
            surface.blit(goal_text, (10, 28))
        
        if self.render_mode == "human":
            self._screen.blit(surface, (0, 0))
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
        
        return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))

    def render_observation(self) -> Optional[np.ndarray]:
        return self.render(show_goal=False)

    def render_image_observation(self) -> Optional[np.ndarray]:
        return self.render(show_goal=True)
    
    def close(self):
        if self._pygame_initialized:
            
            pygame.quit()
            self._pygame_initialized = False


class ContinuousCorridorEnv(ContinuousRoomEnv):
    """Continuous 1D corridor with two discrete direct-velocity actions."""

    ACTION_NAMES = {0: "Left", 1: "Right"}

    def __init__(
        self,
        corridor_length: float = 10.0,
        corridor_width: float = 1.0,
        max_velocity: Optional[float] = None,
        horizon: Optional[int] = None,
        max_steps: int = 200,
        start_position: Optional[Tuple[float, float]] = None,
        goal_position: Optional[Tuple[float, float]] = None,
        **kwargs,
    ):
        if corridor_length <= 0.0:
            raise ValueError(f"corridor_length must be positive, got {corridor_length}")
        if corridor_width <= 0.0:
            raise ValueError(f"corridor_width must be positive, got {corridor_width}")

        self.corridor_length = float(corridor_length)
        self.corridor_width = float(corridor_width)
        resolved_max_steps = int(horizon) if horizon is not None else int(max_steps)
        if resolved_max_steps <= 0:
            raise ValueError(f"horizon/max_steps must be positive, got {resolved_max_steps}")
        self.horizon = resolved_max_steps

        move_delta = kwargs.pop("move_delta", 0.3)
        self.max_velocity = float(move_delta if max_velocity is None else max_velocity)
        if self.max_velocity <= 0.0:
            raise ValueError(f"max_velocity must be positive, got {self.max_velocity}")

        super().__init__(
            move_delta=self.max_velocity,
            max_steps=resolved_max_steps,
            start_position=start_position,
            goal_position=goal_position,
            num_actions=2,
            **kwargs,
        )

        min_x, max_x = self._valid_x_bounds()
        center_y = self._center_y()
        y_eps = np.float32(1e-6)
        self.observation_space = spaces.Box(
            low=np.array([min_x, center_y - y_eps], dtype=np.float32),
            high=np.array([max_x, center_y + y_eps], dtype=np.float32),
            dtype=np.float32,
        )

    def _build_environment(self) -> None:
        wt = self.wall_thickness
        self.width = self.corridor_length + 2.0 * wt
        self.height = self.corridor_width + 2.0 * wt
        self.walkable_areas = [(wt, wt, self.corridor_length, self.corridor_width)]

    def _center_y(self) -> float:
        return self.wall_thickness + 0.5 * self.corridor_width

    def _valid_x_bounds(self) -> Tuple[float, float]:
        min_x = self.wall_thickness + self.agent_radius
        max_x = self.wall_thickness + self.corridor_length - self.agent_radius
        if max_x < min_x:
            raise ValueError(
                "corridor_length must be at least 2 * agent_radius so the agent can fit "
                f"(length={self.corridor_length}, agent_radius={self.agent_radius})"
            )
        return min_x, max_x

    def _center_position(self) -> np.ndarray:
        min_x, max_x = self._valid_x_bounds()
        return np.array([0.5 * (min_x + max_x), self._center_y()], dtype=np.float32)

    def _default_goal_position(self) -> np.ndarray:
        _, max_x = self._valid_x_bounds()
        return np.array([max_x, self._center_y()], dtype=np.float32)

    def _as_corridor_position(self, position, name: str) -> np.ndarray:
        arr = np.asarray(position, dtype=np.float32).reshape(-1)
        if arr.size == 1:
            arr = np.array([arr[0], self._center_y()], dtype=np.float32)
        if arr.size != 2:
            raise ValueError(f"{name} position must have shape (2,) or be an x scalar, got {arr.shape}")

        min_x, max_x = self._valid_x_bounds()
        center_y = self._center_y()
        x_value = float(arr[0])
        eps = 1e-6
        if x_value < min_x - eps or x_value > max_x + eps:
            raise ValueError(
                f"{name} x={x_value:.3f} is outside valid corridor bounds "
                f"[{min_x:.3f}, {max_x:.3f}]"
            )
        if not np.isclose(float(arr[1]), center_y, atol=1e-6):
            raise ValueError(
                f"{name} y={float(arr[1]):.3f} must equal corridor center y={center_y:.3f}"
            )
        return np.array([np.clip(x_value, min_x, max_x), center_y], dtype=np.float32)

    def _validate_position(self, position: Tuple[float, float], name: str):
        self._as_corridor_position(position, name)

    def _move(self, action: int) -> Tuple[np.ndarray, bool]:
        action_idx = int(action)
        if action_idx not in (0, 1):
            raise ValueError("ContinuousCorridor action must be 0 (left) or 1 (right)")

        direction = -1.0 if action_idx == 0 else 1.0
        raw_next_x = float(self.position[0]) + direction * self.max_velocity
        min_x, max_x = self._valid_x_bounds()
        next_x = float(np.clip(raw_next_x, min_x, max_x))
        wall_collision = not np.isclose(raw_next_x, next_x)
        return np.array([next_x, self._center_y()], dtype=np.float32), wall_collision

    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)
        info["xy"] = self.position.copy()
        info["horizon"] = self.horizon
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        gym.Env.reset(self, seed=seed)

        if options is not None and "start_position" in options:
            self.position = self._as_corridor_position(options["start_position"], "start")
        elif self._fixed_start_position is not None:
            self.position = self._as_corridor_position(self._fixed_start_position, "start")
        else:
            self.position = self._center_position()

        if options is not None and "goal_position" in options:
            self.goal = self._as_corridor_position(options["goal_position"], "goal")
        elif self._fixed_goal_position is not None:
            self.goal = self._as_corridor_position(self._fixed_goal_position, "goal")
        else:
            self.goal = self._default_goal_position()

        self.steps = 0
        self.wall_collision = False
        info = {
            "position": self.position.copy(),
            "xy": self.position.copy(),
            "goal": self.goal.copy(),
            "distance_to_goal": self._distance_to_goal(),
            "wall_collision": self.wall_collision,
        }
        return self.position.astype(np.float32).copy(), info

    def step_from_position(self, position: np.ndarray, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        original_position = self.position.copy()
        original_steps = self.steps
        original_collision = self.wall_collision

        self.position = self._as_corridor_position(position, "debug")
        new_position, wall_collision = self._move(action)
        self.position = new_position

        terminated = self._is_goal_reached()
        truncated = False
        if self.dense_reward:
            reward = np.exp(-self._distance_to_goal())
            reward += 1000.0 if terminated else 0.0
            if wall_collision and self.wall_penalty > 0.0:
                reward -= self.wall_penalty
        else:
            reward = 0.0 if terminated else -1.0

        info = {
            "position": new_position.copy(),
            "xy": new_position.copy(),
            "goal": self.goal.copy(),
            "distance_to_goal": self._distance_to_goal(),
            "success": terminated,
            "wall_collision": wall_collision,
        }

        self.position = original_position
        self.steps = original_steps
        self.wall_collision = original_collision
        return new_position, reward, terminated, truncated, info

    def get_debug_coordinates(self) -> Dict[str, np.ndarray]:
        return {
            "xy": self.position.copy(),
            "fixed_start": (
                self._center_position()
                if self._fixed_start_position is None
                else self._as_corridor_position(self._fixed_start_position, "start")
            ),
            "fixed_goal": (
                self._default_goal_position()
                if self._fixed_goal_position is None
                else self._as_corridor_position(self._fixed_goal_position, "goal")
            ),
        }

    def get_debug_maze_layout(self) -> Dict[str, np.ndarray]:
        wt = self.wall_thickness
        min_x, max_x = self._valid_x_bounds()
        lower = np.array([min_x, self._center_y()], dtype=np.float32)
        upper = np.array([max_x, self._center_y()], dtype=np.float32)
        left_wall = np.array([0.0, wt, wt, self.corridor_width], dtype=np.float32)
        right_wall = np.array([wt + self.corridor_length, wt, wt, self.corridor_width], dtype=np.float32)
        return {
            "maze_lower": lower,
            "maze_upper": upper,
            "wall_rectangles": np.stack([left_wall, right_wall], axis=0),
        }

    def get_debug_plot_bounds(self) -> Dict[str, np.ndarray]:
        min_x, max_x = self._valid_x_bounds()
        y_margin = max(0.5 * self.corridor_width, self.agent_radius)
        center_y = self._center_y()
        return {
            "lower": np.array([min_x, center_y - y_margin], dtype=np.float32),
            "upper": np.array([max_x, center_y + y_margin], dtype=np.float32),
        }


class ContinuousSingleRoomEnv(ContinuousRoomEnv):
    """Single room with continuous state space."""
    
    def __init__(self, room_size: float = 5.0, **kwargs):
        self.room_size = room_size
        super().__init__(**kwargs)
    
    def _build_environment(self):
        wt = self.wall_thickness
        self.width = self.room_size + 2 * wt
        self.height = self.room_size + 2 * wt
        self.walkable_areas = [(wt, wt, self.room_size, self.room_size)]


class ContinuousTwoRoomsEnv(ContinuousRoomEnv):
    """Two rooms connected by a corridor."""
    
    def __init__(
        self,
        room_size: float = 5.0,
        corridor_width: float = 1.0,
        corridor_length: float = 2.0,
        **kwargs
    ):
        self.room_size = room_size
        self.corridor_width = corridor_width
        self.corridor_length = corridor_length
        super().__init__(**kwargs)
    
    def _build_environment(self):
        wt = self.wall_thickness
        rs = self.room_size
        cw = self.corridor_width
        cl = self.corridor_length
        
        self.width = 2 * rs + cl + 2 * wt
        self.height = rs + 2 * wt
        
        corridor_y = wt + (rs - cw) / 2
        
        self.walkable_areas = [
            (wt, wt, rs, rs),
            (wt + rs, corridor_y, cl, cw),
            (wt + rs + cl, wt, rs, rs),
        ]


class ContinuousFourRoomsEnv(ContinuousRoomEnv):
    """Four rooms connected by narrow corridors through the walls."""
    
    def __init__(
        self,
        room_size: float,
        corridor_width: float = 1.5,
        corridor_offset: float = 1.2,
        wall_thickness: float = 0.6,
        agent_radius: float = 0.15,
        **kwargs
    ):
        self.room_size = room_size
        self.corridor_width = corridor_width
        self.corridor_offset = corridor_offset
        super().__init__(wall_thickness=wall_thickness, agent_radius=agent_radius, **kwargs)
    
    def _build_environment(self):
        wt = self.wall_thickness
        rs = self.room_size
        cw = self.corridor_width
        offset = self.corridor_offset
        
        # Total size: 2 rooms + internal wall + outer walls
        self.width = 2 * rs + wt + 2 * wt
        self.height = 2 * rs + wt + 2 * wt
        
        # Four rooms
        r_bl = (wt, wt, rs, rs)  # Bottom-left
        r_br = (wt + rs + wt, wt, rs, rs)  # Bottom-right
        r_tl = (wt, wt + rs + wt, rs, rs)  # Top-left
        r_tr = (wt + rs + wt, wt + rs + wt, rs, rs)  # Top-right
        
        # Overlap amount to prevent gaps
        overlap = 0.05
        
        # Narrow corridors through walls - extended to overlap with rooms
        # corridor_offset now refers to the CENTER of the corridor from the edge
        # Left side: corridor between bottom-left and top-left (through horizontal wall)
        left_h_corridor = (wt + offset - cw / 2, wt + rs - overlap, cw, wt + 2 * overlap)
        
        # Right side: corridor between bottom-right and top-right (through horizontal wall)
        right_h_corridor = (wt + rs + wt + rs - offset - cw / 2, wt + rs - overlap, cw, wt + 2 * overlap)
        
        # Bottom: corridor between bottom-left and bottom-right (through vertical wall)
        bottom_v_corridor = (wt + rs - overlap, wt + rs - offset - cw / 2, wt + 2 * overlap, cw)
        
        # Top: corridor between top-left and top-right (through vertical wall)
        top_v_corridor = (wt + rs - overlap, wt + rs + wt + offset - cw / 2, wt + 2 * overlap, cw)
        
        self.walkable_areas = [
            r_bl, r_br, r_tl, r_tr,
            left_h_corridor, right_h_corridor,
            bottom_v_corridor, top_v_corridor
        ]


class ContinuousMultipleRoomsEnv(ContinuousRoomEnv):
    """Multiple rooms connected to a main horizontal corridor."""
    
    def __init__(
        self,
        num_rooms: int = 4,
        room_size: float = 3.0,
        main_corridor_height: float = 1.5,
        connector_width: float = 1.0,
        connector_length: float = 1.0,
        **kwargs
    ):
        self.num_rooms = num_rooms
        self.room_size = room_size
        self.main_corridor_height = main_corridor_height
        self.connector_width = connector_width
        self.connector_length = connector_length
        super().__init__(**kwargs)
    
    def _build_environment(self):
        wt = self.wall_thickness
        rs = self.room_size
        mch = self.main_corridor_height
        conn_w = self.connector_width
        conn_l = self.connector_length
        n = self.num_rooms
        
        room_spacing = rs + wt
        
        self.width = n * room_spacing + wt
        self.height = rs + conn_l + mch + 2 * wt
        
        # Overlap amount to prevent gaps
        overlap = 0.05
        
        main_corridor_y = wt + rs + conn_l
        main_corridor = (wt, main_corridor_y, self.width - 2 * wt, mch)
        
        self.walkable_areas = [main_corridor]
        
        for i in range(n):
            room_x = wt + i * room_spacing
            room = (room_x, wt, rs, rs)
            self.walkable_areas.append(room)
            
            # Connector extended to overlap with both room and main corridor
            conn_x = room_x + (rs - conn_w) / 2
            conn_y = wt + rs - overlap  # Start slightly inside the room
            connector = (conn_x, conn_y, conn_w, conn_l + 2 * overlap)  # Extend into main corridor
            self.walkable_areas.append(connector)


# Register environments
def _register_envs():
    envs_to_register = [
        ("ContinuousCorridor-v0", "env.continuous_rooms:ContinuousCorridorEnv"),
        ("ContinuousSingleRoom-v0", "env.continuous_rooms:ContinuousSingleRoomEnv"),
        ("ContinuousTwoRooms-v0", "env.continuous_rooms:ContinuousTwoRoomsEnv"),
        ("ContinuousFourRooms-v0", "env.continuous_rooms:ContinuousFourRoomsEnv"),
        ("ContinuousMultipleRooms-v0", "env.continuous_rooms:ContinuousMultipleRoomsEnv"),
    ]
    
    for env_id, entry_point in envs_to_register:
        try:
            register(id=env_id, entry_point=entry_point)
        except gym.error.Error:
            pass

_register_envs()


if __name__ == "__main__":
    import imageio
    from pathlib import Path
    
    def test_and_record(env_class, env_name: str, output_dir: str = "videos/continuous_rooms", **kwargs):
        print(f"\nTesting {env_name}...")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        env = env_class(render_mode="rgb_array", show_coordinates=True, **kwargs)
        
        obs, info = env.reset(seed=42)
        print(f"  World: {env.width:.1f} x {env.height:.1f}, Areas: {len(env.walkable_areas)}")
        
        frames = [env.render()]
        actions = [6, 6, 6, 2, 6]
        for action in actions: #for i in range(100):
            # if i < len(actions):
            #     action = actions[i]
            # else:
            #     action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            frames.append(env.render())
            if terminated or truncated:
                break
        
        video_path = f"{output_dir}/{env_name}.mp4"
        imageio.mimsave(video_path, frames, fps=15)
        print(f"  Saved: {video_path}")
        env.close()
    
    environments = [
        # (ContinuousSingleRoomEnv, "ContinuousSingleRoom", {"room_size": 5.0}),
        # (ContinuousTwoRoomsEnv, "ContinuousTwoRooms", {"room_size": 4.0, "corridor_width": 1.0, "corridor_length": 2.0}),
        (ContinuousFourRoomsEnv, "ContinuousFourRooms", {"room_size": 5.0, "max_steps": 100, "move_delta": 0.5, "goal_threshold": 0.5, "goal_position": [4.5, 4.5], "start_position": [0.9, 0.9]}),
        # (ContinuousMultipleRoomsEnv, "ContinuousMultipleRooms", {"num_rooms": 4, "room_size": 3.0}),
    ]
    
    print("="*60)
    print("CONTINUOUS ROOM ENVIRONMENTS - TEST")
    print("="*60)
    
    for env_class, env_name, kwargs in environments:
        try:
            test_and_record(env_class, env_name, **kwargs)
        except Exception as e:
            print(f"✗ {env_name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✓ Done!")
