import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, List


class ErgodicEnv(gym.Wrapper):
    """
    Environment wrapper to ignore the done signal
    assuming the MDP is ergodic.
    """
    def __init__(self, env):
        super().__init__(env)

    def step(self, a):
        state, reward, terminated, truncated, info = super().step(a)
        terminated = False
        return state, reward, terminated, truncated, info

class CoreStateEnv(gym.Wrapper):
    """
    Environment wrapper to filter a subset of the full state space.
    This can be useful in Mujoco to recover the "core" state
    which is normally obtained using the env.env.state_vector() call.
    """
    def __init__(self, env, features_range):
        super().__init__(env)
        self.features_range = features_range
        self.num_features = len(features_range)

    def core_state(self, s):
        return s[self.features_range]

    def reset(self):
        s = super().reset()
        return self.core_state(s)

    def step(self, a):
        s, r, terminated, truncated, i = super().step(a)
        s = self.core_state(s)
        return s, r, terminated, truncated, i

class CustomRewardEnv(gym.Wrapper):
    """
    Environment wrapper to provide a custom reward function
    and episode termination.
    """
    def __init__(self, env, get_reward):
        super().__init__(env)
        self.get_reward = get_reward

    def step(self, a):
        s, r, terminated, truncated, i = super().step(a)
        r, terminated = self.get_reward(s, r, terminated, truncated, i)
        return s, r, terminated, truncated, i

class DiscretizedContinuousEnv(gym.Wrapper):
    """
    Wrapper that discretizes a continuous room environment into a grid-based discrete environment.
    
    This wrapper converts continuous 2D positions into discrete cell indices, making
    continuous environments compatible with algorithms designed for discrete state spaces.
    
    The wrapper:
    - Divides walkable areas into a grid of cells
    - Uses 4 discrete actions (up, down, left, right) that move to adjacent cell centers
    - Provides discrete state observations (cell indices)
    - Maintains state_to_idx, idx_to_state, cells mappings like discrete room envs
    """
    
    DEAD_STATE = (-1, -1)
    
    # Action mapping: 0=up, 1=down, 2=left, 3=right (matching discrete room envs)
    DISCRETE_ACTIONS = {
        0: (0, 1),    # Up
        1: (0, -1),   # Down
        2: (-1, 0),   # Left
        3: (1, 0),    # Right
        4: (-1, 1),   # Up-Left
        5: (1, 1),    # Up-Right
        6: (-1, -1),  # Down-Left
        7: (1, -1),   # Down-Right
    }
    
    def __init__(
        self,
        env: gym.Env,
        cell_size: float = 1.0,
        lava: bool = False
    ):
        """
        Args:
            env: Continuous room environment to wrap
            cell_size: Size of each discrete cell in continuous coordinates
            lava: If True, invalid moves lead to a dead state
        """
        super().__init__(env)
        raise NotImplementedError("DiscretizedContinuousEnv is currently disabled. Da vedere bene anche la discretizzazione dei muri.")
        
        self.cell_size = cell_size
        self.lava = lava
        
        # Access the unwrapped environment for attributes
        self._base_env = env.unwrapped
        
        # Build discrete grid from walkable areas
        self.cells: List[Tuple[int, int]] = []
        self.state_to_idx: Dict[Tuple[int, int], int] = {}
        self.idx_to_state: Dict[int, Tuple[int, int]] = {}
        self._cell_centers: Dict[Tuple[int, int], Tuple[float, float]] = {}
        
        self._build_discrete_grid()
        
        # Add dead state if lava is enabled
        if self.lava:
            self._add_cell(self.DEAD_STATE)
            # Dead state center is outside the environment
            self._cell_centers[self.DEAD_STATE] = (-1.0, -1.0)
        
        self.n_states = len(self.cells)
        
        # Override observation and action spaces
        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(4)  # 4 discrete actions
        
        self._agent_cell: Tuple[int, int] = (0, 0)
        self._goal_cell: Optional[Tuple[int, int]] = None
        self._step_count = 0
    
    @property
    def wall_thickness(self) -> float:
        """Forward wall_thickness to base environment."""
        return self._base_env.wall_thickness
    
    @property
    def walkable_areas(self) -> List[Tuple[float, float, float, float]]:
        """Forward walkable_areas to base environment."""
        return self._base_env.walkable_areas
    
    @property
    def position(self) -> np.ndarray:
        """Get position from base environment."""
        return self._base_env.position
    
    @position.setter
    def position(self, value: np.ndarray):
        """Set position in base environment."""
        self._base_env.position = value
    
    @property
    def goal(self) -> np.ndarray:
        """Get goal from base environment."""
        return self._base_env.goal
    
    @goal.setter
    def goal(self, value: np.ndarray):
        """Set goal in base environment."""
        self._base_env.goal = value
    
    def _build_discrete_grid(self):
        """Build discrete grid cells from continuous walkable areas."""
        # Find bounds of all walkable areas
        min_x = min(area[0] for area in self.walkable_areas)
        min_y = min(area[1] for area in self.walkable_areas)
        max_x = max(area[0] + area[2] for area in self.walkable_areas)
        max_y = max(area[1] + area[3] for area in self.walkable_areas)
        
        # Calculate grid bounds in cell coordinates
        # Offset to start grid at (0,0)
        self._grid_offset_x = min_x
        self._grid_offset_y = min_y
        
        # Generate all potential cells and check if they're walkable
        num_cells_x = int(np.ceil((max_x - min_x) / self.cell_size))
        num_cells_y = int(np.ceil((max_y - min_y) / self.cell_size))
        
        for cx in range(num_cells_x):
            for cy in range(num_cells_y):
                # Calculate cell center in continuous coordinates
                center_x = min_x + (cx + 0.5) * self.cell_size
                center_y = min_y + (cy + 0.5) * self.cell_size
                
                # Check if cell center is in any walkable area
                if self._is_center_walkable(center_x, center_y):
                    cell = (cx, cy)
                    self._add_cell(cell)
                    self._cell_centers[cell] = (center_x, center_y)
    
    def _is_center_walkable(self, x: float, y: float) -> bool:
        """Check if a point is inside any walkable area."""
        for area in self._base_env.walkable_areas:
            ax, ay, aw, ah = area
            if ax <= x <= ax + aw and ay <= y <= ay + ah:
                return True
        return False
    
    def _add_cell(self, cell: Tuple[int, int]):
        """Add a cell to the grid."""
        if cell not in self.state_to_idx:
            idx = len(self.cells)
            self.cells.append(cell)
            self.state_to_idx[cell] = idx
            self.idx_to_state[idx] = cell
    
    def _continuous_to_cell(self, pos: np.ndarray) -> Tuple[int, int]:
        """Convert continuous position to discrete cell coordinates."""
        cx = int((pos[0] - self._grid_offset_x) / self.cell_size)
        cy = int((pos[1] - self._grid_offset_y) / self.cell_size)
        cell = (cx, cy)
        
        # If not a valid cell, find nearest valid cell
        if cell not in self.state_to_idx or cell == self.DEAD_STATE:
            min_dist = float('inf')
            nearest_cell = self.cells[0]
            for c in self.cells:
                if c == self.DEAD_STATE:
                    continue
                center = self._cell_centers[c]
                dist = (pos[0] - center[0])**2 + (pos[1] - center[1])**2
                if dist < min_dist:
                    min_dist = dist
                    nearest_cell = c
            return nearest_cell
        
        return cell
    
    def _cell_to_continuous(self, cell: Tuple[int, int]) -> np.ndarray:
        """Convert discrete cell to continuous position (cell center)."""
        if cell in self._cell_centers:
            center = self._cell_centers[cell]
            return np.array([center[0], center[1]], dtype=np.float32)
        # Fallback
        return np.array([
            self._grid_offset_x + (cell[0] + 0.5) * self.cell_size,
            self._grid_offset_y + (cell[1] + 0.5) * self.cell_size
        ], dtype=np.float32)
    
    def step_from(self, cell: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        Compute next cell from current cell and action.
        Matches the interface of discrete room environments.
        """
        if self.lava and cell == self.DEAD_STATE:
            return self.DEAD_STATE
        
        direction = self.DISCRETE_ACTIONS[action]
        next_cell = (cell[0] + direction[0], cell[1] + direction[1])
        
        if next_cell in self.state_to_idx and next_cell != self.DEAD_STATE:
            return next_cell
        else:
            if self.lava:
                return self.DEAD_STATE
            return cell  # Stay in place
    
    def _get_obs(self) -> int:
        """Get current observation (state index)."""
        return self.state_to_idx[self._agent_cell]
    
    def _get_info(self) -> Dict:
        """Get auxiliary information."""
        return {
            "agent_position": self._agent_cell,
            "state_index": self.state_to_idx[self._agent_cell],
            "step_count": self._step_count,
            "continuous_position": self._cell_to_continuous(self._agent_cell),
            "goal_cell": self._goal_cell,
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment."""
        # Reset underlying continuous environment
        obs, info = self.env.reset(seed=seed, options=options)
        self._step_count = 0
        
        # Convert continuous positions to discrete cells
        self._agent_cell = self._continuous_to_cell(self._base_env.position)
        self._goal_cell = self._continuous_to_cell(self._base_env.goal)
        
        # Handle start_position from options
        if options is not None:
            if "start_state" in options:
                start_idx = options["start_state"]
                self._agent_cell = self.idx_to_state[start_idx]
            elif "start_position" in options:
                start_pos = options["start_position"]
                if isinstance(start_pos, int):
                    self._agent_cell = self.idx_to_state[start_pos]
                else:
                    self._agent_cell = start_pos
        
        # Set underlying environment position to cell center
        self.env.position = self._cell_to_continuous(self._agent_cell)
        self.env.goal = self._cell_to_continuous(self._goal_cell)
        
        # Store goal position for compatibility
        self.goal_position = self._goal_cell
        self.start_position = self._agent_cell
        
        return self._get_obs(), self._get_info()
    
    def step(self, action: int):
        """Execute one discrete step."""
        self._step_count += 1
        
        # Compute next cell
        self._agent_cell = self.step_from(self._agent_cell, action)
        
        # Update underlying environment position
        self.env.position = self._cell_to_continuous(self._agent_cell)
        
        # Check termination
        terminated = self._agent_cell == self._goal_cell
        in_dead_state = self.lava and self._agent_cell == self.DEAD_STATE
        truncated = in_dead_state or self._step_count >= self._base_env.max_steps
        
        # Reward
        reward = 0.0 if terminated else -1.0
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def render(self):
        """Render using underlying environment with discrete grid overlay."""
        # Get base render
        frame = self.env.render()
        
        if frame is None:
            return None
        
        # Add grid overlay
        frame = self._add_grid_overlay(frame)
        
        return frame
    
    def _add_grid_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add discrete cell grid lines to the rendered frame."""
        from PIL import Image, ImageDraw
        
        # Convert to PIL Image for drawing
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        
        # Get rendering scale (frame size / environment size)
        frame_height, frame_width = frame.shape[:2]
        env_width = self._base_env.width
        env_height = self._base_env.height
        
        scale = min(frame_width, frame_height) / max(env_width, env_height)
        offset_x = (frame_width - env_width * scale) / 2
        offset_y = (frame_height - env_height * scale) / 2
        
        def to_screen(x: float, y: float) -> Tuple[int, int]:
            """Convert continuous coordinates to screen coordinates."""
            sx = int(offset_x + x * scale)
            sy = int(frame_height - offset_y - y * scale)
            return (sx, sy)
        
        # Grid line color (gray)
        grid_color = (128, 128, 128, 180)  # Semi-transparent gray
        line_width = 1
        
        # Draw vertical grid lines
        for cx in range(int(np.ceil((self._base_env.width - self._grid_offset_x) / self.cell_size)) + 1):
            x = self._grid_offset_x + cx * self.cell_size
            if x <= self._base_env.width:
                x1, y1 = to_screen(x, 0)
                x2, y2 = to_screen(x, self._base_env.height)
                draw.line([(x1, y1), (x2, y2)], fill=grid_color, width=line_width)
        
        # Draw horizontal grid lines
        for cy in range(int(np.ceil((self._base_env.height - self._grid_offset_y) / self.cell_size)) + 1):
            y = self._grid_offset_y + cy * self.cell_size
            if y <= self._base_env.height:
                x1, y1 = to_screen(0, y)
                x2, y2 = to_screen(self._base_env.width, y)
                draw.line([(x1, y1), (x2, y2)], fill=grid_color, width=line_width)
        
        # Optionally highlight valid cells with a slightly different overlay
        # Draw cell indices or markers for valid cells
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except:
            font = ImageFont.load_default()
        
        # Draw current cell highlight (thicker border)
        if self._agent_cell in self._cell_centers and self._agent_cell != self.DEAD_STATE:
            center = self._cell_centers[self._agent_cell]
            # Calculate cell corners
            cell_x = self._grid_offset_x + self._agent_cell[0] * self.cell_size
            cell_y = self._grid_offset_y + self._agent_cell[1] * self.cell_size
            
            # Convert corners to screen coordinates
            x1, y1 = to_screen(cell_x, cell_y + self.cell_size)
            x2, y2 = to_screen(cell_x + self.cell_size, cell_y)
            
            # Draw highlighted border for current cell
            highlight_color = (255, 255, 0)  # Yellow
            draw.rectangle([(x1, y1), (x2, y2)], outline=highlight_color, width=2)
        
        return np.array(img)

    @property
    def max_steps(self):
        return self._base_env.max_steps


# ============================================================================
# Tests
# ============================================================================
if __name__ == "__main__":
    import imageio
    from pathlib import Path
    
    def test_discretized_movement():
        """Test that agent moves correctly between cells in discretized environment."""
        print("=" * 60)
        print("TESTING DISCRETIZED CONTINUOUS ENVIRONMENT")
        print("=" * 60)
        
        # Import the continuous environment
        from continuous_rooms import ContinuousFourRoomsEnv
        
        # Create output directory
        output_dir = Path("videos/discretized_tests")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create continuous environment
        base_env = ContinuousFourRoomsEnv(
            room_size=4.0,
            corridor_width=0.8,
            corridor_offset=1.2,
            wall_thickness=0.6,
            agent_radius=0.15,
            render_mode="rgb_array",
            show_coordinates=True,
            max_steps=100
        )
        
        # Wrap with discretization
        cell_size = 1.0
        env = DiscretizedContinuousEnv(base_env, cell_size=cell_size, lava=False)
        
        print(f"\nEnvironment Info:")
        print(f"  Number of states: {env.n_states}")
        print(f"  Number of actions: {env.action_space.n}")
        print(f"  Cell size: {cell_size}")
        print(f"  Grid dimensions: {len(set(c[0] for c in env.cells if c != env.DEAD_STATE))} x {len(set(c[1] for c in env.cells if c != env.DEAD_STATE))}")
        
        # Test 1: Reset and check initial state
        print("\n--- Test 1: Reset and Initial State ---")
        obs, info = env.reset(seed=42)
        print(f"  Initial observation (state index): {obs}")
        print(f"  Initial cell: {info['agent_position']}")
        print(f"  Continuous position: {info['continuous_position']}")
        print(f"  Goal cell: {info['goal_cell']}")
        
        # Test 2: Movement in all directions
        print("\n--- Test 2: Movement Test ---")
        action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'UP-LEFT', 5: 'UP-RIGHT', 6: 'DOWN-LEFT', 7: 'DOWN-RIGHT'}
        
        movement_results = []
        for action in range(8):
            env.reset(seed=42)
            initial_cell = env._agent_cell
            initial_obs = env._get_obs()
            
            obs, reward, terminated, truncated, info = env.step(action)
            new_cell = info['agent_position']
            
            moved = initial_cell != new_cell
            movement_results.append({
                'action': action_names[action],
                'from': initial_cell,
                'to': new_cell,
                'moved': moved
            })
            
            status = "✓ MOVED" if moved else "✗ STAYED (wall)"
            print(f"  {action_names[action]}: {initial_cell} -> {new_cell} {status}")
        
        # Test 3: Continuous movement recording
        print("\n--- Test 3: Recording Movement Video ---")
        obs, info = env.reset(seed=42)
        frames = [env.render()]
        
        trajectory = [(info['agent_position'], info['continuous_position'].copy())]
        cells_visited = {info['agent_position']}
        stuck_count = 0
        max_stuck = 5
        
        # Perform random walk to test movement
        np.random.seed(42)
        for step in range(50):
            # Choose action that might lead to a new cell
            action = np.random.randint(8)
            prev_cell = env._agent_cell
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            curr_cell = info['agent_position']
            trajectory.append((curr_cell, info['continuous_position'].copy()))
            cells_visited.add(curr_cell)
            
            if prev_cell == curr_cell:
                stuck_count += 1
            else:
                stuck_count = 0
            
            frames.append(env.render())
            
            if terminated or truncated:
                print(f"  Episode ended at step {step + 1}")
                break
        
        # Save video
        video_path = output_dir / "discretized_movement_test.mp4"
        imageio.mimsave(str(video_path), frames, fps=5)
        print(f"  Video saved to: {video_path}")
        print(f"  Total frames: {len(frames)}")
        print(f"  Unique cells visited: {len(cells_visited)}")
        
        # Test 4: Verify step_from consistency
        print("\n--- Test 4: step_from Consistency ---")
        test_cells = list(env.cells)[:5]  # Test first 5 cells
        for cell in test_cells:
            if cell == env.DEAD_STATE:
                continue
            print(f"  From cell {cell}:")
            for action in range(8):
                next_cell = env.step_from(cell, action)
                direction = env.DISCRETE_ACTIONS[action]
                expected = (cell[0] + direction[0], cell[1] + direction[1])
                if expected in env.state_to_idx and expected != env.DEAD_STATE:
                    assert next_cell == expected, f"Expected {expected}, got {next_cell}"
                    status = "→ valid"
                else:
                    assert next_cell == cell, f"Should stay in place, but got {next_cell}"
                    status = "→ wall (stayed)"
                print(f"    {action_names[action]}: {next_cell} {status}")
        
        # Test 5: Cell center alignment
        print("\n--- Test 5: Cell Center Alignment ---")
        obs, info = env.reset(seed=123)
        for _ in range(10):
            action = np.random.randint(8)
            obs, _, terminated, truncated, info = env.step(action)
            
            cell = info['agent_position']
            cont_pos = info['continuous_position']
            expected_center = env._cell_to_continuous(cell)
            
            pos_diff = np.linalg.norm(cont_pos - expected_center)
            assert pos_diff < 1e-5, f"Position {cont_pos} not at cell center {expected_center}"
            
            if terminated or truncated:
                break
        print("  ✓ All positions correctly aligned to cell centers")
        
        # Test 6: Comprehensive movement video with annotations
        print("\n--- Test 6: Annotated Movement Video ---")
        obs, info = env.reset(seed=0)
        frames = []
        
        # Define a sequence of actions to show clear movement
        action_sequence = [3, 5, 6, 3, 0, 5, 0, 2, 4, 2, 4, 1, 1, 3, 0, 3, 0, 1, 2, 1, 2]  # Right, Right, Up, Up, etc.
        
        for i, action in enumerate(action_sequence):
            prev_cell = env._agent_cell
            obs, reward, terminated, truncated, info = env.step(action)
            curr_cell = info['agent_position']
            
            frame = env.render()
            frames.append(frame)
            
            move_status = "MOVED" if prev_cell != curr_cell else "BLOCKED"
            print(f"  Step {i+1}: {action_names[action]} | {prev_cell} -> {curr_cell} | {move_status}")
            
            if terminated or truncated:
                break
        
        video_path = output_dir / "annotated_movement_test.mp4"
        imageio.mimsave(str(video_path), frames, fps=2)
        print(f"  Annotated video saved to: {video_path}")
        
        env.close()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        
        return True
    
    def test_grid_coverage():
        """Test that all cells in the grid are reachable."""
        print("\n" + "=" * 60)
        print("TESTING GRID COVERAGE")
        print("=" * 60)
        
        from continuous_rooms import ContinuousFourRoomsEnv
        
        base_env = ContinuousFourRoomsEnv(
            room_size=4.0,
            corridor_width=0.8,
            corridor_offset=1.2,
            render_mode="rgb_array",
            max_steps=1000
        )
        env = DiscretizedContinuousEnv(base_env, cell_size=1.0, lava=False)
        
        # BFS to find all reachable cells from each starting cell
        from collections import deque
        
        def bfs_reachable(start_cell):
            visited = {start_cell}
            queue = deque([start_cell])
            while queue:
                cell = queue.popleft()
                for action in range(4):
                    next_cell = env.step_from(cell, action)
                    if next_cell not in visited and next_cell != env.DEAD_STATE:
                        visited.add(next_cell)
                        queue.append(next_cell)
            return visited
        
        # Check from first valid cell
        valid_cells = [c for c in env.cells if c != env.DEAD_STATE]
        start = valid_cells[0]
        reachable = bfs_reachable(start)
        
        print(f"  Total cells: {len(valid_cells)}")
        print(f"  Reachable from {start}: {len(reachable)}")
        
        unreachable = set(valid_cells) - reachable
        if unreachable:
            print(f"  ⚠ Unreachable cells: {unreachable}")
        else:
            print("  ✓ All cells are reachable!")
        
        env.close()
        return len(unreachable) == 0
    
    # Run tests
    print("\n" + "=" * 60)
    print("DISCRETIZED ENVIRONMENT WRAPPER TESTS")
    print("=" * 60 + "\n")
    
    try:
        test_discretized_movement()
        test_grid_coverage()
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()