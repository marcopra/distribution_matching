import numpy as np
import gymnasium as gym
from gymnasium import spaces

import utils
from env.domain_utils import coerce_dict, get_env_id, get_env_module


def is_point_maze_env(reference):
    env_id = get_env_id(reference).lower()
    module_name = get_env_module(reference).lower()
    return "pointmaze" in env_id or "point_maze" in module_name


def pop_point_maze_kwargs(env_kwargs):
    return coerce_dict(env_kwargs.pop("pointmaze", {}), "pointmaze")


def prepare_point_maze_make_kwargs(name, env_kwargs, url=False):
    del name
    pointmaze_kwargs = pop_point_maze_kwargs(env_kwargs)
    env_kwargs["reward_type"] = "dense"
    env_kwargs["reset_target"] = True
    env_kwargs.setdefault("continuing_task", False)
    if url:
        env_kwargs["continuing_task"] = True
    return env_kwargs, pointmaze_kwargs


class PointMazeDiscreteActions(gym.ActionWrapper):
    ACTIONS = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=np.float32,
    )

    def __init__(self, env, action_scale=1.0):
        super().__init__(env)
        self.action_scale = float(action_scale)
        self.action_space = spaces.Discrete(len(self.ACTIONS))

    def action(self, action):
        action_idx = int(action)
        if action_idx < 0 or action_idx >= len(self.ACTIONS):
            raise ValueError(f"PointMaze discrete action must be in [0, {len(self.ACTIONS) - 1}]")
        return self.action_scale * self.ACTIONS[action_idx].copy()

    def __getattr__(self, name):
        return getattr(self.env, name)


class PointMazeDirectVelocityActions(gym.Wrapper):
    def __init__(self, env, max_velocity=1.0, preserve_target_velocity=True):
        super().__init__(env)
        self.max_velocity = float(max_velocity)
        if self.max_velocity <= 0.0:
            raise ValueError(f"pointmaze.max_velocity must be positive, got {self.max_velocity}")

        self.preserve_target_velocity = bool(preserve_target_velocity)
        self.action_space = spaces.Box(
            low=-self.max_velocity,
            high=self.max_velocity,
            shape=(2,),
            dtype=np.float32,
        )

    def _base_env(self):
        return self.env.unwrapped

    def _point_env(self):
        point_env = getattr(self._base_env(), "point_env", None)
        if point_env is None:
            raise AttributeError("PointMazeDirectVelocityActions requires a PointMaze env with point_env")
        return point_env

    def _set_velocity(self, velocity):
        point_env = self._point_env()
        qpos = point_env.data.qpos.copy()
        qvel = point_env.data.qvel.copy()
        qvel[:2] = velocity
        point_env.set_state(qpos, qvel)

    def _refresh_obs(self):
        base_env = self._base_env()
        point_obs, _ = base_env.point_env._get_obs()
        return base_env._get_obs(point_obs)

    def step(self, action):
        velocity = np.asarray(action, dtype=np.float32)
        if velocity.shape != (2,):
            raise ValueError(f"PointMaze direct velocity action must have shape (2,), got {velocity.shape}")

        velocity = np.clip(velocity, self.action_space.low, self.action_space.high).astype(np.float32)
        point_env = self._point_env()
        self._set_velocity(velocity)

        zero_force = np.zeros(point_env.action_space.shape, dtype=point_env.action_space.dtype)
        obs, reward, terminated, truncated, info = self.env.step(zero_force)

        if self.preserve_target_velocity:
            self._set_velocity(velocity)
            obs = self._refresh_obs()

        info = dict(info) if info is not None else {}
        info["direct_velocity_action"] = velocity.copy()
        info["direct_velocity_preserved"] = self.preserve_target_velocity
        return obs, reward, terminated, truncated, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class FixedPointMazeResetWrapper(gym.Wrapper):
    def __init__(self, env, goal_position, start_position):
        super().__init__(env)
        self.fixed_goal = np.asarray(goal_position, dtype=np.float32)
        if self.fixed_goal.shape != (2,):
            raise ValueError(f"PointMaze goal_position must have shape (2,), got {self.fixed_goal.shape}")

        self.fixed_start = np.asarray(start_position, dtype=np.float32)
        if self.fixed_start.shape != (2,):
            raise ValueError(f"PointMaze start_position must have shape (2,), got {self.fixed_start.shape}")

        self.goal_position = self.fixed_goal.copy()
        self.start_position = self.fixed_start.copy()

    def _base_env(self):
        return self.env.unwrapped

    def _refresh_obs(self):
        base_env = self._base_env()
        point_obs, _ = base_env.point_env._get_obs()
        return base_env._get_obs(point_obs)

    def _apply_fixed_task(self):
        base_env = self._base_env()
        base_env.goal = self.fixed_goal.copy()
        base_env.reset_pos = self.fixed_start.copy()
        base_env.point_env.init_qpos[:2] = self.fixed_start
        base_env.point_env.init_qvel[:] = 0.0

        qpos = base_env.point_env.data.qpos.copy()
        qvel = np.zeros_like(base_env.point_env.data.qvel)
        qpos[:2] = self.fixed_start
        base_env.point_env.set_state(qpos, qvel)
        base_env.update_target_site_pos()

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        self._apply_fixed_task()
        obs = self._refresh_obs()

        info = dict(info) if info is not None else {}
        info["fixed_goal_position"] = self.fixed_goal.copy()
        info["fixed_start_position"] = self.fixed_start.copy()
        return obs, info

    def get_debug_coordinates(self):
        obs = self._refresh_obs()
        return {
            "xy": np.asarray(obs["observation"], dtype=np.float32)[:2].copy(),
            "fixed_start": self.fixed_start.copy(),
            "fixed_goal": self.fixed_goal.copy(),
        }

    def get_debug_maze_layout(self):
        base_env = self._base_env()
        maze = getattr(base_env, "maze", None)
        if maze is None or not hasattr(maze, "maze_map") or not hasattr(maze, "cell_rowcol_to_xy"):
            return None

        half_cell = 0.5 * float(getattr(maze, "maze_size_scaling", 1.0))
        wall_rectangles = []
        all_rectangles = []

        for row_idx, row in enumerate(maze.maze_map):
            for col_idx, cell in enumerate(row):
                cell_center = maze.cell_rowcol_to_xy(np.array([row_idx, col_idx], dtype=np.int32))
                x0 = float(cell_center[0] - half_cell)
                y0 = float(cell_center[1] - half_cell)
                rect = np.array([x0, y0, 2.0 * half_cell, 2.0 * half_cell], dtype=np.float32)
                all_rectangles.append(rect)
                if cell == 1:
                    wall_rectangles.append(rect)

        if not all_rectangles:
            return None

        all_rectangles = np.asarray(all_rectangles, dtype=np.float32)
        maze_lower = all_rectangles[:, :2].min(axis=0)
        maze_upper = (all_rectangles[:, :2] + all_rectangles[:, 2:4]).max(axis=0)
        wall_rectangles = np.asarray(wall_rectangles, dtype=np.float32).reshape(-1, 4)

        return {
            "maze_lower": maze_lower,
            "maze_upper": maze_upper,
            "wall_rectangles": wall_rectangles,
        }

    def __getattr__(self, name):
        return getattr(self.env, name)


class PointMazeTopDownCameraWrapper(gym.Wrapper):
    def __init__(self, env, distance=None, elevation=-90.0, azimuth=90.0):
        super().__init__(env)
        self.camera_distance = distance
        self.camera_elevation = float(elevation)
        self.camera_azimuth = float(azimuth)
        self._apply_top_down_camera()

    def _base_env(self):
        return self.env.unwrapped

    def _maze_center(self):
        base_env = self._base_env()
        maze = getattr(base_env, "maze", None)
        if maze is None or not hasattr(maze, "maze_map") or not hasattr(maze, "cell_rowcol_to_xy"):
            return np.zeros(3, dtype=np.float64)

        xy_positions = []
        for row_idx, row in enumerate(maze.maze_map):
            for col_idx, _ in enumerate(row):
                xy_positions.append(maze.cell_rowcol_to_xy(np.array([row_idx, col_idx], dtype=np.int32)))

        if not xy_positions:
            return np.zeros(3, dtype=np.float64)

        xy_positions = np.asarray(xy_positions, dtype=np.float64)
        center_xy = 0.5 * (xy_positions.min(axis=0) + xy_positions.max(axis=0))
        return np.array([center_xy[0], center_xy[1], 0.0], dtype=np.float64)

    def _apply_top_down_camera(self):
        point_env = getattr(self._base_env(), "point_env", None)
        renderer = getattr(point_env, "mujoco_renderer", None)
        if renderer is None:
            return

        camera_config = dict(renderer.default_cam_config or {})
        if self.camera_distance is not None:
            camera_config["distance"] = float(self.camera_distance)
        camera_config["elevation"] = self.camera_elevation
        camera_config["azimuth"] = self.camera_azimuth
        camera_config["lookat"] = self._maze_center()
        renderer.default_cam_config = camera_config

        viewer = getattr(renderer, "viewer", None)
        if viewer is not None:
            renderer._set_cam_config()

    def render(self):
        self._apply_top_down_camera()
        return self.env.render()

    def render_observation(self):
        self._apply_top_down_camera()
        render_fn = getattr(self.env, "render_observation", None)
        return render_fn() if callable(render_fn) else self.env.render()

    def render_image_observation(self):
        self._apply_top_down_camera()
        render_fn = getattr(self.env, "render_image_observation", None)
        return render_fn() if callable(render_fn) else self.env.render()

    def __getattr__(self, name):
        return getattr(self.env, name)


class PointMazeGoalMaskWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._cached_hidden_render = None

    def _render_without_goal(self):
        base_env = self.env.unwrapped
        original_rgba = base_env.model.site_rgba[base_env.target_site_id].copy()
        try:
            base_env.model.site_rgba[base_env.target_site_id, 3] = 0.0
            return self.env.render()
        finally:
            base_env.model.site_rgba[base_env.target_site_id] = original_rgba

    def render_observation(self):
        frame = self._render_without_goal()
        self._cached_hidden_render = frame
        return frame

    def render_image_observation(self):
        if self._cached_hidden_render is None:
            self._cached_hidden_render = self._render_without_goal()

        # Uncomment the next line and comment the return below to render the goal
        # in image_observation while keeping observation goal-hidden.
        # return self.env.render()
        return self._cached_hidden_render.copy()

    def reset(self, **kwargs):
        self._cached_hidden_render = None
        return self.env.reset(**kwargs)

    def step(self, action):
        self._cached_hidden_render = None
        return self.env.step(action)

    def __getattr__(self, name):
        return getattr(self.env, name)


class PointMazeXYObservationWrapper(gym.Wrapper):
    """Expose only the agent XY position from PointMaze state observations."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2,),
            dtype=np.float32,
        )

    @staticmethod
    def _xy_observation(obs):
        if isinstance(obs, dict):
            obs = obs.get("observation")
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if obs.size < 2:
            raise ValueError(f"PointMaze XY observation requires at least 2 values, got shape {obs.shape}")
        return obs[:2].copy()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._xy_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._xy_observation(obs), reward, terminated, truncated, info

    def __getattr__(self, name):
        return getattr(self.env, name)


def wrap_point_maze_env(env, pointmaze_kwargs):
    pointmaze_kwargs = coerce_dict(pointmaze_kwargs, "pointmaze")
    goal_position = pointmaze_kwargs.pop("goal_position", None)
    start_position = pointmaze_kwargs.pop("start_position", None)
    top_down_camera = bool(pointmaze_kwargs.pop("top_down_camera", True))
    camera_distance = pointmaze_kwargs.pop("camera_distance", None)
    camera_elevation = pointmaze_kwargs.pop("camera_elevation", -90.0)
    camera_azimuth = pointmaze_kwargs.pop("camera_azimuth", 90.0)
    discrete_actions = bool(pointmaze_kwargs.pop("discrete_actions", True))
    direct_velocity_actions = bool(pointmaze_kwargs.pop("direct_velocity_actions", False))
    max_velocity = pointmaze_kwargs.pop("max_velocity", 1.0)
    preserve_target_velocity = bool(pointmaze_kwargs.pop("preserve_target_velocity", True))
    only_xy_position = bool(pointmaze_kwargs.pop("only_xy_position", False))

    if goal_position is None:
        raise ValueError("PointMaze environments require pointmaze.goal_position to keep the goal fixed")
    if start_position is None:
        raise ValueError("PointMaze environments require pointmaze.start_position to keep the initial position fixed")
    if pointmaze_kwargs:
        unknown_keys = ", ".join(sorted(pointmaze_kwargs))
        raise TypeError(f"Unknown PointMaze kwargs: {unknown_keys}")

    env = FixedPointMazeResetWrapper(env, goal_position=goal_position, start_position=start_position)
    if top_down_camera:
        env = PointMazeTopDownCameraWrapper(
            env,
            distance=camera_distance,
            elevation=camera_elevation,
            azimuth=camera_azimuth,
        )
    env = PointMazeGoalMaskWrapper(env)
    if direct_velocity_actions:
        env = PointMazeDirectVelocityActions(
            env,
            max_velocity=max_velocity,
            preserve_target_velocity=preserve_target_velocity,
        )
        if discrete_actions:
            env = PointMazeDiscreteActions(env, action_scale=max_velocity)
            action_description = f"4 discrete direct velocity actions scaled to max_velocity={max_velocity}"
        else:
            action_description = "direct velocity actions"
    elif discrete_actions:
        env = PointMazeDiscreteActions(env)
        action_description = "4 discrete force actions"
    else:
        action_description = "continuous force actions"

    if only_xy_position:
        env = PointMazeXYObservationWrapper(env)

    warning = (
        "Warning: PointMaze environment uses fixed goal and initial position, "
        f"{action_description}, dense reward, and goal-hidden pixel observations."
    )
    if only_xy_position:
        warning += " State observations are masked to agent XY position only."
    if getattr(env.unwrapped, "continuing_task", False):
        warning += " continuing_task=True keeps the episode from terminating at success."
    utils.ColorPrint.yellow(warning)
    return env
