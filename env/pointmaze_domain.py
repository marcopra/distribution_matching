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

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(len(self.ACTIONS))

    def action(self, action):
        action_idx = int(action)
        if action_idx < 0 or action_idx >= len(self.ACTIONS):
            raise ValueError(f"PointMaze discrete action must be in [0, {len(self.ACTIONS) - 1}]")
        return self.ACTIONS[action_idx].copy()

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


def wrap_point_maze_env(env, pointmaze_kwargs):
    pointmaze_kwargs = coerce_dict(pointmaze_kwargs, "pointmaze")
    goal_position = pointmaze_kwargs.pop("goal_position", None)
    start_position = pointmaze_kwargs.pop("start_position", None)

    if goal_position is None:
        raise ValueError("PointMaze environments require pointmaze.goal_position to keep the goal fixed")
    if start_position is None:
        raise ValueError("PointMaze environments require pointmaze.start_position to keep the initial position fixed")
    if pointmaze_kwargs:
        unknown_keys = ", ".join(sorted(pointmaze_kwargs))
        raise TypeError(f"Unknown PointMaze kwargs: {unknown_keys}")

    env = FixedPointMazeResetWrapper(env, goal_position=goal_position, start_position=start_position)
    env = PointMazeGoalMaskWrapper(env)
    env = PointMazeDiscreteActions(env)

    warning = (
        "Warning: PointMaze environment uses fixed goal and initial position, "
        "4 discrete actions, dense reward, and goal-hidden pixel observations."
    )
    if getattr(env.unwrapped, "continuing_task", False):
        warning += " continuing_task=True keeps the episode from terminating at success."
    utils.ColorPrint.yellow(warning)
    return env
