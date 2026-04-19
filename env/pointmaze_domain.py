import mujoco
import numpy as np
import gymnasium as gym
from dm_env import specs

from env.domain_utils import get_env_id, get_env_module


def is_point_maze_env(reference):
    env_id = get_env_id(reference).lower()
    module_name = get_env_module(reference).lower()
    return "pointmaze" in env_id or "point_maze" in module_name


def prepare_point_maze_make_kwargs(name, env_kwargs):
    raise NotImplementedError(
        "PointMaze environment support is only scaffolded right now. "
        "Fill in the family-specific kwargs handling and wrappers in env/pointmaze_domain.py."
    )


def wrap_point_maze_env(env, obs_type, action_repeat, resolution, grayscale):
    raise NotImplementedError(
        "PointMaze environment support is only scaffolded right now. "
        "Fill in the family-specific wrappers in env/pointmaze_domain.py."
    )


class PhysicsStateWrapper(gym.Wrapper):
    """Wrapper that simulates the CDMC physics interface for PointMaze."""

    def __init__(self, env):
        super().__init__(env)
        self._physics_state = None

    def _get_physics_state(self):
        if hasattr(self.env, "unwrapped"):
            unwrapped = self.env.unwrapped
            if hasattr(unwrapped, "point_env"):
                point_env = unwrapped.point_env
                qpos = point_env.data.qpos.copy()
                qvel = point_env.data.qvel.copy()
                return np.concatenate([qpos, qvel])

        return self._physics_state if self._physics_state is not None else np.zeros(4)

    def _set_physics_state(self, state):
        if hasattr(self.env, "unwrapped"):
            unwrapped = self.env.unwrapped
            if hasattr(unwrapped, "point_env"):
                point_env = unwrapped.point_env
                mid = len(state) // 2
                point_env.data.qpos[:] = state[:mid]
                point_env.data.qvel[:] = state[mid:]
                mujoco.mj_forward(point_env.model, point_env.data)

    def reset(self, **kwargs):
        time_step = self.env.reset(**kwargs)
        self._physics_state = self._get_physics_state()
        return time_step

    def step(self, action):
        time_step = self.env.step(action)
        self._physics_state = self._get_physics_state()
        return time_step

    @property
    def physics(self):
        class PhysicsInterface:
            def __init__(self, wrapper):
                self.wrapper = wrapper

            def state(self):
                return self.wrapper._get_physics_state()

            def set_state(self, state):
                self.wrapper._set_physics_state(state)

            class ResetContext:
                def __init__(self, physics_interface):
                    self.physics = physics_interface
                    self.original_state = None

                def __enter__(self):
                    self.original_state = self.physics.state()
                    return self

                def __exit__(self, exc_type, exc_val, exc_tb):
                    if self.original_state is not None:
                        self.physics.set_state(self.original_state)

            def reset_context(self):
                return self.ResetContext(self)

        return PhysicsInterface(self)


class RewardSpecWrapper(gym.Wrapper):
    """Add reward and discount specs compatible with CDMC for PointMaze."""

    def __init__(self, env):
        super().__init__(env)
        if not hasattr(self.env, "unwrapped") or not hasattr(self.env.unwrapped, "compute_reward"):
            raise NotImplementedError("RewardSpecWrapper is currently only implemented for PointMaze environments")

    def reward_spec(self):
        return specs.Array(shape=(1,), dtype=np.float32, name="reward")

    def discount_spec(self):
        return specs.Array(shape=(1,), dtype=np.float32, name="discount")

    def compute_reward_from_state_and_action(self, physics_state, action, desired_goal=None):
        unwrapped = self.env.unwrapped
        original_state = self.physics.state()
        original_goal = unwrapped.goal.copy()

        try:
            if desired_goal is not None:
                goal_to_use = desired_goal.copy()
            else:
                goal_to_use = physics_state[-2:].copy()

            achieved_goal = physics_state[:2].copy()
            if hasattr(unwrapped, "compute_reward"):
                reward = unwrapped.compute_reward(achieved_goal, goal_to_use, {})
                return np.array([reward], dtype=np.float32)
            raise NotImplementedError("compute_reward method not found in environment")
        finally:
            self.physics.set_state(original_state)
            unwrapped.goal = original_goal
            if hasattr(unwrapped, "update_target_site_pos"):
                unwrapped.update_target_site_pos()

    def compute_reward_from_obs_dict(self, obs_dict, action=None):
        if not all(key in obs_dict for key in ["achieved_goal", "desired_goal"]):
            raise ValueError("obs_dict must contain 'achieved_goal' and 'desired_goal' keys")

        achieved_goal = obs_dict["achieved_goal"]
        desired_goal = obs_dict["desired_goal"]
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "compute_reward"):
            reward = unwrapped.compute_reward(achieved_goal, desired_goal, {})
            return np.array([reward], dtype=np.float32)
        raise NotImplementedError("compute_reward method not found in environment")

    def __getattr__(self, name):
        return getattr(self.env, name)
