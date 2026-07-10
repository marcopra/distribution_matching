import numpy as np
import gymnasium as gym
from gymnasium import spaces

import utils
from env.domain_utils import coerce_dict, get_env_id, get_env_module


def is_fetch_env(reference):
    env_id = get_env_id(reference).lower()
    module_name = get_env_module(reference).lower()
    return "fetch" in env_id or "fetch" in module_name


def pop_fetch_kwargs(env_kwargs):
    return coerce_dict(env_kwargs.pop("fetch", {}), "fetch")


def prepare_fetch_make_kwargs(env_kwargs):
    fetch_kwargs = pop_fetch_kwargs(env_kwargs)
    env_kwargs["reward_type"] = "sparse"
    return env_kwargs, fetch_kwargs


class FetchDiscreteActions(gym.ActionWrapper):
    ACTIONS = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(len(self.ACTIONS))

    def action(self, action):
        action_idx = int(action)
        if action_idx < 0 or action_idx >= len(self.ACTIONS):
            raise ValueError(f"Fetch discrete action must be in [0, {len(self.ACTIONS) - 1}]")
        return self.ACTIONS[action_idx].copy()

    def __getattr__(self, name):
        return getattr(self.env, name)


class FixedFetchResetWrapper(gym.Wrapper):
    DEFAULT_MOCAP_QUAT = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)

    def __init__(self, env, goal_position, start_position=None):
        super().__init__(env)
        self.fixed_goal = np.asarray(goal_position, dtype=np.float32)
        if self.fixed_goal.shape != (3,):
            raise ValueError(f"Fetch goal_position must have shape (3,), got {self.fixed_goal.shape}")

        self.fixed_start = None
        if start_position is not None:
            self.fixed_start = np.asarray(start_position, dtype=np.float32)
            if self.fixed_start.shape != (3,):
                raise ValueError(
                    f"Fetch start_position must have shape (3,), got {self.fixed_start.shape}"
                )

        self.goal_position = self.fixed_goal.copy()
        self.start_position = None if self.fixed_start is None else self.fixed_start.copy()
        self.has_object = bool(getattr(self.env.unwrapped, "has_object", False))

    def _forward(self):
        base_env = self.env.unwrapped
        if hasattr(base_env, "_mujoco") and hasattr(base_env, "model") and hasattr(base_env, "data"):
            base_env._mujoco.mj_forward(base_env.model, base_env.data)
        elif hasattr(base_env, "sim"):
            base_env.sim.forward()
        else:
            raise NotImplementedError("Unsupported Fetch backend for forward simulation")

    def _set_joint_qpos(self, joint_name, qpos):
        base_env = self.env.unwrapped
        if hasattr(base_env, "_utils") and hasattr(base_env, "model") and hasattr(base_env, "data"):
            base_env._utils.set_joint_qpos(base_env.model, base_env.data, joint_name, qpos)
        elif hasattr(base_env, "sim"):
            base_env.sim.data.set_joint_qpos(joint_name, qpos)
        else:
            raise NotImplementedError("Unsupported Fetch backend for joint position update")

    def _get_joint_qpos(self, joint_name):
        base_env = self.env.unwrapped
        if hasattr(base_env, "_utils") and hasattr(base_env, "model") and hasattr(base_env, "data"):
            return np.asarray(
                base_env._utils.get_joint_qpos(base_env.model, base_env.data, joint_name),
                dtype=np.float32,
            )
        if hasattr(base_env, "sim"):
            return np.asarray(base_env.sim.data.get_joint_qpos(joint_name), dtype=np.float32)
        raise NotImplementedError("Unsupported Fetch backend for joint position query")

    def _set_gripper_position(self, target_position):
        base_env = self.env.unwrapped
        target_position = np.asarray(target_position, dtype=np.float32)

        if hasattr(base_env, "_utils") and hasattr(base_env, "model") and hasattr(base_env, "data"):
            base_env._utils.set_mocap_pos(base_env.model, base_env.data, "robot0:mocap", target_position)
            base_env._utils.set_mocap_quat(
                base_env.model,
                base_env.data,
                "robot0:mocap",
                self.DEFAULT_MOCAP_QUAT,
            )
            for _ in range(10):
                base_env._mujoco.mj_step(base_env.model, base_env.data, nstep=base_env.n_substeps)
            return

        if hasattr(base_env, "sim"):
            base_env.sim.data.set_mocap_pos("robot0:mocap", target_position)
            base_env.sim.data.set_mocap_quat("robot0:mocap", self.DEFAULT_MOCAP_QUAT)
            for _ in range(10):
                base_env.sim.step()
            return

        raise NotImplementedError("Unsupported Fetch backend for gripper reset")

    def _apply_fixed_start(self):
        if self.fixed_start is None:
            return

        if self.has_object:
            object_qpos = self._get_joint_qpos("object0:joint").copy()
            object_qpos[:3] = self.fixed_start
            self._set_joint_qpos("object0:joint", object_qpos)
            self._forward()
        else:
            self._set_gripper_position(self.fixed_start)
            self._forward()

    def _refresh_obs(self):
        return self.env.unwrapped._get_obs()

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        self.env.unwrapped.goal = self.fixed_goal.copy()
        self._apply_fixed_start()
        obs = self._refresh_obs()

        info = dict(info) if info is not None else {}
        info["fixed_goal_position"] = self.fixed_goal.copy()
        if self.fixed_start is not None:
            info["fixed_start_position"] = self.fixed_start.copy()
        return obs, info

    def get_debug_coordinates(self):
        obs = self._refresh_obs()
        proprio = np.asarray(obs["observation"], dtype=np.float32).reshape(-1)
        return {
            "xyz": proprio[:3].copy(),
            "fixed_start": None if self.fixed_start is None else self.fixed_start.copy(),
            "fixed_goal": self.fixed_goal.copy(),
        }

    def __getattr__(self, name):
        return getattr(self.env, name)


class FetchGoalMaskWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._cached_hidden_render = None

    def _get_site_model_and_id(self):
        base_env = self.env.unwrapped

        if hasattr(base_env, "_mujoco") and hasattr(base_env, "model"):
            site_id = base_env._mujoco.mj_name2id(
                base_env.model,
                base_env._mujoco.mjtObj.mjOBJ_SITE,
                "target0",
            )
            return base_env.model, site_id

        if hasattr(base_env, "sim"):
            site_id = base_env.sim.model.site_name2id("target0")
            return base_env.sim.model, site_id

        raise NotImplementedError("Unsupported Fetch backend for goal masking")

    def _render_without_goal(self):
        model, site_id = self._get_site_model_and_id()
        original_rgba = model.site_rgba[site_id].copy()
        try:
            model.site_rgba[site_id, 3] = 0.0
            return self.env.render()
        finally:
            model.site_rgba[site_id] = original_rgba

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


def wrap_fetch_env(env, fetch_kwargs):
    fetch_kwargs = coerce_dict(fetch_kwargs, "fetch")
    goal_position = fetch_kwargs.pop("goal_position", None)
    start_position = fetch_kwargs.pop("start_position", None)

    if goal_position is None:
        raise ValueError("Fetch environments require fetch.goal_position to keep the goal fixed")
    if fetch_kwargs:
        unknown_keys = ", ".join(sorted(fetch_kwargs))
        raise TypeError(f"Unknown Fetch kwargs: {unknown_keys}")

    env = FixedFetchResetWrapper(env, goal_position=goal_position, start_position=start_position)
    env = FetchGoalMaskWrapper(env)
    env = FetchDiscreteActions(env)

    utils.ColorPrint.yellow(
        "Warning: Fetch environment uses fixed goal and initial position, "
        "8 discrete actions, sparse reward, and goal-hidden pixel observations."
    )
    return env
