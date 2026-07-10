import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.domain_utils import get_env_id, get_env_module


def is_classic_minigrid_env(reference):
    env_id = get_env_id(reference).lower()
    module_name = get_env_module(reference).lower()
    return "minigrid" in env_id or "minigrid" in module_name


class ClassicMiniGridInterfaceMixin:
    """
    Shared MiniGrid state metadata for wrappers.

    The discrete state space enumerates agent poses `(x, y, dir)` over all non-wall
    cells. This is exact for static-layout tasks. For tasks with mutable objects
    such as doors or keys, callers should use pixel observations instead.
    """

    DEAD_STATE = None
    _UNSUPPORTED_DYNAMIC_OBJECTS = {"door", "key", "ball", "box"}

    def _init_classic_minigrid_interface(self):
        self._build_classic_minigrid_state_space()

    def _build_classic_minigrid_state_space(self):
        base_env = self.env.unwrapped
        self.cells = []
        self.state_to_idx = {}
        self.idx_to_state = {}
        self.plot_cells = []
        self.plot_state_to_idx = {}

        for y in range(base_env.height):
            for x in range(base_env.width):
                cell = base_env.grid.get(x, y)
                if getattr(cell, "type", None) == "wall":
                    continue

                plot_cell = (x, y)
                if plot_cell not in self.plot_state_to_idx:
                    self.plot_state_to_idx[plot_cell] = len(self.plot_cells)
                    self.plot_cells.append(plot_cell)

                for direction in range(4):
                    state = (x, y, direction)
                    idx = len(self.cells)
                    self.cells.append(state)
                    self.state_to_idx[state] = idx
                    self.idx_to_state[idx] = state

        self.n_states = len(self.cells)

    def _validate_discrete_classic_minigrid_support(self):
        base_env = self.env.unwrapped
        unsupported = set()

        for y in range(base_env.height):
            for x in range(base_env.width):
                cell = base_env.grid.get(x, y)
                cell_type = getattr(cell, "type", None)
                if cell_type in self._UNSUPPORTED_DYNAMIC_OBJECTS:
                    unsupported.add(cell_type)

        if unsupported:
            unsupported_str = ", ".join(sorted(unsupported))
            raise ValueError(
                "MiniGrid discrete one-hot observations are only supported for static-layout "
                f"tasks. Found mutable object types: {unsupported_str}. Use obs_type='pixels' instead."
            )

    def _get_classic_minigrid_state(self):
        base_env = self.env.unwrapped
        pos = tuple(int(v) for v in np.asarray(base_env.agent_pos).tolist())
        direction = int(base_env.agent_dir)
        state = (pos[0], pos[1], direction)

        if state not in self.state_to_idx:
            raise KeyError(f"Agent state {state} is not part of the MiniGrid state space")
        return state

    def _augment_info(self, info):
        info = dict(info) if info is not None else {}
        state = self._get_classic_minigrid_state()
        info.setdefault("agent_position", state[:2])
        info.setdefault("agent_direction", state[2])
        info.setdefault("state_index", self.state_to_idx[state])
        return info

    def render_from_position(self, position):
        """
        Render the current MiniGrid layout from a given agent pose.

        We only vary the agent pose here; the rest of the grid stays unchanged.
        That means for environments with mutable objects this is a snapshot-based
        debugging view rather than an exhaustive rendering of every latent state.
        """

        base_env = self.env.unwrapped
        state = tuple(position)
        if len(state) == 2:
            state = (state[0], state[1], int(base_env.agent_dir))
        if len(state) != 3:
            raise ValueError(f"Expected MiniGrid state as (x, y, dir), got {position}")

        original_pos = np.array(base_env.agent_pos, copy=True)
        original_dir = int(base_env.agent_dir)
        try:
            base_env.agent_pos = np.array(state[:2], dtype=np.int64)
            base_env.agent_dir = int(state[2])
            return self.env.render()
        finally:
            base_env.agent_pos = original_pos
            base_env.agent_dir = original_dir


class ClassicMiniGridDiscreteStateWrapper(ClassicMiniGridInterfaceMixin, gym.Wrapper):
    """Expose the classic MiniGrid benchmark as a discrete agent-pose MDP."""

    def __init__(self, env):
        super().__init__(env)
        self._init_classic_minigrid_interface()
        self._validate_discrete_classic_minigrid_support()
        self.observation_space = spaces.Discrete(self.n_states)

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        state = self._get_classic_minigrid_state()
        return self.state_to_idx[state], self._augment_info(info)

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        state = self._get_classic_minigrid_state()
        return self.state_to_idx[state], reward, terminated, truncated, self._augment_info(info)

    def __getattr__(self, name):
        return getattr(self.env, name)


class ClassicMiniGridTopDownObservationWrapper(ClassicMiniGridInterfaceMixin, gym.Wrapper):
    """Return fully observable top-down RGB observations for the classic MiniGrid benchmark."""

    def __init__(self, env):
        super().__init__(env)
        self._init_classic_minigrid_interface()
        sample = None
        try:
            sample = self.env.render()
        except Exception:
            sample = None

        if not isinstance(sample, np.ndarray) or sample.ndim != 3:
            base_env = self.env.unwrapped
            tile_size = int(getattr(base_env, "tile_size", 32))
            sample = np.zeros(
                (base_env.height * tile_size, base_env.width * tile_size, 3),
                dtype=np.uint8,
            )

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=sample.shape,
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        return self.env.render(), self._augment_info(info)

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        return self.env.render(), reward, terminated, truncated, self._augment_info(info)

    def __getattr__(self, name):
        return getattr(self.env, name)


def wrap_classic_minigrid_env(
    env,
    obs_type,
    resolution,
    grayscale,
    *,
    resize_rendering_cls,
    discrete_observation_wrapper_cls,
):
    if obs_type == "pixels":
        env = resize_rendering_cls(env, resolution=resolution, grayscale=grayscale)
        return ClassicMiniGridTopDownObservationWrapper(env)

    if obs_type == "discrete_states":
        env = ClassicMiniGridDiscreteStateWrapper(env)
        return discrete_observation_wrapper_cls(env)

    return env
