from collections import deque
from typing import Any, NamedTuple
import os
import minigrid
import utils
import gymnasium as gym
from env.rooms import *
from env.multiple_rooms import MultipleRoomsEnv
from env.corridor import CorridorEnv
from env.continuous_rooms import (
    ContinuousSingleRoomEnv,
    ContinuousTwoRoomsEnv, 
    ContinuousFourRoomsEnv,
    ContinuousMultipleRoomsEnv
)
from env.wrappers import DiscretizedContinuousEnv
import numpy as np
from gymnasium import spaces
import mujoco
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)
from dm_env import StepType, specs
from PIL import Image

import ale_py
gym.register_envs(ale_py)
from gymnasium.wrappers import AtariPreprocessing


class ResizeRendering(gym.Wrapper):

    def __init__(self, env, resolution=224, grayscale=False):
        super().__init__(env)
        self.resolution = resolution
        self.grayscale = grayscale
        self.render_resolution = resolution  # Expose for agent access

    def render(self):
        img = super().render()

        # Convert numpy array to PIL Image
        img = Image.fromarray(img.astype(np.uint8))
        if self.grayscale:
            img = img.convert('L')
        
        # Resize the image
        img_resized = img.resize((self.resolution, self.resolution), Image.LANCZOS)
        
        # Convert back to numpy array
        img_array = np.array(img_resized)
        if self.grayscale:
            img_array = img_array[..., None]
        return img_array
    
    def set_task(self, task):
        """Set the task for the environment."""
        # Set the task in the base environment
        self.env.set_task(task)
    
    def __getattr__(self, name):
        """Forward other attributes to the wrapped environment."""
        return getattr(self.env, name)


class AtariScoreMaskWrapper(gym.ObservationWrapper):
    """
    Mask Atari score area by overwriting a top band.
    Defaults are tuned for ALE/Pong, but can be customized per env.
    """

    DEFAULT_BANDS = {
        "ALE/Pong-v5": 10,
        "PongNoFrameskip-v4": 10,
        "ALE/Breakout-v5": 12,
        "BreakoutNoFrameskip-v4": 12,
        "ALE/SpaceInvaders-v5": 12,
        "SpaceInvadersNoFrameskip-v4": 12,
        "TennisNoFrameskip-v4" : 8,
        "BowlingNoFrameskip-v4" : 25,
        "MarioBrosNoFrameskip-v4" : 7,
        "ALE/MarioBros-v5" : 7,
        "ALE/MontezumaRevenge-v5" : 0,

    }

    def __init__(self, env, band_height=None, color=255):
        super().__init__(env)
        self.band_height = band_height
        self.color = color

    def _resolve_band_height(self):
        if self.band_height is not None:
            return self.band_height
        env_id = getattr(self.env.unwrapped, "spec", None)
        env_name = env_id.id if env_id is not None else None
        if env_name in self.DEFAULT_BANDS:
            return self.DEFAULT_BANDS[env_name]
        return 0

    def observation(self, obs):
        if not isinstance(obs, np.ndarray) or obs.ndim != 3:
            return obs
        band = self._resolve_band_height()
        if band <= 0:
            return obs
        out = obs.copy()
        out[:band, :, :] = self.color
        return out

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    proprio_observation: Any
    image_observation: Any
    action: Any
    success: Any = None
    info: Any = None

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)

class DiscreteObservationWrapper(gym.Wrapper):
    """Wrapper that converts discrete observations to one-hot encoding."""
    
    def __init__(self, env):
        super().__init__(env)
        if isinstance(env.observation_space, spaces.Discrete):
            self.n_states = env.observation_space.n
            # TODO non penso ci siano problemi perchè dopo uso floa32
            # assert self.n_states < 256, "Number of discrete states must be less than 256 for uint8 one-hot encoding, otherwise change dtype here."
            self.is_discrete = True
            # Update observation space to one-hot
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(self.n_states,), dtype=np.float32
            )
        else:
            self.is_discrete = False
    
    def _obs_to_onehot(self, obs):
        """Convert discrete observation to one-hot."""
        if self.is_discrete:
            onehot = np.zeros(self.n_states, dtype=np.float32)
            onehot[obs] = 1.0
            return onehot
        return obs
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._obs_to_onehot(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._obs_to_onehot(obs), reward, terminated, truncated, info
    
    def __getattr__(self, name):
        return getattr(self.env, name)


class MiniGridInterfaceMixin:
    """
    Shared MiniGrid state metadata for wrappers.

    The discrete state space enumerates agent poses `(x, y, dir)` over all non-wall
    cells. This is exact for static-layout tasks. For tasks with mutable objects
    such as doors or keys, callers should use pixel observations instead.
    """

    DEAD_STATE = None
    _UNSUPPORTED_DYNAMIC_OBJECTS = {"door", "key", "ball", "box"}

    def _init_minigrid_interface(self):
        self._build_minigrid_state_space()

    def _build_minigrid_state_space(self):
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

    def _validate_discrete_minigrid_support(self):
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

    def _get_minigrid_state(self):
        base_env = self.env.unwrapped
        pos = tuple(int(v) for v in np.asarray(base_env.agent_pos).tolist())
        direction = int(base_env.agent_dir)
        state = (pos[0], pos[1], direction)
        if state not in self.state_to_idx:
            raise KeyError(f"Agent state {state} is not part of the MiniGrid state space")
        return state

    def _augment_info(self, info):
        info = dict(info) if info is not None else {}
        state = self._get_minigrid_state()
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


class MiniGridDiscreteStateWrapper(MiniGridInterfaceMixin, gym.Wrapper):
    """Expose MiniGrid as a discrete fully observable agent-pose MDP."""

    def __init__(self, env):
        super().__init__(env)
        self._init_minigrid_interface()
        self._validate_discrete_minigrid_support()
        self.observation_space = spaces.Discrete(self.n_states)

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        state = self._get_minigrid_state()
        return self.state_to_idx[state], self._augment_info(info)

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        state = self._get_minigrid_state()
        return self.state_to_idx[state], reward, terminated, truncated, self._augment_info(info)

    def __getattr__(self, name):
        return getattr(self.env, name)


class MiniGridTopDownObservationWrapper(MiniGridInterfaceMixin, gym.Wrapper):
    """Return fully observable top-down RGB observations for MiniGrid."""

    def __init__(self, env):
        super().__init__(env)
        self._init_minigrid_interface()
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

class ActionRepeatWrapper(gym.Wrapper):
    MONTEZUMA_ROOM_RAM_INDEX = 3

    def __init__(self, env, num_repeats, obs_type='pixels', data_collection=False):
        super().__init__(env)
        self._num_repeats = num_repeats
        self.data_collection = data_collection
        self.obs_type = obs_type
        self.obs_keys = None
        self._is_montezuma = self._check_is_montezuma()
        self._montezuma_initial_room = None
        self._montezuma_max_room = None
        self._montezuma_visited_second_room = False
        
        # Expose render_resolution if available
        if hasattr(env, 'render_resolution'):
            self.render_resolution = env.render_resolution
        elif hasattr(env, 'resolution'):
            self.render_resolution = env.resolution

    def _check_is_montezuma(self):
        spec = getattr(self.env.unwrapped, 'spec', None)
        env_id = spec.id if spec is not None else ''
        return 'MontezumaRevenge' in env_id

    def _get_montezuma_room_id(self):
        if not self._is_montezuma:
            return None
        ale = getattr(self.env.unwrapped, 'ale', None)
        if ale is None:
            return None
        ram = ale.getRAM()
        if ram is None or len(ram) <= self.MONTEZUMA_ROOM_RAM_INDEX:
            return None
        return int(ram[self.MONTEZUMA_ROOM_RAM_INDEX])

    def _reset_montezuma_tracking(self):
        room_id = self._get_montezuma_room_id()
        self._montezuma_initial_room = room_id
        self._montezuma_max_room = room_id
        self._montezuma_visited_second_room = False
        return room_id

    def _update_montezuma_tracking(self):
        room_id = self._get_montezuma_room_id()
        if room_id is None:
            return None
        if self._montezuma_initial_room is None:
            self._montezuma_initial_room = room_id
        if self._montezuma_max_room is None:
            self._montezuma_max_room = room_id
        else:
            self._montezuma_max_room = max(self._montezuma_max_room, room_id)
        if self._montezuma_initial_room is not None and room_id != self._montezuma_initial_room:
            self._montezuma_visited_second_room = True
        return room_id

    def _augment_info(self, info, room_id):
        info = dict(info) if info is not None else {}
        if self._is_montezuma:
            info['montezuma_room_id'] = room_id
            info['montezuma_visited_second_room'] = self._montezuma_visited_second_room
            info['montezuma_max_room_id'] = self._montezuma_max_room
        return info

    def _process_proprio_obs(self, obs):
        """Process proprioceptive observation, concatenating dict values if needed."""
    
        if isinstance(obs, dict):
            if self.obs_keys is None:
                self.obs_keys = []
                for key in obs.keys():  # Sort for consistent ordering
                    self.obs_keys.append(key)
                print(f"Proprio obs keys order: {self.obs_keys}") 

            # Concatenate all values in the dictionary
            arrays = []
            for key in self.obs_keys:
                value = obs[key]
                if isinstance(value, str):
                    # Text fields such as MiniGrid missions are not part of the numeric
                    # proprio observation. We skip them in this generic fallback path.
                    continue
                arrays.append(np.asarray(value, dtype=np.float32).reshape(-1))
            assert self.obs_keys == list(obs.keys()), f"Expected keys {self.obs_keys}, but got {list(obs.keys())}"  
            return np.concatenate(arrays, dtype=np.float32)
        else:
            return obs

    def step(self, action):
        reward = 0.0
        discount = 1.0
        done = False
        info = {}
        montezuma_room_id = self._get_montezuma_room_id()
        
        for i in range(self._num_repeats):
            obs, reward_step, terminated, truncated, info = self.env.step(action)
            montezuma_room_id = self._update_montezuma_tracking()
            
            done = terminated or truncated
            
            reward += reward_step * discount
            discount *= 0.99  # Standard discount factor
            
            if done:
                break
                
        # Convert gym step to dm_env format for compatibility
        if done:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID
    
        # For Atari (or other envs where obs is already pixels), use obs directly
        if self.obs_type == 'pixels' and len(obs.shape) == 3:
            image_obs = obs
        else:
            image_obs = self.env.render()

        proprio_obs = self._process_proprio_obs(obs)
        info = self._augment_info(info, montezuma_room_id)
        return ExtendedTimeStep(
            step_type=step_type,
            reward=reward,
            discount=discount if not done else 0.0,
            observation=image_obs if self.obs_type == 'pixels' else proprio_obs,  # Use image or proprioceptive observations
            proprio_observation=proprio_obs,
            image_observation=image_obs,
            action=action,
            success=info['success'] if 'success' in info else terminated,
            info=info,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        montezuma_room_id = self._reset_montezuma_tracking()
        # For Atari (or other envs where obs is already pixels), use obs directly
        if self.obs_type == 'pixels' and len(obs.shape) == 3:
            image_obs = obs
        else:
            image_obs = self.env.render()
        proprio_obs = self._process_proprio_obs(obs)
        info = self._augment_info(info, montezuma_room_id)
        # Convert gym reset to dm_env format
        return ExtendedTimeStep(
            step_type=StepType.FIRST,
            reward=0.0,
            discount=1.0,
            observation=image_obs if self.obs_type == 'pixels' else proprio_obs,  # Use image or proprioceptive observations
            proprio_observation=proprio_obs,
            image_observation=image_obs,
            action=np.zeros(self.env.action_space.shape, dtype=self.env.action_space.dtype),
            success=False,
            info=info,
        )
    
    @property
    def physics(self):
        """Forward physics attribute if available."""
        if hasattr(self.env, 'physics'):
            return self.env.physics
        else:
            raise AttributeError(f"{self.__class__.__name__} has no attribute 'physics'")
    
    def __getattr__(self, name):
        """Forward other attributes to the wrapped environment."""
        return getattr(self.env, name)


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, num_frames):
        super().__init__(env)
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        
        # Expose render_resolution if available
        if hasattr(env, 'render_resolution'):
            self.render_resolution = env.render_resolution
        elif hasattr(env, 'resolution'):
            self.render_resolution = env.resolution
        
        # Update observation space to include stacked frames
        obs = env.reset()

        # Get the shape from the observation
        if isinstance(obs.observation, np.ndarray):
            self.orig_obs_shape = obs.observation.shape
            
        else:
            # Handle case where observation might be a different structure
            raise ValueError("Expected observation to be a numpy array")
        
        # Create a new stacked observation space
        channels = self.orig_obs_shape[2] * num_frames
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=255, 
            shape=(channels, self.orig_obs_shape[0], self.orig_obs_shape[1]),
            dtype=np.uint8
        )
        self.proprio_observation_space = env.observation_space

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames, f"Expected {self._num_frames} frames in buffer, but got {len(self._frames)}"
        # Stack frames along the channel dimension (axis 0 after transpose)
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, obs):
        # Transform HWC to CHW format
        if isinstance(obs, np.ndarray):
            return obs.transpose(2, 0, 1).copy()
        else:
            raise ValueError("Expected observation to be a numpy array")

    def reset(self, **kwargs):
        time_step = self.env.reset(**kwargs)
        pixels = self._extract_pixels(time_step.observation)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self.env.step(action)
        pixels = self._extract_pixels(time_step.observation)
        self._frames.append(pixels)
        return self._transform_observation(time_step)
    
    @property
    def physics(self):
        """Forward physics attribute if available."""
        if hasattr(self.env, 'physics'):
            return self.env.physics
        else:
            raise AttributeError(f"{self.__class__.__name__} has no attribute 'physics'")
    
    def __getattr__(self, name):
        """Forward other attributes to the wrapped environment."""
        return getattr(self.env, name)


class ActionDTypeWrapper(gym.Wrapper):
    def __init__(self, env, dtype=np.float32):
        super().__init__(env)
        original_space = env.action_space
        if not isinstance(original_space, gym.spaces.Box):
            self.action_space = gym.spaces.Discrete(original_space.n)
        else:
            self.action_space = gym.spaces.Box(
                low=original_space.low.astype(dtype),
                high=original_space.high.astype(dtype),
                shape=original_space.shape,
                dtype=dtype
            )

    def step(self, action):
        if type(action) != int:
            action = action.astype(self.env.action_space.dtype)
        return self.env.step(action)
    
    def __getattr__(self, name):
        """Forward other attributes to the wrapped environment."""
        return getattr(self.env, name)


class PhysicsStateWrapper(gym.Wrapper):
    """Wrapper che simula l'interfacio physics per il relabelling come in CDMC."""
    
    def __init__(self, env):
        super().__init__(env)
        self._physics_state = None
    
    def _get_physics_state(self):
        """Estrae lo stato fisico dall'ambiente Gymnasium."""
        # Per PointMaze, usiamo la posizione e velocità come stato fisico
        if hasattr(self.env, 'unwrapped'):
            unwrapped = self.env.unwrapped
            if hasattr(unwrapped, 'point_env'):
                # PointMaze environment
                point_env = unwrapped.point_env
                qpos = point_env.data.qpos.copy()
                qvel = point_env.data.qvel.copy()
                return np.concatenate([qpos, qvel])
        
        # Fallback: usa l'osservazione propriocettiva se disponibile
        return self._physics_state if self._physics_state is not None else np.zeros(4)
    
    def _set_physics_state(self, state):
        """Imposta lo stato fisico nell'ambiente."""
        if hasattr(self.env, 'unwrapped'):
            unwrapped = self.env.unwrapped
            if hasattr(unwrapped, 'point_env'):
                # PointMaze environment
                point_env = unwrapped.point_env
                mid = len(state) // 2
                point_env.data.qpos[:] = state[:mid]
                point_env.data.qvel[:] = state[mid:]
                # Forward kinematics to update dependent variables
                mujoco.mj_forward(point_env.model, point_env.data)
    
    def reset(self,**kwargs):
        time_step = self.env.reset(**kwargs)
        self._physics_state = self._get_physics_state()
        return time_step
    
    def step(self, action):
        time_step = self.env.step(action)
        self._physics_state = self._get_physics_state()
        return time_step
    
    @property
    def physics(self):
        """Simula l'interfaccia physics di CDMC."""
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

class IgnoreSuccessTerminationWrapper(gym.Wrapper):
    """Wrapper che ignora la terminazione basata su 'success'."""
    
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Ignora 'success' per la terminazione
        return obs, reward, False, truncated, info
    
    def __getattr__(self, name):
        """Forward other attributes to the wrapped environment."""
        return getattr(self.env, name)
    
class RewardSpecWrapper(gym.Wrapper):
    """Wrapper che aggiunge le specifiche per reward e discount compatibili con CDMC."""
    
    def __init__(self, env):
        super().__init__(env)
        # Verifica che sia un PointMaze environment
        if not hasattr(self.env, 'unwrapped') or not hasattr(self.env.unwrapped, 'compute_reward'):
            raise NotImplementedError("RewardSpecWrapper is currently only implemented for PointMaze environments")
    
    def reward_spec(self):
        """Specifica del reward per compatibilità con replay buffer CDMC."""
        return specs.Array(shape=(1,), dtype=np.float32, name='reward')
    
    def discount_spec(self):
        """Specifica del discount per compatibilità con replay buffer CDMC."""
        return specs.Array(shape=(1,), dtype=np.float32, name='discount')
    
    def compute_reward_from_state_and_action(self, physics_state, action, desired_goal=None):
        """Calcola il reward usando state e goal, senza fare uno step nell'environment."""
        unwrapped = self.env.unwrapped
        
        # Salva lo stato corrente e il goal corrente
        original_state = self.physics.state()
        original_goal = unwrapped.goal.copy()
        
        try:
            # Se desired_goal è fornito, usalo, altrimenti prendilo dagli ultimi elementi dello stato
            if desired_goal is not None:
                goal_to_use = desired_goal.copy()
            else:
                # Assumiamo che il goal sia negli ultimi 2 elementi del physics_state
                goal_to_use = physics_state[-2:].copy()
            
            # Estrai achieved_goal (posizione corrente) dai primi 2 elementi dello stato
            achieved_goal = physics_state[:2].copy()
            
            # Calcola il reward direttamente usando compute_reward
            if hasattr(unwrapped, 'compute_reward'):
                reward = unwrapped.compute_reward(achieved_goal, goal_to_use, {})
                return np.array([reward], dtype=np.float32)
            else:
                raise NotImplementedError("compute_reward method not found in environment")
            
        finally:
            # Ripristina lo stato originale (non necessario in questo caso ma per sicurezza)
            self.physics.set_state(original_state)
            # Ripristina sempre il goal originale
            unwrapped.goal = original_goal
            if hasattr(unwrapped, 'update_target_site_pos'):
                unwrapped.update_target_site_pos()
    
    def compute_reward_from_obs_dict(self, obs_dict, action=None):
        """Calcola il reward da un dizionario di osservazioni (formato PointMaze)."""
        if not all(key in obs_dict for key in ['achieved_goal', 'desired_goal']):
            raise ValueError("obs_dict must contain 'achieved_goal' and 'desired_goal' keys")
        
        achieved_goal = obs_dict['achieved_goal']
        desired_goal = obs_dict['desired_goal']
        
        # Per PointMaze, il reward dipende solo dalla posizione, non dallo stato fisico completo
        # quindi possiamo calcolare direttamente
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, 'compute_reward'):
            reward = unwrapped.compute_reward(achieved_goal, desired_goal, {})
            return np.array([reward], dtype=np.float32)
        else:
            raise NotImplementedError("compute_reward method not found in environment")
    
    
    def __getattr__(self, name):
        """Forward other attributes to the wrapped environment."""
        return getattr(self.env, name)


class ExtendedTimeStepWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # Expose render_resolution if available
        if hasattr(env, 'render_resolution'):
            self.render_resolution = env.render_resolution
        elif hasattr(env, 'resolution'):
            self.render_resolution = env.resolution

    def reset(self, **kwargs):
        time_step = self.env.reset(**kwargs)
        return time_step

    def step(self, action):
        time_step = self.env.step(action)
        return time_step
    
    def reward_spec(self):
        """Reward spec for compatibility with replay buffer."""
        if hasattr(self.env, 'reward_spec'):
            return self.env.reward_spec()
        return specs.Array(shape=(1,), dtype=np.float32, name='reward')
    
    def discount_spec(self):
        """Discount spec for compatibility with replay buffer."""
        if hasattr(self.env, 'discount_spec'):
            return self.env.discount_spec()
        return specs.Array(shape=(1,), dtype=np.float32, name='discount')
    
    @property
    def physics(self):
        """Forward physics attribute if available."""
        if hasattr(self.env, 'physics'):
            return self.env.physics
        else:
            raise AttributeError(f"{self.__class__.__name__} has no attribute 'physics'")
    
    def __getattr__(self, name):
        """Forward other attributes to the wrapped environment."""
        return getattr(self.env, name)

class TerminateOnPoint(gym.Wrapper):
    """Termina l'episodio immediatamente se il reward è -1 (punto perso)."""
    
    def __init__(self, env):
        super().__init__(env)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Se reward < > 1 (cioè -1 in Pong), termina l'episodio
        if reward != 0:
            terminated = True
            
        return obs, reward, terminated, truncated, info

def observation_spec(env):
    """Get observation spec of the environment for agent initialization."""
    shape = env.observation_space.shape
    if len(shape) == 1:
        return specs.Array(shape, np.float32, 'observation')
    elif len(shape) == 3:
        return specs.Array(shape, np.uint8, 'observation')
    # return specs.Array(shape, np.float32, 'observation')


def action_spec(env):
    """Get action spec of the environment for agent initialization."""
    if isinstance(env.action_space, spaces.Discrete):
        # For discrete action space
        return specs.DiscreteArray(env.action_space.n, name='action', dtype=env.action_space.dtype)
    else:
        # For continuous action space
        shape = env.action_space.shape
        min_action = env.action_space.low[0]
        max_action = env.action_space.high[0]
        return specs.BoundedArray(shape, np.float32, min_action, max_action, 'action')


def _normalize_obs_type(obs_type):
    if obs_type is None:
        return obs_type

    normalized = str(obs_type).strip().lower()
    aliases = {
        "state": "discrete_states",
        "states": "discrete_states",
        "discrete_state": "discrete_states",
        "discerete_states": "discrete_states",
    }
    return aliases.get(normalized, normalized)


def _is_minigrid_env(env, name):
    spec = getattr(env.unwrapped, "spec", None)
    env_id = getattr(spec, "id", None) or str(name)
    module_name = getattr(env.unwrapped.__class__, "__module__", "").lower()
    return "minigrid" in env_id.lower() or "minigrid" in module_name


def _is_atari_env(env, name):
    return isinstance(env.unwrapped, ale_py.env.AtariEnv) or str(name).startswith("ALE/")


def _build_make_kwargs(name, obs_type, kwargs):
    make_kwargs = dict(kwargs)
    if obs_type == 'pixels' and 'render_mode' not in make_kwargs and 'minigrid' in str(name).lower():
        make_kwargs['render_mode'] = 'rgb_array'
    return make_kwargs


def _wrap_atari_pixels(env, name, action_repeat, grayscale, score_mask, score_mask_band, score_mask_color, resolution):
    print(f"Applying AtariPreprocessing wrapper for {name} with action_repeat={action_repeat} and resolution={resolution}")
    env = AtariPreprocessing(
        env,
        noop_max=0,
        frame_skip=action_repeat,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=grayscale,
        grayscale_newaxis=grayscale,
        scale_obs=False,
    )
    if score_mask:
        env = AtariScoreMaskWrapper(env, band_height=score_mask_band, color=score_mask_color)
    return env, 1


def _apply_family_wrappers(env, name, obs_type, action_repeat, resolution, grayscale, score_mask, score_mask_band, score_mask_color):
    is_atari = _is_atari_env(env, name)
    is_minigrid = _is_minigrid_env(env, name)

    if is_atari and obs_type == 'pixels':
        env, action_repeat = _wrap_atari_pixels(
            env,
            name,
            action_repeat,
            grayscale,
            score_mask,
            score_mask_band,
            score_mask_color,
            resolution,
        )
    elif is_minigrid:
        if obs_type == 'pixels':
            env = ResizeRendering(env, resolution=resolution, grayscale=grayscale)
            env = MiniGridTopDownObservationWrapper(env)
        elif obs_type == 'discrete_states':
            env = MiniGridDiscreteStateWrapper(env)
            env = DiscreteObservationWrapper(env)

    return env, action_repeat, is_atari, is_minigrid


def _reset_for_observation_detection(env, seed):
    if seed is not None:
        return env.reset(seed=seed)
    return env.reset()


def _maybe_wrap_discrete_observation(env, obs_type, is_minigrid, initial_state):
    if not is_minigrid and (obs_type == 'discrete_states' or isinstance(initial_state, (int, np.integer))):
        return DiscreteObservationWrapper(env)
    return env


def _apply_common_wrappers(env, name, obs_type, action_repeat, frame_stack, resolution, grayscale, url, is_atari, is_minigrid, enable_relabelling):
    if url and not is_atari:
        env = IgnoreSuccessTerminationWrapper(env)

    if obs_type == 'pixels' and not is_atari and not is_minigrid:
        env = ResizeRendering(env, resolution=resolution, grayscale=grayscale)

    env = ActionDTypeWrapper(env, np.float32)

    if enable_relabelling:
        assert name.startswith('PointMaze'), "Relabelling wrappers are only implemented for PointMaze environments"
        env = PhysicsStateWrapper(env)
        env = RewardSpecWrapper(env)

    env = ActionRepeatWrapper(env, action_repeat, obs_type)

    print(f"Action repeat wrapper applied with num_repeats={action_repeat} and obs_type={obs_type}, frame_stack={frame_stack}")
    if obs_type == 'pixels':
        env = FrameStackWrapper(env, frame_stack)

    return ExtendedTimeStepWrapper(env)

def make(name, obs_type, frame_stack=1, action_repeat=1, seed=None, resolution=224, random_init=True, randomize_goal=True, enable_relabelling=False, url = False, discretize=False, cell_size=1.0, lava=False, score_mask=False, score_mask_band=None, score_mask_color=255, grayscale=False, **kwargs):
    """
    Create a Gymnasium environment with wrappers.
    
    Args:
        name: Environment name (e.g., 'PointMaze_Medium-v3')
        frame_stack: Number of frames to stack
        action_repeat: Number of times to repeat each action
        seed: Random seed
        resolution: Image resolution
        random_init: Se True, usa posizioni iniziali casuali
        randomize_goal: Se True, usa goal casuali
        enable_relabelling: Se True, aggiunge i wrapper per il relabelling CDMC
        discretize: Se True, discretizza l'environment continuo
        cell_size: Dimensione delle celle per la discretizzazione
        lava: Se True, le mosse invalide portano a uno stato dead
    
    Returns:
        Wrapped environment
    """
    obs_type = _normalize_obs_type(obs_type)
    make_kwargs = _build_make_kwargs(name, obs_type, kwargs)
    env = gym.make(name, **make_kwargs)
    env, action_repeat, is_atari, is_minigrid = _apply_family_wrappers(
        env,
        name,
        obs_type,
        action_repeat,
        resolution,
        grayscale,
        score_mask,
        score_mask_band,
        score_mask_color,
    )

    # Assert that render_mode is 'rgb_array' if pixels observation is requested
    if obs_type == 'pixels':
        assert env.render_mode == 'rgb_array', \
            f"render_mode must be 'rgb_array' for pixel observations, got {env.render_mode}"

    state, _ = _reset_for_observation_detection(env, seed)
    env = _maybe_wrap_discrete_observation(env, obs_type, is_minigrid, state)

    return _apply_common_wrappers(
        env,
        name,
        obs_type,
        action_repeat,
        frame_stack,
        resolution,
        grayscale,
        url,
        is_atari,
        is_minigrid,
        enable_relabelling,
    )
def make_kwargs(cfg):
    """Return default kwargs for make function."""
    env_kwargs = {}
    
    for key, value in cfg.env.items():
        if key not in ['name']:
            env_kwargs[key] = value
    
    
    if hasattr(cfg.env, 'dense_reward'):
        env_kwargs['dense_reward'] = cfg.env.dense_reward
    if hasattr(cfg.env, 'num_actions'):
        env_kwargs['num_actions'] = cfg.env.num_actions
    if hasattr(cfg, 'grayscale'):
        env_kwargs['grayscale'] = cfg.grayscale
        
    # Add discretization parameters
    if hasattr(cfg.env, 'discretize') and cfg.env.discretize:
        env_kwargs['discretize'] = True
        env_kwargs['cell_size'] = cfg.env.cell_size if hasattr(cfg.env, 'cell_size') else 1.0
        env_kwargs['lava'] = cfg.env.lava if hasattr(cfg.env, 'lava') else False
    
    # Add environment-specific parameters
    if "SingleRoom" in cfg.env.name:
        env_kwargs['room_size'] = cfg.env.room_size
    elif "TwoRooms" in cfg.env.name:
        env_kwargs['room_size'] = cfg.env.room_size
        env_kwargs['corridor_length'] = cfg.env.corridor_length
        if hasattr(cfg.env, 'corridor_y'):
            env_kwargs['corridor_y'] = cfg.env.corridor_y
        # Continuous TwoRooms parameters
        if hasattr(cfg.env, 'corridor_width'):
            env_kwargs['corridor_width'] = cfg.env.corridor_width
    elif "FourRooms" in cfg.env.name:
        env_kwargs['room_size'] = cfg.env.room_size
        # Check if it's continuous or discrete
        if "Continuous" in cfg.env.name:
            if hasattr(cfg.env, 'corridor_width'):
                env_kwargs['corridor_width'] = cfg.env.corridor_width
            if hasattr(cfg.env, 'corridor_offset'):
                env_kwargs['corridor_offset'] = cfg.env.corridor_offset
            if hasattr(cfg.env, 'wall_thickness'):
                env_kwargs['wall_thickness'] = cfg.env.wall_thickness
            if hasattr(cfg.env, 'agent_radius'):
                env_kwargs['agent_radius'] = cfg.env.agent_radius
        else:
            env_kwargs['corridor_length'] = cfg.env.corridor_length
            env_kwargs['corridor_positions'] = {
                'horizontal': cfg.env.corridor_positions.horizontal,
                'vertical': cfg.env.corridor_positions.vertical
            }
    elif "MultipleRooms" in cfg.env.name:
        env_kwargs['num_rooms'] = cfg.env.num_rooms
        env_kwargs['room_size'] = cfg.env.room_size
        
        if "dense_reward" in cfg.env:
            env_kwargs['dense_reward'] = cfg.env.dense_reward
            
         # Check if it's continuous or discrete
        if "Continuous" in cfg.env.name:
            if hasattr(cfg.env, 'corridor_width'):
                env_kwargs['corridor_width'] = cfg.env.corridor_width
            if hasattr(cfg.env, 'corridor_length'):
                env_kwargs['corridor_length'] = cfg.env.corridor_length
            if hasattr(cfg.env, 'main_corridor_height'):
                env_kwargs['main_corridor_height'] = cfg.env.main_corridor_height
            
            if hasattr(cfg.env, 'wall_thickness'):
                env_kwargs['wall_thickness'] = cfg.env.wall_thickness
            if hasattr(cfg.env, 'agent_radius'):
                env_kwargs['agent_radius'] = cfg.env.agent_radius
        else:
            env_kwargs['corridor_height'] = cfg.env.corridor_height if 'corridor_height' in cfg.env else 1
            env_kwargs['connector_position'] = cfg.env.connector_position if 'connector_position' in cfg.env else None
        env_kwargs['connector_length'] = cfg.env.connector_length if 'connector_length' in cfg.env else 1
    elif "Corridor" in cfg.env.name:
        env_kwargs['length'] = cfg.env.length
        env_kwargs['height'] = cfg.env.height
        env_kwargs['num_curves'] = cfg.env.num_curves
        env_kwargs['corridor_width'] = cfg.env.corridor_width if 'corridor_width' in cfg.env else 1
        
    return env_kwargs


if __name__ == "__main__":
    def _save_hwc_image(image, filename):
        if image.ndim == 3 and image.shape[0] in (1, 3, 4):
            image = image.transpose(1, 2, 0)
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]
        Image.fromarray(image.astype(np.uint8)).save(filename)

    def _run_minigrid_one_hot_test():
        env = make(
            "MiniGrid-Empty-16x16-v0",
            obs_type="discrete_states",
            render_mode="rgb_array",
            grayscale=True,
        )
        try:
            time_step = env.reset()
            obs = time_step.observation
            print(f"MiniGrid pixel observation shape: {obs.shape}, dtype: {obs.dtype}, observation: {obs}")

            assert obs.ndim == 1, f"Expected 1D one-hot observation, got shape {obs.shape}"
            assert np.issubdtype(obs.dtype, np.floating), f"Expected float one-hot observation, got {obs.dtype}"
            assert np.isclose(obs.sum(), 1.0), f"Expected one-hot sum 1.0, got {obs.sum()}"
            assert np.count_nonzero(obs) == 1, f"Expected a unique active entry, got {np.count_nonzero(obs)}"
            active_idx = int(np.argmax(obs))
            info_idx = int(time_step.info["state_index"])
            assert active_idx == info_idx, f"One-hot index {active_idx} does not match info state_index {info_idx}"
            print(f"[MiniGrid one-hot] OK: shape={obs.shape}, active_index={active_idx}")
        finally:
            env.close()

    def _run_minigrid_pixel_test():
        env = make(
            "MiniGrid-Empty-16x16-v0",
            obs_type="pixels",
            frame_stack=1,
            resolution=96,
            render_mode="rgb_array",
        )
        try:
            time_step = env.reset()
            obs = time_step.observation
            assert obs.ndim == 3, f"Expected stacked image observation, got shape {obs.shape}"
            assert obs.shape[0] in (1, 3), f"Expected CHW image with 1 or 3 channels, got shape {obs.shape}"
            assert obs.dtype == np.uint8, f"Expected uint8 image observation, got {obs.dtype}"
            out_path = os.path.join(os.getcwd(), "minigrid_empty_pixels.png")
            _save_hwc_image(obs, out_path)
            print(f"[MiniGrid pixels] OK: shape={obs.shape}, saved={out_path}")
        finally:
            env.close()

    try:
        _run_minigrid_one_hot_test()
        _run_minigrid_pixel_test()
    except Exception as exc:
        print(f"MiniGrid smoke tests failed: {exc}")

    test_env = make(
        "ALE/MontezumaRevenge-v5",
        obs_type="pixels",
        frame_stack=1,
        action_repeat=4,
        resolution=84,
        score_mask=True,
        score_mask_band=None,
        render_mode="rgb_array",
        frameskip=1
    )

    def to_hwc(obs):
        if isinstance(obs, np.ndarray) and obs.ndim == 3 and obs.shape[0] in (1, 3, 4):
            return obs.transpose(1, 2, 0)
        
        return obs

    time_step = test_env.reset()
    for i in range(5):
        if i > 0:
            action = test_env.action_space.sample()
            time_step = test_env.step(action)
        obs_hwc = to_hwc(time_step.observation)
        out_path = os.path.join(os.getcwd(), f"pong_score_mask_test_{i:02d}.png")
        Image.fromarray(obs_hwc.astype(np.uint8)).save(out_path)
        print(f"Saved score-masked test image to {out_path}")

    test_env.close()
