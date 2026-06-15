from collections import deque
from typing import Any, NamedTuple

import ale_py
import gymnasium as gym
import gymnasium_robotics
import numpy as np
from dm_env import StepType, specs
from gymnasium import spaces
from gymnasium.spaces import utils as spaces_utils
from PIL import Image

import utils
from env.atari_domain import is_atari_env, pop_atari_kwargs, wrap_atari_pixels
from env.continuous_rooms import (
    ContinuousCorridorEnv,
    ContinuousFourRoomsEnv,
    ContinuousMultipleRoomsEnv,
    ContinuousSingleRoomEnv,
    ContinuousTwoRoomsEnv,
)
from env.corridor import CorridorEnv
from env.maze import MazeEnv
from env.classic_minigrid_domain import (
    is_classic_minigrid_env,
    wrap_classic_minigrid_env,
)
from env.fetch_domain import is_fetch_env, prepare_fetch_make_kwargs, wrap_fetch_env
from env.middle_room import MiddleRoomEnv
from env.multiple_rooms import MultipleRoomsEnv
from env.pointmaze_domain import (
    is_point_maze_env,
    prepare_point_maze_make_kwargs,
    wrap_point_maze_env,
)
from env.rooms import *

gym.register_envs(gymnasium_robotics)
gym.register_envs(ale_py)


def _get_env_method(env, method_name):
    """Find a callable on an env or any wrapped env in the wrapper chain."""
    current = env
    visited = set()

    while current is not None and id(current) not in visited:
        visited.add(id(current))
        method = getattr(current, method_name, None)
        if callable(method):
            return method
        current = getattr(current, "env", None)

    return None


def _zero_action(space):
    if isinstance(space, spaces.Discrete):
        return np.array(0, dtype=space.dtype)
    return np.zeros(space.shape, dtype=space.dtype)


class ResizeRendering(gym.Wrapper):

    def __init__(self, env, resolution=224, grayscale=False):
        super().__init__(env)
        self.resolution = resolution
        self.grayscale = grayscale
        self.render_resolution = resolution  # Expose for agent access

    def _resize_image(self, img):
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

    def render(self):
        return self._resize_image(super().render())

    def render_observation(self):
        render_fn = _get_env_method(self.env, "render_observation")
        img = render_fn() if callable(render_fn) else self.env.render()
        return self._resize_image(img)

    def render_image_observation(self):
        render_fn = _get_env_method(self.env, "render_image_observation")
        img = render_fn() if callable(render_fn) else self.env.render()
        return self._resize_image(img)
    
    def set_task(self, task):
        """Set the task for the environment."""
        # Set the task in the base environment
        self.env.set_task(task)
    
    def __getattr__(self, name):
        """Forward other attributes to the wrapped environment."""
        return getattr(self.env, name)


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
    """Wrapper that converts discrete observations to one-hot, and flattens dict states."""
    
    def __init__(self, env):
        super().__init__(env)
        self._flatten_observation = False
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
            if getattr(env.observation_space, "shape", None) is None:
                self._flatten_observation = True
                flat_space = spaces_utils.flatten_space(env.observation_space)
                self.observation_space = spaces.Box(
                    low=np.asarray(flat_space.low, dtype=np.float32),
                    high=np.asarray(flat_space.high, dtype=np.float32),
                    shape=flat_space.shape,
                    dtype=np.float32,
                )
    
    def _obs_to_onehot(self, obs):
        """Convert discrete observation to one-hot or flatten structured observations."""
        if self.is_discrete:
            onehot = np.zeros(self.n_states, dtype=np.float32)
            onehot[obs] = 1.0
            return onehot
        if self._flatten_observation:
            return spaces_utils.flatten(self.env.observation_space, obs).astype(np.float32, copy=False)
        return obs
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._obs_to_onehot(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._obs_to_onehot(obs), reward, terminated, truncated, info
    
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
                    # Text fields such as classic MiniGrid missions are not part of the numeric
                    # proprio observation. We skip them in this generic fallback path.
                    continue
                arrays.append(np.asarray(value, dtype=np.float32).reshape(-1))
            assert self.obs_keys == list(obs.keys()), f"Expected keys {self.obs_keys}, but got {list(obs.keys())}"  
            return np.concatenate(arrays, dtype=np.float32)
        else:
            return obs

    def _render_pixels(self, include_goal=True):
        render_method_name = "render_image_observation" if include_goal else "render_observation"
        render_method = _get_env_method(self.env, render_method_name)
        if callable(render_method):
            return render_method()
        return self.env.render()

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

        proprio_obs = self._process_proprio_obs(obs)
        # Only render pixels for pixel observations. State-only PointMaze runs should
        # not require an OpenGL context just to build ExtendedTimeStep fields.
        if self.obs_type == 'pixels':
            if isinstance(obs, np.ndarray) and obs.ndim == 3:
                pixel_obs = obs
                image_obs = obs
            else:
                pixel_obs = self._render_pixels(include_goal=False)
                image_obs = self._render_pixels(include_goal=True)
        else:
            pixel_obs = None
            image_obs = None

        info = self._augment_info(info, montezuma_room_id)
        return ExtendedTimeStep(
            step_type=step_type,
            reward=reward,
            discount=discount if not done else 0.0,
            observation=pixel_obs if self.obs_type == 'pixels' else proprio_obs,
            proprio_observation=proprio_obs,
            image_observation=image_obs,
            action=action,
            success=info['success'] if 'success' in info else terminated,
            info=info,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        montezuma_room_id = self._reset_montezuma_tracking()
        proprio_obs = self._process_proprio_obs(obs)
        # For Atari (or other envs where obs is already pixels), use obs directly.
        if self.obs_type == 'pixels':
            if isinstance(obs, np.ndarray) and obs.ndim == 3:
                pixel_obs = obs
                image_obs = obs
            else:
                pixel_obs = self._render_pixels(include_goal=False)
                image_obs = self._render_pixels(include_goal=True)
        else:
            pixel_obs = None
            image_obs = None

        info = self._augment_info(info, montezuma_room_id)
        # Convert gym reset to dm_env format
        return ExtendedTimeStep(
            step_type=StepType.FIRST,
            reward=0.0,
            discount=1.0,
            observation=pixel_obs if self.obs_type == 'pixels' else proprio_obs,
            proprio_observation=proprio_obs,
            image_observation=image_obs,
            action=_zero_action(self.env.action_space),
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


class IgnoreSuccessTerminationWrapper(gym.Wrapper):
    """Ignore episode termination that is based only on a success flag."""
    
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, False, truncated, info
    
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
    """Terminate the episode as soon as a point is scored or lost."""
    
    def __init__(self, env):
        super().__init__(env)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
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


def _prepare_env_kwargs(name, obs_type, kwargs):
    env_kwargs = dict(kwargs)
    if obs_type == 'pixels' and 'render_mode' not in env_kwargs and is_classic_minigrid_env(name):
        env_kwargs['render_mode'] = 'rgb_array'
    return env_kwargs


def _prepare_family_make_kwargs(name, env_kwargs, url):
    atari_kwargs = {}
    fetch_kwargs = {}
    pointmaze_kwargs = {}

    has_atari_config = "atari" in env_kwargs or any(
        key in env_kwargs for key in ("score_mask", "score_mask_band", "score_mask_color")
    )
    if has_atari_config:
        print(f"Name '{name}' has Atari config kwargs, preparing Atari-specific kwargs")
        if not is_atari_env(name):
            raise TypeError("'atari' kwargs are only supported for Atari environments")
        atari_kwargs = pop_atari_kwargs(env_kwargs)

    if is_fetch_env(name):
        env_kwargs, fetch_kwargs = prepare_fetch_make_kwargs(env_kwargs)
    elif is_point_maze_env(name):
        env_kwargs, pointmaze_kwargs = prepare_point_maze_make_kwargs(name, env_kwargs, url=url)

    return env_kwargs, atari_kwargs, fetch_kwargs, pointmaze_kwargs


def _apply_family_wrappers(
    env,
    name,
    obs_type,
    action_repeat,
    resolution,
    grayscale,
    atari_kwargs,
    fetch_kwargs,
    pointmaze_kwargs,
):
    is_atari = is_atari_env(env) or is_atari_env(name)
    is_classic_minigrid = is_classic_minigrid_env(env) or is_classic_minigrid_env(name)
    is_fetch = is_fetch_env(env) or is_fetch_env(name)
    is_point_maze = is_point_maze_env(env) or is_point_maze_env(name)

    if is_fetch:
        env = wrap_fetch_env(env, fetch_kwargs)
    elif is_point_maze:
        env = wrap_point_maze_env(env, pointmaze_kwargs)
    elif is_atari and obs_type == 'pixels':
        env, action_repeat = wrap_atari_pixels(
            env,
            name,
            action_repeat,
            grayscale,
            atari_kwargs,
        )
    elif is_classic_minigrid:
        env = wrap_classic_minigrid_env(
            env,
            obs_type,
            resolution,
            grayscale,
            resize_rendering_cls=ResizeRendering,
            discrete_observation_wrapper_cls=DiscreteObservationWrapper,
        )

    return env, action_repeat, is_atari, is_classic_minigrid


def _reset_for_observation_detection(env, seed):
    if seed is not None:
        return env.reset(seed=seed)
    return env.reset()


def _maybe_wrap_discrete_observation(env, obs_type, is_classic_minigrid, initial_state):
    if not is_classic_minigrid and (obs_type == 'discrete_states' or isinstance(initial_state, (int, np.integer))):
        return DiscreteObservationWrapper(env)
    return env


def _apply_common_wrappers(env, name, obs_type, action_repeat, frame_stack, resolution, grayscale, url, is_atari, is_classic_minigrid):
    if url and not is_atari:
        env = IgnoreSuccessTerminationWrapper(env)

    if obs_type == 'pixels' and not is_atari and not is_classic_minigrid:
        env = ResizeRendering(env, resolution=resolution, grayscale=grayscale)

    env = ActionDTypeWrapper(env, np.float32)

    env = ActionRepeatWrapper(env, action_repeat, obs_type)

    print(f"Action repeat wrapper applied with num_repeats={action_repeat} and obs_type={obs_type}, frame_stack={frame_stack}")
    if obs_type == 'pixels':
        env = FrameStackWrapper(env, frame_stack)

    return ExtendedTimeStepWrapper(env)

def make(name, obs_type, frame_stack=1, action_repeat=1, seed=None, resolution=224, grayscale=False, url=False, **kwargs):
    """
    Create a Gymnasium environment with wrappers.
    
    Args:
        name: Environment name (e.g., 'PointMaze_Medium-v3')
        frame_stack: Number of frames to stack
        action_repeat: Number of times to repeat each action
        seed: Random seed
        resolution: Image resolution
        kwargs: Additional environment-specific kwargs.
    
    Returns:
        Wrapped environment
    """
    obs_type = _normalize_obs_type(obs_type)
    env_kwargs = _prepare_env_kwargs(name, obs_type, kwargs)
    env_kwargs, atari_kwargs, fetch_kwargs, pointmaze_kwargs = _prepare_family_make_kwargs(name, env_kwargs, url)

    env = gym.make(name, **env_kwargs)
    env, action_repeat, is_atari, is_classic_minigrid = _apply_family_wrappers(
        env,
        name,
        obs_type,
        action_repeat,
        resolution,
        grayscale,
        atari_kwargs,
        fetch_kwargs,
        pointmaze_kwargs,
    )

    # Assert that render_mode is 'rgb_array' if pixels observation is requested
    if obs_type == 'pixels':
        assert env.render_mode == 'rgb_array', \
            f"render_mode must be 'rgb_array' for pixel observations, got {env.render_mode}"

    state, _ = _reset_for_observation_detection(env, seed)
    env = _maybe_wrap_discrete_observation(env, obs_type, is_classic_minigrid, state)

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
        is_classic_minigrid,
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Example usage
    env = make('MultipleRooms-v0', obs_type='pixels', frame_stack=1, action_repeat=1, resolution=84, num_rooms=2, room_size=5, corridor_height=1, connector_length=3, render_mode='rgb_array')
    obs = env.reset()
    print("Observation shape:", obs.observation.shape)
    # save observation image for debug
    if isinstance(obs.observation, np.ndarray) and len(obs.observation.shape) == 3:
        obs_image = obs.observation.transpose(1, 2, 0)  # [H, W, C]
        plt.imsave('debug_obs_image.png', obs_image.astype(np.uint8))
        print("Saved observation image to debug_obs_image.png")

    print("Proprioceptive observation shape:", obs.proprio_observation.shape)
    print("Image observation shape:", obs.image_observation.shape)
    print("Action space:", env.action_space)
    print("Reward spec:", env.reward_spec())
    print("Discount spec:", env.discount_spec())
