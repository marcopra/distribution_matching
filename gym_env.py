from collections import deque
from typing import Any, NamedTuple
import os

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

class ResizeRendering(gym.Wrapper):

    def __init__(self, env, resolution=224):
        super().__init__(env)
        self.resolution = resolution
        self.render_resolution = resolution  # Expose for agent access

    def render(self):
        img = super().render()

        # Convert numpy array to PIL Image
        img = Image.fromarray(img.astype(np.uint8))
        
        # Resize the image
        img_resized = img.resize((self.resolution, self.resolution), Image.LANCZOS)
        
        # Convert back to numpy array
        return np.array(img_resized)
    
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

class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env, num_repeats, obs_type='pixels', data_collection=False):
        super().__init__(env)
        self._num_repeats = num_repeats
        self.data_collection = data_collection
        self.obs_type = obs_type
        self.obs_keys = None
        
        # Expose render_resolution if available
        if hasattr(env, 'render_resolution'):
            self.render_resolution = env.render_resolution
        elif hasattr(env, 'resolution'):
            self.render_resolution = env.resolution

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
                arrays.append(obs[key].flatten())
            assert self.obs_keys == list(obs.keys()), f"Expected keys {self.obs_keys}, but got {list(obs.keys())}"  
            return np.concatenate(arrays, dtype=np.float32)
        else:
            return obs

    def step(self, action):
        reward = 0.0
        discount = 1.0
        done = False
        info = {}
        
        for i in range(self._num_repeats):
            obs, reward_step, terminated, truncated, info = self.env.step(action)
            # Handle success as a termination condition in MetaWorld
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
        image_obs = self.env.render()
        proprio_obs = self._process_proprio_obs(obs)
        return ExtendedTimeStep(
            step_type=step_type,
            reward=reward,
            discount=discount if not done else 0.0,
            observation=image_obs if self.obs_type == 'pixels' else proprio_obs,  # Use image or proprioceptive observations
            proprio_observation=proprio_obs,
            image_observation=image_obs,
            action=action,
            success=info['success'] if 'success' in info else terminated,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        image_obs = self.env.render()
        proprio_obs = self._process_proprio_obs(obs)
        # Convert gym reset to dm_env format
        return ExtendedTimeStep(
            step_type=StepType.FIRST,
            reward=0.0,
            discount=1.0,
            observation=image_obs if self.obs_type == 'pixels' else proprio_obs,  # Use image or proprioceptive observations
            proprio_observation=proprio_obs,
            image_observation=image_obs,
            action=np.zeros(self.env.action_space.shape, dtype=self.env.action_space.dtype),
            success=False
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
        assert len(self._frames) == self._num_frames
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

def make(name, obs_type, frame_stack=1, action_repeat=1, seed=None, resolution=224, random_init=True, randomize_goal=True, enable_relabelling=False, url = False, discretize=False, cell_size=1.0, lava=False, **kwargs):
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
    
    
    env = gym.make(name, **kwargs)
    
    # Apply discretization wrapper for continuous environments
    if discretize:
        env = DiscretizedContinuousEnv(env, cell_size=cell_size, lava=lava)
    
    # Assert that render_mode is 'rgb_array' if pixels observation is requested
    if obs_type == 'pixels':
        assert env.render_mode == 'rgb_array', \
            f"render_mode must be 'rgb_array' for pixel observations, got {env.render_mode}"

    if seed is not None:
        state, _ = env.reset(seed=seed)
    else:
        state, _ = env.reset()

    if obs_type == 'discrete_states' or type(state) == int:
        env = DiscreteObservationWrapper(env)
    
    if url:
        env = IgnoreSuccessTerminationWrapper(env)
    
    # Add wrappers
    if obs_type == 'pixels':
        env = ResizeRendering(env, resolution=resolution)   
    env = ActionDTypeWrapper(env, np.float32)
    
    # Add relabelling wrappers if requested
    if enable_relabelling:
        assert name.startswith('PointMaze'), "Relabelling wrappers are only implemented for PointMaze environments"
        env = PhysicsStateWrapper(env)
        env = RewardSpecWrapper(env)
    
    env = ActionRepeatWrapper(env, action_repeat, obs_type)
    
    if obs_type == 'pixels':
        env = FrameStackWrapper(env, frame_stack)
    
    env = ExtendedTimeStepWrapper(env)
    
    return env

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
# Tests
if __name__ == "__main__":
    import pathlib
    from replay_buffer import ReplayBufferStorage
    from dm_env import specs
    
    def test_reward_consistency():
        """Test che il reward calcolato dai wrapper corrisponda a quello nei file npz."""
        print("Testing reward consistency...")
        
        # Crea ambiente con relabelling abilitato
        env = make('PointMaze_MediumDense-v3', enable_relabelling=True)
        
        # Setup specs per replay buffer (usando proprio_observation concatenato)
        proprio_shape = (6,)  # observation + achieved_goal + desired_goal concatenati
        data_specs = (
            observation_spec(env),
            action_spec(env),
            specs.Array((1,), np.float32, 'reward'),
            specs.Array((1,), np.float32, 'discount'),
            specs.Array(proprio_shape, np.float32, 'proprio_observation')
        )
        
        # Directory buffer (assumendo che esista)
        buffer_dir = pathlib.Path('/home/mprattico/Pretrain-TACO/exp_local/prova2/buffer')
        if not buffer_dir.exists():
            print("Buffer directory './buffer' not found. Creating empty test...")
            return
        
        # Carica storage
        replay_storage = ReplayBufferStorage(data_specs, buffer_dir)
        
        if len(replay_storage) == 0:
            print("No episodes found in buffer directory")
            return
        
        # Carica alcuni episodi npz per test
        npz_files = list(buffer_dir.glob('*.npz'))
        if not npz_files:
            print("No .npz files found in buffer directory")
            return
            
        print(f"Found {len(npz_files)} episodes to test")
        
        # Test su primi 3 episodi
        for i, npz_file in enumerate(npz_files[:3]):
            print(f"\nTesting episode {i+1}: {npz_file.name}")
            
            # Carica episodio
            episode_data = np.load(npz_file)
            
            # Verifica chiavi richieste
            required_keys = ['reward', 'proprio_observation']
            if not all(key in episode_data for key in required_keys):
                print(f"Skipping episode {i+1}: missing required keys")
                continue
            
            # Test su 5 transizioni casuali dell'episodio
            episode_len = len(episode_data['reward'])
            test_indices = np.random.choice(episode_len, min(5, episode_len), replace=False)
            
            for j, idx in enumerate(test_indices):
                # Estrai dati originali
                original_reward = episode_data['reward'][idx][0]
                proprio_obs = episode_data['proprio_observation'][idx]
                
                # Decomponi proprio_observation (observation + achieved_goal + desired_goal)
                # Formato: [observation(4), achieved_goal(2), desired_goal(2)] = 8 total
                # Ma dovrebbe essere 6 secondo proprio_shape, quindi probabilmente diverso
                
                if len(proprio_obs) >= 6:
                    # Formato: [observation(4), achieved_goal(2), desired_goal(2)] = 8 total
                    # Ma per il calcolo dobbiamo usare physics_state con goal incluso
                    observation = proprio_obs[:4]  # primi 4 valori (observation)  
                    achieved_goal = proprio_obs[4:6] if len(proprio_obs) >= 8 else proprio_obs[:2]  # achieved_goal
                    desired_goal = proprio_obs[6:8] if len(proprio_obs) >= 8 else proprio_obs[2:4]  # desired_goal
                    
                    # Crea physics_state con goal incluso come ultimi elementi
                    physics_state = np.concatenate([observation, desired_goal])
                    
                    # Estrai action se disponibile
                    if 'action' in episode_data:
                        action = episode_data['action'][idx]
                    else:
                        action = np.zeros(2, dtype=np.float32)  # dummy action per PointMaze
                    
                    try:
                        # Calcola reward usando state e action
                        calculated_reward = env.compute_reward_from_state_and_action(proprio_obs, action)
                        calculated_reward_value = calculated_reward[0]
                        
                        # Confronta rewards (aggiustiamo la tolleranza)
                        reward_diff = abs(original_reward - calculated_reward_value)
                        
                        print(f"  Transition {j+1} (idx {idx}):")
                        print(f"    Achieved goal: {achieved_goal}")
                        print(f"    Desired goal: {desired_goal}")
                        print(f"    Original reward: {original_reward:.6f}")
                        print(f"    Calculated reward: {calculated_reward_value:.6f}")
                        print(f"    Difference: {reward_diff:.6f}")
                        
                        if reward_diff < 1e-5:
                            print(f"    ✓ Match!")
                        else:
                            print(f"    ✗ Mismatch!")
                            
                    except Exception as e:
                        print(f"    Error calculating reward: {e}")
                        import traceback
                        traceback.print_exc()
                        
                else:
                    print(f"  Unexpected proprio_observation shape: {proprio_obs.shape}")
            
        print("\nReward consistency test completed.")
    
    # Esegui test
    test_reward_consistency()