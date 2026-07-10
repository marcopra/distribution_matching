from collections import OrderedDict
import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils
from PIL import Image
import logging
# set logging level to info

def _resolve_torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    dtype = str(dtype).lower()
    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "32": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
        "double": torch.float64,
        "64": torch.float64,
    }
    if dtype not in dtype_map:
        raise ValueError("compute_dtype must be one of: float32, fp32, float64, fp64, double")
    return dtype_map[dtype]


torch.set_default_dtype(_resolve_torch_dtype(os.environ.get("ROVER_COMPUTE_DTYPE", "float32")))

logger = logging.getLogger("myapp")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)

logger.addHandler(handler)
from agent.rover_buffers import EncodedTransitionFIFO
from agent.rover_matchers import DistributionMatcher
from agent.rover_networks import CNNEncoder, Encoder, ProjectSA
from agent.rover_visualization.exploration import ExplorationVisualizer
from agent.rover_visualization.gridworld import EmbeddingDistributionVisualizerV2
from agent.rover_visualization.suite import build_debug_visualizer_suite

# ============================================================================
# Main Agent
# ============================================================================
class RoverAgent:
    def __init__(self,
                 name,
                 obs_type,
                 obs_shape,
                 grayscale,
                 action_shape,
                 lr_actor,
                 discount,
                 lambda_reg,
                 batch_size,
                 batch_size_actor,
                 subsamples,
                 nstep,
                 use_tb,
                 use_wandb,
                 lr_T,
                 lr_encoder,
                 curl,
                 embedding_sum_loss,
                 hidden_dim,
                 feature_dim,
                 update_every_steps,
                 update_actor_every_steps,
                 pmd_steps,
                 num_expl_steps,
                 T_init_steps,
                 total_train_steps,
                 sink_schedule,
                 epsilon_schedule,
                 mode,
                 reward,
                 svd_truncation,
                 embeddings = True,
                 linear_projection = False,
                 pmd_eta_mode: str = "none",
                 pmd_best_iterate: bool = True,
                 pmd_grad_clip_norm: float = 0.0,
                 pmd_adagrad_eps: float = 1e-8,
                 pmd_eta_min: float = 1e-8,
                 pmd_eta_max: float = 1e3,
                 pmd_backtrack_factor: float = 0.5,
                 pmd_backtrack_max_trials: int = 8,
                 compute_dtype: str = "float32",
                 nystrom_synthetic_subsamples: bool = False,
                 synthetic_dataset_sub: Optional[object] = None,
                 syntethic_dataset_sub: Optional[object] = None,
                 synthetic_dataset_exclude_states: Optional[object] = None,
                 encoded_fifo_capacity: Optional[int] = None,
                 encoded_fifo_encode_batch_size: int = 4096,
                 encoded_fifo_cuda_oom_splits: int = 4,
                 device: str = "cpu",
                 ):

        self.compute_dtype = _resolve_torch_dtype(compute_dtype)
        torch.set_default_dtype(self.compute_dtype)

        self.n_states = obs_shape[0]
        self.n_actions = action_shape[0]
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        self.grayscale = grayscale
        self.feature_dim = feature_dim if feature_dim is not None else self.n_states
        self.action_dim = action_shape[0]
        self.latent_a_dim = int(self.action_dim * 1.25) + 1 # From TACO
        self.lr_actor = lr_actor
        self.discount = discount
        self.lr_T = lr_T
        self.T_init_steps = T_init_steps
        self.batch_size = batch_size
        self.batch_size_actor = batch_size_actor
        # assert batch_size_actor >= batch_size, "Actor update batch size must be greater than or equal to encoder update batch size"
        self.update_every_steps = update_every_steps
        self.update_actor_every_steps = update_actor_every_steps
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.device = device
        self.pmd_steps = pmd_steps
        self.embeddings = embeddings
        self.curl = curl
        if curl:
            utils.ColorPrint.red("CURL is enabled, but stromgly suggested to not use it.\nAll the paper results are without CURL, and it may cause poor performance. Use with caution.")

        self.embedding_sum_loss = embedding_sum_loss
        self.reward = reward
        self.pmd_eta_mode = pmd_eta_mode.lower()
        assert self.pmd_eta_mode in ["none", "adagrad", "backtracking", "adadiff"], "pmd_eta_mode must be one of ['none', 'adagrad', 'backtracking', 'adadiff']"
        self.pmd_best_iterate = pmd_best_iterate
        self.pmd_grad_clip_norm = pmd_grad_clip_norm
        self.pmd_adagrad_eps = pmd_adagrad_eps
        self.pmd_eta_min = pmd_eta_min
        self.pmd_eta_max = pmd_eta_max
        self.pmd_backtrack_factor = pmd_backtrack_factor
        self.pmd_backtrack_max_trials = pmd_backtrack_max_trials

        self.mode = mode
        assert self.mode in ['l1', 'l2'], "Mode must be 'l1' or 'l2'"

        self.first_save = False
        self.sink_schedule = sink_schedule
        self.epsilon_schedule = epsilon_schedule
        self.gradient_coeff = None
        self.svd_truncation = svd_truncation

        self.num_expl_steps = num_expl_steps
        self.lambda_reg = lambda_reg
        self.image_channels = 1 if self.grayscale else 3
        self.subsamples = subsamples
        self.nystrom_synthetic_subsamples = bool(nystrom_synthetic_subsamples)
        if synthetic_dataset_sub is not None and syntethic_dataset_sub is not None:
            raise ValueError("Use only one of synthetic_dataset_sub or syntethic_dataset_sub.")
        self.synthetic_dataset_sub = synthetic_dataset_sub
        if syntethic_dataset_sub is not None:
            self.synthetic_dataset_sub = syntethic_dataset_sub
        self.synthetic_dataset_exclude_states = synthetic_dataset_exclude_states
        self._synthetic_nystrom_subsample_batch = None
        self._synthetic_state_indices_cache = None
        self._synthetic_state_stride_cache = None
        min_fifo_capacity = max(
            int(self.batch_size_actor),
            int(self.subsamples) if self.subsamples is not None else 0,
            1,
        )
        if encoded_fifo_capacity is None:
            encoded_fifo_capacity = min_fifo_capacity
        self.encoded_fifo_capacity = int(encoded_fifo_capacity)
        if self.encoded_fifo_capacity < min_fifo_capacity:
            utils.ColorPrint.yellow(
                f"encoded_fifo_capacity={self.encoded_fifo_capacity} is smaller than "
                f"the actor sample size; raising it to {min_fifo_capacity}."
            )
            self.encoded_fifo_capacity = min_fifo_capacity
        self.encoded_fifo_encode_batch_size = int(encoded_fifo_encode_batch_size)
        self.encoded_fifo_cuda_oom_splits = int(encoded_fifo_cuda_oom_splits)
        self._encoded_actor_fifo = EncodedTransitionFIFO(self.encoded_fifo_capacity)
        self._encoded_fifo_replay_marker = None
        
        # Track unique state-action pairs from previous dataset
        self._previous_unique_pairs = set()
        self._previous_unique_next_states = set()
        self._dataset_novelty_stats = {
            'total_current': 0,
            'new_pairs': 0,
            'old_pairs': 0,
            'new_percentage': 0.0,
            'total_previous': 0,
            'new_next_states': 0,
            'old_next_states': 0,
            'next_states_new_percentage': 0.0,
            'total_previous_next_states': 0
        }
        
        if obs_type == 'pixels':
            if self.curl:
                self.aug = utils.RandomShiftsAug(pad=4)
            else:
                self.aug = nn.Identity()
            assert embeddings, "Pixel observations require embeddings to be True"
            self.encoder = CNNEncoder(
                obs_shape,
                feature_dim,
                mode=mode
            ).to(self.device)
            
            self.obs_dim = self.feature_dim
        else:
            # Components
            self.aug = nn.Identity()
            if embeddings == False:
                self.encoder = nn.Identity()
                self.feature_dim = obs_shape[0]
                utils.ColorPrint.yellow("WARNING: Using identity encoder for state observations")
            else:
                self.encoder = Encoder(
                        obs_shape, 
                        hidden_dim, 
                        self.feature_dim,
                    ).to(self.device)
            self.obs_dim = self.feature_dim
       

        self.project_sa = ProjectSA(
            self.obs_dim * self.n_actions,
            hidden_dim,
            self.obs_dim,
            linear=linear_projection
        ).to(self.device)

        
        self.policy_encoder = copy.deepcopy(self.encoder).to(self.device)
        self._freeze_module(self.policy_encoder)
        self._policy_is_synced = True
        
        self.distribution_matcher = DistributionMatcher(
            gamma=self.discount,
            lambda_reg=self.lambda_reg,
            svd_truncation=self.svd_truncation,
            device=self.device  
        )
        
        self.W = None #nn.Parameter(torch.rand(feature_dim, feature_dim).to(self.device))
       
        if self.reward:
            self.reward = nn.Sequential(
                nn.Linear(self.obs_dim * self.n_actions, hidden_dim), 
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1)
            ).to(self.device)
        
        # parameter list:
        parameters = list(self.encoder.parameters()) 
        if self.W is not None:
            parameters+= [self.W]
        else:
            self.W = nn.Identity()
        if self.reward:
            parameters += list(self.reward.parameters())
        
        # Optimizers
        if embeddings:
            self.encoder_optimizer = torch.optim.AdamW(
                parameters,
                lr=lr_encoder,
                weight_decay=1e-5,   # try 1e-6 to 1e-4
                betas=(0.9, 0.999),
                eps=1e-8
            )

            # decay to 10% of initial LR over total training steps
            self.encoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.encoder_optimizer,
                T_max=total_train_steps,   # pass from config
                eta_min=lr_encoder * 0.1)
                        
        else:
            self.encoder_optimizer = None
        self.transition_optimizer = torch.optim.Adam(
            self.project_sa.parameters(),
            lr=lr_T
        )
        
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.training = False

        self.current_action_probs = []
        self.action_probs_history = []  # List of [step, mean_action_probs_array]
        self.policy_deviation_history = []  # List of [step, deviation_value]

    
        self.debug_visualizer = build_debug_visualizer_suite(
            agent=self,
            exploration_visualizer_cls=ExplorationVisualizer,
            gridworld_visualizer_cls=EmbeddingDistributionVisualizerV2,
        )
        self.visualizer = self.debug_visualizer.exploration_visualizer
        self.gridworld_visualizer = None
        self.env = None
        self.wrapped_env = None
        self._discrete_env = None

        # Gradient norm tracking by reward
        self.max_samples_per_reward = 150
        self.gradient_samples = {
            '+1': [],  # List of (step, gradient_norm)
            '-1': [],  # List of (step, gradient_norm)
            '0': []    # List of (step, gradient_norm)
        }
        self.gradient_norm_history = {
            '+1': [],  # List of (step, mean_norm, std_norm)
            '-1': [],
            '0': []
        }
        self.current_eta = 0.0
        self._adagrad_accum = None

        self.subsampled = None

    def _find_discrete_env(self, env):
        current = env
        while current is not None:
            if hasattr(current, "n_states") and hasattr(current, "idx_to_state") and hasattr(current, "state_to_idx"):
                return current

            if hasattr(current, "env"):
                current = current.env
            elif hasattr(current, "unwrapped") and current.unwrapped is not current:
                current = current.unwrapped
            else:
                break
        return env.unwrapped

    def _initial_state_index(self) -> int:
        if self.env is None or not hasattr(self.env, "state_to_idx"):
            raise RuntimeError("A discrete environment must be inserted before building synthetic Nyström subsamples.")

        start_position = getattr(self.env, "start_position", None)
        if start_position is None and hasattr(self.env, "_start_position_param"):
            start_param = getattr(self.env, "_start_position_param")
            if start_param is not None:
                start_position = tuple(start_param)

        if start_position is None and hasattr(self.env, "_set_start_position"):
            start_position = self.env._set_start_position(None)

        if start_position is None:
            start_position = self.env.idx_to_state[0]

        start_position = tuple(start_position)
        if start_position not in self.env.state_to_idx:
            raise ValueError(f"Initial state {start_position} is not a valid state in the environment.")
        return self.env.state_to_idx[start_position]

    def _prepare_rendered_state_image(self, image: np.ndarray, render_resolution: int) -> np.ndarray:
        image = np.asarray(image, dtype=np.uint8)

        if self.grayscale:
            if image.ndim == 3 and image.shape[2] == 1:
                image = image[..., 0]
            elif image.ndim == 3:
                image = np.asarray(Image.fromarray(image).convert('L'))
            elif image.ndim != 2:
                raise ValueError(f"Expected grayscale image to be 2D or HWC, got shape {image.shape}")
        elif image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)

        if image.shape[:2] != (render_resolution, render_resolution):
            image = np.asarray(
                Image.fromarray(image).resize(
                    (render_resolution, render_resolution),
                    Image.LANCZOS,
                )
            )

        if self.grayscale:
            if image.ndim == 2:
                image = image[..., None]
        elif image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)

        if image.ndim != 3 or image.shape[2] != self.image_channels:
            raise ValueError(
                f"Expected image shape [H, W, {self.image_channels}], got {image.shape}"
            )

        return image

    def _build_prerendered_state_observations(self) -> torch.Tensor:
        if self.obs_type != "pixels":
            raise RuntimeError("Prerendered observations are only used for pixel observations.")

        cached = getattr(self.gridworld_visualizer, "_prerendered_states", None)
        if cached is not None:
            return cached.to(self.device)

        required_attrs = ("n_states", "idx_to_state", "render_from_position")
        if self.env is None or not all(hasattr(self.env, attr) for attr in required_attrs):
            raise RuntimeError(
                "Pixel synthetic Nyström subsamples require a discrete env with "
                "n_states, idx_to_state, and render_from_position."
            )

        render_resolution = getattr(self.wrapped_env, "render_resolution", self.obs_shape[-1])
        frame_stack = self.obs_shape[0] // self.image_channels
        rendered_states = []

        for state_idx in range(self.env.n_states):
            state = self.env.idx_to_state[state_idx]
            try:
                image = self.env.render_from_position(state, show_goal=False)
            except TypeError:
                image = self.env.render_from_position(state)
            image = self._prepare_rendered_state_image(image, render_resolution)
            image_chw = image.transpose(2, 0, 1).copy()
            rendered_states.append(np.tile(image_chw, (frame_stack, 1, 1)))

        return torch.from_numpy(np.stack(rendered_states)).float().to(self.device)

    def _build_indexed_state_observations(self) -> torch.Tensor:
        if self.obs_type == "pixels":
            return self._build_prerendered_state_observations()

        obs = np.zeros((self.env.n_states, *self.obs_shape), dtype=np.float32)
        flat = obs.reshape(self.env.n_states, -1)
        if self.env.n_states > flat.shape[1]:
            raise ValueError(
                f"State count {self.env.n_states} does not fit observation shape {self.obs_shape}; "
                "expected one-hot discrete observations."
            )
        flat[np.arange(self.env.n_states), np.arange(self.env.n_states)] = 1.0
        return torch.from_numpy(obs).to(self.device)

    def _synthetic_excluded_state_indices(self):
        excluded = self.synthetic_dataset_exclude_states
        if excluded is None:
            return set()

        if isinstance(excluded, str):
            values = [value.strip() for value in excluded.split(",") if value.strip()]
        elif isinstance(excluded, (list, tuple, set)):
            values = list(excluded)
        else:
            values = [excluded]

        indices = {int(value) for value in values}
        invalid = [idx for idx in indices if idx < 0 or idx >= self.env.n_states]
        if invalid:
            raise ValueError(
                f"synthetic_dataset_exclude_states contains invalid state indices {invalid}; "
                f"valid range is [0, {self.env.n_states - 1}]."
            )
        return indices

    def _synthetic_state_indices(self):
        if self._synthetic_state_indices_cache is not None:
            return self._synthetic_state_indices_cache, self._synthetic_state_stride_cache

        excluded = self._synthetic_excluded_state_indices()
        candidate_indices = [idx for idx in range(self.env.n_states) if idx not in excluded]
        if not candidate_indices:
            raise ValueError("synthetic_dataset_exclude_states removed every state.")

        if self.synthetic_dataset_sub is None:
            indices = candidate_indices
            stride = 1
            self._synthetic_state_indices_cache = indices
            self._synthetic_state_stride_cache = stride
            return indices, stride

        if isinstance(self.synthetic_dataset_sub, str):
            value = self.synthetic_dataset_sub.strip().lower()
            if value in ("half", "n/2", "0.5"):
                num_states = max(1, self.env.n_states // 2)
            else:
                num_states = int(value)
        else:
            num_states = int(self.synthetic_dataset_sub)
        if num_states <= 0:
            raise ValueError("synthetic_dataset_sub must be positive or null.")
        if num_states > len(candidate_indices):
            raise ValueError(
                f"synthetic_dataset_sub={num_states} requests more states than available "
                f"after exclusions ({len(candidate_indices)})."
            )
        if num_states == len(candidate_indices):
            indices = candidate_indices
            stride = 1
            self._synthetic_state_indices_cache = indices
            self._synthetic_state_stride_cache = stride
            return indices, stride

        stride = max(1, len(candidate_indices) // num_states)
        indices = candidate_indices[::stride][:num_states]
        self._synthetic_state_indices_cache = indices
        self._synthetic_state_stride_cache = stride
        return indices, stride

    def _build_synthetic_nystrom_subsample_batch(self):
        """Build one landmark transition for every state-action pair."""
        required_attrs = (
            "n_states",
            "idx_to_state",
            "state_to_idx",
            "step_from",
        )
        if self.env is None or not all(hasattr(self.env, attr) for attr in required_attrs):
            raise RuntimeError(
                "Synthetic Nyström subsamples require a discrete env with "
                "n_states, idx_to_state, state_to_idx, and step_from."
            )

        if self._synthetic_nystrom_subsample_batch is not None:
            return self._synthetic_nystrom_subsample_batch

        state_observations = self._build_indexed_state_observations()
        initial_state_idx = self._initial_state_index()
        state_indices, state_stride = self._synthetic_state_indices()
        transitions = []
        first_transition_idx = None

        for state_idx in state_indices:
            state = self.env.idx_to_state[state_idx]
            for action_idx in range(self.n_actions):
                next_state = self.env.step_from(state, action_idx)
                next_state_idx = self.env.state_to_idx[next_state]
                transition = (state_idx, action_idx, next_state_idx)
                if first_transition_idx is None and next_state_idx == initial_state_idx:
                    first_transition_idx = len(transitions)
                transitions.append(transition)

        if first_transition_idx is None:
            utils.ColorPrint.yellow(
                "Synthetic Nyström subset has no transition whose next state is the initial state; "
                "using the first synthetic transition for alpha[0]."
            )
        else:
            transitions.insert(0, transitions.pop(first_transition_idx))

        obs_indices = torch.as_tensor([s for s, _, _ in transitions], dtype=torch.long, device=self.device)
        action = torch.as_tensor([a for _, a, _ in transitions], dtype=torch.long, device=self.device)
        next_obs_indices = torch.as_tensor([ns for _, _, ns in transitions], dtype=torch.long, device=self.device)
        obs = state_observations.index_select(0, obs_indices)
        next_obs = state_observations.index_select(0, next_obs_indices)

        rewards = []
        discounts = []
        for _, _, next_state_idx in transitions:
            if hasattr(self.env, "compute_reward_from_observation"):
                reward_value = self.env.compute_reward_from_observation(next_state_idx)
            else:
                reward_value = 0.0
            next_state = self.env.idx_to_state[next_state_idx]
            is_terminal = next_state == getattr(self.env, "goal_position", None)
            rewards.append([reward_value])
            discounts.append([0.0 if is_terminal else 1.0])

        reward = torch.as_tensor(rewards, dtype=self.compute_dtype, device=self.device)
        discount = torch.as_tensor(discounts, dtype=self.compute_dtype, device=self.device)

        self._synthetic_nystrom_subsample_batch = (obs, action, reward, discount, next_obs)
        if self.synthetic_dataset_sub is None or len(state_indices) == self.env.n_states:
            utils.ColorPrint.yellow(
                f"Using synthetic Nyström subsamples with all transitions: "
                f"{len(transitions)} = {len(state_indices)} states x {self.n_actions} actions."
            )
        else:
            utils.ColorPrint.yellow(
                f"Using synthetic Nyström subsamples with stride {state_stride}: "
                f"{len(transitions)} = {len(state_indices)}/{self.env.n_states} states x {self.n_actions} actions."
            )
        return self._synthetic_nystrom_subsample_batch

    def _encode_synthetic_nystrom_subsamples(self):
        self._sync_policy_encoder()
        synthetic_batch = self._build_synthetic_nystrom_subsample_batch()
        encoded = self._encode_actor_transition_batch_with_retries(synthetic_batch)
        return encoded, encoded.get("reward")
    
    def insert_env(self, env):
        """
        Insert environment reference for gridworld-specific visualizations.
        Call this from pretrain.py after agent creation.
        """       
        self.wrapped_env = env
        self.env = self._find_discrete_env(env)
        self._synthetic_nystrom_subsample_batch = None
        self._synthetic_state_indices_cache = None
        self._synthetic_state_stride_cache = None
        
        try:
            self.gridworld_visualizer = self.debug_visualizer.attach_env(env)
            if self.gridworld_visualizer is not None:
                print("✓ Domain-specific debug visualizer initialized")
        except Exception as e:
            print(f"⚠ Could not initialize domain-specific debug visualizer: {e}")
            self.gridworld_visualizer = None


    
    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.project_sa.train(training)
        self.policy_encoder.eval()

    def init_meta(self):
        return OrderedDict()

    def get_meta_specs(self):
        return tuple()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        return meta
    
    def _encode_state_action(
        self, 
        encoded_obs: torch.Tensor, 
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Encode (s,a) pairs as ψ(s,a) = φ(s) ⊗ e_a."""
        action_onehot = F.one_hot(actions.long(), self.n_actions).reshape(-1, self.n_actions)  # [B, |A|]
        
        # Outer product: [B, d] ⊗ [B, |A|] -> [B, d*|A|]
        encoded_sa = torch.einsum('bd,ba->bda', encoded_obs, action_onehot)
        return encoded_sa.reshape(encoded_obs.shape[0], -1)

    def _freeze_module(self, module: nn.Module) -> None:
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)

    def _sync_policy_encoder(self) -> None:
        self.policy_encoder.load_state_dict(self.encoder.state_dict())
        self._freeze_module(self.policy_encoder)
        self._policy_is_synced = True

    def ready_for_snapshot(self) -> bool:
        return self._policy_is_synced

    def _encode_with_module(self, module: nn.Module, obs: torch.Tensor, project: bool = False) -> torch.Tensor:
        obs = self.aug(obs)
        if not self.embeddings:
            return module(obs)
        if project:
            return module.encode_and_project(obs)
        return module(obs)

    def _policy_logits_from_H(self, H: torch.Tensor, coeff: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute policy logits for a given kernel matrix H and PMD coefficient vector."""
        coeff = self.gradient_coeff if coeff is None else coeff
        if coeff is None:
            return torch.zeros(H.shape[0], self.n_actions, device=H.device, dtype=H.dtype)
        sink_bias = torch.ones(H.shape[0], self.E.shape[1], device=H.device, dtype=H.dtype) * coeff[-1]
        return H @ (coeff[:-1] * self.E) + sink_bias

    def _policy_from_H(self, H: torch.Tensor, coeff: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Closed-form PMD policy from logits."""
        logits = self._policy_logits_from_H(H, coeff=coeff)
        return torch.softmax(-logits, dim=1, dtype=logits.dtype)
    
    
    def compute_action_probs(self, obs: np.ndarray) -> np.ndarray:
        """Compute π(·|s) for given observation."""
        with torch.no_grad():
            # Handle different observation types
            if self.obs_type == 'pixels':
                # obs should already be an image [C, H, W]
                if obs.ndim == 2:
                    raise ValueError(
                        "For pixel observations, compute_action_probs expects an image [C, H, W], "
                        f"but got shape {obs.shape}. Use render_observation_from_state() first."
                    )
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)  # [1, C, H, W]
            else:
                # State observations: [x, y] -> [1, 2]
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)  # [1, obs_dim]
            
            enc_obs = self._encode_with_module(self.policy_encoder, obs_tensor, project=True)
    
            if self.gradient_coeff is None:
                return np.ones(self.n_actions) / self.n_actions
            
            # Add a zero to enc_obs to account for the extra row in H
            enc_obs_augmented = torch.cat([enc_obs, torch.zeros((1, 1), device=enc_obs.device)], dim=1)  # [1, feature_dim + 1]
            H = enc_obs_augmented @ self._phi_all_obs.T  # [1, num_unique]
            probs = self._policy_from_H(H)

            
            if torch.sum(probs) == 0.0 or torch.isnan(torch.sum(probs)):
                utils.ColorPrint.red(f"Warning: action_probs sum to zero or NaN. Returning uniform distribution. Check training stability and learning rates.{torch.sum(probs)}, {probs}")
                probs = torch.ones_like(probs) / self.n_actions
                # raise ValueError(f"action_probs sum to zero or NaN", torch.sum(probs), probs)
            logger.debug(f"Action probabilities: {probs.cpu().numpy().flatten()}")
            return probs.cpu().numpy().flatten()

    
    def act(self, obs, meta, step, eval_mode):
        if step < self.num_expl_steps or np.random.rand() < utils.schedule(self.epsilon_schedule, step):
            return np.random.randint(self.n_actions)

        # Compute action probabilities
        action_probs = self.compute_action_probs(obs)
        self.current_action_probs.append(action_probs)  # Store for visualization
        # print(f"Step {step}: Action probabilities: {action_probs}")
        # Sample action
        return np.random.choice(self.n_actions, p=action_probs)

    
    def _is_T_sufficiently_initialized(self, step: int) -> bool:
        """Check if transition learning phase is complete."""
        return step >= self.num_expl_steps + self.T_init_steps 
       
    def update_encoders(self, obs, action, next_obs, reward):
        metrics = dict()
        
        # Encode
        obs_en = self.aug_and_encode(obs, project=True)
        with torch.no_grad():
            next_obs_en = self.aug_and_encode(next_obs, project=True)

        encoded_state_action = self._encode_state_action(obs_en, action)
        
        # Predict next state
        projected_sa = self.project_sa(encoded_state_action)  
        
        # Normalize embeddings L2
        if self.mode == 'l1':
            norm_next_obs_en = F.normalize(next_obs_en, p=2, dim=1, eps=1e-10)
        elif self.mode == 'l2':
            norm_next_obs_en = next_obs_en
        norm_projected_sa = F.normalize(projected_sa, p=2, dim=1, eps=1e-10)

        # Compute loss
        # 1. Contrastive loss: 
        # Wz = torch.matmul(self.W, norm_next_obs_en.T)  # [feature_dim, B]
        logits = torch.matmul(norm_projected_sa, norm_next_obs_en.T)  # [B, B]
        logits = logits - torch.max(logits, 1)[0][:, None]  # For numerical stability
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        contrastive_loss = self.cross_entropy_loss(logits, labels)
        
        z_anchor = self.aug_and_encode(obs, project=True)
        with torch.no_grad():
            z_pos = self.aug_and_encode(obs, project=True)

        ### Compute CURL loss
        if self.curl:
            # Normalize embeddings L2
            if self.mode == 'l1':
                z_anchor = F.normalize(z_anchor, p=2, dim=1, eps=1e-10)
                z_pos = F.normalize(z_pos, p=2, dim=1, eps=1e-10)
            # Wz = torch.matmul(self.W, z_pos.T)  # [feature_dim, B]
            logits = torch.matmul(z_anchor, z_pos.T)  # [B, B]
            logits = logits - torch.max(logits, 1)[0][:, None]  # For numerical stability
            labels = torch.arange(logits.shape[0]).long().to(self.device)
            curl_loss = self.cross_entropy_loss(logits, labels)
        else:
            curl_loss = torch.tensor(0.0, device=self.device)

        if self.reward:
            reward_pred = self.reward(encoded_state_action)
            reward_loss = F.mse_loss(reward_pred, reward.to(self.device))
        else:
            reward_loss = torch.tensor(0.0, device=self.device)
        metrics['reward_loss'] = reward_loss.item()


        if self.embedding_sum_loss>0:
            # Sum of embeddings loss = 1
            sum_next_obs_en = torch.sum(next_obs_en, dim=1)  # [B]
            embedding_sum_loss = self.embedding_sum_loss * torch.mean((sum_next_obs_en - 1.0) ** 2)
        else:
            embedding_sum_loss = torch.tensor(0.0, device=self.device)

        loss =  contrastive_loss + curl_loss + embedding_sum_loss+reward_loss
        
        # max_grad_norm = 1.0
        # Optimize
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad()      
            # torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_grad_norm)
        # torch.nn.utils.clip_grad_norm_(self.project_sa.parameters(), max_grad_norm)
        self.transition_optimizer.zero_grad()
        loss.backward()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.step()
            self._policy_is_synced = False
        self.transition_optimizer.step()
        self.encoder_scheduler.step()

        # Print losses
        logger.debug(f"Transition Model Losses: Contrastive={contrastive_loss.item():.4f}, CURL={curl_loss.item():.4f}, Embedding Sum={embedding_sum_loss.item():.4f}, Reward={reward_loss.item():.4f}, Total={loss.item():.4f}")
        if self.use_tb or self.use_wandb:
            metrics['transition_loss'] = loss.item()
            metrics['contrastive_loss'] = contrastive_loss.item()
            metrics['curl_loss'] = curl_loss.item()
        return metrics
    

    def update_actor(self, obs=None, action=None, next_obs=None, step=None, rewards=None, encoded_full=None):
        """Update policy using Projected Mirror Descent."""
        metrics = dict()

        if encoded_full is None:
            self._sync_policy_encoder()
            self._cache_features(obs, action, next_obs, encoder=self.policy_encoder)
        else:
            self._cache_encoded_features(encoded_full)

        self.gradient_coeff = torch.zeros((self._phi_all_obs.shape[0]+1, 1), device=self.device)  # [z_x + 1, 1]
        prev_gradient_coeff = self.gradient_coeff.clone()
        self.H = self._phi_all_obs @ self._phi_all_next.T # [n, n]
        self.K = self.distribution_matcher.state_action_kernel(
            self._phi_all_obs,
            self._phi_all_obs,
            self._all_actions,
            self._all_actions,
        )  # [n, n]
        base_eta = float(utils.schedule(self.lr_actor, step))
        base_eta = float(np.clip(base_eta, self.pmd_eta_min, self.pmd_eta_max))
        self.current_eta = base_eta

        sink_norm = utils.schedule(self.sink_schedule, step)
        self.pi = self._policy_from_H(self.H.T, coeff=self.gradient_coeff)  # [z_x+1, n_actions]

        M = self.H*(self.E@self.pi.T) 

        nu_pi = self.distribution_matcher.compute_nu_pi(
                phi_all_next_obs = self._phi_all_next,
                psi_all_obs_action= self._psi_all,
                K= self.K,
                M = M,
                alpha=self._alpha,
                sink_norm=sink_norm 
        )
        actor_loss = torch.linalg.norm(nu_pi)**2
        print(f"Actor loss (squared norm of occupancy measure): {actor_loss}")
        best_loss = actor_loss
        best_pi = self.pi.clone()
        best_coeff = self.gradient_coeff.clone()

        self._adagrad_accum = 0.0

        for iteration in range(self.pmd_steps):
            grad_update = self.distribution_matcher.compute_gradient_coefficient(
                M, 
                phi_all_next_obs = self._phi_all_next, 
                psi_all_obs_action = self._psi_all, 
                alpha = self._alpha,
                sink_norm=sink_norm,
                K=self.K
            ) 

            # Track gradient norms by reward (only on final iteration)
            if iteration == self.pmd_steps - 1 and rewards is not None:
                self._track_gradient_norms(grad_update, rewards, step)

            if self.pmd_grad_clip_norm > 0:
                grad_norm = torch.linalg.norm(grad_update)
                if grad_norm > self.pmd_grad_clip_norm:
                    grad_update = grad_update * (self.pmd_grad_clip_norm / (grad_norm + 1e-12))

            if self.pmd_eta_mode == "adagrad":
                # Infinite norm for mirror descent
                grad_norm_sq = float(torch.max(grad_update * grad_update).item())
                self._adagrad_accum += grad_norm_sq
                eta_t = base_eta / np.sqrt(self._adagrad_accum + self.pmd_adagrad_eps)
                eta_t = float(np.clip(eta_t, self.pmd_eta_min, self.pmd_eta_max))
            elif self.pmd_eta_mode == "adadiff":
        
                # Infinite norm for mirror descent
                grad_norm_sq = float(torch.max(grad_update * grad_update - prev_gradient_coeff*prev_gradient_coeff).item())
                self._adagrad_accum += grad_norm_sq
                eta_t = base_eta / np.sqrt(self._adagrad_accum + self.pmd_adagrad_eps)
                eta_t = float(np.clip(eta_t, self.pmd_eta_min, self.pmd_eta_max))
            else:
                eta_t = base_eta

            candidate_coeff = self.gradient_coeff + eta_t * grad_update
            candidate_pi = self._policy_from_H(self.H.T, coeff=candidate_coeff)
            candidate_M = self.H * (self.E @ candidate_pi.T)
            candidate_nu = self.distribution_matcher.compute_nu_pi(
                phi_all_next_obs=self._phi_all_next,
                psi_all_obs_action=self._psi_all,
                K=self.K,
                M=candidate_M,
                alpha=self._alpha,
                sink_norm=sink_norm
            )
            candidate_loss = torch.linalg.norm(candidate_nu) ** 2

            if self.pmd_eta_mode == "backtracking":
                trial_eta = eta_t
                trial = 0
                while candidate_loss > actor_loss and trial < self.pmd_backtrack_max_trials:
                    trial_eta *= self.pmd_backtrack_factor
                    trial_eta = float(np.clip(trial_eta, self.pmd_eta_min, self.pmd_eta_max))
                    candidate_coeff = self.gradient_coeff + trial_eta * grad_update
                    candidate_pi = self._policy_from_H(self.H.T, coeff=candidate_coeff)
                    candidate_M = self.H * (self.E @ candidate_pi.T)
                    candidate_nu = self.distribution_matcher.compute_nu_pi(
                        phi_all_next_obs=self._phi_all_next,
                        psi_all_obs_action=self._psi_all,
                        K=self.K,
                        M=candidate_M,
                        alpha=self._alpha,
                        sink_norm=sink_norm
                    )
                    candidate_loss = torch.linalg.norm(candidate_nu) ** 2
                    trial += 1
                eta_t = trial_eta

            self.current_eta = eta_t
            self.gradient_coeff = candidate_coeff
            prev_gradient_coeff = grad_update.clone()
            self.pi = candidate_pi
            M = candidate_M
            actor_loss = candidate_loss

            if actor_loss < best_loss:
                best_loss = actor_loss
                best_pi = self.pi.clone()
                best_coeff = self.gradient_coeff.clone()

            if iteration % 10 == 0 or iteration == self.pmd_steps - 1:
                print(f"  PMD Iteration {iteration}, Actor loss: {actor_loss}, eta: {self.current_eta:.6g}")

        if self.pmd_best_iterate:
            self.pi = best_pi
            self.gradient_coeff = best_coeff
            actor_loss = best_loss
            

        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss
            metrics['actor_eta'] = float(self.current_eta)
            metrics['actor_best_loss'] = float(best_loss)
            metrics['sink_norm'] = float(sink_norm)
   
        return metrics

    def update_actor_nystrom(self,
                             full_obs,
                             full_action,
                             full_next_obs,
                             step,
                             rewards=None,
                             sub_obs=None,
                             sub_action=None,
                             sub_next_obs=None,
                             sub_rewards=None,
                             encoded_full=None,
                             encoded_sub=None):
        """Update policy using Projected Mirror Descent and Nystrom Approximation."""
        metrics = dict()
        if encoded_full is not None or encoded_sub is not None:
            if encoded_full is None or encoded_sub is None:
                raise ValueError("Nyström actor update requires both encoded_full and encoded_sub.")
            self._cache_encoded_features(encoded_full, encoded_sub=encoded_sub)
        elif sub_obs is None or sub_action is None or sub_next_obs is None:
            raise ValueError("Nyström actor update requires subsampled observations, actions, and next observations.")
        else:
            self._sync_policy_encoder()
            self._cache_features(
                full_obs,
                full_action,
                full_next_obs,
                encoder=self.policy_encoder,
                sub_obs=sub_obs,
                sub_action=sub_action,
                sub_next_obs=sub_next_obs,
            )

        utils.ColorPrint.blue(f"Starting Nyström PMD actor update with {self._phi_all_obs.shape[0]} total samples and {self._phi_sub_next.shape[0]} subsampled points.")
        self.gradient_coeff = torch.zeros((self._phi_all_obs.shape[0]+1, 1), device=self.device)  # [z_x + 1, 1]
        prev_gradient_coeff = self.gradient_coeff.clone()
        sub_H = self._phi_all_obs @ self._phi_sub_next.T # [n, m]
        base_eta = float(utils.schedule(self.lr_actor, step))
        base_eta = float(np.clip(base_eta, self.pmd_eta_min, self.pmd_eta_max))
        self.current_eta = base_eta

        sink_norm = utils.schedule(self.sink_schedule, step)
        self.pi = self._policy_from_H(sub_H.T, coeff=self.gradient_coeff)  # [z_x+1, n_actions]

        # M = self.H*(self.E@self.pi.T) # [n, ]

        nu_pi = self.distribution_matcher.compute_nu_pi_nystrom_memory_efficient(
                    phi_all_obs=self._phi_all_obs,
                    phi_sub_next_obs = self._phi_sub_next,
                    psi_sub_obs_action = self._psi_sub,
                    psi_all_obs_action = self._psi_all,
                    H = sub_H,
                    pi = self.pi,
                    E = self.E,
                    alpha=self._sub_alpha,
                    sink_norm=sink_norm,
                    phi_sub_obs=self._phi_sub_obs,
                    all_actions=self._all_actions,
                    sub_actions=self._sub_actions,
                )
        actor_loss = torch.linalg.norm(nu_pi)**2
        print(f"Actor loss (squared norm of occupancy measure): {actor_loss}")
        best_loss = actor_loss
        best_pi = self.pi.clone()
        best_coeff = self.gradient_coeff.clone()

        self._adagrad_accum = 0.0
        B_nystrom = self.distribution_matcher.compute_B_nystrom(
            psi_all_obs_action=self._psi_all,
            psi_sub_obs_action=self._psi_sub,
            svd_truncation=self.svd_truncation,
            phi_all_obs=self._phi_all_obs,
            phi_sub_obs=self._phi_sub_obs,
            all_actions=self._all_actions,
            sub_actions=self._sub_actions,
        )

        for iteration in range(self.pmd_steps):
            grad_update = self.distribution_matcher.compute_gradient_coefficient_nystrom_memory_efficient(
                phi_all_obs=self._phi_all_obs,
                phi_all_next_obs = self._phi_all_next,
                phi_sub_next_obs = self._phi_sub_next,
                psi_all_obs_action = self._psi_all,
                psi_sub_obs_action = self._psi_sub,
                H = sub_H,
                pi=self.pi,
                E=self.E,
                alpha = self._sub_alpha,
                sink_norm=sink_norm,
                B_nystrom=B_nystrom,
                phi_sub_obs=self._phi_sub_obs,
                all_actions=self._all_actions,
                sub_actions=self._sub_actions,

            )           

            # Track gradient norms by reward (only on final iteration)
            # if iteration == self.pmd_steps - 1 and sub_rewards is not None:
            #     self._track_gradient_norms(grad_update, sub_rewards, step)

            if self.pmd_grad_clip_norm > 0:
                grad_norm = torch.linalg.norm(grad_update)
                if grad_norm > self.pmd_grad_clip_norm:
                    grad_update = grad_update * (self.pmd_grad_clip_norm / (grad_norm + 1e-12))

            if self.pmd_eta_mode == "adagrad":
                # Infinite norm for mirror descent
                grad_norm_sq = float(torch.max(grad_update * grad_update).item())
                self._adagrad_accum += grad_norm_sq
                eta_t = base_eta / np.sqrt(self._adagrad_accum + self.pmd_adagrad_eps)
                eta_t = float(np.clip(eta_t, self.pmd_eta_min, self.pmd_eta_max))
            elif self.pmd_eta_mode == "adadiff":
        
                # Infinite norm for mirror descent
                grad_norm_sq = float(torch.max(grad_update * grad_update - prev_gradient_coeff*prev_gradient_coeff).item())
                self._adagrad_accum += grad_norm_sq
                eta_t = base_eta / np.sqrt(self._adagrad_accum + self.pmd_adagrad_eps)
                eta_t = float(np.clip(eta_t, self.pmd_eta_min, self.pmd_eta_max))
            else:
                eta_t = base_eta

            candidate_coeff = self.gradient_coeff + eta_t * grad_update
            candidate_pi = self._policy_from_H(sub_H.T, coeff=candidate_coeff)
            # candidate_M = sub_H * (self.E @ candidate_pi.T)
            candidate_nu = self.distribution_matcher.compute_nu_pi_nystrom_memory_efficient(
                    phi_all_obs=self._phi_all_obs,
                    phi_sub_next_obs = self._phi_sub_next,
                    psi_sub_obs_action = self._psi_sub,
                    psi_all_obs_action = self._psi_all,
                    H = sub_H,
                    pi = candidate_pi,
                    E = self.E,
                    alpha=self._sub_alpha,
                    sink_norm=sink_norm,
                    B_nystrom=B_nystrom,
                    phi_sub_obs=self._phi_sub_obs,
                    all_actions=self._all_actions,
                    sub_actions=self._sub_actions,
                )
            
            candidate_loss = torch.linalg.norm(candidate_nu) ** 2

            if self.pmd_eta_mode == "backtracking":
                trial_eta = eta_t
                trial = 0
                while candidate_loss > actor_loss and trial < self.pmd_backtrack_max_trials:
                    trial_eta *= self.pmd_backtrack_factor
                    trial_eta = float(np.clip(trial_eta, self.pmd_eta_min, self.pmd_eta_max))
                    candidate_coeff = self.gradient_coeff + trial_eta * grad_update
                    candidate_pi = self._policy_from_H(sub_H.T, coeff=candidate_coeff)
                    # candidate_M = sub_H * (self.E @ candidate_pi.T)
                    candidate_nu = self.distribution_matcher.compute_nu_pi_nystrom_memory_efficient(
                            phi_all_obs=self._phi_all_obs,
                            phi_sub_next_obs = self._phi_sub_next,
                            psi_sub_obs_action = self._psi_sub,
                            psi_all_obs_action = self._psi_all,
                            H = sub_H,
                            pi = candidate_pi,
                            E = self.E,
                            alpha=self._sub_alpha,
                            sink_norm=sink_norm,
                            B_nystrom=B_nystrom,
                            phi_sub_obs=self._phi_sub_obs,
                            all_actions=self._all_actions,
                            sub_actions=self._sub_actions,
                        )                    
                    candidate_loss = torch.linalg.norm(candidate_nu) ** 2
                    trial += 1
                eta_t = trial_eta

            self.current_eta = eta_t
            self.gradient_coeff = candidate_coeff
            prev_gradient_coeff = grad_update.clone()
            self.pi = candidate_pi
            actor_loss = candidate_loss

            if actor_loss < best_loss:
                best_loss = actor_loss
                best_pi = self.pi.clone()
                best_coeff = self.gradient_coeff.clone()

            if iteration % 10 == 0 or iteration == self.pmd_steps - 1:
                print(f"  PMD Iteration {iteration}, Actor loss: {actor_loss}, eta: {self.current_eta:.6g}")

        if self.pmd_best_iterate:
            self.pi = best_pi
            self.gradient_coeff = best_coeff
            actor_loss = best_loss
            

        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss
            metrics['actor_eta'] = float(self.current_eta)
            metrics['actor_best_loss'] = float(best_loss)
            metrics['sink_norm'] = float(sink_norm)
   
        return metrics


    def _track_gradient_norms(self, gradient, rewards, step):
        """
        Track gradient norms for samples with different reward values.
        
        Args:
            gradient: [batch_size+1, 1] gradient tensor
            rewards: [batch_size] reward tensor
            step: current training step
        """
        with torch.no_grad():
            # Compute per-sample gradient norm (excluding the last augmented dimension)
            grad_per_sample = gradient[:-1]  # [batch_size, 1]
            
            # Group by reward value
            for reward_val, reward_key in [(1.0, '+1'), (-1.0, '-1'), (0.0, '0')]:
                mask = (rewards == reward_val)
                if mask.sum() > 0:
                    # Get gradients for this reward type
                    grads_for_reward = grad_per_sample[mask]
                    
                    print(f"shapes for reward {reward_key}: grads {grads_for_reward.shape}, rewards {rewards[mask].shape}")
                    # Compute norms
                    norms = torch.norm(grads_for_reward.reshape(grads_for_reward.shape[0], -1), dim=1).cpu().numpy()
                    
                    # Store individual samples (up to max)
                    for norm_val in norms:
                        if len(self.gradient_samples[reward_key]) < self.max_samples_per_reward:
                            self.gradient_samples[reward_key].append((step, float(norm_val)))
                        else:
                            # Replace oldest sample
                            self.gradient_samples[reward_key].pop(0)
                            self.gradient_samples[reward_key].append((step, float(norm_val)))
            
            # Compute statistics for this step
            for reward_key in ['+1', '-1', '0']:
                # Get norms from current samples at this step
                current_norms = [norm for s, norm in self.gradient_samples[reward_key] if s == step]
                if len(current_norms) > 0:
                    mean_norm = np.mean(current_norms)
                    std_norm = np.std(current_norms) if len(current_norms) > 1 else 0.0
                    self.gradient_norm_history[reward_key].append((step, mean_norm, std_norm))
                    
                    print(f"  Reward {reward_key}: {len(current_norms)} samples, "
                          f"mean_norm={mean_norm:.6f}, std={std_norm:.6f}")

    
    def _cache_features(self, obs, action, next_obs, encoder=None, sub_obs=None, sub_action=None, sub_next_obs=None):
        """Pre-compute and cache dataset features."""
        encoder = self.encoder if encoder is None else encoder
       
        with torch.no_grad():
    
            self._phi_all_obs = self._encode_with_module(encoder, obs, project=True)
            self._phi_all_next = self._encode_with_module(encoder, next_obs, project=True)

            action = action #.cpu()
            self._psi_all = self._encode_state_action(self._phi_all_obs, action) #.cpu()
            self._all_actions = action.long().reshape(-1).detach().cpu()
           
            self._alpha = torch.zeros((self._phi_all_next.shape[0], 1), device=self.device)  # [n, 1]
    
            self._alpha[0] = 1.0  # set alpha to 1.0 for the first state
            self.E = F.one_hot(
                action, 
                self.n_actions,
            ).reshape(-1, self.n_actions).to(dtype=self.compute_dtype, device=self.device)

            # ** AUGMENTATION STEP **
            # ψ and Φ are augmented with an additional zero dimension
            zeros_col = torch.zeros(*self._psi_all.shape[:-1], 1, device=self._psi_all.device)
            self._psi_all = torch.cat([self._psi_all, zeros_col], dim=-1)

            zero_col = torch.zeros(*self._phi_all_next.shape[:-1], 1, device=self._phi_all_next.device)
            self._phi_all_next = torch.cat([self._phi_all_next, zero_col], dim=-1)

            zero_col = torch.zeros(*self._phi_all_obs.shape[:-1], 1, device=self._phi_all_obs.device)
            self._phi_all_obs = torch.cat([self._phi_all_obs, zero_col], dim=-1)

            if sub_obs is not None and sub_next_obs is not None and sub_action is not None:
                self._phi_sub_obs = self._encode_with_module(encoder, sub_obs, project=True)
                self._phi_sub_next = self._encode_with_module(encoder, sub_next_obs, project=True)
                self._sub_actions = sub_action.long().reshape(-1).detach().cpu()

                self._psi_sub = self._encode_state_action(self._phi_sub_obs, sub_action)

                zeros_col_sub_next = torch.zeros(*self._phi_sub_next.shape[:-1], 1, device=self._phi_sub_next.device)
                self._phi_sub_next = torch.cat([self._phi_sub_next, zeros_col_sub_next], dim=-1)

                zero_col_sub_obs = torch.zeros(*self._phi_sub_obs.shape[:-1], 1, device=self._phi_sub_obs.device)
                self._phi_sub_obs = torch.cat([self._phi_sub_obs, zero_col_sub_obs], dim=-1)

                zero_col_sub_psi = torch.zeros(*self._psi_sub.shape[:-1], 1, device=self._psi_sub.device)
                self._psi_sub = torch.cat([self._psi_sub, zero_col_sub_psi], dim=-1)

                self._sub_alpha = torch.zeros((self._phi_sub_next.shape[0], 1), device=self.device)  # [m, 1]
                self._sub_alpha[0] = 1.0  # set alpha to 1.0 for the first state

            print(f"dimensions after augmentation: psi_all {self._psi_all.shape}, phi_all_next {self._phi_all_next.shape}, phi_all_obs {self._phi_all_obs.shape}")

    def _append_zero_feature_column(self, tensor):
        zeros_col = torch.zeros(*tensor.shape[:-1], 1, device=tensor.device, dtype=tensor.dtype)
        return torch.cat([tensor, zeros_col], dim=-1)

    def _cache_encoded_features(self, encoded_full, encoded_sub=None):
        with torch.no_grad():
            self._phi_all_obs = self._append_zero_feature_column(encoded_full["phi_obs"])
            self._phi_all_next = self._append_zero_feature_column(encoded_full["phi_next"])
            self._psi_all = self._append_zero_feature_column(encoded_full["psi"])

            self._alpha = torch.zeros((self._phi_all_next.shape[0], 1), device=self.device, dtype=self._phi_all_next.dtype)
            self._alpha[0] = 1.0

            self.E = encoded_full["E"].to(dtype=self.compute_dtype, device=self.device)
            self._all_actions = torch.argmax(encoded_full["E"], dim=1).long().detach().cpu()

            if encoded_sub is not None:
                self._phi_sub_obs = self._append_zero_feature_column(encoded_sub["phi_obs"])
                self._phi_sub_next = self._append_zero_feature_column(encoded_sub["phi_next"])
                self._psi_sub = self._append_zero_feature_column(encoded_sub["psi"])
                self._sub_actions = torch.argmax(encoded_sub["E"], dim=1).long().detach().cpu()

                self._sub_alpha = torch.zeros((self._phi_sub_next.shape[0], 1), device=self.device, dtype=self._phi_sub_next.dtype)
                self._sub_alpha[0] = 1.0

            print(f"dimensions after augmentation: psi_all {self._psi_all.shape}, phi_all_next {self._phi_all_next.shape}, phi_all_obs {self._phi_all_obs.shape}")

    def _make_actor_batch(self, obs, action, next_obs, reward):
        return (
            obs,
            action,
            next_obs,
            reward.reshape(obs.shape[0], -1),
        )

    def _slice_actor_batch(self, actor_batch, index):
        return tuple(field[index] for field in actor_batch)

    def _concat_actor_batches(self, actor_batches, max_samples):
        if not actor_batches:
            raise RuntimeError("No replay samples available for actor update")
        actor_batch = tuple(
            torch.cat([batch[field_idx] for batch in actor_batches], dim=0)
            for field_idx in range(len(actor_batches[0]))
        )
        return self._slice_actor_batch(actor_batch, slice(0, max_samples))

    def _load_first_actor_transition(self, replay_buffer=None, fallback_actor_batch=None):
        if replay_buffer is not None and hasattr(replay_buffer, "get_first_transition"):
            first_batch = replay_buffer.get_first_transition()
            first_obs, first_action, first_reward, _, first_next_obs = utils.to_torch(
                first_batch[:5],
                self.device,
            )
            return self._make_actor_batch(first_obs, first_action, first_next_obs, first_reward)

        if fallback_actor_batch is None:
            return None
        return self._slice_actor_batch(fallback_actor_batch, slice(0, 1))

    def _replace_first_actor_transition(self, actor_batch, first_actor_transition):
        if first_actor_transition is None:
            return actor_batch

        actor_batch = tuple(field.clone() for field in actor_batch)
        for field_idx, first_field in enumerate(first_actor_transition):
            actor_batch[field_idx][:1] = first_field.to(actor_batch[field_idx].device)
        return actor_batch

    def _is_cuda_oom(self, error):
        return (
            isinstance(error, RuntimeError)
            and "out of memory" in str(error).lower()
            and "cuda" in str(error).lower()
        )

    def _encoded_batch_size(self, encoded):
        return next(iter(encoded.values())).shape[0]

    def _concat_encoded_batches(self, encoded_batches):
        return {
            key: torch.cat([batch[key] for batch in encoded_batches], dim=0)
            for key in encoded_batches[0].keys()
        }

    def _slice_raw_transition_batch(self, transitions, index):
        return tuple(field[index] for field in transitions)

    def _encode_actor_transition_batch(self, transitions):
        obs, action, reward, _, next_obs = utils.to_torch(transitions[:5], self.device)
        reward = reward.reshape(obs.shape[0], -1)
        with torch.no_grad():
            phi_obs = self._encode_with_module(self.policy_encoder, obs, project=True)
            phi_next = self._encode_with_module(self.policy_encoder, next_obs, project=True)
            psi = self._encode_state_action(phi_obs, action)
            action_onehot = F.one_hot(
                action.long(),
                self.n_actions,
            ).reshape(-1, self.n_actions).to(dtype=self.compute_dtype, device=self.device)
        return {
            "phi_obs": phi_obs,
            "phi_next": phi_next,
            "psi": psi,
            "E": action_onehot,
            "reward": reward,
        }

    def _encode_actor_transition_batch_with_retries(self, transitions, splits_left=None):
        splits_left = self.encoded_fifo_cuda_oom_splits if splits_left is None else splits_left
        batch_size = transitions[0].shape[0]
        try:
            return self._encode_actor_transition_batch(transitions)
        except RuntimeError as error:
            if not self._is_cuda_oom(error) or splits_left <= 0 or batch_size <= 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            midpoint = batch_size // 2
            left = self._slice_raw_transition_batch(transitions, slice(0, midpoint))
            right = self._slice_raw_transition_batch(transitions, slice(midpoint, None))
            encoded_left = self._encode_actor_transition_batch_with_retries(left, splits_left - 1)
            encoded_right = self._encode_actor_transition_batch_with_retries(right, splits_left - 1)
            return self._concat_encoded_batches([encoded_left, encoded_right])

    def _insert_first_transition_if_available(self, replay_buffer):
        if self._encoded_actor_fifo.has_first:
            return
        if replay_buffer is None or not hasattr(replay_buffer, "get_first_transition"):
            return
        try:
            first_transition = replay_buffer.get_first_transition()
        except RuntimeError:
            return
        encoded = self._encode_actor_transition_batch_with_retries(first_transition)
        self._encoded_actor_fifo.add(np.array([0], dtype=np.int64), encoded)

    def _update_encoded_actor_fifo(self, replay_buffer):
        if replay_buffer is None or not hasattr(replay_buffer, "get_new_transitions_since"):
            return False

        self._sync_policy_encoder()
        inserted = 0
        encode_batch_size = max(1, self.encoded_fifo_encode_batch_size)

        while True:
            transition_ids, transitions = replay_buffer.get_new_transitions_since(
                self._encoded_fifo_replay_marker,
                limit=encode_batch_size,
            )
            if transition_ids is None:
                break

            # Encode only the new replay-buffer transitions, then immediately
            # acknowledge them so raw pending data can be released by storage.
            encoded = self._encode_actor_transition_batch_with_retries(transitions)
            self._encoded_actor_fifo.add(transition_ids, encoded)
            self._encoded_fifo_replay_marker = int(transition_ids[-1])
            if hasattr(replay_buffer, "mark_transitions_encoded"):
                replay_buffer.mark_transitions_encoded(self._encoded_fifo_replay_marker)
            inserted += int(len(transition_ids))

        self._insert_first_transition_if_available(replay_buffer)
        return inserted > 0 or len(self._encoded_actor_fifo) > 0

    def _sample_encoded_actor_data(self, size, include_first):
        encoded = self._encoded_actor_fifo.sample(
            int(size),
            self.device,
            include_first=include_first,
        )
        return encoded, encoded.get("reward")

    def _all_encoded_actor_data(self, include_first=True):
        encoded = self._encoded_actor_fifo.all(
            self.device,
            include_first=include_first,
        )
        return encoded, encoded.get("reward")

    def _load_actor_batch_from_replay_iter(
            self,
            replay_iter,
            obs,
            action,
            next_obs,
            reward,
            max_samples,
            replay_buffer=None):
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")

        actor_batches = [self._make_actor_batch(obs, action, next_obs, reward)]
        collected = obs.shape[0]

        while collected < max_samples:
            batch = next(replay_iter)
            obs_b, action_b, reward_b, _, next_obs_b = utils.to_torch(batch, self.device)
            actor_batch = self._make_actor_batch(obs_b, action_b, next_obs_b, reward_b)
            actor_batches.append(actor_batch)
            collected += obs_b.shape[0]

        actor_batch = self._concat_actor_batches(actor_batches, max_samples)
        first_actor_transition = self._load_first_actor_transition(
            replay_buffer=replay_buffer,
            fallback_actor_batch=actor_batch,
        )
        return self._replace_first_actor_transition(actor_batch, first_actor_transition)

    def _load_actor_subsample_from_replay_iter(
            self,
            replay_iter,
            max_samples,
            replay_buffer=None,
            fallback_actor_batch=None):
        if max_samples <= 0:
            raise ValueError("subsamples must be positive when provided")

        first_actor_transition = self._load_first_actor_transition(
            replay_buffer=replay_buffer,
            fallback_actor_batch=fallback_actor_batch,
        )
        if first_actor_transition is None:
            raise RuntimeError("Could not build a subsample with the first transition in position 0")
        if max_samples == 1:
            return first_actor_transition

        actor_batches = []
        collected = 0
        remaining_samples = max_samples - 1

        while collected < remaining_samples:
            batch = next(replay_iter)
            obs_b, action_b, reward_b, _, next_obs_b = utils.to_torch(batch, self.device)
            actor_batch = self._make_actor_batch(obs_b, action_b, next_obs_b, reward_b)

            if actor_batch[0].shape[0] > 1:
                actor_batch = self._slice_actor_batch(actor_batch, slice(1, None))
            if actor_batch[0].shape[0] == 0:
                continue
            actor_batches.append(actor_batch)
            collected += actor_batch[0].shape[0]

        sampled_actor_batch = self._concat_actor_batches(actor_batches, remaining_samples)
        return self._concat_actor_batches(
            [first_actor_transition, sampled_actor_batch],
            max_samples,
        )

    def _get_actor_update_data(self, replay_iter, obs, action, next_obs, reward, replay_buffer=None):
        if self._update_encoded_actor_fifo(replay_buffer):
            if self.subsamples is None:
                encoded_full, rewards = self._sample_encoded_actor_data(
                    self.batch_size_actor,
                    include_first=True,
                )
                return encoded_full, rewards, None, None

            if self.subsamples <= 0:
                raise ValueError("subsamples must be positive when provided")

            # Nyström uses the FIFO as the full support and samples only the
            # landmark/subsample set. The FIFO capacity controls the maximum
            # retained support size; batch_size_actor is only for non-Nyström.
            encoded_full, rewards = self._all_encoded_actor_data(include_first=True)
            if self.nystrom_synthetic_subsamples:
                encoded_sub, sub_rewards = self._encode_synthetic_nystrom_subsamples()
            else:
                encoded_sub, sub_rewards = self._sample_encoded_actor_data(
                    int(self.subsamples),
                    include_first=True,
                )
            return encoded_full, rewards, encoded_sub, sub_rewards

        full_actor_batch = self._load_actor_batch_from_replay_iter(
            replay_iter,
            obs,
            action,
            next_obs,
            reward,
            max_samples=self.batch_size_actor,
            replay_buffer=replay_buffer,
        )

        if self.subsamples is None:
            return (*full_actor_batch, None, None, None, None)

        if self.subsamples <= 0:
            raise ValueError("subsamples must be positive when provided")

        if self.nystrom_synthetic_subsamples:
            synthetic_batch = self._build_synthetic_nystrom_subsample_batch()
            syn_obs, syn_action, syn_reward, _, syn_next_obs = synthetic_batch
            subsampled_actor_batch = self._make_actor_batch(
                syn_obs,
                syn_action,
                syn_next_obs,
                syn_reward,
            )
        elif self.subsamples >= full_actor_batch[0].shape[0]:
            subsampled_actor_batch = full_actor_batch
        else:
            subsampled_actor_batch = self._load_actor_subsample_from_replay_iter(
                replay_iter,
                max_samples=int(self.subsamples),
                replay_buffer=replay_buffer,
                fallback_actor_batch=full_actor_batch,
            )

        return (*full_actor_batch, *subsampled_actor_batch)

    def _build_debug_visualizer_batch(self, obs, max_observations=2000):
        if obs is None:
            return None, None

        max_observations = min(max_observations, obs.shape[0])
        visualizer_obs = obs[:max_observations]
        with torch.no_grad():
            visualizer_z = self._encode_with_module(
                self.policy_encoder,
                visualizer_obs,
                project=True,
            )
        return visualizer_obs, visualizer_z

    def _compute_mean_action_probs_deviation(self, action_probs: np.ndarray) -> float:
        """
        Compute mean deviation of action probabilities from uniform distribution.
        
        Args:
            action_probs: [batch_size, n_actions] action probabilities from current policy
            
        Returns:
            Mean absolute deviation from uniform (1/n_actions)
        """
        uniform_prob = 1.0 / self.n_actions
        # Average over batch, then compute mean absolute deviation across actions
        mean_probs = np.mean(action_probs, axis=0)  # [n_actions]
        deviation = np.mean(np.abs(mean_probs - uniform_prob))
        return deviation
    
    def plot_gradient_norm_by_reward(self, save_dir: str = './gradient_plots'):
        """
        Plot gradient norms over time, separated by reward value.
        
        Args:
            save_dir: Directory to save the plot
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Check if we have data
        has_data = any(len(self.gradient_norm_history[key]) > 0 for key in self.gradient_norm_history)
        if not has_data:
            print("No gradient norm history to plot yet")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Gradient Norms by Reward Type', fontsize=16, fontweight='bold')
        
        colors = {'+1': 'green', '-1': 'red', '0': 'blue'}
        labels = {'+1': 'Reward +1', '-1': 'Reward -1', '0': 'Reward 0'}
        
        # Plot 1: Mean gradient norms over time
        ax1 = axes[0, 0]
        for reward_key in ['+1', '-1', '0']:
            if len(self.gradient_norm_history[reward_key]) > 0:
                steps, means, stds = zip(*self.gradient_norm_history[reward_key])
                ax1.plot(steps, means, color=colors[reward_key], linewidth=2, 
                        label=labels[reward_key], alpha=0.8)
                means_arr = np.array(means)
                stds_arr = np.array(stds)
                ax1.fill_between(steps, means_arr - stds_arr, means_arr + stds_arr, 
                                color=colors[reward_key], alpha=0.2)
        
        ax1.set_xlabel('Training Steps', fontsize=11)
        ax1.set_ylabel('Mean Gradient Norm', fontsize=11)
        ax1.set_title('Mean Gradient Norms Over Time', fontsize=12, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Current distribution of gradient norms (latest samples)
        ax2 = axes[0, 1]
        current_samples = {k: [norm for _, norm in v[-50:]] for k, v in self.gradient_samples.items() if len(v) > 0}
        
        if any(len(samples) > 0 for samples in current_samples.values()):
            positions = []
            data_to_plot = []
            tick_labels = []
            box_colors = []
            
            for i, (reward_key, samples) in enumerate(current_samples.items()):
                if len(samples) > 0:
                    positions.append(i)
                    data_to_plot.append(samples)
                    tick_labels.append(labels[reward_key])
                    box_colors.append(colors[reward_key])
            
            bp = ax2.boxplot(data_to_plot, positions=positions, patch_artist=True,
                           widths=0.6, showfliers=True)
            
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            ax2.set_xticks(positions)
            ax2.set_xticklabels(tick_labels)
            ax2.set_ylabel('Gradient Norm', fontsize=11)
            ax2.set_title('Current Gradient Norm Distribution\n(Last 50 samples per reward)', 
                         fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
        else:
            ax2.text(0.5, 0.5, 'Not enough data yet', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
        
        # Plot 3: Sample counts over time
        ax3 = axes[1, 0]
        for reward_key in ['+1', '-1', '0']:
            if len(self.gradient_norm_history[reward_key]) > 0:
                steps, _, _ = zip(*self.gradient_norm_history[reward_key])
                # Count cumulative samples at each step
                cumulative_counts = []
                for step in steps:
                    count = len([s for s, _ in self.gradient_samples[reward_key] if s <= step])
                    cumulative_counts.append(count)
                
                ax3.plot(steps, cumulative_counts, color=colors[reward_key], 
                        linewidth=2, label=labels[reward_key], alpha=0.8)
        
        ax3.set_xlabel('Training Steps', fontsize=11)
        ax3.set_ylabel('Cumulative Sample Count', fontsize=11)
        ax3.set_title('Sample Collection Progress', fontsize=12, fontweight='bold')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(self.max_samples_per_reward, color='black', linestyle='--', 
                   linewidth=1, alpha=0.5, label=f'Max ({self.max_samples_per_reward})')
        
        # Plot 4: Ratio comparison
        ax4 = axes[1, 1]
        if len(self.gradient_norm_history['+1']) > 0 and len(self.gradient_norm_history['0']) > 0:
            # Get aligned steps
            steps_pos = [s for s, _, _ in self.gradient_norm_history['+1']]
            steps_zero = [s for s, _, _ in self.gradient_norm_history['0']]
            steps_neg = [s for s, _, _ in self.gradient_norm_history['-1']]
            
            common_steps = sorted(set(steps_pos) & set(steps_zero))
            
            if len(common_steps) > 0:
                ratios_pos_zero = []
                for step in common_steps:
                    mean_pos = [m for s, m, _ in self.gradient_norm_history['+1'] if s == step][0]
                    mean_zero = [m for s, m, _ in self.gradient_norm_history['0'] if s == step][0]
                    if mean_zero > 1e-10:
                        ratios_pos_zero.append(mean_pos / mean_zero)
                    else:
                        ratios_pos_zero.append(np.nan)
                
                ax4.plot(common_steps, ratios_pos_zero, color='purple', linewidth=2, 
                        label='||∇(r=+1)|| / ||∇(r=0)||', alpha=0.8)
                ax4.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            
            # Add negative reward ratio if available
            common_steps_neg = sorted(set(steps_neg) & set(steps_zero))
            if len(common_steps_neg) > 0:
                ratios_neg_zero = []
                for step in common_steps_neg:
                    mean_neg = [m for s, m, _ in self.gradient_norm_history['-1'] if s == step][0]
                    mean_zero = [m for s, m, _ in self.gradient_norm_history['0'] if s == step][0]
                    if mean_zero > 1e-10:
                        ratios_neg_zero.append(mean_neg / mean_zero)
                    else:
                        ratios_neg_zero.append(np.nan)
                
                ax4.plot(common_steps_neg, ratios_neg_zero, color='orange', linewidth=2,
                        label='||∇(r=-1)|| / ||∇(r=0)||', alpha=0.8)
            
            ax4.set_xlabel('Training Steps', fontsize=11)
            ax4.set_ylabel('Gradient Norm Ratio', fontsize=11)
            ax4.set_title('Relative Gradient Magnitudes', fontsize=12, fontweight='bold')
            ax4.legend(loc='best')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Not enough data for comparison', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=12)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'gradient_norms_by_reward.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Gradient norm plot saved to: {save_path}")
        
        # Print summary statistics
        print("\n=== Gradient Norm Summary ===")
        for reward_key in ['+1', '-1', '0']:
            if len(self.gradient_samples[reward_key]) > 0:
                norms = [norm for _, norm in self.gradient_samples[reward_key]]
                print(f"Reward {reward_key:>2}: n={len(norms):3d}, "
                      f"mean={np.mean(norms):.6f}, std={np.std(norms):.6f}, "
                      f"min={np.min(norms):.6f}, max={np.max(norms):.6f}")


    def plot_policy_deviation_history(self, save_dir: str = './policy_plots'):
        """
        Plot cumulative history of policy deviation from uniform distribution.
        
        Args:
            save_dir: Directory to save the plot
        """
        if len(self.policy_deviation_history) == 0:
            print("No policy deviation history to plot yet")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        steps, deviations = zip(*self.policy_deviation_history)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(steps, deviations, color='blue', linewidth=2, alpha=0.8, label='Policy Deviation')
        ax.axhline(0, color='green', linestyle='--', linewidth=1.5, label='Uniform Policy', alpha=0.7)
        
        # Theoretical maximum (when policy is deterministic on one action)
        max_deviation = (self.n_actions - 1) / self.n_actions
        ax.axhline(max_deviation, color='red', linestyle='--', linewidth=1.5, 
                   label=f'Deterministic Policy ({max_deviation:.3f})', alpha=0.7)
        
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Mean |P(a) - 1/|A||', fontsize=12)
        ax.set_title('Policy Concentration Over Time\n(Deviation from Uniform Distribution)', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add current value annotation
        if len(deviations) > 0:
            current_val = deviations[-1]
            ax.text(0.02, 0.98, f'Current: {current_val:.4f}', 
                    transform=ax.transAxes, fontsize=11,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'policy_deviation_history.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Policy deviation plot saved to: {save_path}")

    def aug_and_encode(self, obs, project=False):
        obs = self.aug(obs)
        if not self.embeddings:
            return self.encoder(obs)
        if project:
            return self.encoder.encode_and_project(obs)
        else:
            return self.encoder(obs)

    def _sample_batch(self, replay_iter):
        batch = next(replay_iter)
        return utils.to_torch(batch, self.device)

    def update(self, replay_iter, step, replay_buffer=None):
        metrics = dict()

        if step % self.update_every_steps != 0 and self._is_T_sufficiently_initialized(step) is True:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        if self.use_tb or self.use_wandb:
            metrics['batch_reward'] = reward.mean().item()
        if self.embeddings:
            metrics.update(self.update_encoders(obs, action, next_obs, reward))

        # If T is not sufficiently initialized, skip actor update
        if self._is_T_sufficiently_initialized(step) is False:   
            metrics['actor_loss'] = 100.0  # dummy value
            return metrics
        
        # In ideal mode, we can update actor immediately
        if  step % self.update_actor_every_steps == 0 or step == self.num_expl_steps + self.T_init_steps: # or self.ideal:  
            actor_update_data = self._get_actor_update_data(
                replay_iter,
                obs,
                action,
                next_obs,
                reward,
                replay_buffer=replay_buffer,
            )
            if isinstance(actor_update_data[0], dict):
                encoded_full, all_reward_actor, encoded_sub, subsampled_reward_actor = actor_update_data
                utils.ColorPrint.red(
                    f"samples for actor update: full {self._encoded_batch_size(encoded_full)}, "
                    f"subsampled {self._encoded_batch_size(encoded_sub) if encoded_sub is not None else 'N/A'}"
                )
            else:
                (
                    all_obs_actor,
                    all_action_actor,
                    all_next_obs_actor,
                    all_reward_actor,
                    subsampled_obs_actor,
                    subsampled_action_actor,
                    subsampled_next_obs_actor,
                    subsampled_reward_actor,
                ) = actor_update_data
                encoded_full = None
                encoded_sub = None
                utils.ColorPrint.red(
                    f"samples for actor update: full {all_obs_actor.shape[0]}, "
                    f"subsampled {subsampled_obs_actor.shape[0] if subsampled_obs_actor is not None else 'N/A'}"
                )

            if self.subsamples is None and encoded_full is not None:
                metrics.update(
                    self.update_actor(
                        step=step,
                        rewards=all_reward_actor,
                        encoded_full=encoded_full,
                    )
                )
            elif self.subsamples is None:
                metrics.update(
                    self.update_actor(
                        all_obs_actor,
                        all_action_actor,
                        all_next_obs_actor,
                        step=step,
                        rewards=all_reward_actor,
                    )
                )
            elif encoded_full is not None:
                metrics.update(
                    self.update_actor_nystrom(
                        None,
                        None,
                        None,
                        step=step,
                        rewards=all_reward_actor,
                        sub_rewards=subsampled_reward_actor,
                        encoded_full=encoded_full,
                        encoded_sub=encoded_sub,
                    )
                )
            else:
                metrics.update(
                    self.update_actor_nystrom(
                        all_obs_actor,
                        all_action_actor,
                        all_next_obs_actor,
                        step=step,
                        rewards=all_reward_actor,
                        sub_obs=subsampled_obs_actor,
                        sub_action=subsampled_action_actor,
                        sub_next_obs=subsampled_next_obs_actor,
                        sub_rewards=subsampled_reward_actor,
                    )
                )


            if self.debug_visualizer is not None:
                param_text = (
                    f"Step: {step}\n"
                    f"γ = {self.discount}\n"
                    f"η = {self.current_eta}\n"
                    f"λ = {self.lambda_reg}\n"
                    f"sink norm = {utils.schedule(self.sink_schedule, step):.6f}\n"
                    f"PMD steps = {self.pmd_steps}\n"
                    f"subsamples = {self.subsamples if self.subsamples is not None else 'all'}\n"
                )

            # if  step % 10000 == 0:
                visualizer_obs, visualizer_z = self._build_debug_visualizer_batch(obs)
                metrics.update(
                    self.debug_visualizer.save(
                        step=step,
                        obs_batch=visualizer_obs,
                        z_batch=visualizer_z,
                        param_text=param_text,
                    )
                )
            
                with torch.no_grad():
                
                    if len(self.current_action_probs) == 0:
                        return metrics
                    current_action_probs = np.array(self.current_action_probs)  # [num_recorded, n_actions]
                    # Compute mean deviation from uniform
                    mean_deviation = self._compute_mean_action_probs_deviation(current_action_probs)
                    
                    # Store in history
                    self.policy_deviation_history.append((step, mean_deviation))
                    
                    # Also store the mean action probabilities
                    mean_probs = np.mean(current_action_probs, axis=0)
                    self.action_probs_history.append((step, mean_probs))
                    
                    # Log to metrics
                    metrics['policy_deviation_from_uniform'] = mean_deviation
                    print(f"Policy deviation from uniform: {mean_deviation:.4f} (0=uniform, {(self.n_actions-1)/self.n_actions:.3f}=deterministic)")
                    self.current_action_probs = []  # Clear after processing
                self.plot_policy_deviation_history(save_dir=os.path.join(os.getcwd(), 'policy_plots'))
                self.plot_gradient_norm_by_reward(save_dir=os.path.join(os.getcwd(), 'gradient_plots'))

        return metrics
