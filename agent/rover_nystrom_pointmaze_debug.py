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
from matplotlib import patches
import utils
import logging
from agent.utils import (
    EncodedActorUpdateData,
    PointMazeNystromDebugHelper,
    RawActorUpdateData,
)
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
                 pca_truncation,
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
                 debug_fixed_dataset_updates: bool = False,
                 encoded_fifo_capacity: Optional[int] = None,
                 encoded_fifo_encode_batch_size: int = 4096,
                 encoded_fifo_cuda_oom_splits: int = 4,
                 kernel_type: str = "inner_product",
                 kernel_bandwidth: Optional[float] = None,
                 nystrom_grid_border_margin: float = 0.05,
                 nystrom_grid_oversample: float = 2.0,
                 nystrom_exact_grid: bool = False,
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

        self.sink_schedule = sink_schedule
        self.epsilon_schedule = epsilon_schedule
        self.gradient_coeff = None
        self.pca_truncation = pca_truncation

        self.num_expl_steps = num_expl_steps
        self.lambda_reg = lambda_reg
        self.image_channels = 1 if self.grayscale else 3
        self.kernel_type = str(kernel_type or "inner_product").strip().lower()
        self.kernel_bandwidth = kernel_bandwidth
        self.kernel_fn = utils.build_kernel_fn(
            self.kernel_type,
            bandwidth=self.kernel_bandwidth,
        )
        self.subsamples = subsamples
        self.nystrom_synthetic_subsamples = bool(nystrom_synthetic_subsamples)
        self.debug_fixed_dataset_updates = bool(debug_fixed_dataset_updates)
        #####
        if self.debug_fixed_dataset_updates:
            utils.ColorPrint.yellow("DEBUG: encoder and actor updates use the fixed continuous Nyström dataset.")
        
        self.nystrom_debug = PointMazeNystromDebugHelper(
            border_margin=nystrom_grid_border_margin,
            oversample=nystrom_grid_oversample,
            exact_grid=nystrom_exact_grid,
        )
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
            pca_truncation=self.pca_truncation,
            kernel_type=self.kernel_type,
            kernel_bandwidth=self.kernel_bandwidth,
            device=self.device  
        )
        # TODO: sistemare gestione del kernel- Per ora in distribution_matching c'è lo state-action kernel, while qui c'è lo state kernel
        self.distribution_matcher.state_kernel_fn = self.kernel_fn
        
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

    def insert_env(self, env):
        """
        Insert environment reference for gridworld-specific visualizations.
        Call this from pretrain.py after agent creation.
        """       
        self.wrapped_env = env
        self.env = self._find_discrete_env(env)
        self.nystrom_debug.attach_env(env)
        
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

    def _kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return self.kernel_fn(X, Y)

    def _kernel_status(self, kernel_fn=None) -> str:
        kernel_fn = self.kernel_fn if kernel_fn is None else kernel_fn
        bandwidth = getattr(kernel_fn, "bandwidth", None)
        if bandwidth is None:
            return f"kernel={self.kernel_type}"
        return f"kernel={self.kernel_type}, bandwidth={bandwidth:.6g}"

    @staticmethod
    def _uniform_tensor_indices(n_items, max_items, device):
        if n_items <= max_items:
            return torch.arange(n_items, device=device)
        return torch.round(torch.linspace(0, n_items - 1, max_items, device=device)).long()

    def _kernel_debug_matrix(self, kernel_fn, X, Y, max_points=300):
        x_idx = self._uniform_tensor_indices(X.shape[0], max_points, X.device)
        y_idx = self._uniform_tensor_indices(Y.shape[0], max_points, Y.device)
        with torch.no_grad():
            matrix = kernel_fn(X[x_idx], Y[y_idx]).detach().float().cpu().numpy()
        return matrix

    @staticmethod
    def _unique_rows(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.numel() == 0:
            return tensor
        rows = tensor.detach().cpu().reshape(tensor.shape[0], -1).numpy()
        seen = set()
        keep_indices = []
        for idx, row in enumerate(rows):
            key = row.tobytes()
            if key in seen:
                continue
            seen.add(key)
            keep_indices.append(idx)
        index = torch.as_tensor(keep_indices, dtype=torch.long, device=tensor.device)
        return tensor.index_select(0, index)

    def _save_actor_kernel_debug_plot(
            self,
            step,
            state_X,
            state_Y,
            state_action_X,
            state_action_Y,
            state_action_X_actions=None,
            state_action_Y_actions=None,
            nystrom=False
        ):
        if self.kernel_type != "gaussian":
            return

        save_dir = os.path.join(os.getcwd(), "kernel_debug_plots")
        os.makedirs(save_dir, exist_ok=True)
        suffix = "nystrom" if nystrom else "full"
        save_path = os.path.join(save_dir, f"step_{step}_{suffix}_kernels.png")

        # Plot unique next-state points to avoid action-repeated rows in the state kernel.
        state_matrix = self._kernel_debug_matrix(self.kernel_fn, state_X, state_Y)
        if state_action_X_actions is None or state_action_Y_actions is None:
            action_matrix = self._kernel_debug_matrix(self.distribution_matcher.kernel_fn, state_action_X, state_action_Y)
        else:
            x_idx = self._uniform_tensor_indices(state_action_X.shape[0], 300, state_action_X.device)
            y_idx = self._uniform_tensor_indices(state_action_Y.shape[0], 300, state_action_Y.device)
            x_actions = state_action_X_actions.to(device=state_action_X.device)
            y_actions = state_action_Y_actions.to(device=state_action_Y.device)
            with torch.no_grad():
                action_matrix = self.distribution_matcher.state_action_kernel(
                    state_action_X[x_idx],
                    state_action_Y[y_idx],
                    x_actions[x_idx],
                    y_actions[y_idx],
                ).detach().float().cpu().numpy()
        state_values = state_matrix.reshape(-1)
        action_values = action_matrix.reshape(-1)

        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        panels = [
            (axes[0, 0], state_matrix, "State / next-state kernel heatmap", self._kernel_status(self.kernel_fn)),
            (axes[1, 0], action_matrix, "State-action kernel heatmap", self._kernel_status(self.distribution_matcher.kernel_fn)),
        ]
        for ax, matrix, title, status in panels:
            im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
            ax.set_title(f"{title}\n{status}", fontsize=10)
            ax.set_xlabel("Y support")
            ax.set_ylabel("X support")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for ax, values, title in [
            (axes[0, 1], state_values, "State / next-state kernel values"),
            (axes[1, 1], action_values, "State-action kernel values"),
        ]:
            ax.hist(values, bins=70, range=(0.0, 1.0), color="#2563eb", alpha=0.82)
            ax.axvline(float(np.mean(values)), color="black", linestyle="--", linewidth=1.1, label=f"mean={np.mean(values):.3g}")
            ax.axvline(float(np.median(values)), color="#dc2626", linestyle=":", linewidth=1.3, label=f"median={np.median(values):.3g}")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("kernel value")
            ax.set_ylabel("count")
            ax.legend(fontsize=8)

        fig.suptitle(f"Automatic Gaussian sigma diagnostics at step {step} ({suffix})", fontsize=13)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Kernel sigma debug plot saved to: {save_path}")
    
    
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
            H = self._kernel(enc_obs_augmented, self._phi_all_obs)  # [1, num_unique]
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
            if encoded_full is None:
                raise ValueError("Nyström actor update requires encoded_full when encoded_sub is provided.")
            self._cache_encoded_features(encoded_full, encoded_sub=encoded_sub)
            if encoded_sub is None:
                self._use_full_features_as_subsample()
        elif sub_obs is None or sub_action is None or sub_next_obs is None:
            self._sync_policy_encoder()
            self._cache_features(
                full_obs,
                full_action,
                full_next_obs,
                encoder=self.policy_encoder,
            )
            self._use_full_features_as_subsample()
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
        sub_H = self._kernel(self._phi_all_obs, self._phi_sub_next) # [n, m]
        self._save_actor_kernel_debug_plot(
            step,
            state_X=self._phi_sub_next,
            state_Y=self._phi_sub_next,
            state_action_X=self._phi_sub_obs,
            state_action_Y=self._phi_sub_obs,
            state_action_X_actions=self._sub_actions,
            state_action_Y_actions=self._sub_actions,
            nystrom=True,
        )
        utils.ColorPrint.yellow(f"Actor state kernel: {self._kernel_status(self.kernel_fn)}")
        base_eta = float(utils.schedule(self.lr_actor, step))
        base_eta = float(np.clip(base_eta, self.pmd_eta_min, self.pmd_eta_max))
        self.current_eta = base_eta

        sink_norm = utils.schedule(self.sink_schedule, step)
        self.pi = self._policy_from_H(sub_H.T, coeff=self.gradient_coeff)  # [z_x+1, n_actions]

        K_sub_sub = self.distribution_matcher.state_action_kernel(
            self._phi_sub_obs,
            self._phi_sub_obs,
            self._sub_actions,
            self._sub_actions,
        )  # [m, m]
        K_all_sub = self.distribution_matcher.state_action_kernel(
            self._phi_all_obs,
            self._phi_sub_obs,
            self._all_actions,
            self._sub_actions,
        )  # [n, m]


        B_nystrom, U_r = self.distribution_matcher.compute_B_and_projections(
            K_nm=K_all_sub,
            K_mm=K_sub_sub,
            components=self.pca_truncation
        )

        del K_sub_sub, K_all_sub
        

        utils.ColorPrint.yellow(
            f"Nyström state-action kernel: {self._kernel_status(self.distribution_matcher.kernel_fn)}"
        )

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
                    B_nystrom=B_nystrom,
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

        for iteration in range(self.pmd_steps):
            grad_update = self.distribution_matcher.compute_gradient_coefficient_nystrom_memory_efficient_and_projection(
                phi_sub_next_obs = self._phi_sub_next,
                psi_sub_obs_action = self._psi_sub,
                H = sub_H,
                pi=self.pi,
                E=self.E,
                alpha = self._sub_alpha,
                sink_norm=sink_norm,
                B_nystrom=B_nystrom,
                eig_vecs_r=U_r,
            )           

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

            if iteration % 1 == 0 or iteration == self.pmd_steps - 1:
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

    
    def _cache_features(self, obs, action, next_obs, encoder=None, sub_obs=None, sub_action=None, sub_next_obs=None):
        """Pre-compute and cache dataset features."""
        encoder = self.encoder if encoder is None else encoder
       
        with torch.no_grad():
            
            print(f"encoding obs shape: {obs.shape}, next_obs shape: {next_obs.shape}")
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

    def _use_full_features_as_subsample(self):
        self._phi_sub_obs = self._phi_all_obs
        self._phi_sub_next = self._phi_all_next
        self._psi_sub = self._psi_all
        self._sub_actions = self._all_actions
        self._sub_alpha = self._alpha

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

    def _nystrom_subsample_count(self) -> int:
        if self.subsamples is None:
            raise ValueError("Nyström actor update requires agent.subsamples to be set.")
        count = int(self.subsamples)
        if count <= 0:
            raise ValueError("subsamples must be positive when provided")
        return count

    def _fixed_actor_update_data(self) -> RawActorUpdateData:
        fixed_batch = self.nystrom_debug.fixed_actor_batch(self, n_transitions=int(self.batch_size_actor))
        subsample = (
            self.nystrom_debug.fixed_actor_batch(self, n_transitions=self._nystrom_subsample_count())
            if self.subsamples is not None else None
        )
        return RawActorUpdateData(
            full=fixed_batch,
            subsample=subsample,
            source="fixed PointMaze debug dataset",
        )

    def _xy_points_from_actor_batch(self, actor_batch, expected_size=None):
        if actor_batch is None:
            return None
        if (
            self.debug_fixed_dataset_updates
            and expected_size is not None
            and hasattr(self.nystrom_debug, "fixed_xy_points_for_size")
        ):
            fixed_points = self.nystrom_debug.fixed_xy_points_for_size(int(expected_size))
            if fixed_points is not None:
                return np.asarray(fixed_points, dtype=np.float32).reshape(-1, 2)
        obs = actor_batch[0]
        if obs is None or obs.ndim < 2 or obs.shape[1] < 2:
            return None
        if self.obs_type == "pixels":
            return None
        obs_np = obs.detach().float().cpu().numpy().reshape(obs.shape[0], -1)
        points = obs_np[:, :2]
        if points.shape[0] == 0 or not np.isfinite(points).all():
            return None
        return points.astype(np.float32, copy=False)

    def _save_pointmaze_actor_dataset_plot(self, step, points, filename, title):
        if points is None:
            return
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if points.size == 0:
            return

        save_dir = os.path.join(os.getcwd(), "pointmaze_plots")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        try:
            layout = self.nystrom_debug.maze_layout()
        except Exception:
            layout = None
        if isinstance(layout, dict):
            for x0, y0, width, height in layout["wall_rectangles"]:
                ax.add_patch(
                    patches.Rectangle(
                        (x0, y0),
                        width,
                        height,
                        facecolor="black",
                        edgecolor="black",
                        linewidth=0.5,
                        zorder=1,
                    )
                )
            lower = np.asarray(layout["maze_lower"], dtype=np.float32)
            upper = np.asarray(layout["maze_upper"], dtype=np.float32)
            ax.add_patch(
                patches.Rectangle(
                    (lower[0], lower[1]),
                    upper[0] - lower[0],
                    upper[1] - lower[1],
                    fill=False,
                    edgecolor="black",
                    linewidth=1.2,
                    zorder=2,
                )
            )
            ax.set_xlim(lower[0] - 0.1, upper[0] + 0.1)
            ax.set_ylim(lower[1] - 0.1, upper[1] + 0.1)
        else:
            pad = 0.05 * max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1.0)
            ax.set_xlim(float(points[:, 0].min()) - pad, float(points[:, 0].max()) + pad)
            ax.set_ylim(float(points[:, 1].min()) - pad, float(points[:, 1].max()) + pad)

        ax.scatter(points[:, 0], points[:, 1], s=8, c="#ff7f0e", linewidths=0.0, alpha=0.9, zorder=8)
        ax.scatter(points[0, 0], points[0, 1], marker="*", s=130, c="white", edgecolors="black", linewidths=0.9, zorder=9)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{title}\n{points.shape[0]} states")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"PointMaze actor dataset plot saved to: {save_path}")

    def _save_actor_full_dataset_plot(self, actor_data, step):
        if isinstance(actor_data, EncodedActorUpdateData):
            return
        full_size = actor_data.full[0].shape[0]
        self._save_pointmaze_actor_dataset_plot(
            step,
            self._xy_points_from_actor_batch(actor_data.full, expected_size=full_size),
            f"step_{step}_actor_full_dataset.png",
            "PointMaze actor full dataset",
        )

    def _save_actor_nystrom_subsample_plot(self, actor_data, step):
        if isinstance(actor_data, EncodedActorUpdateData):
            return
        if actor_data.subsample is None:
            return
        subsample_size = actor_data.subsample[0].shape[0]
        self._save_pointmaze_actor_dataset_plot(
            step,
            self._xy_points_from_actor_batch(actor_data.subsample, expected_size=subsample_size),
            f"step_{step}_nystrom_subsamples.png",
            "PointMaze Nyström subsamples",
        )

    def _synthetic_actor_subsample_batch(self):
        return self.nystrom_debug.fixed_actor_batch(self)

    def _encoded_fifo_actor_update_data(self, replay_buffer):
        if not self._update_encoded_actor_fifo(replay_buffer):
            return None

        if self.subsamples is None:
            full, rewards = self._sample_encoded_actor_data(
                self.batch_size_actor,
                include_first=True,
            )
            return EncodedActorUpdateData(
                full=full,
                rewards=rewards,
                source=f"encoded FIFO sample of batch_size_actor={self.batch_size_actor}",
            )

        # Nyström uses the whole encoded FIFO as support and a smaller landmark set.
        count = self._nystrom_subsample_count()
        full, rewards = self._all_encoded_actor_data(include_first=True)
        if self.nystrom_synthetic_subsamples:
            subsample, subsample_rewards = self.nystrom_debug.encode_subsamples(self)
            subsample_source = "fixed PointMaze Nyström landmarks"
        else:
            subsample, subsample_rewards = self._sample_encoded_actor_data(
                count,
                include_first=True,
            )
            subsample_source = f"encoded FIFO Nyström sample of subsamples={count}"
        return EncodedActorUpdateData(
            full=full,
            rewards=rewards,
            subsample=subsample,
            subsample_rewards=subsample_rewards,
            source=f"encoded FIFO full support + {subsample_source}",
        )

    def _replay_actor_subsample_batch(self, replay_iter, full_batch, replay_buffer):
        count = self._nystrom_subsample_count()
        if self.nystrom_synthetic_subsamples:
            return self._synthetic_actor_subsample_batch()
        if count >= full_batch[0].shape[0]:
            return full_batch
        return self._load_actor_subsample_from_replay_iter(
            replay_iter,
            max_samples=count,
            replay_buffer=replay_buffer,
            fallback_actor_batch=full_batch,
        )

    def _replay_actor_update_data(self, replay_iter, obs, action, next_obs, reward, replay_buffer):
        full_batch = self._load_actor_batch_from_replay_iter(
            replay_iter,
            obs,
            action,
            next_obs,
            reward,
            max_samples=self.batch_size_actor,
            replay_buffer=replay_buffer,
        )
        if self.subsamples is None:
            return RawActorUpdateData(
                full=full_batch,
                source=f"replay iterator sample of batch_size_actor={self.batch_size_actor}",
            )
        return RawActorUpdateData(
            full=full_batch,
            subsample=self._replay_actor_subsample_batch(replay_iter, full_batch, replay_buffer),
            source=(
                "replay full support + fixed PointMaze Nyström landmarks"
                if self.nystrom_synthetic_subsamples
                else f"replay full support + replay Nyström subsample of subsamples={self.subsamples}"
            ),
        )

    def _get_actor_update_data(self, replay_iter, obs, action, next_obs, reward, replay_buffer=None):
        """Choose the actor dataset for this step.

        Priority is explicit: fixed debug dataset, encoded FIFO, then raw
        replay. If subsamples is None, the object carries only the full actor
        batch and update_actor is used. Otherwise it also carries Nyström data.
        """
        if self.debug_fixed_dataset_updates:
            return self._fixed_actor_update_data()

        encoded_data = self._encoded_fifo_actor_update_data(replay_buffer)
        if encoded_data is not None:
            return encoded_data

        return self._replay_actor_update_data(
            replay_iter,
            obs,
            action,
            next_obs,
            reward,
            replay_buffer,
        )

    def _log_actor_update_data(self, actor_data):
        if isinstance(actor_data, EncodedActorUpdateData):
            full_size = self._encoded_batch_size(actor_data.full)
            subsample_size = (
                self._encoded_batch_size(actor_data.subsample)
                if actor_data.subsample is not None else "N/A"
            )
            data_kind = "encoded"
        else:
            full_size = actor_data.full[0].shape[0]
            subsample_size = actor_data.subsample[0].shape[0] if actor_data.subsample is not None else "N/A"
            data_kind = "raw"

        utils.ColorPrint.red(
            f"actor update data: update=update_actor_nystrom, kind={data_kind}, "
            f"source={actor_data.source}, full={full_size}, subsampled={subsample_size}"
        )


    def _update_actor_from_data(self, actor_data, step):
        self._log_actor_update_data(actor_data)
        # DEBUG PLOTS: remove these two lines to disable actor dataset dumps.
        self._save_actor_full_dataset_plot(actor_data, step)
        self._save_actor_nystrom_subsample_plot(actor_data, step)

        if isinstance(actor_data, EncodedActorUpdateData):
            return self.update_actor_nystrom(
                None,
                None,
                None,
                step=step,
                rewards=actor_data.rewards,
                sub_rewards=actor_data.subsample_rewards,
                encoded_full=actor_data.full,
                encoded_sub=actor_data.subsample,
            )

        obs, action, next_obs, reward = actor_data.full
        sub_obs = sub_action = sub_next_obs = sub_reward = None
        if actor_data.subsample is not None:
            sub_obs, sub_action, sub_next_obs, sub_reward = actor_data.subsample
        return self.update_actor_nystrom(
            obs,
            action,
            next_obs,
            step=step,
            rewards=reward,
            sub_obs=sub_obs,
            sub_action=sub_action,
            sub_next_obs=sub_next_obs,
            sub_rewards=sub_reward,
        )

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

    def _debug_visualizer_text(self, step) -> str:
        return (
            f"Step: {step}\n"
            f"γ = {self.discount}\n"
            f"η = {self.current_eta}\n"
            f"λ = {self.lambda_reg}\n"
            f"sink norm = {utils.schedule(self.sink_schedule, step):.6f}\n"
            f"PMD steps = {self.pmd_steps}\n"
            f"subsamples = {self.subsamples if self.subsamples is not None else 'all'}\n"
        )

    def _run_debug_visualizers(self, metrics, obs, step):
        if self.debug_visualizer is None:
            return metrics

        visualizer_obs, visualizer_z = self._build_debug_visualizer_batch(obs)
        metrics.update(
            self.debug_visualizer.save(
                step=step,
                obs_batch=visualizer_obs,
                z_batch=self._unique_rows(visualizer_z),
                param_text=self._debug_visualizer_text(step),
            )
        )

        if len(self.current_action_probs) == 0:
            return metrics

        current_action_probs = np.array(self.current_action_probs)
        mean_deviation = self._compute_mean_action_probs_deviation(current_action_probs)
        mean_probs = np.mean(current_action_probs, axis=0)

        self.policy_deviation_history.append((step, mean_deviation))
        self.action_probs_history.append((step, mean_probs))
        self.current_action_probs = []

        metrics['policy_deviation_from_uniform'] = mean_deviation
        print(
            f"Policy deviation from uniform: {mean_deviation:.4f} "
            f"(0=uniform, {(self.n_actions - 1) / self.n_actions:.3f}=deterministic)"
        )
        self.plot_policy_deviation_history(save_dir=os.path.join(os.getcwd(), 'policy_plots'))
        return metrics

    def update(self, replay_iter, step, replay_buffer=None):
        metrics = dict()

        if step % self.update_every_steps != 0 and self._is_T_sufficiently_initialized(step) is True:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)
        if self.debug_fixed_dataset_updates:
            obs, action, next_obs, reward = self.nystrom_debug.fixed_encoder_batch(self)

        if self.use_tb or self.use_wandb:
            metrics['batch_reward'] = reward.mean().item()
        if self.embeddings:
            metrics.update(self.update_encoders(obs, action, next_obs, reward))

        # Train the encoder/transition model first; PMD starts once T is ready.
        if not self._is_T_sufficiently_initialized(step):
            metrics['actor_loss'] = 100.0  # dummy value
            return metrics

        if step % self.update_actor_every_steps == 0 or step == self.num_expl_steps + self.T_init_steps:
            actor_update_data = self._get_actor_update_data(
                replay_iter,
                obs,
                action,
                next_obs,
                reward,
                replay_buffer=replay_buffer,
            )
            metrics.update(self._update_actor_from_data(actor_update_data, step))
            metrics = self._run_debug_visualizers(metrics, obs, step)
        exit()
        return metrics
