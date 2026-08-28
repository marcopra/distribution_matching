from collections import OrderedDict
import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import utils
import logging
from agent.rover_utils.types import EncodedActorUpdateData, RawActorUpdateData
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
from agent.rover_utils.buffers import EncodedTransitionFIFO
from agent.rover_utils.matchers import DistributionMatcher
from agent.rover_utils.networks import CNNEncoder, Encoder, ProjectSA

# ============================================================================
# Main Agent
# ============================================================================
class RoverAgent:
    requires_transition_view = True

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
                 debug: bool = False,
                 debug_config=None,
                 encoded_fifo_capacity: Optional[int] = None,
                 encoded_fifo_encode_batch_size: int = 4096,
                 encoded_fifo_cuda_oom_splits: int = 4,
                 max_pending_transitions: Optional[int] = None,
                 kernel_type: str = "inner_product",
                 kernel_bandwidth: Optional[float] = None,
                 subsampling_strategy: str = "random",
                 nystrom_candidate_multiplier: float = 5.0,
                 nystrom_cholesky_tolerance: float = 1e-6,
                 nystrom_cholesky_progress: bool = True,
                 device: str = "cpu",
                 ):

        self.compute_dtype = _resolve_torch_dtype(compute_dtype)
        torch.set_default_dtype(self.compute_dtype)
        self.device = device

        # ** MDP settings **
        self.n_states = obs_shape[0]
        self.n_actions = action_shape[0]
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        self.grayscale = grayscale
        self.image_channels = 1 if self.grayscale else 3
        self.feature_dim = feature_dim if feature_dim is not None else self.n_states
        self.action_dim = action_shape[0]
        self.discount = discount

        # ** Learning settings **
        self.lr_actor = lr_actor
        self.lr_T = lr_T
        self.num_expl_steps = num_expl_steps # Number of starting exploration steps
        self.T_init_steps = T_init_steps # Number of initial transition learning steps before PMD starts
        self.batch_size = batch_size
        self.batch_size_actor = batch_size_actor
        # assert batch_size_actor >= batch_size, "Actor update batch size must be greater than or equal to encoder update batch size"
        self.update_every_steps = update_every_steps # Update encoder and transition model every N steps
        self.update_actor_every_steps = update_actor_every_steps # Update actor every N steps - PMD update
        self.pmd_steps = pmd_steps
        # PMD settings - Adaptive learning rate for PMD updates
        self.pmd_eta_mode = pmd_eta_mode.lower()
        assert self.pmd_eta_mode in ["none", "adagrad", "backtracking", "adadiff"], "pmd_eta_mode must be one of ['none', 'adagrad', 'backtracking', 'adadiff']"
        self.pmd_best_iterate = pmd_best_iterate
        self.pmd_grad_clip_norm = pmd_grad_clip_norm
        self.pmd_adagrad_eps = pmd_adagrad_eps
        self.pmd_eta_min = pmd_eta_min
        self.pmd_eta_max = pmd_eta_max
        self.pmd_backtrack_factor = pmd_backtrack_factor
        self.pmd_backtrack_max_trials = pmd_backtrack_max_trials

        # ** Logging mode **
        self.use_tb = use_tb
        self.use_wandb = use_wandb

        # ** Encoder settings **
        self.embeddings = embeddings
        self.curl = curl # Stronglu suggest to NOT use CURL, as it may cause poor performance. All the paper results are without CURL.
        if curl:
            utils.ColorPrint.red("CURL is enabled, but stromgly suggested to not use it.\nAll the paper results are without CURL, and it may cause poor performance. Use with caution.")
        self.embedding_sum_loss = embedding_sum_loss # Constraint on the sum of the embeddings to be close to 1.0, as suggested in the paper. It is not very helpful in practice
        self.reward = reward # Constraint the embedder to predict the reward. It is not very helpful in practice in URL
        self.mode = mode # Normalization mode for the embeddings. 'l1' or 'l2'. 'l1' is STRONGLY suggested.
        assert self.mode in ['l1', 'l2'], "Mode must be 'l1' or 'l2'"

        # ** Sink schedule and epsilon schedule for exploration **
        self.sink_schedule = sink_schedule # Sink state norm schedule for exploration
        self.epsilon_schedule = epsilon_schedule # Epsilon-greedy exploration schedule - i.e. probability of taking a random action

        # ** Gradient Parameters for PMD updates **
        self.gradient_coeff = None
        self.pca_truncation = pca_truncation # PCA truncation for matrix inversions in rover gradient computations
        self.lambda_reg = lambda_reg

        # ** Kernel Settings **
        self.kernel_type = str(kernel_type or "inner_product").strip().lower()
        self.kernel_bandwidth = kernel_bandwidth
        
        # ** Subsampling settings for Nyström approximation **
        self.subsampling_strategy = str(subsampling_strategy).lower()
        if self.subsampling_strategy not in ("random", "pivoted_cholesky"):
            raise ValueError(
                "subsampling_strategy must be random or pivoted_cholesky"
            )
        self.nystrom_candidate_multiplier = float(nystrom_candidate_multiplier) # Number of candidates to subsample for Nyström approximation, relative to the number of subsamples. Only used for pivoted_cholesky.
        self.nystrom_cholesky_tolerance = float(nystrom_cholesky_tolerance) # Needed for pivoted Cholesky subsampling
        self.nystrom_cholesky_progress = bool(nystrom_cholesky_progress)
        if self.nystrom_candidate_multiplier < 1.0:
            raise ValueError("nystrom_candidate_multiplier must be at least 1")
        if self.nystrom_cholesky_tolerance < 0.0:
            raise ValueError("nystrom_cholesky_tolerance must be non-negative")
        if self.subsampling_strategy == "pivoted_cholesky":
            # Selector must reproduce actor kernel columns exactly. Gaussian
            # bandwidth is fitted from candidate pool and reused by actor update.
            # Extend FIFO kernel-column computation with this assertion when
            # supporting another kernel.
            assert self.kernel_type in ("inner_product", "gaussian"), (
                "pivoted_cholesky subsampling currently requires kernel_type "
                "inner_product or gaussian"
            )
        self.kernel_fn = utils.build_kernel_fn(
            self.kernel_type,
            bandwidth=self.kernel_bandwidth,
        )
        self.subsamples = subsamples # N of subsamples for Nyström approximation. If None, use all the samples in the FIFO buffer.

        self.debug = bool(debug)
        self.debug_manager = None
        if self.debug:
            from agent.rover_utils.debug import make_debug_manager

            self.debug_manager = make_debug_manager(self, debug_config)

        # ** FIFO buffer for encoded transitions settings **
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
        self.max_pending_transitions = (
            None if max_pending_transitions is None
            else int(max_pending_transitions)
        )
        self._encoded_actor_fifo = EncodedTransitionFIFO(self.encoded_fifo_capacity)
        self._encoded_fifo_replay_marker = None
        

        # ** Neural network components **
        # PIXEL observations
        if obs_type == 'pixels':

            if self.curl: # CURL augmentation for pixel observations
                self.aug = utils.RandomShiftsAug(pad=4)
            else: # No augmentation for pixel observations
                self.aug = nn.Identity()
            assert embeddings, "Pixel observations require embeddings to be True"

            self.encoder = CNNEncoder(
                obs_shape,
                feature_dim,
                mode=mode
            ).to(self.device)
            
            self.obs_dim = self.feature_dim
        # PROPRIO observations
        else:
            # Components
            self.aug = nn.Identity()
            if embeddings == False: # NO embeddings for state observations - Use identity encoder
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
       
        # State-action projection network: W(ψ(s,a)) 
        self.project_sa = ProjectSA(
            self.obs_dim * self.n_actions,
            hidden_dim,
            self.obs_dim,
            linear=linear_projection
        ).to(self.device)

        
        self.policy_encoder = copy.deepcopy(self.encoder).to(self.device)
        self._freeze_module(self.policy_encoder)
        self._policy_is_synced = True

        # ** Distribution matcher for PMD updates **
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

       
        if self.reward:
            self.reward = nn.Sequential(
                nn.Linear(self.obs_dim * self.n_actions, hidden_dim), 
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1)
            ).to(self.device)
        
        # parameter list:
        parameters = list(self.encoder.parameters()) 

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

        # Loss functions
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.training = False

        if self.debug_manager is not None:
            self.debug_manager.preserve_legacy_rng_sequence()

        self.current_eta = 0.0
        self._adagrad_accum = None

        self.subsampled = None

    def insert_env(self, env):
        if self.debug_manager is not None:
            self.debug_manager.attach_env(env)


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

    def _fit_state_kernel_bandwidth(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        del Y
        if self.kernel_type != "gaussian" or self.kernel_bandwidth is not None:
            return
        if self.subsampling_strategy == "pivoted_cholesky" and self.kernel_type == "gaussian":
            bandwidth = self._encoded_actor_fifo.last_pivoted_cholesky_bandwidth
            if bandwidth is not None:
                self.kernel_fn.bandwidth = bandwidth
                self.distribution_matcher.kernel_fn.bandwidth = bandwidth
                utils.ColorPrint.yellow(
                    f"Using pivoted-Cholesky candidate-pool Gaussian bandwidth={bandwidth:.6g}."
                )
                return
        with torch.no_grad():
            candidates = X.detach().reshape(X.shape[0], -1)
            if candidates.shape[0] > 1000:
                indices = torch.randperm(candidates.shape[0], device=candidates.device)[:1000]
                candidates = candidates[indices]
            distances = torch.pdist(candidates, p=2)
            distances = distances[distances > 0]
            bandwidth = 1.0 if distances.numel() == 0 else float(torch.median(distances).item())
        self.kernel_fn.bandwidth = max(bandwidth, 1e-12)
        self.distribution_matcher.kernel_fn.bandwidth = self.kernel_fn.bandwidth
    
    
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
                obs_tensor = torch.as_tensor(obs, device=self.device, dtype=self.compute_dtype).unsqueeze(0)  # [1, C, H, W]
            else:
                # State observations: [x, y] -> [1, 2]
                obs_tensor = torch.as_tensor(obs, device=self.device, dtype=self.compute_dtype).unsqueeze(0)  # [1, obs_dim]
            
            enc_obs = self._encode_with_module(self.policy_encoder, obs_tensor, project=True)
    
            if self.gradient_coeff is None:
                return np.ones(self.n_actions) / self.n_actions
            
            # Add a zero to enc_obs to account for the extra row in H
            enc_obs_augmented = torch.cat([enc_obs, torch.zeros((1, 1), device=enc_obs.device, dtype=enc_obs.dtype)], dim=1)  # [1, feature_dim + 1]
            H = self._kernel(enc_obs_augmented, self._phi_all_obs)  # [1, num_unique]
            probs = self._policy_from_H(H)

            
            if torch.sum(probs) == 0.0 or torch.isnan(torch.sum(probs)):
                utils.ColorPrint.red(f"Warning: action_probs sum to zero or NaN. Returning uniform distribution. Check training stability and learning rates.{torch.sum(probs)}, {probs}")
                probs = torch.ones_like(probs) / self.n_actions
                # raise ValueError(f"action_probs sum to zero or NaN", torch.sum(probs), probs)
            logger.debug(f"Action probabilities: {probs.cpu().numpy().flatten()}")
            return probs.cpu().numpy().flatten()

    def _compute_action_probs_batch(self, observations: np.ndarray) -> np.ndarray:
        """Compute π(·|s) for a batch of observations."""
        observations = np.asarray(observations)
        with torch.no_grad():
            obs_tensor = torch.as_tensor(
                observations,
                device=self.device,
                dtype=self.compute_dtype,
            )
            enc_obs = self._encode_with_module(self.policy_encoder, obs_tensor, project=True)

            if self.gradient_coeff is None:
                return np.full(
                    (observations.shape[0], self.n_actions),
                    1.0 / self.n_actions,
                    dtype=np.float64,
                )

            zeros = torch.zeros(
                (enc_obs.shape[0], 1),
                device=enc_obs.device,
                dtype=enc_obs.dtype,
            )
            enc_obs_augmented = torch.cat([enc_obs, zeros], dim=1)
            H = self._kernel(enc_obs_augmented, self._phi_all_obs)
            probs = self._policy_from_H(H)
            bad_rows = (torch.sum(probs, dim=1) == 0.0) | torch.isnan(torch.sum(probs, dim=1))
            if torch.any(bad_rows):
                utils.ColorPrint.red(
                    "Warning: some batched action_probs sum to zero or NaN. "
                    "Using uniform distribution for those rows."
                )
                probs = probs.clone()
                probs[bad_rows] = torch.ones(
                    self.n_actions,
                    device=probs.device,
                    dtype=probs.dtype,
                ) / self.n_actions
            return probs.detach().cpu().numpy()

    
    def act(self, obs, meta, step, eval_mode):
        if step < self.num_expl_steps or np.random.rand() < utils.schedule(self.epsilon_schedule, step):
            return np.random.randint(self.n_actions)

        # Compute action probabilities
        action_probs = self.compute_action_probs(obs)
        self.debug_manager.record_action_probs(action_probs) if self.debug_manager is not None else None  # Store for visualization
        # print(f"Step {step}: Action probabilities: {action_probs}")
        # Sample action
        return np.random.choice(self.n_actions, p=action_probs)

    def act_parallel(self, observations, metas, step, eval_mode):
        observations = np.asarray(observations)
        num_envs = observations.shape[0]
        steps = np.asarray(step if np.ndim(step) > 0 else [step] * num_envs, dtype=np.int64)
        if steps.shape[0] != num_envs:
            raise ValueError(f"Expected {num_envs} logical steps, got {steps.shape[0]}")

        actions = np.empty(num_envs, dtype=np.int64)
        policy_indices = []
        for env_id, step_i in enumerate(steps):
            epsilon = utils.schedule(self.epsilon_schedule, int(step_i))
            if step_i < self.num_expl_steps or np.random.rand() < epsilon:
                actions[env_id] = np.random.randint(self.n_actions)
            else:
                policy_indices.append(env_id)

        if policy_indices:
            try:
                batch_obs = observations[policy_indices]
                action_probs = self._compute_action_probs_batch(batch_obs)
                if action_probs.shape != (len(policy_indices), self.n_actions):
                    raise ValueError(f"Unexpected action_probs shape {action_probs.shape}")
                for row, env_id in enumerate(policy_indices):
                    probs = action_probs[row]
                    self.debug_manager.record_action_probs(probs) if self.debug_manager is not None else None
                    actions[env_id] = np.random.choice(self.n_actions, p=probs)
            except Exception as exc:
                utils.ColorPrint.yellow(f"Batched act_parallel failed; falling back to looped act: {exc}")
                for env_id in policy_indices:
                    meta = metas[env_id] if metas is not None else None
                    actions[env_id] = self.act(
                        observations[env_id],
                        meta,
                        int(steps[env_id]),
                        eval_mode=eval_mode,
                    )

        return actions

    
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
        self.gradient_coeff = torch.zeros((self._phi_all_obs.shape[0]+1, 1), device=self.device, dtype=self.compute_dtype)  # [z_x + 1, 1]
        prev_gradient_coeff = self.gradient_coeff.clone()
        self._fit_state_kernel_bandwidth(self._phi_all_obs, self._phi_sub_next)
        sub_H = self._kernel(self._phi_all_obs, self._phi_sub_next) # [n, m]
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
    
        U_r = U_r.to(self.compute_dtype)
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
            grad_update = self.distribution_matcher.compute_gradient_coefficient_nystrom_blockwise_and_proj(
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
            

        metrics['actor_loss'] = float(actor_loss)
        metrics['actor_eta'] = float(self.current_eta)
        metrics['actor_best_loss'] = float(best_loss)
        metrics['sink_norm'] = float(sink_norm)
   
        return metrics

    
    def _cache_features(self, obs, action, next_obs, encoder=None, sub_obs=None, sub_action=None, sub_next_obs=None):
        """Pre-compute and cache dataset features."""
        encoder = self.encoder if encoder is None else encoder
       
        with torch.no_grad():
            
            print(f"encoding obs shape: {obs.shape}, next_obs shape: {next_obs.shape}")
            self._phi_all_obs = self._encode_with_module(encoder, obs, project=True).to(dtype=self.compute_dtype)
            self._phi_all_next = self._encode_with_module(encoder, next_obs, project=True).to(dtype=self.compute_dtype)

            action = action #.cpu()
            self._psi_all = self._encode_state_action(self._phi_all_obs, action) #.cpu()
            self._all_actions = action.long().reshape(-1).detach().cpu()
           
            self._alpha = torch.zeros((self._phi_all_next.shape[0], 1), device=self.device, dtype=self.compute_dtype)  # [n, 1]
    
            self._alpha[0] = 1.0  # set alpha to 1.0 for the first state
            self.E = F.one_hot(
                action, 
                self.n_actions,
            ).reshape(-1, self.n_actions).to(dtype=self.compute_dtype, device=self.device)

            # ** AUGMENTATION STEP **
            # ψ and Φ are augmented with an additional zero dimension
            zeros_col = torch.zeros(*self._psi_all.shape[:-1], 1, device=self._psi_all.device, dtype=self._psi_all.dtype)
            self._psi_all = torch.cat([self._psi_all, zeros_col], dim=-1)

            zero_col = torch.zeros(*self._phi_all_next.shape[:-1], 1, device=self._phi_all_next.device, dtype=self._phi_all_next.dtype)
            self._phi_all_next = torch.cat([self._phi_all_next, zero_col], dim=-1)

            zero_col = torch.zeros(*self._phi_all_obs.shape[:-1], 1, device=self._phi_all_obs.device, dtype=self._phi_all_obs.dtype)
            self._phi_all_obs = torch.cat([self._phi_all_obs, zero_col], dim=-1)

            if sub_obs is not None and sub_next_obs is not None and sub_action is not None:
                self._phi_sub_obs = self._encode_with_module(encoder, sub_obs, project=True).to(dtype=self.compute_dtype)
                self._phi_sub_next = self._encode_with_module(encoder, sub_next_obs, project=True).to(dtype=self.compute_dtype)
                self._sub_actions = sub_action.long().reshape(-1).detach().cpu()

                self._psi_sub = self._encode_state_action(self._phi_sub_obs, sub_action)

                zeros_col_sub_next = torch.zeros(*self._phi_sub_next.shape[:-1], 1, device=self._phi_sub_next.device, dtype=self._phi_sub_next.dtype)
                self._phi_sub_next = torch.cat([self._phi_sub_next, zeros_col_sub_next], dim=-1)

                zero_col_sub_obs = torch.zeros(*self._phi_sub_obs.shape[:-1], 1, device=self._phi_sub_obs.device, dtype=self._phi_sub_obs.dtype)
                self._phi_sub_obs = torch.cat([self._phi_sub_obs, zero_col_sub_obs], dim=-1)

                zero_col_sub_psi = torch.zeros(*self._psi_sub.shape[:-1], 1, device=self._psi_sub.device, dtype=self._psi_sub.dtype)
                self._psi_sub = torch.cat([self._psi_sub, zero_col_sub_psi], dim=-1)

                self._sub_alpha = torch.zeros((self._phi_sub_next.shape[0], 1), device=self.device, dtype=self.compute_dtype)  # [m, 1]
                self._sub_alpha[0] = 1.0  # set alpha to 1.0 for the first state

            print(f"dimensions after augmentation: psi_all {self._psi_all.shape}, phi_all_next {self._phi_all_next.shape}, phi_all_obs {self._phi_all_obs.shape}")

    def _append_zero_feature_column(self, tensor):
        zeros_col = torch.zeros(*tensor.shape[:-1], 1, device=tensor.device, dtype=tensor.dtype)
        return torch.cat([tensor, zeros_col], dim=-1)

    def _cache_encoded_features(self, encoded_full, encoded_sub=None):
        with torch.no_grad():
            self._phi_all_obs = self._append_zero_feature_column(encoded_full["phi_obs"].to(dtype=self.compute_dtype, device=self.device))
            self._phi_all_next = self._append_zero_feature_column(encoded_full["phi_next"].to(dtype=self.compute_dtype, device=self.device))
            self._psi_all = self._append_zero_feature_column(encoded_full["psi"].to(dtype=self.compute_dtype, device=self.device))

            self._alpha = torch.zeros((self._phi_all_next.shape[0], 1), device=self.device, dtype=self._phi_all_next.dtype)
            self._alpha[0] = 1.0

            self.E = encoded_full["E"].to(dtype=self.compute_dtype, device=self.device)
            self._all_actions = torch.argmax(encoded_full["E"], dim=1).long().detach().cpu()

            if encoded_sub is not None:
                self._phi_sub_obs = self._append_zero_feature_column(encoded_sub["phi_obs"].to(dtype=self.compute_dtype, device=self.device))
                self._phi_sub_next = self._append_zero_feature_column(encoded_sub["phi_next"].to(dtype=self.compute_dtype, device=self.device))
                self._psi_sub = self._append_zero_feature_column(encoded_sub["psi"].to(dtype=self.compute_dtype, device=self.device))
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
        encoded = {
            "phi_obs": phi_obs,
            "phi_next": phi_next,
            "psi": psi,
            "E": action_onehot,
            "reward": reward,
        }
        # TEMP DEBUG: carry PointMaze XY through encoded FIFO so actor dataset
        # plots still work when using real replay data instead of synthetic
        # Nyström/debug-fixed data. Remove with encoded plotting helpers below.
        if self.obs_type != "pixels" and obs.ndim >= 2 and obs.shape[1] >= 2:
            encoded["debug_xy"] = obs.detach().reshape(obs.shape[0], -1)[:, :2]
        return encoded

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
            terminal_mask = (
                np.asarray(transitions[3]).reshape(len(transition_ids), -1).min(axis=1)
                <= 0.0
            )
            encoded = self._encode_actor_transition_batch_with_retries(transitions)
            self._encoded_actor_fifo.add(
                transition_ids,
                encoded,
                terminal_mask=terminal_mask,
            )
            self._encoded_fifo_replay_marker = int(transition_ids[-1])
            if hasattr(replay_buffer, "mark_transitions_encoded"):
                replay_buffer.mark_transitions_encoded(self._encoded_fifo_replay_marker)
            inserted += int(len(transition_ids))

        self._insert_first_transition_if_available(replay_buffer)
        return inserted > 0 or len(self._encoded_actor_fifo) > 0

    def drain_encoded_actor_fifo(self, replay_buffer):
        """Encode pending replay transitions without running actor/encoder update."""
        return self._update_encoded_actor_fifo(replay_buffer)

    def _sample_encoded_actor_data(self, size, include_first):
        encoded = self._encoded_actor_fifo.sample_by_strategy(
            int(size),
            self.device,
            strategy=self.subsampling_strategy,
            include_first=include_first,
            candidate_multiplier=self.nystrom_candidate_multiplier,
            cholesky_tolerance=self.nystrom_cholesky_tolerance,
            kernel_type=self.kernel_type,
            kernel_bandwidth=self.kernel_bandwidth,
            cholesky_progress=self.nystrom_cholesky_progress,
        )
        if self.subsampling_strategy == "pivoted_cholesky" and self.kernel_type == "gaussian":
            bandwidth = self._encoded_actor_fifo.last_pivoted_cholesky_bandwidth
            self.kernel_fn.bandwidth = bandwidth
            self.distribution_matcher.kernel_fn.bandwidth = bandwidth
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
                source=(
                    f"encoded FIFO {self.subsampling_strategy} sample of "
                    f"batch_size_actor={self.batch_size_actor}"
                ),
            )

        # Nyström uses the whole encoded FIFO as support and a smaller landmark set.
        count = self._nystrom_subsample_count()
        full, rewards = self._all_encoded_actor_data(include_first=True)
        if self.debug_manager is not None and self.debug_manager.nystrom_synthetic_subsamples:
            subsample, subsample_rewards = self.debug_manager.synthetic_encoded_subsample()
            subsample_source = "fixed PointMaze Nyström landmarks"
        else:
            subsample, subsample_rewards = self._sample_encoded_actor_data(
                count,
                include_first=True,
            )
            subsample_source = (
                f"encoded FIFO {self.subsampling_strategy} Nyström sample "
                f"of subsamples={count}"
            )
        return EncodedActorUpdateData(
            full=full,
            rewards=rewards,
            subsample=subsample,
            subsample_rewards=subsample_rewards,
            source=f"encoded FIFO full support + {subsample_source}",
        )

    def _replay_actor_subsample_batch(self, replay_iter, full_batch, replay_buffer):
        count = self._nystrom_subsample_count()
        if self.debug_manager is not None and self.debug_manager.nystrom_synthetic_subsamples:
            return self.debug_manager.synthetic_raw_subsample()
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
                if self.debug_manager is not None and self.debug_manager.nystrom_synthetic_subsamples
                else f"replay full support + replay Nyström subsample of subsamples={self.subsamples}"
            ),
        )

    def _get_actor_update_data(self, replay_iter, obs, action, next_obs, reward, replay_buffer=None):
        """Choose the actor dataset for this step.

        Priority is explicit: fixed debug dataset, encoded FIFO, then raw
        replay. If subsamples is None, the object carries only the full actor
        batch and update_actor is used. Otherwise it also carries Nyström data.
        """
        if self.debug_manager is not None and self.debug_manager.debug_fixed_dataset_updates:
            return self.debug_manager.fixed_actor_update_data()

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

        if isinstance(actor_data, EncodedActorUpdateData):
            metrics = self.update_actor_nystrom(
                None,
                None,
                None,
                step=step,
                rewards=actor_data.rewards,
                sub_rewards=actor_data.subsample_rewards,
                encoded_full=actor_data.full,
                encoded_sub=actor_data.subsample,
            )

        else:
            obs, action, next_obs, reward = actor_data.full
            sub_obs = sub_action = sub_next_obs = sub_reward = None
            if actor_data.subsample is not None:
                sub_obs, sub_action, sub_next_obs, sub_reward = actor_data.subsample
            metrics = self.update_actor_nystrom(
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
        if self.debug_manager is not None:
            self.debug_manager.actor_data_updated(actor_data, step)
        return metrics

    def aug_and_encode(self, obs, project=False):
        obs = self.aug(obs)
        if not self.embeddings:
            return self.encoder(obs)
        if project:
            return self.encoder.encode_and_project(obs)
        else:
            return self.encoder(obs)

    def _run_debug_manager(self, metrics, step):
        if self.debug_manager is not None:
            return self.debug_manager.update(metrics, step)
        return metrics

    def update(self, replay_iter, step, replay_buffer=None):
        metrics = dict()

        if step % self.update_every_steps != 0 and self._is_T_sufficiently_initialized(step) is True:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)
        if self.debug_manager is not None and self.debug_manager.debug_fixed_dataset_updates:
            obs, action, next_obs, reward = self.debug_manager.fixed_encoder_batch()

        metrics['batch_reward'] = reward.mean().item()
        if self.embeddings:
            metrics.update(self.update_encoders(obs, action, next_obs, reward))

        # Train the encoder/transition model first; PMD starts once T is ready.
        if not self._is_T_sufficiently_initialized(step):
            metrics['actor_loss'] = 100.0  # dummy value #TODO check if this is needed
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
            metrics = self._run_debug_manager(metrics, step)

        return metrics
