"""ROVER Nyström agent with separate dynamics and coverage representations."""

from __future__ import annotations

import copy
import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import utils
from agent.utils import pairwise_squared_distance_torch
from agent.rover_nystrom_debug import RoverAgent
from agent.rover_networks import Encoder, FrozenCNNFeatureEncoder, ProjectSA
from agent.rover_subspace_matchers import (
    SubspaceCoverageMatcher,
    normalize_coverage_state_indices,
)


class _SimplexStateKernel:
    """Apply the configured kernel to state decoded from simplex features."""

    def __init__(self, base_kernel, state_dim):
        self.base_kernel = base_kernel
        self.state_dim = int(state_dim)

    @property
    def bandwidth(self):
        return self.base_kernel.bandwidth

    @bandwidth.setter
    def bandwidth(self, value):
        self.base_kernel.bandwidth = value

    @property
    def kernel_type(self):
        return self.base_kernel.kernel_type

    def _decode(self, features):
        # Actor support vectors carry state_dim+1 simplex values plus an
        # optional orthogonal sink column. The final simplex component is the
        # reference logit, so this exactly recovers the original state vector.
        simplex = features[..., : self.state_dim + 1].clamp_min(
            torch.finfo(features.dtype).tiny
        )
        return torch.log(simplex[..., : self.state_dim] / simplex[..., -1:])

    def fit_bandwidth(self, X, Y):
        return self.base_kernel.fit_bandwidth(self._decode(X), self._decode(Y))

    def __call__(self, X, Y):
        return self.base_kernel(self._decode(X), self._decode(Y))


class RoverSubspaceCoverageAgent(RoverAgent):
    """Use separate, configurable representations for dynamics and coverage."""

    def __init__(
        self,
        *args,
        coverage_state_indices=None,
        coverage_kernel_bandwidth=None,
        coverage_kernel_bandwidth_mult=None,
        train_coverage_encoder: bool = False,
        coverage_kernel_input: Optional[str] = None,
        coverage_frame_index: int = -1,
        dynamics_kernel_input: str = "states",
        dynamics_encoder_mode: str = "raw",
        dynamics_kernel_type: Optional[str] = None,
        coverage_kernel_type: Optional[str] = None,
        linear_projection: bool = False,
        hidden_dim: int = 1024,
        lr_encoder: float = 1e-4,
        total_train_steps: int = 1,
        **kwargs,
    ):
        # Subspace experiments freeze the selected dynamics encoder at
        # initialization. Do not allow the base agent's delayed freeze option.
        if "freeze_encoder" in kwargs:
            old_freeze_encoder = kwargs.pop("freeze_encoder")
            if old_freeze_encoder:
                raise ValueError(
                    "freeze_encoder was renamed; use "
                    "dynamics_encoder_mode='frozen' for subspace experiments"
                )
        freeze_observation_encoder_after_warmup = kwargs.pop(
            "freeze_observation_encoder_after_warmup", False
        )
        if freeze_observation_encoder_after_warmup:
            raise ValueError(
                "freeze_observation_encoder_after_warmup is unsupported for "
                "subspace experiments; set "
                "dynamics_encoder_mode='frozen' to freeze the selected dynamics "
                "encoder from initialization"
            )
        kwargs["freeze_observation_encoder_after_warmup"] = False
        if dynamics_kernel_type is not None:
            kwargs["kernel_type"] = dynamics_kernel_type
        super().__init__(
            *args,
            linear_projection=linear_projection,
            hidden_dim=hidden_dim,
            lr_encoder=lr_encoder,
            total_train_steps=total_train_steps,
            **kwargs,
        )
        self.dynamics_kernel_type = self.kernel_fn.kernel_type
        self.kernel_type = self.dynamics_kernel_type
        self.distribution_matcher.kernel_type = self.dynamics_kernel_type
        if not self.embeddings:
            raise ValueError(
                "Subspace ROVER requires normalized learned features for its sink model; "
                "set embeddings=true"
            )
        self.dynamics_kernel_input = str(dynamics_kernel_input).strip().lower()
        self.dynamics_encoder_mode = str(dynamics_encoder_mode).strip().lower()
        if self.dynamics_kernel_input not in ("pixels", "states"):
            raise ValueError(
                "dynamics_kernel_input must be 'pixels' or 'states'"
            )
        if self.dynamics_encoder_mode not in ("raw", "trainable", "frozen"):
            raise ValueError(
                "dynamics_encoder_mode must be 'raw', 'trainable', or 'frozen'; "
                "use 'trainable' or 'frozen' for pixel dynamics"
            )
        if (
            self.dynamics_kernel_input == "pixels"
            and self.dynamics_encoder_mode == "raw"
        ):
            raise ValueError(
                "Pixel dynamics require the CNN encoder; set "
                "dynamics_encoder_mode='trainable' or 'frozen'"
            )
        if self.dynamics_kernel_input == "pixels" and self.obs_type != "pixels":
            raise ValueError(
                "dynamics_kernel_input='pixels' requires pixel observations"
            )
        if coverage_kernel_input is None:
            coverage_kernel_input = "states"
        if coverage_kernel_type is None:
            coverage_kernel_type = self.dynamics_kernel_type
        self.coverage_kernel_type = str(coverage_kernel_type).strip().lower()
        coverage_kernel_input = str(coverage_kernel_input).strip().lower()
        if self.obs_type == "pixels" and (
            self.dynamics_kernel_input == "states"
            or coverage_kernel_input == "states"
        ) and self.proprio_state_dims != 4:
            raise ValueError(
                "PointMaze state kernels with pixel observations require all four "
                "proprioceptive values (x, y, vx, vy); set "
                "proprio_state_dims=4"
            )
        self.dynamics_state_dim = (
            self.proprio_state_dims
            if self.obs_type == "pixels"
            else math.prod(getattr(self, "obs_shape", (self.obs_dim,)))
        )
        self.nystrom_kernel_feature_key = (
            ("phi_obs" if self.dynamics_encoder_mode != "raw" else "dynamics_state")
            if self.dynamics_kernel_input == "states"
            else None
        )
        if self.dynamics_kernel_input == "states" and self.dynamics_state_dim < 2:
            raise ValueError(
                "dynamics_kernel_input='states' requires at least two state values"
            )
        self.uses_state_dynamics_encoder = (
            self.dynamics_kernel_input == "states"
            and self.dynamics_encoder_mode != "raw"
        )
        if self.dynamics_kernel_input == "states" and not self.uses_state_dynamics_encoder:
            state_kernel = _SimplexStateKernel(self.kernel_fn, self.dynamics_state_dim)
            self.kernel_fn = state_kernel
            self.distribution_matcher.kernel_fn = state_kernel
            self.distribution_matcher.state_kernel_fn = state_kernel
        self.dynamics_state_encoder = None
        self.dynamics_state_encoder_optimizer = None
        self.dynamics_state_encoder_scheduler = None
        if self.uses_state_dynamics_encoder:
            # For state observations, reuse Rover's observation encoder. The
            # legacy PointMaze run learned this encoder directly; keeping the
            # same module also preserves its initialization and optimizer path.
            reuse_observation_encoder = self.obs_type != "pixels"
            if reuse_observation_encoder:
                self.dynamics_state_encoder = self.encoder
                self.policy_dynamics_state_encoder = self.policy_encoder
            else:
                self.dynamics_state_encoder = Encoder(
                    (self.dynamics_state_dim,), hidden_dim, self.feature_dim
                ).to(self.device)
                self.policy_dynamics_state_encoder = copy.deepcopy(
                    self.dynamics_state_encoder
                ).to(self.device)
                self._freeze_module(self.policy_dynamics_state_encoder)
            if self.dynamics_encoder_mode == "trainable":
                if reuse_observation_encoder:
                    self.dynamics_state_encoder_optimizer = self.encoder_optimizer
                    self.dynamics_state_encoder_scheduler = self.encoder_scheduler
                else:
                    self.dynamics_state_encoder_optimizer = torch.optim.AdamW(
                        self.dynamics_state_encoder.parameters(),
                        lr=lr_encoder,
                        weight_decay=1e-5,
                    )
                    self.dynamics_state_encoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.dynamics_state_encoder_optimizer,
                        T_max=max(int(total_train_steps), 1),
                        eta_min=lr_encoder * 0.1,
                    )
            else:
                self._freeze_module(self.policy_dynamics_state_encoder)
                self._freeze_module(self.dynamics_state_encoder)
        self.train_coverage_encoder = bool(train_coverage_encoder)
        self.coverage_kernel_input = coverage_kernel_input
        if self.coverage_kernel_input not in ("states", "pixels"):
            raise ValueError(
                "coverage_kernel_input must be 'states' or 'pixels'"
            )

        self.coverage_feature_encoder = None
        if self.coverage_kernel_input == "pixels":
            if self.obs_type != "pixels":
                raise ValueError("coverage_kernel_input='pixels' requires pixel observations")
            if self.train_coverage_encoder:
                raise ValueError(
                    "train_coverage_encoder is incompatible with pixel coverage; "
                    "pixel coverage uses a frozen CNN"
                )
            if coverage_state_indices is not None:
                raise ValueError(
                    "coverage_state_indices applies to raw state features; set it "
                    "to null when using coverage_kernel_input='pixels'"
                )
            if len(self.obs_shape) != 3:
                raise ValueError(
                    "Frozen CNN coverage requires pixel observations with shape [C, H, W]"
                )
            base_channels = int(self.image_channels)
            stacked_channels, image_height, image_width = map(int, self.obs_shape)
            if stacked_channels % base_channels != 0:
                raise ValueError(
                    "Pixel observation channels must be divisible by image_channels "
                    f"({stacked_channels} vs {base_channels})"
                )
            frame_count = stacked_channels // base_channels
            frame_index = int(coverage_frame_index)
            if frame_index < 0:
                frame_index += frame_count
            if frame_index < 0 or frame_index >= frame_count:
                raise ValueError(
                    f"coverage_frame_index must select one of {frame_count} frames, "
                    f"got {coverage_frame_index}"
                )
            self.coverage_frame_index = frame_index
            self.coverage_feature_encoder = FrozenCNNFeatureEncoder(
                (base_channels, image_height, image_width),
                feature_dim=self.feature_dim,
            ).to(device=self.device, dtype=torch.float32)
            self._freeze_module(self.coverage_feature_encoder)
            self.coverage_state_indices = ()
        else:
            if self.obs_type == "pixels" and self.train_coverage_encoder:
                raise ValueError(
                    "train_coverage_encoder with pixel observations and state "
                    "coverage is unsupported; use raw state coordinates"
                )
            raw_state_dim = (
                self.proprio_state_dims
                if self.obs_type == "pixels"
                else math.prod(getattr(self, "obs_shape", (self.obs_dim,)))
            )
            if raw_state_dim < 1:
                raise ValueError(
                    "State coverage requires proprioceptive state for pixels or "
                    "state observations"
                )
            self.coverage_state_indices = normalize_coverage_state_indices(
                coverage_state_indices, raw_state_dim
            )
            self.coverage_frame_index = int(coverage_frame_index)

        if self.train_coverage_encoder and not self.embeddings:
            raise ValueError(
                "train_coverage_encoder=True requires embeddings=True so its transition "
                "features can be trained"
            )
        if coverage_kernel_bandwidth is None:
            self.coverage_kernel_bandwidth = None
        else:
            coverage_kernel_bandwidth = float(coverage_kernel_bandwidth)
            if (
                not math.isfinite(coverage_kernel_bandwidth)
                or coverage_kernel_bandwidth <= 0
            ):
                raise ValueError("coverage_kernel_bandwidth must be positive when set")
            self.coverage_kernel_bandwidth = coverage_kernel_bandwidth
        if coverage_kernel_bandwidth_mult is None:
            self.coverage_kernel_bandwidth_mult = None
        else:
            coverage_kernel_bandwidth_mult = float(coverage_kernel_bandwidth_mult)
            if (
                not math.isfinite(coverage_kernel_bandwidth_mult)
                or coverage_kernel_bandwidth_mult <= 0
            ):
                raise ValueError("coverage_kernel_bandwidth_mult must be positive when set")
            if self.coverage_kernel_bandwidth is not None:
                raise ValueError(
                    "Set only one of coverage_kernel_bandwidth and "
                    "coverage_kernel_bandwidth_mult"
                )
            self.coverage_kernel_bandwidth_mult = coverage_kernel_bandwidth_mult

        self.coverage_kernel_fn = utils.build_kernel_fn(
            self.coverage_kernel_type,
            bandwidth=self.coverage_kernel_bandwidth,
        )
        self.coverage_kernel_type = self.coverage_kernel_fn.kernel_type
        self.coverage_matcher = SubspaceCoverageMatcher(
            gamma=self.discount,
            kernel_fn=self.coverage_kernel_fn,
        )
        self._raw_coverage_sub_next: Optional[torch.Tensor] = None
        self._coverage_sub_next_features: Optional[torch.Tensor] = None

        if self.train_coverage_encoder:
            selected_dim = len(self.coverage_state_indices)
            self.coverage_encoder = Encoder(
                (selected_dim,), hidden_dim, self.feature_dim
            ).to(self.device)
            self.coverage_project_sa = ProjectSA(
                self.feature_dim * self.n_actions,
                hidden_dim,
                self.feature_dim,
                linear=bool(linear_projection),
            ).to(self.device)
            self.coverage_encoder_optimizer = torch.optim.AdamW(
                list(self.coverage_encoder.parameters())
                + list(self.coverage_project_sa.parameters()),
                lr=lr_encoder,
                weight_decay=1e-5,
            )
            self.coverage_encoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.coverage_encoder_optimizer,
                T_max=max(int(total_train_steps), 1),
                eta_min=lr_encoder * 0.1,
            )

        # Base warm-up freezing is disabled here. Subspace mode='frozen' locks
        # selected image CNN from initialization.
        self.freeze_observation_encoder_after_warmup = False
        self._encoder_frozen = (
            self.dynamics_kernel_input == "pixels"
            and self.dynamics_encoder_mode == "frozen"
        )
        if self._encoder_frozen:
            self._freeze_module(self.encoder)

    @staticmethod
    def _unit_mass_operator_features(features: torch.Tensor) -> torch.Tensor:
        """Map nonnegative state embeddings to unit-mass ROVER features."""
        features = features.clamp_min(0.0)
        mass = features.sum(dim=-1, keepdim=True)
        normalized = features / mass.clamp_min(1e-12)
        fallback = torch.full_like(features, 1.0 / features.shape[-1])
        return torch.where(mass > 1e-12, normalized, fallback)

    def _state_dynamics_features(self, states: torch.Tensor) -> torch.Tensor:
        """Embed the full state bijectively in the simplex required by ROVER."""
        states = states.reshape(states.shape[0], -1)
        if states.shape[1] < self.dynamics_state_dim:
            raise ValueError(
                "State dynamics input is missing values: expected at least "
                f"{self.dynamics_state_dim}, got {states.shape[1]}"
            )
        states = states[:, : self.dynamics_state_dim].to(
            device=self.device, dtype=self.compute_dtype
        )
        if self.uses_state_dynamics_encoder:
            return self.dynamics_state_encoder(states)
        logits = torch.cat([states, torch.zeros_like(states[:, :1])], dim=1)
        return torch.softmax(logits, dim=1)

    def _policy_state_dynamics_features(self, states: torch.Tensor) -> torch.Tensor:
        if not self.uses_state_dynamics_encoder:
            return self._state_dynamics_features(states)
        states = states.reshape(states.shape[0], -1)[:, : self.dynamics_state_dim]
        states = states.to(device=self.device, dtype=self.compute_dtype)
        return self.policy_dynamics_state_encoder(states)

    def train(self, training=True):
        super().train(training)
        if self.uses_state_dynamics_encoder:
            self.dynamics_state_encoder.train(
                training and self.dynamics_encoder_mode == "trainable"
            )
            self.policy_dynamics_state_encoder.eval()

    def _sync_policy_encoder(self) -> None:
        super()._sync_policy_encoder()
        if self.uses_state_dynamics_encoder:
            self.policy_dynamics_state_encoder.load_state_dict(
                self.dynamics_state_encoder.state_dict()
            )
            self._freeze_module(self.policy_dynamics_state_encoder)

    def _policy_query_features(self, obs_tensor, metas=None):
        if self.dynamics_kernel_input != "states":
            return super()._policy_query_features(obs_tensor, metas=metas)
        if self.obs_type != "pixels" and obs_tensor.reshape(
            obs_tensor.shape[0], -1
        ).shape[1] >= self.dynamics_state_dim:
            states = obs_tensor.reshape(obs_tensor.shape[0], -1)
        else:
            if metas is None or len(metas) != obs_tensor.shape[0]:
                raise ValueError(
                    "State dynamics policy queries on pixels require one "
                    "proprio_state entry per observation"
                )
            values = []
            for meta in metas:
                value = (
                    meta.get("proprio_state")
                    if isinstance(meta, dict) else None
                )
                if value is None:
                    raise ValueError(
                        "State dynamics policy query is missing "
                        "proprio_state"
                    )
                values.append(np.asarray(value, dtype=np.float32).reshape(-1))
            states = torch.as_tensor(
                np.stack(values), device=self.device, dtype=self.compute_dtype
            )
        return self._policy_state_dynamics_features(states)

    def _fit_dynamics_kernel_bandwidth(self, X, Y) -> None:
        if self.dynamics_kernel_input != "states" or self.uses_state_dynamics_encoder:
            return super()._fit_state_kernel_bandwidth(X, Y)
        if getattr(self, "_active_kernel_bandwidth", None) is not None:
            self.kernel_fn.bandwidth = self._active_kernel_bandwidth
            self.distribution_matcher.kernel_fn.bandwidth = self._active_kernel_bandwidth
            return
        if (
            self.subsampling_strategy == "pivoted_cholesky"
            and self.kernel_type == "gaussian"
        ):
            bandwidth = getattr(
                getattr(self, "_encoded_actor_fifo", None),
                "last_pivoted_cholesky_bandwidth",
                None,
            )
            if bandwidth is not None:
                self.kernel_fn.bandwidth = bandwidth
                self.distribution_matcher.kernel_fn.bandwidth = bandwidth
                return
        if self.kernel_type != "gaussian" or self.kernel_bandwidth_mult is None:
            return
        multiplier = float(self.kernel_bandwidth_mult)
        if multiplier <= 0.0:
            raise ValueError("kernel_bandwidth_mult must be positive when set.")
        with torch.no_grad():
            states_X = self.kernel_fn._decode(X.detach())
            states_Y = self.kernel_fn._decode(Y.detach())
            if states_X.shape[0] > self.max_distances or states_Y.shape[0] > self.max_distances:
                count = min(states_X.shape[0], states_Y.shape[0], self.max_distances)
                indices_X = torch.randperm(states_X.shape[0], device=states_X.device)[:count]
                indices_Y = torch.randperm(states_Y.shape[0], device=states_Y.device)[:count]
                states_X = states_X[indices_X]
                states_Y = states_Y[indices_Y]
            distances = torch.sqrt(
                torch.clamp(
                    pairwise_squared_distance_torch(states_X, states_Y), min=0.0
                )
            )
            distances = distances[distances > 0]
            median_distance = (
                torch.median(distances)
                if distances.numel()
                else torch.tensor(1.0, device=X.device, dtype=X.dtype)
            )
            bandwidth = max(float(median_distance.item()) * multiplier, 1e-12)
        self.kernel_fn.bandwidth = bandwidth
        self.distribution_matcher.kernel_fn.bandwidth = bandwidth
        utils.ColorPrint.yellow(
            f"Fitted state actor Gaussian bandwidth={bandwidth:.6g} "
            f"(median_distance={float(median_distance.item()):.6g}, "
            f"multiplier={multiplier:.6g})."
        )

    def _encode_with_module(self, module, obs, project=False):
        features = super()._encode_with_module(module, obs, project=project)
        if project and self.embeddings:
            # The subspace sink derivation requires unit-mass dynamics features.
            return self._unit_mass_operator_features(features)
        return features

    def _selected_coverage_observations(self, observations: torch.Tensor) -> torch.Tensor:
        if self.coverage_kernel_input == "states" and self.obs_type == "pixels":
            raise RuntimeError(
                "State coverage features for pixel observations must come from "
                "aligned replay metadata"
            )
        if self.coverage_kernel_input == "pixels":
            if observations.ndim != 4 or tuple(observations.shape[1:]) != tuple(self.obs_shape):
                raise ValueError(
                    "Frozen CNN coverage expects observations matching [B, "
                    f"{', '.join(map(str, self.obs_shape))}], got {tuple(observations.shape)}"
                )
            start = self.coverage_frame_index * self.image_channels
            return observations[:, start : start + self.image_channels]

        flattened = observations.reshape(observations.shape[0], -1)
        if flattened.shape[1] != math.prod(self.obs_shape):
            raise ValueError(
                "State coverage expects raw observations matching obs_shape; "
                f"got {flattened.shape[1]} features, expected {math.prod(self.obs_shape)}"
            )
        indices = torch.as_tensor(
            self.coverage_state_indices, device=flattened.device, dtype=torch.long
        )
        return flattened.index_select(1, indices)

    def _cache_features(
        self, obs, action, next_obs, encoder=None,
        sub_obs=None, sub_action=None, sub_next_obs=None,
    ):
        if (
            self.dynamics_kernel_input == "states" and self.obs_type == "pixels"
        ) or (self.coverage_kernel_input == "states" and self.obs_type == "pixels"):
            raise RuntimeError(
                "State kernel inputs for pixel observations require aligned replay metadata; "
                "use the encoded transition replay path"
            )
        super()._cache_features(
            obs, action, next_obs, encoder=encoder,
            sub_obs=sub_obs, sub_action=sub_action, sub_next_obs=sub_next_obs,
        )
        raw_sub_next = next_obs if sub_next_obs is None else sub_next_obs
        self._raw_coverage_sub_next = self._selected_coverage_observations(raw_sub_next)
        self._coverage_sub_next_features = None

    def _encode_actor_transition_batch(self, transitions):
        encoded = super()._encode_actor_transition_batch(transitions)
        if self.dynamics_kernel_input == "states":
            if self.obs_type == "pixels":
                source_states = encoded.get("proprio_state")
                next_states = encoded.get("proprio_next_state")
            else:
                source_states = torch.as_tensor(
                    transitions[0], device=self.device
                ).reshape(len(transitions[0]), -1)
                next_states = torch.as_tensor(
                    transitions[4], device=self.device
                ).reshape(len(transitions[4]), -1)
            if source_states is None or next_states is None:
                raise ValueError(
                    "State dynamics kernel requires source and next-state states "
                    "in replay metadata"
                )
            with torch.no_grad():
                phi_obs = self._policy_state_dynamics_features(
                    source_states.to(device=self.device)
                )
                phi_next = self._policy_state_dynamics_features(
                    next_states.to(device=self.device)
                )
            encoded["phi_next"] = phi_next.detach().to(
                device="cpu", dtype=self.compute_dtype
            )
            encoded["phi_obs"] = phi_obs.detach().to(
                device="cpu", dtype=self.compute_dtype
            )
            encoded["dynamics_state"] = source_states.detach().to(
                device="cpu", dtype=self.compute_dtype
            )

        if self.coverage_kernel_input == "pixels":
            raw_next = torch.as_tensor(transitions[4], device=self.device)
            selected_frame = self._selected_coverage_observations(raw_next)
            with torch.no_grad():
                coverage_features = self.coverage_feature_encoder(selected_frame)
            encoded["coverage_next_features"] = coverage_features.detach().to(
                device="cpu", dtype=torch.float32
            )
        elif self.obs_type == "pixels":
            next_states = encoded.get("proprio_next_state")
            if next_states is None:
                raise ValueError(
                    "State coverage kernel requires next-state values in replay metadata"
                )
            indices = torch.as_tensor(
                self.coverage_state_indices, device=next_states.device, dtype=torch.long
            )
            encoded["coverage_next_raw"] = next_states.index_select(
                1, indices
            ).detach().to(device="cpu", dtype=torch.float32)
        else:
            raw_next = torch.as_tensor(transitions[4], device=self.device)
            encoded["coverage_next_raw"] = self._selected_coverage_observations(
                raw_next
            ).detach().to(device="cpu", dtype=torch.float32)
        return encoded

    def _cache_encoded_features(self, encoded_full, encoded_sub=None):
        super()._cache_encoded_features(encoded_full, encoded_sub=encoded_sub)
        coverage_sub = encoded_sub if encoded_sub is not None else encoded_full
        if self.coverage_kernel_input == "pixels":
            if "coverage_next_features" not in coverage_sub:
                raise ValueError(
                    "Encoded actor batch is missing configured coverage features"
                )
            self._raw_coverage_sub_next = None
            self._coverage_sub_next_features = coverage_sub[
                "coverage_next_features"
            ].to(device=self.device, dtype=self.compute_dtype)
            return

        if "coverage_next_raw" not in coverage_sub:
            raise ValueError(
                "Encoded actor batch is missing filtered state coverage features"
            )
        self._raw_coverage_sub_next = coverage_sub["coverage_next_raw"].to(
            device=self.device, dtype=self.compute_dtype
        )
        self._coverage_sub_next_features = None

    def _prepare_coverage_features(self) -> None:
        if self._raw_coverage_sub_next is None:
            raise RuntimeError("Raw coverage features were not cached for this actor update")
        with torch.no_grad():
            raw = self._raw_coverage_sub_next.to(device=self.device)
            if self.coverage_kernel_input == "pixels":
                self.coverage_feature_encoder.eval()
                features = self.coverage_feature_encoder(raw)
            elif self.train_coverage_encoder:
                self.coverage_encoder.eval()
                features = self.coverage_encoder.encode_and_project(raw)
            else:
                features = raw
            self._coverage_sub_next_features = features.to(dtype=self.compute_dtype)

    def _encode_coverage_state_action(self, encoded_obs, actions):
        action_onehot = F.one_hot(actions.long().reshape(-1), self.n_actions).to(
            device=encoded_obs.device, dtype=encoded_obs.dtype
        )
        return torch.einsum("bd,ba->bda", encoded_obs, action_onehot).reshape(
            encoded_obs.shape[0], -1
        )

    def update_encoders(
        self, obs, action, next_obs, reward,
        proprio_state=None, proprio_next_state=None,
    ):
        metrics = {}
        if self.uses_state_dynamics_encoder:
            # Match legacy feature_learning_loss=infonce with diagonal positives.
            if self.obs_type == "pixels":
                if proprio_state is None or proprio_next_state is None:
                    raise ValueError(
                        "Learned state dynamics embeddings require source and next "
                        "proprio_state replay metadata"
                    )
                states = proprio_state.reshape(proprio_state.shape[0], -1)
                next_states = proprio_next_state.reshape(proprio_next_state.shape[0], -1)
            else:
                states = obs.reshape(obs.shape[0], -1)
                next_states = next_obs.reshape(next_obs.shape[0], -1)
            states = states[:, : self.dynamics_state_dim].to(
                device=self.device, dtype=self.compute_dtype
            )
            next_states = next_states[:, : self.dynamics_state_dim].to(
                device=self.device, dtype=self.compute_dtype
            )
            z_obs = self.dynamics_state_encoder(states)
            with torch.no_grad():
                z_next = self.dynamics_state_encoder(next_states)
            prediction = self.project_sa(self._encode_state_action(z_obs, action))
            targets = z_next
            if self.mode == "l1":
                targets = F.normalize(targets, p=2, dim=1, eps=1e-10)
            predicted = F.normalize(prediction, p=2, dim=1, eps=1e-10)
            logits = predicted @ targets.T
            logits = logits - logits.max(dim=1, keepdim=True).values
            labels = torch.arange(logits.shape[0], device=logits.device)
            loss = self.cross_entropy_loss(logits, labels)
            if self.embedding_sum_loss > 0:
                loss = loss + self.embedding_sum_loss * (
                    z_next.sum(dim=1).sub(1.0).square().mean()
                )
            if self.dynamics_state_encoder_optimizer is not None:
                self.dynamics_state_encoder_optimizer.zero_grad(set_to_none=True)
            self.transition_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.dynamics_state_encoder_optimizer is not None:
                self.dynamics_state_encoder_optimizer.step()
                self.dynamics_state_encoder_scheduler.step()
                self._policy_is_synced = False
            self.transition_optimizer.step()
            metrics["dynamics_state_transition_loss"] = float(loss.detach().item())
        elif self.dynamics_kernel_input == "states" and not self.train_coverage_encoder:
            # Primary image encoder does not define either active representation.
            return {}
        else:
            metrics.update(super().update_encoders(obs, action, next_obs, reward))
        if not self.train_coverage_encoder:
            return metrics

        coverage_obs = self._selected_coverage_observations(obs)
        coverage_next = self._selected_coverage_observations(next_obs)
        z_obs = self.coverage_encoder.encode_and_project(coverage_obs)
        z_next = self.coverage_encoder.encode_and_project(coverage_next)
        prediction = self.coverage_project_sa(
            self._encode_coverage_state_action(z_obs, action)
        )
        targets = z_next.detach()
        if self.mode == "l1":
            targets = F.normalize(targets, p=2, dim=1, eps=1e-10)
        predicted = F.normalize(prediction, p=2, dim=1, eps=1e-10)
        logits = predicted @ targets.T
        logits = logits - logits.max(dim=1, keepdim=True).values
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = self.cross_entropy_loss(logits, labels)
        if self.embedding_sum_loss > 0:
            loss = loss + self.embedding_sum_loss * (
                z_next.sum(dim=1).sub(1.0).square().mean()
            )

        self.coverage_encoder_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.coverage_encoder_optimizer.step()
        self.coverage_encoder_scheduler.step()
        metrics["coverage_transition_loss"] = float(loss.detach().item())
        return metrics

    def _debug_visualizer_text(self, step) -> str:
        text = super()._debug_visualizer_text(step)
        dynamics_space = (
            f"stacked images through {self.dynamics_encoder_mode} CNN"
            if self.dynamics_kernel_input == "pixels"
            else (
                f"{self.dynamics_encoder_mode} embedding of full state "
                f"({self.dynamics_state_dim} values)"
                if self.uses_state_dynamics_encoder
                else f"full proprioceptive state ({self.dynamics_state_dim} values)"
            )
        )
        coverage_space = (
            f"frozen CNN frame {self.coverage_frame_index}"
            if self.coverage_kernel_input == "pixels"
            else f"raw proprioceptive indices {self.coverage_state_indices}"
        )
        return (
            f"{text}"
            f"dynamics kernel = {self.dynamics_kernel_type}\n"
            f"dynamics kernel space = {dynamics_space}\n"
            f"coverage kernel = {self.coverage_kernel_type}\n"
            f"coverage space = {coverage_space}\n"
        )

    def _coverage_uses_legacy_path(self) -> bool:
        """Preserve exact old objective when coverage spans same geometry/kernel."""
        if self.coverage_kernel_input != "states" or self.obs_type == "pixels":
            return False
        if self.coverage_kernel_type != self.dynamics_kernel_type:
            return False
        raw_state_dim = math.prod(getattr(self, "obs_shape", (self.obs_dim,)))
        covers_all_features = self.coverage_state_indices == tuple(range(raw_state_dim))
        if getattr(self, "embeddings", False) or getattr(self, "train_coverage_encoder", False):
            return False
        if not covers_all_features:
            return False
        if getattr(self, "coverage_kernel_bandwidth_mult", None) is not None:
            return False
        if self.coverage_kernel_bandwidth is None:
            return True
        dynamic_bandwidth = self.kernel_fn.bandwidth
        if dynamic_bandwidth is None:
            dynamic_bandwidth = 1.0
        return self.coverage_kernel_bandwidth == float(dynamic_bandwidth)

    def _sync_coverage_kernel_bandwidth(self) -> None:
        if (
            self.coverage_kernel_type == self.dynamics_kernel_type
            and self.coverage_kernel_bandwidth is None
            and getattr(self, "coverage_kernel_bandwidth_mult", None) is None
        ):
            self.coverage_kernel_fn.bandwidth = self.kernel_fn.bandwidth

    def _fit_state_kernel_bandwidth(self, X, Y) -> None:
        if self._coverage_sub_next_features is None:
            self._prepare_coverage_features()
        self._fit_dynamics_kernel_bandwidth(X, Y)
        if (
            self.coverage_kernel_type == "gaussian"
            and self.coverage_kernel_bandwidth is None
            and self.coverage_kernel_bandwidth_mult is not None
        ):
            self._fit_coverage_kernel_bandwidth(self._coverage_sub_next_features)

    def _fit_coverage_kernel_bandwidth(self, features) -> None:
        """Fit coverage bandwidth on selected state features only."""
        with torch.no_grad():
            if features.shape[0] > self.max_distances:
                indices = torch.randperm(features.shape[0], device=features.device)[: self.max_distances]
                features = features[indices]
            distances = torch.sqrt(
                torch.clamp(
                    pairwise_squared_distance_torch(features.detach(), features.detach()),
                    min=0.0,
                )
            )
            distances = distances[distances > 0]
            median_distance = (
                torch.median(distances)
                if distances.numel()
                else torch.tensor(1.0, device=features.device, dtype=features.dtype)
            )
            self.coverage_kernel_fn.bandwidth = max(
                float(median_distance.item()) * self.coverage_kernel_bandwidth_mult,
                1e-12,
            )

    def _compute_actor_occupancy_loss(self, **kwargs):
        if self._coverage_uses_legacy_path():
            return super()._compute_actor_occupancy_loss(**kwargs)
        self._sync_coverage_kernel_bandwidth()
        return self.coverage_matcher.compute_nystrom_subspace_occupancy_loss(
            phi_sub_next_obs=kwargs["phi_sub_next_obs"],
            psi_sub_obs_action=kwargs["psi_sub_obs_action"],
            H=kwargs["H"],
            pi=kwargs["pi"],
            E=kwargs["E"],
            alpha=kwargs["alpha"],
            sink_norm=kwargs["sink_norm"],
            B_nystrom=kwargs["B_nystrom"],
            state_indices=self.coverage_state_indices,
            coverage_features=getattr(self, "_coverage_sub_next_features", None),
        )

    def _compute_actor_occupancy_gradient(self, **kwargs):
        if self._coverage_uses_legacy_path():
            return super()._compute_actor_occupancy_gradient(**kwargs)
        self._sync_coverage_kernel_bandwidth()
        return self.coverage_matcher.compute_gradient_coefficient_nystrom_subspace(
            phi_sub_next_obs=kwargs["phi_sub_next_obs"],
            psi_sub_obs_action=kwargs["psi_sub_obs_action"],
            H=kwargs["H"],
            pi=kwargs["pi"],
            E=kwargs["E"],
            alpha=kwargs["alpha"],
            sink_norm=kwargs["sink_norm"],
            B_nystrom=kwargs["B_nystrom"],
            eig_vecs_r=kwargs["eig_vecs_r"],
            state_indices=self.coverage_state_indices,
            coverage_features=getattr(self, "_coverage_sub_next_features", None),
        )
