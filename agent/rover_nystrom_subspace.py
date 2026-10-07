"""ROVER Nyström agent with a separate state-space coverage geometry."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
import utils
from agent.utils import pairwise_squared_distance_torch
from agent.rover_nystrom_debug import RoverAgent
from agent.rover_networks import Encoder, FrozenCNNFeatureEncoder, ProjectSA
from agent.rover_subspace_matchers import (
    SubspaceCoverageMatcher,
    normalize_coverage_state_filter,
)


class RoverSubspaceCoverageAgent(RoverAgent):
    """Keep full-state dynamics while measuring occupancy in selected features."""

    def __init__(
        self,
        *args,
        coverage_state_filter=None,
        coverage_kernel_bandwidth=None,
        coverage_kernel_bandwidth_mult=None,
        coverage_embeddings: bool = False,
        coverage_encoder_type: str = "raw",
        coverage_frame_index: int = -1,
        linear_projection: bool = False,
        hidden_dim: int = 1024,
        lr_encoder: float = 1e-4,
        total_train_steps: int = 1,
        **kwargs,
    ):
        super().__init__(
            *args,
            linear_projection=linear_projection,
            hidden_dim=hidden_dim,
            lr_encoder=lr_encoder,
            total_train_steps=total_train_steps,
            **kwargs,
        )
        if not self.embeddings:
            raise ValueError(
                "Subspace ROVER requires normalized learned features for its sink model; "
                "set embeddings=true"
            )
        if self.whiten_representations:
            raise ValueError(
                "Subspace ROVER's unit-mass sink features are incompatible with "
                "whiten_representations; disable whitening"
            )
        self.coverage_state_filter = coverage_state_filter
        self.coverage_embeddings = bool(coverage_embeddings)
        self.coverage_encoder_type = str(coverage_encoder_type).strip().lower()
        if self.coverage_encoder_type not in ("raw", "frozen_cnn"):
            raise ValueError(
                "coverage_encoder_type must be 'raw' or 'frozen_cnn'"
            )

        self.coverage_feature_encoder = None
        if self.coverage_encoder_type == "frozen_cnn":
            if self.obs_type != "pixels":
                raise ValueError("coverage_encoder_type='frozen_cnn' requires pixel observations")
            if self.coverage_embeddings:
                raise ValueError(
                    "coverage_embeddings cannot be enabled with the frozen CNN "
                    "coverage encoder"
                )
            if coverage_state_filter is not None:
                raise ValueError(
                    "coverage_state_filter applies to raw state features; set it to null "
                    "when using coverage_encoder_type='frozen_cnn'"
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
            if self.obs_type == "pixels":
                raise ValueError(
                    "Pixel coverage requires coverage_encoder_type='frozen_cnn' "
                    "to define a compact spatial feature map"
                )
            raw_state_dim = math.prod(getattr(self, "obs_shape", (self.obs_dim,)))
            self.coverage_state_indices = normalize_coverage_state_filter(
                coverage_state_filter, raw_state_dim
            )
            self.coverage_frame_index = int(coverage_frame_index)

        if self.coverage_embeddings and not self.embeddings:
            raise ValueError(
                "coverage_embeddings=True requires embeddings=True so its transition "
                "features can be trained"
            )
        if coverage_kernel_bandwidth is None:
            self.coverage_kernel_bandwidth = None
        else:
            coverage_kernel_bandwidth = float(coverage_kernel_bandwidth)
            if not math.isfinite(coverage_kernel_bandwidth) or coverage_kernel_bandwidth <= 0:
                raise ValueError("coverage_kernel_bandwidth must be positive when set")
            self.coverage_kernel_bandwidth = coverage_kernel_bandwidth
        if coverage_kernel_bandwidth_mult is None:
            self.coverage_kernel_bandwidth_mult = None
        else:
            coverage_kernel_bandwidth_mult = float(coverage_kernel_bandwidth_mult)
            if not math.isfinite(coverage_kernel_bandwidth_mult) or coverage_kernel_bandwidth_mult <= 0:
                raise ValueError("coverage_kernel_bandwidth_mult must be positive when set")
            if self.coverage_kernel_bandwidth is not None:
                raise ValueError(
                    "Set only one of coverage_kernel_bandwidth and "
                    "coverage_kernel_bandwidth_mult"
                )
            self.coverage_kernel_bandwidth_mult = coverage_kernel_bandwidth_mult

        self.coverage_kernel_fn = utils.build_kernel_fn(
            self.kernel_type,
            bandwidth=self.coverage_kernel_bandwidth,
        )
        self.coverage_matcher = SubspaceCoverageMatcher(
            gamma=self.discount,
            kernel_fn=self.coverage_kernel_fn,
        )
        self._raw_coverage_sub_next: Optional[torch.Tensor] = None
        self._coverage_sub_next_features: Optional[torch.Tensor] = None

        if self.coverage_embeddings:
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

    @staticmethod
    def _unit_mass_operator_features(features: torch.Tensor) -> torch.Tensor:
        """Map nonnegative state embeddings to unit-mass ROVER features."""
        features = features.clamp_min(0.0)
        mass = features.sum(dim=-1, keepdim=True)
        normalized = features / mass.clamp_min(1e-12)
        fallback = torch.full_like(features, 1.0 / features.shape[-1])
        return torch.where(mass > 1e-12, normalized, fallback)

    def _encode_with_module(self, module, obs, project=False):
        features = super()._encode_with_module(module, obs, project=project)
        if project and self.embeddings:
            # The subspace sink derivation requires unit-mass dynamics features.
            return self._unit_mass_operator_features(features)
        return features

    def _selected_coverage_observations(self, observations: torch.Tensor) -> torch.Tensor:
        if getattr(self, "coverage_encoder_type", "raw") == "frozen_cnn":
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
                "Coverage embeddings expect raw observations matching obs_shape; "
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
        super()._cache_features(
            obs, action, next_obs, encoder=encoder,
            sub_obs=sub_obs, sub_action=sub_action, sub_next_obs=sub_next_obs,
        )
        raw_sub_next = next_obs if sub_next_obs is None else sub_next_obs
        self._raw_coverage_sub_next = self._selected_coverage_observations(raw_sub_next)
        self._coverage_sub_next_features = None

    def _encode_actor_transition_batch(self, transitions):
        encoded = super()._encode_actor_transition_batch(transitions)
        if getattr(self, "coverage_encoder_type", "raw") == "frozen_cnn":
            raw_next = torch.as_tensor(transitions[4], device=self.device)
            selected_frame = self._selected_coverage_observations(raw_next)
            with torch.no_grad():
                coverage_features = self.coverage_feature_encoder(selected_frame)
            encoded["coverage_next_features"] = coverage_features.detach().to(
                device="cpu", dtype=torch.float32
            )
        else:
            raw_next = torch.as_tensor(transitions[4], device=self.device)
            encoded["coverage_next_raw"] = self._selected_coverage_observations(
                raw_next
            ).detach().to(device="cpu", dtype=torch.float32)
        return encoded

    def _cache_encoded_features(self, encoded_full, encoded_sub=None):
        super()._cache_encoded_features(encoded_full, encoded_sub=encoded_sub)
        coverage_sub = encoded_sub if encoded_sub is not None else encoded_full
        if getattr(self, "coverage_encoder_type", "raw") == "frozen_cnn":
            if "coverage_next_features" not in coverage_sub:
                raise ValueError(
                    "Encoded pixel actor batch is missing frozen CNN coverage features"
                )
            self._raw_coverage_sub_next = None
            self._coverage_sub_next_features = coverage_sub[
                "coverage_next_features"
            ].to(device=self.device, dtype=self.compute_dtype)
            return

        if "coverage_next_raw" not in coverage_sub:
            raise ValueError(
                "Encoded PointMaze actor batch is missing raw coverage coordinates"
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
            if getattr(self, "coverage_encoder_type", "raw") == "frozen_cnn":
                self.coverage_feature_encoder.eval()
                features = self.coverage_feature_encoder(raw)
            elif self.coverage_embeddings:
                self.coverage_encoder.eval()
                features = self.coverage_encoder.encode_and_project(
                    raw, normalize=self.feature_learning_loss != "leworld"
                )
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

    def update_encoders(self, obs, action, next_obs, reward):
        metrics = super().update_encoders(obs, action, next_obs, reward)
        if not self.coverage_embeddings:
            return metrics

        coverage_obs = self._selected_coverage_observations(obs)
        coverage_next = self._selected_coverage_observations(next_obs)
        z_obs = self.coverage_encoder.encode_and_project(
            coverage_obs, normalize=self.feature_learning_loss != "leworld"
        )
        z_next = self.coverage_encoder.encode_and_project(
            coverage_next, normalize=self.feature_learning_loss != "leworld"
        )
        prediction = self.coverage_project_sa(
            self._encode_coverage_state_action(z_obs, action)
        )
        zero = z_obs.new_zeros(())
        if self.feature_learning_loss == "infonce":
            targets = z_next.detach()
            if self.mode == "l1":
                targets = F.normalize(targets, p=2, dim=1, eps=1e-10)
            predicted = F.normalize(prediction, p=2, dim=1, eps=1e-10)
            logits = predicted @ targets.T
            logits = logits - logits.max(dim=1, keepdim=True).values
            if self.infonce_positive_mode == "exact_next_obs":
                group_ids = self._exact_observation_group_ids(coverage_next)
                loss = self._multi_positive_infonce(logits, group_ids)
            else:
                labels = torch.arange(logits.shape[0], device=logits.device)
                loss = self.cross_entropy_loss(logits, labels)
            if self.embedding_sum_loss > 0:
                loss = loss + self.embedding_sum_loss * (
                    z_next.sum(dim=1).sub(1.0).square().mean()
                )
        else:
            normalized_prediction = F.normalize(prediction, p=2, dim=1, eps=1e-10)
            normalized_target = F.normalize(z_next, p=2, dim=1, eps=1e-10)
            prediction_loss = F.mse_loss(normalized_prediction, normalized_target)
            sigreg_loss = 0.5 * (self._sigreg(z_obs) + self._sigreg(z_next))
            loss = prediction_loss + self.leworld_sigreg_weight * sigreg_loss

        self.coverage_encoder_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.coverage_encoder_optimizer.step()
        self.coverage_encoder_scheduler.step()
        metrics["coverage_transition_loss"] = float(loss.detach().item())
        return metrics

    def _coverage_uses_legacy_path(self) -> bool:
        """Preserve exact old objective when coverage spans same geometry/kernel."""
        if getattr(self, "coverage_encoder_type", "raw") != "raw":
            return False
        raw_state_dim = math.prod(getattr(self, "obs_shape", (self.obs_dim,)))
        covers_all_features = self.coverage_state_indices == tuple(range(raw_state_dim))
        if getattr(self, "embeddings", False) or getattr(self, "coverage_embeddings", False):
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
            self.coverage_kernel_bandwidth is None
            and getattr(self, "coverage_kernel_bandwidth_mult", None) is None
        ):
            self.coverage_kernel_fn.bandwidth = self.kernel_fn.bandwidth

    def _fit_state_kernel_bandwidth(self, X, Y) -> None:
        if self._coverage_sub_next_features is None:
            self._prepare_coverage_features()
        super()._fit_state_kernel_bandwidth(X, Y)
        if self.coverage_kernel_bandwidth is None and self.coverage_kernel_bandwidth_mult is not None:
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
