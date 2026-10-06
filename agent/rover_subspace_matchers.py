"""Nyström occupancy and gradient calculations for projected state coverage."""

from __future__ import annotations

from numbers import Integral

import torch

from agent.rover_sink import (
    nystrom_resolvent_coefficients,
    orthogonal_sink_residual_gram,
)


def normalize_coverage_state_filter(state_filter, state_dim: int) -> tuple[int, ...]:
    """Resolve ``None``, prefix length, or explicit indices to feature indices."""
    state_dim = int(state_dim)
    if state_dim < 1:
        raise ValueError(f"coverage state dimension must be positive, got {state_dim}")

    if state_filter is None:
        indices = tuple(range(state_dim))
    elif isinstance(state_filter, Integral) and not isinstance(state_filter, bool):
        count = int(state_filter)
        if count < 1 or count > state_dim:
            raise ValueError(
                f"coverage_state_filter prefix length must be in [1, {state_dim}], got {count}"
            )
        indices = tuple(range(count))
    else:
        if isinstance(state_filter, (str, bytes)):
            raise TypeError("coverage_state_filter must be null, an integer, or a list of indices")
        try:
            raw_indices = list(state_filter)
        except TypeError as exc:
            raise TypeError(
                "coverage_state_filter must be null, an integer, or a list of indices"
            ) from exc
        if not raw_indices:
            raise ValueError("coverage_state_filter index list must not be empty")
        if any(not isinstance(index, Integral) or isinstance(index, bool) for index in raw_indices):
            raise TypeError("coverage_state_filter indices must be integers")
        indices = tuple(int(index) for index in raw_indices)
        if len(set(indices)) != len(indices):
            raise ValueError("coverage_state_filter indices must be unique")
        invalid = [index for index in indices if index < 0 or index >= state_dim]
        if invalid:
            raise ValueError(
                f"coverage_state_filter indices must be in [0, {state_dim - 1}], got {invalid}"
            )
    return indices


class SubspaceCoverageMatcher:
    """Evaluate occupancy in selected features while retaining full-state dynamics."""

    def __init__(self, gamma: float, kernel_fn):
        self.gamma = float(gamma)
        self.kernel_fn = kernel_fn

    @staticmethod
    def _resolvent_coefficients(
        H: torch.Tensor,
        pi: torch.Tensor,
        E: torch.Tensor,
        alpha: torch.Tensor,
        B_nystrom: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        # H and E use full-state transition features. Only the coverage Gram
        # matrix below uses the selected state coordinates.
        return nystrom_resolvent_coefficients(
            H, pi, E, alpha, B_nystrom, gamma
        )

    @staticmethod
    def _coverage_state_features(
        phi_sub_next_obs: torch.Tensor,
        state_indices: tuple[int, ...],
        coverage_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if coverage_features is None:
            state_features = phi_sub_next_obs[:, :-1]
            if state_features.shape[1] <= max(state_indices):
                raise ValueError(
                    "coverage_state_filter exceeds state feature dimension: "
                    f"max index {max(state_indices)}, dimension {state_features.shape[1]}"
                )
            index = torch.as_tensor(state_indices, device=state_features.device, dtype=torch.long)
            projected = state_features.index_select(1, index)
        else:
            projected = coverage_features.to(
                device=phi_sub_next_obs.device, dtype=phi_sub_next_obs.dtype
            )
        return projected

    def _coverage_gram(
        self,
        *,
        phi_sub_next_obs: torch.Tensor,
        psi_sub_obs_action: torch.Tensor,
        sink_norm: float,
        state_indices: tuple[int, ...],
        coverage_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        state_features = self._coverage_state_features(
            phi_sub_next_obs, state_indices, coverage_features
        )
        state_gram = self.kernel_fn(state_features, state_features)
        sink_coefficients = psi_sub_obs_action.sum(dim=1)
        return orthogonal_sink_residual_gram(
            state_gram, sink_coefficients, sink_norm
        )

    def compute_nystrom_subspace_occupancy_loss(
        self,
        *,
        phi_sub_next_obs: torch.Tensor,
        psi_sub_obs_action: torch.Tensor,
        H: torch.Tensor,
        pi: torch.Tensor,
        E: torch.Tensor,
        alpha: torch.Tensor,
        sink_norm: float,
        B_nystrom: torch.Tensor,
        state_indices: tuple[int, ...],
        coverage_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return squared RKHS norm of discounted projected occupancy."""
        coefficients = self._resolvent_coefficients(
            H, pi, E, alpha, B_nystrom, self.gamma
        )
        coverage_gram = self._coverage_gram(
            phi_sub_next_obs=phi_sub_next_obs,
            psi_sub_obs_action=psi_sub_obs_action,
            sink_norm=sink_norm,
            state_indices=state_indices,
            coverage_features=coverage_features,
        )
        occupancy_coefficients = (1.0 - self.gamma) * coefficients
        return (occupancy_coefficients.T @ coverage_gram @ occupancy_coefficients).squeeze()

    def compute_gradient_coefficient_nystrom_subspace(
        self,
        *,
        phi_sub_next_obs: torch.Tensor,
        psi_sub_obs_action: torch.Tensor,
        H: torch.Tensor,
        pi: torch.Tensor,
        E: torch.Tensor,
        alpha: torch.Tensor,
        sink_norm: float,
        B_nystrom: torch.Tensor,
        eig_vecs_r: torch.Tensor,
        state_indices: tuple[int, ...],
        coverage_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return Nyström PMD gradient for occupancy in selected state features."""
        m = phi_sub_next_obs.shape[0]
        r = eig_vecs_r.shape[1]
        eps = 1e-6
        beta = 2.0 * self.gamma * ((1.0 - self.gamma) ** 2)

        M = H * (E @ pi.T)
        BM = B_nystrom @ M
        del M
        BM.mul_(-self.gamma)
        diagonal = torch.arange(m, device=BM.device)
        BM[diagonal, diagonal] += 1.0
        S_r = eig_vecs_r.T @ BM @ eig_vecs_r
        del BM

        S_r_reg = S_r.clone()
        diagonal_r = torch.arange(r, device=S_r_reg.device)
        S_r_reg[diagonal_r, diagonal_r] += eps

        coverage_gram = self._coverage_gram(
            phi_sub_next_obs=phi_sub_next_obs,
            psi_sub_obs_action=psi_sub_obs_action,
            sink_norm=sink_norm,
            state_indices=state_indices,
            coverage_features=coverage_features,
        )
        K_rr = eig_vecs_r.T @ coverage_gram[:-1, :-1] @ eig_vecs_r
        k_re = eig_vecs_r.T @ coverage_gram[:-1, -1:]
        k_er_T = coverage_gram[-1:, :-1] @ eig_vecs_r
        k_ee = coverage_gram[-1:, -1:]
        del coverage_gram

        alpha_r = eig_vecs_r.T @ alpha
        alpha_e = torch.ones((1, 1), device=alpha.device, dtype=alpha.dtype)
        z_r = torch.linalg.solve(S_r, alpha_r)
        del S_r, alpha_r
        z_e = alpha_e / (1.0 - self.gamma)

        h_r = K_rr @ z_r
        del K_rr
        h_r.add_(k_re * z_e)
        del k_re
        h_e = k_er_T @ z_r
        del k_er_T, z_r
        h_e.add_(k_ee * z_e)
        del k_ee, z_e

        tmp_main = torch.linalg.solve(S_r_reg.T, h_r)
        del S_r_reg, h_r
        tmp_sink = h_e / (1.0 - self.gamma + eps)
        del h_e

        gradient = torch.empty(
            (B_nystrom.shape[1] + 1, 1),
            device=B_nystrom.device,
            dtype=B_nystrom.dtype,
        )
        gradient[:-1] = B_nystrom.T @ (eig_vecs_r @ tmp_main)
        gradient[-1:] = tmp_sink
        gradient.mul_(beta)
        return gradient
