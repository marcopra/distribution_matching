"""Coverage feature selection and Gram construction for streamed ROVER."""

from __future__ import annotations

from numbers import Integral

import torch

from agent.rover_sink import kernel_sink_gram


def normalize_coverage_state_indices(state_indices, state_dim: int) -> tuple[int, ...]:
    """Resolve ``None``, prefix length, or explicit state-coordinate indices."""
    state_dim = int(state_dim)
    if state_dim < 1:
        raise ValueError(f"coverage state dimension must be positive, got {state_dim}")

    if state_indices is None:
        indices = tuple(range(state_dim))
    elif isinstance(state_indices, Integral) and not isinstance(state_indices, bool):
        count = int(state_indices)
        if count < 1 or count > state_dim:
            raise ValueError(
                f"coverage_state_indices prefix length must be in [1, {state_dim}], got {count}"
            )
        indices = tuple(range(count))
    else:
        if isinstance(state_indices, (str, bytes)):
            raise TypeError("coverage_state_indices must be null, an integer, or a list of indices")
        try:
            raw_indices = list(state_indices)
        except TypeError as exc:
            raise TypeError(
                "coverage_state_indices must be null, an integer, or a list of indices"
            ) from exc
        if not raw_indices:
            raise ValueError("coverage_state_indices list must not be empty")
        if any(not isinstance(index, Integral) or isinstance(index, bool) for index in raw_indices):
            raise TypeError("coverage_state_indices must contain integers")
        indices = tuple(int(index) for index in raw_indices)
        if len(set(indices)) != len(indices):
            raise ValueError("coverage_state_indices must be unique")
        invalid = [index for index in indices if index < 0 or index >= state_dim]
        if invalid:
            raise ValueError(
                f"coverage_state_indices must be in [0, {state_dim - 1}], got {invalid}"
            )
    return indices


class SubspaceCoverageMatcher:
    """Build coverage Gram matrices in the selected state or image features."""

    def __init__(self, gamma: float, kernel_fn):
        self.gamma = float(gamma)
        self.kernel_fn = kernel_fn

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
                    "coverage_state_indices exceeds state feature dimension: "
                    f"max index {max(state_indices)}, dimension {state_features.shape[1]}"
                )
            index = torch.as_tensor(state_indices, device=state_features.device, dtype=torch.long)
            projected = state_features.index_select(1, index)
        else:
            projected = coverage_features.to(
                device=phi_sub_next_obs.device, dtype=phi_sub_next_obs.dtype
            )
        return projected

    def coverage_gram(self, *, phi_sub_next_obs, sink_coefficients, sink_norm,
                      state_indices, coverage_features=None):
        state_features = self._coverage_state_features(
            phi_sub_next_obs, state_indices, coverage_features,
        )
        assert torch.allclose(sink_coefficients, torch.ones_like(sink_coefficients),
                              atol=1e-5, rtol=1e-5), "ROVER sink augmentation requires unit-mass dynamics features"
        return kernel_sink_gram(self.kernel_fn, state_features, sink_coefficients, sink_norm)
