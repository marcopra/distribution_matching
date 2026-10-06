"""Shared algebra for ROVER's orthogonal sink-state augmentation."""

from __future__ import annotations

import torch


def nystrom_resolvent_coefficients(
    H: torch.Tensor,
    pi: torch.Tensor,
    E: torch.Tensor,
    alpha: torch.Tensor,
    B_nystrom: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Solve the Nyström resolvent, including its absorbing sink coefficient."""
    M = H * (E @ pi.T)
    BM = B_nystrom @ M
    system = torch.eye(BM.shape[0], device=BM.device, dtype=BM.dtype)
    system = system - float(gamma) * BM

    augmented = torch.zeros(
        (system.shape[0] + 1, system.shape[1] + 1),
        device=system.device,
        dtype=system.dtype,
    )
    augmented[:-1, :-1] = system
    augmented[-1, -1] = 1.0 - float(gamma)
    alpha_augmented = torch.ones(
        (alpha.shape[0] + 1, 1), device=alpha.device, dtype=alpha.dtype
    )
    alpha_augmented[:-1] = alpha
    return torch.linalg.solve(augmented, alpha_augmented)


def orthogonal_sink_residual_gram(
    state_gram: torch.Tensor,
    sink_feature_coefficients: torch.Tensor,
    sink_norm: float,
) -> torch.Tensor:
    """Build Gram matrix for residual states plus baseline sink.

    For residuals ``r_i = (phi(x_i), -eps * c_i)`` and sink ``e=(0, eps)``,
    this returns their inner-product Gram matrix using the supplied state
    kernel. Applying a nonlinear kernel directly to concatenated residual
    coordinates does not preserve this algebra.
    """
    if state_gram.ndim != 2 or state_gram.shape[0] != state_gram.shape[1]:
        raise ValueError("state_gram must be a square matrix")
    n = state_gram.shape[0]
    coefficients = torch.as_tensor(
        sink_feature_coefficients,
        device=state_gram.device,
        dtype=state_gram.dtype,
    ).reshape(n, 1)
    eps_sq = float(sink_norm) ** 2

    gram = torch.empty(
        (n + 1, n + 1), device=state_gram.device, dtype=state_gram.dtype
    )
    gram[:-1, :-1] = state_gram + eps_sq * (coefficients @ coefficients.T)
    gram[:-1, -1:] = -eps_sq * coefficients
    gram[-1:, :-1] = -eps_sq * coefficients.T
    gram[-1, -1] = eps_sq
    return gram
