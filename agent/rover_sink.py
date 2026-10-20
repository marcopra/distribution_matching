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
    BM = B_nystrom @ (H * (E @ pi.T))
    augmented = _augmented_resolvent(BM, gamma)
    alpha_augmented = torch.ones(
        (alpha.shape[0] + 1, 1), device=alpha.device, dtype=alpha.dtype
    )
    alpha_augmented[:-1] = alpha
    return torch.linalg.solve(augmented, alpha_augmented)


@torch.no_grad()
def nystrom_resolvent_diagnostics(
    H: torch.Tensor,
    pi: torch.Tensor,
    E: torch.Tensor,
    B_nystrom: torch.Tensor,
    gamma: float,
) -> dict[str, float]:
    """Estimate resolvent conditioning and summarize real-state transition entries."""
    BM = B_nystrom @ (H * (E @ pi.T))
    assert BM.ndim == 2 and BM.shape[0] == BM.shape[1]
    assert torch.isfinite(BM).all()
    augmented = _augmented_resolvent(BM, gamma)
    lu, pivots = torch.linalg.lu_factor(augmented)
    inverse_norm_estimate = _estimate_inverse_one_norm(lu, pivots)
    matrix_norm = torch.linalg.matrix_norm(augmented, ord=1)
    negative = BM < 0
    stats = torch.stack((
        matrix_norm * inverse_norm_estimate,
        negative.to(BM.dtype).mean(),
        BM.min(),
        -BM.clamp_max(0).sum(),
        BM.sum(0).min(),
        BM.sum(0).max(),
    )).detach().cpu().tolist()
    diagnostics = dict(zip((
        "resolvent_condition_1_estimate",
        "negative_real_transition_fraction",
        "negative_real_transition_min",
        "negative_real_transition_l1",
        "real_transition_column_mass_min",
        "real_transition_column_mass_max",
    ), map(float, stats)))
    return diagnostics


def _augmented_resolvent(BM: torch.Tensor, gamma: float) -> torch.Tensor:
    system = torch.eye(BM.shape[0], device=BM.device, dtype=BM.dtype)
    system = system - float(gamma) * BM
    augmented = torch.zeros(
        (system.shape[0] + 1, system.shape[1] + 1),
        device=system.device,
        dtype=system.dtype,
    )
    augmented[:-1, :-1] = system
    augmented[-1, -1] = 1.0 - float(gamma)
    return augmented


def _estimate_inverse_one_norm(
    lu: torch.Tensor,
    pivots: torch.Tensor,
    iterations: int = 5,
) -> torch.Tensor:
    """Hager-style 1-norm estimate using an existing LU factorization."""
    size = lu.shape[0]
    x = torch.full((size, 1), 1.0 / size, device=lu.device, dtype=lu.dtype)
    estimate = lu.new_zeros(())
    for _ in range(iterations):
        y = torch.linalg.lu_solve(lu, pivots, x)
        estimate = torch.maximum(estimate, y.abs().sum())
        signs = torch.where(y >= 0, torch.ones_like(y), -torch.ones_like(y))
        z = torch.linalg.lu_solve(lu, pivots, signs, adjoint=True)
        index = z.abs().argmax()
        x = (
            torch.nn.functional.one_hot(index, num_classes=size)
            .to(lu.dtype)
            .reshape(size, 1)
        )
    return estimate


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
