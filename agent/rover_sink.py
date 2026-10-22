"""Shared algebra for ROVER's orthogonal sink-state augmentation."""

from __future__ import annotations

import torch


def nystrom_resolvent_coefficients(BM, alpha, gamma):
    """Solve the Nyström resolvent, including its absorbing sink coefficient."""
    augmented = _augmented_resolvent(BM, gamma)
    alpha_augmented = torch.ones(
        (alpha.shape[0] + 1, 1), device=alpha.device, dtype=alpha.dtype
    )
    alpha_augmented[:-1] = alpha
    return torch.linalg.solve(augmented, alpha_augmented)


@torch.no_grad()
def nystrom_resolvent_diagnostics(BM, gamma):
    """Estimate resolvent conditioning and summarize real-state transition entries."""
    assert BM.ndim == 2 and BM.shape[0] == BM.shape[1]
    assert torch.isfinite(BM).all()
    augmented = _augmented_resolvent(BM, gamma)
    matrix_norm = torch.linalg.matrix_norm(augmented, ord=1)
    lu, pivots = torch.linalg.lu_factor(augmented)
    del augmented
    inverse_norm_estimate = _estimate_inverse_one_norm(lu, pivots)
    del lu, pivots
    column_mass = BM.sum(0)
    stats = torch.stack((
        matrix_norm * inverse_norm_estimate,
        (BM < 0).sum(dtype=BM.dtype) / BM.numel(),
        BM.min(),
        -BM.clamp_max(0).sum(),
        column_mass.min(),
        column_mass.max(),
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
    augmented = BM.new_zeros(BM.shape[0] + 1, BM.shape[1] + 1)
    augmented[:-1, :-1].copy_(BM).mul_(-float(gamma))
    augmented[:-1, :-1].diagonal().add_(1)
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


@torch.no_grad()
def kernel_sink_gram(kernel, features, coefficients, sink_norm, block_size=1024):
    """Construct residual Gram without a second full Gram or outer product."""
    n = len(features)
    assert n > 0 and block_size > 0 and features.ndim == 2
    assert coefficients.shape == (n,)
    assert torch.isfinite(features.new_tensor(sink_norm)) and sink_norm >= 0
    gram = features.new_empty(n + 1, n + 1)
    for start in range(0, n, block_size):
        rows = slice(start, min(n, start + block_size))
        gram[rows, :n] = kernel(features[rows], features)
    c = coefficients.reshape(n, 1)
    gram[:n, :n].addmm_(c, c.T, alpha=float(sink_norm) ** 2)
    gram[:n, -1:] = -float(sink_norm) ** 2 * c
    gram[-1:, :n] = -float(sink_norm) ** 2 * c.T
    gram[-1, -1] = float(sink_norm) ** 2
    return gram
