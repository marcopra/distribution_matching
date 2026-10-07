import pytest
import torch

import utils
from agent.rover_matchers import DistributionMatcher
from agent.rover_networks import FrozenCNNFeatureEncoder
from agent.rover_sink import orthogonal_sink_residual_gram
from agent.rover_subspace_matchers import SubspaceCoverageMatcher


def test_frozen_cnn_spatial_features_are_unit_mass_and_frozen():
    encoder = FrozenCNNFeatureEncoder((1, 84, 84), feature_dim=16)
    observations = torch.randint(0, 256, (3, 1, 84, 84), dtype=torch.uint8)

    features = encoder(observations)

    assert features.shape == (3, 32 * 7 * 7)
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert torch.all(features >= 0)
    torch.testing.assert_close(features.sum(dim=1), torch.ones(3))


def test_subspace_sink_requires_unit_mass_dynamics_features():
    matcher = SubspaceCoverageMatcher(
        gamma=0.6,
        kernel_fn=utils.build_kernel_fn("gaussian", bandwidth=0.8),
    )
    with pytest.raises(ValueError, match="unit-mass dynamics state-action"):
        matcher._coverage_gram(
            phi_sub_next_obs=torch.tensor([[0.2, 0.3, 0.0]]),
            psi_sub_obs_action=torch.tensor([[0.2, 0.3]]),
            sink_norm=0.1,
            state_indices=(0, 1),
        )


def test_orthogonal_sink_residual_gram_matches_linear_feature_construction():
    features = torch.tensor(
        [[0.2, -0.3], [0.7, 0.1], [-0.4, 0.5]], dtype=torch.float64
    )
    sink_coefficients = torch.tensor([1.0, 0.5, 1.2], dtype=torch.float64)
    sink_norm = 0.3
    state_gram = features @ features.T

    residuals = torch.cat(
        [features, -sink_norm * sink_coefficients[:, None]], dim=1
    )
    sink = torch.tensor([[0.0, 0.0, sink_norm]], dtype=torch.float64)
    explicit_features = torch.cat([residuals, sink], dim=0)
    expected = explicit_features @ explicit_features.T

    actual = orthogonal_sink_residual_gram(
        state_gram, sink_coefficients, sink_norm
    )
    torch.testing.assert_close(actual, expected)


def test_gaussian_sink_gram_has_orthogonal_sink_blocks():
    features = torch.tensor(
        [[0.75, 0.75], [1.4, -1.4], [-0.5, 0.8]], dtype=torch.float64
    )
    sink_coefficients = torch.ones(3, dtype=torch.float64)
    sink_norm = 0.2
    state_kernel = utils.build_kernel_fn("gaussian", bandwidth=1.28658)
    state_gram = state_kernel(features, features)

    actual = orthogonal_sink_residual_gram(
        state_gram, sink_coefficients, sink_norm
    )
    expected = torch.zeros((4, 4), dtype=torch.float64)
    expected[:-1, :-1] = state_gram + sink_norm**2
    expected[:-1, -1] = -sink_norm**2
    expected[-1, :-1] = -sink_norm**2
    expected[-1, -1] = sink_norm**2
    torch.testing.assert_close(actual, expected)


def test_gaussian_nystrom_loss_matches_state_plus_sink_decomposition():
    dtype = torch.float64
    device = torch.device("cpu")
    matcher = DistributionMatcher(
        lambda_reg=1e-3,
        gamma=0.6,
        kernel_type="gaussian",
        kernel_bandwidth=0.8,
        device="cpu",
    )
    xy = torch.tensor(
        [[0.2, 0.1], [0.6, -0.3], [-0.4, 0.5]], dtype=dtype, device=device
    )
    phi_sub_next_obs = torch.cat(
        [xy, torch.zeros((3, 1), dtype=dtype, device=device)], dim=1
    )
    psi_sub_obs_action = torch.tensor(
        [[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]], dtype=dtype, device=device
    )
    H = torch.eye(3, dtype=dtype, device=device)
    E = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=dtype, device=device
    )
    pi = torch.tensor(
        [[0.6, 0.4], [0.3, 0.7], [0.8, 0.2]], dtype=dtype, device=device
    )
    alpha = torch.tensor([[1.0], [0.0], [0.0]], dtype=dtype, device=device)
    B_nystrom = torch.eye(3, dtype=dtype, device=device)
    sink_norm = 0.2

    actual = matcher.compute_nu_pi_nystrom_kernel_loss(
        phi_sub_next_obs=phi_sub_next_obs,
        psi_sub_obs_action=psi_sub_obs_action,
        H=H,
        pi=pi,
        E=E,
        alpha=alpha,
        sink_norm=sink_norm,
        B_nystrom=B_nystrom,
    )

    M = H * (E @ pi.T)
    weights = (1.0 - matcher.gamma) * torch.linalg.solve(
        torch.eye(3, dtype=dtype, device=device) - matcher.gamma * (B_nystrom @ M),
        alpha,
    )
    state_gram = matcher.state_kernel(xy, xy)
    sink_mass = 1.0 - (psi_sub_obs_action.sum(dim=1, keepdim=True).T @ weights).squeeze()
    expected = (
        weights.T @ state_gram @ weights + sink_norm**2 * sink_mass.square()
    ).squeeze()
    torch.testing.assert_close(actual, expected)


def test_subspace_matcher_gaussian_loss_uses_orthogonal_sink_decomposition():
    dtype = torch.float64
    matcher = SubspaceCoverageMatcher(
        gamma=0.6,
        kernel_fn=utils.build_kernel_fn("gaussian", bandwidth=0.8),
    )
    xy = torch.tensor(
        [[0.2, 0.1], [0.6, -0.3], [-0.4, 0.5]], dtype=dtype
    )
    phi_sub_next_obs = torch.cat(
        [xy, torch.zeros((3, 1), dtype=dtype)], dim=1
    )
    psi = torch.tensor([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]], dtype=dtype)
    H = torch.eye(3, dtype=dtype)
    E = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=dtype)
    pi = torch.tensor([[0.6, 0.4], [0.3, 0.7], [0.8, 0.2]], dtype=dtype)
    alpha = torch.tensor([[1.0], [0.0], [0.0]], dtype=dtype)
    B_nystrom = torch.eye(3, dtype=dtype)
    sink_norm = 0.2

    actual = matcher.compute_nystrom_subspace_occupancy_loss(
        phi_sub_next_obs=phi_sub_next_obs,
        psi_sub_obs_action=psi,
        H=H,
        pi=pi,
        E=E,
        alpha=alpha,
        sink_norm=sink_norm,
        B_nystrom=B_nystrom,
        state_indices=(0, 1),
        coverage_features=xy,
    )

    coefficients = matcher._resolvent_coefficients(
        H, pi, E, alpha, B_nystrom, matcher.gamma
    )
    weights = (1.0 - matcher.gamma) * coefficients
    state_gram = matcher.kernel_fn(xy, xy)
    sink_mass = 1.0 - (psi.sum(dim=1, keepdim=True).T @ weights[:-1]).squeeze()
    expected = (
        weights[:-1].T @ state_gram @ weights[:-1]
        + sink_norm**2 * sink_mass.square()
    ).squeeze()
    torch.testing.assert_close(actual, expected)


def test_inner_product_nystrom_loss_preserves_explicit_occupancy_norm():
    dtype = torch.float64
    matcher = DistributionMatcher(
        lambda_reg=1e-3,
        gamma=0.6,
        kernel_type="inner_product",
        device="cpu",
    )
    xy = torch.tensor(
        [[0.2, 0.1], [0.6, -0.3], [-0.4, 0.5]], dtype=dtype
    )
    phi_sub_next_obs = torch.cat(
        [xy, torch.zeros((3, 1), dtype=dtype)], dim=1
    )
    psi = torch.tensor([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]], dtype=dtype)
    H = torch.eye(3, dtype=dtype)
    E = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=dtype)
    pi = torch.tensor([[0.6, 0.4], [0.3, 0.7], [0.8, 0.2]], dtype=dtype)
    alpha = torch.tensor([[1.0], [0.0], [0.0]], dtype=dtype)
    B_nystrom = torch.eye(3, dtype=dtype)
    sink_norm = 0.2

    old_occupancy = matcher.compute_nu_pi_nystrom_memory_efficient(
        phi_all_obs=phi_sub_next_obs,
        phi_sub_next_obs=phi_sub_next_obs,
        psi_sub_obs_action=psi,
        psi_all_obs_action=psi,
        H=H,
        pi=pi,
        E=E,
        alpha=alpha,
        sink_norm=sink_norm,
        B_nystrom=B_nystrom,
    )
    old_loss = torch.linalg.norm(old_occupancy).square()
    new_loss = matcher.compute_nu_pi_nystrom_kernel_loss(
        phi_sub_next_obs=phi_sub_next_obs,
        psi_sub_obs_action=psi,
        H=H,
        pi=pi,
        E=E,
        alpha=alpha,
        sink_norm=sink_norm,
        B_nystrom=B_nystrom,
    )
    torch.testing.assert_close(new_loss, old_loss)
