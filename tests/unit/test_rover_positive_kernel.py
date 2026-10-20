"""Probability, adjoint, and warm-start checks for the positive estimator."""

import io

import numpy as np
import pytest
import torch

from agent.rover_positive_kernel import (
    PositiveKernelTransition, RoverPositiveKernelAgent, positive_kernel_logits,
)


@pytest.fixture(autouse=True)
def restore_default_dtype():
    dtype = torch.get_default_dtype()
    yield
    torch.set_default_dtype(dtype)


def make_model(dtype=torch.float64, gamma=.99, sink_norm=.7):
    generator = torch.Generator().manual_seed(10)
    H = torch.rand(13, 7, generator=generator, dtype=dtype)
    # Action 3 is unseen; landmark 0 has no nearby source transitions.
    H[:, 0] = 0
    E = torch.nn.functional.one_hot(torch.arange(13) % 3, 4).to(dtype)
    next_landmark = torch.arange(13) % 7
    gram = torch.eye(8, dtype=dtype)
    gram[-1, -1] = sink_norm ** 2
    alpha = torch.zeros(7, 1, dtype=dtype)
    alpha[2] = 1
    return PositiveKernelTransition(H, E, next_landmark, gram, alpha, gamma, .3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("gamma", [.7, .99])
def test_transition_and_occupancy_are_probabilities(dtype, gamma):
    model = make_model(dtype=dtype, gamma=gamma)
    pi = torch.full((7, 4), .25, dtype=dtype)
    P = model.transition_matrix(pi)
    z, S = model._resolvent(pi)
    d = (1 - gamma) * z
    assert P.min() >= 0 and d.min() >= 0
    torch.testing.assert_close(P.sum(0), torch.ones(8, dtype=dtype))
    torch.testing.assert_close(d.sum(), torch.tensor(1., dtype=dtype))
    torch.testing.assert_close(P[-1, [0, -1]], torch.ones(2, dtype=dtype))
    assert torch.linalg.cond(S, p=1) <= (1 + gamma) / (1 - gamma) * (1 + 1e-5)
    assert (gamma * torch.linalg.eigvals(P).abs().max()).item() == pytest.approx(gamma, rel=1e-5)
    assert 0 <= model.loss(pi) <= 1
    # With no examples for the selected action, every column goes to the sink.
    pi.zero_()
    pi[:, 3] = 1
    expected = torch.zeros_like(P)
    expected[-1] = 1
    torch.testing.assert_close(model.transition_matrix(pi), expected)


@pytest.mark.parametrize("sink_norm", [0., .7, 2.])
def test_normalized_gradient_matches_autograd_and_finite_difference(sink_norm):
    model = make_model(sink_norm=sink_norm)
    torch.manual_seed(12)
    logits = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    pi = torch.softmax(logits, 1)
    loss = model.loss(pi)
    automatic_pi, automatic_logits = torch.autograd.grad(loss, (pi, logits))
    coeff = model.gradient_coefficients(pi.detach())
    Q = positive_kernel_logits(model.H.T, model.E, coeff, model.prior_mass)
    z, _ = model._resolvent(pi.detach())
    torch.testing.assert_close(automatic_pi, z[:-1] * Q, atol=1e-12, rtol=1e-10)
    expected_logits = pi.detach() * z[:-1] * (Q - (pi.detach() * Q).sum(1, keepdim=True))
    torch.testing.assert_close(automatic_logits, expected_logits, atol=1e-12, rtol=1e-10)
    direction = torch.randn_like(logits)
    eps = 1e-5
    difference = (
        model.loss(torch.softmax(logits.detach() + eps * direction, 1))
        - model.loss(torch.softmax(logits.detach() - eps * direction, 1))
    ) / (2 * eps)
    torch.testing.assert_close(difference, (automatic_logits * direction).sum(), atol=1e-9, rtol=1e-6)
    # Sink coefficients matter: action-specific denominators prevent cancellation.
    sink_only = torch.zeros_like(coeff)
    sink_only[-1] = 2
    query = positive_kernel_logits(model.H.T, model.E, sink_only, model.prior_mass)
    torch.testing.assert_close(query, 2 * model.prior_mass / model.denominator)
    assert query[1].std() > 0


def test_invalid_policy_and_nonfinite_coefficients_fail_loudly():
    model = make_model()
    with pytest.raises(AssertionError):
        model.transition_matrix(torch.full((7, 4), .5, dtype=torch.float64))
    coeff = torch.full((14, 1), float("nan"), dtype=torch.float64)
    with pytest.raises(AssertionError, match="Nonfinite"):
        positive_kernel_logits(model.H.T, model.E, coeff, model.prior_mass)


def make_agent(dynamics_encoder_mode):
    return RoverPositiveKernelAgent(
        name="test", obs_type="states", obs_shape=(4,), grayscale=False,
        action_shape=(4,), lr_actor=10., discount=.99, lambda_reg=.001,
        batch_size=4, batch_size_actor=12, subsamples=6, nstep=1,
        use_tb=True, use_wandb=False, lr_T=1e-3, lr_encoder=1e-3,
        curl=False, embedding_sum_loss=0., hidden_dim=8, feature_dim=4,
        update_every_steps=1, update_actor_every_steps=1, pmd_steps=3,
        num_expl_steps=0, T_init_steps=0, total_train_steps=10,
        sink_schedule=".7", epsilon_schedule="0", mode="l1", reward=False,
        pca_truncation=None, embeddings=True, linear_projection=True, device="cpu",
        dynamics_kernel_input="states", dynamics_encoder_mode=dynamics_encoder_mode,
        dynamics_kernel_type="gaussian", coverage_kernel_input="states",
        coverage_state_indices=[0, 1], coverage_kernel_type="gaussian",
        kernel_bandwidth=.6, coverage_kernel_bandwidth=1., pmd_eta_mode="backtracking",
        pmd_best_iterate=True, pmd_warm_start=True, compute_dtype="float64",
        transition_prior_mass=.3,
    )


def encoded_batch(agent, n):
    obs = torch.randn(n, 4)
    next_obs = obs + .2 * torch.randn_like(obs)
    return agent._encode_actor_transition_batch((
        obs, (torch.arange(n) % 4).reshape(-1, 1), torch.zeros(n, 1),
        torch.ones(n, 1), next_obs,
    ))


@pytest.mark.parametrize("dynamics_encoder_mode", ["raw", "trainable"])
def test_actor_warm_start_changes_basis_and_survives_checkpoint(monkeypatch, dynamics_encoder_mode):
    torch.manual_seed(4)
    agent = make_agent(dynamics_encoder_mode)
    agent._sync_policy_encoder()
    monkeypatch.setattr(agent, "_save_actor_kernel_debug_plot", lambda *args, **kwargs: None)
    losses = []
    objective = agent._compute_actor_occupancy_loss

    def record_loss(**kwargs):
        value = objective(**kwargs)
        losses.append(float(value))
        return value

    def reject_regression(**kwargs):
        pytest.fail("Positive transitions must not construct a KRR inverse")

    monkeypatch.setattr(agent, "_compute_actor_occupancy_loss", record_loss)
    monkeypatch.setattr(agent.distribution_matcher, "compute_B_and_projections", reject_regression)
    for step, (n, m, steps) in enumerate([(7, 4, 3), (11, 6, 0), (5, 3, 3)]):
        full = encoded_batch(agent, n)
        sub = {key: value[:m] for key, value in full.items()}
        previous_support, previous_pi = agent._pmd_previous_support, agent._pmd_previous_policy
        losses.clear()
        agent.pmd_steps = steps
        metrics = agent.update_actor_nystrom(None, None, None, step, encoded_full=full, encoded_sub=sub)
        assert agent._positive_transition is None
        assert agent.gradient_coeff.shape == (n + 1, 1) and agent.pi.shape == (m, 4)
        assert float(metrics["actor_loss"]) <= losses[0]
        if steps:
            assert float(metrics["actor_loss"]) < losses[0]
        else:
            nearest = agent._pmd_nearest_rows(agent._pmd_previous_support, previous_support)
            torch.testing.assert_close(agent.pi, previous_pi[nearest])
        query = sub["proprio_next_state"].numpy()
        batched = agent._compute_action_probs_batch(query)
        singles = np.stack([agent.compute_action_probs(q) for q in query])
        np.testing.assert_allclose(batched, agent.pi.numpy(), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(singles, batched, rtol=1e-10, atol=1e-12)

    # Remove instance monkeypatches before pickling the real agent.
    monkeypatch.undo()
    buffer = io.BytesIO()
    torch.save(agent, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, map_location="cpu", weights_only=False)
    assert restored._positive_transition is None
    np.testing.assert_allclose(restored._compute_action_probs_batch(query), batched)


def test_failed_actor_update_releases_transition_model(monkeypatch):
    agent = make_agent("raw")
    agent._sync_policy_encoder()
    monkeypatch.setattr(agent, "_save_actor_kernel_debug_plot", lambda *args, **kwargs: None)

    def fail(**kwargs):
        assert agent._positive_transition is not None
        raise RuntimeError("deliberate test failure")

    monkeypatch.setattr(agent, "_compute_actor_occupancy_loss", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        agent.update_actor_nystrom(None, None, None, 0, encoded_full=encoded_batch(agent, 5))
    assert agent._positive_transition is None
