"""Streamed algebra against independent dense Nyström equations."""
import gc
import io
import weakref

import pytest
import torch
import torch.nn.functional as F

from agent.rover_matchers import DistributionMatcher, NystromActorModel
from agent.rover_sink import kernel_sink_gram, orthogonal_sink_residual_gram, nystrom_resolvent_diagnostics


@pytest.fixture(autouse=True)
def restore_default_dtype():
    previous = torch.get_default_dtype()
    yield
    torch.set_default_dtype(previous)


def case(kernel="gaussian", rank=3, block_size=2, device="cpu", duplicate=False, dtype=torch.float64):
    generator = torch.Generator().manual_seed(19)
    states = torch.rand(9, 4, generator=generator, dtype=dtype).to(device)
    states /= states.sum(1, keepdim=True)
    landmarks = states[:5].clone()
    if duplicate:
        landmarks[2] = landmarks[0]
    next_states = states[3:8].clone()
    actions = torch.arange(9, device=device) % 3
    sub_actions = actions[:5].clone()
    E = F.one_hot(actions, 3).to(states.dtype)
    alpha = states.new_zeros(5, 1)
    alpha[0] = 1
    matcher = DistributionMatcher(.001, gamma=.7, kernel_type=kernel, kernel_bandwidth=.8)
    gram = kernel_sink_gram(matcher.state_kernel, next_states, landmarks.sum(1), .3, block_size)
    model = NystromActorModel(matcher, states, landmarks, next_states, actions,
                             sub_actions, E, alpha, gram, rank, block_size)
    return model


def dense(model, pi):
    matcher, x, y = model.matcher, model.states, model.landmarks
    K = matcher.state_action_kernel(x, y, model.actions, model.landmark_actions)
    K_mm = matcher.state_action_kernel(y, y, model.landmark_actions, model.landmark_actions)
    A = K.T @ K + matcher.lambda_reg * K_mm
    A.diagonal().add_(1e-6)
    B = torch.linalg.solve(A, K.T)
    H = matcher.state_kernel(x, model.next_landmarks)
    BM = B @ (H * (model.E @ pi.T))
    S = torch.eye(len(y) + 1, dtype=x.dtype, device=x.device)
    S[:-1, :-1] -= matcher.gamma * BM
    S[-1, -1] -= matcher.gamma
    alpha = torch.cat((model.alpha, model.alpha.new_ones(1, 1)))
    z = torch.linalg.solve(S, alpha)
    loss = ((1 - matcher.gamma) ** 2 * (z.T @ model.gram @ z)).squeeze()
    U = F.pad(model.U, (0, 1, 0, 1))
    U[-1, -1] = 1
    S_r = U.T @ S @ U
    z_r = torch.linalg.solve(S_r, U.T @ alpha)
    tmp = torch.linalg.solve(S_r.T + 1e-6 * torch.eye(len(S_r), device=x.device, dtype=x.dtype),
                             U.T @ model.gram @ U @ z_r)
    tilde_B = F.pad(B, (0, 1, 0, 1))
    tilde_B[-1, -1] = 1
    gradient = 2 * matcher.gamma * (1 - matcher.gamma) ** 2 * tilde_B.T @ U @ tmp
    return A, B, H, BM, loss, gradient


@pytest.mark.parametrize("kernel", ["gaussian", "inner_product"])
@pytest.mark.parametrize("rank", [None, 3])
@pytest.mark.parametrize("block_size", [1, 4, 20])
@pytest.mark.parametrize("duplicate", [False, True])
def test_streamed_matches_dense(kernel, rank, block_size, duplicate):
    model = case(kernel, rank, block_size, duplicate=duplicate)
    coeff = torch.linspace(-.1, .1, 10, dtype=torch.float64).reshape(-1, 1)
    offset = model.states.new_full((5, 3), .2)
    pi = model.policy(coeff, offset)
    A, B, H, BM, loss, gradient = dense(model, pi)
    torch.testing.assert_close(model._solve(A), torch.eye(5, dtype=A.dtype), atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(pi, torch.softmax(-(H.T @ (coeff[:-1] * model.E) + coeff[-1] + offset), 1))
    torch.testing.assert_close(model.transition_matrix(pi), BM, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(model.adjoint(model.U), B.T @ model.U, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(model.loss(pi), loss, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(model.gradient_coefficients(pi), gradient, atol=1e-8, rtol=1e-8)
    assert all(torch.isfinite(torch.tensor(value)) for value in model.diagnostics(pi).values())


def test_kernel_blocks_are_bounded_and_policy_survives_pickle(monkeypatch):
    calls = []
    original = DistributionMatcher.kernel
    def checked(self, X, Y):
        calls.append(len(X))
        assert len(X) <= 2
        return original(self, X, Y)
    monkeypatch.setattr(DistributionMatcher, "kernel", checked)
    state_kernel = DistributionMatcher.state_kernel
    def checked_state(self, X, Y):
        assert len(X) <= 2
        return state_kernel(self, X, Y)
    monkeypatch.setattr(DistributionMatcher, "state_kernel", checked_state)
    model = case(block_size=2)
    coeff = model.states.new_zeros(10, 1)
    pi = model.policy(coeff)
    model.loss(pi)
    model.gradient_coefficients(pi)
    assert calls and max(calls) == 2
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)
    torch.testing.assert_close(restored.policy(coeff), pi)


def test_factorization_failure_propagates(monkeypatch):
    def fail(*args, **kwargs):
        raise torch.linalg.LinAlgError("singular test system")
    monkeypatch.setattr(torch.linalg, "lu_factor", fail)
    with pytest.raises(torch.linalg.LinAlgError, match="singular test system"):
        case()


def test_nonfinite_inputs_and_invalid_policy_fail():
    model = case()
    with pytest.raises(AssertionError):
        model.policy(model.states.new_full((10, 1), float("nan")))
    with pytest.raises(AssertionError):
        model.transition_matrix(model.states.new_zeros(5, 3))


def test_regression_and_eigendecomposition_use_action_blocks(monkeypatch):
    sizes = {"lu_factor": [], "eigh": []}
    for name in sizes:
        original = getattr(torch.linalg, name)
        def checked(matrix, *args, _name=name, _original=original, **kwargs):
            sizes[_name].append(len(matrix))
            return _original(matrix, *args, **kwargs)
        monkeypatch.setattr(torch.linalg, name, checked)
    model = case()
    assert sizes == {"lu_factor": [2, 2, 1], "eigh": [2, 2, 1]}
    assert sum(block["transition"].numel() for block in model.blocks) == 25
    assert sum(block["lu"].numel() for block in model.blocks) == 9
    assert sum(block["projected_transition"].numel() for block in model.blocks) == 15
    K = model.matcher.state_action_kernel(model.landmarks, model.landmarks,
                                          model.landmark_actions, model.landmark_actions)
    K = K.cpu().clone()
    K.diagonal().add_(1e-8)
    reference = torch.linalg.eigh(K.float())[1][:, -3:].to(model.U)
    torch.testing.assert_close(model.U @ model.U.T, reference @ reference.T, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(model.U_gram, model.U.T @ model.U)


def test_cached_transition_never_recomputes_kernels(monkeypatch):
    model = case()
    pi = model.policy(model.states.new_zeros(10, 1))
    expected = model.transition_matrix(pi)
    def fail(*args, **kwargs):
        raise AssertionError("Transition recomputed a kernel")
    monkeypatch.setattr(model.matcher, "kernel", fail)
    monkeypatch.setattr(model.matcher, "state_kernel", fail)
    torch.testing.assert_close(model.transition_matrix(pi), expected)
    assert torch.isfinite(model.loss(pi))
    assert model.diagnostics(pi)


def test_projected_gradient_never_builds_full_transition(monkeypatch):
    model = case()
    pi = model.policy(model.states.new_zeros(10, 1))
    expected = dense(model, pi)[-1]
    def fail(*args, **kwargs):
        raise AssertionError("Projected gradient built a full transition")
    monkeypatch.setattr(model, "transition_matrix", fail)
    torch.testing.assert_close(model.gradient_coefficients(pi), expected, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))])
def test_unbalanced_projection_and_action_without_landmarks(device):
    x = torch.tensor([[3., 0.], [2., 0.], [0., .2], [0., .1], [.3, .1]], dtype=torch.float64, device=device)
    actions = torch.tensor([0, 0, 1, 1, 2], device=device)
    indices = torch.tensor([2, 0, 3, 1], device=device)
    y, b = x[indices], actions[indices]
    E = F.one_hot(actions, 3).to(x.dtype)
    alpha = x.new_zeros(4, 1)
    alpha[0] = 1
    matcher = DistributionMatcher(.001, gamma=.7, kernel_type="inner_product")
    gram = kernel_sink_gram(matcher.state_kernel, y, y.sum(1), .2)
    model = NystromActorModel(matcher, x, y, y, actions, b, E, alpha, gram, 1, block_size=2)
    assert [len(block["basis"]) for block in model.blocks] == [1, 0]
    pi = model.policy(x.new_zeros(6, 1))
    _, B, _, BM, loss, gradient = dense(model, pi)
    torch.testing.assert_close(model.transition_matrix(pi), BM, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(model.adjoint(model.U), B.T @ model.U, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(model.loss(pi), loss, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(model.gradient_coefficients(pi), gradient, atol=1e-8, rtol=1e-8)
    assert model.gradient_coefficients(pi)[-2] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_repeated_cuda_updates_release_live_memory():
    baselines = []
    for _ in range(5):
        model = case(device="cuda")
        pi = model.policy(model.states.new_zeros(10, 1))
        model.loss(pi)
        model.gradient_coefficients(pi)
        ref = weakref.ref(model)
        del model, pi
        gc.collect()
        torch.cuda.synchronize()
        assert ref() is None
        baselines.append(torch.cuda.memory_allocated())
    assert max(baselines[1:]) - min(baselines[1:]) < 1024 * 1024


@pytest.mark.parametrize("mode", ["raw", "trainable"])
def test_regression_actor_checkpoint_preserves_queries_without_update_model(monkeypatch, mode):
    from agent.rover_nystrom_subspace import RoverSubspaceCoverageAgent
    from tests.unit.test_rover_positive_kernel import agent_kwargs, encoded_batch
    torch.manual_seed(0)  # Keep the tiny ReLU fixture's landmark features nonzero.
    agent = RoverSubspaceCoverageAgent(**agent_kwargs(mode))
    agent._sync_policy_encoder()
    monkeypatch.setattr(agent, "_save_actor_kernel_debug_plot", lambda *args, **kwargs: None)
    prepare = agent._prepare_actor_model
    models = []
    def measured(sink_norm):
        model = prepare(sink_norm)
        models.append(weakref.ref(model))
        return model
    monkeypatch.setattr(agent, "_prepare_actor_model", measured)
    full = encoded_batch(agent, 9)
    sub = {key: value[:5] for key, value in full.items()}
    agent.update_actor_nystrom(None, None, None, 0, encoded_full=full, encoded_sub=sub)
    assert models[0]() is None
    assert not any(isinstance(value, NystromActorModel) for value in vars(agent).values())
    assert not hasattr(agent, "_psi_all") and not hasattr(agent, "_phi_all_next")
    query = sub["proprio_next_state"].numpy()
    expected = agent._compute_action_probs_batch(query)
    monkeypatch.undo()
    buffer = io.BytesIO()
    torch.save(agent, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, map_location="cpu", weights_only=False)
    torch.testing.assert_close(torch.from_numpy(restored._compute_action_probs_batch(query)), torch.from_numpy(expected))


@pytest.mark.parametrize("block_size", [1, 2, 20])
def test_chunked_coverage_gram_matches_nonunit_sink_features(block_size):
    x = torch.tensor([[.1, .2], [.3, .4], [.7, .1]], dtype=torch.float64)
    coefficients = x.sum(1)
    matcher = DistributionMatcher(.001, kernel_type="gaussian", kernel_bandwidth=.7)
    expected = orthogonal_sink_residual_gram(matcher.state_kernel(x, x), coefficients, .4)
    torch.testing.assert_close(kernel_sink_gram(matcher.state_kernel, x, coefficients, .4, block_size), expected)


def test_diagnostics_match_dense_transition_statistics():
    BM = torch.tensor([[.1, -.2], [.3, .4]], dtype=torch.float64)
    gamma = .7
    S = torch.eye(3, dtype=BM.dtype)
    S[:-1, :-1] -= gamma * BM
    S[-1, -1] -= gamma
    actual = list(nystrom_resolvent_diagnostics(BM, gamma).values())
    expected = [torch.linalg.cond(S, p=1).item(), .25, -.2, .2, .2, .4]
    torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected))


def test_float32_configuration_retains_dtype_and_dense_algebra():
    model = case(dtype=torch.float32)
    pi = model.policy(model.states.new_zeros(10, 1))
    _, _, _, BM, loss, gradient = dense(model, pi)
    assert all(block["lu"].dtype == torch.float32 for block in model.blocks)
    assert model.U.dtype == pi.dtype == torch.float32
    torch.testing.assert_close(model.transition_matrix(pi), BM, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(model.loss(pi), loss, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(model.gradient_coefficients(pi), gradient, rtol=1e-3, atol=1e-4)
