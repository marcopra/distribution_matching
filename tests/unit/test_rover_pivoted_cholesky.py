"""Action-local CPU/CUDA pivots against an independent dense Gram reference."""
import numpy as np
import pytest
import torch

from agent.rover_buffers import EncodedTransitionFIFO


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))]


def dense_pivots(features, size, tolerance, force_first, kernel, actions, bandwidth):
    x = features.detach().cpu().float()
    gram = (x @ x.T if kernel == "inner_product" else
            torch.exp(-(x[:, None] - x[None, :]).square().sum(-1) / (2 * bandwidth ** 2)))
    if actions is not None:
        gram *= actions[:, None] == actions[None, :]
    residual = (x.square().sum(1) if kernel == "inner_product" else torch.ones(len(x), dtype=x.dtype))
    threshold = tolerance * max(residual.max().item(), torch.finfo(x.dtype).eps)
    factor = x.new_zeros((len(x), min(size, len(x))))
    available = torch.ones(len(x), dtype=torch.bool)
    indices, values = [], []
    for j in range(factor.shape[1]):
        pivot = 0 if j == 0 and force_first else int(residual.masked_fill(~available, -torch.inf).argmax())
        value = max(float(residual[pivot]), 0.)
        if not (j == 0 and force_first) and tolerance > 0 and value <= threshold:
            break
        indices.append(pivot)
        values.append(value)
        available[pivot] = False
        if value > torch.finfo(x.dtype).eps:
            column = gram[:, pivot] - factor[:, :j] @ factor[pivot, :j]
            factor[:, j] = column / np.sqrt(value)
            residual.sub_(factor[:, j].square()).clamp_(min=0)
        residual[pivot] = 0
    return torch.tensor(indices, dtype=torch.long), np.asarray(values, dtype=np.float32)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kernel", ["inner_product", "gaussian"])
@pytest.mark.parametrize("force_first", [False, True])
@pytest.mark.parametrize("size", [1, 17, 40])
def test_pivots_match_dense(device, kernel, force_first, size):
    x = torch.randn(29, 31, generator=torch.Generator().manual_seed(19), dtype=torch.float32)
    actions = torch.tensor([7, 0, 2] * 9 + [0, 0])
    kwargs = dict(size=size, tolerance=1e-6, force_first=force_first,
                  kernel_type=kernel, actions=actions, bandwidth=3.)
    expected, residuals = dense_pivots(x, size, 1e-6, force_first, kernel, actions, 3.)
    actual, values = EncodedTransitionFIFO._pivoted_cholesky_indices(x, device=device, **kwargs)
    torch.testing.assert_close(actual, expected)
    np.testing.assert_allclose(values, residuals, atol=1e-5, rtol=1e-5)
    assert actual.device.type == "cpu" and values.dtype == np.float32


@pytest.mark.parametrize("device", DEVICES)
def test_global_ties_and_unbalanced_factor_growth(device):
    x = torch.eye(73, dtype=torch.float32)
    actions = torch.tensor([9] * 68 + [0] * 5)
    indices, values = EncodedTransitionFIFO._pivoted_cholesky_indices(
        x, 71, 0., True, actions=actions, device=device)
    torch.testing.assert_close(indices, torch.arange(71))
    np.testing.assert_array_equal(values, np.ones(71, dtype=np.float32))


@pytest.mark.parametrize("device", DEVICES)
def test_duplicates_tolerance_and_action_ties(device):
    x = torch.ones(6, 3, dtype=torch.float64)
    actions = torch.tensor([3, 0, 3, 1, 0, 1])
    indices, values = EncodedTransitionFIFO._pivoted_cholesky_indices(
        x, 6, 1e-6, False, kernel_type="gaussian", actions=actions, bandwidth=1., device=device)
    torch.testing.assert_close(indices, torch.tensor([0, 1, 3]))
    np.testing.assert_array_equal(values, np.ones(3, dtype=np.float32))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("force_first", [False, True])
def test_unmasked_linear_and_zero_initial_point(device, force_first):
    x = torch.tensor([[0., 0.], [1., 0.], [0., 2.], [3., 0.], [0., 1.]])
    expected, residuals = dense_pivots(x, 5, 1e-6, force_first, "inner_product", None, None)
    actual, values = EncodedTransitionFIFO._pivoted_cholesky_indices(
        x, 5, 1e-6, force_first, device=device)
    torch.testing.assert_close(actual, expected)
    np.testing.assert_allclose(values, residuals)


@pytest.mark.parametrize("device", DEVICES)
def test_zero_tolerance_selects_distinct_zero_rows(device):
    indices, values = EncodedTransitionFIFO._pivoted_cholesky_indices(
        torch.zeros(7, 3), 10, 0., True, device=device)
    torch.testing.assert_close(indices, torch.arange(7))
    np.testing.assert_array_equal(values, np.zeros(7, dtype=np.float32))


@pytest.mark.parametrize("device", DEVICES)
def test_products_use_only_action_rows_and_contiguous_directions(device, monkeypatch):
    original = torch.Tensor.__matmul__
    products = []
    def checked(left, right):
        assert left.device.type == device == right.device.type
        assert left.shape[0] <= 11
        if left.shape[1] != 41:
            assert left.T.is_contiguous()
        products.append(left.shape)
        return original(left, right)
    monkeypatch.setattr(torch.Tensor, "__matmul__", checked)
    indices, _ = EncodedTransitionFIFO._pivoted_cholesky_indices(
        torch.eye(41, dtype=torch.float32), 35, 0., True,
        actions=torch.arange(41) % 4, device=device)
    torch.testing.assert_close(indices, torch.arange(35))
    assert products


@pytest.mark.parametrize("features,actions,message", [
    (torch.tensor([[float("nan")]]), torch.tensor([0]), "Non-finite"),
    (torch.ones(1, 1), torch.tensor([-1]), "Negative"),
])
def test_invalid_inputs_fail(features, actions, message):
    with pytest.raises(AssertionError, match=message):
        EncodedTransitionFIFO._pivoted_cholesky_indices(features, 1, 0., True, actions=actions)


def test_allocation_errors_propagate(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("allocation failed")
    monkeypatch.setattr(torch.Tensor, "new_empty", fail)
    with pytest.raises(RuntimeError, match="allocation failed"):
        EncodedTransitionFIFO._pivoted_cholesky_indices(torch.ones(2, 1), 1, 0., True)
