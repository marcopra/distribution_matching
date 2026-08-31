"""Behavior locks for release-facing Rover FIFO and kernel configuration."""

import numpy as np
import pytest
import torch

from agent.rover_utils.buffers import EncodedTransitionFIFO
from agent.rover_utils.kernels import KernelManager
from agent.rover_utils.actor_data import RawTransitions


def _encoded(values):
    values = torch.as_tensor(values, dtype=torch.float32).reshape(-1, 1)
    actions = torch.zeros(values.shape[0], 2, dtype=torch.float32)
    actions[:, 0] = 1
    return {
        "phi_obs": values,
        "phi_next": values + 1,
        "psi": values,
        "E": actions,
        "reward": torch.zeros_like(values),
    }


def test_fifo_pins_first_transition_and_evicts_oldest():
    fifo = EncodedTransitionFIFO(capacity=3)
    fifo.add(np.arange(5), _encoded(range(5)))

    stored = fifo.all("cpu")

    assert len(fifo) == 3
    assert stored["phi_obs"].flatten().tolist() == [0.0, 3.0, 4.0]


def test_random_landmark_selection_is_reproducible():
    fifo = EncodedTransitionFIFO(capacity=6)
    fifo.add(np.arange(6), _encoded(range(6)))
    manager = KernelManager(
        {
            "name": "inner_product",
            "subsampling_strategy": "random",
        }
    )

    torch.manual_seed(7)
    first = manager.select(fifo, size=4, device="cpu")
    torch.manual_seed(7)
    second = manager.select(fifo, size=4, device="cpu")

    assert torch.equal(first["phi_obs"], second["phi_obs"])
    assert first["phi_obs"][0].item() == 0.0


def test_pivoted_cholesky_preserves_first_landmark():
    fifo = EncodedTransitionFIFO(capacity=5)
    fifo.add(np.arange(5), _encoded(range(5)))
    manager = KernelManager(
        {
            "name": "inner_product",
            "subsampling_strategy": "pivoted_cholesky",
            "cholesky_progress": False,
            "cholesky_tolerance": 0.0,
        }
    )

    selected = manager.select(fifo, size=3, device="cpu")

    assert selected["phi_obs"][0].item() == 0.0
    assert selected["phi_obs"].shape[0] <= 3


def test_gaussian_manager_requires_and_uses_fixed_bandwidth():
    manager = KernelManager(
        {
            "name": "gaussian",
            "bandwidth": 0.3,
            "subsampling_strategy": "random",
        }
    )
    assert manager.bandwidth == 0.3

    with pytest.raises(ValueError, match="bandwidth is required"):
        KernelManager({"name": "gaussian", "bandwidth": None})


def test_raw_transitions_pin_initial_row():
    support = RawTransitions(
        obs=torch.tensor([[4.0], [5.0]]),
        action=torch.tensor([[1], [1]]),
        next_obs=torch.tensor([[5.0], [6.0]]),
        reward=torch.tensor([[0.0], [0.0]]),
    )
    initial = RawTransitions(
        obs=torch.tensor([[0.0]]),
        action=torch.tensor([[0]]),
        next_obs=torch.tensor([[1.0]]),
        reward=torch.tensor([[0.0]]),
    )

    pinned = support.with_first(initial)

    assert pinned.obs[0].item() == 0.0
    assert support.obs[0].item() == 4.0  # Frozen dataclass operation is non-mutating.
