"""Regression checks for reset mass and per-state policy confidence."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agent.rover import RoverAgent as StandardRover
from agent.rover_nystrom_debug import RoverAgent
from agent.rover_nystrom_subspace import RoverSubspaceCoverageAgent
from tests.unit.test_rover_actor_encoder_drift import make_agent


@pytest.mark.parametrize('cls', [StandardRover, RoverAgent])
def test_opposing_deterministic_policies_do_not_cancel(cls):
    agent = cls.__new__(cls)
    agent.n_actions = 4
    assert agent._compute_mean_action_probs_deviation(np.eye(4)) == .75
    assert agent._compute_mean_action_probs_deviation(np.full((8, 4), .25)) == 0.
    mixed = .85 * np.eye(4) + .15 / 4
    assert agent._compute_mean_action_probs_deviation(mixed) == pytest.approx(.6375)


def test_pointmaze_actor_anchor_keeps_replay_real_and_coverage_aligned():
    agent = RoverSubspaceCoverageAgent.__new__(RoverSubspaceCoverageAgent)
    agent.__dict__.update(make_agent().__dict__)
    agent.obs_shape = (4,)
    agent.dynamics_kernel_input = "states"
    agent.uses_state_dynamics_encoder = False
    agent.dynamics_state_dim = 4
    agent.compute_dtype = torch.float32
    agent.coverage_kernel_input = "states"
    agent.coverage_state_indices = (0, 1)
    agent.nystrom_debug = SimpleNamespace(_has_point_env=lambda: True)
    obs = torch.tensor([[-1., 1., 0., 0.], [.5, 1., .2, 0.]])
    following = obs + .02
    encoded = agent._encode_actor_transition_batch((
        obs, torch.tensor([[0], [1]]), torch.ones(2, 1),
        torch.ones(2, 1), following, obs[:, :2],
    ))
    before = {k: v.clone() for k, v in encoded.items()}
    anchored = agent._anchor_pointmaze_actor_batch(encoded)
    torch.testing.assert_close(anchored['next_obs_raw'][0], obs[0])
    torch.testing.assert_close(anchored['coverage_next_raw'][0], obs[0, :2])
    torch.testing.assert_close(anchored['phi_next'][0], anchored['phi_obs'][0])
    assert anchored['reward'][0].item() == 0.
    for key in encoded:
        torch.testing.assert_close(encoded[key], before[key])
        torch.testing.assert_close(anchored[key][1:], before[key][1:])


def test_raw_pointmaze_actor_anchor_uses_reset_source():
    agent = make_agent()
    agent.nystrom_debug = SimpleNamespace(_has_point_env=lambda: True)
    obs = torch.tensor([[-1., 1.]])
    following = obs + .02
    replay = SimpleNamespace(get_first_transition=lambda: (
        obs, torch.tensor([[0]]), torch.ones(1, 1), torch.ones(1, 1), following,
    ))
    source, action, target, reward = agent._load_first_actor_transition(replay)
    torch.testing.assert_close(source, obs)
    torch.testing.assert_close(target, obs)
    assert reward.item() == 0.
    torch.testing.assert_close(following, obs + .02)


def test_explicit_synthetic_first_transition_keeps_its_reset_target():
    agent = make_agent()
    agent.nystrom_debug = SimpleNamespace(_has_point_env=lambda: True)
    source = torch.zeros(1, 2)
    reset = torch.tensor([[-1., 1.]])
    replay = SimpleNamespace(get_first_transition=lambda: (
        source, torch.tensor([[0]]), torch.zeros(1, 1), torch.ones(1, 1), reset,
    ))
    _, _, target, _ = agent._load_first_actor_transition(replay)
    torch.testing.assert_close(target, reset)
