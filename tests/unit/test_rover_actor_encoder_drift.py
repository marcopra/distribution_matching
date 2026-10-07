"""Regression for changing state encoders with an encoded actor FIFO."""
import numpy as np
import torch
from agent.rover_buffers import EncodedTransitionFIFO
from agent.rover_nystrom_debug import RoverAgent
from agent.rover_nystrom_subspace import RoverSubspaceCoverageAgent


class ChangingEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = 1.

    def encode_and_project(self, obs, normalize=True):
        raw = torch.stack([obs[:, 0].abs()+self.offset, obs[:, 1].abs()+1], 1)
        return torch.nn.functional.normalize(raw, p=1, dim=1) if normalize else raw


def make_agent():
    agent = RoverAgent.__new__(RoverAgent)
    agent.obs_type = "states"
    agent.embeddings = True
    agent.feature_learning_loss = "infonce"
    agent.device = "cpu"
    agent.aug = torch.nn.Identity()
    agent.policy_encoder = ChangingEncoder()
    agent.encoded_fifo_cuda_oom_splits = 2
    agent.batch_size_actor = 3
    agent.subsamples = 2
    agent.nystrom_synthetic_subsamples = False
    agent.subsampling_strategy = "random"
    agent.discount = .99
    agent.nystrom_candidate_multiplier = 1.
    agent.nystrom_cholesky_tolerance = 1e-6
    agent.nystrom_cholesky_progress = False
    agent.kernel_type = "gaussian"
    agent.kernel_bandwidth_mult = 1.
    agent._encoded_actor_fifo = EncodedTransitionFIFO(10)
    agent._update_encoded_actor_fifo = lambda replay: True
    return agent


def test_actor_support_and_landmarks_use_current_encoder_including_pinned_reset():
    agent = make_agent()
    obs = torch.tensor([[1., 0.], [0., 1.], [2., 1.]])
    next_obs = obs+.25
    transitions = (obs, torch.tensor([[0], [1], [0]]), torch.zeros(3,1),
                   torch.ones(3,1), next_obs)
    encoded = agent._encode_actor_transition_batch(transitions)
    assert "obs_raw" in encoded and "next_obs_raw" in encoded
    agent._encoded_actor_fifo.add(np.arange(3), encoded)
    original_pinned = agent._encoded_actor_fifo._first["phi_obs"].clone()
    agent.policy_encoder.offset = 10.

    data = agent._encoded_fifo_actor_update_data(object())
    for batch in [data.full, data.subsample]:
        torch.testing.assert_close(batch["phi_obs"],
            agent.policy_encoder.encode_and_project(batch["obs_raw"]))
        torch.testing.assert_close(batch["phi_next"],
            agent.policy_encoder.encode_and_project(batch["next_obs_raw"]))
        torch.testing.assert_close(batch["obs_raw"][0], obs[0])
        assert batch["phi_obs"].device.type == "cpu"
    assert not torch.allclose(data.full["phi_obs"][:1], original_pinned)
    # Sampling refreshes the bounded actor batch, without rewriting FIFO history.
    torch.testing.assert_close(agent._encoded_actor_fifo._first["phi_obs"], original_pinned)


def test_old_encoded_batches_without_raw_states_remain_loadable():
    agent = make_agent()
    legacy = {"phi_obs": torch.ones(2,2), "phi_next": torch.ones(2,2)}
    assert agent._refresh_encoded_state_features(legacy) is legacy


def test_subspace_refresh_keeps_raw_xy_coverage_with_full_state_dynamics():
    agent = RoverSubspaceCoverageAgent.__new__(RoverSubspaceCoverageAgent)
    agent.__dict__.update(make_agent().__dict__)
    agent.obs_shape = (4,)
    agent.coverage_state_indices = (0, 1)
    obs = torch.tensor([[1., 0., .2, -.1], [0., 1., -.3, .2]])
    next_obs = obs+.25
    transitions = (obs, torch.tensor([[0], [1]]), torch.zeros(2,1),
                   torch.ones(2,1), next_obs)
    cached = agent._encode_actor_transition_batch(transitions)
    agent.policy_encoder.offset = 10.
    refreshed = agent._refresh_encoded_state_features(cached)
    assert not torch.allclose(cached['phi_obs'], refreshed['phi_obs'])
    torch.testing.assert_close(refreshed['coverage_next_raw'], next_obs[:,:2])
    torch.testing.assert_close(refreshed['next_obs_raw'], next_obs)
