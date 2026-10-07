"""Regression for changing state encoders with an encoded actor FIFO."""
import numpy as np
import pytest
import torch
from agent.rover_buffers import EncodedTransitionFIFO
from agent.rover_nystrom_debug import RoverAgent
from agent.rover_nystrom_subspace import RoverSubspaceCoverageAgent


class ChangingEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("offset", torch.tensor(1.))

    def encode_and_project(self, obs, normalize=True):
        obs = obs.reshape(obs.shape[0], -1).float()
        raw = torch.stack([obs[:, 0].abs()+self.offset, obs[:, 1].abs()+1], 1)
        return torch.nn.functional.normalize(raw, p=1, dim=1) if normalize else raw


def make_agent():
    agent = RoverAgent.__new__(RoverAgent)
    agent.obs_type = "states"
    agent.embeddings = True
    agent.feature_learning_loss = "infonce"
    agent.device = "cpu"
    agent.aug = torch.nn.Identity()
    agent.encoder = ChangingEncoder()
    agent.policy_encoder = ChangingEncoder()
    agent.encoded_fifo_encode_batch_size = 2
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
    agent.encoder.offset.fill_(10.)

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
    agent.policy_encoder.offset.fill_(10.)
    refreshed = agent._refresh_encoded_state_features(cached)
    assert not torch.allclose(cached['phi_obs'], refreshed['phi_obs'])
    torch.testing.assert_close(refreshed['coverage_next_raw'], next_obs[:,:2])
    torch.testing.assert_close(refreshed['next_obs_raw'], next_obs)


class QueuedReplay:
    def __init__(self, ids=None, transitions=None):
        self.pending = None if ids is None else (ids, transitions)
        self.acknowledged = []

    def get_new_transitions_since(self, marker, limit):
        if self.pending is None:
            return None, None
        result, self.pending = self.pending, None
        return result

    def mark_transitions_encoded(self, marker):
        self.acknowledged.append(marker)


@pytest.mark.parametrize("obs_type", ["states", "pixels"])
def test_fifo_drain_keeps_acting_encoder_until_actor_support_is_refreshed(obs_type):
    agent = make_agent()
    del agent._update_encoded_actor_fifo  # Exercise real replay drain.
    agent.obs_type = obs_type
    agent.batch_size_actor = 4
    agent._encoded_fifo_replay_marker = None
    obs = torch.tensor([[1., 0.], [0., 1.], [2., 1.], [1., 2.]])
    if obs_type == "pixels":
        obs = obs.to(torch.uint8).reshape(4, 1, 1, 2).repeat(1, 3, 2, 1)
    transitions = (obs, torch.tensor([[0], [1], [0], [1]]), torch.zeros(4, 1),
                   torch.ones(4, 1), obs + 1)
    initial = agent._encode_actor_transition_batch(tuple(x[:2] for x in transitions))
    agent._encoded_actor_fifo.add(np.arange(2), initial)
    pinned = initial["phi_obs"][:1].clone()
    agent._phi_all_obs = initial["phi_obs"].clone()
    agent.gradient_coeff = torch.ones(3, 1)
    agent.encoder.offset.fill_(10.)
    replay = QueuedReplay(np.arange(2, 4), tuple(x[2:] for x in transitions))

    assert agent.drain_encoded_actor_fifo(replay)
    torch.testing.assert_close(agent.policy_encoder.offset, torch.tensor(1.))
    torch.testing.assert_close(agent.policy_encoder.encode_and_project(obs[:1]), pinned)
    torch.testing.assert_close(agent._phi_all_obs, initial["phi_obs"])
    torch.testing.assert_close(agent.gradient_coeff, torch.ones(3, 1))
    assert replay.acknowledged == [3]

    # No new replay is required for the next actor fit to refresh old support.
    data = agent._encoded_fifo_actor_update_data(replay)
    torch.testing.assert_close(agent.policy_encoder.offset, agent.encoder.offset)
    for batch in (data.full, data.subsample):
        torch.testing.assert_close(batch["phi_obs"],
            agent.policy_encoder.encode_and_project(batch["obs_raw"]))
        torch.testing.assert_close(batch["phi_next"],
            agent.policy_encoder.encode_and_project(batch["next_obs_raw"]))
        torch.testing.assert_close(batch["obs_raw"][:1], obs[:1])
        assert batch["obs_raw"].device.type == "cpu"
        if obs_type == "pixels":
            assert batch["obs_raw"].dtype == torch.uint8
            assert batch["next_obs_raw"].dtype == torch.uint8
    assert not torch.allclose(data.full["phi_obs"][:1], pinned)
    agent.encoder.offset.fill_(20.)
    assert agent.drain_encoded_actor_fifo(replay)
    torch.testing.assert_close(agent.policy_encoder.offset, torch.tensor(10.))


def test_legacy_pixel_fifo_cannot_resume_training_without_raw_observations():
    agent = make_agent()
    del agent._update_encoded_actor_fifo
    agent.obs_type = "pixels"
    agent._encoded_actor_fifo.add(np.arange(2), {
        "phi_obs": torch.ones(2, 2), "phi_next": torch.ones(2, 2),
        "action": torch.zeros(2, 1, dtype=torch.long), "reward": torch.zeros(2, 1),
    })
    with pytest.raises(RuntimeError, match="Start a fresh training run"):
        agent.drain_encoded_actor_fifo(QueuedReplay())


def test_pixel_support_refresh_is_chunked_and_preserves_cached_coverage():
    agent = make_agent()
    agent.obs_type = "pixels"
    observations = torch.arange(36, dtype=torch.uint8).reshape(3, 3, 2, 2)
    transitions = (observations, torch.tensor([[0], [1], [0]]), torch.zeros(3, 1),
                   torch.ones(3, 1), observations + 1)
    encoded = agent._encode_actor_transition_batch(transitions)
    encoded["coverage_next_features"] = torch.randn(3, 4)
    original_encode = agent._encode_actor_transition_batch
    sizes = []

    def bounded_encode(batch):
        sizes.append(len(batch[0]))
        return original_encode(batch)

    agent._encode_actor_transition_batch = bounded_encode
    agent.policy_encoder.offset.fill_(10.)
    refreshed = agent._refresh_encoded_state_features(encoded)
    assert sizes == [2, 1]
    assert refreshed["obs_raw"] is encoded["obs_raw"]
    assert refreshed["coverage_next_features"] is encoded["coverage_next_features"]
    torch.testing.assert_close(refreshed["phi_obs"],
        agent.policy_encoder.encode_and_project(observations))
