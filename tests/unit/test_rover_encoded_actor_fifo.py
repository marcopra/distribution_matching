"""Unit tests for ROVER encoded actor FIFO sampling and visualization helpers.

These tests use temporary replay-buffer data and do not write persistent
artifacts; any future outputs should be placed under tests/outputs/.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from dm_env import specs

from agent.rover_nystrom import EncodedTransitionFIFO, EmbeddingDistributionVisualizerV2, RoverAgent
from replay_buffer import ReplayBuffer, ReplayBufferStorage


class FakeTimeStep:
    def __init__(self, observation, action, reward, discount=1.0, last=False):
        self._data = {
            "observation": np.asarray(observation, dtype=np.float32),
            "action": np.asarray(action, dtype=np.int64),
            "reward": np.asarray([reward], dtype=np.float32),
            "discount": np.asarray([discount], dtype=np.float32),
        }
        self._last = last

    def __getitem__(self, key):
        return self._data[key]

    def last(self):
        return self._last


def make_replay(tmp_path, save_snapshot=False):
    data_specs = (
        specs.Array((2,), np.float32, "observation"),
        specs.Array((), np.int64, "action"),
        specs.Array((1,), np.float32, "reward"),
        specs.Array((1,), np.float32, "discount"),
    )
    storage = ReplayBufferStorage(data_specs, tuple(), Path(tmp_path))
    replay = ReplayBuffer(
        storage,
        max_size=100,
        num_workers=1,
        nstep=1,
        discount=0.9,
        fetch_every=1000,
        save_snapshot=save_snapshot,
    )
    return storage, replay


def encoded(ids):
    ids = torch.as_tensor(ids, dtype=torch.float32).reshape(-1, 1)
    return {
        "phi_obs": ids.clone(),
        "phi_next": ids.clone() + 100,
        "psi": ids.clone() + 200,
        "E": torch.ones(ids.shape[0], 2),
        "reward": ids.clone() + 300,
    }


class EncodedActorFIFOTest(unittest.TestCase):
    def test_fifo_pins_first_transition_without_duplicate(self):
        fifo = EncodedTransitionFIFO(capacity=5)
        fifo.add(np.arange(6), encoded(np.arange(6)))

        self.assertEqual(len(fifo), 5)
        sample = fifo.sample(5, "cpu", include_first=True)
        self.assertEqual(float(sample["phi_obs"][0, 0]), 0.0)
        self.assertEqual((sample["phi_obs"] == 0).sum().item(), 1)

    def test_replay_stream_inserts_only_new_mid_episode_without_npz(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage, replay = make_replay(tmp, save_snapshot=False)
            storage.add(FakeTimeStep([0, 0], 0, 0.0), {})
            self.assertEqual(replay.get_new_transitions_since()[0], None)

            storage.add(FakeTimeStep([1, 0], 1, 1.0), {})
            ids, transitions = replay.get_new_transitions_since()
            np.testing.assert_array_equal(ids, np.array([0]))
            np.testing.assert_array_equal(transitions[0], np.array([[0, 0]], dtype=np.float32))
            np.testing.assert_array_equal(transitions[4], np.array([[1, 0]], dtype=np.float32))

            replay.mark_transitions_encoded(ids[-1])
            ids, transitions = replay.get_new_transitions_since(ids[-1])
            self.assertIsNone(ids)
            self.assertIsNone(transitions)

            storage.add(FakeTimeStep([2, 0], 0, 2.0, last=True), {})
            for fn in Path(tmp).glob("*.npz"):
                fn.unlink()
            ids, transitions = replay.get_new_transitions_since(0)
            np.testing.assert_array_equal(ids, np.array([1]))
            np.testing.assert_array_equal(transitions[0], np.array([[1, 0]], dtype=np.float32))
            np.testing.assert_array_equal(transitions[4], np.array([[2, 0]], dtype=np.float32))

    def test_replay_buffer_sampling_api_still_loads_completed_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage, replay = make_replay(tmp, save_snapshot=True)
            storage.add(FakeTimeStep([0, 0], 0, 0.0), {})
            storage.add(FakeTimeStep([1, 0], 1, 1.0), {})
            storage.add(FakeTimeStep([2, 0], 0, 2.0, last=True), {})

            all_data = replay.get_all_data()
            self.assertEqual(all_data[0].shape[0], 2)
            sample = replay._sample()
            self.assertEqual(len(sample), 5)

    def test_normal_actor_data_samples_first_at_index_zero(self):
        agent = RoverAgent.__new__(RoverAgent)
        agent.subsamples = None
        agent.batch_size_actor = 5
        agent.device = "cpu"
        agent._encoded_actor_fifo = EncodedTransitionFIFO(capacity=10)
        agent._encoded_actor_fifo.add(np.arange(10), encoded(np.arange(10)))
        agent._update_encoded_actor_fifo = lambda replay_buffer: True

        encoded_full, rewards, encoded_sub, sub_rewards = agent._get_actor_update_data(
            None, None, None, None, None, replay_buffer=object()
        )

        self.assertIsNone(encoded_sub)
        self.assertIsNone(sub_rewards)
        self.assertEqual(float(encoded_full["phi_obs"][0, 0]), 0.0)
        self.assertEqual(float(rewards[0, 0]), 300.0)
        self.assertEqual((encoded_full["phi_obs"] == 0).sum().item(), 1)

    def test_nystrom_actor_data_uses_all_fifo_and_subsamples_first_at_index_zero(self):
        agent = RoverAgent.__new__(RoverAgent)
        agent.subsamples = 4
        agent.batch_size_actor = 6
        agent.device = "cpu"
        agent._encoded_actor_fifo = EncodedTransitionFIFO(capacity=10)
        agent._encoded_actor_fifo.add(np.arange(10), encoded(np.arange(10)))
        agent._update_encoded_actor_fifo = lambda replay_buffer: True

        encoded_full, rewards, encoded_sub, sub_rewards = agent._get_actor_update_data(
            None, None, None, None, None, replay_buffer=object()
        )

        self.assertEqual(encoded_full["phi_obs"].shape[0], 10)
        self.assertEqual(float(encoded_full["phi_obs"][0, 0]), 0.0)
        self.assertEqual((encoded_full["phi_obs"] == 0).sum().item(), 1)
        self.assertEqual(encoded_sub["phi_obs"].shape[0], 4)
        self.assertEqual(float(encoded_sub["phi_obs"][0, 0]), 0.0)
        self.assertEqual(float(sub_rewards[0, 0]), 300.0)
        self.assertEqual((encoded_sub["phi_obs"] == 0).sum().item(), 1)

    def test_visualizer_initial_distribution_uses_subsample_alpha(self):
        agent = type("Agent", (), {})()
        agent.obs_type = "states"
        agent.encoder = torch.nn.Identity()
        agent.subsamples = 2
        agent._phi_all_next = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        agent._alpha = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
        agent._phi_sub_next = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        agent._sub_alpha = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

        visualizer = EmbeddingDistributionVisualizerV2.__new__(EmbeddingDistributionVisualizerV2)
        visualizer.agent = agent
        visualizer.n_states = 2
        visualizer.all_state_ids_one_hot = torch.eye(2, dtype=torch.float32)

        np.testing.assert_allclose(
            visualizer._compute_initial_distribution(),
            np.array([0.0, 1.0], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
