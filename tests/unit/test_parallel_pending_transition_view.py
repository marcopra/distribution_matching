import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from dm_env import specs

from replay_buffer_parallel import ReplayBuffer, ReplayBufferStorageParallel


class FakeTimeStep:
    def __init__(self, value, last=False):
        self._data = {
            "observation": np.full((3, 8, 8), value, dtype=np.uint8),
            "action": np.asarray(value % 4, dtype=np.int64),
            "reward": np.asarray([float(value)], dtype=np.float32),
            "discount": np.asarray([1.0], dtype=np.float32),
        }
        self._last = last

    def __getitem__(self, key):
        return self._data[key]

    def last(self):
        return self._last


def make_storage(path, num_envs=1):
    data_specs = (
        specs.Array((3, 8, 8), np.uint8, "observation"),
        specs.Array((), np.int64, "action"),
        specs.Array((1,), np.float32, "reward"),
        specs.Array((1,), np.float32, "discount"),
    )
    return ReplayBufferStorageParallel(
        data_specs,
        tuple(),
        Path(path),
        num_envs=num_envs,
        retain_episodes=True,
    )


def make_replay(storage, **kwargs):
    return ReplayBuffer(
        storage,
        max_size=100,
        num_workers=1,
        nstep=1,
        discount=0.99,
        fetch_every=1000,
        save_snapshot=False,
        transition_view=True,
        **kwargs,
    )


class FakeReplayDataset:
    def __init__(self, pending_count):
        self.pending_count = pending_count

    def pending_transition_count(self):
        return self.pending_count


class FakeReplayLoader:
    def __init__(self, dataset):
        self.dataset = dataset


class FakeAgent:
    update_every_steps = 5
    max_pending_transitions = 100

    def __init__(self):
        self.drain_calls = 0

    def drain_encoded_actor_fifo(self, replay_dataset):
        self.drain_calls += 1
        replay_dataset.pending_count = 0
        return True


class FakeMetricLogger:
    def __init__(self):
        self.buffered = []
        self.immediate = []

    def log_metrics(self, metrics, step, ty):
        self.buffered.append((metrics, step, ty))

    def log_immediate_metrics(self, metrics, step, ty):
        self.immediate.append((metrics, step, ty))


class FakeMetricAgent:
    def update(self, replay_iter, step):
        return {
            "actor_loss": 1.0,
            "unique_images_actor_batch": 17,
            "unique_images_subsamples": 11,
        }


class FakeDrainMetricAgent(FakeAgent):
    unique_images_all_training = 23


class ParallelPendingTransitionViewTest(unittest.TestCase):
    def test_end_episode_stores_partial_stream_without_fake_transition(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = make_storage(tmp)
            storage.add(FakeTimeStep(0), {})
            storage.add(FakeTimeStep(1), {})
            storage.add(FakeTimeStep(2), {})

            storage.end_episode(0)

            self.assertEqual(len(storage), 2)
            episode_files = list(Path(tmp).glob('*.npz'))
            self.assertEqual(len(episode_files), 1)
            with np.load(episode_files[0]) as episode:
                np.testing.assert_array_equal(
                    episode['observation'][:, 0, 0, 0],
                    np.array([0, 1, 2], dtype=np.uint8),
                )

    def test_pending_transition_count_stays_bounded_by_hard_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = make_storage(tmp)
            replay = make_replay(
                storage,
                max_pending_transitions=3,
                drop_oldest_pending_on_overflow=True,
            )

            storage.add(FakeTimeStep(0), {})
            for step in range(1, 10):
                storage.add(FakeTimeStep(step), {})
                self.assertLessEqual(replay.pending_transition_count(), 3)

            ids, _ = replay.get_new_transitions_since()
            np.testing.assert_array_equal(ids, np.array([6, 7, 8], dtype=np.int64))

    def test_mark_transitions_encoded_discards_raw_pending_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = make_storage(tmp)
            replay = make_replay(storage)

            storage.add(FakeTimeStep(0), {})
            for step in range(1, 5):
                storage.add(FakeTimeStep(step), {})
            self.assertEqual(replay.pending_transition_count(), 4)

            ids, transitions = replay.get_new_transitions_since(limit=2)
            self.assertEqual(transitions[0].shape[0], 2)
            replay.mark_transitions_encoded(ids[-1])
            self.assertEqual(replay.pending_transition_count(), 2)

            ids, _ = replay.get_new_transitions_since(ids[-1])
            np.testing.assert_array_equal(ids, np.array([2, 3], dtype=np.int64))

    def test_seed_collection_drain_runs_on_update_every_steps(self):
        from pretrain_parallel import Workspace

        workspace = Workspace.__new__(Workspace)
        workspace.agent_requires_replay = True
        workspace.agent = FakeAgent()
        workspace.replay_loader = FakeReplayLoader(FakeReplayDataset(pending_count=50))

        drained = workspace._drain_encoded_actor_fifo_if_due([1, 2, 3, 4, 5])

        self.assertTrue(drained)
        self.assertEqual(workspace.agent.drain_calls, 1)
        self.assertEqual(workspace.replay_loader.dataset.pending_transition_count(), 0)

    def test_actor_unique_metrics_log_immediately(self):
        from pretrain_parallel import Workspace

        workspace = Workspace.__new__(Workspace)
        workspace.agent = FakeMetricAgent()
        workspace.replay_loader = None
        workspace._replay_iter = None
        workspace._global_step = 7
        workspace.cfg = SimpleNamespace(action_repeat=4)
        workspace.logger = FakeMetricLogger()

        workspace._update_agent_once(logical_step=7)

        self.assertEqual(
            workspace.logger.buffered,
            [({"actor_loss": 1.0}, 28, "train")],
        )
        self.assertEqual(
            workspace.logger.immediate,
            [({
                "unique_images_actor_batch": 17,
                "unique_images_subsamples": 11,
            }, 28, "train")],
        )

    def test_lifetime_unique_metric_logs_after_fifo_drain(self):
        from pretrain_parallel import Workspace

        workspace = Workspace.__new__(Workspace)
        workspace.agent_requires_replay = True
        workspace.agent = FakeDrainMetricAgent()
        workspace.replay_loader = FakeReplayLoader(FakeReplayDataset(pending_count=5))
        workspace._global_step = 9
        workspace.cfg = SimpleNamespace(action_repeat=4)
        workspace.logger = FakeMetricLogger()

        workspace._drain_encoded_actor_fifo_if_due([5])

        self.assertEqual(
            workspace.logger.immediate,
            [({"unique_images_all_training": 23}, 36, "train")],
        )

    def test_random_and_non_nystrom_paths_do_not_register_transition_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = make_storage(tmp)
            ReplayBuffer(
                storage,
                max_size=100,
                num_workers=1,
                nstep=1,
                discount=0.99,
                fetch_every=1000,
                save_snapshot=False,
                transition_view=False,
            )
            storage.add(FakeTimeStep(0), {})
            storage.add(FakeTimeStep(1), {})
            self.assertFalse(storage._transition_views)


if __name__ == "__main__":
    unittest.main()
