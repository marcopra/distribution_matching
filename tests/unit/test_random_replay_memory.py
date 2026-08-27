import tempfile
import unittest
from pathlib import Path

import numpy as np
from dm_env import specs

from replay_buffer_parallel import ReplayBuffer, ReplayBufferStorageParallel


class FakeTimeStep:
    def __init__(self, value, last=False):
        self._data = {
            "observation": np.full((4, 8, 8), value, dtype=np.uint8),
            "action": np.asarray(value % 4, dtype=np.int64),
            "reward": np.asarray([0.0], dtype=np.float32),
            "discount": np.asarray([1.0], dtype=np.float32),
        }
        self._last = last

    def __getitem__(self, key):
        return self._data[key]

    def last(self):
        return self._last


class RandomReplayMemoryTest(unittest.TestCase):
    def make_storage(self, path, retain_episodes):
        data_specs = (
            specs.Array((4, 8, 8), np.uint8, "observation"),
            specs.Array((), np.int64, "action"),
            specs.Array((1,), np.float32, "reward"),
            specs.Array((1,), np.float32, "discount"),
        )
        return ReplayBufferStorageParallel(
            data_specs,
            tuple(),
            Path(path),
            num_envs=2,
            retain_episodes=retain_episodes,
        )

    def test_counting_only_storage_does_not_retain_pixels_or_write_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = self.make_storage(tmp, retain_episodes=False)
            for step in range(10_000):
                storage.add(FakeTimeStep(step, last=step % 100 == 99), {}, step % 2)

            self.assertFalse(storage._transition_views)
            self.assertTrue(all(not episode for episode in storage._current_episodes))
            self.assertFalse(list(Path(tmp).glob("*.npz")))
            self.assertGreater(len(storage), 0)

    def test_first_transition_sampling_does_not_implicitly_register_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = self.make_storage(tmp, retain_episodes=True)
            ReplayBuffer(
                storage,
                max_size=100,
                num_workers=1,
                nstep=1,
                discount=0.99,
                fetch_every=1000,
                save_snapshot=False,
                first_transition=True,
                batch_size=2,
            )
            for step in range(100):
                storage.add(FakeTimeStep(step), {}, step % 2)

            self.assertFalse(storage._transition_views)


if __name__ == "__main__":
    unittest.main()
