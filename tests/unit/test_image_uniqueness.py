import unittest

import numpy as np
import torch

from agent.image_uniqueness import BoundedUniqueCounter, last_frame_fingerprints
from agent.rover_buffers import EncodedTransitionFIFO


class LastFrameFingerprintTests(unittest.TestCase):
    def test_uses_only_last_grayscale_frame(self):
        observations = np.zeros((3, 3, 4, 4), dtype=np.uint8)
        observations[0, :2] = 11
        observations[1, :2] = 99
        observations[2, :2] = 11
        observations[2, -1] = 7

        fingerprints = last_frame_fingerprints(observations, image_channels=1)

        self.assertEqual(fingerprints[0], fingerprints[1])
        self.assertNotEqual(fingerprints[0], fingerprints[2])

    def test_uses_last_rgb_frame(self):
        observations = np.zeros((2, 9, 3, 3), dtype=np.uint8)
        observations[0, :6] = 1
        observations[1, :6] = 2

        fingerprints = last_frame_fingerprints(observations, image_channels=3)

        self.assertEqual(fingerprints[0], fingerprints[1])


class BoundedUniqueCounterTests(unittest.TestCase):
    def test_exact_duplicates_and_mode_transition(self):
        counter = BoundedUniqueCounter(exact_limit=3, precision=10)

        self.assertEqual(counter.update([1, 1, 2]), 2)
        self.assertTrue(counter.is_exact)
        self.assertEqual(counter.update([3]), 3)
        self.assertTrue(counter.is_exact)
        self.assertGreaterEqual(counter.update([4]), 4)
        self.assertFalse(counter.is_exact)

    def test_hll_estimate_is_accurate_and_monotonic(self):
        counter = BoundedUniqueCounter(exact_limit=10, precision=14)
        fingerprints = np.random.default_rng(7).integers(
            np.iinfo(np.int64).min,
            np.iinfo(np.int64).max,
            size=20_000,
            dtype=np.int64,
        )
        counts = []
        for start in range(0, 20_000, 1_000):
            counts.append(counter.update(fingerprints[start:start + 1_000]))

        self.assertEqual(counts, sorted(counts))
        self.assertLess(abs(counts[-1] - 20_000) / 20_000, 0.03)

    def test_state_survives_torch_snapshot_round_trip(self):
        counter = BoundedUniqueCounter(exact_limit=2, precision=10)
        counter.update([1, 2, 3, 4])

        import io

        buffer = io.BytesIO()
        torch.save(counter, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=False)

        self.assertFalse(restored.is_exact)
        self.assertEqual(restored.count(), counter.count())


class FingerprintFifoAlignmentTests(unittest.TestCase):
    def test_fifo_sampling_keeps_hashes_aligned(self):
        fifo = EncodedTransitionFIFO(capacity=5)
        ids = np.arange(5, dtype=np.int64)
        encoded = {
            "phi_obs": torch.arange(5, dtype=torch.float32).reshape(-1, 1),
            "phi_next": torch.arange(5, dtype=torch.float32).reshape(-1, 1),
            "psi": torch.arange(5, dtype=torch.float32).reshape(-1, 1),
            "E": torch.ones((5, 1)),
            "reward": torch.zeros((5, 1)),
            "image_hash": torch.arange(100, 105, dtype=torch.int64),
        }
        fifo.add(ids, encoded)

        sampled = fifo.sample_by_strategy(
            5, "cpu", strategy="random", include_first=True
        )

        mapping = {
            int(phi): int(image_hash)
            for phi, image_hash in zip(
                sampled["phi_obs"].reshape(-1), sampled["image_hash"].reshape(-1)
            )
        }
        self.assertEqual(mapping, {index: 100 + index for index in range(5)})


if __name__ == "__main__":
    unittest.main()
