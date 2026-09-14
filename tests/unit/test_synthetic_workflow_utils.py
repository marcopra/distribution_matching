import unittest

import numpy as np

from tests.diagnostics.pointmaze.synthetic_workflow_utils import (
    shuffled_encoder_index_batches,
)


class ShuffledEncoderIndexBatchesTest(unittest.TestCase):
    def test_each_epoch_visits_every_transition_once(self):
        batches = list(shuffled_encoder_index_batches(10, 4, updates=6, seed=7))

        np.testing.assert_array_equal(np.sort(np.concatenate(batches[:3])), np.arange(10))
        np.testing.assert_array_equal(np.sort(np.concatenate(batches[3:])), np.arange(10))
        self.assertEqual([len(batch) for batch in batches], [4, 4, 2, 4, 4, 2])

    def test_seed_is_reproducible_but_epochs_are_reshuffled(self):
        first = list(shuffled_encoder_index_batches(12, 4, updates=6, seed=3))
        second = list(shuffled_encoder_index_batches(12, 4, updates=6, seed=3))

        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        self.assertFalse(np.array_equal(np.concatenate(first[:3]), np.concatenate(first[3:])))

    def test_rejects_invalid_sizes(self):
        with self.assertRaises(ValueError):
            next(shuffled_encoder_index_batches(0, 4, updates=1, seed=0))
        with self.assertRaises(ValueError):
            next(shuffled_encoder_index_batches(4, 0, updates=1, seed=0))


if __name__ == "__main__":
    unittest.main()
