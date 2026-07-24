import unittest

import gymnasium as gym
import numpy as np

from env.atari_domain import AtariScoreMaskWrapper


class DummyPixelEnv(gym.Env):
    observation_space = gym.spaces.Box(
        low=0,
        high=255,
        shape=(84, 84, 1),
        dtype=np.uint8,
    )
    action_space = gym.spaces.Discrete(2)


class AtariScoreMaskTest(unittest.TestCase):
    def test_montezuma_default_masks_score_and_lives_band(self):
        self.assertEqual(
            AtariScoreMaskWrapper.DEFAULT_BANDS["ALE/MontezumaRevenge-v5"],
            10,
        )

    def test_mask_replaces_only_configured_top_rows(self):
        wrapper = AtariScoreMaskWrapper(
            DummyPixelEnv(),
            band_height=10,
            color=0,
        )
        observation = np.full((84, 84, 1), 127, dtype=np.uint8)

        masked = wrapper.observation(observation)

        np.testing.assert_array_equal(masked[:10], 0)
        np.testing.assert_array_equal(masked[10:], 127)
        np.testing.assert_array_equal(observation, 127)


if __name__ == "__main__":
    unittest.main()
