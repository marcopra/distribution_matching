import unittest

import gymnasium as gym
import numpy as np

from env.atari_domain import AtariActionSetWrapper, AtariScoreMaskWrapper


class DummyPixelEnv(gym.Env):
    observation_space = gym.spaces.Box(
        low=0,
        high=255,
        shape=(84, 84, 1),
        dtype=np.uint8,
    )
    action_space = gym.spaces.Discrete(2)


class DummyActionEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Discrete(18)

    def __init__(self):
        self.last_action = None

    def step(self, action):
        self.last_action = action
        return np.zeros(1, np.float32), 0.0, False, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, np.float32), {}

    def get_action_meanings(self):
        return [f"ACTION_{index}" for index in range(18)]


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


class AtariActionSetTest(unittest.TestCase):
    def test_restricted_index_maps_to_full_ale_action(self):
        env = DummyActionEnv()
        wrapper = AtariActionSetWrapper(env, [0, 1, 2, 3, 4, 5])

        self.assertEqual(wrapper.action_space.n, 6)
        wrapper.step(4)

        self.assertEqual(env.last_action, 4)
        self.assertEqual(
            wrapper.get_action_meanings(),
            [f"ACTION_{index}" for index in range(6)],
        )

    def test_non_contiguous_action_set_maps_by_position(self):
        env = DummyActionEnv()
        wrapper = AtariActionSetWrapper(env, [0, 2, 11, 12])

        wrapper.step(2)

        self.assertEqual(env.last_action, 11)

    def test_rejects_empty_duplicate_and_out_of_range_sets(self):
        env = DummyActionEnv()
        with self.assertRaises(ValueError):
            AtariActionSetWrapper(env, [])
        with self.assertRaises(ValueError):
            AtariActionSetWrapper(env, [0, 0])
        with self.assertRaises(ValueError):
            AtariActionSetWrapper(env, [18])


if __name__ == "__main__":
    unittest.main()
