import unittest
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.utils import PointMazeNystromDebugHelper
from env.continuous_rooms import ContinuousCorridorEnv


class ContinuousCorridorEnvTest(unittest.TestCase):
    def make_env(self, **kwargs):
        defaults = dict(
            corridor_length=4.0,
            corridor_width=1.0,
            wall_thickness=0.3,
            agent_radius=0.15,
            max_velocity=0.3,
            goal_threshold=0.05,
            render_mode="rgb_array",
        )
        defaults.update(kwargs)
        return ContinuousCorridorEnv(**defaults)

    def test_default_start_is_corridor_center(self):
        env = self.make_env()
        obs, info = env.reset(seed=0)

        np.testing.assert_allclose(obs, env._center_position(), atol=1e-6)
        np.testing.assert_allclose(info["position"], env._center_position(), atol=1e-6)
        self.assertEqual(env.action_space.n, 2)

    def test_custom_start_position_is_used_and_validated(self):
        env = self.make_env(start_position=[1.0, 0.8])
        obs, _ = env.reset(seed=0)
        np.testing.assert_allclose(obs, np.array([1.0, env._center_y()], dtype=np.float32), atol=1e-6)

        with self.assertRaises(ValueError):
            self.make_env(start_position=[1.0, 0.9])

    def test_wall_collision_clips_continuously(self):
        env = self.make_env(start_position=[0.5, 0.8], max_velocity=0.3)
        env.reset(seed=0)
        min_x, _ = env._valid_x_bounds()

        obs, _, _, _, info = env.step(0)
        np.testing.assert_allclose(obs, np.array([min_x, env._center_y()], dtype=np.float32), atol=1e-6)
        self.assertTrue(info["wall_collision"])

        obs, _, _, _, info = env.step(1)
        np.testing.assert_allclose(obs, np.array([min_x + 0.3, env._center_y()], dtype=np.float32), atol=1e-6)
        self.assertFalse(info["wall_collision"])

    def test_horizon_alias_controls_truncation(self):
        env = self.make_env(horizon=3, max_steps=50, max_velocity=0.1)
        env.reset(seed=0)

        truncated_values = []
        for _ in range(3):
            _, _, terminated, truncated, _ = env.step(0)
            self.assertFalse(terminated)
            truncated_values.append(truncated)

        self.assertEqual(truncated_values, [False, False, True])
        self.assertEqual(env.max_steps, 3)
        self.assertEqual(env.horizon, 3)

    def test_fixed_debug_helper_builds_corridor_batch_without_point_env(self):
        env = self.make_env(render_resolution=32)
        env.reset(seed=0)

        class FakeAgent:
            obs_type = "pixels"
            obs_shape = (3, 32, 32)
            grayscale = False
            image_channels = 3
            device = "cpu"
            compute_dtype = torch.float32
            subsamples = None
            batch_size_actor = 6
            n_actions = 2

            def _kernel_status(self):
                return "test kernel"

        helper = PointMazeNystromDebugHelper(border_margin=0.0)
        helper.save_fixed_points_plot = lambda n_actions: None
        helper.attach_env(env)

        obs, action, reward, discount, next_obs = helper.build_subsample_batch(FakeAgent())

        self.assertEqual(tuple(obs.shape), (6, 3, 32, 32))
        self.assertEqual(tuple(next_obs.shape), (6, 3, 32, 32))
        self.assertEqual(tuple(reward.shape), (6, 1))
        self.assertEqual(tuple(discount.shape), (6, 1))
        self.assertEqual(torch.bincount(action, minlength=2).tolist(), [3, 3])
        min_x, max_x = env._valid_x_bounds()
        np.testing.assert_allclose(helper.fixed_xy_points[0], [min_x, env._center_y()], atol=1e-6)
        np.testing.assert_allclose(helper.fixed_xy_points[-1], [max_x, env._center_y()], atol=1e-6)
        self.assertTrue(np.all(np.diff(helper.fixed_xy_points[:, 0]) > 0.0))


if __name__ == "__main__":
    unittest.main()
