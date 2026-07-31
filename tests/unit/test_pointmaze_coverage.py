import unittest

import numpy as np

from agent.rover_visualization.domains import (
    CoverageProgress,
    pointmaze_evaluation_seed,
    pointmaze_free_space_coverage,
)
from agent.rover_visualization.suite import RoverDebugVisualizerSuite


class PointMazeFakeEnv:
    def get_debug_maze_layout(self):
        return {
            "maze_lower": np.asarray([0.0, 0.0], dtype=np.float32),
            "maze_upper": np.asarray([1.0, 1.0], dtype=np.float32),
            "wall_rectangles": np.asarray([[0.4, 0.4, 0.2, 0.2]], dtype=np.float32),
        }


class PointMazeCoverageTest(unittest.TestCase):
    def test_free_space_coverage_excludes_walls(self):
        covered, free, percentage = pointmaze_free_space_coverage(
            PointMazeFakeEnv(),
            [np.asarray([[0.0, 0.0]], dtype=np.float32)],
            grid_size=3,
            radius=0.01,
        )
        self.assertEqual((covered, free), (1, 8))
        self.assertAlmostEqual(percentage, 12.5)

    def test_coverage_rejects_empty_and_invalid_inputs(self):
        env = PointMazeFakeEnv()
        with self.assertRaisesRegex(ValueError, "valid trajectory"):
            pointmaze_free_space_coverage(env, [], grid_size=3, radius=0.1)
        with self.assertRaisesRegex(ValueError, "grid_size"):
            pointmaze_free_space_coverage(env, [[[0.0, 0.0]]], grid_size=1, radius=0.1)
        with self.assertRaisesRegex(ValueError, "radius"):
            pointmaze_free_space_coverage(env, [[[0.0, 0.0]]], grid_size=3, radius=0.0)

    def test_progress_logs_delta_best_gain_and_expansion(self):
        progress = CoverageProgress(tolerance=0.25)
        first = progress.update(10.0, 0)
        second = progress.update(10.2, 100_000)
        third = progress.update(11.0, 200_000)
        fourth = progress.update(10.5, 300_000)

        self.assertEqual(first["coverage_delta"], 0.0)
        self.assertEqual(second["coverage_expanding"], 0.0)
        self.assertAlmostEqual(third["coverage_gain_per_100k"], 0.8)
        self.assertEqual(third["coverage_expanding"], 1.0)
        self.assertEqual(fourth["coverage_best"], 11.0)

    def test_evaluation_seed_is_deterministic_and_checkpoint_specific(self):
        self.assertEqual(
            pointmaze_evaluation_seed(3, 100_000, 7),
            pointmaze_evaluation_seed(3, 100_000, 7),
        )
        self.assertNotEqual(
            pointmaze_evaluation_seed(3, 100_000, 7),
            pointmaze_evaluation_seed(3, 200_000, 7),
        )

    def test_pointmaze_actor_update_domain_visualizer_is_disabled(self):
        suite = RoverDebugVisualizerSuite(
            agent=object(),
            exploration_visualizer=None,
            gridworld_visualizer_factory=lambda agent: self.fail("unexpected gridworld visualizer"),
        )
        self.assertIsNone(suite.attach_env(PointMazeFakeEnv()))


if __name__ == "__main__":
    unittest.main()
