import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from agent.rover_buffers import EncodedTransitionFIFO
from agent.rover_nystrom_pointmaze_debug import RoverAgent as PointMazeRoverAgent
from agent.rover_nystrom_gridworld_debug import GridWorldSyntheticData, RoverAgent
from agent.rover_visualization.gridworld import EmbeddingDistributionVisualizerV2
from sampling import sample_time_steps


def encoded(values):
    values = torch.as_tensor(values, dtype=torch.float32).reshape(-1, 1)
    return {
        "phi_obs": values,
        "phi_next": values + 10,
        "psi": values + 20,
        "E": torch.nn.functional.one_hot(values.long().reshape(-1) % 4, 4).float(),
        "reward": torch.zeros_like(values),
    }


class SamplingTest(unittest.TestCase):
    def test_vector_horizons_are_reproducible_and_bounded(self):
        horizons = np.array([0, 1, 3, 7, 12], dtype=np.int64)
        first = sample_time_steps(0.9, len(horizons), seed=7, horizon=horizons)
        second = sample_time_steps(0.9, len(horizons), seed=7, horizon=horizons)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first >= 0))
        self.assertTrue(np.all(first <= horizons))

    def test_vector_horizon_shape_is_validated(self):
        with self.assertRaisesRegex(ValueError, "num_samples"):
            sample_time_steps(0.9, 3, seed=0, horizon=np.array([1, 2]))


class TrajectoryFIFOTest(unittest.TestCase):
    def make_fifo(self, capacity=8):
        fifo = EncodedTransitionFIFO(capacity)
        fifo.add(
            np.arange(5),
            encoded(np.arange(5)),
            terminal_mask=np.array([False, True, False, False, True]),
        )
        return fifo

    def test_variable_horizon_gamma_sampling_is_exact_and_pins_first(self):
        fifo = self.make_fifo()
        np.random.seed(4)
        sample = fifo.sample_by_strategy(100, "cpu", "gamma_h", gamma=0.9)
        self.assertEqual(sample["phi_obs"].shape[0], 100)
        self.assertEqual(float(sample["phi_obs"][0]), 0.0)
        self.assertEqual(fifo.last_sampled_time_steps.shape, (100,))
        self.assertTrue(np.all(fifo.last_sampled_time_steps <= fifo.last_sampled_horizons))
        self.assertLess(np.unique(sample["phi_obs"].numpy()).size, 100)

    def test_reverse_uses_h_minus_same_gamma_draw(self):
        gamma_fifo = self.make_fifo()
        reverse_fifo = self.make_fifo()
        np.random.seed(11)
        gamma_fifo.sample_by_strategy(50, "cpu", "gamma_h", gamma=0.9)
        np.random.seed(11)
        reverse_fifo.sample_by_strategy(50, "cpu", "reverse_gamma_h", gamma=0.9)
        np.testing.assert_array_equal(
            gamma_fifo.last_sampled_trajectory_ids,
            reverse_fifo.last_sampled_trajectory_ids,
        )
        np.testing.assert_array_equal(
            gamma_fifo.last_sampled_time_steps[1:] + reverse_fifo.last_sampled_time_steps[1:],
            gamma_fifo.last_sampled_horizons[1:],
        )

    def test_fifo_eviction_keeps_metadata_aligned(self):
        fifo = self.make_fifo(capacity=4)
        self.assertEqual(len(fifo), 4)
        np.random.seed(2)
        sample = fifo.sample_by_strategy(20, "cpu", "gamma_h", gamma=0.9)
        self.assertEqual(sample["phi_obs"].shape[0], 20)
        self.assertTrue(set(sample["phi_obs"].reshape(-1).tolist()).issubset({0.0, 2.0, 3.0, 4.0}))

    def test_pivoted_cholesky_pins_first_and_removes_redundant_directions(self):
        fifo = EncodedTransitionFIFO(8)
        batch = encoded([0, 1, 2, 3, 4])
        batch["psi"] = torch.tensor(
            [[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0], [1.0, 1.0]]
        )
        fifo.add(np.arange(5), batch)

        torch.manual_seed(3)
        sample = fifo.sample_by_strategy(
            5,
            "cpu",
            "pivoted_cholesky",
            candidate_multiplier=1.0,
            cholesky_tolerance=1e-6,
        )

        self.assertEqual(float(sample["phi_obs"][0]), 0.0)
        self.assertEqual(sample["psi"].shape[0], 2)
        self.assertEqual(fifo.last_pivoted_cholesky_candidate_count, 5)
        self.assertEqual(fifo.last_pivoted_cholesky_residuals.shape, (2,))

    def test_pivoted_cholesky_candidate_pool_has_no_duplicate_rows(self):
        fifo = EncodedTransitionFIFO(20)
        values = np.arange(20)
        batch = encoded(values)
        batch["psi"] = torch.eye(20)
        fifo.add(values, batch)

        torch.manual_seed(5)
        sample = fifo.sample_by_strategy(
            6,
            "cpu",
            "pivoted_cholesky",
            candidate_multiplier=2.0,
            cholesky_tolerance=0.0,
        )

        self.assertEqual(sample["psi"].shape[0], 6)
        self.assertEqual(torch.unique(sample["phi_obs"]).numel(), 6)
        self.assertEqual(fifo.last_pivoted_cholesky_candidate_count, 12)

    def test_gaussian_pivoted_cholesky_fits_candidate_bandwidth(self):
        fifo = EncodedTransitionFIFO(8)
        batch = encoded([0, 0, 10, 10])
        batch["E"] = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(4, 1)
        fifo.add(np.arange(4), batch)

        sample = fifo.sample_by_strategy(
            4,
            "cpu",
            "pivoted_cholesky",
            candidate_multiplier=1.0,
            cholesky_tolerance=1e-6,
            kernel_type="gaussian",
            kernel_bandwidth=None,
            kernel_bandwidth_mult=0.5,
        )

        self.assertEqual(float(sample["phi_obs"][0]), 0.0)
        self.assertEqual(sample["phi_obs"].shape[0], 2)
        self.assertAlmostEqual(fifo.last_pivoted_cholesky_bandwidth, 5.0)

    def test_gaussian_pivoted_cholesky_masks_different_actions(self):
        fifo = EncodedTransitionFIFO(4)
        batch = encoded([1, 1])
        batch["E"] = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )
        fifo.add(np.arange(2), batch)

        sample = fifo.sample_by_strategy(
            2,
            "cpu",
            "pivoted_cholesky",
            candidate_multiplier=1.0,
            cholesky_tolerance=1e-6,
            kernel_type="gaussian",
            kernel_bandwidth=1.0,
        )

        self.assertEqual(sample["phi_obs"].shape[0], 2)
        self.assertEqual(torch.argmax(sample["E"], dim=1).tolist(), [0, 1])
        self.assertEqual(fifo.last_pivoted_cholesky_bandwidth, 1.0)


class PointMazeSamplingStrategyTest(unittest.TestCase):
    def make_agent(self):
        agent = PointMazeRoverAgent.__new__(PointMazeRoverAgent)
        agent.device = "cpu"
        agent.discount = 0.9
        agent.nystrom_candidate_multiplier = 1.0
        agent.nystrom_cholesky_tolerance = 1e-6
        agent.kernel_type = "inner_product"
        agent.kernel_bandwidth = None
        agent.kernel_bandwidth_mult = None
        agent._encoded_actor_fifo = EncodedTransitionFIFO(8)
        agent._encoded_actor_fifo.add(
            np.arange(5),
            encoded(np.arange(5)),
            terminal_mask=np.array([False, True, False, False, True]),
        )
        return agent

    def test_pointmaze_uses_all_fifo_sampling_strategies(self):
        expected_sizes = {
            "random": 4,
            "gamma_h": 4,
            "reverse_gamma_h": 4,
        }
        for strategy, expected_size in expected_sizes.items():
            with self.subTest(strategy=strategy):
                agent = self.make_agent()
                agent.subsampling_strategy = strategy
                sampled, _ = agent._sample_encoded_actor_data(4, include_first=True)
                self.assertEqual(sampled["phi_obs"].shape[0], expected_size)
                self.assertEqual(float(sampled["phi_obs"][0]), 0.0)

        agent = self.make_agent()
        agent.subsampling_strategy = "pivoted_cholesky"
        sampled, _ = agent._sample_encoded_actor_data(4, include_first=True)
        self.assertGreaterEqual(sampled["phi_obs"].shape[0], 1)
        self.assertLessEqual(sampled["phi_obs"].shape[0], 4)
        self.assertEqual(float(sampled["phi_obs"][0]), 0.0)

    def test_pointmaze_fifo_update_records_episode_boundaries(self):
        agent = PointMazeRoverAgent.__new__(PointMazeRoverAgent)
        agent._encoded_actor_fifo = EncodedTransitionFIFO(8)
        agent._encoded_fifo_replay_marker = None
        agent.encoded_fifo_encode_batch_size = 8
        agent._sync_policy_encoder = lambda: None
        agent._insert_first_transition_if_available = lambda replay: None
        agent._encode_actor_transition_batch_with_retries = (
            lambda transitions: encoded(np.arange(transitions[0].shape[0]))
        )

        ids = np.arange(5)
        transitions = (
            np.zeros((5, 1), dtype=np.float32),
            np.zeros((5, 1), dtype=np.int64),
            np.zeros((5, 1), dtype=np.float32),
            np.array([[1.0], [0.0], [1.0], [1.0], [0.0]], dtype=np.float32),
            np.zeros((5, 1), dtype=np.float32),
        )

        class Replay:
            marked = None

            def get_new_transitions_since(self, marker, limit=None):
                del limit
                return (ids, transitions) if marker is None else (None, None)

            def mark_transitions_encoded(self, marker):
                self.marked = marker

        replay = Replay()
        self.assertTrue(agent._update_encoded_actor_fifo(replay))
        _, trajectory_ids, _ = agent._encoded_actor_fifo._all_with_trajectory_metadata()
        self.assertEqual(trajectory_ids.tolist(), [0, 0, 1, 1, 1])
        self.assertEqual(replay.marked, 4)


class FakeGridWorld:
    n_states = 3
    idx_to_state = {0: (0, 0), 1: (1, 0), 2: (2, 0)}
    state_to_idx = {(0, 0): 0, (1, 0): 1, (2, 0): 2}
    start_position = (0, 0)

    def step_from(self, cell, action):
        if action == 3:
            return (min(cell[0] + 1, 2), 0)
        if action == 2:
            return (max(cell[0] - 1, 0), 0)
        return cell


class FakeAgent:
    n_actions = 4
    n_states = 3
    obs_type = "states"
    obs_shape = (3,)
    grayscale = False
    device = "cpu"
    compute_dtype = torch.float32
    batch_size = 20
    subsamples = 5
    synthetic_dataset_exclude_state_idxs = [1]
    synthetic_dataset_exclusion_mode = "one_action"
    synthetic_dataset_excluded_action = 0
    synthetic_subsample_exclude_state_idxs = [2]
    synthetic_subsample_exclusion_mode = "all_actions"
    synthetic_subsample_excluded_action = 0

    @staticmethod
    def _make_actor_batch(obs, action, next_obs, reward):
        return obs, action, next_obs, reward

    @staticmethod
    def _slice_actor_batch(batch, index):
        return tuple(field[index] for field in batch)

    def _nystrom_subsample_count(self):
        return self.subsamples


class SyntheticGridWorldTest(unittest.TestCase):
    def make_helper(self):
        helper = GridWorldSyntheticData(FakeAgent())
        helper.attach_env(FakeGridWorld())
        return helper

    def test_separate_exclusions_and_exact_subsample_size(self):
        helper = self.make_helper()
        full = helper.full_actor_batch()
        subsample = helper.subsample_actor_batch()
        self.assertEqual(full[0].shape[0], 11)
        self.assertEqual(subsample[0].shape[0], 8)
        self.assertEqual(int(torch.argmax(full[0][0])), 0)
        torch.testing.assert_close(full[0][0], full[2][0])
        state_one_actions = full[1][torch.argmax(full[0], dim=1) == 1]
        self.assertNotIn(0, state_one_actions.tolist())
        self.assertNotIn(2, torch.argmax(subsample[0], dim=1).tolist())

    def test_synthetic_subsamples_ignore_requested_count(self):
        agent = FakeAgent()
        agent.subsamples = 1
        helper = GridWorldSyntheticData(agent)
        helper.attach_env(FakeGridWorld())
        # State 2 is excluded with all four actions, leaving 2 states x 4 actions.
        self.assertEqual(helper.subsample_actor_batch()[0].shape[0], 8)

    def test_all_action_exclusion_rejects_initial_state(self):
        agent = FakeAgent()
        agent.synthetic_dataset_exclude_state_idxs = [0]
        agent.synthetic_dataset_exclusion_mode = "all_actions"
        helper = GridWorldSyntheticData(agent)
        helper.attach_env(FakeGridWorld())
        with self.assertRaisesRegex(ValueError, "initial state"):
            helper.full_actor_batch()

    def test_omegaconf_exclusion_list_is_supported(self):
        agent = FakeAgent()
        agent.synthetic_dataset_exclude_state_idxs = OmegaConf.create([1])
        helper = GridWorldSyntheticData(agent)
        helper.attach_env(FakeGridWorld())
        self.assertEqual(helper.full_actor_batch()[0].shape[0], 11)

    def test_histogram_plot_smoke(self):
        agent = RoverAgent.__new__(RoverAgent)
        agent.subsampling_strategy = "gamma_h"
        agent._encoded_actor_fifo = type(
            "FIFO",
            (),
            {"last_sampled_time_steps": np.array([0, 1, 1, 2])},
        )()
        visualizer = type(
            "Visualizer",
            (),
            {
                "n_states": 3,
                "n_actions": 4,
                "action_colors": ["red", "blue", "green", "orange"],
                "_compute_batch_and_subsample_state_counts": lambda self: (
                    np.array([2, 3, 1]), np.array([1, 2, 0])
                ),
                "_compute_batch_and_subsample_state_action_counts": lambda self: (
                    np.ones((4, 3)), np.eye(4, 3)
                ),
            },
        )()
        agent.debug_visualizer = type(
            "Suite",
            (),
            {"domain_visualizer": type("Adapter", (), {"visualizer": visualizer})()},
        )()
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                agent._save_gridworld_histograms(10)
                self.assertTrue(os.path.exists("gridworld_plots/step_10_dataset_histograms.png"))
            finally:
                os.chdir(old_cwd)

    def test_gaussian_candidate_bandwidth_is_reused_by_actor_update(self):
        agent = RoverAgent.__new__(RoverAgent)
        agent.subsampling_strategy = "pivoted_cholesky"
        agent.kernel_type = "gaussian"
        agent._encoded_actor_fifo = type(
            "FIFO", (), {"last_pivoted_cholesky_bandwidth": 2.5}
        )()
        agent.kernel_fn = type("Kernel", (), {"bandwidth": None})()
        matcher_kernel = type("Kernel", (), {"bandwidth": None})()
        agent.distribution_matcher = type(
            "Matcher", (), {"kernel_fn": matcher_kernel}
        )()

        agent._fit_state_kernel_bandwidth(None, None)

        self.assertEqual(agent.kernel_fn.bandwidth, 2.5)
        self.assertEqual(agent.distribution_matcher.kernel_fn.bandwidth, 2.5)

    def test_dataset_subsample_and_policy_heatmap_smoke(self):
        visualizer = EmbeddingDistributionVisualizerV2.__new__(EmbeddingDistributionVisualizerV2)
        visualizer.n_states = 4
        visualizer.n_actions = 4
        visualizer.grid_width = 2
        visualizer.grid_height = 2
        visualizer.action_names = ["Up", "Down", "Left", "Right"]
        visualizer.state_adapter = type(
            "Adapter",
            (),
            {"values_to_grid": lambda self, values, reduce="sum": np.asarray(values).reshape(2, 2)},
        )()
        visualizer._compute_batch_and_subsample_state_counts = lambda: (
            np.array([4, 3, 2, 1], dtype=np.float32),
            np.array([1, 0, 2, 1], dtype=np.float32),
        )
        visualizer._get_policy_per_state = lambda: np.array(
            [
                [0.7, 0.1, 0.1, 0.1],
                [0.1, 0.7, 0.1, 0.1],
                [0.1, 0.1, 0.7, 0.1],
                [0.1, 0.1, 0.1, 0.7],
            ],
            dtype=np.float32,
        )
        visualizer._plot_policy_bars_per_cell = (
            lambda ax, policy, annotate_probabilities=False: ax.text(
                0.5, 0.5, f"bars={annotate_probabilities}", transform=ax.transAxes
            )
        )
        visualizer._plot_policy_arrows = (
            lambda ax, policy: ax.text(0.5, 0.5, "arrows", transform=ax.transAxes)
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "heatmaps.png")
            returned = visualizer.save_dataset_subsample_policy_heatmaps(20, output)
            self.assertEqual(returned, output)
            self.assertTrue(os.path.exists(output))

    def test_actor_occupancy_counts_current_not_next_state(self):
        agent = type("Agent", (), {})()
        agent.obs_type = "states"
        agent.device = "cpu"
        agent.encoder = torch.nn.Identity()
        agent.subsamples = 2
        # Last column is actor augmentation; source states differ from next states.
        agent._phi_all_obs = torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32
        )
        agent._phi_all_next = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        )
        agent._phi_sub_obs = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        )
        agent._phi_sub_next = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32
        )
        agent._all_actions = torch.tensor([0, 1])
        agent._sub_actions = torch.tensor([1, 1])
        visualizer = EmbeddingDistributionVisualizerV2.__new__(EmbeddingDistributionVisualizerV2)
        visualizer.agent = agent
        visualizer.n_states = 2
        visualizer.n_actions = 4
        visualizer.all_state_ids_one_hot = torch.eye(2)
        visualizer._prerendered_states = None

        full_counts, subsample_counts = visualizer._compute_batch_and_subsample_state_counts()
        np.testing.assert_array_equal(full_counts, np.array([2.0, 0.0]))
        np.testing.assert_array_equal(subsample_counts, np.array([1.0, 1.0]))
        full_sa, subsample_sa = visualizer._compute_batch_and_subsample_state_action_counts()
        np.testing.assert_array_equal(
            full_sa,
            np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        )
        np.testing.assert_array_equal(
            subsample_sa,
            np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]),
        )


class GridWorldConfigTest(unittest.TestCase):
    def test_four_rooms_preset_composes(self):
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
            cfg = compose(config_name="pretrain/pretrain_rover_nystrom_four_rooms_debug")
        self.assertEqual(
            cfg.agent._target_,
            "agent.rover_nystrom_gridworld_debug.RoverAgent",
        )
        self.assertEqual(cfg.env.name, "FourRooms-v0")
        self.assertIn(
            cfg.agent.subsampling_strategy,
            ("random", "gamma_h", "reverse_gamma_h", "pivoted_cholesky"),
        )
        self.assertGreaterEqual(cfg.agent.nystrom_candidate_multiplier, 1.0)
        self.assertGreaterEqual(cfg.agent.nystrom_cholesky_tolerance, 0.0)


if __name__ == "__main__":
    unittest.main()
