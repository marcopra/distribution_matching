import unittest
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import torch
import utils

from agent.rover_buffers import EncodedTransitionFIFO
from agent.rover_nystrom_debug import RoverAgent
from agent.rover_nystrom_subspace import (
    RoverSubspaceCoverageAgent,
    _SimplexStateKernel,
)


class _PixelProjector(torch.nn.Module):
    def encode_and_project(self, observation):
        return observation.float().reshape(observation.shape[0], -1)


class _PixelsToTwoFeatures(torch.nn.Module):
    def encode_and_project(self, observation):
        flat = observation.float().reshape(observation.shape[0], -1)
        return torch.stack((flat[:, 0] + 1.0, flat[:, 1] + 1.0), dim=1)


class RoverPixelProprioStateTest(unittest.TestCase):
    def test_replay_metadata_uses_coordinates_from_stored_time_step(self):
        agent = RoverAgent.__new__(RoverAgent)
        agent.proprio_state_dims = 2
        time_step = SimpleNamespace(
            proprio_observation=np.asarray([1.25, -0.75, 9.0], dtype=np.float32)
        )
        policy_meta = OrderedDict(proprio_state=np.zeros(2, dtype=np.float32))

        replay_meta = agent.prepare_replay_meta(time_step, policy_meta)

        np.testing.assert_array_equal(
            replay_meta["proprio_state"],
            np.asarray([1.25, -0.75], dtype=np.float32),
        )
        np.testing.assert_array_equal(policy_meta["proprio_state"], np.zeros(2))

    def test_pixel_encoding_preserves_coordinate_sidecar(self):
        agent = RoverAgent.__new__(RoverAgent)
        agent.device = "cpu"
        agent.obs_type = "pixels"
        agent.embeddings = True
        agent.aug = torch.nn.Identity()
        agent.policy_encoder = _PixelProjector()

        observations = np.arange(12, dtype=np.uint8).reshape(3, 1, 2, 2)
        coordinates = np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
        transitions = (
            observations,
            np.asarray([0, 1, 0], dtype=np.int64),
            np.zeros((3, 1), dtype=np.float32),
            np.ones((3, 1), dtype=np.float32),
            observations + 1,
            coordinates,
        )

        encoded = agent._encode_actor_transition_batch(transitions)

        torch.testing.assert_close(encoded["debug_xy"], torch.from_numpy(coordinates))
        torch.testing.assert_close(encoded["obs_raw"], torch.from_numpy(observations))
        self.assertEqual(encoded["obs_raw"].dtype, torch.uint8)

    def test_nystrom_selection_keeps_coordinates_aligned(self):
        fifo = EncodedTransitionFIFO(capacity=4)
        ids = torch.arange(4, dtype=torch.float32).reshape(-1, 1)
        encoded = {
            "phi_obs": ids.clone(),
            "phi_next": ids.clone(),
            "action": torch.zeros((4, 1), dtype=torch.long),
            "reward": torch.zeros((4, 1)),
            "debug_xy": torch.cat((ids, -ids), dim=1),
        }
        fifo.add(np.arange(4), encoded)

        selected = fifo.sample_by_strategy(
            3,
            "cpu",
            strategy="pivoted_cholesky",
            include_first=True,
            candidate_multiplier=2.0,
            cholesky_tolerance=0.0,
            kernel_type="inner_product",
            cholesky_progress=False,
        )

        torch.testing.assert_close(selected["debug_xy"][:, :1], selected["phi_obs"])
        torch.testing.assert_close(selected["debug_xy"][:, 1:], -selected["phi_obs"])

    def test_state_kernel_round_trips_four_coordinates_and_ignores_sink_columns(self):
        agent = RoverSubspaceCoverageAgent.__new__(RoverSubspaceCoverageAgent)
        agent.device = "cpu"
        agent.compute_dtype = torch.float64
        agent.dynamics_state_dim = 4
        states = torch.tensor(
            [[-1.0, 1.0, 0.75, -0.5], [0.25, -0.75, -0.2, 1.1]],
            dtype=torch.float32,
        )
        features = agent._state_dynamics_features(states)
        base_kernel = utils.build_kernel_fn("gaussian", bandwidth=0.7)
        kernel = _SimplexStateKernel(base_kernel, state_dim=4)
        padded = torch.cat((features, torch.zeros((2, 2), dtype=features.dtype)), dim=1)

        torch.testing.assert_close(kernel._decode(padded), states.double(), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(kernel(padded[:1], padded[1:]), base_kernel(states.double()[:1], states.double()[1:]))

    def test_state_dynamics_bandwidth_fits_decoded_state_and_pixel_coverage(self):
        agent = RoverSubspaceCoverageAgent.__new__(RoverSubspaceCoverageAgent)
        agent.dynamics_kernel_input = "states"
        agent.coverage_kernel_input = "pixels"
        agent.kernel_type = "gaussian"
        agent.subsampling_strategy = "random"
        agent.kernel_bandwidth_mult = 1.0
        agent.max_distances = 100
        agent._active_kernel_bandwidth = None
        agent._encoded_actor_fifo = None
        agent.coverage_kernel_bandwidth = None
        agent.coverage_kernel_bandwidth_mult = 2.0
        agent._coverage_sub_next_features = torch.tensor(
            [[0.0], [1.0], [4.0]], dtype=torch.float64
        )

        base_kernel = utils.build_kernel_fn("gaussian", bandwidth=0.1)
        agent.kernel_fn = _SimplexStateKernel(base_kernel, state_dim=4)
        agent.distribution_matcher = SimpleNamespace(kernel_fn=agent.kernel_fn)
        agent.coverage_kernel_fn = utils.build_kernel_fn("gaussian", bandwidth=0.1)

        states_x = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        states_y = torch.tensor(
            [[0.0, 1.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        X = torch.softmax(torch.cat((states_x, torch.zeros((2, 1))), dim=1), dim=1)
        Y = torch.softmax(torch.cat((states_y, torch.zeros((2, 1))), dim=1), dim=1)
        expected_dynamics = torch.cdist(states_x, states_y)
        expected_dynamics = torch.median(expected_dynamics[expected_dynamics > 0])

        agent._fit_state_kernel_bandwidth(X, Y)

        self.assertAlmostEqual(base_kernel.bandwidth, expected_dynamics.item())
        self.assertAlmostEqual(agent.coverage_kernel_fn.bandwidth, 6.0)

    def test_state_dynamics_and_xy_coverage_use_aligned_full_state_sidecars(self):
        agent = RoverSubspaceCoverageAgent.__new__(RoverSubspaceCoverageAgent)
        agent.device = "cpu"
        agent.compute_dtype = torch.float64
        agent.obs_type = "pixels"
        agent.obs_shape = (3, 1, 2)
        agent.embeddings = True
        agent.aug = torch.nn.Identity()
        agent.policy_encoder = _PixelsToTwoFeatures()
        agent.proprio_state_dims = 4
        agent.dynamics_state_dim = 4
        agent.dynamics_kernel_input = "states"
        agent.coverage_kernel_input = "states"
        agent.coverage_state_indices = (0, 1)
        agent.train_coverage_encoder = False

        observations = np.arange(12, dtype=np.uint8).reshape(2, 3, 1, 2)
        states = np.asarray(
            [[-1.0, 1.0, 0.1, -0.2], [0.5, -0.5, 0.8, 0.4]],
            dtype=np.float32,
        )
        next_states = np.asarray(
            [[0.5, -0.5, 0.3, 0.2], [0.5, -0.5, -0.8, -0.4]],
            dtype=np.float32,
        )
        encoded = agent._encode_actor_transition_batch((
            observations,
            np.asarray([0, 1], dtype=np.int64),
            np.zeros((2, 1), dtype=np.float32),
            np.ones((2, 1), dtype=np.float32),
            observations + 1,
            states,
            next_states,
        ))

        state_kernel = _SimplexStateKernel(
            utils.build_kernel_fn("gaussian", bandwidth=0.5), state_dim=4
        )
        torch.testing.assert_close(
            state_kernel._decode(encoded["phi_obs"]), torch.from_numpy(states).double()
        )
        torch.testing.assert_close(
            state_kernel._decode(encoded["phi_next"]), torch.from_numpy(next_states).double()
        )
        torch.testing.assert_close(
            encoded["coverage_next_raw"], torch.from_numpy(next_states[:, :2])
        )
        torch.testing.assert_close(
            encoded["coverage_next_raw"][0], encoded["coverage_next_raw"][1]
        )
        self.assertFalse(torch.allclose(encoded["phi_next"][0], encoded["phi_next"][1]))

    def test_pixel_dynamics_policy_query_needs_no_proprio_metadata(self):
        agent = RoverSubspaceCoverageAgent.__new__(RoverSubspaceCoverageAgent)
        agent.device = "cpu"
        agent.compute_dtype = torch.float32
        agent.obs_type = "pixels"
        agent.dynamics_kernel_input = "pixels"
        agent.embeddings = True
        agent.aug = torch.nn.Identity()
        agent.policy_encoder = _PixelProjector()
        observation = torch.arange(12, dtype=torch.float32).reshape(1, 3, 1, 4)

        actual = agent._policy_query_features(observation, metas=[None])

        torch.testing.assert_close(
            actual,
            agent._unit_mass_operator_features(observation.reshape(1, -1)),
        )


if __name__ == "__main__":
    unittest.main()
