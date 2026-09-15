import unittest
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import torch

from agent.rover_buffers import EncodedTransitionFIFO
from agent.rover_nystrom_debug import RoverAgent


class _PixelProjector(torch.nn.Module):
    def encode_and_project(self, observation):
        return observation.float().reshape(observation.shape[0], -1)


class RoverPixelDiagnosticCoordinatesTest(unittest.TestCase):
    def test_replay_metadata_uses_coordinates_from_stored_time_step(self):
        agent = RoverAgent.__new__(RoverAgent)
        agent.diagnostic_coordinate_dims = 2
        time_step = SimpleNamespace(
            proprio_observation=np.asarray([1.25, -0.75, 9.0], dtype=np.float32)
        )
        policy_meta = OrderedDict(diagnostic_coordinates=np.zeros(2, dtype=np.float32))

        replay_meta = agent.prepare_replay_meta(time_step, policy_meta)

        np.testing.assert_array_equal(
            replay_meta["diagnostic_coordinates"],
            np.asarray([1.25, -0.75], dtype=np.float32),
        )
        np.testing.assert_array_equal(policy_meta["diagnostic_coordinates"], np.zeros(2))

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


if __name__ == "__main__":
    unittest.main()
