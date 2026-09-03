import unittest
from types import SimpleNamespace

import torch

import utils
from agent.rover_nystrom_debug import RoverAgent


class PointMazeWhiteningTest(unittest.TestCase):
    def make_agent(self, enabled=True):
        agent = RoverAgent.__new__(RoverAgent)
        agent.whiten_representations = enabled
        agent.whitening_mean = None
        agent.whitening_components = None
        agent.whitening_eigenvalues = None
        agent.whitening_eigenvalue_floor = None
        agent.whitening_explained_variance = None
        agent.n_actions = 2
        agent.device = "cpu"
        return agent

    @staticmethod
    def augmented(values):
        return torch.cat([values, torch.zeros(values.shape[0], 1)], dim=1)

    def install_actor_features(self, agent, offset=0.0):
        x = torch.tensor(
            [
                [-3.0, -1.0, 0.2],
                [-2.0, 0.5, -0.4],
                [-1.0, 1.5, 0.7],
                [0.0, -0.5, -0.8],
                [1.0, 0.7, 0.3],
                [2.0, -1.3, 0.9],
                [3.0, 0.2, -0.2],
                [4.0, 1.1, 0.5],
            ]
        ) + offset
        agent._phi_all_obs = self.augmented(x)
        agent._phi_all_next = self.augmented(x + 0.25)
        agent._phi_sub_obs = self.augmented(x[[1, 5]])
        agent._phi_sub_next = self.augmented(x[[1, 5]] + 0.25)
        agent._all_actions = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        agent._sub_actions = torch.tensor([1, 1])
        agent._psi_all = torch.empty(0)
        agent._psi_sub = torch.empty(0)
        return x

    def test_fit_whitens_full_support_and_reuses_transform_everywhere(self):
        agent = self.make_agent()
        raw = self.install_actor_features(agent)
        raw_next = raw + 0.25
        raw_sub = raw[[1, 5]]

        agent._fit_actor_whitening()

        whitened = agent._phi_all_obs[:, :-1]
        self.assertGreaterEqual(agent.whitening_explained_variance, 0.99)
        torch.testing.assert_close(whitened.mean(dim=0), torch.zeros(whitened.shape[1]), atol=1e-6, rtol=0)
        covariance = torch.cov(whitened.T)
        expected = torch.eye(whitened.shape[1]) / whitened.shape[1]
        torch.testing.assert_close(covariance, expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(agent._phi_all_next[:, :-1], agent._apply_whitening(raw_next))
        torch.testing.assert_close(agent._phi_sub_obs[:, :-1], agent._apply_whitening(raw_sub))
        torch.testing.assert_close(agent._psi_all[:, :-1], agent._encode_state_action(whitened, agent._all_actions))

    def test_refit_replaces_transform_but_disabled_path_is_identity(self):
        agent = self.make_agent()
        self.install_actor_features(agent)
        agent._fit_actor_whitening()
        first_mean = agent.whitening_mean.clone()
        self.install_actor_features(agent, offset=10.0)
        agent._fit_actor_whitening()
        self.assertFalse(torch.equal(first_mean, agent.whitening_mean))

        disabled = self.make_agent(enabled=False)
        values = torch.randn(4, 3)
        self.assertIs(disabled._apply_whitening(values), values)


class PointMazeBandwidthScheduleTest(unittest.TestCase):
    def make_agent(self, bandwidth):
        agent = RoverAgent.__new__(RoverAgent)
        agent.kernel_bandwidth = bandwidth
        agent.kernel_fn = utils.build_kernel_fn("gaussian")
        matcher_fn = utils.build_kernel_fn("gaussian")
        agent.distribution_matcher = SimpleNamespace(
            kernel_bandwidth=None,
            kernel_fn=matcher_fn,
            state_kernel_fn=None,
        )
        return agent

    def test_linear_schedule_updates_both_kernels_and_clamps_zero(self):
        agent = self.make_agent("linear(0.0, 0.3, 500000)")
        self.assertEqual(agent._resolve_kernel_bandwidth(0), 1e-12)
        self.assertAlmostEqual(agent._resolve_kernel_bandwidth(250000), 0.15)
        self.assertAlmostEqual(agent.kernel_fn.bandwidth, 0.15)
        self.assertAlmostEqual(agent.distribution_matcher.kernel_fn.bandwidth, 0.15)
        self.assertIs(agent.distribution_matcher.state_kernel_fn, agent.kernel_fn)

    def test_numeric_and_null_bandwidths_remain_supported(self):
        fixed = self.make_agent(0.2)
        self.assertAlmostEqual(fixed._resolve_kernel_bandwidth(123), 0.2)
        automatic = self.make_agent(None)
        self.assertIsNone(automatic._resolve_kernel_bandwidth(123))
        self.assertIsNone(automatic._active_kernel_bandwidth)


if __name__ == "__main__":
    unittest.main()
