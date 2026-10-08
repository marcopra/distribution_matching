import unittest
from types import SimpleNamespace

import utils
from agent.rover_nystrom_debug import RoverAgent


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
