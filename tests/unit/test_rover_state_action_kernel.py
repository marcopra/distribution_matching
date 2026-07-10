import unittest

import torch

from agent.rover_matchers import DistributionMatcher


class RoverStateActionKernelTest(unittest.TestCase):
    def test_linear_kernel_masks_different_actions(self):
        matcher = DistributionMatcher(lambda_reg=1e-3, kernel_type="linear")
        states = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        actions = torch.tensor([0, 1, 0])

        kernel = matcher.state_action_kernel(states, states, actions, actions)
        expected_state_kernel = states @ states.T
        expected = expected_state_kernel * (actions[:, None] == actions[None, :]).float()

        torch.testing.assert_close(kernel, expected)
        self.assertEqual(kernel[0, 1].item(), 0.0)

    def test_gaussian_kernel_masks_different_actions(self):
        matcher = DistributionMatcher(lambda_reg=1e-3, kernel_type="gaussian", kernel_bandwidth=1.0)
        states = torch.tensor([[0.0], [1.0], [2.0]])
        actions = torch.tensor([0, 1, 0])

        kernel = matcher.state_action_kernel(states, states, actions, actions)

        self.assertEqual(kernel[0, 1].item(), 0.0)
        self.assertGreater(kernel[0, 2].item(), 0.0)
        torch.testing.assert_close(torch.diag(kernel), torch.ones(3))

    def test_one_hot_actions_are_compared_without_index_conversion(self):
        matcher = DistributionMatcher(lambda_reg=1e-3, kernel_type="linear")
        states = torch.tensor([[1.0], [2.0], [3.0]])
        actions = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        )

        kernel = matcher.state_action_kernel(states, states, actions, actions)

        self.assertEqual(kernel[0, 1].item(), 0.0)
        self.assertEqual(kernel[0, 2].item(), 3.0)


if __name__ == "__main__":
    unittest.main()
