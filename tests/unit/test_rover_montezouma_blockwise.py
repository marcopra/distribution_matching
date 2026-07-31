import unittest

import torch
import torch.nn.functional as F

from agent.rover_matchers import DistributionMatcher
from agent.rover_networks import CNNEncoder, Encoder
from agent.rover_nystrom_debug import RoverAgent


class CompactBlockwiseNystromTest(unittest.TestCase):
    def _case(self, dtype, block_size, kernel_type="inner_product"):
        torch.manual_seed(7)
        n, m, d, n_actions = 7, 3, 4, 2
        all_phi = torch.randn(n, d, dtype=dtype)
        sub_phi = torch.randn(m, d, dtype=dtype)
        sub_next = torch.randn(m, d, dtype=dtype)
        all_actions = torch.tensor([0, 1, 0, 0, 1, 1, 0])
        sub_actions = torch.tensor([0, 1, 0])
        matcher = DistributionMatcher(
            lambda_reg=1e-3,
            gamma=0.91,
            kernel_type=kernel_type,
            kernel_bandwidth=1.7 if kernel_type == "gaussian" else None,
            device="cpu",
        )
        matcher.state_kernel_fn = matcher.kernel_fn

        K_nm = matcher.state_action_kernel(
            all_phi, sub_phi, all_actions, sub_actions
        )
        K_mm = matcher.state_action_kernel(
            sub_phi, sub_phi, sub_actions, sub_actions
        )
        expected_A = K_nm.T @ K_nm + matcher.lambda_reg * K_mm
        expected_A.diagonal().add_(1e-6)
        A, U_r = matcher.compute_nystrom_system_blockwise(
            all_phi,
            sub_phi,
            all_actions,
            sub_actions,
            components=m,
            block_size=block_size,
        )
        torch.testing.assert_close(A, expected_A)

        coeff = torch.randn(n + 1, 1, dtype=dtype)
        H = matcher.state_kernel(all_phi, sub_next)
        E = F.one_hot(all_actions, n_actions).to(dtype=dtype)
        expected_pi = torch.softmax(
            -(H.T @ (coeff[:-1] * E) + coeff[-1]), dim=1
        )
        pi = matcher.policy_from_support_blockwise(
            all_phi,
            sub_next,
            all_actions,
            coeff,
            n_actions,
            block_size,
        )
        torch.testing.assert_close(pi, expected_pi)

        M = H * (E @ pi.T)
        expected_B = torch.linalg.solve(A, K_nm.T)
        expected_BM = expected_B @ M
        BM = matcher.compute_BM_blockwise(
            A,
            all_phi,
            sub_phi,
            sub_next,
            all_actions,
            sub_actions,
            pi,
            block_size,
        )
        torch.testing.assert_close(BM, expected_BM)

        alpha = torch.zeros((m, 1), dtype=dtype)
        alpha[0] = 1
        psi_sub = torch.einsum(
            "bd,ba->bda",
            sub_phi,
            F.one_hot(sub_actions, n_actions).to(dtype=dtype),
        ).reshape(m, -1)
        expected_nu = matcher.compute_nu_pi_nystrom_memory_efficient(
            phi_all_obs=all_phi,
            phi_sub_next_obs=sub_next,
            psi_sub_obs_action=psi_sub,
            psi_all_obs_action=torch.empty((n, 1), dtype=dtype),
            H=H,
            pi=pi,
            E=E,
            alpha=alpha,
            sink_norm=0.2,
            B_nystrom=expected_B,
        )
        nu = matcher.compute_nu_from_BM_compact(
            BM, sub_next, sub_phi, alpha, sink_norm=0.2
        )
        torch.testing.assert_close(nu, expected_nu)

        expected_gradient = (
            matcher.compute_gradient_coefficient_nystrom_blockwise_and_proj(
                phi_sub_next_obs=sub_next,
                psi_sub_obs_action=psi_sub,
                H=H,
                pi=pi,
                E=E,
                alpha=alpha,
                sink_norm=0.2,
                B_nystrom=expected_B,
                eig_vecs_r=U_r,
            )
        )
        gradient = matcher.compute_gradient_coefficient_compact_blockwise(
            BM,
            A,
            all_phi,
            sub_phi,
            sub_next,
            all_actions,
            sub_actions,
            alpha,
            0.2,
            U_r,
            block_size,
        )
        torch.testing.assert_close(gradient, expected_gradient)

    def test_inner_product_float64_multiple_blocks(self):
        for block_size in (1, 4, 20):
            with self.subTest(block_size=block_size):
                self._case(torch.float64, block_size)

    def test_gaussian_float64(self):
        self._case(torch.float64, 2, kernel_type="gaussian")

    def test_inner_product_float32(self):
        self._case(torch.float32, 3)


class FreshReplayRoutingTest(unittest.TestCase):
    @staticmethod
    def _raw(values):
        values = torch.as_tensor(values, dtype=torch.float32).reshape(-1, 1)
        count = values.shape[0]
        return (
            values,
            torch.zeros((count, 1), dtype=torch.long),
            torch.zeros((count, 1)),
            torch.ones((count, 1)),
            values + 100,
        )

    def test_fifo_disabled_routes_to_fresh_encoder(self):
        agent = RoverAgent.__new__(RoverAgent)
        agent.debug_fixed_dataset_updates = False
        agent.use_encoded_fifo = False
        marker = object()
        agent._fresh_replay_actor_update_data = (
            lambda replay_iter, initial_batch, replay_buffer=None: marker
        )
        obs = torch.zeros((2, 1))
        action = torch.zeros((2, 1), dtype=torch.long)
        next_obs = torch.ones((2, 1))
        reward = torch.zeros((2, 1))
        result = agent._get_actor_update_data(
            iter(()), obs, action, next_obs, reward, replay_buffer=object()
        )
        self.assertIs(result, marker)

    def test_fifo_drain_is_noop_when_disabled(self):
        agent = RoverAgent.__new__(RoverAgent)
        agent.use_encoded_fifo = False
        agent._update_encoded_actor_fifo = lambda replay: self.fail("FIFO touched")
        self.assertFalse(agent.drain_encoded_actor_fifo(object()))

    def test_first_transition_falls_back_when_main_replay_is_empty(self):
        agent = RoverAgent.__new__(RoverAgent)

        class EmptyMainReplay:
            def get_first_transition(self):
                raise RuntimeError("Replay buffer is empty")

        fallback = (
            torch.tensor([[7.0]]),
            torch.tensor([[1]]),
            torch.tensor([[8.0]]),
            torch.tensor([[2.0]]),
        )
        result = agent._load_first_actor_transition(
            replay_buffer=EmptyMainReplay(),
            fallback_actor_batch=fallback,
        )
        for actual, expected in zip(result, fallback):
            torch.testing.assert_close(actual, expected)

    def test_fresh_collection_has_exact_size_float32_and_pinned_first(self):
        agent = RoverAgent.__new__(RoverAgent)
        agent.batch_size_actor = 5
        agent.actor_encode_batch_size = 2
        agent._sync_actor_encode_encoder = lambda: None
        agent._encode_fresh_actor_chunk = lambda batch: {
            "phi_obs": batch[0].float().cpu(),
            "phi_next": batch[4].float().cpu(),
            "action": batch[1].long().cpu(),
            "reward": batch[2].float().cpu(),
        }
        agent._load_first_actor_transition = lambda replay_buffer=None, fallback_actor_batch=None: (
            torch.tensor([[999.0]]),
            torch.tensor([[1]]),
            torch.tensor([[1000.0]]),
            torch.tensor([[3.0]]),
        )
        replay_iter = iter([self._raw([2, 3, 4]), self._raw([5, 6])])
        result = agent._collect_fresh_encoded_actor_data(
            replay_iter,
            self._raw([0, 1]),
            replay_buffer=object(),
        )
        self.assertEqual(result["phi_obs"].shape, (5, 1))
        self.assertEqual(result["phi_obs"].dtype, torch.float32)
        self.assertEqual(float(result["phi_obs"][0]), 999.0)
        self.assertEqual(float(result["phi_next"][0]), 1000.0)
        self.assertEqual(int(result["action"][0]), 1)

    def test_fresh_encoder_reuses_oom_splitting(self):
        agent = RoverAgent.__new__(RoverAgent)
        agent.encoded_fifo_cuda_oom_splits = 3
        agent._is_cuda_oom = lambda error: "cuda out of memory" in str(error).lower()

        def encode(batch, encoder=None):
            if batch[0].shape[0] > 2:
                raise RuntimeError("CUDA out of memory")
            return {"phi_obs": batch[0].float()}

        agent._encode_actor_transition_batch = encode
        result = agent._encode_actor_transition_batch_with_retries(
            self._raw([0, 1, 2, 3, 4]),
            encoder=object(),
        )
        torch.testing.assert_close(
            result["phi_obs"],
            torch.arange(5, dtype=torch.float32).reshape(-1, 1),
        )

    def test_encoder_input_follows_module_dtype_not_global_default(self):
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            cnn = CNNEncoder((3, 84, 84), feature_dim=8).float()
            cnn_output = cnn.encode_and_project(
                torch.zeros((2, 3, 84, 84), dtype=torch.float64)
            )
            self.assertEqual(cnn_output.dtype, torch.float32)

            mlp = Encoder((4,), hidden_dim=8, feature_dim=3).float()
            mlp_output = mlp(torch.zeros((2, 4), dtype=torch.float64))
            self.assertEqual(mlp_output.dtype, torch.float32)
        finally:
            torch.set_default_dtype(previous_dtype)


if __name__ == "__main__":
    unittest.main()
