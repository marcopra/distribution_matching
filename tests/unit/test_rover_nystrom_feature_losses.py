import unittest

import torch

from agent.rover_networks import CNNEncoder, Encoder
from agent.rover_nystrom_debug import RoverAgent


class NystromFeatureLossTest(unittest.TestCase):
    def test_exact_observation_groups_find_duplicate_images(self):
        observations = torch.tensor(
            [
                [[[0, 1], [2, 3]]],
                [[[4, 5], [6, 7]]],
                [[[0, 1], [2, 3]]],
            ],
            dtype=torch.uint8,
        )

        groups = RoverAgent._exact_observation_group_ids(observations)

        self.assertEqual(int(groups[0]), int(groups[2]))
        self.assertNotEqual(int(groups[0]), int(groups[1]))

    def test_multi_positive_infonce_does_not_repel_duplicate_targets(self):
        logits = torch.tensor(
            [[3.0, -2.0, 3.0], [-2.0, 3.0, -2.0], [3.0, -2.0, 3.0]],
            requires_grad=True,
        )
        groups = torch.tensor([0, 1, 0])

        loss = RoverAgent._multi_positive_infonce(logits, groups)
        diagonal_loss = torch.nn.functional.cross_entropy(logits, torch.arange(3))
        loss.backward()

        self.assertLess(float(loss.detach()), float(diagonal_loss.detach()))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_sigreg_is_finite_and_differentiable(self):
        agent = RoverAgent.__new__(RoverAgent)
        agent.leworld_num_projections = 12
        agent.leworld_num_knots = 5
        agent.leworld_projection_chunk_size = 4
        embeddings = torch.randn(16, 6, requires_grad=True)

        loss = agent._sigreg(embeddings)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(embeddings.grad).all())

    def test_encoders_can_return_raw_projected_features(self):
        state_encoder = Encoder((3,), hidden_dim=8, feature_dim=4)
        state_obs = torch.randn(5, 3)
        raw_state = state_encoder.encode_and_project(state_obs, normalize=False)
        normalized_state = state_encoder.encode_and_project(state_obs)
        self.assertTrue(torch.allclose(raw_state, state_encoder.encode_raw(state_obs)))
        self.assertTrue(
            torch.allclose(
                normalized_state,
                torch.nn.functional.normalize(raw_state, p=1, dim=1),
            )
        )

        pixel_encoder = CNNEncoder((1, 84, 84), feature_dim=4, mode="l2")
        pixel_obs = torch.randint(0, 256, (2, 1, 84, 84), dtype=torch.uint8)
        raw_pixels = pixel_encoder.encode_and_project(pixel_obs, normalize=False)
        normalized_pixels = pixel_encoder.encode_and_project(pixel_obs)
        self.assertEqual(raw_pixels.shape, normalized_pixels.shape)
        self.assertTrue(
            torch.allclose(normalized_pixels.norm(dim=1), torch.ones(2), atol=1e-5)
        )

    def test_leworld_agent_updates_with_raw_rover_features(self):
        agent = RoverAgent(
            name="test",
            obs_type="states",
            obs_shape=(3,),
            grayscale=False,
            action_shape=(2,),
            lr_actor=1e-3,
            discount=0.99,
            lambda_reg=0.0,
            batch_size=4,
            batch_size_actor=4,
            subsamples=4,
            nstep=1,
            use_tb=True,
            use_wandb=False,
            lr_T=1e-3,
            lr_encoder=1e-3,
            curl=False,
            embedding_sum_loss=0.0,
            hidden_dim=8,
            feature_dim=4,
            update_every_steps=1,
            update_actor_every_steps=1,
            pmd_steps=1,
            num_expl_steps=0,
            T_init_steps=0,
            total_train_steps=10,
            sink_schedule="0",
            epsilon_schedule="0",
            mode="l2",
            reward=False,
            pca_truncation=0,
            embeddings=True,
            linear_projection=True,
            feature_learning_loss="leworld",
            leworld_num_projections=8,
            leworld_num_knots=4,
            leworld_projection_chunk_size=4,
            device="cpu",
        )
        obs = torch.randn(4, 3)
        next_obs = torch.randn(4, 3)
        actions = torch.tensor([[0], [1], [0], [1]])
        rewards = torch.zeros(4, 1)

        raw_features = agent.aug_and_encode(obs, project=True)
        expected_raw = agent.encoder.encode_and_project(obs, normalize=False)
        metrics = agent.update_encoders(obs, actions, next_obs, rewards)

        self.assertTrue(torch.allclose(raw_features, expected_raw))
        self.assertTrue(torch.isfinite(torch.tensor(metrics["transition_loss"])))
        self.assertGreaterEqual(metrics["leworld_prediction_loss"], 0.0)
        self.assertGreaterEqual(metrics["sigreg_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
