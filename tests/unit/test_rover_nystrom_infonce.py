import torch

from agent.rover_nystrom_subspace import RoverSubspaceCoverageAgent


def test_coverage_encoder_keeps_the_diagonal_infonce_update():
    torch.manual_seed(0)
    agent = RoverSubspaceCoverageAgent(
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
        mode="l1",
        reward=False,
        pca_truncation=0,
        embeddings=True,
        linear_projection=True,
        device="cpu",
        coverage_embeddings=True,
    )
    obs = torch.randn(4, 3)
    next_obs = torch.randn(4, 3)
    actions = torch.tensor([[0], [1], [0], [1]])
    rewards = torch.zeros(4, 1)
    before = [parameter.detach().clone() for parameter in agent.coverage_encoder.parameters()]

    metrics = agent.update_encoders(obs, actions, next_obs, rewards)

    assert torch.isfinite(torch.tensor(metrics["coverage_transition_loss"]))
    assert any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, agent.coverage_encoder.parameters())
    )
