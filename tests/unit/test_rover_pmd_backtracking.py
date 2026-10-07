"""Exercise PMD acceptance through the real actor loop with controlled losses."""

import numpy as np
import pytest
import torch

import utils
from agent.rover_matchers import DistributionMatcher
from agent.rover_nystrom_debug import RoverAgent


def make_actor(losses, best_iterate):
    agent = RoverAgent.__new__(RoverAgent)
    agent.device = "cpu"
    agent.compute_dtype = torch.float64
    agent.discount = .99
    agent.kernel_type = "gaussian"
    agent.kernel_fn = utils.build_kernel_fn("gaussian", bandwidth=.8)
    agent.distribution_matcher = DistributionMatcher(
        gamma=.99, lambda_reg=.001, pca_truncation=3, kernel_type="gaussian",
        kernel_bandwidth=.8, device="cpu",
    )
    agent.pca_truncation = 3
    agent.n_actions = 2
    agent.lr_actor = 10.
    agent.pmd_steps = 2
    agent.pmd_eta_min = 1e-8
    agent.pmd_eta_max = 1e8
    agent.sink_schedule = .32
    agent.pmd_grad_clip_norm = 0.
    agent.pmd_eta_mode = "backtracking"
    agent.pmd_backtrack_factor = .25
    agent.pmd_backtrack_max_trials = 2
    agent.pmd_best_iterate = best_iterate
    agent.use_tb = True
    agent.use_wandb = False
    agent._fit_actor_whitening = lambda: None
    agent._fit_state_kernel_bandwidth = lambda *args: None
    agent._save_actor_kernel_debug_plot = lambda *args, **kwargs: None
    agent._encode_state_action = lambda phi, actions: torch.einsum(
        "bd,ba->bda", phi,
        torch.nn.functional.one_hot(actions, 2).to(phi.dtype),
    ).reshape(len(phi), -1)
    sequence = iter(losses)
    evaluated = []

    def objective(**kwargs):
        evaluated.append(kwargs["pi"].clone())
        return torch.tensor(next(sequence), dtype=torch.float64)

    agent._compute_actor_occupancy_loss = objective
    agent._compute_actor_occupancy_gradient = lambda **kwargs: torch.tensor(
        [[1.], [-1.], [1.], [-1.], [0.]], dtype=torch.float64,
    )
    encoded = {
        "phi_obs": torch.tensor([[.2, .8], [.3, .7], [.6, .4], [.8, .2]]),
        "phi_next": torch.tensor([[.3, .7], [.6, .4], [.8, .2], [.2, .8]]),
        "action": torch.tensor([[0], [1], [0], [1]]),
    }
    return agent, encoded, evaluated


@pytest.mark.parametrize("best_iterate", [False, True])
@pytest.mark.parametrize("bad_loss", [2., float("nan"), float("inf"), -float("inf")])
def test_exhausted_backtracking_keeps_last_accepted_policy(best_iterate, bad_loss):
    agent, encoded, evaluated = make_actor([1., bad_loss, bad_loss, bad_loss], best_iterate)
    metrics = agent.update_actor_nystrom(None, None, None, 100000, encoded_full=encoded)
    assert len(evaluated) == 4  # Initial objective, candidate, two retries.
    assert metrics["actor_loss"] == 1.
    assert metrics["actor_eta"] == 0.
    assert metrics["actor_backtracking_rejections"] == 1
    torch.testing.assert_close(agent.gradient_coeff, torch.zeros_like(agent.gradient_coeff))
    torch.testing.assert_close(agent.pi, evaluated[0])


@pytest.mark.parametrize("best_iterate", [False, True])
def test_backtracking_accepts_smaller_step_then_preserves_it_on_rejection(best_iterate):
    agent, encoded, evaluated = make_actor([1., 2., .8, 2., 2., 2.], best_iterate)
    metrics = agent.update_actor_nystrom(None, None, None, 100000, encoded_full=encoded)
    assert metrics["actor_loss"] == .8
    assert metrics["actor_backtracking_rejections"] == 1
    # Best restoration reports the eta that produced its policy; otherwise
    # the final failed iteration reports no accepted step.
    assert metrics["actor_eta"] == (2.5 if best_iterate else 0.)
    torch.testing.assert_close(agent.gradient_coeff,
        torch.tensor([[2.5], [-2.5], [2.5], [-2.5], [0.]], dtype=torch.float64))
    torch.testing.assert_close(agent.pi, evaluated[2])


def test_backtracking_recovers_from_nonfinite_candidate_with_finite_smaller_step():
    agent, encoded, evaluated = make_actor([1., np.nan, .8], True)
    agent.pmd_steps = 1
    metrics = agent.update_actor_nystrom(None, None, None, 100000, encoded_full=encoded)
    assert metrics["actor_loss"] == .8
    assert metrics["actor_eta"] == 2.5
    assert metrics["actor_backtracking_rejections"] == 0
    torch.testing.assert_close(agent.pi, evaluated[2])


def test_nonfinite_initial_objective_fails_before_any_pmd_step():
    agent, encoded, evaluated = make_actor([np.nan], True)
    with pytest.raises(RuntimeError, match="Initial actor occupancy loss is not finite"):
        agent.update_actor_nystrom(None, None, None, 100000, encoded_full=encoded)
    assert len(evaluated) == 1
