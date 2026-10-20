"""The positive subspace training preset must select the normalized estimator."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import get_class
from omegaconf import OmegaConf

from agent.rover_positive_kernel import RoverPositiveKernelAgent


def test_positive_state_training_selects_probability_preserving_agent():
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="pretrain_parallel/pointmaze_subspace_positive_states")
    # The inherited state preset otherwise selects signed KRR dynamics with an
    # inner-product kernel. Both the class and kernel must reach the real job.
    assert get_class(cfg.agent._target_) is RoverPositiveKernelAgent
    assert cfg.agent.dynamics_kernel_type == cfg.agent.kernel_type == "gaussian"
    assert cfg.agent.pmd_warm_start
    assert cfg.agent.transition_prior_mass > 0
    assert OmegaConf.to_container(cfg.agent.coverage_state_indices) == [0, 1]
