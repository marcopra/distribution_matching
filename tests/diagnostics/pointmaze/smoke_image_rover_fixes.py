#!/usr/bin/env python3
"""Check real PointMaze observations, encoder updates, actor fits, and FIFO drains.

Uses the experiment CNNs and kernels with bounded support (32 rows/16 landmarks).
This checks implementation consistency, not 100k/200k training performance.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

import gym_env
import utils


class PendingReplay:
    def __init__(self):
        self.pending = None

    def get_new_transitions_since(self, marker, limit):
        if self.pending is None:
            return None, None
        result, self.pending = self.pending, None
        return result

    def mark_transitions_encoded(self, marker):
        pass


def check_mode(mode: str, device: str) -> dict:
    config_name = ("pretrain_pointmaze_umaze_1_pixels_subspace" if mode == "image"
                   else "pretrain_pointmaze_umaze_1_subspace")
    with hydra.initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.1"):
        cfg = hydra.compose(config_name=f"pretrain_parallel/{config_name}", overrides=[
            "use_wandb=false", f"device={device}", "agent.batch_size_actor=32",
            "agent.subsamples=16", "agent.pca_truncation=16", "agent.pmd_steps=2",
            "agent.encoded_fifo_capacity=64", "agent.encoded_fifo_encode_batch_size=8",
            "agent.nystrom_cholesky_progress=false", "agent.feature_learning_loss=infonce",
        ])
    utils.set_seed_everywhere(1)
    kwargs = OmegaConf.to_container(cfg.env, resolve=True)
    name = kwargs.pop("name")
    env = gym_env.make(name, cfg.obs_type, frame_stack=cfg.frame_stack,
                       action_repeat=cfg.action_repeat, seed=1, resolution=cfg.resolution,
                       grayscale=cfg.grayscale, url=True, **kwargs)
    try:
        cfg.agent.obs_type = cfg.obs_type
        cfg.agent.obs_shape = gym_env.observation_spec(env).shape
        cfg.agent.action_shape = (gym_env.action_spec(env).num_values,)
        cfg.agent.num_expl_steps = cfg.num_seed_frames
        agent = hydra.utils.instantiate(cfg.agent)
        agent._save_actor_kernel_debug_plot = lambda *args, **kwargs: None
        ts = env.reset(seed=4)
        observations, following, actions = [], [], []
        for index in range(40):
            action = index % agent.n_actions
            nxt = env.step(action)
            observations.append(ts.observation)
            following.append(nxt.observation)
            actions.append(action)
            ts = nxt
        observations, following = np.stack(observations), np.stack(following)
        transitions = (observations, np.asarray(actions).reshape(-1, 1), np.zeros((40, 1)),
                       np.ones((40, 1)), following)
        replay = PendingReplay()
        for start, stop in ((0, 16), (16, 32)):
            replay.pending = (np.arange(start, stop), tuple(x[start:stop] for x in transitions))
            agent.drain_encoded_actor_fifo(replay)

        training_batch = utils.to_torch(tuple(x[:8] for x in transitions), device)
        agent.update_encoders(training_batch[0], training_batch[1], training_batch[4], training_batch[2])
        coverage_weights = (None if agent.coverage_feature_encoder is None else
                            {key: value.clone() for key, value in
                             agent.coverage_feature_encoder.state_dict().items()})
        data = agent._encoded_fifo_actor_update_data(replay)
        max_error = 0.
        with torch.no_grad():
            for batch in (data.full, data.subsample):
                for key, raw_key in (("phi_obs", "obs_raw"), ("phi_next", "next_obs_raw")):
                    fresh = agent._encode_with_module(agent.policy_encoder,
                        batch[raw_key].to(device), project=True).float().cpu()
                    torch.testing.assert_close(batch[key], fresh, atol=1e-6, rtol=1e-5)
                    max_error = max(max_error, float((batch[key] - fresh).abs().max()))
                if mode == "image":
                    assert batch["obs_raw"].dtype == torch.uint8
                    newest = batch["next_obs_raw"][:, -1:].to(device)
                    expected = agent.coverage_feature_encoder(newest).cpu()
                    torch.testing.assert_close(batch["coverage_next_features"], expected,
                                               atol=1e-7, rtol=1e-5)

        metrics = agent.update_actor_nystrom(None, None, None, 11000,
            encoded_full=data.full, encoded_sub=data.subsample)
        before = agent._compute_action_probs_batch(observations[:4])
        agent.update_encoders(training_batch[0], training_batch[1], training_batch[4], training_batch[2])
        replay.pending = (np.arange(32, 40), tuple(x[32:40] for x in transitions))
        agent.drain_encoded_actor_fifo(replay)
        after = agent._compute_action_probs_batch(observations[:4])
        np.testing.assert_array_equal(before, after)
        assert np.isfinite(float(metrics["actor_loss"]))
        if coverage_weights is not None:
            for key, value in agent.coverage_feature_encoder.state_dict().items():
                torch.testing.assert_close(value, coverage_weights[key], atol=0., rtol=0.)
        return dict(mode=mode, verified=True, support_max_abs_error=max_error,
                    action_probability_drift_after_encoder_update_and_drain=float(np.max(abs(after-before))),
                    actor_loss=float(metrics["actor_loss"]),
                    backtracking_rejections=metrics["actor_backtracking_rejections"])
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "tests/outputs/pointmaze/image_rover_verification/smoke")
    args = parser.parse_args()
    torch.set_num_threads(4)
    # Compare frozen features across batch sizes without TF32 convolution
    # rounding masking the frame-selection check. Training config is unchanged.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    results = [check_mode(mode, args.device) for mode in ("image", "state")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
