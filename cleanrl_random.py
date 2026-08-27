"""Uniform-random policy baseline for EnvPool Atari environments.

This script intentionally performs no learning. At every vectorized environment
step, each environment samples an action independently and uniformly from the
full discrete action space.
"""

import csv
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import envpool
import gym
import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """Name used for the run directory."""
    seed: int = 1
    """Random seed."""
    env_id: str = "MontezumaRevenge-v5"
    """EnvPool environment ID."""
    total_timesteps: int = 2_000_000_000
    """Total environment transitions to collect across all environments."""
    num_envs: int = 128
    """Number of parallel environments."""
    track: bool = False
    """Track the run with Weights & Biases."""
    wandb_project_name: str = "cleanRL"
    """Weights & Biases project name."""
    wandb_entity: str | None = None
    """Weights & Biases entity/team."""
    log_episode_csv: bool = True
    """Write one CSV row for every completed full episode."""
    log_interval: int = 100
    """Log throughput and action frequencies every N vectorized steps."""


class RecordEpisodeStatistics(gym.Wrapper):
    """Track full-game returns while EnvPool uses episodic-life termination."""

    def __init__(self, env):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths

        # EnvPool's `terminated` marks the end of the full game. With
        # episodic_life=True, `dones` may also be true when only a life is lost.
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return observations, rewards, dones, infos


def main() -> None:
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(str(run_dir))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )
    writer.add_text("policy/type", "uniform_random")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
        repeat_action_probability=0.25,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    assert isinstance(envs.action_space, gym.spaces.Discrete), (
        "This baseline only supports discrete action spaces."
    )

    num_actions = envs.single_action_space.n
    action_counts = np.zeros(num_actions, dtype=np.int64)
    avg_returns = deque(maxlen=20)
    episode_count = 0
    global_step = 0
    vector_step = 0
    start_time = time.time()

    csv_file = None
    csv_writer = None
    if args.log_episode_csv:
        csv_file = (run_dir / "episodes.csv").open("w", newline="", buffering=1)
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "global_step",
                "episode",
                "env_index",
                "return",
                "length",
                "elapsed_seconds",
            ],
        )
        csv_writer.writeheader()

    envs.reset()

    try:
        while global_step < args.total_timesteps:
            # Uniform random policy: P(a | s) = 1 / |A| for every action.
            actions = rng.integers(
                low=0,
                high=num_actions,
                size=args.num_envs,
                dtype=np.int64,
            )
            action_counts += np.bincount(actions, minlength=num_actions)

            _, _, dones, info = envs.step(actions)
            global_step += args.num_envs
            vector_step += 1

            for env_idx, done in enumerate(dones):
                # Only log when the full game ends, not at each life loss.
                if done and info["lives"][env_idx] == 0:
                    episode_count += 1
                    episode_return = float(info["r"][env_idx])
                    episode_length = int(info["l"][env_idx])
                    avg_returns.append(episode_return)
                    rolling_return = float(np.mean(avg_returns))

                    print(
                        f"global_step={global_step}, episode={episode_count}, "
                        f"episodic_return={episode_return}, "
                        f"episodic_length={episode_length}, "
                        f"avg_return_20={rolling_return:.3f}"
                    )
                    writer.add_scalar("charts/episodic_return", episode_return, global_step)
                    writer.add_scalar("charts/episodic_length", episode_length, global_step)
                    writer.add_scalar("charts/avg_episodic_return", rolling_return, global_step)
                    writer.add_scalar("charts/episodes", episode_count, global_step)

                    if csv_writer is not None:
                        csv_writer.writerow(
                            {
                                "global_step": global_step,
                                "episode": episode_count,
                                "env_index": env_idx,
                                "return": episode_return,
                                "length": episode_length,
                                "elapsed_seconds": time.time() - start_time,
                            }
                        )

            if vector_step % args.log_interval == 0:
                elapsed = max(time.time() - start_time, 1e-9)
                sps = int(global_step / elapsed)
                writer.add_scalar("charts/SPS", sps, global_step)

                total_actions = action_counts.sum()
                if total_actions > 0:
                    for action_idx, count in enumerate(action_counts):
                        writer.add_scalar(
                            f"policy/action_frequency_{action_idx}",
                            count / total_actions,
                            global_step,
                        )
                print(f"global_step={global_step}, SPS={sps}")
    finally:
        envs.close()
        writer.close()
        if csv_file is not None:
            csv_file.close()


if __name__ == "__main__":
    main()