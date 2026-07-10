#!/usr/bin/env python3
"""Collect a homogeneous MultipleRooms dataset for Nyström encoder debugging.
FUNZIONA SOLO SU MINIGRID -> prendo una posizione random nello spazio e faccio azione random -> tendenzialmente dovrebbe essere uniforme.

Examples
--------
python tests/diagnostics/gridworld/collect_multiplerooms_dataset.py \
    --n-samples 50000 \
    --obs-mode pixels \
    --output-dir tests/outputs/gridworld/multiplerooms_pixels

python tests/diagnostics/gridworld/collect_multiplerooms_dataset.py \
    --n-samples 4096 \
    --obs-mode discrete_states \
    --seed 7 \
    --output-dir tests/outputs/gridworld/multiplerooms_states
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
import numpy as np
from omegaconf import OmegaConf
from tqdm.auto import tqdm

import gym_env
import utils
from replay_buffer import save_episode

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect a homogeneous dataset from the default MultipleRooms config."
    )
    parser.add_argument(
        "--config-name",
        default="pretrain/pretrain_rover_multiplerooms",
        help="Hydra config used to recover the default MultipleRooms environment.",
    )
    parser.add_argument(
        "--obs-mode",
        choices=["pixels", "discrete_states"],
        default="pixels",
        help="Observation mode to save inside the dataset.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        required=True,
        help="Exact number of one-step transitions to collect.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for dataset, replay episodes, plots, and summaries.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed used for balanced pair shuffling and reproducibility.",
    )
    parser.add_argument(
        "--sample-grid-size",
        type=int,
        default=16,
        help="Number of sample observations to visualize in sample_grid.png.",
    )
    return parser.parse_args()


def find_discrete_env(env):
    current = env
    visited = set()

    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "n_states") and hasattr(current, "idx_to_state"):
            return current
        current = getattr(current, "env", None)

    return env.unwrapped


def compose_cfg(config_name: str, obs_mode: str, seed: int):
    def _resolve_group_path(group_value: str, group_name: str) -> Path:
        direct = CONFIG_DIR / f"{group_value}.yaml"
        if direct.exists():
            return direct
        grouped = CONFIG_DIR / group_name / f"{group_value}.yaml"
        if grouped.exists():
            return grouped
        raise FileNotFoundError(f"Could not resolve {group_name} config '{group_value}'")

    config_path = CONFIG_DIR / f"{config_name}.yaml"
    cfg = OmegaConf.load(config_path)

    defaults = OmegaConf.to_container(cfg.get("defaults", []), resolve=False)
    env_default = None
    for item in defaults:
        if isinstance(item, dict) and "/env" in item:
            env_default = item["/env"]
            break
        if isinstance(item, dict) and "env" in item:
            env_default = item["env"]
            break

    if env_default is None:
        raise ValueError(f"Could not recover env default from {config_path}")

    env_cfg = OmegaConf.load(_resolve_group_path(env_default, "env"))
    return OmegaConf.merge(cfg, env_cfg, {"obs_type": obs_mode, "seed": seed})


def make_env_from_cfg(cfg):
    env_kwargs = OmegaConf.to_container(cfg.env, resolve=True)
    env_kwargs.pop("name", None)
    return gym_env.make(
        cfg.env.name,
        cfg.obs_type,
        frame_stack=cfg.frame_stack,
        action_repeat=cfg.action_repeat,
        seed=cfg.seed,
        resolution=cfg.resolution,
        grayscale=cfg.grayscale,
        url=True,
        **env_kwargs,
    )


def make_balanced_pair_schedule(
    state_indices: Sequence[int],
    n_actions: int,
    n_samples: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """Build a nearly uniform state-action schedule and shuffle it once.

    Using all (state, action) pairs is the closest practical approximation to
    simultaneous state and action homogeneity under an exact sample budget.
    """

    pair_count = len(state_indices) * n_actions
    base = n_samples // pair_count
    remainder = n_samples % pair_count

    counts = np.full((len(state_indices), n_actions), base, dtype=np.int64)
    if remainder > 0:
        extra_pairs = rng.permutation(pair_count)[:remainder]
        counts.reshape(-1)[extra_pairs] += 1

    schedule: List[Tuple[int, int]] = []
    for local_state_idx, state_idx in enumerate(state_indices):
        for action in range(n_actions):
            schedule.extend([(state_idx, action)] * int(counts[local_state_idx, action]))

    rng.shuffle(schedule)
    return schedule


def scalar_action_dtype(env) -> np.dtype:
    return np.dtype(env.action_space.dtype)


def make_episode(
    obs: np.ndarray,
    action: int,
    reward: float,
    discount: float,
    next_obs: np.ndarray,
    action_dtype: np.dtype,
) -> Dict[str, np.ndarray]:
    reward_dtype = np.float32
    discount_dtype = np.float32
    return {
        "observation": np.stack([obs, next_obs]).astype(obs.dtype, copy=False),
        "action": np.asarray([0, action], dtype=action_dtype),
        "reward": np.asarray([[0.0], [reward]], dtype=reward_dtype),
        "discount": np.asarray([[1.0], [discount]], dtype=discount_dtype),
    }


def positions_to_heatmap(base_env, state_counts: np.ndarray) -> np.ndarray:
    valid_cells = [cell for cell in base_env.cells if cell != getattr(base_env, "DEAD_STATE", None)]
    min_x = min(cell[0] for cell in valid_cells)
    max_x = max(cell[0] for cell in valid_cells)
    min_y = min(cell[1] for cell in valid_cells)
    max_y = max(cell[1] for cell in valid_cells)

    grid = np.full((max_y - min_y + 1, max_x - min_x + 1), np.nan, dtype=np.float64)
    for state_idx, count in enumerate(state_counts):
        if state_idx not in base_env.idx_to_state:
            continue
        cell = base_env.idx_to_state[state_idx]
        if cell == getattr(base_env, "DEAD_STATE", None):
            continue
        x, y = cell
        grid[y - min_y, x - min_x] = count
    return grid


def plot_state_heatmap(
    path: Path,
    base_env,
    state_counts: np.ndarray,
    title: str,
):
    grid = positions_to_heatmap(base_env, state_counts)
    valid_values = grid[~np.isnan(grid)]
    vmin = float(valid_values.min()) if valid_values.size else 0.0
    vmax = float(valid_values.max()) if valid_values.size else 1.0

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(grid, cmap="viridis", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="count")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_action_histogram(path: Path, action_counts: np.ndarray):
    actions = np.arange(len(action_counts))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(actions, action_counts, color="#4477AA")
    ax.set_xticks(actions)
    ax.set_xlabel("action")
    ax.set_ylabel("count")
    ax.set_title("Action Frequency")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_source_next_comparison(
    path: Path,
    base_env,
    source_counts: np.ndarray,
    next_counts: np.ndarray,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, counts, title in zip(
        axes,
        (source_counts, next_counts),
        ("Source State Coverage", "Next-State Coverage"),
    ):
        grid = positions_to_heatmap(base_env, counts)
        valid_values = grid[~np.isnan(grid)]
        vmin = float(valid_values.min()) if valid_values.size else 0.0
        vmax = float(valid_values.max()) if valid_values.size else 1.0
        im = ax.imshow(grid, cmap="magma", interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sample_grid(path: Path, observations: np.ndarray, obs_mode: str):
    if obs_mode != "pixels" or len(observations) == 0:
        return

    num_items = len(observations)
    side = int(np.ceil(np.sqrt(num_items)))
    fig, axes = plt.subplots(side, side, figsize=(3 * side, 3 * side))
    axes = np.asarray(axes).reshape(-1)

    for ax, obs in zip(axes, observations):
        image = obs.transpose(1, 2, 0)
        if image.shape[2] == 1:
            ax.imshow(image[..., 0], cmap="gray")
        else:
            ax.imshow(image.astype(np.uint8))
        ax.axis("off")

    for ax in axes[num_items:]:
        ax.axis("off")

    fig.suptitle("Sample Observations", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def deviation_summary(counts: np.ndarray) -> Dict[str, float]:
    expected = float(np.mean(counts))
    deviations = counts.astype(np.float64) - expected
    return {
        "expected_uniform_count": expected,
        "min_count": float(np.min(counts)),
        "max_count": float(np.max(counts)),
        "max_abs_deviation": float(np.max(np.abs(deviations))),
        "mean_abs_deviation": float(np.mean(np.abs(deviations))),
        "std": float(np.std(counts.astype(np.float64))),
    }


def main():
    args = parse_args()
    utils.set_seed_everywhere(args.seed)
    rng = np.random.default_rng(args.seed)

    output_dir = args.output_dir.resolve()
    replay_dir = output_dir / "replay_buffer"
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_dir.mkdir(parents=True, exist_ok=True)

    cfg = compose_cfg(args.config_name, args.obs_mode, args.seed)
    env = make_env_from_cfg(cfg)
    base_env = find_discrete_env(env)

    valid_state_indices = [
        state_idx
        for state_idx, cell in base_env.idx_to_state.items()
        if cell != getattr(base_env, "DEAD_STATE", None)
    ]
    schedule = make_balanced_pair_schedule(
        state_indices=valid_state_indices,
        n_actions=env.action_space.n,
        n_samples=args.n_samples,
        rng=rng,
    )

    print(
        f"Collecting {args.n_samples} transitions with a balanced state-action schedule "
        f"over {len(valid_state_indices)} states x {env.action_space.n} actions."
    )
    print(
        "Exact state/action uniformity is only possible when n_samples is divisible by "
        "num_states * num_actions; otherwise counts differ by at most one."
    )

    action_dtype = scalar_action_dtype(env)
    source_counts = np.zeros(base_env.n_states, dtype=np.int64)
    next_counts = np.zeros(base_env.n_states, dtype=np.int64)
    action_counts = np.zeros(env.action_space.n, dtype=np.int64)
    state_action_counts = np.zeros((base_env.n_states, env.action_space.n), dtype=np.int64)

    observations: List[np.ndarray] = []
    next_observations: List[np.ndarray] = []
    proprio_observations: List[np.ndarray] = []
    next_proprio_observations: List[np.ndarray] = []
    actions: List[int] = []
    rewards: List[float] = []
    discounts: List[float] = []
    source_state_indices: List[int] = []
    next_state_indices: List[int] = []
    source_positions: List[Tuple[int, int]] = []
    next_positions: List[Tuple[int, int]] = []

    sample_observations: List[np.ndarray] = []
    sample_count = min(args.sample_grid_size, args.n_samples)
    sampled_preview_ids = set(rng.choice(args.n_samples, size=sample_count, replace=False).tolist())

    for sample_idx, (state_idx, action) in enumerate(tqdm(schedule, desc="Collecting dataset")):
        start_position = base_env.idx_to_state[state_idx]
        time_step = env.reset(options={"start_position": start_position})
        next_time_step = env.step(action)

        obs = np.asarray(time_step.observation).copy()
        next_obs = np.asarray(next_time_step.observation).copy()
        proprio_obs = np.asarray(time_step.proprio_observation).copy()
        next_proprio_obs = np.asarray(next_time_step.proprio_observation).copy()
        reward = float(next_time_step.reward)
        discount = float(next_time_step.discount)
        next_state_idx = int(np.argmax(next_proprio_obs))
        next_position = base_env.idx_to_state[next_state_idx]

        episode = make_episode(
            obs=obs,
            action=int(action),
            reward=reward,
            discount=discount,
            next_obs=next_obs,
            action_dtype=action_dtype,
        )
        save_episode(episode, replay_dir / f"collected_{sample_idx:08d}_1.npz")

        observations.append(obs)
        next_observations.append(next_obs)
        proprio_observations.append(proprio_obs)
        next_proprio_observations.append(next_proprio_obs)
        actions.append(int(action))
        rewards.append(reward)
        discounts.append(discount)
        source_state_indices.append(state_idx)
        next_state_indices.append(next_state_idx)
        source_positions.append(tuple(map(int, start_position)))
        next_positions.append(tuple(map(int, next_position)))

        source_counts[state_idx] += 1
        next_counts[next_state_idx] += 1
        action_counts[action] += 1
        state_action_counts[state_idx, action] += 1

        if sample_idx in sampled_preview_ids:
            sample_observations.append(obs)

    observations_arr = np.stack(observations)
    next_observations_arr = np.stack(next_observations)
    proprio_arr = np.stack(proprio_observations)
    next_proprio_arr = np.stack(next_proprio_observations)
    actions_arr = np.asarray(actions, dtype=action_dtype)
    rewards_arr = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)
    discounts_arr = np.asarray(discounts, dtype=np.float32).reshape(-1, 1)
    source_state_arr = np.asarray(source_state_indices, dtype=np.int64)
    next_state_arr = np.asarray(next_state_indices, dtype=np.int64)
    source_positions_arr = np.asarray(source_positions, dtype=np.int64)
    next_positions_arr = np.asarray(next_positions, dtype=np.int64)

    env_cfg = OmegaConf.to_container(cfg.env, resolve=True)
    summary = {
        "n_samples": args.n_samples,
        "obs_mode": args.obs_mode,
        "seed": args.seed,
        "config_name": args.config_name,
        "task_name": str(cfg.env.name),
        "frame_stack": int(cfg.frame_stack),
        "resolution": int(cfg.resolution),
        "grayscale": bool(cfg.grayscale),
        "env": env_cfg,
        "num_states": int(base_env.n_states),
        "num_actions": int(env.action_space.n),
        "state_coverage": deviation_summary(source_counts[valid_state_indices]),
        "action_coverage": deviation_summary(action_counts),
        "next_state_coverage": deviation_summary(next_counts[valid_state_indices]),
        "replay_buffer_dir": str(replay_dir),
    }

    dataset_path = output_dir / "dataset.npz"
    np.savez_compressed(
        dataset_path,
        observation=observations_arr,
        action=actions_arr,
        reward=rewards_arr,
        discount=discounts_arr,
        next_observation=next_observations_arr,
        proprio_observation=proprio_arr,
        next_proprio_observation=next_proprio_arr,
        state_index=source_state_arr,
        next_state_index=next_state_arr,
        state_position=source_positions_arr,
        next_state_position=next_positions_arr,
        state_counts=source_counts,
        next_state_counts=next_counts,
        action_counts=action_counts,
        state_action_counts=state_action_counts,
        metadata_json=np.asarray(json.dumps(summary)),
    )

    plot_state_heatmap(
        output_dir / "state_hist.png",
        base_env,
        source_counts,
        title="Source State Coverage",
    )
    plot_action_histogram(output_dir / "action_hist.png", action_counts)
    plot_source_next_comparison(
        output_dir / "coverage_comparison.png",
        base_env,
        source_counts,
        next_counts,
    )
    plot_sample_grid(
        output_dir / "sample_grid.png",
        np.asarray(sample_observations),
        args.obs_mode,
    )

    summary_path = output_dir / "coverage_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Saved flat dataset to {dataset_path}")
    print(f"Saved replay-compatible one-step episodes to {replay_dir}")
    print(f"Saved coverage summary to {summary_path}")
    print("State coverage:", json.dumps(summary["state_coverage"], indent=2))
    print("Action coverage:", json.dumps(summary["action_coverage"], indent=2))


if __name__ == "__main__":
    main()
