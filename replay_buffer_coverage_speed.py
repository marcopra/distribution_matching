import csv
import math
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

import gym_env


def load_episode(fn):
    with fn.open("rb") as f:
        episode = np.load(f)
        return {k: episode[k] for k in episode.keys()}


def decode_state_indices(observations):
    observations = np.asarray(observations)
    if observations.ndim != 2:
        raise ValueError(
            f"Expected one-hot observations with shape [T, n_states], got {observations.shape}"
        )
    return np.argmax(observations, axis=1)


def extract_spatial_cell(state):
    state = tuple(np.asarray(state).tolist())
    if len(state) < 2:
        raise ValueError(f"Expected a state with at least two coordinates, got {state}")
    return tuple(state[:2])


def get_dead_spatial_cell(env):
    dead_state = getattr(env.unwrapped, "DEAD_STATE", None)
    if dead_state is None:
        return None
    return extract_spatial_cell(dead_state)


def get_spatial_cells(env):
    unwrapped = env.unwrapped
    dead_cell = get_dead_spatial_cell(env)

    if hasattr(unwrapped, "plot_cells") and len(unwrapped.plot_cells) > 0:
        cells = [tuple(cell[:2]) for cell in unwrapped.plot_cells]
    elif hasattr(unwrapped, "cells") and len(unwrapped.cells) > 0:
        cells = [extract_spatial_cell(cell) for cell in unwrapped.cells]
    else:
        cells = [
            extract_spatial_cell(unwrapped.idx_to_state[idx])
            for idx in range(unwrapped.n_states)
        ]

    unique_cells = []
    seen = set()
    for cell in cells:
        if dead_cell is not None and cell == dead_cell:
            continue
        if cell not in seen:
            unique_cells.append(cell)
            seen.add(cell)
    return unique_cells


def build_state_to_spatial_id(env):
    unwrapped = env.unwrapped
    dead_cell = get_dead_spatial_cell(env)
    spatial_cells = get_spatial_cells(env)
    spatial_cell_to_id = {cell: idx for idx, cell in enumerate(spatial_cells)}

    state_to_spatial_id = np.full(unwrapped.n_states, -1, dtype=np.int64)
    for state_idx in range(unwrapped.n_states):
        cell = extract_spatial_cell(unwrapped.idx_to_state[state_idx])
        if dead_cell is not None and cell == dead_cell:
            continue
        state_to_spatial_id[state_idx] = spatial_cell_to_id[cell]

    return state_to_spatial_id, spatial_cells


def normalize_target_percentages(values):
    targets = []
    for value in values:
        pct = float(value)
        if 0.0 < pct < 1.0:
            pct *= 100.0
        if pct <= 0.0 or pct > 100.0:
            raise ValueError(
                "target_coverage_pcts must contain percentages in (0, 100] "
                "or fractions in (0, 1)"
            )
        targets.append(pct)
    return sorted(set(targets))


def get_max_samples(cfg):
    max_samples = cfg.get("max_samples", None)
    if max_samples is None:
        return None

    max_samples = int(max_samples)
    if max_samples <= 0:
        raise ValueError(f"max_samples must be positive or null, got {max_samples}")
    return max_samples


def reached_result(
    target_pct,
    required_cells,
    num_samples,
    num_episodes,
    episode_file,
    visited_cells,
    total_cells,
):
    coverage_pct = 100.0 * visited_cells / total_cells
    return {
        "target_coverage_pct": target_pct,
        "required_cells": required_cells,
        "reached": True,
        "num_samples": num_samples,
        "num_episodes": num_episodes,
        "episode_file": episode_file,
        "visited_cells": visited_cells,
        "total_cells": total_cells,
        "coverage_pct": coverage_pct,
    }


def unreached_result(target_pct, required_cells, num_samples, num_episodes, visited_cells, total_cells):
    coverage_pct = 100.0 * visited_cells / total_cells
    return {
        "target_coverage_pct": target_pct,
        "required_cells": required_cells,
        "reached": False,
        "num_samples": num_samples,
        "num_episodes": num_episodes,
        "episode_file": "",
        "visited_cells": visited_cells,
        "total_cells": total_cells,
        "coverage_pct": coverage_pct,
    }


def compute_coverage_speed(npz_files, env, target_pcts, max_samples):
    state_to_spatial_id, spatial_cells = build_state_to_spatial_id(env)
    total_cells = len(spatial_cells)
    if total_cells == 0:
        raise ValueError("Could not infer any valid spatial cells from the environment")

    targets = [
        {
            "target_pct": pct,
            "required_cells": int(math.ceil(total_cells * pct / 100.0)),
        }
        for pct in target_pcts
    ]

    visited = np.zeros(total_cells, dtype=bool)
    visited_count = 0
    total_samples = 0
    num_episodes = 0
    results = []
    target_idx = 0

    for npz_file in npz_files:
        if max_samples is not None and total_samples >= max_samples:
            break

        episode = load_episode(npz_file)
        if "observation" not in episode:
            raise KeyError(f"Missing 'observation' in {npz_file}")

        state_indices = decode_state_indices(episode["observation"])
        if np.any(state_indices >= env.unwrapped.n_states):
            raise ValueError(
                f"Found decoded states outside the environment range in {npz_file}: "
                f"max index {state_indices.max()}, env has {env.unwrapped.n_states} states"
            )

        spatial_ids = state_to_spatial_id[state_indices]
        if max_samples is not None:
            spatial_ids = spatial_ids[: max_samples - total_samples]

        num_episodes += 1
        for spatial_id in spatial_ids:
            total_samples += 1

            if spatial_id >= 0 and not visited[spatial_id]:
                visited[spatial_id] = True
                visited_count += 1

            while (
                target_idx < len(targets)
                and visited_count >= targets[target_idx]["required_cells"]
            ):
                target = targets[target_idx]
                results.append(
                    reached_result(
                        target["target_pct"],
                        target["required_cells"],
                        total_samples,
                        num_episodes,
                        npz_file.name,
                        visited_count,
                        total_cells,
                    )
                )
                target_idx += 1

            if target_idx >= len(targets):
                return results, total_samples, num_episodes, visited_count, total_cells

    for target in targets[target_idx:]:
        results.append(
            unreached_result(
                target["target_pct"],
                target["required_cells"],
                total_samples,
                num_episodes,
                visited_count,
                total_cells,
            )
        )

    return results, total_samples, num_episodes, visited_count, total_cells


def write_results_csv(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target_coverage_pct",
        "required_cells",
        "reached",
        "num_samples",
        "num_episodes",
        "episode_file",
        "visited_cells",
        "total_cells",
        "coverage_pct",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def plot_samples_to_coverage(results, output_path, title=None):
    reached = [row for row in results if row["reached"]]
    unreached = [row for row in results if not row["reached"]]

    if not reached and not unreached:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    if reached:
        target_pcts = [row["target_coverage_pct"] for row in reached]
        num_samples = [row["num_samples"] for row in reached]
        ax.plot(num_samples, target_pcts, marker="o", linewidth=2, label="Reached")
        for x, y in zip(num_samples, target_pcts):
            ax.annotate(
                f"{x}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
            )

    if unreached:
        target_pcts = [row["target_coverage_pct"] for row in unreached]
        num_samples = [row["num_samples"] for row in unreached]
        ax.scatter(num_samples, target_pcts, marker="x", s=60, label="Not reached")

    ax.set_xlabel("Samples needed")
    ax.set_ylabel("Coverage reached (%)")
    ax.set_title(title or "Replay Buffer Coverage Speed")
    ax.set_ylim(0, 102)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@hydra.main(config_path="configs", config_name="replay_buffer_coverage_speed", version_base="1.1")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    replay_dir = Path(cfg.replay_buffer_dir).resolve()
    npz_files = sorted(replay_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {replay_dir}")

    target_pcts = normalize_target_percentages(cfg.target_coverage_pcts)
    max_samples = get_max_samples(cfg)

    env_kwargs = OmegaConf.to_container(cfg.env, resolve=True) if hasattr(cfg, "env") else {}
    env_kwargs.pop("name", None)
    env = gym_env.make(
        cfg.task_name,
        cfg.obs_type,
        frame_stack=cfg.frame_stack,
        action_repeat=cfg.action_repeat,
        seed=cfg.seed,
        resolution=cfg.resolution,
        grayscale=cfg.grayscale,
        url=False,
        **env_kwargs,
    )

    results, total_samples, num_episodes, visited_count, total_cells = compute_coverage_speed(
        npz_files,
        env,
        target_pcts,
        max_samples,
    )

    output_path = Path(cfg.output_path) if cfg.output_path is not None else None
    if output_path is not None:
        write_results_csv(results, output_path)
        print(f"Saved coverage speed summary to {output_path.resolve()}")

    plot_path = Path(cfg.plot_path) if cfg.get("plot_path", None) is not None else None
    if plot_path is not None:
        title = f"Samples to Reach Coverage ({total_samples} samples scanned)"
        plot_samples_to_coverage(results, plot_path, title=title)
        print(f"Saved coverage speed plot to {plot_path.resolve()}")

    print(f"Loaded episodes from {replay_dir}")
    print(f"Processed {total_samples} observations across {num_episodes} episodes")
    print(f"Final coverage: {visited_count}/{total_cells} cells ({100.0 * visited_count / total_cells:.2f}%)")
    print("")
    print("target_coverage_pct, reached, num_samples, num_episodes, visited_cells, total_cells, coverage_pct")
    for row in results:
        print(
            f"{row['target_coverage_pct']:.2f}, {row['reached']}, "
            f"{row['num_samples']}, {row['num_episodes']}, "
            f"{row['visited_cells']}, {row['total_cells']}, "
            f"{row['coverage_pct']:.2f}"
        )


if __name__ == "__main__":
    main()
