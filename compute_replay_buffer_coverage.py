import csv
from pathlib import Path

import hydra
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
        cells = [extract_spatial_cell(unwrapped.idx_to_state[idx]) for idx in range(unwrapped.n_states)]

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


def update_visited_mask(visited_cells, spatial_ids):
    valid_ids = spatial_ids[spatial_ids >= 0]
    if valid_ids.size > 0:
        visited_cells[np.unique(valid_ids)] = True


def checkpoint_result(num_samples, visited_cells, total_cells):
    visited_count = int(visited_cells.sum())
    return {
        "num_samples": int(num_samples),
        "visited_cells": visited_count,
        "total_cells": int(total_cells),
        "coverage_pct": 100.0 * visited_count / total_cells,
    }


def write_results_csv(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["num_samples", "visited_cells", "total_cells", "coverage_pct"],
        )
        writer.writeheader()
        writer.writerows(results)


@hydra.main(config_path="configs", config_name="compute_replay_buffer_coverage", version_base="1.1")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    replay_dir = Path(cfg.replay_buffer_dir).resolve()
    npz_files = sorted(replay_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {replay_dir}")

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

    checkpoints = sorted({int(x) for x in cfg.coverage_intervals if int(x) > 0})
    if not checkpoints:
        raise ValueError("coverage_intervals must contain at least one positive integer")

    state_to_spatial_id, spatial_cells = build_state_to_spatial_id(env)
    total_cells = len(spatial_cells)
    if total_cells == 0:
        raise ValueError("Could not infer any valid spatial cells from the environment")

    visited_cells = np.zeros(total_cells, dtype=bool)
    total_observations = 0
    results = []
    checkpoint_idx = 0

    for npz_file in npz_files:
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
        cursor = 0

        while cursor < len(spatial_ids):
            if checkpoint_idx >= len(checkpoints):
                update_visited_mask(visited_cells, spatial_ids[cursor:])
                total_observations += len(spatial_ids) - cursor
                break

            next_checkpoint = checkpoints[checkpoint_idx]
            remaining = next_checkpoint - total_observations
            take = min(remaining, len(spatial_ids) - cursor)

            update_visited_mask(visited_cells, spatial_ids[cursor:cursor + take])
            total_observations += take
            cursor += take

            if total_observations == next_checkpoint:
                results.append(checkpoint_result(total_observations, visited_cells, total_cells))
                checkpoint_idx += 1

    skipped_checkpoints = checkpoints[checkpoint_idx:]

    if cfg.include_final_coverage and (not results or results[-1]["num_samples"] != total_observations):
        results.append(checkpoint_result(total_observations, visited_cells, total_cells))

    if cfg.output_path is not None:
        output_path = Path(cfg.output_path)
        write_results_csv(results, output_path)
        print(f"Saved coverage summary to {output_path.resolve()}")

    print(f"Loaded {len(npz_files)} episodes from {replay_dir}")
    print(f"Processed {total_observations} one-hot observations")
    print(f"Coverage denominator: {total_cells} spatial cells")
    if get_dead_spatial_cell(env) is not None:
        print("Ignored dead-state coverage in the denominator")
    print("")
    print("num_samples, visited_cells, total_cells, coverage_pct")
    for row in results:
        print(
            f"{row['num_samples']}, {row['visited_cells']}, {row['total_cells']}, "
            f"{row['coverage_pct']:.2f}"
        )

    if skipped_checkpoints:
        print("")
        print(
            "Skipped checkpoints beyond dataset size: "
            + ", ".join(str(checkpoint) for checkpoint in skipped_checkpoints)
        )


if __name__ == "__main__":
    main()
