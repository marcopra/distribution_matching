"""Convert saved replay buffers into lineplot-ready coverage CSV files."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import env.multiple_rooms  # noqa: F401 - registers MultipleRooms-v0
import env.maze  # noqa: F401 - registers Maze-v0
import env.rooms  # noqa: F401 - registers room envs

try:
    from .utils import Logger
except ImportError:
    from new_plot_maker.utils import Logger


RUN_PREFIX_RE = re.compile(r"^\d{6}_\d+_(?P<algorithm>.+)$")
BUFFER_FILE_RE = re.compile(r".*_(?P<index>\d+)_(?P<declared_len>\d+)\.npz$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare cumulative state-coverage CSVs for make_lineplot.py."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data_coverage/raw/maze/2026.05.05_states"),
        help="Folder containing Hydra run subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_coverage/processed/maze_108_states_coverage"),
        help="Directory where lineplot-ready CSV files will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV files in the output directory.",
    )
    parser.add_argument(
        "--pixel-nearest-batch-size",
        type=int,
        default=64,
        help="Batch size for nearest-rendered-state fallback on pixel observations.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-run diagnostics about replay files, sample counts, and env metadata.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def infer_algorithm(run_dir: Path, cfg: dict[str, Any]) -> str:
    agent_name = cfg.get("agent", {}).get("name")
    if agent_name:
        return str(agent_name)

    match = RUN_PREFIX_RE.match(run_dir.name)
    if match:
        return match.group("algorithm")
    return run_dir.name


def make_env_from_run_config(cfg: dict[str, Any]) -> gym.Env:
    env_cfg = dict(cfg.get("env", {}))
    env_name = env_cfg.pop("name")
    return gym.make(env_name, **env_cfg)


def make_wrapped_env_from_run_config(cfg: dict[str, Any]) -> gym.Env:
    import gym_env

    env_cfg = dict(cfg.get("env", {}))
    env_name = env_cfg.pop("name")
    task_name = cfg.get("task_name", env_name)
    if not task_name or str(task_name).startswith("${"):
        task_name = env_name
    return gym_env.make(
        task_name,
        cfg.get("obs_type", "states"),
        frame_stack=int(cfg.get("frame_stack", 1)),
        action_repeat=int(cfg.get("action_repeat", 1)),
        seed=cfg.get("seed", None),
        resolution=int(cfg.get("resolution", 84)),
        grayscale=bool(cfg.get("grayscale", False)),
        url=False,
        **env_cfg,
    )


def decode_state_indices(observations: np.ndarray) -> np.ndarray:
    observations = np.asarray(observations)
    if observations.ndim != 2:
        raise ValueError(
            f"Expected one-hot observations with shape [T, n_states], got {observations.shape}"
        )
    return np.argmax(observations, axis=1)


def extract_spatial_cell(state: Any) -> tuple[int, int]:
    state_tuple = tuple(np.asarray(state).tolist())
    if len(state_tuple) < 2:
        raise ValueError(f"Expected a state with at least two coordinates, got {state_tuple}")
    return tuple(state_tuple[:2])


def get_dead_spatial_cell(env: gym.Env) -> tuple[int, int] | None:
    dead_state = getattr(env.unwrapped, "DEAD_STATE", None)
    if dead_state is None:
        return None
    return extract_spatial_cell(dead_state)


def get_spatial_cells(env: gym.Env) -> list[tuple[int, int]]:
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


def build_state_to_spatial_id(env: gym.Env) -> tuple[np.ndarray, list[tuple[int, int]]]:
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


def update_visited_mask(visited_cells: np.ndarray, spatial_ids: np.ndarray) -> None:
    valid_ids = spatial_ids[spatial_ids >= 0]
    if valid_ids.size > 0:
        visited_cells[np.unique(valid_ids)] = True


def observation_key(observation: np.ndarray) -> tuple[tuple[int, ...], str, bytes]:
    observation = np.ascontiguousarray(observation)
    return (observation.shape, observation.dtype.str, observation.tobytes())


def render_all_state_observations(env: gym.Env) -> tuple[np.ndarray, dict[Any, int]]:
    rendered_observations = []
    key_to_state = {}
    duplicate_states = []

    for state_idx in range(env.unwrapped.n_states):
        time_step = env.reset(options={"start_state": state_idx})
        observation = np.asarray(time_step.observation)
        rendered_observations.append(observation)

        key = observation_key(observation)
        if key in key_to_state:
            duplicate_states.append((key_to_state[key], state_idx))
        key_to_state[key] = state_idx

    if duplicate_states:
        pairs = ", ".join(f"{a}/{b}" for a, b in duplicate_states[:5])
        raise ValueError(
            "Pixel matching is ambiguous because some rendered states are identical: "
            f"{pairs}"
        )

    return np.stack(rendered_observations, axis=0), key_to_state


def nearest_rendered_state_indices(
    observations: np.ndarray,
    rendered_observations: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    observations = observations.reshape(observations.shape[0], -1).astype(np.float32)
    rendered = rendered_observations.reshape(rendered_observations.shape[0], -1).astype(np.float32)

    rendered_norms = np.sum(rendered * rendered, axis=1)
    nearest_indices = []
    nearest_distances = []
    batch_size = max(1, int(batch_size))

    for start in range(0, observations.shape[0], batch_size):
        batch = observations[start:start + batch_size]
        batch_norms = np.sum(batch * batch, axis=1, keepdims=True)
        distances = batch_norms + rendered_norms[None, :] - 2.0 * batch @ rendered.T
        distances = np.maximum(distances, 0.0)
        indices = np.argmin(distances, axis=1)
        nearest_indices.append(indices)
        nearest_distances.append(distances[np.arange(len(indices)), indices])

    return np.concatenate(nearest_indices), np.concatenate(nearest_distances)


def looks_like_one_hot_states(observations: np.ndarray, n_states: int) -> bool:
    observations = np.asarray(observations)
    if observations.ndim != 2 or observations.shape[1] != n_states:
        return False
    if observations.size == 0:
        return False

    row_sums = observations.sum(axis=1)
    row_max = observations.max(axis=1)
    row_min = observations.min(axis=1)
    return bool(
        np.allclose(row_sums, 1.0)
        and np.allclose(row_max, 1.0)
        and np.allclose(row_min, 0.0)
    )


def load_episode(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as f:
        episode = np.load(f)
        return {key: episode[key] for key in episode.files}


def csv_safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def buffer_file_metadata(path: Path) -> tuple[int | None, int | None]:
    match = BUFFER_FILE_RE.match(path.name)
    if match is None:
        return None, None
    return int(match.group("index")), int(match.group("declared_len"))


def summarize_sequence(values: list[int]) -> str:
    if not values:
        return "none"
    unique_values = sorted(set(values))
    if len(unique_values) == 1:
        return str(unique_values[0])
    return f"{unique_values[0]}..{unique_values[-1]} ({len(unique_values)} unique)"


def log_run_debug_header(
    run_dir: Path,
    algorithm: str,
    cfg: dict[str, Any],
    npz_files: list[Path],
    debug: bool,
) -> None:
    if not debug:
        return

    indices = []
    declared_lengths = []
    for npz_file in npz_files:
        index, declared_len = buffer_file_metadata(npz_file)
        if index is not None:
            indices.append(index)
        if declared_len is not None:
            declared_lengths.append(declared_len)

    Logger.subsection(run_dir.name)
    Logger.detail(f"algorithm: {algorithm}")
    Logger.detail(
        "env: "
        f"{cfg.get('env', {}).get('name', 'unknown')} | "
        f"obs_type: {cfg.get('obs_type', 'unknown')} | "
        f"num_train_frames: {cfg.get('num_train_frames', 'unknown')} | "
        f"max_steps: {cfg.get('env', {}).get('max_steps', 'unknown')}"
    )
    Logger.detail(f"buffer files: {len(npz_files)}")
    if npz_files:
        Logger.detail(f"first buffer: {npz_files[0].name}")
        Logger.detail(f"last buffer: {npz_files[-1].name}")
    if indices:
        missing = sorted(set(range(min(indices), max(indices) + 1)) - set(indices))
        Logger.detail(f"buffer indices: {summarize_sequence(indices)}")
        if missing:
            preview = ", ".join(str(index) for index in missing[:12])
            suffix = "..." if len(missing) > 12 else ""
            Logger.detail(f"missing buffer indices: {preview}{suffix}", color="yellow")
    if declared_lengths:
        Logger.detail(f"declared file lengths: {summarize_sequence(declared_lengths)}")


def log_run_debug_footer(
    run_dir: Path,
    cfg: dict[str, Any],
    episode_lengths: list[int],
    total_samples: int,
    env: gym.Env,
    total_cells: int,
    coverage_source: str,
    debug: bool,
) -> None:
    if not debug:
        return

    expected_samples = cfg.get("num_train_frames")
    Logger.detail(
        "observed episode lengths: "
        f"{summarize_sequence(episode_lengths)} | total_samples: {total_samples}"
    )
    Logger.detail(
        f"env states: {getattr(env.unwrapped, 'n_states', 'unknown')} | "
        f"coverage cells: {total_cells} | coverage source: {coverage_source}"
    )
    if isinstance(expected_samples, int) and expected_samples > 0:
        ratio = total_samples / expected_samples
        Logger.detail(f"processed/expected samples: {ratio:.1%}")
        if ratio < 0.9:
            Logger.item(
                f"{run_dir.name}: processed only {total_samples} samples, "
                f"but config num_train_frames is {expected_samples}",
                color="yellow",
            )


def coverage_rows(
    run_dir: Path,
    algorithm: str,
    cfg: dict[str, Any],
    pixel_nearest_batch_size: int,
    debug: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    npz_files = sorted((run_dir / "buffer").glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz replay files found in {run_dir / 'buffer'}")
    log_run_debug_header(run_dir, algorithm, cfg, npz_files, debug)

    first_episode = load_episode(npz_files[0])
    if "observation" not in first_episode:
        raise KeyError(f"Missing 'observation' in {npz_files[0]}")

    raw_env = make_env_from_run_config(cfg)
    uses_one_hot = looks_like_one_hot_states(first_episode["observation"], raw_env.unwrapped.n_states)
    env = raw_env
    rendered_observations = None
    key_to_state = None
    rendered_shape = None
    nearest_fallback_unique = 0
    nearest_fallback_total = 0
    nearest_max_squared_distance = 0.0
    nearest_distance_sum = 0.0

    if uses_one_hot:
        coverage_source = "one_hot_observation"
    else:
        coverage_source = "pixel_render_matching"
        env = make_wrapped_env_from_run_config(cfg)
        rendered_observations, key_to_state = render_all_state_observations(env)
        rendered_shape = rendered_observations.shape[1:]

    state_to_spatial_id, spatial_cells = build_state_to_spatial_id(env)
    total_cells = len(spatial_cells)
    visited_cells = np.zeros(total_cells, dtype=bool)
    total_samples = 0
    episode_lengths: list[int] = []

    rows: list[dict[str, Any]] = [
        {
            "num_samples": 0,
            "visited_cells": 0,
            "total_cells": total_cells,
            "coverage_pct": 0.0,
            "__group": algorithm,
            "__run_id": run_dir.name,
            "__run_name": run_dir.name,
            "__source_dir": str(run_dir),
            "coverage_source": coverage_source,
        }
    ]

    for npz_file in npz_files:
        episode = load_episode(npz_file)
        if "observation" not in episode:
            raise KeyError(f"Missing 'observation' in {npz_file}")

        observations = np.asarray(episode["observation"])
        episode_lengths.append(len(observations))
        if uses_one_hot:
            state_indices = decode_state_indices(observations)
        else:
            if tuple(observations.shape[1:]) != tuple(rendered_shape):
                raise ValueError(
                    f"Observation shape mismatch in {npz_file}: buffer has "
                    f"{observations.shape[1:]}, rendered states have {rendered_shape}. "
                    "Check obs_type, frame_stack, resolution, grayscale, and env config."
                )

            state_indices = np.empty(len(observations), dtype=np.int64)
            unmatched_indices = []
            unmatched_observations = []
            for idx, observation in enumerate(observations):
                state_idx = key_to_state.get(observation_key(observation))
                if state_idx is None:
                    unmatched_indices.append(idx)
                    unmatched_observations.append(observation)
                else:
                    state_indices[idx] = state_idx

            if unmatched_observations:
                unmatched_array = np.stack(unmatched_observations, axis=0)
                nearest_indices, nearest_distances = nearest_rendered_state_indices(
                    unmatched_array,
                    rendered_observations,
                    pixel_nearest_batch_size,
                )
                state_indices[np.asarray(unmatched_indices, dtype=np.int64)] = nearest_indices
                nearest_fallback_unique += len({observation_key(obs) for obs in unmatched_observations})
                nearest_fallback_total += len(unmatched_observations)
                nearest_max_squared_distance = max(
                    nearest_max_squared_distance,
                    float(np.max(nearest_distances)),
                )
                nearest_distance_sum += float(np.sum(nearest_distances))

        if np.any(state_indices >= env.unwrapped.n_states):
            raise ValueError(
                f"Decoded state index {int(state_indices.max())} is outside "
                f"{run_dir.name}'s environment with {env.unwrapped.n_states} states"
            )

        update_visited_mask(visited_cells, state_to_spatial_id[state_indices])
        total_samples += len(state_indices)
        visited_count = int(visited_cells.sum())

        rows.append(
            {
                "num_samples": total_samples,
                "visited_cells": visited_count,
                "total_cells": total_cells,
                "coverage_pct": 100.0 * visited_count / total_cells,
                "__group": algorithm,
                "__run_id": run_dir.name,
                "__run_name": run_dir.name,
                "__source_dir": str(run_dir),
                "coverage_source": coverage_source,
            }
        )

    if not uses_one_hot and nearest_fallback_total > 0:
        mean_distance = nearest_distance_sum / nearest_fallback_total
        Logger.item(
            f"{run_dir.name}: nearest pixel fallback used for "
            f"{nearest_fallback_total} observations "
            f"({nearest_fallback_unique} unique, mean sq dist {mean_distance:.3f}, "
            f"max sq dist {nearest_max_squared_distance:.3f})",
            color="yellow",
        )
    log_run_debug_footer(
        run_dir=run_dir,
        cfg=cfg,
        episode_lengths=episode_lengths,
        total_samples=total_samples,
        env=env,
        total_cells=total_cells,
        coverage_source=coverage_source,
        debug=debug,
    )
    return rows, coverage_source


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "num_samples",
        "visited_cells",
        "total_cells",
        "coverage_pct",
        "__group",
        "__run_id",
        "__run_name",
        "__source_dir",
        "coverage_source",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clear_existing_output_csvs(output_dir: Path) -> None:
    if not output_dir.exists():
        return

    csv_paths = sorted(output_dir.glob("*.csv"))
    if not csv_paths:
        return

    for csv_path in csv_paths:
        csv_path.unlink()
    Logger.item(f"removed {len(csv_paths)} existing CSV file(s) from {output_dir}", color="yellow")


def log_aggregate_diagnostics(summaries: list[dict[str, Any]], debug: bool) -> None:
    if not summaries:
        return

    max_samples = max(int(summary["num_samples"]) for summary in summaries)
    short_runs = [
        summary for summary in summaries
        if int(summary["num_samples"]) < max_samples
    ]
    total_cells = sorted({int(summary["total_cells"]) for summary in summaries})
    env_names = sorted({str(summary["env_name"]) for summary in summaries})

    Logger.subsection("Coverage Diagnostics")
    Logger.detail(f"max processed samples: {max_samples}")
    Logger.detail(f"env names: {', '.join(env_names)}")
    Logger.detail(f"coverage cell counts: {', '.join(str(value) for value in total_cells)}")

    if short_runs:
        Logger.item(
            f"{len(short_runs)} run(s) end before the max processed sample count",
            color="yellow",
        )
        for summary in sorted(short_runs, key=lambda item: int(item["num_samples"])):
            Logger.detail(
                f"{summary['run_id']} ({summary['algorithm']}, {summary['env_name']}): "
                f"{summary['num_samples']} / {max_samples} samples, "
                f"total_cells={summary['total_cells']}",
                color="yellow",
            )
    elif debug:
        Logger.detail("all runs end at the same processed sample count")

    if len(total_cells) > 1:
        Logger.item(
            "runs have different coverage cell counts; this usually means mixed envs/configs",
            color="yellow",
        )
        for summary in summaries:
            Logger.detail(
                f"{summary['run_id']}: env={summary['env_name']}, "
                f"total_cells={summary['total_cells']}"
            )


def discover_run_dirs(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_dir() and (path / ".hydra" / "config.yaml").exists()
    )


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    Logger.section("Replay Coverage Preparation")
    Logger.item(f"input: {input_dir}")
    Logger.item(f"output: {output_dir}")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if output_dir.exists() and not args.overwrite and any(output_dir.glob("*.csv")):
        raise FileExistsError(
            f"{output_dir} already contains CSV files. Re-run with --overwrite."
        )
    if args.overwrite:
        clear_existing_output_csvs(output_dir)

    written = []
    metadata_rows = []
    run_summaries = []
    for run_dir in discover_run_dirs(input_dir):
        cfg = load_yaml(run_dir / ".hydra" / "config.yaml")
        algorithm = infer_algorithm(run_dir, cfg)
        rows, coverage_source = coverage_rows(
            run_dir,
            algorithm,
            cfg,
            pixel_nearest_batch_size=args.pixel_nearest_batch_size,
            debug=args.debug,
        )

        output_path = output_dir / f"{csv_safe(algorithm)}___{csv_safe(run_dir.name)}.csv"
        write_csv(output_path, rows)
        written.append(output_path)

        final = rows[-1]
        env_name = str(cfg.get("env", {}).get("name", "unknown"))
        buffer_file_count = max(0, len(rows) - 1)
        metadata_rows.append(
            {
                "run_id": run_dir.name,
                "algorithm": algorithm,
                "env_name": env_name,
                "buffer_files": buffer_file_count,
                "num_samples": final["num_samples"],
                "visited_cells": final["visited_cells"],
                "total_cells": final["total_cells"],
                "coverage_pct": f"{final['coverage_pct']:.6f}",
                "coverage_source": coverage_source,
                "csv_path": str(output_path),
            }
        )
        run_summaries.append(
            {
                "run_id": run_dir.name,
                "algorithm": algorithm,
                "env_name": env_name,
                "num_samples": final["num_samples"],
                "total_cells": final["total_cells"],
                "csv_path": str(output_path),
            }
        )
        Logger.item(
            f"{run_dir.name}: {algorithm}, final coverage "
            f"{final['coverage_pct']:.2f}% ({final['visited_cells']}/{final['total_cells']})",
            color="green",
        )

    if not written:
        raise FileNotFoundError(f"No Hydra run directories found below {input_dir}")

    metadata_path = output_dir / "runs_metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "algorithm",
                "env_name",
                "buffer_files",
                "num_samples",
                "visited_cells",
                "total_cells",
                "coverage_pct",
                "coverage_source",
                "csv_path",
            ],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    log_aggregate_diagnostics(run_summaries, args.debug)
    Logger.item(f"wrote {len(written)} run CSV files", color="green")
    Logger.item(f"metadata: {metadata_path}", color="green")


if __name__ == "__main__":
    main()
