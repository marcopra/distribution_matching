"""Resample trained snapshots and write lineplot-ready coverage CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
import numpy.random._pickle as numpy_random_pickle

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import env.maze  # noqa: F401 - registers Maze-v0
import env.multiple_rooms  # noqa: F401 - registers MultipleRooms-v0
import env.rooms  # noqa: F401 - registers room envs
import gym_env
import utils

try:
    from .prepare_replay_coverage import (
        build_state_to_spatial_id,
        csv_safe,
        get_spatial_cells,
    )
    from .utils import Logger
except ImportError:
    from new_plot_maker.prepare_replay_coverage import (
        build_state_to_spatial_id,
        csv_safe,
        get_spatial_cells,
    )
    from new_plot_maker.utils import Logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load trained snapshots, resample their policies in the environment, "
            "and save cumulative coverage CSVs for make_lineplot.py."
        )
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
        default=Path("data_coverage/processed/maze_108_states_resampling_test"),
        help="Output folder for lineplot-ready CSV files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing CSVs in the output folder before writing new results.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=110_000,
        help="Maximum sampled observations per snapshot and sampling repeat.",
    )
    parser.add_argument(
        "--num-sampling-runs",
        type=int,
        default=1,
        help="Number of independent resampling repeats per snapshot.",
    )
    parser.add_argument(
        "--log-every-samples",
        type=int,
        default=1_000,
        help="Print progress every this many sampled observations.",
    )
    parser.add_argument(
        "--save-every-samples",
        type=int,
        default=151,
        help="Record a CSV row every this many sampled observations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for environment and stochastic policy sampling.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device used for loaded agents. CPU is safest for old CUDA snapshots.",
    )
    parser.add_argument(
        "--eval-mode",
        action="store_true",
        help="Use deterministic/eval actions where the agent supports it.",
    )
    parser.add_argument(
        "--random-start",
        action="store_true",
        help="Reset each episode from a random valid start state.",
    )
    parser.add_argument(
        "--random-goal",
        action="store_true",
        help="Reset each episode with a random valid goal state.",
    )
    parser.add_argument(
        "--include-reset-observation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Count each reset observation as a sampled observation.",
    )
    parser.add_argument(
        "--snapshot-name",
        default="snapshot.pt",
        help="Snapshot filename to load below each run's models folder.",
    )
    parser.add_argument(
        "--limit-runs",
        type=int,
        default=None,
        help="Optional limit on the number of run folders to process.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def infer_algorithm(run_dir: Path, cfg: dict[str, Any]) -> str:
    agent_name = cfg.get("agent", {}).get("name")
    return str(agent_name) if agent_name else run_dir.name.split("_", 2)[-1]


def discover_run_dirs(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_dir() and (path / ".hydra" / "config.yaml").exists()
    )


def discover_snapshot(run_dir: Path, snapshot_name: str) -> Path | None:
    candidates = sorted((run_dir / "models").rglob(snapshot_name))
    if candidates:
        return candidates[-1]
    fallback = sorted(run_dir.rglob(snapshot_name))
    return fallback[-1] if fallback else None


def make_env_from_config(cfg: dict[str, Any], seed: int):
    env_cfg = dict(cfg.get("env", {}))
    env_name = env_cfg.pop("name")
    return gym_env.make(
        env_name,
        cfg.get("obs_type", "states"),
        frame_stack=int(cfg.get("frame_stack", 1)),
        action_repeat=int(cfg.get("action_repeat", 1)),
        seed=seed,
        resolution=int(cfg.get("resolution", 84)),
        grayscale=bool(cfg.get("grayscale", False)),
        url=False,
        **env_cfg,
    )


def move_agent_to_device(agent: Any, device: str) -> None:
    for value in vars(agent).values():
        if isinstance(value, torch.nn.Module):
            value.to(device)
            value.eval()
    if hasattr(agent, "device"):
        agent.device = device


def load_agent(snapshot_path: Path, device: str) -> tuple[Any, dict[str, Any]]:
    # Some snapshots pickle NumPy bit generators by class object rather than name.
    # NumPy 1.x's unpickler accepts names only unless we add these compatibility keys.
    for bit_generator in (np.random.PCG64, np.random.PCG64DXSM, np.random.MT19937):
        numpy_random_pickle.BitGenerators.setdefault(bit_generator, bit_generator)
    payload = torch.load(snapshot_path, map_location=device, weights_only=False)
    if "agent" not in payload:
        raise KeyError(f"Snapshot {snapshot_path} does not contain an 'agent' key")
    agent = payload["agent"]
    move_agent_to_device(agent, device)
    return agent, payload


def valid_state_indices(env) -> list[int]:
    unwrapped = env.unwrapped
    dead_state = getattr(unwrapped, "DEAD_STATE", None)
    indices = []
    for state_idx in range(unwrapped.n_states):
        if dead_state is not None and tuple(unwrapped.idx_to_state[state_idx]) == dead_state:
            continue
        indices.append(state_idx)
    return indices


def reset_options(env, rng: np.random.Generator, random_start: bool, random_goal: bool) -> dict[str, Any]:
    options: dict[str, Any] = {}
    indices = valid_state_indices(env)
    if random_start:
        options["start_state"] = int(rng.choice(indices))
    if random_goal:
        options["goal_position"] = int(rng.choice(indices))
    return options


def state_index_from_observation(observation: Any) -> int:
    array = np.asarray(observation)
    if array.ndim == 0 or array.size == 1:
        return int(array.item())
    return int(np.argmax(array))


def record_visit(
    time_step: Any,
    state_to_spatial_id: np.ndarray,
    visited_cells: np.ndarray,
) -> bool:
    state_idx = state_index_from_observation(time_step.proprio_observation)
    if state_idx < 0 or state_idx >= len(state_to_spatial_id):
        raise ValueError(
            f"Observed state index {state_idx} outside state map of size {len(state_to_spatial_id)}"
        )
    spatial_id = int(state_to_spatial_id[state_idx])
    if spatial_id < 0:
        return False
    was_new = not bool(visited_cells[spatial_id])
    visited_cells[spatial_id] = True
    return was_new


def should_record_row(num_samples: int, save_every_samples: int, reached_full_now: bool) -> bool:
    if num_samples == 0:
        return True
    if reached_full_now:
        return True
    return save_every_samples > 0 and num_samples % save_every_samples == 0


def append_row(
    rows: list[dict[str, Any]],
    num_samples: int,
    visited_cells: np.ndarray,
    total_cells: int,
    algorithm: str,
    run_id: str,
    sample_run_id: str,
    run_dir: Path,
    snapshot_path: Path,
) -> None:
    visited_count = int(visited_cells.sum())
    rows.append(
        {
            "num_samples": num_samples,
            "visited_cells": visited_count,
            "total_cells": total_cells,
            "coverage_pct": 100.0 * visited_count / total_cells,
            "__group": algorithm,
            "__run_id": sample_run_id,
            "__run_name": sample_run_id,
            "__source_dir": str(run_dir),
            "__snapshot_path": str(snapshot_path),
        }
    )


def call_update_meta(agent: Any, meta: Any, step: int, time_step: Any) -> Any:
    if not hasattr(agent, "update_meta"):
        return meta
    update_meta = agent.update_meta
    try:
        signature = inspect.signature(update_meta)
        if "finetune" in signature.parameters:
            return update_meta(meta, step, time_step, finetune=False)
    except (TypeError, ValueError):
        pass
    return update_meta(meta, step, time_step)


def sample_action(agent: Any, time_step: Any, meta: Any, step: int, eval_mode: bool) -> Any:
    with torch.no_grad(), utils.eval_mode(agent):
        return agent.act(time_step.observation, meta, step, eval_mode=eval_mode)


def sample_snapshot_coverage(
    run_dir: Path,
    cfg: dict[str, Any],
    algorithm: str,
    snapshot_path: Path,
    agent: Any,
    payload: dict[str, Any],
    sample_index: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_hash = int(hashlib.sha1(run_dir.name.encode("utf-8")).hexdigest()[:8], 16)
    sample_seed = int(args.seed + sample_index + 10_000 * (run_hash % 10_000))
    random.seed(sample_seed)
    np.random.seed(sample_seed)
    torch.manual_seed(sample_seed)
    rng = np.random.default_rng(sample_seed)

    env = make_env_from_config(cfg, seed=sample_seed)
    state_to_spatial_id, spatial_cells = build_state_to_spatial_id(env)
    total_cells = len(spatial_cells)
    visited_cells = np.zeros(total_cells, dtype=bool)
    rows: list[dict[str, Any]] = []
    sample_run_id = f"{run_dir.name}_sample{sample_index:03d}"
    policy_step = int(payload.get("_global_step", 0))
    num_samples = 0
    episode = 0
    first_full_coverage_samples = None

    append_row(
        rows,
        num_samples,
        visited_cells,
        total_cells,
        algorithm,
        run_dir.name,
        sample_run_id,
        run_dir,
        snapshot_path,
    )

    Logger.detail(
        f"{sample_run_id}: starting sampling with seed={sample_seed}, "
        f"policy_step={policy_step}, total_cells={total_cells}"
    )

    time_step = env.reset(
        seed=sample_seed,
        options=reset_options(env, rng, args.random_start, args.random_goal) or None,
    )
    meta = agent.init_meta() if hasattr(agent, "init_meta") else {}

    while num_samples < args.max_samples and first_full_coverage_samples is None:
        if args.include_reset_observation and time_step.first():
            reached_new = record_visit(time_step, state_to_spatial_id, visited_cells)
            num_samples += 1
            reached_full_now = bool(visited_cells.all() and reached_new)
            if reached_full_now:
                first_full_coverage_samples = num_samples
            if should_record_row(num_samples, args.save_every_samples, reached_full_now):
                append_row(
                    rows,
                    num_samples,
                    visited_cells,
                    total_cells,
                    algorithm,
                    run_dir.name,
                    sample_run_id,
                    run_dir,
                    snapshot_path,
                )

        if time_step.last():
            episode += 1
            time_step = env.reset(
                options=reset_options(env, rng, args.random_start, args.random_goal) or None
            )
            meta = agent.init_meta() if hasattr(agent, "init_meta") else {}
            continue

        meta = call_update_meta(agent, meta, policy_step + num_samples, time_step)
        action = sample_action(agent, time_step, meta, policy_step + num_samples, args.eval_mode)
        time_step = env.step(action)

        reached_new = record_visit(time_step, state_to_spatial_id, visited_cells)
        num_samples += 1
        reached_full_now = bool(visited_cells.all() and reached_new)
        if reached_full_now:
            first_full_coverage_samples = num_samples

        if should_record_row(num_samples, args.save_every_samples, reached_full_now):
            append_row(
                rows,
                num_samples,
                visited_cells,
                total_cells,
                algorithm,
                run_dir.name,
                sample_run_id,
                run_dir,
                snapshot_path,
            )

        if args.log_every_samples > 0 and num_samples % args.log_every_samples == 0:
            Logger.detail(
                f"{sample_run_id}: samples={num_samples}, "
                f"coverage={int(visited_cells.sum())}/{total_cells} "
                f"({100.0 * visited_cells.sum() / total_cells:.2f}%), "
                f"episodes={episode}"
            )

    if rows[-1]["num_samples"] != num_samples:
        append_row(
            rows,
            num_samples,
            visited_cells,
            total_cells,
            algorithm,
            run_dir.name,
            sample_run_id,
            run_dir,
            snapshot_path,
        )

    summary = {
        "run_id": run_dir.name,
        "sample_run_id": sample_run_id,
        "algorithm": algorithm,
        "env_name": cfg.get("env", {}).get("name", "unknown"),
        "snapshot_path": str(snapshot_path),
        "num_samples": num_samples,
        "episodes": episode,
        "visited_cells": int(visited_cells.sum()),
        "total_cells": total_cells,
        "coverage_pct": f"{100.0 * visited_cells.sum() / total_cells:.6f}",
        "first_full_coverage_samples": first_full_coverage_samples or "",
        "reached_full_coverage": first_full_coverage_samples is not None,
    }
    return rows, summary


def write_run_csv(output_dir: Path, algorithm: str, sample_run_id: str, rows: list[dict[str, Any]]) -> Path:
    output_path = output_dir / f"{csv_safe(algorithm)}___{csv_safe(sample_run_id)}.csv"
    fieldnames = [
        "num_samples",
        "visited_cells",
        "total_cells",
        "coverage_pct",
        "__group",
        "__run_id",
        "__run_name",
        "__source_dir",
        "__snapshot_path",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_metadata(output_dir: Path, summaries: list[dict[str, Any]]) -> Path:
    metadata_path = output_dir / "runs_metadata.csv"
    fieldnames = [
        "run_id",
        "sample_run_id",
        "algorithm",
        "env_name",
        "snapshot_path",
        "num_samples",
        "episodes",
        "visited_cells",
        "total_cells",
        "coverage_pct",
        "first_full_coverage_samples",
        "reached_full_coverage",
        "csv_path",
    ]
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    return metadata_path


def clear_existing_csvs(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    csv_paths = sorted(output_dir.glob("*.csv"))
    for path in csv_paths:
        path.unlink()
    if csv_paths:
        Logger.item(f"removed {len(csv_paths)} existing CSV file(s) from {output_dir}", color="yellow")


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if args.output_dir.exists() and any(args.output_dir.glob("*.csv")) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} already contains CSV files. Use --overwrite.")
    if args.overwrite:
        clear_existing_csvs(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    Logger.section("Resampling Test")
    Logger.item(f"input: {args.input_dir}")
    Logger.item(f"output: {args.output_dir}")
    Logger.item(f"max samples: {args.max_samples}")
    Logger.item(f"sampling repeats per snapshot: {args.num_sampling_runs}")
    Logger.item(f"eval mode: {args.eval_mode}")
    Logger.item(f"random start: {args.random_start}, random goal: {args.random_goal}")

    run_dirs = discover_run_dirs(args.input_dir)
    if args.limit_runs is not None:
        run_dirs = run_dirs[: args.limit_runs]
    Logger.item(f"run folders: {len(run_dirs)}", color="green")

    summaries = []
    skipped = 0
    for run_dir in run_dirs:
        cfg = load_yaml(run_dir / ".hydra" / "config.yaml")
        algorithm = infer_algorithm(run_dir, cfg)
        snapshot_path = discover_snapshot(run_dir, args.snapshot_name)
        if snapshot_path is None:
            Logger.item(f"{run_dir.name}: no {args.snapshot_name} found, skipping", color="yellow")
            skipped += 1
            continue

        Logger.subsection(run_dir.name)
        Logger.detail(f"algorithm: {algorithm}")
        Logger.detail(f"snapshot: {snapshot_path.relative_to(run_dir)}")
        Logger.detail(
            f"env: {cfg.get('env', {}).get('name', 'unknown')} | "
            f"obs_type: {cfg.get('obs_type', 'unknown')}"
        )
        try:
            agent, payload = load_agent(snapshot_path, args.device)
        except Exception as exc:
            Logger.item(f"{run_dir.name}: failed to load snapshot: {exc}", color="red")
            skipped += 1
            continue
        Logger.detail(
            f"loaded agent: {type(agent).__name__}, "
            f"snapshot_step={payload.get('_global_step', 'unknown')}"
        )

        for sample_index in range(args.num_sampling_runs):
            try:
                rows, summary = sample_snapshot_coverage(
                    run_dir=run_dir,
                    cfg=cfg,
                    algorithm=algorithm,
                    snapshot_path=snapshot_path,
                    agent=agent,
                    payload=payload,
                    sample_index=sample_index,
                    args=args,
                )
            except Exception as exc:
                Logger.item(
                    f"{run_dir.name} sample {sample_index}: sampling failed: {exc}",
                    color="red",
                )
                skipped += 1
                continue
            csv_path = write_run_csv(args.output_dir, algorithm, summary["sample_run_id"], rows)
            summary["csv_path"] = str(csv_path)
            summaries.append(summary)
            if summary["reached_full_coverage"]:
                Logger.item(
                    f"{summary['sample_run_id']}: full coverage at "
                    f"{summary['first_full_coverage_samples']} samples",
                    color="green",
                )
            else:
                Logger.item(
                    f"{summary['sample_run_id']}: stopped at "
                    f"{summary['visited_cells']}/{summary['total_cells']} cells "
                    f"after {summary['num_samples']} samples",
                    color="yellow",
                )

    if not summaries:
        raise RuntimeError("No snapshots were sampled")

    metadata_path = write_metadata(args.output_dir, summaries)
    reached = sum(1 for summary in summaries if summary["reached_full_coverage"])
    Logger.subsection("Summary")
    Logger.item(f"sampled runs: {len(summaries)}", color="green")
    Logger.item(f"reached full coverage: {reached}/{len(summaries)}")
    Logger.item(f"skipped run folders: {skipped}")
    Logger.item(f"metadata: {metadata_path}", color="green")


if __name__ == "__main__":
    main()
