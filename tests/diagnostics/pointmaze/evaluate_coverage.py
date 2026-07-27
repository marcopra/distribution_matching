"""Print PointMaze policy coverage from a snapshot or training-run path.

Examples:
    python tests/diagnostics/pointmaze/evaluate_coverage.py path/to/snapshot.pt
    python tests/diagnostics/pointmaze/evaluate_coverage.py path/to/run_directory
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf

from plot_pointmaze_snapshot_trajectories import (
    find_run_config,
    load_config,
    load_snapshot,
    make_env,
    snapshot_step,
)
from tests.diagnostics.pointmaze.evaluate_nystrom_coverage import (
    collect_trajectories,
    find_final_snapshot,
    free_space_coverage,
)


def resolve_model_path(path: Path) -> tuple[Path, Path]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if path.is_file():
        if path.suffix != ".pt":
            raise ValueError(f"Expected snapshot .pt file or run directory, got: {path}")
        snapshot = path
        config = find_run_config(snapshot)
    else:
        snapshot = find_final_snapshot(path)
        config = path / ".hydra" / "config.yaml"
        if not config.exists() and snapshot is not None:
            config = find_run_config(snapshot)

    if snapshot is None:
        raise FileNotFoundError(f"No snapshot*.pt found under: {path}")
    if config is None or not config.exists():
        raise FileNotFoundError(
            f"No .hydra/config.yaml found for snapshot: {snapshot}"
        )
    return snapshot, config


def environment_name(config_path: Path) -> str:
    cfg = OmegaConf.load(config_path)
    name = str(cfg.env.name).lower()
    if "largedense" in name:
        return "largedense"
    if "umaze" in name:
        return "umaze"
    raise ValueError(f"Unsupported PointMaze environment: {cfg.env.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a PointMaze policy and print wall-free coverage percentage."
    )
    parser.add_argument("path", type=Path, help="Snapshot .pt file or training-run directory.")
    parser.add_argument("--num-trajectories", type=int, default=50)
    parser.add_argument("--largedense-horizon", type=int, default=3000)
    parser.add_argument("--umaze-horizon", type=int, default=1000)
    parser.add_argument("--grid-size", type=int, default=90)
    parser.add_argument("--coverage-radius", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.num_trajectories < 1:
        parser.error("--num-trajectories must be positive")
    if args.largedense_horizon < 1 or args.umaze_horizon < 1:
        parser.error("horizons must be positive")
    if args.grid_size < 2:
        parser.error("--grid-size must be at least 2")
    if args.coverage_radius <= 0:
        parser.error("--coverage-radius must be positive")
    return args


def main() -> None:
    args = parse_args()
    snapshot, config_path = resolve_model_path(args.path)
    environment = environment_name(config_path)
    horizon = (
        args.largedense_horizon
        if environment == "largedense"
        else args.umaze_horizon
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = load_config(config_path)
    env = make_env(cfg, seed=args.seed)
    try:
        agent, payload = load_snapshot(snapshot, torch.device(args.device))
        trajectories = collect_trajectories(
            agent,
            env,
            count=args.num_trajectories,
            horizon=horizon,
            policy_step=snapshot_step(snapshot, payload),
            seed=args.seed,
        )
        if len(trajectories) != args.num_trajectories:
            raise RuntimeError(
                f"Collected {len(trajectories)}/{args.num_trajectories} trajectories"
            )
        covered, free, coverage_pct = free_space_coverage(
            env, trajectories, args.grid_size, args.coverage_radius
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    print(f"Coverage: {coverage_pct:.2f}%")
    print(
        f"Covered points: {covered}/{free} | trajectories: {len(trajectories)} "
        f"| horizon: {horizon} | grid: {args.grid_size}x{args.grid_size} "
        f"| radius: {args.coverage_radius:g}"
    )


if __name__ == "__main__":
    main()
