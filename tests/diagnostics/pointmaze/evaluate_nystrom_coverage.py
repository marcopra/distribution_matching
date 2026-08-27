"""Evaluate PointMaze coverage for Rover Nyström sweeps and URL baselines.

Full evaluation:
    python3 tests/diagnostics/pointmaze/evaluate_nystrom_coverage.py \
        --rover-root tests/diagnostics/pointmaze/models \
        --baseline-root models/pointmaze \
        --output-dir tests/outputs/pointmaze/nystrom_coverage \
        --device cuda

Fast graphics/installation check:
    python3 tests/diagnostics/pointmaze/evaluate_nystrom_coverage.py \
        --smoke-test --device cpu
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from omegaconf import OmegaConf

import utils
from agent.rover_visualization.domains import (
    extract_eval_trajectory_point,
    pointmaze_free_space_coverage,
    save_maze_trajectory_overlay_plot,
)
from plot_pointmaze_snapshot_trajectories import (
    _agent_latent_probs,
    _pointmaze_wall_rectangles,
    _reset_valid_pointmaze_start,
    _sample_one_hot,
    load_config,
    load_snapshot,
    make_env,
    random_action,
    snapshot_step,
)


ENVIRONMENTS = ("largedense", "umaze")
ENV_LABELS = {"largedense": "LargeDense", "umaze": "UMaze"}
TRAJECTORY_COUNTS = {"largedense": 50, "umaze": 50}
BASELINES = (
    ("random", "Random"),
    ("smm", "SMM"),
    ("cic", "CIC"),
    ("icm_apt", "APT"),
    ("maxent", "MaxEnt"),
    ("rnd", "RND"),
)
CONFIG_NAME = ".hydra/config.yaml"
SNAPSHOT_PATTERN = re.compile(r"snapshot_(\d+)\.*pt$")


@dataclass(frozen=True)
class PolicySpec:
    environment: str
    algorithm: str
    run_name: str
    config_path: Path
    snapshot_path: Path | None
    nystrom_points: int | None = None
    bandwidth: float | None = None
    checkpoint_step: int = 0


@dataclass
class Result:
    environment: str
    algorithm: str
    run_name: str
    nystrom_points: int | None
    bandwidth: float | None
    trajectories: int
    horizon: int
    covered_points: int
    free_points: int
    coverage_pct: float


def _resolved_float(cfg: Any, key: str) -> float | None:
    value = cfg.agent.get(key)
    if value is None or isinstance(value, str):
        return None
    return float(value)


def _snapshot_sort_key(path: Path) -> tuple[int, int, str]:
    if path.name == "snapshot.pt":
        return (1, 0, path.name)
    match = SNAPSHOT_PATTERN.fullmatch(path.name)
    return (0, int(match.group(1)), path.name) if match else (-1, -1, path.name)


def find_final_snapshot(root: Path) -> Path | None:
    candidates = [
        path
        for path in root.rglob("snapshot*.pt")
        if path.is_file() and _snapshot_sort_key(path)[0] >= 0
    ]
    return max(candidates, key=_snapshot_sort_key) if candidates else None


def discover_rover_specs(root: Path, strict: bool) -> list[PolicySpec]:
    specs: list[PolicySpec] = []
    missing: list[Path] = []
    for environment in ENVIRONMENTS:
        env_root = root / environment
        for config_path in sorted(env_root.glob(f"*/{CONFIG_NAME}")):
            cfg = OmegaConf.load(config_path)
            run_root = config_path.parents[1]
            snapshot = find_final_snapshot(run_root)
            if snapshot is None:
                missing.append(run_root)
                continue
            specs.append(
                PolicySpec(
                    environment=environment,
                    algorithm="Rover",
                    run_name=run_root.name,
                    config_path=config_path,
                    snapshot_path=snapshot,
                    nystrom_points=int(cfg.agent.subsamples),
                    bandwidth=_resolved_float(cfg, "kernel_bandwidth"),
                    checkpoint_step=_snapshot_sort_key(snapshot)[1],
                )
            )

    if missing and strict:
        listing = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Rover sweep checkpoints missing. Restore a final snapshot*.pt under each run:\n"
            f"{listing}"
        )
    if missing:
        print(f"Smoke warning: skipped {len(missing)} Rover runs without checkpoints.")
    return specs


def discover_baseline_specs(root: Path) -> list[PolicySpec]:
    specs: list[PolicySpec] = []
    missing: list[Path] = []
    for environment in ENVIRONMENTS:
        for directory, label in BASELINES:
            run_root = root / environment / "states" / directory
            if directory == "random":
                config_path = (
                    Path("configs/env/pointmaze/pointmaze_largedense_goal_1.yaml")
                    if environment == "largedense"
                    else Path("configs/env/pointmaze/pointmaze_umaze_goal_1.yaml")
                )
                snapshot = None
            else:
                config_path = run_root / CONFIG_NAME
                snapshot = find_final_snapshot(run_root)
                if not config_path.exists():
                    config_path = (
                        Path("configs/env/pointmaze/pointmaze_largedense_goal_1.yaml")
                        if environment == "largedense"
                        else Path("configs/env/pointmaze/pointmaze_umaze_goal_1.yaml")
                    )
                if snapshot is None:
                    missing.append(run_root)
                    continue
            specs.append(
                PolicySpec(
                    environment=environment,
                    algorithm=label,
                    run_name=directory,
                    config_path=config_path,
                    snapshot_path=snapshot,
                    checkpoint_step=0 if snapshot is None else _snapshot_sort_key(snapshot)[1],
                )
            )
    if missing:
        listing = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Baseline checkpoint missing:\n{listing}")
    return specs


def _sample_meta(agent, seed: int):
    meta = agent.init_meta() if callable(getattr(agent, "init_meta", None)) else {}
    if not isinstance(meta, dict):
        return meta, False
    rng = np.random.default_rng(seed)
    sampled = dict(meta)
    fixed = False
    if "z" in sampled:
        z = np.asarray(sampled["z"])
        if z.ndim == 1 and z.size:
            sampled["z"] = _sample_one_hot(z.size, rng, _agent_latent_probs(agent, z.size))
            fixed = True
    if "skill" in sampled:
        skill = np.asarray(sampled["skill"])
        if skill.ndim == 1 and skill.size:
            sampled["skill"] = rng.uniform(0.0, 1.0, skill.shape).astype(np.float32)
            fixed = True
    return sampled, fixed


def collect_trajectories(
    agent,
    env,
    *,
    count: int,
    horizon: int,
    policy_step: int,
    seed: int,
) -> list[np.ndarray]:
    """Collect only XY positions; unlike plotting helper, retain no rendered frames."""
    trajectories: list[np.ndarray] = []
    walls = _pointmaze_wall_rectangles(env)
    for episode in range(count):
        episode_seed = seed + episode
        time_step, _ = _reset_valid_pointmaze_start(env, episode_seed, walls)
        meta, fixed_meta = _sample_meta(agent, seed + episode * 1_000_003) if agent else ({}, False)
        points: list[np.ndarray] = []
        point = extract_eval_trajectory_point(env, time_step)
        if point is not None:
            points.append(point)

        rng = np.random.default_rng(seed + episode * 100_000)
        for _ in range(horizon):
            if agent is None:
                action = random_action(env.action_space, rng)
            else:
                with torch.no_grad(), utils.eval_mode(agent):
                    action = agent.act(time_step.observation, meta, policy_step, eval_mode=False)
            time_step = env.step(action)
            update_meta = getattr(agent, "update_meta", None) if agent is not None else None
            if callable(update_meta) and not fixed_meta:
                meta = update_meta(meta, policy_step, time_step)
            point = extract_eval_trajectory_point(env, time_step)
            if point is not None:
                points.append(point)
        if points:
            trajectories.append(np.asarray(points, dtype=np.float32))
    return trajectories


def evaluate_spec(spec: PolicySpec, args, index: int) -> Result:
    cfg = load_config(spec.config_path.resolve())
    env = make_env(cfg, seed=args.seed + index)
    agent = None
    payload = {}
    try:
        if spec.snapshot_path is not None:
            agent, payload = load_snapshot(spec.snapshot_path.resolve(), torch.device(args.device))
        horizon = (
            int(args.umaze_horizon)
            if spec.environment == "umaze"
            else int(args.largedense_horizon)
        )
        if args.smoke_test:
            horizon = min(horizon, args.smoke_horizon)
        policy_step = 0 if spec.snapshot_path is None else snapshot_step(spec.snapshot_path, payload)
        trajectories = collect_trajectories(
            agent,
            env,
            count=args.smoke_trajectories if args.smoke_test else TRAJECTORY_COUNTS[spec.environment],
            horizon=horizon,
            policy_step=policy_step,
            seed=args.seed + index * 10_000,
        )
        if not trajectories:
            raise RuntimeError(f"No trajectories collected for {spec.run_name}")
        covered, free, percentage = pointmaze_free_space_coverage(
            env, trajectories, args.grid_size, args.coverage_radius
        )
        if spec.algorithm == "Rover":
            overlay_dir = (
                args.output_dir
                / "trajectory_overlays"
                / spec.environment
                / f"nystrom_{spec.nystrom_points}"
                / spec.run_name
            )
            save_maze_trajectory_overlay_plot(
                trajectories, env, step=policy_step, save_dir=overlay_dir
            )
        return Result(
            environment=spec.environment,
            algorithm=spec.algorithm,
            run_name=spec.run_name,
            nystrom_points=spec.nystrom_points,
            bandwidth=spec.bandwidth,
            trajectories=len(trajectories),
            horizon=horizon,
            covered_points=covered,
            free_points=free,
            coverage_pct=percentage,
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
        del agent, payload
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def aggregate_results(results: list[Result]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int | None], list[Result]] = {}
    for result in results:
        key = (result.environment, result.algorithm, result.nystrom_points)
        groups.setdefault(key, []).append(result)
    rows = []
    for (environment, algorithm, nystrom), group in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2] or -1)
    ):
        coverage = np.asarray([item.coverage_pct for item in group])
        rows.append(
            {
                "environment": environment,
                "algorithm": algorithm,
                "nystrom_points": nystrom,
                "runs": len(group),
                "trajectories": group[0].trajectories,
                "horizon": group[0].horizon,
                "covered_points": float(np.mean([item.covered_points for item in group])),
                "free_points": group[0].free_points,
                "coverage_pct": float(coverage.mean()),
                "coverage_std": float(coverage.std(ddof=1)) if len(group) > 1 else 0.0,
            }
        )
    return rows


def set_paper_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette="colorblind",
        font="serif",
        rc={
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "grid.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )


def save_coverage_plots(rows: list[dict[str, Any]], output_dir: Path) -> None:
    set_paper_style()
    palette = sns.color_palette("colorblind", n_colors=len(BASELINES) + 1)
    for environment in ENVIRONMENTS:
        env_rows = [row for row in rows if row["environment"] == environment]
        rover = sorted(
            (row for row in env_rows if row["algorithm"] == "Rover"),
            key=lambda row: row["nystrom_points"],
        )
        fig, ax = plt.subplots(figsize=(3.35, 2.45))
        if rover:
            x = [row["nystrom_points"] for row in rover]
            y = [row["coverage_pct"] for row in rover]
            sns.lineplot(x=x, y=y, marker="o", linewidth=1.8, color=palette[0], ax=ax, label="Rover")
            std = np.asarray([row["coverage_std"] for row in rover])
            if np.any(std):
                ax.fill_between(x, np.asarray(y) - std, np.asarray(y) + std, color=palette[0], alpha=0.18)
            x_limits = (min(x), max(x)) if len(x) > 1 else (x[0] * 0.9, x[0] * 1.1)
        else:
            x_limits = (0.0, 1.0)

        for idx, (_, label) in enumerate(BASELINES, start=1):
            baseline = next((row for row in env_rows if row["algorithm"] == label), None)
            if baseline is None:
                continue
            sns.lineplot(
                x=list(x_limits),
                y=[baseline["coverage_pct"]] * 2,
                linestyle="--",
                linewidth=1.0,
                color=palette[idx],
                ax=ax,
                label=label,
            )
        ax.set_xlabel("Number of Nyström points")
        ax.set_ylabel("Coverage (%)")
        ax.set_title(ENV_LABELS[environment])
        ax.set_ylim(0.0, 100.0)
        sns.despine(ax=ax)
        ax.legend(frameon=False, ncol=2, loc="best")
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(output_dir / f"{environment}_coverage.{suffix}", bbox_inches="tight")
        plt.close(fig)


def save_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# PointMaze Nyström coverage",
        "",
        "| Environment | Algorithm | Nyström points | Runs | Trajectories | Horizon | Covered / free | Coverage (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        nystrom = "—" if row["nystrom_points"] is None else str(row["nystrom_points"])
        covered = f"{row['covered_points']:.1f}" if row["runs"] > 1 else str(int(row["covered_points"]))
        coverage = f"{row['coverage_pct']:.2f}"
        if row["runs"] > 1:
            coverage += f" ± {row['coverage_std']:.2f}"
        lines.append(
            f"| {ENV_LABELS[row['environment']]} | {row['algorithm']} | {nystrom} | "
            f"{row['runs']} | {row['trajectories']} | {row['horizon']} | "
            f"{covered} / {row['free_points']} | {coverage} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rover-root", type=Path, default=Path("tests/diagnostics/pointmaze/models"))
    parser.add_argument("--baseline-root", type=Path, default=Path("models/pointmaze"))
    parser.add_argument("--output-dir", type=Path, default=Path("tests/outputs/pointmaze/nystrom_coverage"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=90)
    parser.add_argument("--coverage-radius", type=float, default=0.08)
    parser.add_argument(
        "--umaze-horizon",
        type=int,
        default=1000,
        help="Evaluation horizon for UMaze (default: 1000).",
    )
    parser.add_argument(
        "--largedense-horizon",
        type=int,
        default=3000,
        help="Evaluation horizon for LargeDense (default: 3000).",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-trajectories", type=int, default=2)
    parser.add_argument("--smoke-horizon", type=int, default=20)
    args = parser.parse_args()
    if args.grid_size < 2:
        parser.error("--grid-size must be at least 2")
    if args.coverage_radius <= 0:
        parser.error("--coverage-radius must be positive")
    if args.umaze_horizon < 1 or args.largedense_horizon < 1:
        parser.error("environment horizons must be positive")
    if args.smoke_trajectories < 1 or args.smoke_horizon < 1:
        parser.error("smoke trajectory count and horizon must be positive")
    if args.smoke_test and args.output_dir == Path("tests/outputs/pointmaze/nystrom_coverage"):
        args.output_dir = Path("tests/outputs/pointmaze/nystrom_coverage_smoke")
    return args


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rover = discover_rover_specs(args.rover_root, strict=not args.smoke_test)
    baselines = discover_baseline_specs(args.baseline_root)
    if args.smoke_test:
        rover = [next((spec for spec in rover if spec.environment == env), None) for env in ENVIRONMENTS]
        rover = [spec for spec in rover if spec is not None]
        selected_baselines = []
        for env in ENVIRONMENTS:
            selected_baselines.append(
                next(spec for spec in baselines if spec.environment == env and spec.algorithm == "Random")
            )
        baselines = selected_baselines

    specs = rover + baselines
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, spec in enumerate(specs):
        print(
            f"[{index + 1}/{len(specs)}] {ENV_LABELS[spec.environment]} "
            f"{spec.algorithm} {spec.run_name}"
        )
        results.append(evaluate_spec(spec, args, index))

    rows = aggregate_results(results)
    save_coverage_plots(rows, args.output_dir)
    save_markdown(rows, args.output_dir / "coverage_summary.md")
    print(f"Saved results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
