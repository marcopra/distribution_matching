"""Batch-evaluate every PointMaze snapshot under ``models/pointmaze_new``.

Default run:
    python evaluate_pointmaze_models.py

The script is intentionally fault tolerant: one missing or broken policy produces a
table row and an error-summary entry without stopping the remaining evaluations.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

import utils
from agent.rover_visualization.domains import (
    pointmaze_free_space_coverage,
    save_maze_trajectory_overlay_plot,
)
from plot_pointmaze_snapshot_trajectories import (
    default_episode_steps,
    load_config,
    load_snapshot,
    make_env,
    sample_trajectories,
    snapshot_step,
)
from tests.diagnostics.pointmaze.evaluate_nystrom_coverage import collect_trajectories


ENVIRONMENTS = ("umaze", "largedense")
OBSERVATION_TYPES = ("states", "pixels")
ALGORITHMS = ("cic", "icm_apt", "maxent", "random", "rnd", "rover", "smm")
NUMBERED_SNAPSHOT_RE = re.compile(r"snapshot_(\d+)\.pt$")


@dataclass(frozen=True)
class PolicySpec:
    environment: str
    observation_type: str
    algorithm: str
    algorithm_dir: Path
    config_path: Path | None
    snapshot_path: Path | None
    snapshot_id: str
    numbered_step: int | None = None


@dataclass
class EvaluationResult:
    spec: PolicySpec
    status: str
    step: int | None = None
    horizon: int | None = None
    coverage_values: list[float] = field(default_factory=list)
    coverage_mean: float | None = None
    coverage_se: float | None = None
    expansion_delta: float | None = None
    expanding: bool | None = None
    kernel_type: str | None = None
    kernel_bandwidth: float | str | None = None
    plot_path: Path | None = None
    error: str | None = None


def snapshot_identifier(models_dir: Path, snapshot: Path) -> str:
    """Return stable filesystem-safe identifier based on relative snapshot path."""
    relative = snapshot.relative_to(models_dir).with_suffix("")
    return "__".join(re.sub(r"[^A-Za-z0-9_.-]+", "_", part) for part in relative.parts)


def snapshot_discovery_key(path: Path) -> tuple[int, int, str]:
    match = NUMBERED_SNAPSHOT_RE.fullmatch(path.name)
    if match:
        return (0, int(match.group(1)), str(path))
    if path.name == "best_snapshot.pt":
        return (1, 0, str(path))
    if path.name == "final_snapshot.pt":
        return (2, 0, str(path))
    return (3, 0, str(path))


def _find_matrix_config(root: Path, environment: str, observation_type: str) -> Path | None:
    matrix_root = root / environment / observation_type
    for candidate in sorted(matrix_root.glob("*/.hydra/config.yaml")):
        if candidate.is_file():
            return candidate
    return None


def discover_policy_specs(root: Path) -> list[PolicySpec]:
    """Discover full fixed matrix, including explicit missing rows."""
    specs: list[PolicySpec] = []
    for environment in ENVIRONMENTS:
        for observation_type in OBSERVATION_TYPES:
            fallback_config = _find_matrix_config(root, environment, observation_type)
            for algorithm in ALGORITHMS:
                algorithm_dir = root / environment / observation_type / algorithm
                config = algorithm_dir / ".hydra" / "config.yaml"
                if not config.is_file():
                    config = fallback_config

                if algorithm == "random":
                    exists = algorithm_dir.is_dir()
                    specs.append(
                        PolicySpec(
                            environment=environment,
                            observation_type=observation_type,
                            algorithm=algorithm,
                            algorithm_dir=algorithm_dir,
                            config_path=config if exists else None,
                            snapshot_path=None,
                            snapshot_id="random" if exists else "missing",
                        )
                    )
                    continue

                models_dir = algorithm_dir / "models"
                snapshots = sorted(
                    (path for path in models_dir.rglob("*.pt") if path.is_file()),
                    key=snapshot_discovery_key,
                ) if models_dir.is_dir() else []
                if not snapshots:
                    specs.append(
                        PolicySpec(
                            environment=environment,
                            observation_type=observation_type,
                            algorithm=algorithm,
                            algorithm_dir=algorithm_dir,
                            config_path=config if algorithm_dir.is_dir() else None,
                            snapshot_path=None,
                            snapshot_id="missing",
                        )
                    )
                    continue

                for snapshot in snapshots:
                    match = NUMBERED_SNAPSHOT_RE.fullmatch(snapshot.name)
                    specs.append(
                        PolicySpec(
                            environment=environment,
                            observation_type=observation_type,
                            algorithm=algorithm,
                            algorithm_dir=algorithm_dir,
                            config_path=config,
                            snapshot_path=snapshot,
                            snapshot_id=snapshot_identifier(models_dir, snapshot),
                            numbered_step=int(match.group(1)) if match else None,
                        )
                    )
    return specs


def coverage_statistics(values: Iterable[float]) -> tuple[float, float]:
    samples = np.asarray(list(values), dtype=np.float64)
    if samples.size == 0:
        raise ValueError("coverage statistics require at least one value")
    mean = float(samples.mean())
    standard_error = (
        float(samples.std(ddof=1) / math.sqrt(samples.size)) if samples.size > 1 else 0.0
    )
    return mean, standard_error


def rover_kernel_metadata(agent: Any, policy_step: int) -> tuple[str | None, float | str | None]:
    """Read Rover kernel metadata from loaded policy, never from Hydra config."""
    kernel_type = getattr(agent, "kernel_type", None)
    matcher = getattr(agent, "distribution_matcher", None)
    if kernel_type is None and matcher is not None:
        kernel_type = getattr(matcher, "kernel_type", None)
    if kernel_type is None:
        return None, None

    kernel_type = str(kernel_type)
    if kernel_type.lower() != "gaussian":
        return kernel_type, None

    active = getattr(agent, "_active_kernel_bandwidth", None)
    if active is None and matcher is not None:
        active = getattr(matcher, "kernel_bandwidth", None)
    if active is None:
        configured = getattr(agent, "kernel_bandwidth", None)
        if configured is not None:
            try:
                active = float(utils.schedule(configured, policy_step))
            except (TypeError, ValueError, NotImplementedError):
                active = configured
    if active is None:
        multiplier = getattr(agent, "kernel_bandwidth_mult", None)
        return kernel_type, None if multiplier is None else f"adaptive x{float(multiplier):g}"
    try:
        active = float(active)
    except (TypeError, ValueError):
        active = str(active)
    return kernel_type, active


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _attach_env(agent: Any, env: Any) -> None:
    insert_env = getattr(agent, "insert_env", None) if agent is not None else None
    if callable(insert_env):
        insert_env(env)


def _clear_runtime_history(agent: Any) -> None:
    history = getattr(agent, "current_action_probs", None) if agent is not None else None
    if isinstance(history, list):
        history.clear()


def _close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def evaluate_policy(spec: PolicySpec, args: argparse.Namespace) -> EvaluationResult:
    if spec.snapshot_id == "missing":
        return EvaluationResult(spec=spec, status="missing")
    if spec.config_path is None or not spec.config_path.is_file():
        raise FileNotFoundError(f"No Hydra config available for {spec.algorithm_dir}")

    cfg = load_config(spec.config_path)
    device = torch.device(args.device)
    agent = None
    payload: dict[str, Any] = {}
    if spec.snapshot_path is not None:
        agent, payload = load_snapshot(spec.snapshot_path, device)
    step = 0 if agent is None else snapshot_step(spec.snapshot_path, payload)
    kernel_type = None
    kernel_bandwidth = None
    if spec.algorithm == "rover" and agent is not None:
        kernel_type, kernel_bandwidth = rover_kernel_metadata(agent, step)

    plot_path = None
    plot_env = None
    try:
        _seed_everything(args.plot_seed)
        plot_env = make_env(cfg, seed=args.plot_seed)
        _attach_env(agent, plot_env)
        _clear_runtime_history(agent)
        horizon = int(args.horizon_override or default_episode_steps(plot_env))
        trajectories, _ = sample_trajectories(
            agent,
            plot_env,
            num_trajectories=args.plot_trajectories,
            episode_steps=horizon,
            policy_step=step,
            seed=args.plot_seed,
            deterministic=False,
        )
        if len(trajectories) != args.plot_trajectories:
            raise RuntimeError(
                f"Collected {len(trajectories)}/{args.plot_trajectories} plot trajectories"
            )
        plot_dir = spec.algorithm_dir / "trajectory_plots" / spec.snapshot_id
        paths = save_maze_trajectory_overlay_plot(trajectories, plot_env, step, plot_dir)
        if "maze_overlay" not in paths:
            raise RuntimeError("Maze-overlay plot was not created")
        plot_path = Path(paths["maze_overlay"])
    finally:
        _clear_runtime_history(agent)
        if plot_env is not None:
            _close_env(plot_env)

    coverage_values: list[float] = []
    for evaluation_index in range(args.coverage_runs):
        seed = args.coverage_seed + evaluation_index
        coverage_env = None
        try:
            _seed_everything(seed)
            coverage_env = make_env(cfg, seed=seed)
            _attach_env(agent, coverage_env)
            _clear_runtime_history(agent)
            trajectories = collect_trajectories(
                agent,
                coverage_env,
                count=args.coverage_trajectories,
                horizon=horizon,
                policy_step=step,
                seed=seed,
            )
            if len(trajectories) != args.coverage_trajectories:
                raise RuntimeError(
                    f"Collected {len(trajectories)}/{args.coverage_trajectories} coverage trajectories"
                )
            _, _, percentage = pointmaze_free_space_coverage(
                coverage_env,
                trajectories,
                grid_size=args.coverage_grid_size,
                radius=args.coverage_radius,
            )
            coverage_values.append(float(percentage))
        finally:
            _clear_runtime_history(agent)
            if coverage_env is not None:
                _close_env(coverage_env)

    mean, standard_error = coverage_statistics(coverage_values)
    return EvaluationResult(
        spec=spec,
        status="ok",
        step=step,
        horizon=horizon,
        coverage_values=coverage_values,
        coverage_mean=mean,
        coverage_se=standard_error,
        kernel_type=kernel_type,
        kernel_bandwidth=kernel_bandwidth,
        plot_path=plot_path,
    )


def run_evaluations(
    specs: list[PolicySpec],
    evaluator: Callable[[PolicySpec], EvaluationResult],
) -> tuple[list[EvaluationResult], dict[str, list[tuple[str, str]]]]:
    results: list[EvaluationResult] = []
    errors: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for index, spec in enumerate(specs, start=1):
        label = f"{spec.environment}/{spec.observation_type}/{spec.algorithm}/{spec.snapshot_id}"
        print(f"[{index}/{len(specs)}] {label}")
        try:
            result = evaluator(spec)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            results.append(EvaluationResult(spec=spec, status="error", error=message))
            errors[spec.algorithm].append((label, message))
        else:
            results.append(result)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return results, dict(errors)


def apply_expansion_status(results: list[EvaluationResult], tolerance: float) -> None:
    groups: dict[tuple[str, str, str], list[EvaluationResult]] = defaultdict(list)
    for result in results:
        if (
            result.status == "ok"
            and result.spec.numbered_step is not None
            and result.coverage_mean is not None
        ):
            key = (
                result.spec.environment,
                result.spec.observation_type,
                result.spec.algorithm,
            )
            groups[key].append(result)
    for group in groups.values():
        previous: EvaluationResult | None = None
        for result in sorted(group, key=lambda item: item.spec.numbered_step or -1):
            if previous is not None:
                result.expansion_delta = result.coverage_mean - previous.coverage_mean
                result.expanding = result.expansion_delta > tolerance
            previous = result


def _display(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value).replace("|", "\\|")


def render_markdown(
    results: list[EvaluationResult], args: argparse.Namespace, report_directory: Path
) -> str:
    lines = [
        "# PointMaze policy coverage",
        "",
        "## Evaluation settings",
        "",
        f"- Plot trajectories: {args.plot_trajectories}",
        f"- Coverage evaluations per policy: {args.coverage_runs}",
        f"- Coverage trajectories per evaluation: {args.coverage_trajectories}",
        f"- Coverage grid: {args.coverage_grid_size} × {args.coverage_grid_size}",
        f"- Coverage radius: {args.coverage_radius:g}",
        f"- Expansion tolerance: {args.coverage_expansion_tolerance:g} percentage points",
        f"- Coverage seeds: {', '.join(str(args.coverage_seed + i) for i in range(args.coverage_runs))}",
        "- Horizon: environment default" if args.horizon_override is None else f"- Horizon override: {args.horizon_override}",
        "- Coverage summary: mean ± standard error (sample standard deviation / √n)",
        "",
        "## Results",
        "",
        "| Environment | Observation | Algorithm | Snapshot | Step | Horizon | Runs | Coverage values (%) | Mean (%) | SE (%) | Δ (%) | Expanding | Kernel | Bandwidth | Plot | Status |",
        "|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---|---|",
    ]
    for result in results:
        if result.status == "ok":
            values = ", ".join(f"{value:.3f}" for value in result.coverage_values)
            plot = "—"
            if result.plot_path is not None:
                try:
                    relative = Path(
                        os.path.relpath(result.plot_path.resolve(), report_directory.resolve())
                    )
                    plot = f"[{result.plot_path.name}]({relative.as_posix()})"
                except (OSError, ValueError):
                    plot = str(result.plot_path.resolve())
            fields = (
                _display(result.step), _display(result.horizon), str(len(result.coverage_values)),
                values, _display(result.coverage_mean), _display(result.coverage_se),
                _display(result.expansion_delta), _display(result.expanding),
                _display(result.kernel_type), _display(result.kernel_bandwidth), plot,
            )
        else:
            marker = result.status
            fields = tuple(marker for _ in range(11))
        lines.append(
            f"| {result.spec.environment} | {result.spec.observation_type} | "
            f"{result.spec.algorithm} | {result.spec.snapshot_id} | "
            + " | ".join(fields)
            + f" | {result.status} |"
        )
    return "\n".join(lines) + "\n"


def print_error_summary(errors: dict[str, list[tuple[str, str]]]) -> None:
    print("\n=== Error summary ===")
    if not errors:
        print("none")
        return
    for algorithm in sorted(errors):
        print(f"{algorithm}:")
        for label, message in errors[algorithm]:
            print(f"  - {label}: {message}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("models/pointmaze_new"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--plot-trajectories", type=int, default=15)
    parser.add_argument("--plot-seed", type=int, default=0)
    parser.add_argument("--coverage-runs", type=int, default=5)
    parser.add_argument("--coverage-seed", type=int, default=0)
    parser.add_argument("--coverage-trajectories", type=int, default=50)
    parser.add_argument("--coverage-grid-size", type=int, default=90)
    parser.add_argument("--coverage-radius", type=float, default=0.08)
    parser.add_argument("--coverage-expansion-tolerance", type=float, default=0.25)
    parser.add_argument(
        "--horizon-override",
        type=int,
        default=None,
        help="Testing override; production default comes from each environment.",
    )
    parser.add_argument(
        "--max-policies",
        type=int,
        default=None,
        help="Evaluate only first N discovered entries; useful for smoke tests.",
    )
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = args.root / "coverage_results.md"
    for name in ("plot_trajectories", "coverage_runs", "coverage_trajectories", "coverage_grid_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.coverage_grid_size < 2:
        parser.error("--coverage-grid-size must be at least 2")
    if args.coverage_radius <= 0:
        parser.error("--coverage-radius must be positive")
    if args.coverage_expansion_tolerance < 0:
        parser.error("--coverage-expansion-tolerance must be non-negative")
    if args.horizon_override is not None and args.horizon_override < 1:
        parser.error("--horizon-override must be positive")
    if args.max_policies is not None and args.max_policies < 1:
        parser.error("--max-policies must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"PointMaze model root not found: {root}")
    specs = discover_policy_specs(root)
    if args.max_policies is not None:
        specs = specs[: args.max_policies]
    results, errors = run_evaluations(specs, lambda spec: evaluate_policy(spec, args))
    apply_expansion_status(results, args.coverage_expansion_tolerance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_markdown(results, args, args.output.parent), encoding="utf-8"
    )
    print(f"\nSaved table: {args.output.resolve()}")
    print_error_summary(errors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
