"""Evaluate discrete-maze state coverage across scaling checkpoints.

For every algorithm and maze size, sample 50 trajectories for each evaluation
seed, compute unique-state coverage, and report mean ± standard error.

Maze-108 baseline checkpoints missing from the scaling tree automatically fall
back to models/maze/108/pixels/<algorithm>.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

import utils
from plot_pointmaze_snapshot_trajectories import (
    find_run_config,
    load_config,
    load_snapshot,
    make_env,
    snapshot_step,
)


MAZE_SIZES = (108, 200, 500)
ENV_CONFIGS = {
    108: Path("configs/env/gridworld/maze_108_seed7_env.yaml"),
    200: Path("configs/env/gridworld/maze_200_seed7_env.yaml"),
    500: Path("configs/env/gridworld/maze_500_seed7_env.yaml"),
}
DISPLAY_NAMES = {
    "random": "Random",
    "rover": "Rover",
    "cic": "CIC",
    "rnd": "RND",
    "smm": "SMM",
    "icm_apt": "ICM-APT",
    "maxent": "MaxEnt",
}
ALGORITHM_ORDER = ("rover", "cic", "rnd", "smm", "icm_apt", "maxent", "random")
SNAPSHOT_RE = re.compile(r"snapshot_(\d+)\.pt$")
MAZE_RE = re.compile(r"maze_(\d+)(?:_|$)")
SEED_RE = re.compile(r"seed_(\d+)")


@dataclass(frozen=True)
class PolicySpec:
    algorithm: str
    maze_size: int
    model_seed: int
    snapshot_path: Path
    config_path: Path
    source: str


@dataclass(frozen=True)
class SeedResult:
    algorithm: str
    maze_size: int
    evaluation_seed: int
    model_seed: int | None
    trajectories: int
    horizon: int
    visited_states: int
    total_states: int
    coverage_pct: float
    source: str


def snapshot_sort_key(path: Path) -> tuple[int, int, str]:
    match = SNAPSHOT_RE.fullmatch(path.name)
    if match:
        return (0, int(match.group(1)), str(path))
    if path.name == "snapshot.pt":
        return (1, 0, str(path))
    return (-1, -1, str(path))


def model_seed_from_path(path: Path) -> int:
    for part in path.parts:
        match = SEED_RE.fullmatch(part)
        if match:
            return int(match.group(1))
    return 0


def find_final_snapshot(root: Path) -> Path | None:
    snapshots = [
        path
        for path in root.rglob("snapshot*.pt")
        if path.is_file() and snapshot_sort_key(path)[0] >= 0
    ]
    return max(snapshots, key=snapshot_sort_key) if snapshots else None


def algorithm_sort_key(algorithm: str) -> tuple[int, str]:
    try:
        return (ALGORITHM_ORDER.index(algorithm), algorithm)
    except ValueError:
        return (len(ALGORITHM_ORDER), algorithm)


def discover_scaling_specs(root: Path) -> list[PolicySpec]:
    specs: list[PolicySpec] = []
    if not root.exists():
        raise FileNotFoundError(f"Scaling model root not found: {root}")

    for algorithm_root in sorted(path for path in root.iterdir() if path.is_dir()):
        algorithm = algorithm_root.name
        for maze_root in sorted(path for path in algorithm_root.iterdir() if path.is_dir()):
            match = MAZE_RE.search(maze_root.name)
            if not match:
                continue
            maze_size = int(match.group(1))
            if maze_size not in MAZE_SIZES:
                continue
            seed_roots = sorted(
                (path for path in maze_root.glob("seed_*") if path.is_dir()),
                key=model_seed_from_path,
            )
            for seed_root in seed_roots:
                snapshot = find_final_snapshot(seed_root)
                if snapshot is None:
                    print(f"Warning: no snapshot under {seed_root}", file=sys.stderr)
                    continue
                config = find_run_config(snapshot)
                if config is None:
                    print(f"Warning: no Hydra config for {snapshot}", file=sys.stderr)
                    continue
                specs.append(
                    PolicySpec(
                        algorithm=algorithm,
                        maze_size=maze_size,
                        model_seed=model_seed_from_path(seed_root),
                        snapshot_path=snapshot,
                        config_path=config,
                        source="scaling",
                    )
                )
    return specs


def discover_legacy_108_specs(
    root: Path, algorithms: set[str], existing: set[tuple[str, int]]
) -> list[PolicySpec]:
    specs: list[PolicySpec] = []
    for algorithm in sorted(algorithms, key=algorithm_sort_key):
        if algorithm == "rover" or (algorithm, 108) in existing:
            continue
        algorithm_root = root / algorithm
        snapshot = find_final_snapshot(algorithm_root) if algorithm_root.exists() else None
        config = algorithm_root / ".hydra" / "config.yaml"
        if snapshot is None or not config.exists():
            print(
                f"Warning: missing legacy pixel checkpoint/config for {algorithm} at {algorithm_root}",
                file=sys.stderr,
            )
            continue
        specs.append(
            PolicySpec(
                algorithm=algorithm,
                maze_size=108,
                model_seed=int(OmegaConf.load(config).get("seed", 0)),
                snapshot_path=snapshot,
                config_path=config,
                source="legacy-108-pixels",
            )
        )
    return specs


def discrete_env(env):
    current = env
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if all(hasattr(current, name) for name in ("n_states", "state_to_idx", "idx_to_state")):
            return current
        current = getattr(current, "env", None)
    raise AttributeError("Could not find discrete maze environment under wrappers")


def current_state_index(env, time_step) -> int:
    info = getattr(time_step, "info", None)
    if isinstance(info, Mapping) and "state_index" in info:
        return int(info["state_index"])
    base = discrete_env(env)
    location = getattr(base, "_agent_location", None)
    if location in base.state_to_idx:
        return int(base.state_to_idx[location])
    observation = np.asarray(time_step.observation)
    if observation.ndim == 1 and observation.size == base.n_states:
        return int(np.argmax(observation))
    raise ValueError("Could not infer current discrete state index")


def latent_probabilities(agent, dimension: int) -> np.ndarray:
    for name in ("z_probs", "z_prob", "p_z", "pz", "skill_probs", "p_skill"):
        value = getattr(agent, name, None)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        probabilities = np.asarray(value, dtype=np.float64).reshape(-1)
        if probabilities.size != dimension:
            continue
        probabilities = np.clip(probabilities, 0.0, None)
        total = probabilities.sum()
        if np.isfinite(total) and total > 0:
            return probabilities / total
    return np.full(dimension, 1.0 / dimension, dtype=np.float64)


def sample_episode_meta(agent, rng: np.random.Generator):
    if agent is None or not callable(getattr(agent, "init_meta", None)):
        return {}, False
    meta = agent.init_meta()
    if not isinstance(meta, Mapping):
        return meta, False
    sampled = dict(meta)
    fixed = False
    if "z" in sampled:
        z = np.asarray(sampled["z"])
        if z.ndim == 1 and z.size:
            sampled_z = np.zeros(z.size, dtype=np.float32)
            sampled_z[int(rng.choice(z.size, p=latent_probabilities(agent, z.size)))] = 1.0
            sampled["z"] = sampled_z
            fixed = True
    if "skill" in sampled:
        skill = np.asarray(sampled["skill"])
        if skill.ndim == 1 and skill.size:
            sampled["skill"] = rng.uniform(0.0, 1.0, skill.shape).astype(np.float32)
            fixed = True
    return sampled, fixed


def random_action(env, rng: np.random.Generator):
    action_space = env.action_space
    seed_fn = getattr(action_space, "seed", None)
    if callable(seed_fn):
        seed_fn(int(rng.integers(0, 2**31 - 1)))
    return action_space.sample()


def environment_horizon(env) -> int:
    base = discrete_env(env)
    return int(getattr(base, "max_steps", 300))


def collect_coverage(
    agent,
    env,
    *,
    trajectories: int,
    evaluation_seed: int,
    policy_step: int,
    horizon_override: int | None,
) -> tuple[int, int, int]:
    base = discrete_env(env)
    horizon = horizon_override or environment_horizon(env)
    visited_states: set[int] = set()

    for episode in range(trajectories):
        episode_seed = evaluation_seed * 1_000_003 + episode
        time_step = env.reset(seed=episode_seed)
        visited_states.add(current_state_index(env, time_step))
        meta, fixed_meta = sample_episode_meta(
            agent, np.random.default_rng(episode_seed + 17)
        )
        action_rng = np.random.default_rng(episode_seed + 31)

        for _ in range(horizon):
            if agent is None:
                action = random_action(env, action_rng)
            else:
                with torch.no_grad(), utils.eval_mode(agent):
                    action = agent.act(
                        time_step.observation,
                        meta,
                        policy_step,
                        eval_mode=False,
                    )
            time_step = env.step(action)
            visited_states.add(current_state_index(env, time_step))
            update_meta = getattr(agent, "update_meta", None) if agent is not None else None
            if callable(update_meta) and not fixed_meta:
                meta = update_meta(meta, policy_step, time_step)
            if time_step.last():
                break

    return len(visited_states), int(base.n_states), horizon


def evaluate_policy(
    spec: PolicySpec | None,
    *,
    algorithm: str,
    maze_size: int,
    evaluation_seed: int,
    trajectories: int,
    horizon: int | None,
    device: torch.device,
) -> SeedResult:
    if spec is None:
        cfg = load_config((REPO_ROOT / ENV_CONFIGS[maze_size]).resolve())
        source = "random"
        model_seed = None
        agent = None
        payload: Any = {}
    else:
        cfg = load_config(spec.config_path.resolve())
        source = spec.source
        model_seed = spec.model_seed
        agent, payload = load_snapshot(spec.snapshot_path.resolve(), device)

    env = make_env(cfg, seed=evaluation_seed)
    try:
        policy_step = (
            0
            if spec is None
            else snapshot_step(spec.snapshot_path, payload)
        )
        visited, total, used_horizon = collect_coverage(
            agent,
            env,
            trajectories=trajectories,
            evaluation_seed=evaluation_seed,
            policy_step=policy_step,
            horizon_override=horizon,
        )
        return SeedResult(
            algorithm=algorithm,
            maze_size=maze_size,
            evaluation_seed=evaluation_seed,
            model_seed=model_seed,
            trajectories=trajectories,
            horizon=used_horizon,
            visited_states=visited,
            total_states=total,
            coverage_pct=100.0 * visited / total,
            source=source,
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
        del agent, payload
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def aggregate(results: list[SeedResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[SeedResult]] = defaultdict(list)
    for result in results:
        grouped[(result.algorithm, result.maze_size)].append(result)

    rows: list[dict[str, Any]] = []
    for (algorithm, maze_size), group in sorted(
        grouped.items(), key=lambda item: (algorithm_sort_key(item[0][0]), item[0][1])
    ):
        values = np.asarray([result.coverage_pct for result in group], dtype=np.float64)
        standard_error = (
            float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0
        )
        rows.append(
            {
                "algorithm": algorithm,
                "maze_size": maze_size,
                "mean": float(values.mean()),
                "sem": standard_error,
                "n": int(values.size),
                "trajectories": group[0].trajectories,
                "horizon": group[0].horizon,
                "source": ", ".join(sorted({result.source for result in group})),
                "model_seeds": ", ".join(
                    str(seed)
                    for seed in sorted(
                        {result.model_seed for result in group if result.model_seed is not None}
                    )
                )
                or "—",
            }
        )
    return rows


def write_raw_csv(results: list[SeedResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(SeedResult.__dataclass_fields__))
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def write_markdown(rows: list[dict[str, Any]], args, path: Path) -> None:
    lookup = {(row["algorithm"], row["maze_size"]): row for row in rows}
    algorithms = sorted({row["algorithm"] for row in rows}, key=algorithm_sort_key)
    lines = [
        "# Maze scaling coverage",
        "",
        (
            f"Coverage is unique visited states / |S| after {args.num_trajectories} "
            f"trajectories per evaluation seed. Values are mean ± standard error over "
            f"{len(args.eval_seeds)} evaluation seeds ({', '.join(map(str, args.eval_seeds))})."
        ),
        "",
        "| Algorithm | |S| = 108 | |S| = 200 | |S| = 500 |",
        "|---|---:|---:|---:|",
    ]
    for algorithm in algorithms:
        values = []
        for maze_size in MAZE_SIZES:
            row = lookup.get((algorithm, maze_size))
            values.append("—" if row is None else f"{row['mean']:.2f} ± {row['sem']:.2f}%")
        lines.append(f"| {DISPLAY_NAMES.get(algorithm, algorithm)} | {' | '.join(values)} |")

    lines.extend(
        [
            "",
            "## Evaluation details",
            "",
            "| Algorithm | |S| | Model seed(s) | Source | Horizon | Seeds (n) |",
            "|---|---:|---:|---|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {DISPLAY_NAMES.get(row['algorithm'], row['algorithm'])} | "
            f"{row['maze_size']} | {row['model_seeds']} | {row['source']} | "
            f"{row['horizon']} | {row['n']} |"
        )
    lines.extend(
        [
            "",
            "Maze-108 non-Rover baselines use legacy pixel-observation checkpoints when "
            "no scaling checkpoint exists. Random uses the same maze, horizon, trajectory "
            "count, and evaluation seeds as trained policies.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plot(rows: list[dict[str, Any]], output_dir: Path) -> None:
    algorithms = sorted({row["algorithm"] for row in rows}, key=algorithm_sort_key)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    colors = plt.get_cmap("tab10").colors
    markers = ("o", "s", "^", "D", "P", "X", "*")

    for index, algorithm in enumerate(algorithms):
        algorithm_rows = sorted(
            (row for row in rows if row["algorithm"] == algorithm),
            key=lambda row: row["maze_size"],
        )
        x = np.asarray([row["maze_size"] for row in algorithm_rows])
        y = np.asarray([row["mean"] for row in algorithm_rows])
        sem = np.asarray([row["sem"] for row in algorithm_rows])
        ax.errorbar(
            x,
            y,
            yerr=sem,
            marker=markers[index % len(markers)],
            color=colors[index % len(colors)],
            linewidth=1.8,
            markersize=5,
            capsize=3,
            label=DISPLAY_NAMES.get(algorithm, algorithm),
        )

    ax.set_xlabel(r"Number of states $|S|$")
    ax.set_ylabel("State coverage (%)")
    ax.set_xticks(MAZE_SIZES)
    ax.set_xlim(90, 520)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"maze_scaling_coverage.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-root", type=Path, default=Path("models/maze/maze_scaling")
    )
    parser.add_argument(
        "--legacy-108-root", type=Path, default=Path("models/maze/108/pixels")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/maze_scaling_coverage")
    )
    parser.add_argument("--num-trajectories", type=int, default=50)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=None,
        help="Override each maze config's max_steps.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        help="Optional subset, e.g. rover cic rnd smm.",
    )
    parser.add_argument(
        "--maze-sizes",
        type=int,
        nargs="+",
        default=list(MAZE_SIZES),
    )
    args = parser.parse_args()
    if args.num_trajectories < 1:
        parser.error("--num-trajectories must be positive")
    if not args.eval_seeds:
        parser.error("--eval-seeds cannot be empty")
    if args.episode_steps is not None and args.episode_steps < 1:
        parser.error("--episode-steps must be positive")
    invalid_sizes = sorted(set(args.maze_sizes) - set(MAZE_SIZES))
    if invalid_sizes:
        parser.error(f"unsupported maze sizes: {invalid_sizes}")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    scaling_specs = discover_scaling_specs(args.models_root)
    discovered_algorithms = {spec.algorithm for spec in scaling_specs}
    if args.algorithms is not None:
        discovered_algorithms &= set(args.algorithms) - {"random"}
    random_requested = args.algorithms is None or "random" in args.algorithms
    if not discovered_algorithms and not random_requested:
        raise RuntimeError("No trained algorithms discovered for requested filters")

    existing = {(spec.algorithm, spec.maze_size) for spec in scaling_specs}
    legacy_specs = discover_legacy_108_specs(
        args.legacy_108_root, discovered_algorithms, existing
    )
    all_specs = scaling_specs + legacy_specs
    grouped_specs: dict[tuple[str, int], list[PolicySpec]] = defaultdict(list)
    for spec in all_specs:
        if spec.algorithm not in discovered_algorithms or spec.maze_size not in args.maze_sizes:
            continue
        grouped_specs[(spec.algorithm, spec.maze_size)].append(spec)
    for specs in grouped_specs.values():
        specs.sort(key=lambda spec: (spec.model_seed, str(spec.snapshot_path)))

    tasks: list[tuple[str, int, int, PolicySpec | None]] = []
    for algorithm in sorted(discovered_algorithms, key=algorithm_sort_key):
        for maze_size in sorted(args.maze_sizes):
            specs = grouped_specs.get((algorithm, maze_size), [])
            if not specs:
                print(f"Warning: skipping missing {algorithm} |S|={maze_size}", file=sys.stderr)
                continue
            for seed_index, evaluation_seed in enumerate(args.eval_seeds):
                tasks.append((algorithm, maze_size, evaluation_seed, specs[seed_index % len(specs)]))
    if random_requested:
        for maze_size in sorted(args.maze_sizes):
            for evaluation_seed in args.eval_seeds:
                tasks.append(("random", maze_size, evaluation_seed, None))

    results: list[SeedResult] = []
    for index, (algorithm, maze_size, evaluation_seed, spec) in enumerate(tasks, start=1):
        model_text = "random" if spec is None else f"model seed {spec.model_seed}"
        print(
            f"[{index}/{len(tasks)}] {DISPLAY_NAMES.get(algorithm, algorithm)} "
            f"|S|={maze_size}, eval seed={evaluation_seed}, {model_text}",
            flush=True,
        )
        np.random.seed(evaluation_seed)
        torch.manual_seed(evaluation_seed)
        result = evaluate_policy(
            spec,
            algorithm=algorithm,
            maze_size=maze_size,
            evaluation_seed=evaluation_seed,
            trajectories=args.num_trajectories,
            horizon=args.episode_steps,
            device=device,
        )
        results.append(result)
        print(
            f"  coverage={result.coverage_pct:.2f}% "
            f"({result.visited_states}/{result.total_states})",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = aggregate(results)
    write_raw_csv(results, args.output_dir / "coverage_raw.csv")
    write_markdown(rows, args, args.output_dir / "coverage_summary.md")
    write_plot(rows, args.output_dir)
    print(f"Saved table: {(args.output_dir / 'coverage_summary.md').resolve()}")
    print(f"Saved plot: {(args.output_dir / 'maze_scaling_coverage.png').resolve()}")


if __name__ == "__main__":
    main()
