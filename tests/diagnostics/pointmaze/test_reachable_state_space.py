"""Random-policy reachability diagnostic for PointMaze U-Maze.

Default run collects 10 random-policy trajectories of 10k steps each and
updates one cumulative maze-overlay plot after every trajectory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from gymnasium import spaces
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gym_env
from agent.utils_debug_visualization import (
    extract_eval_trajectory_point,
    save_maze_trajectory_overlay_plot,
)


CONFIG_DIR = REPO_ROOT / "configs"


def resolve_group_path(group_value: str, group_name: str) -> Path:
    direct = CONFIG_DIR / f"{group_value}.yaml"
    if direct.exists():
        return direct
    grouped = CONFIG_DIR / group_name / f"{group_value}.yaml"
    if grouped.exists():
        return grouped
    raise FileNotFoundError(f"Could not resolve {group_name} config '{group_value}'")


def resolve_config_path(config_name: str | Path) -> Path:
    value = Path(config_name)
    candidates = []
    if value.suffix in {".yaml", ".yml"}:
        candidates.extend([value, REPO_ROOT / value, CONFIG_DIR / value])
    else:
        candidates.extend([CONFIG_DIR / f"{value}.yaml", REPO_ROOT / f"{value}.yaml"])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve config '{config_name}'")


def compose_pretrain_cfg(config_name: str | Path, seed: int):
    cfg = OmegaConf.load(resolve_config_path(config_name))
    defaults = OmegaConf.to_container(cfg.get("defaults", []), resolve=False)
    env_default = None
    for item in defaults:
        if isinstance(item, dict) and "/env" in item:
            env_default = item["/env"]
        elif isinstance(item, dict) and "env" in item:
            env_default = item["env"]

    if env_default is None:
        raise ValueError(f"Could not recover env default from config {config_name}")

    env_cfg = OmegaConf.load(resolve_group_path(env_default, "env"))
    cfg = OmegaConf.merge(env_cfg, cfg)
    cfg.seed = int(seed)
    return cfg


def apply_pointmaze_overrides(cfg, start_position: list[float] | None):
    if start_position is None:
        return cfg
    if "pointmaze" not in cfg.env:
        raise ValueError("Config does not contain env.pointmaze")
    cfg.env.pointmaze.start_position = [float(start_position[0]), float(start_position[1])]
    return cfg


def make_env(cfg, max_episode_steps: int):
    env_kwargs = OmegaConf.to_container(cfg.env, resolve=True)
    env_kwargs.pop("name", None)
    env_kwargs.pop("synthetic_first_transition", None)
    env_kwargs["max_episode_steps"] = int(max_episode_steps)
    return gym_env.make(
        cfg.task_name,
        cfg.obs_type,
        frame_stack=cfg.frame_stack,
        action_repeat=cfg.action_repeat,
        seed=cfg.seed,
        resolution=cfg.resolution,
        grayscale=bool(getattr(cfg, "grayscale", False)),
        url=True,
        **env_kwargs,
    )


def sample_random_action(action_space, rng: np.random.Generator):
    if isinstance(action_space, spaces.Discrete):
        return int(rng.integers(action_space.n))
    if isinstance(action_space, spaces.Box):
        return rng.uniform(action_space.low, action_space.high).astype(action_space.dtype)
    return action_space.sample()


def collect_random_trajectory(env, steps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    env.action_space.seed(seed)
    time_step = env.reset(seed=seed)
    points = []

    point = extract_eval_trajectory_point(env, time_step)
    if point is not None:
        points.append(point)

    for step in range(steps):
        action = sample_random_action(env.action_space, rng)
        time_step = env.step(action)
        point = extract_eval_trajectory_point(env, time_step)
        if point is not None:
            points.append(point)

        if time_step.last() and step < steps - 1:
            raise RuntimeError(
                f"PointMaze episode ended after {step + 1} steps; "
                f"expected one uninterrupted {steps}-step episode. "
                "Increase --max-episode-steps."
            )

    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def get_env_method(env, method_name: str):
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        method = getattr(current, method_name, None)
        if callable(method):
            return method
        current = getattr(current, "env", None)
    return None


def free_space_coverage(env, trajectories: list[np.ndarray], grid_size: int, radius: float) -> dict:
    layout_fn = get_env_method(env, "get_debug_maze_layout")
    if not callable(layout_fn):
        return {}

    layout = layout_fn()
    lower = np.asarray(layout["maze_lower"], dtype=np.float32)
    upper = np.asarray(layout["maze_upper"], dtype=np.float32)
    wall_rectangles = np.asarray(layout["wall_rectangles"], dtype=np.float32).reshape(-1, 4)

    xs = np.linspace(float(lower[0]), float(upper[0]), int(grid_size), dtype=np.float32)
    ys = np.linspace(float(lower[1]), float(upper[1]), int(grid_size), dtype=np.float32)
    grid = np.asarray([(x, y) for x in xs for y in ys], dtype=np.float32)

    free_mask = np.ones(grid.shape[0], dtype=bool)
    for x0, y0, width, height in wall_rectangles:
        inside_x = (grid[:, 0] >= x0) & (grid[:, 0] <= x0 + width)
        inside_y = (grid[:, 1] >= y0) & (grid[:, 1] <= y0 + height)
        free_mask &= ~(inside_x & inside_y)

    free_points = grid[free_mask]
    if not trajectories or free_points.size == 0:
        return {"free_points": int(free_points.shape[0]), "covered_points": 0, "coverage": 0.0}

    samples = np.concatenate([trajectory for trajectory in trajectories if trajectory.size > 0], axis=0)
    covered = np.zeros(free_points.shape[0], dtype=bool)
    chunk_size = 2048
    radius_sq = float(radius) ** 2
    for start in range(0, free_points.shape[0], chunk_size):
        chunk = free_points[start : start + chunk_size]
        distances_sq = ((chunk[:, None, :] - samples[None, :, :]) ** 2).sum(axis=2)
        covered[start : start + chunk.shape[0]] = distances_sq.min(axis=1) <= radius_sq

    return {
        "free_points": int(free_points.shape[0]),
        "covered_points": int(covered.sum()),
        "coverage": float(covered.mean()),
        "grid_size": int(grid_size),
        "coverage_radius": float(radius),
    }


def save_summary(output_path: Path, summary: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="pretrain/pretrain_umaze_baselines")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-trajectories", type=int, default=10)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("tests/outputs/pointmaze/reachable_state_space"))
    parser.add_argument("--grid-size", type=int, default=90)
    parser.add_argument("--coverage-radius", type=float, default=0.08)
    parser.add_argument("--start-position", type=float, nargs=2, default=[1.0, 0.0])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = compose_pretrain_cfg(args.config_name, args.seed)
    cfg = apply_pointmaze_overrides(cfg, args.start_position)
    max_episode_steps = int(args.max_episode_steps or args.steps)
    env = make_env(cfg, max_episode_steps=max_episode_steps)
    trajectories: list[np.ndarray] = []
    summaries = []

    try:
        for idx in range(args.num_trajectories):
            trajectory_seed = int(args.seed + idx)
            trajectory = collect_random_trajectory(env, args.steps, trajectory_seed)
            trajectories.append(trajectory)

            save_maze_trajectory_overlay_plot(
                trajectories=trajectories,
                env=env,
                step=(idx + 1) * args.steps,
                save_dir=args.output_dir,
            )

            coverage = free_space_coverage(
                env,
                trajectories,
                grid_size=args.grid_size,
                radius=args.coverage_radius,
            )
            summary = {
                "trajectory_index": idx,
                "trajectory_seed": trajectory_seed,
                "trajectory_steps": int(args.steps),
                "trajectory_points": int(trajectory.shape[0]),
                "num_trajectories_collected": idx + 1,
                "max_episode_steps": max_episode_steps,
                "single_episode": True,
                **coverage,
            }
            summaries.append(summary)
            save_summary(args.output_dir / "reachability_summary.json", {"runs": summaries})

            coverage_text = ""
            if "coverage" in coverage:
                coverage_text = f", coverage={coverage['coverage']:.3f}"
            print(
                f"Finished trajectory {idx + 1}/{args.num_trajectories}: "
                f"points={trajectory.shape[0]}{coverage_text}"
            )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    print(f"Saved cumulative plots and summary to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
