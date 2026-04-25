from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from gymnasium import spaces
from omegaconf import OmegaConf

import gym_env


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


def extract_xy(env, time_step) -> np.ndarray | None:
    debug_coordinates = get_env_method(env, "get_debug_coordinates")
    if callable(debug_coordinates):
        info = debug_coordinates()
        if isinstance(info, dict) and "xy" in info:
            xy = np.asarray(info["xy"], dtype=np.float32).reshape(-1)
            if xy.size >= 2:
                return xy[:2]

    proprio = np.asarray(getattr(time_step, "proprio_observation", []), dtype=np.float32).reshape(-1)
    if proprio.size >= 2:
        return proprio[:2]
    return None


def get_debug_point(env, key: str) -> np.ndarray | None:
    debug_coordinates = get_env_method(env, "get_debug_coordinates")
    if not callable(debug_coordinates):
        return None

    info = debug_coordinates()
    if not isinstance(info, dict) or key not in info:
        return None

    point = np.asarray(info[key], dtype=np.float32).reshape(-1)
    if point.size < 2:
        return None
    return point[:2]


def get_maze_layout(env) -> dict[str, np.ndarray] | None:
    debug_layout = get_env_method(env, "get_debug_maze_layout")
    if not callable(debug_layout):
        return None

    layout = debug_layout()
    if not isinstance(layout, dict):
        return None

    maze_lower = layout.get("maze_lower")
    maze_upper = layout.get("maze_upper")
    wall_rectangles = layout.get("wall_rectangles")
    if maze_lower is None or maze_upper is None or wall_rectangles is None:
        return None

    maze_lower = np.asarray(maze_lower, dtype=np.float32).reshape(-1)
    maze_upper = np.asarray(maze_upper, dtype=np.float32).reshape(-1)
    wall_rectangles = np.asarray(wall_rectangles, dtype=np.float32).reshape(-1, 4)
    if maze_lower.size != 2 or maze_upper.size != 2:
        return None

    return {
        "maze_lower": maze_lower,
        "maze_upper": maze_upper,
        "wall_rectangles": wall_rectangles,
    }


def sample_random_action(action_space, rng: np.random.Generator):
    if isinstance(action_space, spaces.Discrete):
        return int(rng.integers(action_space.n))
    if isinstance(action_space, spaces.Box):
        return rng.uniform(action_space.low, action_space.high).astype(action_space.dtype)
    return action_space.sample()


def collect_random_rollout(env, steps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    env.action_space.seed(seed)
    time_step = env.reset(seed=seed)
    coords = []
    reset_seed = seed

    for step in range(steps):
        xy = extract_xy(env, time_step)
        if xy is not None:
            coords.append(xy)

        action = sample_random_action(env.action_space, rng)
        time_step = env.step(action)
        if time_step.last():
            reset_seed += step + 1
            time_step = env.reset(seed=reset_seed)

    if not coords:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(coords, dtype=np.float32)


def get_plot_bounds(coords: np.ndarray, layout: dict[str, np.ndarray] | None):
    if layout is not None:
        return layout["maze_lower"], layout["maze_upper"]

    lower = coords.min(axis=0)
    upper = coords.max(axis=0)
    margin = np.maximum((upper - lower) * 0.05, 1e-3)
    return lower - margin, upper + margin


def overlay_maze_walls(ax, layout: dict[str, np.ndarray] | None) -> None:
    if layout is None:
        return

    for x0, y0, width, height in layout["wall_rectangles"]:
        ax.add_patch(
            patches.Rectangle(
                (x0, y0),
                width,
                height,
                facecolor="black",
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )
        )

    lower = layout["maze_lower"]
    upper = layout["maze_upper"]
    ax.add_patch(
        patches.Rectangle(
            (lower[0], lower[1]),
            upper[0] - lower[0],
            upper[1] - lower[1],
            fill=False,
            edgecolor="black",
            linewidth=1.5,
            zorder=4,
        )
    )


def plot_heatmap(env, coords: np.ndarray, bins: int, output_path: Path) -> None:
    if coords.size == 0:
        raise RuntimeError("No XY coordinates were collected from the PointMaze rollout.")

    layout = get_maze_layout(env)
    lower, upper = get_plot_bounds(coords, layout)
    heatmap, _, _ = np.histogram2d(
        coords[:, 0],
        coords[:, 1],
        bins=bins,
        range=[[lower[0], upper[0]], [lower[1], upper[1]]],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(
        heatmap.T,
        origin="lower",
        aspect="equal",
        extent=[lower[0], upper[0], lower[1], upper[1]],
        cmap="viridis",
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Visit count")

    overlay_maze_walls(ax, layout)

    start = get_debug_point(env, "fixed_start")
    if start is not None:
        ax.scatter(
            start[0],
            start[1],
            marker="*",
            s=190,
            c="white",
            edgecolors="black",
            linewidths=0.9,
            zorder=5,
            label="start",
        )

    goal = get_debug_point(env, "fixed_goal")
    if goal is not None:
        ax.scatter(
            goal[0],
            goal[1],
            marker="X",
            s=120,
            c="tab:red",
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
            label="goal",
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"PointMaze random-policy visitation (n={len(coords)})")
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    if start is not None or goal is not None:
        ax.legend(loc="upper right")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_env(args):
    cfg = OmegaConf.load(args.env_config)
    env_cfg = cfg.env if "env" in cfg else cfg
    env_kwargs = OmegaConf.to_container(env_cfg, resolve=True)
    task_name = env_kwargs.pop("name")

    return gym_env.make(
        task_name,
        args.obs_type,
        frame_stack=1,
        action_repeat=args.action_repeat,
        seed=args.seed,
        resolution=args.resolution,
        grayscale=False,
        url=args.continuing_task,
        **env_kwargs,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect a random-policy rollout in PointMaze and save an XY visitation heatmap."
    )
    parser.add_argument(
        "--env-config",
        type=Path,
        default=Path("configs/env/pointmaze/pointmaze_umaze_goal_1.yaml"),
        help="Path to a PointMaze env yaml, e.g. configs/env/pointmaze/pointmaze_umaze_goal_1.yaml.",
    )
    parser.add_argument("--output", type=Path, default=Path("pointmaze_random_policy_heatmap.png"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--bins", type=int, default=36)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=84)
    parser.add_argument("--obs-type", type=str, default="proprio")
    parser.add_argument(
        "--continuing-task",
        action="store_true",
        help="Use gym_env's url=True path, which keeps PointMaze episodes from ending at success.",
    )
    return parser.parse_args()

import time
def main():
    args = parse_args()
    env = make_env(args)
    try:
        # measure time in seconds
        start_time = time.time()
        coords = collect_random_rollout(env, steps=args.steps, seed=args.seed)
        plot_heatmap(env, coords, bins=args.bins, output_path=args.output)
        end_time = time.time()
        print(f"Total time taken: {end_time - start_time:.2f} seconds")
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    print(f"Collected {len(coords)} PointMaze positions with a random policy.")
    print(f"Saved heatmap to {args.output.resolve()}")


if __name__ == "__main__":
    main()
