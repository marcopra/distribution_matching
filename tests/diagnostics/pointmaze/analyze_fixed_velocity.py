"""Diagnose fixed-velocity PointMaze reachability and reward difficulty.

Runs a shortest-cell-path controller for several velocities, measures actual
per-step displacement, and writes plots plus a machine-readable JSON summary.
Uses the same wrappers as training.

Example:
  python tests/diagnostics/pointmaze/analyze_fixed_velocity.py \
    --goal -3 3 --horizon 2000 --velocities 0.5 1 2 3 5 7.5 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gym_env


ACTIONS = np.asarray([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.float32)


def make_env(args, velocity: float, reward_type: str):
    return gym_env.make(
        args.env_name,
        obs_type="states",
        frame_stack=1,
        action_repeat=1,
        seed=args.seed,
        url=False,
        render_mode="rgb_array",
        max_episode_steps=args.horizon,
        pointmaze={
            "top_down_camera": False,
            "discrete_actions": True,
            "direct_velocity_actions": True,
            "max_velocity": velocity,
            "only_xy_position": True,
            "start_position": args.start,
            "start_position_variance": 0.0,
            "reward_type": reward_type,
            "goal_position": args.goal,
        },
    )


def base_maze(env):
    return env.unwrapped.maze


def closest_free_cell(maze, xy):
    candidates = []
    for row, values in enumerate(maze.maze_map):
        for col, value in enumerate(values):
            if value != 1:
                center = np.asarray(maze.cell_rowcol_to_xy(np.asarray([row, col])))
                candidates.append((float(np.linalg.norm(center - xy)), (row, col)))
    return min(candidates)[1]


def shortest_cell_path(env, start, goal):
    maze = base_maze(env)
    source = closest_free_cell(maze, np.asarray(start))
    target = closest_free_cell(maze, np.asarray(goal))
    queue = deque([source])
    parent = {source: None}
    while queue:
        cell = queue.popleft()
        if cell == target:
            break
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (cell[0] + dr, cell[1] + dc)
            if (
                0 <= nxt[0] < len(maze.maze_map)
                and 0 <= nxt[1] < len(maze.maze_map[0])
                and maze.maze_map[nxt[0]][nxt[1]] != 1
                and nxt not in parent
            ):
                parent[nxt] = cell
                queue.append(nxt)
    if target not in parent:
        raise RuntimeError(f"No free-cell path from {source} to {target}")
    cells = []
    current = target
    while current is not None:
        cells.append(current)
        current = parent[current]
    cells.reverse()
    centers = [np.asarray(maze.cell_rowcol_to_xy(np.asarray(cell)), dtype=np.float32) for cell in cells]
    return cells, np.asarray(centers)


def compress_waypoints(centers, goal):
    """Keep turn points, then exact goal; avoids needless cell-center dithering."""
    if len(centers) <= 1:
        return np.asarray([goal], dtype=np.float32)
    kept = []
    old_direction = None
    for idx in range(1, len(centers)):
        direction = np.sign(centers[idx] - centers[idx - 1]).astype(int)
        if old_direction is not None and not np.array_equal(direction, old_direction):
            kept.append(centers[idx - 1])
        old_direction = direction
    kept.append(np.asarray(goal, dtype=np.float32))
    return np.asarray(kept, dtype=np.float32)


def run_controller(env, waypoints, goal, horizon, seed):
    ts = env.reset(seed=seed)
    positions = [np.asarray(ts.observation, dtype=np.float32).copy()]
    rewards = [0.0]
    distances = [float(np.linalg.norm(positions[-1] - goal))]
    waypoint_idx = 0
    success_step = None

    for step in range(1, horizon + 1):
        pos = positions[-1]
        target = waypoints[min(waypoint_idx, len(waypoints) - 1)]
        delta = target - pos
        tolerance = 0.08
        if np.linalg.norm(delta) <= tolerance and waypoint_idx < len(waypoints) - 1:
            waypoint_idx += 1
            target = waypoints[waypoint_idx]
            delta = target - pos
        axis = int(np.argmax(np.abs(delta)))
        desired = np.zeros(2, dtype=np.float32)
        desired[axis] = 1.0 if delta[axis] >= 0 else -1.0
        action = int(np.argmax(ACTIONS @ desired))
        ts = env.step(action)
        pos = np.asarray(ts.observation, dtype=np.float32).copy()
        positions.append(pos)
        rewards.append(float(ts.reward))
        distances.append(float(np.linalg.norm(pos - goal)))
        if bool(ts.success) and success_step is None:
            success_step = step
            break
        if ts.last():
            break
    return {
        "positions": np.asarray(positions),
        "rewards": np.asarray(rewards),
        "distances": np.asarray(distances),
        "success_step": success_step,
    }


def measure_open_step(env, seed):
    ts = env.reset(seed=seed)
    start = np.asarray(ts.observation, dtype=np.float32)
    moves = []
    for action in range(4):
        env.reset(seed=seed)
        nxt = np.asarray(env.step(action).observation, dtype=np.float32)
        moves.append(float(np.linalg.norm(nxt - start)))
    positive = [move for move in moves if move > 1e-5]
    return float(np.median(positive)), moves


def random_success_rate(env, episodes, horizon, seed):
    rng = np.random.default_rng(seed)
    hit_steps = []
    minimum_distances = []
    for episode in range(episodes):
        ts = env.reset(seed=seed + episode)
        best = float(np.linalg.norm(np.asarray(ts.observation) - env.fixed_goal))
        hit = None
        for step in range(1, horizon + 1):
            ts = env.step(int(rng.integers(4)))
            distance = float(np.linalg.norm(np.asarray(ts.observation) - env.fixed_goal))
            best = min(best, distance)
            if bool(ts.success):
                hit = step
                break
            if ts.last():
                break
        hit_steps.append(hit)
        minimum_distances.append(best)
    successes = [step for step in hit_steps if step is not None]
    return {
        "episodes": episodes,
        "success_rate": len(successes) / episodes if episodes else None,
        "median_success_step": float(np.median(successes)) if successes else None,
        "median_minimum_distance": float(np.median(minimum_distances)) if episodes else None,
        "hit_steps": hit_steps,
    }


def draw_maze(ax, layout):
    for x, y, w, h in np.asarray(layout["wall_rectangles"]):
        ax.add_patch(Rectangle((x, y), w, h, color="0.25"))
    ax.set_aspect("equal")
    ax.set_xlim(layout["maze_lower"][0], layout["maze_upper"][0])
    ax.set_ylim(layout["maze_lower"][1], layout["maze_upper"][1])


def save_plots(output_dir, layout, runs, start, goal, horizon):
    fig, axes = plt.subplots(1, len(runs), figsize=(4 * len(runs), 4), squeeze=False)
    for ax, (velocity, run) in zip(axes[0], runs.items()):
        draw_maze(ax, layout)
        path = run["positions"]
        ax.plot(path[:, 0], path[:, 1], lw=1.2)
        ax.scatter(*start, marker="o", color="tab:blue", label="start")
        ax.scatter(*goal, marker="*", s=100, color="tab:red", label="goal")
        ax.set_title(f"v={velocity:g}, hit={run['success_step']}")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "trajectories.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for velocity, run in runs.items():
        ax.plot(run["distances"], label=f"v={velocity:g}")
    ax.axhline(0.45, ls="--", color="k", lw=1, label="sparse success radius (0.45)")
    ax.set(xlabel="environment step", ylabel="Euclidean distance to goal", xlim=(0, horizon))
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "distance_to_goal.png", dpi=180)
    plt.close(fig)

    velocities = list(runs)
    hit_steps = [runs[v]["success_step"] or horizon for v in velocities]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(velocities, hit_steps, "o-")
    ax.axhline(horizon, ls="--", color="tab:red", label=f"horizon={horizon}")
    ax.set(xlabel="max velocity", ylabel="controller steps to success")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "velocity_sweep.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    rates = [runs[v]["random"]["success_rate"] for v in velocities]
    ax.bar([str(v) for v in velocities], rates)
    ax.set(xlabel="max velocity", ylabel="random-policy success rate", ylim=(0, 1))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "random_policy_success.png", dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="PointMaze_LargeDense-v3")
    parser.add_argument("--start", nargs=2, type=float, default=[0.5, 0.0])
    parser.add_argument("--goal", nargs=2, type=float, default=[-3.0, 3.0])
    parser.add_argument("--velocities", nargs="+", type=float, default=[0.5, 1, 2, 3, 5, 7.5, 10])
    parser.add_argument("--horizon", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-episodes", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("tests/outputs/pointmaze/fixed_velocity"))
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = {}
    metadata = {}
    layout = None
    route_cells = None
    waypoints = None
    try:
        for velocity in args.velocities:
            env = make_env(args, velocity, "sparse")
            try:
                layout = env.get_debug_maze_layout()
                cells, centers = shortest_cell_path(env, args.start, args.goal)
                route_cells = cells
                waypoints = compress_waypoints(centers, args.goal)
                step_size, directional_moves = measure_open_step(env, args.seed)
                run = run_controller(env, waypoints, np.asarray(args.goal), args.horizon, args.seed)
                run["random"] = random_success_rate(
                    env, args.random_episodes, args.horizon, args.seed + 10_000
                )
                runs[velocity] = run
                metadata[str(velocity)] = {
                    "measured_open_step": step_size,
                    "directional_first_steps": directional_moves,
                    "success": run["success_step"] is not None,
                    "success_step": run["success_step"],
                    "minimum_distance": float(run["distances"].min()),
                    "final_distance": float(run["distances"][-1]),
                    "horizon_fraction": None if run["success_step"] is None else run["success_step"] / args.horizon,
                    "random_policy": run["random"],
                }
            finally:
                env.close()
    finally:
        plt.close("all")

    save_plots(args.output_dir, layout, runs, np.asarray(args.start), np.asarray(args.goal), args.horizon)
    summary = {
        "environment": args.env_name,
        "start": args.start,
        "goal": args.goal,
        "horizon": args.horizon,
        "reward_type_tested": "sparse",
        "sparse_success_radius": 0.45,
        "shortest_cell_route": [list(cell) for cell in route_cells],
        "route_cell_count": len(route_cells),
        "controller_waypoints": waypoints.tolist(),
        "runs": metadata,
        "notes": {
            "dense_reward": "Gymnasium Robotics PointMaze dense reward is negative Euclidean goal distance.",
            "sparse_reward": "Sparse reward is 1 inside 0.45 goal distance, otherwise 0.",
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Plots and summary saved to {args.output_dir}")


if __name__ == "__main__":
    main()
