#!/usr/bin/env python3
"""Render PointMaze configs and assert configured goals survive env wrapping."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

import gym_env


CONFIG_DIR = ROOT / "configs/env/pointmaze"
OUTPUT_DIR = Path(__file__).resolve().parent / "goal_demos"


def find_method(env, name):
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        method = getattr(current, name, None)
        if callable(method):
            return method
        current = getattr(current, "env", None)
    raise AttributeError(f"PointMaze wrapper has no {name}()")


def make_env(config):
    kwargs = OmegaConf.to_container(config.env, resolve=True)
    name = kwargs.pop("name")
    return gym_env.make(
        name,
        "states",
        frame_stack=1,
        action_repeat=1,
        seed=0,
        resolution=84,
        grayscale=False,
        url=True,
        **kwargs,
    )


def render_config(config_path: Path) -> Path:
    config = OmegaConf.load(config_path)
    expected_goal = np.asarray(config.env.pointmaze.goal_position, dtype=np.float32)
    expected_start = np.asarray(config.env.pointmaze.start_position, dtype=np.float32)
    env = make_env(config)
    try:
        env.reset(seed=0)
        coordinates = find_method(env, "get_debug_coordinates")()
        layout = find_method(env, "get_debug_maze_layout")()
        actual_goal = np.asarray(coordinates["fixed_goal"], dtype=np.float32)
        if not np.allclose(actual_goal, expected_goal):
            raise AssertionError(
                f"{config_path.name}: configured goal {expected_goal} became {actual_goal}"
            )

        fig, ax = plt.subplots(figsize=(8, 6))
        for x, y, width, height in layout["wall_rectangles"]:
            ax.add_patch(
                patches.Rectangle(
                    (x, y), width, height, facecolor="#2f3f58", edgecolor="black"
                )
            )
        ax.scatter(*expected_start, s=130, c="#2ca02c", marker="o", label="start", zorder=3)
        ax.scatter(*actual_goal, s=190, c="#d62728", marker="*", label="goal", zorder=4)
        ax.annotate(
            f"goal = ({actual_goal[0]:g}, {actual_goal[1]:g})",
            actual_goal,
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=11,
            weight="bold",
        )
        lower = np.asarray(layout["maze_lower"])
        upper = np.asarray(layout["maze_upper"])
        ax.set_xlim(lower[0] - 0.25, upper[0] + 0.25)
        ax.set_ylim(lower[1] - 0.25, upper[1] + 0.25)
        ax.set_aspect("equal")
        ax.set_title(config_path.stem)
        ax.legend(loc="upper right")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.tight_layout()

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{config_path.stem}.png"
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        print(f"PASS {config_path.name}: goal={actual_goal.tolist()} -> {output_path}")
        return output_path
    finally:
        env.close()


def main() -> None:
    for config_path in sorted(CONFIG_DIR.glob("pointmaze_*_goal_*.yaml")):
        render_config(config_path)


if __name__ == "__main__":
    main()
