#!/usr/bin/env python3
"""Plot maze environment configurations ordered by start-to-goal distance."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIGS = [
    REPO_ROOT / "configs/env/gridworld/maze_108_seed7_env.yaml",
    REPO_ROOT / "configs/env/gridworld/maze_108_seed7_env_4.yaml",
    REPO_ROOT / "configs/env/gridworld/maze_108_seed7_env_3.yaml",
]


Cell = tuple[int, int]


@dataclass(frozen=True)
class MazeConfig:
    config_path: Path
    maze_path: Path
    cells: tuple[Cell, ...]
    start: Cell
    goal: Cell
    distance: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one image with multiple maze environment plots, ordered by "
            "shortest-path distance from the start position to the goal position."
        )
    )
    parser.add_argument(
        "configs",
        nargs="*",
        type=Path,
        default=DEFAULT_CONFIGS,
        help="Environment YAML configs to plot. Defaults to the three maze_108_seed7 configs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data_plot/maze_108_seed7_goal_configurations.png",
        help="Where to save the generated image.",
    )
    parser.add_argument(
        "--coordinates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show x/y axes, tick labels, and point coordinate annotations.",
    )
    parser.add_argument(
        "--order",
        choices=("ascending", "descending"),
        default="ascending",
        help="Sort panels by start-to-goal shortest-path distance.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Saved image resolution.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path, *, relative_to: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        relative_to.parent / path,
        REPO_ROOT / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def as_cell(value: Iterable[int], name: str) -> Cell:
    values = list(value)
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two coordinates, got {value}")
    return int(values[0]), int(values[1])


def load_maze_config(config_path: Path) -> MazeConfig:
    config_path = resolve_path(config_path, relative_to=REPO_ROOT)
    config = load_yaml(config_path)
    env_config = config.get("env")
    if not isinstance(env_config, dict):
        raise ValueError(f"{config_path} must contain an 'env' mapping")

    maze_file = env_config.get("maze_file")
    if maze_file is None:
        raise ValueError(f"{config_path} is missing env.maze_file")

    maze_path = resolve_path(maze_file, relative_to=config_path)
    maze = load_yaml(maze_path)
    raw_cells = maze.get("cells")
    if not raw_cells:
        raise ValueError(f"{maze_path} must define a non-empty cells list")

    cells = tuple(sorted(as_cell(cell, "cell") for cell in raw_cells))
    start = as_cell(env_config.get("start_position", maze.get("start_position")), "start_position")
    goal = as_cell(env_config.get("goal_position", maze.get("goal_position")), "goal_position")

    cell_set = set(cells)
    if start not in cell_set:
        raise ValueError(f"Start position {start} in {config_path} is not a valid maze cell")
    if goal not in cell_set:
        raise ValueError(f"Goal position {goal} in {config_path} is not a valid maze cell")

    return MazeConfig(
        config_path=config_path,
        maze_path=maze_path,
        cells=cells,
        start=start,
        goal=goal,
        distance=shortest_path_distance(cells, start, goal),
    )


def shortest_path_distance(cells: Iterable[Cell], start: Cell, goal: Cell) -> int | None:
    cell_set = set(cells)
    distances = {start: 0}
    queue = deque([start])

    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            return distances[(x, y)]

        for neighbor in ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)):
            if neighbor in cell_set and neighbor not in distances:
                distances[neighbor] = distances[(x, y)] + 1
                queue.append(neighbor)

    return None


def sort_key(config: MazeConfig) -> tuple[float, str]:
    distance = float("inf") if config.distance is None else config.distance
    return distance, config.config_path.name


def draw_maze(ax, config: MazeConfig, show_coordinates: bool) -> None:
    cells = set(config.cells)
    min_x = min(x for x, _ in cells)
    max_x = max(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    max_y = max(y for _, y in cells)

    ax.set_facecolor("#8a8a8a")
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            if (x, y) not in cells:
                continue
            ax.add_patch(
                Rectangle(
                    (x - 0.5, y - 0.5),
                    1.0,
                    1.0,
                    facecolor="#161616",
                    edgecolor="#b9b6b6",
                    linewidth=0.25,
                )
            )

    ax.scatter(
        [config.start[0]],
        [config.start[1]],
        marker="s",
        s=150,
        c="#d62728",
        # edgecolors="white",
        linewidths=1.0,
        zorder=5,
    )
    ax.scatter(
        [config.goal[0]],
        [config.goal[1]],
        marker="*",
        s=270,
        c="#2ca02c",
        # edgecolors="white",
        linewidths=0.8,
        zorder=6,
    )

    if show_coordinates:
        ax.annotate(
            f"S {config.start}",
            xy=config.start,
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            color="#d62728",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#d62728", "lw": 0.6},
        )
        ax.annotate(
            f"G {config.goal}",
            xy=config.goal,
            xytext=(0, 15),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#2ca02c",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#2ca02c", "lw": 0.6},
        )

    distance_label = "unreachable" if config.distance is None else str(config.distance)
    ax.set_title(
        f"{config.config_path.stem}\ngoal={config.goal}, distance={distance_label}",
        fontsize=10,
    )

    ax.set_xlim(min_x - 0.5, max_x + 0.5)
    ax.set_ylim(max_y + 0.5, min_y - 0.5)
    ax.set_aspect("equal")

    ax.set_xticks(range(min_x, max_x + 1))
    ax.set_yticks(range(min_y, max_y + 1))

    if show_coordinates:
        ax.set_xlabel("x coordinate")
        ax.set_ylabel("y coordinate")
        ax.tick_params(axis="both", labelsize=8)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)


def save_figure(configs: list[MazeConfig], output: Path, show_coordinates: bool, dpi: int) -> None:
    fig_width = max(5.0, 4.3 * len(configs))
    fig, axes = plt.subplots(1, len(configs), figsize=(fig_width, 5.2), squeeze=False)
    axes = axes[0]

    for ax, config in zip(axes, configs):
        draw_maze(ax, config, show_coordinates=show_coordinates)

    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#d62728",
               markeredgecolor="white", markersize=9, label="Start"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#2ca02c",
               markeredgecolor="white", markersize=12, label="Goal"),
        Rectangle((0, 0), 1, 1, facecolor="#161616", edgecolor="#f2f2f2", label="Valid cell"),
        Rectangle((0, 0), 1, 1, facecolor="#8a8a8a", edgecolor="#8a8a8a", label="Wall"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Maze goal configurations ordered by shortest-path distance", fontsize=14)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configs = [load_maze_config(path) for path in args.configs]
    configs = sorted(configs, key=sort_key, reverse=args.order == "descending")
    save_figure(configs, args.output, show_coordinates=args.coordinates, dpi=args.dpi)

    print(f"Saved {args.output.resolve()}")
    for idx, config in enumerate(configs, start=1):
        distance = "unreachable" if config.distance is None else config.distance
        print(
            f"{idx}. {config.config_path.name}: start={config.start}, "
            f"goal={config.goal}, distance={distance}"
        )


if __name__ == "__main__":
    main()
