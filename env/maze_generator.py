"""
Utilities for generating discrete grid mazes and their config files.

The generated architecture YAML stores exactly ``n_states`` traversable cells.
``MazeEnv`` can load that architecture and expose it as a Gymnasium env.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import deque
from pathlib import Path
from typing import Iterable

import yaml


Cell = tuple[int, int]


class LiteralString(str):
    """String marker for YAML literal block output."""


def _literal_string_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.SafeDumper.add_representer(LiteralString, _literal_string_representer)


def _neighbors(cell: Cell) -> list[Cell]:
    x, y = cell
    return [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]


def _inside(cell: Cell, width: int, height: int) -> bool:
    x, y = cell
    return 0 <= x < width and 0 <= y < height


def _auto_dimensions(n_states: int) -> tuple[int, int]:
    """Choose a roomy rectangle so the generated state set can include walls."""
    side = math.sqrt(max(1, n_states) * 1.8)
    width = max(3, math.ceil(side))
    height = max(3, math.ceil(n_states * 1.8 / width))
    while width * height < n_states:
        width += 1
    return width, height


def _normalize_cells(cells: Iterable[Cell]) -> list[Cell]:
    cells = list(cells)
    min_x = min(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    normalized = sorted((x - min_x, y - min_y) for x, y in cells)
    return normalized


def _grow_connected_cells(
    n_states: int,
    width: int,
    height: int,
    rng: random.Random,
) -> list[Cell]:
    """Grow a connected, mostly tree-like set of cells with exactly n_states."""
    start = (rng.randrange(width), rng.randrange(height))
    cells: set[Cell] = {start}
    frontier: set[Cell] = {
        nbr for nbr in _neighbors(start) if _inside(nbr, width, height)
    }

    while len(cells) < n_states:
        if not frontier:
            raise RuntimeError("frontier exhausted before reaching requested state count")

        candidates = []
        for cell in frontier:
            existing_degree = sum(nbr in cells for nbr in _neighbors(cell))
            candidates.append((existing_degree, cell))

        min_degree = min(degree for degree, _ in candidates)
        low_degree = [cell for degree, cell in candidates if degree == min_degree]
        next_cell = rng.choice(low_degree)
        frontier.remove(next_cell)
        cells.add(next_cell)

        for nbr in _neighbors(next_cell):
            if _inside(nbr, width, height) and nbr not in cells:
                frontier.add(nbr)

    return _normalize_cells(cells)


def _shortest_path_distances(cells: list[Cell], start: Cell) -> dict[Cell, int]:
    cell_set = set(cells)
    distances = {start: 0}
    queue = deque([start])

    while queue:
        cell = queue.popleft()
        for nbr in _neighbors(cell):
            if nbr in cell_set and nbr not in distances:
                distances[nbr] = distances[cell] + 1
                queue.append(nbr)

    return distances


def _farthest_cell(cells: list[Cell], start: Cell) -> Cell:
    distances = _shortest_path_distances(cells, start)
    return max(distances, key=lambda cell: (distances[cell], cell[1], cell[0]))


def render_ascii(cells: list[Cell], start: Cell | None = None, goal: Cell | None = None) -> str:
    """Render traversable cells as '.', walls as '#', start as 'S', and goal as 'G'."""
    cell_set = set(cells)
    max_x = max(x for x, _ in cells)
    max_y = max(y for _, y in cells)
    rows = []
    for y in range(max_y + 1):
        row = []
        for x in range(max_x + 1):
            cell = (x, y)
            if cell == start:
                row.append("S")
            elif cell == goal:
                row.append("G")
            elif cell in cell_set:
                row.append(".")
            else:
                row.append("#")
        rows.append("".join(row))
    return "\n".join(rows)


def generate_maze_architecture(
    n_states: int,
    seed: int | None = None,
    width: int | None = None,
    height: int | None = None,
) -> dict:
    """
    Generate a connected discrete maze architecture with exactly ``n_states`` cells.

    The start is chosen as one end of the generated graph diameter approximation,
    and the goal is chosen near the opposite end.
    """
    if n_states < 2:
        raise ValueError("n_states must be at least 2")

    rng = random.Random(seed)
    if width is None or height is None:
        width, height = _auto_dimensions(n_states)
    if width * height < n_states:
        raise ValueError("width * height must be at least n_states")

    while True:
        try:
            cells = _grow_connected_cells(n_states, width, height, rng)
            break
        except RuntimeError:
            width += 1
            height += 1

    first = rng.choice(cells)
    start = _farthest_cell(cells, first)
    goal = _farthest_cell(cells, start)

    return {
        "version": 1,
        "kind": "discrete_maze",
        "metadata": {
            "n_states": n_states,
            "seed": seed,
            "width": max(x for x, _ in cells) + 1,
            "height": max(y for _, y in cells) + 1,
        },
        "start_position": list(start),
        "goal_position": list(goal),
        "goal_positions": [list(goal)],
        "cells": [[x, y] for x, y in cells],
        "ascii": LiteralString(render_ascii(cells, start=start, goal=goal)),
    }


def write_maze_files(
    n_states: int,
    architecture_path: str | Path,
    env_config_path: str | Path | None = None,
    seed: int | None = None,
    width: int | None = None,
    height: int | None = None,
    max_steps: int | None = None,
) -> dict:
    """Generate a maze architecture YAML and optionally a Hydra env config YAML."""
    architecture = generate_maze_architecture(
        n_states=n_states,
        seed=seed,
        width=width,
        height=height,
    )

    architecture_path = Path(architecture_path)
    architecture_path.parent.mkdir(parents=True, exist_ok=True)
    with architecture_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(architecture, f, sort_keys=False)

    if env_config_path is not None:
        env_config_path = Path(env_config_path)
        env_config_path.parent.mkdir(parents=True, exist_ok=True)
        env_config = {
            "env": {
                "name": "Maze-v0",
                "maze_file": str(architecture_path),
                "max_steps": max_steps if max_steps is not None else 4 * n_states,
                "render_mode": "rgb_array",
                "show_coordinates": False,
                "goal_position": architecture["goal_position"],
                "start_position": architecture["start_position"],
            }
        }
        with env_config_path.open("w", encoding="utf-8") as f:
            f.write("# @package _global_\n")
            f.write("# Generated Maze Environment Configuration\n\n")
            yaml.safe_dump(env_config, f, sort_keys=False)

    return architecture


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a discrete maze YAML.")
    parser.add_argument("n_states", type=int, help="Number of traversable discrete states")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument(
        "--architecture-path",
        default="configs/env/gridworld/mazes/generated_maze.yaml",
    )
    parser.add_argument("--env-config-path", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    architecture = write_maze_files(
        n_states=args.n_states,
        architecture_path=args.architecture_path,
        env_config_path=args.env_config_path,
        seed=args.seed,
        width=args.width,
        height=args.height,
        max_steps=args.max_steps,
    )
    print(render_ascii(
        [(x, y) for x, y in architecture["cells"]],
        start=tuple(architecture["start_position"]),
        goal=tuple(architecture["goal_position"]),
    ))


if __name__ == "__main__":
    main()
