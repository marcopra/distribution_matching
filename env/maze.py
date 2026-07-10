"""
Discrete maze environment loaded from a generated architecture YAML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import gymnasium as gym
import yaml

from env.rooms import BaseRoomEnv


Cell = tuple[int, int]


def _as_cell(value, name: str) -> Cell:
    if isinstance(value, int):
        raise TypeError(f"{name} must be a coordinate pair, not a state index")
    if value is None:
        raise ValueError(f"{name} cannot be None")
    if len(value) != 2:
        raise ValueError(f"{name} must have exactly two coordinates")
    return (int(value[0]), int(value[1]))


def _resolve_maze_path(maze_file: str | Path) -> Path:
    path = Path(maze_file).expanduser()
    if path.exists() or path.is_absolute():
        return path

    repo_root = Path(__file__).resolve().parents[1]
    repo_path = repo_root / path
    if repo_path.exists():
        return repo_path

    return path


def load_maze_architecture(maze_file: str | Path) -> dict:
    path = _resolve_maze_path(maze_file)
    with path.open("r", encoding="utf-8") as f:
        architecture = yaml.safe_load(f)
    if not isinstance(architecture, dict):
        raise ValueError(f"Maze architecture at {path} must contain a YAML mapping")
    return architecture


class MazeEnv(BaseRoomEnv):
    """
    A 4-action discrete maze loaded from YAML.

    Args:
        maze_file: Path to an architecture YAML produced by ``maze_generator``.
        maze: Optional in-memory architecture mapping.
        cells: Optional direct list of traversable cells, used when no file is given.
        goal_position: Goal override as ``(x, y)`` or state index.
        start_position: Start override as ``(x, y)`` or state index.
    """

    def __init__(
        self,
        maze_file: Optional[str] = None,
        maze: Optional[dict] = None,
        cells: Optional[list[list[int]]] = None,
        goal_position: Optional[Tuple[int, int]] = None,
        start_position: Optional[Tuple[int, int]] = None,
        max_steps: int = 300,
        render_mode: Optional[str] = None,
        show_coordinates: bool = False,
        lava: bool = False,
        dense_reward: bool = False,
    ):
        if maze is not None and maze_file is not None:
            raise ValueError("Pass either maze or maze_file, not both")

        if maze_file is not None:
            maze = load_maze_architecture(maze_file)
        elif maze is None:
            maze = {"cells": cells}

        self.maze_file = maze_file
        self.maze = maze
        self._architecture_cells = self._load_cells(maze)
        self.goal_positions = self._load_goal_positions(maze)
        self._architecture_goal = self._load_optional_cell(maze.get("goal_position"))
        if self._architecture_goal is None and self.goal_positions:
            self._architecture_goal = self.goal_positions[0]
        self._architecture_start = self._load_optional_cell(maze.get("start_position"))
        self._architecture_ascii = maze.get("ascii")

        if goal_position is None:
            goal_position = self._architecture_goal
        if start_position is None:
            start_position = self._architecture_start

        super().__init__(
            goal_position=goal_position,
            start_position=start_position,
            max_steps=max_steps,
            render_mode=render_mode,
            show_coordinates=show_coordinates,
            lava=lava,
            dense_reward=dense_reward,
        )

        expected_n_states = maze.get("metadata", {}).get("n_states")
        if expected_n_states is not None and int(expected_n_states) != len(self._architecture_cells):
            raise ValueError(
                f"Maze metadata says n_states={expected_n_states}, "
                f"but cells contains {len(self._architecture_cells)} entries"
            )

    def _load_cells(self, maze: dict) -> list[Cell]:
        cells = maze.get("cells")
        if not cells:
            raise ValueError("Maze architecture must define a non-empty cells list")

        parsed = [_as_cell(cell, "cell") for cell in cells]
        if len(set(parsed)) != len(parsed):
            raise ValueError("Maze architecture contains duplicate cells")
        return sorted(parsed)

    def _load_optional_cell(self, value) -> Optional[Cell]:
        if value is None:
            return None
        return _as_cell(value, "position")

    def _load_goal_positions(self, maze: dict) -> list[Cell]:
        values = maze.get("goal_positions") or []
        goals = [_as_cell(value, "goal_position") for value in values]
        invalid_goals = [goal for goal in goals if goal not in self._architecture_cells]
        if invalid_goals:
            raise ValueError(f"goal_positions contains invalid cells: {invalid_goals}")
        return goals

    def _build_cells(self):
        for cell in self._architecture_cells:
            self._add_cell(cell)

    def _get_default_goal(self) -> Cell:
        if self._architecture_goal is not None:
            return self._architecture_goal
        return self.cells[-1]

    def get_ascii_map(self) -> str:
        """Return the saved ASCII map, or render the current architecture as ASCII."""
        if self._architecture_ascii:
            return str(self._architecture_ascii)
        return self._render_ansi(show_goal=True)


gym.register(
    id="Maze-v0",
    entry_point="env.maze:MazeEnv",
    max_episode_steps=300,
)
