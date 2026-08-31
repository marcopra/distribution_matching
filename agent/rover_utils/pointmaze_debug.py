"""PointMaze synthetic-data provider for optional Rover diagnostics.

Kept beside Rover debug orchestration instead of generic agent utilities.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
import torch

from utils import ColorPrint
from agent.rover_utils.actor_data import RawTransitions

class PointMazeNystromDebugHelper:
    """Build fixed PointMaze landmark transitions for Nyström debugging."""

    def __init__(self, border_margin: float = 0.05, oversample: float = 2.0, exact_grid: bool = False):
        self.border_margin = float(border_margin)
        self.oversample = float(oversample)
        self.exact_grid = bool(exact_grid)
        self.wrapped_env = None
        self.env = None
        self._subsample_batch = None
        self._subsample_batches = {}
        self._batch_xy_points = {}
        self._fixed_xy_points = None
        self._fixed_actions = None
        self._fixed_plot_stats = None
        self._last_grid_spacing = None

    @property
    def fixed_xy_points(self):
        return self._fixed_xy_points

    @property
    def fixed_actions(self):
        return self._fixed_actions

    @property
    def fixed_plot_stats(self):
        return self._fixed_plot_stats

    def __getstate__(self):
        state = self.__dict__.copy()
        state["wrapped_env"] = None
        state["env"] = None
        return state

    def attach_env(self, env):
        self.wrapped_env = env
        self.env = self._find_discrete_env(env)
        self.clear_cache()

    def clear_cache(self):
        self._subsample_batch = None
        self._subsample_batches = {}
        self._batch_xy_points = {}
        self._fixed_xy_points = None
        self._fixed_actions = None
        self._fixed_plot_stats = None
        self._last_grid_spacing = None

    @staticmethod
    def _find_discrete_env(env):
        current = env
        while current is not None:
            if all(hasattr(current, attr) for attr in ("n_states", "idx_to_state", "state_to_idx")):
                return current
            if hasattr(current, "env"):
                current = current.env
            elif hasattr(current, "unwrapped") and current.unwrapped is not current:
                current = current.unwrapped
            else:
                break
        return getattr(env, "unwrapped", env)

    def _iter_env_chain(self):
        current = self.wrapped_env
        visited = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            yield current
            current = getattr(current, "env", None)

    def _env_method(self, method_name):
        for current in self._iter_env_chain():
            method = getattr(current, method_name, None)
            if callable(method):
                return method
        return None

    def _has_point_env(self):
        base_env = getattr(self.wrapped_env, "unwrapped", None)
        return base_env is not None and getattr(base_env, "point_env", None) is not None

    def _base_and_point_env(self):
        if self.wrapped_env is None:
            raise RuntimeError("PointMaze Nyström grid requires insert_env(env) before actor updates.")
        base_env = getattr(self.wrapped_env, "unwrapped", None)
        point_env = getattr(base_env, "point_env", None)
        if base_env is None or point_env is None:
            raise RuntimeError("PointMaze Nyström grid requires unwrapped.point_env.")
        return base_env, point_env

    def _set_state(self, xy, velocity=(0.0, 0.0)):
        base_env, point_env = self._base_and_point_env()
        qpos = point_env.data.qpos.copy()
        qvel = np.zeros_like(point_env.data.qvel)
        qpos[:2] = np.asarray(xy, dtype=np.float64)
        qvel[:2] = np.asarray(velocity, dtype=np.float64)
        point_env.set_state(qpos, qvel)
        if hasattr(base_env, "update_target_site_pos"):
            base_env.update_target_site_pos()

    @contextmanager
    def _preserve_state(self):
        base_env, point_env = self._base_and_point_env()
        snapshot = {
            "qpos": point_env.data.qpos.copy(),
            "qvel": point_env.data.qvel.copy(),
            "wrappers": [],
        }
        for current in self._iter_env_chain():
            wrapper_state = {}
            if hasattr(current, "_frames"):
                wrapper_state["frames"] = [frame.copy() for frame in list(current._frames)]
            if hasattr(current, "_cached_hidden_render"):
                cached = current._cached_hidden_render
                wrapper_state["cached_hidden_render"] = None if cached is None else cached.copy()
            if wrapper_state:
                snapshot["wrappers"].append((current, wrapper_state))

        try:
            yield
        finally:
            point_env.set_state(snapshot["qpos"], snapshot["qvel"])
            if hasattr(base_env, "update_target_site_pos"):
                base_env.update_target_site_pos()
            for wrapper, wrapper_state in snapshot["wrappers"]:
                if "frames" in wrapper_state and hasattr(wrapper, "_frames"):
                    wrapper._frames.clear()
                    wrapper._frames.extend([frame.copy() for frame in wrapper_state["frames"]])
                if "cached_hidden_render" in wrapper_state and hasattr(wrapper, "_cached_hidden_render"):
                    cached = wrapper_state["cached_hidden_render"]
                    wrapper._cached_hidden_render = None if cached is None else cached.copy()

    def _proprio_observation(self) -> np.ndarray:
        base_env, _ = self._base_and_point_env()
        point_obs, _ = base_env.point_env._get_obs()
        raw_obs = base_env._get_obs(point_obs)
        xy_obs_fn = self._env_method("_xy_observation")
        if callable(xy_obs_fn):
            return np.asarray(xy_obs_fn(raw_obs), dtype=np.float32)

        process_fn = self._env_method("_process_proprio_obs")
        if callable(process_fn):
            return np.asarray(process_fn(raw_obs), dtype=np.float32)

        if isinstance(raw_obs, dict):
            arrays = [
                np.asarray(value, dtype=np.float32).reshape(-1)
                for value in raw_obs.values()
                if not isinstance(value, str)
            ]
            return np.concatenate(arrays, dtype=np.float32)
        return np.asarray(raw_obs, dtype=np.float32)

    def _prepare_rendered_image(self, agent, image: np.ndarray, render_resolution: int) -> np.ndarray:
        image = np.asarray(image, dtype=np.uint8)
        if agent.grayscale:
            if image.ndim == 3 and image.shape[2] == 1:
                image = image[..., 0]
            elif image.ndim == 3:
                image = np.asarray(Image.fromarray(image).convert("L"))
            elif image.ndim != 2:
                raise ValueError(f"Expected grayscale image to be 2D or HWC, got shape {image.shape}")
        elif image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)

        if image.shape[:2] != (render_resolution, render_resolution):
            image = np.asarray(
                Image.fromarray(image).resize((render_resolution, render_resolution), Image.LANCZOS)
            )

        if agent.grayscale and image.ndim == 2:
            image = image[..., None]
        elif not agent.grayscale and image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)

        if image.ndim != 3 or image.shape[2] != agent.image_channels:
            raise ValueError(f"Expected image shape [H, W, {agent.image_channels}], got {image.shape}")
        return image

    def _observation(self, agent) -> np.ndarray:
        if agent.obs_type != "pixels":
            return self._proprio_observation().reshape(agent.obs_shape)

        render_fn = self._env_method("render_observation") or self._env_method("render")
        if not callable(render_fn):
            raise RuntimeError("PointMaze pixel Nyström grid requires render_observation() or render().")

        render_resolution = getattr(self.wrapped_env, "render_resolution", agent.obs_shape[-1])
        frame_stack = agent.obs_shape[0] // agent.image_channels
        image = self._prepare_rendered_image(agent, render_fn(), render_resolution)
        image_chw = image.transpose(2, 0, 1).copy()
        return np.tile(image_chw, (frame_stack, 1, 1))

    def _observation_from_xy(self, agent, xy) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float32).reshape(-1)[:2]
        if agent.obs_type != "pixels":
            return xy.reshape(agent.obs_shape)

        render_from_position = self._env_method("render_from_position")
        if not callable(render_from_position):
            raise RuntimeError(
                "Fixed continuous Nyström dataset requires render_from_position(position, show_goal=False)."
            )

        try:
            image = render_from_position(xy, show_goal=False)
        except TypeError:
            image = render_from_position(xy)

        render_resolution = getattr(self.wrapped_env, "render_resolution", agent.obs_shape[-1])
        frame_stack = agent.obs_shape[0] // agent.image_channels
        image = self._prepare_rendered_image(agent, image, render_resolution)
        image_chw = image.transpose(2, 0, 1).copy()
        return np.tile(image_chw, (frame_stack, 1, 1))

    def observation_from_xy(self, agent, xy) -> np.ndarray:
        """Return the full agent observation for a PointMaze XY probe."""
        xy = np.asarray(xy, dtype=np.float32).reshape(-1)[:2]
        if not self._has_point_env():
            return self._observation_from_xy(agent, xy)

        with self._preserve_state():
            self._set_state(xy)
            return self._observation(agent)

    def maze_layout(self):
        layout_fn = self._env_method("get_debug_maze_layout")
        layout = layout_fn() if callable(layout_fn) else None
        layout = layout if isinstance(layout, dict) else self._layout_from_unwrapped_maze()
        if not isinstance(layout, dict):
            raise RuntimeError(
                "PointMaze Nyström grid requires get_debug_maze_layout() or maze.maze_map."
            )

        maze_lower = np.asarray(layout.get("maze_lower"), dtype=np.float32).reshape(-1)
        maze_upper = np.asarray(layout.get("maze_upper"), dtype=np.float32).reshape(-1)
        wall_rectangles = np.asarray(layout.get("wall_rectangles"), dtype=np.float32).reshape(-1, 4)
        walkable_rectangles = np.asarray(layout.get("walkable_rectangles", []), dtype=np.float32).reshape(-1, 4)
        if maze_lower.size != 2 or maze_upper.size != 2:
            raise RuntimeError("PointMaze layout must provide 2D maze_lower and maze_upper bounds.")
        return {
            "maze_lower": maze_lower[:2],
            "maze_upper": maze_upper[:2],
            "wall_rectangles": wall_rectangles,
            "walkable_rectangles": walkable_rectangles,
        }

    def _layout_from_unwrapped_maze(self):
        base_env = getattr(self.wrapped_env, "unwrapped", None)
        maze = getattr(base_env, "maze", None)
        if maze is None or not hasattr(maze, "maze_map") or not hasattr(maze, "cell_rowcol_to_xy"):
            return None

        half_cell = 0.5 * float(getattr(maze, "maze_size_scaling", 1.0))
        all_rectangles, wall_rectangles, walkable_rectangles = [], [], []
        for row_idx, row in enumerate(maze.maze_map):
            for col_idx, cell in enumerate(row):
                center = maze.cell_rowcol_to_xy(np.array([row_idx, col_idx], dtype=np.int32))
                rect = np.array(
                    [center[0] - half_cell, center[1] - half_cell, 2.0 * half_cell, 2.0 * half_cell],
                    dtype=np.float32,
                )
                all_rectangles.append(rect)
                if cell == 1:
                    wall_rectangles.append(rect)
                else:
                    walkable_rectangles.append(rect)

        if not all_rectangles:
            return None
        all_rectangles = np.asarray(all_rectangles, dtype=np.float32)
        return {
            "maze_lower": all_rectangles[:, :2].min(axis=0),
            "maze_upper": (all_rectangles[:, :2] + all_rectangles[:, 2:4]).max(axis=0),
            "wall_rectangles": np.asarray(wall_rectangles, dtype=np.float32).reshape(-1, 4),
            "walkable_rectangles": np.asarray(walkable_rectangles, dtype=np.float32).reshape(-1, 4),
        }

    @staticmethod
    def _points_outside_walls(points: np.ndarray, wall_rectangles: np.ndarray, margin: float) -> np.ndarray:
        if wall_rectangles.size == 0:
            return np.ones(points.shape[0], dtype=bool)
        wall_lower = wall_rectangles[:, :2] - margin
        wall_upper = wall_rectangles[:, :2] + wall_rectangles[:, 2:4] + margin
        in_wall = ((points[:, None, :] > wall_lower) & (points[:, None, :] < wall_upper)).all(axis=2).any(axis=1)
        return ~in_wall

    @staticmethod
    def _points_inside_rectangles(points: np.ndarray, rectangles: np.ndarray, margin: float) -> np.ndarray:
        if rectangles.size == 0:
            return np.ones(points.shape[0], dtype=bool)
        rect_lower = rectangles[:, :2] + margin
        rect_upper = rectangles[:, :2] + rectangles[:, 2:4] - margin
        valid_rectangles = np.all(rect_upper >= rect_lower, axis=1)
        if not np.any(valid_rectangles):
            rect_lower = rectangles[:, :2]
            rect_upper = rectangles[:, :2] + rectangles[:, 2:4]
        else:
            rect_lower = rect_lower[valid_rectangles]
            rect_upper = rect_upper[valid_rectangles]
        return ((points[:, None, :] >= rect_lower) & (points[:, None, :] <= rect_upper)).all(axis=2).any(axis=1)

    def _feasible_points_mask(
            self,
            points: np.ndarray,
            wall_rectangles: np.ndarray,
            walkable_rectangles: np.ndarray,
            margin: float,
        ) -> np.ndarray:
        inside_walkable = self._points_inside_rectangles(points, walkable_rectangles, margin)
        outside_walls = self._points_outside_walls(points, wall_rectangles, margin)
        return inside_walkable & outside_walls

    @staticmethod
    def _xy_grid(lower: np.ndarray, upper: np.ndarray, n_x: int, n_y: int) -> np.ndarray:
        xs = np.linspace(lower[0], upper[0], n_x, dtype=np.float32)
        ys = np.linspace(lower[1], upper[1], n_y, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        return np.column_stack([grid_x.ravel(), grid_y.ravel()])

    @staticmethod
    def _anchored_equispaced_grid(lower: np.ndarray, upper: np.ndarray, spacing: float, anchor: np.ndarray) -> np.ndarray:
        spacing = float(spacing)
        if spacing <= 0.0:
            raise ValueError(f"PointMaze grid spacing must be positive, got {spacing}")
        anchor = np.asarray(anchor, dtype=np.float32).reshape(-1)[:2]
        lower = np.asarray(lower, dtype=np.float32).reshape(-1)[:2]
        upper = np.asarray(upper, dtype=np.float32).reshape(-1)[:2]

        x_min = int(np.ceil((lower[0] - anchor[0]) / spacing))
        x_max = int(np.floor((upper[0] - anchor[0]) / spacing))
        y_min = int(np.ceil((lower[1] - anchor[1]) / spacing))
        y_max = int(np.floor((upper[1] - anchor[1]) / spacing))
        xs = anchor[0] + spacing * np.arange(x_min, x_max + 1, dtype=np.float32)
        ys = anchor[1] + spacing * np.arange(y_min, y_max + 1, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
        return np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32, copy=False)

    @staticmethod
    def _nearest_neighbor_distances(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if points.shape[0] < 2:
            return np.empty((0,), dtype=np.float32)
        deltas = points[:, None, :] - points[None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        np.fill_diagonal(distances, np.inf)
        nearest = distances.min(axis=1)
        return nearest[np.isfinite(nearest)].astype(np.float32, copy=False)

    @staticmethod
    def _spacing_summary(points: np.ndarray) -> Dict[str, float]:
        nearest = PointMazeNystromDebugHelper._nearest_neighbor_distances(points)
        if nearest.size == 0:
            return {"min": float("nan"), "median": float("nan"), "max": float("nan")}
        return {
            "min": float(np.min(nearest)),
            "median": float(np.median(nearest)),
            "max": float(np.max(nearest)),
        }

    @staticmethod
    def _regular_grid_spacing(lower: np.ndarray, upper: np.ndarray, n_x: int, n_y: int) -> Dict[str, float]:
        spacing_x = float((upper[0] - lower[0]) / max(n_x - 1, 1))
        spacing_y = float((upper[1] - lower[1]) / max(n_y - 1, 1))
        return {"x": spacing_x, "y": spacing_y}

    @staticmethod
    def _walkable_area(rectangles: np.ndarray, margin: float) -> float:
        if rectangles.size == 0:
            return 0.0
        rectangles = np.asarray(rectangles, dtype=np.float32).reshape(-1, 4)
        widths = np.maximum(rectangles[:, 2] - 2.0 * margin, 0.0)
        heights = np.maximum(rectangles[:, 3] - 2.0 * margin, 0.0)
        return float(np.sum(widths * heights))

    @staticmethod
    def _exact_grid_shape(n_points: int, lower: np.ndarray, upper: np.ndarray) -> Tuple[int, int]:
        span = np.maximum(np.asarray(upper, dtype=np.float32) - np.asarray(lower, dtype=np.float32), 1e-6)
        aspect = float(span[0] / span[1])
        n_x = max(2, int(round(np.sqrt(max(n_points, 1) * aspect))))
        n_y = max(2, int(round(n_points / n_x)))
        candidates = []
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                cand_x = max(2, n_x + dx)
                cand_y = max(2, n_y + dy)
                product = cand_x * cand_y
                spacing = PointMazeNystromDebugHelper._regular_grid_spacing(lower, upper, cand_x, cand_y)
                mean_spacing = 0.5 * (spacing["x"] + spacing["y"])
                spacing_mismatch = abs(spacing["x"] - spacing["y"]) / max(mean_spacing, 1e-12)
                candidates.append((abs(product - n_points), spacing_mismatch, product, cand_x, cand_y))
        _, _, _, best_x, best_y = min(candidates)
        return int(best_x), int(best_y)

    def _exact_feasible_grid_candidates(
            self,
            lower: np.ndarray,
            upper: np.ndarray,
            wall_rectangles: np.ndarray,
            walkable_rectangles: np.ndarray,
            margin: float,
            n_points: int,
            anchor: Optional[np.ndarray] = None,
        ) -> Tuple[np.ndarray, Dict[str, float]]:
        if anchor is None:
            anchor = lower
        area = self._walkable_area(walkable_rectangles, margin)
        if area <= 0.0:
            span = np.maximum(upper - lower, 1e-6)
            area = float(span[0] * span[1])
        base_spacing = float(np.sqrt(area / max(n_points, 1)))
        best = None
        # Search one scalar spacing. This keeps dx == dy; count may adjust if maze holes make exact count impossible.
        for radius in (0.35, 0.6, 0.9):
            for factor in np.linspace(max(0.05, 1.0 - radius), 1.0 + radius, 241):
                spacing_value = base_spacing * float(factor)
                candidates = self._anchored_equispaced_grid(lower, upper, spacing_value, anchor)
                valid_points = candidates[
                    self._feasible_points_mask(candidates, wall_rectangles, walkable_rectangles, margin)
                ]
                valid_count = valid_points.shape[0]
                if valid_count == 0:
                    continue
                unique_x = np.unique(np.round(candidates[:, 0], decimals=6)).size
                unique_y = np.unique(np.round(candidates[:, 1], decimals=6)).size
                score = (abs(valid_count - n_points), abs(float(factor) - 1.0), candidates.shape[0])
                if best is None or score < best[0]:
                    best = (score, valid_points, spacing_value, unique_x, unique_y, valid_count)
            if best is not None and best[0][0] == 0:
                break

        if best is None:
            raise RuntimeError("Could not build any feasible PointMaze exact-grid points.")

        _, valid_points, spacing_value, n_x, n_y, valid_count = best
        grid_spacing = {
            "x": float(spacing_value),
            "y": float(spacing_value),
            "n_x": int(n_x),
            "n_y": int(n_y),
            "requested_points": int(n_points),
            "adjusted_points": int(valid_count),
            "exact_grid": True,
            "equispaced": True,
            "feasible_points": int(valid_count),
        }
        return valid_points.astype(np.float32, copy=False), grid_spacing

    @staticmethod
    def _put_start_first(points: np.ndarray, start_xy: Optional[np.ndarray]) -> np.ndarray:
        if start_xy is None or points.shape[0] == 0:
            return points
        start_xy = np.asarray(start_xy, dtype=np.float32).reshape(1, 2)
        selected = points.astype(np.float32, copy=True)
        matches = np.where(np.all(np.isclose(selected, start_xy, atol=1e-6), axis=1))[0]
        if matches.size > 0:
            start_idx = int(matches[0])
            selected[[0, start_idx]] = selected[[start_idx, 0]]
        else:
            selected[0] = start_xy[0]
        return selected

    @staticmethod
    def _put_existing_start_first(points: np.ndarray, start_xy: Optional[np.ndarray]) -> np.ndarray:
        if start_xy is None or points.shape[0] == 0:
            return points
        start_xy = np.asarray(start_xy, dtype=np.float32).reshape(1, 2)
        selected = points.astype(np.float32, copy=True)
        matches = np.where(np.all(np.isclose(selected, start_xy, atol=1e-5), axis=1))[0]
        if matches.size > 0:
            start_idx = int(matches[0])
            selected[[0, start_idx]] = selected[[start_idx, 0]]
        return selected

    @staticmethod
    def _thin_regular_grid_points(valid_points: np.ndarray, n_points: int) -> np.ndarray:
        if valid_points.shape[0] <= n_points:
            return valid_points[:n_points].astype(np.float32, copy=False)
        indices = np.round(np.linspace(0, valid_points.shape[0] - 1, n_points)).astype(np.int64)
        return valid_points[indices].astype(np.float32, copy=False)

    def _regular_grid_candidates(
            self,
            lower: np.ndarray,
            upper: np.ndarray,
            wall_rectangles: np.ndarray,
            walkable_rectangles: np.ndarray,
            margin: float,
            n_points: int,
        ) -> Tuple[np.ndarray, Dict[str, float]]:
        span = np.maximum(upper - lower, 1e-6)
        base_n_x = max(2, int(np.ceil(np.sqrt(n_points * (span[0] / span[1]) * self.oversample))))
        base_n_y = max(2, int(np.ceil(n_points * self.oversample / base_n_x)))

        best = None
        for radius in (12, 28, 56):
            for n_x in range(max(2, base_n_x - radius), base_n_x + radius + 1):
                for n_y in range(max(2, base_n_y - radius), base_n_y + radius + 1):
                    candidates = self._xy_grid(lower, upper, n_x, n_y)
                    valid_points = candidates[
                        self._feasible_points_mask(candidates, wall_rectangles, walkable_rectangles, margin)
                    ]
                    valid_count = valid_points.shape[0]
                    if valid_count < n_points:
                        continue
                    spacing = self._regular_grid_spacing(lower, upper, n_x, n_y)
                    mean_spacing = 0.5 * (spacing["x"] + spacing["y"])
                    spacing_mismatch = abs(spacing["x"] - spacing["y"]) / max(mean_spacing, 1e-12)
                    extra = valid_count - n_points
                    score = (spacing_mismatch, extra, n_x * n_y)
                    if best is None or score < best[0]:
                        best = (score, valid_points, spacing, n_x, n_y, valid_count)
            if best is not None:
                break

        if best is None:
            n_x, n_y = base_n_x, base_n_y
            valid_points = np.empty((0, 2), dtype=np.float32)
            for _ in range(8):
                candidates = self._xy_grid(lower, upper, n_x, n_y)
                valid_points = candidates[
                    self._feasible_points_mask(candidates, wall_rectangles, walkable_rectangles, margin)
                ]
                if valid_points.shape[0] >= n_points:
                    break
                n_x, n_y = int(np.ceil(n_x * 1.25)) + 1, int(np.ceil(n_y * 1.25)) + 1

            if valid_points.shape[0] < n_points:
                raise RuntimeError(
                    f"Could only place {valid_points.shape[0]} reachable PointMaze grid points; "
                    f"requested {n_points}. Try reducing nystrom_grid_border_margin."
                )
            spacing = self._regular_grid_spacing(lower, upper, n_x, n_y)
            return valid_points, {"x": spacing["x"], "y": spacing["y"], "n_x": int(n_x), "n_y": int(n_y)}

        _, valid_points, spacing, n_x, n_y, _ = best
        return valid_points, {"x": spacing["x"], "y": spacing["y"], "n_x": int(n_x), "n_y": int(n_y)}

    def _fixed_start_xy(self):
        debug_fn = self._env_method("get_debug_coordinates")
        debug_info = debug_fn() if callable(debug_fn) else {}
        start = debug_info.get("fixed_start") if isinstance(debug_info, dict) else None
        return None if start is None else np.asarray(start, dtype=np.float32).reshape(-1)[:2]

    def build_grid_points(self, n_points: int) -> np.ndarray:
        if n_points <= 0:
            raise ValueError("Nyström PointMaze grid requires a positive number of points.")

        layout = self.maze_layout()
        raw_lower = layout["maze_lower"].astype(np.float32, copy=False)
        raw_upper = layout["maze_upper"].astype(np.float32, copy=False)
        raw_span = raw_upper - raw_lower
        if self.exact_grid and abs(float(raw_span[1])) > 1e-6:
            exact_margin = 0.0
            start_xy = self._fixed_start_xy()
            if start_xy is not None and not self._points_outside_walls(start_xy[None, :], layout["wall_rectangles"], margin=0.0)[0]:
                start_xy = None
            selected, self._last_grid_spacing = self._exact_feasible_grid_candidates(
                raw_lower,
                raw_upper,
                layout["wall_rectangles"],
                layout["walkable_rectangles"],
                exact_margin,
                n_points,
                anchor=start_xy,
            )
            selected = self._put_existing_start_first(selected, start_xy)
            if selected.shape[0] != n_points:
                ColorPrint.yellow(
                    f"Exact PointMaze grid adjusted states from {n_points} to {selected.shape[0]} "
                    f"({self._last_grid_spacing['n_x']} x {self._last_grid_spacing['n_y']} equispaced lattice, "
                    f"{selected.shape[0]} feasible)."
                )
            return selected.astype(np.float32, copy=False)

        if abs(float(raw_span[1])) <= 1e-6:
            margin = 0.0 if self.exact_grid else max(self.border_margin, 0.0)
            x_lower = float(raw_lower[0] + margin)
            x_upper = float(raw_upper[0] - margin)
            if x_upper <= x_lower:
                x_lower, x_upper = float(raw_lower[0]), float(raw_upper[0])
            start_xy = self._fixed_start_xy()
            if start_xy is not None and n_points > 1:
                xs = np.linspace(x_lower, x_upper, n_points - 1, dtype=np.float32)
                ordered_points = np.column_stack([xs, np.full(n_points - 1, raw_lower[1], dtype=np.float32)])
                selected = np.concatenate([start_xy.reshape(1, 2).astype(np.float32), ordered_points], axis=0)
                return selected.astype(np.float32, copy=False)

            xs = np.linspace(x_lower, x_upper, n_points, dtype=np.float32)
            selected = np.column_stack([xs, np.full(n_points, raw_lower[1], dtype=np.float32)])
            return selected.astype(np.float32, copy=False)

        margin = max(self.border_margin, 0.0)
        lower, upper = layout["maze_lower"] + margin, layout["maze_upper"] - margin
        if np.any(upper <= lower):
            lower, upper, margin = layout["maze_lower"], layout["maze_upper"], 0.0

        valid_points, grid_spacing = self._regular_grid_candidates(
            lower,
            upper,
            layout["wall_rectangles"],
            layout["walkable_rectangles"],
            margin,
            n_points,
        )

        start_xy = self._fixed_start_xy()
        if start_xy is not None and not self._points_outside_walls(start_xy[None, :], layout["wall_rectangles"], margin)[0]:
            start_xy = None
        selected = self._thin_regular_grid_points(valid_points, n_points)
        if start_xy is not None:
            selected[0] = start_xy
        self._last_grid_spacing = grid_spacing
        return selected.astype(np.float32, copy=False)

    def build_policy_grid_points(self, n_points: int) -> np.ndarray:
        """Return ordered, equally spaced feasible probes plus exact start."""
        if n_points <= 0:
            raise ValueError("PointMaze policy grid requires a positive number of points.")

        layout = self.maze_layout()
        margin = max(self.border_margin, 0.0)
        lower = layout["maze_lower"].astype(np.float32, copy=False) + margin
        upper = layout["maze_upper"].astype(np.float32, copy=False) - margin
        if np.any(upper <= lower):
            lower = layout["maze_lower"].astype(np.float32, copy=False)
            upper = layout["maze_upper"].astype(np.float32, copy=False)
            margin = 0.0

        points, _ = self._exact_feasible_grid_candidates(
            lower,
            upper,
            layout["wall_rectangles"],
            layout["walkable_rectangles"],
            margin,
            n_points,
            anchor=lower,
        )
        order = np.lexsort((points[:, 0], points[:, 1]))
        points = points[order]

        start = self._fixed_start_xy()
        if start is not None:
            start = np.asarray(start, dtype=np.float32).reshape(1, 2)
            matches = np.all(np.isclose(points, start, atol=1e-5), axis=1)
            points = np.concatenate([points[~matches], start], axis=0)
        return points.astype(np.float32, copy=False)

    def _landmark_transition(self, agent, xy, action_idx):
        if not self._has_point_env():
            step_from_position = self._env_method("step_from_position")
            if not callable(step_from_position):
                raise RuntimeError(
                    "Fixed continuous Nyström dataset requires step_from_position(position, action)."
                )

            obs = self._observation_from_xy(agent, xy)
            next_position, reward_value, terminated, truncated, _ = step_from_position(
                np.asarray(xy, dtype=np.float32),
                int(action_idx),
            )
            next_obs = self._observation_from_xy(agent, next_position)
            reward = [float(reward_value)]
            discount = [0.0 if bool(terminated or truncated) else 1.0]
            return obs, next_obs, reward, discount, np.asarray(next_position, dtype=np.float32).reshape(-1)[:2]

        self._set_state(xy)
        obs = self._observation(agent)
        time_step = self.wrapped_env.step(int(action_idx))
        next_obs = self._observation(agent)
        next_xy = self._proprio_observation().reshape(-1)[:2]
        reward = [float(getattr(time_step, "reward", 0.0))]
        discount = [float(getattr(time_step, "discount", 1.0))]
        return obs, next_obs, reward, discount, np.asarray(next_xy, dtype=np.float32)

    def _synthetic_transition_to_xy(self, agent, source_xy, target_xy):
        source_xy = np.asarray(source_xy, dtype=np.float32).reshape(-1)[:2]
        target_xy = np.asarray(target_xy, dtype=np.float32).reshape(-1)[:2]
        obs = self.observation_from_xy(agent, source_xy)
        next_obs = self.observation_from_xy(agent, target_xy)
        return obs, next_obs, [0.0], [1.0], target_xy.astype(np.float32, copy=False)

    @staticmethod
    def _step_summary(source_xy: np.ndarray, next_xy: np.ndarray) -> Dict[str, float]:
        source_xy = np.asarray(source_xy, dtype=np.float32).reshape(-1, 2)
        next_xy = np.asarray(next_xy, dtype=np.float32).reshape(-1, 2)
        distances = np.linalg.norm(next_xy - source_xy, axis=1)
        finite = distances[np.isfinite(distances)]
        if finite.size == 0:
            return {"min": float("nan"), "median": float("nan"), "max": float("nan")}
        return {
            "min": float(np.min(finite)),
            "median": float(np.median(finite)),
            "max": float(np.max(finite)),
        }

    def build_subsample_batch(self, agent, n_transitions: Optional[int] = None):
        n_transitions = int(
            n_transitions
            if n_transitions is not None
            else agent.subsamples if agent.subsamples is not None else agent.batch_size_actor
        )
        requested_transitions = n_transitions
        if n_transitions in self._subsample_batches:
            self._subsample_batch = self._subsample_batches[n_transitions]
            self._fixed_xy_points = self._batch_xy_points.get(n_transitions)
            return self._subsample_batch

        if n_transitions % agent.n_actions != 0 and self.exact_grid:
            n_states = max(1, int(round(n_transitions / agent.n_actions)))
            ColorPrint.yellow(
                f"Exact PointMaze grid adjusted transitions from {n_transitions} "
                f"to {n_states * agent.n_actions} so every state has all actions."
            )
            n_transitions = n_states * agent.n_actions
        elif n_transitions % agent.n_actions != 0:
            raise ValueError(
                f"Fixed continuous debug dataset size={n_transitions} must be divisible by "
                f"n_actions={agent.n_actions} so each sampled state can include all actions. "
                "Set agent.subsamples or agent.batch_size_actor accordingly."
            )

        n_states = n_transitions // agent.n_actions
        state_points = self.build_grid_points(n_states)
        n_states = state_points.shape[0]
        adjusted_transitions = n_states * agent.n_actions
        if adjusted_transitions != n_transitions:
            ColorPrint.yellow(
                f"Exact PointMaze grid adjusted transitions from {n_transitions} "
                f"to {adjusted_transitions}."
            )
            n_transitions = adjusted_transitions
        xy_points = np.repeat(state_points, agent.n_actions, axis=0)
        actions_np = np.tile(np.arange(agent.n_actions, dtype=np.int64), n_states)

        if self._has_point_env():
            with self._preserve_state():
                transitions = [
                    self._landmark_transition(agent, xy, action_idx)
                    for xy, action_idx in zip(xy_points, actions_np)
                ]
        else:
            transitions = [
                self._landmark_transition(agent, xy, action_idx)
                for xy, action_idx in zip(xy_points, actions_np)
            ]
        next_xy_points = np.stack([transition[4] for transition in transitions]).astype(np.float32, copy=False)
        step_source_xy = xy_points[1:] if xy_points.shape[0] > 1 else xy_points
        step_next_xy = next_xy_points[1:] if next_xy_points.shape[0] > 1 else next_xy_points
        if transitions:
            transitions[0] = self._synthetic_transition_to_xy(agent, state_points[0], state_points[0])
        obs_list, next_obs_list, rewards, discounts, _ = zip(*transitions)

        obs = torch.as_tensor(np.stack(obs_list), dtype=torch.float32, device=agent.device)
        next_obs = torch.as_tensor(np.stack(next_obs_list), dtype=torch.float32, device=agent.device)
        action = torch.as_tensor(actions_np, dtype=torch.long, device=agent.device)
        reward = torch.as_tensor(rewards, dtype=agent.compute_dtype, device=agent.device)
        discount = torch.as_tensor(discounts, dtype=agent.compute_dtype, device=agent.device)

        self._subsample_batch = (obs, action, reward, discount, next_obs)
        self._subsample_batches[n_transitions] = self._subsample_batch
        self._subsample_batches[requested_transitions] = self._subsample_batch
        self._batch_xy_points[n_transitions] = state_points
        self._batch_xy_points[requested_transitions] = state_points
        self._fixed_xy_points = state_points
        self._fixed_actions = actions_np
        self._fixed_plot_stats = {
            "border_margin": float(self.border_margin),
            "oversample": float(self.oversample),
            "grid_spacing": getattr(self, "_last_grid_spacing", None),
            "point_spacing": self._spacing_summary(state_points),
            "step_size": self._step_summary(step_source_xy, step_next_xy),
        }
        dataset_name = "Nyström grid" if agent.subsamples is not None else "debug grid"
        ColorPrint.yellow(
            f"Using fixed continuous {dataset_name} with {n_states} reachable XY states "
            f"x {agent.n_actions} actions = {n_transitions} state-action landmarks "
            f"({agent._kernel_status()})."
        )
        self.save_fixed_points_plot(agent.n_actions)
        return self._subsample_batch

    def fixed_actor_batch(self, agent, n_transitions: Optional[int] = None):
        obs, action, reward, _, next_obs = self.build_subsample_batch(agent, n_transitions=n_transitions)
        # Match raw replay's actor boundary: state, action, next state, reward.
        return obs, action, next_obs, reward

    def fixed_xy_points_for_size(self, n_transitions: int):
        return self._batch_xy_points.get(int(n_transitions))

    def fixed_encoder_batch(self, agent):
        actor_batch = self.fixed_actor_batch(agent)
        size = min(int(agent.batch_size), actor_batch[0].shape[0])
        total_size = actor_batch[0].shape[0]
        n_actions = int(getattr(agent, "n_actions", 1))
        if size == total_size:
            index = torch.arange(total_size, device=agent.device)
        elif n_actions > 0 and total_size % n_actions == 0 and size % n_actions == 0:
            n_states = total_size // n_actions
            n_selected_states = size // n_actions
            state_index = torch.round(
                torch.linspace(0, n_states - 1, n_selected_states, device=agent.device)
            ).long()
            action_offsets = torch.arange(n_actions, device=agent.device)
            index = (state_index[:, None] * n_actions + action_offsets[None, :]).reshape(-1)
        else:
            index = torch.round(
                torch.linspace(0, total_size - 1, size, device=agent.device)
            ).long()
        return tuple(field[index] for field in actor_batch)

    def encode_subsamples(self, agent):
        """Encode fixed landmarks through the agent's normal encoder path."""
        agent._sync_policy_encoder()
        encoded = agent.transition_encoder.encode_raw(
            RawTransitions(*self.fixed_actor_batch(agent))
        )
        return encoded.tensors, encoded.reward

    def save_fixed_points_plot(self, n_actions: int):
        if self._fixed_xy_points is None:
            return
        from agent.utils_debug_visualization import PointMazeNystromDebugVisualizer

        PointMazeNystromDebugVisualizer(
            save_dir=os.path.join(os.getcwd(), "pointmaze_plots")
        ).save_fixed_points_plot(
            layout=self.maze_layout(),
            points=self._fixed_xy_points,
            n_actions=n_actions,
            stats=self._fixed_plot_stats,
        )
