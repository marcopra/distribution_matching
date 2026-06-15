from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import patches
import numpy as np
from PIL import Image


def _get_env_id(reference) -> str:
    if isinstance(reference, str):
        return reference

    env = getattr(reference, "unwrapped", reference)
    spec = getattr(env, "spec", None)
    env_id = getattr(spec, "id", None)
    if env_id is not None:
        return env_id
    return env.__class__.__name__


def _get_env_module(reference) -> str:
    if isinstance(reference, str):
        return ""

    env = getattr(reference, "unwrapped", reference)
    return getattr(env.__class__, "__module__", "")


def _is_fetch_env(reference) -> bool:
    env_id = _get_env_id(reference).lower()
    module_name = _get_env_module(reference).lower()
    return "fetch" in env_id or "fetch" in module_name


def _is_point_maze_env(reference) -> bool:
    env_id = _get_env_id(reference).lower()
    module_name = _get_env_module(reference).lower()
    return "pointmaze" in env_id or "point_maze" in module_name


def _find_discrete_env(reference):
    current = reference
    while current is not None:
        if all(hasattr(current, attr) for attr in ("n_states", "idx_to_state", "state_to_idx")):
            return current

        if hasattr(current, "env"):
            current = current.env
        elif hasattr(current, "unwrapped") and current.unwrapped is not current:
            current = current.unwrapped
        else:
            break
    return None


def _get_env_method(env, method_name: str):
    current = env
    visited = set()

    while current is not None and id(current) not in visited:
        visited.add(id(current))
        method = getattr(current, method_name, None)
        if callable(method):
            return method
        current = getattr(current, "env", None)

    return None


def _has_debug_xy_env(reference) -> bool:
    return callable(_get_env_method(reference, "get_debug_coordinates"))


def _as_xy(value) -> Optional[np.ndarray]:
    if value is None:
        return None

    xy = np.asarray(value, dtype=np.float32).reshape(-1)
    if xy.size < 2 or not np.all(np.isfinite(xy[:2])):
        return None
    return xy[:2]


def extract_eval_trajectory_point(env, time_step) -> Optional[np.ndarray]:
    """Extract an x/y point from an evaluation time step when available."""
    info = getattr(time_step, "info", None)
    if isinstance(info, dict):
        for key in ("agent_position", "position", "xy"):
            xy = _as_xy(info.get(key))
            if xy is not None:
                return xy

    method = _get_env_method(env, "get_debug_coordinates")
    if callable(method):
        debug_info = method()
        if isinstance(debug_info, dict):
            for key in ("xy", "xyz", "agent_position", "position"):
                xy = _as_xy(debug_info.get(key))
                if xy is not None:
                    return xy

    discrete_env = _find_discrete_env(env)
    raw_proprio = getattr(time_step, "proprio_observation", [])
    proprio_array = np.asarray(raw_proprio, dtype=np.float32)
    proprio = proprio_array.reshape(-1)
    if discrete_env is not None and proprio.size == getattr(discrete_env, "n_states", -1):
        state_idx = int(np.argmax(proprio))
        return _as_xy(discrete_env.idx_to_state.get(state_idx))

    if proprio_array.ndim <= 1:
        return _as_xy(proprio)
    return None


def _prepare_trajectories(trajectories) -> list[np.ndarray]:
    prepared = []
    for trajectory in trajectories:
        if trajectory is None:
            continue
        points = [_as_xy(point) for point in trajectory]
        points = [point for point in points if point is not None]
        if points:
            prepared.append(np.asarray(points, dtype=np.float32))
    return prepared


def _trajectory_colors(n_trajectories: int) -> list:
    if n_trajectories <= 10:
        cmap = plt.get_cmap("tab10")
        return [cmap(i) for i in range(n_trajectories)]
    if n_trajectories <= 20:
        cmap = plt.get_cmap("tab20")
        return [cmap(i) for i in range(n_trajectories)]
    cmap = plt.get_cmap("turbo")
    return [cmap(i / max(n_trajectories - 1, 1)) for i in range(n_trajectories)]


def _get_discrete_plot_cells(env) -> Optional[list[tuple[int, int]]]:
    discrete_env = _find_discrete_env(env)
    if discrete_env is None:
        return None

    cells = []
    dead_state = getattr(discrete_env, "DEAD_STATE", None)
    for state in getattr(discrete_env, "cells", []):
        if dead_state is not None and state == dead_state:
            continue
        xy = _as_xy(state)
        if xy is not None:
            cells.append((int(xy[0]), int(xy[1])))
    return cells or None


def _plot_bounds(env, trajectories: list[np.ndarray]) -> tuple[float, float, float, float]:
    cells = _get_discrete_plot_cells(env)
    if cells is not None:
        xs = [cell[0] for cell in cells]
        ys = [cell[1] for cell in cells]
        return min(xs) - 0.5, max(xs) + 0.5, min(ys) - 0.5, max(ys) + 0.5

    all_points = np.concatenate(trajectories, axis=0)
    min_x, min_y = all_points.min(axis=0)
    max_x, max_y = all_points.max(axis=0)
    span_x = max(max_x - min_x, 1e-3)
    span_y = max(max_y - min_y, 1e-3)
    margin_x = max(0.05 * span_x, 1e-3)
    margin_y = max(0.05 * span_y, 1e-3)
    return min_x - margin_x, max_x + margin_x, min_y - margin_y, max_y + margin_y


def _draw_discrete_background(ax, env) -> None:
    cells = _get_discrete_plot_cells(env)
    if cells is None:
        return

    for x, y in cells:
        ax.add_patch(
            patches.Rectangle(
                (x - 0.5, y - 0.5),
                1.0,
                1.0,
                facecolor="#f7f7f7",
                edgecolor="#d9d9d9",
                linewidth=0.35,
                zorder=0,
            )
        )


def _style_trajectory_axis(ax, env, trajectories: list[np.ndarray], title: str) -> None:
    min_x, max_x, min_y, max_y = _plot_bounds(env, trajectories)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    if _get_discrete_plot_cells(env) is not None:
        ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title, pad=4)
    ax.tick_params(direction="out", length=3, width=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _draw_start_goal_markers(ax, env, trajectories: list[np.ndarray]) -> None:
    if trajectories:
        starts = np.asarray([trajectory[0] for trajectory in trajectories], dtype=np.float32)
        ends = np.asarray([trajectory[-1] for trajectory in trajectories], dtype=np.float32)
        ax.scatter(
            starts[:, 0],
            starts[:, 1],
            marker="o",
            s=22,
            facecolors="white",
            edgecolors="black",
            linewidths=0.7,
            zorder=7,
            label="start",
        )
        ax.scatter(
            ends[:, 0],
            ends[:, 1],
            marker="x",
            s=28,
            c="black",
            linewidths=0.9,
            zorder=8,
            label="end",
        )

    goal = None
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        goal = getattr(current, "goal_position", None)
        if goal is not None:
            break
        current = getattr(current, "env", None)
    goal_xy = _as_xy(goal)
    # if goal_xy is not None:
    #     ax.scatter(
    #         goal_xy[0],
    #         goal_xy[1],
    #         marker="*",
    #         s=90,
    #         facecolors="#ffd92f",
    #         edgecolors="black",
    #         linewidths=0.6,
    #         zorder=9,
    #         label="goal",
    #     )


def _draw_colored_trajectories(ax, trajectories: list[np.ndarray], alpha: float = 0.92) -> None:
    colors = _trajectory_colors(len(trajectories))
    for idx, trajectory in enumerate(trajectories):
        color = colors[idx]
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=color,
            linewidth=1.35,
            alpha=alpha,
            solid_capstyle="round",
            zorder=4,
        )
        ax.scatter(
            trajectory[:, 0],
            trajectory[:, 1],
            s=7,
            color=color,
            alpha=min(alpha + 0.05, 1.0),
            linewidths=0,
            zorder=5,
        )


def _draw_visit_heatmap(fig, ax, env, trajectories: list[np.ndarray], cmap: str = "magma"):
    all_points = np.concatenate(trajectories, axis=0)
    cells = _get_discrete_plot_cells(env)
    if cells is not None:
        xs = [cell[0] for cell in cells]
        ys = [cell[1] for cell in cells]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        heatmap = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=np.float32)
        valid = set(cells)
        for point in all_points:
            cell = (int(round(point[0])), int(round(point[1])))
            if cell in valid:
                heatmap[cell[1] - min_y, cell[0] - min_x] += 1
        heatmap = np.ma.masked_where(heatmap <= 0, heatmap)
        im = ax.imshow(
            heatmap,
            extent=[min_x - 0.5, max_x + 0.5, max_y + 0.5, min_y - 0.5],
            cmap=cmap,
            norm=mcolors.LogNorm(vmin=1, vmax=max(float(heatmap.max()), 1.0)),
            interpolation="nearest",
            zorder=1,
        )
    else:
        min_x, max_x, min_y, max_y = _plot_bounds(env, trajectories)
        heatmap, xedges, yedges = np.histogram2d(
            all_points[:, 0],
            all_points[:, 1],
            bins=48,
            range=[[min_x, max_x], [min_y, max_y]],
        )
        heatmap = np.ma.masked_where(heatmap.T <= 0, heatmap.T)
        im = ax.imshow(
            heatmap,
            origin="lower",
            extent=[min_x, max_x, min_y, max_y],
            cmap=cmap,
            norm=mcolors.LogNorm(vmin=1, vmax=max(float(heatmap.max()), 1.0)),
            interpolation="nearest",
            zorder=1,
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cbar.set_label("visit count")
    return im


def _plot_colored_overlay(fig, ax, env, trajectories: list[np.ndarray]) -> None:
    _draw_discrete_background(ax, env)
    _draw_colored_trajectories(ax, trajectories)
    _draw_start_goal_markers(ax, env, trajectories)
    _style_trajectory_axis(ax, env, trajectories, "Colored trajectories")


def _plot_heatmap_overlay(fig, ax, env, trajectories: list[np.ndarray]) -> None:
    _draw_discrete_background(ax, env)
    _draw_visit_heatmap(fig, ax, env, trajectories, cmap="magma")
    _draw_colored_trajectories(ax, trajectories, alpha=0.72)
    _draw_start_goal_markers(ax, env, trajectories)
    _style_trajectory_axis(ax, env, trajectories, "Log visit heatmap + trajectories")


def _plot_occupancy_only(fig, ax, env, trajectories: list[np.ndarray]) -> None:
    _draw_discrete_background(ax, env)
    _draw_visit_heatmap(fig, ax, env, trajectories, cmap="viridis")
    _draw_start_goal_markers(ax, env, trajectories)
    _style_trajectory_axis(ax, env, trajectories, "Aggregated visitation")


def _plot_endpoint_summary(fig, ax, env, trajectories: list[np.ndarray]) -> None:
    _draw_discrete_background(ax, env)
    _draw_colored_trajectories(ax, trajectories, alpha=0.22)
    colors = _trajectory_colors(len(trajectories))
    for idx, trajectory in enumerate(trajectories):
        end = trajectory[-1]
        ax.scatter(
            end[0],
            end[1],
            s=36,
            color=colors[idx],
            edgecolors="black",
            linewidths=0.45,
            zorder=8,
        )
    _draw_start_goal_markers(ax, env, trajectories)
    _style_trajectory_axis(ax, env, trajectories, "Endpoints with faint paths")


def _save_single_eval_trajectory_style(
    trajectories: list[np.ndarray],
    env,
    save_path: Path,
    style: str,
    step: int,
) -> None:
    plotters = {
        "colored": _plot_colored_overlay,
        "heatmap_overlay": _plot_heatmap_overlay,
        "occupancy": _plot_occupancy_only,
        "endpoints": _plot_endpoint_summary,
    }
    plotter = plotters.get(style)
    if plotter is None:
        return

    with plt.rc_context(_paper_trajectory_rc()):
        fig, ax = plt.subplots(figsize=(3.25, 3.05), constrained_layout=True)
        plotter(fig, ax, env, trajectories)
        ax.text(
            0.01,
            0.99,
            f"step {step}, n={len(trajectories)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.5),
            zorder=10,
        )
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles))
            ax.legend(
                unique.values(),
                unique.keys(),
                loc="lower right",
                frameon=True,
                framealpha=0.88,
                fontsize=6,
                borderpad=0.25,
                handlelength=1.0,
            )
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _save_eval_trajectory_suite(
    trajectories: list[np.ndarray],
    env,
    save_path: Path,
    step: int,
) -> None:
    with plt.rc_context(_paper_trajectory_rc()):
        fig, axes = plt.subplots(2, 2, figsize=(6.9, 6.3), constrained_layout=True)
        plotters = (
            _plot_colored_overlay,
            _plot_heatmap_overlay,
            _plot_occupancy_only,
            _plot_endpoint_summary,
        )
        for ax, plotter in zip(axes.flat, plotters):
            plotter(fig, ax, env, trajectories)
        fig.suptitle(f"Evaluation trajectories at step {step} (n={len(trajectories)})", fontsize=9)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _save_eval_trajectory_small_multiples(
    trajectories: list[np.ndarray],
    env,
    save_path: Path,
    step: int,
    max_episodes: int = 16,
) -> None:
    shown = trajectories[:max_episodes]
    if not shown:
        return

    ncols = min(4, len(shown))
    nrows = int(np.ceil(len(shown) / ncols))
    colors = _trajectory_colors(len(shown))
    with plt.rc_context(_paper_trajectory_rc()):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(1.75 * ncols, 1.65 * nrows),
            squeeze=False,
            constrained_layout=True,
        )
        for idx, ax in enumerate(axes.flat):
            if idx >= len(shown):
                ax.axis("off")
                continue
            trajectory = shown[idx]
            _draw_discrete_background(ax, env)
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=colors[idx],
                linewidth=1.4,
                alpha=0.95,
                zorder=4,
            )
            ax.scatter(trajectory[0, 0], trajectory[0, 1], s=16, facecolor="white", edgecolor="black", linewidth=0.6, zorder=6)
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], s=20, marker="x", color="black", linewidth=0.8, zorder=7)
            _style_trajectory_axis(ax, env, trajectories, f"episode {idx}")
            ax.tick_params(labelsize=5.5)
        fig.suptitle(f"Per-episode evaluation trajectories at step {step}", fontsize=8)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _paper_trajectory_rc() -> dict:
    return {
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "figure.titlesize": 9,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def save_eval_trajectory_plots(
    trajectories,
    env,
    step: int,
    save_dir: str | os.PathLike = "eval_trajectory_plots",
    styles: Optional[tuple[str, ...]] = None,
) -> dict[str, str]:
    """
    Save paper-style evaluation trajectory candidates.

    Styles:
    - colored: one color per episode, useful when individual paths matter.
    - heatmap_overlay: log visitation heatmap plus colored paths, best for overlaps.
    - occupancy: aggregate visitation only, cleanest for density/coverage.
    - endpoints: faint paths with emphasized final states, useful for success modes.
    - small_multiples: one subplot per episode, useful when overlays are too dense.
    - suite: a 2x2 comparison panel of the first four styles.
    """
    trajectories = _prepare_trajectories(trajectories)
    if not trajectories:
        return {}

    if styles is None:
        styles = ("suite", "colored", "heatmap_overlay", "occupancy", "endpoints", "small_multiples")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    filename_prefix = f"eval_trajectories_step_{step}_ntraj_{len(trajectories)}"

    saved_paths = {}
    for style in styles:
        if style == "suite":
            save_path = save_dir / f"{filename_prefix}_suite.png"
            _save_eval_trajectory_suite(trajectories, env, save_path, step)
        elif style == "small_multiples":
            save_path = save_dir / f"{filename_prefix}_small_multiples.png"
            _save_eval_trajectory_small_multiples(trajectories, env, save_path, step)
        else:
            save_path = save_dir / f"{filename_prefix}_{style}.png"
            _save_single_eval_trajectory_style(trajectories, env, save_path, style, step)
            if not save_path.exists():
                continue
        saved_paths[style] = str(save_path)

    if saved_paths:
        print(f"✓ Evaluation trajectory plots saved in: {save_dir}")
    return saved_paths


class BaseDomainDebugVisualizer:
    def save(self, step: int) -> None:
        raise NotImplementedError


class GridworldVisualizerAdapter(BaseDomainDebugVisualizer):
    def __init__(self, visualizer, save_dir: str = "gridworld_plots"):
        self.visualizer = visualizer
        self.save_dir = Path(save_dir)

    def save(self, step: int) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.save_dir / f"step_{step}.png"
        self.visualizer.plot_results(step, str(save_path))


class ContinuousCoverageVisualizer(BaseDomainDebugVisualizer):
    def __init__(self, agent, env, save_dir: str, rollout_steps: int = 256, bins: int = 40):
        self.agent = agent
        self.env = env
        self.save_dir = Path(save_dir)
        self.rollout_steps = rollout_steps
        self.bins = bins
        self._running_lower_bounds: Optional[np.ndarray] = None
        self._running_upper_bounds: Optional[np.ndarray] = None

    def _sample_policy_rollout(self, step: int) -> np.ndarray:
        rng = np.random.default_rng(int(step))
        reset_seed = int(step)
        time_step = self.env.reset(seed=reset_seed)
        coords = []

        for rollout_step in range(self.rollout_steps):
            coord = self._extract_coordinates(time_step)
            if coord is not None:
                coords.append(coord)

            probs = np.asarray(self.agent.compute_action_probs(time_step.observation), dtype=np.float64)
            probs = np.clip(probs, 0.0, None)
            probs = probs / max(probs.sum(), 1e-12)
            action = rng.choice(self.agent.n_actions, p=probs)

            time_step = self.env.step(action)
            if time_step.last():
                reset_seed += rollout_step + 1
                time_step = self.env.reset(seed=reset_seed)

        if not coords:
            return np.zeros((0, 0), dtype=np.float32)
        return np.asarray(coords, dtype=np.float32)

    def _extract_coordinates(self, time_step) -> Optional[np.ndarray]:
        raise NotImplementedError

    def _get_env_plot_bounds(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if not self._use_env_plot_bounds():
            return None

        method = _get_env_method(self.env, "get_debug_plot_bounds")
        if not callable(method):
            return None

        bounds = method()
        if isinstance(bounds, dict):
            lower = bounds.get("lower")
            upper = bounds.get("upper")
        elif isinstance(bounds, (tuple, list)) and len(bounds) == 2:
            lower, upper = bounds
        else:
            return None

        if lower is None or upper is None:
            return None

        lower = np.asarray(lower, dtype=np.float32).reshape(-1)
        upper = np.asarray(upper, dtype=np.float32).reshape(-1)
        if lower.shape != upper.shape:
            return None
        return self._expand_bounds(lower, upper)

    def _update_running_bounds(self, coords: np.ndarray) -> None:
        if coords.size == 0:
            return

        coords_min = coords.min(axis=0)
        coords_max = coords.max(axis=0)
        if self._running_lower_bounds is None:
            self._running_lower_bounds = coords_min.copy()
            self._running_upper_bounds = coords_max.copy()
            return

        self._running_lower_bounds = np.minimum(self._running_lower_bounds, coords_min)
        self._running_upper_bounds = np.maximum(self._running_upper_bounds, coords_max)

    def _get_plot_bounds(self, coords: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
        env_bounds = self._get_env_plot_bounds()
        if env_bounds is not None:
            return env_bounds

        self._update_running_bounds(coords)
        if self._running_lower_bounds is None or self._running_upper_bounds is None:
            return None
        return self._expand_bounds(self._running_lower_bounds, self._running_upper_bounds)

    @staticmethod
    def _expand_bounds(
        lower: np.ndarray,
        upper: np.ndarray,
        relative_margin: float = 0.05,
        minimum_margin: float = 1e-3,
    ) -> tuple[np.ndarray, np.ndarray]:
        lower = lower.astype(np.float32, copy=True)
        upper = upper.astype(np.float32, copy=True)

        span = upper - lower
        margin = np.maximum(np.abs(span) * relative_margin, minimum_margin)
        lower -= margin
        upper += margin

        degenerate = upper <= lower
        if np.any(degenerate):
            lower[degenerate] -= minimum_margin
            upper[degenerate] += minimum_margin
        return lower, upper

    def _use_env_plot_bounds(self) -> bool:
        return True


class FetchCoverageVisualizer(ContinuousCoverageVisualizer):
    def __init__(self, agent, env, save_dir: str = "fetch_plots", rollout_steps: int = 256, bins: int = 36):
        super().__init__(agent, env, save_dir=save_dir, rollout_steps=rollout_steps, bins=bins)

    def _extract_coordinates(self, time_step) -> Optional[np.ndarray]:
        method = _get_env_method(self.env, "get_debug_coordinates")
        if callable(method):
            debug_info = method()
            if isinstance(debug_info, dict) and "xyz" in debug_info:
                xyz = np.asarray(debug_info["xyz"], dtype=np.float32).reshape(-1)
                if xyz.size >= 3:
                    return xyz[:3]

        proprio = np.asarray(getattr(time_step, "proprio_observation", []), dtype=np.float32).reshape(-1)
        if proprio.size >= 3:
            return proprio[:3]
        return None

    def save(self, step: int) -> None:
        coords = self._sample_policy_rollout(step)
        if coords.size == 0:
            return
        bounds = self._get_plot_bounds(coords)
        if bounds is None:
            return
        lower, upper = bounds

        self.save_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
        projections = (
            (0, 1, "x", "y", "XY end-effector coverage"),
            (0, 2, "x", "z", "XZ end-effector coverage"),
            (1, 2, "y", "z", "YZ end-effector coverage"),
        )

        for ax, (i, j, xlabel, ylabel, title) in zip(axes, projections):
            heatmap, xedges, yedges = np.histogram2d(
                coords[:, i],
                coords[:, j],
                bins=self.bins,
                range=[[lower[i], upper[i]], [lower[j], upper[j]]],
            )
            im = ax.imshow(
                heatmap.T,
                origin="lower",
                aspect="auto",
                extent=[lower[i], upper[i], lower[j], upper[j]],
                cmap="magma",
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xlim(lower[i], upper[i])
            ax.set_ylim(lower[j], upper[j])

        fig.suptitle(f"Fetch coverage rollout at step {step}, n samples: {coords.shape[0]}", fontsize=14)
        save_path = self.save_dir / f"step_{step}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Fetch coverage plot saved: {os.getcwd()}/{save_path}")


class XYCoverageVisualizer(ContinuousCoverageVisualizer):
    ACTION_COLORS = ("#2563eb", "#dc2626", "#16a34a", "#ca8a04", "#7c3aed", "#0891b2")
    POINTMAZE_ACTION_VECTORS = {
        0: np.array([1.0, 0.0], dtype=np.float32),
        1: np.array([-1.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 1.0], dtype=np.float32),
        3: np.array([0.0, -1.0], dtype=np.float32),
    }

    def __init__(
        self,
        agent,
        env,
        save_dir: str = "continuous_xy_plots",
        rollout_steps: int = 1000,
        bins: int = 36,
        title_prefix: str = "Continuous XY",
        policy_eval_points: int = 10,
    ):
        super().__init__(agent, env, save_dir=save_dir, rollout_steps=rollout_steps, bins=bins)
        self.title_prefix = title_prefix
        self.policy_eval_points = int(policy_eval_points)

    def _extract_coordinates(self, time_step) -> Optional[np.ndarray]:
        method = _get_env_method(self.env, "get_debug_coordinates")
        if callable(method):
            debug_info = method()
            if isinstance(debug_info, dict) and "xy" in debug_info:
                xy = np.asarray(debug_info["xy"], dtype=np.float32).reshape(-1)
                if xy.size >= 2:
                    return xy[:2]

        proprio = np.asarray(getattr(time_step, "proprio_observation", []), dtype=np.float32).reshape(-1)
        if proprio.size >= 2:
            return proprio[:2]
        return None

    def _get_initial_position(self) -> Optional[np.ndarray]:
        method = _get_env_method(self.env, "get_debug_coordinates")
        if callable(method):
            debug_info = method()
            if isinstance(debug_info, dict) and "fixed_start" in debug_info:
                fixed_start = np.asarray(debug_info["fixed_start"], dtype=np.float32).reshape(-1)
                if fixed_start.size >= 2:
                    return fixed_start[:2]

        current = self.env
        visited = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            for attr_name in ("start_position", "fixed_start"):
                start_position = getattr(current, attr_name, None)
                if start_position is None:
                    continue
                start_position = np.asarray(start_position, dtype=np.float32).reshape(-1)
                if start_position.size >= 2:
                    return start_position[:2]
            current = getattr(current, "env", None)

        return None

    def _get_maze_layout(self):
        method = _get_env_method(self.env, "get_debug_maze_layout")
        if not callable(method):
            return None

        layout = method()
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

    def _overlay_maze_walls(self, ax) -> None:
        layout = self._get_maze_layout()
        if layout is None:
            return

        maze_lower = layout["maze_lower"]
        maze_upper = layout["maze_upper"]
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

        ax.add_patch(
            patches.Rectangle(
                (maze_lower[0], maze_lower[1]),
                maze_upper[0] - maze_lower[0],
                maze_upper[1] - maze_lower[1],
                fill=False,
                edgecolor="black",
                linewidth=1.5,
                zorder=4,
            )
        )

    def _overlay_nystrom_points(self, ax) -> None:
        debug_helper = getattr(self.agent, "nystrom_debug", None)
        points = getattr(debug_helper, "fixed_xy_points", None)
        if points is None:
            return
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if points.size == 0:
            return
        ax.scatter(
            points[:, 0],
            points[:, 1],
            marker=".",
            s=5,
            c="#f4a261",
            alpha=0.42,
            linewidths=0,
            zorder=6,
            label="Nyström points",
        )

    def _policy_probe_points(self) -> Optional[np.ndarray]:
        n_points = int(self.policy_eval_points)
        if n_points <= 0:
            return None

        layout = self._get_maze_layout()
        if layout is not None:
            lower = np.asarray(layout["maze_lower"], dtype=np.float32).reshape(-1)
            upper = np.asarray(layout["maze_upper"], dtype=np.float32).reshape(-1)
        else:
            bounds = self._get_env_plot_bounds()
            if bounds is None:
                return None
            lower, upper = bounds

        if lower.size < 2 or upper.size < 2:
            return None

        xs = np.linspace(float(lower[0]), float(upper[0]), n_points, dtype=np.float32)
        if abs(float(upper[1] - lower[1])) <= 1e-6:
            ys = np.full(n_points, float(lower[1]), dtype=np.float32)
        else:
            ys = np.full(n_points, 0.5 * float(lower[1] + upper[1]), dtype=np.float32)
        return np.column_stack([xs, ys]).astype(np.float32, copy=False)

    def _policy_probe_scale(self, points: np.ndarray) -> float:
        layout = self._get_maze_layout()
        if layout is not None:
            span = np.asarray(layout["maze_upper"] - layout["maze_lower"], dtype=np.float32)
        else:
            bounds = self._get_env_plot_bounds()
            if bounds is None:
                span = np.ptp(points, axis=0)
            else:
                lower, upper = bounds
                span = np.asarray(upper - lower, dtype=np.float32)

        finite_span = span[np.isfinite(span) & (span > 1e-6)]
        base_scale = float(np.min(finite_span)) if finite_span.size else 1.0

        if points.shape[0] > 1:
            deltas = points[:, None, :] - points[None, :, :]
            distances = np.linalg.norm(deltas, axis=2)
            distances = distances[distances > 1e-6]
            if distances.size:
                base_scale = min(base_scale, float(np.min(distances)) * 1.35)

        return max(base_scale * 0.45, 0.08)

    def _format_policy_probe_observation(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.uint8)
        obs_shape = tuple(getattr(self.agent, "obs_shape", ()))
        grayscale = bool(getattr(self.agent, "grayscale", False))
        image_channels = int(getattr(self.agent, "image_channels", 1 if grayscale else 3))
        target_hw = obs_shape[-2:] if len(obs_shape) == 3 else image.shape[:2]

        if grayscale:
            if image.ndim == 3 and image.shape[2] == 1:
                image = image[..., 0]
            elif image.ndim == 3:
                image = np.asarray(Image.fromarray(image).convert("L"))
        elif image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)

        if image.shape[:2] != tuple(target_hw):
            image = np.asarray(Image.fromarray(image).resize((int(target_hw[1]), int(target_hw[0])), Image.LANCZOS))

        if grayscale and image.ndim == 2:
            image = image[..., None]
        elif not grayscale and image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)

        if image.ndim != 3 or image.shape[2] != image_channels:
            raise ValueError(f"Expected policy probe image with {image_channels} channels, got {image.shape}")

        image_chw = image.transpose(2, 0, 1).copy()
        if len(obs_shape) == 3 and image_chw.shape[0] != obs_shape[0]:
            frame_stack = obs_shape[0] // image_channels
            image_chw = np.tile(image_chw, (frame_stack, 1, 1))
        return image_chw

    def _observation_for_policy_probe_point(self, xy: np.ndarray) -> np.ndarray:
        debug_helper = getattr(self.agent, "nystrom_debug", None)
        observation_from_xy = getattr(debug_helper, "observation_from_xy", None)
        if callable(observation_from_xy):
            return observation_from_xy(self.agent, xy)

        if getattr(self.agent, "obs_type", None) != "pixels":
            return np.asarray(xy, dtype=np.float32).reshape(getattr(self.agent, "obs_shape", (2,)))

        render_from_position = _get_env_method(self.env, "render_from_position")
        if not callable(render_from_position):
            raise RuntimeError("Policy action-probability probes require render_from_position(position).")

        try:
            image = render_from_position(np.asarray(xy, dtype=np.float32), show_goal=False)
        except TypeError:
            image = render_from_position(np.asarray(xy, dtype=np.float32))
        return self._format_policy_probe_observation(image)

    def _set_policy_axis_limits(self, ax, points: np.ndarray) -> None:
        layout = self._get_maze_layout()
        if layout is not None:
            lower = np.asarray(layout["maze_lower"], dtype=np.float32)
            upper = np.asarray(layout["maze_upper"], dtype=np.float32)
        else:
            lower = points.min(axis=0)
            upper = points.max(axis=0)
        lower, upper = self._expand_bounds(lower, upper, relative_margin=0.04, minimum_margin=0.15)
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    def _plot_policy_probe_bars(
        self,
        ax,
        points: np.ndarray,
        probabilities: np.ndarray,
        scale: float,
        highlight_index: Optional[int] = None,
    ) -> None:
        n_actions = probabilities.shape[1]
        colors = self.ACTION_COLORS
        bar_spacing = scale / max(n_actions, 1)
        bar_width = bar_spacing * 0.78
        max_bar_height = scale * 0.78
        box_height = scale * 0.92
        box_width = scale * 1.08

        for point_idx, (point, probs) in enumerate(zip(points, probabilities)):
            x, y = float(point[0]), float(point[1])
            is_highlighted = highlight_index is not None and point_idx == int(highlight_index)
            ax.add_patch(
                patches.Rectangle(
                    (x - box_width / 2.0, y - box_height / 2.0),
                    box_width,
                    box_height,
                    facecolor="white",
                    edgecolor="#111827" if is_highlighted else "#4b5563",
                    linewidth=2.4 if is_highlighted else 0.45,
                    alpha=0.92 if is_highlighted else 0.78,
                    zorder=4 if is_highlighted else 1,
                )
            )
            start_x = x - 0.5 * bar_spacing * (n_actions - 1)
            bottom_y = y - box_height / 2.0 + scale * 0.08
            for action_idx, prob in enumerate(probs):
                bar_height = float(prob) * max_bar_height
                bar_x = start_x + action_idx * bar_spacing
                ax.add_patch(
                    patches.Rectangle(
                        (bar_x - bar_width / 2.0, bottom_y),
                        bar_width,
                        bar_height,
                        facecolor=colors[action_idx % len(colors)],
                        edgecolor="none",
                        alpha=0.95,
                        zorder=5 if is_highlighted else 2,
                    )
                )

    def _plot_policy_probe_arrows(self, ax, points: np.ndarray, probabilities: np.ndarray, scale: float) -> None:
        arrow_length = scale * 0.58
        for point, probs in zip(points, probabilities):
            action_idx = int(np.argmax(probs))
            vector = self.POINTMAZE_ACTION_VECTORS.get(action_idx)
            if vector is None:
                continue
            color = self.ACTION_COLORS[action_idx % len(self.ACTION_COLORS)]
            dx, dy = (vector * arrow_length).astype(float)
            ax.arrow(
                float(point[0]) - 0.5 * dx,
                float(point[1]) - 0.5 * dy,
                dx,
                dy,
                width=scale * 0.035,
                head_width=scale * 0.22,
                head_length=scale * 0.18,
                length_includes_head=True,
                facecolor=color,
                edgecolor=color,
                alpha=0.95,
                zorder=2,
            )

    def _add_policy_action_legend(self, ax, n_actions: int) -> None:
        labels = {
            0: "0: right (+x)",
            1: "1: left (-x)",
            2: "2: up (+y)",
            3: "3: down (-y)",
        }
        handles = [
            patches.Patch(
                facecolor=self.ACTION_COLORS[action_idx % len(self.ACTION_COLORS)],
                edgecolor="none",
                label=labels.get(action_idx, f"action {action_idx}"),
            )
            for action_idx in range(n_actions)
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    def _save_policy_action_probability_plot(self, step: int) -> None:
        points = self._policy_probe_points()
        if points is None or points.size == 0:
            return

        probabilities = []
        for xy in points:
            obs = self._observation_for_policy_probe_point(xy)
            probs = np.asarray(self.agent.compute_action_probs(obs), dtype=np.float64).reshape(-1)
            probs = np.clip(probs, 0.0, None)
            probs = probs / max(probs.sum(), 1e-12)
            probabilities.append(probs)
        probabilities = np.asarray(probabilities, dtype=np.float64)

        self.save_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        x_values = points[:, 0]
        for action_idx in range(probabilities.shape[1]):
            ax.plot(
                x_values,
                probabilities[:, action_idx],
                marker="o",
                linewidth=1.8,
                markersize=4,
                color=self.ACTION_COLORS[action_idx % len(self.ACTION_COLORS)],
                label=f"action {action_idx}",
            )

        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("x position")
        ax.set_ylabel("policy probability")
        ax.set_title(
            f"{self.title_prefix} policy action probabilities at step {step}\n"
            f"{points.shape[0]} evenly spaced probe points"
        )
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)

        save_path = self.save_dir / f"step_{step}_action_probs.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ {self.title_prefix} policy action-probability plot saved: {save_path}")

    def save(self, step: int) -> None:
        coords = self._sample_policy_rollout(step)
        if coords.size == 0:
            return
        bounds = self._get_plot_bounds(coords)
        if bounds is None:
            return
        lower, upper = bounds

        self.save_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        heatmap, xedges, yedges = np.histogram2d(
            coords[:, 0],
            coords[:, 1],
            bins=self.bins,
            range=[[lower[0], upper[0]], [lower[1], upper[1]]],
        )
        im = ax.imshow(
            heatmap.T,
            origin="lower",
            aspect="auto",
            extent=[lower[0], upper[0], lower[1], upper[1]],
            cmap="viridis",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{self.title_prefix} rollout coverage at step {step} (n samples: {coords.shape[0]})")
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        self._overlay_maze_walls(ax)
        self._overlay_nystrom_points(ax)
        initial_position = self._get_initial_position()
        if initial_position is not None:
            ax.scatter(
                initial_position[0],
                initial_position[1],
                marker="*",
                s=180,
                c="white",
                edgecolors="black",
                linewidths=0.9,
                zorder=5,
                label="initial position",
            )

        save_path = self.save_dir / f"step_{step}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ {self.title_prefix} rollout coverage plot saved: {os.getcwd()}/{save_path}")
        self._save_policy_action_probability_plot(step)


class PointMazeCoverageVisualizer(XYCoverageVisualizer):
    def __init__(
        self,
        agent,
        env,
        save_dir: str = "pointmaze_plots",
        rollout_steps: int = 10000,
        bins: int = 36,
        policy_eval_points: int = 128,
    ):
        super().__init__(
            agent,
            env,
            save_dir=save_dir,
            rollout_steps=rollout_steps,
            bins=bins,
            title_prefix="PointMaze XY",
            policy_eval_points=policy_eval_points,
        )

    def _use_env_plot_bounds(self) -> bool:
        return False

    def _policy_probe_points(self) -> Optional[np.ndarray]:
        n_points = int(self.policy_eval_points)
        if n_points <= 0:
            return None

        debug_helper = getattr(self.agent, "nystrom_debug", None)
        build_grid_points = getattr(debug_helper, "build_grid_points", None)
        if callable(build_grid_points):
            return np.asarray(build_grid_points(n_points), dtype=np.float32).reshape(-1, 2)

        return super()._policy_probe_points()

    def _save_policy_action_probability_plot(self, step: int) -> None:
        points = self._policy_probe_points()
        if points is None or points.size == 0:
            return

        probabilities = []
        for xy in points:
            obs = self._observation_for_policy_probe_point(xy)
            probs = np.asarray(self.agent.compute_action_probs(obs), dtype=np.float64).reshape(-1)
            probs = np.clip(probs, 0.0, None)
            probs = probs / max(probs.sum(), 1e-12)
            probabilities.append(probs)
        probabilities = np.asarray(probabilities, dtype=np.float64)

        self.save_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
        scale = self._policy_probe_scale(points)
        for ax, title in zip(
            axes,
            (
                "Policy probabilities by XY probe",
                "Most probable action by XY probe",
            ),
        ):
            self._overlay_maze_walls(ax)
            self._set_policy_axis_limits(ax, points)
            ax.set_title(title)
            ax.grid(True, alpha=0.18, linewidth=0.5)

        self._plot_policy_probe_bars(axes[0], points, probabilities, scale, highlight_index=0)
        self._plot_policy_probe_arrows(axes[1], points, probabilities, scale)
        self._add_policy_action_legend(axes[0], probabilities.shape[1])
        self._add_policy_action_legend(axes[1], probabilities.shape[1])
        fig.suptitle(
            f"{self.title_prefix} policy action probabilities at step {step} "
            f"({points.shape[0]} feasible probe points)",
            fontsize=13,
        )

        save_path = self.save_dir / f"step_{step}_action_probs.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ {self.title_prefix} policy action-probability plot saved: {save_path}")


class PointMazeNystromDebugVisualizer:
    """Standalone plots for fixed continuous Nyström landmarks."""

    def __init__(self, save_dir: str = "pointmaze_plots"):
        self.save_dir = Path(save_dir)

    def save_fixed_points_plot(self, layout: dict, points: np.ndarray, n_actions: int) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.save_dir / "fixed_nystrom_points.png"
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)

        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        for x0, y0, width, height in layout["wall_rectangles"]:
            ax.add_patch(
                patches.Rectangle(
                    (x0, y0),
                    width,
                    height,
                    facecolor="black",
                    edgecolor="black",
                    linewidth=0.5,
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
                linewidth=1.2,
            )
        )
        ax.scatter(points[:, 0], points[:, 1], s=7, c="#ff7f0e", linewidths=0.0, alpha=1.0)
        ax.scatter(
            points[0, 0],
            points[0, 1],
            marker="*",
            s=140,
            c="white",
            edgecolors="black",
            linewidths=0.9,
            zorder=5,
        )
        ax.set_xlim(lower[0] - 0.1, upper[0] + 0.1)
        ax.set_ylim(lower[1] - 0.1, upper[1] + 0.1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Fixed Nyström continuous grid ({points.shape[0]} states, all {n_actions} actions each)")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Fixed Nyström continuous grid plot saved to: {save_path}")
