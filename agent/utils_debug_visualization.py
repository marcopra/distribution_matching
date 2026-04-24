from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np


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

        fig.suptitle(f"Fetch coverage rollout at step {step}", fontsize=14)
        save_path = self.save_dir / f"step_{step}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Fetch coverage plot saved: {save_path}")


class PointMazeCoverageVisualizer(ContinuousCoverageVisualizer):
    def __init__(self, agent, env, save_dir: str = "pointmaze_plots", rollout_steps: int = 256, bins: int = 36):
        super().__init__(agent, env, save_dir=save_dir, rollout_steps=rollout_steps, bins=bins)

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

    def _use_env_plot_bounds(self) -> bool:
        return False

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
        ax.set_title(f"PointMaze XY coverage at step {step}")
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        self._overlay_maze_walls(ax)

        save_path = self.save_dir / f"step_{step}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ PointMaze coverage plot saved: {save_path}")


class RoverDebugVisualizerSuite:
    def __init__(
        self,
        agent,
        exploration_visualizer,
        gridworld_visualizer_factory: Callable,
    ):
        self.agent = agent
        self.exploration_visualizer = exploration_visualizer
        self._gridworld_visualizer_factory = gridworld_visualizer_factory
        self.domain_visualizer: Optional[BaseDomainDebugVisualizer] = None

    def attach_env(self, env) -> Optional[BaseDomainDebugVisualizer]:
        if _find_discrete_env(env) is not None:
            self.domain_visualizer = GridworldVisualizerAdapter(
                self._gridworld_visualizer_factory(self.agent)
            )
        elif _is_fetch_env(env):
            self.domain_visualizer = FetchCoverageVisualizer(self.agent, env, rollout_steps=500, bins=10)
        elif _is_point_maze_env(env):
            self.domain_visualizer = PointMazeCoverageVisualizer(self.agent, env, rollout_steps=1200, bins=20)
        else:
            self.domain_visualizer = None
        return self.domain_visualizer

    def save(self, step: int, obs_batch=None, z_batch=None, param_text: str = "") -> dict:
        metrics = {}
        if self.exploration_visualizer is not None and obs_batch is not None and z_batch is not None:
            vis_metrics = self.exploration_visualizer.update(
                obs_batch=obs_batch,
                z_batch=z_batch,
                step=step,
            )
            metrics.update(vis_metrics)
            self.exploration_visualizer.plot_all(step, param_text=param_text)

            if step % (self.agent.update_actor_every_steps * 3) == 0:
                try:
                    self.exploration_visualizer.plot_tsne(
                        z_batch,
                        step,
                        method="tsne",
                    )
                except Exception as exc:
                    print(f"⚠ Could not generate t-SNE plot at step {step}: {exc}")

        if self.domain_visualizer is not None:
            try:
                self.domain_visualizer.save(step)
            except Exception as exc:
                print(f"⚠ Could not generate domain debug plot at step {step}: {exc}")

        return metrics


def build_debug_visualizer_suite(
    agent,
    exploration_visualizer_cls,
    gridworld_visualizer_cls,
):
    exploration_visualizer = exploration_visualizer_cls(
        obs_shape=agent.obs_shape,
        obs_type=agent.obs_type,
        feature_dim=agent.feature_dim,
        hash_dim=1024,
        k_neighbors=5,
        occupancy_window=agent.update_actor_every_steps * 3,
        save_dir=os.path.join("exploration_plots", os.getcwd()),
        device=agent.device,
    )
    return RoverDebugVisualizerSuite(
        agent=agent,
        exploration_visualizer=exploration_visualizer,
        gridworld_visualizer_factory=gridworld_visualizer_cls,
    )
