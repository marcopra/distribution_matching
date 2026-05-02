'''
python eval_occupancy_heatmaps.py --snapshot exp_local/2026.05.02/143749_062710_dist_matching/models/pixels/gym/dist_matching/1/snapshot.pt --env-config configs/env/gridworld/middle_room.yaml obs_type=pixels
'''
from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import patches
import numpy as np
from omegaconf import OmegaConf
import torch

import gym_env
import utils


def find_wrapped_env(env, predicate):
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if predicate(current):
            return current
        current = getattr(current, "env", None)
    return None


def find_discrete_env(env):
    return find_wrapped_env(
        env,
        lambda candidate: all(
            hasattr(candidate, attr)
            for attr in ("n_states", "idx_to_state", "state_to_idx")
        ),
    )


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


def as_xy(value) -> np.ndarray | None:
    if value is None:
        return None
    xy = np.asarray(value, dtype=np.float32).reshape(-1)
    if xy.size < 2 or not np.all(np.isfinite(xy[:2])):
        return None
    return xy[:2]


def discrete_plot_cells(env) -> list[tuple[int, int]] | None:
    discrete_env = find_discrete_env(env)
    if discrete_env is None:
        return None

    cells = []
    dead_state = getattr(discrete_env, "DEAD_STATE", None)
    for state in getattr(discrete_env, "cells", []):
        if dead_state is not None and state == dead_state:
            continue
        xy = as_xy(state)
        if xy is not None:
            cells.append((int(xy[0]), int(xy[1])))
    return cells or None


def extract_occupancy_trajectory_point(env, time_step) -> np.ndarray | None:
    info = getattr(time_step, "info", None)
    if isinstance(info, dict):
        for key in ("agent_position", "position", "xy"):
            xy = as_xy(info.get(key))
            if xy is not None:
                return xy

    debug_coordinates = get_env_method(env, "get_debug_coordinates")
    if callable(debug_coordinates):
        debug_info = debug_coordinates()
        if isinstance(debug_info, dict):
            for key in ("xy", "xyz", "agent_position", "position"):
                xy = as_xy(debug_info.get(key))
                if xy is not None:
                    return xy

    discrete_env = find_discrete_env(env)
    raw_proprio = getattr(time_step, "proprio_observation", [])
    proprio_array = np.asarray(raw_proprio, dtype=np.float32)
    proprio = proprio_array.reshape(-1)
    if discrete_env is not None and proprio.size == getattr(discrete_env, "n_states", -1):
        state_idx = int(np.argmax(proprio))
        return as_xy(discrete_env.idx_to_state.get(state_idx))

    if proprio_array.ndim <= 1:
        return as_xy(proprio)
    return None


def draw_discrete_background(ax, cells: Iterable[tuple[int, int]] | None) -> None:
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


def plot_bounds(env, trajectory_groups: list[list[np.ndarray]]) -> tuple[float, float, float, float]:
    cells = discrete_plot_cells(env)
    if cells is not None:
        xs = [cell[0] for cell in cells]
        ys = [cell[1] for cell in cells]
        return min(xs) - 0.5, max(xs) + 0.5, min(ys) - 0.5, max(ys) + 0.5

    all_trajectories = [
        trajectory
        for group in trajectory_groups
        for trajectory in group
        if trajectory.size > 0
    ]
    if not all_trajectories:
        raise RuntimeError("No valid trajectories were collected.")

    all_points = np.concatenate(all_trajectories, axis=0)
    min_x, min_y = all_points.min(axis=0)
    max_x, max_y = all_points.max(axis=0)
    span_x = max(max_x - min_x, 1e-3)
    span_y = max(max_y - min_y, 1e-3)
    margin_x = max(0.05 * span_x, 1e-3)
    margin_y = max(0.05 * span_y, 1e-3)
    return min_x - margin_x, max_x + margin_x, min_y - margin_y, max_y + margin_y


def prepare_trajectory(points: list[np.ndarray]) -> np.ndarray | None:
    prepared = [as_xy(point) for point in points]
    prepared = [point for point in prepared if point is not None]
    if not prepared:
        return None
    return np.asarray(prepared, dtype=np.float32)


def move_agent_to_device(agent, device: torch.device):
    agent.device = str(device)
    compute_dtype = getattr(agent, "compute_dtype", None)
    if isinstance(compute_dtype, torch.dtype):
        torch.set_default_dtype(compute_dtype)

    for name, value in list(vars(agent).items()):
        if isinstance(value, torch.nn.Module):
            value.to(device)
        elif isinstance(value, torch.Tensor):
            setattr(agent, name, value.to(device))

    train = getattr(agent, "train", None)
    if callable(train):
        train(False)
    return agent


def load_snapshot_agent(snapshot_path: Path, device: torch.device):
    import agent.rover  # noqa: F401
    import agent.rover_nystrom  # noqa: F401

    payload = torch.load(snapshot_path, map_location=device, weights_only=False)
    agent = payload["agent"] if isinstance(payload, dict) and "agent" in payload else payload
    return move_agent_to_device(agent, device)


def call_agent_act(agent, observation, meta, step: int, eval_mode: bool):
    signature = inspect.signature(agent.act)
    params = signature.parameters
    if "meta" in params or len(params) >= 4:
        return agent.act(observation, meta, step, eval_mode=eval_mode)
    return agent.act(observation, step, eval_mode=eval_mode)


def make_env(args):
    cfg = OmegaConf.load(args.env_config)
    env_cfg = cfg.env if "env" in cfg else cfg
    env_kwargs = OmegaConf.to_container(env_cfg, resolve=True)
    task_name = env_kwargs.pop("name")
    env_kwargs.pop("synthetic_first_transition", None)

    return gym_env.make(
        task_name,
        args.obs_type,
        frame_stack=args.frame_stack,
        action_repeat=args.action_repeat,
        seed=args.seed,
        resolution=args.resolution,
        grayscale=args.grayscale,
        url=not args.no_url,
        **env_kwargs,
    )


def collect_one_trajectory(env, agent, meta, step: int, seed: int, eval_mode: bool):
    time_step = env.reset(seed=seed)
    trajectory = []
    point = extract_occupancy_trajectory_point(env, time_step)
    if point is not None:
        trajectory.append(point)

    while not time_step.last():
        with torch.no_grad(), utils.eval_mode(agent):
            action = call_agent_act(agent, time_step.observation, meta, step, eval_mode)
        time_step = env.step(action)
        point = extract_occupancy_trajectory_point(env, time_step)
        if point is not None:
            trajectory.append(point)

    return prepare_trajectory(trajectory)


def collect_trajectory_groups(env, agent, args) -> list[list[np.ndarray]]:
    insert_env = getattr(agent, "insert_env", None)
    if callable(insert_env):
        insert_env(env)

    meta = agent.init_meta() if hasattr(agent, "init_meta") else {}
    groups = []
    for plot_idx in range(args.n_plots):
        group = []
        for traj_idx in range(args.k_trajectories):
            seed = args.seed + plot_idx * args.k_trajectories + traj_idx
            trajectory = collect_one_trajectory(
                env=env,
                agent=agent,
                meta=meta,
                step=args.step,
                seed=seed,
                eval_mode=args.eval_mode,
            )
            if trajectory is not None:
                group.append(trajectory)
        if not group:
            raise RuntimeError(f"No valid trajectories collected for plot {plot_idx}.")
        groups.append(group)
        print(f"Collected plot {plot_idx + 1}/{args.n_plots}: {len(group)} trajectories")
    return groups


def trajectory_group_to_heatmap(
    env,
    group: list[np.ndarray],
    bounds: tuple[float, float, float, float],
    bins: int,
) -> tuple[np.ndarray, list[float], bool, np.ndarray | None]:
    cells = discrete_plot_cells(env)
    if cells is not None:
        xs = [cell[0] for cell in cells]
        ys = [cell[1] for cell in cells]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        heatmap = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=np.float32)
        valid_mask = np.zeros_like(heatmap, dtype=bool)
        valid = set(cells)
        for x, y in valid:
            valid_mask[y - min_y, x - min_x] = True
        for point in np.concatenate(group, axis=0):
            cell = (int(round(point[0])), int(round(point[1])))
            if cell in valid:
                heatmap[cell[1] - min_y, cell[0] - min_x] += 1
        extent = [min_x - 0.5, max_x + 0.5, max_y + 0.5, min_y - 0.5]
        return heatmap, extent, True, valid_mask

    min_x, max_x, min_y, max_y = bounds
    points = np.concatenate(group, axis=0)
    heatmap, _, _ = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=bins,
        range=[[min_x, max_x], [min_y, max_y]],
    )
    return heatmap.T, [min_x, max_x, min_y, max_y], False, None


def center_index(shape: tuple[int, int]) -> tuple[int, int]:
    return shape[0] // 2, shape[1] // 2


def donor_capacity(
    heatmap: np.ndarray,
    center: tuple[int, int],
    valid_mask: np.ndarray,
) -> int:
    capacity = 0
    for row in range(heatmap.shape[0]):
        for col in range(heatmap.shape[1]):
            if (row, col) == center:
                continue
            if not valid_mask[row, col]:
                continue
            value = int(heatmap[row, col])
            if value > 1:
                capacity += value - 1
    return capacity


def add_zero_mass_preserving(
    heatmap: np.ndarray,
    probability: float,
    rng: np.random.Generator,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    adjusted = np.array(heatmap, dtype=np.float32, copy=True)
    if probability <= 0.0:
        return adjusted

    if valid_mask is None:
        valid_mask = np.ones_like(adjusted, dtype=bool)

    center = center_index(adjusted.shape)
    zero_locations = np.argwhere((adjusted == 0) & valid_mask)
    if zero_locations.size == 0:
        return adjusted

    selected_mask = rng.random(len(zero_locations)) < probability
    selected_zero_locations = zero_locations[selected_mask]
    if len(selected_zero_locations) == 0:
        return adjusted

    capacity = donor_capacity(adjusted, center, valid_mask)
    n_add = min(len(selected_zero_locations), capacity)
    if n_add == 0:
        print("Warning: no non-center heatmap locations with value > 1; zero-fill skipped.")
        return adjusted
    if n_add < len(selected_zero_locations):
        print(
            "Warning: capped zero-fill from "
            f"{len(selected_zero_locations)} to {n_add} cells to keep total heat constant."
        )
        selected_indices = rng.choice(len(selected_zero_locations), size=n_add, replace=False)
        selected_zero_locations = selected_zero_locations[selected_indices]

    for row, col in selected_zero_locations:
        adjusted[row, col] = 1

    remaining = n_add
    while remaining > 0:
        donor_locations = np.argwhere((adjusted > 1) & valid_mask)
        if donor_locations.size == 0:
            raise RuntimeError("Could not preserve heatmap mass: donor pool was exhausted.")
        donor_locations = np.asarray(
            [loc for loc in donor_locations if tuple(loc) != center],
            dtype=np.int64,
        )
        if donor_locations.size == 0:
            raise RuntimeError("Could not preserve heatmap mass: only the center can donate.")

        donor_idx = int(rng.integers(len(donor_locations)))
        row, col = donor_locations[donor_idx]
        adjusted[row, col] -= 1
        remaining -= 1

    if not np.isclose(adjusted.sum(), heatmap.sum()):
        raise RuntimeError("Heatmap mass changed after zero-fill adjustment.")
    return adjusted


def style_axis(ax, env, bounds: tuple[float, float, float, float]) -> None:
    min_x, max_x, min_y, max_y = bounds
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    if discrete_plot_cells(env) is not None:
        ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.tick_params(direction="out", length=3, width=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def draw_start_end_markers(ax, group: list[np.ndarray]) -> None:
    starts = np.asarray([trajectory[0] for trajectory in group], dtype=np.float32)
    ends = np.asarray([trajectory[-1] for trajectory in group], dtype=np.float32)
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


def save_occupancy_plots(env, trajectory_groups: list[list[np.ndarray]], args) -> list[Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bounds = plot_bounds(env, trajectory_groups)
    heatmaps = [
        trajectory_group_to_heatmap(env, group, bounds, args.bins)
        for group in trajectory_groups
    ]
    adjusted_heatmaps = []
    for idx, (heatmap, extent, is_discrete, valid_mask) in enumerate(heatmaps):
        rng = np.random.default_rng(args.seed + 10_000 + idx)
        adjusted = add_zero_mass_preserving(
            heatmap,
            probability=args.zero_fill_probability,
            rng=rng,
            valid_mask=valid_mask,
        )
        adjusted_heatmaps.append((adjusted, extent, is_discrete))
    heatmaps = adjusted_heatmaps
    global_max = max(float(heatmap.max()) for heatmap, _, _ in heatmaps)
    norm = mcolors.LogNorm(vmin=1, vmax=max(global_max, 1.0))
    cells = discrete_plot_cells(env)

    saved_paths = []
    rc = {
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
    with plt.rc_context(rc):
        for idx, (group, (heatmap, extent, is_discrete)) in enumerate(
            zip(trajectory_groups, heatmaps)
        ):
            masked = np.ma.masked_where(heatmap <= 0, heatmap)
            fig, ax = plt.subplots(figsize=(3.25, 3.05), constrained_layout=True)
            draw_discrete_background(ax, cells)
            imshow_kwargs = {}
            if not is_discrete:
                imshow_kwargs["origin"] = "lower"
            im = ax.imshow(
                masked,
                extent=extent,
                cmap=args.cmap,
                norm=norm,
                interpolation="nearest",
                zorder=1,
                **imshow_kwargs,
            )
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
            cbar.set_label("visit count")

            draw_start_end_markers(ax, group)
            style_axis(ax, env, bounds)
            ax.set_title("Aggregated visitation", pad=4)
            ax.text(
                0.01,
                0.99,
                f"step {args.step}, n={args.k_trajectories}",
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

            save_path = output_dir / f"occupancy_step_{args.step}_n{args.k_trajectories}_sample_{idx:02d}.png"
            fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(save_path)
    return saved_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sample k evaluation trajectories n times and save n occupancy heatmaps "
            "with a shared visit-count color scale."
        )
    )
    parser.add_argument("--snapshot", type=Path, required=True, help="Path to snapshot.pt or snapshot_<step>.pt.")
    parser.add_argument(
        "--env-config",
        type=Path,
        default=Path("configs/env/gridworld/middle_room.yaml"),
        help="Environment yaml containing an env.name field and env kwargs.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("eval_occupancy_heatmaps"))
    parser.add_argument("--n-plots", type=int, default=20, help="Number of occupancy plots to create.")
    parser.add_argument("--k-trajectories", type=int, default=20, help="Trajectories sampled per occupancy plot.")
    parser.add_argument("--step", type=int, default=50000, help="Step value passed to the policy and shown in the plot label.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--obs-type", type=str, default="discrete_states")
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=84)
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--no-url", action="store_true", help="Disable the url=True continuing-task wrapper used by pretrain.py.")
    parser.add_argument("--eval-mode", action="store_true", help="Use deterministic/eval policy mode. Default matches pretrain.py eval_mode=False.")
    parser.add_argument("--bins", type=int, default=48, help="Histogram bins for continuous-coordinate environments.")
    parser.add_argument("--zero-fill-probability", type=float, default=0.7)
    parser.add_argument("--cmap", type=str, default="viridis")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    env = make_env(args)
    try:
        agent = load_snapshot_agent(args.snapshot, device)
        groups = collect_trajectory_groups(env, agent, args)
        saved_paths = save_occupancy_plots(env, groups, args)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    print(f"Saved {len(saved_paths)} occupancy heatmaps in {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
