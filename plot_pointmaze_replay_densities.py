"""Plot XY occupancy and joint position/velocity densities from replay episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


STATE_NAMES = ("x", "y", "vx", "vy")


def load_replay_states(replay_dir: Path) -> tuple[np.ndarray, int, int]:
    episode_paths = sorted(replay_dir.glob("*.npz"))
    if not episode_paths:
        raise FileNotFoundError(f"No replay episodes found in {replay_dir}")

    batches = []
    transition_count = 0
    skipped = 0
    for path in episode_paths:
        with np.load(path) as episode:
            if "proprio_state" not in episode:
                skipped += 1
                continue
            states = np.asarray(episode["proprio_state"], dtype=np.float32)
            episode_transitions = int(states.shape[0]) - 1
            if states.ndim != 2 or states.shape[1] < 4:
                raise ValueError(
                    f"{path} has proprio_state shape {states.shape}; expected [T, >=4]"
                )
            if states.shape[0] < episode_transitions:
                raise ValueError(
                    f"{path} has {states.shape[0]} state rows for "
                    f"{episode_transitions} transitions"
                )
            # Observation row i is the source state for transition i.
            batches.append(states[:episode_transitions, :4])
            transition_count += episode_transitions

    if not batches:
        raise ValueError(
            "No episodes contain proprio_state. Collect pixel replay with the "
            "updated ReplayBufferStorage first."
        )
    states = np.concatenate(batches, axis=0)
    states = states[np.isfinite(states).all(axis=1)]
    if states.size == 0:
        raise ValueError("Replay contains no finite PointMaze states")
    return states, transition_count, skipped


def _plot_xy_density(states: np.ndarray, output_path: Path, bins: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    counts, x_edges, y_edges = np.histogram2d(
        states[:, 0], states[:, 1], bins=int(bins)
    )
    positive = counts[counts > 0]
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        counts.T,
        shading="auto",
        cmap="magma",
        norm=LogNorm(vmin=1, vmax=max(2, float(positive.max()))),
    )
    fig.colorbar(mesh, ax=ax, label="Transitions per XY bin (log scale)")
    ax.set(xlabel="x", ylabel="y", title="RND replay density in physical XY")
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_xyz_density(
    states: np.ndarray,
    velocity_index: int,
    output_path: Path,
    bins: int,
    max_points: int,
    seed: int,
) -> None:
    values = states[:, [0, 1, velocity_index]]
    bin_count = max(8, int(bins))
    counts, edges = np.histogramdd(
        values,
        bins=(bin_count, bin_count, bin_count),
    )
    rng = np.random.default_rng(seed)
    sample_count = min(len(values), max(1, int(max_points)))
    sample_indices = rng.choice(len(values), size=sample_count, replace=False)
    sample = values[sample_indices]

    bins_by_axis = [
        np.clip(
            np.searchsorted(edges[axis], sample[:, axis], side="right") - 1,
            0,
            counts.shape[axis] - 1,
        )
        for axis in range(3)
    ]
    local_counts = counts[tuple(bins_by_axis)]
    density_colors = np.log10(local_counts + 1.0)

    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        sample[:, 0],
        sample[:, 1],
        sample[:, 2],
        c=density_colors,
        cmap="viridis",
        s=5,
        alpha=0.55,
        linewidths=0,
    )
    fig.colorbar(scatter, ax=ax, pad=0.1, label="log10(local voxel count + 1)")
    velocity_name = STATE_NAMES[velocity_index]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel(velocity_name)
    ax.set_title(f"RND replay density in (x, y, {velocity_name})")
    ax.set_xlim(edges[0][0], edges[0][-1])
    ax.set_ylim(edges[1][0], edges[1][-1])
    ax.set_zlim(edges[2][0], edges[2][-1])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--xy-bins", type=int, default=90)
    parser.add_argument("--joint-bins", type=int, default=40)
    parser.add_argument("--max-points", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    states, transition_count, skipped = load_replay_states(args.replay_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _plot_xyz_density(
        states, 2, args.output_dir / "density_xy_vx_3d.png",
        args.joint_bins, args.max_points, args.seed,
    )
    _plot_xyz_density(
        states, 3, args.output_dir / "density_xy_vy_3d.png",
        args.joint_bins, args.max_points, args.seed + 1,
    )
    _plot_xy_density(
        states, args.output_dir / "density_xy_2d.png", args.xy_bins
    )

    summary = {
        "replay_dir": str(args.replay_dir.resolve()),
        "transitions_in_state_episodes": int(transition_count),
        "finite_states_plotted": int(states.shape[0]),
        "episodes_missing_proprio_state": int(skipped),
        "state_order": list(STATE_NAMES),
        "mean": states.mean(axis=0).tolist(),
        "std": states.std(axis=0).tolist(),
        "min": states.min(axis=0).tolist(),
        "max": states.max(axis=0).tolist(),
    }
    (args.output_dir / "density_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"plotted {states.shape[0]:,} finite states from "
        f"{transition_count:,} transitions; skipped {skipped} episodes"
    )
    print(f"outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
