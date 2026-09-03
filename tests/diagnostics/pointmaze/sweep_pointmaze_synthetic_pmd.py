#!/usr/bin/env python3
"""Run synthetic-only PMD sweeps with frozen PointMaze pixel encoders."""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import utils
from agent.rover_visualization.domains import (
    extract_eval_trajectory_point,
    pointmaze_free_space_coverage,
)
from tests.diagnostics.pointmaze.sweep_pointmaze_encoder_embeddings import (
    build_agent,
    compose_pretrain_cfg,
    make_pretrain_env,
)
from tests.diagnostics.pointmaze.synthetic_workflow_utils import (
    assert_module_unchanged,
    fit_pca_whitening,
    load_dataset,
    load_encoder_checkpoint,
)


def parse_optional_floats(values):
    result = []
    for value in values:
        result.append(None if str(value).lower() in {"none", "null", "auto"} else float(value))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflow-dir",
        type=Path,
        default=Path("tests/outputs/pointmaze/synthetic_workflow"),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Shared cached dataset directory. Defaults to WORKFLOW_DIR/dataset.",
    )
    parser.add_argument(
        "--config-name",
        default="pretrain_parallel/pretrain_pointmaze_umaze_1_pixels",
    )
    parser.add_argument("--feature-dims", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--kernels", nargs="+", default=["gaussian"])
    parser.add_argument("--bandwidths", nargs="+", default=["none"])
    parser.add_argument("--bandwidth-mults", nargs="+", default=["0.3"])
    parser.add_argument("--lambda-regs", type=float, nargs="+", default=[1e-6])
    parser.add_argument("--landmarks", type=int, nargs="+", default=[8000])
    parser.add_argument("--pmd-steps", type=int, nargs="+", default=[10])
    parser.add_argument("--etas", type=float, nargs="+", default=[10.0])
    parser.add_argument("--eta-mode", choices=["none", "adagrad", "backtracking", "adadiff"], default="backtracking")
    parser.add_argument("--pca-truncation", type=int, default=500)
    parser.add_argument(
        "--feature-whitening",
        choices=("none", "pca"),
        default="none",
        help="Fit a fixed PCA whitening transform on unique synthetic images before PMD.",
    )
    parser.add_argument(
        "--whitening-variance",
        type=float,
        default=0.99,
        help="Variance retained when --whitening-components is 0.",
    )
    parser.add_argument(
        "--whitening-components",
        type=int,
        default=0,
        help="Fixed PCA rank; 0 chooses rank from --whitening-variance.",
    )
    parser.add_argument(
        "--whitening-epsilon",
        type=float,
        default=1e-5,
        help="Eigenvalue floor relative to largest retained eigenvalue.",
    )
    parser.add_argument(
        "--whitening-unit-trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Divide whitened vectors by sqrt(retained rank) for dimension-stable distances.",
    )
    parser.add_argument("--sink", type=float, default=0.8)
    parser.add_argument("--eval-trajectories", type=int, default=50)
    parser.add_argument("--coverage-grid-size", type=int, default=90)
    parser.add_argument("--coverage-radius", type=float, default=0.08)
    parser.add_argument("--eval-epsilon", type=float, default=0.0)
    parser.add_argument(
        "--action-prob-plot-points",
        type=int,
        default=250,
        help="Synthetic dataset points shown in action-probability plot; 0 shows all points.",
    )
    parser.add_argument(
        "--action-prob-batch-size",
        type=int,
        default=1024,
        help="Batch size for evaluating cached synthetic images.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=256,
        help="Images encoded per GPU forward pass. Raw pixels remain on CPU.",
    )
    parser.add_argument(
        "--allow-dataset-mismatch",
        action="store_true",
        help="Allow an old encoder checkpoint trained on a different cached dataset; shape and feature dimension remain strict.",
    )
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def kernel_settings(kernel: str, bandwidths, multipliers):
    if kernel not in {"gaussian", "gaussian_chunked", "laplacian"}:
        return [(None, None)]
    settings = []
    settings.extend((bandwidth, None) for bandwidth in bandwidths if bandwidth is not None)
    if kernel in {"gaussian", "gaussian_chunked"} and any(bandwidth is None for bandwidth in bandwidths):
        settings.extend((None, multiplier) for multiplier in multipliers if multiplier is not None)
    if not settings:
        settings.append((1.0, None))
    return settings


def landmark_indices(total_transitions: int, count: int, n_actions: int) -> np.ndarray:
    if count <= 0 or count > total_transitions:
        raise ValueError(f"landmarks must be in [1, {total_transitions}], got {count}")
    if count % n_actions != 0:
        raise ValueError(f"landmarks={count} must be divisible by n_actions={n_actions}")
    total_states = total_transitions // n_actions
    landmark_states = count // n_actions
    states = np.rint(np.linspace(0, total_states - 1, landmark_states)).astype(np.int64)
    return (states[:, None] * n_actions + np.arange(n_actions)[None, :]).reshape(-1)


def configure_kernel(agent, kernel: str, bandwidth, multiplier) -> None:
    agent.kernel_type = kernel
    agent.kernel_bandwidth = bandwidth
    agent.kernel_bandwidth_mult = multiplier
    agent.kernel_fn = utils.build_kernel_fn(kernel, bandwidth=bandwidth)
    agent.distribution_matcher.kernel_type = kernel
    agent.distribution_matcher.kernel_bandwidth = bandwidth
    agent.distribution_matcher.kernel_fn = utils.build_kernel_fn(kernel, bandwidth=bandwidth)
    agent.distribution_matcher.state_kernel_fn = agent.kernel_fn


def evaluate_coverage(agent, env, args) -> float:
    trajectories = []
    previous_epsilon = agent.epsilon_schedule
    agent.epsilon_schedule = float(args.eval_epsilon)
    try:
        for episode in range(int(args.eval_trajectories)):
            time_step = env.reset(seed=int(args.seed) + episode)
            trajectory = []
            point = extract_eval_trajectory_point(env, time_step)
            if point is not None:
                trajectory.append(point)
            while not time_step.last():
                with torch.no_grad():
                    action = agent.act(
                        time_step.observation,
                        agent.init_meta(),
                        step=agent.num_expl_steps + agent.T_init_steps + 1,
                        eval_mode=True,
                    )
                time_step = env.step(action)
                point = extract_eval_trajectory_point(env, time_step)
                if point is not None:
                    trajectory.append(point)
            if trajectory:
                trajectories.append(trajectory)
    finally:
        agent.epsilon_schedule = previous_epsilon
    _, _, coverage = pointmaze_free_space_coverage(
        env,
        trajectories,
        grid_size=int(args.coverage_grid_size),
        radius=float(args.coverage_radius),
    )
    return float(coverage)


def evaluate_synthetic_action_probabilities(
    agent, arrays, n_actions: int, batch_size: int, max_points: int
):
    """Evaluate a deterministic grid subset of cached images; never rerender probes."""
    all_xy = np.asarray(arrays["xy"], dtype=np.float32).reshape(-1, 2)
    all_observations = np.asarray(arrays["obs"])[::n_actions]
    if all_observations.shape[0] != all_xy.shape[0]:
        raise ValueError(
            f"Synthetic image/XY mismatch: observations={all_observations.shape[0]}, xy={all_xy.shape[0]}"
        )
    state_indices = spatial_grid_plot_indices(all_xy, int(max_points))
    xy = all_xy[state_indices]
    observations = all_observations[state_indices]
    chunks = []
    effective_batch_size = max(1, int(batch_size))
    for start in range(0, observations.shape[0], effective_batch_size):
        chunks.append(agent._compute_action_probs_batch(observations[start : start + effective_batch_size]))
    probabilities = np.concatenate(chunks, axis=0).astype(np.float64, copy=False)
    probabilities = np.clip(probabilities, 0.0, None)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    bad_rows = (~np.isfinite(row_sums[:, 0])) | (row_sums[:, 0] <= 0.0)
    probabilities[bad_rows] = 1.0 / n_actions
    probabilities[~bad_rows] /= row_sums[~bad_rows]
    return xy, probabilities, state_indices


def spatial_grid_plot_indices(xy: np.ndarray, max_items: int) -> np.ndarray:
    """Select a coarse, regular sub-lattice instead of subsampling flattened order."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    n_items = xy.shape[0]
    if max_items <= 0 or n_items <= max_items:
        return np.arange(n_items, dtype=np.int64)

    # Synthetic exact-grid coordinates have small float32 noise at most. Rounding
    # recovers lattice levels while keeping returned indices tied to original data.
    rounded = np.round(xy, decimals=6)
    x_levels = np.unique(rounded[:, 0])
    y_levels = np.unique(rounded[:, 1])
    if x_levels.size < 2 or y_levels.size < 2:
        return np.rint(np.linspace(0, n_items - 1, max_items)).astype(np.int64)

    dx = float(np.median(np.diff(x_levels)))
    dy = float(np.median(np.diff(y_levels)))
    base_stride = max(1, int(round(np.sqrt(n_items / max_items))))
    max_stride = max(base_stride + 4, int(np.ceil(np.sqrt(n_items / max_items))) + 3)
    best = None
    for x_stride in range(1, max_stride + 1):
        for y_stride in range(1, max_stride + 1):
            coarse_dx = abs(dx * x_stride)
            coarse_dy = abs(dy * y_stride)
            spacing_mismatch = abs(coarse_dx - coarse_dy) / max(
                0.5 * (coarse_dx + coarse_dy), 1e-12
            )
            for x_offset in range(x_stride):
                selected_x = x_levels[x_offset::x_stride]
                x_mask = np.isin(rounded[:, 0], selected_x)
                for y_offset in range(y_stride):
                    selected_y = y_levels[y_offset::y_stride]
                    indices = np.flatnonzero(x_mask & np.isin(rounded[:, 1], selected_y))
                    if indices.size == 0:
                        continue
                    # Never exceed requested plot budget. Among valid lattices,
                    # prioritize isotropic spacing, then use as much budget as possible.
                    score = (
                        indices.size > max_items,
                        spacing_mismatch,
                        abs(indices.size - max_items),
                        x_stride + y_stride,
                    )
                    if best is None or score < best[0]:
                        best = (score, indices, x_stride, y_stride, x_offset, y_offset)

    if best is None:
        return np.rint(np.linspace(0, n_items - 1, max_items)).astype(np.int64)
    _, indices, x_stride, y_stride, x_offset, y_offset = best
    print(
        "Action-probability probes use spatial sub-lattice: "
        f"{indices.size}/{n_items} states, x stride={x_stride} offset={x_offset}, "
        f"y stride={y_stride} offset={y_offset}."
    )
    return indices.astype(np.int64, copy=False)


def encode_synthetic_transitions_in_batches(agent, arrays, batch_size: int):
    """Encode cached pixels once in bounded GPU batches; keep raw arrays on CPU."""
    total = int(np.asarray(arrays["obs"]).shape[0])
    if total == 0:
        raise ValueError("Cannot encode an empty synthetic dataset")
    batch_size = max(1, int(batch_size))
    encoded_chunks = []
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        transitions = (
            torch.as_tensor(arrays["obs"][start:end]),
            torch.as_tensor(arrays["action"][start:end], dtype=torch.long),
            torch.as_tensor(arrays["reward"][start:end], dtype=agent.compute_dtype),
            torch.as_tensor(arrays["discount"][start:end], dtype=agent.compute_dtype),
            torch.as_tensor(arrays["next_obs"][start:end]),
        )
        encoded_chunks.append(agent._encode_actor_transition_batch_with_retries(transitions))
    return agent._concat_encoded_batches(encoded_chunks)


def index_encoded_batch(encoded, index: torch.Tensor):
    return {name: value.index_select(0, index) for name, value in encoded.items()}


def save_synthetic_grid_plot(agent, points, output_path: Path, title: str) -> None:
    """Plot cached grid coordinates with same maze overlay used by Rover debug plots."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] == 0:
        raise ValueError("Cannot plot empty synthetic XY grid")
    domain_visualizer = getattr(getattr(agent, "debug_visualizer", None), "domain_visualizer", None)
    if domain_visualizer is None:
        raise RuntimeError("PointMaze domain visualizer is unavailable")

    fig, ax = plt.subplots(figsize=(6, 5.4), constrained_layout=True)
    domain_visualizer._overlay_maze_walls(ax)
    domain_visualizer._set_policy_axis_limits(ax, points)
    marker_size = max(1.0, min(9.0, 18000.0 / points.shape[0]))
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=marker_size,
        c="#ff7f0e",
        linewidths=0.0,
        alpha=0.95,
        zorder=8,
    )
    ax.scatter(
        points[0, 0],
        points[0, 1],
        marker="*",
        s=130,
        c="white",
        edgecolors="black",
        linewidths=0.9,
        zorder=9,
    )
    ax.set_title(f"{title}\n{points.shape[0]} states")
    ax.grid(True, alpha=0.15, linewidth=0.4)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved synthetic XY grid plot: {output_path}")


def save_synthetic_action_probability_plot(agent, arrays, run_dir: Path, n_actions: int, args):
    points, probabilities, state_indices = evaluate_synthetic_action_probabilities(
        agent,
        arrays,
        n_actions=n_actions,
        batch_size=args.action_prob_batch_size,
        max_points=args.action_prob_plot_points,
    )
    np.savez_compressed(
        run_dir / "synthetic_dataset_action_probs.npz",
        xy=points,
        action_probabilities=probabilities.astype(np.float32),
        argmax_action=np.argmax(probabilities, axis=1).astype(np.int64),
        state_indices=state_indices,
    )

    plot_points = points
    plot_probabilities = probabilities
    domain_visualizer = getattr(getattr(agent, "debug_visualizer", None), "domain_visualizer", None)
    required_methods = (
        "_overlay_maze_walls",
        "_set_policy_axis_limits",
        "_policy_probe_scale",
        "_plot_policy_probe_bars",
        "_plot_policy_probe_arrows",
        "_add_policy_action_legend",
    )
    if domain_visualizer is None or not all(hasattr(domain_visualizer, name) for name in required_methods):
        raise RuntimeError("PointMaze domain visualizer does not expose action-probability plotting helpers")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    scale = domain_visualizer._policy_probe_scale(plot_points)
    for ax, title in zip(
        axes,
        ("Policy probabilities by synthetic XY point", "Most probable action by synthetic XY point"),
    ):
        domain_visualizer._overlay_maze_walls(ax)
        domain_visualizer._set_policy_axis_limits(ax, plot_points)
        ax.set_title(title)
        ax.grid(True, alpha=0.18, linewidth=0.5)

    domain_visualizer._plot_policy_probe_bars(
        axes[0], plot_points, plot_probabilities, scale, highlight_index=0
    )
    domain_visualizer._plot_policy_probe_arrows(axes[1], plot_points, plot_probabilities, scale)
    domain_visualizer._add_policy_action_legend(axes[0], n_actions)
    domain_visualizer._add_policy_action_legend(axes[1], n_actions)
    fig.suptitle(
        "Synthetic PointMaze image policy action probabilities "
        f"({plot_points.shape[0]} grid test points)",
        fontsize=13,
    )
    output_path = run_dir / "synthetic_dataset_action_probs.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved synthetic action-probability plot: {output_path}")
    return output_path


def run_id(config) -> str:
    bandwidth = "auto" if config["bandwidth"] is None else f"{config['bandwidth']:g}"
    multiplier = "none" if config["bandwidth_mult"] is None else f"{config['bandwidth_mult']:g}"
    whitening = str(config["feature_whitening"])
    if whitening == "pca":
        whitening += (
            f"_var{config['whitening_variance']:g}_c{config['whitening_components']}_"
            f"eps{config['whitening_epsilon']:g}_ut{int(config['whitening_unit_trace'])}"
        )
    return (
        f"d{config['feature_dim']}_{config['kernel']}_bw{bandwidth}_mult{multiplier}_"
        f"lam{config['lambda_reg']:g}_m{config['landmarks']}_pmd{config['pmd_steps']}_"
        f"eta{config['eta']:g}_sink{config['sink']:g}_white{whitening}"
    ).replace("+", "")


def scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    if isinstance(value, np.generic):
        return value.item()
    return value


def append_result(path: Path, row) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = []
    fieldnames = list(row)
    if path.exists():
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            existing_rows = list(reader)
            existing_fields = list(reader.fieldnames or [])
        fieldnames = existing_fields + [name for name in fieldnames if name not in existing_fields]
        if fieldnames != existing_fields:
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(existing_rows)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not path.exists() or path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    args.workflow_dir = args.workflow_dir.resolve()
    dataset_dir = (
        args.dataset_dir.resolve()
        if args.dataset_dir is not None
        else args.workflow_dir / "dataset"
    )
    arrays, dataset_metadata = load_dataset(dataset_dir)
    if dataset_metadata.get("config_name") != str(args.config_name):
        raise ValueError(
            f"Dataset config={dataset_metadata.get('config_name')!r} does not match "
            f"requested config={args.config_name!r}"
        )
    cfg = compose_pretrain_cfg(args.config_name, args.seed)
    cfg.grayscale = bool(dataset_metadata["grayscale"])
    n_actions = int(dataset_metadata["n_actions"])
    bandwidths = parse_optional_floats(args.bandwidths)
    multipliers = parse_optional_floats(args.bandwidth_mults)
    sweep_root = args.workflow_dir / "pmd_sweeps"
    results_path = sweep_root / "results.csv"

    combinations = []
    for feature_dim, kernel, lambda_reg, landmarks, pmd_steps, eta in itertools.product(
        args.feature_dims,
        args.kernels,
        args.lambda_regs,
        args.landmarks,
        args.pmd_steps,
        args.etas,
    ):
        for bandwidth, multiplier in kernel_settings(kernel, bandwidths, multipliers):
            combinations.append(
                {
                    "feature_dim": feature_dim,
                    "kernel": kernel,
                    "bandwidth": bandwidth,
                    "bandwidth_mult": multiplier,
                    "lambda_reg": lambda_reg,
                    "landmarks": landmarks,
                    "pmd_steps": pmd_steps,
                    "eta": eta,
                    "sink": float(args.sink),
                    "feature_whitening": str(args.feature_whitening),
                    "whitening_variance": float(args.whitening_variance),
                    "whitening_components": int(args.whitening_components),
                    "whitening_epsilon": float(args.whitening_epsilon),
                    "whitening_unit_trace": bool(args.whitening_unit_trace),
                }
            )

    for run_number, config in enumerate(combinations, start=1):
        identifier = run_id(config)
        run_dir = sweep_root / "individual_run_outputs" / identifier
        result_file = run_dir / "result.json"
        if args.skip_existing and result_file.exists():
            print(f"[{run_number}/{len(combinations)}] skip {identifier}")
            continue
        print(f"[{run_number}/{len(combinations)}] run {identifier}")
        run_dir.mkdir(parents=True, exist_ok=True)
        utils.set_seed_everywhere(args.seed)
        torch.manual_seed(args.seed)
        env = make_pretrain_env(cfg)
        old_cwd = Path.cwd()
        try:
            env.reset(seed=int(args.seed))
            build_args = argparse.Namespace(
                n_states=int(dataset_metadata["actual_n_states"]),
                batch_size=min(1024, arrays["obs"].shape[0]),
                exact_grid=bool(dataset_metadata.get("exact_grid", False)),
                border_margin=dataset_metadata.get("border_margin"),
                oversample=dataset_metadata.get("oversample"),
                device=args.device,
                updates=1,
            )
            agent = build_agent(cfg, env, int(config["feature_dim"]), build_args)
            checkpoint_path = args.workflow_dir / f"featuredim_{config['feature_dim']}" / "encoder.pt"
            checkpoint_payload = load_encoder_checkpoint(
                checkpoint_path,
                agent.encoder,
                expected_feature_dim=int(config["feature_dim"]),
                expected_obs_shape=agent.obs_shape,
                expected_dataset_checksum=dataset_metadata["checksum"],
                device=args.device,
                allow_dataset_mismatch=args.allow_dataset_mismatch,
            )
            agent.mode = str(checkpoint_payload["mode"])
            agent.encoder.mode = agent.mode
            agent.policy_encoder.mode = agent.mode
            agent._sync_policy_encoder()
            whitening_metadata = None
            if config["feature_whitening"] == "pca":
                unique_observations = np.asarray(arrays["obs"])[::n_actions]
                agent.encoder, whitening_metadata = fit_pca_whitening(
                    agent.encoder,
                    unique_observations,
                    device=args.device,
                    batch_size=args.action_prob_batch_size,
                    explained_variance=config["whitening_variance"],
                    components=config["whitening_components"],
                    epsilon=config["whitening_epsilon"],
                    unit_trace=config["whitening_unit_trace"],
                )
                agent.policy_encoder = copy.deepcopy(agent.encoder).to(args.device)
                agent._freeze_module(agent.encoder)
                agent._freeze_module(agent.policy_encoder)
                agent._policy_is_synced = True
                np.savez_compressed(
                    run_dir / "whitening_transform.npz",
                    mean=whitening_metadata["mean"],
                    components=whitening_metadata["components"],
                    eigenvalues=whitening_metadata["eigenvalues"],
                    all_eigenvalues=whitening_metadata["all_eigenvalues"],
                )
                print(
                    "PCA whitening: "
                    f"{whitening_metadata['input_dim']} -> {whitening_metadata['output_dim']} dims, "
                    f"explained_variance={whitening_metadata['explained_variance']:.6f}, "
                    f"unit_trace={whitening_metadata['unit_trace']}"
                )
            encoder_reference = {name: tensor.detach().clone() for name, tensor in agent.encoder.state_dict().items()}
            agent.encoder_optimizer = None
            agent.encoder_scheduler = None
            agent.lambda_reg = float(config["lambda_reg"])
            agent.distribution_matcher.lambda_reg = float(config["lambda_reg"])
            agent.pmd_steps = int(config["pmd_steps"])
            agent.lr_actor = float(config["eta"])
            agent.pmd_eta_mode = args.eta_mode
            agent.pca_truncation = min(int(args.pca_truncation), int(config["landmarks"]))
            agent.distribution_matcher.pca_truncation = agent.pca_truncation
            agent.sink_schedule = float(args.sink)
            configure_kernel(agent, config["kernel"], config["bandwidth"], config["bandwidth_mult"])

            total_transitions = int(np.asarray(arrays["obs"]).shape[0])
            sub_index_np = landmark_indices(total_transitions, int(config["landmarks"]), n_actions)
            sub_index = torch.as_tensor(sub_index_np, dtype=torch.long, device=args.device)
            full_xy = np.asarray(arrays["xy"], dtype=np.float32).reshape(-1, 2)
            landmark_state_indices = np.unique(sub_index_np // n_actions)
            landmark_xy = full_xy[landmark_state_indices]
            np.savez_compressed(
                run_dir / "synthetic_grid_points.npz",
                full_xy=full_xy,
                landmark_xy=landmark_xy,
                landmark_state_indices=landmark_state_indices,
            )
            save_synthetic_grid_plot(
                agent,
                full_xy,
                run_dir / "synthetic_actor_full_dataset.png",
                "PointMaze synthetic actor full dataset",
            )
            save_synthetic_grid_plot(
                agent,
                landmark_xy,
                run_dir / "synthetic_nystrom_subsamples.png",
                "PointMaze synthetic Nyström subsamples (uniform grid order)",
            )
            agent.use_tb = True
            os.chdir(run_dir)
            started = time.perf_counter()
            encoded_full = encode_synthetic_transitions_in_batches(
                agent, arrays, batch_size=args.encode_batch_size
            )
            encoded_sub = index_encoded_batch(encoded_full, sub_index)
            metrics = agent.update_actor_nystrom(
                None,
                None,
                None,
                step=agent.num_expl_steps + agent.T_init_steps,
                rewards=encoded_full["reward"],
                sub_rewards=encoded_sub["reward"],
                encoded_full=encoded_full,
                encoded_sub=encoded_sub,
            )
            elapsed = time.perf_counter() - started
            assert_module_unchanged(agent.encoder, encoder_reference)
            action_probability_plot = save_synthetic_action_probability_plot(
                agent,
                arrays,
                run_dir,
                n_actions,
                args,
            )
            coverage = evaluate_coverage(agent, env, args) if args.eval_trajectories > 0 else float("nan")
            fitted_bandwidth = getattr(agent.kernel_fn, "bandwidth", None)
            row = {
                "run_id": identifier,
                **config,
                "fitted_bandwidth": fitted_bandwidth,
                "whitening_output_dim": (
                    whitening_metadata["output_dim"] if whitening_metadata is not None else int(config["feature_dim"])
                ),
                "whitening_explained_variance": (
                    whitening_metadata["explained_variance"] if whitening_metadata is not None else 1.0
                ),
                "actor_loss": scalar(metrics.get("actor_loss", float("nan"))),
                "actor_best_loss": scalar(metrics.get("actor_best_loss", float("nan"))),
                "actor_eta": scalar(metrics.get("actor_eta", float("nan"))),
                "coverage_pct": coverage,
                "action_probability_plot": str(action_probability_plot),
                "elapsed_seconds": elapsed,
                "dataset_checksum": dataset_metadata["checksum"],
                "seed": int(args.seed),
            }
            result_file.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
            os.chdir(old_cwd)
            append_result(results_path, row)
        finally:
            os.chdir(old_cwd)
            env.close()

    print(f"Saved PMD sweep to {sweep_root}")


if __name__ == "__main__":
    main()
