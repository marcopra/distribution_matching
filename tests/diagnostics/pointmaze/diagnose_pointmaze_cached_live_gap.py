#!/usr/bin/env python3
"""Compare cached synthetic PointMaze observations with live environment observations."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from agent.rover_visualization.domains import extract_eval_trajectory_point
from tests.diagnostics.pointmaze.sweep_pointmaze_encoder_embeddings import (
    build_agent,
    compose_pretrain_cfg,
    make_pretrain_env,
)
from tests.diagnostics.pointmaze.synthetic_workflow_utils import (
    load_dataset,
    load_encoder_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflow-dir",
        type=Path,
        default=Path("tests/outputs/pointmaze/synthetic_workflow"),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="PMD individual-run directory containing result.json and optional whitening_transform.npz.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--live-samples", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--distance-batch-size", type=int, default=256)
    parser.add_argument("--worst-pairs", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-dataset-mismatch", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def encode(encoder, observations: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    chunks = []
    encoder.eval()
    with torch.no_grad():
        for start in range(0, observations.shape[0], max(1, batch_size)):
            batch = torch.as_tensor(observations[start : start + batch_size], device=device)
            chunks.append(encoder.encode_and_project(batch.float()).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float64, copy=False)


def apply_whitening(features: np.ndarray, run_dir: Path, result: dict) -> np.ndarray:
    if result.get("feature_whitening", "none") == "none":
        return features.copy()
    transform_path = run_dir / "whitening_transform.npz"
    if not transform_path.exists():
        raise FileNotFoundError(f"Missing whitening transform: {transform_path}")
    with np.load(transform_path, allow_pickle=False) as transform:
        mean = transform["mean"].astype(np.float64)
        components = transform["components"].astype(np.float64)
        eigenvalues = transform["eigenvalues"].astype(np.float64)
    floor = float(result.get("whitening_epsilon", 1e-5)) * max(float(eigenvalues[0]), np.finfo(float).eps)
    whitened = (features - mean) @ components.T
    whitened /= np.sqrt(np.maximum(eigenvalues, floor))
    if bool(result.get("whitening_unit_trace", True)):
        whitened /= np.sqrt(max(whitened.shape[1], 1))
    return whitened


def nearest(reference: np.ndarray, queries: np.ndarray, batch_size: int):
    """Return nearest reference index and Euclidean distance without a large pair matrix."""
    indices, distances = [], []
    ref_sq = np.sum(reference * reference, axis=1)
    for start in range(0, queries.shape[0], max(1, batch_size)):
        query = queries[start : start + batch_size]
        squared = np.maximum(
            np.sum(query * query, axis=1, keepdims=True) + ref_sq[None] - 2.0 * query @ reference.T,
            0.0,
        )
        index = np.argmin(squared, axis=1)
        indices.append(index)
        distances.append(np.sqrt(squared[np.arange(index.size), index]))
    return np.concatenate(indices), np.concatenate(distances)


def collect_live(env, count: int, episodes: int, seed: int, n_actions: int):
    rng = np.random.default_rng(seed)
    observations, xy, episode_ids, steps = [], [], [], []
    for episode in range(max(1, episodes)):
        time_step = env.reset(seed=seed + episode)
        step = 0
        while True:
            point = extract_eval_trajectory_point(env, time_step)
            if point is not None:
                observations.append(np.asarray(time_step.observation, dtype=np.uint8))
                xy.append(point)
                episode_ids.append(episode)
                steps.append(step)
                if len(observations) >= count:
                    return np.stack(observations), np.stack(xy), np.asarray(episode_ids), np.asarray(steps)
            if time_step.last():
                break
            time_step = env.step(int(rng.integers(n_actions)))
            step += 1
    if not observations:
        raise RuntimeError("No live observations had extractable XY coordinates")
    print(f"WARNING: requested {count} live samples, collected {len(observations)}")
    return np.stack(observations), np.stack(xy), np.asarray(episode_ids), np.asarray(steps)


def image2d(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    return np.squeeze(image)


def save_metric_maps(output_dir: Path, xy: np.ndarray, metrics: dict) -> None:
    names = list(metrics)
    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.2 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, name in zip(axes, names):
        values = np.asarray(metrics[name])
        scatter = ax.scatter(xy[:, 0], xy[:, 1], c=values, s=13, cmap="viridis")
        fig.colorbar(scatter, ax=ax, shrink=0.82)
        ax.set_title(name.replace("_", " "))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15)
    for ax in axes[len(names) :]:
        ax.set_visible(False)
    fig.savefig(output_dir / "live_metric_maps.png", dpi=170)
    plt.close(fig)


def save_worst_pairs(output_dir, cached_obs, cached_xy, live_obs, live_xy, pair_index, score, label, count):
    count = min(max(1, count), live_obs.shape[0])
    selected = np.argsort(score)[-count:][::-1]
    fig, axes = plt.subplots(count, 2, figsize=(7.5, 2.55 * count), squeeze=False, constrained_layout=True)
    for row, live_index in enumerate(selected):
        cached_index = int(pair_index[live_index])
        axes[row, 0].imshow(image2d(live_obs[live_index]), cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_title(f"live {live_index} XY={live_xy[live_index].round(3)}")
        axes[row, 1].imshow(image2d(cached_obs[cached_index]), cmap="gray", vmin=0, vmax=255)
        axes[row, 1].set_title(f"cached {cached_index} XY={cached_xy[cached_index].round(3)}\n{label}={score[live_index]:.5g}")
        for ax in axes[row]:
            ax.axis("off")
    fig.savefig(output_dir / f"worst_pairs_{label}.png", dpi=150)
    plt.close(fig)


def summarize(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": np.min(values),
        "p25": np.percentile(values, 25),
        "median": np.median(values),
        "p75": np.percentile(values, 75),
        "p95": np.percentile(values, 95),
        "max": np.max(values),
        "mean": np.mean(values),
    }


def main() -> None:
    args = parse_args()
    workflow_dir = args.workflow_dir.resolve()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "cached_live_diagnostic").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = json.loads((run_dir / "result.json").read_text())
    dataset_dir = args.dataset_dir.resolve() if args.dataset_dir else workflow_dir / "dataset"
    arrays, metadata = load_dataset(dataset_dir)
    if result.get("dataset_checksum") != metadata["checksum"]:
        raise ValueError("PMD result and cached dataset checksums differ")

    n_actions = int(metadata["n_actions"])
    cached_obs = np.asarray(arrays["obs"])[::n_actions]
    cached_xy = np.asarray(arrays["xy"], dtype=np.float64)
    cfg = compose_pretrain_cfg(metadata["config_name"], args.seed)
    cfg.grayscale = bool(metadata["grayscale"])
    env = make_pretrain_env(cfg)
    try:
        build_args = argparse.Namespace(
            n_states=int(metadata["actual_n_states"]),
            batch_size=min(1024, arrays["obs"].shape[0]),
            exact_grid=bool(metadata.get("exact_grid", False)),
            border_margin=metadata.get("border_margin"),
            oversample=metadata.get("oversample"),
            device=args.device,
            updates=1,
        )
        agent = build_agent(cfg, env, int(result["feature_dim"]), build_args)
        load_encoder_checkpoint(
            workflow_dir / f"featuredim_{result['feature_dim']}" / "encoder.pt",
            agent.encoder,
            expected_feature_dim=int(result["feature_dim"]),
            expected_obs_shape=agent.obs_shape,
            expected_dataset_checksum=metadata["checksum"],
            device=args.device,
            allow_dataset_mismatch=args.allow_dataset_mismatch,
        )
        live_obs, live_xy, episode_ids, live_steps = collect_live(
            env, args.live_samples, args.episodes, args.seed, n_actions
        )
        cached_raw = encode(agent.encoder, cached_obs, args.device, args.encode_batch_size)
        live_raw = encode(agent.encoder, live_obs, args.device, args.encode_batch_size)
    finally:
        env.close()

    cached_white = apply_whitening(cached_raw, run_dir, result)
    live_white = apply_whitening(live_raw, run_dir, result)
    spatial_idx, spatial_dist = nearest(cached_xy, live_xy, args.distance_batch_size)
    raw_idx, raw_dist = nearest(cached_raw, live_raw, args.distance_batch_size)
    white_idx, white_dist = nearest(cached_white, live_white, args.distance_batch_size)

    spatial_raw_dist = np.linalg.norm(live_raw - cached_raw[spatial_idx], axis=1)
    spatial_white_dist = np.linalg.norm(live_white - cached_white[spatial_idx], axis=1)
    raw_xy_error = np.linalg.norm(live_xy - cached_xy[raw_idx], axis=1)
    white_xy_error = np.linalg.norm(live_xy - cached_xy[white_idx], axis=1)
    pixel_delta = live_obs.astype(np.float64) - cached_obs[spatial_idx].astype(np.float64)
    pixel_mae = np.mean(np.abs(pixel_delta), axis=tuple(range(1, pixel_delta.ndim)))
    pixel_rmse = np.sqrt(np.mean(pixel_delta * pixel_delta, axis=tuple(range(1, pixel_delta.ndim))))
    bandwidth = float(result["fitted_bandwidth"])
    if bandwidth <= 0 or not np.isfinite(bandwidth):
        raise ValueError(f"Invalid fitted Gaussian bandwidth: {bandwidth}")
    kernel_max = np.exp(-(white_dist ** 2) / (2.0 * bandwidth ** 2))
    kernel_spatial = np.exp(-(spatial_white_dist ** 2) / (2.0 * bandwidth ** 2))

    columns = {
        "episode": episode_ids,
        "step": live_steps,
        "live_x": live_xy[:, 0],
        "live_y": live_xy[:, 1],
        "spatial_cached_index": spatial_idx,
        "spatial_cached_x": cached_xy[spatial_idx, 0],
        "spatial_cached_y": cached_xy[spatial_idx, 1],
        "spatial_xy_distance": spatial_dist,
        "pixel_mae": pixel_mae,
        "pixel_rmse": pixel_rmse,
        "spatial_raw_latent_distance": spatial_raw_dist,
        "spatial_whitened_latent_distance": spatial_white_dist,
        "raw_nearest_cached_index": raw_idx,
        "raw_nearest_latent_distance": raw_dist,
        "raw_nearest_xy_error": raw_xy_error,
        "whitened_nearest_cached_index": white_idx,
        "whitened_nearest_latent_distance": white_dist,
        "whitened_nearest_xy_error": white_xy_error,
        "gaussian_kernel_max": kernel_max,
        "gaussian_kernel_spatial": kernel_spatial,
    }
    np.savez_compressed(output_dir / "cached_live_metrics.npz", live_observations=live_obs, live_xy=live_xy, **columns)
    with (output_dir / "cached_live_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(zip(*(columns[name] for name in columns)))

    mapped = {
        "spatial XY distance": spatial_dist,
        "pixel MAE": pixel_mae,
        "raw latent distance at spatial NN": spatial_raw_dist,
        "whitened latent distance at spatial NN": spatial_white_dist,
        "whitened latent NN XY error": white_xy_error,
        "Gaussian kernel maximum": kernel_max,
        "Gaussian kernel at spatial NN": kernel_spatial,
    }
    save_metric_maps(output_dir, live_xy, mapped)
    save_worst_pairs(output_dir, cached_obs, cached_xy, live_obs, live_xy, spatial_idx, pixel_mae, "pixel_mae", args.worst_pairs)
    save_worst_pairs(output_dir, cached_obs, cached_xy, live_obs, live_xy, spatial_idx, spatial_white_dist, "spatial_whitened_distance", args.worst_pairs)
    save_worst_pairs(output_dir, cached_obs, cached_xy, live_obs, live_xy, white_idx, white_xy_error, "latent_nn_xy_error", args.worst_pairs)

    summary_metrics = {name: summarize(value) for name, value in columns.items() if name not in {"episode", "step"} and "index" not in name and not name.endswith("_x") and not name.endswith("_y")}
    summary = {
        "workflow_dir": workflow_dir,
        "run_dir": run_dir,
        "output_dir": output_dir,
        "dataset_checksum": metadata["checksum"],
        "feature_dim": int(result["feature_dim"]),
        "feature_whitening": result.get("feature_whitening", "none"),
        "raw_latent_dim": cached_raw.shape[1],
        "whitened_latent_dim": cached_white.shape[1],
        "gaussian_bandwidth": bandwidth,
        "cached_states": cached_obs.shape[0],
        "live_samples": live_obs.shape[0],
        "metrics": summary_metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n")
    print(f"Saved cached/live diagnostic: {output_dir}")
    print(json.dumps(jsonable({"live_samples": live_obs.shape[0], "metrics": summary_metrics}), indent=2))


if __name__ == "__main__":
    main()
