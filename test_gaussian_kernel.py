from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

import gym_env
from agent.utils import PointMazeNystromDebugHelper


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = REPO_ROOT / "configs"


def resolve_group_path(group_value: str, group_name: str) -> Path:
    direct = CONFIG_DIR / f"{group_value}.yaml"
    if direct.exists():
        return direct
    grouped = CONFIG_DIR / group_name / f"{group_value}.yaml"
    if grouped.exists():
        return grouped
    raise FileNotFoundError(f"Could not resolve {group_name} config {group_value!r}")


def resolve_config_path(config_name: str | Path) -> Path:
    value = Path(config_name)
    candidates = []
    if value.suffix in {".yaml", ".yml"}:
        candidates.extend([value, REPO_ROOT / value, CONFIG_DIR / value])
    else:
        candidates.extend([CONFIG_DIR / f"{value}.yaml", REPO_ROOT / f"{value}.yaml"])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve config {config_name!r}")


def compose_pretrain_cfg(config_name: str | Path, seed: int | None):
    cfg = OmegaConf.load(resolve_config_path(config_name))
    defaults = OmegaConf.to_container(cfg.get("defaults", []), resolve=False)
    env_default = None
    for item in defaults:
        if isinstance(item, dict) and "/env" in item:
            env_default = item["/env"]
        elif isinstance(item, dict) and "env" in item:
            env_default = item["env"]
    if env_default is None:
        raise ValueError(f"Could not recover env default from config {config_name}")

    env_cfg = OmegaConf.load(resolve_group_path(env_default, "env"))
    cfg = OmegaConf.merge(env_cfg, cfg)
    if seed is not None:
        cfg.seed = int(seed)
    return cfg


def make_pretrain_env(cfg):
    env_kwargs = OmegaConf.to_container(cfg.env, resolve=True)
    env_kwargs.pop("name", None)
    return gym_env.make(
        cfg.task_name,
        cfg.obs_type,
        frame_stack=cfg.frame_stack,
        action_repeat=cfg.action_repeat,
        seed=cfg.seed,
        resolution=cfg.resolution,
        grayscale=bool(getattr(cfg, "grayscale", False)),
        url=True,
        **env_kwargs,
    )


def load_snapshot_agent(snapshot_path: Path, device: torch.device):
    # Import likely Rover modules before unpickling snapshots.
    import agent.rover  # noqa: F401
    import agent.rover_nystrom  # noqa: F401
    import agent.rover_nystrom_debug  # noqa: F401

    payload = torch.load(snapshot_path, map_location=device, weights_only=False)
    agent = payload["agent"] if isinstance(payload, dict) and "agent" in payload else payload
    agent.device = str(device)

    compute_dtype = getattr(agent, "compute_dtype", torch.float32)
    if isinstance(compute_dtype, torch.dtype):
        torch.set_default_dtype(compute_dtype)

    for value in vars(agent).values():
        if isinstance(value, torch.nn.Module):
            if isinstance(compute_dtype, torch.dtype):
                value.to(device=device, dtype=compute_dtype)
            else:
                value.to(device=device)
            value.eval()
    if hasattr(agent, "train"):
        agent.train(False)
    return agent


def select_encoder(agent, source: str):
    encoder = getattr(agent, source, None)
    if encoder is None:
        raise AttributeError(f"Snapshot agent does not have {source!r}")
    encoder.eval()
    return encoder


def append_zero_feature_column(tensor: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros(*tensor.shape[:-1], 1, device=tensor.device, dtype=tensor.dtype)
    return torch.cat([tensor, zero], dim=-1)


def encode_observations(agent, encoder, obs: torch.Tensor, batch_size: int, project: bool = True) -> torch.Tensor:
    chunks = []
    try:
        encoder_dtype = next(encoder.parameters()).dtype
    except StopIteration:
        encoder_dtype = obs.dtype
    with torch.no_grad():
        for start in range(0, obs.shape[0], batch_size):
            batch = obs[start:start + batch_size].to(dtype=encoder_dtype)
            chunks.append(agent._encode_with_module(encoder, batch, project=project))
    return torch.cat(chunks, dim=0)


def encode_fixed_dataset(agent, encoder, transitions, batch_size: int):
    obs, action, _, _, next_obs = transitions
    phi_obs = encode_observations(agent, encoder, obs, batch_size=batch_size, project=True)
    phi_next = encode_observations(agent, encoder, next_obs, batch_size=batch_size, project=True)
    psi = agent._encode_state_action(phi_obs, action)
    return {
        "phi_obs": append_zero_feature_column(phi_obs),
        "phi_next": append_zero_feature_column(phi_next),
        "psi": append_zero_feature_column(psi),
        "action": action.detach().cpu().numpy().reshape(-1),
    }


def pairwise_distance(X: torch.Tensor, Y: torch.Tensor, distance_norm: str) -> torch.Tensor:
    if distance_norm == "l1":
        return torch.cdist(X, Y, p=1)
    x_norm = (X * X).sum(dim=1, keepdim=True)
    y_norm = (Y * Y).sum(dim=1, keepdim=True).T
    squared = torch.clamp(x_norm + y_norm - 2.0 * (X @ Y.T), min=0.0)
    return torch.sqrt(squared)


def gaussian_kernel(X: torch.Tensor, Y: torch.Tensor, sigma: float, distance_norm: str) -> torch.Tensor:
    sigma = max(float(sigma), 1e-12)
    distance = pairwise_distance(X, Y, distance_norm)
    return torch.exp(-(distance * distance) / (2.0 * sigma * sigma))


def uniform_indices(n_items: int, max_items: int) -> torch.Tensor:
    if n_items <= max_items:
        return torch.arange(n_items, dtype=torch.long)
    return torch.round(torch.linspace(0, n_items - 1, max_items)).long()


def sample_pair_indices(n_x: int, n_y: int, n_pairs: int, same_matrix: bool, rng: np.random.Generator):
    x_idx = rng.integers(0, n_x, size=n_pairs)
    y_idx = rng.integers(0, n_y, size=n_pairs)
    if same_matrix and n_x == n_y:
        same = x_idx == y_idx
        while same.any():
            y_idx[same] = rng.integers(0, n_y, size=int(same.sum()))
            same = x_idx == y_idx
    return torch.from_numpy(x_idx).long(), torch.from_numpy(y_idx).long()


def sample_distances_and_similarities(
    X: torch.Tensor,
    Y: torch.Tensor,
    sigma: float,
    distance_norm: str,
    n_pairs: int,
    same_matrix: bool,
    rng: np.random.Generator,
):
    n_pairs = min(n_pairs, int(X.shape[0]) * int(Y.shape[0]))
    x_idx, y_idx = sample_pair_indices(X.shape[0], Y.shape[0], n_pairs, same_matrix, rng)
    x_idx = x_idx.to(X.device)
    y_idx = y_idx.to(Y.device)
    diff = X[x_idx] - Y[y_idx]
    ord_value = 1 if distance_norm == "l1" else 2
    distances = torch.linalg.vector_norm(diff, ord=ord_value, dim=1)
    similarities = torch.exp(-(distances * distances) / (2.0 * float(sigma) * float(sigma)))
    return distances.detach().cpu().numpy(), similarities.detach().cpu().numpy()


def neighbor_stats(
    X: torch.Tensor,
    Y: torch.Tensor,
    sigma: float,
    distance_norm: str,
    max_rows: int,
    batch_size: int,
    same_matrix: bool,
):
    row_idx = uniform_indices(X.shape[0], max_rows).to(X.device)
    nearest = []
    row_mean = []
    effective_neighbors = []
    with torch.no_grad():
        for start in range(0, row_idx.numel(), batch_size):
            idx = row_idx[start:start + batch_size]
            K = gaussian_kernel(X[idx], Y, sigma, distance_norm)
            if same_matrix and X.shape[0] == Y.shape[0]:
                matches = idx[:, None] == torch.arange(Y.shape[0], device=Y.device)[None, :]
                K = K.masked_fill(matches, -float("inf"))
            nearest.append(K.max(dim=1).values)
            finite = torch.where(torch.isfinite(K), K, torch.zeros_like(K))
            row_mean.append(finite.mean(dim=1))
            effective_neighbors.append(finite.sum(dim=1))
    return {
        "nearest": torch.cat(nearest).detach().cpu().numpy(),
        "row_mean": torch.cat(row_mean).detach().cpu().numpy(),
        "effective_neighbors": torch.cat(effective_neighbors).detach().cpu().numpy(),
    }


def percentiles(values: np.ndarray, qs=(0, 1, 5, 10, 15, 25, 50, 75, 85, 90, 95, 99, 100)):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    return {str(q): float(np.percentile(values, q)) for q in qs}


def plot_heatmap(K: np.ndarray, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(K, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="kernel similarity")
    ax.set_title(title)
    ax.set_xlabel("Y index")
    ax.set_ylabel("X index")
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_histogram(values: np.ndarray, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.hist(values, bins=80, range=(0.0, 1.0), color="#2563eb", alpha=0.82)
    ax.axvline(np.mean(values), color="black", linestyle="--", linewidth=1.2, label=f"mean={np.mean(values):.3g}")
    ax.axvline(np.median(values), color="#dc2626", linestyle=":", linewidth=1.4, label=f"median={np.median(values):.3g}")
    ax.set_title(title)
    ax.set_xlabel("kernel similarity")
    ax.set_ylabel("sampled pair count")
    ax.legend()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_similarity_vs_distance(distances: np.ndarray, similarities: np.ndarray, sigma: float, distance_norm: str, title: str, output_path: Path) -> None:
    order = np.argsort(distances)
    sorted_dist = distances[order]
    theoretical = np.exp(-(sorted_dist * sorted_dist) / (2.0 * sigma * sigma))

    max_scatter = min(8000, distances.size)
    scatter_idx = np.linspace(0, distances.size - 1, max_scatter).astype(np.int64)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.scatter(distances[scatter_idx], similarities[scatter_idx], s=4, alpha=0.18, color="#2563eb", label="sampled pairs")
    ax.plot(sorted_dist, theoretical, color="black", linewidth=2.0, label="exp(-d^2 / 2 sigma^2)")
    ax.axvline(sigma, color="#dc2626", linestyle="--", linewidth=1.2, label=f"sigma={sigma:g}")
    ax.set_title(title)
    ax.set_xlabel(f"{distance_norm.upper()} distance in encoded space")
    ax.set_ylabel("kernel similarity")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_neighbor_summary(stats: dict, title: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    entries = [
        ("nearest", "nearest non-self similarity"),
        ("row_mean", "mean row similarity"),
        ("effective_neighbors", "sum of row similarities"),
    ]
    for ax, (key, label) in zip(axes, entries):
        values = stats[key]
        ax.hist(values, bins=50, color="#16a34a", alpha=0.82)
        ax.axvline(np.median(values), color="black", linestyle="--", linewidth=1.1, label=f"median={np.median(values):.3g}")
        ax.set_title(label)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_reference_xy_similarity(points: np.ndarray, similarities: np.ndarray, sigma: float, output_path: Path) -> None:
    if points is None or points.shape[0] != similarities.shape[1] or similarities.shape[1] == 0:
        return

    center = np.median(points, axis=0)
    ref_indices = [0, int(np.argmin(np.sum((points - center) ** 2, axis=1)))]
    titles = ["first point", "center-nearest point"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for row_idx, (ax, ref_idx, title) in enumerate(zip(axes, ref_indices, titles)):
        sc = ax.scatter(points[:, 0], points[:, 1], c=similarities[row_idx], s=16, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.scatter(points[ref_idx, 0], points[ref_idx, 1], marker="*", s=160, c="white", edgecolors="black")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{title}, sigma={sigma:g}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def analyze_kernel_target(
    name: str,
    X: torch.Tensor,
    Y: torch.Tensor,
    sigma: float,
    distance_norm: str,
    output_dir: Path,
    same_matrix: bool,
    rng: np.random.Generator,
    max_heatmap_points: int,
    sample_pairs: int,
    neighbor_rows: int,
    batch_size: int,
    xy_points: np.ndarray | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    heat_x = uniform_indices(X.shape[0], max_heatmap_points).to(X.device)
    heat_y = heat_x if same_matrix and X.shape[0] == Y.shape[0] else uniform_indices(Y.shape[0], max_heatmap_points).to(Y.device)
    with torch.no_grad():
        heatmap = gaussian_kernel(X[heat_x], Y[heat_y], sigma, distance_norm).detach().cpu().numpy()

    distances, similarities = sample_distances_and_similarities(
        X, Y, sigma=sigma, distance_norm=distance_norm, n_pairs=sample_pairs, same_matrix=same_matrix, rng=rng
    )
    neighbors = neighbor_stats(
        X, Y, sigma=sigma, distance_norm=distance_norm, max_rows=neighbor_rows, batch_size=batch_size, same_matrix=same_matrix
    )

    stem = f"{name}_sigma_{sigma:g}".replace(".", "p")
    plot_heatmap(
        heatmap,
        title=f"{name} Gaussian kernel heatmap, sigma={sigma:g}",
        output_path=output_dir / f"{stem}_heatmap.png",
    )
    plot_histogram(
        similarities,
        title=f"{name} sampled kernel values, sigma={sigma:g}",
        output_path=output_dir / f"{stem}_histogram.png",
    )
    plot_similarity_vs_distance(
        distances,
        similarities,
        sigma=sigma,
        distance_norm=distance_norm,
        title=f"{name} similarity vs encoded {distance_norm.upper()} distance",
        output_path=output_dir / f"{stem}_similarity_vs_distance.png",
    )
    plot_neighbor_summary(
        neighbors,
        title=f"{name} row summaries, sigma={sigma:g}",
        output_path=output_dir / f"{stem}_neighbor_summary.png",
    )

    if xy_points is not None and same_matrix:
        center = np.median(xy_points, axis=0)
        ref_indices = torch.tensor(
            [0, int(np.argmin(np.sum((xy_points - center) ** 2, axis=1)))],
            dtype=torch.long,
            device=X.device,
        )
        with torch.no_grad():
            ref_similarities = gaussian_kernel(X[ref_indices], X, sigma, distance_norm).detach().cpu().numpy()
        plot_reference_xy_similarity(
            xy_points,
            ref_similarities,
            sigma=sigma,
            output_path=output_dir / f"{stem}_xy_reference_similarity.png",
        )

    return {
        "target": name,
        "sigma": float(sigma),
        "distance_norm": distance_norm,
        "shape": [int(X.shape[0]), int(Y.shape[0])],
        "heatmap_shape": list(heatmap.shape),
        "sampled_distance_percentiles": percentiles(distances),
        "sampled_similarity_percentiles": percentiles(similarities),
        "nearest_similarity_percentiles": percentiles(neighbors["nearest"]),
        "row_mean_similarity_percentiles": percentiles(neighbors["row_mean"]),
        "effective_neighbor_percentiles": percentiles(neighbors["effective_neighbors"]),
        "sampled_similarity_mean": float(np.mean(similarities)),
        "sampled_similarity_median": float(np.median(similarities)),
        "nearest_similarity_median": float(np.median(neighbors["nearest"])),
        "effective_neighbors_median": float(np.median(neighbors["effective_neighbors"])),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a Rover encoder, build a fixed PointMaze all-actions dataset, and diagnose Gaussian kernel sigma."
    )
    parser.add_argument("--snapshot", type=Path, required=True, help="Path to snapshot.pt or snapshot_<step>.pt")
    parser.add_argument("--config-name", type=str, default="pretrain/pretrain_rover_pointmaze_umaze_1")
    parser.add_argument("--output-dir", type=Path, default=Path("gaussian_kernel_test"))
    parser.add_argument("--n-states", type=int, default=256, help="Number of reachable XY grid states. Total transitions = n_states * n_actions.")
    parser.add_argument("--sigma", type=float, nargs="+", required=True, help="One or more Gaussian kernel sigma values to test.")
    parser.add_argument("--encoder-source", choices=("policy_encoder", "encoder"), default="policy_encoder")
    parser.add_argument(
        "--distance-norm",
        choices=("auto", "l1", "l2"),
        default="auto",
        help="Distance used inside the Gaussian. auto uses snapshot agent.mode.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=("unique_state", "state", "transition", "action"),
        default=("unique_state", "transition", "action"),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encode-batch-size", type=int, default=512)
    parser.add_argument("--kernel-batch-size", type=int, default=128)
    parser.add_argument("--max-heatmap-points", type=int, default=1200)
    parser.add_argument("--sample-pairs", type=int, default=50_000)
    parser.add_argument("--neighbor-rows", type=int, default=512)
    parser.add_argument("--border-margin", type=float, default=0.05)
    parser.add_argument("--oversample", type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    cfg = compose_pretrain_cfg(args.config_name, seed=args.seed)
    env = make_pretrain_env(cfg)
    env.reset()

    agent = load_snapshot_agent(args.snapshot, device)
    encoder = select_encoder(agent, args.encoder_source)
    n_actions = int(getattr(agent, "n_actions"))
    distance_norm = str(getattr(agent, "mode", "l2")).lower() if args.distance_norm == "auto" else args.distance_norm
    if distance_norm not in ("l1", "l2"):
        raise ValueError(f"Unsupported distance norm {distance_norm!r}; expected 'l1' or 'l2'.")

    original_subsamples = getattr(agent, "subsamples", None)
    agent.subsamples = int(args.n_states * n_actions)
    helper = PointMazeNystromDebugHelper(border_margin=args.border_margin, oversample=args.oversample)
    helper.attach_env(env)
    transitions = helper.build_subsample_batch(agent)
    agent.subsamples = original_subsamples

    encoded = encode_fixed_dataset(agent, encoder, transitions, batch_size=args.encode_batch_size)
    action = encoded["action"]
    xy_points = np.asarray(helper.fixed_xy_points, dtype=np.float32)

    targets = {}
    if "unique_state" in args.targets:
        targets["unique_state"] = {
            "X": encoded["phi_obs"][::n_actions],
            "Y": encoded["phi_obs"][::n_actions],
            "same": True,
            "xy": xy_points,
        }
    if "state" in args.targets:
        targets["state"] = {
            "X": encoded["phi_obs"],
            "Y": encoded["phi_obs"],
            "same": True,
            "xy": None,
        }
    if "transition" in args.targets:
        targets["transition"] = {
            "X": encoded["phi_obs"],
            "Y": encoded["phi_next"],
            "same": False,
            "xy": None,
        }
    if "action" in args.targets:
        targets["action"] = {
            "X": encoded["psi"],
            "Y": encoded["psi"],
            "same": True,
            "xy": None,
        }

    print(
        f"Fixed dataset: {args.n_states} XY states x {n_actions} actions = "
        f"{encoded['phi_obs'].shape[0]} transitions"
    )
    print(f"Action counts: {np.bincount(action, minlength=n_actions).tolist()}")
    print(
        f"Gaussian kernel tested here is K(x, y) = exp(-d(x,y)^2 / (2 sigma^2)), "
        f"with d = {distance_norm.upper()} distance."
    )

    all_stats = {
        "snapshot": str(args.snapshot),
        "config_name": args.config_name,
        "encoder_source": args.encoder_source,
        "n_states": int(args.n_states),
        "n_actions": n_actions,
        "n_transitions": int(encoded["phi_obs"].shape[0]),
        "distance_norm": distance_norm,
        "feature_shapes": {key: list(value.shape) for key, value in encoded.items() if isinstance(value, torch.Tensor)},
        "sigmas": [float(sigma) for sigma in args.sigma],
        "targets": [],
    }

    for sigma in args.sigma:
        sigma_dir = args.output_dir / f"sigma_{sigma:g}".replace(".", "p")
        for target_name, target in targets.items():
            print(f"Analyzing target={target_name}, sigma={sigma:g}, shape={tuple(target['X'].shape)} x {tuple(target['Y'].shape)}")
            stats = analyze_kernel_target(
                name=target_name,
                X=target["X"],
                Y=target["Y"],
                sigma=float(sigma),
                distance_norm=distance_norm,
                output_dir=sigma_dir,
                same_matrix=bool(target["same"]),
                rng=rng,
                max_heatmap_points=args.max_heatmap_points,
                sample_pairs=args.sample_pairs,
                neighbor_rows=args.neighbor_rows,
                batch_size=args.kernel_batch_size,
                xy_points=target["xy"],
            )
            all_stats["targets"].append(stats)

    stats_path = args.output_dir / "gaussian_kernel_stats.json"
    with stats_path.open("w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Saved diagnostics to {args.output_dir}")
    print(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
