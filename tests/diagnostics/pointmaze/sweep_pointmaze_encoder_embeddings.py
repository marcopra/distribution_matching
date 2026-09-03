#!/usr/bin/env python3
"""Train PointMaze Rover encoders on one cached synthetic pixel dataset.

Example
-------
python tests/diagnostics/pointmaze/sweep_pointmaze_encoder_embeddings.py \
    --feature-dims 16 32 64 128 \
    --updates 1000 \
    --n-points 32000 \
    --batch-size 1024 \
    --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Sequence

os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from tqdm.rich import tqdm

import gym_env
import utils
from agent.rover_pointmaze_debug import RoverAgent
from tests.diagnostics.pointmaze.synthetic_workflow_utils import (
    arrays_to_tensors,
    fixed_encoder_indices,
    load_dataset,
    save_dataset,
    save_encoder_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep PointMaze Rover encoder latent sizes on the fixed Nyström debug dataset."
    )
    parser.add_argument("--feature-dims", type=int, nargs="+", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/outputs/pointmaze/synthetic_workflow"),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Shared cached dataset directory. Defaults to OUTPUT_DIR/dataset.",
    )
    parser.add_argument(
        "--config-name",
        default="pretrain_parallel/pretrain_pointmaze_umaze_1_pixels",
    )
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--n-points",
        type=int,
        default=32000,
        help=(
            "Requested total dataset transitions. Each reachable XY state contributes one "
            "transition per action, so target XY states = n_points / n_actions. Equispaced "
            "feasibility may adjust the final count slightly."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("l1", "l2"),
        default=None,
        help="Encoder output normalization. Defaults to mode from agent config.",
    )
    parser.add_argument(
        "--grayscale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render one-channel observations by default; use --no-grayscale for RGB.",
    )
    parser.add_argument(
        "--regenerate-dataset",
        action="store_true",
        help="Replace cached synthetic transitions. Default reuses and verifies them.",
    )
    parser.add_argument("--checkpoint-fractions", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Refresh metrics and figures every N updates; 0 saves only checkpoints/final output.",
    )
    parser.add_argument("--border-margin", type=float, default=None)
    parser.add_argument("--oversample", type=float, default=None)
    parser.add_argument(
        "--exact-grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use an equally spaced feasible XY lattice; enabled by default.",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=4000,
        help="Uniformly subsample XY states for PCA/t-SNE scatter if fixed grid is larger.",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="Base t-SNE perplexity. It is clipped to fit the number of plotted points.",
    )
    return parser.parse_args()


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


def compose_pretrain_cfg(config_name: str | Path, seed: int):
    cfg = OmegaConf.load(resolve_config_path(config_name))
    defaults = OmegaConf.to_container(cfg.get("defaults", []), resolve=False)
    env_default = None
    agent_default = None
    for item in defaults:
        if isinstance(item, dict) and "/env" in item:
            env_default = item["/env"]
        elif isinstance(item, dict) and "env" in item:
            env_default = item["env"]
        elif isinstance(item, dict) and "/agent" in item:
            agent_default = item["/agent"]
        elif isinstance(item, dict) and "agent" in item:
            agent_default = item["agent"]
    if env_default is None:
        raise ValueError(f"Could not recover env default from {config_name}")
    if agent_default is None:
        raise ValueError(f"Could not recover agent default from {config_name}")

    env_cfg = OmegaConf.load(resolve_group_path(env_default, "env"))
    agent_cfg = OmegaConf.load(resolve_group_path(agent_default, "agent"))
    merged = OmegaConf.merge({"agent": agent_cfg}, env_cfg, cfg)
    merged.seed = int(seed)
    return merged


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


def build_agent(cfg, env, feature_dim: int, args: argparse.Namespace) -> RoverAgent:
    obs_spec = gym_env.observation_spec(env)
    action_spec = gym_env.action_spec(env)
    action_shape = (action_spec.num_values,) if hasattr(action_spec, "num_values") else action_spec.shape
    n_actions = int(action_shape[0])
    if hasattr(args, "n_points"):
        n_transitions = int(args.n_points)
    elif hasattr(args, "n_states"):
        # Compatibility for PMD and cached/live diagnostic scripts importing this helper.
        n_transitions = int(args.n_states) * n_actions
    else:
        raise AttributeError("build_agent requires args.n_points or args.n_states")

    agent_cfg = OmegaConf.to_container(cfg.agent, resolve=False)
    agent_cfg.pop("_target_", None)
    agent_cfg.update(
        {
            "obs_type": cfg.obs_type,
            "obs_shape": obs_spec.shape if obs_spec.shape else (1,),
            "action_shape": action_shape,
            "grayscale": bool(cfg.grayscale),
            "discount": float(cfg.discount),
            "feature_dim": int(feature_dim),
            "subsamples": n_transitions,
            "batch_size": int(args.batch_size),
            "batch_size_actor": n_transitions,
            "debug_fixed_dataset_updates": True,
            "nystrom_exact_grid": bool(args.exact_grid or agent_cfg.get("nystrom_exact_grid", False)),
            "device": args.device,
            "total_train_steps": max(1, int(args.updates)),
            "num_expl_steps": int(cfg.num_seed_frames // cfg.action_repeat),
            "use_tb": False,
            "use_wandb": False,
        }
    )
    if args.border_margin is not None:
        agent_cfg["nystrom_grid_border_margin"] = float(args.border_margin)
    if args.oversample is not None:
        agent_cfg["nystrom_grid_oversample"] = float(args.oversample)

    agent = RoverAgent(**agent_cfg)
    agent.insert_env(env)
    agent.train(True)
    return agent


def checkpoint_steps(updates: int, fractions: Sequence[float]) -> List[int]:
    steps = sorted({int(round(float(frac) * updates)) for frac in fractions})
    steps = [step for step in steps if 0 <= step <= updates]
    if 0 not in steps:
        steps.insert(0, 0)
    if updates not in steps:
        steps.append(updates)
    return steps


def create_shared_dataset(cfg, args: argparse.Namespace, feature_dim: int):
    """Render fixed transitions once; later feature dimensions never rerender them."""
    env = make_pretrain_env(cfg)
    try:
        env.reset(seed=int(args.seed))
        agent = build_agent(cfg, env, feature_dim, args)
        obs, action, reward, discount, next_obs = agent.nystrom_debug.build_subsample_batch(agent)
        xy_points = np.asarray(agent.nystrom_debug.fixed_xy_points, dtype=np.float32).reshape(-1, 2)
        stored_obs = obs.to(torch.uint8) if agent.obs_type == "pixels" else obs
        stored_next_obs = next_obs.to(torch.uint8) if agent.obs_type == "pixels" else next_obs
        arrays = {
            "obs": stored_obs.detach().cpu().numpy(),
            "action": action.detach().cpu().numpy(),
            "reward": reward.detach().cpu().numpy(),
            "discount": discount.detach().cpu().numpy(),
            "next_obs": stored_next_obs.detach().cpu().numpy(),
            "xy": xy_points,
        }
        metadata = {
            "config_name": str(args.config_name),
            "seed": int(args.seed),
            "n_actions": int(agent.n_actions),
            "requested_n_points": int(args.n_points),
            "requested_n_states": int(round(args.n_points / agent.n_actions)),
            "actual_n_states": int(xy_points.shape[0]),
            "actual_n_points": int(stored_obs.shape[0]),
            "obs_shape": list(agent.obs_shape),
            "obs_type": str(agent.obs_type),
            "grayscale": bool(agent.grayscale),
            "resolution": int(cfg.resolution),
            "frame_stack": int(cfg.frame_stack),
            "action_repeat": int(cfg.action_repeat),
            "exact_grid": bool(args.exact_grid),
            "border_margin": args.border_margin,
            "oversample": args.oversample,
            "grid_stats": agent.nystrom_debug.fixed_plot_stats,
        }
        return arrays, metadata
    finally:
        env.close()


def prepare_shared_dataset(cfg, args: argparse.Namespace):
    dataset_dir = args.dataset_dir if args.dataset_dir is not None else args.output_dir / "dataset"
    dataset_dir = dataset_dir.resolve()
    dataset_file = dataset_dir / "transitions.npz"
    if dataset_file.exists() and not args.regenerate_dataset:
        arrays, metadata = load_dataset(dataset_dir)
    else:
        arrays, metadata = create_shared_dataset(cfg, args, int(args.feature_dims[0]))
        metadata = save_dataset(dataset_dir, arrays, metadata)

    expected = {
        "config_name": str(args.config_name),
        "seed": int(args.seed),
        "requested_n_points": int(args.n_points),
        "grayscale": bool(args.grayscale),
    }
    mismatches = [
        f"{key}: cached={metadata.get(key)!r}, requested={value!r}"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "Cached dataset does not match request; use --regenerate-dataset: "
            + "; ".join(mismatches)
        )
    return arrays, metadata


def update_encoders_with_metrics(agent: RoverAgent, obs, action, next_obs, reward) -> Dict[str, float]:
    """Run encoder update and force loss values into returned metrics.

    RoverAgent.update_encoders only includes most losses in its return dict when
    logging flags are enabled. This script has no TensorBoard/W&B writer, so the
    flags only control metric collection here.
    """
    old_use_tb = agent.use_tb
    old_use_wandb = agent.use_wandb
    agent.use_tb = True
    agent.use_wandb = False
    try:
        metrics = agent.update_encoders(obs, action, next_obs, reward)
    finally:
        agent.use_tb = old_use_tb
        agent.use_wandb = old_use_wandb
    return metrics


def uniform_indices(n_items: int, max_items: int) -> np.ndarray:
    if n_items <= max_items:
        return np.arange(n_items, dtype=np.int64)
    return np.round(np.linspace(0, n_items - 1, max_items)).astype(np.int64)


def encode_unique_states(agent: RoverAgent, full_obs: torch.Tensor, n_actions: int, batch_size: int) -> np.ndarray:
    unique_obs = full_obs[::n_actions]
    chunks = []
    agent.encoder.eval()
    with torch.no_grad():
        for start in range(0, unique_obs.shape[0], batch_size):
            batch = unique_obs[start : start + batch_size]
            chunks.append(agent.aug_and_encode(batch, project=True).detach().float().cpu())
    agent.encoder.train(True)
    return torch.cat(chunks, dim=0).numpy()


def pca_projection(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    if float(np.linalg.norm(centered)) <= 1e-12:
        return np.zeros((embeddings.shape[0], 2), dtype=np.float32)
    if embeddings.shape[1] >= 2 and embeddings.shape[0] >= 2:
        return PCA(n_components=2, random_state=0).fit_transform(embeddings)
    if embeddings.shape[1] == 1:
        return np.column_stack([embeddings[:, 0], np.zeros(embeddings.shape[0], dtype=embeddings.dtype)])
    return np.zeros((embeddings.shape[0], 2), dtype=np.float32)


def tsne_projection(embeddings: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    if embeddings.shape[0] < 3 or float(np.linalg.norm(centered)) <= 1e-12:
        return np.zeros((embeddings.shape[0], 2), dtype=np.float32)
    effective_perplexity = min(float(perplexity), max(1.0, (embeddings.shape[0] - 1) / 3.0))
    return TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=int(seed),
    ).fit_transform(embeddings)


def save_metrics(path: Path, metrics_log: Dict[str, List[float]]) -> None:
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in metrics_log.items()})


def save_dataset_xy_figure(path: Path, xy_points: np.ndarray, metadata: dict) -> None:
    """Save full generated XY lattice, without plot subsampling."""
    xy_points = np.asarray(xy_points, dtype=np.float32).reshape(-1, 2)
    if xy_points.shape[0] == 0:
        raise ValueError("Cannot plot empty PointMaze XY dataset")
    grid_stats = metadata.get("grid_stats") or {}
    grid_spacing = grid_stats.get("grid_spacing") or {}
    spacing_x = grid_spacing.get("x", float("nan"))
    spacing_y = grid_spacing.get("y", float("nan"))

    fig, ax = plt.subplots(figsize=(6.4, 6.0), constrained_layout=True)
    marker_size = max(1.0, min(14.0, 24000.0 / xy_points.shape[0]))
    ax.scatter(xy_points[:, 0], xy_points[:, 1], s=marker_size, linewidths=0, alpha=0.9)
    ax.scatter(
        xy_points[0, 0],
        xy_points[0, 1],
        marker="*",
        s=130,
        color="tab:red",
        edgecolors="black",
        linewidths=0.7,
        label="first/start point",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        "PointMaze generated dataset XY locations\n"
        f"{xy_points.shape[0]} states × {metadata['n_actions']} actions = "
        f"{xy_points.shape[0] * metadata['n_actions']} points; "
        f"dx={spacing_x:.5g}, dy={spacing_y:.5g}"
    )
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.legend(loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_evolution_figure(
    path: Path,
    checkpoints: Sequence[int],
    embeddings_by_step: Dict[int, np.ndarray],
    xy_points: np.ndarray,
    metrics_log: Dict[str, List[float]],
    feature_dim: int,
    max_plot_points: int,
    projection_name: str,
    perplexity: float,
    seed: int,
) -> None:
    n_cols = len(checkpoints)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.3 * n_cols, 7.2), squeeze=False)
    plot_idx = uniform_indices(xy_points.shape[0], max_plot_points)
    colors = xy_points[plot_idx, 0] + 0.37 * xy_points[plot_idx, 1]

    for col, step in enumerate(checkpoints):
        ax = axes[0, col]
        embeddings = embeddings_by_step.get(int(step))
        if embeddings is None:
            ax.text(0.5, 0.5, "pending", ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"step {step}")
            continue
        plot_embeddings = embeddings[plot_idx]
        if projection_name == "pca":
            projected = pca_projection(plot_embeddings)
            x_label, y_label = "PC1", "PC2"
        elif projection_name == "tsne":
            projected = tsne_projection(plot_embeddings, perplexity=perplexity, seed=seed + int(step))
            x_label, y_label = "t-SNE 1", "t-SNE 2"
        else:
            raise ValueError(f"Unknown projection {projection_name!r}")
        scatter = ax.scatter(
            projected[:, 0],
            projected[:, 1],
            c=colors,
            s=8,
            cmap="viridis",
            alpha=0.82,
            linewidths=0,
        )
        ax.set_title(f"step {step}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if col == n_cols - 1:
            fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="XY color")

    loss_ax = axes[1, 0]
    steps = np.asarray(metrics_log["step"], dtype=np.float32)
    if steps.size > 0:
        loss_ax.plot(steps, metrics_log["transition_loss"], label="transition", linewidth=2.0)
        loss_ax.plot(steps, metrics_log["contrastive_loss"], label="contrastive", linewidth=1.7)
        if np.any(np.asarray(metrics_log["curl_loss"], dtype=np.float32) != 0.0):
            loss_ax.plot(steps, metrics_log["curl_loss"], label="curl", linewidth=1.2)
    loss_ax.set_title("encoder losses")
    loss_ax.set_xlabel("update")
    loss_ax.set_ylabel("loss")
    loss_ax.grid(alpha=0.25, linewidth=0.5)
    if steps.size > 0:
        loss_ax.legend(loc="best")

    for col in range(1, n_cols):
        axes[1, col].axis("off")

    fig.suptitle(
        f"PointMaze fixed-data encoder {projection_name.upper()} | feature_dim={feature_dim}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_projection_figures(
    experiment_dir: Path,
    checkpoints: Sequence[int],
    embeddings_by_step: Dict[int, np.ndarray],
    xy_points: np.ndarray,
    metrics_log: Dict[str, List[float]],
    feature_dim: int,
    args: argparse.Namespace,
) -> None:
    for projection_name in ("pca", "tsne"):
        save_evolution_figure(
            experiment_dir / f"embedding_evolution_{projection_name}.png",
            checkpoints,
            embeddings_by_step,
            xy_points,
            metrics_log,
            feature_dim,
            args.max_plot_points,
            projection_name=projection_name,
            perplexity=args.tsne_perplexity,
            seed=args.seed,
        )


def write_config(path: Path, args: argparse.Namespace, feature_dim: int, n_actions: int, actual_n_states: int) -> None:
    payload = {
        "feature_dim": int(feature_dim),
        "n_actions": int(n_actions),
        "requested_n_points": int(args.n_points),
        "requested_n_states": int(round(args.n_points / n_actions)),
        "actual_n_states": int(actual_n_states),
        "actual_n_points": int(actual_n_states * n_actions),
        "transitions": int(actual_n_states * n_actions),
        "updates": int(args.updates),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "device": str(args.device),
        "mode": str(args.mode),
        "dataset_dir": str(
            (args.dataset_dir if args.dataset_dir is not None else args.output_dir / "dataset").resolve()
        ),
        "config_name": str(args.config_name),
        "checkpoint_fractions": [float(value) for value in args.checkpoint_fractions],
        "save_every": int(args.save_every),
        "tsne_perplexity": float(args.tsne_perplexity),
    }
    path.write_text(json.dumps(payload, indent=2))


def run_one_feature_dim(cfg, args: argparse.Namespace, feature_dim: int, arrays, dataset_metadata) -> None:
    env = make_pretrain_env(cfg)
    try:
        utils.set_seed_everywhere(args.seed)
        torch.manual_seed(args.seed)
        env.reset(seed=int(args.seed))
        agent = build_agent(cfg, env, feature_dim, args)
        n_actions = int(agent.n_actions)
        tensors = arrays_to_tensors(arrays, args.device, agent.compute_dtype)
        indices = fixed_encoder_indices(tensors["obs"].shape[0], args.batch_size, n_actions)
        index = torch.as_tensor(indices, dtype=torch.long, device=args.device)
        obs = tensors["obs"].index_select(0, index)
        action = tensors["action"].index_select(0, index)
        next_obs = tensors["next_obs"].index_select(0, index)
        reward = tensors["reward"].index_select(0, index)
        full_obs = tensors["obs"]
        xy_points = np.asarray(arrays["xy"], dtype=np.float32).reshape(-1, 2)
        if full_obs.shape[0] != xy_points.shape[0] * n_actions:
            raise RuntimeError(
                f"Fixed dataset mismatch: full_obs={full_obs.shape[0]}, xy={xy_points.shape[0]}, n_actions={n_actions}"
            )

        experiment_dir = args.output_dir / f"featuredim_{feature_dim}"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        write_config(experiment_dir / "config.json", args, feature_dim, n_actions, xy_points.shape[0])
        np.savez_compressed(experiment_dir / "fixed_xy_points.npz", xy=xy_points)

        checkpoints = checkpoint_steps(args.updates, args.checkpoint_fractions)
        checkpoint_set = set(checkpoints)
        embeddings_by_step: Dict[int, np.ndarray] = {
            0: encode_unique_states(agent, full_obs, n_actions=n_actions, batch_size=args.batch_size)
        }
        metrics_log: Dict[str, List[float]] = {
            "step": [],
            "transition_loss": [],
            "contrastive_loss": [],
            "curl_loss": [],
            "embedding_sum_loss": [],
            "reward_loss": [],
        }

        metrics_path = experiment_dir / "metrics.npz"
        save_metrics(metrics_path, metrics_log)
        save_projection_figures(
            experiment_dir,
            checkpoints,
            embeddings_by_step,
            xy_points,
            metrics_log,
            feature_dim,
            args,
        )

        progress = tqdm(range(1, args.updates + 1), desc=f"feature_dim={feature_dim}")
        for step in progress:
            metrics = update_encoders_with_metrics(agent, obs, action, next_obs, reward)
            metrics_log["step"].append(step)
            for key in ("transition_loss", "contrastive_loss", "curl_loss", "embedding_sum_loss", "reward_loss"):
                metrics_log[key].append(float(metrics.get(key, 0.0)))

            should_snapshot = step in checkpoint_set
            should_refresh = args.save_every > 0 and step % args.save_every == 0
            if should_snapshot:
                embeddings_by_step[step] = encode_unique_states(
                    agent,
                    full_obs,
                    n_actions=n_actions,
                    batch_size=args.batch_size,
                )
                save_encoder_checkpoint(
                    experiment_dir / f"encoder_step_{step}.pt",
                    agent.encoder,
                    feature_dim=feature_dim,
                    obs_shape=agent.obs_shape,
                    mode=agent.mode,
                    grayscale=agent.grayscale,
                    dataset_checksum_value=dataset_metadata["checksum"],
                    training_updates=step,
                )
            if should_snapshot or should_refresh or step == args.updates:
                save_metrics(metrics_path, metrics_log)
                save_projection_figures(
                    experiment_dir,
                    checkpoints,
                    embeddings_by_step,
                    xy_points,
                    metrics_log,
                    feature_dim,
                    args,
                )
        save_encoder_checkpoint(
            experiment_dir / "encoder.pt",
            agent.encoder,
            feature_dim=feature_dim,
            obs_shape=agent.obs_shape,
            mode=agent.mode,
            grayscale=agent.grayscale,
            dataset_checksum_value=dataset_metadata["checksum"],
            training_updates=args.updates,
        )
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    if args.n_points <= 0:
        raise ValueError("--n-points must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    utils.set_seed_everywhere(args.seed)
    torch.manual_seed(args.seed)
    cfg = compose_pretrain_cfg(args.config_name, args.seed)
    cfg.grayscale = bool(args.grayscale)
    if args.mode is not None:
        cfg.agent.mode = str(args.mode)
    else:
        args.mode = str(cfg.agent.mode)
    arrays, dataset_metadata = prepare_shared_dataset(cfg, args)
    dataset_dir = (
        args.dataset_dir if args.dataset_dir is not None else args.output_dir / "dataset"
    ).resolve()
    dataset_plot = dataset_dir / "dataset_xy_locations.png"
    save_dataset_xy_figure(dataset_plot, arrays["xy"], dataset_metadata)
    output_plot = args.output_dir.resolve() / "dataset_xy_locations.png"
    if output_plot != dataset_plot:
        save_dataset_xy_figure(output_plot, arrays["xy"], dataset_metadata)
    print(f"Saved generated dataset XY plot to {dataset_plot} and {output_plot}")

    for feature_dim in args.feature_dims:
        run_one_feature_dim(cfg, args, feature_dim, arrays, dataset_metadata)

    print(f"Saved PointMaze encoder sweep outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
