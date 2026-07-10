#!/usr/bin/env python3
"""Train PointMaze Rover encoders on fixed debug data and plot embeddings.

Example
-------
python tests/diagnostics/pointmaze/sweep_pointmaze_encoder_embeddings.py \
    --feature-dims 2 8 16 32 64 \
    --updates 1000 \
    --n-states 1000 \
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
from agent.rover_nystrom_pointmaze_debug import RoverAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep PointMaze Rover encoder latent sizes on the fixed Nyström debug dataset."
    )
    parser.add_argument("--feature-dims", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("tests/outputs/pointmaze/encoder_latent_sweep"))
    parser.add_argument("--config-name", default="pretrain/pretrain_rover_pointmaze_umaze_1")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--n-states",
        type=int,
        default=1000,
        help="Number of fixed reachable XY states. Dataset has n_states * n_actions transitions.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-fractions", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    parser.add_argument(
        "--save-every",
        type=int,
        default=25,
        help="Refresh metrics and figure every N encoder updates.",
    )
    parser.add_argument("--border-margin", type=float, default=None)
    parser.add_argument("--oversample", type=float, default=None)
    parser.add_argument(
        "--exact-grid",
        action="store_true",
        help="Use exact feasible fixed-grid mode from PointMazeNystromDebugHelper.",
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
    n_transitions = int(args.n_states) * int(action_shape[0])

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


def fixed_encoder_tensors(agent: RoverAgent):
    obs, action, next_obs, reward = agent.nystrom_debug.fixed_encoder_batch(agent)
    full_obs, full_action, full_next_obs, full_reward = agent.nystrom_debug.fixed_actor_batch(agent)
    return (obs, action, next_obs, reward), (full_obs, full_action, full_next_obs, full_reward)


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
        "requested_n_states": int(args.n_states),
        "actual_n_states": int(actual_n_states),
        "transitions": int(actual_n_states * n_actions),
        "updates": int(args.updates),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "device": str(args.device),
        "config_name": str(args.config_name),
        "checkpoint_fractions": [float(value) for value in args.checkpoint_fractions],
        "save_every": int(args.save_every),
        "tsne_perplexity": float(args.tsne_perplexity),
    }
    path.write_text(json.dumps(payload, indent=2))


def run_one_feature_dim(cfg, args: argparse.Namespace, feature_dim: int) -> None:
    env = make_pretrain_env(cfg)
    try:
        env.reset()
        agent = build_agent(cfg, env, feature_dim, args)
        n_actions = int(agent.n_actions)
        train_batch, full_batch = fixed_encoder_tensors(agent)
        obs, action, next_obs, reward = train_batch
        full_obs, _, _, _ = full_batch
        xy_points = np.asarray(agent.nystrom_debug.fixed_xy_points, dtype=np.float32).reshape(-1, 2)
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
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    utils.set_seed_everywhere(args.seed)
    torch.manual_seed(args.seed)
    cfg = compose_pretrain_cfg(args.config_name, args.seed)

    for feature_dim in args.feature_dims:
        run_one_feature_dim(cfg, args, feature_dim)

    print(f"Saved PointMaze encoder sweep outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
