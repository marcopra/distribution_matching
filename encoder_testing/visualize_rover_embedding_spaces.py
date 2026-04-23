#!/usr/bin/env python3
"""Train low-dimensional ROVER encoders and visualize state/state-action geometry.

Examples
--------
python encoder_testing/visualize_rover_embedding_spaces.py \
    --dataset encoder_testing/outputs/multiplerooms_states \
    --output-dir encoder_testing/outputs/embedding_space_debug \
    --feature-dims 2 3 \
    --encoder-update-mode nystrom \
    --subsamples 100 \
    --epochs 10 \
    --batch-size 1024 \
    --curl-weight 0.001 \
    --device cuda \
    --config-name pretrain/pretrain_rover_multiplerooms2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from tqdm.rich import tqdm

import utils
from encoder_testing.test_rover_nystrom_encoder import (
    DYN_LOSS_CHOICES,
    build_agent,
    compose_cfg,
    copied_update_encoders,
    copied_update_encoders_nystrom,
    load_dataset,
    make_env_from_cfg,
    resolve_dataset_npz,
    save_loss_curves,
    tensor_from_numpy,
    compute_epoch_batches,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ACTION_TO_ARROW = {
    0: "↑",
    1: "↓",
    2: "←",
    3: "→",
}
ACTION_TO_COLOR = {
    0: "#D81B60",
    1: "#1E88E5",
    2: "#43A047",
    3: "#FB8C00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train low-dimensional ROVER encoders and visualize state / state-action embeddings."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the dataset folder produced by collect_multiplerooms_dataset.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where experiment outputs will be stored.",
    )
    parser.add_argument(
        "--config-name",
        default="pretrain/pretrain_rover_multiplerooms",
        help="Base config used to recover the default MultipleRooms environment.",
    )
    parser.add_argument(
        "--feature-dims",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Low-dimensional feature sizes to train and visualize. Default: 2 3.",
    )
    parser.add_argument(
        "--encoder-update-mode",
        choices=["nystrom", "full"],
        default="nystrom",
        help="Choose between copied update_encoders_nystrom and copied full update_encoders.",
    )
    parser.add_argument(
        "--dyn-loss",
        choices=DYN_LOSS_CHOICES,
        default="classic_contrastive",
        help="Dynamics loss used inside copied_update_encoders_nystrom. Only used in nystrom mode.",
    )
    parser.add_argument(
        "--subsamples",
        type=int,
        default=None,
        help="Number of fixed Nyström subsamples. Required only for nystrom mode.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of passes over the fixed dataset per experiment.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training mini-batch size.",
    )
    parser.add_argument(
        "--curl-weight",
        type=float,
        default=1e-3,
        help="Weight applied to the CURL term during training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for training and plot-point selection.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device, for example cpu or cuda.",
    )
    parser.add_argument(
        "--num-plot-points",
        type=int,
        default=8,
        help="How many transitions to visualize per experiment.",
    )
    parser.add_argument(
        "--plot-indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit dataset indices to visualize instead of auto-selecting them.",
    )
    return parser.parse_args()


def get_state_indices(dataset: Dict[str, np.ndarray]) -> np.ndarray:
    if "state_index" in dataset:
        return dataset["state_index"].astype(np.int64)
    if "proprio_observation" in dataset:
        return np.argmax(dataset["proprio_observation"], axis=1).astype(np.int64)
    raise KeyError("Dataset does not contain state_index or proprio_observation.")


def get_next_state_indices(dataset: Dict[str, np.ndarray]) -> np.ndarray:
    if "next_state_index" in dataset:
        return dataset["next_state_index"].astype(np.int64)
    if "next_proprio_observation" in dataset:
        return np.argmax(dataset["next_proprio_observation"], axis=1).astype(np.int64)
    raise KeyError("Dataset does not contain next_state_index or next_proprio_observation.")


def select_plot_indices(
    actions: np.ndarray,
    state_indices: np.ndarray,
    num_points: int,
    seed: int,
    explicit_indices: Sequence[int] | None = None,
) -> np.ndarray:
    if explicit_indices:
        return np.asarray(explicit_indices, dtype=np.int64)

    rng = np.random.default_rng(seed)
    actions = actions.astype(np.int64)
    state_indices = state_indices.astype(np.int64)
    unique_actions = sorted(np.unique(actions).tolist())
    per_action = max(1, num_points // max(len(unique_actions), 1))

    selected: List[int] = []
    for action in unique_actions:
        candidates = np.where(actions == action)[0]
        shuffled = rng.permutation(candidates)
        seen_states = set()
        for idx in shuffled:
            state_idx = int(state_indices[idx])
            if state_idx in seen_states:
                continue
            selected.append(int(idx))
            seen_states.add(state_idx)
            if len([x for x in selected if actions[x] == action]) >= per_action:
                break

    if len(selected) < num_points:
        remaining = [idx for idx in rng.permutation(len(actions)) if idx not in selected]
        selected.extend(int(idx) for idx in remaining[: num_points - len(selected)])

    return np.asarray(sorted(selected[:num_points]), dtype=np.int64)


def normalize_state_for_plot(agent, state_embeddings: torch.Tensor) -> torch.Tensor:
    if agent.mode == "l1":
        return F.normalize(state_embeddings, p=2, dim=1, eps=1e-10)
    if agent.mode == "l2":
        return state_embeddings
    raise ValueError(f"Unsupported mode: {agent.mode}")


def compute_projected_state_action(
    agent,
    encoded_state_action: torch.Tensor,
    sub_obs: torch.Tensor | None,
    sub_action: torch.Tensor | None,
    sub_next_obs: torch.Tensor | None,
    encoder_update_mode: str,
) -> torch.Tensor:
    if encoder_update_mode == "nystrom":
        if sub_obs is None or sub_action is None or sub_next_obs is None:
            raise ValueError("Nyström projection requires fixed subsamples.")
        with torch.no_grad():
            sub_obs_en = agent.aug_and_encode(sub_obs, project=True)
            sub_next_obs_en = agent.aug_and_encode(sub_next_obs, project=True)
            sub_encoded_state_action = agent._encode_state_action(sub_obs_en, sub_action)
            sub_norm_next_obs_en = normalize_state_for_plot(agent, sub_next_obs_en)
        return agent.project_sa(
            phi_x=encoded_state_action,
            phi_sub_x=sub_encoded_state_action,
            psi_sub_y=sub_norm_next_obs_en,
        )
    return agent.project_sa(encoded_state_action)


def compute_visualization_tensors(
    agent,
    obs: torch.Tensor,
    action: torch.Tensor,
    next_obs: torch.Tensor,
    encoder_update_mode: str,
    sub_obs: torch.Tensor | None,
    sub_action: torch.Tensor | None,
    sub_next_obs: torch.Tensor | None,
):
    with torch.no_grad():
        obs_en = agent.aug_and_encode(obs, project=True)
        next_obs_en = agent.aug_and_encode(next_obs, project=True)
        encoded_state_action = agent._encode_state_action(obs_en, action)
        projected_sa = compute_projected_state_action(
            agent,
            encoded_state_action,
            sub_obs=sub_obs,
            sub_action=sub_action,
            sub_next_obs=sub_next_obs,
            encoder_update_mode=encoder_update_mode,
        )

        obs_plot = normalize_state_for_plot(agent, obs_en).detach().cpu().numpy()
        next_obs_plot = normalize_state_for_plot(agent, next_obs_en).detach().cpu().numpy()
        projected_plot = F.normalize(projected_sa, p=2, dim=1, eps=1e-10).detach().cpu().numpy()
        state_action_plot = encoded_state_action.detach().cpu().numpy()

    return obs_plot, next_obs_plot, projected_plot, state_action_plot


def reduce_state_action_space(points: np.ndarray, target_dim: int) -> np.ndarray:
    n_samples, original_dim = points.shape
    n_components = min(target_dim, n_samples, original_dim)
    if n_components <= 0:
        raise ValueError("Need at least one state-action point to visualize.")

    reduced = PCA(n_components=n_components).fit_transform(points)
    if n_components == target_dim:
        return reduced

    padded = np.zeros((n_samples, target_dim), dtype=reduced.dtype)
    padded[:, :n_components] = reduced
    return padded


def add_text_label(ax, point: np.ndarray, label: str, feature_dim: int, color: str):
    if feature_dim == 3:
        ax.text(point[0], point[1], point[2], label, fontsize=8, color=color)
    else:
        ax.text(point[0], point[1], label, fontsize=8, color=color)


def scatter_with_labels(
    ax,
    points: np.ndarray,
    labels: Sequence[str],
    feature_dim: int,
    color: str,
    marker: str,
    legend_label: str,
):
    if feature_dim == 3:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, marker=marker, s=70, label=legend_label)
    else:
        ax.scatter(points[:, 0], points[:, 1], c=color, marker=marker, s=70, label=legend_label)

    for point, label in zip(points, labels):
        add_text_label(ax, point, label, feature_dim, color)


def connect_projected_to_positive(ax, projected_points: np.ndarray, positive_points: np.ndarray, feature_dim: int):
    for pair_idx, (projected, positive) in enumerate(zip(projected_points, positive_points)):
        line_label = "target link: project_sa(s,a) -> expected φ(s')" if pair_idx == 0 else None
        if feature_dim == 3:
            ax.plot(
                [projected[0], positive[0]],
                [projected[1], positive[1]],
                [projected[2], positive[2]],
                linestyle="--",
                linewidth=0.8,
                color="#777777",
                alpha=0.7,
                label=line_label,
            )
        else:
            ax.plot(
                [projected[0], positive[0]],
                [projected[1], positive[1]],
                linestyle="--",
                linewidth=0.8,
                color="#777777",
                alpha=0.7,
                label=line_label,
            )


def draw_unit_reference(ax, feature_dim: int):
    """Draw a very light unit circle/sphere as a geometric reference."""
    if feature_dim == 3:
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 20)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(
            x,
            y,
            z,
            rstride=2,
            cstride=2,
            color="#BBBBBB",
            linewidth=0.4,
            alpha=1,
        )
    else:
        circle = plt.Circle(
            (0.0, 0.0),
            1.0,
            facecolor="none",
            edgecolor="#BBBBBB",
            linewidth=0.8,
            alpha=1,
        )
        ax.add_patch(circle)
        ax.set_aspect("equal", adjustable="box")


def plot_embedding_views(
    output_path: Path,
    feature_dim: int,
    next_points: np.ndarray,
    projected_points: np.ndarray,
    state_action_points: np.ndarray,
    next_state_labels: Sequence[str],
    state_action_labels: Sequence[str],
    actions: np.ndarray,
    title: str,
    relation_note: str,
):
    if feature_dim == 3:
        fig = plt.figure(figsize=(16, 7))
        ax_state = fig.add_subplot(1, 2, 1, projection="3d")
        ax_sa = fig.add_subplot(1, 2, 2, projection="3d")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        ax_state, ax_sa = axes

    scatter_with_labels(
        ax_state,
        next_points,
        next_state_labels,
        feature_dim,
        color="#54A24B",
        marker="^",
        legend_label="φ(s')",
    )
    scatter_with_labels(
        ax_state,
        projected_points,
        state_action_labels,
        feature_dim,
        color="#E45756",
        marker="s",
        legend_label="project_sa(ψ(s,a))",
    )
    draw_unit_reference(ax_state, feature_dim)
    connect_projected_to_positive(ax_state, projected_points, next_points, feature_dim)

    ax_state.set_title("Projected State-Action vs Positive Target")
    ax_state.legend(loc="best")
    ax_state.set_xlabel("dim 1")
    ax_state.set_ylabel("dim 2")
    if feature_dim == 3:
        ax_state.set_zlabel("dim 3")
        ax_state.text2D(
            0.02,
            0.02,
            relation_note,
            transform=ax_state.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#BBBBBB"),
        )
    else:
        ax_state.text(
            0.02,
            0.02,
            relation_note,
            transform=ax_state.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#BBBBBB"),
            verticalalignment="bottom",
        )

    sa_colors = [ACTION_TO_COLOR.get(int(action), "#333333") for action in actions]
    if feature_dim == 3:
        ax_sa.scatter(
            state_action_points[:, 0],
            state_action_points[:, 1],
            state_action_points[:, 2],
            c=sa_colors,
            marker="o",
            s=70,
        )
    else:
        ax_sa.scatter(
            state_action_points[:, 0],
            state_action_points[:, 1],
            c=sa_colors,
            marker="o",
            s=70,
        )
    for point, label, color in zip(state_action_points, state_action_labels, sa_colors):
        add_text_label(ax_sa, point, label, feature_dim, color)

    ax_sa.set_title("State-Action Embedding Space (PCA)")
    ax_sa.set_xlabel("dim 1")
    ax_sa.set_ylabel("dim 2")
    if feature_dim == 3:
        ax_sa.set_zlabel("dim 3")

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def train_agent_for_feature_dim(
    agent,
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_observations: np.ndarray,
    batch_plan: Sequence[np.ndarray],
    encoder_update_mode: str,
    dyn_loss: str,
    curl_weight: float,
    device: str,
    sub_obs_np: np.ndarray | None,
    sub_action_np: np.ndarray | None,
    sub_next_obs_np: np.ndarray | None,
) -> Dict[str, np.ndarray]:
    metrics_log: Dict[str, List[float]] = {
        "step": [],
        "epoch": [],
        "transition_loss": [],
        "contrastive_loss": [],
        "curl_loss": [],
        "embedding_sum_loss": [],
        "reward_loss": [],
    }

    if encoder_update_mode == "nystrom":
        sub_obs = tensor_from_numpy(sub_obs_np, device)
        sub_action = tensor_from_numpy(sub_action_np, device).long()
        sub_next_obs = tensor_from_numpy(sub_next_obs_np, device)
    else:
        sub_obs = sub_action = sub_next_obs = None

    batches_per_epoch = len(batch_plan)
    for step_idx, batch_indices in enumerate(tqdm(batch_plan, desc="Training", leave=False), start=1):
        epoch = int(np.ceil(step_idx / max(batches_per_epoch, 1)))
        obs_batch = tensor_from_numpy(observations[batch_indices], device)
        action_batch = tensor_from_numpy(actions[batch_indices], device).long()
        next_obs_batch = tensor_from_numpy(next_observations[batch_indices], device)
        reward_batch = tensor_from_numpy(rewards[batch_indices], device)

        if encoder_update_mode == "nystrom":
            metrics = copied_update_encoders_nystrom(
                agent,
                obs_batch,
                action_batch,
                next_obs_batch,
                reward_batch,
                sub_obs,
                sub_action,
                sub_next_obs,
                curl_loss_weight=curl_weight,
                dyn_loss=dyn_loss,
            )
        else:
            metrics = copied_update_encoders(
                agent,
                obs_batch,
                action_batch,
                next_obs_batch,
                reward_batch,
                curl_loss_weight=curl_weight,
            )

        metrics_log["step"].append(step_idx)
        metrics_log["epoch"].append(epoch)
        for key in ("transition_loss", "contrastive_loss", "curl_loss", "embedding_sum_loss", "reward_loss"):
            metrics_log[key].append(metrics.get(key, 0.0))

    return {key: np.asarray(value) for key, value in metrics_log.items()}


def main():
    args = parse_args()
    utils.set_seed_everywhere(args.seed)
    torch.manual_seed(args.seed)
    if args.encoder_update_mode == "nystrom" and args.subsamples is None:
        raise ValueError("--subsamples is required when --encoder-update-mode=nystrom")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = resolve_dataset_npz(args.dataset)
    dataset = load_dataset(dataset_path)
    obs_mode = dataset["metadata"]["obs_mode"]
    observations = dataset["observation"]
    actions = dataset["action"].astype(np.int64)
    rewards = dataset["reward"]
    next_observations = dataset["next_observation"]
    state_indices = get_state_indices(dataset)
    next_state_indices = get_next_state_indices(dataset)

    plot_indices = select_plot_indices(
        actions=actions,
        state_indices=state_indices,
        num_points=args.num_plot_points,
        seed=args.seed,
        explicit_indices=args.plot_indices,
    )

    cfg = compose_cfg(args.config_name, obs_mode, args.seed)
    batch_plan = compute_epoch_batches(
        num_samples=observations.shape[0],
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    )
    total_steps = len(batch_plan)
    nystrom_name_suffix = f"_dynloss_{args.dyn_loss}" if args.encoder_update_mode == "nystrom" else ""

    for feature_dim in args.feature_dims:
        experiment_dir = output_dir / f"{args.encoder_update_mode}{nystrom_name_suffix}_featuredim_{feature_dim}"
        experiment_dir.mkdir(parents=True, exist_ok=True)

        env = make_env_from_cfg(cfg)
        agent = build_agent(
            cfg=cfg,
            env=env,
            feature_dim=feature_dim,
            device=args.device,
            total_steps=total_steps,
            subsamples=args.subsamples,
            encoder_update_mode=args.encoder_update_mode,
        )

        fixed_subsample_path = None
        sub_obs_np = sub_action_np = sub_next_obs_np = None
        if args.encoder_update_mode == "nystrom":
            fixed_rng = np.random.default_rng(args.seed)
            subsample_indices = np.sort(
                fixed_rng.choice(observations.shape[0], size=args.subsamples, replace=False)
            )
            sub_obs_np = observations[subsample_indices]
            sub_action_np = actions[subsample_indices]
            sub_next_obs_np = next_observations[subsample_indices]
            fixed_subsample_path = experiment_dir / "fixed_subsamples.npz"
            np.savez_compressed(
                fixed_subsample_path,
                indices=subsample_indices,
                observation=sub_obs_np,
                action=sub_action_np,
                next_observation=sub_next_obs_np,
                reward=rewards[subsample_indices],
            )

        metrics = train_agent_for_feature_dim(
            agent=agent,
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            batch_plan=batch_plan,
            encoder_update_mode=args.encoder_update_mode,
            dyn_loss=args.dyn_loss,
            curl_weight=args.curl_weight,
            device=args.device,
            sub_obs_np=sub_obs_np,
            sub_action_np=sub_action_np,
            sub_next_obs_np=sub_next_obs_np,
        )

        np.savez_compressed(experiment_dir / "metrics.npz", **metrics)
        save_loss_curves(experiment_dir / "loss_curves.png", metrics)

        obs_plot_batch = tensor_from_numpy(observations[plot_indices], args.device)
        action_plot_batch = tensor_from_numpy(actions[plot_indices], args.device).long()
        next_obs_plot_batch = tensor_from_numpy(next_observations[plot_indices], args.device)

        if args.encoder_update_mode == "nystrom":
            sub_obs_plot = tensor_from_numpy(sub_obs_np, args.device)
            sub_action_plot = tensor_from_numpy(sub_action_np, args.device).long()
            sub_next_obs_plot = tensor_from_numpy(sub_next_obs_np, args.device)
        else:
            sub_obs_plot = sub_action_plot = sub_next_obs_plot = None

        _, next_points, projected_points, state_action_points = compute_visualization_tensors(
            agent=agent,
            obs=obs_plot_batch,
            action=action_plot_batch,
            next_obs=next_obs_plot_batch,
            encoder_update_mode=args.encoder_update_mode,
            sub_obs=sub_obs_plot,
            sub_action=sub_action_plot,
            sub_next_obs=sub_next_obs_plot,
        )
        state_action_points_reduced = reduce_state_action_space(state_action_points, target_dim=feature_dim)

        plot_actions = actions[plot_indices]
        pair_ids = [f"p{i}" for i in range(len(plot_indices))]
        next_state_labels = [
            f"{pair_id}:{int(idx)}"
            for pair_id, idx in zip(pair_ids, next_state_indices[plot_indices])
        ]
        state_action_labels = [
            f"{pair_id}:{int(state_idx)},{ACTION_TO_ARROW.get(int(action), str(int(action)))}"
            for pair_id, state_idx, action in zip(pair_ids, state_indices[plot_indices], plot_actions)
        ]

        plot_embedding_views(
            output_path=experiment_dir / "embedding_spaces.png",
            feature_dim=feature_dim,
            next_points=next_points,
            projected_points=projected_points,
            state_action_points=state_action_points_reduced,
            next_state_labels=next_state_labels,
            state_action_labels=state_action_labels,
            actions=plot_actions,
            title=(
                f"Embedding Spaces | mode={args.encoder_update_mode} | "
                f"feature_dim={feature_dim} | curl_weight={args.curl_weight:g}"
            ),
            relation_note=(
                "Pair ids p0, p1, ... identify matching transitions.\n"
                "Red = project_sa(ψ(s,a)), green = expected target φ(s').\n"
                "Dashed links connect each projected point to its own positive target.\n"
                "Right panel shows raw ψ(s,a) in a PCA view."
            ),
        )

        selected_transitions = [
            {
                "dataset_index": int(dataset_idx),
                "pair_id": pair_id,
                "state_index": int(state_idx),
                "action": int(action),
                "action_arrow": ACTION_TO_ARROW.get(int(action), str(int(action))),
                "next_state_index": int(next_state_idx),
            }
            for pair_id, dataset_idx, state_idx, action, next_state_idx in zip(
                pair_ids,
                plot_indices,
                state_indices[plot_indices],
                plot_actions,
                next_state_indices[plot_indices],
            )
        ]
        config_dump = {
            "dataset": str(args.dataset.resolve()),
            "dataset_npz": str(dataset_path),
            "config_name": args.config_name,
            "feature_dim": int(feature_dim),
            "encoder_update_mode": args.encoder_update_mode,
            "dyn_loss": args.dyn_loss if args.encoder_update_mode == "nystrom" else None,
            "curl_weight": float(args.curl_weight),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "subsamples": int(args.subsamples) if args.subsamples is not None else None,
            "seed": int(args.seed),
            "device": args.device,
            "fixed_subsample_file": str(fixed_subsample_path) if fixed_subsample_path else None,
            "selected_transitions": selected_transitions,
        }
        (experiment_dir / "config.json").write_text(json.dumps(config_dump, indent=2))

        env.close()

    print(f"Saved embedding-space visualizations to {output_dir}")


if __name__ == "__main__":
    main()
