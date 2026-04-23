#!/usr/bin/env python3
"""Run fixed-subsample Nyström encoder sweeps on a saved MultipleRooms dataset.

Examples
--------
python pretrain.py --config-name=pretrain/pretrain_rover_multiplerooms agent.embeddings=true discount=0.99 agent=rover_nystrom agent.subsamples=1000 agent.pmd_steps=100 agent.feature_dim=109 num_seed_frames=4000 agent.lr_actor=100 obs_type=pixels 




python encoder_testing/test_rover_nystrom_encoder.py --dataset encoder_testing/outputs/multiplerooms_pixels --output-dir encoder_testing/outputs/pixels_s100 --dyn-loss classic_contrastive --feature-dims 2 10 20 32 64 109 --curl-weights 0.0 0.001 0.01 1.0 --encoder-update-mode nystrom --subsamples 100 --epochs 150 --batch-size 1024 --device cuda --config-name pretrain/pretrain_rover_multiplerooms2
python encoder_testing/test_rover_nystrom_encoder.py --dataset encoder_testing/outputs/multiplerooms_pixels --output-dir encoder_testing/outputs/pixels_s100 --dyn-loss distance_ratio --feature-dims 2 10 20 32 64 109 --curl-weights 0.0 0.001 0.01 1.0 --encoder-update-mode nystrom --subsamples 100 --epochs 150 --batch-size 1024 --device cuda --config-name pretrain/pretrain_rover_multiplerooms2
python encoder_testing/test_rover_nystrom_encoder.py --dataset encoder_testing/outputs/multiplerooms_pixels --output-dir encoder_testing/outputs/pixels_s100 --dyn-loss distance_positive_only --feature-dims 2 10 20 32 64 109 --curl-weights 0.0 0.001 0.01 1.0 --encoder-update-mode nystrom --subsamples 100 --epochs 150 --batch-size 1024 --device cuda --config-name pretrain/pretrain_rover_multiplerooms2

python encoder_testing/test_rover_nystrom_encoder.py \
    --dataset encoder_testing/outputs/multiplerooms_states \
    --output-dir encoder_testing/outputs/nystrom_debug_states_s100_l2_prova \
    --feature-dims 32 64 109 200\
    --curl-weights 0.0 0.001 0.01 1.0 \
    --encoder-update-mode nystrom \
    --dyn-loss distance_positive_only \
    --subsamples 100 \
    --epochs 10 \
    --batch-size 1024 \
    --device cuda \
    --config-name pretrain/pretrain_rover_multiplerooms2

python encoder_testing/test_rover_nystrom_encoder.py \
    --dataset encoder_testing/outputs/multiplerooms_states \
    --output-dir encoder_testing/outputs/rank_test \
    --feature-dims 109\
    --curl-weights 0.0 \
    --encoder-update-mode nystrom \
    --dyn-loss distance_positive_only \
    --subsamples 100 \
    --epochs 10 \
    --batch-size 1024 \
    --device cuda \
    --config-name pretrain/pretrain_rover_multiplerooms2

python encoder_testing/test_rover_nystrom_encoder.py \
    --dataset encoder_testing/outputs/multiplerooms_states \
    --output-dir encoder_testing/outputs/full_debug_states \
    --feature-dims 32 64 109 \
    --curl-weights 0.0 0.001 0.01 1.0 \
    --encoder-update-mode full \
    --subsamples 10 \
    --epochs 100 \
    --batch-size 1024 \
    --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm.rich import tqdm

import gym_env
import utils
from agent.rover_nystrom_t_new import (
    EmbeddingDistributionVisualizerV2,
    ProjectSA,
    RoverAgent,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DYN_LOSS_CHOICES = (
    "classic_contrastive",
    "distance_ratio",
    "distance_positive_only",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep Nyström encoder updates on a saved MultipleRooms dataset."
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
        help="Base Hydra config used to recover the default MultipleRooms environment.",
    )
    parser.add_argument(
        "--feature-dims",
        type=int,
        nargs="+",
        required=True,
        help="Grid values for feature_dim.",
    )
    parser.add_argument(
        "--curl-weights",
        type=float,
        nargs="+",
        default=[1e-3],
        help="Grid values for the CURL loss multiplier. Default keeps CURL active.",
    )
    parser.add_argument(
        "--subsamples",
        type=int,
        default=None,
        help="Number of fixed Nyström subsamples m. Required only for --encoder-update-mode nystrom.",
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
        "--epochs",
        type=int,
        default=10,
        help="Number of passes over the fixed dataset per experiment.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training mini-batch size for update_encoders_nystrom.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for subsampling and batch ordering.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device, for example cpu or cuda.",
    )
    return parser.parse_args()


def compose_cfg(config_name: str, obs_mode: str, seed: int):
    def _resolve_group_path(group_value: str, group_name: str) -> Path:
        direct = CONFIG_DIR / f"{group_value}.yaml"
        if direct.exists():
            return direct
        grouped = CONFIG_DIR / group_name / f"{group_value}.yaml"
        if grouped.exists():
            return grouped
        raise FileNotFoundError(f"Could not resolve {group_name} config '{group_value}'")

    config_path = CONFIG_DIR / f"{config_name}.yaml"
    cfg = OmegaConf.load(config_path)

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
        raise ValueError(f"Could not recover env default from {config_path}")
    if agent_default is None:
        raise ValueError(f"Could not recover agent default from {config_path}")

    env_cfg = OmegaConf.load(_resolve_group_path(env_default, "env"))
    agent_cfg = OmegaConf.load(_resolve_group_path(agent_default, "agent"))
    return OmegaConf.merge(
        {"agent": agent_cfg},
        env_cfg,
        cfg,
        {"obs_type": obs_mode, "seed": seed},
    )


def make_env_from_cfg(cfg):
    env_kwargs = OmegaConf.to_container(cfg.env, resolve=True)
    env_kwargs.pop("name", None)
    return gym_env.make(
        cfg.env.name,
        cfg.obs_type,
        frame_stack=cfg.frame_stack,
        action_repeat=cfg.action_repeat,
        seed=cfg.seed,
        resolution=cfg.resolution,
        grayscale=cfg.grayscale,
        url=True,
        **env_kwargs,
    )


def copied_update_encoders_nystrom(
    agent: RoverAgent,
    obs: torch.Tensor,
    action: torch.Tensor,
    next_obs: torch.Tensor,
    reward: torch.Tensor,
    sub_obs: torch.Tensor,
    sub_action: torch.Tensor,
    sub_next_obs: torch.Tensor,
    curl_loss_weight: float = 1e-3,
    dyn_loss: str = "classic_contrastive",
):
    """Copied from RoverAgent.update_encoders_nystrom with explicit CURL weighting."""
    metrics = {}

    obs_en = agent.aug_and_encode(obs, project=True)
    with torch.no_grad():
        next_obs_en = agent.aug_and_encode(next_obs, project=True)

    unique_obs = torch.unique(obs, dim=0)
    obs_en_unique = agent.aug_and_encode(unique_obs, project=True)

    cov_matrix = obs_en_unique @ obs_en_unique.T
    rank_estimate = torch.linalg.matrix_rank(cov_matrix, tol=1e-5)
    S,V,D = torch.linalg.svd(cov_matrix)

    print(f" Full Rank: {cov_matrix.shape[0]}, Estimated Rank: {rank_estimate}, Last 30 minor eigenvalues: {V[-30:]}")

    encoded_state_action = agent._encode_state_action(obs_en, action)
    if agent.mode == "l1":
        norm_next_obs_en = F.normalize(next_obs_en, p=2, dim=1, eps=1e-10)
    elif agent.mode == "l2":
        norm_next_obs_en = next_obs_en
    else:
        raise ValueError(f"Unsupported mode: {agent.mode}")

    with torch.no_grad():
        sub_obs_en = agent.aug_and_encode(sub_obs, project=True)
        sub_next_obs_en = agent.aug_and_encode(sub_next_obs, project=True)
        if agent.mode == "l1":
            sub_norm_next_obs_en = F.normalize(sub_next_obs_en, p=2, dim=1, eps=1e-10)
        elif agent.mode == "l2":
            sub_norm_next_obs_en = sub_next_obs_en
        else:
            raise ValueError(f"Unsupported mode: {agent.mode}")

    sub_encoded_state_action = agent._encode_state_action(sub_obs_en, sub_action)
    projected_sa = agent.project_sa(
        phi_x=encoded_state_action,
        phi_sub_x=sub_encoded_state_action,
        psi_sub_y=sub_norm_next_obs_en,
    )
    norm_projected_sa = F.normalize(projected_sa, p=2, dim=1, eps=1e-10)
    
    if dyn_loss == "classic_contrastive":
        logits = torch.matmul(norm_projected_sa, norm_next_obs_en.T)
        logits = logits - torch.max(logits, 1)[0][:, None]
        labels = torch.arange(logits.shape[0], device=agent.device).long()
        contrastive_loss = agent.cross_entropy_loss(logits, labels)
    elif dyn_loss == "distance_ratio":
        logits = torch.cdist(norm_projected_sa, norm_next_obs_en, p=2) ** 2
        pos = logits.diagonal()
        mask = ~torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
        neg_sum = logits.masked_select(mask).view(logits.shape[0], -1).mean(dim=1)
        contrastive_loss = (pos / (neg_sum + 1e-8)).mean()
    elif dyn_loss == "distance_positive_only":
        logits = torch.cdist(norm_projected_sa, norm_next_obs_en, p=2) ** 2
        contrastive_loss = logits.diagonal().mean()
    else:
        raise ValueError(f"Unknown dyn_loss: {dyn_loss}")

    z_anchor = agent.aug_and_encode(sub_obs, project=True)
    with torch.no_grad():
        z_pos = agent.aug_and_encode(sub_obs, project=True)

    # curl_loss = torch.tensor(0.0, device=agent.device)
    
    if agent.curl:
        if agent.mode == "l1":
            z_anchor = F.normalize(z_anchor, p=2, dim=1, eps=1e-10)
            z_pos = F.normalize(z_pos, p=2, dim=1, eps=1e-10)
        curl_logits = torch.matmul(z_anchor, z_pos.T)
        curl_logits = curl_logits - torch.max(curl_logits, 1)[0][:, None]
        curl_labels = torch.arange(curl_logits.shape[0], device=agent.device).long()
        curl_loss = agent.cross_entropy_loss(curl_logits, curl_labels)
    else:
        curl_loss = torch.tensor(0.0, device=agent.device)
        
    if agent.reward:
        reward_pred = agent.reward(encoded_state_action)
        reward_loss = F.mse_loss(reward_pred, reward.to(agent.device))
    else:
        reward_loss = torch.tensor(0.0, device=agent.device)

    if agent.embedding_sum_loss > 0:
        sum_next_obs_en = torch.sum(next_obs_en, dim=1)
        embedding_sum_loss = agent.embedding_sum_loss * torch.mean((sum_next_obs_en - 1.0) ** 2)
    else:
        embedding_sum_loss = torch.tensor(0.0, device=agent.device)

    loss = contrastive_loss + float(curl_loss_weight) * curl_loss + embedding_sum_loss + reward_loss

    if agent.encoder_optimizer is not None:
        agent.encoder_optimizer.zero_grad()
    agent.transition_optimizer.zero_grad()
    loss.backward()
    if agent.encoder_optimizer is not None:
        agent.encoder_optimizer.step()
        agent._policy_is_synced = False
    agent.transition_optimizer.step()
    agent.encoder_scheduler.step()

    metrics["transition_loss"] = float(loss.item())
    metrics["contrastive_loss"] = float(contrastive_loss.item())
    metrics["curl_loss"] = float(curl_loss.item())
    metrics["embedding_sum_loss"] = float(embedding_sum_loss.item())
    metrics["reward_loss"] = float(reward_loss.item())
    metrics["curl_loss_weight"] = float(curl_loss_weight)
    metrics["dyn_loss"] = dyn_loss
    return metrics


def copied_update_encoders(
    agent: RoverAgent,
    obs: torch.Tensor,
    action: torch.Tensor,
    next_obs: torch.Tensor,
    reward: torch.Tensor,
    curl_loss_weight: float = 1e-3,
):
    """Copied from RoverAgent.update_encoders with explicit CURL weighting."""
    metrics = {}

    obs_en = agent.aug_and_encode(obs, project=True)
    with torch.no_grad():
        next_obs_en = agent.aug_and_encode(next_obs, project=True)

    encoded_state_action = agent._encode_state_action(obs_en, action)
    projected_sa = agent.project_sa(encoded_state_action)

    if agent.mode == "l1":
        norm_next_obs_en = F.normalize(next_obs_en, p=2, dim=1, eps=1e-10)
    elif agent.mode == "l2":
        norm_next_obs_en = next_obs_en
    else:
        raise ValueError(f"Unsupported mode: {agent.mode}")
    norm_projected_sa = F.normalize(projected_sa, p=2, dim=1, eps=1e-10)

    logits = torch.matmul(norm_projected_sa, norm_next_obs_en.T)
    logits = logits - torch.max(logits, 1)[0][:, None]
    labels = torch.arange(logits.shape[0], device=agent.device).long()
    contrastive_loss = agent.cross_entropy_loss(logits, labels)

    z_anchor = agent.aug_and_encode(obs, project=True)
    with torch.no_grad():
        z_pos = agent.aug_and_encode(obs, project=True)

    if agent.curl:
        if agent.mode == "l1":
            z_anchor = F.normalize(z_anchor, p=2, dim=1, eps=1e-10)
            z_pos = F.normalize(z_pos, p=2, dim=1, eps=1e-10)
        curl_logits = torch.matmul(z_anchor, z_pos.T)
        curl_logits = curl_logits - torch.max(curl_logits, 1)[0][:, None]
        curl_labels = torch.arange(curl_logits.shape[0], device=agent.device).long()
        curl_loss = agent.cross_entropy_loss(curl_logits, curl_labels)
    else:
        curl_loss = torch.tensor(0.0, device=agent.device)

    if agent.reward:
        reward_pred = agent.reward(encoded_state_action)
        reward_loss = F.mse_loss(reward_pred, reward.to(agent.device))
    else:
        reward_loss = torch.tensor(0.0, device=agent.device)

    if agent.embedding_sum_loss > 0:
        sum_next_obs_en = torch.sum(next_obs_en, dim=1)
        embedding_sum_loss = agent.embedding_sum_loss * torch.mean((sum_next_obs_en - 1.0) ** 2)
    else:
        embedding_sum_loss = torch.tensor(0.0, device=agent.device)

    loss = contrastive_loss + float(curl_loss_weight) * curl_loss + embedding_sum_loss + reward_loss

    if agent.encoder_optimizer is not None:
        agent.encoder_optimizer.zero_grad()
    agent.transition_optimizer.zero_grad()
    loss.backward()
    if agent.encoder_optimizer is not None:
        agent.encoder_optimizer.step()
        agent._policy_is_synced = False
    agent.transition_optimizer.step()
    agent.encoder_scheduler.step()

    metrics["transition_loss"] = float(loss.item())
    metrics["contrastive_loss"] = float(contrastive_loss.item())
    metrics["curl_loss"] = float(curl_loss.item())
    metrics["embedding_sum_loss"] = float(embedding_sum_loss.item())
    metrics["reward_loss"] = float(reward_loss.item())
    metrics["curl_loss_weight"] = float(curl_loss_weight)
    return metrics


def load_dataset(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        dataset = {key: data[key] for key in data.files}
    dataset["metadata"] = json.loads(str(dataset["metadata_json"].item()))
    return dataset


def resolve_dataset_npz(dataset_dir: Path) -> Path:
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise ValueError(f"--dataset must point to a dataset folder, got: {dataset_dir}")

    dataset_path = dataset_dir / "dataset.npz"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Could not find dataset file at {dataset_path}. "
            "Expected a folder created by collect_multiplerooms_dataset.py."
        )
    return dataset_path


def build_agent(
    cfg,
    env,
    feature_dim: int,
    device: str,
    total_steps: int,
    subsamples: int | None,
    encoder_update_mode: str,
):
    obs_spec = gym_env.observation_spec(env)
    action_spec = gym_env.action_spec(env)
    action_shape = (action_spec.num_values,) if hasattr(action_spec, "num_values") else action_spec.shape
    agent_cfg = OmegaConf.to_container(cfg.agent, resolve=False)
    agent_cfg.pop("_target_", None)
    init_subsamples = int(subsamples) if subsamples is not None else 1
    agent_cfg.update(
        {
            "obs_type": cfg.obs_type,
            "obs_shape": obs_spec.shape if obs_spec.shape else (1,),
            "action_shape": action_shape,
            "grayscale": bool(cfg.grayscale),
            "discount": float(cfg.discount),
            "feature_dim": int(feature_dim),
            "subsamples": init_subsamples,
            "device": device,
            "total_train_steps": max(1, int(total_steps)),
            "num_expl_steps": int(cfg.num_seed_frames // cfg.action_repeat),
            "use_tb": False,
            "use_wandb": False,
        }
    )
    agent = RoverAgent(**agent_cfg)
    if encoder_update_mode == "full":
        agent.project_sa = ProjectSA(
            agent.obs_dim * agent.n_actions,
            agent_cfg["hidden_dim"],
            agent.obs_dim,
        ).to(agent.device)
        agent.transition_optimizer = torch.optim.Adam(
            agent.project_sa.parameters(),
            lr=agent.lr_T,
        )
    agent.insert_env(env)
    agent.train(True)
    if agent.gridworld_visualizer is None or not hasattr(
        agent.gridworld_visualizer, "_compute_state_correlation_matrix"
    ):
        agent.gridworld_visualizer = EmbeddingDistributionVisualizerV2(agent)
    return agent


def tensor_from_numpy(array: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(array).to(device)


def compute_epoch_batches(num_samples: int, batch_size: int, epochs: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    all_batches: List[np.ndarray] = []
    for _ in range(epochs):
        order = rng.permutation(num_samples)
        for start in range(0, num_samples, batch_size):
            all_batches.append(order[start : start + batch_size])
    return all_batches


def save_loss_curves(path: Path, metrics: Dict[str, np.ndarray]):
    steps = metrics["step"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, metrics["transition_loss"], label="total loss", linewidth=2)
    ax.plot(steps, metrics["contrastive_loss"], label="contrastive loss", linewidth=2)
    # ax.plot(steps, metrics["curl_loss"], label="curl loss", linewidth=2)
    # ax.plot(steps, metrics["embedding_sum_loss"], label="embedding sum loss", linewidth=2)
    # ax.plot(steps, metrics["reward_loss"], label="reward loss", linewidth=2)
    ax.set_xlabel("update step")
    ax.set_ylabel("loss")
    ax.set_title("Nyström Encoder Losses")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_correlation_matrix(path: Path, visualizer, correlation_matrix: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    visualizer._plot_state_correlations(ax, correlation_matrix)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def update_loss_grid_figure(
    path: Path,
    metrics_by_experiment: Dict[Tuple[float, int], Dict[str, np.ndarray]],
    feature_dims: Sequence[int],
    curl_weights: Sequence[float],
):
    n_rows = len(curl_weights)
    n_cols = len(feature_dims)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), squeeze=False)

    for row, curl_weight in enumerate(curl_weights):
        for col, feature_dim in enumerate(feature_dims):
            ax = axes[row, col]
            metrics = metrics_by_experiment.get((float(curl_weight), int(feature_dim)))
            if metrics is None:
                ax.text(0.5, 0.5, "pending", ha="center", va="center", fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                steps = metrics["step"]
                ax.plot(steps, metrics["contrastive_loss"], label="contrastive", linewidth=1.8)
                ax.grid(alpha=0.25, linewidth=0.5)
            ax.set_title(f"fd={feature_dim}, cw={curl_weight:g}")
            if row == n_rows - 1:
                ax.set_xlabel("step")
            if col == 0:
                ax.set_ylabel("contrastive loss")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=1, frameon=True)
    fig.suptitle("Contrastive Loss Grid", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def update_grid_figure(
    path: Path,
    matrices: Dict[Tuple[float, int], np.ndarray],
    feature_dims: Sequence[int],
    curl_weights: Sequence[float],
):
    n_rows = len(curl_weights)
    n_cols = len(feature_dims)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    for row, curl_weight in enumerate(curl_weights):
        for col, feature_dim in enumerate(feature_dims):
            ax = axes[row, col]
            matrix = matrices.get((float(curl_weight), int(feature_dim)))
            if matrix is None:
                ax.text(0.5, 0.5, "pending", ha="center", va="center", fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"fd={feature_dim}, cw={curl_weight:g}")
            if row == n_rows - 1:
                ax.set_xlabel("state index")
            if col == 0:
                ax.set_ylabel("state index")

    fig.suptitle("Correlation Matrices Grid", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    utils.set_seed_everywhere(args.seed)
    torch.manual_seed(args.seed)
    if args.encoder_update_mode == "nystrom" and args.subsamples is None:
        raise ValueError("--subsamples is required when --encoder-update-mode=nystrom")
    output_dir = Path(f"{args.output_dir}_{args.encoder_update_mode}")

    if args.encoder_update_mode == "nystrom":
        output_dir = Path(f"{output_dir}_dynloss_{args.dyn_loss}")

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = resolve_dataset_npz(args.dataset)
    dataset = load_dataset(dataset_path)
    obs_mode = dataset["metadata"]["obs_mode"]
    observations = dataset["observation"]
    actions = dataset["action"]
    rewards = dataset["reward"]
    next_observations = dataset["next_observation"]

    num_samples = observations.shape[0]
    if args.subsamples is not None and args.subsamples > num_samples:
        raise ValueError(f"subsamples={args.subsamples} exceeds dataset size {num_samples}")

    batch_plan = compute_epoch_batches(
        num_samples=num_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    )

    cfg = compose_cfg(args.config_name, obs_mode, args.seed)
    total_steps = len(batch_plan)
    nystrom_name_suffix = f"_dynloss_{args.dyn_loss}" if args.encoder_update_mode == "nystrom" else ""

    fixed_subsample_path = None
    subsample_indices = None
    sub_obs_np = None
    sub_action_np = None
    sub_next_obs_np = None
    if args.encoder_update_mode == "nystrom":
        fixed_rng = np.random.default_rng(args.seed)
        subsample_indices = np.sort(fixed_rng.choice(num_samples, size=args.subsamples, replace=False))
        sub_obs_np = observations[subsample_indices]
        sub_action_np = actions[subsample_indices]
        sub_next_obs_np = next_observations[subsample_indices]
        fixed_subsample_path = output_dir / "fixed_subsamples.npz"
        np.savez_compressed(
            fixed_subsample_path,
            indices=subsample_indices,
            observation=sub_obs_np,
            action=sub_action_np,
            next_observation=sub_next_obs_np,
            reward=rewards[subsample_indices],
            batch_lengths=np.asarray([len(batch) for batch in batch_plan], dtype=np.int64),
        )
        print(
            f"Selected {args.subsamples} fixed Nyström subsamples once at the beginning and saved them to "
            f"{fixed_subsample_path}"
        )

    completed_matrices: Dict[Tuple[float, int], np.ndarray] = {}
    completed_metrics: Dict[Tuple[float, int], Dict[str, np.ndarray]] = {}
    grid_path = output_dir / f"grid_correlation_matrices_{args.encoder_update_mode}{nystrom_name_suffix}.png"
    loss_grid_path = output_dir / f"grid_loss_curves_{args.encoder_update_mode}{nystrom_name_suffix}.png"

    experiment_pairs = [
        (feature_dim, curl_weight)
        for curl_weight in args.curl_weights
        for feature_dim in args.feature_dims
    ]

    progress = tqdm(experiment_pairs, desc="Nyström encoder sweep")
    for feature_dim, curl_weight in progress:
        progress.set_description(f"fd={feature_dim}, cw={curl_weight:g}")
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

        if args.encoder_update_mode == "nystrom":
            sub_obs = tensor_from_numpy(sub_obs_np, args.device)
            sub_action = tensor_from_numpy(sub_action_np, args.device).long()
            sub_next_obs = tensor_from_numpy(sub_next_obs_np, args.device)

        metrics_log: Dict[str, List[float]] = {
            "step": [],
            "epoch": [],
            "transition_loss": [],
            "contrastive_loss": [],
            "curl_loss": [],
            "embedding_sum_loss": [],
            "reward_loss": [],
        }

        global_step = 0
        epoch = 0
        batches_per_epoch = int(np.ceil(num_samples / args.batch_size))
        for batch_idx, batch_indices in enumerate(batch_plan):
            if batch_idx % batches_per_epoch == 0:
                epoch += 1

            obs_batch = tensor_from_numpy(observations[batch_indices], args.device)
            action_batch = tensor_from_numpy(actions[batch_indices], args.device).long()
            next_obs_batch = tensor_from_numpy(next_observations[batch_indices], args.device)
            reward_batch = tensor_from_numpy(rewards[batch_indices], args.device)

            if args.encoder_update_mode == "nystrom":
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
                    dyn_loss=args.dyn_loss,
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
            global_step += 1

            metrics_log["step"].append(global_step)
            metrics_log["epoch"].append(epoch)
            for key in ("transition_loss", "contrastive_loss", "curl_loss", "embedding_sum_loss", "reward_loss"):
                metrics_log[key].append(metrics.get(key, 0.0))

        experiment_dir = output_dir / (
            f"{args.encoder_update_mode}{nystrom_name_suffix}_featuredim_{feature_dim}_curlweight_{curl_weight:g}"
        )
        experiment_dir.mkdir(parents=True, exist_ok=True)

        correlation_matrix = agent.gridworld_visualizer._compute_state_correlation_matrix()
        completed_matrices[(float(curl_weight), int(feature_dim))] = correlation_matrix

        save_correlation_matrix(
            experiment_dir / "correlation_matrix.png",
            agent.gridworld_visualizer,
            correlation_matrix,
            title=(
                f"State Embedding Correlations\n"
                f"mode={args.encoder_update_mode}, feature_dim={feature_dim}, curl_weight={curl_weight:g}"
            ),
        )
        save_loss_curves(
            experiment_dir / "loss_curves.png",
            {key: np.asarray(value) for key, value in metrics_log.items()},
        )
        completed_metrics[(float(curl_weight), int(feature_dim))] = {
            key: np.asarray(value) for key, value in metrics_log.items()
        }

        np.savez_compressed(
            experiment_dir / "metrics.npz",
            **{key: np.asarray(value) for key, value in metrics_log.items()},
        )
        experiment_config = {
            "feature_dim": int(feature_dim),
            "curl_weight": float(curl_weight),
            "encoder_update_mode": args.encoder_update_mode,
            "dyn_loss": args.dyn_loss if args.encoder_update_mode == "nystrom" else None,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "subsamples": int(args.subsamples) if args.subsamples is not None else None,
            "seed": int(args.seed),
            "device": args.device,
            "dataset": str(args.dataset.resolve()),
            "dataset_npz": str(dataset_path),
            "config_name": args.config_name,
            "obs_mode": obs_mode,
            "fixed_subsample_file": str(fixed_subsample_path) if fixed_subsample_path is not None else None,
        }
        (experiment_dir / "config.json").write_text(json.dumps(experiment_config, indent=2))

        update_grid_figure(
            grid_path,
            matrices=completed_matrices,
            feature_dims=args.feature_dims,
            curl_weights=args.curl_weights,
        )
        update_loss_grid_figure(
            loss_grid_path,
            metrics_by_experiment=completed_metrics,
            feature_dims=args.feature_dims,
            curl_weights=args.curl_weights,
        )
        env.close()

    print(f"Saved final grid correlation figure to {grid_path}")
    print(f"Saved final grid loss figure to {loss_grid_path}")


if __name__ == "__main__":
    main()
