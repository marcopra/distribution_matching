"""Pretrain Rover's PointMaze pixel CNN with two-view InfoNCE on replay images."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import utils
from agent.rover_networks import CNNEncoder


def load_image_pool(episode_paths: list[Path]) -> tuple[np.ndarray, int]:
    images = []
    transition_count = 0
    for path in episode_paths:
        with np.load(path) as episode:
            if "observation" not in episode:
                continue
            episode_images = np.asarray(episode["observation"][:-1])
            if episode_images.ndim != 4 or episode_images.dtype != np.uint8:
                raise ValueError(
                    f"{path} has observation shape/dtype "
                    f"{episode_images.shape}/{episode_images.dtype}; "
                    "expected uint8 stacked pixels [T,C,H,W]"
                )
            if episode_images.shape[0] == 0:
                continue
            images.append(episode_images.copy())
            transition_count += int(episode_images.shape[0])
    if not images:
        raise ValueError("Replay directory contains no pixel transitions")
    return np.concatenate(images, axis=0), transition_count


def make_episode_pools(
    replay_dir: Path,
    pool_size: int,
    seed: int,
) -> list[list[Path]]:
    paths = list(replay_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No replay episodes found in {replay_dir}")
    random.Random(seed).shuffle(paths)
    pools = []
    current_pool = []
    current_size = 0
    for path in paths:
        try:
            # ReplayBufferStorage filenames end in ``_<transition-count>.npz``.
            # Avoid decompressing image arrays just to assemble the pools.
            size = max(0, int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            with np.load(path) as episode:
                if "observation" not in episode:
                    continue
                size = max(0, int(episode["observation"].shape[0]) - 1)
        if size == 0:
            continue
        current_pool.append(path)
        current_size += size
        if current_size >= pool_size:
            pools.append(current_pool)
            current_pool = []
            current_size = 0
    if current_pool:
        pools.append(current_pool)
    return pools


def train_pool(
    images: np.ndarray,
    encoder: CNNEncoder,
    optimizer: torch.optim.Optimizer,
    augmenter: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    epochs: int,
    temperature: float,
    step_offset: int,
) -> tuple[int, float]:
    if len(images) < 2:
        raise ValueError("InfoNCE pool must contain at least two transitions")
    steps_per_epoch = max(1, int(np.ceil(len(images) / batch_size)))
    final_loss = float("nan")
    step = step_offset
    encoder.train()
    for _ in range(epochs):
        for _ in range(steps_per_epoch):
            indices = np.random.randint(0, len(images), size=batch_size)
            batch = torch.as_tensor(images[indices], device=device)
            view_a = augmenter(batch)
            view_b = augmenter(batch)
            z_a = F.normalize(encoder.encode_and_project(view_a), p=2, dim=1)
            z_b = F.normalize(encoder.encode_and_project(view_b), p=2, dim=1)
            logits = (z_a @ z_b.T) / temperature
            labels = torch.arange(batch_size, device=device)
            loss = 0.5 * (
                F.cross_entropy(logits, labels)
                + F.cross_entropy(logits.T, labels)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().item())
            step += 1
    return step, final_loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--mode", choices=("l1", "l2"), default="l1")
    parser.add_argument("--grayscale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frame-stack", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.batch_size < 2 or args.epochs < 1 or args.pool_size < 2:
        raise ValueError("batch-size and pool-size must be >=2; epochs must be >=1")
    if args.temperature <= 0 or args.learning_rate <= 0:
        raise ValueError("temperature and learning-rate must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {args.device!r}, but CUDA is unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    pools = make_episode_pools(args.replay_dir, args.pool_size, args.seed)
    first_pool, _ = load_image_pool(pools[0])
    obs_shape = tuple(int(dim) for dim in first_pool.shape[1:])
    del first_pool
    if args.frame_stack < 1:
        raise ValueError("frame-stack must be positive")
    expected_channels = args.frame_stack * (1 if args.grayscale else 3)
    if len(obs_shape) != 3 or obs_shape[0] != expected_channels:
        raise ValueError(
            f"Replay observation shape is {obs_shape}; expected "
            f"({expected_channels}, H, W) for grayscale={args.grayscale}"
        )

    device = torch.device(args.device)
    encoder = CNNEncoder(obs_shape, args.feature_dim, mode=args.mode).to(device)
    optimizer = torch.optim.AdamW(
        encoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    augmenter = utils.RandomShiftsAug(pad=4)
    step = 0
    transition_count = 0
    last_loss = float("nan")
    for pool_index, pool_paths in enumerate(pools, start=1):
        images, count = load_image_pool(pool_paths)
        if tuple(images.shape[1:]) != obs_shape:
            raise ValueError(
                f"Replay episode shape changed: {images.shape[1:]} vs {obs_shape}"
            )
        transition_count += count
        step, last_loss = train_pool(
            images,
            encoder,
            optimizer,
            augmenter,
            device,
            args.batch_size,
            args.epochs,
            args.temperature,
            step,
        )
        print(
            f"pool {pool_index}/{len(pools)}: {count:,} images, "
            f"steps={step}, InfoNCE={last_loss:.5f}"
        )
        del images

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "encoder_state_dict": encoder.cpu().state_dict(),
        "feature_dim": int(args.feature_dim),
        "obs_shape": obs_shape,
        "mode": args.mode,
        "grayscale": bool(args.grayscale),
        "pretraining": {
            "method": "two-view symmetric InfoNCE",
            "replay_dir": str(args.replay_dir.resolve()),
            "transitions": int(transition_count),
            "steps": int(step),
            "epochs_per_pool": int(args.epochs),
            "pool_size": int(args.pool_size),
            "batch_size": int(args.batch_size),
            "temperature": float(args.temperature),
            "learning_rate": float(args.learning_rate),
            "final_loss": float(last_loss),
            "seed": int(args.seed),
        },
    }
    torch.save(payload, args.output)
    print(f"saved compatible Rover encoder checkpoint: {args.output.resolve()}")


if __name__ == "__main__":
    main()
