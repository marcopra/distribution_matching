#!/usr/bin/env python3
"""Compare saved PointMaze encoders on exactly the same cached observations.

Each ``--model`` value has form ``LABEL=PATH``. PATH may be a full training
``snapshot.pt`` or an encoder-only checkpoint written by
``sweep_pointmaze_encoder_embeddings.py``.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch

from tests.diagnostics.encoders.evaluate_pointmaze_encoder import (
    load_snapshot_agent,
)
from tests.diagnostics.pointmaze.synthetic_workflow_utils import (
    CHECKPOINT_FORMAT,
    load_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PointMaze snapshot/standalone encoders on one fixed cached dataset."
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Model label and checkpoint path. Repeat for every encoder.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing transitions.npz and metadata.json.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-plot-points", type=int, default=4000)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--encoder-source",
        choices=("encoder", "policy_encoder"),
        default="encoder",
        help="Module used from full snapshots. Standalone checkpoints always represent encoder.",
    )
    return parser.parse_args()


def parse_model_specs(values: list[str]) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    labels: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --model {value!r}; expected LABEL=PATH")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        path = Path(raw_path).expanduser().resolve()
        if not label or not raw_path.strip():
            raise ValueError(f"Invalid --model {value!r}; label and path must be non-empty")
        if label in labels:
            raise ValueError(f"Duplicate model label: {label!r}")
        if not path.is_file():
            raise FileNotFoundError(path)
        labels.add(label)
        specs.append((label, path))
    return specs


def load_checkpoint_payload(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def is_encoder_only_checkpoint(path: Path) -> bool:
    if path.name.startswith("snapshot") or path.name in {"best_snapshot.pt", "final_snapshot.pt"}:
        return False
    # weights_only avoids unpickling the large agent/environment graph when probing.
    try:
        payload = load_checkpoint_payload(path, torch.device("cpu"))
    except Exception:
        return False
    return isinstance(payload, dict) and payload.get("format") == CHECKPOINT_FORMAT


def load_encoders(
    specs: list[tuple[str, Path]],
    device: torch.device,
    encoder_source: str,
    dataset_checksum: str,
) -> tuple[dict[str, torch.nn.Module], dict[str, dict[str, Any]]]:
    encoders: dict[str, torch.nn.Module] = {}
    details: dict[str, dict[str, Any]] = {}
    standalone: list[tuple[str, Path]] = []

    # Load full snapshots first. Their encoder supplies architecture for standalone state dicts.
    for label, path in specs:
        if is_encoder_only_checkpoint(path):
            standalone.append((label, path))
            continue
        print(f"Loading snapshot: {label} <- {path}")
        agent = load_snapshot_agent(path, device)
        encoder = getattr(agent, encoder_source, None)
        if encoder is None:
            raise AttributeError(f"{path} has no {encoder_source!r}")
        encoder.eval()
        encoders[label] = encoder
        details[label] = {
            "path": str(path),
            "checkpoint_type": "snapshot",
            "encoder_source": encoder_source,
        }

    if standalone and not encoders:
        raise ValueError(
            "At least one full snapshot.pt is required as architecture template when an "
            "encoder-only checkpoint is supplied"
        )
    template = next(iter(encoders.values()), None)
    for label, path in standalone:
        print(f"Loading standalone encoder: {label} <- {path}")
        payload = load_checkpoint_payload(path, device)
        checkpoint_checksum = payload.get("dataset_checksum")
        if checkpoint_checksum != dataset_checksum:
            raise ValueError(
                f"{path}: dataset checksum {checkpoint_checksum!r} does not match "
                f"fixed dataset {dataset_checksum!r}"
            )
        encoder = copy.deepcopy(template)
        encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
        if hasattr(encoder, "mode"):
            encoder.mode = str(payload["mode"])
        encoder.to(device)
        encoder.eval()
        encoders[label] = encoder
        details[label] = {
            "path": str(path),
            "checkpoint_type": "encoder_only",
            "feature_dim": int(payload["feature_dim"]),
            "training_updates": int(payload["training_updates"]),
            "dataset_checksum": checkpoint_checksum,
        }

    # Restore command-line order.
    return ({label: encoders[label] for label, _ in specs}, details)


def module_dtype(module: torch.nn.Module) -> torch.dtype:
    for parameter in module.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return torch.float32


def encode_observations(
    encoder: torch.nn.Module,
    observations: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    chunks = []
    dtype = module_dtype(encoder)
    with torch.inference_mode():
        for start in range(0, observations.shape[0], batch_size):
            batch = torch.as_tensor(
                observations[start : start + batch_size], device=device, dtype=dtype
            )
            chunks.append(encoder.encode_and_project(batch).detach().float().cpu())
    return torch.cat(chunks).numpy()


def uniform_indices(size: int, maximum: int) -> np.ndarray:
    if maximum <= 0:
        raise ValueError("--max-plot-points must be positive")
    if size <= maximum:
        return np.arange(size, dtype=np.int64)
    return np.rint(np.linspace(0, size - 1, maximum)).astype(np.int64)


def project_pca(embedding: np.ndarray) -> np.ndarray:
    return PCA(n_components=2, random_state=0).fit_transform(embedding)


def project_tsne(embedding: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    effective = min(float(perplexity), max(1.0, (embedding.shape[0] - 1) / 3.0))
    return TSNE(
        n_components=2,
        perplexity=effective,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(embedding)


def save_comparison_figure(
    path: Path,
    projections: dict[str, np.ndarray],
    colors: np.ndarray,
    projection_name: str,
) -> None:
    count = len(projections)
    fig, axes = plt.subplots(1, count, figsize=(4.6 * count, 4.4), squeeze=False)
    scatter = None
    for axis, (label, projected) in zip(axes[0], projections.items()):
        scatter = axis.scatter(
            projected[:, 0], projected[:, 1], c=colors, s=9, cmap="viridis",
            alpha=0.82, linewidths=0,
        )
        axis.set_title(label)
        axis.set_xlabel("PC1" if projection_name == "pca" else "t-SNE 1")
        axis.set_ylabel("PC2" if projection_name == "pca" else "t-SNE 2")
    if scatter is not None:
        fig.colorbar(scatter, ax=axes.ravel().tolist(), fraction=0.025, pad=0.025, label="x + 0.37 y")
    fig.suptitle(f"PointMaze fixed-data encoder {projection_name.upper()} comparison")
    fig.subplots_adjust(left=0.06, right=0.93, bottom=0.13, top=0.84, wspace=0.28)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    specs = parse_model_specs(args.model)
    arrays, metadata = load_dataset(args.dataset_dir.resolve())
    n_actions = int(metadata["n_actions"])
    observations = np.asarray(arrays["obs"])[::n_actions]
    xy = np.asarray(arrays["xy"], dtype=np.float32).reshape(-1, 2)
    if observations.shape[0] != xy.shape[0]:
        raise ValueError(
            f"Dataset ordering mismatch: {observations.shape[0]} unique observations, {xy.shape[0]} XY rows"
        )

    device = torch.device(args.device)
    encoders, model_details = load_encoders(
        specs, device, args.encoder_source, str(metadata["checksum"])
    )
    plot_indices = uniform_indices(xy.shape[0], args.max_plot_points)
    colors = xy[plot_indices, 0] + 0.37 * xy[plot_indices, 1]
    embeddings: dict[str, np.ndarray] = {}
    for label, encoder in encoders.items():
        print(f"Encoding {observations.shape[0]} fixed states with {label}")
        embedding = encode_observations(encoder, observations, device, args.batch_size)
        if embedding.shape[0] != observations.shape[0]:
            raise RuntimeError(f"{label}: encoder returned wrong row count {embedding.shape}")
        embeddings[label] = embedding
        model_details[label]["embedding_shape"] = list(embedding.shape)
        del encoder
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pca: dict[str, np.ndarray] = {}
    tsne: dict[str, np.ndarray] = {}
    archive: dict[str, np.ndarray] = {"xy": xy, "plot_indices": plot_indices}
    for index, (label, embedding) in enumerate(embeddings.items()):
        plotted = embedding[plot_indices]
        pca[label] = project_pca(plotted)
        tsne[label] = project_tsne(plotted, args.tsne_perplexity, args.seed + index)
        archive[f"embedding_{index}"] = embedding
        archive[f"pca_{index}"] = pca[label]
        archive[f"tsne_{index}"] = tsne[label]

    save_comparison_figure(args.output_dir / "encoder_comparison_pca.png", pca, colors, "pca")
    save_comparison_figure(args.output_dir / "encoder_comparison_tsne.png", tsne, colors, "tsne")
    np.savez_compressed(args.output_dir / "encoder_comparison_data.npz", **archive)
    manifest = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "dataset_checksum": metadata["checksum"],
        "n_fixed_states": int(observations.shape[0]),
        "plot_indices": int(plot_indices.shape[0]),
        "seed": int(args.seed),
        "tsne_perplexity": float(args.tsne_perplexity),
        "models": model_details,
        "archive_index_to_label": {str(i): label for i, (label, _) in enumerate(specs)},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved comparison outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
