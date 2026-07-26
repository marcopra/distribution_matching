"""Evaluate semantic coherence of a checkpoint encoder on Montezuma data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def encode(encoder, observations, device, batch_size):
    result = []
    parameter = next(encoder.parameters(), None)
    dtype = parameter.dtype if parameter is not None else torch.float32
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            batch = torch.as_tensor(
                observations[start:start + batch_size], device=device, dtype=dtype
            )
            z = (
                encoder.encode_and_project(batch)
                if hasattr(encoder, "encode_and_project")
                else encoder(batch)
            )
            result.append(F.normalize(z, dim=1).cpu().numpy())
    return np.concatenate(result)


def episode_split(episodes, fraction=0.7):
    unique = np.unique(episodes)
    cut = max(1, int(len(unique) * fraction))
    train_eps = set(unique[:cut])
    train = np.asarray([value in train_eps for value in episodes])
    test = ~train
    if not np.any(test):  # Tiny smoke datasets.
        cut = max(1, int(len(episodes) * fraction))
        train = np.arange(len(episodes)) < cut
        test = ~train
    return train, test


def probe_metrics(z, data):
    train, test = episode_split(data["episode"])
    scaler = StandardScaler().fit(z[train])
    x_train, x_test = scaler.transform(z[train]), scaler.transform(z[test])
    metrics = {}

    for name in ("x", "y"):
        model = Ridge(alpha=1.0).fit(x_train, data[name][train])
        metrics[f"{name}_r2_episode_holdout"] = float(
            r2_score(data[name][test], model.predict(x_test))
        )

    labels = data["room"]
    train_classes = np.unique(labels[train])
    test_known = test & np.isin(labels, train_classes)
    if len(train_classes) >= 2 and np.any(test_known):
        model = LogisticRegression(max_iter=2000, class_weight="balanced")
        model.fit(x_train, labels[train])
        metrics["room_accuracy_episode_holdout"] = float(
            accuracy_score(labels[test_known], model.predict(scaler.transform(z[test_known])))
        )
        metrics["room_majority_baseline"] = float(
            np.max(np.bincount(labels[test_known].astype(int))) / np.sum(test_known)
        )
    else:
        metrics["room_accuracy_episode_holdout"] = None
        metrics["room_majority_baseline"] = None
    return metrics


def neighborhood_metrics(z, data, xy_bin, k):
    count = min(k + 1, len(z))
    indices = NearestNeighbors(n_neighbors=count, metric="cosine").fit(z).kneighbors(
        return_distance=False
    )[:, 1:]
    same_room = data["room"][indices] == data["room"][:, None]
    same_bin = same_room & (
        data["x"][indices] // xy_bin == data["x"][:, None] // xy_bin
    ) & (data["y"][indices] // xy_bin == data["y"][:, None] // xy_bin)
    return {
        f"knn_{count - 1}_room_purity": float(same_room.mean()),
        f"knn_{count - 1}_position_bin_purity": float(same_bin.mean()),
    }, indices


def geometry_metrics(z, next_z, data, rng, pairs=50_000):
    n = len(z)
    i = rng.integers(n, size=min(pairs, n * 20))
    j = rng.integers(n, size=len(i))
    same_room = data["room"][i] == data["room"][j]
    spatial = np.hypot(data["x"][i] - data["x"][j], data["y"][i] - data["y"][j])
    latent = 1.0 - np.sum(z[i] * z[j], axis=1)
    valid = same_room & (i != j)
    correlation = spearmanr(spatial[valid], latent[valid]).statistic if valid.sum() > 5 else np.nan
    temporal = np.sum(z * next_z, axis=1)
    random_next = np.sum(z * next_z[rng.permutation(n)], axis=1)
    covariance = np.cov(z, rowvar=False)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)
    effective_rank = (eigenvalues.sum() ** 2) / (np.square(eigenvalues).sum() + 1e-12)
    return {
        "same_room_spatial_vs_latent_spearman": float(correlation),
        "temporal_cosine_mean": float(temporal.mean()),
        "random_pair_cosine_mean": float(random_next.mean()),
        "temporal_random_cosine_gap": float(temporal.mean() - random_next.mean()),
        "effective_rank": float(effective_rank),
        "embedding_dim": int(z.shape[1]),
    }, temporal, random_next


def save_plots(z, data, temporal, random_next, output):
    pca = PCA(n_components=2).fit_transform(z)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].scatter(pca[:, 0], pca[:, 1], c=data["room"], s=4, cmap="tab20")
    axes[0].set_title("PCA colored by room")
    axes[1].scatter(pca[:, 0], pca[:, 1], c=data["x"], s=4, cmap="viridis")
    axes[1].set_title("PCA colored by player x")
    axes[2].hist(temporal, bins=50, alpha=.65, label="successive")
    axes[2].hist(random_next, bins=50, alpha=.65, label="random")
    axes[2].set_title("Cosine similarity")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output / "semantic_summary.png", dpi=180)
    plt.close(fig)


def save_neighbor_sheet(data, neighbors, output, queries=12, columns=6):
    picks = np.linspace(0, len(neighbors) - 1, min(queries, len(neighbors)), dtype=int)
    tile = 84
    canvas = Image.new("RGB", (columns * tile, len(picks) * tile), "white")
    draw = ImageDraw.Draw(canvas)
    for row, query in enumerate(picks):
        selected = [query] + neighbors[query, :columns - 1].tolist()
        for col, index in enumerate(selected):
            frame = data["obs"][index, -1]
            image = Image.fromarray(frame).convert("RGB")
            canvas.paste(image, (col * tile, row * tile))
            draw.text(
                (col * tile + 2, row * tile + 2),
                f"r{int(data['room'][index])} {int(data['x'][index])},{int(data['y'][index])}",
                fill=(255, 40, 40),
            )
    canvas.save(output / "nearest_neighbors.png")


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    raw = np.load(args.dataset, allow_pickle=False)
    data = {key: raw[key] for key in raw.files}
    if args.max_samples and len(data["obs"]) > args.max_samples:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(len(data["obs"]), args.max_samples, replace=False))
        data = {key: value[keep] for key, value in data.items()}
    rng = np.random.default_rng(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    snapshot = torch.load(args.checkpoint, map_location=device, weights_only=False)
    agent = snapshot["agent"] if isinstance(snapshot, dict) and "agent" in snapshot else snapshot
    encoder = agent.policy_encoder if args.policy_encoder and hasattr(agent, "policy_encoder") else agent.encoder
    # CNNEncoder.forward currently casts observations to torch default dtype.
    # Normalize checkpoint dtype here because actor computations may have
    # converted saved modules to float64.
    encoder = encoder.to(device=device, dtype=torch.float32).eval()
    z = encode(encoder, data["obs"], device, args.batch_size)
    next_z = encode(encoder, data["next_obs"], device, args.batch_size)

    metrics = probe_metrics(z, data)
    neighbor_result, neighbors = neighborhood_metrics(z, data, args.xy_bin, args.k)
    metrics.update(neighbor_result)
    geometry_result, temporal, random_next = geometry_metrics(z, next_z, data, rng)
    metrics.update(geometry_result)
    metrics["samples"] = len(z)
    metrics["rooms_present"] = np.unique(data["room"]).astype(int).tolist()

    save_plots(z, data, temporal, random_next, output)
    save_neighbor_sheet(data, neighbors, output)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=True))
    np.save(output / "embeddings.npy", z)
    print(json.dumps(metrics, indent=2, allow_nan=True))
    print(f"Outputs: {output.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="tests/outputs/encoders/montezuma_semantics")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=20_000)
    parser.add_argument("--xy-bin", type=int, default=4)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--policy-encoder", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
