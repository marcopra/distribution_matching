"""Test Montezuma transition prediction and conditional-mean operator quality.

Compares three predictors on episode-held-out transitions:
1. persistence: phi(s') = phi(s);
2. checkpoint nonlinear ProjectSA(phi(s), a);
3. Gaussian KRR conditional-mean operator fitted in checkpoint representation.
"""

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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def encode(encoder, observations, device, batch_size):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            obs = torch.as_tensor(
                observations[start:start + batch_size],
                device=device,
                dtype=torch.float32,
            )
            z = (
                encoder.encode_and_project(obs)
                if hasattr(encoder, "encode_and_project")
                else encoder(obs)
            )
            chunks.append(F.normalize(z, dim=1).cpu())
    return torch.cat(chunks)


def episode_split(episodes, seed, train_episodes_fraction):
    unique = np.unique(episodes)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    cut = max(1, min(len(unique) - 1, int(len(unique) * train_episodes_fraction)))
    train_episodes = set(unique[:cut])
    train = np.flatnonzero(np.asarray([episode in train_episodes for episode in episodes]))
    test = np.flatnonzero(np.asarray([episode not in train_episodes for episode in episodes]))
    if len(test) == 0:
        raise ValueError("Dataset needs at least two episodes for held-out evaluation")
    return train, test


def stratified_limit(indices, actions, maximum, rng):
    if len(indices) <= maximum:
        return indices
    groups = []
    unique_actions = np.unique(actions[indices])
    quota = max(1, maximum // len(unique_actions))
    for action in unique_actions:
        candidates = indices[actions[indices] == action]
        groups.append(rng.choice(candidates, min(quota, len(candidates)), replace=False))
    selected = np.concatenate(groups)
    if len(selected) < maximum:
        remaining = np.setdiff1d(indices, selected, assume_unique=False)
        selected = np.concatenate(
            [selected, rng.choice(remaining, min(maximum - len(selected), len(remaining)), replace=False)]
        )
    return np.sort(selected[:maximum])


def nonlinear_predictions(agent, z, actions, device, batch_size):
    predictions = []
    project_sa = agent.project_sa.to(device=device, dtype=torch.float32).eval()
    with torch.no_grad():
        for start in range(0, len(z), batch_size):
            state = z[start:start + batch_size].to(device)
            action = torch.as_tensor(
                actions[start:start + batch_size], device=device, dtype=torch.long
            )
            psi = agent._encode_state_action(state, action)
            predictions.append(F.normalize(project_sa(psi), dim=1).cpu())
    return torch.cat(predictions)


def gaussian_kernel(x, y, bandwidth):
    distance = torch.cdist(x, y).square()
    return torch.exp(-distance / (2.0 * bandwidth * bandwidth))


def fit_predict_krr(z, next_z, actions, train, test, bandwidth, regularization):
    prediction = torch.empty((len(test), z.shape[1]), dtype=z.dtype)
    test_actions = actions[test]
    for action in np.unique(test_actions):
        train_a = train[actions[train] == action]
        test_positions = np.flatnonzero(test_actions == action)
        test_a = test[test_positions]
        if len(train_a) == 0:
            prediction[test_positions] = z[test_a]
            continue
        x_train = z[train_a]
        kernel = gaussian_kernel(x_train, x_train, bandwidth)
        kernel.diagonal().add_(regularization)
        coefficients = torch.linalg.solve(kernel, next_z[train_a])
        prediction[test_positions] = gaussian_kernel(z[test_a], x_train, bandwidth) @ coefficients
    return F.normalize(prediction, dim=1)


def predictor_metrics(name, prediction, target, current, next_x, next_y,
                      coordinate_probe_x, coordinate_probe_y, neighbor_index,
                      bank_x, bank_y):
    cosine = torch.sum(prediction * target, dim=1).numpy()
    persistence_cosine = torch.sum(current * target, dim=1).numpy()
    pred_np = prediction.numpy()
    neighbor = neighbor_index.kneighbors(pred_np, return_distance=False)[:, 0]
    spatial_error = np.hypot(bank_x[neighbor] - next_x, bank_y[neighbor] - next_y)
    return {
        "name": name,
        "cosine_mean": float(cosine.mean()),
        "cosine_median": float(np.median(cosine)),
        "mse": float(torch.mean((prediction - target) ** 2)),
        "cosine_gain_over_persistence": float((cosine - persistence_cosine).mean()),
        "decoded_x_r2": float(r2_score(next_x, coordinate_probe_x.predict(pred_np))),
        "decoded_y_r2": float(r2_score(next_y, coordinate_probe_y.predict(pred_np))),
        "embedding_nn_position_error_mean": float(spatial_error.mean()),
        "embedding_nn_position_error_median": float(np.median(spatial_error)),
    }


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    data_file = np.load(args.dataset, allow_pickle=False)
    data = {key: data_file[key] for key in data_file.files}
    train, test = episode_split(data["episode"], args.seed, args.train_episode_fraction)
    train = stratified_limit(train, data["action"], args.train_samples, rng)
    test = stratified_limit(test, data["action"], args.test_samples, rng)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    agent = checkpoint["agent"] if isinstance(checkpoint, dict) and "agent" in checkpoint else checkpoint
    encoder = agent.encoder.to(device=device, dtype=torch.float32).eval()
    all_indices = np.concatenate([train, test])
    all_z = encode(encoder, data["obs"][all_indices], device, args.batch_size)
    all_next_z = encode(encoder, data["next_obs"][all_indices], device, args.batch_size)
    z_train, z_test = all_z[:len(train)], all_z[len(train):]
    next_train, next_test = all_next_z[:len(train)], all_next_z[len(train):]

    nonlinear_all = nonlinear_predictions(
        agent, all_z, data["action"][all_indices], device, args.batch_size
    )
    nonlinear_test = nonlinear_all[len(train):]

    probe_x = Ridge(alpha=1.0).fit(next_train.numpy(), data["next_x"][train])
    probe_y = Ridge(alpha=1.0).fit(next_train.numpy(), data["next_y"][train])
    neighbor_index = NearestNeighbors(n_neighbors=1, metric="cosine").fit(next_train.numpy())

    common = dict(
        target=next_test,
        current=z_test,
        next_x=data["next_x"][test],
        next_y=data["next_y"][test],
        coordinate_probe_x=probe_x,
        coordinate_probe_y=probe_y,
        neighbor_index=neighbor_index,
        bank_x=data["next_x"][train],
        bank_y=data["next_y"][train],
    )
    results = [
        predictor_metrics("persistence_phi_s", z_test, **common),
        predictor_metrics("checkpoint_nonlinear_project_sa", nonlinear_test, **common),
    ]

    checkpoint_bandwidth = getattr(getattr(agent, "kernel_fn", None), "bandwidth", None)
    base_bandwidth = args.bandwidth if args.bandwidth is not None else checkpoint_bandwidth
    if base_bandwidth is None:
        raise ValueError("No fitted checkpoint bandwidth; pass --bandwidth")
    for multiplier in args.bandwidth_multipliers:
        bandwidth = float(base_bandwidth) * multiplier
        for regularization in args.regularizations:
            combined_z = torch.cat([z_train, z_test])
            combined_next = torch.cat([next_train, next_test])
            combined_actions = data["action"][all_indices]
            prediction = fit_predict_krr(
                combined_z, combined_next, combined_actions,
                np.arange(len(train)), np.arange(len(train), len(all_indices)),
                bandwidth, regularization,
            )
            name = f"gaussian_krr_bw_x{multiplier:g}_lambda_{regularization:g}"
            result = predictor_metrics(name, prediction, **common)
            result["bandwidth"] = bandwidth
            result["regularization"] = regularization
            results.append(result)

    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": checkpoint.get("_global_step") if isinstance(checkpoint, dict) else None,
        "checkpoint_kernel_type": getattr(agent, "kernel_type", None),
        "checkpoint_fitted_bandwidth": checkpoint_bandwidth,
        "train_samples": len(train),
        "test_samples": len(test),
        "results": results,
    }
    (output / "operator_metrics.json").write_text(json.dumps(payload, indent=2))

    names = [item["name"] for item in results]
    cosine = [item["cosine_mean"] for item in results]
    error = [item["embedding_nn_position_error_mean"] for item in results]
    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(names) * 1.5), 5))
    axes[0].bar(names, cosine)
    axes[0].set_ylabel("cosine(predicted phi(s'), phi(s'))")
    axes[0].set_ylim(-1, 1)
    axes[1].bar(names, error)
    axes[1].set_ylabel("decoded next-position error")
    for axis in axes:
        axis.tick_params(axis="x", rotation=65)
    fig.tight_layout()
    fig.savefig(output / "operator_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(payload, indent=2))
    print(f"Outputs: {output.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="tests/outputs/encoders/montezuma_operator")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-samples", type=int, default=4000)
    parser.add_argument("--test-samples", type=int, default=2000)
    parser.add_argument("--train-episode-fraction", type=float, default=0.7)
    parser.add_argument("--bandwidth", type=float)
    parser.add_argument("--bandwidth-multipliers", type=float, nargs="+", default=[1.0, 5.0])
    parser.add_argument("--regularizations", type=float, nargs="+", default=[1e-4, 1e-2])
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
