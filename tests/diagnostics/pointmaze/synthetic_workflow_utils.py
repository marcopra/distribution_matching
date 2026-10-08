"""Shared data and encoder-checkpoint helpers for synthetic PointMaze studies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch


DATASET_FILENAME = "transitions.npz"
METADATA_FILENAME = "metadata.json"
CHECKPOINT_FORMAT = "pointmaze_synthetic_encoder_v1"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def dataset_checksum(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash names, dtypes, shapes, and bytes in stable key order."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def save_dataset(directory: Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> Dict[str, Any]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    checksum = dataset_checksum(normalized)
    payload = dict(metadata)
    payload.update(
        {
            "checksum": checksum,
            "array_shapes": {name: list(value.shape) for name, value in normalized.items()},
            "array_dtypes": {name: str(value.dtype) for name, value in normalized.items()},
        }
    )
    np.savez_compressed(directory / DATASET_FILENAME, **normalized)
    (directory / METADATA_FILENAME).write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n"
    )
    return payload


def load_dataset(directory: Path, verify: bool = True):
    directory = Path(directory)
    with np.load(directory / DATASET_FILENAME, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    metadata = json.loads((directory / METADATA_FILENAME).read_text())
    if verify:
        actual = dataset_checksum(arrays)
        expected = metadata.get("checksum")
        if actual != expected:
            raise ValueError(f"Dataset checksum mismatch: expected {expected}, got {actual}")
    return arrays, metadata


def arrays_to_tensors(arrays: Mapping[str, np.ndarray], device: str, compute_dtype: torch.dtype):
    return {
        "obs": torch.as_tensor(arrays["obs"], dtype=torch.float32, device=device),
        "action": torch.as_tensor(arrays["action"], dtype=torch.long, device=device),
        "reward": torch.as_tensor(arrays["reward"], dtype=compute_dtype, device=device),
        "discount": torch.as_tensor(arrays["discount"], dtype=compute_dtype, device=device),
        "next_obs": torch.as_tensor(arrays["next_obs"], dtype=torch.float32, device=device),
    }


def fixed_encoder_indices(total_size: int, batch_size: int, n_actions: int) -> np.ndarray:
    """Match PointMazeNystromDebugHelper.fixed_encoder_batch ordering."""
    size = min(int(batch_size), int(total_size))
    if size == total_size:
        return np.arange(total_size, dtype=np.int64)
    if n_actions > 0 and total_size % n_actions == 0 and size % n_actions == 0:
        n_states = total_size // n_actions
        n_selected_states = size // n_actions
        state_index = np.rint(np.linspace(0, n_states - 1, n_selected_states)).astype(np.int64)
        return (state_index[:, None] * n_actions + np.arange(n_actions)[None, :]).reshape(-1)
    return np.rint(np.linspace(0, total_size - 1, size)).astype(np.int64)


def shuffled_encoder_index_batches(
    total_size: int,
    batch_size: int,
    updates: int,
    seed: int,
):
    """Yield shuffled mini-batches, reshuffling after every full dataset pass."""
    total_size = int(total_size)
    batch_size = int(batch_size)
    updates = int(updates)
    if total_size <= 0:
        raise ValueError("total_size must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if updates < 0:
        raise ValueError("updates must be non-negative")

    rng = np.random.default_rng(int(seed))
    yielded = 0
    while yielded < updates:
        permutation = rng.permutation(total_size)
        for start in range(0, total_size, batch_size):
            yield permutation[start : start + batch_size]
            yielded += 1
            if yielded == updates:
                return


def save_encoder_checkpoint(
    path: Path,
    encoder: torch.nn.Module,
    *,
    feature_dim: int,
    obs_shape,
    mode: str,
    grayscale: bool,
    dataset_checksum_value: str,
    training_updates: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {name: tensor.detach().cpu() for name, tensor in encoder.state_dict().items()}
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "encoder_state_dict": state_dict,
            "feature_dim": int(feature_dim),
            "obs_shape": tuple(int(value) for value in obs_shape),
            "mode": str(mode),
            "grayscale": bool(grayscale),
            "dataset_checksum": str(dataset_checksum_value),
            "training_updates": int(training_updates),
        },
        path,
    )


def load_encoder_checkpoint(
    path: Path,
    encoder: torch.nn.Module,
    *,
    expected_feature_dim: int,
    expected_obs_shape,
    expected_dataset_checksum: str,
    device: str,
    allow_dataset_mismatch: bool = False,
) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported encoder checkpoint format in {path}")
    checks = {
        "feature_dim": (int(payload["feature_dim"]), int(expected_feature_dim)),
        "obs_shape": (tuple(payload["obs_shape"]), tuple(expected_obs_shape)),
    }
    mismatches = [f"{name}: checkpoint={actual!r}, expected={expected!r}" for name, (actual, expected) in checks.items() if actual != expected]
    if mismatches:
        raise ValueError("Encoder checkpoint mismatch: " + "; ".join(mismatches))
    checkpoint_checksum = payload.get("dataset_checksum")
    if checkpoint_checksum != expected_dataset_checksum:
        message = (
            "dataset_checksum: "
            f"checkpoint={checkpoint_checksum!r}, expected={expected_dataset_checksum!r}"
        )
        if not allow_dataset_mismatch:
            raise ValueError(
                "Encoder checkpoint mismatch: " + message
                + ". Pass --allow-dataset-mismatch to reuse this encoder on a different dataset."
            )
        print(f"WARNING: reusing encoder with {message}")
    encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    if hasattr(encoder, "mode"):
        encoder.mode = str(payload["mode"])
    encoder.to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return payload


def assert_module_unchanged(module: torch.nn.Module, reference: Mapping[str, torch.Tensor]) -> None:
    for name, tensor in module.state_dict().items():
        expected = reference[name].to(device=tensor.device, dtype=tensor.dtype)
        if not torch.equal(tensor, expected):
            raise RuntimeError(f"Frozen encoder parameter changed during PMD: {name}")
