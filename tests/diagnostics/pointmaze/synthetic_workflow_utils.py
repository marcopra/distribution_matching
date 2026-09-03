"""Shared data and encoder-checkpoint helpers for synthetic PointMaze studies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch
import torch.nn as nn


DATASET_FILENAME = "transitions.npz"
METADATA_FILENAME = "metadata.json"
CHECKPOINT_FORMAT = "pointmaze_synthetic_encoder_v1"


class PCAWhitenedEncoder(nn.Module):
    """Apply a fixed PCA whitening transform after a frozen encoder."""

    def __init__(
        self,
        encoder: nn.Module,
        mean: torch.Tensor,
        components: torch.Tensor,
        eigenvalues: torch.Tensor,
        eigenvalue_floor: float,
        unit_trace: bool,
    ):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("whitening_mean", mean.detach().clone())
        self.register_buffer("whitening_components", components.detach().clone())
        self.register_buffer("whitening_eigenvalues", eigenvalues.detach().clone())
        self.eigenvalue_floor = float(eigenvalue_floor)
        self.unit_trace = bool(unit_trace)
        self.repr_dim = int(components.shape[0])
        self.mode = getattr(encoder, "mode", None)

    def forward(self, obs):
        return self.encoder(obs)

    def encode_and_project(self, obs):
        features = self.encoder.encode_and_project(obs)
        centered = features - self.whitening_mean
        projected = centered @ self.whitening_components.T
        denominator = torch.sqrt(
            torch.clamp(self.whitening_eigenvalues, min=self.eigenvalue_floor)
        )
        whitened = projected / denominator
        if self.unit_trace:
            whitened = whitened / np.sqrt(max(self.repr_dim, 1))
        return whitened


def fit_pca_whitening(
    encoder: nn.Module,
    observations,
    *,
    device: str,
    batch_size: int = 1024,
    explained_variance: float = 0.99,
    components: int = 0,
    epsilon: float = 1e-5,
    unit_trace: bool = True,
):
    """Fit PCA whitening on encoder outputs and return frozen wrapper plus metadata."""
    if not 0.0 < float(explained_variance) <= 1.0:
        raise ValueError("whitening explained_variance must be in (0, 1]")
    if float(epsilon) <= 0.0:
        raise ValueError("whitening epsilon must be positive")
    observations = np.asarray(observations)
    if observations.shape[0] < 2:
        raise ValueError("PCA whitening requires at least two observations")

    encoded_chunks = []
    effective_batch_size = max(1, int(batch_size))
    encoder.eval()
    with torch.no_grad():
        for start in range(0, observations.shape[0], effective_batch_size):
            batch = torch.as_tensor(
                observations[start : start + effective_batch_size],
                dtype=torch.float32,
                device=device,
            )
            encoded_chunks.append(encoder.encode_and_project(batch).detach().double().cpu())
    encoded = torch.cat(encoded_chunks, dim=0)
    mean = encoded.mean(dim=0)
    centered = encoded - mean
    covariance = centered.T @ centered / max(encoded.shape[0] - 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = torch.clamp(eigenvalues[order], min=0.0)
    eigenvectors = eigenvectors[:, order]

    max_rank = min(int(encoded.shape[0] - 1), int(encoded.shape[1]))
    if int(components) > 0:
        retained = min(int(components), max_rank)
    else:
        total_variance = float(eigenvalues.sum().item())
        if total_variance <= 0.0:
            raise ValueError("Cannot whiten constant encoder features")
        cumulative = torch.cumsum(eigenvalues, dim=0) / total_variance
        retained = int(torch.searchsorted(cumulative, float(explained_variance)).item()) + 1
        retained = min(max(retained, 1), max_rank)

    retained_values = eigenvalues[:retained]
    retained_components = eigenvectors[:, :retained].T
    largest = max(float(retained_values[0].item()), torch.finfo(torch.float64).eps)
    eigenvalue_floor = float(epsilon) * largest

    try:
        parameter = next(encoder.parameters())
        target_dtype = parameter.dtype
        target_device = parameter.device
    except StopIteration:
        target_dtype = torch.float32
        target_device = torch.device(device)
    wrapper = PCAWhitenedEncoder(
        encoder,
        mean.to(device=target_device, dtype=target_dtype),
        retained_components.to(device=target_device, dtype=target_dtype),
        retained_values.to(device=target_device, dtype=target_dtype),
        eigenvalue_floor=eigenvalue_floor,
        unit_trace=unit_trace,
    ).to(target_device)
    wrapper.eval()
    for parameter in wrapper.parameters():
        parameter.requires_grad_(False)

    retained_fraction = float(retained_values.sum().item() / max(eigenvalues.sum().item(), 1e-30))
    metadata = {
        "input_dim": int(encoded.shape[1]),
        "output_dim": int(retained),
        "explained_variance": retained_fraction,
        "requested_explained_variance": float(explained_variance),
        "epsilon": float(epsilon),
        "eigenvalue_floor": float(eigenvalue_floor),
        "unit_trace": bool(unit_trace),
        "mean": mean.numpy(),
        "components": retained_components.numpy(),
        "eigenvalues": retained_values.numpy(),
        "all_eigenvalues": eigenvalues.numpy(),
    }
    return wrapper, metadata


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
