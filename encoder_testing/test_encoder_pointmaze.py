#!/usr/bin/env python3
"""Evaluate a saved Rover encoder on PointMaze transitions and controlled samples.

The script loads a pretrain snapshot in the same spirit as ``pretrain.py``:
``torch.load(snapshot)["agent"]``.  It then uses the saved agent methods for
phi(s), psi(s,a), and the learned transition projector:

    projected(psi(s,a)) = agent.project_sa(agent._encode_state_action(phi(s), a))
    phi(s')              = encoder.encode_and_project(s')

For the velocity diagnostic, PointMaze does not expose velocity directly in the
pixel observation.  With frame stacking, velocity is visible only through the
history of rendered positions.  To isolate this from the action, the script
constructs each stacked observation by rendering a short kinematic history:

    xy_k = xy_current - (num_frames - 1 - k) * velocity * env_dt

This tests whether phi(s) separates stacks that imply different velocities,
without mixing in the next-state effect of a discrete action.  If you want to
compare next states too, the transition distance statistics already test the
action-conditioned projector on real episode transitions.

Examples
--------
python encoder_testing/test_encoder_pointmaze.py \
    --snapshot exp_local/.../models/pixels/gym/dist_matching/1/snapshot_50000.pt \
    --episode-path exp_local/.../buffer \
    --output-dir encoder_testing/outputs/pointmaze_encoder_eval \
    --config-name pretrain/pretrain_pointmaze_umaze_1 \
    --device cuda

python encoder_testing/test_encoder_pointmaze.py \
    --snapshot /home/mprattico-iit.local/distribution_matching/exp_local/2026.04.27/155642_655032_PointMaze_UMaze-v3/models/pixels/gym/dist_matching/1/snapshot.pt \
    --episode-path exp_local/2026.04.27/155642_655032_PointMaze_UMaze-v3/buffer \
    --output-dir encoder_testing/outputs/pointmaze_encoder_eval \
    --config-name pretrain/pretrain_pointmaze_umaze_1 \
    --device cuda
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ROVER_IMPORT_ERRORS = {}
for module_name in (
    "agent.rover",
    "agent.rover_nystrom",
    "agent.rover_nystrom_memory_efficient",
):
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        ROVER_IMPORT_ERRORS[module_name] = str(exc)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from sklearn.manifold import TSNE
import torch
import torch.nn.functional as F

import gym_env
from replay_buffer import episode_len, load_episode


DEFAULT_SPATIAL_LOCATIONS = [
    (-1.0, 1.0),
    (-0.25, 1.0),
    (1.0, 1.0),
    (-1.0, 0.0),
    (-1.0, -1.0),
    (0.25, -1.0),
    (1.0, -1.0),
]

DEFAULT_VELOCITY_LOCATIONS = [
    (-1.0, 1.0),
    (1.0, -1.0),
    (-1.0, -1.0),
    (0, -1.0),
]

DEFAULT_VELOCITIES = [
    (0.0, 0.0),
    (5.0, 0.0),
    (0.0, 5.0),
    (5.0, 5.0),
]


def parse_xy_pairs(values: Optional[Sequence[str]], default: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not values:
        return list(default)
    if len(values) % 2 != 0:
        raise ValueError("Expected an even number of values: x1 y1 x2 y2 ...")
    return [(float(values[i]), float(values[i + 1])) for i in range(0, len(values), 2)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved Rover encoder on PointMaze episodes and controlled states."
    )
    parser.add_argument("--snapshot", type=Path, required=True, help="Path to snapshot.pt or snapshot_<frame>.pt.")
    parser.add_argument(
        "--episode-path",
        type=Path,
        required=True,
        help="Path to one replay .npz episode or a directory containing replay .npz files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for JSON and plot outputs.")
    parser.add_argument(
        "--config-name",
        default="pretrain/pretrain_pointmaze_umaze_1",
        help="Hydra config name used to recreate the PointMaze env for diagnostic sampling.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for sampling.")
    parser.add_argument("--nstep", type=int, default=1, help="n-step transition offset used for episode loading.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Encoding batch size.")
    parser.add_argument("--max-transitions", type=int, default=4096, help="Maximum episode transitions used for statistics.")
    parser.add_argument(
        "--negatives-per-positive",
        type=int,
        default=32,
        help="Number of mismatched next states sampled per positive transition.",
    )
    parser.add_argument(
        "--encoder-source",
        choices=["encoder", "policy_encoder"],
        default="encoder",
        help="Which saved encoder module to evaluate.",
    )
    parser.add_argument(
        "--apply-aug",
        action="store_true",
        help="Apply agent.aug before encoding. By default diagnostics use raw observations without random shifts.",
    )
    parser.add_argument(
        "--spatial-locations",
        nargs="*",
        help="Spatial diagnostic centers as x y pairs. Defaults to U-Maze waypoints.",
    )
    parser.add_argument("--spatial-samples-per-location", type=int, default=8)
    parser.add_argument("--spatial-jitter", type=float, default=0.04)
    parser.add_argument(
        "--velocity-locations",
        nargs="*",
        help="Velocity diagnostic starting positions as x y pairs. Defaults to two U-Maze waypoints.",
    )
    parser.add_argument(
        "--velocities",
        nargs="*",
        help="Velocity values as vx vy pairs. Defaults to four velocities.",
    )
    parser.add_argument("--velocity-samples-per-setting", type=int, default=2)
    parser.add_argument("--velocity-position-jitter", type=float, default=0.0)
    parser.add_argument("--velocity-gif-duration-ms", type=int, default=2000)
    parser.add_argument(
        "--save-sampled-frames",
        action="store_true",
        help="Save sampled frame stacks as per-frame PNGs. Disabled by default.",
    )
    parser.add_argument(
        "--save-velocity-gif",
        action="store_true",
        help="Save an animated GIF of velocity frame-stack histories. Disabled by default.",
    )
    parser.add_argument(
        "--history-dt",
        type=float,
        default=None,
        help="Time delta between stacked rendered frames. Defaults to PointMaze point_env.dt.",
    )
    parser.add_argument(
        "--history-dt-multiplier",
        type=float,
        default=1.0,
        help="Multiplier applied to history-dt when constructing velocity stacks.",
    )
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    return parser.parse_args()


def _forward_getattr_safely(self, name):
    env = self.__dict__.get("env", None)
    if env is None:
        raise AttributeError(name)
    return getattr(env, name)


def resolve_torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None:
        return None
    dtype = str(dtype).lower()
    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "32": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
        "double": torch.float64,
        "64": torch.float64,
    }
    if dtype not in dtype_map:
        return None
    return dtype_map[dtype]


def infer_agent_dtype(agent):
    dtype = resolve_torch_dtype(getattr(agent, "compute_dtype", None))
    if dtype is not None:
        return dtype
    for module_name in ("encoder", "policy_encoder", "project_sa"):
        module = getattr(agent, module_name, None)
        if module is None or not hasattr(module, "parameters"):
            continue
        for param in module.parameters():
            if param.is_floating_point():
                return param.dtype
    return torch.get_default_dtype()


@contextmanager
def temporarily_safe_wrapper_getattrs():
    """Avoid recursive __getattr__ calls while unpickling env-containing snapshots."""
    class_specs = [
        ("env.pointmaze_domain", "FixedPointMazeResetWrapper"),
        ("env.pointmaze_domain", "PointMazeTopDownCameraWrapper"),
        ("env.pointmaze_domain", "PointMazeGoalMaskWrapper"),
        ("env.pointmaze_domain", "PointMazeDiscreteActions"),
        ("gym_env", "ResizeRendering"),
        ("gym_env", "DiscreteObservationWrapper"),
        ("gym_env", "ActionRepeatWrapper"),
        ("gym_env", "FrameStackWrapper"),
        ("gym_env", "ActionDTypeWrapper"),
        ("gym_env", "IgnoreSuccessTerminationWrapper"),
        ("gym_env", "ExtendedTimeStepWrapper"),
        ("gym_env", "TerminateOnPoint"),
    ]
    originals = []
    for module_name, class_name in class_specs:
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
        except (ImportError, AttributeError):
            continue
        if "__getattr__" in cls.__dict__:
            originals.append((cls, cls.__dict__["__getattr__"]))
            cls.__getattr__ = _forward_getattr_safely
    try:
        yield
    finally:
        for cls, original in originals:
            cls.__getattr__ = original


def scrub_snapshot_env_refs(agent) -> None:
    """Drop environment/debug references after loading; only encoders are needed here."""
    for attr in (
        "env",
        "wrapped_env",
        "_discrete_env",
        "visualizer",
        "gridworld_visualizer",
    ):
        if attr in getattr(agent, "__dict__", {}):
            setattr(agent, attr, None)

    debug_visualizer = getattr(agent, "__dict__", {}).get("debug_visualizer", None)
    if debug_visualizer is not None:
        for attr in ("env", "wrapped_env", "exploration_visualizer", "gridworld_visualizer"):
            if attr in getattr(debug_visualizer, "__dict__", {}):
                setattr(debug_visualizer, attr, None)


def load_snapshot_agent(snapshot_path: Path, device: torch.device):
    def _torch_load(weights_only: Optional[bool]):
        kwargs = {"map_location": device}
        if weights_only is not None:
            kwargs["weights_only"] = weights_only
        return torch.load(snapshot_path, **kwargs)

    try:
        with temporarily_safe_wrapper_getattrs():
            try:
                payload = _torch_load(weights_only=False)
            except TypeError:
                payload = _torch_load(weights_only=None)
    except ModuleNotFoundError as exc:
        details = "\n".join(f"  {name}: {err}" for name, err in ROVER_IMPORT_ERRORS.items())
        raise ModuleNotFoundError(
            f"Could not load snapshot because a Python module is missing: {exc}. "
            "Run this script from the same environment used for pretraining. "
            f"Earlier rover import errors were:\n{details}"
        ) from exc

    agent = payload["agent"] if isinstance(payload, dict) and "agent" in payload else payload
    scrub_snapshot_env_refs(agent)
    agent_dtype = infer_agent_dtype(agent)
    torch.set_default_dtype(agent_dtype)
    agent.device = str(device)
    for name in ("encoder", "policy_encoder", "project_sa"):
        module = getattr(agent, name, None)
        if module is not None and hasattr(module, "to"):
            module.to(device=device, dtype=agent_dtype)
            module.eval()
    if hasattr(agent, "train"):
        agent.train(False)
    return agent


def select_encoder(agent, source: str):
    if source == "policy_encoder":
        module = getattr(agent, "policy_encoder", None)
        if module is None:
            raise AttributeError("Snapshot agent does not have policy_encoder")
        return module
    return agent.encoder


def resolve_group_path(group_value: str, group_name: str) -> Path:
    direct = CONFIG_DIR / f"{group_value}.yaml"
    if direct.exists():
        return direct
    grouped = CONFIG_DIR / group_name / f"{group_value}.yaml"
    if grouped.exists():
        return grouped
    raise FileNotFoundError(f"Could not resolve {group_name} config '{group_value}'")


def resolve_config_path(config_name: str) -> Path:
    value = Path(config_name)
    candidates = []
    if value.suffix in {".yaml", ".yml"}:
        candidates.extend([value, REPO_ROOT / value, CONFIG_DIR / value])
    else:
        candidates.extend([CONFIG_DIR / f"{config_name}.yaml", REPO_ROOT / f"{config_name}.yaml"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve config '{config_name}'")


def compose_cfg(config_name: str, seed: int):
    cfg = OmegaConf.load(resolve_config_path(config_name))
    defaults = OmegaConf.to_container(cfg.get("defaults", []), resolve=False)
    env_default = None
    for item in defaults:
        if isinstance(item, dict) and "/env" in item:
            env_default = item["/env"]
        elif isinstance(item, dict) and "env" in item:
            env_default = item["env"]
    if env_default is None:
        raise ValueError(f"Could not recover env default from config {config_name}")
    env_cfg = OmegaConf.load(resolve_group_path(env_default, "env"))
    return OmegaConf.merge(env_cfg, cfg, {"seed": seed})


def make_pointmaze_env(cfg):
    env_kwargs = OmegaConf.to_container(cfg.env, resolve=True)
    env_kwargs.pop("name", None)
    return gym_env.make(
        cfg.env.name,
        cfg.obs_type,
        frame_stack=cfg.frame_stack,
        action_repeat=cfg.action_repeat,
        seed=cfg.seed,
        resolution=cfg.resolution,
        grayscale=bool(getattr(cfg, "grayscale", False)),
        url=True,
        **env_kwargs,
    )


def episode_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz episodes found in {path}")
    return files


def load_transitions(path: Path, nstep: int, max_transitions: int, rng: np.random.Generator):
    obs_chunks, action_chunks, next_obs_chunks = [], [], []
    for fn in episode_files(path):
        episode = load_episode(fn)
        transition_count = episode_len(episode) - nstep + 1
        if transition_count <= 0:
            continue
        obs_chunks.append(episode["observation"][:transition_count])
        action_chunks.append(episode["action"][1 : transition_count + 1])
        next_obs_chunks.append(episode["observation"][nstep : transition_count + nstep])

    if not obs_chunks:
        raise RuntimeError(f"No valid transitions found in {path}")

    obs = np.concatenate(obs_chunks, axis=0)
    actions = np.concatenate(action_chunks, axis=0)
    next_obs = np.concatenate(next_obs_chunks, axis=0)

    if max_transitions is not None and obs.shape[0] > max_transitions:
        indices = rng.choice(obs.shape[0], size=max_transitions, replace=False)
        obs, actions, next_obs = obs[indices], actions[indices], next_obs[indices]
    return obs, actions, next_obs


def to_torch_obs(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(obs, device=device)
    if tensor.dtype == torch.uint8:
        return tensor
    return tensor.to(dtype=torch.get_default_dtype())


@torch.no_grad()
def encode_phi(
    agent,
    encoder,
    obs: np.ndarray | torch.Tensor,
    device: torch.device,
    batch_size: int,
    apply_aug: bool = False,
) -> torch.Tensor:
    if isinstance(obs, np.ndarray):
        obs_tensor = to_torch_obs(obs, device)
    else:
        obs_tensor = obs.to(device)

    outputs = []
    for start in range(0, obs_tensor.shape[0], batch_size):
        batch = obs_tensor[start : start + batch_size]
        if apply_aug:
            batch = agent.aug(batch)
        if getattr(agent, "embeddings", True) and hasattr(encoder, "encode_and_project"):
            z = encoder.encode_and_project(batch)
        else:
            z = encoder(batch)
        outputs.append(z.detach())
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def project_state_action(agent, phi_obs: torch.Tensor, actions: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    action_tensor = torch.as_tensor(actions, device=device).reshape(-1).long()
    psi = agent._encode_state_action(phi_obs, action_tensor)
    return agent.project_sa(psi).detach()


def sample_negative_indices(num_samples: int, negatives_per_positive: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if num_samples < 2:
        raise ValueError("Need at least two transitions to construct negative pairs")
    if negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be at least 1")
    anchors = np.repeat(np.arange(num_samples), negatives_per_positive)
    negatives = rng.integers(0, num_samples - 1, size=anchors.shape[0])
    negatives = negatives + (negatives >= anchors)
    return anchors, negatives


def summarize(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def compute_transition_statistics(
    agent,
    encoder,
    obs: np.ndarray,
    actions: np.ndarray,
    next_obs: np.ndarray,
    device: torch.device,
    batch_size: int,
    negatives_per_positive: int,
    apply_aug: bool,
    rng: np.random.Generator,
) -> Dict[str, Dict[str, float]]:
    phi_obs = encode_phi(agent, encoder, obs, device, batch_size, apply_aug=apply_aug)
    phi_next = encode_phi(agent, encoder, next_obs, device, batch_size, apply_aug=apply_aug)
    projected_sa = project_state_action(agent, phi_obs, actions, device)

    anchors_np, negatives_np = sample_negative_indices(phi_next.shape[0], negatives_per_positive, rng)
    anchors = torch.as_tensor(anchors_np, device=device)
    negatives = torch.as_tensor(negatives_np, device=device)

    pos_l2_raw = torch.linalg.norm(projected_sa - phi_next, dim=1)
    neg_l2_raw = torch.linalg.norm(projected_sa[anchors] - phi_next[negatives], dim=1)

    projected_norm = F.normalize(projected_sa, p=2, dim=1, eps=1e-10)
    phi_next_norm = F.normalize(phi_next, p=2, dim=1, eps=1e-10)
    pos_l2_norm = torch.linalg.norm(projected_norm - phi_next_norm, dim=1)
    neg_l2_norm = torch.linalg.norm(projected_norm[anchors] - phi_next_norm[negatives], dim=1)

    pos_cos = F.cosine_similarity(projected_sa, phi_next, dim=1, eps=1e-10)
    neg_cos = F.cosine_similarity(projected_sa[anchors], phi_next[negatives], dim=1, eps=1e-10)

    result = {
        "num_transitions": {"value": int(phi_next.shape[0])},
        "negatives_per_positive": {"value": int(negatives_per_positive)},
        "l2_without_extra_l2_normalization_positive": summarize(pos_l2_raw.cpu().numpy()),
        "l2_without_extra_l2_normalization_negative": summarize(neg_l2_raw.cpu().numpy()),
        "l2_with_extra_l2_normalization_positive": summarize(pos_l2_norm.cpu().numpy()),
        "l2_with_extra_l2_normalization_negative": summarize(neg_l2_norm.cpu().numpy()),
        "cosine_without_pre_l2_normalization_positive": summarize(pos_cos.cpu().numpy()),
        "cosine_without_pre_l2_normalization_negative": summarize(neg_cos.cpu().numpy()),
    }
    result["separation"] = {
        "l2_raw_neg_minus_pos_mean": (
            result["l2_without_extra_l2_normalization_negative"]["mean"]
            - result["l2_without_extra_l2_normalization_positive"]["mean"]
        ),
        "l2_norm_neg_minus_pos_mean": (
            result["l2_with_extra_l2_normalization_negative"]["mean"]
            - result["l2_with_extra_l2_normalization_positive"]["mean"]
        ),
        "cosine_pos_minus_neg_mean": (
            result["cosine_without_pre_l2_normalization_positive"]["mean"]
            - result["cosine_without_pre_l2_normalization_negative"]["mean"]
        ),
    }
    return result


def base_point_env(env):
    base = env.unwrapped
    point_env = getattr(base, "point_env", None)
    if point_env is None:
        raise AttributeError("Could not find PointMaze point_env through env.unwrapped")
    return base, point_env


def pointmaze_dt(env) -> float:
    _, point_env = base_point_env(env)
    dt = getattr(point_env, "dt", None)
    if dt is not None:
        return float(dt)
    model = getattr(point_env, "model", None)
    frame_skip = float(getattr(point_env, "frame_skip", 1))
    if model is not None and hasattr(model, "opt"):
        return float(model.opt.timestep) * frame_skip
    return 0.05


def set_point_state(env, xy: Tuple[float, float], velocity: Tuple[float, float] = (0.0, 0.0)) -> None:
    base, point_env = base_point_env(env)
    qpos = point_env.data.qpos.copy()
    qvel = np.zeros_like(point_env.data.qvel)
    qpos[:2] = np.asarray(xy, dtype=np.float64)
    qvel[:2] = np.asarray(velocity, dtype=np.float64)
    point_env.set_state(qpos, qvel)
    if hasattr(base, "update_target_site_pos"):
        base.update_target_site_pos()


def render_chw(env) -> np.ndarray:
    render_fn = getattr(env, "render_observation", None)
    frame = render_fn() if callable(render_fn) else env.render()
    if frame.ndim == 2:
        frame = frame[..., None]
    return frame.transpose(2, 0, 1).astype(np.uint8, copy=True)


def infer_frame_stack(agent, cfg) -> int:
    if hasattr(cfg, "frame_stack"):
        return int(cfg.frame_stack)
    obs_shape = getattr(agent, "obs_shape", None)
    grayscale = bool(getattr(agent, "grayscale", False))
    channels = 1 if grayscale else 3
    if obs_shape is not None and len(obs_shape) == 3:
        return int(obs_shape[0] // channels)
    return 1


def stacked_observation_from_history(
    env,
    xy: Tuple[float, float],
    velocity: Tuple[float, float],
    frame_stack: int,
    history_dt: float,
) -> np.ndarray:
    frames = rendered_history_frames(env, xy, velocity, frame_stack, history_dt)
    return np.concatenate(frames, axis=0)


def rendered_history_frames(
    env,
    xy: Tuple[float, float],
    velocity: Tuple[float, float],
    frame_stack: int,
    history_dt: float,
) -> List[np.ndarray]:
    xy_arr = np.asarray(xy, dtype=np.float64)
    velocity_arr = np.asarray(velocity, dtype=np.float64)
    frames = []
    for frame_idx in range(frame_stack):
        steps_back = frame_stack - 1 - frame_idx
        frame_xy = xy_arr - steps_back * history_dt * velocity_arr
        set_point_state(env, (float(frame_xy[0]), float(frame_xy[1])), velocity)
        frames.append(render_chw(env))
    return frames


def chw_to_hwc(frame: np.ndarray) -> np.ndarray:
    if frame.shape[0] == 1:
        return np.repeat(frame.transpose(1, 2, 0), 3, axis=2)
    return frame.transpose(1, 2, 0)


def float_token(value: float) -> str:
    return f"{value:.3f}".replace("-", "m").replace(".", "p")


def save_sampled_stack_frames(
    frames: Sequence[np.ndarray],
    root_dir: Optional[Path],
    position_group: int,
    group_xy: Tuple[float, float],
    real_xy: Tuple[float, float],
    sample_idx: int,
    velocity: Optional[Tuple[float, float]] = None,
) -> None:
    if root_dir is None:
        return

    folder_parts = [
        f"position_group_{position_group:02d}",
        f"group_xy_x{float_token(group_xy[0])}_y{float_token(group_xy[1])}",
        f"real_xy_x{float_token(real_xy[0])}_y{float_token(real_xy[1])}",
        f"sample_{sample_idx:03d}",
    ]
    if velocity is not None:
        folder_parts.insert(
            3,
            f"velocity_vx{float_token(velocity[0])}_vy{float_token(velocity[1])}",
        )

    sample_dir = root_dir / "__".join(folder_parts)
    sample_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "position_group": int(position_group),
        "group_xy": [float(group_xy[0]), float(group_xy[1])],
        "real_xy": [float(real_xy[0]), float(real_xy[1])],
        "sample_idx": int(sample_idx),
        "num_frames": int(len(frames)),
    }
    if velocity is not None:
        metadata["velocity"] = [float(velocity[0]), float(velocity[1])]

    with (sample_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    for frame_idx, frame in enumerate(frames):
        image = Image.fromarray(chw_to_hwc(frame).astype(np.uint8))
        image.save(sample_dir / f"frame_{frame_idx:02d}.png")


def save_velocity_state_gif(
    env,
    locations: Sequence[Tuple[float, float]],
    velocities: Sequence[Tuple[float, float]],
    frame_stack: int,
    history_dt: float,
    output_path: Path,
    duration_ms: int = 700,
) -> None:
    histories = {}
    for loc_idx, xy in enumerate(locations):
        for vel_idx, velocity in enumerate(velocities):
            histories[(loc_idx, vel_idx)] = rendered_history_frames(env, xy, velocity, frame_stack, history_dt)

    sample_frame = next(iter(histories.values()))[0]
    cell_hwc = chw_to_hwc(sample_frame)
    img_h, img_w = cell_hwc.shape[:2]
    label_h = 34
    title_h = 28
    pad = 8
    rows = len(locations)
    cols = len(velocities)
    canvas_w = cols * img_w + (cols + 1) * pad
    canvas_h = title_h + rows * (img_h + label_h) + (rows + 1) * pad

    gif_frames = []
    for frame_idx in range(frame_stack):
        canvas = Image.new("RGB", (canvas_w, canvas_h), color=(245, 245, 245))
        draw = ImageDraw.Draw(canvas)
        draw.text((pad, 6), f"Velocity frame-stack history: frame {frame_idx + 1}/{frame_stack}", fill=(20, 20, 20))

        for row, xy in enumerate(locations):
            for col, velocity in enumerate(velocities):
                x0 = pad + col * (img_w + pad)
                y0 = title_h + pad + row * (img_h + label_h + pad)
                frame = chw_to_hwc(histories[(row, col)][frame_idx]).astype(np.uint8)
                canvas.paste(Image.fromarray(frame), (x0, y0))
                label = f"xy=({xy[0]:.1f},{xy[1]:.1f})  v=({velocity[0]:.1f},{velocity[1]:.1f})"
                draw.text((x0, y0 + img_h + 3), label, fill=(20, 20, 20))

        gif_frames.append(canvas)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration_ms,
        loop=0,
    )


def tsne_2d(embeddings: np.ndarray, requested_perplexity: float, seed: int) -> np.ndarray:
    n = embeddings.shape[0]
    if n < 2:
        raise ValueError("Need at least two points for t-SNE")
    perplexity = min(float(requested_perplexity), max(1.0, (n - 1) / 3.0))
    init = "pca" if n > 3 and embeddings.shape[1] > 2 else "random"
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init=init,
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(embeddings)


def maze_layout(env):
    layout_fn = getattr(env, "get_debug_maze_layout", None)
    if callable(layout_fn):
        return layout_fn()
    return None


def draw_maze(ax, layout) -> None:
    if layout is None:
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        return
    for x0, y0, width, height in layout["wall_rectangles"]:
        ax.add_patch(Rectangle((x0, y0), width, height, facecolor="0.2", edgecolor="0.2", alpha=0.85))
    lower = layout["maze_lower"]
    upper = layout["maze_upper"]
    ax.set_xlim(float(lower[0]) - 0.2, float(upper[0]) + 0.2)
    ax.set_ylim(float(lower[1]) - 0.2, float(upper[1]) + 0.2)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)


def color_cycle(n: int):
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    return [cmap(i % cmap.N) for i in range(n)]


def plot_spatial_tsne(
    agent,
    encoder,
    env,
    locations: Sequence[Tuple[float, float]],
    samples_per_location: int,
    jitter: float,
    frame_stack: int,
    history_dt: float,
    device: torch.device,
    batch_size: int,
    apply_aug: bool,
    perplexity: float,
    seed: int,
    output_path: Path,
    frames_output_dir: Optional[Path],
    rng: np.random.Generator,
) -> None:
    observations, labels, xy_points = [], [], []
    for loc_idx, (x, y) in enumerate(locations):
        for sample_idx in range(samples_per_location):
            dx, dy = rng.normal(0.0, jitter, size=2) if jitter > 0 else (0.0, 0.0)
            xy = (float(x + dx), float(y + dy))
            frames = rendered_history_frames(env, xy, (0.0, 0.0), frame_stack, history_dt)
            save_sampled_stack_frames(
                frames,
                frames_output_dir,
                position_group=loc_idx,
                group_xy=(x, y),
                real_xy=xy,
                sample_idx=sample_idx,
                velocity=(0.0, 0.0),
            )
            observations.append(np.concatenate(frames, axis=0))
            labels.append(loc_idx)
            xy_points.append(xy)

    embeddings = encode_phi(agent, encoder, np.stack(observations), device, batch_size, apply_aug=apply_aug)
    coords = tsne_2d(embeddings.cpu().numpy(), perplexity, seed)
    colors = color_cycle(len(locations))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    ax_tsne, ax_map = axes

    for loc_idx, loc in enumerate(locations):
        mask = np.asarray(labels) == loc_idx
        label = f"({loc[0]:.2f}, {loc[1]:.2f})"
        ax_tsne.scatter(coords[mask, 0], coords[mask, 1], s=42, color=colors[loc_idx], label=label, alpha=0.85)
        ax_map.scatter(
            [xy_points[i][0] for i in np.where(mask)[0]],
            [xy_points[i][1] for i in np.where(mask)[0]],
            s=38,
            color=colors[loc_idx],
            alpha=0.85,
        )
        ax_map.text(loc[0], loc[1] + 0.08, label, color=colors[loc_idx], fontsize=8, ha="center")

    ax_tsne.set_title("t-SNE of phi(s)")
    ax_tsne.set_xlabel("t-SNE 1")
    ax_tsne.set_ylabel("t-SNE 2")
    ax_tsne.legend(fontsize=8, loc="best")

    draw_maze(ax_map, maze_layout(env))
    ax_map.set_title("Sampled U-Maze xy locations")
    ax_map.set_xlabel("x")
    ax_map.set_ylabel("y")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_velocity_tsne(
    agent,
    encoder,
    env,
    locations: Sequence[Tuple[float, float]],
    velocities: Sequence[Tuple[float, float]],
    samples_per_setting: int,
    position_jitter: float,
    frame_stack: int,
    history_dt: float,
    device: torch.device,
    batch_size: int,
    apply_aug: bool,
    perplexity: float,
    seed: int,
    output_path: Path,
    frames_output_dir: Optional[Path],
    rng: np.random.Generator,
) -> None:
    observations, velocity_labels, xy_labels = [], [], []
    for loc_idx, xy in enumerate(locations):
        for vel_idx, velocity in enumerate(velocities):
            for sample_idx in range(samples_per_setting):
                jitter = rng.normal(0.0, position_jitter, size=2) if position_jitter > 0 else np.zeros(2)
                sample_xy = (float(xy[0] + jitter[0]), float(xy[1] + jitter[1]))
                frames = rendered_history_frames(env, sample_xy, velocity, frame_stack, history_dt)
                save_sampled_stack_frames(
                    frames,
                    frames_output_dir,
                    position_group=loc_idx,
                    group_xy=xy,
                    real_xy=sample_xy,
                    sample_idx=sample_idx,
                    velocity=velocity,
                )
                observations.append(np.concatenate(frames, axis=0))
                velocity_labels.append(vel_idx)
                xy_labels.append(xy)

    embeddings = encode_phi(agent, encoder, np.stack(observations), device, batch_size, apply_aug=apply_aug)
    coords = tsne_2d(embeddings.cpu().numpy(), perplexity, seed)
    colors = color_cycle(len(velocities))

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    velocity_labels_np = np.asarray(velocity_labels)
    for vel_idx, velocity in enumerate(velocities):
        mask = velocity_labels_np == vel_idx
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=46,
            color=colors[vel_idx],
            alpha=0.85,
            label=f"vx={velocity[0]:.2f}, vy={velocity[1]:.2f}",
        )

    for (x, y), (tx, ty) in zip(xy_labels, coords):
        ax.text(tx, ty + 0.35, f"({x:.1f},{y:.1f})", fontsize=7, ha="center", alpha=0.8)

    ax.set_title("t-SNE of phi(s) from frame stacks with controlled velocities")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(title="Velocity", fontsize=8, title_fontsize=9, loc="best")
    ax.grid(True, alpha=0.2)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    agent = load_snapshot_agent(args.snapshot, device)
    encoder = select_encoder(agent, args.encoder_source)

    obs, actions, next_obs = load_transitions(args.episode_path, args.nstep, args.max_transitions, rng)
    print(f"Loaded {obs.shape[0]} transitions from {args.episode_path}")

    stats = compute_transition_statistics(
        agent=agent,
        encoder=encoder,
        obs=obs,
        actions=actions,
        next_obs=next_obs,
        device=device,
        batch_size=args.batch_size,
        negatives_per_positive=args.negatives_per_positive,
        apply_aug=args.apply_aug,
        rng=rng,
    )
    stats_path = args.output_dir / "transition_pair_statistics.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved transition statistics to {stats_path}")
    print(json.dumps(stats["separation"], indent=2))

    cfg = compose_cfg(args.config_name, args.seed)
    env = make_pointmaze_env(cfg)
    env.reset()

    frame_stack = infer_frame_stack(agent, cfg)
    history_dt = args.history_dt if args.history_dt is not None else pointmaze_dt(env)
    history_dt = float(history_dt) * float(args.history_dt_multiplier)
    print(
        "Velocity-stack construction uses "
        f"frame_stack={frame_stack}, history_dt={history_dt:.6f}. "
        "Each stack is rendered from previous positions implied by vx,vy."
    )

    spatial_locations = parse_xy_pairs(args.spatial_locations, DEFAULT_SPATIAL_LOCATIONS)
    velocity_locations = parse_xy_pairs(args.velocity_locations, DEFAULT_VELOCITY_LOCATIONS)
    velocities = parse_xy_pairs(args.velocities, DEFAULT_VELOCITIES)

    spatial_path = args.output_dir / "spatial_tsne_with_umaze_xy.png"
    spatial_frames_dir = args.output_dir / "spatial_sampled_state_frames" if args.save_sampled_frames else None
    plot_spatial_tsne(
        agent=agent,
        encoder=encoder,
        env=env,
        locations=spatial_locations,
        samples_per_location=args.spatial_samples_per_location,
        jitter=args.spatial_jitter,
        frame_stack=frame_stack,
        history_dt=history_dt,
        device=device,
        batch_size=args.batch_size,
        apply_aug=args.apply_aug,
        perplexity=args.tsne_perplexity,
        seed=args.seed,
        output_path=spatial_path,
        frames_output_dir=spatial_frames_dir,
        rng=rng,
    )
    print(f"Saved spatial t-SNE plot to {spatial_path}")
    if spatial_frames_dir is not None:
        print(f"Saved spatial sampled frames to {spatial_frames_dir}")

    velocity_path = args.output_dir / "velocity_tsne.png"
    velocity_frames_dir = args.output_dir / "velocity_sampled_state_frames" if args.save_sampled_frames else None
    plot_velocity_tsne(
        agent=agent,
        encoder=encoder,
        env=env,
        locations=velocity_locations,
        velocities=velocities,
        samples_per_setting=args.velocity_samples_per_setting,
        position_jitter=args.velocity_position_jitter,
        frame_stack=frame_stack,
        history_dt=history_dt,
        device=device,
        batch_size=args.batch_size,
        apply_aug=args.apply_aug,
        perplexity=args.tsne_perplexity,
        seed=args.seed,
        output_path=velocity_path,
        frames_output_dir=velocity_frames_dir,
        rng=rng,
    )
    print(f"Saved velocity t-SNE plot to {velocity_path}")
    if velocity_frames_dir is not None:
        print(f"Saved velocity sampled frames to {velocity_frames_dir}")

    if args.save_velocity_gif:
        velocity_gif_path = args.output_dir / "velocity_sampled_state_stacks.gif"
        save_velocity_state_gif(
            env=env,
            locations=velocity_locations,
            velocities=velocities,
            frame_stack=frame_stack,
            history_dt=history_dt,
            output_path=velocity_gif_path,
            duration_ms=args.velocity_gif_duration_ms,
        )
        print(f"Saved velocity sampled-state GIF to {velocity_gif_path}")

    close = getattr(env, "close", None)
    if callable(close):
        close()


if __name__ == "__main__":
    main()
