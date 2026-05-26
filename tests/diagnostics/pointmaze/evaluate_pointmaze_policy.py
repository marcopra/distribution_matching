"""Evaluate and visualize a saved policy on controlled PointMaze locations.

Requires a trained policy snapshot; writes policy plots, rollout videos, and
summaries to tests/outputs/pointmaze/policy_eval by default.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

import gym_env
import utils

# Keep these locations explicit so the diagnostic is easy to edit.
# They are chosen near the navigable borders of PointMaze_UMaze-v3.
BORDER_LOCATIONS = [
    {"name": "reference", "xy": [-1.45, 0.75]},
    {"name": "left_top_border", "xy": [-1.45, 1.0]},
    {"name": "top_corridor_border", "xy": [0.0, 1.45]},
    {"name": "right_top_border", "xy": [1.45, 1.0]},
    {"name": "left_lower_border", "xy": [-1.0, -0.45]},
    {"name": "right_lower_border", "xy": [1.0, -0.45]},
]

ACTION_LABELS_4 = ["+x", "-x", "+y", "-y"]

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"


def resolve_group_path(group_value: str, group_name: str) -> Path:
    direct = CONFIG_DIR / f"{group_value}.yaml"
    if direct.exists():
        return direct
    grouped = CONFIG_DIR / group_name / f"{group_value}.yaml"
    if grouped.exists():
        return grouped
    raise FileNotFoundError(f"Could not resolve {group_name} config '{group_value}'")


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
    raise FileNotFoundError(f"Could not resolve config '{config_name}'")


def compose_pretrain_cfg(config_name: str | Path, seed: int | None = None):
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
    cfg = OmegaConf.merge(env_cfg, cfg)
    if seed is not None:
        cfg.seed = int(seed)
    return cfg


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


def get_env_method(env, method_name: str):
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        method = getattr(current, method_name, None)
        if callable(method):
            return method
        current = getattr(current, "env", None)
    return None


def point_env(env):
    point = getattr(env.unwrapped, "point_env", None)
    if point is None:
        raise AttributeError("Could not find PointMaze point_env")
    return point


def set_point_xy(env, xy, qvel=None):
    point = point_env(env)
    qpos = point.data.qpos.copy()
    qvel_arr = point.data.qvel.copy()
    qpos[:2] = np.asarray(xy, dtype=np.float64)
    qvel_arr[:] = 0.0 if qvel is None else np.asarray(qvel, dtype=np.float64)
    point.set_state(qpos, qvel_arr)


def format_frame(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = frame[..., None]
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    return frame.astype(np.uint8)


def render_frame(env) -> np.ndarray:
    return format_frame(env.render())


def render_policy_input_frame(env) -> np.ndarray:
    render_observation = get_env_method(env, "render_observation")
    if callable(render_observation):
        return format_frame(render_observation())
    return render_frame(env)


def frame_to_chw(frame: np.ndarray, grayscale: bool) -> np.ndarray:
    frame = format_frame(frame)
    if grayscale:
        gray = np.asarray(np.round(frame[..., :3].mean(axis=2)), dtype=np.uint8)
        return gray[None, ...]
    return np.transpose(frame[..., :3], (2, 0, 1))


def stacked_static_observation(env, cfg) -> np.ndarray:
    frame = render_policy_input_frame(env)
    chw = frame_to_chw(frame, bool(getattr(cfg, "grayscale", False)))
    return np.concatenate([chw.copy() for _ in range(int(cfg.frame_stack))], axis=0)


def load_snapshot_agent(snapshot_path: Path, device: torch.device):
    # Import likely agent modules before unpickling snapshots saved by pretrain.py.
    import agent.rover  # noqa: F401
    import agent.rover_nystrom  # noqa: F401

    payload = torch.load(snapshot_path, map_location=device, weights_only=False)
    agent = payload["agent"] if isinstance(payload, dict) and "agent" in payload else payload
    agent.device = str(device)
    compute_dtype = getattr(agent, "compute_dtype", torch.float32)
    if isinstance(compute_dtype, torch.dtype):
        torch.set_default_dtype(compute_dtype)

    for value in vars(agent).values():
        if isinstance(value, torch.nn.Module):
            value.to(device)

    for name, value in list(vars(agent).items()):
        if isinstance(value, torch.Tensor):
            setattr(agent, name, value.to(device))

    train = getattr(agent, "train", None)
    if callable(train):
        train(False)
    return agent


def policy_probs(agent, obs: np.ndarray) -> np.ndarray:
    probs = np.asarray(agent.compute_action_probs(obs), dtype=np.float64).reshape(-1)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        return np.ones_like(probs) / len(probs)
    return probs / total


def sample_policy_episode(agent, env, cfg, steps: int, output_path: Path, seed: int, deterministic: bool) -> None:
    rng = np.random.default_rng(seed)
    time_step = env.reset(seed=seed)
    frames = [render_frame(env)]
    meta = agent.init_meta() if callable(getattr(agent, "init_meta", None)) else {}

    for step in range(steps):
        obs = np.asarray(time_step.observation)
        probs = policy_probs(agent, obs)
        action = int(np.argmax(probs)) if deterministic else int(rng.choice(len(probs), p=probs))
        time_step = env.step(action)
        frames.append(render_frame(env))

        update_meta = getattr(agent, "update_meta", None)
        if callable(update_meta):
            meta = update_meta(meta, step, time_step)
        if time_step.last():
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=20, macro_block_size=1)


def evaluate_border_locations(agent, env, cfg):
    rows = []
    env.reset(seed=int(cfg.seed))
    for item in BORDER_LOCATIONS:
        xy = np.asarray(item["xy"], dtype=np.float64)
        set_point_xy(env, xy)
        obs = stacked_static_observation(env, cfg)
        probs = policy_probs(agent, obs)
        image = render_frame(env)
        rows.append({"name": item["name"], "xy": xy, "probs": probs, "image": image})
    return rows


def action_labels(n_actions: int) -> list[str]:
    if n_actions == 4:
        return ACTION_LABELS_4
    return [str(i) for i in range(n_actions)]


def plot_location_policy_probs(rows, output_path: Path) -> None:
    if not rows:
        raise ValueError("No policy rows to plot")

    labels = action_labels(len(rows[0]["probs"]))
    fig_height = max(3.0, 2.4 * len(rows))
    fig, axes = plt.subplots(len(rows), 2, figsize=(11, fig_height), squeeze=False)
    colors = ["#2563eb", "#dc2626", "#16a34a", "#ca8a04", "#7c3aed", "#0891b2"]

    for row_idx, row in enumerate(rows):
        ax_probs, ax_img = axes[row_idx]
        probs = row["probs"]
        ax_probs.bar(np.arange(len(probs)), probs, color=colors[: len(probs)])
        ax_probs.set_ylim(0.0, 1.0)
        ax_probs.set_xticks(np.arange(len(probs)))
        ax_probs.set_xticklabels(labels)
        ax_probs.set_ylabel("prob")
        ax_probs.grid(True, axis="y", alpha=0.25)
        ax_probs.set_title(f"{row['name']}  xy=({row['xy'][0]:.2f}, {row['xy'][1]:.2f})")
        for idx, prob in enumerate(probs):
            ax_probs.text(idx, prob + 0.025, f"{prob:.2f}", ha="center", va="bottom", fontsize=9)

        ax_img.imshow(row["image"])
        ax_img.set_title("rendered state")
        ax_img.axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def print_policy_table(rows) -> None:
    labels = action_labels(len(rows[0]["probs"])) if rows else []
    print("\nPolicy probabilities near maze borders")
    print("location              xy                  " + "  ".join(f"{label:>8}" for label in labels))
    for row in rows:
        xy_text = f"({row['xy'][0]: .2f}, {row['xy'][1]: .2f})"
        probs_text = "  ".join(f"{prob:8.4f}" for prob in row["probs"])
        print(f"{row['name']:<21} {xy_text:<18} {probs_text}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a Rover pretrain snapshot, sample PointMaze policy video, and plot border policy probabilities."
    )
    parser.add_argument("--snapshot", type=Path, required=True, help="Path to snapshot.pt or snapshot_<step>.pt")
    parser.add_argument(
        "--config-name",
        type=str,
        default="pretrain/pretrain_pointmaze_umaze_1",
        help="Pretrain config used to recreate the PointMaze env.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("tests/outputs/pointmaze/policy_eval"))
    parser.add_argument("--episode-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--deterministic", action="store_true", help="Use argmax actions instead of sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cfg = compose_pretrain_cfg(args.config_name, seed=args.seed)
    env = make_pretrain_env(cfg)
    agent = load_snapshot_agent(args.snapshot, device)

    insert_env = getattr(agent, "insert_env", None)
    if callable(insert_env):
        try:
            insert_env(env)
        except Exception as exc:
            utils.ColorPrint.yellow(f"Could not attach env to agent visualizer: {exc}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode_path = args.output_dir / "policy_episode.mp4"
    plot_path = args.output_dir / "border_policy_probs.png"

    try:
        sample_policy_episode(
            agent,
            env,
            cfg,
            steps=args.episode_steps,
            output_path=episode_path,
            seed=int(cfg.seed),
            deterministic=args.deterministic,
        )
        rows = evaluate_border_locations(agent, env, cfg)
        plot_location_policy_probs(rows, plot_path)
        print_policy_table(rows)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    print(f"\nSaved sampled policy episode to {episode_path.resolve()}")
    print(f"Saved border policy probability plot to {plot_path.resolve()}")


if __name__ == "__main__":
    main()
