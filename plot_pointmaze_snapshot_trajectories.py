"""
 python plot_pointmaze_snapshot_trajectories.py   \
    --snapshot models/pointmaze/largedense/states/rover/models/states/gym/dist_matching/1/snapshot.pt  \
    --num-trajectories 10 \
    --episode-steps 500 \
    --start-position-variance 0.1
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

import gym_env
import utils
from agent.rover_visualization.domains import (
    extract_eval_trajectory_point,
    save_maze_trajectory_overlay_plot,
)


def find_run_config(snapshot_path: Path) -> Path | None:
    for parent in [snapshot_path.parent, *snapshot_path.parents]:
        candidate = parent / ".hydra" / "config.yaml"
        if candidate.exists():
            return candidate
    return None


def load_config(config_path: Path):
    return OmegaConf.load(config_path)


def make_env(cfg, seed: int, start_position_variance: float | None = None):
    env_kwargs = OmegaConf.to_container(cfg.env, resolve=True)
    env_kwargs.pop("name", None)
    if start_position_variance is not None:
        pointmaze_kwargs = dict(env_kwargs.get("pointmaze", {}))
        pointmaze_kwargs["start_position_variance"] = float(start_position_variance)
        env_kwargs["pointmaze"] = pointmaze_kwargs
    return gym_env.make(
        cfg.task_name,
        cfg.obs_type,
        frame_stack=int(cfg.frame_stack),
        action_repeat=int(cfg.action_repeat),
        seed=seed,
        resolution=int(cfg.resolution),
        grayscale=bool(getattr(cfg, "grayscale", False)),
        url=True,
        **env_kwargs,
    )


def load_snapshot(snapshot_path: Path, device: torch.device):
    payload = torch.load(snapshot_path, map_location=device, weights_only=False)
    agent = payload["agent"] if isinstance(payload, dict) and "agent" in payload else payload
    agent.device = str(device)
    compute_dtype = getattr(agent, "compute_dtype", None)
    if isinstance(compute_dtype, torch.dtype):
        torch.set_default_dtype(compute_dtype)

    for name, value in list(vars(agent).items()):
        if isinstance(value, torch.nn.Module):
            value.to(device)
        elif isinstance(value, torch.Tensor):
            setattr(agent, name, value.to(device))

    train = getattr(agent, "train", None)
    if callable(train):
        train(False)
    return agent, payload


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


def format_frame(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = frame[..., None]
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    return frame[..., :3].astype(np.uint8)


def render_goal_hidden_frame(env) -> np.ndarray:
    render_observation = get_env_method(env, "render_observation")
    if callable(render_observation):
        return format_frame(render_observation())
    return format_frame(env.render())


def save_gif(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        raise ValueError("Cannot save GIF without frames.")

    duration_ms = max(int(round(1000.0 / float(fps))), 1)
    images = [Image.fromarray(format_frame(frame)) for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


def default_episode_steps(env) -> int:
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        max_steps = getattr(current, "_max_episode_steps", None)
        if max_steps is not None:
            return int(max_steps)
        current = getattr(current, "env", None)

    spec = getattr(getattr(env, "unwrapped", env), "spec", None)
    max_steps = getattr(spec, "max_episode_steps", None)
    if max_steps is not None:
        return int(max_steps)
    return 300


def snapshot_step(snapshot_path: Path, payload) -> int:
    match = re.search(r"snapshot_(\d+)\.pt$", snapshot_path.name)
    if match:
        return int(match.group(1))
    if isinstance(payload, dict) and "_global_step" in payload:
        return int(payload["_global_step"])
    return 0


def sample_trajectories(
    agent,
    env,
    *,
    num_trajectories: int,
    episode_steps: int,
    policy_step: int,
    seed: int,
    deterministic: bool,
):
    trajectories = []
    frames = []

    for episode in range(num_trajectories):
        episode_seed = seed + episode
        time_step = env.reset(seed=episode_seed)
        meta = agent.init_meta() if callable(getattr(agent, "init_meta", None)) else {}
        trajectory = []

        point = extract_eval_trajectory_point(env, time_step)
        if point is not None:
            trajectory.append(point)
        frames.append(render_goal_hidden_frame(env))

        for step in range(episode_steps):
            with torch.no_grad(), utils.eval_mode(agent):
                action = agent.act(
                    time_step.observation,
                    meta,
                    policy_step,
                    eval_mode=deterministic,
                )
            time_step = env.step(action)

            update_meta = getattr(agent, "update_meta", None)
            if callable(update_meta):
                meta = update_meta(meta, policy_step, time_step)

            point = extract_eval_trajectory_point(env, time_step)
            if point is not None:
                trajectory.append(point)
            frames.append(render_goal_hidden_frame(env))

            if time_step.last():
                break

        if trajectory:
            trajectories.append(np.asarray(trajectory, dtype=np.float32))

    return trajectories, frames


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a PointMaze policy snapshot, sample trajectories, save maze overlay plot and goal-hidden GIF."
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--num-trajectories", type=int, default=10)
    parser.add_argument("--episode-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--gif-fps", type=int, default=20)
    parser.add_argument(
        "--start-position-variance",
        type=float,
        default=None,
        help="Override env.pointmaze.start_position_variance from the loaded config.",
    )
    parser.add_argument(
        "--policy-step",
        type=int,
        default=None,
        help="Step value passed to agent.act. Defaults to the snapshot step parsed from filename/payload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot_path = args.snapshot.expanduser().resolve()
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    config_path = args.config.expanduser().resolve() if args.config else find_run_config(snapshot_path)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(
            "Could not find .hydra/config.yaml above snapshot. Pass --config explicitly."
        )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = snapshot_path.parent / "trajectory_samples"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    cfg = load_config(config_path)
    env = make_env(cfg, seed=args.seed, start_position_variance=args.start_position_variance)
    agent, payload = load_snapshot(snapshot_path, device)
    step = snapshot_step(snapshot_path, payload)
    policy_step = int(args.policy_step) if args.policy_step is not None else step
    episode_steps = int(args.episode_steps) if args.episode_steps is not None else default_episode_steps(env)

    insert_env = getattr(agent, "insert_env", None)
    if callable(insert_env):
        try:
            insert_env(env)
        except Exception as exc:
            utils.ColorPrint.yellow(f"Could not attach env to agent visualizer: {exc}")

    try:
        trajectories, frames = sample_trajectories(
            agent,
            env,
            num_trajectories=int(args.num_trajectories),
            episode_steps=episode_steps,
            policy_step=policy_step,
            seed=int(args.seed),
            deterministic=bool(args.deterministic),
        )
        if not trajectories:
            raise RuntimeError("No trajectory XY points collected.")

        plot_paths = save_maze_trajectory_overlay_plot(
            trajectories=trajectories,
            env=env,
            step=step,
            save_dir=output_dir,
        )

        gif_path = output_dir / f"{snapshot_path.stem}_ntraj_{len(trajectories)}_rollouts.gif"
        save_gif(gif_path, frames, fps=int(args.gif_fps))
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    print(f"Config: {config_path}")
    if args.start_position_variance is not None:
        print(f"Start position variance override: {float(args.start_position_variance)}")
    print(f"Policy step: {policy_step}")
    print(f"Deterministic: {bool(args.deterministic)}")
    print(f"Sampled {len(trajectories)} trajectories, {len(frames)} GIF frames.")
    for style, path in plot_paths.items():
        print(f"Saved {style} plot: {Path(path).resolve()}")
    print(f"Saved goal-hidden GIF: {gif_path}")


if __name__ == "__main__":
    main()
