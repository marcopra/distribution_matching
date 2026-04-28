from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

import gym_env

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = REPO_ROOT / "configs"


def load_env_kwargs(config_path: Path) -> tuple[str, dict]:
    cfg = OmegaConf.load(config_path)
    env_cfg = cfg.env if "env" in cfg else cfg
    env_kwargs = OmegaConf.to_container(env_cfg, resolve=True)
    task_name = env_kwargs.pop("name")
    return task_name, env_kwargs


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


def compose_pretrain_cfg(config_name: str | Path):
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
    return OmegaConf.merge(env_cfg, cfg)


def make_pretrain_pointmaze_env(cfg):
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


def make_pointmaze_env(
    config_path: Path,
    *,
    seed: int,
    resolution: int,
    direct_velocity_actions: bool,
    discrete_actions: bool,
    max_velocity: float | None,
):
    task_name, env_kwargs = load_env_kwargs(config_path)
    pointmaze_kwargs = dict(env_kwargs.get("pointmaze", {}))
    pointmaze_kwargs["direct_velocity_actions"] = direct_velocity_actions
    pointmaze_kwargs["discrete_actions"] = discrete_actions
    if max_velocity is not None:
        pointmaze_kwargs["max_velocity"] = max_velocity
    env_kwargs["pointmaze"] = pointmaze_kwargs

    return gym_env.make(
        task_name,
        obs_type="proprio",
        frame_stack=1,
        action_repeat=1,
        seed=seed,
        resolution=resolution,
        grayscale=False,
        url=False,
        **env_kwargs,
    )


def point_env(env):
    base_env = env.unwrapped
    point = getattr(base_env, "point_env", None)
    if point is None:
        raise AttributeError("Could not find PointMaze point_env")
    return point


def point_state(env) -> tuple[np.ndarray, np.ndarray]:
    point = point_env(env)
    return point.data.qpos.copy(), point.data.qvel.copy()


def render_frame(env) -> np.ndarray:
    frame = np.asarray(env.render())
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    return frame.astype(np.uint8)


def resize_frame(frame: np.ndarray, size: int) -> np.ndarray:
    img = Image.fromarray(frame.astype(np.uint8))
    return np.asarray(img.resize((size, size), Image.NEAREST), dtype=np.uint8)


def stacked_observation_frame(time_step, channels_per_frame: int, render_size: int) -> np.ndarray:
    obs = np.asarray(time_step.observation)
    if obs.ndim != 3:
        raise ValueError(f"Expected stacked pixel observation with shape (C, H, W), got {obs.shape}")
    if obs.shape[0] % channels_per_frame != 0:
        raise ValueError(
            f"Stacked observation channels {obs.shape[0]} are not divisible by {channels_per_frame}"
        )

    frame_count = obs.shape[0] // channels_per_frame
    frames = []
    separator = np.full((render_size, 2, 3), 255, dtype=np.uint8)
    for frame_idx in range(frame_count):
        start = frame_idx * channels_per_frame
        frame = obs[start : start + channels_per_frame]
        frame = np.transpose(frame, (1, 2, 0))
        if channels_per_frame == 1:
            frame = np.repeat(frame, 3, axis=-1)
        frames.append(resize_frame(frame, render_size))
        if frame_idx < frame_count - 1:
            frames.append(separator)
    return np.concatenate(frames, axis=1).astype(np.uint8)


def save_direct_velocity_video(env, output_path: Path, seed: int, fps: int) -> None:
    env.reset(seed=seed)
    actions = (
        [np.array([1.0, 0.0], dtype=np.float32)] * 100
        # + [np.array([0.0, 5.0], dtype=np.float32)] * 50
        # + [np.array([-1.0, 0.0], dtype=np.float32)] * 50
        # + [np.array([0.0, -1.0], dtype=np.float32)] * 50
    )

    frames = [render_frame(env)]
    for action in actions:
        env.step(action)
        frames.append(render_frame(env))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps, macro_block_size=1)


def save_pretrain_discrete_rollout_videos(
    cfg,
    render_output_path: Path,
    stacked_output_path: Path,
    fps: int,
) -> None:
    env = make_pretrain_pointmaze_env(cfg)
    try:
        if not hasattr(env.action_space, "n"):
            raise ValueError(f"Expected discrete action space from pretrain config, got {env.action_space}")

        action_sequence = [0] * 35 + [2] * 35 + [1] * 35 + [3] * 35
        channels_per_frame = 1 if bool(getattr(cfg, "grayscale", False)) else 3
        render_frames = []
        stacked_frames = []

        time_step = env.reset(seed=cfg.seed)
        render_frames.append(render_frame(env))
        stacked_frames.append(stacked_observation_frame(time_step, channels_per_frame, int(cfg.resolution)))

        for action in action_sequence:
            time_step = env.step(action)
            render_frames.append(render_frame(env))
            stacked_frames.append(stacked_observation_frame(time_step, channels_per_frame, int(cfg.resolution)))

        render_output_path.parent.mkdir(parents=True, exist_ok=True)
        stacked_output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(render_output_path, render_frames, fps=fps, macro_block_size=1)
        imageio.mimsave(stacked_output_path, stacked_frames, fps=fps, macro_block_size=1)

        final_qpos, final_qvel = point_state(env)
        print("\nPretrain config discrete rollout")
        print(f"config task={cfg.task_name}, obs_type={cfg.obs_type}, frame_stack={cfg.frame_stack}")
        print(f"action_space={env.action_space}, actions=0x35, 2x35, 1x35, 3x35")
        print(f"final qpos={final_qpos}, qvel={final_qvel}")
        print(f"Saved pretrain render video to {render_output_path.resolve()}")
        print(f"Saved pretrain stacked-observation video to {stacked_output_path.resolve()}")
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def predict_normal_pointmaze_velocity(env, velocity: np.ndarray, action: np.ndarray) -> np.ndarray:
    point = point_env(env)
    velocity = np.clip(np.asarray(velocity, dtype=np.float64), -5.0, 5.0)
    action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

    h = float(point.model.opt.timestep)
    mass = float(point.model.body_mass[1])
    damping = point.model.dof_damping[:2].astype(np.float64)
    gear = point.model.actuator_gear[:2, 0].astype(np.float64)
    return (mass * velocity + h * gear * action) / (mass + h * damping)


def run_normal_velocity_test(env, seed: int) -> None:
    env.reset(seed=seed)
    point = point_env(env)
    qpos, _ = point_state(env)
    zero_velocity = np.zeros(2, dtype=np.float64)
    point.set_state(qpos, zero_velocity)

    action = np.array([1.0, 0.0], dtype=np.float32)

    predicted_1 = predict_normal_pointmaze_velocity(env, zero_velocity, action)
    env.step(action)
    _, measured_1 = point_state(env)

    predicted_2 = predict_normal_pointmaze_velocity(env, measured_1, action)
    env.step(action)
    _, measured_2 = point_state(env)

    h = float(point.model.opt.timestep)
    mass = float(point.model.body_mass[1])
    damping = point.model.dof_damping[:2]
    gear = point.model.actuator_gear[:2, 0]

    print("\nNormal PointMaze force-control velocity test")
    print(f"dt={h:.6f}, mass={mass:.9f}, damping={damping}, gear={gear}")
    print("MuJoCo/Gymnasium Robotics clips v to [-5, 5] and action u to [-1, 1].")
    print("For this unconstrained point body, one Euler step is:")
    print("  v_next = (mass * clip(v, -5, 5) + dt * gear * clip(u, -1, 1)) / (mass + dt * damping)")
    print("  q_next = q + dt * v_next")
    print(f"from rest, action [1, 0]: predicted v={predicted_1}, measured v={measured_1}")
    print(f"second action [1, 0]:    predicted v={predicted_2}, measured v={measured_2}")


def parse_args():
    parser = argparse.ArgumentParser(description="Test PointMaze direct velocity actions and normal force dynamics.")
    parser.add_argument(
        "--env-config",
        type=Path,
        default=Path("configs/env/pointmaze/pointmaze_umaze_goal_1.yaml"),
    )
    parser.add_argument("--output", type=Path, default=Path("pointmaze_direct_velocity_test.mp4"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--pretrain-config",
        type=str,
        default="pretrain/pretrain_pointmaze_umaze_1",
        help="Hydra-style pretrain config used for the discrete frame-stack rollout.",
    )
    parser.add_argument(
        "--pretrain-render-output",
        type=Path,
        default=Path("pointmaze_pretrain_discrete_render.mp4"),
    )
    parser.add_argument(
        "--pretrain-stacked-output",
        type=Path,
        default=Path("pointmaze_pretrain_discrete_framestack.mp4"),
    )
    parser.add_argument(
        "--max-velocity",
        type=float,
        default=None,
        help="Override env.pointmaze.max_velocity. Defaults to the YAML value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    direct_env = make_pointmaze_env(
        args.env_config,
        seed=args.seed,
        resolution=args.resolution,
        direct_velocity_actions=True,
        discrete_actions=False,
        max_velocity=args.max_velocity,
    )
    normal_env = make_pointmaze_env(
        args.env_config,
        seed=args.seed,
        resolution=args.resolution,
        direct_velocity_actions=False,
        discrete_actions=False,
        max_velocity=args.max_velocity,
    )

    try:
        save_direct_velocity_video(direct_env, args.output, args.seed, args.fps)
        print(f"Saved direct-velocity video to {args.output.resolve()}")
        run_normal_velocity_test(normal_env, args.seed)

        pretrain_cfg = compose_pretrain_cfg(args.pretrain_config)
        save_pretrain_discrete_rollout_videos(
            pretrain_cfg,
            args.pretrain_render_output,
            args.pretrain_stacked_output,
            args.fps,
        )
    finally:
        for env in (direct_env, normal_env):
            close = getattr(env, "close", None)
            if callable(close):
                close()


if __name__ == "__main__":
    main()
