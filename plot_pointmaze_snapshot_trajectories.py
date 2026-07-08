"""
 python plot_pointmaze_snapshot_trajectories.py   \
    --snapshot models/pointmaze/largedense/states/rover/models/states/gym/dist_matching/1/snapshot.pt  \
    --num-trajectories 10 \
    --episode-steps 1000 \
    

python plot_pointmaze_snapshot_trajectories.py   \
    --snapshot models/pointmaze/largedense/states/rover/models/states/gym/dist_matching/1/snapshot.pt  \
    --num-trajectories 50 \
    --episode-steps 3000 \
    --start-position-variance 0.01


python plot_pointmaze_snapshot_trajectories.py --config configs/env/pointmaze/pointmaze_largedense_goal_1.yaml --num-trajectories 15  --episode-steps 4000  

python plot_pointmaze_snapshot_trajectories.py --snapshot models/pointmaze/largedense/states/cic/models/states/gym/cic/0/snapshot_1000000.pt  --num-trajectories 15 --episode-steps 4000 
python plot_pointmaze_snapshot_trajectories.py --snapshot models/pointmaze/largedense/states/icm_apt/models/states/gym/icm_apt/0/snapshot_1000000.pt  --num-trajectories 15 --episode-steps 4000 
python plot_pointmaze_snapshot_trajectories.py --snapshot models/pointmaze/largedense/states/rnd/models/states/gym/rnd/0/snapshot_1000000.pt  --num-trajectories 15 --episode-steps 4000 
python plot_pointmaze_snapshot_trajectories.py --snapshot models/pointmaze/largedense/states/smm/models/states/gym/smm/0/snapshot_1000000.pt  --num-trajectories 15 --episode-steps 4000 
python plot_pointmaze_snapshot_trajectories.py --snapshot models/pointmaze/largedense/states/maxent/models/states/gym/maxent/0/snapshot_1000000.pt  --num-trajectories 15 --episode-steps 4000 
python plot_pointmaze_snapshot_trajectories.py --snapshot models/pointmaze/largedense/states/rover/models/states/gym/dist_matching/1/snapshot.pt  --num-trajectories 15 --episode-steps 4000 

python plot_pointmaze_snapshot_trajectories.py --config configs/env/pointmaze/pointmaze_largedense_goal_1.yaml --num-trajectories 15  --episode-steps 5000  

python plot_pointmaze_snapshot_trajectories.py --snapshot models/pointmaze/largedense/states/cic/models/states/gym/cic/0/snapshot_1000000.pt  --num-trajectories 15 --episode-steps 5000 
python plot_pointmaze_snapshot_trajectories.py --snapshot models/pointmaze/largedense/states/icm_apt/models/states/gym/icm_apt/0/snapshot_1000000.pt  --num-trajectories 15 --episode-steps 5000 
python plot_pointmaze_snapshot_trajectories.py --snapshot models/pointmaze/largedense/states/rnd/models/states/gym/rnd/0/snapshot_1000000.pt  --num-trajectories 15 --episode-steps 5000 
python plot_pointmaze_snapshot_trajectories.py --snapshot models/pointmaze/largedense/states/smm/models/states/gym/smm/0/snapshot_1000000.pt  --num-trajectories 15 --episode-steps 5000 
python plot_pointmaze_snapshot_trajectories.py --snapshot models/pointmaze/largedense/states/maxent/models/states/gym/maxent/0/snapshot_1000000.pt  --num-trajectories 15 --episode-steps 5000 
python plot_pointmaze_snapshot_trajectories.py --snapshot models/pointmaze/largedense/states/rover/models/states/gym/dist_matching/1/snapshot.pt  --num-trajectories 15 --episode-steps 5000 
"""
from __future__ import annotations

import argparse
import os
import re
import types
from collections.abc import Mapping
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
    cfg = OmegaConf.load(config_path)
    if "env" in cfg and "task_name" in cfg:
        return cfg

    env_cfg = cfg.env if "env" in cfg else cfg
    return OmegaConf.create(
        {
            "env": OmegaConf.to_container(env_cfg, resolve=True),
            "task_name": env_cfg.name,
            "obs_type": "states",
            "frame_stack": 1,
            "action_repeat": 1,
            "resolution": 84,
            "grayscale": False,
        }
    )


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
    patch_runtime_rover_action_dtype(agent)
    return agent, payload


def patch_runtime_rover_action_dtype(agent) -> None:
    if not all(hasattr(agent, name) for name in ("_kernel", "_phi_all_obs", "_policy_from_H", "_encode_with_module")):
        return

    def compute_action_probs(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            dtype = getattr(self, "compute_dtype", torch.float32)
            obs_tensor = torch.as_tensor(obs, device=self.device, dtype=dtype).unsqueeze(0)
            enc_obs = self._encode_with_module(self.policy_encoder, obs_tensor, project=True)

            if self.gradient_coeff is None:
                return np.ones(self.n_actions) / self.n_actions

            enc_obs_augmented = torch.cat(
                [enc_obs, torch.zeros((1, 1), device=enc_obs.device, dtype=enc_obs.dtype)],
                dim=1,
            )
            H = self._kernel(enc_obs_augmented, self._phi_all_obs)
            probs = self._policy_from_H(H)

            if torch.sum(probs) == 0.0 or torch.isnan(torch.sum(probs)):
                utils.ColorPrint.red(
                    "Warning: action_probs sum to zero or NaN. Returning uniform distribution. "
                    f"Check training stability and learning rates.{torch.sum(probs)}, {probs}"
                )
                probs = torch.ones_like(probs) / self.n_actions
            return probs.cpu().numpy().flatten()

    agent.compute_action_probs = types.MethodType(compute_action_probs, agent)


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


def random_action(action_space, rng: np.random.Generator):
    sample = getattr(action_space, "sample", None)
    if not callable(sample):
        raise TypeError(f"Unsupported action space without sample(): {action_space}")

    seed = getattr(action_space, "seed", None)
    if callable(seed):
        seed(int(rng.integers(0, 2**31 - 1)))
    return action_space.sample()


def _pointmaze_wall_rectangles(env) -> np.ndarray:
    get_layout = get_env_method(env, "get_debug_maze_layout")
    if not callable(get_layout):
        return np.empty((0, 4), dtype=np.float32)
    layout = get_layout()
    if not isinstance(layout, dict) or "wall_rectangles" not in layout:
        return np.empty((0, 4), dtype=np.float32)
    return np.asarray(layout["wall_rectangles"], dtype=np.float32).reshape(-1, 4)


def _points_inside_rectangles(points: np.ndarray, rectangles: np.ndarray, margin: float = 0.0) -> np.ndarray:
    if rectangles.size == 0:
        return np.zeros(len(points), dtype=bool)

    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    x = points[:, 0:1]
    y = points[:, 1:2]
    x0 = rectangles[:, 0] - margin
    y0 = rectangles[:, 1] - margin
    x1 = rectangles[:, 0] + rectangles[:, 2] + margin
    y1 = rectangles[:, 1] + rectangles[:, 3] + margin
    return ((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)).any(axis=1)


def _reset_valid_pointmaze_start(env, seed: int, wall_rectangles: np.ndarray, max_attempts: int = 100):
    """Reject noisy PointMaze starts that land in/too near walls."""
    last_time_step = None
    for attempt in range(max_attempts):
        time_step = env.reset(seed=seed + attempt * 1009)
        last_time_step = time_step
        point = extract_eval_trajectory_point(env, time_step)
        if point is None or not _points_inside_rectangles(np.asarray([point]), wall_rectangles, margin=0.05)[0]:
            return time_step, attempt
    return last_time_step, max_attempts


def _as_numpy_1d(value) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    return value if value.size else None


def _agent_latent_probs(agent, dim: int) -> np.ndarray:
    """Best-effort plot-only prior for categorical latent variables."""
    for name in (
        "z_probs",
        "z_prob",
        "p_z",
        "pz",
        "skill_probs",
        "skill_prob",
        "p_skill",
        "latent_probs",
        "latent_prob",
    ):
        probs = _as_numpy_1d(getattr(agent, name, None))
        if probs is None or probs.size != dim:
            continue
        probs = np.clip(probs, 0.0, None)
        total = probs.sum()
        if np.isfinite(total) and total > 0:
            return (probs / total).astype(np.float64)
    return np.full(dim, 1.0 / dim, dtype=np.float64)


def _sample_one_hot(dim: int, rng: np.random.Generator, probs: np.ndarray) -> np.ndarray:
    sample = np.zeros(dim, dtype=np.float32)
    sample[int(rng.choice(dim, p=probs))] = 1.0
    return sample


def _sample_latent_meta_for_plot(agent, meta: Mapping, rng: np.random.Generator):
    if not isinstance(meta, Mapping):
        return meta, False

    sampled_meta = dict(meta)
    has_latent_meta = False

    if "z" in sampled_meta:
        z = np.asarray(sampled_meta["z"], dtype=np.float32)
        if z.ndim == 1 and z.size > 0:
            sampled_meta["z"] = _sample_one_hot(z.size, rng, _agent_latent_probs(agent, z.size))
            has_latent_meta = True

    if "skill" in sampled_meta:
        skill = np.asarray(sampled_meta["skill"], dtype=np.float32)
        if skill.ndim == 1 and skill.size > 0:
            # CIC uses continuous latent skill vectors. Sample one per trajectory and keep it fixed.
            sampled_meta["skill"] = rng.uniform(0.0, 1.0, size=skill.shape).astype(np.float32)
            has_latent_meta = True

    return sampled_meta, has_latent_meta


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
    wall_rectangles = _pointmaze_wall_rectangles(env)
    rejected_starts = 0
    wall_points = 0

    for episode in range(num_trajectories):
        episode_seed = seed + episode
        time_step, rejected = _reset_valid_pointmaze_start(env, episode_seed, wall_rectangles)
        rejected_starts += rejected
        meta = agent.init_meta() if agent is not None and callable(getattr(agent, "init_meta", None)) else {}
        meta, fixed_latent_meta = _sample_latent_meta_for_plot(
            agent,
            meta,
            np.random.default_rng(seed + episode * 1000003),
        )
        trajectory = []

        point = extract_eval_trajectory_point(env, time_step)
        if point is not None:
            trajectory.append(point)
        frames.append(render_goal_hidden_frame(env))

        for step in range(episode_steps):
            if agent is None:
                action = random_action(env.action_space, rng=np.random.default_rng(seed + episode * 100000 + step))
            else:
                with torch.no_grad(), utils.eval_mode(agent):
                    action = agent.act(
                        time_step.observation,
                        meta,
                        policy_step,
                        eval_mode=deterministic,
                    )
                    # policy_prob = getattr(agent, "compute_action_probs", None)
                    # if callable(policy_prob):
                    #     probs = policy_prob(time_step.observation)
                    #     if not np.all(np.isfinite(probs)) or np.sum(probs) <= 0.0:
                    #         utils.ColorPrint.red(
                    #             f"Warning: action_probs sum to zero or NaN. Returning uniform distribution. "
                    #             f"Check training stability and learning rates.{np.sum(probs)}, {probs}"
                    #         )
                    #         probs = np.ones_like(probs) / len(probs)
                    #     if np.random.rand() < 0.5:
                    #         action = np.random.choice(len(probs), p=probs)
                    #     else:
                    #         action = np.argmax(probs) if deterministic else np.random.choice(len(probs), p=probs)
            time_step = env.step(action)

            update_meta = getattr(agent, "update_meta", None)
            if agent is not None and callable(update_meta) and not fixed_latent_meta:
                meta = update_meta(meta, policy_step, time_step)

            point = extract_eval_trajectory_point(env, time_step)
            if point is not None:
                if _points_inside_rectangles(np.asarray([point]), wall_rectangles)[0]:
                    wall_points += 1
                trajectory.append(point)
            frames.append(render_goal_hidden_frame(env))

            # if time_step.last():
            #     break

        if trajectory:
            trajectories.append(np.asarray(trajectory, dtype=np.float32))

    if rejected_starts:
        print(f"Rejected {rejected_starts} noisy PointMaze starts in/near walls.")
    if wall_points:
        print(f"Warning: collected {wall_points} trajectory points inside wall rectangles after reset.")

    return trajectories, frames


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a PointMaze policy snapshot, sample trajectories, save maze overlay plot and goal-hidden GIF."
    )
    parser.add_argument("--snapshot", type=Path, default=None)
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
    snapshot_path = args.snapshot.expanduser().resolve() if args.snapshot is not None else None
    if snapshot_path is not None and not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    config_path = args.config.expanduser().resolve() if args.config else None
    if config_path is None and snapshot_path is not None:
        config_path = find_run_config(snapshot_path)
    if config_path is None:
        config_path = Path("configs/env/pointmaze/pointmaze_umaze_goal_1.yaml").resolve()
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(
            "Could not find .hydra/config.yaml above snapshot. Pass --config explicitly."
        )

    output_dir = args.output_dir
    if output_dir is None:
        if snapshot_path is None:
            output_dir = Path("pointmaze_random_policy_samples")
        else:
            output_dir = snapshot_path.parent / "trajectory_samples"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    cfg = load_config(config_path)
    env = make_env(cfg, seed=args.seed, start_position_variance=args.start_position_variance)
    if snapshot_path is None:
        agent = None
        payload = {}
        step = 0
    else:
        agent, payload = load_snapshot(snapshot_path, device)
        step = snapshot_step(snapshot_path, payload)
    policy_step = int(args.policy_step) if args.policy_step is not None else step
    episode_steps = int(args.episode_steps) if args.episode_steps is not None else default_episode_steps(env)

    insert_env = getattr(agent, "insert_env", None) if agent is not None else None
    if agent is not None and callable(insert_env):
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

        prefix = snapshot_path.stem if snapshot_path is not None else "random_policy"
        gif_path = output_dir / f"{prefix}_ntraj_{len(trajectories)}_rollouts.gif"
        # save_gif(gif_path, frames, fps=int(args.gif_fps))
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    print(f"Config: {config_path}")
    print(f"Policy: {'random' if agent is None else snapshot_path}")
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
