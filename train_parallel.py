"""Parallel-environment finetuning entry point.

Uses pretrain_parallel's vector stepping/replay implementation, adding pretrained
agent initialization and PointMaze seed-trajectory logging needed by finetuning.
"""

import os
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from omegaconf import open_dict

import utils
from agent.rover_visualization.domains import save_maze_trajectory_overlay_plot
from pretrain_parallel import (
    NullContext,
    Workspace as ParallelWorkspace,
    enable_console_log,
)


class Workspace(ParallelWorkspace):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._seed_trajectories = [[] for _ in range(self.num_envs)]
        self._completed_seed_trajectories = []
        self._seed_trajectories_logged = False
        self._initialize_from_pretrained()

    def _initialize_from_pretrained(self):
        payload = None
        if int(getattr(self.cfg, "snapshot_ts", 0)) > 0:
            payload = self._load_configured_snapshot()
        else:
            path = str(getattr(self.cfg, "p_path", "none"))
            if path and path != "none":
                if path.endswith(".npy"):
                    self.agent = utils.load_policy_weights_into_agent(
                        self.agent, path, device=self.device
                    )
                else:
                    payload = self._load_snapshot_path(path)

        if payload is not None:
            pretrained_agent = payload["agent"]
            self.agent.init_from(pretrained_agent)

        if hasattr(self.agent, "insert_env"):
            self.agent.insert_env(self.eval_env)

    def _load_configured_snapshot(self):
        snapshot_dir = (
            Path(self.cfg.snapshot_base_dir)
            / self.cfg.obs_type
            / self.cfg.domain
            / self.cfg.agent.name
            / str(self.cfg.seed)
        )
        return self._load_snapshot_path(
            snapshot_dir / f"snapshot_{int(self.cfg.snapshot_ts)}.pt"
        )

    @staticmethod
    def _load_snapshot_path(path):
        snapshot = Path(path).expanduser()
        print(f"loading pretrained snapshot: {snapshot.resolve()}")
        if not snapshot.exists():
            raise FileNotFoundError(f"Pretrained snapshot not found: {snapshot}")
        with snapshot.open("rb") as stream:
            payload = torch.load(stream, weights_only=False, map_location="cpu")
        if not isinstance(payload, dict) or "agent" not in payload:
            raise ValueError(f"Snapshot must contain an 'agent' key: {snapshot}")
        return payload

    @property
    def _is_pointmaze(self):
        task_name = str(self.cfg.task_name).lower()
        return "pointmaze" in task_name or "point_maze" in task_name

    @staticmethod
    def _point_from_time_step(time_step):
        proprio = np.asarray(
            getattr(time_step, "proprio_observation", []), dtype=np.float32
        ).reshape(-1)
        if proprio.size < 2 or not np.all(np.isfinite(proprio[:2])):
            return None
        return proprio[:2].copy()

    def _on_transition_collected(self, time_step, logical_step, env_id):
        logical_frame = logical_step * int(self.cfg.action_repeat)
        if not self._is_pointmaze or logical_frame >= int(self.cfg.num_seed_frames):
            return
        point = self._point_from_time_step(time_step)
        if point is not None:
            self._seed_trajectories[env_id].append(point)
        if time_step.last():
            trajectory = self._seed_trajectories[env_id]
            if trajectory:
                self._completed_seed_trajectories.append(
                    np.asarray(trajectory, dtype=np.float32)
                )
            self._seed_trajectories[env_id] = []

    def _on_parallel_steps_completed(self, logical_steps):
        if self._seed_trajectories_logged or not self._is_pointmaze:
            return
        if self.global_frame < int(self.cfg.num_seed_frames):
            return

        trajectories = list(self._completed_seed_trajectories)
        trajectories.extend(
            np.asarray(trajectory, dtype=np.float32)
            for trajectory in self._seed_trajectories
            if trajectory
        )
        self._seed_trajectories_logged = True
        if not trajectories:
            print("No PointMaze seed trajectory points available to log")
            return

        save_dir = self.work_dir / str(
            getattr(self.cfg, "seed_trajectory_dir", "seed_trajectories")
        )
        try:
            plot_paths = save_maze_trajectory_overlay_plot(
                trajectories=trajectories,
                env=self.eval_env,
                step=int(self.cfg.num_seed_frames),
                save_dir=save_dir,
            )
            if self.cfg.use_wandb and wandb.run is not None:
                wandb.log(
                    {
                        f"seed_trajectories/{style}": wandb.Image(str(path))
                        for style, path in plot_paths.items()
                    },
                    step=int(self.cfg.num_seed_frames),
                )
            print(
                f"Logged {len(trajectories)} PointMaze trajectories from "
                f"{int(self.cfg.num_seed_frames)} seed frames"
            )
        except Exception as exc:
            print(f"Could not log PointMaze seed trajectories: {exc}")


@hydra.main(
    config_path="configs",
    config_name="train_parallel/finetune_maze",
    version_base="1.1",
)
def main(cfg):
    root_dir = Path.cwd()
    if not hasattr(cfg, "save_log"):
        with open_dict(cfg):
            cfg.save_log = True

    log_context = (
        enable_console_log(root_dir / "train_parallel.log")
        if cfg.save_log
        else NullContext()
    )
    with log_context:
        workspace = Workspace(cfg)
        try:
            snapshot = root_dir / "snapshot.pt"
            if snapshot.exists():
                print(f"resuming: {snapshot}")
                workspace.load_snapshot()
            workspace.train()
        finally:
            workspace.close()


if __name__ == "__main__":
    main()
