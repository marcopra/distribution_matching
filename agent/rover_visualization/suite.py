from __future__ import annotations

import os
from typing import Callable, Optional

from .domains import (
    BaseDomainDebugVisualizer,
    FetchCoverageVisualizer,
    GridworldVisualizerAdapter,
    PointMazeCoverageVisualizer,
    XYCoverageVisualizer,
    _find_discrete_env,
    _has_debug_xy_env,
    _is_fetch_env,
    _is_point_maze_env,
)


class RoverDebugVisualizerSuite:
    def __init__(
        self,
        agent,
        exploration_visualizer,
        gridworld_visualizer_factory: Callable,
    ):
        self.agent = agent
        self.exploration_visualizer = exploration_visualizer
        self._gridworld_visualizer_factory = gridworld_visualizer_factory
        self.domain_visualizer: Optional[BaseDomainDebugVisualizer] = None

    def attach_env(self, env) -> Optional[BaseDomainDebugVisualizer]:
        if _find_discrete_env(env) is not None:
            self.domain_visualizer = GridworldVisualizerAdapter(
                self._gridworld_visualizer_factory(self.agent)
            )
        elif _is_fetch_env(env):
            self.domain_visualizer = FetchCoverageVisualizer(self.agent, env, rollout_steps=500, bins=10)
        elif _is_point_maze_env(env):
            self.domain_visualizer = PointMazeCoverageVisualizer(self.agent, env, rollout_steps=10000, bins=20)
        elif _has_debug_xy_env(env):
            self.domain_visualizer = XYCoverageVisualizer(
                self.agent,
                env,
                save_dir="continuous_xy_plots",
                rollout_steps=1000,
                bins=40,
                title_prefix="Continuous XY",
                policy_eval_points=100,
            )
        else:
            self.domain_visualizer = None
        return self.domain_visualizer

    def save(self, step: int, obs_batch=None, z_batch=None, param_text: str = "") -> dict:
        metrics = {}
        if self.exploration_visualizer is not None and obs_batch is not None and z_batch is not None:
            vis_metrics = self.exploration_visualizer.update(
                obs_batch=obs_batch,
                z_batch=z_batch,
                step=step,
            )
            metrics.update(vis_metrics)
            self.exploration_visualizer.plot_all(step, param_text=param_text)

            
            try:
                self.exploration_visualizer.plot_tsne(
                    z_batch,
                    step,
                    method="tsne",
                )
            except Exception as exc:
                print(f"⚠ Could not generate t-SNE plot at step {step}: {exc}")

        if self.domain_visualizer is not None:
            try:
                self.domain_visualizer.save(step)
            except Exception as exc:
                print(f"⚠ Could not generate domain debug plot at step {step}: {exc}")

        return metrics


def build_debug_visualizer_suite(
    agent,
    exploration_visualizer_cls,
    gridworld_visualizer_cls,
):
    exploration_visualizer = exploration_visualizer_cls(
        obs_shape=agent.obs_shape,
        obs_type=agent.obs_type,
        feature_dim=agent.feature_dim,
        hash_dim=1024,
        k_neighbors=5,
        occupancy_window=agent.update_actor_every_steps * 3,
        save_dir=os.path.join("exploration_plots", os.getcwd()),
        device=agent.device,
    )
    return RoverDebugVisualizerSuite(
        agent=agent,
        exploration_visualizer=exploration_visualizer,
        gridworld_visualizer_factory=gridworld_visualizer_cls,
    )
