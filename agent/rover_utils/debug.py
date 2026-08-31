"""Optional diagnostics and debug data providers for Rover Nyström.

This module is imported lazily only when ``RoverAgent(debug=True)``.  It owns
all plotting state and live environment references so the algorithm module
remains usable without visualization dependencies.
"""

from __future__ import annotations

import os
import copy
from contextlib import contextmanager

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import torch
import torch.nn as nn

from agent.rover_utils.pointmaze_debug import PointMazeNystromDebugHelper
from agent.rover_utils.types import EncodedActorUpdateData, RawActorUpdateData
from agent.rover_utils.visualization.gridworld import EmbeddingDistributionVisualizerV2
from agent.rover_utils.visualization.suite import RoverDebugVisualizerSuite


def _config_dict(config):
    if config is None:
        return {}
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(config):
            return OmegaConf.to_container(config, resolve=True)
    except ImportError:
        pass
    return dict(config)


class RoverDebugManager:
    """Common Rover diagnostics plus domain-specific data/visualizers."""

    def __init__(self, agent, config=None):
        self.agent = agent
        self.config = _config_dict(config)
        self.domain = str(self.config.get("domain", "generic")).lower()
        self.debug_fixed_dataset_updates = bool(
            self.config.get("debug_fixed_dataset_updates", False)
        )
        self.nystrom_synthetic_subsamples = bool(
            self.config.get("nystrom_synthetic_subsamples", False)
        )
        self.action_probs = []
        self.action_probs_history = []
        self.policy_deviation_history = []
        self.wrapped_env = None
        self.env = None
        self.visualizer_suite = RoverDebugVisualizerSuite(
            agent=agent,
            exploration_visualizer=None,
            gridworld_visualizer_factory=EmbeddingDistributionVisualizerV2,
        )
        self.data_helper = self._build_data_helper()

    def preserve_legacy_rng_sequence(self):
        """Consume RNG exactly like removed ExplorationVisualizer witness.

        Historical pivoted-Cholesky sampling used global Torch RNG after this
        debug witness was initialized. Keeping this one-time sequence preserves
        seeded legacy runs without retaining visualization state or plots.
        """
        if not bool(self.config.get("preserve_legacy_rng_sequence", False)):
            return

        obs_shape = tuple(self.agent.obs_shape)
        if self.agent.obs_type == "pixels":
            layers = nn.Sequential(
                nn.Conv2d(obs_shape[0], 32, 3, stride=2), nn.ReLU(),
                nn.Conv2d(32, 32, 3), nn.ReLU(),
                nn.Conv2d(32, 32, 3), nn.ReLU(),
                nn.Conv2d(32, 32, 3), nn.ReLU(),
            )
            representation_dim = 32 * 7 * 7
        else:
            input_dim = int(obs_shape[0] if len(obs_shape) == 1 else np.prod(obs_shape))
            hidden_dim = max(128, input_dim * 2)
            layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 256), nn.ReLU(),
            )
            representation_dim = 256

        # Original construction order: layer defaults, projection, Kaiming re-init.
        torch.randn(1024, representation_dim)
        for module in layers.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _build_data_helper(self):
        if self.domain == "pointmaze":
            return PointMazeNystromDebugHelper(
                border_margin=float(self.config.get("nystrom_grid_border_margin", 0.05)),
                oversample=float(self.config.get("nystrom_grid_oversample", 2.0)),
                exact_grid=bool(self.config.get("nystrom_exact_grid", False)),
            )
        if self.domain == "gridworld":
            # Lazy import avoids loading legacy debug code in normal runs.
            from agent.rover_utils.gridworld_debug import GridWorldSyntheticData

            helper = GridWorldSyntheticData(self.agent)
            for name, default in (
                ("synthetic_dataset_exclude_state_idxs", []),
                ("synthetic_dataset_exclusion_mode", "one_action"),
                ("synthetic_dataset_excluded_action", 0),
                ("synthetic_subsample_exclude_state_idxs", []),
                ("synthetic_subsample_exclusion_mode", "one_action"),
                ("synthetic_subsample_excluded_action", 0),
            ):
                setattr(self.agent, name, self.config.get(name, default))
            return helper
        return None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["wrapped_env"] = None
        state["env"] = None
        suite = state.get("visualizer_suite")
        if suite is not None:
            suite = copy.copy(suite)
            suite.domain_visualizer = None
            state["visualizer_suite"] = suite
        return state

    def attach_env(self, env):
        self.wrapped_env = env
        self.env = getattr(env, "unwrapped", env)
        if self.data_helper is not None:
            self.data_helper.attach_env(env)
        self.visualizer_suite.attach_env(env)

    @contextmanager
    def detached_for_snapshot(self):
        wrapped_env, env = self.wrapped_env, self.env
        domain_visualizer = self.visualizer_suite.domain_visualizer
        helper_refs = None
        if self.data_helper is not None:
            helper_refs = (
                getattr(self.data_helper, "wrapped_env", None),
                getattr(self.data_helper, "env", None),
            )
            self.data_helper.wrapped_env = None
            self.data_helper.env = None
        self.wrapped_env = None
        self.env = None
        self.visualizer_suite.domain_visualizer = None
        try:
            yield
        finally:
            self.wrapped_env, self.env = wrapped_env, env
            self.visualizer_suite.domain_visualizer = domain_visualizer
            if self.data_helper is not None and helper_refs is not None:
                self.data_helper.wrapped_env, self.data_helper.env = helper_refs

    def record_action_probs(self, probabilities):
        self.action_probs.append(np.asarray(probabilities))

    def fixed_actor_update_data(self):
        if self.data_helper is None:
            raise RuntimeError(f"Fixed debug dataset is unavailable for domain={self.domain}")
        if self.domain == "gridworld":
            full = self.data_helper.full_actor_batch()
            subsample = self.data_helper.subsample_actor_batch() if self.agent.subsamples is not None else None
            source = "fixed GridWorld debug dataset"
        else:
            full = self.data_helper.fixed_actor_batch(
                self.agent, n_transitions=int(self.agent.batch_size_actor)
            )
            subsample = (
                self.data_helper.fixed_actor_batch(
                    self.agent, n_transitions=self.agent._nystrom_subsample_count()
                )
                if self.agent.subsamples is not None else None
            )
            source = "fixed PointMaze debug dataset"
        return RawActorUpdateData(full=full, subsample=subsample, source=source)

    def fixed_encoder_batch(self):
        if self.data_helper is None:
            raise RuntimeError(f"Fixed debug dataset is unavailable for domain={self.domain}")
        return self.data_helper.fixed_encoder_batch(self.agent)

    def synthetic_raw_subsample(self):
        if self.data_helper is None:
            raise RuntimeError(f"Synthetic landmarks are unavailable for domain={self.domain}")
        if self.domain == "gridworld":
            return self.data_helper.subsample_actor_batch()
        return self.data_helper.fixed_actor_batch(self.agent)

    def synthetic_encoded_subsample(self):
        if self.data_helper is None:
            raise RuntimeError(f"Synthetic landmarks are unavailable for domain={self.domain}")
        return self.data_helper.encode_subsamples(self.agent)

    def actor_data_updated(self, actor_data, step):
        if self.domain == "pointmaze":
            self._plot_pointmaze_actor_data(actor_data, step)
        self._plot_kernel_diagnostics(step)
        if self.domain == "gridworld":
            adapter = self.visualizer_suite.domain_visualizer
            visualizer = getattr(adapter, "visualizer", None)
            if visualizer is not None and hasattr(visualizer, "save_dataset_subsample_policy_heatmaps"):
                path = os.path.join(
                    os.getcwd(), "gridworld_plots",
                    f"step_{step}_dataset_subsamples_policy_heatmaps.png",
                )
                visualizer.save_dataset_subsample_policy_heatmaps(step, path)

    def _actor_points(self, data):
        if data is None:
            return None
        if isinstance(data, dict):
            xy = data.get("debug_xy")
            if xy is None:
                return None
            return xy.detach().float().cpu().numpy().reshape(xy.shape[0], -1)[:, :2]
        obs = data[0]
        if self.agent.obs_type == "pixels" or obs.ndim < 2 or obs.shape[1] < 2:
            return None
        return obs.detach().float().cpu().numpy().reshape(obs.shape[0], -1)[:, :2]

    def _plot_pointmaze_actor_data(self, actor_data, step):
        save_dir = os.path.join(
            os.getcwd(), self.config.get("pointmaze_plot_dir", "pointmaze_plots")
        )
        for data, filename, title in (
            (actor_data.full, f"step_{step}_actor_full_dataset.png", "Actor full dataset"),
            (actor_data.subsample, f"step_{step}_nystrom_subsamples.png", "Nyström landmarks"),
        ):
            points = self._actor_points(data)
            if points is None or points.size == 0:
                continue
            os.makedirs(save_dir, exist_ok=True)
            fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
            try:
                layout = self.data_helper.maze_layout()
            except Exception:
                layout = None
            if isinstance(layout, dict):
                for x0, y0, width, height in layout["wall_rectangles"]:
                    ax.add_patch(patches.Rectangle((x0, y0), width, height, color="black"))
                lower = np.asarray(layout["maze_lower"])
                upper = np.asarray(layout["maze_upper"])
                ax.set_xlim(lower[0] - 0.1, upper[0] + 0.1)
                ax.set_ylim(lower[1] - 0.1, upper[1] + 0.1)
            ax.scatter(points[:, 0], points[:, 1], s=8, color="#ff7f0e", alpha=0.9)
            ax.scatter(points[0, 0], points[0, 1], marker="*", s=130, color="white", edgecolor="black")
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{title}\n{points.shape[0]} states")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches="tight")
            plt.close(fig)

    @staticmethod
    def _bounded_indices(size, maximum, device):
        if size <= maximum:
            return torch.arange(size, device=device)
        return torch.linspace(0, size - 1, maximum, device=device).round().long()

    def _plot_kernel_diagnostics(self, step):
        if self.agent.kernel_type != "gaussian" or not hasattr(self.agent, "_phi_sub_next"):
            return
        state = self.agent._phi_sub_next
        state_action = self.agent._phi_sub_obs
        actions = self.agent._sub_actions.to(device=state_action.device)
        state_idx = self._bounded_indices(state.shape[0], 300, state.device)
        action_idx = self._bounded_indices(state_action.shape[0], 300, state_action.device)
        with torch.no_grad():
            state_matrix = self.agent.kernel_fn(
                state[state_idx], state[state_idx]
            ).detach().float().cpu().numpy()
            action_matrix = self.agent.distribution_matcher.state_action_kernel(
                state_action[action_idx], state_action[action_idx],
                actions[action_idx], actions[action_idx],
            ).detach().float().cpu().numpy()
        save_dir = os.path.join(os.getcwd(), "kernel_debug_plots")
        os.makedirs(save_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
        for ax, matrix, title in (
            (axes[0], state_matrix, "State kernel"),
            (axes[1], action_matrix, "State-action kernel"),
        ):
            image = ax.imshow(matrix, cmap="viridis", aspect="auto")
            ax.set_title(title)
            fig.colorbar(image, ax=ax)
        fig.savefig(
            os.path.join(save_dir, f"step_{step}_nystrom_kernels.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    def update(self, metrics, step):
        metrics.update(self.visualizer_suite.save(step=step))
        if not self.action_probs:
            return metrics
        probabilities = np.asarray(self.action_probs)
        mean_probs = probabilities.mean(axis=0)
        deviation = float(np.mean(np.abs(mean_probs - 1.0 / self.agent.n_actions)))
        self.action_probs_history.append((step, mean_probs))
        self.policy_deviation_history.append((step, deviation))
        self.action_probs.clear()
        metrics["policy_deviation_from_uniform"] = deviation
        self._plot_policy_deviation()
        return metrics

    def _plot_policy_deviation(self):
        save_dir = os.path.join(os.getcwd(), self.config.get("policy_plot_dir", "policy_plots"))
        os.makedirs(save_dir, exist_ok=True)
        steps, deviations = zip(*self.policy_deviation_history)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(steps, deviations, color="blue", linewidth=2, label="Policy deviation")
        ax.axhline(0, color="green", linestyle="--", label="Uniform policy")
        ax.set_xlabel("Training steps")
        ax.set_ylabel("Mean |P(a) - 1/|A||")
        ax.set_title("Policy concentration over time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "policy_deviation_history.png"), dpi=150)
        plt.close(fig)


def make_debug_manager(agent, config=None):
    return RoverDebugManager(agent, config)
