"""GridWorld-specific Nyström debug agent.

This keeps the PointMaze debug agent's PMD implementation, but replaces its
synthetic landmark builder and actor-data diagnostics with discrete GridWorld
versions.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

import utils
from agent.rover_nystrom_pointmaze_debug import RoverAgent as PointMazeRoverAgent
from agent.utils import EncodedActorUpdateData, RawActorUpdateData


class GridWorldSyntheticData:
    """Build fixed state-action transitions from a discrete GridWorld."""

    def __init__(self, agent):
        self.agent = agent
        self.wrapped_env = None
        self.env = None
        self._full_batch = None
        self._subsample_batch = None
        self.full_state_indices = None
        self.subsample_state_indices = None

    def attach_env(self, env):
        self.wrapped_env = env
        self.env = self._find_discrete_env(env)
        self.clear_cache()

    def clear_cache(self):
        self._full_batch = None
        self._subsample_batch = None
        self.full_state_indices = None
        self.subsample_state_indices = None

    @staticmethod
    def _find_discrete_env(env):
        current = env
        visited = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if all(hasattr(current, key) for key in ("n_states", "idx_to_state", "state_to_idx")):
                return current
            current = getattr(current, "env", None)
        raise RuntimeError("GridWorld synthetic data requires n_states, idx_to_state, and state_to_idx")

    def _initial_state_index(self):
        start = getattr(self.env, "start_position", None)
        if start is None:
            start = getattr(self.env, "_start_position_param", None)
        if start is None:
            return 0
        if isinstance(start, (int, np.integer)):
            return int(start)
        start = tuple(start)
        if start not in self.env.state_to_idx:
            raise ValueError(f"Initial state {start} is not a valid GridWorld state")
        return int(self.env.state_to_idx[start])

    @staticmethod
    def _parse_indices(values):
        if values is None:
            return set()
        if isinstance(values, str):
            values = [value.strip() for value in values.split(",") if value.strip()]
        elif isinstance(values, Iterable):
            values = list(values)
        else:
            values = [values]
        return {int(value) for value in values}

    def _eligible_pairs(self, scope):
        excluded = self._parse_indices(getattr(self.agent, f"synthetic_{scope}_exclude_state_idxs"))
        invalid = sorted(index for index in excluded if index < 0 or index >= self.env.n_states)
        if invalid:
            raise ValueError(
                f"synthetic_{scope}_exclude_state_idxs contains invalid indices {invalid}; "
                f"valid range is [0, {self.env.n_states - 1}]"
            )

        mode = getattr(self.agent, f"synthetic_{scope}_exclusion_mode")
        excluded_action = getattr(self.agent, f"synthetic_{scope}_excluded_action")
        if mode not in ("one_action", "all_actions"):
            raise ValueError(
                f"synthetic_{scope}_exclusion_mode must be one_action or all_actions"
            )
        if excluded_action < 0 or excluded_action >= self.agent.n_actions:
            raise ValueError(
                f"synthetic_{scope}_excluded_action={excluded_action} is outside "
                f"[0, {self.agent.n_actions - 1}]"
            )

        initial_state = self._initial_state_index()
        if mode == "all_actions" and initial_state in excluded:
            raise ValueError("Cannot exclude all actions from initial state required by Nyström alpha")

        state_indices = np.repeat(np.arange(self.env.n_states, dtype=np.int64), self.agent.n_actions)
        actions = np.tile(np.arange(self.agent.n_actions, dtype=np.int64), self.env.n_states)
        excluded_states = np.isin(state_indices, np.asarray(sorted(excluded), dtype=np.int64))
        remove = excluded_states if mode == "all_actions" else excluded_states & (actions == excluded_action)
        state_indices = state_indices[~remove]
        actions = actions[~remove]
        if state_indices.size == 0:
            raise ValueError(f"Synthetic {scope} exclusions removed every state-action pair")

        initial_rows = np.flatnonzero(state_indices == initial_state)
        if initial_rows.size == 0:
            raise ValueError("Synthetic data has no initial-state support")
        first = int(initial_rows[0])
        if first != 0:
            order = np.concatenate(([first], np.delete(np.arange(state_indices.size), first)))
            state_indices = state_indices[order]
            actions = actions[order]
        return state_indices, actions

    def _prepare_image(self, image):
        image = np.asarray(image, dtype=np.uint8)
        if self.agent.grayscale:
            image = np.asarray(Image.fromarray(image).convert("L"))
        elif image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        resolution = int(self.agent.obs_shape[-1])
        if image.shape[:2] != (resolution, resolution):
            image = np.asarray(Image.fromarray(image).resize((resolution, resolution), Image.LANCZOS))
        if image.ndim == 2:
            image = image[..., None]
        chw = image.transpose(2, 0, 1).copy()
        frame_stack = self.agent.obs_shape[0] // chw.shape[0]
        return np.tile(chw, (frame_stack, 1, 1))

    def _observations(self, state_indices):
        if self.agent.obs_type != "pixels":
            observations = np.zeros((state_indices.size, *self.agent.obs_shape), dtype=np.float32)
            flat = observations.reshape(state_indices.size, -1)
            if self.env.n_states > flat.shape[1]:
                raise ValueError("GridWorld state count does not fit one-hot observation shape")
            flat[np.arange(state_indices.size), state_indices] = 1.0
            return observations

        rendered = []
        for state_index in state_indices:
            state = self.env.idx_to_state[int(state_index)]
            try:
                image = self.env.render_from_position(state, show_goal=False)
            except TypeError:
                image = self.env.render_from_position(state)
            rendered.append(self._prepare_image(image))
        return np.stack(rendered).astype(np.float32, copy=False)

    def _build(self, scope):
        if self.env is None:
            raise RuntimeError("insert_env(env) must run before synthetic GridWorld data is built")
        state_indices, actions = self._eligible_pairs(scope)

        next_state_indices = np.fromiter(
            (
                self.env.state_to_idx[self.env.step_from(self.env.idx_to_state[int(state)], int(action))]
                for state, action in zip(state_indices, actions)
            ),
            dtype=np.int64,
            count=state_indices.size,
        )
        # Actor alpha treats row zero as initial distribution support.
        next_state_indices[0] = state_indices[0]
        obs = torch.as_tensor(self._observations(state_indices), dtype=torch.float32, device=self.agent.device)
        next_obs = torch.as_tensor(self._observations(next_state_indices), dtype=torch.float32, device=self.agent.device)
        action = torch.as_tensor(actions, dtype=torch.long, device=self.agent.device)
        reward = torch.zeros((state_indices.size, 1), dtype=self.agent.compute_dtype, device=self.agent.device)
        batch = self.agent._make_actor_batch(obs, action, next_obs, reward)
        return batch, state_indices

    def full_actor_batch(self):
        if self._full_batch is None:
            self._full_batch, self.full_state_indices = self._build("dataset")
        return self._full_batch

    def subsample_actor_batch(self):
        if self._subsample_batch is None:
            self._subsample_batch, self.subsample_state_indices = self._build("subsample")
        return self._subsample_batch

    def fixed_actor_batch(self, agent, n_transitions: Optional[int] = None):
        del agent, n_transitions
        return self.full_actor_batch()

    def fixed_encoder_batch(self, agent):
        batch = self.full_actor_batch()
        size = min(int(agent.batch_size), batch[0].shape[0])
        return agent._slice_actor_batch(batch, slice(0, size))

    def encode_subsamples(self, agent):
        agent._sync_policy_encoder()
        batch = self.subsample_actor_batch()
        transitions = (batch[0], batch[1], batch[3], torch.ones_like(batch[3]), batch[2])
        encoded = agent._encode_actor_transition_batch_with_retries(transitions)
        return encoded, encoded.get("reward")


class RoverAgent(PointMazeRoverAgent):
    """PointMaze Nyström implementation adapted to finite GridWorlds."""

    def __init__(
        self,
        *args,
        synthetic_dataset_exclude_state_idxs=None,
        synthetic_dataset_exclusion_mode="one_action",
        synthetic_dataset_excluded_action=0,
        synthetic_subsample_exclude_state_idxs=None,
        synthetic_subsample_exclusion_mode="one_action",
        synthetic_subsample_excluded_action=0,
        subsampling_strategy="random",
        nystrom_candidate_multiplier=5.0,
        nystrom_cholesky_tolerance=1e-6,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.synthetic_dataset_exclude_state_idxs = synthetic_dataset_exclude_state_idxs or []
        self.synthetic_dataset_exclusion_mode = str(synthetic_dataset_exclusion_mode).lower()
        self.synthetic_dataset_excluded_action = int(synthetic_dataset_excluded_action)
        self.synthetic_subsample_exclude_state_idxs = synthetic_subsample_exclude_state_idxs or []
        self.synthetic_subsample_exclusion_mode = str(synthetic_subsample_exclusion_mode).lower()
        self.synthetic_subsample_excluded_action = int(synthetic_subsample_excluded_action)
        self.subsampling_strategy = str(subsampling_strategy).lower()
        if self.subsampling_strategy not in (
            "random", "gamma_h", "reverse_gamma_h", "pivoted_cholesky"
        ):
            raise ValueError(
                "subsampling_strategy must be random, gamma_h, reverse_gamma_h, "
                "or pivoted_cholesky"
            )
        self.nystrom_candidate_multiplier = float(nystrom_candidate_multiplier)
        self.nystrom_cholesky_tolerance = float(nystrom_cholesky_tolerance)
        if self.nystrom_candidate_multiplier < 1.0:
            raise ValueError("nystrom_candidate_multiplier must be at least 1")
        if self.nystrom_cholesky_tolerance < 0.0:
            raise ValueError("nystrom_cholesky_tolerance must be non-negative")
        if self.subsampling_strategy == "pivoted_cholesky":
            # Selector supports kernels whose candidate Gram columns it can
            # reproduce exactly. Gaussian bandwidth is fitted from candidate
            # pool before selection, then reused by actor update so geometries
            # cannot silently differ. Extend this assertion together with FIFO
            # kernel-column computation when adding another kernel.
            assert self.kernel_type in ("inner_product", "gaussian"), (
                "pivoted_cholesky subsampling currently requires kernel_type "
                "inner_product or gaussian"
            )
        if self.n_actions != 4:
            raise ValueError("GridWorld Nyström debug agent requires exactly four actions")
        self.nystrom_debug = GridWorldSyntheticData(self)

    def _fixed_actor_update_data(self):
        full = self.nystrom_debug.full_actor_batch()
        subsample = self.nystrom_debug.subsample_actor_batch() if self.subsamples is not None else None
        return RawActorUpdateData(
            full=full,
            subsample=subsample,
            source="fixed GridWorld synthetic dataset",
        )

    def _synthetic_actor_subsample_batch(self):
        return self.nystrom_debug.subsample_actor_batch()

    def _update_encoded_actor_fifo(self, replay_buffer):
        if replay_buffer is None or not hasattr(replay_buffer, "get_new_transitions_since"):
            return False
        self._sync_policy_encoder()
        inserted = 0
        encode_batch_size = max(1, self.encoded_fifo_encode_batch_size)
        while True:
            transition_ids, transitions = replay_buffer.get_new_transitions_since(
                self._encoded_fifo_replay_marker,
                limit=encode_batch_size,
            )
            if transition_ids is None:
                break
            terminal_mask = np.asarray(transitions[3]).reshape(len(transition_ids), -1).min(axis=1) <= 0.0
            encoded = self._encode_actor_transition_batch_with_retries(transitions)
            self._encoded_actor_fifo.add(transition_ids, encoded, terminal_mask=terminal_mask)
            self._encoded_fifo_replay_marker = int(transition_ids[-1])
            if hasattr(replay_buffer, "mark_transitions_encoded"):
                replay_buffer.mark_transitions_encoded(self._encoded_fifo_replay_marker)
            inserted += int(len(transition_ids))
        self._insert_first_transition_if_available(replay_buffer)
        return inserted > 0 or len(self._encoded_actor_fifo) > 0

    def _sample_encoded_actor_data(self, size, include_first):
        encoded = self._encoded_actor_fifo.sample_by_strategy(
            int(size),
            self.device,
            strategy=self.subsampling_strategy,
            gamma=self.discount,
            include_first=include_first,
            candidate_multiplier=self.nystrom_candidate_multiplier,
            cholesky_tolerance=self.nystrom_cholesky_tolerance,
            kernel_type=self.kernel_type,
            kernel_bandwidth=self.kernel_bandwidth,
            kernel_bandwidth_mult=self.kernel_bandwidth_mult,
        )
        if self.subsampling_strategy == "pivoted_cholesky" and self.kernel_type == "gaussian":
            bandwidth = self._encoded_actor_fifo.last_pivoted_cholesky_bandwidth
            self.kernel_fn.bandwidth = bandwidth
            self.distribution_matcher.kernel_fn.bandwidth = bandwidth
        return encoded, encoded.get("reward")

    def _fit_state_kernel_bandwidth(self, X, Y):
        if self.subsampling_strategy == "pivoted_cholesky" and self.kernel_type == "gaussian":
            bandwidth = self._encoded_actor_fifo.last_pivoted_cholesky_bandwidth
            if bandwidth is not None:
                self.kernel_fn.bandwidth = bandwidth
                self.distribution_matcher.kernel_fn.bandwidth = bandwidth
                utils.ColorPrint.yellow(
                    f"Using pivoted-Cholesky candidate-pool Gaussian bandwidth={bandwidth:.6g}."
                )
                return
            # Synthetic/fixed landmark paths bypass FIFO candidate selection.
            # Preserve their existing Gaussian bandwidth behavior.
        super()._fit_state_kernel_bandwidth(X, Y)

    def _encoded_fifo_actor_update_data(self, replay_buffer):
        if not self._update_encoded_actor_fifo(replay_buffer):
            return None
        if self.subsamples is None:
            full, rewards = self._sample_encoded_actor_data(self.batch_size_actor, include_first=True)
            return EncodedActorUpdateData(
                full=full,
                rewards=rewards,
                source=f"encoded FIFO {self.subsampling_strategy} sample",
            )
        full, rewards = self._all_encoded_actor_data(include_first=True)
        if self.nystrom_synthetic_subsamples:
            subsample, subsample_rewards = self.nystrom_debug.encode_subsamples(self)
            source = "synthetic GridWorld landmarks"
        else:
            subsample, subsample_rewards = self._sample_encoded_actor_data(
                self._nystrom_subsample_count(), include_first=True
            )
            source = f"{self.subsampling_strategy} landmarks"
        return EncodedActorUpdateData(
            full=full,
            rewards=rewards,
            subsample=subsample,
            subsample_rewards=subsample_rewards,
            source=f"encoded FIFO full support + {source}",
        )

    # PointMaze scatter plots are intentionally replaced by GridWorld plots.
    def _save_actor_full_dataset_plot(self, actor_data, step):
        del actor_data, step

    def _save_actor_nystrom_subsample_plot(self, actor_data, step):
        del actor_data, step

    def _save_gridworld_histograms(self, step):
        adapter = getattr(self.debug_visualizer, "domain_visualizer", None)
        visualizer = getattr(adapter, "visualizer", None)
        if visualizer is None or not hasattr(visualizer, "_compute_batch_and_subsample_state_counts"):
            return
        full_counts, subsample_counts = visualizer._compute_batch_and_subsample_state_counts()
        if full_counts is None:
            return
        full_state_action, subsample_state_action = (
            visualizer._compute_batch_and_subsample_state_action_counts()
        )
        save_dir = os.path.join(os.getcwd(), "gridworld_plots")
        os.makedirs(save_dir, exist_ok=True)
        panels = 3 if self._encoded_actor_fifo.last_sampled_time_steps is not None else 2
        fig, axes = plt.subplots(2, panels, figsize=(7 * panels, 10), squeeze=False)
        state_ids = np.arange(full_counts.shape[0])
        axes[0, 0].bar(state_ids, full_counts, color="#2563eb")
        axes[0, 0].set_title("Actor dataset state-index histogram")
        axes[0, 0].set_xlabel("GridWorld state index")
        axes[0, 0].set_ylabel("Count")
        if subsample_counts is not None:
            axes[0, 1].bar(state_ids, subsample_counts, color="#f97316")
        axes[0, 1].set_title("Nyström subsample state-index histogram")
        axes[0, 1].set_xlabel("GridWorld state index")
        axes[0, 1].set_ylabel("Count")
        if panels == 3:
            sampled_t = self._encoded_actor_fifo.last_sampled_time_steps
            bins = np.arange(int(sampled_t.max()) + 2) - 0.5
            axes[0, 2].hist(sampled_t, bins=bins, color="#16a34a", edgecolor="black")
            axes[0, 2].set_title(f"{self.subsampling_strategy} within-trajectory indices")
            axes[0, 2].set_xlabel("t")
            axes[0, 2].set_ylabel("Count")

        state_action_size = visualizer.n_states * visualizer.n_actions
        state_action_ids = np.arange(state_action_size)
        action_ids = state_action_ids % visualizer.n_actions
        action_colors = np.asarray(visualizer.action_colors)[action_ids]
        if full_state_action is not None:
            full_state_action_hist = full_state_action.T.reshape(-1)
            axes[1, 0].bar(state_action_ids, full_state_action_hist, color=action_colors)
        axes[1, 0].set_title("Actor dataset state-action-index histogram")
        axes[1, 0].set_xlabel("State-action index = state × |A| + action")
        axes[1, 0].set_ylabel("Count")
        if subsample_state_action is not None:
            subsample_state_action_hist = subsample_state_action.T.reshape(-1)
            axes[1, 1].bar(state_action_ids, subsample_state_action_hist, color=action_colors)
        axes[1, 1].set_title("Nyström subsample state-action-index histogram")
        axes[1, 1].set_xlabel("State-action index = state × |A| + action")
        axes[1, 1].set_ylabel("Count")
        if panels == 3:
            axes[1, 2].axis("off")
        fig.suptitle(f"GridWorld actor data diagnostics (step {step})")
        fig.tight_layout()
        path = os.path.join(save_dir, f"step_{step}_dataset_histograms.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"GridWorld dataset histograms saved to: {path}")

    def _save_gridworld_heatmaps(self, step):
        adapter = getattr(self.debug_visualizer, "domain_visualizer", None)
        visualizer = getattr(adapter, "visualizer", None)
        if visualizer is None or not hasattr(visualizer, "save_dataset_subsample_policy_heatmaps"):
            return
        save_path = os.path.join(
            os.getcwd(),
            "gridworld_plots",
            f"step_{step}_dataset_subsamples_policy_heatmaps.png",
        )
        try:
            visualizer.save_dataset_subsample_policy_heatmaps(step, save_path)
        except Exception as exc:
            utils.ColorPrint.red(f"Could not save GridWorld dataset/policy heatmaps: {exc}")

    def _update_actor_from_data(self, actor_data, step):
        metrics = super()._update_actor_from_data(actor_data, step)
        self._save_gridworld_heatmaps(step)
        self._save_gridworld_histograms(step)
        return metrics

    def _debug_visualizer_text(self, step):
        return super()._debug_visualizer_text(step) + f"subsampling = {self.subsampling_strategy}\n"
