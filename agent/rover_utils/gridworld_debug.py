"""GridWorld synthetic-data provider for optional Rover diagnostics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import numpy as np
from PIL import Image
import torch

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
