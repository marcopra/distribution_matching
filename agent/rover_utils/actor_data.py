"""Actor-data plumbing kept separate from Rover's numerical algorithm.

Two immutable dataclasses describe data at the only two useful boundaries:
raw replay transitions and actor-ready encoded transitions. Helper classes own
replay batching, incremental FIFO synchronization, and memory-safe encoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import utils
from agent.rover_utils.buffers import EncodedTransitionFIFO


@dataclass(frozen=True)
class RawTransitions:
    """Raw actor transitions, always ordered as ``(s, a, s', r)``."""

    obs: torch.Tensor
    action: torch.Tensor
    next_obs: torch.Tensor
    reward: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.obs.shape[0])

    def slice(self, index) -> "RawTransitions":
        return RawTransitions(
            self.obs[index], self.action[index], self.next_obs[index], self.reward[index]
        )

    @classmethod
    def concatenate(
        cls, batches: Sequence["RawTransitions"], limit: Optional[int] = None
    ) -> "RawTransitions":
        if not batches:
            raise RuntimeError("No replay samples available for actor update")
        result = cls(*(
            torch.cat([getattr(batch, field) for batch in batches], dim=0)
            for field in ("obs", "action", "next_obs", "reward")
        ))
        return result if limit is None else result.slice(slice(0, limit))

    def with_first(self, first: "RawTransitions") -> "RawTransitions":
        """Put initial transition at row zero, matching alpha[0] = 1."""
        fields = []
        for name in ("obs", "action", "next_obs", "reward"):
            value = getattr(self, name).clone()
            value[:1] = getattr(first, name).to(value.device)
            fields.append(value)
        return RawTransitions(*fields)

    def as_tuple(self):
        return self.obs, self.action, self.next_obs, self.reward


@dataclass(frozen=True)
class EncodedTransitions:
    """Actor-ready features used by Nyström support and landmarks."""

    tensors: Dict[str, torch.Tensor]

    @property
    def size(self) -> int:
        return int(self.tensors["phi_obs"].shape[0])

    @property
    def reward(self) -> Optional[torch.Tensor]:
        return self.tensors.get("reward")


class TransitionEncoder:
    """Encode raw transitions; split only when a CUDA allocation fails."""

    def __init__(
        self,
        device: str,
        dtype: torch.dtype,
        n_actions: int,
        obs_type: str,
        encode_observations: Callable[[torch.Tensor], torch.Tensor],
        encode_state_action: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        max_oom_splits: int,
    ):
        self.device = device
        self.dtype = dtype
        self.n_actions = n_actions
        self.obs_type = obs_type
        self.encode_observations = encode_observations
        self.encode_state_action = encode_state_action
        self.max_oom_splits = int(max_oom_splits)

    def from_replay(self, transitions) -> RawTransitions:
        obs, action, reward, _, next_obs = utils.to_torch(transitions[:5], self.device)
        return RawTransitions(obs, action, next_obs, reward.reshape(obs.shape[0], -1))

    def encode(self, transitions) -> EncodedTransitions:
        return self.encode_raw(self.from_replay(transitions))

    def encode_raw(self, raw: RawTransitions) -> EncodedTransitions:
        """Encode an already materialized raw actor batch."""
        with torch.no_grad():
            phi_obs = self.encode_observations(raw.obs)
            phi_next = self.encode_observations(raw.next_obs)
            psi = self.encode_state_action(phi_obs, raw.action)
            action_one_hot = F.one_hot(
                raw.action.long(), self.n_actions
            ).reshape(-1, self.n_actions).to(dtype=self.dtype, device=self.device)
        tensors = {
            "phi_obs": phi_obs,
            "phi_next": phi_next,
            "psi": psi,
            "E": action_one_hot,
            "reward": raw.reward,
        }
        # Optional observation metadata exists only for diagnostics.
        if self.obs_type != "pixels" and raw.obs.ndim >= 2 and raw.obs.shape[1] >= 2:
            tensors["debug_xy"] = raw.obs.detach().reshape(raw.size, -1)[:, :2]
        return EncodedTransitions(tensors)

    def encode_safely(self, transitions, splits_left=None) -> EncodedTransitions:
        splits_left = self.max_oom_splits if splits_left is None else splits_left
        try:
            return self.encode(transitions)
        except torch.OutOfMemoryError:
            size = int(transitions[0].shape[0])
            if splits_left <= 0 or size <= 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            midpoint = size // 2
            left = tuple(field[:midpoint] for field in transitions)
            right = tuple(field[midpoint:] for field in transitions)
            encoded = [
                self.encode_safely(left, splits_left - 1),
                self.encode_safely(right, splits_left - 1),
            ]
            keys = encoded[0].tensors.keys()
            return EncodedTransitions({
                key: torch.cat([batch.tensors[key] for batch in encoded], dim=0)
                for key in keys
            })


class ActorBuffer:
    """Synchronize unseen replay transitions into a bounded encoded FIFO."""

    def __init__(self, capacity: int, encode_batch_size: int):
        self.fifo = EncodedTransitionFIFO(capacity)
        self.encode_batch_size = max(1, int(encode_batch_size))
        self.replay_marker = None

    def sync(
        self,
        replay_buffer,
        encoder: TransitionEncoder,
        sync_policy_encoder: Callable[[], None],
    ) -> bool:
        """Copy every unseen replay transition into the encoded actor FIFO.

        ``replay_marker`` stores the last transition ID processed by this
        buffer. Each call therefore encodes only transitions collected since
        the previous call. Raw transitions are fetched and encoded in bounded
        chunks so synchronization does not require one large forward pass.

        Returns ``True`` when the FIFO contains actor data after the call. It
        may return ``True`` even when no new transition was inserted, because
        data encoded by an earlier call can still be present in the FIFO.
        """
        # FIFO mode needs replay's ordered transition-stream API. An ordinary
        # random replay iterator cannot tell us which transitions are unseen.
        if replay_buffer is None or not hasattr(replay_buffer, "get_new_transitions_since"):
            return False

        # Freeze a current copy of the learned encoder for policy data. Every
        # transition inserted during this synchronization uses that same copy.
        sync_policy_encoder()

        inserted = 0
        while True:
            # Read transitions strictly after replay_marker. limit bounds GPU
            # encoding memory; loop continues until pending stream is empty.
            ids, transitions = replay_buffer.get_new_transitions_since(
                self.replay_marker, limit=self.encode_batch_size
            )
            if ids is None:
                break

            # Replay discount is non-positive at episode boundaries. FIFO uses
            # this mask only to retain trajectory metadata for diagnostics.
            terminal_mask = (
                np.asarray(transitions[3]).reshape(len(ids), -1).min(axis=1) <= 0.0
            )

            # Convert raw observations/actions to actor-ready phi/psi features.
            # encode_safely splits the chunk recursively after CUDA OOM.
            encoded = encoder.encode_safely(transitions)

            # Storage moves encoded tensors to CPU, pins transition zero, and
            # evicts oldest ordinary transitions when capacity is exceeded.
            self.fifo.add(ids, encoded.tensors, terminal_mask=terminal_mask)

            # Marker advances only after successful encoding and insertion.
            # A failed encode therefore remains retryable on the next call.
            self.replay_marker = int(ids[-1])

            # Raw pending data is no longer needed after its encoded version is
            # safely stored. Acknowledgment lets replay release that memory.
            if hasattr(replay_buffer, "mark_transitions_encoded"):
                replay_buffer.mark_transitions_encoded(self.replay_marker)
            inserted += len(ids)

        # Alpha places initial-state mass at row zero. Ensure that transition
        # is present even when it was absent from the streamed chunks above.
        self._ensure_first(replay_buffer, encoder)
        return inserted > 0 or len(self.fifo) > 0

    def _ensure_first(self, replay_buffer, encoder: TransitionEncoder) -> None:
        if self.fifo.has_first or not hasattr(replay_buffer, "get_first_transition"):
            return
        try:
            first = replay_buffer.get_first_transition()
        except RuntimeError:
            return
        self.fifo.add(np.array([0], dtype=np.int64), encoder.encode_safely(first).tensors)

    def full(self, device: str) -> EncodedTransitions:
        return EncodedTransitions(self.fifo.all(device, include_first=True))


class RawReplayActorSource:
    """Build raw support/landmark batches when encoded FIFO is disabled."""

    def __init__(self, device: str):
        self.device = device

    def _batch(self, replay_sample) -> RawTransitions:
        obs, action, reward, _, next_obs = utils.to_torch(replay_sample[:5], self.device)
        return RawTransitions(obs, action, next_obs, reward.reshape(obs.shape[0], -1))

    def first(self, replay_buffer, fallback: RawTransitions) -> RawTransitions:
        if replay_buffer is not None and hasattr(replay_buffer, "get_first_transition"):
            return self._batch(replay_buffer.get_first_transition())
        return fallback.slice(slice(0, 1))

    def load_support(self, replay_iter, initial: RawTransitions, size: int, replay_buffer):
        batches, count = [initial], initial.size
        while count < size:
            batch = self._batch(next(replay_iter))
            batches.append(batch)
            count += batch.size
        support = RawTransitions.concatenate(batches, limit=size)
        return support.with_first(self.first(replay_buffer, support))

    def load_landmarks(self, replay_iter, support: RawTransitions, size: int, replay_buffer):
        first = self.first(replay_buffer, support)
        if size == 1:
            return first
        batches, count = [], 0
        while count < size - 1:
            batch = self._batch(next(replay_iter)).slice(slice(1, None))
            if batch.size:
                batches.append(batch)
                count += batch.size
        remainder = RawTransitions.concatenate(batches, limit=size - 1)
        return RawTransitions.concatenate([first, remainder], limit=size)
