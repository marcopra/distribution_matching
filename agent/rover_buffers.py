from __future__ import annotations

import torch


class EncodedTransitionFIFO:
    """FIFO storage for actor-ready encoded transitions.

    The first transition is pinned separately so actor batches can always place
    it at index 0 without duplicating it in the random sample.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("encoded FIFO capacity must be positive")
        self.capacity = int(capacity)
        self._data = None
        self._ids = None
        self._first = None
        self._first_id = None

    def __len__(self):
        size = 0 if self._ids is None else int(self._ids.numel())
        return size + (1 if self._first is not None else 0)

    @property
    def has_first(self):
        return self._first is not None

    @property
    def data_count(self):
        return 0 if self._ids is None else int(self._ids.numel())

    @staticmethod
    def _index(encoded, index):
        return {key: value[index] for key, value in encoded.items()}

    @staticmethod
    def _cat(encoded_batches):
        keys = encoded_batches[0].keys()
        return {
            key: torch.cat([batch[key] for batch in encoded_batches], dim=0)
            for key in keys
        }

    def add(self, transition_ids, encoded):
        transition_ids = torch.as_tensor(transition_ids, dtype=torch.long, device='cpu')
        encoded = {
            key: value.detach().to('cpu')
            for key, value in encoded.items()
        }

        first_mask = transition_ids == 0
        if first_mask.any():
            first_idx = int(torch.nonzero(first_mask, as_tuple=False)[0].item())
            self._first = self._index(encoded, slice(first_idx, first_idx + 1))
            self._first_id = int(transition_ids[first_idx].item())

        if self._first_id is None:
            keep_mask = torch.ones_like(transition_ids, dtype=torch.bool)
        else:
            keep_mask = transition_ids != self._first_id
        if not keep_mask.any():
            return

        new_ids = transition_ids[keep_mask]
        new_data = self._index(encoded, keep_mask)
        if self._data is None:
            self._ids = new_ids
            self._data = new_data
        else:
            self._ids = torch.cat([self._ids, new_ids], dim=0)
            self._data = self._cat([self._data, new_data])

        overflow = int(self._ids.numel()) - max(0, self.capacity - (1 if self._first is not None else 0))
        if overflow > 0:
            self._ids = self._ids[overflow:]
            self._data = self._index(self._data, slice(overflow, None))

    def sample(self, size, device, include_first=True):
        if size <= 0:
            raise ValueError("sample size must be positive")
        if len(self) == 0:
            raise RuntimeError("Encoded actor FIFO is empty")

        batches = []
        remaining = int(size)
        if include_first and self._first is not None:
            batches.append(self._first)
            remaining -= 1

        data_size = 0 if self._ids is None else int(self._ids.numel())
        if remaining > 0 and data_size > 0:
            take = min(remaining, data_size)
            indices = torch.randperm(data_size)[:take]
            batches.append(self._index(self._data, indices))

        if not batches:
            raise RuntimeError("Encoded actor FIFO does not contain a first transition yet")

        sampled = batches[0] if len(batches) == 1 else self._cat(batches)
        return {
            key: value.to(device)
            for key, value in sampled.items()
        }

    def all(self, device, include_first=True):
        if len(self) == 0:
            raise RuntimeError("Encoded actor FIFO is empty")

        batches = []
        if include_first and self._first is not None:
            batches.append(self._first)
        if self._data is not None and self.data_count > 0:
            batches.append(self._data)

        if not batches:
            raise RuntimeError("Encoded actor FIFO does not contain a first transition yet")

        encoded = batches[0] if len(batches) == 1 else self._cat(batches)
        return {
            key: value.to(device)
            for key, value in encoded.items()
        }


