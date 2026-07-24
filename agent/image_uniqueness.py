import hashlib
import math

import numpy as np
import torch


UINT64_MASK = (1 << 64) - 1


def last_frame_fingerprints(observations, image_channels):
    """Return signed int64 BLAKE2 fingerprints for final stacked image."""
    if torch.is_tensor(observations):
        observations = observations.detach().to("cpu").numpy()
    observations = np.asarray(observations)
    if observations.ndim < 4:
        raise ValueError(
            "pixel observations must have shape [batch, channels, height, width]"
        )
    image_channels = int(image_channels)
    if image_channels <= 0 or observations.shape[1] < image_channels:
        raise ValueError("image_channels must fit observation channel dimension")

    frames = np.ascontiguousarray(observations[:, -image_channels:, ...])
    hashes = np.empty(frames.shape[0], dtype=np.int64)
    for index, frame in enumerate(frames):
        digest = hashlib.blake2b(memoryview(frame), digest_size=8).digest()
        hashes[index] = int.from_bytes(digest, byteorder="little", signed=True)
    return hashes


class BoundedUniqueCounter:
    """Exact unique counter which permanently falls back to HyperLogLog."""

    def __init__(self, exact_limit=5_000_000, precision=14):
        if exact_limit <= 0:
            raise ValueError("exact_limit must be positive")
        if precision < 4 or precision > 20:
            raise ValueError("precision must be between 4 and 20")
        self.exact_limit = int(exact_limit)
        self.precision = int(precision)
        self._exact = set()
        self._registers = None
        self._last_count = 0

    @property
    def is_exact(self):
        return self._registers is None

    def _add_hll(self, value):
        value = int(value) & UINT64_MASK
        index = value >> (64 - self.precision)
        suffix_bits = 64 - self.precision
        suffix = value & ((1 << suffix_bits) - 1)
        rank = (
            suffix_bits + 1
            if suffix == 0
            else suffix_bits - suffix.bit_length() + 1
        )
        if rank > self._registers[index]:
            self._registers[index] = rank

    def _convert_to_hll(self):
        self._registers = np.zeros(1 << self.precision, dtype=np.uint8)
        for value in self._exact:
            self._add_hll(value)
        self._last_count = len(self._exact)
        self._exact = None

    def update(self, values):
        values = np.asarray(values, dtype=np.int64).reshape(-1)
        if self.is_exact:
            self._exact.update(map(int, values))
            if len(self._exact) > self.exact_limit:
                self._convert_to_hll()
        else:
            for value in values:
                self._add_hll(value)
        return self.count()

    def _hll_estimate(self):
        register_count = int(self._registers.size)
        alpha = 0.7213 / (1.0 + 1.079 / register_count)
        inverse_sum = np.exp2(-self._registers.astype(np.float64)).sum()
        estimate = alpha * register_count * register_count / inverse_sum
        zero_count = int(np.count_nonzero(self._registers == 0))
        if zero_count:
            linear_count = register_count * math.log(register_count / zero_count)
            if linear_count <= 2.5 * register_count:
                estimate = linear_count
        return int(round(estimate))

    def count(self):
        if self.is_exact:
            self._last_count = len(self._exact)
        else:
            self._last_count = max(self._last_count, self._hll_estimate())
        return self._last_count
