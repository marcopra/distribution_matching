"""Kernel construction, bandwidth fitting, and Nyström landmark selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import utils


def _config_value(config: Any, name: str, default: Any) -> Any:
    """Read one value from a dict, Hydra DictConfig, or simple object."""
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


@dataclass(frozen=True)
class KernelSettings:
    """Small immutable view of public ``agent.kernel`` configuration."""

    name: str = "gaussian"
    bandwidth: Optional[float] = None
    subsampling_strategy: str = "random"
    candidate_multiplier: float = 5.0
    cholesky_tolerance: float = 1e-6
    cholesky_progress: bool = True

    @classmethod
    def from_config(cls, config: Any) -> "KernelSettings":
        settings = cls(
            name=str(_config_value(config, "name", "gaussian")).lower(),
            bandwidth=_config_value(config, "bandwidth", None),
            subsampling_strategy=str(
                _config_value(config, "subsampling_strategy", "random")
            ).lower(),
            candidate_multiplier=float(
                _config_value(config, "candidate_multiplier", 5.0)
            ),
            cholesky_tolerance=float(
                _config_value(config, "cholesky_tolerance", 1e-6)
            ),
            cholesky_progress=bool(
                _config_value(config, "cholesky_progress", True)
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.name not in {"inner_product", "gaussian", "laplacian", "dirac"}:
            raise ValueError(f"Unsupported kernel: {self.name}")
        if self.subsampling_strategy not in {"random", "pivoted_cholesky"}:
            raise ValueError(
                "kernel.subsampling_strategy must be random or pivoted_cholesky"
            )
        if self.subsampling_strategy == "pivoted_cholesky" and self.name not in {
            "inner_product",
            "gaussian",
        }:
            raise ValueError(
                "pivoted_cholesky supports inner_product and gaussian kernels"
            )
        if self.name == "gaussian" and self.bandwidth is None:
            raise ValueError("kernel.bandwidth is required for gaussian kernels")
        if self.bandwidth is not None and float(self.bandwidth) <= 0:
            raise ValueError("kernel.bandwidth must be positive")
        if self.candidate_multiplier < 1:
            raise ValueError("kernel.candidate_multiplier must be at least 1")
        if self.cholesky_tolerance < 0:
            raise ValueError("kernel.cholesky_tolerance must be non-negative")


class KernelManager:
    """Own kernel state shared by policy evaluation and landmark selection."""

    def __init__(self, config: Any):
        self.settings = KernelSettings.from_config(config)
        self.kernel_fn = utils.build_kernel_fn(
            self.settings.name,
            bandwidth=self.settings.bandwidth,
        )

    @property
    def bandwidth(self) -> Optional[float]:
        return getattr(self.kernel_fn, "bandwidth", None)

    def attach_matcher(self, matcher: Any) -> None:
        """Make matcher and policy use same state-kernel object."""
        matcher.state_kernel_fn = self.kernel_fn
        matcher.kernel_fn = utils.build_kernel_fn(
            self.settings.name,
            bandwidth=self.bandwidth,
        )

    def select(self, fifo: Any, size: int, device: str, include_first: bool = True):
        """Select landmarks with kernel-aware random or pivoted-Cholesky logic."""
        sample = fifo.sample_by_strategy(
            int(size),
            device,
            strategy=self.settings.subsampling_strategy,
            include_first=include_first,
            candidate_multiplier=self.settings.candidate_multiplier,
            cholesky_tolerance=self.settings.cholesky_tolerance,
            kernel_type=self.settings.name,
            kernel_bandwidth=self.settings.bandwidth,
            cholesky_progress=self.settings.cholesky_progress,
        )
        return sample

    def status(self) -> str:
        bandwidth = self.bandwidth
        if bandwidth is None:
            return f"kernel={self.settings.name}"
        return f"kernel={self.settings.name}, bandwidth={bandwidth:.6g}"
