"""Data transfer objects shared by Rover algorithm and debug components."""

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass(frozen=True)
class RawActorUpdateData:
    full: tuple
    source: str = "unknown"
    subsample: Optional[tuple] = None


@dataclass(frozen=True)
class EncodedActorUpdateData:
    full: Dict[str, torch.Tensor]
    rewards: torch.Tensor
    source: str = "unknown"
    subsample: Optional[Dict[str, torch.Tensor]] = None
    subsample_rewards: Optional[torch.Tensor] = None
