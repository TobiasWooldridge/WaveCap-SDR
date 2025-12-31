"""Typed interfaces for trunking encoders.

Provides protocol definitions for control-channel and traffic-channel encoders
along with lightweight frame containers that validate dibit inputs. Encoders
map dibits to baseband-friendly float symbols so higher-level tests can plug in
future protocol backends without changing call sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from wavecapsdr.trunking.config import TrunkingProtocol

DibitArray = NDArray[np.uint8]


def _normalize_dibits(values: Sequence[int] | DibitArray) -> DibitArray:
    """Normalize dibit input to a 1D uint8 NumPy array with valid values."""
    dibits = np.asarray(values, dtype=np.uint8)
    if dibits.ndim != 1:
        raise ValueError(f"Dibit array must be 1-dimensional, got shape {dibits.shape}")
    if dibits.size == 0:
        raise ValueError("Dibit array cannot be empty")
    if np.any(dibits > 3):
        raise ValueError("Dibits must be in range [0, 3]")
    return dibits


@dataclass(frozen=True)
class ControlChannelFrame:
    """Container for control-channel dibits before modulation."""
    dibits: DibitArray
    symbol_rate: int

    def __init__(self, dibits: Sequence[int] | DibitArray, symbol_rate: int) -> None:
        if symbol_rate <= 0:
            raise ValueError("symbol_rate must be positive")
        object.__setattr__(self, "dibits", _normalize_dibits(dibits))
        object.__setattr__(self, "symbol_rate", int(symbol_rate))


@dataclass(frozen=True)
class TrafficChannelFrame:
    """Container for traffic-channel dibits before modulation."""
    dibits: DibitArray
    symbol_rate: int

    def __init__(self, dibits: Sequence[int] | DibitArray, symbol_rate: int) -> None:
        if symbol_rate <= 0:
            raise ValueError("symbol_rate must be positive")
        object.__setattr__(self, "dibits", _normalize_dibits(dibits))
        object.__setattr__(self, "symbol_rate", int(symbol_rate))


@runtime_checkable
class ControlChannelEncoder(Protocol):
    """Protocol for control-channel encoders."""
    protocol: TrunkingProtocol
    symbol_rate: int

    def encode(self, frame: ControlChannelFrame, sample_rate: int) -> NDArray[np.float32]:
        """Encode a control-channel frame to baseband symbols."""
        ...


@runtime_checkable
class TrafficChannelEncoder(Protocol):
    """Protocol for traffic-channel encoders."""
    protocol: TrunkingProtocol
    symbol_rate: int

    def encode(self, frame: TrafficChannelFrame, sample_rate: int) -> NDArray[np.float32]:
        """Encode a traffic-channel frame to baseband symbols."""
        ...
