"""P25 trunking encoders."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from wavecapsdr.encoders.trunking.base import (
    ControlChannelEncoder,
    ControlChannelFrame,
    TrafficChannelEncoder,
    TrafficChannelFrame,
)
from wavecapsdr.trunking.config import TrunkingProtocol

P25_SYMBOLS = np.array([1.0, 3.0, -1.0, -3.0], dtype=np.float32)


def _samples_per_symbol(sample_rate: int, symbol_rate: int) -> int:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if sample_rate % symbol_rate != 0:
        raise ValueError(f"sample_rate {sample_rate} is not divisible by symbol_rate {symbol_rate}")
    return sample_rate // symbol_rate


class P25ControlChannelEncoder(ControlChannelEncoder):
    """Control-channel encoder for P25 Phase I/II systems."""

    def __init__(self, protocol: TrunkingProtocol) -> None:
        if protocol not in (TrunkingProtocol.P25_PHASE1, TrunkingProtocol.P25_PHASE2):
            raise ValueError(f"P25 encoder does not support protocol {protocol}")
        self.protocol = protocol
        self.symbol_rate = 4800

    def encode(self, frame: ControlChannelFrame, sample_rate: int) -> NDArray[np.float32]:
        if frame.symbol_rate != self.symbol_rate:
            raise ValueError(
                f"Control frame symbol_rate {frame.symbol_rate} does not match encoder "
                f"symbol_rate {self.symbol_rate}"
            )
        samples_per_symbol = _samples_per_symbol(sample_rate, frame.symbol_rate)
        symbols = P25_SYMBOLS[frame.dibits]
        return np.repeat(symbols, samples_per_symbol)


class P25TrafficChannelEncoder(TrafficChannelEncoder):
    """Traffic-channel encoder for P25 Phase I/II systems."""

    def __init__(self, protocol: TrunkingProtocol) -> None:
        if protocol not in (TrunkingProtocol.P25_PHASE1, TrunkingProtocol.P25_PHASE2):
            raise ValueError(f"P25 encoder does not support protocol {protocol}")
        self.protocol = protocol
        # Phase II uses 6000 symbols/s; Phase I uses 4800 symbols/s.
        self.symbol_rate = 6000 if protocol == TrunkingProtocol.P25_PHASE2 else 4800

    def encode(self, frame: TrafficChannelFrame, sample_rate: int) -> NDArray[np.float32]:
        if frame.symbol_rate != self.symbol_rate:
            raise ValueError(
                f"Traffic frame symbol_rate {frame.symbol_rate} does not match encoder "
                f"symbol_rate {self.symbol_rate}"
            )
        samples_per_symbol = _samples_per_symbol(sample_rate, frame.symbol_rate)
        symbols = P25_SYMBOLS[frame.dibits]
        return np.repeat(symbols, samples_per_symbol)
