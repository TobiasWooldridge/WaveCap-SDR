from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Protocol
import numpy as np


@dataclass(frozen=True)
class DeviceInfo:
    id: str
    driver: str
    label: str
    freq_min_hz: float
    freq_max_hz: float
    sample_rates: tuple[int, ...]
    gains: tuple[str, ...]


class StreamHandle(Protocol):
    def read(self, num_samples: int) -> tuple[np.ndarray, bool]:  # (samples, overrun)
        ...

    def close(self) -> None:
        ...


class Device(Protocol):
    info: DeviceInfo

    def configure(
        self,
        center_hz: float,
        sample_rate: int,
        gain: Optional[float] = None,
        bandwidth: Optional[float] = None,
        ppm: Optional[float] = None,
    ) -> None:
        ...

    def start_stream(self) -> StreamHandle:
        ...

    def close(self) -> None:
        ...


class DeviceDriver(Protocol):
    name: str

    def enumerate(self) -> Iterable[DeviceInfo]:
        ...

    def open(self, id_or_args: Optional[str] = None) -> Device:
        ...
