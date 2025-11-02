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
    gain_min: Optional[float] = None
    gain_max: Optional[float] = None
    bandwidth_min: Optional[float] = None
    bandwidth_max: Optional[float] = None
    ppm_min: Optional[float] = None
    ppm_max: Optional[float] = None
    antennas: tuple[str, ...] = ()


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
        antenna: Optional[str] = None,
    ) -> None:
        ...

    def start_stream(self) -> StreamHandle:
        ...

    def get_antenna(self) -> Optional[str]:
        """Return the currently configured antenna, if any."""
        ...

    def reconfigure_running(
        self,
        center_hz: Optional[float] = None,
        gain: Optional[float] = None,
        bandwidth: Optional[float] = None,
        ppm: Optional[float] = None,
    ) -> None:
        """Reconfigure device while stream is running (hot reconfiguration).

        This method allows updating certain parameters without stopping/restarting
        the stream. Not all parameters can be changed without restart (sample_rate
        and antenna typically require restart).

        Args:
            center_hz: New center frequency (if provided)
            gain: New gain setting (if provided)
            bandwidth: New bandwidth (if provided)
            ppm: New PPM correction (if provided)
        """
        ...

    def close(self) -> None:
        ...


class DeviceDriver(Protocol):
    name: str

    def enumerate(self) -> Iterable[DeviceInfo]:
        ...

    def open(self, id_or_args: Optional[str] = None) -> Device:
        ...
