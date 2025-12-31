from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from wavecapsdr.typing import NDArrayComplex


@dataclass(frozen=True)
class DeviceInfo:
    id: str
    driver: str
    label: str
    freq_min_hz: float
    freq_max_hz: float
    sample_rates: tuple[int, ...]
    gains: tuple[str, ...]
    gain_min: float | None = None
    gain_max: float | None = None
    bandwidth_min: float | None = None
    bandwidth_max: float | None = None
    ppm_min: float | None = None
    ppm_max: float | None = None
    antennas: tuple[str, ...] = ()


class StreamHandle(Protocol):
    def read(self, num_samples: int) -> tuple[NDArrayComplex, bool]:  # (samples, overrun)
        ...

    def close(self) -> None: ...


class Device(Protocol):
    info: DeviceInfo

    def configure(
        self,
        center_hz: float,
        sample_rate: int,
        gain: float | None = None,
        bandwidth: float | None = None,
        ppm: float | None = None,
        antenna: str | None = None,
        device_settings: dict[str, Any] | None = None,
        element_gains: dict[str, float] | None = None,
        agc_enabled: bool = False,
        stream_format: str | None = None,
        dc_offset_auto: bool = True,
        iq_balance_auto: bool = True,
    ) -> None: ...

    def start_stream(self) -> StreamHandle: ...

    def get_antenna(self) -> str | None:
        """Return the currently configured antenna, if any."""
        ...

    def reconfigure_running(
        self,
        center_hz: float | None = None,
        gain: float | None = None,
        bandwidth: float | None = None,
        ppm: float | None = None,
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

    def close(self) -> None: ...


class DeviceDriver(Protocol):
    name: str

    def enumerate(self) -> Iterable[DeviceInfo]: ...

    def open(self, id_or_args: str | None = None) -> Device: ...
