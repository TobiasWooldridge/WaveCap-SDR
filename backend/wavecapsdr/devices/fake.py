from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .base import Device, DeviceDriver, DeviceInfo, StreamHandle


@dataclass
class _FakeStream(StreamHandle):
    sample_rate: int
    freq: float
    running: bool = True

    def read(self, num_samples: int) -> tuple[np.ndarray, bool]:
        if not self.running:
            return np.empty(0, dtype=np.complex64), False
        t = np.arange(num_samples, dtype=np.float32) / float(self.sample_rate)
        # Simple complex exponential at freq
        sig = np.exp(1j * 2.0 * np.pi * self.freq * t).astype(np.complex64)
        return sig, False

    def close(self) -> None:
        self.running = False


@dataclass
class _FakeDevice(Device):
    info: DeviceInfo
    sample_rate: int = 1_000_000
    center_hz: float = 100_000_000.0

    def configure(
        self,
        center_hz: float,
        sample_rate: int,
        gain: Optional[float] = None,
        bandwidth: Optional[float] = None,
        ppm: Optional[float] = None,
    ) -> None:
        self.center_hz = center_hz
        self.sample_rate = sample_rate

    def start_stream(self) -> StreamHandle:
        return _FakeStream(sample_rate=self.sample_rate, freq=5_000.0)

    def close(self) -> None:
        pass


class FakeDriver(DeviceDriver):
    name = "fake"

    def enumerate(self) -> Iterable[DeviceInfo]:
        return [
            DeviceInfo(
                id="fake0",
                driver="fake",
                label="FakeSDR-0",
                freq_min_hz=50.0,
                freq_max_hz=6e9,
                sample_rates=(250_000, 1_000_000, 2_000_000),
                gains=("LNA", "VGA"),
            )
        ]

    def open(self, id_or_args: Optional[str] = None) -> Device:
        return _FakeDevice(
            info=DeviceInfo(
                id=id_or_args or "fake0",
                driver="fake",
                label="FakeSDR-0",
                freq_min_hz=50.0,
                freq_max_hz=6e9,
                sample_rates=(250_000, 1_000_000, 2_000_000),
                gains=("LNA", "VGA"),
            )
        )
