from __future__ import annotations

import contextlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from .base import Device, DeviceDriver, DeviceInfo, StreamHandle


def _import_pyrtlsdr() -> type[Any]:
    try:
        from rtlsdr import RtlSdr  # type: ignore

        return cast(type[Any], RtlSdr)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pyrtlsdr not available; install with pip install pyrtlsdr") from exc


@dataclass
class _RtlStream(StreamHandle):
    sdr: Any  # RtlSdr dynamically imported

    def read(self, num_samples: int) -> tuple[np.ndarray, bool]:
        data = self.sdr.read_samples(num_samples)
        # pyrtlsdr returns np.complex64
        return data.astype(np.complex64, copy=False), False

    def close(self) -> None:
        # No separate stream handle; close device elsewhere
        pass


@dataclass
class _RtlDevice(Device):
    info: DeviceInfo
    sdr: Any  # RtlSdr dynamically imported

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
    ) -> None:
        self.sdr.sample_rate = sample_rate
        self.sdr.center_freq = center_hz
        if ppm is not None:
            self.sdr.freq_correction = int(ppm)
        if gain is not None:
            # Manual gain
            with contextlib.suppress(Exception):
                self.sdr.set_gain_mode(True)
            self.sdr.gain = gain
        else:
            # Prefer automatic gain if available
            with contextlib.suppress(Exception):
                self.sdr.set_gain_mode(False)

    def start_stream(self) -> StreamHandle:
        return _RtlStream(self.sdr)

    def get_antenna(self) -> str | None:
        """RTL-SDR has no antenna selection."""
        return None

    def reconfigure_running(
        self,
        center_hz: float | None = None,
        gain: float | None = None,
        bandwidth: float | None = None,
        ppm: float | None = None,
    ) -> None:
        """Reconfigure device while running."""
        if center_hz is not None:
            self.sdr.center_freq = center_hz
        if gain is not None:
            self.sdr.gain = gain
        if ppm is not None:
            self.sdr.freq_correction = int(ppm)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.sdr.close()


class RtlDriver(DeviceDriver):
    name = "rtl"

    def __init__(self) -> None:
        self._RtlSdr = _import_pyrtlsdr()

    def enumerate(self) -> Iterable[DeviceInfo]:
        RtlSdr = self._RtlSdr
        try:
            count = RtlSdr.get_device_count()
        except Exception:
            count = 0
        infos = []
        for i in range(count):
            try:
                serials = RtlSdr.get_device_serial_addresses()
                label = f"RTL-SDR-{serials[i] if i < len(serials) else i}"
            except Exception:
                label = f"RTL-SDR-{i}"
            infos.append(
                DeviceInfo(
                    id=str(i),
                    driver="rtlsdr",
                    label=label,
                    freq_min_hz=24e6,
                    freq_max_hz=1.766e9,
                    sample_rates=(250_000, 1_024_000, 2_048_000),
                    gains=("LNA",),
                )
            )
        return infos

    def open(self, id_or_args: str | None = None) -> Device:
        RtlSdr = self._RtlSdr
        index = 0
        try:
            if id_or_args is not None and id_or_args.isdigit():
                index = int(id_or_args)
        except Exception:
            index = 0
        sdr = RtlSdr(index)
        info = DeviceInfo(
            id=str(index),
            driver="rtlsdr",
            label="RTL-SDR",
            freq_min_hz=24e6,
            freq_max_hz=1.766e9,
            sample_rates=(250_000, 1_024_000, 2_048_000),
            gains=("LNA",),
        )
        return _RtlDevice(info=info, sdr=sdr)
