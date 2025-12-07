from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Type, cast

import numpy as np

from .base import Device, DeviceDriver, DeviceInfo, StreamHandle


def _import_pyrtlsdr() -> Type[Any]:
    try:
        from rtlsdr import RtlSdr  # type: ignore

        return cast(Type[Any], RtlSdr)
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
        gain: Optional[float] = None,
        bandwidth: Optional[float] = None,
        ppm: Optional[float] = None,
        antenna: Optional[str] = None,
        device_settings: Optional[dict[str, Any]] = None,
        element_gains: Optional[dict[str, float]] = None,
        stream_format: Optional[str] = None,
        dc_offset_auto: bool = True,
        iq_balance_auto: bool = True,
    ) -> None:
        self.sdr.sample_rate = sample_rate
        self.sdr.center_freq = center_hz
        if ppm is not None:
            self.sdr.freq_correction = int(ppm)
        if gain is not None:
            # Manual gain
            try:
                self.sdr.set_gain_mode(True)
            except Exception:
                pass
            self.sdr.gain = gain
        else:
            # Prefer automatic gain if available
            try:
                self.sdr.set_gain_mode(False)
            except Exception:
                pass

    def start_stream(self) -> StreamHandle:
        return _RtlStream(self.sdr)

    def get_antenna(self) -> Optional[str]:
        """RTL-SDR has no antenna selection."""
        return None

    def reconfigure_running(
        self,
        center_hz: Optional[float] = None,
        gain: Optional[float] = None,
        bandwidth: Optional[float] = None,
        ppm: Optional[float] = None,
    ) -> None:
        """Reconfigure device while running."""
        if center_hz is not None:
            self.sdr.center_freq = center_hz
        if gain is not None:
            self.sdr.gain = gain
        if ppm is not None:
            self.sdr.freq_correction = int(ppm)

    def close(self) -> None:
        try:
            self.sdr.close()
        except Exception:
            pass


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

    def open(self, id_or_args: Optional[str] = None) -> Device:
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
