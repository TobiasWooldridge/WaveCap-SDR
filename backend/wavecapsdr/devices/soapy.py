from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

from ..config import DeviceConfig
from .base import Device, DeviceDriver, DeviceInfo, StreamHandle


def _import_soapy():
    try:
        import SoapySDR  # type: ignore

        return SoapySDR
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "SoapySDR Python bindings not available. Install extra 'soapy' or ensure SoapySDR is present."
        ) from exc


@dataclass
class _SoapyStream(StreamHandle):
    sdr: "SoapySDR.Device"
    stream: "SoapySDR.Stream"

    def read(self, num_samples: int) -> tuple[np.ndarray, bool]:
        import SoapySDR  # type: ignore

        buff = np.empty(num_samples, dtype=np.complex64)
        flags = SoapySDR.SDR_READ_FLAG_NONE
        # Use a tuple for single channel 0
        sr = self.sdr.readStream(self.stream, [buff.view(np.float32)], num_samples)
        if isinstance(sr, tuple):
            (ret, flags, _timeNs, _) = sr + (0, 0, 0, 0)  # type: ignore[assignment]
        else:  # pragma: no cover - depends on binding variant
            ret, flags, _timeNs = sr, 0, 0
        if ret < 0:
            # Negative indicates error; represent as empty with overrun flag
            return np.empty(0, dtype=np.complex64), True
        return buff[:ret], bool(flags & SoapySDR.SDR_READ_FLAG_OVERFLOW)

    def close(self) -> None:
        self.sdr.deactivateStream(self.stream)
        self.sdr.closeStream(self.stream)


@dataclass
class _SoapyDevice(Device):
    info: DeviceInfo
    sdr: "SoapySDR.Device"

    def configure(
        self,
        center_hz: float,
        sample_rate: int,
        gain: Optional[float] = None,
        bandwidth: Optional[float] = None,
        ppm: Optional[float] = None,
    ) -> None:
        import SoapySDR  # type: ignore

        self.sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, float(sample_rate))
        self.sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, center_hz)
        if gain is not None:
            try:
                self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, gain)
            except Exception:
                pass
        if bandwidth is not None:
            try:
                self.sdr.setBandwidth(SoapySDR.SOAPY_SDR_RX, 0, bandwidth)
            except Exception:
                pass
        if ppm is not None:
            try:
                self.sdr.setFrequencyCorrection(SoapySDR.SOAPY_SDR_RX, 0, ppm)
            except Exception:
                pass

    def start_stream(self) -> StreamHandle:
        import SoapySDR  # type: ignore

        self.sdr.setAntenna(SoapySDR.SOAPY_SDR_RX, 0, self.sdr.listAntennas(SoapySDR.SOAPY_SDR_RX, 0)[0])
        stream = self.sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32, [0])
        self.sdr.activateStream(stream)
        return _SoapyStream(self.sdr, stream)

    def close(self) -> None:
        self.sdr = None  # type: ignore[assignment]


class SoapyDriver(DeviceDriver):
    name = "soapy"

    def __init__(self, cfg: DeviceConfig):
        self._cfg = cfg
        self._SoapySDR = _import_soapy()

    def enumerate(self) -> Iterable[DeviceInfo]:
        SoapySDR = self._SoapySDR
        results = []
        for args in SoapySDR.Device.enumerate():  # type: ignore[attr-defined]
            # args behaves like a dict-like
            driver = str(args.get("driver", "unknown"))
            label = str(args.get("label", "SDR"))
            serial = str(args.get("serial", ""))
            id_ = serial or label

            # Capability probing is device-specific; provide broad ranges if unavailable
            freq_min = float(args.get("rfmin", 1e4))
            freq_max = float(args.get("rfmax", 6e9))
            # Sample rates often not listed here; expose common rates
            sample_rates: tuple[int, ...] = (250_000, 1_000_000, 2_000_000, 2_400_000)
            gains: tuple[str, ...] = ("LNA", "VGA")
            results.append(
                DeviceInfo(
                    id=id_,
                    driver=driver,
                    label=label,
                    freq_min_hz=freq_min,
                    freq_max_hz=freq_max,
                    sample_rates=sample_rates,
                    gains=gains,
                )
            )
        return results

    def open(self, id_or_args: Optional[str] = None) -> Device:
        SoapySDR = self._SoapySDR
        args = id_or_args or self._cfg.device_args or ""
        sdr = SoapySDR.Device(args)
        # Build DeviceInfo from the live device
        try:
            rx_ranges = sdr.getFrequencyRange(SoapySDR.SOAPY_SDR_RX, 0)
            freq_min = float(rx_ranges[0].minimum()) if rx_ranges else 1e4
            freq_max = float(rx_ranges[0].maximum()) if rx_ranges else 6e9
        except Exception:
            freq_min, freq_max = 1e4, 6e9
        try:
            srs = tuple(int(r) for r in sdr.listSampleRates(SoapySDR.SOAPY_SDR_RX, 0))
            sample_rates = srs or (250_000, 1_000_000, 2_000_000)
        except Exception:
            sample_rates = (250_000, 1_000_000, 2_000_000)
        info = DeviceInfo(
            id=str(id_or_args or "device0"),
            driver=str(sdr.getDriverKey()),
            label=str(sdr.getHardwareKey()),
            freq_min_hz=freq_min,
            freq_max_hz=freq_max,
            sample_rates=sample_rates,
            gains=tuple(g.name for g in []),
        )
        return _SoapyDevice(info=info, sdr=sdr)
