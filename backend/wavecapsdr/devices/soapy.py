from __future__ import annotations

from dataclasses import dataclass
import multiprocessing
import signal
import time
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
        buff = np.empty(num_samples, dtype=np.complex64)
        # readStream returns StreamResult with ret, flags, timeNs attributes
        # ret = number of samples read (or negative error code)
        # flags = stream flags (bit 1 = overflow)
        # Use integer constants since not all SoapySDR bindings expose named constants
        SOAPY_SDR_READ_FLAG_OVERFLOW = (1 << 1)
        sr = self.sdr.readStream(self.stream, [buff.view(np.float32)], num_samples, flags=0)
        # Handle both tuple and StreamResult object
        if hasattr(sr, 'ret'):
            # StreamResult object
            ret = sr.ret
            flags = sr.flags if hasattr(sr, 'flags') else 0
        elif isinstance(sr, tuple):
            (ret, flags, _timeNs, _) = sr + (0, 0, 0, 0)  # type: ignore[assignment]
        else:
            # Assume it's just the return count
            ret = sr
            flags = 0
        if ret < 0:
            # Negative indicates error; represent as empty with overrun flag
            return np.empty(0, dtype=np.complex64), True
        return buff[:ret], bool(flags & SOAPY_SDR_READ_FLAG_OVERFLOW)

    def close(self) -> None:
        self.sdr.deactivateStream(self.stream)
        self.sdr.closeStream(self.stream)


@dataclass
class _SoapyDevice(Device):
    info: DeviceInfo
    sdr: "SoapySDR.Device"
    _antenna: Optional[str] = None

    def configure(
        self,
        center_hz: float,
        sample_rate: int,
        gain: Optional[float] = None,
        bandwidth: Optional[float] = None,
        ppm: Optional[float] = None,
        antenna: Optional[str] = None,
    ) -> None:
        import SoapySDR  # type: ignore

        self.sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, float(sample_rate))
        self.sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, center_hz)
        if gain is not None:
            # Manual gain
            try:
                try:
                    self.sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, False)  # manual
                except Exception:
                    pass
                self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, gain)
            except Exception:
                pass
        else:
            # Prefer automatic gain control when supported
            try:
                self.sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, True)  # auto
            except Exception:
                # Not all drivers support automatic gain; skip
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
        if antenna is not None:
            self._antenna = antenna

    def start_stream(self) -> StreamHandle:
        import SoapySDR  # type: ignore

        # Set antenna: use configured antenna if specified, otherwise use first available
        available_antennas = self.sdr.listAntennas(SoapySDR.SOAPY_SDR_RX, 0)
        if self._antenna is not None:
            antenna = self._antenna
        else:
            antenna = available_antennas[0] if available_antennas else "RX"
        self.sdr.setAntenna(SoapySDR.SOAPY_SDR_RX, 0, antenna)
        # Update _antenna to reflect what was actually set
        self._antenna = antenna

        stream = self.sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32, [0])
        self.sdr.activateStream(stream)
        return _SoapyStream(self.sdr, stream)

    def get_antenna(self) -> Optional[str]:
        """Return the currently configured antenna, if any."""
        return self._antenna

    def close(self) -> None:
        self.sdr = None  # type: ignore[assignment]


class SoapyDriver(DeviceDriver):
    name = "soapy"

    def __init__(self, cfg: DeviceConfig):
        self._cfg = cfg
        self._SoapySDR = _import_soapy()
        self._enumerate_cache: Optional[tuple[float, list[DeviceInfo]]] = None
        self._enumerate_cache_ttl = 30.0  # seconds

    def _enumerate_driver(self, driver_name: str, timeout: float = 5.0) -> list[DeviceInfo]:
        """Enumerate devices for a specific driver with timeout protection."""
        def enumerate_worker(driver_name: str, queue: multiprocessing.Queue) -> None:
            """Worker function to enumerate in subprocess."""
            try:
                import SoapySDR  # type: ignore
                # Unload and reload modules to reset any stuck state
                try:
                    SoapySDR.Device.unmake()  # type: ignore[attr-defined]
                except Exception:
                    pass
                # Only enumerate for this specific driver
                results = []
                for args in SoapySDR.Device.enumerate(f"driver={driver_name}"):  # type: ignore[attr-defined]
                    driver = str(args["driver"]) if "driver" in args else "unknown"
                    label = str(args["label"]) if "label" in args else "SDR"
                    # Build a canonical args string
                    try:
                        items = []
                        for k in sorted(args.keys()):
                            v = args[k] if k in args else None
                            if v is None:
                                continue
                            items.append(f"{k}={v}")
                        id_ = ",".join(items) if items else driver
                    except Exception:
                        id_ = driver

                    freq_min = float(args["rfmin"]) if "rfmin" in args else 1e4
                    freq_max = float(args["rfmax"]) if "rfmax" in args else 6e9
                    sample_rates = (250_000, 1_000_000, 2_000_000, 2_400_000)
                    gains = ("LNA", "VGA")
                    results.append({
                        "id": id_,
                        "driver": driver,
                        "label": label,
                        "freq_min_hz": freq_min,
                        "freq_max_hz": freq_max,
                        "sample_rates": sample_rates,
                        "gains": gains,
                    })
                queue.put(("success", results))
            except Exception as e:
                queue.put(("error", str(e)))

        # Use multiprocessing to isolate and timeout the enumeration
        queue: multiprocessing.Queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=enumerate_worker, args=(driver_name, queue))
        process.start()
        process.join(timeout=timeout)

        results = []
        if process.is_alive():
            # Timeout - kill the process
            process.terminate()
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join()
            print(f"Warning: Enumeration for driver '{driver_name}' timed out after {timeout}s", flush=True)
        else:
            # Check results
            if not queue.empty():
                status, data = queue.get()
                if status == "success":
                    results = [DeviceInfo(**d) for d in data]
                else:
                    print(f"Warning: Enumeration for driver '{driver_name}' failed: {data}", flush=True)

        return results

    def enumerate(self) -> Iterable[DeviceInfo]:
        # Check cache first
        now = time.time()
        if self._enumerate_cache is not None:
            cache_time, cached_results = self._enumerate_cache
            if now - cache_time < self._enumerate_cache_ttl:
                return cached_results

        # Enumerate specific drivers we support, with timeout protection
        # This prevents hangs from problematic drivers (audio, uhd, etc.)
        results = []
        for driver_name in ["rtlsdr", "sdrplay"]:
            try:
                driver_results = self._enumerate_driver(driver_name, timeout=5.0)
                results.extend(driver_results)
            except Exception as e:
                print(f"Warning: Failed to enumerate driver '{driver_name}': {e}", flush=True)

        # Cache the results
        self._enumerate_cache = (now, results)
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
