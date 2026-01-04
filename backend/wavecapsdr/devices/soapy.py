from __future__ import annotations

import contextlib
import functools
import multiprocessing
import queue as queue_module
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from multiprocessing import Queue as MPQueue
from typing import Any, Callable, Iterator, TypeVar, cast

import numpy as np
from wavecapsdr.typing import NDArrayComplex

from ..config import DeviceConfig

# Re-enabled for proactive recovery during enumeration (not during streaming)
from ..sdrplay_recovery import attempt_recovery, get_recovery
from .base import Device, DeviceDriver, DeviceInfo, StreamHandle
from .sdrplay_lock import acquire_sdrplay_lock, release_sdrplay_lock

# Global lock for SDRplay device operations to prevent "deletion in-progress" errors
# The SDRplay API cannot handle concurrent open/close operations
_sdrplay_device_lock = threading.Lock()

# Cooldown period after closing an SDRplay device before opening another
# The SDRplay API needs time to fully release the device
_SDRPLAY_CLOSE_COOLDOWN = 1.0  # seconds
_sdrplay_last_close_time: float = 0.0

# SDRplay health tracking for proactive recovery
_sdrplay_last_enumeration_success: float = 0.0
_sdrplay_consecutive_timeouts: int = 0
_SDRPLAY_MAX_TIMEOUTS_BEFORE_RECOVERY = 2  # Attempt recovery after 2 consecutive timeouts

# Health cache for pre-flight checks
_SDRPLAY_HEALTH_CACHE_TTL = 5.0  # seconds - how long to trust cached health status
_sdrplay_last_health_check: float = 0.0
_sdrplay_health_status: bool = False  # True = healthy, False = unknown/unhealthy

# Track active SDRplay captures to prevent recovery while streaming
# Recovery during active streaming would kill the captures
_sdrplay_active_captures: int = 0
_sdrplay_active_captures_lock = threading.Lock()

# Track expected device count for stability monitoring
_sdrplay_expected_device_count: int | None = None


@contextlib.contextmanager
def _sdrplay_operation_lock(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    with _sdrplay_device_lock:
        acquire_sdrplay_lock()
        try:
            yield
        finally:
            release_sdrplay_lock()


def increment_sdrplay_active_captures() -> None:
    """Increment active SDRplay capture count (called when capture starts)."""
    global _sdrplay_active_captures
    with _sdrplay_active_captures_lock:
        _sdrplay_active_captures += 1
        print(f"[SOAPY] SDRplay active captures: {_sdrplay_active_captures}", flush=True)


def decrement_sdrplay_active_captures() -> None:
    """Decrement active SDRplay capture count (called when capture stops)."""
    global _sdrplay_active_captures
    with _sdrplay_active_captures_lock:
        _sdrplay_active_captures = max(0, _sdrplay_active_captures - 1)
        print(f"[SOAPY] SDRplay active captures: {_sdrplay_active_captures}", flush=True)


def get_sdrplay_active_captures() -> int:
    """Get count of active SDRplay captures."""
    with _sdrplay_active_captures_lock:
        return _sdrplay_active_captures


class SDRplayServiceError(Exception):
    """SDRplay API service is unresponsive or device open failed.

    This exception indicates the SDRplay service needs to be restarted.
    Use: POST /api/v1/devices/sdrplay/restart-service
    """

    pass


def get_sdrplay_health_status() -> dict[str, Any]:
    """Get current SDRplay service health status for monitoring.

    Returns:
        Dictionary with health metrics:
        - is_healthy: True if last enumeration succeeded
        - consecutive_timeouts: Number of consecutive enumeration failures
        - last_success_timestamp: Unix timestamp of last successful enumeration
        - seconds_since_success: Seconds since last successful enumeration (or None if never)
    """
    now = time.time()
    return {
        "is_healthy": _sdrplay_consecutive_timeouts == 0,
        "consecutive_timeouts": _sdrplay_consecutive_timeouts,
        "last_success_timestamp": _sdrplay_last_enumeration_success
        if _sdrplay_last_enumeration_success > 0
        else None,
        "seconds_since_success": (now - _sdrplay_last_enumeration_success)
        if _sdrplay_last_enumeration_success > 0
        else None,
        "recovery_threshold": _SDRPLAY_MAX_TIMEOUTS_BEFORE_RECOVERY,
    }


def reset_sdrplay_health_counters() -> None:
    """Reset SDRplay health counters after manual intervention."""
    global _sdrplay_consecutive_timeouts, _sdrplay_last_enumeration_success
    _sdrplay_consecutive_timeouts = 0
    _sdrplay_last_enumeration_success = time.time()


# Global reference to driver instance for cache invalidation from other modules
_soapy_driver_instance: SoapyDriver | None = None


def invalidate_sdrplay_caches() -> None:
    """Clear all SDRplay enumeration caches after failure or recovery.

    Call this from other modules (e.g., capture.py) when SDRplay device
    operations fail to ensure stale data isn't served.
    """
    global _sdrplay_health_status, _sdrplay_last_health_check, _sdrplay_expected_device_count

    print("[SOAPY] Invalidating SDRplay caches", flush=True)
    _sdrplay_health_status = False
    _sdrplay_last_health_check = 0.0
    _sdrplay_expected_device_count = None

    # Also invalidate driver instance cache if available
    if _soapy_driver_instance is not None:
        _soapy_driver_instance.invalidate_cache()


F = TypeVar("F", bound=Callable[..., Any])


def _device_open_worker(device_args: str, result_queue: MPQueue[Any]) -> None:
    """Worker function to open SoapySDR device in subprocess.

    This isolates the blocking SoapySDR.Device() call so it can be
    terminated if it hangs (unlike threads which cannot be killed).

    The subprocess opens the device, queries its capabilities, then closes it.
    If successful, the main process knows it's safe to open the device.
    """
    try:
        import SoapySDR

        # Open the device with timeout (10 seconds = 10,000,000 microseconds)
        # Uses TobiasWooldridge/SoapySDR fork's timeout API
        DEVICE_OPEN_TIMEOUT_US = 10_000_000
        try:
            sdr = SoapySDR.Device.make(device_args, DEVICE_OPEN_TIMEOUT_US)
        except AttributeError:
            # Fallback for SoapySDR without timeout support
            sdr = SoapySDR.Device(device_args)

        # Query device info (these can also hang)
        driver = str(sdr.getDriverKey())
        hardware = str(sdr.getHardwareKey())

        # Get frequency range
        try:
            rx_ranges = sdr.getFrequencyRange(SoapySDR.SOAPY_SDR_RX, 0)
            freq_min = float(rx_ranges[0].minimum()) if rx_ranges else 1e4
            freq_max = float(rx_ranges[0].maximum()) if rx_ranges else 6e9
        except Exception:
            freq_min, freq_max = 1e4, 6e9

        # Get sample rates
        try:
            sample_rates = list(sdr.listSampleRates(SoapySDR.SOAPY_SDR_RX, 0))
        except Exception:
            sample_rates = []

        # Get antennas
        try:
            antennas = list(sdr.listAntennas(SoapySDR.SOAPY_SDR_RX, 0))
        except Exception:
            antennas = []

        # Close device in subprocess (main process will reopen)
        with contextlib.suppress(Exception):
            SoapySDR.Device.unmake(sdr)

        result_queue.put(
            (
                "success",
                {
                    "driver": driver,
                    "hardware": hardware,
                    "freq_min": freq_min,
                    "freq_max": freq_max,
                    "sample_rates": sample_rates,
                    "antennas": antennas,
                },
            )
        )
    except Exception as e:
        result_queue.put(("error", str(e)))


class TimeoutError(Exception):
    """Raised when an operation times out."""

    pass


def with_timeout(timeout_seconds: float) -> Callable[[F], F]:
    """Decorator to add timeout protection to device operations.

    Args:
        timeout_seconds: Maximum time to wait for operation to complete

    Returns:
        Decorated function that will raise TimeoutError if it takes too long
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result: list[Any] = []
            exception: list[Exception] = []

            def target() -> None:
                try:
                    result.append(func(*args, **kwargs))
                except Exception as e:
                    exception.append(e)

            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            thread.join(timeout=timeout_seconds)

            if thread.is_alive():
                # Thread is still running - operation timed out
                # Note: We can't kill the thread, but marking it as daemon means
                # it won't prevent program exit
                raise TimeoutError(f"{func.__name__} timed out after {timeout_seconds} seconds")

            if exception:
                raise exception[0]

            return result[0] if result else None

        return wrapper  # type: ignore[return-value]

    return decorator


def _import_soapy() -> Any:
    try:
        import SoapySDR

        return SoapySDR
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "SoapySDR Python bindings not available. Install extra 'soapy' or ensure SoapySDR is present."
        ) from exc


@dataclass
class _SoapyStream(StreamHandle):
    sdr: Any  # SoapySDR.Device (dynamically imported)
    stream: Any  # SoapySDR.Stream (dynamically imported)
    max_samples: int | None = None
    # Pre-allocated buffer pool for zero-allocation reads
    _preallocated_buffer: np.ndarray | None = None
    _preallocated_size: int = 0
    # Statistics tracking
    _read_count: int = 0
    _overflow_count: int = 0
    _timeout_count: int = 0
    _error_count: int = 0
    _total_samples: int = 0
    _last_stats_log: float = 0.0
    _stats_log_interval: float = 30.0  # Log stats every 30 seconds

    def _ensure_buffer(self, num_samples: int) -> np.ndarray:
        """Get or create a pre-allocated buffer of sufficient size."""
        if self._preallocated_buffer is None or self._preallocated_size < num_samples:
            # Allocate with some headroom to reduce reallocations
            alloc_size = max(num_samples, 65536)  # At least 64K samples
            self._preallocated_buffer = np.empty(alloc_size, dtype=np.complex64)
            self._preallocated_size = alloc_size
        return self._preallocated_buffer

    def read(self, num_samples: int) -> tuple[NDArrayComplex, bool]:
        import time as _time

        self._read_count += 1
        buff = self._ensure_buffer(num_samples)

        # Use readStreamIntoBuffers for pre-allocated buffer reads
        # This avoids buffer allocation overhead in the SoapySDR binding
        SOAPY_SDR_READ_FLAG_OVERFLOW = 1 << 1

        try:
            # readStreamIntoBuffers reads directly into our pre-allocated buffer
            sr = self.sdr.readStreamIntoBuffers(
                self.stream,
                [buff[:num_samples].view(np.float32)],
                flags=0,
                timeoutUs=500000,
            )
        except AttributeError:
            # Fallback to legacy readStream if readStreamIntoBuffers not available
            sr = self.sdr.readStream(
                self.stream,
                [buff[:num_samples].view(np.float32)],
                num_samples,
                flags=0,
                timeoutUs=500000,
            )

        # Handle both tuple and StreamResult object
        if hasattr(sr, "ret"):
            ret = sr.ret
            flags = sr.flags if hasattr(sr, "flags") else 0
        elif isinstance(sr, tuple):
            padded = (*sr, 0, 0, 0, 0)
            ret, flags, _timeNs, _ = padded[0], padded[1], padded[2], padded[3]
        else:
            ret = sr
            flags = 0

        # SoapySDR error codes
        SOAPY_SDR_OVERFLOW = -4
        SOAPY_SDR_TIMEOUT = -1

        if ret < 0:
            if ret == SOAPY_SDR_OVERFLOW:
                self._overflow_count += 1
                # Overflow: try to drain
                try:
                    sr2 = self.sdr.readStreamIntoBuffers(
                        self.stream,
                        [buff[:num_samples].view(np.float32)],
                        flags=0,
                        timeoutUs=100000,
                    )
                    ret2 = sr2.ret if hasattr(sr2, "ret") else (sr2[0] if isinstance(sr2, tuple) else sr2)
                except AttributeError:
                    sr2 = self.sdr.readStream(
                        self.stream,
                        [buff[:num_samples].view(np.float32)],
                        num_samples,
                        flags=0,
                        timeoutUs=100000,
                    )
                    ret2 = sr2.ret if hasattr(sr2, "ret") else (sr2[0] if isinstance(sr2, tuple) else sr2)

                if ret2 > 0:
                    self._total_samples += ret2
                    return buff[:ret2].copy(), True
                return np.empty(0, dtype=np.complex64), True
            elif ret == SOAPY_SDR_TIMEOUT:
                self._timeout_count += 1
                return np.empty(0, dtype=np.complex64), False
            else:
                self._error_count += 1
                if self._error_count <= 10 or self._error_count % 100 == 0:
                    error_names = {-2: "STREAM_ERROR", -3: "CORRUPTION", -5: "NOT_SUPPORTED", -6: "TIME_ERROR", -7: "UNDERFLOW"}
                    error_name = error_names.get(ret, f"UNKNOWN({ret})")
                    print(f"[SOAPY] readStream error: {error_name} (count={self._error_count})", flush=True)
                return np.empty(0, dtype=np.complex64), True

        # Success path
        self._total_samples += ret

        # Periodic stats logging (every 30 seconds)
        now = _time.time()
        if now - self._last_stats_log >= self._stats_log_interval:
            self._log_stats()
            self._last_stats_log = now

        # Return a copy of the pre-allocated buffer slice to avoid mutation
        return buff[:ret].copy(), bool(flags & SOAPY_SDR_READ_FLAG_OVERFLOW)

    def _log_stats(self) -> None:
        """Log streaming statistics periodically."""
        # Try to get driver-level stats if available
        try:
            stats = self.sdr.getStreamStats(self.stream)
            driver_stats = f", driver_overflows={stats.overflows}, driver_underflows={stats.underflows}" if hasattr(stats, "overflows") else ""
        except Exception:
            driver_stats = ""

        print(
            f"[SOAPY] Stream stats: reads={self._read_count}, samples={self._total_samples:,}, "
            f"overflows={self._overflow_count}, timeouts={self._timeout_count}, errors={self._error_count}{driver_stats}",
            flush=True,
        )

    def get_stats(self) -> dict[str, int]:
        """Get streaming statistics."""
        stats = {
            "reads": self._read_count,
            "total_samples": self._total_samples,
            "overflows": self._overflow_count,
            "timeouts": self._timeout_count,
            "errors": self._error_count,
        }
        # Include driver-level stats if available
        try:
            driver_stats = self.sdr.getStreamStats(self.stream)
            if hasattr(driver_stats, "overflows"):
                stats["driver_overflows"] = driver_stats.overflows
            if hasattr(driver_stats, "underflows"):
                stats["driver_underflows"] = driver_stats.underflows
            if hasattr(driver_stats, "bytesTransferred"):
                stats["bytes_transferred"] = driver_stats.bytesTransferred
        except Exception:
            pass
        return stats

    def close(self) -> None:
        """Close stream with timeout protection."""

        @with_timeout(5.0)
        def _deactivate() -> None:
            self.sdr.deactivateStream(self.stream)

        @with_timeout(5.0)
        def _close() -> None:
            self.sdr.closeStream(self.stream)

        try:
            _deactivate()
        except TimeoutError:
            print("Warning: deactivateStream timed out", flush=True)
        except Exception as e:
            print(f"Warning: deactivateStream failed: {e}", flush=True)

        try:
            _close()
        except TimeoutError:
            print("Warning: closeStream timed out", flush=True)
        except Exception as e:
            print(f"Warning: closeStream failed: {e}", flush=True)


@dataclass
class _SoapyDevice(Device):
    info: DeviceInfo
    sdr: Any  # SoapySDR.Device (dynamically imported)
    _antenna: str | None = None
    _stream_format: str | None = None  # Store stream format for start_stream()

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
        """Configure device with timeout protection."""
        # Store stream format for use in start_stream()
        self._stream_format = stream_format
        is_sdrplay = self.info.driver == "sdrplay"

        @with_timeout(10.0)
        def _do_configure() -> None:
            import SoapySDR

            with _sdrplay_operation_lock(is_sdrplay):
                # Basic configuration
                self.sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, float(sample_rate))
                self.sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, center_hz)

                # Apply device-specific settings via SoapySDR writeSetting API
                # Examples: bias_tee, rf_notch, dab_notch, agc_setpoint, etc.
                if device_settings:
                    # Query available settings for validation
                    available_settings = set()
                    try:
                        settings_info = self.sdr.getSettingInfo()
                        for setting in settings_info:
                            setting_key = setting.key if hasattr(setting, "key") else str(setting)
                            available_settings.add(setting_key)
                    except Exception:
                        pass

                    for key, value in device_settings.items():
                        # Validate setting exists (if we got the list)
                        if available_settings and key not in available_settings:
                            print(
                                f"[WARNING] Setting '{key}' may not be supported by this device",
                                flush=True,
                            )

                        try:
                            # Convert value to string as required by writeSetting
                            self.sdr.writeSetting(key, str(value))
                            print(f"[SOAPY] Applied device setting: {key}={value}", flush=True)
                        except Exception as e:
                            print(
                                f"[WARNING] Failed to apply device setting {key}={value}: {e}",
                                flush=True,
                            )

                # Per-element gain control (e.g., LNA, VGA, TIA)
                if element_gains:
                    # Query available gain elements for validation
                    available_gains = []
                    with contextlib.suppress(Exception):
                        available_gains = list(self.sdr.listGains(SoapySDR.SOAPY_SDR_RX, 0))

                    for element_name, element_gain in element_gains.items():
                        # Validate element exists
                        if available_gains and element_name not in available_gains:
                            print(
                                f"[WARNING] Gain element '{element_name}' may not be supported by this device. Available: {available_gains}",
                                flush=True,
                            )

                        # Validate gain value is in range
                        try:
                            gain_range = self.sdr.getGainRange(
                                SoapySDR.SOAPY_SDR_RX, 0, element_name
                            )
                            if gain_range:
                                gain_min = float(gain_range.minimum())
                                gain_max = float(gain_range.maximum())
                                if element_gain < gain_min or element_gain > gain_max:
                                    print(
                                        f"[WARNING] {element_name} gain {element_gain} dB is outside valid range [{gain_min}, {gain_max}] dB",
                                        flush=True,
                                    )
                        except Exception:
                            pass

                        try:
                            self.sdr.setGain(
                                SoapySDR.SOAPY_SDR_RX, 0, element_name, element_gain
                            )
                            print(
                                f"[SOAPY] Set {element_name} gain to {element_gain} dB", flush=True
                            )
                        except Exception as e:
                            print(f"[WARNING] Failed to set {element_name} gain: {e}", flush=True)

                # Overall gain (only if no per-element gains specified)
                if gain is not None and not element_gains:
                    # Manual gain
                    print(f"[SOAPY] Setting gain to {gain} dB (manual mode)", flush=True)
                    try:
                        try:
                            self.sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, False)  # manual
                        except Exception:
                            pass
                        self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, gain)
                        actual_gain = self.sdr.getGain(SoapySDR.SOAPY_SDR_RX, 0)
                        print(f"[SOAPY] Gain set to {gain} dB, readback: {actual_gain} dB", flush=True)
                    except Exception as e:
                        print(f"[SOAPY] Failed to set gain: {e}", flush=True)
                elif gain is None and not element_gains:
                    # Prefer automatic gain control when supported
                    try:
                        self.sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, True)  # auto
                    except Exception:
                        # Not all drivers support automatic gain; skip
                        pass

                # DC offset auto-correction
                try:
                    self.sdr.setDCOffsetMode(SoapySDR.SOAPY_SDR_RX, 0, dc_offset_auto)
                    print(f"[SOAPY] DC offset auto-correction: {dc_offset_auto}", flush=True)
                except Exception:
                    # Not all devices support DC offset correction
                    pass

                # IQ balance auto-correction
                try:
                    self.sdr.setIQBalanceMode(SoapySDR.SOAPY_SDR_RX, 0, iq_balance_auto)
                    print(f"[SOAPY] IQ balance auto-correction: {iq_balance_auto}", flush=True)
                except Exception:
                    # Not all devices support IQ balance correction
                    pass

                if bandwidth is not None:
                    with contextlib.suppress(Exception):
                        self.sdr.setBandwidth(SoapySDR.SOAPY_SDR_RX, 0, bandwidth)
                if ppm is not None:
                    with contextlib.suppress(Exception):
                        self.sdr.setFrequencyCorrection(SoapySDR.SOAPY_SDR_RX, 0, ppm)
                if antenna is not None:
                    self._antenna = antenna

        _do_configure()
        with _sdrplay_operation_lock(is_sdrplay):
            self._verify_settings_applied(
                center_hz=center_hz,
                sample_rate=sample_rate,
                gain=gain,
                bandwidth=bandwidth,
                ppm=ppm,
                antenna=antenna,
                device_settings=device_settings,
                element_gains=element_gains,
            )

    def _verify_settings_applied(
        self,
        *,
        center_hz: float | None,
        sample_rate: int | None,
        gain: float | None,
        bandwidth: float | None,
        ppm: float | None,
        antenna: str | None,
        device_settings: dict[str, Any] | None,
        element_gains: dict[str, float] | None,
    ) -> None:
        """Read back SDR settings to confirm they were applied."""
        import SoapySDR

        def _safe_get(getter: Callable[[], Any]) -> Any | None:
            with contextlib.suppress(Exception):
                return getter()
            return None

        def _check_numeric(
            label: str,
            expected: float | int | None,
            actual: float | int | None,
            tolerance: float = 0.0,
            unit: str = "",
            precision: int = 2,
        ) -> tuple[str, str | None]:
            if actual is None:
                return "", None
            actual_text = (
                f"{actual:.{precision}f}{unit}" if isinstance(actual, float) else f"{actual}{unit}"
            )
            mismatch: str | None = None
            if expected is not None and abs(float(actual) - float(expected)) > tolerance:
                mismatch = f"{label} expected {expected}{unit}, got {actual_text}"
            return f"{label}={actual_text}", mismatch

        readback_parts: list[str] = []
        mismatches: list[str] = []

        actual_sample_rate = _safe_get(
            lambda: float(self.sdr.getSampleRate(SoapySDR.SOAPY_SDR_RX, 0))
        )
        text, mismatch = _check_numeric(
            "sample_rate",
            sample_rate,
            actual_sample_rate,
            tolerance=1.0,
            unit="Hz",
            precision=0,
        )
        if text:
            readback_parts.append(text)
        if mismatch:
            mismatches.append(mismatch)

        actual_center = _safe_get(lambda: float(self.sdr.getFrequency(SoapySDR.SOAPY_SDR_RX, 0)))
        text, mismatch = _check_numeric(
            "center_hz",
            center_hz,
            actual_center,
            tolerance=1.0,
            unit="Hz",
            precision=1,
        )
        if text:
            readback_parts.append(text)
        if mismatch:
            mismatches.append(mismatch)

        actual_gain = _safe_get(lambda: float(self.sdr.getGain(SoapySDR.SOAPY_SDR_RX, 0)))
        text, mismatch = _check_numeric(
            "gain",
            gain,
            actual_gain,
            tolerance=0.25,
            unit="dB",
            precision=2,
        )
        if text:
            readback_parts.append(text)
        if mismatch:
            mismatches.append(mismatch)

        actual_bandwidth = _safe_get(lambda: float(self.sdr.getBandwidth(SoapySDR.SOAPY_SDR_RX, 0)))
        text, mismatch = _check_numeric(
            "bandwidth",
            bandwidth,
            actual_bandwidth,
            tolerance=1.0,
            unit="Hz",
            precision=0,
        )
        if text:
            readback_parts.append(text)
        if mismatch:
            mismatches.append(mismatch)

        actual_ppm = _safe_get(
            lambda: float(self.sdr.getFrequencyCorrection(SoapySDR.SOAPY_SDR_RX, 0))
        )
        text, mismatch = _check_numeric(
            "ppm",
            ppm,
            actual_ppm,
            tolerance=0.05,
            unit="ppm",
            precision=2,
        )
        if text:
            readback_parts.append(text)
        if mismatch:
            mismatches.append(mismatch)

        actual_antenna = _safe_get(lambda: str(self.sdr.getAntenna(SoapySDR.SOAPY_SDR_RX, 0)))
        if actual_antenna is not None:
            readback_parts.append(f"antenna={actual_antenna}")
            if antenna is not None and actual_antenna != antenna:
                mismatches.append(f"antenna expected {antenna}, got {actual_antenna}")

        for key, expected_value in (device_settings or {}).items():
            actual_value = _safe_get(lambda: self.sdr.readSetting(key))
            if actual_value is None:
                continue
            actual_text = str(actual_value)
            readback_parts.append(f"setting[{key}]={actual_text}")
            if str(expected_value) != actual_text:
                mismatches.append(f"setting {key} expected {expected_value}, got {actual_text}")

        for element, expected_gain in (element_gains or {}).items():
            actual_element_gain = _safe_get(
                lambda: float(self.sdr.getGain(SoapySDR.SOAPY_SDR_RX, 0, element))
            )
            if actual_element_gain is None:
                continue
            readback_parts.append(f"gain[{element}]={actual_element_gain:.2f}dB")
            if abs(actual_element_gain - expected_gain) > 0.25:
                mismatches.append(
                    f"{element} gain expected {expected_gain}dB, got {actual_element_gain:.2f}dB"
                )

        if readback_parts:
            print(f"[SOAPY] Readback: {', '.join(readback_parts)}", flush=True)
        if mismatches:
            print(
                f"[SOAPY] WARNING: Setting mismatches detected: {'; '.join(mismatches)}", flush=True
            )

    def start_stream(self) -> StreamHandle:
        """Start stream with timeout protection."""
        is_sdrplay = self.info.driver == "sdrplay"

        @with_timeout(10.0)
        def _do_start_stream() -> StreamHandle:
            import SoapySDR

            with _sdrplay_operation_lock(is_sdrplay):
                # Set antenna: use configured antenna if specified, otherwise use first available
                available_antennas = self.sdr.listAntennas(SoapySDR.SOAPY_SDR_RX, 0)
                print(
                    f"[DEBUG] Device {self.info.driver}: Available antennas: {available_antennas}",
                    flush=True,
                )
                if self._antenna is not None:
                    antenna = self._antenna
                    print(
                        f"[DEBUG] Device {self.info.driver}: Requested antenna: {antenna}",
                        flush=True,
                    )
                else:
                    antenna = available_antennas[0] if available_antennas else "RX"
                    print(
                        f"[DEBUG] Device {self.info.driver}: Using default antenna: {antenna}",
                        flush=True,
                    )
                self.sdr.setAntenna(SoapySDR.SOAPY_SDR_RX, 0, antenna)
                # Verify what was actually set by querying the device
                actual_antenna = self.sdr.getAntenna(SoapySDR.SOAPY_SDR_RX, 0)
                print(
                    f"[DEBUG] Device {self.info.driver}: Antenna set to: {antenna}, device reports: {actual_antenna}",
                    flush=True,
                )
                # Update _antenna to reflect what was actually set
                self._antenna = actual_antenna

                # Select stream format based on configuration
                # Default to CF32 (complex float32) if not specified
                format_map = {
                    "CF32": SoapySDR.SOAPY_SDR_CF32,  # Complex float32 (default)
                    "CS16": SoapySDR.SOAPY_SDR_CS16,  # Complex int16
                    "CS8": SoapySDR.SOAPY_SDR_CS8,  # Complex int8
                }
                stream_fmt = format_map.get(
                    self._stream_format or "CF32", SoapySDR.SOAPY_SDR_CF32
                )
                print(
                    f"[SOAPY] Using stream format: {self._stream_format or 'CF32'}",
                    flush=True,
                )

                # Setup stream with optimized buffer configuration
                stream = self.sdr.setupStream(SoapySDR.SOAPY_SDR_RX, stream_fmt, [0])

                # Query and log buffer information
                mtu = None
                try:
                    mtu = int(self.sdr.getStreamMTU(stream))
                    if mtu > 0:
                        print(f"[SOAPY] Stream MTU: {mtu} samples", flush=True)
                    else:
                        mtu = None
                except Exception:
                    mtu = None

                # Check for direct buffer access support (zero-copy)
                has_direct_access = False
                try:
                    has_direct_access = self.sdr.hasDirectBufferAccess(stream)
                    if has_direct_access:
                        num_bufs, buf_size = self.sdr.getDirectBuffersInfo(stream)
                        print(
                            f"[SOAPY] Direct buffer access: enabled ({num_bufs} buffers, {buf_size} bytes each)",
                            flush=True,
                        )
                    else:
                        print("[SOAPY] Direct buffer access: not supported by driver", flush=True)
                except Exception:
                    # API not available in this SoapySDR version
                    pass

                # Legacy direct access check (older API)
                try:
                    num_direct_bufs = self.sdr.getNumDirectAccessBuffers(stream)
                    if num_direct_bufs > 0 and not has_direct_access:
                        print(f"[SOAPY] Legacy direct access buffers: {num_direct_bufs}", flush=True)
                except Exception:
                    pass

                # Activate stream with optional buffer size hint
                # Most drivers will use sensible defaults if not specified
                self.sdr.activateStream(stream)

                # Verify antenna didn't reset after stream activation (SDRplay issue)
                post_activate_antenna = self.sdr.getAntenna(SoapySDR.SOAPY_SDR_RX, 0)
                if post_activate_antenna != self._antenna:
                    print(
                        f"[SOAPY] WARNING: Antenna changed after activateStream! Expected {self._antenna}, got {post_activate_antenna}",
                        flush=True,
                    )
                    # Try to re-set the antenna
                    self.sdr.setAntenna(SoapySDR.SOAPY_SDR_RX, 0, self._antenna)
                    final_antenna = self.sdr.getAntenna(SoapySDR.SOAPY_SDR_RX, 0)
                    print(
                        f"[SOAPY] Re-set antenna to {self._antenna}, device now reports: {final_antenna}",
                        flush=True,
                    )
                else:
                    print(
                        f"[SOAPY] Antenna verified after activateStream: {post_activate_antenna}",
                        flush=True,
                    )

                return _SoapyStream(self.sdr, stream, max_samples=mtu)

        return _do_start_stream()

    def get_antenna(self) -> str | None:
        """Return the currently configured antenna, if any."""
        return self._antenna

    def get_capabilities(self) -> dict[str, Any]:
        """Query and return device capabilities dynamically.

        This provides runtime information about what the device supports,
        useful for validation and debugging.
        """
        import SoapySDR

        caps: dict[str, Any] = {}

        try:
            # Query available gain elements
            gain_elements = self.sdr.listGains(SoapySDR.SOAPY_SDR_RX, 0)
            caps["gain_elements"] = list(gain_elements) if gain_elements else []

            # Query gain ranges for each element
            caps["gain_ranges"] = {}
            for elem in caps["gain_elements"]:
                try:
                    gain_range = self.sdr.getGainRange(SoapySDR.SOAPY_SDR_RX, 0, elem)
                    if gain_range:
                        caps["gain_ranges"][elem] = {
                            "min": float(gain_range.minimum()),
                            "max": float(gain_range.maximum()),
                        }
                except Exception:
                    pass

            # Query overall gain range
            try:
                overall_gain = self.sdr.getGainRange(SoapySDR.SOAPY_SDR_RX, 0)
                if overall_gain:
                    caps["overall_gain_range"] = {
                        "min": float(overall_gain.minimum()),
                        "max": float(overall_gain.maximum()),
                    }
            except Exception:
                pass

            # Query available settings
            try:
                settings_info = self.sdr.getSettingInfo()
                caps["settings"] = {}
                for setting in settings_info:
                    setting_key = setting.key if hasattr(setting, "key") else str(setting)
                    caps["settings"][setting_key] = {
                        "description": setting.description
                        if hasattr(setting, "description")
                        else "",
                        "type": setting.type if hasattr(setting, "type") else "unknown",
                    }
            except Exception:
                caps["settings"] = {}

            # Query available antennas
            try:
                antennas = self.sdr.listAntennas(SoapySDR.SOAPY_SDR_RX, 0)
                caps["antennas"] = list(antennas) if antennas else []
            except Exception:
                caps["antennas"] = []

            # Query frequency ranges
            try:
                freq_ranges = self.sdr.getFrequencyRange(SoapySDR.SOAPY_SDR_RX, 0)
                caps["frequency_ranges"] = []
                for fr in freq_ranges:
                    caps["frequency_ranges"].append(
                        {"min": float(fr.minimum()), "max": float(fr.maximum())}
                    )
            except Exception:
                caps["frequency_ranges"] = []

            # Query sample rate ranges
            try:
                rate_ranges = self.sdr.getSampleRateRange(SoapySDR.SOAPY_SDR_RX, 0)
                caps["sample_rate_ranges"] = []
                for rr in rate_ranges:
                    caps["sample_rate_ranges"].append(
                        {"min": float(rr.minimum()), "max": float(rr.maximum())}
                    )
            except Exception:
                # Try discrete sample rates instead
                try:
                    rates = self.sdr.listSampleRates(SoapySDR.SOAPY_SDR_RX, 0)
                    caps["sample_rates"] = [int(r) for r in rates] if rates else []
                except Exception:
                    caps["sample_rates"] = []

            # Query bandwidth ranges
            try:
                bw_ranges = self.sdr.getBandwidthRange(SoapySDR.SOAPY_SDR_RX, 0)
                caps["bandwidth_ranges"] = []
                for bw in bw_ranges:
                    caps["bandwidth_ranges"].append(
                        {"min": float(bw.minimum()), "max": float(bw.maximum())}
                    )
            except Exception:
                caps["bandwidth_ranges"] = []

            # Query available stream formats
            try:
                formats = self.sdr.getStreamFormats(SoapySDR.SOAPY_SDR_RX, 0)
                caps["stream_formats"] = list(formats) if formats else []
            except Exception:
                caps["stream_formats"] = ["CF32"]  # Default

            # Query available sensors
            try:
                sensors = self.sdr.listSensors()
                caps["sensors"] = list(sensors) if sensors else []
            except Exception:
                caps["sensors"] = []

        except Exception as e:
            caps["error"] = str(e)

        return caps

    def read_sensors(self) -> dict[str, Any]:
        """Read all available sensors from the device.

        Returns a dict mapping sensor names to their current values.
        Common sensors include temperature, voltage, current, etc.
        """

        sensor_values: dict[str, Any] = {}

        try:
            # Get list of available sensors
            sensors = self.sdr.listSensors()

            for sensor_name in sensors:
                try:
                    # Read sensor value
                    value = self.sdr.readSensor(sensor_name)
                    sensor_values[sensor_name] = value
                except Exception as e:
                    sensor_values[sensor_name] = f"Error: {e}"

        except Exception as e:
            sensor_values["_error"] = str(e)

        return sensor_values

    def reconfigure_running(
        self,
        center_hz: float | None = None,
        gain: float | None = None,
        bandwidth: float | None = None,
        ppm: float | None = None,
    ) -> None:
        """Reconfigure device while stream is running (hot reconfiguration)."""
        is_sdrplay = self.info.driver == "sdrplay"

        @with_timeout(10.0)
        def _do_reconfigure() -> None:
            import SoapySDR

            with _sdrplay_operation_lock(is_sdrplay):
                if center_hz is not None:
                    self.sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, center_hz)

                if gain is not None:
                    try:
                        try:
                            self.sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, False)  # manual
                        except Exception:
                            pass
                        self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, gain)
                    except Exception:
                        pass

                if bandwidth is not None:
                    with contextlib.suppress(Exception):
                        self.sdr.setBandwidth(SoapySDR.SOAPY_SDR_RX, 0, bandwidth)

                if ppm is not None:
                    with contextlib.suppress(Exception):
                        self.sdr.setFrequencyCorrection(SoapySDR.SOAPY_SDR_RX, 0, ppm)

        _do_reconfigure()
        self._verify_settings_applied(
            center_hz=center_hz,
            sample_rate=None,
            gain=gain,
            bandwidth=bandwidth,
            ppm=ppm,
            antenna=None,
            device_settings=None,
            element_gains=None,
        )

    def close(self) -> None:
        """Close device with timeout protection and cooldown for SDRplay."""
        global _sdrplay_last_close_time

        if self.sdr is None:
            return

        is_sdrplay = self.info.driver == "sdrplay"

        @with_timeout(5.0)
        def _unmake() -> None:
            # Properly close the SoapySDR device
            import SoapySDR

            with _sdrplay_operation_lock(is_sdrplay):
                with contextlib.suppress(Exception):
                    SoapySDR.Device.unmake(self.sdr)

        try:
            _unmake()
        except TimeoutError:
            if is_sdrplay:
                print("Warning: SDRplay device unmake timed out", flush=True)
            else:
                print("Warning: Device unmake timed out", flush=True)
        except Exception as e:
            if is_sdrplay:
                print(f"Warning: SDRplay device unmake failed: {e}", flush=True)
            else:
                print(f"Warning: Device unmake failed: {e}", flush=True)
        finally:
            self.sdr = None
            if is_sdrplay:
                _sdrplay_last_close_time = time.time()


class SoapyDriver(DeviceDriver):
    name = "soapy"

    def __init__(self, cfg: DeviceConfig, *, use_sdrplay_proxy: bool = True):
        global _soapy_driver_instance
        self._cfg = cfg
        self._use_sdrplay_proxy = use_sdrplay_proxy
        self._SoapySDR = _import_soapy()
        self._enumerate_cache: tuple[float, list[DeviceInfo]] | None = None
        self._enumerate_cache_ttl = 30.0  # seconds
        # Cache of SDRplay devices for use when devices are busy
        self._sdrplay_device_cache: list[DeviceInfo] = []
        # Register this instance for global cache invalidation
        _soapy_driver_instance = self

    def invalidate_cache(self) -> None:
        """Invalidate enumeration cache, forcing re-enumeration on next call."""
        print("[SOAPY] Driver cache invalidated", flush=True)
        self._enumerate_cache = None
        self._sdrplay_device_cache.clear()

    def _parse_driver_arg(self, args: str) -> str | None:
        for part in args.split(","):
            part = part.strip()
            if part.lower().startswith("driver="):
                return part.split("=", 1)[1].strip().lower()
        return None

    def _resolve_driver_hint(self, identifier: str | None) -> str | None:
        if not identifier:
            return None
        driver = self._parse_driver_arg(identifier)
        if driver:
            return driver
        if self._enumerate_cache is not None:
            _cache_time, cached_results = self._enumerate_cache
            for device in cached_results:
                if device.id == identifier or device.label == identifier:
                    return device.driver
        try:
            for device in self.enumerate():
                if device.id == identifier or device.label == identifier:
                    return device.driver
        except Exception:
            return None
        return None

    def _enumerate_driver(self, driver_name: str, timeout: float = 5.0) -> list[DeviceInfo]:
        """Enumerate devices for a specific driver with timeout protection.

        Uses native SoapySDR timeout API instead of multiprocessing isolation.
        """
        SoapySDR = self._SoapySDR
        timeout_us = int(timeout * 1_000_000)

        results: list[DeviceInfo] = []
        seen_devices: set[str] = set()

        try:
            # Use native timeout API - returns empty list on timeout (logs warning)
            device_list = SoapySDR.Device.enumerate(f"driver={driver_name}", timeout_us)

            for args in device_list:
                driver = str(args["driver"]) if "driver" in args else "unknown"
                label = str(args["label"]) if "label" in args else "SDR"

                # Create stable device identifier (excludes volatile fields like 'tuner')
                serial = str(args["serial"]) if "serial" in args else ""
                stable_id = f"{driver}:{serial}" if serial else f"{driver}:{label}"

                # Skip duplicate devices
                if stable_id in seen_devices:
                    continue
                seen_devices.add(stable_id)

                # Build canonical args string, excluding volatile fields
                volatile_fields = {"tuner"}
                try:
                    items = []
                    for k in sorted(args.keys()):
                        if k in volatile_fields:
                            continue
                        items.append(f"{k}={args[k]}")
                    id_ = ",".join(items) if items else driver
                except Exception:
                    id_ = driver

                freq_min = float(args["rfmin"]) if "rfmin" in args else 1e4
                freq_max = float(args["rfmax"]) if "rfmax" in args else 6e9

                # Set device-specific limits based on driver
                if driver == "rtlsdr":
                    sample_rates: tuple[int, ...] = (
                        250_000,
                        1_000_000,
                        1_024_000,
                        1_800_000,
                        1_920_000,
                        2_000_000,
                        2_048_000,
                        2_400_000,
                        2_560_000,
                    )
                    gains: tuple[str, ...] = ("LNA", "VGA")
                    gain_min, gain_max = 0.0, 49.6
                    bandwidth_min, bandwidth_max = 200_000.0, 3_200_000.0
                    antennas: tuple[str, ...] = ("RX",)
                elif driver == "sdrplay":
                    sample_rates = (
                        200_000,
                        250_000,
                        500_000,
                        1_000_000,
                        2_000_000,
                        3_000_000,
                        4_000_000,
                        5_000_000,
                        6_000_000,
                        7_000_000,
                        8_000_000,
                        9_000_000,
                        10_000_000,
                    )
                    gains = ("LNA", "VGA")
                    gain_min, gain_max = 0.0, 59.0
                    bandwidth_min, bandwidth_max = 200_000.0, 8_000_000.0
                    antennas = ("Antenna A", "Antenna B", "Antenna C")
                else:
                    sample_rates = (250_000, 1_000_000, 2_000_000, 2_400_000)
                    gains = ("LNA", "VGA")
                    gain_min, gain_max = None, None
                    bandwidth_min, bandwidth_max = None, None
                    antennas = ()

                results.append(
                    DeviceInfo(
                        id=id_,
                        driver=driver,
                        label=label,
                        freq_min_hz=freq_min,
                        freq_max_hz=freq_max,
                        sample_rates=sample_rates,
                        gains=gains,
                        gain_min=gain_min,
                        gain_max=gain_max,
                        bandwidth_min=bandwidth_min,
                        bandwidth_max=bandwidth_max,
                        ppm_min=-100.0,
                        ppm_max=100.0,
                        antennas=antennas,
                    )
                )
        except Exception as e:
            print(
                f"Warning: Enumeration for driver '{driver_name}' failed: {e}",
                flush=True,
            )

        return results

    def enumerate(self) -> Iterable[DeviceInfo]:
        global _sdrplay_consecutive_timeouts, _sdrplay_last_enumeration_success

        # Check cache first
        now = time.time()
        if self._enumerate_cache is not None:
            cache_time, cached_results = self._enumerate_cache
            if now - cache_time < self._enumerate_cache_ttl:
                return cached_results

        # Enumerate specific drivers we support, with timeout protection
        # This prevents hangs from problematic drivers (audio, uhd, etc.)
        results = []

        # Enumerate RTL-SDR first (usually reliable)
        try:
            rtlsdr_results = self._enumerate_driver("rtlsdr", timeout=5.0)
            results.extend(rtlsdr_results)
        except Exception as e:
            print(f"Warning: Failed to enumerate driver 'rtlsdr': {e}", flush=True)

        # Enumerate SDRplay with proactive recovery on timeout
        # OPTIMIZATION: Skip enumeration if SDRplay devices are busy with active captures
        # The SDRplay API locks devices during streaming, so enumeration will timeout
        # Use cached device list instead to avoid the 5-second timeout penalty
        active_captures = get_sdrplay_active_captures()
        if active_captures > 0 and self._sdrplay_device_cache:
            # Use cached SDRplay devices when captures are active
            print(
                f"[SOAPY] Skipping SDRplay enumeration ({active_captures} active captures), using cached devices",
                flush=True,
            )
            results.extend(self._sdrplay_device_cache)
        else:
            try:
                sdrplay_results = self._enumerate_driver("sdrplay", timeout=5.0)
                if sdrplay_results:
                    results.extend(sdrplay_results)
                    # Cache successful results for use when devices are busy
                    self._sdrplay_device_cache = list(sdrplay_results)
                    # Success - reset timeout counter
                    _sdrplay_consecutive_timeouts = 0
                    _sdrplay_last_enumeration_success = now
                    print(
                        f"[SOAPY] SDRplay enumeration success: {len(sdrplay_results)} device(s)",
                        flush=True,
                    )
                else:
                    # Empty result could indicate stuck service
                    _sdrplay_consecutive_timeouts += 1
                    # Invalidate cache on empty result - don't serve stale data
                    self._enumerate_cache = None
                    print(
                        f"[SOAPY] SDRplay enumeration returned no devices (timeout count: {_sdrplay_consecutive_timeouts})",
                        flush=True,
                    )
            except Exception as e:
                _sdrplay_consecutive_timeouts += 1
                # Invalidate cache on failure - don't serve stale data
                self._enumerate_cache = None
                print(
                    f"Warning: Failed to enumerate driver 'sdrplay': {e} (timeout count: {_sdrplay_consecutive_timeouts})",
                    flush=True,
                )

        # Proactive recovery: If SDRplay has timed out multiple times, attempt service restart
        # CRITICAL: Only do this when NO active SDRplay captures are running!
        # Recovery while streaming would kill active captures.
        sdrplay_found = any(d.driver == "sdrplay" for d in results)

        # Note: active_captures check is now handled above (skip enumeration when busy)
        # Recovery is only triggered when enumeration actually fails/times out
        if (
            not sdrplay_found
            and active_captures == 0
            and _sdrplay_consecutive_timeouts >= _SDRPLAY_MAX_TIMEOUTS_BEFORE_RECOVERY
        ):
            recovery = get_recovery()
            allowed, reason = recovery.can_restart()
            if allowed:
                print(
                    f"[SOAPY] SDRplay stuck detected ({_sdrplay_consecutive_timeouts} consecutive failures), "
                    f"attempting automatic recovery...",
                    flush=True,
                )
                success = attempt_recovery(reason="Enumeration timeout - proactive recovery")
                if success:
                    print(
                        "[SOAPY] Recovery successful, clearing caches and retrying enumeration...",
                        flush=True,
                    )
                    # Clear all caches before retry
                    self._enumerate_cache = None
                    self._sdrplay_device_cache.clear()
                    # Retry enumeration after recovery
                    try:
                        sdrplay_results = self._enumerate_driver("sdrplay", timeout=5.0)
                        if sdrplay_results:
                            results.extend(sdrplay_results)
                            _sdrplay_consecutive_timeouts = 0
                            _sdrplay_last_enumeration_success = now
                            print(
                                f"[SOAPY] SDRplay recovery enumeration success: {len(sdrplay_results)} device(s)",
                                flush=True,
                            )
                    except Exception as e:
                        print(f"[SOAPY] SDRplay enumeration failed after recovery: {e}", flush=True)
                else:
                    print(f"[SOAPY] Recovery failed: {recovery.stats.last_error}", flush=True)
            else:
                print(f"[SOAPY] Cannot attempt recovery: {reason}", flush=True)
        elif not sdrplay_found and active_captures == 0:
            print(
                f"[SOAPY] No SDRplay devices found (timeout count: {_sdrplay_consecutive_timeouts})",
                flush=True,
            )

        # Device count stability tracking
        global _sdrplay_expected_device_count
        sdrplay_count = len([d for d in results if d.driver == "sdrplay"])
        if _sdrplay_expected_device_count is not None:
            if sdrplay_count < _sdrplay_expected_device_count:
                print(
                    f"[SOAPY] WARNING: SDRplay device count dropped from {_sdrplay_expected_device_count} to {sdrplay_count}",
                    flush=True,
                )
                # Clear device cache to avoid serving stale device list
                self._sdrplay_device_cache.clear()
        if sdrplay_count > 0:
            _sdrplay_expected_device_count = sdrplay_count

        # Cache the results
        self._enumerate_cache = (now, results)
        return results

    def _ensure_sdrplay_healthy(self, max_recovery_attempts: int = 2) -> bool:
        """Ensure SDRplay service is healthy before device operations.

        This is a MANDATORY pre-flight check that always runs (uses cache if recent).
        If unhealthy, attempts automatic recovery.

        Args:
            max_recovery_attempts: Maximum recovery attempts before giving up

        Returns:
            True if service is healthy (or was recovered), False otherwise.
        """
        global _sdrplay_consecutive_timeouts, _sdrplay_health_status, _sdrplay_last_health_check

        now = time.time()

        # Use cache if fresh (< 5 seconds old)
        if (
            _sdrplay_health_status
            and (now - _sdrplay_last_health_check) < _SDRPLAY_HEALTH_CACHE_TTL
        ):
            return True

        # Fresh health check via subprocess enumeration
        for attempt in range(max_recovery_attempts + 1):
            try:
                results = self._enumerate_driver("sdrplay", timeout=3.0)
                if results:
                    _sdrplay_health_status = True
                    _sdrplay_last_health_check = now
                    _sdrplay_consecutive_timeouts = 0
                    print(
                        f"[SOAPY] SDRplay health check passed ({len(results)} device(s))",
                        flush=True,
                    )
                    return True
                else:
                    _sdrplay_consecutive_timeouts += 1
                    print(
                        f"[SOAPY] SDRplay health check: no devices (attempt {attempt + 1})",
                        flush=True,
                    )
            except Exception as e:
                _sdrplay_consecutive_timeouts += 1
                print(
                    f"[SOAPY] SDRplay health check failed: {e} (attempt {attempt + 1})", flush=True
                )

            # Try recovery (except on last attempt)
            if attempt < max_recovery_attempts:
                recovery = get_recovery()
                allowed, reason = recovery.can_restart()
                if allowed:
                    print(
                        f"[SOAPY] Attempting recovery ({attempt + 1}/{max_recovery_attempts})...",
                        flush=True,
                    )
                    if attempt_recovery(f"Health check failed (attempt {attempt + 1})"):
                        print("[SOAPY] Recovery completed, retrying health check...", flush=True)
                        continue
                    else:
                        print(f"[SOAPY] Recovery failed: {recovery.stats.last_error}", flush=True)
                else:
                    print(f"[SOAPY] Recovery not allowed: {reason}", flush=True)
                    break

        _sdrplay_health_status = False
        return False

    def _open_device_subprocess(self, args: str, timeout: float = 15.0) -> dict[str, Any]:
        """Open device in subprocess with timeout protection.

        If subprocess hangs (SDRplay service stuck), it can be killed cleanly.
        This is safer than threads which cannot be forcibly terminated.

        Args:
            args: Device arguments string
            timeout: Maximum seconds to wait for device open

        Returns:
            Device info dict on success

        Raises:
            TimeoutError: If device open times out
            SDRplayServiceError: If device open fails
        """
        global _sdrplay_health_status

        queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
        process = multiprocessing.Process(target=_device_open_worker, args=(args, queue))
        process.start()
        process.join(timeout=timeout)

        if process.is_alive():
            # Timeout - kill the subprocess cleanly
            print(
                f"[SOAPY] Device open timed out after {timeout}s, killing subprocess...", flush=True
            )
            process.terminate()
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join()

            # Invalidate health cache
            _sdrplay_health_status = False

            raise TimeoutError(
                f"Device open timed out after {timeout}s. "
                "SDRplay service may be stuck. "
                "Try: POST /api/v1/devices/sdrplay/restart-service"
            )

        try:
            status, data = queue.get(timeout=0.2)
        except queue_module.Empty:
            _sdrplay_health_status = False
            raise SDRplayServiceError("Device open failed - no response from subprocess")
        if status == "error":
            raise SDRplayServiceError(f"Device open failed: {data}")

        return cast(dict[str, Any], data)

    def _build_device_info_subprocess(self, args: str, timeout: float = 10.0) -> dict[str, Any]:
        """Query device info via subprocess without consuming API slot.

        Uses the existing _device_open_worker which opens, queries, and closes
        the device in a subprocess. This provides device info for SDRplayProxyDevice
        without blocking the SDRplay API for the actual capture.

        Args:
            args: Device arguments string

        Returns:
            Device info dict with driver, hardware, freq_min, freq_max, sample_rates, antennas
        """
        try:
            # Reuse existing subprocess worker
            return self._open_device_subprocess(args, timeout=timeout)
        except Exception as e:
            # Return fallback info if subprocess fails
            print(f"[SOAPY] Device info query failed: {e}, using defaults", flush=True)
            return {
                "driver": "sdrplay",
                "hardware": "SDRplay RSP",
                "freq_min": 1e4,
                "freq_max": 6e9,
                "sample_rates": [200_000, 500_000, 1_000_000, 2_000_000, 4_000_000, 6_000_000],
                "antennas": ["Antenna A", "Antenna B", "Antenna C"],
            }

    def open(self, id_or_args: str | None = None) -> Device:
        global _sdrplay_last_close_time

        SoapySDR = self._SoapySDR
        args = id_or_args or self._cfg.device_args or ""

        # Check if this is an SDRplay device
        driver_hint = self._resolve_driver_hint(id_or_args or args)
        is_sdrplay = driver_hint == "sdrplay" or "sdrplay" in args.lower()

        # For SDRplay, use subprocess proxy to bypass API single-device limitation
        # Each SDRplay device runs in its own isolated subprocess
        #
        # Two proxy modes available:
        # 1. Native driver proxy (SoapySDRPlay3 with ENABLE_SUBPROCESS_MULTIDEV)
        #    - Enabled by adding proxy=true to device args
        #    - Driver spawns worker subprocess transparently
        #    - More robust recovery and health monitoring
        # 2. Python subprocess proxy (SDRplayProxyDevice)
        #    - Fallback when native proxy unavailable
        #    - Uses multiprocessing + shared memory
        if is_sdrplay and self._use_sdrplay_proxy:
            # Try native driver proxy first by adding proxy=true to args
            native_proxy_args = self._add_proxy_arg(args)
            if self._check_native_proxy_available(native_proxy_args):
                print(f"[SOAPY] Using native driver proxy for SDRplay: {native_proxy_args}", flush=True)
                # Use direct open with proxy=true - driver handles subprocess
                return self._open_with_native_proxy(native_proxy_args)

            # Fallback to Python subprocess proxy
            from .sdrplay_proxy import SDRplayProxyDevice

            print(f"[SOAPY] Using Python subprocess proxy for SDRplay device: {args}", flush=True)

            # IMPORTANT: DO NOT probe device here! Device probing via subprocess
            # happens BEFORE the global SDRplay lock is acquired, which can corrupt
            # the SDRplay API state when multiple captures start simultaneously.
            #
            # Use hardcoded defaults instead - the worker subprocess will query
            # actual device capabilities when it opens (inside the lock).
            info = DeviceInfo(
                id=str(id_or_args or "device0"),
                driver="sdrplay",
                label="SDRplay RSP",
                freq_min_hz=1e4,
                freq_max_hz=6e9,
                sample_rates=(
                    200_000,
                    250_000,
                    500_000,
                    1_000_000,
                    2_000_000,
                    3_000_000,
                    4_000_000,
                    5_000_000,
                    6_000_000,
                ),
                gains=("IFGR", "RFGR"),
                gain_min=0.0,
                gain_max=59.0,
                bandwidth_min=200_000.0,
                bandwidth_max=8_000_000.0,
                ppm_min=-100.0,
                ppm_max=100.0,
                antennas=("Antenna A", "Antenna B", "Antenna C"),
            )

            return SDRplayProxyDevice(info=info, device_args=args)

        if is_sdrplay:
            print(f"[SOAPY] Using direct SDRplay access (proxy disabled): {args}", flush=True)

        def _open_device() -> tuple[Any, DeviceInfo]:
            # Open with timeout (10 seconds = 10,000,000 microseconds)
            DEVICE_OPEN_TIMEOUT_US = 10_000_000
            try:
                sdr = SoapySDR.Device.make(args, DEVICE_OPEN_TIMEOUT_US)
            except AttributeError:
                sdr = SoapySDR.Device(args)
            # Build DeviceInfo from the live device
            driver = str(sdr.getDriverKey())

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

            # Query gain range
            try:
                gain_range = sdr.getGainRange(SoapySDR.SOAPY_SDR_RX, 0)
                if gain_range:
                    gain_min = float(gain_range.minimum())
                    gain_max = float(gain_range.maximum())
                else:
                    # Fallback to driver-specific defaults
                    if driver == "rtlsdr":
                        gain_min, gain_max = 0.0, 49.6
                    elif driver == "sdrplay":
                        gain_min, gain_max = 0.0, 59.0
                    else:
                        gain_min, gain_max = None, None
            except Exception:
                # Fallback to driver-specific defaults
                if driver == "rtlsdr":
                    gain_min, gain_max = 0.0, 49.6
                elif driver == "sdrplay":
                    gain_min, gain_max = 0.0, 59.0
                else:
                    gain_min, gain_max = None, None

            # Query bandwidth range
            try:
                bandwidth_range = sdr.getBandwidthRange(SoapySDR.SOAPY_SDR_RX, 0)
                if bandwidth_range:
                    bandwidth_min = (
                        float(bandwidth_range[0].minimum()) if bandwidth_range else None
                    )
                    bandwidth_max = (
                        float(bandwidth_range[0].maximum()) if bandwidth_range else None
                    )
                    # If min == max, the driver didn't report a useful range - use fallbacks
                    if (
                        bandwidth_min is not None
                        and bandwidth_max is not None
                        and bandwidth_min >= bandwidth_max
                    ):
                        if driver == "rtlsdr":
                            bandwidth_min, bandwidth_max = 200_000.0, 3_200_000.0
                        elif driver == "sdrplay":
                            bandwidth_min, bandwidth_max = 200_000.0, 8_000_000.0
                else:
                    # Fallback to driver-specific defaults
                    if driver == "rtlsdr":
                        bandwidth_min, bandwidth_max = 200_000.0, 3_200_000.0
                    elif driver == "sdrplay":
                        bandwidth_min, bandwidth_max = 200_000.0, 8_000_000.0
                    else:
                        bandwidth_min, bandwidth_max = None, None
            except Exception:
                # Fallback to driver-specific defaults
                if driver == "rtlsdr":
                    bandwidth_min, bandwidth_max = 200_000.0, 3_200_000.0
                elif driver == "sdrplay":
                    bandwidth_min, bandwidth_max = 200_000.0, 8_000_000.0
                else:
                    bandwidth_min, bandwidth_max = None, None

            # PPM correction range (standard across SDRs)
            ppm_min, ppm_max = -100.0, 100.0

            # Query available antennas
            try:
                antennas = tuple(sdr.listAntennas(SoapySDR.SOAPY_SDR_RX, 0))
            except Exception:
                antennas = ()

            info = DeviceInfo(
                id=str(id_or_args or "device0"),
                driver=driver,
                label=str(sdr.getHardwareKey()),
                freq_min_hz=freq_min,
                freq_max_hz=freq_max,
                sample_rates=sample_rates,
                gains=(),
                gain_min=gain_min,
                gain_max=gain_max,
                bandwidth_min=bandwidth_min,
                bandwidth_max=bandwidth_max,
                ppm_min=ppm_min,
                ppm_max=ppm_max,
                antennas=antennas,
            )
            return sdr, info

        if is_sdrplay:
            with _sdrplay_operation_lock(True):
                elapsed = time.time() - _sdrplay_last_close_time
                if elapsed < _SDRPLAY_CLOSE_COOLDOWN:
                    time.sleep(_SDRPLAY_CLOSE_COOLDOWN - elapsed)
                sdr, info = _open_device()
        else:
            sdr, info = _open_device()

        return _SoapyDevice(info=info, sdr=sdr)

    def _add_proxy_arg(self, args: str) -> str:
        """Add proxy=true to device args string for native driver proxy."""
        if "proxy=true" in args.lower():
            return args
        if args:
            return f"{args},proxy=true"
        return "proxy=true"

    def _check_native_proxy_available(self, args: str) -> bool:
        """Check if SoapySDRPlay3's native proxy mode is available.

        The driver must be compiled with ENABLE_SUBPROCESS_MULTIDEV for this to work.
        We detect availability by checking if enumeration returns proxy-tagged devices.
        """
        try:
            SoapySDR = self._SoapySDR

            # Quick check: enumerate with proxy=true and see if results have proxy tag
            # This is fast because proxy mode uses cached results
            results = SoapySDR.Device.enumerate(args)

            # If we got results and they have proxy=true, native proxy is available
            for r in results:
                if r.get("proxy") == "true":
                    return True

            # No proxy devices found - driver doesn't support native proxy
            return False
        except Exception as e:
            print(f"[SOAPY] Native proxy check failed: {e}", flush=True)
            return False

    def _open_with_native_proxy(self, args: str) -> Device:
        """Open SDRplay device using native driver proxy mode.

        The SoapySDRPlay3 driver handles subprocess spawning transparently.
        We just open the device normally - the driver does the rest.
        """
        SoapySDR = self._SoapySDR

        # Open with timeout
        DEVICE_OPEN_TIMEOUT_US = 10_000_000
        try:
            sdr = SoapySDR.Device.make(args, DEVICE_OPEN_TIMEOUT_US)
        except AttributeError:
            sdr = SoapySDR.Device(args)

        # Query device info
        driver = str(sdr.getDriverKey())
        hardware = str(sdr.getHardwareKey())

        # Get frequency range
        try:
            rx_ranges = sdr.getFrequencyRange(SoapySDR.SOAPY_SDR_RX, 0)
            freq_min = float(rx_ranges[0].minimum()) if rx_ranges else 1e4
            freq_max = float(rx_ranges[0].maximum()) if rx_ranges else 6e9
        except Exception:
            freq_min, freq_max = 1e4, 6e9

        # Get sample rates
        try:
            sample_rates = tuple(sdr.listSampleRates(SoapySDR.SOAPY_SDR_RX, 0))
        except Exception:
            sample_rates = (2_000_000, 3_000_000, 4_000_000, 5_000_000, 6_000_000)

        # Get gain range
        try:
            gain_range = sdr.getGainRange(SoapySDR.SOAPY_SDR_RX, 0)
            gain_min = float(gain_range.minimum())
            gain_max = float(gain_range.maximum())
        except Exception:
            gain_min, gain_max = 0.0, 59.0

        # Get antennas
        try:
            antennas = tuple(sdr.listAntennas(SoapySDR.SOAPY_SDR_RX, 0))
        except Exception:
            antennas = ("Antenna A", "Antenna B", "Antenna C")

        info = DeviceInfo(
            id=args,
            driver=driver,
            label=f"{hardware} (native proxy)",
            freq_min_hz=freq_min,
            freq_max_hz=freq_max,
            sample_rates=sample_rates,
            gains=("IFGR", "RFGR"),
            gain_min=gain_min,
            gain_max=gain_max,
            bandwidth_min=200_000.0,
            bandwidth_max=8_000_000.0,
            ppm_min=-100.0,
            ppm_max=100.0,
            antennas=antennas,
        )

        return _SoapyDevice(info=info, sdr=sdr)
