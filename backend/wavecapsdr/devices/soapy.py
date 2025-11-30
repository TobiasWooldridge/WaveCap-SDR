from __future__ import annotations

from dataclasses import dataclass
import functools
import multiprocessing
import signal
import threading
import time
from typing import Any, Callable, Iterable, Optional, Tuple, TypeVar

import numpy as np

from ..config import DeviceConfig
# Re-enabled for proactive recovery during enumeration (not during streaming)
from ..sdrplay_recovery import attempt_recovery, get_recovery
from .base import Device, DeviceDriver, DeviceInfo, StreamHandle

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


class SDRplayServiceError(Exception):
    """SDRplay API service is unresponsive or device open failed.

    This exception indicates the SDRplay service needs to be restarted.
    Use: POST /api/v1/devices/sdrplay/restart-service
    """
    pass


def get_sdrplay_health_status() -> dict:
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
        "last_success_timestamp": _sdrplay_last_enumeration_success if _sdrplay_last_enumeration_success > 0 else None,
        "seconds_since_success": (now - _sdrplay_last_enumeration_success) if _sdrplay_last_enumeration_success > 0 else None,
        "recovery_threshold": _SDRPLAY_MAX_TIMEOUTS_BEFORE_RECOVERY,
    }


def reset_sdrplay_health_counters() -> None:
    """Reset SDRplay health counters after manual intervention."""
    global _sdrplay_consecutive_timeouts, _sdrplay_last_enumeration_success
    _sdrplay_consecutive_timeouts = 0
    _sdrplay_last_enumeration_success = time.time()


F = TypeVar('F', bound=Callable[..., Any])


def _enumerate_worker(driver_name: str, queue: multiprocessing.Queue) -> None:
    """Worker function to enumerate SoapySDR devices in subprocess.

    This is a module-level function so it can be pickled for multiprocessing.
    """
    try:
        import SoapySDR  # type: ignore
        # Unload and reload modules to reset any stuck state
        try:
            SoapySDR.Device.unmake()  # type: ignore[attr-defined]
        except Exception:
            pass
        # Only enumerate for this specific driver
        results = []
        seen_devices = set()  # Track unique devices by stable identifier
        for args in SoapySDR.Device.enumerate(f"driver={driver_name}"):  # type: ignore[attr-defined]
            driver = str(args["driver"]) if "driver" in args else "unknown"
            label = str(args["label"]) if "label" in args else "SDR"

            # Create a stable device identifier that doesn't change based on availability
            # Use serial number if available, otherwise fall back to label
            # Exclude volatile fields like 'tuner' which changes when device is busy
            # Note: SoapySDRKwargs doesn't have .get(), use 'in' check instead
            serial = str(args["serial"]) if "serial" in args else ""
            stable_id = f"{driver}:{serial}" if serial else f"{driver}:{label}"

            # Skip duplicate devices (e.g., same device with tuner=unavailable vs tuner=R828D)
            if stable_id in seen_devices:
                continue
            seen_devices.add(stable_id)

            # Build a canonical args string, excluding volatile fields
            # 'tuner' field changes to 'unavailable' when device is busy
            volatile_fields = {'tuner'}
            try:
                items = []
                for k in sorted(args.keys()):
                    if k in volatile_fields:
                        continue
                    v = args[k] if k in args else None
                    if v is None:
                        continue
                    items.append(f"{k}={v}")
                id_ = ",".join(items) if items else driver
            except Exception:
                id_ = driver

            freq_min = float(args["rfmin"]) if "rfmin" in args else 1e4
            freq_max = float(args["rfmax"]) if "rfmax" in args else 6e9
            gains = ("LNA", "VGA")

            # Set device-specific limits based on driver
            if driver == "rtlsdr":
                # RTL-SDR common sample rates
                sample_rates = (250_000, 1_000_000, 1_024_000, 1_800_000, 1_920_000, 2_000_000, 2_048_000, 2_400_000, 2_560_000)
                gain_min, gain_max = 0.0, 49.6
                bandwidth_min, bandwidth_max = 200_000.0, 3_200_000.0
                antennas = ("RX",)
            elif driver == "sdrplay":
                # SDRplay supports a wide range of sample rates from 200 kHz to 10 MHz
                sample_rates = (
                    200_000, 250_000, 500_000, 1_000_000, 2_000_000,
                    3_000_000, 4_000_000, 5_000_000, 6_000_000,
                    7_000_000, 8_000_000, 9_000_000, 10_000_000
                )
                gain_min, gain_max = 0.0, 59.0
                bandwidth_min, bandwidth_max = 200_000.0, 8_000_000.0
                antennas = ("Antenna A", "Antenna B", "Antenna C")
            else:
                # Default for unknown devices
                sample_rates = (250_000, 1_000_000, 2_000_000, 2_400_000)
                gain_min, gain_max = None, None
                bandwidth_min, bandwidth_max = None, None
                antennas = ()

            ppm_min, ppm_max = -100.0, 100.0

            results.append({
                "id": id_,
                "driver": driver,
                "label": label,
                "freq_min_hz": freq_min,
                "freq_max_hz": freq_max,
                "sample_rates": sample_rates,
                "gains": gains,
                "gain_min": gain_min,
                "gain_max": gain_max,
                "bandwidth_min": bandwidth_min,
                "bandwidth_max": bandwidth_max,
                "ppm_min": ppm_min,
                "ppm_max": ppm_max,
                "antennas": antennas,
            })
        queue.put(("success", results))
    except Exception as e:
        queue.put(("error", str(e)))


def _device_open_worker(device_args: str, result_queue: multiprocessing.Queue) -> None:
    """Worker function to open SoapySDR device in subprocess.

    This isolates the blocking SoapySDR.Device() call so it can be
    terminated if it hangs (unlike threads which cannot be killed).

    The subprocess opens the device, queries its capabilities, then closes it.
    If successful, the main process knows it's safe to open the device.
    """
    try:
        import SoapySDR  # type: ignore

        # Open the device (this is the blocking call that can hang)
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
        try:
            SoapySDR.Device.unmake(sdr)
        except Exception:
            pass

        result_queue.put(("success", {
            "driver": driver,
            "hardware": hardware,
            "freq_min": freq_min,
            "freq_max": freq_max,
            "sample_rates": sample_rates,
            "antennas": antennas,
        }))
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
                raise TimeoutError(
                    f"{func.__name__} timed out after {timeout_seconds} seconds"
                )

            if exception:
                raise exception[0]

            return result[0] if result else None

        return wrapper  # type: ignore[return-value]
    return decorator


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
            print(f"Warning: deactivateStream timed out", flush=True)
        except Exception as e:
            print(f"Warning: deactivateStream failed: {e}", flush=True)

        try:
            _close()
        except TimeoutError:
            print(f"Warning: closeStream timed out", flush=True)
        except Exception as e:
            print(f"Warning: closeStream failed: {e}", flush=True)


@dataclass
class _SoapyDevice(Device):
    info: DeviceInfo
    sdr: "SoapySDR.Device"
    _antenna: Optional[str] = None
    _stream_format: Optional[str] = None  # Store stream format for start_stream()

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
        """Configure device with timeout protection."""
        # Store stream format for use in start_stream()
        self._stream_format = stream_format

        @with_timeout(10.0)
        def _do_configure() -> None:
            import SoapySDR  # type: ignore

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
                        setting_key = setting.key if hasattr(setting, 'key') else str(setting)
                        available_settings.add(setting_key)
                except Exception:
                    pass

                for key, value in device_settings.items():
                    # Validate setting exists (if we got the list)
                    if available_settings and key not in available_settings:
                        print(f"[WARNING] Setting '{key}' may not be supported by this device", flush=True)

                    try:
                        # Convert value to string as required by writeSetting
                        self.sdr.writeSetting(key, str(value))
                        print(f"[SOAPY] Applied device setting: {key}={value}", flush=True)
                    except Exception as e:
                        print(f"[WARNING] Failed to apply device setting {key}={value}: {e}", flush=True)

            # Per-element gain control (e.g., LNA, VGA, TIA)
            if element_gains:
                # Query available gain elements for validation
                available_gains = []
                try:
                    available_gains = list(self.sdr.listGains(SoapySDR.SOAPY_SDR_RX, 0))
                except Exception:
                    pass

                for element_name, element_gain in element_gains.items():
                    # Validate element exists
                    if available_gains and element_name not in available_gains:
                        print(f"[WARNING] Gain element '{element_name}' may not be supported by this device. Available: {available_gains}", flush=True)

                    # Validate gain value is in range
                    try:
                        gain_range = self.sdr.getGainRange(SoapySDR.SOAPY_SDR_RX, 0, element_name)
                        if gain_range:
                            gain_min = float(gain_range.minimum())
                            gain_max = float(gain_range.maximum())
                            if element_gain < gain_min or element_gain > gain_max:
                                print(f"[WARNING] {element_name} gain {element_gain} dB is outside valid range [{gain_min}, {gain_max}] dB", flush=True)
                    except Exception:
                        pass

                    try:
                        self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, element_name, element_gain)
                        print(f"[SOAPY] Set {element_name} gain to {element_gain} dB", flush=True)
                    except Exception as e:
                        print(f"[WARNING] Failed to set {element_name} gain: {e}", flush=True)

            # Overall gain (only if no per-element gains specified)
            if gain is not None and not element_gains:
                # Manual gain
                try:
                    try:
                        self.sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, False)  # manual
                    except Exception:
                        pass
                    self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, gain)
                except Exception:
                    pass
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
            except Exception as e:
                # Not all devices support DC offset correction
                pass

            # IQ balance auto-correction
            try:
                self.sdr.setIQBalanceMode(SoapySDR.SOAPY_SDR_RX, 0, iq_balance_auto)
                print(f"[SOAPY] IQ balance auto-correction: {iq_balance_auto}", flush=True)
            except Exception as e:
                # Not all devices support IQ balance correction
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

        _do_configure()

    def start_stream(self) -> StreamHandle:
        """Start stream with timeout protection."""
        @with_timeout(10.0)
        def _do_start_stream() -> StreamHandle:
            import SoapySDR  # type: ignore

            # Set antenna: use configured antenna if specified, otherwise use first available
            available_antennas = self.sdr.listAntennas(SoapySDR.SOAPY_SDR_RX, 0)
            print(f"[DEBUG] Device {self.info.driver}: Available antennas: {available_antennas}", flush=True)
            if self._antenna is not None:
                antenna = self._antenna
                print(f"[DEBUG] Device {self.info.driver}: Requested antenna: {antenna}", flush=True)
            else:
                antenna = available_antennas[0] if available_antennas else "RX"
                print(f"[DEBUG] Device {self.info.driver}: Using default antenna: {antenna}", flush=True)
            self.sdr.setAntenna(SoapySDR.SOAPY_SDR_RX, 0, antenna)
            # Verify what was actually set by querying the device
            actual_antenna = self.sdr.getAntenna(SoapySDR.SOAPY_SDR_RX, 0)
            print(f"[DEBUG] Device {self.info.driver}: Antenna set to: {antenna}, device reports: {actual_antenna}", flush=True)
            # Update _antenna to reflect what was actually set
            self._antenna = actual_antenna

            # Select stream format based on configuration
            # Default to CF32 (complex float32) if not specified
            format_map = {
                "CF32": SoapySDR.SOAPY_SDR_CF32,  # Complex float32 (default)
                "CS16": SoapySDR.SOAPY_SDR_CS16,  # Complex int16
                "CS8": SoapySDR.SOAPY_SDR_CS8,    # Complex int8
            }
            stream_fmt = format_map.get(self._stream_format or "CF32", SoapySDR.SOAPY_SDR_CF32)
            print(f"[SOAPY] Using stream format: {self._stream_format or 'CF32'}", flush=True)

            # Setup stream with optimized buffer configuration
            stream = self.sdr.setupStream(SoapySDR.SOAPY_SDR_RX, stream_fmt, [0])

            # Query and log buffer information
            try:
                mtu = self.sdr.getStreamMTU(stream)
                print(f"[SOAPY] Stream MTU: {mtu} samples", flush=True)
            except Exception:
                pass

            try:
                num_direct_bufs = self.sdr.getNumDirectAccessBuffers(stream)
                if num_direct_bufs > 0:
                    print(f"[SOAPY] Direct access buffers: {num_direct_bufs}", flush=True)
            except Exception:
                pass

            # Activate stream with optional buffer size hint
            # Most drivers will use sensible defaults if not specified
            self.sdr.activateStream(stream)
            return _SoapyStream(self.sdr, stream)

        return _do_start_stream()

    def get_antenna(self) -> Optional[str]:
        """Return the currently configured antenna, if any."""
        return self._antenna

    def get_capabilities(self) -> dict[str, Any]:
        """Query and return device capabilities dynamically.

        This provides runtime information about what the device supports,
        useful for validation and debugging.
        """
        import SoapySDR  # type: ignore

        caps = {}

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
                            "max": float(gain_range.maximum())
                        }
                except Exception:
                    pass

            # Query overall gain range
            try:
                overall_gain = self.sdr.getGainRange(SoapySDR.SOAPY_SDR_RX, 0)
                if overall_gain:
                    caps["overall_gain_range"] = {
                        "min": float(overall_gain.minimum()),
                        "max": float(overall_gain.maximum())
                    }
            except Exception:
                pass

            # Query available settings
            try:
                settings_info = self.sdr.getSettingInfo()
                caps["settings"] = {}
                for setting in settings_info:
                    setting_key = setting.key if hasattr(setting, 'key') else str(setting)
                    caps["settings"][setting_key] = {
                        "description": setting.description if hasattr(setting, 'description') else "",
                        "type": setting.type if hasattr(setting, 'type') else "unknown"
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
                    caps["frequency_ranges"].append({
                        "min": float(fr.minimum()),
                        "max": float(fr.maximum())
                    })
            except Exception:
                caps["frequency_ranges"] = []

            # Query sample rate ranges
            try:
                rate_ranges = self.sdr.getSampleRateRange(SoapySDR.SOAPY_SDR_RX, 0)
                caps["sample_rate_ranges"] = []
                for rr in rate_ranges:
                    caps["sample_rate_ranges"].append({
                        "min": float(rr.minimum()),
                        "max": float(rr.maximum())
                    })
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
                    caps["bandwidth_ranges"].append({
                        "min": float(bw.minimum()),
                        "max": float(bw.maximum())
                    })
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
        import SoapySDR  # type: ignore

        sensor_values = {}

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
        center_hz: Optional[float] = None,
        gain: Optional[float] = None,
        bandwidth: Optional[float] = None,
        ppm: Optional[float] = None,
    ) -> None:
        """Reconfigure device while stream is running (hot reconfiguration)."""
        @with_timeout(10.0)
        def _do_reconfigure() -> None:
            import SoapySDR  # type: ignore

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
                try:
                    self.sdr.setBandwidth(SoapySDR.SOAPY_SDR_RX, 0, bandwidth)
                except Exception:
                    pass

            if ppm is not None:
                try:
                    self.sdr.setFrequencyCorrection(SoapySDR.SOAPY_SDR_RX, 0, ppm)
                except Exception:
                    pass

        _do_reconfigure()

    def close(self) -> None:
        """Close device with timeout protection and cooldown for SDRplay."""
        global _sdrplay_last_close_time

        if self.sdr is None:
            return

        is_sdrplay = self.info.driver == "sdrplay"

        @with_timeout(5.0)
        def _unmake() -> None:
            # Properly close the SoapySDR device
            import SoapySDR  # type: ignore
            try:
                SoapySDR.Device.unmake(self.sdr)
            except Exception:
                pass

        # For SDRplay, acquire lock and track close time
        if is_sdrplay:
            with _sdrplay_device_lock:
                try:
                    _unmake()
                except TimeoutError:
                    print(f"Warning: SDRplay device unmake timed out", flush=True)
                except Exception as e:
                    print(f"Warning: SDRplay device unmake failed: {e}", flush=True)
                finally:
                    self.sdr = None  # type: ignore[assignment]
                    _sdrplay_last_close_time = time.time()
        else:
            try:
                _unmake()
            except TimeoutError:
                print(f"Warning: Device unmake timed out", flush=True)
            except Exception as e:
                print(f"Warning: Device unmake failed: {e}", flush=True)
            finally:
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
        # Use multiprocessing to isolate and timeout the enumeration
        # Note: _enumerate_worker is a module-level function so it can be pickled
        queue: multiprocessing.Queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=_enumerate_worker, args=(driver_name, queue))
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
        try:
            sdrplay_results = self._enumerate_driver("sdrplay", timeout=5.0)
            if sdrplay_results:
                results.extend(sdrplay_results)
                # Success - reset timeout counter
                _sdrplay_consecutive_timeouts = 0
                _sdrplay_last_enumeration_success = now
                print(f"[SOAPY] SDRplay enumeration success: {len(sdrplay_results)} device(s)", flush=True)
            else:
                # Empty result could indicate stuck service
                _sdrplay_consecutive_timeouts += 1
                print(f"[SOAPY] SDRplay enumeration returned no devices (timeout count: {_sdrplay_consecutive_timeouts})", flush=True)
        except Exception as e:
            _sdrplay_consecutive_timeouts += 1
            print(f"Warning: Failed to enumerate driver 'sdrplay': {e} (timeout count: {_sdrplay_consecutive_timeouts})", flush=True)

        # Proactive recovery: If SDRplay has timed out multiple times, attempt service restart
        # This is safe during enumeration (no active streams to disrupt)
        sdrplay_found = any(d.driver == "sdrplay" for d in results)
        if not sdrplay_found and _sdrplay_consecutive_timeouts >= _SDRPLAY_MAX_TIMEOUTS_BEFORE_RECOVERY:
            recovery = get_recovery()
            allowed, reason = recovery.can_restart()
            if allowed:
                print(
                    f"[SOAPY] SDRplay stuck detected ({_sdrplay_consecutive_timeouts} consecutive failures), "
                    f"attempting automatic recovery...",
                    flush=True
                )
                success = attempt_recovery(reason="Enumeration timeout - proactive recovery")
                if success:
                    print("[SOAPY] Recovery successful, retrying enumeration...", flush=True)
                    # Retry enumeration after recovery
                    try:
                        sdrplay_results = self._enumerate_driver("sdrplay", timeout=5.0)
                        if sdrplay_results:
                            results.extend(sdrplay_results)
                            _sdrplay_consecutive_timeouts = 0
                            _sdrplay_last_enumeration_success = now
                            print(f"[SOAPY] SDRplay recovery enumeration success: {len(sdrplay_results)} device(s)", flush=True)
                    except Exception as e:
                        print(f"[SOAPY] SDRplay enumeration failed after recovery: {e}", flush=True)
                else:
                    print(f"[SOAPY] Recovery failed: {recovery.stats.last_error}", flush=True)
            else:
                print(f"[SOAPY] Cannot attempt recovery: {reason}", flush=True)
        elif not sdrplay_found:
            print("[SOAPY] No SDRplay devices found - may need to restart SDRplay service", flush=True)

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
        if _sdrplay_health_status and (now - _sdrplay_last_health_check) < _SDRPLAY_HEALTH_CACHE_TTL:
            return True

        # Fresh health check via subprocess enumeration
        for attempt in range(max_recovery_attempts + 1):
            try:
                results = self._enumerate_driver("sdrplay", timeout=3.0)
                if results:
                    _sdrplay_health_status = True
                    _sdrplay_last_health_check = now
                    _sdrplay_consecutive_timeouts = 0
                    print(f"[SOAPY] SDRplay health check passed ({len(results)} device(s))", flush=True)
                    return True
                else:
                    _sdrplay_consecutive_timeouts += 1
                    print(f"[SOAPY] SDRplay health check: no devices (attempt {attempt + 1})", flush=True)
            except Exception as e:
                _sdrplay_consecutive_timeouts += 1
                print(f"[SOAPY] SDRplay health check failed: {e} (attempt {attempt + 1})", flush=True)

            # Try recovery (except on last attempt)
            if attempt < max_recovery_attempts:
                recovery = get_recovery()
                allowed, reason = recovery.can_restart()
                if allowed:
                    print(f"[SOAPY] Attempting recovery ({attempt + 1}/{max_recovery_attempts})...", flush=True)
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

    def _open_device_subprocess(self, args: str, timeout: float = 15.0) -> dict:
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

        queue: multiprocessing.Queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_device_open_worker,
            args=(args, queue)
        )
        process.start()
        process.join(timeout=timeout)

        if process.is_alive():
            # Timeout - kill the subprocess cleanly
            print(f"[SOAPY] Device open timed out after {timeout}s, killing subprocess...", flush=True)
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

        if queue.empty():
            _sdrplay_health_status = False
            raise SDRplayServiceError("Device open failed - no response from subprocess")

        status, data = queue.get()
        if status == "error":
            raise SDRplayServiceError(f"Device open failed: {data}")

        return data

    def open(self, id_or_args: Optional[str] = None) -> Device:
        global _sdrplay_last_close_time

        SoapySDR = self._SoapySDR
        args = id_or_args or self._cfg.device_args or ""

        # Check if this is an SDRplay device
        is_sdrplay = "sdrplay" in args.lower()

        # For SDRplay, use subprocess isolation for reliability
        if is_sdrplay:
            with _sdrplay_device_lock:
                # 1. MANDATORY pre-flight health check (always runs, uses cache)
                if not self._ensure_sdrplay_healthy():
                    raise SDRplayServiceError(
                        "SDRplay service is unresponsive after recovery attempts. "
                        "Try manually: POST /api/v1/devices/sdrplay/restart-service"
                    )

                # 2. Wait for cooldown if needed
                elapsed = time.time() - _sdrplay_last_close_time
                if elapsed < _SDRPLAY_CLOSE_COOLDOWN:
                    wait_time = _SDRPLAY_CLOSE_COOLDOWN - elapsed
                    print(f"[SOAPY] Waiting {wait_time:.1f}s for SDRplay API cooldown", flush=True)
                    time.sleep(wait_time)

                # 3. Verify device opens in subprocess (can be killed if hangs)
                # This protects against the main hang point: SoapySDR.Device()
                print("[SOAPY] Testing device open in subprocess...", flush=True)
                device_info = self._open_device_subprocess(args, timeout=15.0)
                print(f"[SOAPY] Subprocess test successful: {device_info.get('driver', 'unknown')}", flush=True)

                # 4. Now safe to open in main process (service confirmed responsive)
                sdr = SoapySDR.Device(args)
        else:
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
                bandwidth_min = float(bandwidth_range[0].minimum()) if bandwidth_range else None
                bandwidth_max = float(bandwidth_range[0].maximum()) if bandwidth_range else None
                # If min == max, the driver didn't report a useful range - use fallbacks
                if bandwidth_min is not None and bandwidth_max is not None and bandwidth_min >= bandwidth_max:
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
            gains=tuple(g.name for g in []),
            gain_min=gain_min,
            gain_max=gain_max,
            bandwidth_min=bandwidth_min,
            bandwidth_max=bandwidth_max,
            ppm_min=ppm_min,
            ppm_max=ppm_max,
            antennas=antennas,
        )
        return _SoapyDevice(info=info, sdr=sdr)
