"""SDRplay subprocess worker for isolated device control.

The SDRplay API v3.15 has a fundamental limitation: only ONE device can be
selected per process. This worker runs in a separate subprocess to bypass
that limitation, allowing multiple SDRplay devices to be used simultaneously.

Communication:
- SharedMemory: Zero-copy IQ data transfer (ring buffer)
- Pipe: Commands (configure, start, stop, shutdown)
- Pipe: Status updates (overflow, errors, sample count)

The worker owns the SoapySDR device and streams IQ samples to shared memory.
The main process reads from shared memory via SDRplayProxyStream.
"""

from __future__ import annotations

import contextlib
import logging
import logging.handlers
import os
import signal
import struct
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Callable

import numpy as np
from wavecapsdr.typing import NDArrayComplex

# Module-level logger (configured per-process)
logger: logging.Logger | None = None


def _setup_subprocess_logging(device_serial: str) -> logging.Logger:
    """Configure file + IPC logging for subprocess.

    Args:
        device_serial: Serial number of the device (for log file naming)

    Returns:
        Configured logger instance
    """
    # Create log directory
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # Create logger for this worker
    worker_logger = logging.getLogger(f"sdrplay_worker.{os.getpid()}")
    worker_logger.setLevel(logging.DEBUG)

    # Clear any existing handlers
    worker_logger.handlers.clear()

    # File handler with rotation (5MB max, 3 backups)
    log_file = log_dir / f"sdrplay_worker_{device_serial}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] [PID:%(process)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    worker_logger.addHandler(file_handler)

    # Also log to stderr (captured by parent if configured)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.INFO)
    stderr_handler.setFormatter(logging.Formatter(
        '[SDRplayWorker:%(process)d] %(levelname)s: %(message)s'
    ))
    worker_logger.addHandler(stderr_handler)

    return worker_logger


def _log_to_pipe(status_pipe: Connection, level: str, message: str) -> None:
    """Send log message to main process via IPC pipe.

    Args:
        status_pipe: Connection to main process
        level: Log level (debug, info, warning, error)
        message: Log message
    """
    try:
        status_pipe.send({
            "type": "log",
            "level": level,
            "message": message,
        })
    except Exception:
        pass  # Don't crash if pipe is broken


# Ring buffer header format (64 bytes total):
# - write_idx: uint64 (8 bytes) - samples written by worker
# - read_idx: uint64 (8 bytes) - samples consumed by main process (unused for now)
# - sample_count: uint64 (8 bytes) - total samples written (for stats)
# - overflow_count: uint64 (8 bytes) - number of overflows
# - sample_rate: uint32 (4 bytes)
# - flags: uint32 (4 bytes) - bit 0: overflow, bit 1: error, bit 2: running
# - timestamp: float64 (8 bytes) - last write time
# - reserved: 16 bytes padding
HEADER_SIZE = 64
HEADER_FORMAT = "<QQQQIId16x"  # 8+8+8+8+4+4+8+16 = 64 bytes

# Buffer size for IQ samples (64MB = 8,388,608 complex64 samples)
# At 6 MHz sample rate, this is ~1.4s of buffer for DSP/CPU bursts.
BUFFER_SAMPLES = 8 * 1024 * 1024
BUFFER_SIZE = BUFFER_SAMPLES * 8  # complex64 = 8 bytes per sample

# Total shared memory size
SHM_SIZE = HEADER_SIZE + BUFFER_SIZE

# Flag bits
FLAG_OVERFLOW = 1 << 0
FLAG_ERROR = 1 << 1
FLAG_RUNNING = 1 << 2
FLAG_DATA_READY = 1 << 3  # Set after first successful sample write


@dataclass
class WorkerConfig:
    """Configuration sent to worker subprocess."""
    device_args: str
    shm_name: str
    center_hz: float = 100e6
    sample_rate: int = 2_000_000
    gain: float | None = None
    bandwidth: float | None = None
    ppm: float | None = None
    antenna: str | None = None
    dc_offset_auto: bool = True
    iq_balance_auto: bool = True
    device_settings: dict[str, Any] | None = None
    element_gains: dict[str, float] | None = None
    stream_format: str | None = None


def _write_header(
    shm: SharedMemory,
    write_idx: int = 0,
    read_idx: int = 0,
    sample_count: int = 0,
    overflow_count: int = 0,
    sample_rate: int = 0,
    flags: int = 0,
    timestamp: float = 0.0,
) -> None:
    """Write header to shared memory."""
    if shm.buf is None:
        raise RuntimeError("Shared memory buffer is not available")
    struct.pack(
        HEADER_FORMAT,
        write_idx,
        read_idx,
        sample_count,
        overflow_count,
        sample_rate,
        flags,
        timestamp,
    )
    # Use struct.pack_into to write directly to the buffer
    struct.pack_into(
        HEADER_FORMAT,
        shm.buf,
        0,
        write_idx,
        read_idx,
        sample_count,
        overflow_count,
        sample_rate,
        flags,
        timestamp,
    )


def _read_header(shm: SharedMemory) -> tuple[int, int, int, int, int, int, float] | None:
    """Read header from shared memory.

    Returns: (write_idx, read_idx, sample_count, overflow_count, sample_rate, flags, timestamp)
             or None if the shared memory buffer is not available
    """
    buf = getattr(shm, "buf", None)
    if buf is None:
        return None
    try:
        header = bytes(buf[:HEADER_SIZE])
        return struct.unpack(HEADER_FORMAT, header)
    except (TypeError, ValueError, AttributeError):
        # Buffer became invalid
        return None


def sdrplay_worker_main(
    device_args: str,
    shm_name: str,
    cmd_pipe: Connection,
    status_pipe: Connection,
) -> None:
    """Main entry point for SDRplay worker subprocess.

    This function runs in an isolated process and has exclusive access to
    one SDRplay device via the SDRplay API.

    Args:
        device_args: SoapySDR device arguments string (e.g., "driver=sdrplay,serial=...")
        shm_name: Name of shared memory region for IQ data
        cmd_pipe: Pipe for receiving commands from main process
        status_pipe: Pipe for sending status updates to main process
    """
    global logger

    # Extract serial from device_args for log file naming
    serial = "unknown"
    for part in device_args.split(","):
        if part.strip().startswith("serial="):
            serial = part.strip().split("=")[1]
            break

    # Setup logging FIRST
    logger = _setup_subprocess_logging(serial)
    logger.info(f"Worker starting for device: {device_args}")
    logger.info(f"Shared memory name: {shm_name}")
    logger.info(f"PID: {os.getpid()}")

    # Forward startup to main process
    _log_to_pipe(status_pipe, "info", f"Worker started for device serial={serial}")

    # Ignore SIGINT in worker - let main process handle shutdown
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    worker = SDRplayWorker(device_args, shm_name, cmd_pipe, status_pipe)
    try:
        worker.run()
    except Exception as e:
        error_msg = f"Worker crashed: {e}\n{traceback.format_exc()}"
        logger.error(error_msg)
        _log_to_pipe(status_pipe, "error", error_msg)
        with contextlib.suppress(Exception):
            status_pipe.send({"type": "error", "message": str(e)})
    finally:
        logger.info("Worker shutting down, cleaning up...")
        worker.cleanup()
        logger.info("Worker cleanup complete")


class SDRplayWorker:
    """Subprocess worker for isolated SDRplay device control.

    Runs in a separate process to bypass SDRplay API single-device limit.
    """

    def __init__(
        self,
        device_args: str,
        shm_name: str,
        cmd_pipe: Connection,
        status_pipe: Connection,
    ):
        self.device_args = device_args
        self.shm_name = shm_name
        self.cmd_pipe = cmd_pipe
        self.status_pipe = status_pipe

        self.sdr: Any | None = None
        self.stream: Any | None = None
        self.shm: SharedMemory | None = None
        self.running = False
        self.streaming = False

        # Stats
        self.sample_count = 0
        self.overflow_count = 0
        self.write_idx = 0

        # Config
        self.sample_rate = 2_000_000
        self.center_hz = 100e6
        self._read_buffer: np.ndarray | None = None
        self._iq_buffer: np.ndarray | None = None

    def run(self) -> None:
        """Main loop: process commands and stream IQ to shared memory."""
        # Attach to shared memory (created by main process)
        self.shm = SharedMemory(name=self.shm_name)
        if self.shm.buf is not None:
            self._iq_buffer = np.ndarray(
                shape=(BUFFER_SAMPLES,),
                dtype=np.complex64,
                buffer=self.shm.buf,
                offset=HEADER_SIZE,
            )

        # Initialize header
        _write_header(
            self.shm,
            write_idx=0,
            sample_rate=self.sample_rate,
            flags=0,
            timestamp=time.time(),
        )

        self.status_pipe.send({"type": "ready"})
        self.running = True

        loop_count = 0
        while self.running:
            loop_count += 1
            if loop_count <= 5 or loop_count % 10000 == 0:
                if logger:
                    logger.debug(f"Main loop[{loop_count}]: running={self.running}, streaming={self.streaming}, stream={self.stream is not None}")

            # Check for commands (non-blocking)
            while self.cmd_pipe.poll(timeout=0.001):
                try:
                    cmd = self.cmd_pipe.recv()
                    if logger:
                        logger.debug(f"Received command: {cmd.get('type', 'unknown')}")
                    self._handle_command(cmd)
                except EOFError:
                    self.running = False
                    break

            # Stream IQ samples if active
            if self.streaming and self.stream is not None:
                try:
                    self._stream_samples()
                except Exception as e:
                    if logger:
                        logger.error(f"_stream_samples exception: {e}")
                    self.status_pipe.send({"type": "stream_error", "message": str(e)})
                    self.streaming = False

    def _handle_command(self, cmd: dict[str, Any]) -> None:
        """Process a command from the main process."""
        cmd_type = cmd.get("type", "")

        if cmd_type == "open":
            self._open_device()
        elif cmd_type == "configure":
            self._configure(cmd)
        elif cmd_type == "start":
            self._start_stream()
        elif cmd_type == "stop":
            self._stop_stream()
        elif cmd_type == "shutdown":
            self.running = False
        else:
            self.status_pipe.send({"type": "error", "message": f"Unknown command: {cmd_type}"})

    def _open_device(self) -> None:
        """Open SoapySDR device."""
        try:
            import SoapySDR

            if logger:
                logger.info(f"Opening device: {self.device_args}")
            _log_to_pipe(self.status_pipe, "info", f"Opening device: {self.device_args}")

            sdr = SoapySDR.Device(self.device_args)
            self.sdr = sdr

            # Query device info
            driver = str(sdr.getDriverKey())
            hardware = str(sdr.getHardwareKey())
            antennas: list[str] = list(sdr.listAntennas(SoapySDR.SOAPY_SDR_RX, 0))

            if logger:
                logger.info(f"Device opened: driver={driver}, hardware={hardware}, antennas={antennas}")

            self.status_pipe.send({
                "type": "opened",
                "driver": driver,
                "hardware": hardware,
                "antennas": antennas,
            })

        except Exception as e:
            error_msg = f"Failed to open device: {e}\n{traceback.format_exc()}"
            if logger:
                logger.error(error_msg)
            _log_to_pipe(self.status_pipe, "error", f"Failed to open device: {e}")
            self.status_pipe.send({"type": "open_error", "message": str(e)})
            raise

    def _readback_settings(
        self,
        *,
        expected_sample_rate: int | None,
        expected_center_hz: float | None,
        expected_gain: float | None,
        expected_bandwidth: float | None,
        expected_ppm: float | None,
        expected_antenna: str | None,
        expected_device_settings: dict[str, Any] | None,
        expected_element_gains: dict[str, float] | None,
    ) -> None:
        """Query the device after configuration to confirm applied settings."""
        if self.sdr is None:
            return

        import SoapySDR
        sdr = self.sdr
        assert sdr is not None

        def _safe_get(getter: Callable[[], Any]) -> Any | None:
            with contextlib.suppress(Exception):
                return getter()
            return None

        def _check_numeric(
            label: str,
            expected: float | int | None,
            getter: Callable[[], float | int],
            tolerance: float,
            unit: str,
            precision: int = 2,
        ) -> tuple[str, str | None]:
            actual = _safe_get(lambda: float(getter()))
            if actual is None:
                return "", None
            actual_text = f"{actual:.{precision}f}{unit}" if isinstance(actual, float) else f"{actual}{unit}"
            mismatch: str | None = None
            if expected is not None and abs(float(actual) - float(expected)) > tolerance:
                mismatch = f"{label} expected {expected}{unit}, got {actual_text}"
            return f"{label}={actual_text}", mismatch

        readback_parts: list[str] = []
        mismatches: list[str] = []

        text, mismatch = _check_numeric(
            "sample_rate",
            expected_sample_rate,
            lambda: sdr.getSampleRate(SoapySDR.SOAPY_SDR_RX, 0),
            tolerance=1.0,
            unit="Hz",
            precision=0,
        )
        if text:
            readback_parts.append(text)
        if mismatch:
            mismatches.append(mismatch)

        text, mismatch = _check_numeric(
            "center_hz",
            expected_center_hz,
            lambda: sdr.getFrequency(SoapySDR.SOAPY_SDR_RX, 0),
            tolerance=1.0,
            unit="Hz",
            precision=1,
        )
        if text:
            readback_parts.append(text)
        if mismatch:
            mismatches.append(mismatch)

        text, mismatch = _check_numeric(
            "gain",
            expected_gain,
            lambda: sdr.getGain(SoapySDR.SOAPY_SDR_RX, 0),
            tolerance=0.25,
            unit="dB",
        )
        if text:
            readback_parts.append(text)
        if mismatch:
            mismatches.append(mismatch)

        text, mismatch = _check_numeric(
            "bandwidth",
            expected_bandwidth,
            lambda: sdr.getBandwidth(SoapySDR.SOAPY_SDR_RX, 0),
            tolerance=1.0,
            unit="Hz",
            precision=0,
        )
        if text:
            readback_parts.append(text)
        if mismatch:
            mismatches.append(mismatch)

        text, mismatch = _check_numeric(
            "ppm",
            expected_ppm,
            lambda: sdr.getFrequencyCorrection(SoapySDR.SOAPY_SDR_RX, 0),
            tolerance=0.05,
            unit="ppm",
        )
        if text:
            readback_parts.append(text)
        if mismatch:
            mismatches.append(mismatch)

        actual_antenna = _safe_get(lambda: str(sdr.getAntenna(SoapySDR.SOAPY_SDR_RX, 0)))
        if actual_antenna is not None:
            readback_parts.append(f"antenna={actual_antenna}")
            if expected_antenna is not None and actual_antenna != expected_antenna:
                mismatches.append(f"antenna expected {expected_antenna}, got {actual_antenna}")

        for key, expected_value in (expected_device_settings or {}).items():
            actual_value = _safe_get(lambda: sdr.readSetting(key))
            if actual_value is None:
                continue
            actual_text = str(actual_value)
            readback_parts.append(f"setting[{key}]={actual_text}")
            if str(expected_value) != actual_text:
                mismatches.append(f"setting {key} expected {expected_value}, got {actual_text}")

        for element, expected_gain in (expected_element_gains or {}).items():
            actual_element_gain = _safe_get(lambda: float(sdr.getGain(SoapySDR.SOAPY_SDR_RX, 0, element)))
            if actual_element_gain is None:
                continue
            readback_parts.append(f"gain[{element}]={actual_element_gain:.2f}dB")
            if abs(actual_element_gain - expected_gain) > 0.25:
                mismatches.append(
                    f"{element} gain expected {expected_gain}dB, got {actual_element_gain:.2f}dB"
                )

        if readback_parts:
            if logger:
                logger.info("Readback: %s", ", ".join(readback_parts))
        if mismatches:
            warning_msg = f"Setting mismatches detected: {'; '.join(mismatches)}"
            if logger:
                logger.warning(warning_msg)
            _log_to_pipe(self.status_pipe, "warning", warning_msg)

    def _configure(self, cmd: dict[str, Any]) -> None:
        """Configure the device."""
        if self.sdr is None:
            if logger:
                logger.error("Configure called but device not open")
            self.status_pipe.send({"type": "error", "message": "Device not open"})
            return

        sdr = self.sdr  # Local reference for type checker
        try:
            import SoapySDR

            # Extract configuration
            center_hz = cmd.get("center_hz", self.center_hz)
            sample_rate = cmd.get("sample_rate", self.sample_rate)
            gain = cmd.get("gain")
            bandwidth = cmd.get("bandwidth")
            ppm = cmd.get("ppm")
            antenna = cmd.get("antenna")
            dc_offset_auto = cmd.get("dc_offset_auto", True)
            iq_balance_auto = cmd.get("iq_balance_auto", True)
            device_settings = cmd.get("device_settings", {})
            element_gains = cmd.get("element_gains", {})
            agc_enabled = cmd.get("agc_enabled", False)

            if logger:
                logger.info(f"Configuring: center={center_hz/1e6:.3f}MHz, rate={sample_rate/1e6:.3f}MHz, antenna={antenna}")
                logger.debug(f"Full config: gain={gain}, bandwidth={bandwidth}, ppm={ppm}")
                logger.debug(f"Device settings: {device_settings}, element_gains: {element_gains}")

            # Apply configuration
            if logger:
                logger.debug("Setting sample rate...")
            sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, float(sample_rate))
            if logger:
                logger.debug("Setting frequency...")
            sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, center_hz)

            if device_settings:
                if logger:
                    logger.info(f"Applying device settings: {device_settings}")
                for key, value in device_settings.items():
                    try:
                        sdr.writeSetting(key, str(value))
                        if logger:
                            logger.info(f"Applied setting {key}={value}")
                    except Exception as e:
                        if logger:
                            logger.warning(f"Setting {key}={value} failed: {e}")

            if element_gains:
                # Disable AGC first so we can set element gains
                with contextlib.suppress(Exception):
                    sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, False)
                if logger:
                    logger.info(f"Applying element gains: {element_gains}")
                for elem, g in element_gains.items():
                    try:
                        sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, elem, g)
                        if logger:
                            logger.info(f"Set gain element {elem}={g}")
                    except Exception as e:
                        if logger:
                            logger.warning(f"Gain {elem}={g} failed: {e}")
                # Read back to verify
                try:
                    ifgr = sdr.getGain(SoapySDR.SOAPY_SDR_RX, 0, "IFGR")
                    rfgr = sdr.getGain(SoapySDR.SOAPY_SDR_RX, 0, "RFGR")
                    if logger:
                        logger.info(f"Element gains applied: IFGR={ifgr:.0f}, RFGR={rfgr:.0f}")
                except Exception:
                    pass
                # Now enable AGC if requested (AGC will control IFGR, RFGR stays fixed)
                if agc_enabled:
                    try:
                        sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, True)
                        if logger:
                            logger.info("AGC enabled (controls IFGR, RFGR stays at configured value)")
                    except Exception as e:
                        if logger:
                            logger.warning(f"Failed to enable AGC: {e}")
            elif gain is not None:
                with contextlib.suppress(Exception):
                    sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, False)
                sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, gain)
                # Read back gain distribution to verify driver mapping
                try:
                    actual_gain = sdr.getGain(SoapySDR.SOAPY_SDR_RX, 0)
                    ifgr = sdr.getGain(SoapySDR.SOAPY_SDR_RX, 0, "IFGR")
                    rfgr = sdr.getGain(SoapySDR.SOAPY_SDR_RX, 0, "RFGR")
                    if logger:
                        logger.info(f"Gain set: requested={gain:.1f}dB -> actual={actual_gain:.1f}dB (IFGR={ifgr:.0f}, RFGR={rfgr:.0f})")
                except Exception as e:
                    if logger:
                        logger.info(f"Set overall gain={gain}dB (readback failed: {e})")
                # Enable AGC if requested
                if agc_enabled:
                    try:
                        sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, True)
                        if logger:
                            logger.info("AGC enabled after setting initial gain")
                    except Exception as e:
                        if logger:
                            logger.warning(f"Failed to enable AGC: {e}")
            elif agc_enabled:
                # No specific gain, just enable AGC
                try:
                    sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, True)
                    if logger:
                        logger.info("AGC enabled (automatic gain control)")
                except Exception as e:
                    if logger:
                        logger.warning(f"Failed to enable AGC: {e}")
            else:
                # No gain or AGC specified - enable AGC by default
                try:
                    sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, True)
                    if logger:
                        logger.debug("Enabled AGC mode (default)")
                except Exception:
                    pass

            if bandwidth is not None:
                try:
                    sdr.setBandwidth(SoapySDR.SOAPY_SDR_RX, 0, bandwidth)
                    if logger:
                        logger.debug(f"Set bandwidth={bandwidth}")
                except Exception as e:
                    if logger:
                        logger.warning(f"setBandwidth failed: {e}")

            if ppm is not None:
                try:
                    sdr.setFrequencyCorrection(SoapySDR.SOAPY_SDR_RX, 0, ppm)
                    if logger:
                        logger.debug(f"Set PPM correction={ppm}")
                except Exception as e:
                    if logger:
                        logger.warning(f"setFrequencyCorrection failed: {e}")

            with contextlib.suppress(Exception):
                sdr.setDCOffsetMode(SoapySDR.SOAPY_SDR_RX, 0, dc_offset_auto)

            with contextlib.suppress(Exception):
                sdr.setIQBalanceMode(SoapySDR.SOAPY_SDR_RX, 0, iq_balance_auto)

            if antenna is not None:
                sdr.setAntenna(SoapySDR.SOAPY_SDR_RX, 0, antenna)
                if logger:
                    logger.debug(f"Set antenna={antenna}")

            self.sample_rate = sample_rate
            self.center_hz = center_hz
            self._readback_settings(
                expected_sample_rate=sample_rate,
                expected_center_hz=center_hz,
                expected_gain=gain,
                expected_bandwidth=bandwidth,
                expected_ppm=ppm,
                expected_antenna=antenna,
                expected_device_settings=device_settings,
                expected_element_gains=element_gains,
            )

            # Update header with sample rate
            if self.shm is None:
                raise RuntimeError("Shared memory not initialized")
            _write_header(
                self.shm,
                write_idx=self.write_idx,
                sample_count=self.sample_count,
                overflow_count=self.overflow_count,
                sample_rate=self.sample_rate,
                flags=FLAG_RUNNING if self.streaming else 0,
                timestamp=time.time(),
            )

            self.status_pipe.send({"type": "configured"})
            if logger:
                logger.info("Configuration complete")

        except Exception as e:
            error_msg = f"Configure failed: {e}\n{traceback.format_exc()}"
            if logger:
                logger.error(error_msg)
            _log_to_pipe(self.status_pipe, "error", f"Configure failed: {e}")
            self.status_pipe.send({"type": "configure_error", "message": str(e)})
            raise

    def _start_stream(self) -> None:
        """Start IQ streaming."""
        if self.sdr is None:
            if logger:
                logger.error("Start stream called but device not open")
            self.status_pipe.send({"type": "error", "message": "Device not open"})
            return

        sdr = self.sdr  # Local reference for type checker
        if self.streaming:
            if logger:
                logger.debug("Start stream called but already streaming - returning current state")
            # Still send success response since we're in the desired state
            try:
                import SoapySDR
                current_antenna = sdr.getAntenna(SoapySDR.SOAPY_SDR_RX, 0)
            except Exception:
                current_antenna = None
            self.status_pipe.send({"type": "started", "antenna": current_antenna})
            return

        try:
            import SoapySDR

            if logger:
                logger.info("Starting stream...")
            _log_to_pipe(self.status_pipe, "info", "Starting IQ stream")

            # Set default antenna if not set
            antennas = sdr.listAntennas(SoapySDR.SOAPY_SDR_RX, 0)
            current_antenna = sdr.getAntenna(SoapySDR.SOAPY_SDR_RX, 0)
            if logger:
                logger.info(f"Antenna: current={current_antenna}, available={antennas}")

            # Setup stream with retry for transient SDRplay API errors
            # Error -5 (sdrplay_api_Fail) often occurs when:
            # - Another process just released the device
            # - USB settling time needed
            # - API cleanup still in progress
            max_retries = 3
            retry_delay = 0.5  # seconds, doubles each retry

            for attempt in range(max_retries):
                if logger:
                    logger.debug("Calling setupStream...")
                self.stream = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32, [0])
                if logger:
                    logger.debug("Calling activateStream...")
                result = sdr.activateStream(self.stream)

                if result == 0:
                    # Success!
                    break

                # activateStream failed - clean up and potentially retry
                if logger:
                    logger.warning(f"activateStream failed with code {result} (attempt {attempt + 1}/{max_retries})")
                sdr.closeStream(self.stream)
                self.stream = None

                # Error -5 is sdrplay_api_Fail - often transient, worth retrying
                if result == -5 and attempt < max_retries - 1:
                    delay = retry_delay * (2 ** attempt)
                    if logger:
                        logger.info(f"Retrying in {delay:.1f}s...")
                    _log_to_pipe(self.status_pipe, "warning", f"Stream activation failed, retrying in {delay:.1f}s")
                    time.sleep(delay)
                else:
                    # Non-retryable error or exhausted retries
                    raise RuntimeError(f"activateStream failed with error code {result}")

            self.streaming = True
            self.write_idx = 0
            self.sample_count = 0
            self.overflow_count = 0

            # Update header
            if self.shm is None:
                raise RuntimeError("Shared memory not initialized")
            _write_header(
                self.shm,
                write_idx=0,
                sample_count=0,
                overflow_count=0,
                sample_rate=self.sample_rate,
                flags=FLAG_RUNNING,
                timestamp=time.time(),
            )

            self.status_pipe.send({"type": "started", "antenna": current_antenna})
            if logger:
                logger.info(f"Stream started successfully, rate={self.sample_rate}")
            _log_to_pipe(self.status_pipe, "info", f"Stream started at {self.sample_rate/1e6:.1f} MHz")

        except Exception as e:
            error_msg = f"Start stream failed: {e}\n{traceback.format_exc()}"
            if logger:
                logger.error(error_msg)
            _log_to_pipe(self.status_pipe, "error", f"Start stream failed: {e}")
            self.status_pipe.send({"type": "start_error", "message": str(e)})
            raise

    def _stop_stream(self) -> None:
        """Stop IQ streaming."""
        if not self.streaming or self.stream is None or self.sdr is None:
            return

        sdr = self.sdr  # Local reference for type checker
        try:
            if logger:
                logger.info("Stopping stream...")
            self.streaming = False

            sdr.deactivateStream(self.stream)
            sdr.closeStream(self.stream)
            self.stream = None

            # Update header
            if self.shm is not None:
                _write_header(
                    self.shm,
                    write_idx=self.write_idx,
                    sample_count=self.sample_count,
                    overflow_count=self.overflow_count,
                    sample_rate=self.sample_rate,
                    flags=0,
                    timestamp=time.time(),
                )

            self.status_pipe.send({"type": "stopped"})
            if logger:
                logger.info(f"Stream stopped, total samples: {self.sample_count}, overflows: {self.overflow_count}")

        except Exception as e:
            if logger:
                logger.error(f"Error stopping stream: {e}")
            self.status_pipe.send({"type": "stop_error", "message": str(e)})

    def _stream_samples(self) -> None:
        """Read samples from device and write to shared memory."""
        if self.stream is None or self.sdr is None or self.shm is None:
            return

        sdr = self.sdr  # Local reference for type checker
        # Read samples from device
        chunk_size = min(32768, BUFFER_SAMPLES // 8)  # Larger read reduces per-call overhead
        if self._read_buffer is None or self._read_buffer.size < chunk_size:
            self._read_buffer = np.empty(chunk_size, dtype=np.complex64)
        buff = self._read_buffer[:chunk_size]

        sr = sdr.readStream(self.stream, [buff.view(np.float32)], chunk_size, flags=0)

        # Handle StreamResult
        if hasattr(sr, 'ret'):
            ret = sr.ret
            flags = sr.flags if hasattr(sr, 'flags') else 0
        elif isinstance(sr, tuple):
            ret = sr[0]
            flags = sr[1] if len(sr) > 1 else 0
        else:
            ret = sr
            flags = 0

        # Log sample reads periodically (every 1000 calls or on error)
        self._stream_call_count = getattr(self, '_stream_call_count', 0) + 1
        if ret <= 0 or self._stream_call_count <= 5 or self._stream_call_count % 1000 == 0:
            if logger:
                logger.debug(f"_stream_samples[{self._stream_call_count}]: ret={ret}, flags={flags}")

        if ret <= 0:
            return

        # Check for overflow
        SOAPY_SDR_READ_FLAG_OVERFLOW = (1 << 1)
        if flags & SOAPY_SDR_READ_FLAG_OVERFLOW:
            self.overflow_count += 1

        # Write samples to ring buffer using numpy array backed by shared memory
        samples = buff[:ret]
        num_samples = ret

        # Create numpy array view of the IQ buffer (after header)
        iq_buffer = self._iq_buffer
        if iq_buffer is None:
            buf = getattr(self.shm, "buf", None)
            if buf is None:
                return
            iq_buffer = np.ndarray(
                shape=(BUFFER_SAMPLES,),
                dtype=np.complex64,
                buffer=buf,
                offset=HEADER_SIZE,
            )
            self._iq_buffer = iq_buffer

        # Calculate write position in ring buffer (wrap around)
        write_pos = self.write_idx % BUFFER_SAMPLES

        # Handle wrap-around
        samples_to_end = BUFFER_SAMPLES - write_pos
        if num_samples <= samples_to_end:
            # Single write (no wrap)
            iq_buffer[write_pos:write_pos + num_samples] = samples
        else:
            # Split write (wrap around)
            iq_buffer[write_pos:] = samples[:samples_to_end]
            iq_buffer[:num_samples - samples_to_end] = samples[samples_to_end:]

        # Update indices
        self.write_idx += num_samples
        self.sample_count += num_samples

        # Update header
        header_flags = FLAG_RUNNING | (FLAG_OVERFLOW if self.overflow_count > 0 else 0) | (FLAG_DATA_READY if self.sample_count > 0 else 0)
        _write_header(
            self.shm,
            write_idx=self.write_idx,
            sample_count=self.sample_count,
            overflow_count=self.overflow_count,
            sample_rate=self.sample_rate,
            flags=header_flags,
            timestamp=time.time(),
        )

        # Force memory synchronization - Python's mmap on macOS may need this
        if self.shm is not None and self.shm.buf is not None:
            with contextlib.suppress(Exception):
                if hasattr(self.shm.buf, "flush"):
                    self.shm.buf.flush()

        # Log first few header writes for debugging
        if self._stream_call_count <= 5:
            if logger:
                logger.debug(f"Header write: shm={self.shm_name}, write_idx={self.write_idx}, flags={header_flags}")

    def cleanup(self) -> None:
        """Clean up resources."""
        if logger:
            logger.info("Cleaning up resources...")

        if self.streaming:
            self._stop_stream()

        if self.sdr is not None:
            try:
                import SoapySDR
                if logger:
                    logger.debug("Calling SoapySDR.Device.unmake...")
                SoapySDR.Device.unmake(self.sdr)
                if logger:
                    logger.debug("Device unmake succeeded")
            except Exception as e:
                if logger:
                    logger.error(f"Device unmake error: {e}")
            self.sdr = None

        if self.shm is not None:
            with contextlib.suppress(Exception):
                self.shm.close()
            self.shm = None

        if logger:
            logger.info("Cleanup complete")
