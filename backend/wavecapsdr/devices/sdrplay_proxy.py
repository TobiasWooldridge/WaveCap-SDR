"""SDRplay proxy device for subprocess-isolated SDR control.

Presents the same Device interface as _SoapyDevice but routes all calls
to a subprocess worker via IPC. This allows multiple SDRplay devices to
run simultaneously by bypassing the SDRplay API's single-device limitation.

Architecture:
    Main Process                          Subprocess (Worker)
    ─────────────────────────────────────────────────────────────
    SDRplayProxyDevice
    ├── cmd_pipe ─────────────────────→  SDRplayWorker
    ├── status_pipe ←─────────────────   (owns SoapySDR device)
    └── SharedMemory (IQ data) ←──────
                                         writes IQ samples

    SDRplayProxyStream
    └── reads from SharedMemory (zero-copy)
"""

from __future__ import annotations

import contextlib
import logging
import time
import uuid
from dataclasses import dataclass
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import numpy as np
from wavecapsdr.typing import NDArrayComplex

from .base import Device, DeviceInfo, StreamHandle
from .sdrplay_lock import acquire_sdrplay_lock, release_sdrplay_lock
from .sdrplay_worker import (
    BUFFER_SAMPLES,
    FLAG_DATA_READY,
    HEADER_SIZE,
    SHM_SIZE,
    _read_header,
    sdrplay_worker_main,
)
from .soapy import decrement_sdrplay_active_captures, increment_sdrplay_active_captures

logger = logging.getLogger(__name__)

# Global SDRplay operations are serialized via an inter-process lock
# (implemented in sdrplay_lock.py).


def _generate_shm_name() -> str:
    """Generate a unique shared memory name."""
    return f"wavecap_sdrplay_{uuid.uuid4().hex[:8]}"


@dataclass
class SDRplayProxyStream(StreamHandle):
    """Stream handle that reads IQ samples from subprocess's shared memory.

    The worker subprocess writes IQ samples to a ring buffer in shared memory.
    This stream reads from that buffer with minimal copying.
    """

    shm: SharedMemory
    status_pipe: Connection
    sample_rate: int = 2_000_000
    _last_read_idx: int = 0
    _closed: bool = False
    _data_ready: bool = False
    _last_data_time: float = 0.0  # Time when we last received data

    _debug_counter: int = 0
    _last_overflow_count: int = 0

    def is_ready(self) -> bool:
        """Check if worker has written first samples (non-blocking).

        Returns:
            True if data is ready, False if still waiting for first samples.
        """
        if self._data_ready:
            return True
        header = _read_header(self.shm)
        if header is None:
            return False
        write_idx, _, sample_count, _, _, flags, _ = header
        logger.info(
            f"is_ready() check: write_idx={write_idx}, sample_count={sample_count}, flags={flags}, shm={self.shm.name}, stream_id={id(self)}"
        )
        if flags & FLAG_DATA_READY:
            self._data_ready = True
            return True
        return False

    def read(self, num_samples: int) -> tuple[NDArrayComplex, bool]:
        """Read IQ samples from shared memory ring buffer.

        Args:
            num_samples: Number of complex samples to read

        Returns:
            Tuple of (samples array, overflow flag)
        """
        self._debug_counter += 1

        # CRITICAL DEBUG: Log at start to confirm read() is being called
        if self._debug_counter <= 30:
            logger.info(f"SDRplayProxyStream.read() ENTRY: counter={self._debug_counter}")

        if self._closed:
            if self._debug_counter <= 10 or self._debug_counter % 1000 == 0:
                logger.warning(
                    f"SDRplayProxyStream.read()[{self._debug_counter}]: stream is closed"
                )
            return np.empty(0, dtype=np.complex64), False

        # Read header to get current write position
        header = _read_header(self.shm)
        if header is None:
            # Shared memory buffer not available - log only periodically
            self._buf_none_count = getattr(self, "_buf_none_count", 0) + 1
            if self._buf_none_count <= 5 or self._buf_none_count % 5000 == 0:
                logger.warning(
                    f"SDRplayProxyStream.read()[{self._debug_counter}]: "
                    f"shared memory buffer unavailable (count={self._buf_none_count})"
                )
            return np.empty(0, dtype=np.complex64), False
        buf = getattr(self.shm, "buf", None)
        if buf is None:
            self._buf_none_count = getattr(self, "_buf_none_count", 0) + 1
            if self._buf_none_count <= 5 or self._buf_none_count % 5000 == 0:
                logger.warning(
                    f"SDRplayProxyStream.read()[{self._debug_counter}]: "
                    f"shared memory buffer unavailable (count={self._buf_none_count})"
                )
            return np.empty(0, dtype=np.complex64), False
        self._buf_none_count = 0  # Reset counter on successful read

        write_idx, _, sample_count, overflow_count, sample_rate, flags, _timestamp = header

        # Debug logging for first few reads and then very infrequently
        if self._debug_counter <= 5 or self._debug_counter % 100000 == 0:
            logger.debug(
                f"SDRplayProxyStream.read()[{self._debug_counter}]: "
                f"write_idx={write_idx}, _last_read_idx={self._last_read_idx}, "
                f"requested={num_samples}, sample_count={sample_count}, flags={flags}, stream_id={id(self)}, shm={self.shm.name}"
            )

        # Calculate available samples
        available = write_idx - self._last_read_idx
        if available <= 0:
            # No new samples yet - track consecutive empty reads
            self._empty_reads = getattr(self, "_empty_reads", 0) + 1

            # Workaround for macOS SharedMemory coherency issue:
            # Re-attach to the shared memory to force a fresh view when:
            # 1. Early startup: worker reports ready but we see stale header
            # 2. After initial data: coherency broke, need to recover quickly
            import time as time_mod

            now = time_mod.time()
            time_since_data = now - self._last_data_time if self._last_data_time > 0 else 0

            # Be AGGRESSIVE with re-attach after we've received data before:
            # - If we had data, re-attach every 0.3 seconds while empty (not every 2s)
            # - If startup, re-attach after 30 empty reads (~0.6s)
            last_reattach = getattr(self, "_last_reattach_time", 0)
            time_since_reattach = now - last_reattach if last_reattach > 0 else float("inf")

            should_reattach = (
                # Throttle: don't re-attach more than once per 0.2 seconds
                time_since_reattach > 0.2
                and (
                    # Case 1: Early startup - worker reports ready but we see stale header
                    (self._empty_reads >= 30 and (flags & FLAG_DATA_READY))
                    or
                    # Case 2: After initial data - coherency broke, re-attach quickly
                    (self._last_data_time > 0 and time_since_data > 0.3 and self._empty_reads >= 5)
                )
            )
            # Debug: log every 50 empty reads to understand why re-attach isn't triggering
            if self._empty_reads % 50 == 0:
                case1 = self._empty_reads >= 30 and (flags & FLAG_DATA_READY)
                case2 = (
                    self._last_data_time > 0 and time_since_data > 0.3 and self._empty_reads >= 5
                )
                throttle_ok = time_since_reattach > 0.2
                logger.info(
                    f"SDRplayProxyStream: DEBUG empty={self._empty_reads}, throttle_ok={throttle_ok}, "
                    f"case1={case1}, case2={case2}, should_reattach={should_reattach}, flags={flags}"
                )
            if should_reattach:
                reattach_count = getattr(self, "_reattach_count", 0)
                # Allow many more reattach attempts for ongoing operation (100 vs 5 during startup)
                max_reattach = 100 if self._last_data_time > 0 else 5
                if reattach_count < max_reattach:
                    self._reattach_count = reattach_count + 1
                    self._last_reattach_time = now  # Track when we last tried
                    try:
                        shm_name = self.shm.name
                        logger.warning(
                            f"SDRplayProxyStream: {self._empty_reads} empty reads, time_since_data={time_since_data:.1f}s. "
                            f"Re-attaching to shared memory {shm_name}... (attempt {reattach_count + 1}/{max_reattach})"
                        )
                        # Close current reference (don't unlink!)
                        old_shm = self.shm
                        # Re-attach to same shared memory
                        self.shm = SharedMemory(name=shm_name)
                        old_shm.close()
                        logger.info(f"SDRplayProxyStream: Re-attached to {shm_name}")
                        # DON'T reset empty_reads here - we only reset when we get data
                        # Try reading again with fresh attachment
                        header = _read_header(self.shm)
                        if header is None:
                            return np.empty(0, dtype=np.complex64), False
                        (
                            write_idx,
                            _,
                            sample_count,
                            overflow_count,
                            sample_rate,
                            flags,
                            _timestamp,
                        ) = header
                        self._buf_none_count = 0
                        available = write_idx - self._last_read_idx
                        if available > 0:
                            logger.info(
                                f"SDRplayProxyStream: Re-attach successful! Now have {available} samples available"
                            )
                            self._empty_reads = 0  # Reset only on success
                            # Continue to process samples below
                        else:
                            return np.empty(0, dtype=np.complex64), False
                    except Exception as e:
                        logger.error(f"SDRplayProxyStream: Re-attach failed: {e}")
                        return np.empty(0, dtype=np.complex64), False
                else:
                    return np.empty(0, dtype=np.complex64), False
            else:
                # Log if this keeps happening (every 10000 reads)
                if self._debug_counter % 10000 == 0:
                    logger.warning(
                        f"SDRplayProxyStream.read(): no available samples, "
                        f"write_idx={write_idx}, _last_read_idx={self._last_read_idx}"
                    )
                return np.empty(0, dtype=np.complex64), False
        else:
            # Reset empty read counter when we get samples
            self._empty_reads = 0
            self._reattach_count = 0  # Reset reattach counter on success
            import time as time_mod

            self._last_data_time = time_mod.time()

        # Detect ring buffer overrun: if available > BUFFER_SAMPLES, the writer
        # has wrapped around and overwritten data before we could read it.
        # Reset read position to catch up with writer, discarding missed samples.
        # overflow_count is cumulative; only report overflow when it increases.
        overflow = overflow_count > self._last_overflow_count
        self._last_overflow_count = overflow_count
        if available > BUFFER_SAMPLES:
            # Calculate how many samples we missed
            missed = available - BUFFER_SAMPLES
            # Jump forward to just behind the writer (leave some margin)
            # Position ourselves BUFFER_SAMPLES/2 behind writer for safety
            self._last_read_idx = write_idx - (BUFFER_SAMPLES // 2)
            available = BUFFER_SAMPLES // 2
            overflow = True
            logger.warning(
                f"Ring buffer overrun: reader fell behind by {missed} samples "
                f"({missed * 1000 / sample_rate:.1f}ms), resetting position"
            )

        # Limit to requested amount
        to_read = min(num_samples, available, BUFFER_SAMPLES // 2)

        # Calculate read position in ring buffer
        buffer_start = HEADER_SIZE
        read_offset = self._last_read_idx % BUFFER_SAMPLES

        # Read samples from ring buffer (single-copy into output)
        samples = np.empty(to_read, dtype=np.complex64)
        if read_offset + to_read <= BUFFER_SAMPLES:
            view = np.ndarray(
                shape=(to_read,),
                dtype=np.complex64,
                buffer=buf,
                offset=buffer_start + (read_offset * 8),
            )
            samples[:] = view
        else:
            first_count = BUFFER_SAMPLES - read_offset
            view1 = np.ndarray(
                shape=(first_count,),
                dtype=np.complex64,
                buffer=buf,
                offset=buffer_start + (read_offset * 8),
            )
            samples[:first_count] = view1

            second_count = to_read - first_count
            view2 = np.ndarray(
                shape=(second_count,),
                dtype=np.complex64,
                buffer=buf,
                offset=buffer_start,
            )
            samples[first_count:] = view2

        # Update read position
        self._last_read_idx += to_read

        # [DIAG-STAGE1] Ring buffer statistics for pipeline diagnosis
        if not hasattr(self, "_diag_read_counter"):
            self._diag_read_counter = 0
            self._diag_total_samples = 0
            self._diag_overflow_count = 0
            self._diag_empty_streak = 0
        self._diag_read_counter += 1
        self._diag_total_samples += len(samples)
        if overflow:
            self._diag_overflow_count += 1

        # Log every 100 reads
        if self._diag_read_counter % 100 == 0:
            iq_power = float(np.mean(np.abs(samples) ** 2)) if len(samples) > 0 else 0.0
            iq_peak = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0
            logger.info(
                f"[DIAG-STAGE1] reads={self._diag_read_counter}, "
                f"available={available}, returned={len(samples)}, "
                f"overflow_total={self._diag_overflow_count}, "
                f"iq_power={iq_power:.6f}, iq_peak={iq_peak:.4f}"
            )

        # Log successful reads for debugging (very infrequently)
        if self._debug_counter <= 5 or self._debug_counter % 100000 == 0:
            logger.debug(
                f"SDRplayProxyStream.read()[{self._debug_counter}]: "
                f"returning {len(samples)} samples, new _last_read_idx={self._last_read_idx}"
            )

        return samples, overflow

    def close(self) -> None:
        """Mark stream as closed."""
        self._closed = True


@dataclass
class SDRplayProxyDevice(Device):
    """Proxy for SDRplay device running in subprocess.

    Presents the same Device interface as _SoapyDevice but routes all
    calls to a subprocess worker. This allows multiple SDRplay devices
    to run simultaneously.
    """

    info: DeviceInfo
    device_args: str
    operation_cooldown: float = (
        1.0  # Minimum seconds between operations (SDRplay API needs time to stabilize)
    )
    # Reduced from 2.0s to 1.0s for faster multi-device startup (~3s vs ~6s for 3 devices)
    _worker_process: Process | None = None
    _shm: SharedMemory | None = None
    _cmd_pipe: Connection | None = None
    _status_pipe: Connection | None = None
    _worker_cmd_pipe: Connection | None = None
    _worker_status_pipe: Connection | None = None
    _antenna: str | None = None
    _stream_format: str | None = None
    _started: bool = False
    _configured: bool = False

    def _handle_worker_message(self, msg: dict[str, Any]) -> None:
        """Handle a message from the worker subprocess.

        Args:
            msg: Message dictionary from worker
        """
        msg_type = msg.get("type", "")

        # Handle log messages from worker
        if msg_type == "log":
            level = msg.get("level", "info")
            message = msg.get("message", "")
            log_func = getattr(logger, level, logger.info)
            log_func(f"[Worker] {message}")

    def _ensure_worker(self, timeout: float = 30.0, already_have_lock: bool = False) -> None:
        """Ensure worker subprocess is running.

        Args:
            timeout: Maximum time to wait for worker to be ready
            already_have_lock: If True, caller already holds global lock (skip acquire)
        """
        if self._worker_process is not None and self._worker_process.is_alive():
            return

        # Acquire global lock - serializes all SDRplay device operations
        if not already_have_lock:
            acquire_sdrplay_lock(self.operation_cooldown)
        try:
            # Re-check after acquiring lock (another thread may have started worker)
            if self._worker_process is not None and self._worker_process.is_alive():
                return

            logger.info(f"Starting worker subprocess for {self.device_args}")

            # Create shared memory for IQ buffer
            _generate_shm_name()
            self._shm = SharedMemory(create=True, size=SHM_SIZE)
            logger.debug(f"Created shared memory: {self._shm.name}, size={SHM_SIZE}")

            # Create IPC pipes
            self._cmd_pipe, self._worker_cmd_pipe = Pipe()
            self._worker_status_pipe, self._status_pipe = Pipe()

            # Launch worker process
            self._worker_process = Process(
                target=sdrplay_worker_main,
                args=(
                    self.device_args,
                    self._shm.name,
                    self._worker_cmd_pipe,
                    self._worker_status_pipe,
                ),
                daemon=True,
            )
            self._worker_process.start()
            logger.info(f"Worker process started, PID={self._worker_process.pid}")
            if self._worker_cmd_pipe is not None:
                self._worker_cmd_pipe.close()
                self._worker_cmd_pipe = None
            if self._worker_status_pipe is not None:
                self._worker_status_pipe.close()
                self._worker_status_pipe = None

            # Wait for worker to be ready
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self._status_pipe.poll(timeout=0.1):
                    msg = self._status_pipe.recv()
                    self._handle_worker_message(msg)
                    if msg.get("type") == "ready":
                        logger.info("Worker ready")
                        break
                    elif msg.get("type") == "error":
                        error_msg = msg.get("message", "Unknown error")
                        logger.error(f"Worker startup error: {error_msg}")
                        raise RuntimeError(f"Worker startup error: {error_msg}")
                # Check if worker died
                if not self._worker_process.is_alive():
                    exit_code = self._worker_process.exitcode
                    logger.error(f"Worker process died during startup, exit_code={exit_code}")
                    self._cleanup_worker()
                    raise RuntimeError(
                        f"Worker process died during startup (exit_code={exit_code})"
                    )
            else:
                logger.error(f"Worker failed to start within {timeout}s timeout")
                self._cleanup_worker()
                raise TimeoutError("Worker failed to start within timeout")

            # Send open command
            logger.debug("Sending 'open' command to worker")
            self._cmd_pipe.send({"type": "open"})

            # Wait for device to open
            while time.time() - start_time < timeout:
                if self._status_pipe.poll(timeout=0.1):
                    msg = self._status_pipe.recv()
                    self._handle_worker_message(msg)
                    if msg.get("type") == "opened":
                        logger.info(
                            f"Device opened: driver={msg.get('driver')}, hardware={msg.get('hardware')}"
                        )
                        break
                    elif msg.get("type") == "open_error":
                        error_msg = msg.get("message", "Unknown error")
                        logger.error(f"Failed to open device: {error_msg}")
                        self._cleanup_worker()
                        raise RuntimeError(f"Failed to open device: {error_msg}")
                # Check if worker died
                if not self._worker_process.is_alive():
                    exit_code = self._worker_process.exitcode
                    logger.error(f"Worker process died while opening device, exit_code={exit_code}")
                    self._cleanup_worker()
                    raise RuntimeError(
                        f"Worker process died while opening device (exit_code={exit_code})"
                    )
            else:
                logger.error(f"Device open timed out after {timeout}s")
                self._cleanup_worker()
                raise TimeoutError("Device open timed out")
        finally:
            if not already_have_lock:
                release_sdrplay_lock()

    def _cleanup_worker(self) -> None:
        """Clean up worker subprocess and resources."""
        logger.debug("Cleaning up worker subprocess and resources")

        if self._cmd_pipe is not None:
            with contextlib.suppress(Exception):
                self._cmd_pipe.send({"type": "shutdown"})

        if self._worker_process is not None:
            pid = self._worker_process.pid
            self._worker_process.join(timeout=2.0)
            if self._worker_process.is_alive():
                logger.warning(f"Worker {pid} did not exit gracefully, terminating")
                self._worker_process.terminate()
                self._worker_process.join(timeout=1.0)
                if self._worker_process.is_alive():
                    logger.warning(f"Worker {pid} did not terminate, killing")
                    self._worker_process.kill()
            else:
                exit_code = self._worker_process.exitcode
                logger.debug(f"Worker {pid} exited with code {exit_code}")
            self._worker_process = None

        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
                logger.debug("Shared memory cleaned up")
            except Exception as e:
                logger.warning(f"Error cleaning up shared memory: {e}")
            self._shm = None

        if self._cmd_pipe is not None:
            with contextlib.suppress(Exception):
                self._cmd_pipe.close()
            self._cmd_pipe = None

        if self._status_pipe is not None:
            with contextlib.suppress(Exception):
                self._status_pipe.close()
            self._status_pipe = None

        if self._worker_cmd_pipe is not None:
            with contextlib.suppress(Exception):
                self._worker_cmd_pipe.close()
            self._worker_cmd_pipe = None

        if self._worker_status_pipe is not None:
            with contextlib.suppress(Exception):
                self._worker_status_pipe.close()
            self._worker_status_pipe = None

        logger.debug("Worker cleanup complete")

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
        """Configure device via worker subprocess."""
        # Acquire global lock for SDRplay operations
        acquire_sdrplay_lock(self.operation_cooldown)
        try:
            logger.info(
                f"Configuring device: center={center_hz / 1e6:.3f}MHz, rate={sample_rate / 1e6:.3f}MHz, antenna={antenna}"
            )
            self._ensure_worker(already_have_lock=True)  # We already hold the lock

            self._antenna = antenna
            self._stream_format = stream_format

            if self._cmd_pipe is None or self._status_pipe is None:
                raise RuntimeError("Worker not initialized")

            # Send configure command
            self._cmd_pipe.send(
                {
                    "type": "configure",
                    "center_hz": center_hz,
                    "sample_rate": sample_rate,
                    "gain": gain,
                    "bandwidth": bandwidth,
                    "ppm": ppm,
                    "antenna": antenna,
                    "device_settings": device_settings or {},
                    "element_gains": element_gains or {},
                    "agc_enabled": agc_enabled,
                    "dc_offset_auto": dc_offset_auto,
                    "iq_balance_auto": iq_balance_auto,
                }
            )

            # Wait for confirmation
            timeout = 10.0
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self._status_pipe.poll(timeout=0.1):
                    msg = self._status_pipe.recv()
                    self._handle_worker_message(msg)
                    if msg.get("type") == "configured":
                        self._configured = True
                        logger.info("Device configured successfully")
                        return
                    elif msg.get("type") == "configure_error":
                        error_msg = msg.get("message", "Unknown error")
                        logger.error(f"Configure failed: {error_msg}")
                        raise RuntimeError(f"Configure failed: {error_msg}")
                # Check if worker died
                if self._worker_process and not self._worker_process.is_alive():
                    exit_code = self._worker_process.exitcode
                    logger.error(f"Worker process died during configure, exit_code={exit_code}")
                    raise RuntimeError(
                        f"Worker process died during configure (exit_code={exit_code})"
                    )

            logger.error("Configure timed out")
            raise TimeoutError("Configure timed out")
        finally:
            release_sdrplay_lock()

    def start_stream(self, already_have_lock: bool = False) -> StreamHandle:
        """Start IQ streaming via worker subprocess.

        Args:
            already_have_lock: If True, caller already holds global lock (skip acquire/release)
        """
        # Acquire global lock - activateStream() cannot be called concurrently
        if not already_have_lock:
            acquire_sdrplay_lock(self.operation_cooldown)
        try:
            logger.info("Starting IQ stream")
            self._ensure_worker(already_have_lock=True)  # We already hold the lock

            if not self._configured:
                logger.error("start_stream called but device not configured")
                raise RuntimeError("Device not configured")

            if self._cmd_pipe is None or self._status_pipe is None or self._shm is None:
                raise RuntimeError("Worker not initialized")

            # Send start command
            self._cmd_pipe.send({"type": "start"})

            # Wait for confirmation
            timeout = 10.0
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self._status_pipe.poll(timeout=0.1):
                    msg = self._status_pipe.recv()
                    self._handle_worker_message(msg)
                    if msg.get("type") == "started":
                        self._started = True
                        self._antenna = msg.get("antenna")
                        logger.info(f"Stream started, antenna={self._antenna}")

                        # Track active capture to prevent recovery during streaming
                        increment_sdrplay_active_captures()

                        # Return proxy stream
                        logger.info(f"Creating SDRplayProxyStream with shm.name={self._shm.name}")
                        return SDRplayProxyStream(
                            shm=self._shm,
                            status_pipe=self._status_pipe,
                        )
                    elif msg.get("type") == "start_error":
                        error_msg = msg.get("message", "Unknown error")
                        logger.error(f"Start stream failed: {error_msg}")
                        raise RuntimeError(f"Start stream failed: {error_msg}")
                # Check if worker died
                if self._worker_process and not self._worker_process.is_alive():
                    exit_code = self._worker_process.exitcode
                    logger.error(f"Worker process died during start_stream, exit_code={exit_code}")
                    raise RuntimeError(
                        f"Worker process died during start_stream (exit_code={exit_code})"
                    )

            logger.error("Start stream timed out")
            raise TimeoutError("Start stream timed out")
        finally:
            # Only release lock if we acquired it
            if not already_have_lock:
                release_sdrplay_lock()

    def configure_and_start(
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
    ) -> StreamHandle:
        """Configure device AND start stream atomically under a single lock.

        This prevents race conditions where another SDRplay device opens while
        this device is between configure and start_stream.

        Returns:
            StreamHandle for reading IQ samples
        """
        # Acquire global lock for entire configure+start sequence
        acquire_sdrplay_lock(self.operation_cooldown)
        try:
            logger.info(
                f"Atomic configure+start: center={center_hz / 1e6:.3f}MHz, rate={sample_rate / 1e6:.3f}MHz"
            )

            # Configure (without releasing lock)
            self._ensure_worker(already_have_lock=True)
            self._antenna = antenna
            self._stream_format = stream_format

            if self._cmd_pipe is None or self._status_pipe is None:
                raise RuntimeError("Worker not initialized")

            self._cmd_pipe.send(
                {
                    "type": "configure",
                    "center_hz": center_hz,
                    "sample_rate": sample_rate,
                    "gain": gain,
                    "bandwidth": bandwidth,
                    "ppm": ppm,
                    "antenna": antenna,
                    "device_settings": device_settings or {},
                    "element_gains": element_gains or {},
                    "agc_enabled": agc_enabled,
                    "dc_offset_auto": dc_offset_auto,
                    "iq_balance_auto": iq_balance_auto,
                }
            )

            # Wait for configure confirmation
            timeout = 10.0
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self._status_pipe.poll(timeout=0.1):
                    msg = self._status_pipe.recv()
                    self._handle_worker_message(msg)
                    if msg.get("type") == "configured":
                        self._configured = True
                        logger.info("Device configured, starting stream...")
                        break
                    elif msg.get("type") == "configure_error":
                        error_msg = msg.get("message", "Unknown error")
                        raise RuntimeError(f"Configure failed: {error_msg}")
                if self._worker_process and not self._worker_process.is_alive():
                    raise RuntimeError(
                        f"Worker died during configure (exit_code={self._worker_process.exitcode})"
                    )
            else:
                raise TimeoutError("Configure timed out")

            # Start stream (still holding lock)
            return self.start_stream(already_have_lock=True)
        finally:
            # Always release lock when done (success or failure)
            release_sdrplay_lock()

    def get_antenna(self) -> str | None:
        """Return the currently configured antenna."""
        return self._antenna

    def get_capabilities(self) -> dict[str, Any]:
        """Query device capabilities (basic info only for proxy)."""
        return {
            "proxy": True,
            "device_args": self.device_args,
            "driver": self.info.driver,
            "antennas": list(self.info.antennas) if self.info.antennas else [],
        }

    def read_sensors(self) -> dict[str, Any]:
        """Read sensors (not implemented for proxy)."""
        return {"proxy": True, "message": "Sensor reading not available via proxy"}

    def reconfigure_running(
        self,
        center_hz: float | None = None,
        gain: float | None = None,
        bandwidth: float | None = None,
        ppm: float | None = None,
    ) -> None:
        """Hot reconfigure while streaming."""
        if not self._started:
            return

        if self._cmd_pipe is None or self._status_pipe is None:
            raise RuntimeError("Worker not initialized")

        # Send configure command (worker handles running reconfiguration)
        cmd: dict[str, Any] = {"type": "configure"}
        if center_hz is not None:
            cmd["center_hz"] = center_hz
        if gain is not None:
            cmd["gain"] = gain
        if bandwidth is not None:
            cmd["bandwidth"] = bandwidth
        if ppm is not None:
            cmd["ppm"] = ppm

        acquire_sdrplay_lock(self.operation_cooldown)
        try:
            self._cmd_pipe.send(cmd)

            # Wait for confirmation
            timeout = 5.0
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self._status_pipe.poll(timeout=0.1):
                    msg = self._status_pipe.recv()
                    if msg.get("type") == "configured":
                        return
                    elif msg.get("type") == "configure_error":
                        raise RuntimeError(f"Reconfigure failed: {msg.get('message')}")

            raise TimeoutError("Reconfigure timed out")
        finally:
            release_sdrplay_lock()

    def close(self) -> None:
        """Close device and terminate worker subprocess."""
        # Acquire global lock for SDRplay operations
        acquire_sdrplay_lock(self.operation_cooldown)
        try:
            logger.info("Closing device")

            if self._started and self._cmd_pipe is not None and self._status_pipe is not None:
                try:
                    logger.debug("Sending stop command to worker")
                    self._cmd_pipe.send({"type": "stop"})
                    # Wait briefly for stop confirmation
                    start_time = time.time()
                    while time.time() - start_time < 2.0:
                        if self._status_pipe.poll(timeout=0.1):
                            msg = self._status_pipe.recv()
                            self._handle_worker_message(msg)
                            if msg.get("type") == "stopped":
                                logger.debug("Stream stopped confirmation received")
                                break
                except Exception as e:
                    logger.warning(f"Error stopping stream: {e}")

            self._cleanup_worker()

            # Decrement active capture count if we were streaming
            if self._started:
                decrement_sdrplay_active_captures()

            self._started = False
            self._configured = False
            logger.info("Device closed")
        finally:
            release_sdrplay_lock()
