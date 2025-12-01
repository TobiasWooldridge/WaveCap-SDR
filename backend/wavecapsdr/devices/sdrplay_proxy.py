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

import struct
import time
import uuid
from dataclasses import dataclass, field
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import Device, DeviceInfo, StreamHandle
from .sdrplay_worker import (
    sdrplay_worker_main,
    HEADER_SIZE,
    HEADER_FORMAT,
    BUFFER_SAMPLES,
    SHM_SIZE,
    FLAG_OVERFLOW,
    FLAG_ERROR,
    FLAG_RUNNING,
    _read_header,
)


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

    def read(self, num_samples: int) -> Tuple[np.ndarray, bool]:
        """Read IQ samples from shared memory ring buffer.

        Args:
            num_samples: Number of complex samples to read

        Returns:
            Tuple of (samples array, overflow flag)
        """
        if self._closed:
            return np.empty(0, dtype=np.complex64), False

        # Read header to get current write position
        write_idx, _, sample_count, overflow_count, sample_rate, flags, timestamp = _read_header(self.shm)

        # Calculate available samples
        available = write_idx - self._last_read_idx
        if available <= 0:
            # No new samples yet - return empty
            return np.empty(0, dtype=np.complex64), False

        # Limit to requested amount
        to_read = min(num_samples, available, BUFFER_SAMPLES // 2)

        # Calculate read position in ring buffer
        buffer_start = HEADER_SIZE
        read_offset = (self._last_read_idx % BUFFER_SAMPLES) * 8  # 8 bytes per complex64

        # Read samples from ring buffer
        bytes_to_read = to_read * 8
        bytes_to_end = (BUFFER_SAMPLES * 8) - read_offset

        if bytes_to_read <= bytes_to_end:
            # Single read (no wrap)
            samples_bytes = bytes(self.shm.buf[buffer_start + read_offset:buffer_start + read_offset + bytes_to_read])
        else:
            # Split read (wrap around)
            part1 = bytes(self.shm.buf[buffer_start + read_offset:buffer_start + BUFFER_SAMPLES * 8])
            part2 = bytes(self.shm.buf[buffer_start:buffer_start + bytes_to_read - bytes_to_end])
            samples_bytes = part1 + part2

        # Convert to numpy array (copy to make writeable)
        samples = np.frombuffer(samples_bytes, dtype=np.complex64).copy()

        # Update read position
        self._last_read_idx += to_read

        # Check for overflow
        overflow = bool(flags & FLAG_OVERFLOW)

        # Check for status messages (non-blocking)
        while self.status_pipe.poll(timeout=0):
            try:
                msg = self.status_pipe.recv()
                if msg.get("type") == "overflow":
                    overflow = True
            except EOFError:
                break

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
    _worker_process: Optional[Process] = None
    _shm: Optional[SharedMemory] = None
    _cmd_pipe: Optional[Connection] = None
    _status_pipe: Optional[Connection] = None
    _worker_cmd_pipe: Optional[Connection] = None
    _worker_status_pipe: Optional[Connection] = None
    _antenna: Optional[str] = None
    _stream_format: Optional[str] = None
    _started: bool = False
    _configured: bool = False

    def _ensure_worker(self, timeout: float = 30.0) -> None:
        """Ensure worker subprocess is running."""
        if self._worker_process is not None and self._worker_process.is_alive():
            return

        print(f"[SDRplayProxy] Starting worker for {self.device_args}", flush=True)

        # Create shared memory for IQ buffer
        shm_name = _generate_shm_name()
        self._shm = SharedMemory(create=True, size=SHM_SIZE)

        # Create IPC pipes
        self._cmd_pipe, self._worker_cmd_pipe = Pipe()
        self._worker_status_pipe, self._status_pipe = Pipe()

        # Launch worker process
        self._worker_process = Process(
            target=sdrplay_worker_main,
            args=(self.device_args, self._shm.name, self._worker_cmd_pipe, self._worker_status_pipe),
            daemon=True,
        )
        self._worker_process.start()

        # Wait for worker to be ready
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._status_pipe.poll(timeout=0.1):
                msg = self._status_pipe.recv()
                if msg.get("type") == "ready":
                    print("[SDRplayProxy] Worker ready", flush=True)
                    break
                elif msg.get("type") == "error":
                    raise RuntimeError(f"Worker startup error: {msg.get('message')}")
        else:
            self._cleanup_worker()
            raise TimeoutError("Worker failed to start within timeout")

        # Send open command
        self._cmd_pipe.send({"type": "open"})

        # Wait for device to open
        while time.time() - start_time < timeout:
            if self._status_pipe.poll(timeout=0.1):
                msg = self._status_pipe.recv()
                if msg.get("type") == "opened":
                    print(f"[SDRplayProxy] Device opened: {msg.get('driver')}", flush=True)
                    break
                elif msg.get("type") == "open_error":
                    self._cleanup_worker()
                    raise RuntimeError(f"Failed to open device: {msg.get('message')}")
        else:
            self._cleanup_worker()
            raise TimeoutError("Device open timed out")

    def _cleanup_worker(self) -> None:
        """Clean up worker subprocess and resources."""
        if self._cmd_pipe is not None:
            try:
                self._cmd_pipe.send({"type": "shutdown"})
            except Exception:
                pass

        if self._worker_process is not None:
            self._worker_process.join(timeout=2.0)
            if self._worker_process.is_alive():
                self._worker_process.terminate()
                self._worker_process.join(timeout=1.0)
                if self._worker_process.is_alive():
                    self._worker_process.kill()
            self._worker_process = None

        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception:
                pass
            self._shm = None

        if self._cmd_pipe is not None:
            try:
                self._cmd_pipe.close()
            except Exception:
                pass
            self._cmd_pipe = None

        if self._status_pipe is not None:
            try:
                self._status_pipe.close()
            except Exception:
                pass
            self._status_pipe = None

    def configure(
        self,
        center_hz: float,
        sample_rate: int,
        gain: Optional[float] = None,
        bandwidth: Optional[float] = None,
        ppm: Optional[float] = None,
        antenna: Optional[str] = None,
        device_settings: Optional[Dict[str, Any]] = None,
        element_gains: Optional[Dict[str, float]] = None,
        stream_format: Optional[str] = None,
        dc_offset_auto: bool = True,
        iq_balance_auto: bool = True,
    ) -> None:
        """Configure device via worker subprocess."""
        self._ensure_worker()

        self._antenna = antenna
        self._stream_format = stream_format

        # Send configure command
        self._cmd_pipe.send({
            "type": "configure",
            "center_hz": center_hz,
            "sample_rate": sample_rate,
            "gain": gain,
            "bandwidth": bandwidth,
            "ppm": ppm,
            "antenna": antenna,
            "device_settings": device_settings or {},
            "element_gains": element_gains or {},
            "dc_offset_auto": dc_offset_auto,
            "iq_balance_auto": iq_balance_auto,
        })

        # Wait for confirmation
        timeout = 10.0
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._status_pipe.poll(timeout=0.1):
                msg = self._status_pipe.recv()
                if msg.get("type") == "configured":
                    self._configured = True
                    print("[SDRplayProxy] Device configured", flush=True)
                    return
                elif msg.get("type") == "configure_error":
                    raise RuntimeError(f"Configure failed: {msg.get('message')}")

        raise TimeoutError("Configure timed out")

    def start_stream(self) -> StreamHandle:
        """Start IQ streaming via worker subprocess."""
        self._ensure_worker()

        if not self._configured:
            raise RuntimeError("Device not configured")

        # Send start command
        self._cmd_pipe.send({"type": "start"})

        # Wait for confirmation
        timeout = 10.0
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._status_pipe.poll(timeout=0.1):
                msg = self._status_pipe.recv()
                if msg.get("type") == "started":
                    self._started = True
                    self._antenna = msg.get("antenna")
                    print(f"[SDRplayProxy] Stream started, antenna={self._antenna}", flush=True)

                    # Return proxy stream
                    return SDRplayProxyStream(
                        shm=self._shm,
                        status_pipe=self._status_pipe,
                    )
                elif msg.get("type") == "start_error":
                    raise RuntimeError(f"Start stream failed: {msg.get('message')}")

        raise TimeoutError("Start stream timed out")

    def get_antenna(self) -> Optional[str]:
        """Return the currently configured antenna."""
        return self._antenna

    def get_capabilities(self) -> Dict[str, Any]:
        """Query device capabilities (basic info only for proxy)."""
        return {
            "proxy": True,
            "device_args": self.device_args,
            "driver": self.info.driver,
            "antennas": list(self.info.antennas) if self.info.antennas else [],
        }

    def read_sensors(self) -> Dict[str, Any]:
        """Read sensors (not implemented for proxy)."""
        return {"proxy": True, "message": "Sensor reading not available via proxy"}

    def reconfigure_running(
        self,
        center_hz: Optional[float] = None,
        gain: Optional[float] = None,
        bandwidth: Optional[float] = None,
        ppm: Optional[float] = None,
    ) -> None:
        """Hot reconfigure while streaming."""
        if not self._started:
            return

        # Send configure command (worker handles running reconfiguration)
        cmd = {"type": "configure"}
        if center_hz is not None:
            cmd["center_hz"] = center_hz
        if gain is not None:
            cmd["gain"] = gain
        if bandwidth is not None:
            cmd["bandwidth"] = bandwidth
        if ppm is not None:
            cmd["ppm"] = ppm

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

    def close(self) -> None:
        """Close device and terminate worker subprocess."""
        print("[SDRplayProxy] Closing device", flush=True)

        if self._started and self._cmd_pipe is not None:
            try:
                self._cmd_pipe.send({"type": "stop"})
                # Wait briefly for stop confirmation
                start_time = time.time()
                while time.time() - start_time < 2.0:
                    if self._status_pipe.poll(timeout=0.1):
                        msg = self._status_pipe.recv()
                        if msg.get("type") == "stopped":
                            break
            except Exception:
                pass

        self._cleanup_worker()
        self._started = False
        self._configured = False
        print("[SDRplayProxy] Device closed", flush=True)
