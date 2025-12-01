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

import multiprocessing
import signal
import struct
import time
from dataclasses import dataclass
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Dict, Optional, Tuple

import numpy as np


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

# Buffer size for IQ samples (2MB = 262144 complex64 samples)
# At 6 MHz sample rate, this is ~44ms of buffer
BUFFER_SAMPLES = 262144
BUFFER_SIZE = BUFFER_SAMPLES * 8  # complex64 = 8 bytes per sample

# Total shared memory size
SHM_SIZE = HEADER_SIZE + BUFFER_SIZE

# Flag bits
FLAG_OVERFLOW = 1 << 0
FLAG_ERROR = 1 << 1
FLAG_RUNNING = 1 << 2


@dataclass
class WorkerConfig:
    """Configuration sent to worker subprocess."""
    device_args: str
    shm_name: str
    center_hz: float = 100e6
    sample_rate: int = 2_000_000
    gain: Optional[float] = None
    bandwidth: Optional[float] = None
    ppm: Optional[float] = None
    antenna: Optional[str] = None
    dc_offset_auto: bool = True
    iq_balance_auto: bool = True
    device_settings: Dict[str, Any] = None
    element_gains: Dict[str, float] = None
    stream_format: Optional[str] = None


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
    header = struct.pack(
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


def _read_header(shm: SharedMemory) -> Tuple[int, int, int, int, int, int, float]:
    """Read header from shared memory.

    Returns: (write_idx, read_idx, sample_count, overflow_count, sample_rate, flags, timestamp)
    """
    header = bytes(shm.buf[:HEADER_SIZE])
    return struct.unpack(HEADER_FORMAT, header)


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
    # Ignore SIGINT in worker - let main process handle shutdown
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    worker = SDRplayWorker(device_args, shm_name, cmd_pipe, status_pipe)
    try:
        worker.run()
    except Exception as e:
        status_pipe.send({"type": "error", "message": str(e)})
    finally:
        worker.cleanup()


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

        self.sdr = None
        self.stream = None
        self.shm: Optional[SharedMemory] = None
        self.running = False
        self.streaming = False

        # Stats
        self.sample_count = 0
        self.overflow_count = 0
        self.write_idx = 0

        # Config
        self.sample_rate = 2_000_000
        self.center_hz = 100e6

    def run(self) -> None:
        """Main loop: process commands and stream IQ to shared memory."""
        # Attach to shared memory (created by main process)
        self.shm = SharedMemory(name=self.shm_name)

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

        while self.running:
            # Check for commands (non-blocking)
            while self.cmd_pipe.poll(timeout=0.001):
                try:
                    cmd = self.cmd_pipe.recv()
                    self._handle_command(cmd)
                except EOFError:
                    self.running = False
                    break

            # Stream IQ samples if active
            if self.streaming and self.stream is not None:
                try:
                    self._stream_samples()
                except Exception as e:
                    self.status_pipe.send({"type": "stream_error", "message": str(e)})
                    self.streaming = False

    def _handle_command(self, cmd: dict) -> None:
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
            import SoapySDR  # type: ignore

            print(f"[SDRplayWorker] Opening device: {self.device_args}", flush=True)
            self.sdr = SoapySDR.Device(self.device_args)

            # Query device info
            driver = str(self.sdr.getDriverKey())
            hardware = str(self.sdr.getHardwareKey())
            antennas = list(self.sdr.listAntennas(SoapySDR.SOAPY_SDR_RX, 0))

            self.status_pipe.send({
                "type": "opened",
                "driver": driver,
                "hardware": hardware,
                "antennas": antennas,
            })
            print(f"[SDRplayWorker] Device opened: {driver} / {hardware}", flush=True)

        except Exception as e:
            self.status_pipe.send({"type": "open_error", "message": str(e)})
            raise

    def _configure(self, cmd: dict) -> None:
        """Configure the device."""
        if self.sdr is None:
            self.status_pipe.send({"type": "error", "message": "Device not open"})
            return

        try:
            import SoapySDR  # type: ignore

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

            print(f"[SDRplayWorker] Configuring: center={center_hz/1e6}MHz, rate={sample_rate/1e6}MHz", flush=True)

            # Apply configuration
            self.sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, float(sample_rate))
            self.sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, center_hz)

            if device_settings:
                for key, value in device_settings.items():
                    try:
                        self.sdr.writeSetting(key, str(value))
                    except Exception as e:
                        print(f"[SDRplayWorker] Setting {key}={value} failed: {e}", flush=True)

            if element_gains:
                for elem, g in element_gains.items():
                    try:
                        self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, elem, g)
                    except Exception as e:
                        print(f"[SDRplayWorker] Gain {elem}={g} failed: {e}", flush=True)
            elif gain is not None:
                try:
                    self.sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, False)
                except Exception:
                    pass
                self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, gain)
            else:
                try:
                    self.sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, True)
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

            try:
                self.sdr.setDCOffsetMode(SoapySDR.SOAPY_SDR_RX, 0, dc_offset_auto)
            except Exception:
                pass

            try:
                self.sdr.setIQBalanceMode(SoapySDR.SOAPY_SDR_RX, 0, iq_balance_auto)
            except Exception:
                pass

            if antenna is not None:
                self.sdr.setAntenna(SoapySDR.SOAPY_SDR_RX, 0, antenna)

            self.sample_rate = sample_rate
            self.center_hz = center_hz

            # Update header with sample rate
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
            print("[SDRplayWorker] Configuration complete", flush=True)

        except Exception as e:
            self.status_pipe.send({"type": "configure_error", "message": str(e)})
            raise

    def _start_stream(self) -> None:
        """Start IQ streaming."""
        if self.sdr is None:
            self.status_pipe.send({"type": "error", "message": "Device not open"})
            return

        if self.streaming:
            return

        try:
            import SoapySDR  # type: ignore

            print("[SDRplayWorker] Starting stream...", flush=True)

            # Set default antenna if not set
            antennas = self.sdr.listAntennas(SoapySDR.SOAPY_SDR_RX, 0)
            current_antenna = self.sdr.getAntenna(SoapySDR.SOAPY_SDR_RX, 0)
            print(f"[SDRplayWorker] Current antenna: {current_antenna}, available: {antennas}", flush=True)

            # Setup stream
            self.stream = self.sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32, [0])
            self.sdr.activateStream(self.stream)

            self.streaming = True
            self.write_idx = 0
            self.sample_count = 0
            self.overflow_count = 0

            # Update header
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
            print("[SDRplayWorker] Stream started", flush=True)

        except Exception as e:
            self.status_pipe.send({"type": "start_error", "message": str(e)})
            raise

    def _stop_stream(self) -> None:
        """Stop IQ streaming."""
        if not self.streaming or self.stream is None:
            return

        try:
            print("[SDRplayWorker] Stopping stream...", flush=True)
            self.streaming = False

            self.sdr.deactivateStream(self.stream)
            self.sdr.closeStream(self.stream)
            self.stream = None

            # Update header
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
            print("[SDRplayWorker] Stream stopped", flush=True)

        except Exception as e:
            self.status_pipe.send({"type": "stop_error", "message": str(e)})

    def _stream_samples(self) -> None:
        """Read samples from device and write to shared memory."""
        if self.stream is None:
            return

        # Read samples from device
        chunk_size = min(8192, BUFFER_SAMPLES // 4)  # Read in smaller chunks
        buff = np.empty(chunk_size, dtype=np.complex64)

        sr = self.sdr.readStream(self.stream, [buff.view(np.float32)], chunk_size, flags=0)

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
        iq_buffer = np.ndarray(
            shape=(BUFFER_SAMPLES,),
            dtype=np.complex64,
            buffer=self.shm.buf,
            offset=HEADER_SIZE,
        )

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
        _write_header(
            self.shm,
            write_idx=self.write_idx,
            sample_count=self.sample_count,
            overflow_count=self.overflow_count,
            sample_rate=self.sample_rate,
            flags=FLAG_RUNNING | (FLAG_OVERFLOW if self.overflow_count > 0 else 0),
            timestamp=time.time(),
        )

    def cleanup(self) -> None:
        """Clean up resources."""
        print("[SDRplayWorker] Cleaning up...", flush=True)

        if self.streaming:
            self._stop_stream()

        if self.sdr is not None:
            try:
                import SoapySDR  # type: ignore
                SoapySDR.Device.unmake(self.sdr)
            except Exception as e:
                print(f"[SDRplayWorker] Device unmake error: {e}", flush=True)
            self.sdr = None

        if self.shm is not None:
            try:
                self.shm.close()
            except Exception:
                pass
            self.shm = None

        print("[SDRplayWorker] Cleanup complete", flush=True)
