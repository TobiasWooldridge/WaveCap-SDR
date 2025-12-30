"""IMBE voice decoder for P25 Phase 1 using DSD-FME with threaded I/O.

This version uses dedicated threads for subprocess I/O to avoid asyncio overhead
and provides much better real-time performance.

Input: 48kHz mono discriminator audio (instantaneous frequency from FM demod)
Output: 8kHz 16-bit PCM (resampled to target rate)
"""

from __future__ import annotations

import asyncio
import logging
import queue
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


class IMBEDecoderError(Exception):
    """Raised when IMBE decoder fails."""


@dataclass
class IMBEDecoderThreaded:
    """
    Decodes P25 Phase 1 IMBE voice frames to PCM audio using DSD-FME.

    Uses dedicated threads for subprocess I/O to avoid asyncio overhead.
    Audio is collected in batches for more efficient processing.
    """

    # DSD-FME outputs 8kHz 16-bit mono PCM
    DSD_OUTPUT_RATE = 8000
    DSD_INPUT_RATE = 48000

    # Larger queue to handle bursty input
    INPUT_QUEUE_SIZE = 512
    OUTPUT_QUEUE_SIZE = 256

    output_rate: int = 48000
    input_rate: int = 48000

    # State
    process: subprocess.Popen[bytes] | None = field(default=None, repr=False)
    running: bool = False
    _write_thread: threading.Thread | None = field(default=None, repr=False)
    _read_thread: threading.Thread | None = field(default=None, repr=False)

    # Thread-safe queues
    _input_queue: queue.Queue[np.ndarray | None] = field(
        default_factory=lambda: queue.Queue(maxsize=512),
        repr=False,
    )
    _output_queue: queue.Queue[np.ndarray] = field(
        default_factory=lambda: queue.Queue(maxsize=256),
        repr=False,
    )

    # Callback for output audio (called from read thread)
    on_audio: Callable[[np.ndarray], None] | None = field(default=None, repr=False)

    # Resampling
    _resample_up: int = field(default=0, init=False)
    _resample_down: int = field(default=0, init=False)

    # Statistics
    frames_decoded: int = 0
    frames_dropped: int = 0
    bytes_written: int = 0
    _last_stats_time: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        # Calculate resampling ratio
        self._resample_up = self.output_rate
        self._resample_down = self.DSD_OUTPUT_RATE
        gcd = np.gcd(self._resample_up, self._resample_down)
        self._resample_up //= gcd
        self._resample_down //= gcd

    @staticmethod
    def is_available() -> bool:
        """Check if DSD-FME is installed and available."""
        return shutil.which("dsd-fme") is not None

    def start(self) -> None:
        """Start the DSD-FME decoder subprocess with threaded I/O."""
        if self.running:
            return

        if not self.is_available():
            raise IMBEDecoderError(
                "DSD-FME not found in PATH. Install with: "
                "git clone https://github.com/lwvmobile/dsd-fme && "
                "cd dsd-fme && mkdir build && cd build && cmake .. && make && sudo make install"
            )

        # Start subprocess
        args = self._get_dsd_args()
        logger.debug(f"Starting DSD-FME: {' '.join(args)}")

        self.process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,  # Unbuffered for lower latency
        )

        self.running = True
        self.frames_decoded = 0
        self.frames_dropped = 0
        self.bytes_written = 0
        self._last_stats_time = time.time()

        # Start I/O threads
        self._write_thread = threading.Thread(
            target=self._write_loop,
            name="imbe-writer",
            daemon=True
        )
        self._read_thread = threading.Thread(
            target=self._read_loop,
            name="imbe-reader",
            daemon=True
        )

        self._write_thread.start()
        self._read_thread.start()

        logger.info(
            f"IMBE decoder started (input={self.input_rate}Hz, output={self.output_rate}Hz, threaded)"
        )

    def stop(self) -> None:
        """Stop the decoder subprocess."""
        if not self.running:
            return

        self.running = False

        # Signal threads to exit
        try:
            self._input_queue.put(None, timeout=0.5)
        except queue.Full:
            pass

        # Wait for threads
        if self._write_thread and self._write_thread.is_alive():
            self._write_thread.join(timeout=1.0)
        if self._read_thread and self._read_thread.is_alive():
            self._read_thread.join(timeout=1.0)

        # Terminate process
        if self.process:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            except Exception as e:
                logger.debug(f"Exception during IMBE decoder cleanup: {e}")
            finally:
                self.process = None

        logger.info(
            f"IMBE decoder stopped (decoded={self.frames_decoded}, "
            f"dropped={self.frames_dropped}, bytes_written={self.bytes_written})"
        )

    def decode(self, discriminator_audio: np.ndarray) -> None:
        """
        Queue discriminator audio for decoding (thread-safe).

        Args:
            discriminator_audio: Instantaneous frequency values from FM discriminator
        """
        if not self.running:
            return

        try:
            self._input_queue.put_nowait(discriminator_audio)
        except queue.Full:
            self.frames_dropped += 1
            if self.frames_dropped == 1 or self.frames_dropped % 100 == 0:
                logger.warning(
                    f"IMBE input queue full (dropped={self.frames_dropped}, "
                    f"chunk_size={len(discriminator_audio)})"
                )
            # Try to drop oldest and add new
            try:
                self._input_queue.get_nowait()
                self._input_queue.put_nowait(discriminator_audio)
            except queue.Empty:
                pass
            except queue.Full:
                pass

    def get_audio(self) -> np.ndarray | None:
        """Get decoded audio if available (non-blocking)."""
        try:
            return self._output_queue.get_nowait()
        except queue.Empty:
            return None

    def get_audio_blocking(self, timeout: float = 0.5) -> np.ndarray | None:
        """Get decoded audio, waiting up to timeout seconds."""
        try:
            return self._output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _get_dsd_args(self) -> list[str]:
        """Get DSD-FME command line arguments."""
        return [
            "dsd-fme",
            "-i", "-",           # Read from stdin
            "-o", "-",           # Write to stdout
            "-f1",               # P25 Phase 1 only
            "-N",                # No audio output to speaker
            "-g", "0",           # No gain adjustment
            "-u", "0",           # No upsampling (we handle it)
            "-L",                # Low CPU mode
        ]

    def _write_loop(self) -> None:
        """Thread loop that writes audio to DSD-FME stdin."""
        logger.debug("IMBE writer thread started")

        # Accumulation buffer for batching
        batch_samples = 4800  # 100ms at 48kHz
        batch_buffer = np.zeros(batch_samples, dtype=np.float32)
        batch_pos = 0

        while self.running and self.process and self.process.stdin:
            try:
                # Get audio from queue with timeout
                audio = self._input_queue.get(timeout=0.1)

                if audio is None:  # Shutdown signal
                    break

                # Add to batch buffer
                samples_to_add = len(audio)
                space_available = batch_samples - batch_pos

                if samples_to_add <= space_available:
                    # Fits in current batch
                    batch_buffer[batch_pos:batch_pos + samples_to_add] = audio
                    batch_pos += samples_to_add
                else:
                    # Fill current batch and flush
                    batch_buffer[batch_pos:batch_pos + space_available] = audio[:space_available]
                    self._flush_batch(batch_buffer)
                    batch_pos = 0

                    # Handle remaining samples
                    remaining = audio[space_available:]
                    while len(remaining) >= batch_samples:
                        self._flush_batch(remaining[:batch_samples])
                        remaining = remaining[batch_samples:]

                    if len(remaining) > 0:
                        batch_buffer[:len(remaining)] = remaining
                        batch_pos = len(remaining)

                # Flush if batch is full
                if batch_pos >= batch_samples:
                    self._flush_batch(batch_buffer)
                    batch_pos = 0

            except queue.Empty:
                # Flush partial batch on timeout to maintain low latency
                if batch_pos > 0:
                    self._flush_batch(batch_buffer[:batch_pos])
                    batch_pos = 0
            except Exception as e:
                logger.error(f"IMBE writer error: {e}")
                break

        logger.debug("IMBE writer thread exiting")

    def _flush_batch(self, audio: np.ndarray) -> None:
        """Write audio batch to DSD-FME stdin."""
        if not self.process or not self.process.stdin:
            return

        try:
            # Scale discriminator audio to int16 range for DSD-FME
            # The discriminator output is in radians per sample from phase differentiation
            # C4FM at 48kHz: ±1800 Hz = ±0.236 radians/sample for full deviation
            # DSD-FME works best with ~±20000 amplitude for clear C4FM
            #
            # Use fixed scale factor optimized for C4FM at 48kHz:
            # 20000 / 0.236 ≈ 85000, but real signals have margin, so use lower
            # Scale factor of 50000 works well with typical signal levels
            scaled = np.clip(audio * 50000.0, -32767, 32767).astype(np.int16)
            audio_bytes = scaled.tobytes()

            self.process.stdin.write(audio_bytes)
            self.bytes_written += len(audio_bytes)

        except (BrokenPipeError, OSError) as e:
            logger.error(f"IMBE write error: {e}")
            self.running = False

    def _read_loop(self) -> None:
        """Thread loop that reads decoded audio from DSD-FME stdout."""
        logger.debug("IMBE reader thread started")

        # DSD-FME outputs 8kHz 16-bit PCM
        chunk_samples = 160  # 20ms at 8kHz = one IMBE frame
        chunk_bytes = chunk_samples * 2

        while self.running and self.process and self.process.stdout:
            try:
                data = self.process.stdout.read(chunk_bytes)
                if not data:
                    break

                if len(data) < chunk_bytes:
                    # Partial read, wait for more
                    continue

                # Convert to int16 samples
                samples = np.frombuffer(data, dtype=np.int16)

                # Convert to float32 normalized to [-1, 1]
                audio_f32 = samples.astype(np.float32) / 32768.0

                # Resample from 8kHz to target rate
                if self._resample_up != self._resample_down:
                    audio_f32 = signal.resample_poly(
                        audio_f32, self._resample_up, self._resample_down
                    ).astype(np.float32)

                self.frames_decoded += 1

                # Call callback if set
                if self.on_audio:
                    try:
                        self.on_audio(audio_f32)
                    except Exception as e:
                        logger.error(f"IMBE audio callback error: {e}")

                # Queue for output
                try:
                    self._output_queue.put_nowait(audio_f32)
                except queue.Full:
                    # Drop oldest
                    try:
                        self._output_queue.get_nowait()
                        self._output_queue.put_nowait(audio_f32)
                    except queue.Empty:
                        pass

            except Exception as e:
                logger.error(f"IMBE reader error: {e}")
                break

        logger.debug("IMBE reader thread exiting")


def check_imbe_available() -> tuple[bool, str]:
    """Check if IMBE decoding is available."""
    if IMBEDecoderThreaded.is_available():
        return True, "DSD-FME is available for IMBE decoding (threaded)"
    else:
        return False, (
            "DSD-FME not found. Install with: "
            "git clone https://github.com/lwvmobile/dsd-fme && "
            "cd dsd-fme && mkdir build && cd build && cmake .. && make && sudo make install"
        )
