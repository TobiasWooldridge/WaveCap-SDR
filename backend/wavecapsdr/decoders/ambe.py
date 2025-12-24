"""AMBE+2 voice decoder for P25 Phase II using DSD-FME subprocess.

DSD-FME (Digital Speech Decoder - Florida Man Edition) handles the actual
AMBE+2 voice codec decoding. We pipe discriminator audio to it and get
decoded PCM audio back.

P25 Phase II uses TDMA with two timeslots, each carrying AMBE+2 encoded voice.
- 49-bit AMBE+2 frames (compared to 88-bit IMBE for Phase I)
- 12000 symbols/sec CQPSK modulation
- 20ms voice frames

Input: 48kHz mono discriminator audio (instantaneous frequency from FM demod)
Output: 8kHz 16-bit PCM (resampled to target rate)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import subprocess

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


class AMBEDecoderError(Exception):
    """Raised when AMBE decoder fails."""


class AMBEDecoder:
    """
    Decodes P25 Phase II AMBE+2 voice frames to PCM audio using DSD-FME.

    DSD-FME expects discriminator audio at 48kHz and outputs 8kHz PCM.
    We resample the output to the target sample rate (typically 48kHz).

    For Phase II TDMA, each timeslot produces separate audio streams.
    """

    # DSD-FME outputs 8kHz 16-bit mono PCM
    DSD_OUTPUT_RATE = 8000
    DSD_INPUT_RATE = 48000

    # AMBE+2 frame parameters
    FRAME_BITS = 49
    FRAME_SAMPLES = 160  # 20ms at 8kHz

    def __init__(
        self,
        output_rate: int = 48000,
        input_rate: int = 48000,
        timeslot: int = 0,  # 0 = both, 1 or 2 = specific slot
    ):
        """
        Initialize AMBE+2 decoder.

        Args:
            output_rate: Target audio output sample rate (default 48000)
            input_rate: Input discriminator audio sample rate (default 48000)
            timeslot: TDMA timeslot to decode (0=both, 1 or 2 for specific)
        """
        self.output_rate = output_rate
        self.input_rate = input_rate
        self.timeslot = timeslot
        self.process: subprocess.Popen[bytes] | None = None
        self.running = False
        self._decoder_task: asyncio.Task[None] | None = None

        # I/O queues
        self._input_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=64)
        self._output_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=64)

        # Resampling ratio from DSD output to target rate
        self._resample_up = output_rate
        self._resample_down = self.DSD_OUTPUT_RATE
        # Simplify ratio
        gcd = np.gcd(self._resample_up, self._resample_down)
        self._resample_up //= gcd
        self._resample_down //= gcd

        # Statistics
        self.frames_decoded = 0
        self.frames_dropped = 0

    @staticmethod
    def is_available() -> bool:
        """Check if DSD-FME is installed and available."""
        return shutil.which("dsd-fme") is not None

    async def start(self) -> None:
        """Start the DSD-FME decoder subprocess."""
        if self.running:
            return

        if not self.is_available():
            raise AMBEDecoderError(
                "DSD-FME not found in PATH. Install with: "
                "git clone https://github.com/lwvmobile/dsd-fme && "
                "cd dsd-fme && mkdir build && cd build && cmake .. && make && sudo make install"
            )

        self.running = True
        self._decoder_task = asyncio.create_task(self._run_decoder())
        logger.info(
            f"AMBE+2 decoder started (input={self.input_rate}Hz, output={self.output_rate}Hz, "
            f"timeslot={self.timeslot})"
        )

    async def stop(self) -> None:
        """Stop the decoder subprocess."""
        if not self.running:
            return

        self.running = False

        # Cancel decoder task
        if self._decoder_task:
            self._decoder_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._decoder_task

        # Terminate process
        if self.process:
            try:
                if self.process.stdin:
                    with contextlib.suppress(BrokenPipeError, OSError):
                        self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            except Exception as e:
                logger.debug(f"Exception during AMBE decoder cleanup: {e}")
            finally:
                self.process = None

        logger.info(
            f"AMBE+2 decoder stopped (decoded={self.frames_decoded}, dropped={self.frames_dropped})"
        )

    async def decode(self, discriminator_audio: np.ndarray) -> None:
        """
        Queue discriminator audio for decoding.

        Args:
            discriminator_audio: Instantaneous frequency values from FM discriminator
                                 (float array, will be converted to int16)
        """
        if not self.running:
            return

        try:
            self._input_queue.put_nowait(discriminator_audio)
        except asyncio.QueueFull:
            # Drop oldest to make room
            self.frames_dropped += 1
            try:
                self._input_queue.get_nowait()
                self._input_queue.put_nowait(discriminator_audio)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass

    async def get_audio(self) -> np.ndarray | None:
        """
        Get decoded and resampled audio.

        Returns:
            PCM audio samples as float32 array normalized to [-1, 1],
            or None if no audio available.
        """
        try:
            return self._output_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def get_audio_blocking(self, timeout: float = 0.5) -> np.ndarray | None:
        """
        Get decoded audio, waiting up to timeout seconds.

        Returns:
            PCM audio samples or None if timeout.
        """
        try:
            return await asyncio.wait_for(self._output_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    # --- Sync methods for use from capture thread ---

    def decode_sync(self, discriminator_audio: np.ndarray) -> None:
        """
        Queue discriminator audio for decoding (sync version).

        Can be called from any thread - uses non-blocking queue put.
        The async decoder task will process queued data.

        Args:
            discriminator_audio: Instantaneous frequency values from FM discriminator
        """
        if not self.running:
            return

        try:
            self._input_queue.put_nowait(discriminator_audio)
        except asyncio.QueueFull:
            # Drop oldest to make room
            self.frames_dropped += 1
            try:
                self._input_queue.get_nowait()
                self._input_queue.put_nowait(discriminator_audio)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass

    def get_audio_sync(self) -> np.ndarray | None:
        """
        Get decoded audio if available (sync version).

        Can be called from any thread - uses non-blocking queue get.

        Returns:
            PCM audio samples as float32 array normalized to [-1, 1],
            or None if no audio available yet.
        """
        try:
            return self._output_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def start_in_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Start the decoder from a sync context.

        Schedules the async start() method on the provided event loop.
        The decoder subprocess will run in the event loop's thread.

        Args:
            loop: Event loop to run the decoder in
        """
        if self.running:
            return

        if not self.is_available():
            logger.warning(
                "DSD-FME not found - AMBE+2 voice decoding disabled. "
                "Install from: https://github.com/lwvmobile/dsd-fme"
            )
            return

        try:
            asyncio.run_coroutine_threadsafe(self.start(), loop)
        except Exception as e:
            logger.error(f"Failed to start AMBE decoder: {e}")

    def _get_dsd_args(self) -> list[str]:
        """Get DSD-FME command line arguments for P25 Phase II."""
        args = [
            "dsd-fme",
            "-i", "-",           # Read from stdin
            "-o", "-",           # Write to stdout
            "-fp",               # Force P25 (includes Phase II)
            "-N",                # No audio output to speaker
            "-g", "0",           # No gain adjustment
            "-u", "0",           # No upsampling (we handle it)
            "-L",                # Low CPU mode (skip some features)
        ]

        # Specify timeslot if needed
        if self.timeslot in (1, 2):
            args.extend(["-s", str(self.timeslot)])

        return args

    async def _run_decoder(self) -> None:
        """Run the decoder subprocess and handle I/O."""
        try:
            args = self._get_dsd_args()
            logger.debug(f"Starting DSD-FME (AMBE+2): {' '.join(args)}")

            self.process = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )

            # Create tasks for reading and writing
            write_task = asyncio.create_task(self._write_input())
            read_task = asyncio.create_task(self._read_output())

            await asyncio.gather(write_task, read_task, return_exceptions=True)

        except Exception as e:
            logger.error(f"AMBE+2 decoder error: {e}", exc_info=True)
        finally:
            if self.process:
                with contextlib.suppress(Exception):
                    self.process.terminate()

    async def _write_input(self) -> None:
        """Write discriminator audio to DSD-FME stdin."""
        loop = asyncio.get_running_loop()

        while self.running and self.process:
            try:
                # Get audio from queue
                audio = await asyncio.wait_for(self._input_queue.get(), timeout=0.5)

                if self.process and self.process.stdin:
                    # Convert float discriminator values to int16
                    # For Phase II CQPSK, the deviation is different from Phase I C4FM
                    # Scale factor adjusted for CQPSK symbol spacing
                    scaled = np.clip(audio * 12.0, -32767, 32767).astype(np.int16)
                    audio_bytes = scaled.tobytes()

                    await loop.run_in_executor(
                        None, self.process.stdin.write, audio_bytes
                    )
                    await loop.run_in_executor(None, self.process.stdin.flush)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error writing to AMBE decoder: {e}")
                break

    async def _read_output(self) -> None:
        """Read decoded PCM from DSD-FME stdout."""
        loop = asyncio.get_running_loop()

        # DSD-FME outputs 8kHz 16-bit PCM
        # Read enough samples for meaningful chunks
        chunk_samples = 160  # 20ms at 8kHz
        chunk_bytes = chunk_samples * 2  # 16-bit = 2 bytes

        while self.running and self.process:
            try:
                if self.process and self.process.stdout:
                    data = await loop.run_in_executor(
                        None, self.process.stdout.read, chunk_bytes
                    )
                    if not data:
                        break

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

                    # Queue for output
                    try:
                        self._output_queue.put_nowait(audio_f32)
                    except asyncio.QueueFull:
                        # Drop oldest
                        try:
                            self._output_queue.get_nowait()
                            self._output_queue.put_nowait(audio_f32)
                        except (asyncio.QueueEmpty, asyncio.QueueFull):
                            pass

            except Exception as e:
                logger.error(f"Error reading from AMBE decoder: {e}")
                break


def check_ambe_available() -> tuple[bool, str]:
    """
    Check if AMBE+2 decoding is available.

    Returns:
        (available, message) tuple
    """
    if AMBEDecoder.is_available():
        return True, "DSD-FME is available for AMBE+2 decoding"
    else:
        return False, (
            "DSD-FME not found. Install it for P25 Phase II voice decoding:\n"
            "  git clone https://github.com/lwvmobile/dsd-fme\n"
            "  cd dsd-fme && mkdir build && cd build\n"
            "  cmake .. && make -j4 && sudo make install\n"
            "\n"
            "Note: mbelib is also required:\n"
            "  git clone https://github.com/szechyjs/mbelib\n"
            "  cd mbelib && mkdir build && cd build\n"
            "  cmake .. && make -j4 && sudo make install"
        )


class DMRVoiceDecoder(AMBEDecoder):
    """
    Decodes DMR AMBE+2 voice to PCM audio using DSD-FME.

    DMR uses 4-FSK modulation at 4800 baud with AMBE+2 codec.
    This is a subclass of AMBEDecoder with DMR-specific DSD-FME flags.
    """

    def __init__(
        self,
        output_rate: int = 48000,
        input_rate: int = 48000,
        slot: int = 0,  # 0 = both, 1 or 2 = specific slot
    ):
        """
        Initialize DMR voice decoder.

        Args:
            output_rate: Target audio output sample rate (default 48000)
            input_rate: Input discriminator audio sample rate (default 48000)
            slot: DMR timeslot to decode (0=both, 1 or 2 for specific)
        """
        super().__init__(output_rate, input_rate, timeslot=slot)

    def _get_dsd_args(self) -> list[str]:
        """Get DSD-FME command line arguments for DMR."""
        args = [
            "dsd-fme",
            "-i", "-",           # Read from stdin
            "-o", "-",           # Write to stdout
            "-fd",               # DMR/DMR+ mode (4FSK)
            "-N",                # No audio output to speaker
            "-g", "0",           # No gain adjustment
            "-u", "0",           # No upsampling (we handle it)
            "-L",                # Low CPU mode (skip some features)
        ]

        # Specify timeslot if needed
        if self.timeslot in (1, 2):
            args.extend(["-s", str(self.timeslot)])

        return args

    def start_in_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Start the decoder from a sync context.

        Schedules the async start() method on the provided event loop.
        The decoder subprocess will run in the event loop's thread.

        Args:
            loop: Event loop to run the decoder in
        """
        if self.running:
            return

        if not self.is_available():
            logger.warning(
                "DSD-FME not found - DMR voice decoding disabled. "
                "Install from: https://github.com/lwvmobile/dsd-fme"
            )
            return

        try:
            asyncio.run_coroutine_threadsafe(self.start(), loop)
        except Exception as e:
            logger.error(f"Failed to start DMR voice decoder: {e}")

    async def start(self) -> None:
        """Start the DSD-FME decoder subprocess for DMR."""
        if self.running:
            return

        if not self.is_available():
            raise AMBEDecoderError(
                "DSD-FME not found in PATH. Install with: "
                "git clone https://github.com/lwvmobile/dsd-fme && "
                "cd dsd-fme && mkdir build && cd build && cmake .. && make && sudo make install"
            )

        self.running = True
        self._decoder_task = asyncio.create_task(self._run_decoder())
        logger.info(
            f"DMR voice decoder started (input={self.input_rate}Hz, output={self.output_rate}Hz, "
            f"slot={self.timeslot})"
        )

    async def stop(self) -> None:
        """Stop the decoder subprocess."""
        if not self.running:
            return

        self.running = False

        # Cancel decoder task
        if self._decoder_task:
            self._decoder_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._decoder_task

        # Terminate process
        if self.process:
            try:
                if self.process.stdin:
                    with contextlib.suppress(BrokenPipeError, OSError):
                        self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            except Exception as e:
                logger.debug(f"Exception during DMR voice decoder cleanup: {e}")
            finally:
                self.process = None

        logger.info(
            f"DMR voice decoder stopped (decoded={self.frames_decoded}, dropped={self.frames_dropped})"
        )

    async def _write_input(self) -> None:
        """Write discriminator audio to DSD-FME stdin."""
        loop = asyncio.get_running_loop()

        while self.running and self.process:
            try:
                # Get audio from queue
                audio = await asyncio.wait_for(self._input_queue.get(), timeout=0.5)

                if self.process and self.process.stdin:
                    # Convert float discriminator values to int16
                    # DMR uses ±1944 Hz deviation for 4FSK
                    # Scale factor: 32767 / 1944 ≈ 17
                    scaled = np.clip(audio * 17.0, -32767, 32767).astype(np.int16)
                    audio_bytes = scaled.tobytes()

                    await loop.run_in_executor(
                        None, self.process.stdin.write, audio_bytes
                    )
                    await loop.run_in_executor(None, self.process.stdin.flush)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error writing to DMR voice decoder: {e}")
                break


def check_dmr_voice_available() -> tuple[bool, str]:
    """
    Check if DMR voice decoding is available.

    Returns:
        (available, message) tuple
    """
    if DMRVoiceDecoder.is_available():
        return True, "DSD-FME is available for DMR voice decoding"
    else:
        return False, (
            "DSD-FME not found. Install it for DMR voice decoding:\n"
            "  git clone https://github.com/lwvmobile/dsd-fme\n"
            "  cd dsd-fme && mkdir build && cd build\n"
            "  cmake .. && make -j4 && sudo make install\n"
            "\n"
            "Note: mbelib is also required:\n"
            "  git clone https://github.com/szechyjs/mbelib\n"
            "  cd mbelib && mkdir build && cd build\n"
            "  cmake .. && make -j4 && sudo make install"
        )
