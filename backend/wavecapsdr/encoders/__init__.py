"""Audio encoders for streaming in various formats.

Encoders lazily spawn when clients subscribe and terminate when no subscribers remain.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EncoderConfig:
    """Configuration for an audio encoder."""

    format: str  # mp3, aac, opus, etc.
    bitrate: int = 128000  # bits per second
    sample_rate: int = 48000
    channels: int = 1


class AudioEncoder(ABC):
    """Base class for audio encoders."""

    def __init__(self, config: EncoderConfig):
        self.config = config
        self.process: subprocess.Popen[bytes] | None = None
        self.running = False
        self._input_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=32)
        self._output_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=32)
        self._encoder_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the encoder process."""
        if self.running:
            return

        self.running = True
        self._encoder_task = asyncio.create_task(self._run_encoder())
        logger.info(f"Started {self.config.format} encoder at {self.config.bitrate}bps")

    async def stop(self) -> None:
        """Stop the encoder process."""
        if not self.running:
            return

        self.running = False

        # Cancel encoder task
        if self._encoder_task:
            self._encoder_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._encoder_task

        # Terminate process
        if self.process:
            try:
                # Close stdin first to avoid BrokenPipeError
                if self.process.stdin:
                    try:
                        self.process.stdin.close()
                    except (BrokenPipeError, OSError):
                        pass  # Already closed or broken

                self.process.terminate()
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            except Exception as e:
                logger.debug(f"Exception during encoder process cleanup: {e}")
            finally:
                self.process = None

        logger.info(f"Stopped {self.config.format} encoder")

    async def encode(self, pcm_data: bytes) -> None:
        """Queue PCM data for encoding."""
        try:
            self._input_queue.put_nowait(pcm_data)
        except asyncio.QueueFull:
            # Drop oldest packet
            try:
                self._input_queue.get_nowait()
                self._input_queue.put_nowait(pcm_data)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass

    async def get_encoded(self) -> bytes:
        """Get encoded audio data."""
        return await self._output_queue.get()

    @abstractmethod
    def _get_ffmpeg_args(self) -> list[str]:
        """Get ffmpeg arguments for this encoder."""
        ...

    async def _run_encoder(self) -> None:
        """Run the encoder process and handle I/O."""
        try:
            # Start ffmpeg process
            args = self._get_ffmpeg_args()
            self.process = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )

            # Create tasks for reading and writing
            write_task = asyncio.create_task(self._write_input())
            read_task = asyncio.create_task(self._read_output())

            # Wait for both tasks
            await asyncio.gather(write_task, read_task, return_exceptions=True)

        except Exception as e:
            logger.error(f"Encoder error for {self.config.format}: {e}", exc_info=True)
        finally:
            if self.process:
                with contextlib.suppress(Exception):
                    self.process.terminate()

    async def _write_input(self) -> None:
        """Write PCM data to encoder stdin."""
        loop = asyncio.get_running_loop()
        while self.running and self.process:
            try:
                data = await asyncio.wait_for(self._input_queue.get(), timeout=0.5)
                if self.process and self.process.stdin:
                    await loop.run_in_executor(None, self.process.stdin.write, data)
                    await loop.run_in_executor(None, self.process.stdin.flush)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error writing to encoder: {e}")
                break

    async def _read_output(self) -> None:
        """Read encoded data from encoder stdout."""
        loop = asyncio.get_running_loop()
        chunk_size = 4096

        while self.running and self.process:
            try:
                if self.process and self.process.stdout:
                    data = await loop.run_in_executor(None, self.process.stdout.read, chunk_size)
                    if not data:
                        break
                    try:
                        self._output_queue.put_nowait(data)
                    except asyncio.QueueFull:
                        # Drop oldest
                        try:
                            self._output_queue.get_nowait()
                            self._output_queue.put_nowait(data)
                        except (asyncio.QueueEmpty, asyncio.QueueFull):
                            pass
            except Exception as e:
                logger.error(f"Error reading from encoder: {e}")
                break


class MP3Encoder(AudioEncoder):
    """MP3 audio encoder using ffmpeg."""

    def _get_ffmpeg_args(self) -> list[str]:
        return [
            "ffmpeg",
            "-f",
            "s16le",  # Input format: signed 16-bit little-endian PCM
            "-ar",
            str(self.config.sample_rate),
            "-ac",
            str(self.config.channels),
            "-i",
            "pipe:0",  # Read from stdin
            "-f",
            "mp3",
            "-b:a",
            str(self.config.bitrate),
            "-",  # Write to stdout
        ]


class OpusEncoder(AudioEncoder):
    """Opus audio encoder using ffmpeg."""

    def _get_ffmpeg_args(self) -> list[str]:
        return [
            "ffmpeg",
            "-f",
            "s16le",
            "-ar",
            str(self.config.sample_rate),
            "-ac",
            str(self.config.channels),
            "-i",
            "pipe:0",
            "-f",
            "opus",
            "-b:a",
            str(self.config.bitrate),
            "-",
        ]


class AACEncoder(AudioEncoder):
    """AAC audio encoder using ffmpeg."""

    def _get_ffmpeg_args(self) -> list[str]:
        return [
            "ffmpeg",
            "-f",
            "s16le",
            "-ar",
            str(self.config.sample_rate),
            "-ac",
            str(self.config.channels),
            "-i",
            "pipe:0",
            "-f",
            "adts",  # AAC ADTS format for streaming
            "-c:a",
            "aac",
            "-b:a",
            str(self.config.bitrate),
            "-",
        ]


def create_encoder(format: str, sample_rate: int = 48000, bitrate: int = 128000) -> AudioEncoder:
    """Factory function to create an encoder for the specified format."""
    config = EncoderConfig(format=format, sample_rate=sample_rate, bitrate=bitrate)

    if format == "mp3":
        return MP3Encoder(config)
    elif format == "opus":
        return OpusEncoder(config)
    elif format == "aac":
        return AACEncoder(config)
    else:
        raise ValueError(f"Unsupported encoder format: {format}")
