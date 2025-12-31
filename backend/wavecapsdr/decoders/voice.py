"""Unified P25 voice decoder for trunking applications.

Provides a single interface for decoding both:
- P25 Phase I: IMBE (88-bit frames, C4FM modulation)
- P25 Phase II: AMBE+2 (49-bit frames, CQPSK/TDMA modulation)

This module wraps the IMBEDecoder and AMBEDecoder classes to provide
a unified interface for the trunking system.

Decoder backends:
- Native (mbelib-neo): Direct ctypes bindings for maximum performance
- Threaded (DSD-FME): Subprocess-based, fallback when mbelib-neo unavailable
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np
from wavecapsdr.typing import NDArrayFloat

from wavecapsdr.decoders.ambe import AMBEDecoder, AMBEDecoderError, check_ambe_available
from wavecapsdr.decoders.imbe import IMBEDecoderError, check_imbe_available
from wavecapsdr.decoders.imbe_threaded import IMBEDecoderThreaded
from wavecapsdr.decoders.imbe_native import (
    IMBEDecoderNative,
    IMBENativeError,
    check_native_imbe_available,
)
from wavecapsdr.decoders.mbelib_neo import (
    is_available as mbelib_neo_available,
    check_available as check_mbelib_neo,
)

logger = logging.getLogger(__name__)


class VocoderType(str, Enum):
    """P25 vocoder types."""

    IMBE = "imbe"  # Phase I
    AMBE2 = "ambe2"  # Phase II


class VoiceDecoderError(Exception):
    """Raised when voice decoder fails."""


@dataclass
class VoiceDecoderStats:
    """Voice decoder statistics."""

    frames_decoded: int = 0
    frames_dropped: int = 0
    frames_errored: int = 0
    audio_chunks_output: int = 0
    last_decode_time: float = 0.0


@dataclass
class VoiceDecoder:
    """Unified P25 voice decoder.

    Wraps IMBE (Phase I) and AMBE+2 (Phase II) decoders to provide a
    single interface for trunking voice channel decoding.

    Automatically selects the best available decoder backend:
    - Native (mbelib-neo): Direct ctypes bindings, maximum performance
    - Threaded (DSD-FME): Subprocess-based fallback

    Usage:
        decoder = VoiceDecoder(vocoder_type=VocoderType.IMBE)
        await decoder.start()

        # Process discriminator audio
        await decoder.decode(audio_samples)

        # Get decoded PCM
        pcm = await decoder.get_audio()

        await decoder.stop()
    """

    vocoder_type: VocoderType = VocoderType.IMBE
    output_rate: int = 48000
    input_rate: int = 48000
    timeslot: int = 0  # For Phase II TDMA: 0=both, 1 or 2 for specific slot
    prefer_native: bool = True  # Prefer mbelib-neo native decoder when available

    # Internal decoders
    _imbe_decoder: IMBEDecoderThreaded | None = field(default=None, repr=False)
    _imbe_native: IMBEDecoderNative | None = field(default=None, repr=False)
    _ambe_decoder: AMBEDecoder | None = field(default=None, repr=False)
    _using_native: bool = field(default=False, repr=False)

    # State
    running: bool = False
    _stats: VoiceDecoderStats = field(default_factory=VoiceDecoderStats)

    # Callbacks
    on_audio: Callable[[NDArrayFloat], None] | None = None
    on_error: Callable[[str], None] | None = None

    def __post_init__(self) -> None:
        """Initialize internal state."""
        self._stats = VoiceDecoderStats()
        self._using_native = False

    @staticmethod
    def is_available(vocoder_type: VocoderType, native_only: bool = False) -> bool:
        """Check if a vocoder type is available.

        Args:
            vocoder_type: The vocoder type to check
            native_only: If True, only check for native mbelib-neo availability
        """
        if vocoder_type == VocoderType.IMBE:
            # Check for native first (preferred)
            if IMBEDecoderNative.is_available():
                return True
            # Fall back to DSD-FME threaded decoder
            if not native_only:
                return IMBEDecoderThreaded.is_available()
            return False
        return AMBEDecoder.is_available()

    @staticmethod
    def check_availability() -> dict[str, Any]:
        """Check availability of all vocoders.

        Returns:
            Dictionary with availability status and messages for each vocoder.
        """
        # Check native IMBE (mbelib-neo)
        native_available, native_msg = check_native_imbe_available()

        # Check threaded IMBE (DSD-FME)
        threaded_available, threaded_msg = check_imbe_available()

        # IMBE is available if either native or threaded works
        imbe_available = native_available or threaded_available
        if native_available:
            imbe_msg = f"Native: {native_msg}"
        elif threaded_available:
            imbe_msg = f"Threaded: {threaded_msg}"
        else:
            imbe_msg = f"Native: {native_msg}; Threaded: {threaded_msg}"

        # Check AMBE
        ambe_available, ambe_msg = check_ambe_available()

        return {
            "imbe": {
                "available": imbe_available,
                "native_available": native_available,
                "native_message": native_msg,
                "threaded_available": threaded_available,
                "threaded_message": threaded_msg,
                "message": imbe_msg,
            },
            "ambe2": {
                "available": ambe_available,
                "message": ambe_msg,
            },
            "any_available": imbe_available or ambe_available,
        }

    async def start(self) -> None:
        """Start the voice decoder."""
        if self.running:
            return

        # Create and start the appropriate decoder
        try:
            if self.vocoder_type == VocoderType.IMBE:
                # Try native decoder first if preferred and available
                use_native = self.prefer_native and IMBEDecoderNative.is_available()

                if use_native:
                    self._imbe_native = IMBEDecoderNative(
                        output_rate=self.output_rate,
                        input_rate=self.input_rate,
                    )
                    self._imbe_native.start()
                    self._using_native = True
                    logger.info("VoiceDecoder: Started IMBE decoder (native/mbelib-neo)")
                else:
                    # Fall back to threaded DSD-FME decoder
                    self._imbe_decoder = IMBEDecoderThreaded(
                        output_rate=self.output_rate,
                        input_rate=self.input_rate,
                    )
                    self._imbe_decoder.start()
                    self._using_native = False
                    logger.info("VoiceDecoder: Started IMBE decoder (threaded/DSD-FME)")

            elif self.vocoder_type == VocoderType.AMBE2:
                self._ambe_decoder = AMBEDecoder(
                    output_rate=self.output_rate,
                    input_rate=self.input_rate,
                    timeslot=self.timeslot,
                )
                await self._ambe_decoder.start()
                logger.info(f"VoiceDecoder: Started AMBE+2 decoder (timeslot={self.timeslot})")

            self.running = True

        except (IMBEDecoderError, IMBENativeError, AMBEDecoderError) as e:
            raise VoiceDecoderError(str(e)) from e

    async def stop(self) -> None:
        """Stop the voice decoder."""
        if not self.running:
            return

        self.running = False
        backend = "native" if self._using_native else "threaded"

        # Stop the active decoder
        if self._imbe_native:
            self._imbe_native.stop()
            self._stats.frames_decoded = self._imbe_native.frames_decoded
            self._stats.frames_dropped = self._imbe_native.frames_dropped
            self._imbe_native = None

        if self._imbe_decoder:
            # Threaded decoder uses sync stop
            self._imbe_decoder.stop()
            self._stats.frames_decoded = self._imbe_decoder.frames_decoded
            self._stats.frames_dropped = self._imbe_decoder.frames_dropped
            self._imbe_decoder = None

        if self._ambe_decoder:
            await self._ambe_decoder.stop()
            self._stats.frames_decoded = self._ambe_decoder.frames_decoded
            self._stats.frames_dropped = self._ambe_decoder.frames_dropped
            self._ambe_decoder = None

        logger.info(
            f"VoiceDecoder: Stopped ({self.vocoder_type.value}/{backend}, "
            f"decoded={self._stats.frames_decoded}, dropped={self._stats.frames_dropped})"
        )

    async def decode(self, discriminator_audio: NDArrayFloat) -> None:
        """
        Queue discriminator audio for decoding.

        Args:
            discriminator_audio: Instantaneous frequency values from FM discriminator
        """
        if not self.running:
            return

        if self._imbe_native:
            # Native decoder uses sync decode (thread-safe)
            self._imbe_native.decode(discriminator_audio)
        elif self._imbe_decoder:
            # Threaded decoder uses sync decode (thread-safe)
            self._imbe_decoder.decode(discriminator_audio)
        elif self._ambe_decoder:
            await self._ambe_decoder.decode(discriminator_audio)

    async def get_audio(self) -> NDArrayFloat | None:
        """
        Get decoded audio if available.

        Returns:
            PCM audio samples as float32 [-1, 1] or None
        """
        if not self.running:
            return None

        audio = None
        if self._imbe_native:
            # Native decoder uses sync get_audio (non-blocking)
            audio = self._imbe_native.get_audio()
        elif self._imbe_decoder:
            # Threaded decoder uses sync get_audio (non-blocking)
            audio = self._imbe_decoder.get_audio()
        elif self._ambe_decoder:
            audio = await self._ambe_decoder.get_audio()

        if audio is not None:
            self._stats.audio_chunks_output += 1
            if self.on_audio:
                try:
                    self.on_audio(audio)
                except Exception as e:
                    logger.error(f"VoiceDecoder: on_audio callback error: {e}")

        return audio

    async def get_audio_blocking(self, timeout: float = 0.5) -> NDArrayFloat | None:
        """
        Get decoded audio, waiting up to timeout seconds.

        Returns:
            PCM audio samples or None if timeout
        """
        if not self.running:
            return None

        audio = None
        if self._imbe_native:
            # Native decoder uses sync get_audio_blocking
            audio = self._imbe_native.get_audio_blocking(timeout)
        elif self._imbe_decoder:
            # Threaded decoder uses sync get_audio_blocking
            audio = self._imbe_decoder.get_audio_blocking(timeout)
        elif self._ambe_decoder:
            audio = await self._ambe_decoder.get_audio_blocking(timeout)

        if audio is not None:
            self._stats.audio_chunks_output += 1
            if self.on_audio:
                try:
                    self.on_audio(audio)
                except Exception as e:
                    logger.error(f"VoiceDecoder: on_audio callback error: {e}")

        return audio

    def get_stats(self) -> dict[str, Any]:
        """Get decoder statistics."""
        # Update stats from active decoder
        if self._imbe_native:
            self._stats.frames_decoded = self._imbe_native.frames_decoded
            self._stats.frames_dropped = self._imbe_native.frames_dropped
        elif self._imbe_decoder:
            self._stats.frames_decoded = self._imbe_decoder.frames_decoded
            self._stats.frames_dropped = self._imbe_decoder.frames_dropped
        elif self._ambe_decoder:
            self._stats.frames_decoded = self._ambe_decoder.frames_decoded
            self._stats.frames_dropped = self._ambe_decoder.frames_dropped

        return {
            "vocoder_type": self.vocoder_type.value,
            "backend": "native" if self._using_native else "threaded",
            "running": self.running,
            "frames_decoded": self._stats.frames_decoded,
            "frames_dropped": self._stats.frames_dropped,
            "frames_errored": self._stats.frames_errored,
            "audio_chunks_output": self._stats.audio_chunks_output,
        }


def create_voice_decoder(
    protocol: str,
    output_rate: int = 48000,
    input_rate: int = 48000,
    timeslot: int = 0,
    prefer_native: bool = True,
) -> VoiceDecoder:
    """Create a voice decoder for the specified protocol.

    Args:
        protocol: "p25_phase1" or "p25_phase2"
        output_rate: Target audio output sample rate
        input_rate: Input discriminator audio sample rate
        timeslot: For Phase II, which TDMA slot to decode
        prefer_native: Use mbelib-neo native decoder when available

    Returns:
        Configured VoiceDecoder instance
    """
    if protocol in ("p25_phase1", "imbe"):
        vocoder_type = VocoderType.IMBE
    elif protocol in ("p25_phase2", "ambe2", "ambe+2"):
        vocoder_type = VocoderType.AMBE2
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    return VoiceDecoder(
        vocoder_type=vocoder_type,
        output_rate=output_rate,
        input_rate=input_rate,
        timeslot=timeslot,
        prefer_native=prefer_native,
    )
