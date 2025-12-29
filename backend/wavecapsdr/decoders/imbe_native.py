"""Native IMBE decoder using mbelib-neo with integrated P25 frame detection.

This decoder bypasses DSD-FME entirely by:
1. Using our C4FM demodulator for symbol extraction
2. Using our P25 frame parser for LDU detection
3. Using mbelib-neo for IMBE voice synthesis

This gives maximum performance and eliminates subprocess overhead.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from wavecapsdr.decoders.mbelib_neo import IMBEDecoderNeo, MbelibNeoError, is_available
from wavecapsdr.decoders.p25_frames import DUID, decode_ldu1, decode_ldu2, decode_nid
from wavecapsdr.decoders.nac_tracker import NACTracker
from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator

logger = logging.getLogger(__name__)


class IMBENativeError(Exception):
    """Raised when native IMBE decoder fails."""


@dataclass
class IMBEDecoderNative:
    """Native IMBE decoder using mbelib-neo with integrated P25 frame processing.

    This decoder processes discriminator audio directly:
    1. C4FM demodulates to dibits
    2. Detects P25 LDU frames (voice units)
    3. Extracts IMBE voice codewords
    4. Decodes via mbelib-neo to PCM

    This eliminates the DSD-FME subprocess entirely for maximum performance.

    Usage:
        decoder = IMBEDecoderNative(output_rate=48000)
        decoder.start()

        # Process discriminator audio from FM demodulator
        decoder.decode(disc_audio)

        # Get decoded audio
        while (audio := decoder.get_audio()) is not None:
            process_audio(audio)

        decoder.stop()
    """

    # I/O rates
    output_rate: int = 48000
    input_rate: int = 48000  # Discriminator audio sample rate

    # Queue sizes
    INPUT_QUEUE_SIZE: int = 256
    OUTPUT_QUEUE_SIZE: int = 128

    # State
    running: bool = False
    _c4fm: C4FMDemodulator | None = field(default=None, repr=False)
    _mbelib: IMBEDecoderNeo | None = field(default=None, repr=False)

    # Threading
    _process_thread: threading.Thread | None = field(default=None, repr=False)
    _input_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=256), repr=False)
    _output_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=128), repr=False)

    # Callback for output audio
    on_audio: Callable[[np.ndarray], None] | None = field(default=None, repr=False)

    # P25 frame state
    _dibit_buffer: np.ndarray | None = field(default=None, repr=False)
    _sync_state: str = "searching"  # searching, synced
    _frame_position: int = 0
    _nac_tracker: NACTracker | None = field(default=None, repr=False)

    # Resampling (input_rate -> 50kHz for C4FM)
    _input_resample_up: int = field(default=0, init=False)
    _input_resample_down: int = field(default=0, init=False)

    # Statistics
    frames_decoded: int = 0
    frames_dropped: int = 0
    ldu_frames: int = 0
    imbe_frames: int = 0
    bytes_processed: int = 0

    def __post_init__(self) -> None:
        # C4FM at 48kHz with 4800 baud = exactly 10 samples/symbol
        # No resampling needed if input is already 48kHz
        self._c4fm_rate = 48000
        self._input_resample_up = self._c4fm_rate
        self._input_resample_down = self.input_rate
        gcd = np.gcd(self._input_resample_up, self._input_resample_down)
        self._input_resample_up //= gcd
        self._input_resample_down //= gcd

    @staticmethod
    def is_available() -> bool:
        """Check if mbelib-neo is available."""
        return is_available()

    def start(self) -> None:
        """Start the native IMBE decoder."""
        if self.running:
            return

        if not self.is_available():
            raise IMBENativeError(
                "mbelib-neo not found. Install from: "
                "https://github.com/arancormonk/mbelib-neo"
            )

        # Initialize C4FM demodulator at 48kHz (10 samples/symbol at 4800 baud)
        self._c4fm = C4FMDemodulator(sample_rate=self._c4fm_rate)

        # Initialize mbelib decoder
        self._mbelib = IMBEDecoderNeo(output_rate=self.output_rate)
        self._mbelib.start()

        # Initialize NAC tracker
        self._nac_tracker = NACTracker()

        # Initialize dibit buffer for frame accumulation
        self._dibit_buffer = np.zeros(2000, dtype=np.uint8)
        self._frame_position = 0
        self._sync_state = "searching"

        # Reset statistics
        self.frames_decoded = 0
        self.frames_dropped = 0
        self.ldu_frames = 0
        self.imbe_frames = 0
        self.bytes_processed = 0

        # Start processing thread
        self.running = True
        self._process_thread = threading.Thread(
            target=self._process_loop,
            name="imbe-native",
            daemon=True,
        )
        self._process_thread.start()

        logger.info(
            f"IMBEDecoderNative started (input={self.input_rate}Hz, output={self.output_rate}Hz)"
        )

    def stop(self) -> None:
        """Stop the decoder."""
        if not self.running:
            return

        self.running = False

        # Signal thread to exit
        try:
            self._input_queue.put(None, timeout=0.5)
        except queue.Full:
            pass

        # Wait for thread
        if self._process_thread and self._process_thread.is_alive():
            self._process_thread.join(timeout=2.0)

        # Stop mbelib decoder
        if self._mbelib:
            self._mbelib.stop()
            self._mbelib = None

        self._c4fm = None

        logger.info(
            f"IMBEDecoderNative stopped (ldu={self.ldu_frames}, imbe={self.imbe_frames}, "
            f"decoded={self.frames_decoded}, dropped={self.frames_dropped})"
        )

    def decode(self, discriminator_audio: np.ndarray) -> None:
        """Queue discriminator audio for decoding (thread-safe)."""
        if not self.running:
            return

        try:
            self._input_queue.put_nowait(discriminator_audio)
        except queue.Full:
            self.frames_dropped += 1
            # Drop oldest and add new
            try:
                self._input_queue.get_nowait()
                self._input_queue.put_nowait(discriminator_audio)
            except (queue.Empty, queue.Full):
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

    def _process_loop(self) -> None:
        """Main processing loop (runs in thread)."""
        logger.debug("IMBEDecoderNative process thread started")

        while self.running:
            try:
                # Get audio from queue
                audio = self._input_queue.get(timeout=0.1)

                if audio is None:  # Shutdown signal
                    break

                self.bytes_processed += len(audio) * 4  # float32

                # Process audio
                self._process_audio(audio)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"IMBEDecoderNative error: {e}", exc_info=True)

        logger.debug("IMBEDecoderNative process thread exiting")

    def _process_audio(self, disc_audio: np.ndarray) -> None:
        """Process discriminator audio through the pipeline."""
        if self._c4fm is None or self._mbelib is None:
            return

        # Resample to C4FM sample rate if needed
        # C4FM demodulator expects 48kHz for 10 samples/symbol at 4800 baud
        if self._input_resample_up != self._input_resample_down:
            disc_audio = signal.resample_poly(
                disc_audio, self._input_resample_up, self._input_resample_down
            ).astype(np.float32)

        # C4FM demodulate discriminator audio to dibits
        # Use the discriminator-specific method that skips FM demod
        dibits, soft_symbols = self._c4fm.demodulate_discriminator(disc_audio)

        if len(dibits) == 0:
            return

        # Process dibits through P25 frame parser
        self._process_dibits(dibits)

    def _process_dibits(self, dibits: np.ndarray) -> None:
        """Process dibits through P25 frame detection and IMBE extraction."""
        # Add to buffer
        if self._dibit_buffer is None:
            return

        # Simple approach: slide through looking for frame sync
        for dibit in dibits:
            # Add to circular buffer
            if self._frame_position < len(self._dibit_buffer):
                self._dibit_buffer[self._frame_position] = dibit
                self._frame_position += 1
            else:
                # Shift buffer and add new dibit
                self._dibit_buffer[:-1] = self._dibit_buffer[1:]
                self._dibit_buffer[-1] = dibit

            # Check for frame sync every N dibits
            if self._frame_position >= 32 and self._frame_position % 10 == 0:
                self._check_for_frame()

    def _check_for_frame(self) -> None:
        """Check if we have a complete P25 frame."""
        if self._dibit_buffer is None or self._frame_position < 32:
            return

        # Get last ~900 dibits (LDU frame size)
        frame_start = max(0, self._frame_position - 900)
        frame_dibits = self._dibit_buffer[frame_start:self._frame_position]

        if len(frame_dibits) < 32:
            return

        # Try to decode NID to identify frame type
        nid = decode_nid(frame_dibits[:32], nac_tracker=self._nac_tracker)
        if nid is None:
            return

        # Track NAC
        if self._nac_tracker and nid.nac != 0:
            self._nac_tracker.update(nid.nac)

        # Process based on DUID
        if nid.duid == DUID.LDU1:
            self._process_ldu(frame_dibits, is_ldu1=True)
        elif nid.duid == DUID.LDU2:
            self._process_ldu(frame_dibits, is_ldu1=False)

    def _process_ldu(self, dibits: np.ndarray, is_ldu1: bool) -> None:
        """Process an LDU frame and extract IMBE voice."""
        if self._mbelib is None:
            return

        # Decode LDU
        if is_ldu1:
            ldu = decode_ldu1(dibits)
        else:
            ldu = decode_ldu2(dibits)

        if ldu is None:
            return

        self.ldu_frames += 1

        # Process each IMBE frame
        for imbe_bytes in ldu.imbe_frames:
            if len(imbe_bytes) == 0:
                continue

            self.imbe_frames += 1

            # Convert bytes to bits for mbelib
            # IMBE frame is 88 bits = 11 bytes
            # mbelib wants it as char[8][23] = 184 bits
            # We need to expand/pad the 88-bit frame to 184 bits
            imbe_bits = self._expand_imbe_frame(imbe_bytes)

            # Decode with mbelib
            try:
                audio = self._mbelib.decode_frame(imbe_bits)
                if audio is not None and len(audio) > 0:
                    self.frames_decoded += 1

                    # Callback
                    if self.on_audio:
                        try:
                            self.on_audio(audio)
                        except Exception as e:
                            logger.error(f"on_audio callback error: {e}")

                    # Queue for output
                    try:
                        self._output_queue.put_nowait(audio)
                    except queue.Full:
                        try:
                            self._output_queue.get_nowait()
                            self._output_queue.put_nowait(audio)
                        except queue.Empty:
                            pass

            except Exception as e:
                logger.debug(f"IMBE decode error: {e}")

    def _expand_imbe_frame(self, imbe_bytes: bytes) -> np.ndarray:
        """Expand 88-bit IMBE frame to 184-bit format for mbelib.

        mbelib expects char imbe_fr[8][23] which is 184 bits.
        The actual IMBE voice data is 88 bits.

        The 8x23 structure represents the interleaved format:
        - 8 codewords of 23 bits each
        - These are FEC-protected voice parameters

        For now, we'll use a simple mapping. A proper implementation
        would handle the full interleaving/de-interleaving.
        """
        # Convert bytes to bits
        bits = np.unpackbits(np.frombuffer(imbe_bytes, dtype=np.uint8))

        # Pad to 184 bits (8 * 23)
        result = np.zeros(184, dtype=np.uint8)
        result[:min(len(bits), 184)] = bits[:184] if len(bits) >= 184 else bits

        return result


def check_native_imbe_available() -> tuple[bool, str]:
    """Check if native IMBE decoding is available."""
    if IMBEDecoderNative.is_available():
        return True, "mbelib-neo is available for native IMBE decoding"
    else:
        return False, (
            "mbelib-neo not found. Install from: "
            "https://github.com/arancormonk/mbelib-neo"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    available, msg = check_native_imbe_available()
    print(f"Native IMBE available: {available}")
    print(f"Message: {msg}")

    if available:
        decoder = IMBEDecoderNative(output_rate=48000, input_rate=48000)
        decoder.start()

        # Test with silence
        silence = np.zeros(4800, dtype=np.float32)
        decoder.decode(silence)

        import time
        time.sleep(0.5)

        print(f"Frames decoded: {decoder.frames_decoded}")
        print(f"LDU frames: {decoder.ldu_frames}")
        print(f"IMBE frames: {decoder.imbe_frames}")

        decoder.stop()
