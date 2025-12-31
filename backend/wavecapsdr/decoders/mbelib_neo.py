"""Direct Python ctypes bindings to mbelib-neo for IMBE/AMBE decoding.

This bypasses DSD-FME entirely, providing maximum performance for P25 voice
decoding by calling mbelib-neo's SIMD-accelerated routines directly.

mbelib-neo provides:
- FFT-based unvoiced synthesis (cleaner audio)
- SIMD acceleration (SSE2/NEON)
- Adaptive smoothing for corrupted frames
- WOLA for smooth frame transitions
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
from ctypes import POINTER, Structure, c_char, c_float, c_int, c_short, c_uint32
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
from wavecapsdr.typing import NDArrayAny

logger = logging.getLogger(__name__)


# ============================================================================
# mbelib-neo C structure definitions
# ============================================================================


class MbeParms(Structure):
    """mbe_parms structure from mbelib-neo.

    Contains vocoder parameters for one frame plus synthesis state.
    """

    _fields_ = [
        # Core vocoder parameters
        ("w0", c_float),  # Fundamental radian frequency
        ("L", c_int),  # Number of harmonic bands
        ("K", c_int),  # Number of voiced bands
        ("Vl", c_int * 57),  # Voiced/unvoiced flags per band
        ("Ml", c_float * 57),  # Magnitude per band
        ("log2Ml", c_float * 57),  # Base-2 log magnitude per band
        ("PHIl", c_float * 57),  # Absolute phase per band
        ("PSIl", c_float * 57),  # Smoothed phase per band
        ("gamma", c_float),  # Spectral amplitude enhancement scale
        ("un", c_int),  # Legacy/unused
        ("repeat", c_int),  # Repeat frame flag (legacy)
        ("swn", c_int),  # Sine wave increment for tone synthesis
        # Adaptive smoothing state
        ("localEnergy", c_float),  # Local energy tracking
        ("amplitudeThreshold", c_int),  # Amplitude threshold for scaling
        ("errorRate", c_float),  # Bit error rate (0.0 to 1.0)
        ("errorCountTotal", c_int),  # Total bit errors in frame
        ("errorCount4", c_int),  # Coset 4 error count (IMBE)
        # Frame repeat/muting state
        ("repeatCount", c_int),  # Consecutive repeat count
        ("mutingThreshold", c_float),  # Muting threshold
        # FFT-based unvoiced synthesis state
        ("previousUw", c_float * 256),  # Previous frame inverse FFT output
        ("noiseSeed", c_float),  # LCG noise generator state
        ("noiseOverlap", c_float * 96),  # Noise buffer overlap
    ]


# IMBE frame types
IMBE_FRAME_7200x4400 = (8, 23)  # 8 rows x 23 bits = 184 bits
IMBE_FRAME_7100x4400 = (7, 24)  # 7 rows x 24 bits = 168 bits

# Audio output
SAMPLES_PER_FRAME = 160  # 20ms at 8kHz
OUTPUT_SAMPLE_RATE = 8000


class MbelibNeoError(Exception):
    """Raised when mbelib-neo operations fail."""


def _find_mbelib_neo() -> str | None:
    """Find the mbelib-neo shared library."""
    system = platform.system()

    # Possible library names and paths
    if system == "Darwin":
        lib_names = ["libmbe-neo.dylib", "libmbe-neo.1.dylib", "libmbe-neo.1.1.dylib"]
        search_paths = [
            "/usr/local/lib",
            "/opt/homebrew/lib",
            str(Path.home() / ".local/lib"),
            # Local build path
            "/Users/thw/Projects/mbelib-neo/build/dev-release",
        ]
    elif system == "Linux":
        lib_names = ["libmbe-neo.so", "libmbe-neo.so.1", "libmbe-neo.so.1.1"]
        search_paths = [
            "/usr/local/lib",
            "/usr/lib",
            "/usr/lib/x86_64-linux-gnu",
            str(Path.home() / ".local/lib"),
        ]
    elif system == "Windows":
        lib_names = ["mbe-neo.dll", "libmbe-neo.dll"]
        search_paths = os.environ.get("PATH", "").split(os.pathsep)
    else:
        return None

    # Search for library
    for path in search_paths:
        for name in lib_names:
            full_path = Path(path) / name
            if full_path.exists():
                return str(full_path)

    return None


# Global library handle (lazy loaded)
_lib: ctypes.CDLL | None = None
_lib_path: str | None = None


def _get_lib() -> ctypes.CDLL:
    """Get or load the mbelib-neo library."""
    global _lib, _lib_path

    if _lib is not None:
        return _lib

    lib_path = _find_mbelib_neo()
    if lib_path is None:
        raise MbelibNeoError(
            "mbelib-neo not found. Install from: https://github.com/arancormonk/mbelib-neo"
        )

    try:
        _lib = ctypes.CDLL(lib_path)
        _lib_path = lib_path
        _setup_function_signatures(_lib)
        logger.info(f"Loaded mbelib-neo from {lib_path}")
        return _lib
    except OSError as e:
        raise MbelibNeoError(f"Failed to load mbelib-neo from {lib_path}: {e}")


def _setup_function_signatures(lib: ctypes.CDLL) -> None:
    """Set up ctypes function signatures for type safety."""

    # Version
    lib.mbe_versionString.argtypes = []
    lib.mbe_versionString.restype = ctypes.c_char_p

    # Initialization
    lib.mbe_initMbeParms.argtypes = [
        POINTER(MbeParms),  # cur_mp
        POINTER(MbeParms),  # prev_mp
        POINTER(MbeParms),  # prev_mp_enhanced
    ]
    lib.mbe_initMbeParms.restype = None

    # Thread RNG seed
    lib.mbe_setThreadRngSeed.argtypes = [c_uint32]
    lib.mbe_setThreadRngSeed.restype = None

    # IMBE 7200x4400 (P25 Phase 1)
    # char imbe_fr[8][23], char imbe_d[88]
    ImbeFrame = c_char * 23 * 8
    ImbeData = c_char * 88

    lib.mbe_processImbe7200x4400Framef.argtypes = [
        POINTER(c_float),  # aout_buf (160 floats)
        POINTER(c_int),  # errs
        POINTER(c_int),  # errs2
        ctypes.c_char_p,  # err_str
        ImbeFrame,  # imbe_fr[8][23]
        ImbeData,  # imbe_d[88]
        POINTER(MbeParms),  # cur_mp
        POINTER(MbeParms),  # prev_mp
        POINTER(MbeParms),  # prev_mp_enhanced
        c_int,  # uvquality
    ]
    lib.mbe_processImbe7200x4400Framef.restype = None

    # IMBE 7100x4400 (ProVoice)
    ImbeFrame7100 = c_char * 24 * 7

    lib.mbe_processImbe7100x4400Framef.argtypes = [
        POINTER(c_float),  # aout_buf (160 floats)
        POINTER(c_int),  # errs
        POINTER(c_int),  # errs2
        ctypes.c_char_p,  # err_str
        ImbeFrame7100,  # imbe_fr[7][24]
        ImbeData,  # imbe_d[88]
        POINTER(MbeParms),  # cur_mp
        POINTER(MbeParms),  # prev_mp
        POINTER(MbeParms),  # prev_mp_enhanced
        c_int,  # uvquality
    ]
    lib.mbe_processImbe7100x4400Framef.restype = None

    # Float to short conversion
    lib.mbe_floattoshort.argtypes = [
        POINTER(c_float),  # float_buf (160 floats)
        POINTER(c_short),  # aout_buf (160 shorts)
    ]
    lib.mbe_floattoshort.restype = None

    # Silence synthesis
    lib.mbe_synthesizeSilencef.argtypes = [POINTER(c_float)]
    lib.mbe_synthesizeSilencef.restype = None

    # Comfort noise
    lib.mbe_synthesizeComfortNoisef.argtypes = [POINTER(c_float)]
    lib.mbe_synthesizeComfortNoisef.restype = None

    # Muting check
    lib.mbe_requiresMuting.argtypes = [POINTER(MbeParms)]
    lib.mbe_requiresMuting.restype = c_int


def get_version() -> str:
    """Get mbelib-neo version string."""
    lib = _get_lib()
    version_bytes = cast(bytes, lib.mbe_versionString())
    return version_bytes.decode("utf-8")


def is_available() -> bool:
    """Check if mbelib-neo is available."""
    try:
        _get_lib()
        return True
    except MbelibNeoError:
        return False


def check_available() -> tuple[bool, str]:
    """Check availability with message."""
    try:
        lib = _get_lib()
        version_bytes = cast(bytes, lib.mbe_versionString())
        version = version_bytes.decode("utf-8")
        return True, f"mbelib-neo {version} available at {_lib_path}"
    except MbelibNeoError as e:
        return False, str(e)


# ============================================================================
# IMBE Decoder using mbelib-neo
# ============================================================================


@dataclass
class IMBEDecoderNeo:
    """IMBE decoder using mbelib-neo for P25 Phase 1 voice.

    This decoder processes IMBE voice frames directly, without needing
    DSD-FME or any external process. It uses mbelib-neo's SIMD-accelerated
    routines for maximum performance.

    Usage:
        decoder = IMBEDecoderNeo()
        decoder.start()

        # Process IMBE frames (from P25 LDU1/LDU2)
        for imbe_frame in voice_frames:
            pcm = decoder.decode_frame(imbe_frame)
            if pcm is not None:
                # 160 samples of 8kHz audio
                process_audio(pcm)

        decoder.stop()
    """

    # Output configuration
    output_rate: int = 48000  # Target sample rate (will resample from 8kHz)
    uvquality: int = 3  # Unvoiced synthesis quality (1-64, 3 is good balance)

    # State
    _lib: ctypes.CDLL | None = field(default=None, repr=False)
    _cur_mp: MbeParms | None = field(default=None, repr=False)
    _prev_mp: MbeParms | None = field(default=None, repr=False)
    _prev_mp_enhanced: MbeParms | None = field(default=None, repr=False)

    # Buffers (pre-allocated for performance)
    _audio_buf: NDArrayAny | None = field(default=None, repr=False)
    _audio_buf_ptr: ctypes._Pointer[c_float] | None = field(default=None, repr=False)
    _imbe_frame: ctypes.Array[ctypes.Array[c_char]] | None = field(default=None, repr=False)
    _imbe_data: ctypes.Array[c_char] | None = field(default=None, repr=False)
    _errs: ctypes.c_int = field(default_factory=lambda: ctypes.c_int(0), repr=False)
    _errs2: ctypes.c_int = field(default_factory=lambda: ctypes.c_int(0), repr=False)
    _err_str: ctypes.Array[c_char] | None = field(default=None, repr=False)

    # Resampling (8kHz -> output_rate)
    _resample_ratio: float = field(default=1.0, init=False)

    # Statistics
    frames_decoded: int = 0
    frames_errored: int = 0
    total_errors: int = 0
    running: bool = False

    def __post_init__(self) -> None:
        self._resample_ratio = self.output_rate / OUTPUT_SAMPLE_RATE

    @staticmethod
    def is_available() -> bool:
        """Check if mbelib-neo is available."""
        return is_available()

    def start(self) -> None:
        """Initialize the decoder."""
        if self.running:
            return

        # Load library
        self._lib = _get_lib()
        lib = self._lib

        # Set thread RNG seed for deterministic output
        lib.mbe_setThreadRngSeed(12345)

        # Allocate parameter structures
        self._cur_mp = MbeParms()
        self._prev_mp = MbeParms()
        self._prev_mp_enhanced = MbeParms()
        cur_mp = self._cur_mp
        prev_mp = self._prev_mp
        prev_mp_enhanced = self._prev_mp_enhanced

        # Initialize parameters
        lib.mbe_initMbeParms(
            ctypes.byref(cur_mp),
            ctypes.byref(prev_mp),
            ctypes.byref(prev_mp_enhanced),
        )

        # Allocate audio buffer (160 samples at 8kHz = 20ms)
        self._audio_buf = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)
        self._audio_buf_ptr = self._audio_buf.ctypes.data_as(POINTER(c_float))

        # Allocate IMBE frame buffer (8 x 23 chars)
        ImbeFrame = c_char * 23 * 8
        self._imbe_frame = ImbeFrame()

        # Allocate IMBE data buffer (88 chars)
        ImbeData = c_char * 88
        self._imbe_data = ImbeData()

        # Error string buffer
        self._err_str = ctypes.create_string_buffer(64)

        # Reset statistics
        self.frames_decoded = 0
        self.frames_errored = 0
        self.total_errors = 0
        self.running = True

        version_bytes = cast(bytes, self._lib.mbe_versionString())
        version = version_bytes.decode("utf-8")
        logger.info(
            f"IMBEDecoderNeo started (mbelib-neo {version}, output_rate={self.output_rate})"
        )

    def stop(self) -> None:
        """Stop the decoder and release resources."""
        if not self.running:
            return

        self.running = False

        # Clear references
        self._cur_mp = None
        self._prev_mp = None
        self._prev_mp_enhanced = None
        self._audio_buf = None
        self._audio_buf_ptr = None
        self._imbe_frame = None
        self._imbe_data = None

        logger.info(
            f"IMBEDecoderNeo stopped (decoded={self.frames_decoded}, "
            f"errored={self.frames_errored}, total_errors={self.total_errors})"
        )

    def decode_frame(self, imbe_bits: NDArrayAny | bytes) -> NDArrayAny | None:
        """Decode a single IMBE 7200x4400 frame to PCM audio.

        Args:
            imbe_bits: IMBE frame as either:
                - numpy array of 144 bits (0/1 values)
                - bytes of 18 bytes (packed bits)
                - numpy array shaped (8, 23) of bits

        Returns:
            PCM audio as float32 array normalized to [-1, 1], resampled to
            output_rate. Returns None if decoding fails or frame is muted.
        """
        if not self.running or self._lib is None:
            return None

        # Convert input to 8x23 bit array for mbelib
        try:
            frame_bits = self._prepare_imbe_frame(imbe_bits)
        except ValueError as e:
            logger.warning(f"Invalid IMBE frame: {e}")
            self.frames_errored += 1
            return None

        lib = self._lib
        audio_buf = self._audio_buf
        audio_buf_ptr = self._audio_buf_ptr
        imbe_frame = self._imbe_frame
        imbe_data = self._imbe_data
        cur_mp = self._cur_mp
        prev_mp = self._prev_mp
        prev_mp_enhanced = self._prev_mp_enhanced
        err_str = self._err_str
        if (
            lib is None
            or audio_buf is None
            or audio_buf_ptr is None
            or imbe_frame is None
            or imbe_data is None
            or cur_mp is None
            or prev_mp is None
            or prev_mp_enhanced is None
            or err_str is None
        ):
            return None

        # Copy bits to ctypes buffer
        for row in range(8):
            for col in range(23):
                bit_idx = row * 23 + col
                if bit_idx < len(frame_bits):
                    imbe_frame[row][col] = bytes([frame_bits[bit_idx]])
                else:
                    imbe_frame[row][col] = b"\x00"

        # Decode frame
        lib.mbe_processImbe7200x4400Framef(
            audio_buf_ptr,
            ctypes.byref(self._errs),
            ctypes.byref(self._errs2),
            err_str,
            imbe_frame,
            imbe_data,
            ctypes.byref(cur_mp),
            ctypes.byref(prev_mp),
            ctypes.byref(prev_mp_enhanced),
            self.uvquality,
        )

        # Update statistics
        self.frames_decoded += 1
        self.total_errors += self._errs.value + self._errs2.value

        # Check if frame should be muted
        if lib.mbe_requiresMuting(ctypes.byref(cur_mp)):
            self.frames_errored += 1
            # Return comfort noise instead of silence
            lib.mbe_synthesizeComfortNoisef(audio_buf_ptr)

        # Get audio (already in self._audio_buf)
        audio = np.asarray(audio_buf.copy(), dtype=np.float32)

        # Normalize to [-1, 1] range (mbelib outputs ~[-8000, 8000])
        audio = audio / 8000.0
        audio = np.clip(audio, -1.0, 1.0)

        # Resample if needed
        if self._resample_ratio != 1.0:
            from scipy import signal

            # Calculate rational resampling factors
            up = self.output_rate
            down = OUTPUT_SAMPLE_RATE
            gcd = np.gcd(up, down)
            up //= gcd
            down //= gcd
            audio = signal.resample_poly(audio, up, down).astype(np.float32)

        return audio

    def _prepare_imbe_frame(self, imbe_bits: NDArrayAny | bytes) -> NDArrayAny:
        """Convert various input formats to flat bit array."""
        if isinstance(imbe_bits, bytes):
            # Unpack bytes to bits
            bits = np.unpackbits(np.frombuffer(imbe_bits, dtype=np.uint8))
            return bits[:184]  # 8*23 = 184 bits

        imbe_bits = np.asarray(imbe_bits)

        if imbe_bits.ndim == 2 and imbe_bits.shape == (8, 23):
            # Already in correct shape
            return imbe_bits.flatten().astype(np.uint8)

        if imbe_bits.ndim == 1:
            if len(imbe_bits) >= 144:
                # Pad to 184 bits if needed
                if len(imbe_bits) < 184:
                    padded = np.zeros(184, dtype=np.uint8)
                    padded[: len(imbe_bits)] = imbe_bits
                    return padded
                return imbe_bits[:184].astype(np.uint8)

        raise ValueError(
            f"Invalid IMBE frame shape: {imbe_bits.shape}. "
            f"Expected (8, 23), (144,), (184,), or 18 bytes."
        )

    def decode_silence(self) -> NDArrayAny:
        """Generate a frame of silence."""
        if (
            not self.running
            or self._lib is None
            or self._audio_buf is None
            or self._audio_buf_ptr is None
        ):
            return np.zeros(int(SAMPLES_PER_FRAME * self._resample_ratio), dtype=np.float32)

        self._lib.mbe_synthesizeSilencef(self._audio_buf_ptr)
        audio = np.asarray(self._audio_buf.copy(), dtype=np.float32) / 8000.0

        if self._resample_ratio != 1.0:
            from scipy import signal

            up = self.output_rate
            down = OUTPUT_SAMPLE_RATE
            gcd = np.gcd(up, down)
            audio = signal.resample_poly(audio, up // gcd, down // gcd).astype(np.float32)

        return audio


# ============================================================================
# Convenience functions
# ============================================================================


def create_imbe_decoder(output_rate: int = 48000) -> IMBEDecoderNeo:
    """Create an IMBE decoder instance."""
    return IMBEDecoderNeo(output_rate=output_rate)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    available, msg = check_available()
    print(f"mbelib-neo available: {available}")
    print(f"Message: {msg}")

    if available:
        print(f"Version: {get_version()}")

        # Test decoder
        decoder = IMBEDecoderNeo(output_rate=48000)
        decoder.start()

        # Decode a silence frame (all zeros)
        silence = decoder.decode_silence()
        print(f"Silence frame: {len(silence)} samples")

        # Try decoding a dummy frame
        dummy_frame = np.zeros(184, dtype=np.uint8)
        audio = decoder.decode_frame(dummy_frame)
        if audio is not None:
            print(
                f"Decoded frame: {len(audio)} samples, range [{audio.min():.3f}, {audio.max():.3f}]"
            )

        decoder.stop()
