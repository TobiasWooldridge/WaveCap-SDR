"""pyFFTW backend - SIMD-optimized CPU FFT.

Uses FFTW (Fastest Fourier Transform in the West) via pyFFTW bindings.
Provides 2-3x speedup over scipy/numpy through:
- Pre-planned FFT operations
- SIMD instructions (SSE2, AVX, AVX2)
- Multi-threaded execution

Install: pip install pyfftw
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .base import FFTBackend, FFTResult

logger = logging.getLogger(__name__)

# Try to import pyFFTW
_pyfftw: Any | None = None
PYFFTW_AVAILABLE = False

try:
    import pyfftw

    _pyfftw = pyfftw
    PYFFTW_AVAILABLE = True
except ImportError:
    pass


class FFTWBackend(FFTBackend):
    """FFTW-based FFT backend with pre-planned operations.

    FFTW uses "wisdom" to optimize FFT for specific sizes.
    Pre-planning amortizes the planning cost over many executions.

    Provides 2-3x speedup over scipy through SIMD and threading.
    """

    def __init__(self, fft_size: int = 2048, threads: int = 2):
        """Initialize FFTW backend with pre-planned FFT.

        Args:
            fft_size: FFT size in samples
            threads: Number of threads for FFT computation

        Raises:
            ImportError: If pyFFTW is not installed
        """
        if not PYFFTW_AVAILABLE or _pyfftw is None:
            raise ImportError(
                "pyFFTW not available. Install with: pip install pyfftw"
            )

        super().__init__(fft_size)
        self._pyfftw = _pyfftw
        self._threads = threads

        # Create aligned input/output arrays for SIMD
        self._input = self._pyfftw.empty_aligned(fft_size, dtype="complex64")
        self._output = self._pyfftw.empty_aligned(fft_size, dtype="complex64")

        # Pre-plan FFT (this is where FFTW gains performance)
        # FFTW_MEASURE takes longer to plan but runs faster
        self._fft_plan = self._pyfftw.FFTW(
            self._input,
            self._output,
            direction="FFTW_FORWARD",
            flags=("FFTW_MEASURE",),
            threads=threads,
        )

        logger.info(
            f"FFTW backend initialized (fft_size={fft_size}, threads={threads})"
        )

    def execute(self, iq: np.ndarray, sample_rate: int) -> FFTResult:
        """Compute FFT using pre-planned FFTW.

        Args:
            iq: Complex IQ samples
            sample_rate: Sample rate in Hz

        Returns:
            FFTResult with power spectrum in dB
        """
        if iq.size < self.fft_size:
            return FFTResult(
                power_db=np.zeros(self.fft_size, dtype=np.float32),
                freqs=np.zeros(self.fft_size, dtype=np.float32),
                bin_hz=sample_rate / self.fft_size,
            )

        # Copy windowed data to aligned input buffer
        chunk = iq[: self.fft_size].astype(np.complex64)
        np.multiply(chunk, self.window, out=self._input)

        # Execute pre-planned FFT
        self._fft_plan()

        # Get result and shift
        fft_shifted = np.fft.fftshift(self._output)

        # Calculate power spectrum in dB
        magnitude = np.abs(fft_shifted)
        power_db = 20.0 * np.log10(magnitude + 1e-10)

        # Generate frequency array
        freqs = np.fft.fftshift(
            np.fft.fftfreq(self.fft_size, 1.0 / sample_rate)
        ).astype(np.float32)

        return FFTResult(
            power_db=power_db.astype(np.float32),
            freqs=freqs,
            bin_hz=sample_rate / self.fft_size,
        )

    @property
    def name(self) -> str:
        """Return backend name."""
        return "fftw"


def is_available() -> bool:
    """Check if FFTW backend is available."""
    return PYFFTW_AVAILABLE
