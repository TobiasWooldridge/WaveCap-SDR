"""SciPy FFT backend - the default CPU implementation.

Uses scipy.fft which is 2-3x faster than numpy.fft.
Falls back to numpy.fft if scipy is unavailable.
"""

from __future__ import annotations

import numpy as np
from wavecapsdr.typing import NDArrayComplex

from .base import FFTBackend, FFTResult

# Try to import scipy.fft (faster than numpy.fft)
try:
    from scipy.fft import fft, fftfreq, fftshift

    SCIPY_FFT_AVAILABLE = True
except ImportError:
    SCIPY_FFT_AVAILABLE = False


class ScipyFFTBackend(FFTBackend):
    """Default FFT backend using scipy.fft.

    Falls back to numpy.fft if scipy is unavailable.
    This is the baseline implementation - always available.
    """

    def __init__(self, fft_size: int = 2048):
        """Initialize scipy FFT backend.

        Args:
            fft_size: FFT size in samples
        """
        super().__init__(fft_size)

    def execute(self, iq: NDArrayComplex, sample_rate: int) -> FFTResult:
        """Compute FFT using scipy.fft.

        Args:
            iq: Complex IQ samples
            sample_rate: Sample rate in Hz

        Returns:
            FFTResult with power spectrum in dB
        """
        if iq.size < self.fft_size:
            # Return empty result for insufficient samples
            return FFTResult(
                power_db=np.zeros(self.fft_size, dtype=np.float32),
                freqs=np.zeros(self.fft_size, dtype=np.float32),
                bin_hz=sample_rate / self.fft_size,
            )

        # Take chunk and apply window
        chunk = iq[: self.fft_size]
        windowed = chunk * self.window

        # Compute FFT
        if SCIPY_FFT_AVAILABLE:
            fft_result = fft(windowed)
            fft_shifted = fftshift(fft_result)
            freqs = fftshift(fftfreq(self.fft_size, 1.0 / sample_rate))
        else:
            # Fallback to numpy
            fft_result = np.fft.fft(windowed)
            fft_shifted = np.fft.fftshift(fft_result)
            freqs = np.fft.fftshift(np.fft.fftfreq(self.fft_size, 1.0 / sample_rate))

        # Calculate power spectrum in dB
        magnitude = np.abs(fft_shifted)
        power_db = 20.0 * np.log10(magnitude + 1e-10)

        return FFTResult(
            power_db=power_db.astype(np.float32),
            freqs=freqs.astype(np.float32),
            bin_hz=sample_rate / self.fft_size,
        )

    @property
    def name(self) -> str:
        """Return backend name."""
        return "scipy" if SCIPY_FFT_AVAILABLE else "numpy"
