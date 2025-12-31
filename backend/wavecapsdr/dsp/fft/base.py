"""Base classes for FFT backends.

This module defines the abstract interface for FFT backends,
allowing pluggable implementations (scipy, pyFFTW, MLX, CuPy).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from wavecapsdr.typing import NDArrayComplex, NDArrayFloat


@dataclass
class FFTResult:
    """Result from FFT computation.

    Attributes:
        power_db: Power spectrum in dB (fftshifted, 0 Hz at center)
        freqs: Frequency array in Hz (fftshifted)
        bin_hz: Hz per bin
    """

    power_db: NDArrayFloat
    freqs: NDArrayFloat
    bin_hz: float


class FFTBackend(ABC):
    """Abstract base class for FFT backends.

    All backends must implement:
    - execute(): Compute FFT and return FFTResult
    - name property: Return backend identifier

    Backends handle:
    - Windowing (Hanning by default)
    - FFT computation
    - Magnitude and dB conversion
    - FFT shift (0 Hz at center)
    """

    def __init__(self, fft_size: int = 2048):
        """Initialize FFT backend.

        Args:
            fft_size: FFT size in samples (power of 2 recommended)
        """
        self.fft_size = fft_size
        self._window: NDArrayFloat | None = None

    @property
    def window(self) -> NDArrayFloat:
        """Get cached Hanning window."""
        if self._window is None or len(self._window) != self.fft_size:
            self._window = np.hanning(self.fft_size).astype(np.float32)
        return self._window

    @abstractmethod
    def execute(self, iq: NDArrayComplex, sample_rate: int) -> FFTResult:
        """Compute FFT and return power spectrum in dB.

        Args:
            iq: Complex IQ samples (at least fft_size samples)
            sample_rate: Sample rate in Hz

        Returns:
            FFTResult with power_db, freqs, and bin_hz
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return backend identifier (e.g., 'scipy', 'mlx', 'cuda')."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fft_size={self.fft_size})"
