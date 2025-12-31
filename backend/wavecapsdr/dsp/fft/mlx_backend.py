"""MLX FFT backend - Apple Metal GPU acceleration.

Uses Apple's MLX framework for GPU-accelerated FFT on Apple Silicon.
MLX provides native Metal support with numpy-like API.

Install: pip install mlx
Requires: macOS with Apple Silicon (M1/M2/M3)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from wavecapsdr.typing import NDArrayComplex

from .base import FFTBackend, FFTResult

logger = logging.getLogger(__name__)

# Try to import MLX
_mx: Any | None = None
MLX_AVAILABLE = False

try:
    import mlx.core as mx

    _mx = mx
    MLX_AVAILABLE = True
except ImportError:
    pass


class MLXFFTBackend(FFTBackend):
    """Apple Metal GPU FFT backend using MLX framework.

    MLX is Apple's ML framework with native Metal support.
    Provides 5-10x speedup over CPU for large FFT sizes.

    Note: Optimal for batch processing. Single FFT may have
    CPU transfer overhead that reduces benefit for small sizes.
    """

    def __init__(self, fft_size: int = 2048):
        """Initialize MLX FFT backend.

        Args:
            fft_size: FFT size in samples

        Raises:
            ImportError: If MLX is not installed
        """
        if not MLX_AVAILABLE or _mx is None:
            raise ImportError(
                "MLX not available. Install with: pip install mlx\n"
                "Requires macOS with Apple Silicon (M1/M2/M3)"
            )

        super().__init__(fft_size)
        self._mx = _mx

        # Pre-allocate window on GPU
        self._window_mx = self._mx.array(self.window)

        logger.info(f"MLX FFT backend initialized (fft_size={fft_size})")

    def execute(self, iq: NDArrayComplex, sample_rate: int) -> FFTResult:
        """Compute FFT using MLX on Metal GPU.

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

        # Take chunk and transfer to GPU
        chunk = iq[: self.fft_size].astype(np.complex64)
        chunk_mx = self._mx.array(chunk)

        # Apply window on GPU
        # MLX doesn't support complex * real directly, so split and combine
        chunk_real = self._mx.real(chunk_mx) * self._window_mx
        chunk_imag = self._mx.imag(chunk_mx) * self._window_mx

        # Reconstruct complex array (MLX way)
        # MLX complex support is limited, use real FFT workaround or cast
        windowed_np = (np.array(chunk_real) + 1j * np.array(chunk_imag)).astype(
            np.complex64
        )

        # MLX FFT (as of MLX 0.5+)
        windowed_mx = self._mx.array(windowed_np)
        fft_result = self._mx.fft.fft(windowed_mx)

        # Compute magnitude on GPU
        magnitude_mx = self._mx.abs(fft_result)

        # Transfer back to CPU for fftshift and log
        magnitude = np.array(magnitude_mx)

        # FFT shift (move 0 Hz to center)
        magnitude_shifted = np.fft.fftshift(magnitude)

        # Compute power in dB
        power_db = 20.0 * np.log10(magnitude_shifted + 1e-10)

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
        return "mlx"


def is_available() -> bool:
    """Check if MLX backend is available."""
    return MLX_AVAILABLE
