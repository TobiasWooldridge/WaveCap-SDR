"""CuPy FFT backend - NVIDIA CUDA GPU acceleration.

Uses CuPy for GPU-accelerated FFT on NVIDIA GPUs.
Provides 10-20x speedup for large FFT sizes.

Install: pip install cupy-cuda12x (adjust for your CUDA version)
Requires: NVIDIA GPU with CUDA support
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from wavecapsdr.typing import NDArrayComplex

from .base import FFTBackend, FFTResult

logger = logging.getLogger(__name__)

# Try to import CuPy
_cp: Any | None = None
CUPY_AVAILABLE = False

try:
    import cupy as cp

    _cp = cp
    CUPY_AVAILABLE = True
except ImportError:
    pass


class CuPyFFTBackend(FFTBackend):
    """NVIDIA CUDA GPU FFT backend using CuPy.

    CuPy provides numpy-like API with CUDA acceleration.
    Uses cuFFT under the hood for optimal GPU performance.

    Best for high sample rates (>10 MSPS) where GPU parallelism
    overcomes CPU-GPU transfer overhead.
    """

    def __init__(self, fft_size: int = 2048):
        """Initialize CuPy FFT backend.

        Args:
            fft_size: FFT size in samples

        Raises:
            ImportError: If CuPy is not installed or no CUDA GPU
        """
        if not CUPY_AVAILABLE or _cp is None:
            raise ImportError(
                "CuPy not available. Install with: pip install cupy-cuda12x\n"
                "Requires NVIDIA GPU with CUDA support"
            )

        super().__init__(fft_size)
        self._cp = _cp

        # Pre-allocate window on GPU
        self._window_gpu = self._cp.array(self.window)

        # Get GPU info
        try:
            device = self._cp.cuda.Device()
            gpu_name = device.attributes.get("Name", "Unknown")
            logger.info(f"CuPy FFT backend initialized on {gpu_name} (fft_size={fft_size})")
        except Exception:
            logger.info(f"CuPy FFT backend initialized (fft_size={fft_size})")

    def execute(self, iq: NDArrayComplex, sample_rate: int) -> FFTResult:
        """Compute FFT using CuPy on CUDA GPU.

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

        # Transfer to GPU
        chunk = iq[: self.fft_size].astype(np.complex64)
        chunk_gpu = self._cp.array(chunk)

        # Apply window on GPU
        windowed_gpu = chunk_gpu * self._window_gpu

        # Compute FFT on GPU (uses cuFFT)
        fft_result_gpu = self._cp.fft.fft(windowed_gpu)

        # Compute magnitude on GPU
        magnitude_gpu = self._cp.abs(fft_result_gpu)

        # FFT shift on GPU
        magnitude_shifted_gpu = self._cp.fft.fftshift(magnitude_gpu)

        # Transfer back to CPU
        magnitude_shifted = self._cp.asnumpy(magnitude_shifted_gpu)

        # Compute power in dB (on CPU - log is fast enough)
        power_db = 20.0 * np.log10(magnitude_shifted + 1e-10)

        # Generate frequency array (CPU)
        freqs = np.fft.fftshift(np.fft.fftfreq(self.fft_size, 1.0 / sample_rate)).astype(np.float32)

        return FFTResult(
            power_db=power_db.astype(np.float32),
            freqs=freqs,
            bin_hz=sample_rate / self.fft_size,
        )

    @property
    def name(self) -> str:
        """Return backend name."""
        return "cuda"


def is_available() -> bool:
    """Check if CuPy backend is available."""
    return CUPY_AVAILABLE
