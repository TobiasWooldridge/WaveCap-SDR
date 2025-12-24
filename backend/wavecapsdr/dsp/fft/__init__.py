"""Pluggable FFT backends for spectrum analysis.

This module provides hardware-accelerated FFT implementations:
- scipy: Default CPU implementation (always available)
- fftw: SIMD-optimized CPU via pyFFTW (2-3x faster)
- mlx: Apple Metal GPU via MLX (macOS only, 5-10x faster)
- cuda: NVIDIA CUDA GPU via CuPy (10-20x faster)

Usage:
    from wavecapsdr.dsp.fft import get_backend, available_backends

    # Auto-detect best backend
    backend = get_backend()

    # Or specify explicitly
    backend = get_backend("mlx", fft_size=4096)

    # Compute FFT
    result = backend.execute(iq_samples, sample_rate)
    power_db = result.power_db  # Spectrum in dB
    freqs = result.freqs        # Frequency array
"""

from .base import FFTBackend, FFTResult
from .registry import available_backends, get_backend
from .scipy_backend import ScipyFFTBackend

__all__ = [
    "FFTBackend",
    "FFTResult",
    "ScipyFFTBackend",
    "available_backends",
    "get_backend",
]
