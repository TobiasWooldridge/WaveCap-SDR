"""FFT backend registry with auto-detection.

Manages available FFT backends and provides automatic selection
based on available hardware and libraries.

Priority order (auto mode):
1. MLX (Apple Metal) - macOS with Apple Silicon
2. CuPy (CUDA) - NVIDIA GPU
3. pyFFTW - SIMD-optimized CPU
4. scipy - Default fallback (always available)
"""

from __future__ import annotations

import logging
import platform
from typing import Any, Callable

from .base import FFTBackend

logger = logging.getLogger(__name__)

# Backend registry
_BACKENDS: dict[str, type[FFTBackend]] = {}


def register(name: str) -> Callable[[type[FFTBackend]], type[FFTBackend]]:
    """Decorator to register an FFT backend.

    Args:
        name: Backend identifier (e.g., 'scipy', 'mlx', 'cuda')
    """

    def decorator(cls: type[FFTBackend]) -> type[FFTBackend]:
        _BACKENDS[name] = cls
        return cls

    return decorator


def _try_create_backend(name: str, fft_size: int, **kwargs: Any) -> FFTBackend | None:
    """Try to create a backend, returning None if unavailable."""
    if name not in _BACKENDS:
        return None

    try:
        return _BACKENDS[name](fft_size=fft_size, **kwargs)
    except ImportError as e:
        logger.debug(f"Backend '{name}' not available: {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize backend '{name}': {e}")
        return None


def get_backend(
    accelerator: str = "auto",
    fft_size: int = 2048,
    **kwargs: Any,
) -> FFTBackend:
    """Get an FFT backend by name or auto-detect best available.

    Args:
        accelerator: Backend name or 'auto' for auto-detection
            Options: 'auto', 'scipy', 'fftw', 'mlx', 'cuda'
        fft_size: FFT size in samples
        **kwargs: Additional backend-specific arguments

    Returns:
        FFTBackend instance

    Auto-detection priority (size-aware):
        For small FFTs (≤4096): scipy/fftw (CPU faster due to transfer overhead)
        For large FFTs (>4096): MLX/CUDA (GPU parallelism pays off)
    """
    # Ensure backends are registered
    _ensure_registered()

    if accelerator == "auto":
        # Platform-aware auto-detection
        is_macos = platform.system() == "Darwin"

        # For small FFTs, CPU is faster (GPU transfer overhead dominates)
        # For large FFTs, GPU parallelism pays off
        use_gpu = fft_size > 4096

        if is_macos:
            if use_gpu:
                # Large FFT: try Metal GPU first
                priority = ["mlx", "fftw", "scipy"]
            else:
                # Small FFT: CPU is faster
                priority = ["fftw", "scipy"]
        else:
            if use_gpu:
                # Large FFT: try CUDA GPU first
                priority = ["cuda", "fftw", "scipy"]
            else:
                # Small FFT: CPU is faster
                priority = ["fftw", "scipy"]

        for name in priority:
            backend = _try_create_backend(name, fft_size, **kwargs)
            if backend is not None:
                logger.info(
                    f"Auto-selected FFT backend: {backend.name} "
                    f"(fft_size={fft_size}, use_gpu={use_gpu})"
                )
                return backend

        # Should never reach here (scipy is always available)
        raise RuntimeError("No FFT backend available")

    # Specific backend requested
    backend = _try_create_backend(accelerator, fft_size, **kwargs)
    if backend is not None:
        return backend

    # Requested backend not available, fall back to scipy
    logger.warning(f"Requested FFT backend '{accelerator}' not available, falling back to scipy")
    return _BACKENDS["scipy"](fft_size=fft_size)


def available_backends() -> list[str]:
    """Get list of available backend names."""
    _ensure_registered()

    available = []
    for name in _BACKENDS:
        try:
            backend = _BACKENDS[name](fft_size=256)
            available.append(name)
            del backend
        except (ImportError, Exception):
            pass

    return available


def _ensure_registered() -> None:
    """Ensure all backends are registered."""
    if _BACKENDS:
        return

    # Import backends to trigger registration
    # scipy is always available
    from .scipy_backend import ScipyFFTBackend

    _BACKENDS["scipy"] = ScipyFFTBackend

    # Optional backends
    try:
        from .mlx_backend import MLXFFTBackend, is_available

        if is_available():
            _BACKENDS["mlx"] = MLXFFTBackend
    except ImportError:
        pass

    try:
        from .fftw_backend import FFTWBackend, is_available

        if is_available():
            _BACKENDS["fftw"] = FFTWBackend
    except ImportError:
        pass

    try:
        from .cupy_backend import CuPyFFTBackend, is_available

        if is_available():
            _BACKENDS["cuda"] = CuPyFFTBackend
    except ImportError:
        pass

    logger.debug(f"Registered FFT backends: {list(_BACKENDS.keys())}")
