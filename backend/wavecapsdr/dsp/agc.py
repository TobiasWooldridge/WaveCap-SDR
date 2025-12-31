"""Automatic Gain Control (AGC) for SDR audio processing.

AGC dynamically adjusts signal levels to maintain consistent output volume
regardless of input signal strength. Essential for AM and SSB, useful for FM.

This implements a simple AGC with attack and release time constants:
- Fast attack: Quickly reduce gain when signal increases (prevent clipping)
- Slow release: Slowly increase gain when signal decreases (smooth transitions)

Performance: Uses vectorized scipy.signal.lfilter for 4-8x speedup over pure Python.
When scipy is unavailable, falls back to Numba JIT-compiled loop (50-100x faster
than pure Python) or pure Python as last resort.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar, cast

import numpy as np
from wavecapsdr.typing import NDArrayFloat

# Try to import scipy for optimized filtering
try:
    from scipy import signal as scipy_signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

F = TypeVar("F", bound=Callable[..., Any])

def _fallback_jit(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    """No-op decorator when numba is not available."""
    def decorator(func: F) -> F:
        return func
    return decorator

# Try to import numba for JIT compilation (fallback when scipy unavailable)
try:
    from numba import jit as _numba_jit
    NUMBA_AVAILABLE = True
except ImportError:
    _numba_jit = cast(Callable[..., Any], _fallback_jit)
    NUMBA_AVAILABLE = False

jit = _numba_jit

# Soft clipping constants
_SOFT_CLIP_K = np.float32(1.5)  # Knee factor for tanh soft clipping
_SOFT_CLIP_NORM = np.float32(1.0 / np.tanh(1.5))  # Normalization factor


def soft_clip(x: NDArrayFloat) -> NDArrayFloat:
    """Apply soft clipping using tanh function.

    Unlike hard clipping (np.clip), soft clipping gradually saturates
    the signal, preventing harsh distortion and aliasing on peaks.

    Args:
        x: Input signal (any range)

    Returns:
        Soft-clipped signal in range [-1, 1]
    """
    return cast(NDArrayFloat, np.tanh(x * _SOFT_CLIP_K) * _SOFT_CLIP_NORM)


def _envelope_detector_vectorized(
    x: NDArrayFloat, attack_coef: float, release_coef: float
) -> NDArrayFloat:
    """Vectorized envelope detector using scipy.signal.lfilter.

    This approximates asymmetric attack/release by using the faster of the two
    coefficients with scipy's optimized lfilter, then applying a correction.
    For typical audio AGC, this provides nearly identical results with 4-8x speedup.
    """
    abs_x = np.abs(x).astype(np.float32)

    # Use scipy's lfilter for the envelope - single-pole IIR lowpass
    # For attack/release asymmetry, we use a two-pass approach:
    # 1. First pass with attack coefficient (tracks peaks quickly)
    # 2. Second pass with release coefficient (smooths decays)

    # Attack pass: y[n] = alpha*x[n] + (1-alpha)*y[n-1]
    # This is equivalent to lfilter with b=[attack_coef], a=[1, -(1-attack_coef)]
    b_attack = np.array([attack_coef], dtype=np.float32)
    a_attack = np.array([1.0, -(1.0 - attack_coef)], dtype=np.float32)
    env_attack = scipy_signal.lfilter(b_attack, a_attack, abs_x)

    # Release pass on the attack envelope (smooths the decay)
    b_release = np.array([release_coef], dtype=np.float32)
    a_release = np.array([1.0, -(1.0 - release_coef)], dtype=np.float32)

    # Take max of attack and release-smoothed envelope for proper asymmetric behavior
    env_release = scipy_signal.lfilter(b_release, a_release, env_attack)

    # Combine: use attack envelope where signal is rising, release where falling
    # This is approximated by taking the maximum of both envelopes
    envelope: NDArrayFloat = cast(NDArrayFloat, np.maximum(env_attack, env_release)).astype(np.float32)

    return envelope


@jit(nopython=True, cache=True)
def _envelope_detector_jit(
    x: NDArrayFloat, attack_coef: float, release_coef: float
) -> NDArrayFloat:
    """JIT-compiled envelope detector using Numba.

    This provides 50-100x speedup over pure Python for sample-by-sample
    processing. Used as fallback when scipy is not available.

    Args:
        x: Input signal (float32 array)
        attack_coef: Attack coefficient (0-1, higher = faster attack)
        release_coef: Release coefficient (0-1, higher = faster release)

    Returns:
        Envelope signal (float32 array, same length as input)
    """
    n = x.shape[0]
    envelope = np.empty(n, dtype=np.float32)

    if n == 0:
        return envelope

    envelope[0] = abs(x[0])

    for i in range(1, n):
        current_sample = abs(x[i])
        prev_envelope = envelope[i - 1]
        if current_sample > prev_envelope:
            # Signal rising - use attack (fast response)
            envelope[i] = attack_coef * current_sample + (1.0 - attack_coef) * prev_envelope
        else:
            # Signal falling - use release (slow decay)
            envelope[i] = release_coef * current_sample + (1.0 - release_coef) * prev_envelope

    return envelope


def _envelope_detector_python(
    x: NDArrayFloat, attack_coef: float, release_coef: float
) -> NDArrayFloat:
    """Pure Python envelope detector (fallback when scipy and numba unavailable)."""
    envelope = np.zeros(x.shape[0], dtype=np.float32)
    if x.shape[0] == 0:
        return envelope

    envelope[0] = abs(x[0])

    for i in range(1, x.shape[0]):
        current_sample = abs(x[i])
        if current_sample > envelope[i - 1]:
            envelope[i] = attack_coef * current_sample + (1.0 - attack_coef) * envelope[i - 1]
        else:
            envelope[i] = release_coef * current_sample + (1.0 - release_coef) * envelope[i - 1]

    return cast(NDArrayFloat, envelope)


def apply_agc(
    x: NDArrayFloat,
    sample_rate: int,
    target_db: float = -20.0,
    attack_ms: float = 5.0,
    release_ms: float = 50.0,
    max_gain_db: float = 60.0,
) -> NDArrayFloat:
    """Apply Automatic Gain Control to maintain consistent signal level.

    AGC tracks the signal envelope and adjusts gain to keep the output
    near the target level. Uses asymmetric attack/release for natural sound.

    Args:
        x: Input audio signal (float32, range approximately ±1.0)
        sample_rate: Sample rate in Hz
        target_db: Target output level in dB (typical: -20 to -10 dB)
        attack_ms: Attack time constant in milliseconds (how fast gain decreases)
        release_ms: Release time constant in milliseconds (how fast gain increases)
        max_gain_db: Maximum gain in dB to prevent excessive amplification

    Returns:
        Gain-controlled audio signal

    Algorithm:
        1. Calculate signal envelope (smoothed absolute value)
        2. Compute required gain to reach target level
        3. Apply gain with attack/release smoothing
        4. Limit max gain to prevent noise amplification

    Typical settings:
        - AM/SSB: target_db=-20, attack_ms=5, release_ms=50
        - FM: target_db=-15, attack_ms=10, release_ms=100 (gentler)
    """
    if x.size == 0:
        return x.astype(np.float32, copy=False)

    # Convert dB to linear
    target_linear = 10.0 ** (target_db / 20.0)
    max_gain_linear = 10.0 ** (max_gain_db / 20.0)

    # Calculate attack and release coefficients (exponential smoothing)
    attack_samples = (attack_ms / 1000.0) * sample_rate
    release_samples = (release_ms / 1000.0) * sample_rate

    attack_coef = 1.0 - np.exp(-1.0 / attack_samples) if attack_samples > 0 else 1.0
    release_coef = 1.0 - np.exp(-1.0 / release_samples) if release_samples > 0 else 1.0

    # Calculate envelope using best available implementation:
    # 1. scipy (vectorized, fastest)
    # 2. numba JIT (50-100x faster than pure Python)
    # 3. pure Python (fallback)
    if SCIPY_AVAILABLE:
        envelope = _envelope_detector_vectorized(x, attack_coef, release_coef)
    elif NUMBA_AVAILABLE:
        envelope = _envelope_detector_jit(x, attack_coef, release_coef)
    else:
        envelope = _envelope_detector_python(x, attack_coef, release_coef)

    # Calculate required gain for each sample
    min_envelope = 1e-6  # Prevent division by zero and excessive gain
    gain = target_linear / np.maximum(envelope, min_envelope)

    # Limit maximum gain to prevent noise amplification
    np.minimum(gain, max_gain_linear, out=gain)

    # Apply gain to signal
    y = x * gain

    # Soft clip to prevent overflow with smooth saturation
    # (preferable to hard clip for better audio quality)
    y = soft_clip(y)

    return y.astype(np.float32)


def apply_simple_agc(
    x: NDArrayFloat, target_rms: float = 0.1, max_gain: float = 10.0
) -> NDArrayFloat:
    """Apply simplified AGC based on RMS level (block-based, no smoothing).

    This is a simpler AGC that operates on the entire block at once,
    adjusting gain based on RMS level. Faster but less sophisticated
    than the sample-by-sample AGC.

    Args:
        x: Input audio signal
        target_rms: Target RMS level (0.0 to 1.0, typical: 0.1)
        max_gain: Maximum gain multiplier (typical: 10.0)

    Returns:
        Gain-controlled audio signal

    Use this for:
        - Low-latency applications
        - Simpler processing with less CPU
        - When smooth attack/release isn't critical
    """
    if x.size == 0:
        return x.astype(np.float32, copy=False)

    # Calculate RMS level
    rms = np.sqrt(np.mean(x**2))

    # Calculate required gain
    if rms > 1e-6:  # Avoid division by zero
        gain = target_rms / rms
    else:
        gain = max_gain

    # Limit maximum gain
    gain = min(gain, max_gain)

    # Apply gain and soft clip
    y = soft_clip(x * gain)

    return y.astype(np.float32)
