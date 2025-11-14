"""Automatic Gain Control (AGC) for SDR audio processing.

AGC dynamically adjusts signal levels to maintain consistent output volume
regardless of input signal strength. Essential for AM and SSB, useful for FM.

This implements a simple AGC with attack and release time constants:
- Fast attack: Quickly reduce gain when signal increases (prevent clipping)
- Slow release: Slowly increase gain when signal decreases (smooth transitions)
"""

from __future__ import annotations

import numpy as np


def apply_agc(
    x: np.ndarray,
    sample_rate: int,
    target_db: float = -20.0,
    attack_ms: float = 5.0,
    release_ms: float = 50.0,
    max_gain_db: float = 60.0,
) -> np.ndarray:
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
    # alpha = 1 - exp(-T / tau) where T is sample period, tau is time constant
    attack_samples = (attack_ms / 1000.0) * sample_rate
    release_samples = (release_ms / 1000.0) * sample_rate

    attack_coef = 1.0 - np.exp(-1.0 / attack_samples) if attack_samples > 0 else 1.0
    release_coef = (
        1.0 - np.exp(-1.0 / release_samples) if release_samples > 0 else 1.0
    )

    # Calculate envelope (peak detector with attack/release)
    envelope = np.zeros(x.shape[0], dtype=np.float32)
    envelope[0] = abs(x[0])

    for i in range(1, x.shape[0]):
        current_sample = abs(x[i])

        # Use attack coefficient if signal is increasing, release if decreasing
        if current_sample > envelope[i - 1]:
            # Attack: fast response to increases
            envelope[i] = (
                attack_coef * current_sample + (1.0 - attack_coef) * envelope[i - 1]
            )
        else:
            # Release: slow response to decreases
            envelope[i] = (
                release_coef * current_sample + (1.0 - release_coef) * envelope[i - 1]
            )

    # Calculate required gain for each sample
    # gain = target / envelope (with protection against division by zero)
    min_envelope = 1e-6  # Prevent division by zero and excessive gain
    gain = target_linear / np.maximum(envelope, min_envelope)

    # Limit maximum gain to prevent noise amplification
    gain = np.minimum(gain, max_gain_linear)

    # Apply gain to signal
    y = x * gain

    # Hard clip to prevent overflow (should rarely be needed with proper AGC)
    y = np.clip(y, -1.0, 1.0)

    return y.astype(np.float32)


def apply_simple_agc(
    x: np.ndarray, target_rms: float = 0.1, max_gain: float = 10.0
) -> np.ndarray:
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

    # Apply gain and clip
    y = np.clip(x * gain, -1.0, 1.0)

    return y.astype(np.float32)
