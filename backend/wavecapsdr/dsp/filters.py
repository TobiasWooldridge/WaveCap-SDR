"""Common audio filter library for SDR signal processing.

This module provides standard audio filters used across different demodulation modes:
- Highpass filters: DC blocking, rumble removal
- Lowpass filters: Bandwidth limiting, anti-aliasing
- Bandpass filters: Voice/SSB frequency selection
- Notch filters: Tone removal

All filters use scipy.signal.butter (Butterworth) for maximally flat passband.
"""

from __future__ import annotations

import numpy as np


def highpass_filter(
    x: np.ndarray, sample_rate: int, cutoff: float, order: int = 5
) -> np.ndarray:
    """Apply highpass filter to remove DC offset and low-frequency noise.

    Used for:
    - DC blocking in all analog modes
    - Rumble/wind noise removal
    - AM carrier offset removal

    Args:
        x: Input signal
        sample_rate: Sample rate in Hz
        cutoff: Cutoff frequency in Hz (frequencies below are attenuated)
        order: Filter order (higher = sharper cutoff, default 5)

    Returns:
        Filtered signal

    Example cutoffs:
        - 100 Hz: AM DC removal
        - 300 Hz: Voice communications (removes rumble)
    """
    if x.size == 0:
        return x.astype(np.float32, copy=False)

    try:
        from scipy import signal

        nyquist = sample_rate / 2.0
        normalized_cutoff = cutoff / nyquist

        # Validate cutoff frequency
        if normalized_cutoff <= 0 or normalized_cutoff >= 1.0:
            return x.astype(np.float32, copy=False)

        b, a = signal.butter(order, normalized_cutoff, btype="high")
        y = signal.lfilter(b, a, x).astype(np.float32)
        return y
    except ImportError:
        # Fallback: no filtering if scipy not available
        return x.astype(np.float32, copy=False)


def lowpass_filter(
    x: np.ndarray, sample_rate: int, cutoff: float, order: int = 5
) -> np.ndarray:
    """Apply lowpass filter to limit bandwidth and reduce high-frequency noise.

    Used for:
    - Voice bandwidth limiting
    - Anti-aliasing before resampling
    - Noise reduction

    Args:
        x: Input signal
        sample_rate: Sample rate in Hz
        cutoff: Cutoff frequency in Hz (frequencies above are attenuated)
        order: Filter order (higher = sharper cutoff, default 5)

    Returns:
        Filtered signal

    Example cutoffs:
        - 3000 Hz: NBFM voice bandwidth
        - 5000 Hz: AM broadcast audio bandwidth
        - 15000 Hz: FM broadcast MPX filter (remove pilot tone)
    """
    if x.size == 0:
        return x.astype(np.float32, copy=False)

    try:
        from scipy import signal

        nyquist = sample_rate / 2.0
        normalized_cutoff = cutoff / nyquist

        # Validate cutoff frequency
        if normalized_cutoff <= 0 or normalized_cutoff >= 1.0:
            return x.astype(np.float32, copy=False)

        b, a = signal.butter(order, normalized_cutoff, btype="low")
        y = signal.lfilter(b, a, x).astype(np.float32)
        return y
    except ImportError:
        # Fallback: no filtering if scipy not available
        return x.astype(np.float32, copy=False)


def bandpass_filter(
    x: np.ndarray, sample_rate: int, low: float, high: float, order: int = 5
) -> np.ndarray:
    """Apply bandpass filter to select a specific frequency range.

    Used for:
    - SSB voice bandwidth (300-3000 Hz)
    - Narrow signal extraction
    - Interference rejection

    Args:
        x: Input signal
        sample_rate: Sample rate in Hz
        low: Low cutoff frequency in Hz
        high: High cutoff frequency in Hz
        order: Filter order (higher = sharper cutoff, default 5)

    Returns:
        Filtered signal

    Example ranges:
        - 300-3000 Hz: SSB voice communications
        - 500-2500 Hz: Narrowband voice (reduced bandwidth)
    """
    if x.size == 0:
        return x.astype(np.float32, copy=False)

    try:
        from scipy import signal

        nyquist = sample_rate / 2.0
        normalized_low = low / nyquist
        normalized_high = high / nyquist

        # Validate cutoff frequencies
        if (
            normalized_low <= 0
            or normalized_high >= 1.0
            or normalized_low >= normalized_high
        ):
            return x.astype(np.float32, copy=False)

        b, a = signal.butter(order, [normalized_low, normalized_high], btype="band")
        y = signal.lfilter(b, a, x).astype(np.float32)
        return y
    except ImportError:
        # Fallback: no filtering if scipy not available
        return x.astype(np.float32, copy=False)


def notch_filter(
    x: np.ndarray, sample_rate: int, freq: float, q: float = 30.0
) -> np.ndarray:
    """Apply notch filter to remove a specific frequency tone.

    Used for:
    - Removing carrier residual
    - Removing interfering tones (power line hum, etc.)
    - Cleaning up signals with known interference

    Args:
        x: Input signal
        sample_rate: Sample rate in Hz
        freq: Frequency to notch out in Hz
        q: Quality factor (higher = narrower notch, default 30)

    Returns:
        Filtered signal

    Example frequencies:
        - 60 Hz / 50 Hz: Power line hum
        - 1000 Hz: Test tone removal
    """
    if x.size == 0:
        return x.astype(np.float32, copy=False)

    try:
        from scipy import signal

        nyquist = sample_rate / 2.0
        normalized_freq = freq / nyquist

        # Validate frequency
        if normalized_freq <= 0 or normalized_freq >= 1.0:
            return x.astype(np.float32, copy=False)

        # Design IIR notch filter
        b, a = signal.iirnotch(normalized_freq, q)
        y = signal.lfilter(b, a, x).astype(np.float32)
        return y
    except ImportError:
        # Fallback: no filtering if scipy not available
        return x.astype(np.float32, copy=False)


def noise_blanker(
    x: np.ndarray, threshold_db: float = 10.0, blanking_width: int = 3
) -> np.ndarray:
    """Apply noise blanker to suppress impulse noise (lightning, ignition, etc.).

    The noise blanker detects short-duration high-amplitude spikes in the signal
    and replaces them with zeros or interpolated values. This is effective against:
    - Lightning static
    - Automotive ignition noise
    - Power line arcing
    - Other impulse/burst interference

    Algorithm:
    1. Compute median absolute amplitude as baseline
    2. Detect samples exceeding threshold_db above baseline
    3. Blank detected impulses and surrounding samples

    Args:
        x: Input signal (real or complex)
        threshold_db: Detection threshold in dB above median level (default 10 dB)
        blanking_width: Number of samples to blank on each side of impulse (default 3)

    Returns:
        Signal with impulses blanked (replaced with zeros)

    Example usage:
        # Aggressive blanking for heavy QRN (static)
        clean = noise_blanker(signal, threshold_db=8, blanking_width=5)

        # Light blanking for occasional pops
        clean = noise_blanker(signal, threshold_db=12, blanking_width=2)
    """
    if x.size == 0:
        return x.astype(np.float32, copy=False)

    # Compute signal magnitude (handles both real and complex)
    if np.iscomplexobj(x):
        mag = np.abs(x)
    else:
        mag = np.abs(x)

    # Calculate baseline level using median (robust to impulses)
    median_level = np.median(mag)

    # Avoid division by zero
    if median_level < 1e-10:
        return x.astype(np.float32, copy=False)

    # Convert threshold from dB to linear scale
    threshold_linear = median_level * (10 ** (threshold_db / 20.0))

    # Detect impulses exceeding threshold
    impulse_mask = mag > threshold_linear

    # Expand blanking region around each impulse
    if blanking_width > 0 and np.any(impulse_mask):
        # Create expanded mask by dilating the impulse mask
        expanded_mask = np.copy(impulse_mask)
        impulse_indices = np.where(impulse_mask)[0]

        for idx in impulse_indices:
            start = max(0, idx - blanking_width)
            end = min(len(expanded_mask), idx + blanking_width + 1)
            expanded_mask[start:end] = True

        # Blank the detected impulses (set to zero)
        y = np.copy(x)
        y[expanded_mask] = 0

        return y.astype(np.float32 if not np.iscomplexobj(x) else np.complex64)
    else:
        # No impulses detected or no expansion needed
        y = np.copy(x)
        y[impulse_mask] = 0
        return y.astype(np.float32 if not np.iscomplexobj(x) else np.complex64)
