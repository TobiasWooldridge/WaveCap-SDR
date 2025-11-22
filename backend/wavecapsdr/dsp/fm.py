from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np

from .filters import highpass_filter, lowpass_filter, notch_filter, noise_blanker


# Pre-allocated zero for quadrature demod (avoids allocation per call)
_ZERO_F32 = np.zeros(1, dtype=np.float32)


def quadrature_demod(iq: np.ndarray, sample_rate: int) -> np.ndarray:
    """Quadrature FM demodulation.

    Converts complex IQ samples to instantaneous frequency deviations.
    Returns audio samples in approximate range [-1, 1] for typical FM signals.

    Performance: Uses in-place operations and pre-allocated arrays.

    Args:
        iq: Complex IQ samples (FM modulated signal centered at 0 Hz)
        sample_rate: Sample rate in Hz

    Returns:
        Demodulated FM audio (instantaneous frequency / max_deviation)
    """
    if iq.size == 0:
        return np.empty(0, dtype=np.float32)

    # y[n] = angle(x[n] * conj(x[n-1]))
    x = iq.astype(np.complex64, copy=False)

    # Compute product with conjugate of previous sample
    prod = x[1:] * np.conj(x[:-1])

    # Pre-allocate output array (avoids concatenation)
    out = np.empty(iq.size, dtype=np.float32)
    out[0] = 0.0

    # Extract phase and scale in one step
    scale = np.float32(sample_rate / (2.0 * np.pi * 75000.0))
    out[1:] = np.angle(prod) * scale

    return out


# Cache for deemphasis filter coefficients
@lru_cache(maxsize=32)
def _get_deemphasis_coeffs(sample_rate: int, tau_us: int) -> Tuple[np.ndarray, np.ndarray]:
    """Get cached deemphasis filter coefficients."""
    tau = tau_us * 1e-6  # Convert from microseconds
    alpha = 1.0 / (1.0 + (1.0 / (2.0 * np.pi * tau * sample_rate)))
    b = np.array([alpha], dtype=np.float32)
    a = np.array([1.0, -(1.0 - alpha)], dtype=np.float32)
    return b, a


def deemphasis_filter(x: np.ndarray, sample_rate: int, tau: float = 75e-6) -> np.ndarray:
    """Apply deemphasis filter to FM audio.

    Performance: Uses cached filter coefficients.
    """
    try:
        from scipy import signal

        # Convert tau to microseconds for cache key (integer)
        tau_us = int(tau * 1e6)
        b, a = _get_deemphasis_coeffs(sample_rate, tau_us)

        y = signal.lfilter(b, a, x).astype(np.float32)
        return y
    except ImportError:
        return x.astype(np.float32, copy=False)


# Cache for MPX lowpass filter coefficients
@lru_cache(maxsize=32)
def _get_lpf_coeffs(sample_rate: int, cutoff: int, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Get cached lowpass filter coefficients for MPX filtering."""
    from scipy import signal

    nyquist = sample_rate / 2.0
    normalized_cutoff = cutoff / nyquist

    if normalized_cutoff >= 1.0:
        return None, None

    b, a = signal.butter(order, normalized_cutoff, btype='low')
    return b, a


def lpf_audio(x: np.ndarray, sample_rate: int, cutoff: float = 15_000) -> np.ndarray:
    """Low-pass filter to remove FM stereo pilot tone and subcarriers.

    Removes 19 kHz pilot tone, 38 kHz stereo subcarrier, and 57 kHz RDS.
    Keeps only 0-15 kHz (mono audio bandwidth for broadcast FM).
    This eliminates the high-pitch whine commonly heard in FM demodulation.

    Performance: Uses cached filter coefficients.

    Args:
        x: Input audio signal
        sample_rate: Sample rate in Hz
        cutoff: Cutoff frequency in Hz (default 15 kHz)

    Returns:
        Filtered audio signal
    """
    if x.size == 0:
        return x.astype(np.float32, copy=False)

    try:
        from scipy import signal

        # Use integer cutoff for cache key
        cutoff_int = int(cutoff)
        b, a = _get_lpf_coeffs(sample_rate, cutoff_int, 5)

        if b is None:
            return x.astype(np.float32, copy=False)

        y = signal.lfilter(b, a, x).astype(np.float32)
        return y
    except ImportError:
        return x.astype(np.float32, copy=False)


def resample_poly(x: np.ndarray, in_rate: int, out_rate: int) -> np.ndarray:
    """Resample using polyphase filtering (3-5x faster than linear interpolation).

    Uses scipy.signal.resample_poly which performs polyphase resampling with
    an anti-aliasing FIR filter, providing both better quality and performance.

    Args:
        x: Input signal
        in_rate: Input sample rate in Hz
        out_rate: Output sample rate in Hz

    Returns:
        Resampled signal as float32
    """
    if x.size == 0 or in_rate == out_rate:
        return x.astype(np.float32, copy=False)

    try:
        from scipy.signal import resample_poly as scipy_resample_poly
        from math import gcd

        # Find GCD to minimize up/down factors
        g = gcd(int(in_rate), int(out_rate))
        up = out_rate // g
        down = in_rate // g

        # resample_poly handles anti-aliasing automatically
        y = scipy_resample_poly(x.astype(np.float64), up, down)
        return y.astype(np.float32)
    except ImportError:
        # Fallback to linear interpolation if scipy not available
        t_in = np.arange(x.shape[0], dtype=np.float64) / float(in_rate)
        duration = t_in[-1] if x.shape[0] > 0 else 0.0
        n_out = max(1, int(round(duration * out_rate)))
        t_out = np.arange(n_out, dtype=np.float64) / float(out_rate)
        y = np.interp(t_out, t_in, x.astype(np.float64))
        return y.astype(np.float32)


# Keep alias for backwards compatibility
resample_linear = resample_poly


def wbfm_demod(
    iq: np.ndarray,
    sample_rate: int,
    audio_rate: int = 48_000,
    enable_deemphasis: bool = True,
    deemphasis_tau: float = 75e-6,
    enable_mpx_filter: bool = True,
    mpx_cutoff_hz: float = 15_000,
    enable_highpass: bool = False,
    highpass_hz: float = 100,
    enable_noise_blanker: bool = False,
    noise_blanker_threshold_db: float = 10.0,
    notch_frequencies: list[float] | None = None,
) -> np.ndarray:
    """Demodulate wideband FM (broadcast radio).

    Args:
        iq: Complex IQ samples (FM modulated signal centered at 0 Hz)
        sample_rate: Sample rate of IQ data in Hz
        audio_rate: Desired audio output sample rate in Hz (default 48 kHz)
        enable_deemphasis: Enable deemphasis filter (default True)
        deemphasis_tau: Deemphasis time constant in seconds (default 75µs for US, 50µs for EU)
        enable_mpx_filter: Enable MPX filter to remove stereo pilot/subcarriers (default True)
        mpx_cutoff_hz: MPX filter cutoff frequency in Hz (default 15 kHz)
        enable_highpass: Enable highpass filter for DC removal (default False)
        highpass_hz: Highpass cutoff frequency in Hz (default 100 Hz)
        enable_noise_blanker: Enable noise blanker for impulse noise suppression (default False)
        noise_blanker_threshold_db: Noise blanker threshold in dB above median level (default 10 dB)
        notch_frequencies: List of frequencies to notch out (Hz) for interference rejection (default None)

    Returns:
        Demodulated audio samples (float32, mono, clipped to ±1.0)

    Pipeline:
        1. Quadrature demodulation (FM → audio)
        2. Optional noise blanker (impulse noise suppression)
        3. Optional deemphasis filter (75µs or 50µs)
        4. Optional MPX filter (removes 19 kHz pilot tone, 38 kHz stereo, 57 kHz RDS)
        5. Optional highpass filter (DC blocking)
        6. Optional notch filters (interference rejection)
        7. Normalize
        8. Resample to audio_rate
        9. Clip to ±1.0
    """
    fm = quadrature_demod(iq, sample_rate)

    # Apply noise blanker to suppress impulse noise (lightning, ignition, etc.)
    if enable_noise_blanker:
        fm = noise_blanker(fm, threshold_db=noise_blanker_threshold_db, blanking_width=3)

    # Apply deemphasis filter (compensates for FM pre-emphasis)
    if enable_deemphasis:
        fm = deemphasis_filter(fm, sample_rate, tau=deemphasis_tau)

    # MPX filter: Remove 19 kHz pilot tone and stereo subcarriers
    # This eliminates the high-pitch whine in FM broadcast audio
    if enable_mpx_filter:
        fm = lpf_audio(fm, sample_rate, cutoff=mpx_cutoff_hz)

    # Optional highpass filter for DC removal
    if enable_highpass and highpass_hz > 0:
        fm = highpass_filter(fm, sample_rate, highpass_hz)

    # Apply notch filters to remove interfering tones
    if notch_frequencies:
        for freq in notch_frequencies:
            if 0 < freq < sample_rate / 2:
                fm = notch_filter(fm, sample_rate, freq, q=30.0)

    # Normalize roughly
    if fm.size:
        fm = fm / max(1e-6, np.max(np.abs(fm)))

    audio = resample_linear(fm, sample_rate, audio_rate)

    # Hard clip to [-1,1]
    np.clip(audio, -1.0, 1.0, out=audio)
    return audio


def nbfm_demod(
    iq: np.ndarray,
    sample_rate: int,
    audio_rate: int = 48_000,
    enable_deemphasis: bool = False,
    deemphasis_tau: float = 75e-6,
    enable_highpass: bool = False,
    highpass_hz: float = 300,
    enable_lowpass: bool = False,
    lowpass_hz: float = 3_000,
    enable_noise_blanker: bool = False,
    noise_blanker_threshold_db: float = 10.0,
    notch_frequencies: list[float] | None = None,
) -> np.ndarray:
    """Demodulate narrowband FM (voice communications, public safety, amateur radio).

    Args:
        iq: Complex IQ samples (FM modulated signal centered at 0 Hz)
        sample_rate: Sample rate of IQ data in Hz
        audio_rate: Desired audio output sample rate in Hz (default 48 kHz)
        enable_deemphasis: Enable deemphasis filter (default False, NBFM often doesn't use it)
        deemphasis_tau: Deemphasis time constant in seconds (default 75µs)
        enable_highpass: Enable highpass filter for rumble/DC removal (default False)
        highpass_hz: Highpass cutoff frequency in Hz (default 300 Hz for voice)
        enable_lowpass: Enable lowpass filter for voice bandwidth limiting (default False)
        lowpass_hz: Lowpass cutoff frequency in Hz (default 3 kHz for voice)
        enable_noise_blanker: Enable noise blanker for impulse noise suppression (default False)
        noise_blanker_threshold_db: Noise blanker threshold in dB above median level (default 10 dB)
        notch_frequencies: List of frequencies to notch out (Hz) for interference rejection (default None)

    Returns:
        Demodulated audio samples (float32, mono, clipped to ±1.0)

    Pipeline:
        1. Quadrature demodulation (FM → audio)
        2. Optional noise blanker (impulse noise suppression)
        3. Optional deemphasis filter
        4. Optional highpass filter (removes rumble, DC offset)
        5. Optional lowpass filter (voice bandwidth limiting)
        6. Optional notch filters (interference rejection)
        7. Normalize
        8. Resample to audio_rate
        9. Clip to ±1.0

    Typical settings:
        - Voice comms: highpass 300 Hz, lowpass 3000 Hz, no deemphasis
        - Public safety: highpass 300 Hz, lowpass 2500 Hz, no deemphasis
    """
    fm = quadrature_demod(iq, sample_rate)

    # Apply noise blanker to suppress impulse noise (lightning, ignition, etc.)
    if enable_noise_blanker:
        fm = noise_blanker(fm, threshold_db=noise_blanker_threshold_db, blanking_width=3)

    # Optional deemphasis (NBFM typically doesn't use it)
    if enable_deemphasis:
        fm = deemphasis_filter(fm, sample_rate, tau=deemphasis_tau)

    # Optional highpass filter for rumble/DC removal
    if enable_highpass and highpass_hz > 0:
        fm = highpass_filter(fm, sample_rate, highpass_hz)

    # Optional lowpass filter for voice bandwidth limiting
    if enable_lowpass and lowpass_hz > 0:
        fm = lowpass_filter(fm, sample_rate, lowpass_hz)

    # Apply notch filters to remove interfering tones
    if notch_frequencies:
        for freq in notch_frequencies:
            if 0 < freq < sample_rate / 2:
                fm = notch_filter(fm, sample_rate, freq, q=30.0)

    # Normalize
    if fm.size:
        fm = fm / max(1e-6, np.max(np.abs(fm)))

    audio = resample_linear(fm, sample_rate, audio_rate)

    # Hard clip to [-1,1]
    np.clip(audio, -1.0, 1.0, out=audio)
    return audio

