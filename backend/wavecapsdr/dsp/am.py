"""AM and SSB demodulation for SDR signal processing.

This module implements:
- AM (Amplitude Modulation): Used for aviation, broadcast AM radio
- SSB (Single Sideband): Used for amateur radio, marine, military

AM demodulation uses envelope detection.
SSB demodulation uses frequency shifting and filtering.
"""

from __future__ import annotations

import numpy as np

from .agc import apply_agc
from .filters import bandpass_filter, highpass_filter, lowpass_filter, notch_filter, noise_blanker
from .fm import resample_linear


def freq_shift(iq: np.ndarray, offset_hz: float, sample_rate: int) -> np.ndarray:
    """Shift signal in frequency domain (mix with complex exponential).

    Args:
        iq: Complex IQ samples
        offset_hz: Frequency shift in Hz (positive = up, negative = down)
        sample_rate: Sample rate in Hz

    Returns:
        Frequency-shifted IQ samples
    """
    if iq.size == 0:
        return iq

    # Generate complex exponential at offset frequency
    t = np.arange(iq.shape[0], dtype=np.float64) / float(sample_rate)
    shift = np.exp(2j * np.pi * offset_hz * t).astype(np.complex64)

    # Multiply to shift frequency
    return (iq * shift).astype(np.complex64)


def am_demod(
    iq: np.ndarray,
    sample_rate: int,
    audio_rate: int = 48_000,
    enable_agc: bool = True,
    enable_highpass: bool = True,
    highpass_hz: float = 100,
    enable_lowpass: bool = True,
    lowpass_hz: float = 5000,
    enable_noise_blanker: bool = False,
    noise_blanker_threshold_db: float = 10.0,
    agc_target_db: float = -20.0,
    notch_frequencies: list[float] | None = None,
) -> np.ndarray:
    """Demodulate AM (Amplitude Modulation) signal.

    AM demodulation uses envelope detection: the signal amplitude
    carries the audio information.

    Args:
        iq: Complex IQ samples (AM modulated signal centered at 0 Hz)
        sample_rate: Sample rate of IQ data in Hz
        audio_rate: Desired audio output sample rate in Hz (default 48 kHz)
        enable_agc: Enable automatic gain control (default True)
        enable_highpass: Enable highpass filter for DC removal (default True)
        highpass_hz: Highpass cutoff frequency in Hz (default 100 Hz)
        enable_lowpass: Enable lowpass filter for bandwidth limiting (default True)
        lowpass_hz: Lowpass cutoff frequency in Hz (default 5 kHz for broadcast AM)
        enable_noise_blanker: Enable noise blanker for impulse noise suppression (default False)
        noise_blanker_threshold_db: Noise blanker threshold in dB above median level (default 10 dB)
        agc_target_db: AGC target level in dB (default -20 dB)
        notch_frequencies: List of frequencies to notch out (Hz) for interference rejection (default None)

    Returns:
        Demodulated audio samples (float32, mono, clipped to ±1.0)

    Pipeline:
        1. Envelope detection (magnitude of IQ)
        2. Optional noise blanker (impulse noise suppression)
        3. Remove DC offset (highpass filter)
        4. Bandwidth limiting (lowpass filter)
        5. Optional notch filters (interference rejection)
        6. Optional AGC
        7. Resample to audio_rate
        8. Clip to ±1.0

    Typical settings:
        - Aviation AM: lowpass_hz=5000, enable_agc=True
        - Broadcast AM: lowpass_hz=5000, enable_agc=True
    """
    if iq.size == 0:
        return np.empty(0, dtype=np.float32)

    # 1. Envelope detection: Take magnitude of complex IQ signal
    # This extracts the amplitude modulation
    audio = np.abs(iq).astype(np.float32)

    # 2. Apply noise blanker to suppress impulse noise (lightning, ignition, etc.)
    if enable_noise_blanker:
        audio = noise_blanker(audio, threshold_db=noise_blanker_threshold_db, blanking_width=3)

    # 3. Remove DC component (AM carrier offset)
    # The carrier creates a DC offset that must be removed
    if enable_highpass and highpass_hz > 0:
        audio = highpass_filter(audio, sample_rate, highpass_hz)

    # 3. Bandwidth limiting (remove high-frequency noise)
    # AM broadcast audio bandwidth is typically 5 kHz
    if enable_lowpass and lowpass_hz > 0:
        audio = lowpass_filter(audio, sample_rate, lowpass_hz)

    # 4. Apply notch filters to remove interfering tones
    if notch_frequencies:
        for freq in notch_frequencies:
            if 0 < freq < sample_rate / 2:
                audio = notch_filter(audio, sample_rate, freq, q=30.0)

    # 5. Automatic Gain Control
    # AM signals vary greatly in strength, AGC is very important
    if enable_agc:
        audio = apply_agc(
            audio,
            sample_rate,
            target_db=agc_target_db,
            attack_ms=5.0,
            release_ms=50.0,
        )

    # 6. Resample to audio output rate
    audio = resample_linear(audio, sample_rate, audio_rate)

    # 7. Hard clip to prevent overflow
    np.clip(audio, -1.0, 1.0, out=audio)

    return audio


def ssb_demod(
    iq: np.ndarray,
    sample_rate: int,
    audio_rate: int = 48_000,
    mode: str = "usb",
    enable_agc: bool = True,
    enable_bandpass: bool = True,
    bandpass_low: float = 300,
    bandpass_high: float = 3000,
    enable_noise_blanker: bool = False,
    noise_blanker_threshold_db: float = 10.0,
    agc_target_db: float = -20.0,
    notch_frequencies: list[float] | None = None,
) -> np.ndarray:
    """Demodulate SSB (Single Sideband) signal.

    SSB transmits only one sideband (USB or LSB) to save bandwidth.
    Demodulation requires frequency shifting and filtering.

    Args:
        iq: Complex IQ samples (SSB signal centered at 0 Hz)
        sample_rate: Sample rate of IQ data in Hz
        audio_rate: Desired audio output sample rate in Hz (default 48 kHz)
        mode: "usb" for Upper Sideband or "lsb" for Lower Sideband
        enable_agc: Enable automatic gain control (default True)
        enable_bandpass: Enable bandpass filter for voice bandwidth (default True)
        bandpass_low: Bandpass low cutoff in Hz (default 300 Hz)
        bandpass_high: Bandpass high cutoff in Hz (default 3000 Hz)
        enable_noise_blanker: Enable noise blanker for impulse noise suppression (default False)
        noise_blanker_threshold_db: Noise blanker threshold in dB above median level (default 10 dB)
        agc_target_db: AGC target level in dB (default -20 dB)
        notch_frequencies: List of frequencies to notch out (Hz) for interference rejection (default None)

    Returns:
        Demodulated audio samples (float32, mono, clipped to ±1.0)

    Pipeline:
        1. Frequency shift (USB: +1500 Hz, LSB: -1500 Hz)
        2. Take real component (demodulate)
        3. Optional noise blanker (impulse noise suppression)
        4. Bandpass filter (voice bandwidth 300-3000 Hz)
        5. Optional notch filters (interference rejection)
        6. Optional AGC (very important for SSB!)
        7. Resample to audio_rate
        8. Clip to ±1.0

    Typical settings:
        - Amateur radio SSB: bandpass 300-3000 Hz, enable_agc=True
        - Marine SSB: bandpass 300-2500 Hz, enable_agc=True
    """
    if iq.size == 0:
        return np.empty(0, dtype=np.float32)

    # 1. Frequency shift to center the audio in baseband
    # USB: shift up by +1.5 kHz
    # LSB: shift down by -1.5 kHz
    shift_hz = 1500.0 if mode.lower() == "usb" else -1500.0
    iq_shifted = freq_shift(iq, shift_hz, sample_rate)

    # 2. Take real component to demodulate
    # This is the key to SSB demodulation - we just need the real part
    audio = np.real(iq_shifted).astype(np.float32)

    # 3. Apply noise blanker to suppress impulse noise (lightning, QRN, etc.)
    if enable_noise_blanker:
        audio = noise_blanker(audio, threshold_db=noise_blanker_threshold_db, blanking_width=3)

    # 4. Bandpass filter for voice bandwidth
    # SSB voice communications use 300-3000 Hz typically
    if enable_bandpass:
        audio = bandpass_filter(audio, sample_rate, bandpass_low, bandpass_high)

    # 4. Apply notch filters to remove interfering tones
    if notch_frequencies:
        for freq in notch_frequencies:
            if 0 < freq < sample_rate / 2:
                audio = notch_filter(audio, sample_rate, freq, q=30.0)

    # 5. Automatic Gain Control
    # SSB signals vary greatly in strength, AGC is CRITICAL
    if enable_agc:
        audio = apply_agc(
            audio,
            sample_rate,
            target_db=agc_target_db,
            attack_ms=5.0,
            release_ms=50.0,
        )

    # 6. Resample to audio output rate
    audio = resample_linear(audio, sample_rate, audio_rate)

    # 7. Hard clip to prevent overflow
    np.clip(audio, -1.0, 1.0, out=audio)

    return audio


def resample_linear(x: np.ndarray, in_rate: int, out_rate: int) -> np.ndarray:
    """Resample audio using linear interpolation.

    This is a simple, fast resampler suitable for audio.
    For higher quality, consider using scipy.signal.resample_poly.

    Args:
        x: Input signal
        in_rate: Input sample rate in Hz
        out_rate: Output sample rate in Hz

    Returns:
        Resampled signal
    """
    if x.size == 0 or in_rate == out_rate:
        return x.astype(np.float32, copy=False)

    t_in = np.arange(x.shape[0], dtype=np.float64) / float(in_rate)
    duration = t_in[-1] if x.shape[0] > 0 else 0.0
    n_out = max(1, int(round(duration * out_rate)))
    t_out = np.arange(n_out, dtype=np.float64) / float(out_rate)
    y = np.interp(t_out, t_in, x.astype(np.float64))

    return y.astype(np.float32)
