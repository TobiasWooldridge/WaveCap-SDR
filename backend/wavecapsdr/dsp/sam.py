"""Synchronous AM (SAM) demodulation with carrier recovery PLL.

SAM provides superior performance over envelope detection AM by:
1. Tracking and locking onto the carrier with a Phase-Locked Loop (PLL)
2. Performing coherent (synchronous) detection
3. Providing AFC (Automatic Frequency Correction) for drifting transmitters
4. Supporting sideband selection (SAM-U, SAM-L, SAM-D)

Essential for HF/shortwave AM reception with fading signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np

from .agc import apply_agc, soft_clip
from .filters import highpass_filter, lowpass_filter, noise_blanker, notch_filter
from .fm import resample_poly


@dataclass
class CarrierRecoveryPLL:
    """Second-order Phase-Locked Loop for AM carrier recovery.

    Uses a type-2 PLL with proportional-integral loop filter for
    robust carrier tracking with zero steady-state phase error.

    Attributes:
        sample_rate: Sample rate in Hz
        loop_bandwidth: PLL loop bandwidth in Hz (controls tracking speed)
        damping: Damping factor (0.707 = critically damped)
    """
    sample_rate: float
    loop_bandwidth: float = 50.0  # Hz - wider = faster tracking, more noise
    damping: float = 0.707  # Critically damped

    # PLL state
    _phase: float = field(default=0.0, init=False)
    _frequency: float = field(default=0.0, init=False)
    _integrator: float = field(default=0.0, init=False)

    # Computed coefficients
    _alpha: float = field(default=0.0, init=False)
    _beta: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        """Compute PLL loop filter coefficients."""
        self._compute_coefficients()

    def _compute_coefficients(self) -> None:
        """Compute loop filter coefficients from bandwidth and damping.

        Uses standard 2nd-order PLL design equations:
        omega_n = 2 * pi * loop_bandwidth
        alpha (proportional) = 2 * damping * omega_n / sample_rate
        beta (integral) = omega_n^2 / sample_rate^2
        """
        omega_n = 2 * np.pi * self.loop_bandwidth
        self._alpha = 2 * self.damping * omega_n / self.sample_rate
        self._beta = (omega_n ** 2) / (self.sample_rate ** 2)

    def set_bandwidth(self, bandwidth_hz: float) -> None:
        """Dynamically adjust loop bandwidth."""
        self.loop_bandwidth = bandwidth_hz
        self._compute_coefficients()

    def process(self, iq: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Process IQ samples and recover carrier phase.

        Args:
            iq: Complex IQ samples

        Returns:
            Tuple of:
            - coherent_i: In-phase (carrier-locked) component
            - coherent_q: Quadrature component
            - freq_offset: Estimated carrier frequency offset in Hz
        """
        if iq.size == 0:
            return (
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
                0.0
            )

        n_samples = len(iq)
        coherent_i = np.zeros(n_samples, dtype=np.float32)
        coherent_q = np.zeros(n_samples, dtype=np.float32)

        # Process sample-by-sample for PLL tracking
        for i in range(n_samples):
            # Generate local oscillator at current phase estimate
            lo = np.exp(-1j * self._phase)

            # Mix input with LO to get coherent output
            mixed = iq[i] * lo
            coherent_i[i] = float(np.real(mixed))
            coherent_q[i] = float(np.imag(mixed))

            # Phase error detector: atan2(Q, I) when locked
            # For carrier recovery, use: arg(input * conj(lo)) = imag(input * conj(lo)) for small angles
            # More robust: full atan2 for large errors
            phase_error = float(np.arctan2(np.imag(mixed), np.abs(np.real(mixed)) + 1e-10))

            # Loop filter (proportional + integral)
            self._integrator += self._beta * phase_error
            freq_correction = self._alpha * phase_error + self._integrator

            # Update phase and frequency estimates
            self._frequency = freq_correction
            self._phase += freq_correction

            # Wrap phase to [-pi, pi]
            if self._phase > np.pi:
                self._phase -= 2 * np.pi
            elif self._phase < -np.pi:
                self._phase += 2 * np.pi

        # Compute frequency offset in Hz
        freq_offset_hz = self._frequency * self.sample_rate / (2 * np.pi)

        return coherent_i, coherent_q, freq_offset_hz

    def reset(self) -> None:
        """Reset PLL state."""
        self._phase = 0.0
        self._frequency = 0.0
        self._integrator = 0.0


def sam_demod(
    iq: np.ndarray,
    sample_rate: int,
    audio_rate: int = 48_000,
    sideband: str = "dsb",
    pll_bandwidth: float = 50.0,
    pll_damping: float = 0.707,
    enable_agc: bool = True,
    enable_highpass: bool = True,
    highpass_hz: float = 100.0,
    enable_lowpass: bool = True,
    lowpass_hz: float = 5000.0,
    enable_noise_blanker: bool = False,
    noise_blanker_threshold_db: float = 10.0,
    agc_target_db: float = -20.0,
    notch_frequencies: list[float] | None = None,
    pll_state: CarrierRecoveryPLL | None = None,
) -> tuple[np.ndarray, float, CarrierRecoveryPLL | None]:
    """Demodulate AM using Synchronous AM (SAM) with carrier recovery PLL.

    SAM provides superior performance over envelope detection by:
    - Tracking carrier phase with a PLL for coherent detection
    - Providing AFC for drifting transmitters
    - Supporting sideband selection (USB, LSB, or DSB)

    Args:
        iq: Complex IQ samples (AM signal centered at 0 Hz)
        sample_rate: Sample rate of IQ data in Hz
        audio_rate: Desired audio output sample rate (default 48 kHz)
        sideband: Sideband selection:
            - "dsb": Double sideband (both sidebands, like normal AM)
            - "usb": Upper sideband only (SAM-U)
            - "lsb": Lower sideband only (SAM-L)
        pll_bandwidth: PLL loop bandwidth in Hz (default 50 Hz)
            - Wider = faster tracking, more noise
            - Narrower = slower tracking, cleaner audio
            - Typical: 30-100 Hz
        pll_damping: PLL damping factor (default 0.707 = critically damped)
        enable_agc: Enable automatic gain control (default True)
        enable_highpass: Enable highpass filter for DC removal (default True)
        highpass_hz: Highpass cutoff in Hz (default 100 Hz)
        enable_lowpass: Enable lowpass filter (default True)
        lowpass_hz: Lowpass cutoff in Hz (default 5000 Hz)
        enable_noise_blanker: Enable noise blanker (default False)
        noise_blanker_threshold_db: Noise blanker threshold in dB (default 10 dB)
        agc_target_db: AGC target level in dB (default -20 dB)
        notch_frequencies: List of frequencies to notch out (default None)
        pll_state: Optional PLL state for continuous processing

    Returns:
        Tuple of:
        - audio: Demodulated audio samples (float32, mono)
        - freq_offset: Estimated carrier frequency offset in Hz (for AFC display)
        - pll_state: PLL state for continuous processing

    Pipeline:
        1. PLL carrier recovery (phase tracking)
        2. Coherent detection (multiply by recovered carrier)
        3. Sideband selection (DSB, USB, or LSB)
        4. Optional noise blanker
        5. DC removal (highpass filter)
        6. Bandwidth limiting (lowpass filter)
        7. Optional notch filters
        8. Optional AGC
        9. Resample to audio_rate
        10. Clip to +/-1.0

    Typical settings:
        - Shortwave broadcast: sideband="dsb", pll_bandwidth=50, lowpass_hz=5000
        - Amateur SSB: sideband="usb"/"lsb", pll_bandwidth=30, lowpass_hz=3000
        - Fading signals: pll_bandwidth=100 (faster tracking)
    """
    if iq.size == 0:
        return (
            cast(np.ndarray, np.empty(0, dtype=np.float32)),
            0.0,
            pll_state
        )

    # 1. Initialize or reuse PLL state
    if pll_state is None:
        pll_state = CarrierRecoveryPLL(
            sample_rate=float(sample_rate),
            loop_bandwidth=pll_bandwidth,
            damping=pll_damping
        )
    else:
        # Update bandwidth if changed
        if pll_state.loop_bandwidth != pll_bandwidth:
            pll_state.set_bandwidth(pll_bandwidth)

    # 2. PLL carrier recovery and coherent detection
    coherent_i, coherent_q, freq_offset = pll_state.process(iq)

    # 3. Sideband selection
    sideband = sideband.lower()
    if sideband == "usb":
        # Upper sideband: I + Q (after Hilbert transform approximation)
        # For proper USB, we'd use a Hilbert transform, but I+Q works reasonably
        audio = coherent_i + coherent_q
    elif sideband == "lsb":
        # Lower sideband: I - Q
        audio = coherent_i - coherent_q
    else:  # dsb (double sideband, like normal AM)
        # DSB uses just the in-phase component
        audio = coherent_i

    # 4. Apply noise blanker to suppress impulse noise
    if enable_noise_blanker:
        audio = noise_blanker(audio, threshold_db=noise_blanker_threshold_db, blanking_width=3)

    # 5. Remove DC component (carrier residual)
    if enable_highpass and highpass_hz > 0:
        audio = highpass_filter(audio, sample_rate, highpass_hz)

    # 6. Bandwidth limiting
    if enable_lowpass and lowpass_hz > 0:
        audio = lowpass_filter(audio, sample_rate, lowpass_hz)

    # 7. Apply notch filters for interference rejection
    if notch_frequencies:
        for freq in notch_frequencies:
            if 0 < freq < sample_rate / 2:
                audio = notch_filter(audio, sample_rate, freq, q=30.0)

    # 8. Automatic Gain Control
    if enable_agc:
        audio = apply_agc(
            audio,
            sample_rate,
            target_db=agc_target_db,
            attack_ms=5.0,
            release_ms=50.0,
        )

    # 9. Resample to audio output rate
    audio = resample_poly(audio, sample_rate, audio_rate)

    # 10. Soft clip to prevent overflow
    if not enable_agc:
        audio = soft_clip(audio)

    return audio, freq_offset, pll_state


def sam_demod_simple(
    iq: np.ndarray,
    sample_rate: int,
    audio_rate: int = 48_000,
    sideband: str = "dsb",
    pll_bandwidth: float = 50.0,
    enable_agc: bool = True,
    enable_highpass: bool = True,
    highpass_hz: float = 100.0,
    enable_lowpass: bool = True,
    lowpass_hz: float = 5000.0,
    agc_target_db: float = -20.0,
) -> np.ndarray:
    """Simplified SAM demodulator returning only audio (stateless).

    Wrapper around sam_demod for simple use cases where PLL state
    and frequency offset are not needed.

    Args:
        iq: Complex IQ samples
        sample_rate: Sample rate in Hz
        audio_rate: Audio output rate (default 48 kHz)
        sideband: "dsb", "usb", or "lsb" (default "dsb")
        pll_bandwidth: PLL bandwidth in Hz (default 50)
        enable_agc: Enable AGC (default True)
        enable_highpass: Enable highpass filter (default True)
        highpass_hz: Highpass cutoff (default 100 Hz)
        enable_lowpass: Enable lowpass filter (default True)
        lowpass_hz: Lowpass cutoff (default 5000 Hz)
        agc_target_db: AGC target level (default -20 dB)

    Returns:
        Demodulated audio samples (float32)
    """
    audio, _, _ = sam_demod(
        iq=iq,
        sample_rate=sample_rate,
        audio_rate=audio_rate,
        sideband=sideband,
        pll_bandwidth=pll_bandwidth,
        enable_agc=enable_agc,
        enable_highpass=enable_highpass,
        highpass_hz=highpass_hz,
        enable_lowpass=enable_lowpass,
        lowpass_hz=lowpass_hz,
        agc_target_db=agc_target_db,
    )
    return audio
