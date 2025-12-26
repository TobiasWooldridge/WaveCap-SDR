"""C4FM (Continuous 4-level FM) demodulator for P25 Phase I.

C4FM is the modulation used by P25 Phase I systems. It transmits 4 frequency
deviation levels at 4800 baud, encoding 2 bits (dibits) per symbol:

    Dibit   Symbol   Phase (radians)   Deviation
    ------  ------   ---------------   ---------
    01      +3       +3π/4             +1800 Hz
    00      +1       +π/4              +600 Hz
    10      -1       -π/4              -600 Hz
    11      -3       -3π/4             -1800 Hz

The demodulation process (per SDRTrunk P25P1DecoderC4FM):
1. Low-pass baseband filter (5.2 kHz passband)
2. Root-Raised Cosine (RRC) pulse shaping filter (alpha=0.2)
3. PI/4 DQPSK differential demodulation (phase change over 1 symbol period)
4. Soft sync detection with timing optimization
5. Symbol slicing using π/2 quadrant boundaries

Reference: TIA-102.BAAA-A (P25 Common Air Interface)
Based on: SDRTrunk DifferentialDemodulatorFloatScalar.java, P25P1DemodulatorC4FM.java
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


def design_rrc_filter(
    samples_per_symbol: float,
    num_taps: int = 101,
    alpha: float = 0.2,
) -> np.ndarray:
    """Design a Root-Raised Cosine (RRC) matched filter.

    The RRC filter is the standard matched filter for P25 C4FM. Both transmitter
    and receiver use RRC filters, and together they form a Raised Cosine response
    with zero ISI at symbol centers.

    Args:
        samples_per_symbol: Number of samples per symbol
        num_taps: Filter length (odd number for symmetry)
        alpha: Roll-off factor (0.2 for P25 Phase I)

    Returns:
        RRC filter coefficients normalized to unit energy
    """
    # Ensure odd number of taps
    if num_taps % 2 == 0:
        num_taps += 1

    # Time vector centered at zero
    n = np.arange(num_taps) - (num_taps - 1) / 2
    t = n / samples_per_symbol

    # RRC impulse response
    h = np.zeros(num_taps, dtype=np.float64)

    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = (1 - alpha + 4 * alpha / np.pi)
        elif abs(ti) == 1 / (4 * alpha):
            h[i] = (alpha / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
            )
        else:
            num = np.sin(np.pi * ti * (1 - alpha)) + 4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha))
            den = np.pi * ti * (1 - (4 * alpha * ti) ** 2)
            h[i] = num / den

    # Normalize to DC gain of 1.0 to preserve symbol amplitude
    # (Unit energy normalization amplifies symbols by ~3-4x)
    h = h / np.sum(h)

    return h.astype(np.float32)


class C4FMDemodulator:
    """C4FM (4-level FSK) demodulator for P25 Phase I.

    This implements a C4FM receiver chain:
    1. FM discriminator (instantaneous frequency)
    2. RRC matched filter (alpha=0.2)
    3. Gardner timing recovery
    4. Symbol slicing using ±2 boundaries (maps to ±3, ±1 levels)

    C4FM symbol mapping (per TIA-102.BAAA-A):
        Symbol   Deviation    Dibit
        +3       +1800 Hz     01 (1)
        +1       +600 Hz      00 (0)
        -1       -600 Hz      10 (2)
        -3       -1800 Hz     11 (3)

    Example usage:
        demod = C4FMDemodulator(sample_rate=48000)
        dibits, soft = demod.demodulate(iq_samples)
    """

    # C4FM symbol levels (normalized to ±3, ±1)
    SYMBOL_LEVELS = np.array([3.0, 1.0, -1.0, -3.0], dtype=np.float32)
    # Symbol level to dibit mapping
    DIBIT_MAP = np.array([1, 0, 2, 3], dtype=np.uint8)  # +3→1, +1→0, -1→2, -3→3

    # Decision boundaries at ±2 (midpoints between ±3/±1)
    BOUNDARY_HIGH = 2.0   # Between +3 and +1
    BOUNDARY_LOW = -2.0   # Between -1 and -3

    def __init__(
        self,
        sample_rate: int = 48000,
        symbol_rate: int = 4800,
        rrc_alpha: float = 0.2,
        rrc_taps: int = 101,
        loop_bw: float = 0.01,
    ):
        """Initialize C4FM demodulator.

        Args:
            sample_rate: Input sample rate in Hz
            symbol_rate: Symbol rate (4800 baud for P25)
            rrc_alpha: RRC filter roll-off factor (0.2 for P25)
            rrc_taps: Number of RRC filter taps
            loop_bw: Timing recovery loop bandwidth (controls tracking speed)
        """
        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.samples_per_symbol = sample_rate / symbol_rate

        # Design RRC matched filter
        self._rrc = design_rrc_filter(self.samples_per_symbol, rrc_taps, rrc_alpha)
        self._rrc_delay = (len(self._rrc) - 1) // 2

        # FM discriminator state
        self._prev_iq = complex(1, 0)

        # Scaling: FM discriminator outputs instantaneous frequency
        # For C4FM: +1800 Hz → +3 symbol level, so scale = 3/1800
        self._freq_scale = 3.0 / 1800.0

        # Timing recovery state
        self._timing_phase = 0.0  # Fractional sample offset
        self._prev_symbol = 0.0

        # Gardner timing recovery loop coefficients
        damping = 1.0
        theta = loop_bw / (damping + 1.0 / (4.0 * damping))
        d = 1.0 + 2.0 * damping * theta + theta * theta
        self._kp = (4.0 * damping * theta) / d
        self._ki = (4.0 * theta * theta) / d
        self._loop_integrator = 0.0
        self._integrator_max = self.samples_per_symbol / 4.0

        logger.debug(
            f"C4FM demod initialized: {sample_rate} Hz, {symbol_rate} baud, "
            f"{self.samples_per_symbol:.2f} sps"
        )

    def reset(self) -> None:
        """Reset demodulator state for a new signal."""
        self._prev_iq = complex(1, 0)
        self._timing_phase = 0.0
        self._prev_symbol = 0.0
        self._loop_integrator = 0.0

    def _fm_discriminator(self, iq: np.ndarray) -> np.ndarray:
        """Quadrature FM discriminator.

        Computes instantaneous frequency from IQ samples using the
        derivative of phase (atan2 of conjugate product).

        Args:
            iq: Complex IQ samples

        Returns:
            Instantaneous frequency scaled to symbol levels (±3, ±1)
        """
        if len(iq) == 0:
            return np.array([], dtype=np.float32)

        # Prepend previous sample for continuity
        iq_ext = np.concatenate([[self._prev_iq], iq])
        self._prev_iq = iq[-1]

        # Conjugate product gives phase difference between adjacent samples
        prod = iq_ext[1:] * np.conj(iq_ext[:-1])

        # Extract phase (radians per sample)
        phase = np.angle(prod)

        # Convert to frequency in Hz, then scale to symbol levels
        # phase (rad/sample) * sample_rate / (2π) = freq (Hz)
        # freq (Hz) * scale = symbol level
        freq_hz = phase * self.sample_rate / (2 * np.pi)
        symbols = freq_hz * self._freq_scale

        return symbols.astype(np.float32)

    def _to_dibit(self, symbol: float) -> int:
        """Convert symbol level to dibit.

        Uses ±2 as decision boundaries:
        - symbol > +2: dibit 1 (+3)
        - 0 < symbol <= +2: dibit 0 (+1)
        - -2 <= symbol <= 0: dibit 2 (-1)
        - symbol < -2: dibit 3 (-3)

        Args:
            symbol: Symbol level (nominally ±3, ±1)

        Returns:
            Dibit value (0-3)
        """
        if symbol > 0:
            return 1 if symbol > self.BOUNDARY_HIGH else 0
        else:
            return 3 if symbol < self.BOUNDARY_LOW else 2

    def demodulate(self, iq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Demodulate C4FM signal to dibits.

        Processes IQ samples through:
        1. FM discriminator → symbol levels
        2. RRC matched filter
        3. Gardner timing recovery
        4. Symbol slicing

        Args:
            iq: Complex IQ samples (np.complex64 or np.complex128)

        Returns:
            Tuple of (dibits, soft_symbols):
            - dibits: Hard decision dibits (0-3), shape (N,)
            - soft_symbols: Soft symbol levels (±3, ±1), shape (N,)
        """
        if len(iq) == 0:
            return (
                np.array([], dtype=np.uint8),
                np.array([], dtype=np.float32),
            )

        # FM discriminator → symbol levels
        symbols = self._fm_discriminator(iq)

        # RRC matched filter
        filtered = signal.lfilter(self._rrc, 1.0, symbols).astype(np.float32)

        # Timing recovery and symbol extraction
        dibits = []
        soft_symbols = []
        sps = self.samples_per_symbol

        # Start after RRC filter delay, at center of first symbol
        # Symbol i has center at: rrc_delay + (i + 0.5) * sps
        buffer_pos = float(self._rrc_delay) + sps / 2.0

        while buffer_pos < len(filtered) - 2:
            # Sample position with fractional timing adjustment
            sample_pos = buffer_pos + self._timing_phase
            idx = int(sample_pos)
            mu = sample_pos - idx

            if idx + 1 >= len(filtered):
                break

            # Linear interpolation for current symbol
            soft_symbol = filtered[idx] + (filtered[idx + 1] - filtered[idx]) * mu

            # Gardner timing error detector (TED)
            # Needs mid-point sample between previous and current symbol
            mid_pos = buffer_pos - sps / 2.0 + self._timing_phase
            mid_idx = int(mid_pos)
            mid_mu = mid_pos - mid_idx

            if mid_idx >= 0 and mid_idx + 1 < len(filtered) and len(dibits) > 0:
                mid_sample = filtered[mid_idx] + (filtered[mid_idx + 1] - filtered[mid_idx]) * mid_mu
                error = mid_sample * (self._prev_symbol - soft_symbol)

                if np.isfinite(error):
                    # PI loop filter update
                    self._loop_integrator += self._ki * error
                    self._loop_integrator = np.clip(
                        self._loop_integrator, -self._integrator_max, self._integrator_max
                    )
                    # Update timing phase
                    phase_adj = self._kp * error + self._loop_integrator
                    self._timing_phase += np.clip(phase_adj, -0.5, 0.5)

                    # Keep timing phase in reasonable range
                    while self._timing_phase > sps / 2:
                        self._timing_phase -= sps
                    while self._timing_phase < -sps / 2:
                        self._timing_phase += sps

            # Symbol decision
            dibit = self._to_dibit(soft_symbol)
            dibits.append(dibit)
            soft_symbols.append(soft_symbol)

            self._prev_symbol = soft_symbol

            # Advance by exactly one symbol period
            buffer_pos += sps

        return (
            np.array(dibits, dtype=np.uint8),
            np.array(soft_symbols, dtype=np.float32),
        )

    def get_timing_offset(self) -> float:
        """Get current timing offset.

        Returns:
            Timing offset in samples
        """
        return self._loop_integrator


def c4fm_demod_simple(
    iq: np.ndarray,
    sample_rate: int = 48000,
    symbol_rate: int = 4800,
) -> np.ndarray:
    """Simplified C4FM demodulator (no state, single-shot).

    This is a stateless version for processing complete frames.
    For streaming operation, use C4FMDemodulator class.

    Args:
        iq: Complex IQ samples
        sample_rate: Sample rate in Hz
        symbol_rate: Symbol rate in baud

    Returns:
        Dibits (0-3)
    """
    demod = C4FMDemodulator(sample_rate=sample_rate, symbol_rate=symbol_rate)
    dibits, _ = demod.demodulate(iq)
    return dibits
