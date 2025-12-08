"""C4FM (Continuous 4-level FM) demodulator for P25 Phase I.

C4FM is the modulation used by P25 Phase I systems. It transmits 4 frequency
deviation levels at 4800 baud, encoding 2 bits (dibits) per symbol:

    Dibit   Symbol   Deviation
    ------  ------   ---------
    01      +3       +1800 Hz
    00      +1       +600 Hz
    10      -1       -600 Hz
    11      -3       -1800 Hz

The demodulation process:
1. FM discriminator (quadrature demod) to recover instantaneous frequency
2. Root-Raised Cosine (RRC) matched filter (alpha=0.2)
3. Gardner timing recovery for symbol synchronization
4. Symbol slicing to dibits

Reference: TIA-102.BAAA-A (P25 Common Air Interface)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, cast

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

    # Normalize to unit energy
    h = h / np.sqrt(np.sum(h**2))

    return h.astype(np.float32)


class C4FMDemodulator:
    """C4FM demodulator with Gardner timing recovery for P25 Phase I.

    This implements a complete C4FM receiver chain:
    1. Quadrature FM discriminator
    2. RRC matched filter
    3. Gardner timing error detector (TED)
    4. Symbol decision and dibit mapping

    The demodulator maintains state between calls for streaming operation.

    Example usage:
        demod = C4FMDemodulator(sample_rate=48000)
        dibits, soft = demod.demodulate(iq_samples)
    """

    # C4FM symbol deviation levels (normalized to +/-1 after FM demod scaling)
    # Maps to dibits: +3 -> 01, +1 -> 00, -1 -> 10, -3 -> 11
    SYMBOL_LEVELS = np.array([3.0, 1.0, -1.0, -3.0], dtype=np.float32)
    DIBIT_MAP = np.array([1, 0, 2, 3], dtype=np.uint8)  # Symbol index to dibit

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

        # Gardner timing recovery state
        self._ted_phase = 0.0  # Fractional sample phase (0 to 1)
        self._loop_bw = loop_bw

        # Calculate loop filter coefficients (proportional-integral)
        # Using standard formulas for 2nd order loop
        damping = 1.0  # Critical damping
        theta = loop_bw / (damping + 1 / (4 * damping))
        d = 1 + 2 * damping * theta + theta**2
        self._kp = 4 * damping * theta / d  # Proportional gain
        self._ki = 4 * theta**2 / d  # Integral gain

        # Loop filter state
        self._loop_integrator = 0.0

        # Previous sample for interpolation
        self._prev_samples = np.zeros(4, dtype=np.float32)
        self._prev_symbol = 0.0

        # FM discriminator state
        self._prev_iq = complex(1, 0)

        # Deviation scaling: map FM output to symbol levels
        # P25 C4FM uses +/- 1800 Hz deviation for +/- 3 symbols
        self._deviation_scale = symbol_rate / (1800 * 2)

        logger.debug(
            f"C4FM demod initialized: {sample_rate} Hz, {symbol_rate} baud, "
            f"{self.samples_per_symbol:.2f} sps, loop_bw={loop_bw}"
        )

    def reset(self) -> None:
        """Reset demodulator state for a new signal."""
        self._ted_phase = 0.0
        self._loop_integrator = 0.0
        self._prev_samples = np.zeros(4, dtype=np.float32)
        self._prev_symbol = 0.0
        self._prev_iq = complex(1, 0)

    def _fm_discriminator(self, iq: np.ndarray) -> np.ndarray:
        """Quadrature FM discriminator.

        Computes instantaneous frequency from IQ samples using the
        derivative of phase (atan2 of conjugate product).

        Args:
            iq: Complex IQ samples

        Returns:
            Instantaneous frequency (scaled to symbol levels)
        """
        if len(iq) == 0:
            return np.array([], dtype=np.float32)

        # Prepend previous sample for continuity
        iq_ext = np.concatenate([[self._prev_iq], iq])
        self._prev_iq = iq[-1]

        # Conjugate product gives phase difference
        prod = iq_ext[1:] * np.conj(iq_ext[:-1])

        # Extract phase (instantaneous frequency)
        phase = np.angle(prod)

        # Scale to symbol levels
        freq = phase * self.sample_rate / (2 * np.pi) * self._deviation_scale

        return freq.astype(np.float32)

    def _interpolate(self, samples: np.ndarray, mu: float) -> float:
        """Cubic interpolation for fractional delay.

        Uses Farrow structure for efficient interpolation.

        Args:
            samples: 4 consecutive samples
            mu: Fractional delay (0 to 1)

        Returns:
            Interpolated sample value
        """
        # Cubic Lagrange interpolation
        s0, s1, s2, s3 = samples
        c0 = s1
        c1 = (s2 - s0) / 2
        c2 = s0 - 5 * s1 / 2 + 2 * s2 - s3 / 2
        c3 = (s3 - s0) / 2 + 3 * (s1 - s2) / 2

        return c0 + mu * (c1 + mu * (c2 + mu * c3))

    def demodulate(self, iq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Demodulate C4FM signal to dibits.

        Processes IQ samples through FM discriminator, matched filter,
        and timing recovery to produce symbol decisions.

        Args:
            iq: Complex IQ samples (np.complex64 or np.complex128)

        Returns:
            Tuple of (dibits, soft_symbols):
            - dibits: Hard decision dibits (0-3), shape (N,)
            - soft_symbols: Soft symbol values, shape (N,)
        """
        if len(iq) == 0:
            return (
                np.array([], dtype=np.uint8),
                np.array([], dtype=np.float32),
            )

        # FM discriminator
        freq = self._fm_discriminator(iq)

        # Matched filter (RRC)
        filtered = signal.lfilter(self._rrc, 1.0, freq).astype(np.float32)

        # Symbol timing recovery with Gardner TED
        dibits = []
        soft_symbols = []

        # Process samples
        i = 0
        while i < len(filtered) - 4:
            # Advance to next symbol timing position
            advance = self.samples_per_symbol + self._loop_integrator
            next_pos = int(self._ted_phase + advance)

            if next_pos >= len(filtered) - 3:
                break

            # Fractional interpolation position
            mu = self._ted_phase + advance - next_pos

            # Get interpolated samples for TED
            idx = next_pos
            if idx + 3 < len(filtered):
                samples = filtered[idx : idx + 4]

                # Interpolate current symbol
                current = self._interpolate(samples, mu)

                # Gardner TED (uses mid-point between symbols)
                # Error = mid * (prev - current)
                mid_idx = int(next_pos - self.samples_per_symbol / 2)
                if mid_idx >= 0 and mid_idx + 3 < len(filtered):
                    mid_samples = filtered[mid_idx : mid_idx + 4]
                    mid = self._interpolate(mid_samples, mu)
                    error = mid * (self._prev_symbol - current)

                    # Loop filter
                    self._loop_integrator += self._ki * error
                    self._loop_integrator = np.clip(
                        self._loop_integrator, -self.samples_per_symbol / 4, self.samples_per_symbol / 4
                    )

                # Symbol decision
                distances = np.abs(self.SYMBOL_LEVELS - current)
                symbol_idx = int(np.argmin(distances))
                dibit = self.DIBIT_MAP[symbol_idx]

                dibits.append(dibit)
                soft_symbols.append(current)

                self._prev_symbol = current

            # Update phase
            self._ted_phase = mu
            i = next_pos + 1

        return (
            np.array(dibits, dtype=np.uint8),
            np.array(soft_symbols, dtype=np.float32),
        )

    def get_timing_offset(self) -> float:
        """Get current timing offset estimate.

        Returns:
            Timing offset in samples (for debugging/visualization)
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
