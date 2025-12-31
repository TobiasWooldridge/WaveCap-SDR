"""Symbol Timing Recovery for P25 demodulation.

This module provides timing error detectors (TEDs) and loop filters for
symbol synchronization in P25 demodulators:

- Gardner TED: Used for C4FM (Phase I), works with real-valued FM discriminator output
- Mueller-Muller TED: Used for CQPSK (Phase II), works with complex baseband

Both use a 2nd-order proportional-integral (PI) loop filter to track timing drift.

Reference: Rice, Michael. "Digital Communications: A Discrete-Time Approach"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from wavecapsdr.typing import NDArrayAny

logger = logging.getLogger(__name__)


@dataclass
class TimingLoopConfig:
    """Configuration for timing recovery loop filter."""

    loop_bw: float = 0.01  # Normalized loop bandwidth (0.01 = 1% of symbol rate)
    damping: float = 1.0  # Damping factor (1.0 = critically damped)
    max_deviation: float = 0.25  # Maximum timing deviation in symbols


def calculate_loop_coefficients(
    samples_per_symbol: float, loop_bw: float = 0.01, damping: float = 1.0
) -> tuple[float, float]:
    """Calculate PI loop filter coefficients.

    Uses standard 2nd-order loop design equations.

    Args:
        samples_per_symbol: Samples per symbol
        loop_bw: Normalized loop bandwidth
        damping: Damping factor (1.0 = critical damping)

    Returns:
        Tuple of (proportional gain, integral gain)
    """
    # Normalized bandwidth in radians
    theta = loop_bw / (damping + 1 / (4 * damping))

    # Loop filter gains
    d = 1 + 2 * damping * theta + theta**2
    kp = 4 * damping * theta / d
    ki = 4 * theta**2 / d

    return kp, ki


class GardnerTED:
    """Gardner Timing Error Detector for real-valued signals.

    The Gardner TED is a decision-directed timing error detector that uses
    the midpoint between two symbols to estimate timing error:

        e[n] = x[n-1/2] * (x[n-1] - x[n])

    It's self-normalizing and works well for 4-level FSK signals like C4FM.

    Features:
    - Non-data-aided (doesn't need symbol decisions)
    - Requires 2 samples per symbol minimum
    - Works with any pulse shape

    Example usage:
        ted = GardnerTED(samples_per_symbol=10)
        for sample in samples:
            symbol, ready = ted.process(sample)
            if ready:
                # Use symbol
    """

    def __init__(
        self,
        samples_per_symbol: float,
        loop_bw: float = 0.01,
        damping: float = 1.0,
    ):
        """Initialize Gardner TED.

        Args:
            samples_per_symbol: Number of samples per symbol
            loop_bw: Loop bandwidth (0.01 = 1% of symbol rate)
            damping: Loop damping factor
        """
        self.samples_per_symbol = samples_per_symbol

        # Loop filter coefficients
        self._kp, self._ki = calculate_loop_coefficients(samples_per_symbol, loop_bw, damping)

        # State
        self._phase = 0.0  # Fractional sample phase
        self._integrator = 0.0  # Loop filter integrator
        self._max_deviation = samples_per_symbol / 4

        # Sample buffer for interpolation
        self._buffer = np.zeros(4, dtype=np.float64)
        self._buf_idx = 0

        # Previous symbol for TED
        self._prev_symbol = 0.0
        self._prev_mid = 0.0

        # Sample counter
        self._sample_count = 0

        logger.debug(
            f"Gardner TED: sps={samples_per_symbol:.2f}, kp={self._kp:.4f}, ki={self._ki:.4f}"
        )

    def reset(self) -> None:
        """Reset TED state."""
        self._phase = 0.0
        self._integrator = 0.0
        self._buffer.fill(0)
        self._buf_idx = 0
        self._prev_symbol = 0.0
        self._prev_mid = 0.0
        self._sample_count = 0

    def _interpolate(self, mu: float) -> float:
        """Cubic interpolation at fractional sample position.

        Args:
            mu: Fractional position (0 to 1)

        Returns:
            Interpolated value
        """
        # Get samples in correct order from circular buffer
        idx = self._buf_idx
        s = [
            self._buffer[(idx - 3) % 4],
            self._buffer[(idx - 2) % 4],
            self._buffer[(idx - 1) % 4],
            self._buffer[idx % 4],
        ]

        # Cubic Lagrange interpolation
        c0 = s[1]
        c1 = (s[2] - s[0]) / 2
        c2 = s[0] - 5 * s[1] / 2 + 2 * s[2] - s[3] / 2
        c3 = (s[3] - s[0]) / 2 + 3 * (s[1] - s[2]) / 2

        return float(c0 + mu * (c1 + mu * (c2 + mu * c3)))

    def process_block(self, samples: NDArrayAny) -> tuple[NDArrayAny, NDArrayAny]:
        """Process a block of samples.

        Args:
            samples: Input samples

        Returns:
            Tuple of (symbols, timing_errors)
        """
        symbols = []
        errors = []

        for sample in samples:
            # Update circular buffer
            self._buf_idx = (self._buf_idx + 1) % 4
            self._buffer[self._buf_idx] = sample
            self._sample_count += 1

            # Check if we're at a symbol timing point
            self._phase += 1.0

            if self._phase >= self.samples_per_symbol:
                self._phase -= self.samples_per_symbol

                # Interpolate current symbol
                mu = self._phase / self.samples_per_symbol
                current = self._interpolate(mu)

                # Interpolate midpoint (half symbol back)
                mid_phase = self._phase + self.samples_per_symbol / 2
                if mid_phase >= 1.0:
                    mid = self._interpolate(mid_phase - int(mid_phase))
                else:
                    mid = self._prev_mid

                # Gardner timing error
                error = mid * (self._prev_symbol - current)

                # Update loop filter
                self._integrator += self._ki * error
                self._integrator = np.clip(
                    self._integrator, -self._max_deviation, self._max_deviation
                )

                # Adjust timing
                adjustment = self._kp * error + self._integrator
                self._phase += adjustment

                symbols.append(current)
                errors.append(error)

                self._prev_symbol = current
                self._prev_mid = mid

        return np.array(symbols, dtype=np.float64), np.array(errors, dtype=np.float64)


class MuellerMullerTED:
    """Mueller-Muller Timing Error Detector for complex signals.

    The M&M TED is designed for complex baseband signals and uses
    symbol decisions in the error computation:

        e[n] = Re{ d*[n-1] * x[n] - d*[n] * x[n-1] }

    where d[n] is the symbol decision and x[n] is the received sample.

    Best suited for:
    - QPSK and higher-order QAM
    - Phase II P25 (CQPSK/H-DQPSK)

    Example usage:
        ted = MuellerMullerTED(samples_per_symbol=4)
        symbols = ted.process_block(complex_samples)
    """

    # Standard QPSK constellation (rotated by pi/4 for H-DQPSK)
    QPSK_CONSTELLATION = np.array(
        [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex128
    ) / np.sqrt(2)

    def __init__(
        self,
        samples_per_symbol: float,
        loop_bw: float = 0.01,
        damping: float = 1.0,
    ):
        """Initialize Mueller-Muller TED.

        Args:
            samples_per_symbol: Number of samples per symbol
            loop_bw: Loop bandwidth
            damping: Loop damping factor
        """
        self.samples_per_symbol = samples_per_symbol

        # Loop filter coefficients
        self._kp, self._ki = calculate_loop_coefficients(samples_per_symbol, loop_bw, damping)

        # State
        self._phase = 0.0
        self._integrator = 0.0
        self._max_deviation = samples_per_symbol / 4

        # Sample buffer for interpolation (complex)
        self._buffer = np.zeros(4, dtype=np.complex128)
        self._buf_idx = 0

        # Previous values for M&M
        self._prev_symbol = complex(0, 0)
        self._prev_decision = complex(0, 0)

        logger.debug(f"M&M TED: sps={samples_per_symbol:.2f}, kp={self._kp:.4f}, ki={self._ki:.4f}")

    def reset(self) -> None:
        """Reset TED state."""
        self._phase = 0.0
        self._integrator = 0.0
        self._buffer.fill(0)
        self._buf_idx = 0
        self._prev_symbol = complex(0, 0)
        self._prev_decision = complex(0, 0)

    def _interpolate(self, mu: float) -> complex:
        """Cubic interpolation for complex samples.

        Args:
            mu: Fractional position (0 to 1)

        Returns:
            Interpolated complex value
        """
        idx = self._buf_idx
        s = [
            self._buffer[(idx - 3) % 4],
            self._buffer[(idx - 2) % 4],
            self._buffer[(idx - 1) % 4],
            self._buffer[idx % 4],
        ]

        # Interpolate real and imaginary separately
        def interp_1d(vals: list[float], mu: float) -> float:
            c0 = vals[1]
            c1 = (vals[2] - vals[0]) / 2
            c2 = vals[0] - 5 * vals[1] / 2 + 2 * vals[2] - vals[3] / 2
            c3 = (vals[3] - vals[0]) / 2 + 3 * (vals[1] - vals[2]) / 2
            return float(c0 + mu * (c1 + mu * (c2 + mu * c3)))

        real_part = interp_1d([x.real for x in s], mu)
        imag_part = interp_1d([x.imag for x in s], mu)

        return complex(real_part, imag_part)

    def _decide(self, symbol: complex) -> complex:
        """Make symbol decision (nearest constellation point).

        Args:
            symbol: Received symbol

        Returns:
            Nearest constellation point
        """
        distances = np.abs(self.QPSK_CONSTELLATION - symbol)
        return complex(self.QPSK_CONSTELLATION[np.argmin(distances)])

    def process_block(self, samples: NDArrayAny) -> tuple[NDArrayAny, NDArrayAny, NDArrayAny]:
        """Process a block of complex samples.

        Args:
            samples: Complex input samples

        Returns:
            Tuple of (symbols, decisions, timing_errors)
        """
        symbols = []
        decisions = []
        errors = []

        for sample in samples:
            # Update circular buffer
            self._buf_idx = (self._buf_idx + 1) % 4
            self._buffer[self._buf_idx] = sample

            # Check symbol timing
            self._phase += 1.0

            if self._phase >= self.samples_per_symbol:
                self._phase -= self.samples_per_symbol

                # Interpolate
                mu = self._phase / self.samples_per_symbol
                current = self._interpolate(mu)

                # Symbol decision
                decision = self._decide(current)

                # Mueller-Muller timing error
                # e = Re{ d*[n-1] * x[n] - d*[n] * x[n-1] }
                error = np.real(
                    np.conj(self._prev_decision) * current - np.conj(decision) * self._prev_symbol
                )

                # Update loop filter
                self._integrator += self._ki * error
                self._integrator = np.clip(
                    self._integrator, -self._max_deviation, self._max_deviation
                )

                # Adjust timing
                adjustment = self._kp * error + self._integrator
                self._phase += adjustment

                symbols.append(current)
                decisions.append(decision)
                errors.append(error)

                self._prev_symbol = current
                self._prev_decision = decision

        return (
            np.array(symbols, dtype=np.complex128),
            np.array(decisions, dtype=np.complex128),
            np.array(errors, dtype=np.float64),
        )
