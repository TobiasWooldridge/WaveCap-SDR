"""CQPSK (H-DQPSK) demodulator for P25 Phase II.

P25 Phase II uses H-CPM (Harmonized Continuous Phase Modulation) in the uplink
and H-DQPSK (Harmonized Differential QPSK) in the downlink. Both are variants
of pi/4-DQPSK at 12000 baud with 2 TDMA slots.

Key differences from Phase I C4FM:
- Higher symbol rate: 12000 baud vs 4800 baud
- TDMA: 2 slots per channel (each 30ms)
- Coherent demodulation with Costas loop carrier recovery
- pi/4-DQPSK differential encoding

The demodulation process:
1. RRC matched filter (alpha=1.0 for Phase II)
2. Costas loop carrier recovery
3. Mueller-Muller timing recovery
4. Differential decode to dibits

Reference: TIA-102.BBAC (P25 Phase II)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from scipy import signal

from wavecapsdr.dsp.p25.symbol_timing import MuellerMullerTED

logger = logging.getLogger(__name__)


def design_rrc_filter_phase2(
    samples_per_symbol: float,
    num_taps: int = 65,
    alpha: float = 1.0,
) -> np.ndarray:
    """Design RRC filter for Phase II CQPSK.

    Phase II uses a wider filter (alpha=1.0) compared to Phase I (alpha=0.2)
    due to the higher symbol rate and different pulse shaping requirements.

    Args:
        samples_per_symbol: Samples per symbol
        num_taps: Filter length
        alpha: Roll-off factor (1.0 for Phase II)

    Returns:
        Filter coefficients
    """
    if num_taps % 2 == 0:
        num_taps += 1

    n = np.arange(num_taps) - (num_taps - 1) / 2
    t = n / samples_per_symbol

    h = np.zeros(num_taps, dtype=np.float64)

    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = 1 - alpha + 4 * alpha / np.pi
        elif abs(ti * 4 * alpha) == 1:
            h[i] = (alpha / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
            )
        else:
            num = np.sin(np.pi * ti * (1 - alpha)) + 4 * alpha * ti * np.cos(
                np.pi * ti * (1 + alpha)
            )
            den = np.pi * ti * (1 - (4 * alpha * ti) ** 2)
            if abs(den) > 1e-10:
                h[i] = num / den
            else:
                h[i] = 0

    # Normalize
    h = h / np.sqrt(np.sum(h**2))
    return h.astype(np.float32)


class CostasLoop:
    """Costas Loop for carrier recovery in QPSK signals.

    The Costas loop is a PLL variant that can lock onto suppressed-carrier
    signals like QPSK. It uses a 4th-order phase detector for QPSK.

    The loop tracks carrier frequency offset and phase, allowing coherent
    demodulation of the received signal.
    """

    def __init__(
        self,
        loop_bw: float = 0.01,
        damping: float = 0.707,
        max_freq: float = 0.1,
    ):
        """Initialize Costas loop.

        Args:
            loop_bw: Normalized loop bandwidth
            damping: Damping factor (0.707 = optimal for tracking)
            max_freq: Maximum normalized frequency offset
        """
        # Loop gains (2nd-order loop)
        theta = loop_bw / (damping + 1 / (4 * damping))
        d = 1 + 2 * damping * theta + theta**2
        self._kp = 4 * damping * theta / d
        self._ki = 4 * theta**2 / d

        self._max_freq = max_freq

        # State
        self._phase = 0.0
        self._freq = 0.0

        logger.debug(f"Costas loop: bw={loop_bw}, kp={self._kp:.4f}, ki={self._ki:.4f}")

    def reset(self) -> None:
        """Reset loop state."""
        self._phase = 0.0
        self._freq = 0.0

    def process(self, sample: complex) -> complex:
        """Process one sample through the Costas loop.

        Args:
            sample: Input complex sample

        Returns:
            Phase-corrected sample
        """
        # Apply current phase correction
        corrected = sample * np.exp(-1j * self._phase)

        # QPSK phase detector (4th power removes modulation)
        # For pi/4-DQPSK, use decision-directed
        error = self._phase_detector_qpsk(corrected)

        # Loop filter (PI controller)
        self._freq += self._ki * error
        self._freq = np.clip(self._freq, -self._max_freq, self._max_freq)

        phase_adj = self._kp * error + self._freq
        self._phase += phase_adj

        # Wrap phase
        while self._phase > np.pi:
            self._phase -= 2 * np.pi
        while self._phase < -np.pi:
            self._phase += 2 * np.pi

        return corrected

    def _phase_detector_qpsk(self, sample: complex) -> float:
        """QPSK phase detector.

        Uses decision-directed detection for better performance.
        """
        # Decision on ideal constellation point
        phase = np.angle(sample)

        # Quantize to nearest pi/4 multiple (QPSK constellation)
        ideal_phase = np.round(phase / (np.pi / 4)) * (np.pi / 4)

        # Phase error
        error = phase - ideal_phase

        # Wrap to [-pi, pi]
        while error > np.pi:
            error -= 2 * np.pi
        while error < -np.pi:
            error += 2 * np.pi

        return float(error)

    def process_block(self, samples: np.ndarray) -> np.ndarray:
        """Process a block of samples.

        Args:
            samples: Complex input samples

        Returns:
            Phase-corrected samples
        """
        output = np.zeros(len(samples), dtype=np.complex128)
        for i, sample in enumerate(samples):
            output[i] = self.process(sample)
        return output

    @property
    def frequency_offset(self) -> float:
        """Get current frequency offset estimate."""
        return self._freq


class CQPSKDemodulator:
    """CQPSK (H-DQPSK) demodulator for P25 Phase II.

    This implements a complete Phase II receiver chain:
    1. RRC matched filter (alpha=1.0)
    2. Costas loop carrier recovery
    3. Mueller-Muller timing recovery
    4. Differential pi/4-DQPSK decoding

    The demodulator maintains state for streaming operation.

    Example usage:
        demod = CQPSKDemodulator(sample_rate=48000)
        dibits = demod.demodulate(iq_samples)
    """

    # pi/4-DQPSK differential phase to dibit mapping
    # Phase change: 45°, 135°, -135°, -45° -> dibits 0, 1, 2, 3
    PHASE_TO_DIBIT = {
        0: 0,  # +45° (pi/4)
        1: 1,  # +135° (3*pi/4)
        2: 2,  # -135° (-3*pi/4)
        3: 3,  # -45° (-pi/4)
    }

    def __init__(
        self,
        sample_rate: int = 48000,
        symbol_rate: int = 12000,
        rrc_alpha: float = 1.0,
        rrc_taps: int = 65,
        carrier_loop_bw: float = 0.01,
        timing_loop_bw: float = 0.01,
    ):
        """Initialize CQPSK demodulator.

        Args:
            sample_rate: Input sample rate in Hz
            symbol_rate: Symbol rate (12000 baud for Phase II)
            rrc_alpha: RRC filter roll-off (1.0 for Phase II)
            rrc_taps: Number of filter taps
            carrier_loop_bw: Costas loop bandwidth
            timing_loop_bw: Timing recovery loop bandwidth
        """
        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.samples_per_symbol = sample_rate / symbol_rate

        # RRC matched filter
        self._rrc = design_rrc_filter_phase2(
            self.samples_per_symbol, rrc_taps, rrc_alpha
        )

        # Carrier recovery (Costas loop)
        self._carrier_loop = CostasLoop(loop_bw=carrier_loop_bw)

        # Timing recovery (Mueller-Muller)
        self._timing_recovery = MuellerMullerTED(
            samples_per_symbol=self.samples_per_symbol,
            loop_bw=timing_loop_bw,
        )

        # Previous symbol phase for differential decode
        self._prev_phase = 0.0

        # Filter state
        self._filter_state = signal.lfilter_zi(self._rrc, 1.0).astype(np.complex128)

        logger.info(
            f"CQPSK demod initialized: {sample_rate} Hz, {symbol_rate} baud, "
            f"{self.samples_per_symbol:.2f} sps"
        )

    def reset(self) -> None:
        """Reset demodulator state."""
        self._carrier_loop.reset()
        self._timing_recovery.reset()
        self._prev_phase = 0.0
        self._filter_state = signal.lfilter_zi(self._rrc, 1.0).astype(np.complex128)

    def demodulate(self, iq: np.ndarray) -> np.ndarray:
        """Demodulate CQPSK signal to dibits.

        Args:
            iq: Complex IQ samples

        Returns:
            Dibits (0-3)
        """
        if len(iq) == 0:
            return np.array([], dtype=np.uint8)

        # Matched filter
        filtered, self._filter_state = signal.lfilter(
            self._rrc, 1.0, iq, zi=self._filter_state * iq[0]
        )

        # Carrier recovery
        corrected = self._carrier_loop.process_block(filtered)

        # Timing recovery
        symbols, decisions, _ = self._timing_recovery.process_block(corrected)

        if len(symbols) == 0:
            return np.array([], dtype=np.uint8)

        # Differential decode (pi/4-DQPSK)
        dibits = self._differential_decode(symbols)

        return dibits

    def _differential_decode(self, symbols: np.ndarray) -> np.ndarray:
        """Decode pi/4-DQPSK symbols to dibits.

        In pi/4-DQPSK, information is encoded in the phase change between
        successive symbols, not the absolute phase.

        Args:
            symbols: Complex symbol values

        Returns:
            Dibits (0-3)
        """
        dibits = []

        for symbol in symbols:
            # Get phase
            phase = np.angle(symbol)

            # Phase difference
            delta_phase = phase - self._prev_phase

            # Normalize to [-pi, pi]
            while delta_phase > np.pi:
                delta_phase -= 2 * np.pi
            while delta_phase < -np.pi:
                delta_phase += 2 * np.pi

            # Quantize to nearest pi/4 multiple
            # Phase changes: +pi/4, +3pi/4, -3pi/4, -pi/4
            phase_idx = int(np.round((delta_phase + np.pi) / (np.pi / 4))) % 8

            # Map to dibit
            # Index 1 (+45°) -> 0
            # Index 3 (+135°) -> 1
            # Index 5 (-135°) -> 2
            # Index 7 (-45°) -> 3
            dibit_map = {1: 0, 3: 1, 5: 2, 7: 3, 0: 0, 2: 1, 4: 2, 6: 3}
            dibit = dibit_map.get(phase_idx, 0)

            dibits.append(dibit)
            self._prev_phase = phase

        return np.array(dibits, dtype=np.uint8)

    def get_carrier_offset(self) -> float:
        """Get current carrier frequency offset estimate.

        Returns:
            Frequency offset in Hz
        """
        return self._carrier_loop.frequency_offset * self.sample_rate / (2 * np.pi)


def cqpsk_demod_simple(
    iq: np.ndarray,
    sample_rate: int = 48000,
    symbol_rate: int = 12000,
) -> np.ndarray:
    """Simplified CQPSK demodulator (stateless, single-shot).

    Args:
        iq: Complex IQ samples
        sample_rate: Sample rate in Hz
        symbol_rate: Symbol rate in baud

    Returns:
        Dibits (0-3)
    """
    demod = CQPSKDemodulator(sample_rate=sample_rate, symbol_rate=symbol_rate)
    return demod.demodulate(iq)
