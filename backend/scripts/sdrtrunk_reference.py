#!/usr/bin/env python3
"""SDRTrunk-compatible P25 C4FM demodulation reference implementation.

This provides a Python port of SDRTrunk's P25P1DemodulatorC4FM for comparison
with WaveCap-SDR's implementation. The key differences are:

1. SDRTrunk works with FM phase values (radians): ±π/4 (inner), ±3π/4 (outer)
2. WaveCap works with normalized symbols: ±1 (inner), ±3 (outer)
3. SDRTrunk uses gain range [1.0, 1.25] with initial 1.219
4. WaveCap uses wider gain range [0.01, 10.0] to handle different input scales

This module allows testing both approaches on the same IQ data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


# Constants from SDRTrunk P25P1DemodulatorC4FM.java
SYMBOL_RATE = 4800
TWO_PI = 2.0 * np.pi

# Equalizer constants
EQUALIZER_LOOP_GAIN = 0.15  # 15% per update
EQUALIZER_MAXIMUM_PLL = np.pi / 3.0  # ±800 Hz max frequency offset
EQUALIZER_MAXIMUM_GAIN = 1.25
EQUALIZER_INITIAL_GAIN = 1.219  # SDRTrunk's empirically determined initial gain

# Sync detection threshold
SYNC_THRESHOLD = 80.0

# P25 Phase 1 sync pattern: 24 dibits (48 bits)
# The sync word 0x5575F5FF55FF in dibit form maps to these ideal phase values
# Pattern dibits: [1,0,1,0,1,1,1,0,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1]
# Which maps to symbols: [+3,+1,+3,+1,+3,+3,+3,+1,+3,+1,+3,+3,+3,+3,+3,+1,+3,+3,+3,+3,+3,+3,+3,+3]
SYNC_PATTERN_DIBITS = [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]

# Map dibit to symbol (+3, +1, -1, -3)
DIBIT_TO_SYMBOL = {0: 1.0, 1: 3.0, 2: -1.0, 3: -3.0}

# SDRTrunk uses phase values (radians) for symbols
SYMBOL_TO_PHASE = {
    3.0: 3.0 * np.pi / 4.0,   # +3 -> +3π/4
    1.0: np.pi / 4.0,         # +1 -> +π/4
    -1.0: -np.pi / 4.0,       # -1 -> -π/4
    -3.0: -3.0 * np.pi / 4.0, # -3 -> -3π/4
}


def sync_pattern_to_phases() -> np.ndarray:
    """Convert sync pattern dibits to ideal phase values (radians)."""
    phases = []
    for dibit in SYNC_PATTERN_DIBITS:
        symbol = DIBIT_TO_SYMBOL[dibit]
        phase = SYMBOL_TO_PHASE[symbol]
        phases.append(phase)
    return np.array(phases, dtype=np.float32)


SYNC_PATTERN_PHASES = sync_pattern_to_phases()


@dataclass
class Dibit:
    """P25 dibit with ideal phase value."""
    value: int  # 0, 1, 2, or 3
    symbol: float  # +1, +3, -1, or -3
    ideal_phase: float  # Phase in radians

    @staticmethod
    def from_soft_symbol(soft: float) -> 'Dibit':
        """Decide dibit from soft symbol (phase in radians)."""
        # SDRTrunk decision boundary is π/2
        boundary = np.pi / 2.0

        if soft >= boundary:
            return Dibit(1, 3.0, 3.0 * np.pi / 4.0)  # +3
        elif soft >= 0:
            return Dibit(0, 1.0, np.pi / 4.0)        # +1
        elif soft >= -boundary:
            return Dibit(2, -1.0, -np.pi / 4.0)      # -1
        else:
            return Dibit(3, -3.0, -3.0 * np.pi / 4.0) # -3


class LinearInterpolator:
    """Linear interpolation between two samples (SDRTrunk compatible)."""

    @staticmethod
    def calculate(sample1: float, sample2: float, mu: float) -> float:
        """Interpolate between sample1 and sample2 at fractional position mu."""
        return sample1 + (sample2 - sample1) * mu


class SDRTrunkEqualizer:
    """Port of SDRTrunk's P25P1DemodulatorC4FM.Equalizer.

    Applies PLL (frequency offset correction) and gain adjustment to
    FM-demodulated phase samples.
    """

    def __init__(self):
        self.pll = 0.0
        self.gain = EQUALIZER_INITIAL_GAIN
        self.initialized = False

    def reset(self):
        """Reset equalizer to initial state."""
        self.pll = 0.0
        self.gain = EQUALIZER_INITIAL_GAIN
        self.initialized = False

    def equalize(self, sample: float) -> float:
        """Apply equalization: (sample + pll) * gain"""
        return (sample + self.pll) * self.gain

    def get_equalized_symbol(
        self,
        sample1: float,
        sample2: float,
        mu: float
    ) -> float:
        """Equalize samples and interpolate at mu."""
        s1 = self.equalize(sample1)
        s2 = self.equalize(sample2)
        return LinearInterpolator.calculate(s1, s2, mu)

    def apply_correction(self, pll_adj: float, gain_adj: float):
        """Apply correction with loop gain."""
        if self.initialized:
            self.pll += pll_adj * EQUALIZER_LOOP_GAIN
            self.gain += gain_adj * EQUALIZER_LOOP_GAIN
        else:
            self.pll += pll_adj
            self.gain += gain_adj
            self.initialized = True

        # Constrain values
        self.pll = np.clip(self.pll, -EQUALIZER_MAXIMUM_PLL, EQUALIZER_MAXIMUM_PLL)
        self.gain = np.clip(self.gain, 1.0, EQUALIZER_MAXIMUM_GAIN)


class SDRTrunkFMDemodulator:
    """Port of SDRTrunk's ScalarFMDemodulator.

    Converts complex IQ samples to phase differences using arctangent.
    """

    def __init__(self, gain: float = 1.0):
        self.gain = gain
        self.prev_i = 0.0
        self.prev_q = 0.0

    def reset(self):
        self.prev_i = 0.0
        self.prev_q = 0.0

    def demodulate(self, i: np.ndarray, q: np.ndarray) -> np.ndarray:
        """FM demodulate I/Q samples to phase values.

        SDRTrunk uses: atan(demodQ / demodI) where
        demodI = i[n] * i[n-1] - q[n] * (-q[n-1])
        demodQ = q[n] * i[n-1] + i[n] * (-q[n-1])

        This is essentially the angle of the product of current sample
        with the conjugate of the previous sample.
        """
        n = len(i)
        demodulated = np.zeros(n, dtype=np.float32)

        # First sample
        demod_i = i[0] * self.prev_i - q[0] * (-self.prev_q)
        demod_q = q[0] * self.prev_i + i[0] * (-self.prev_q)
        if demod_i != 0:
            demodulated[0] = np.arctan(demod_q / demod_i)
        else:
            demodulated[0] = np.arctan(demod_q / np.finfo(float).eps)

        # Remaining samples
        for x in range(1, n):
            demod_i = i[x] * i[x-1] - q[x] * (-q[x-1])
            demod_q = q[x] * i[x-1] + i[x] * (-q[x-1])
            if demod_i != 0:
                demodulated[x] = np.arctan(demod_q / demod_i)
            else:
                demodulated[x] = np.arctan(demod_q / np.finfo(float).eps)

        # Store last sample
        self.prev_i = i[-1]
        self.prev_q = q[-1]

        return demodulated


class SDRTrunkC4FMDemodulator:
    """Port of SDRTrunk's P25P1DemodulatorC4FM.

    This is a reference implementation for comparison with WaveCap-SDR.
    """

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.samples_per_symbol = sample_rate / SYMBOL_RATE

        # FM demodulator
        self.fm_demod = SDRTrunkFMDemodulator()

        # Equalizer (PLL + gain)
        self.equalizer = SDRTrunkEqualizer()

        # Symbol timing state
        self.sample_point = self.samples_per_symbol
        self.buffer = np.zeros(2048, dtype=np.float32)
        self.buffer_pointer = 0

        # Statistics
        self.symbols_processed = 0
        self.syncs_detected = 0

    def reset(self):
        """Reset demodulator state."""
        self.fm_demod.reset()
        self.equalizer.reset()
        self.sample_point = self.samples_per_symbol
        self.buffer.fill(0)
        self.buffer_pointer = 0
        self.symbols_processed = 0
        self.syncs_detected = 0

    def process_iq(self, iq: np.ndarray) -> Iterator[tuple[Dibit, float]]:
        """Process IQ samples and yield (dibit, soft_symbol) tuples.

        Args:
            iq: Complex IQ samples

        Yields:
            (dibit, soft_symbol) for each recovered symbol
        """
        # FM demodulate to phase values
        i = iq.real.astype(np.float32)
        q = iq.imag.astype(np.float32)
        phases = self.fm_demod.demodulate(i, q)

        # Process phases through symbol recovery
        for phase in phases:
            self.buffer_pointer += 1
            self.sample_point -= 1.0

            if self.buffer_pointer >= len(self.buffer) - 1:
                # Shift buffer
                shift = len(self.buffer) // 2
                self.buffer[:-shift] = self.buffer[shift:]
                self.buffer[-shift:] = 0
                self.buffer_pointer -= shift

            # Store sample
            self.buffer[self.buffer_pointer] = phase

            # Symbol decision point
            if self.sample_point < 1.0:
                self.symbols_processed += 1

                # Get equalized, interpolated symbol
                idx = self.buffer_pointer
                mu = self.sample_point

                if idx > 0 and idx < len(self.buffer) - 1:
                    soft_symbol = self.equalizer.get_equalized_symbol(
                        self.buffer[idx],
                        self.buffer[idx + 1],
                        mu
                    )

                    # Make decision
                    dibit = Dibit.from_soft_symbol(soft_symbol)

                    yield dibit, soft_symbol

                # Advance to next symbol
                self.sample_point += self.samples_per_symbol

    def demodulate(self, iq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Batch demodulate IQ samples.

        Args:
            iq: Complex IQ samples

        Returns:
            dibits: Array of dibit values (0-3)
            soft_symbols: Array of soft symbol values (phase in radians)
        """
        results = list(self.process_iq(iq))
        if not results:
            return np.array([], dtype=np.uint8), np.array([], dtype=np.float32)

        dibits = np.array([d.value for d, _ in results], dtype=np.uint8)
        soft_symbols = np.array([s for _, s in results], dtype=np.float32)

        return dibits, soft_symbols


def compare_fm_demodulators(iq: np.ndarray, sample_rate: int = 48000):
    """Compare SDRTrunk and WaveCap FM demodulation approaches.

    Args:
        iq: Complex IQ samples
        sample_rate: Sample rate in Hz

    Returns:
        Dictionary with comparison results
    """
    from wavecapsdr.dsp.fm import quadrature_fm_demod

    # SDRTrunk approach: atan(demodQ / demodI)
    sdrtrunk_fm = SDRTrunkFMDemodulator()
    sdrtrunk_phases = sdrtrunk_fm.demodulate(iq.real.astype(np.float32),
                                              iq.imag.astype(np.float32))

    # WaveCap approach: quadrature demod with proper scaling
    wavecap_phases = quadrature_fm_demod(iq)

    # The key difference: SDRTrunk outputs raw atan values (radians),
    # WaveCap scales to frequency then normalizes for symbols

    return {
        'sdrtrunk_phases': sdrtrunk_phases,
        'wavecap_phases': wavecap_phases,
        'sdrtrunk_std': np.std(sdrtrunk_phases),
        'wavecap_std': np.std(wavecap_phases),
        'sdrtrunk_mean': np.mean(sdrtrunk_phases),
        'wavecap_mean': np.mean(wavecap_phases),
    }


def compare_symbol_recovery(
    iq: np.ndarray,
    sample_rate: int = 48000,
) -> dict:
    """Compare SDRTrunk and WaveCap symbol recovery.

    Args:
        iq: Complex IQ samples
        sample_rate: Sample rate in Hz

    Returns:
        Dictionary with comparison results
    """
    from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator

    # SDRTrunk reference
    sdrtrunk = SDRTrunkC4FMDemodulator(sample_rate)
    sdrtrunk_dibits, sdrtrunk_soft = sdrtrunk.demodulate(iq)

    # WaveCap
    wavecap = C4FMDemodulator(sample_rate=sample_rate)
    wavecap_dibits, wavecap_soft = wavecap.demodulate(iq)

    # Compare results
    min_len = min(len(sdrtrunk_dibits), len(wavecap_dibits))

    if min_len > 0:
        matches = np.sum(sdrtrunk_dibits[:min_len] == wavecap_dibits[:min_len])
        match_rate = matches / min_len * 100
    else:
        match_rate = 0.0

    return {
        'sdrtrunk_symbols': len(sdrtrunk_dibits),
        'wavecap_symbols': len(wavecap_dibits),
        'sdrtrunk_soft_std': np.std(sdrtrunk_soft) if len(sdrtrunk_soft) > 0 else 0,
        'wavecap_soft_std': np.std(wavecap_soft) if len(wavecap_soft) > 0 else 0,
        'match_rate': match_rate,
        'sdrtrunk_dibits': sdrtrunk_dibits,
        'wavecap_dibits': wavecap_dibits,
        'sdrtrunk_soft': sdrtrunk_soft,
        'wavecap_soft': wavecap_soft,
    }


if __name__ == '__main__':
    import argparse
    import wave

    parser = argparse.ArgumentParser(description='SDRTrunk reference P25 demodulator')
    parser.add_argument('input', help='Input WAV file (stereo IQ)')
    parser.add_argument('--compare', action='store_true', help='Compare with WaveCap')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load IQ from WAV file
    with wave.open(args.input, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        n_channels = wf.getnchannels()

        if n_channels != 2:
            print(f"Error: Expected stereo WAV (I/Q), got {n_channels} channels")
            exit(1)

        raw = wf.readframes(n_frames)
        samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
        iq = (samples[:, 0] + 1j * samples[:, 1]).astype(np.complex64) / 32768.0

    print(f"Loaded {len(iq)} IQ samples at {sample_rate} Hz")

    if args.compare:
        print("\n=== FM Demodulator Comparison ===")
        fm_results = compare_fm_demodulators(iq, sample_rate)
        print(f"SDRTrunk phases: std={fm_results['sdrtrunk_std']:.4f}, mean={fm_results['sdrtrunk_mean']:.4f}")
        print(f"WaveCap phases:  std={fm_results['wavecap_std']:.4f}, mean={fm_results['wavecap_mean']:.4f}")

        print("\n=== Symbol Recovery Comparison ===")
        sym_results = compare_symbol_recovery(iq, sample_rate)
        print(f"SDRTrunk: {sym_results['sdrtrunk_symbols']} symbols, soft std={sym_results['sdrtrunk_soft_std']:.4f}")
        print(f"WaveCap:  {sym_results['wavecap_symbols']} symbols, soft std={sym_results['wavecap_soft_std']:.4f}")
        print(f"Dibit match rate: {sym_results['match_rate']:.1f}%")
    else:
        # Just run SDRTrunk reference
        demod = SDRTrunkC4FMDemodulator(sample_rate)
        dibits, soft = demod.demodulate(iq)
        print(f"Recovered {len(dibits)} symbols")
        print(f"Soft symbol std: {np.std(soft):.4f} (expected ~0.785 for ±π/4, ±3π/4)")

        # Show first few symbols
        if len(dibits) > 0:
            print(f"First 20 dibits: {dibits[:20]}")
