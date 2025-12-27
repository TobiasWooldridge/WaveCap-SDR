#!/usr/bin/env python3
"""Compare P25 C4FM demodulation pipelines between SDRTrunk and WaveCap-SDR.

This script performs a detailed comparison of:
1. FM discriminator output
2. Symbol timing recovery
3. Symbol decisions (dibits)
4. Sync detection
5. NID/TSBK decoding

Usage:
    # Compare on a captured IQ file (48kHz P25 channel)
    python scripts/compare_p25_pipelines.py capture.wav

    # Compare on a high-rate IQ capture with offset
    python scripts/compare_p25_pipelines.py --sample-rate 6000000 --offset 175000 capture.wav

    # Just test WaveCap pipeline
    python scripts/compare_p25_pipelines.py --wavecap-only capture.wav
"""

from __future__ import annotations

import argparse
import logging
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import signal

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator, design_rrc_filter
from wavecapsdr.dsp.fm import quadrature_demod

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineStageResult:
    """Results from a single pipeline stage."""
    name: str
    output: np.ndarray
    stats: dict


def load_iq_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load complex IQ from a stereo WAV file.

    Args:
        path: Path to WAV file

    Returns:
        Complex IQ samples, sample rate
    """
    with wave.open(str(path), 'rb') as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        n_channels = wf.getnchannels()

        if n_channels != 2:
            raise ValueError(f"Expected stereo WAV (I/Q), got {n_channels} channels")

        raw = wf.readframes(n_frames)
        samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
        iq = (samples[:, 0] + 1j * samples[:, 1]).astype(np.complex64) / 32768.0

    return iq, sample_rate


def shift_and_decimate(
    iq: np.ndarray,
    sample_rate: int,
    offset_hz: float = 0.0,
    target_rate: int = 48000,
) -> tuple[np.ndarray, int]:
    """Frequency shift and decimate IQ to target rate.

    Args:
        iq: Complex IQ samples
        sample_rate: Input sample rate
        offset_hz: Frequency offset to shift
        target_rate: Target sample rate

    Returns:
        Shifted/decimated IQ, new sample rate
    """
    # Frequency shift
    if offset_hz != 0:
        t = np.arange(len(iq)) / sample_rate
        iq = iq * np.exp(-2j * np.pi * offset_hz * t).astype(np.complex64)

    # Decimate if needed
    if sample_rate > target_rate:
        decimation = sample_rate // target_rate
        # Design anti-alias filter
        nyq = sample_rate / 2
        cutoff = target_rate / 2 * 0.8  # 80% of new Nyquist
        order = 101
        h = signal.firwin(order, cutoff / nyq)
        iq = signal.lfilter(h, 1.0, iq)
        iq = iq[::decimation]
        sample_rate = sample_rate // decimation

    return iq, sample_rate


class SDRTrunkReferencePipeline:
    """SDRTrunk-compatible P25 demodulation pipeline.

    This implements the SDRTrunk approach:
    1. FM discriminator outputs phase in radians
    2. Equalizer applies PLL + gain (initial gain 1.219, range [1.0, 1.25])
    3. Linear interpolation for symbol timing
    4. Decision boundaries at ±π/2
    """

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.samples_per_symbol = sample_rate / 4800

        # FM demod state
        self.prev_sample = 0.0 + 0.0j

        # Equalizer state
        self.pll = 0.0
        self.gain = 1.219  # SDRTrunk's initial value

    def fm_discriminator(self, iq: np.ndarray) -> np.ndarray:
        """SDRTrunk-style FM discriminator.

        Computes: atan(Im(z[n] * conj(z[n-1])) / Re(z[n] * conj(z[n-1])))
        """
        # Prepend previous sample
        iq_ext = np.concatenate([[self.prev_sample], iq])
        self.prev_sample = iq[-1]

        # Differential: z[n] * conj(z[n-1])
        diff = iq_ext[1:] * np.conj(iq_ext[:-1])

        # Arctangent (not atan2 - SDRTrunk uses atan(Q/I))
        # Handle division carefully
        i_part = diff.real
        q_part = diff.imag
        i_part = np.where(i_part == 0, np.finfo(float).eps, i_part)
        phase = np.arctan(q_part / i_part)

        return phase.astype(np.float32)

    def equalize(self, phase: np.ndarray) -> np.ndarray:
        """Apply PLL and gain correction."""
        return (phase + self.pll) * self.gain

    def recover_symbols(self, phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Recover symbols with linear interpolation timing.

        Returns:
            dibits: Symbol decisions (0-3)
            soft: Soft symbol values (radians)
        """
        # Apply equalization
        eq_phase = self.equalize(phase)

        # Simple symbol recovery at fixed sample points
        # (A full implementation would include timing recovery)
        n_symbols = int(len(eq_phase) / self.samples_per_symbol)
        dibits = []
        soft = []

        sample_point = self.samples_per_symbol / 2  # Start at mid-symbol

        for _ in range(n_symbols):
            idx = int(sample_point)
            if idx < len(eq_phase) - 1:
                mu = sample_point - idx
                # Linear interpolation
                symbol = eq_phase[idx] + (eq_phase[idx + 1] - eq_phase[idx]) * mu

                soft.append(symbol)

                # Decision at ±π/2 boundaries
                if symbol >= np.pi / 2:
                    dibits.append(1)  # +3
                elif symbol >= 0:
                    dibits.append(0)  # +1
                elif symbol >= -np.pi / 2:
                    dibits.append(2)  # -1
                else:
                    dibits.append(3)  # -3

            sample_point += self.samples_per_symbol

        return np.array(dibits, dtype=np.uint8), np.array(soft, dtype=np.float32)

    def demodulate(self, iq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Full demodulation pipeline."""
        phase = self.fm_discriminator(iq)
        dibits, soft = self.recover_symbols(phase)
        return dibits, soft


class WaveCapPipeline:
    """WaveCap-SDR P25 demodulation pipeline.

    This implements WaveCap's approach:
    1. C4FM-specific FM discriminator with P25 scaling (±1800 Hz → ±3)
    2. RRC matched filter
    3. Gardner timing recovery
    4. Adaptive gain normalization (range [0.01, 10.0])
    5. Decision boundaries at ±2
    """

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.demod = C4FMDemodulator(sample_rate=sample_rate)

    def fm_discriminator(self, iq: np.ndarray) -> np.ndarray:
        """WaveCap-style FM discriminator for P25 C4FM.

        Uses the C4FM demodulator's internal FM discriminator which
        scales properly for P25 deviation (±1800 Hz → ±3 symbols).
        """
        # Use the C4FM demodulator's internal FM discriminator
        return self.demod._fm_discriminator(iq)

    def demodulate(self, iq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Full demodulation pipeline."""
        return self.demod.demodulate(iq)


def compare_pipelines(
    iq: np.ndarray,
    sample_rate: int,
    verbose: bool = False,
) -> dict:
    """Compare SDRTrunk and WaveCap P25 pipelines.

    Args:
        iq: Complex IQ samples at channel rate
        sample_rate: Sample rate in Hz
        verbose: Print detailed comparison

    Returns:
        Comparison results dictionary
    """
    # Initialize pipelines
    sdrtrunk = SDRTrunkReferencePipeline(sample_rate)
    wavecap = WaveCapPipeline(sample_rate)

    # Stage 1: FM Discriminator
    print("\n=== Stage 1: FM Discriminator ===")
    sdt_phase = sdrtrunk.fm_discriminator(iq)
    wc_phase = wavecap.fm_discriminator(iq)

    print(f"SDRTrunk: mean={np.mean(sdt_phase):.4f}, std={np.std(sdt_phase):.4f}, "
          f"min={np.min(sdt_phase):.4f}, max={np.max(sdt_phase):.4f}")
    print(f"WaveCap:  mean={np.mean(wc_phase):.4f}, std={np.std(wc_phase):.4f}, "
          f"min={np.min(wc_phase):.4f}, max={np.max(wc_phase):.4f}")

    # Key insight: SDRTrunk outputs radians (±π/4, ±3π/4 for ideal C4FM)
    # WaveCap outputs normalized symbols (±1, ±3 for ideal C4FM)
    sdt_expected_std = np.pi * np.sqrt(5) / 4  # ~1.76 for uniform ±π/4, ±3π/4
    wc_expected_std = np.sqrt(5)  # ~2.24 for uniform ±1, ±3

    print(f"\nExpected std - SDRTrunk: ~{sdt_expected_std:.2f} (radians), WaveCap: ~{wc_expected_std:.2f} (symbols)")

    # Normalize WaveCap FM output to radians for comparison
    # WaveCap scales by sample_rate/(2π) * 3/1800, so to convert back:
    # symbol_to_radian = π/4 per symbol level (±1 → ±π/4, ±3 → ±3π/4)
    wc_phase_normalized = wc_phase / (3.0 * 4.0 / np.pi)  # Convert symbols back to radians
    print(f"\nNormalized comparison (radians):")
    print(f"SDRTrunk: std={np.std(sdt_phase):.4f}")
    print(f"WaveCap:  std={np.std(wc_phase_normalized):.4f} (normalized from symbols)")

    # Stage 2: Full Symbol Recovery
    print("\n=== Stage 2: Symbol Recovery ===")
    sdt_dibits, sdt_soft = sdrtrunk.demodulate(iq)
    wc_dibits, wc_soft = wavecap.demodulate(iq)

    print(f"SDRTrunk: {len(sdt_dibits)} symbols, soft std={np.std(sdt_soft):.4f} (radians)")
    print(f"WaveCap:  {len(wc_dibits)} symbols, soft std={np.std(wc_soft):.4f} (symbol levels)")

    # Normalize SDRTrunk radians to symbol levels: ±π/4 → ±1, ±3π/4 → ±3
    sdt_soft_symbols = sdt_soft * (4.0 / np.pi)
    print(f"\nNormalized to symbol levels:")
    print(f"SDRTrunk: std={np.std(sdt_soft_symbols):.4f}")
    print(f"WaveCap:  std={np.std(wc_soft):.4f}")
    print(f"Expected: ~{np.sqrt(5):.4f}")

    # Compare dibit decisions
    min_len = min(len(sdt_dibits), len(wc_dibits))
    if min_len > 0:
        matches = np.sum(sdt_dibits[:min_len] == wc_dibits[:min_len])
        match_rate = matches / min_len * 100
        print(f"\nDibit agreement: {matches}/{min_len} ({match_rate:.1f}%)")

        # Show disagreement breakdown
        for i in range(4):
            sdt_count = np.sum(sdt_dibits[:min_len] == i)
            wc_count = np.sum(wc_dibits[:min_len] == i)
            print(f"  Dibit {i}: SDRTrunk={sdt_count}, WaveCap={wc_count}")

    # Stage 3: Symbol Quality Analysis
    print("\n=== Stage 3: Symbol Quality ===")

    # WaveCap soft symbols should cluster around ±1, ±3
    wc_inner = np.sum((np.abs(wc_soft) > 0.5) & (np.abs(wc_soft) < 2.0))
    wc_outer = np.sum(np.abs(wc_soft) >= 2.0)
    print(f"WaveCap inner symbols (|x|<2): {wc_inner} ({wc_inner/len(wc_soft)*100:.1f}%)")
    print(f"WaveCap outer symbols (|x|>=2): {wc_outer} ({wc_outer/len(wc_soft)*100:.1f}%)")

    # SDRTrunk soft symbols should cluster around ±π/4, ±3π/4
    boundary = np.pi / 2
    sdt_inner = np.sum(np.abs(sdt_soft) < boundary)
    sdt_outer = np.sum(np.abs(sdt_soft) >= boundary)
    print(f"SDRTrunk inner symbols (|x|<π/2): {sdt_inner} ({sdt_inner/len(sdt_soft)*100:.1f}%)")
    print(f"SDRTrunk outer symbols (|x|>=π/2): {sdt_outer} ({sdt_outer/len(sdt_soft)*100:.1f}%)")

    return {
        'sdrtrunk_phase': sdt_phase,
        'wavecap_phase': wc_phase,
        'sdrtrunk_dibits': sdt_dibits,
        'wavecap_dibits': wc_dibits,
        'sdrtrunk_soft': sdt_soft,
        'wavecap_soft': wc_soft,
        'match_rate': match_rate if min_len > 0 else 0,
    }


def test_wavecap_only(iq: np.ndarray, sample_rate: int) -> dict:
    """Test just WaveCap's P25 pipeline."""
    from wavecapsdr.decoders.p25 import P25Decoder

    print("\n=== WaveCap-SDR P25 Pipeline ===")

    # C4FM demodulation
    demod = C4FMDemodulator(sample_rate=sample_rate)
    dibits, soft = demod.demodulate(iq)

    print(f"C4FM: {len(dibits)} symbols recovered")
    print(f"Soft symbol stats: mean={np.mean(soft):.4f}, std={np.std(soft):.4f}")

    # Expected std for C4FM symbols
    expected_std = np.sqrt(5)  # ~2.24
    print(f"Expected std: {expected_std:.4f} (deviation: {abs(np.std(soft) - expected_std):.4f})")

    # Symbol distribution
    print("\nSymbol distribution:")
    for i in range(4):
        count = np.sum(dibits == i)
        pct = count / len(dibits) * 100
        print(f"  Dibit {i}: {count} ({pct:.1f}%)")

    # Try P25 decoding
    print("\n=== P25 Frame Decoding ===")
    decoder = P25Decoder(sample_rate=sample_rate)
    frames = decoder.process_iq(iq)
    print(f"Decoded {len(frames)} P25 frames")

    for frame in frames[:10]:
        print(f"  {frame}")

    return {
        'dibits': dibits,
        'soft': soft,
        'frames': frames,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compare P25 demodulation pipelines',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('input', type=Path, help='Input WAV file (stereo IQ)')
    parser.add_argument('--sample-rate', type=int, help='Override sample rate from WAV')
    parser.add_argument('--offset', type=float, default=0, help='Frequency offset in Hz')
    parser.add_argument('--target-rate', type=int, default=48000, help='Target sample rate for P25')
    parser.add_argument('--wavecap-only', action='store_true', help='Only test WaveCap pipeline')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    # Load IQ
    logger.info(f"Loading {args.input}")
    iq, sample_rate = load_iq_wav(args.input)

    if args.sample_rate:
        sample_rate = args.sample_rate

    logger.info(f"Loaded {len(iq)} samples at {sample_rate} Hz")

    # Shift and decimate if needed
    if sample_rate != args.target_rate or args.offset != 0:
        logger.info(f"Shifting by {args.offset} Hz, decimating to {args.target_rate} Hz")
        iq, sample_rate = shift_and_decimate(iq, sample_rate, args.offset, args.target_rate)
        logger.info(f"After processing: {len(iq)} samples at {sample_rate} Hz")

    # Run comparison or just WaveCap
    if args.wavecap_only:
        results = test_wavecap_only(iq, sample_rate)
    else:
        results = compare_pipelines(iq, sample_rate, args.verbose)

    print("\n=== Summary ===")
    if 'match_rate' in results:
        print(f"Pipeline agreement: {results['match_rate']:.1f}%")

    if 'frames' in results:
        print(f"P25 frames decoded: {len(results['frames'])}")


if __name__ == '__main__':
    main()
