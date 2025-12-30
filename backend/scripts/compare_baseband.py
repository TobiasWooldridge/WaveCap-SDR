#!/usr/bin/env python3
"""Compare baseband IQ recordings from WaveCap and SDRTrunk.

This script analyzes two baseband recordings and identifies differences
at each processing stage to help debug discrepancies.

Usage:
    python scripts/compare_baseband.py wavecap.wav sdrtrunk.wav

Or compare a single recording against expected behavior:
    python scripts/compare_baseband.py --analyze sdrtrunk.wav
"""

from __future__ import annotations

import argparse
import logging
import sys
import wave
from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def load_baseband_wav(filepath: str) -> tuple[np.ndarray, int]:
    """Load baseband WAV file (stereo int16 or float32)."""
    with wave.open(filepath, 'rb') as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()

        logger.info(f"Loading: {filepath}")
        logger.info(f"  Format: {n_channels} ch, {sample_width} bytes/sample, {sample_rate} Hz, {n_frames} frames")

        raw = wf.readframes(n_frames)

        if n_channels == 2 and sample_width == 4:
            # Interleaved I/Q as float32
            data = np.frombuffer(raw, dtype=np.float32)
            i = data[0::2]
            q = data[1::2]
            return (i + 1j * q).astype(np.complex64), sample_rate
        elif n_channels == 2 and sample_width == 2:
            # Interleaved I/Q as int16
            data = np.frombuffer(raw, dtype=np.int16)
            i = data[0::2].astype(np.float32) / 32768.0
            q = data[1::2].astype(np.float32) / 32768.0
            return (i + 1j * q).astype(np.complex64), sample_rate
        else:
            raise ValueError(f"Unexpected format: {n_channels} ch, {sample_width} bytes")


def analyze_iq_statistics(iq: np.ndarray, name: str) -> dict:
    """Analyze IQ sample statistics."""
    stats = {
        "name": name,
        "length": len(iq),
        "i_mean": float(np.mean(iq.real)),
        "i_std": float(np.std(iq.real)),
        "i_min": float(np.min(iq.real)),
        "i_max": float(np.max(iq.real)),
        "q_mean": float(np.mean(iq.imag)),
        "q_std": float(np.std(iq.imag)),
        "q_min": float(np.min(iq.imag)),
        "q_max": float(np.max(iq.imag)),
        "power_mean": float(np.mean(np.abs(iq)**2)),
        "power_std": float(np.std(np.abs(iq)**2)),
        "dc_offset": float(np.abs(np.mean(iq))),
    }
    return stats


def print_iq_stats(stats: dict) -> None:
    """Print IQ statistics."""
    logger.info(f"\n=== IQ Statistics: {stats['name']} ===")
    logger.info(f"  Length: {stats['length']} samples")
    logger.info(f"  I: mean={stats['i_mean']:.6f}, std={stats['i_std']:.6f}, range=[{stats['i_min']:.4f}, {stats['i_max']:.4f}]")
    logger.info(f"  Q: mean={stats['q_mean']:.6f}, std={stats['q_std']:.6f}, range=[{stats['q_min']:.4f}, {stats['q_max']:.4f}]")
    logger.info(f"  Power: mean={stats['power_mean']:.6f}, std={stats['power_std']:.6f}")
    logger.info(f"  DC offset: {stats['dc_offset']:.6f}")


def analyze_spectrum(iq: np.ndarray, sample_rate: int, name: str) -> None:
    """Analyze and log spectrum characteristics."""
    # Compute PSD
    f, psd = scipy_signal.welch(iq, fs=sample_rate, nperseg=min(4096, len(iq)))

    # Find peak
    peak_idx = np.argmax(psd)
    peak_freq = f[peak_idx]
    peak_power = 10 * np.log10(psd[peak_idx])

    # Estimate signal bandwidth (3dB points)
    threshold = psd[peak_idx] / 2
    above_threshold = psd > threshold
    if np.any(above_threshold):
        first = np.argmax(above_threshold)
        last = len(above_threshold) - np.argmax(above_threshold[::-1]) - 1
        bandwidth = f[last] - f[first]
    else:
        bandwidth = 0

    logger.info(f"\n=== Spectrum: {name} ===")
    logger.info(f"  Peak frequency: {peak_freq:.1f} Hz")
    logger.info(f"  Peak power: {peak_power:.1f} dB")
    logger.info(f"  Estimated bandwidth: {bandwidth:.1f} Hz")


def fm_demodulate(iq: np.ndarray) -> np.ndarray:
    """FM demodulation (same as C4FM demodulator)."""
    # Differential phase
    diff = iq[1:] * np.conj(iq[:-1])
    phase = np.angle(diff)
    return phase


def analyze_fm_demod(iq: np.ndarray, sample_rate: int, name: str) -> np.ndarray:
    """Analyze FM demodulated signal."""
    fm = fm_demodulate(iq)

    # Normalize to symbol levels
    # P25 C4FM: +3, +1, -1, -3 deviation
    # At 4800 symbols/sec and 50000 sample rate, we have 10.4 samples per symbol
    samples_per_symbol = sample_rate / 4800

    logger.info(f"\n=== FM Demod: {name} ===")
    logger.info(f"  Samples per symbol: {samples_per_symbol:.2f}")
    logger.info(f"  Mean: {np.mean(fm):.6f}")
    logger.info(f"  Std: {np.std(fm):.6f}")
    logger.info(f"  Range: [{np.min(fm):.4f}, {np.max(fm):.4f}]")

    # Histogram of FM values (should show 4 peaks for C4FM)
    hist, bin_edges = np.histogram(fm, bins=50)
    peak_bins = np.argsort(hist)[-4:]
    peak_values = [(bin_edges[i] + bin_edges[i+1])/2 for i in sorted(peak_bins)]
    logger.info(f"  Top 4 histogram peaks: {[f'{v:.3f}' for v in peak_values]}")

    return fm


def find_sync_pattern(iq: np.ndarray, sample_rate: int) -> list[int]:
    """Find P25 sync pattern locations in demodulated signal."""
    # P25 frame sync as dibits: 1 1 1 1 1 3 1 1 3 3 1 1 3 3 3 3 1 3 1 3 3 3 3 3
    # Convert to deviation levels: +3 +3 +3 +3 +3 -3 +3 +3 -3 -3 +3 +3 -3 -3 -3 -3 +3 -3 +3 -3 -3 -3 -3 -3
    sync_dibits = np.array([1, 1, 1, 1, 1, 3, 1, 1, 3, 3, 1, 1, 3, 3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3])
    # Map to levels: 0->+3, 1->+1, 2->-1, 3->-3
    dibit_to_level = {0: 3, 1: 1, 2: -1, 3: -3}
    sync_levels = np.array([dibit_to_level.get(d, 0) for d in sync_dibits])

    # FM demodulate
    fm = fm_demodulate(iq)

    # Samples per symbol
    sps = sample_rate / 4800

    # Create sync pattern at sample rate
    sync_pattern = np.repeat(sync_levels, int(sps))

    # Normalize
    fm_norm = fm / np.std(fm) if np.std(fm) > 0 else fm
    sync_norm = sync_pattern / np.std(sync_pattern)

    # Correlate
    if len(fm_norm) > len(sync_norm):
        corr = np.correlate(fm_norm, sync_norm, mode='valid')
        threshold = 0.5 * len(sync_norm)

        # Find peaks above threshold
        sync_positions = []
        for i in range(len(corr) - 1):
            if corr[i] > threshold and corr[i] > corr[i-1] and corr[i] > corr[i+1]:
                sync_positions.append(i)

        return sync_positions
    return []


def analyze_sync_detection(iq: np.ndarray, sample_rate: int, name: str) -> None:
    """Analyze sync pattern detection."""
    sync_positions = find_sync_pattern(iq, sample_rate)

    logger.info(f"\n=== Sync Detection: {name} ===")
    logger.info(f"  Found {len(sync_positions)} sync patterns")

    if len(sync_positions) >= 2:
        # Analyze sync spacing
        spacings = np.diff(sync_positions)
        expected_spacing = 1728 * (sample_rate / 4800)  # 1728 symbols per frame
        logger.info(f"  Expected spacing: {expected_spacing:.1f} samples ({1728} symbols)")
        logger.info(f"  Actual spacing: mean={np.mean(spacings):.1f}, std={np.std(spacings):.1f}")

        if len(sync_positions) <= 10:
            logger.info(f"  Positions: {sync_positions}")
    elif len(sync_positions) == 1:
        logger.info(f"  Single sync at position {sync_positions[0]}")
    else:
        logger.info("  No sync patterns found!")


def run_decoder_test(iq: np.ndarray, sample_rate: int, name: str) -> dict:
    """Run through WaveCap decoder and return stats."""
    from wavecapsdr.trunking.control_channel import create_control_monitor, P25Modulation
    from wavecapsdr.trunking.config import TrunkingProtocol

    monitor = create_control_monitor(
        protocol=TrunkingProtocol.P25_PHASE1,
        sample_rate=sample_rate,
        modulation=P25Modulation.C4FM,
    )

    results = monitor.process_iq(iq)
    stats = monitor.get_stats()

    logger.info(f"\n=== Decoder Test: {name} ===")
    logger.info(f"  Sync state: {stats['sync_state']}")
    logger.info(f"  Frames decoded: {stats['frames_decoded']}")
    logger.info(f"  TSBK attempts: {stats['tsbk_attempts']}")
    logger.info(f"  TSBK CRC pass: {stats['tsbk_crc_pass']}")
    logger.info(f"  TSBK CRC pass rate: {stats['tsbk_crc_pass_rate']:.1f}%")
    logger.info(f"  Sync losses: {stats['sync_losses']}")
    logger.info(f"  Parsed results: {len(results)}")

    return stats


def compare_recordings(file1: str, file2: str) -> None:
    """Compare two baseband recordings."""
    # Load both
    iq1, sr1 = load_baseband_wav(file1)
    iq2, sr2 = load_baseband_wav(file2)

    if sr1 != sr2:
        logger.warning(f"Sample rates differ: {sr1} vs {sr2}")

    # Truncate to same length
    min_len = min(len(iq1), len(iq2))
    iq1 = iq1[:min_len]
    iq2 = iq2[:min_len]
    logger.info(f"\nComparing first {min_len} samples ({min_len/sr1:.2f} sec)")

    # Basic stats
    stats1 = analyze_iq_statistics(iq1, Path(file1).name)
    stats2 = analyze_iq_statistics(iq2, Path(file2).name)
    print_iq_stats(stats1)
    print_iq_stats(stats2)

    # Spectrum analysis
    analyze_spectrum(iq1, sr1, Path(file1).name)
    analyze_spectrum(iq2, sr2, Path(file2).name)

    # FM demod analysis
    fm1 = analyze_fm_demod(iq1, sr1, Path(file1).name)
    fm2 = analyze_fm_demod(iq2, sr2, Path(file2).name)

    # Cross-correlation of FM signals
    if len(fm1) == len(fm2):
        corr = np.corrcoef(fm1, fm2)[0, 1]
        logger.info(f"\n=== FM Cross-correlation ===")
        logger.info(f"  Correlation coefficient: {corr:.6f}")

        # Time alignment check
        cross_corr = np.correlate(fm1[:10000], fm2[:10000], mode='full')
        lag = np.argmax(cross_corr) - len(fm1[:10000]) + 1
        logger.info(f"  Time lag (samples): {lag}")

    # Sync detection
    analyze_sync_detection(iq1, sr1, Path(file1).name)
    analyze_sync_detection(iq2, sr2, Path(file2).name)

    # Decoder tests
    logger.info("\n" + "="*60)
    logger.info("DECODER COMPARISON")
    logger.info("="*60)
    stats1 = run_decoder_test(iq1, sr1, Path(file1).name)
    stats2 = run_decoder_test(iq2, sr2, Path(file2).name)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"  {Path(file1).name}: {stats1['tsbk_crc_pass_rate']:.1f}% CRC")
    logger.info(f"  {Path(file2).name}: {stats2['tsbk_crc_pass_rate']:.1f}% CRC")


def analyze_single(filepath: str) -> None:
    """Analyze a single baseband recording in detail."""
    iq, sample_rate = load_baseband_wav(filepath)

    stats = analyze_iq_statistics(iq, Path(filepath).name)
    print_iq_stats(stats)

    analyze_spectrum(iq, sample_rate, Path(filepath).name)
    analyze_fm_demod(iq, sample_rate, Path(filepath).name)
    analyze_sync_detection(iq, sample_rate, Path(filepath).name)
    run_decoder_test(iq, sample_rate, Path(filepath).name)


def main():
    parser = argparse.ArgumentParser(description="Compare baseband recordings")
    parser.add_argument("file1", nargs="?", help="First WAV file (WaveCap recording)")
    parser.add_argument("file2", nargs="?", help="Second WAV file (SDRTrunk recording)")
    parser.add_argument("--analyze", type=str, help="Analyze a single recording in detail")

    args = parser.parse_args()

    if args.analyze:
        analyze_single(args.analyze)
    elif args.file1 and args.file2:
        compare_recordings(args.file1, args.file2)
    elif args.file1:
        analyze_single(args.file1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
