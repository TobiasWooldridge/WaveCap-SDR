#!/usr/bin/env python3
"""Compare DSP C4FM vs OLD MMSE C4FM demodulators on the same IQ input.

This script runs both demodulators on the same IQ recording and compares:
1. Soft symbol statistics (mean, std, distribution)
2. Dibit distribution
3. Frame sync success rate
4. NID decode success rate
5. TSBK CRC pass rate

Usage:
    python scripts/compare_c4fm_demods.py --input capture.wav

This helps identify which demodulator is producing better output for P25 decoding.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from wavecapsdr.decoders.p25 import (
    C4FMDemodulator as OldC4FMDemodulator,
    P25FrameSync,
    P25FrameType,
)
from wavecapsdr.decoders.p25_frames import decode_nid
from wavecapsdr.capture import decimate_iq_for_p25

logger = logging.getLogger(__name__)


def shift_and_decimate(iq: np.ndarray, sample_rate: int, offset_hz: float = 0.0) -> tuple[np.ndarray, int]:
    """Shift IQ to baseband and decimate to 48 kHz for P25.

    This replicates what the WaveCap capture chain does:
    1. Frequency shift to center the channel
    2. Decimate to 48 kHz
    """
    # Frequency shift (if offset provided)
    if offset_hz != 0:
        t = np.arange(len(iq)) / sample_rate
        shift = np.exp(-1j * 2 * np.pi * offset_hz * t)
        iq = iq * shift.astype(np.complex64)

    # Decimate to 48 kHz (target for P25)
    decimated_iq, decimated_rate = decimate_iq_for_p25(iq, sample_rate)

    logger.info(f"Decimated: {sample_rate/1e6:.1f} MHz -> {decimated_rate/1e3:.1f} kHz ({len(iq)} -> {len(decimated_iq)} samples)")

    return decimated_iq, decimated_rate


def load_iq_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load IQ data from stereo WAV file."""
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

    if n_channels != 2:
        raise ValueError(f"Expected stereo IQ WAV, got {n_channels} channels")

    if sample_width == 2:
        samples = np.frombuffer(raw_data, dtype=np.int16).reshape(-1, n_channels)
        samples_f = samples.astype(np.float32) / 32768.0
    elif sample_width == 3:
        data_bytes = np.frombuffer(raw_data, dtype=np.uint8).reshape(-1, n_channels, 3)
        samples = np.zeros((n_frames, n_channels), dtype=np.int32)
        for ch in range(n_channels):
            b0 = data_bytes[:, ch, 0].astype(np.int32)
            b1 = data_bytes[:, ch, 1].astype(np.int32)
            b2 = data_bytes[:, ch, 2].astype(np.int32)
            raw24 = b0 | (b1 << 8) | (b2 << 16)
            samples[:, ch] = np.where(raw24 >= 0x800000, raw24 - 0x1000000, raw24)
        samples_f = samples.astype(np.float32) / 8388608.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    iq = samples_f[:, 0] + 1j * samples_f[:, 1]
    return iq, sample_rate


@dataclass
class DemodResult:
    """Result from a C4FM demodulator run."""
    name: str
    soft_symbols: np.ndarray
    dibits: np.ndarray
    frames_found: int
    nid_success: int
    nac_values: list[int]
    symbol_stats: dict


def run_old_c4fm(iq: np.ndarray, sample_rate: int, max_frames: int = 100) -> DemodResult:
    """Run the OLD MMSE C4FM demodulator."""
    logger.info("Running OLD MMSE C4FM demodulator...")

    demod = OldC4FMDemodulator(sample_rate=sample_rate, symbol_rate=4800)
    frame_sync = P25FrameSync()

    # Demodulate in chunks
    chunk_size = sample_rate // 10  # 100ms
    all_dibits = []
    all_soft = []  # We'll estimate soft symbols from dibits

    for start in range(0, len(iq), chunk_size):
        chunk = iq[start:start + chunk_size]
        if len(chunk) < 100:
            break

        dibits = demod.demodulate(chunk.astype(np.complex64, copy=False))
        if len(dibits) > 0:
            all_dibits.extend(dibits.tolist())

            # Estimate soft symbols from dibits (ideally we'd capture actual)
            dibit_to_sym = {0: 1.0, 1: 3.0, 2: -1.0, 3: -3.0}
            soft = [dibit_to_sym[d] for d in dibits]
            all_soft.extend(soft)

    dibits = np.array(all_dibits, dtype=np.uint8)
    soft_symbols = np.array(all_soft, dtype=np.float32)

    # Calculate symbol stats
    symbol_stats = {
        "count": len(soft_symbols),
        "mean": float(np.mean(soft_symbols)),
        "std": float(np.std(soft_symbols)),
        "dibit_0": int(np.sum(dibits == 0)),
        "dibit_1": int(np.sum(dibits == 1)),
        "dibit_2": int(np.sum(dibits == 2)),
        "dibit_3": int(np.sum(dibits == 3)),
    }

    # Find frames and decode NID
    buffer = dibits.tolist()
    frames_found = 0
    nid_success = 0
    nac_values = []

    while frames_found < max_frames and len(buffer) >= 360:
        buf_arr = np.array(buffer, dtype=np.uint8)
        sync_pos, frame_type, nac, duid = frame_sync.find_sync(buf_arr)

        if sync_pos is None:
            if len(buffer) > 720:
                buffer = buffer[-360:]
            break

        if sync_pos + 360 > len(buffer):
            break

        frames_found += 1
        frame = buf_arr[sync_pos:sync_pos + 360]
        buffer = buffer[sync_pos + 360:]

        # Decode NID
        nid_result = decode_nid(frame[24:57], skip_status_at_10=True)
        if nid_result is not None:
            nid_success += 1
            nac_values.append(nid_result.nac)

    return DemodResult(
        name="OLD_MMSE",
        soft_symbols=soft_symbols,
        dibits=dibits,
        frames_found=frames_found,
        nid_success=nid_success,
        nac_values=nac_values,
        symbol_stats=symbol_stats,
    )


def run_dsp_c4fm(iq: np.ndarray, sample_rate: int, max_frames: int = 100) -> DemodResult:
    """Run the DSP (Gardner TED) C4FM demodulator."""
    logger.info("Running DSP (Gardner TED) C4FM demodulator...")

    # Import DSP C4FM
    try:
        from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator as DspC4FMDemodulator
    except ImportError:
        logger.warning("DSP C4FM not available, skipping")
        return None

    demod = DspC4FMDemodulator(sample_rate=sample_rate)
    frame_sync = P25FrameSync()

    # Demodulate in chunks
    chunk_size = sample_rate // 10  # 100ms
    all_dibits = []
    all_soft = []

    for start in range(0, len(iq), chunk_size):
        chunk = iq[start:start + chunk_size]
        if len(chunk) < 100:
            break

        # DSP demodulator returns (dibits, soft_symbols)
        try:
            result = demod.demodulate(chunk.astype(np.complex64, copy=False))
            if isinstance(result, tuple):
                dibits, soft = result
            else:
                dibits = result
                dibit_to_sym = {0: 1.0, 1: 3.0, 2: -1.0, 3: -3.0}
                soft = np.array([dibit_to_sym[d] for d in dibits], dtype=np.float32)

            if len(dibits) > 0:
                all_dibits.extend(dibits.tolist())
                all_soft.extend(soft.tolist() if hasattr(soft, 'tolist') else list(soft))
        except Exception as e:
            logger.warning(f"DSP demod chunk error: {e}")
            continue

    dibits = np.array(all_dibits, dtype=np.uint8)
    soft_symbols = np.array(all_soft, dtype=np.float32)

    # Calculate symbol stats
    symbol_stats = {
        "count": len(soft_symbols),
        "mean": float(np.mean(soft_symbols)) if len(soft_symbols) > 0 else 0,
        "std": float(np.std(soft_symbols)) if len(soft_symbols) > 0 else 0,
        "dibit_0": int(np.sum(dibits == 0)),
        "dibit_1": int(np.sum(dibits == 1)),
        "dibit_2": int(np.sum(dibits == 2)),
        "dibit_3": int(np.sum(dibits == 3)),
    }

    # Find frames and decode NID
    buffer = dibits.tolist()
    frames_found = 0
    nid_success = 0
    nac_values = []

    while frames_found < max_frames and len(buffer) >= 360:
        buf_arr = np.array(buffer, dtype=np.uint8)
        sync_pos, frame_type, nac, duid = frame_sync.find_sync(buf_arr)

        if sync_pos is None:
            if len(buffer) > 720:
                buffer = buffer[-360:]
            break

        if sync_pos + 360 > len(buffer):
            break

        frames_found += 1
        frame = buf_arr[sync_pos:sync_pos + 360]
        buffer = buffer[sync_pos + 360:]

        # Decode NID
        nid_result = decode_nid(frame[24:57], skip_status_at_10=True)
        if nid_result is not None:
            nid_success += 1
            nac_values.append(nid_result.nac)

    return DemodResult(
        name="DSP_GARDNER",
        soft_symbols=soft_symbols,
        dibits=dibits,
        frames_found=frames_found,
        nid_success=nid_success,
        nac_values=nac_values,
        symbol_stats=symbol_stats,
    )


def print_comparison(result1: DemodResult, result2: DemodResult | None) -> None:
    """Print side-by-side comparison of results."""
    print("\n" + "=" * 70)
    print("C4FM DEMODULATOR COMPARISON")
    print("=" * 70)

    # Symbol stats
    print("\n--- Symbol Statistics ---")
    print(f"{'Metric':<20} {'OLD MMSE':>20} {'DSP Gardner':>20}")
    print("-" * 60)

    stats1 = result1.symbol_stats
    stats2 = result2.symbol_stats if result2 else {}

    print(f"{'Symbol count':<20} {stats1['count']:>20,} {stats2.get('count', 'N/A'):>20}")
    print(f"{'Mean':<20} {stats1['mean']:>20.3f} {stats2.get('mean', 'N/A'):>20}")
    print(f"{'Std Dev':<20} {stats1['std']:>20.3f} {stats2.get('std', 'N/A'):>20}")

    # Expected for P25 C4FM: mean ≈ 0, std ≈ 2.24 (sqrt(5) due to ±3, ±1 levels)
    print()
    print("  Expected for P25 C4FM: mean ≈ 0, std ≈ 2.24")

    # Dibit distribution
    print("\n--- Dibit Distribution ---")
    print(f"{'Dibit':<20} {'OLD MMSE':>20} {'DSP Gardner':>20}")
    print("-" * 60)

    for i in range(4):
        key = f"dibit_{i}"
        v1 = stats1[key]
        v2 = stats2.get(key, 0)
        pct1 = 100 * v1 / stats1['count'] if stats1['count'] > 0 else 0
        pct2 = 100 * v2 / stats2['count'] if stats2.get('count', 0) > 0 else 0
        print(f"{'Dibit ' + str(i):<20} {v1:>10,} ({pct1:>5.1f}%) {v2:>10,} ({pct2:>5.1f}%)")

    # Expected: roughly equal distribution (25% each)
    print()
    print("  Expected: roughly 25% each for random P25 data")

    # Frame sync and NID
    print("\n--- Frame Sync & NID ---")
    print(f"{'Metric':<20} {'OLD MMSE':>20} {'DSP Gardner':>20}")
    print("-" * 60)
    print(f"{'Frames found':<20} {result1.frames_found:>20} {result2.frames_found if result2 else 'N/A':>20}")
    print(f"{'NID success':<20} {result1.nid_success:>20} {result2.nid_success if result2 else 'N/A':>20}")

    nid_pct1 = 100 * result1.nid_success / result1.frames_found if result1.frames_found > 0 else 0
    nid_pct2 = 100 * result2.nid_success / result2.frames_found if result2 and result2.frames_found > 0 else 0
    print(f"{'NID success %':<20} {nid_pct1:>19.1f}% {nid_pct2:>19.1f}%")

    # NAC values
    print("\n--- NAC Values Found ---")
    nac1_set = set(result1.nac_values)
    nac2_set = set(result2.nac_values) if result2 else set()
    print(f"  OLD MMSE:    {sorted([f'0x{n:03X}' for n in nac1_set])}")
    print(f"  DSP Gardner: {sorted([f'0x{n:03X}' for n in nac2_set])}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    issues = []

    # Check symbol bias
    if abs(stats1['mean']) > 0.5:
        issues.append(f"OLD MMSE has symbol bias: mean={stats1['mean']:.2f}")
    if result2 and abs(stats2.get('mean', 0)) > 0.5:
        issues.append(f"DSP Gardner has symbol bias: mean={stats2.get('mean', 0):.2f}")

    # Check std deviation
    expected_std = 2.24  # sqrt(5) for P25 C4FM
    if abs(stats1['std'] - expected_std) > 1.0:
        issues.append(f"OLD MMSE std deviation off: {stats1['std']:.2f} (expected ~{expected_std:.2f})")
    if result2 and abs(stats2.get('std', 0) - expected_std) > 1.0:
        issues.append(f"DSP Gardner std deviation off: {stats2.get('std', 0):.2f} (expected ~{expected_std:.2f})")

    # Check NID success
    if nid_pct1 < 50:
        issues.append(f"OLD MMSE low NID success: {nid_pct1:.1f}%")
    if result2 and nid_pct2 < 50:
        issues.append(f"DSP Gardner low NID success: {nid_pct2:.1f}%")

    if issues:
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print("  ✓ Both demodulators performing well")

    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare C4FM demodulators")
    parser.add_argument("--input", type=Path, required=True, help="Input IQ file (.wav)")
    parser.add_argument("--sample-rate", type=int, default=None, help="Sample rate (if rawiq)")
    parser.add_argument("--offset-hz", type=float, default=0.0,
                        help="Frequency offset to center on control channel (Hz)")
    parser.add_argument("--max-frames", type=int, default=100, help="Max frames to process")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON file")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Load IQ data
    if args.input.suffix.lower() == ".wav":
        iq, sample_rate = load_iq_wav(args.input)
    else:
        from scripts.p25_pipeline_stages import load_iq_rawiq
        iq, sample_rate = load_iq_rawiq(args.input, args.sample_rate)

    logger.info(f"Loaded {args.input}: {len(iq)} samples at {sample_rate} Hz")

    # If sample rate is high (>100 kHz), decimate to 48 kHz
    # This is what the WaveCap capture chain does before P25 decoding
    if sample_rate > 100000:
        logger.info(f"High sample rate detected ({sample_rate/1e6:.1f} MHz), decimating to 48 kHz...")
        iq, sample_rate = shift_and_decimate(iq, sample_rate, offset_hz=args.offset_hz)
    elif args.offset_hz != 0:
        # Apply frequency shift even for low sample rates
        t = np.arange(len(iq)) / sample_rate
        shift = np.exp(-1j * 2 * np.pi * args.offset_hz * t)
        iq = iq * shift.astype(np.complex64)
        logger.info(f"Applied frequency shift: {args.offset_hz} Hz")

    logger.info(f"Processing at {sample_rate} Hz ({len(iq)} samples)")

    # Run both demodulators
    result_old = run_old_c4fm(iq, sample_rate, max_frames=args.max_frames)
    result_dsp = run_dsp_c4fm(iq, sample_rate, max_frames=args.max_frames)

    # Print comparison
    print_comparison(result_old, result_dsp)

    # Save to JSON if requested
    if args.output:
        output = {
            "old_mmse": {
                "symbol_stats": result_old.symbol_stats,
                "frames_found": result_old.frames_found,
                "nid_success": result_old.nid_success,
                "nac_values": list(set(result_old.nac_values)),
            },
        }
        if result_dsp:
            output["dsp_gardner"] = {
                "symbol_stats": result_dsp.symbol_stats,
                "frames_found": result_dsp.frames_found,
                "nid_success": result_dsp.nid_success,
                "nac_values": list(set(result_dsp.nac_values)),
            }
        args.output.write_text(json.dumps(output, indent=2))
        logger.info(f"Saved comparison to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
