#!/usr/bin/env python3
"""Debug sync detection loop by manually stepping through what demodulate() does."""

import sys
import wave
import numpy as np

sys.path.insert(0, '/Users/thw/Projects/WaveCap-SDR/backend')

from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator, _SoftSyncDetector


def load_baseband(filepath: str, max_samples: int = None) -> tuple[np.ndarray, int]:
    with wave.open(filepath, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        if max_samples:
            n_frames = min(n_frames, max_samples)
        raw = wf.readframes(n_frames)
        data = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
        iq = (data[:, 0] + 1j * data[:, 1]).astype(np.complex64) / 32768.0
        return iq, sample_rate


def main(filepath: str):
    print("=" * 70)
    print("Sync Detection Loop Debug")
    print("=" * 70)

    iq, sample_rate = load_baseband(filepath, int(10 * 50000))
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Samples: {len(iq)}")

    # Create demodulator
    demod = C4FMDemodulator(sample_rate=sample_rate)

    # Get the threshold value
    SYNC_THRESHOLD = 130.0  # from _SoftSyncDetector.SYNC_THRESHOLD

    print(f"Sync threshold: {SYNC_THRESHOLD}")
    print(f"Expected max score (perfect match): 216")

    # Run demodulate to get soft symbols
    print("\nRunning demodulate()...")
    dibits, soft_symbols = demod.demodulate(iq)
    print(f"Dibits extracted: {len(dibits)}")
    print(f"Soft symbols: {len(soft_symbols)}")
    print(f"Soft symbol range: [{soft_symbols.min():.2f}, {soft_symbols.max():.2f}]")
    print(f"Sync count from demodulator: {demod._sync_count}")

    # Now manually replay through sync detector to see scores
    print("\n" + "=" * 70)
    print("Manual sync detection replay")
    print("=" * 70)

    detector = _SoftSyncDetector()
    high_scores = []
    all_scores = []

    for i, soft_sym in enumerate(soft_symbols):
        score = detector.process(soft_sym)
        all_scores.append(score)
        if score >= SYNC_THRESHOLD:
            high_scores.append((i, score))

    print(f"Total symbols processed: {len(soft_symbols)}")
    print(f"Scores >= {SYNC_THRESHOLD}: {len(high_scores)}")

    if high_scores:
        print("\nFirst 10 high-score positions:")
        for pos, score in high_scores[:10]:
            print(f"  Position {pos}: score={score:.1f}")

    # Check score statistics
    scores_arr = np.array(all_scores)
    print(f"\nScore statistics:")
    print(f"  Min: {scores_arr.min():.1f}")
    print(f"  Max: {scores_arr.max():.1f}")
    print(f"  Mean: {scores_arr.mean():.1f}")
    print(f"  Std: {scores_arr.std():.1f}")

    # Now let's trace what the ACTUAL sync detector does during demodulate()
    # by checking the buffer state
    print("\n" + "=" * 70)
    print("Checking internal state from demodulator")
    print("=" * 70)

    demod_detector = demod._sync_detector
    print(f"Demodulator sync detector pointer: {demod_detector._pointer}")
    print(f"Demodulator sync detector buffer (first 10): {demod_detector._buffer[:10]}")

    # The mystery: why does manual replay show high scores but demodulate() doesn't trigger?
    # Let's check if the soft_symbols from demodulate() match what we'd expect

    print("\n" + "=" * 70)
    print("Checking soft symbol normalization")
    print("=" * 70)

    # Check if symbols are in expected range
    outer_high = soft_symbols[soft_symbols > 2.5]
    outer_low = soft_symbols[soft_symbols < -2.5]
    inner_high = soft_symbols[(soft_symbols > 0.5) & (soft_symbols <= 2.5)]
    inner_low = soft_symbols[(soft_symbols < -0.5) & (soft_symbols >= -2.5)]

    print(f"Symbol distribution:")
    print(f"  Outer high (+3): {len(outer_high)} symbols, mean={outer_high.mean():.2f}" if len(outer_high) > 0 else "  Outer high: 0 symbols")
    print(f"  Outer low (-3): {len(outer_low)} symbols, mean={outer_low.mean():.2f}" if len(outer_low) > 0 else "  Outer low: 0 symbols")
    print(f"  Inner high (+1): {len(inner_high)} symbols, mean={inner_high.mean():.2f}" if len(inner_high) > 0 else "  Inner high: 0 symbols")
    print(f"  Inner low (-1): {len(inner_low)} symbols, mean={inner_low.mean():.2f}" if len(inner_low) > 0 else "  Inner low: 0 symbols")

    # CRITICAL CHECK: Are the sync patterns actually in the data?
    print("\n" + "=" * 70)
    print("Looking for sync pattern in soft symbols")
    print("=" * 70)

    # P25 sync pattern expected symbols: [+3, +3, +3, +3, +3, -3, +3, +3, -3, -3, +3, +3, -3, -3, -3, -3, +3, -3, +3, -3, -3, -3, -3, -3]
    expected_sync = np.array([3, 3, 3, 3, 3, -3, 3, 3, -3, -3, 3, 3, -3, -3, -3, -3, 3, -3, 3, -3, -3, -3, -3, -3], dtype=np.float32)

    # Scan for best match positions
    best_matches = []
    for i in range(len(soft_symbols) - 24):
        window = soft_symbols[i:i+24]
        # Compute correlation
        corr = np.sum(window * expected_sync)
        if corr > 180:  # Very high match
            best_matches.append((i, corr))

    print(f"Positions with correlation > 180: {len(best_matches)}")
    if best_matches:
        for pos, corr in best_matches[:10]:
            print(f"  Position {pos}: correlation={corr:.1f}")
            if corr > 200:
                # Show the actual symbols
                window = soft_symbols[pos:pos+24]
                print(f"    Actual symbols: {window[:8].round(1)} ...")
                print(f"    Expected:       {expected_sync[:8]} ...")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("/Users/thw/SDRTrunk/recordings/20251227_121743_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav")
