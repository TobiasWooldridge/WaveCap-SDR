#!/usr/bin/env python3
"""Use exact soft symbols from demodulator and trace sync."""

import sys
import wave
import numpy as np

sys.path.insert(0, '/Users/thw/Projects/WaveCap-SDR/backend')


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
    print("Exact Soft Symbol Sync Detection")
    print("=" * 70)

    iq, sample_rate = load_baseband(filepath, int(10 * 50000))
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Samples: {len(iq)}")

    from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator, _SoftSyncDetector

    # Create demodulator and run
    demod = C4FMDemodulator(sample_rate=sample_rate)
    dibits, soft_symbols = demod.demodulate(iq)

    print(f"\nDemodulator output:")
    print(f"  Dibits: {len(dibits)}")
    print(f"  Soft symbols: {len(soft_symbols)}")
    print(f"  Soft symbol range: [{soft_symbols.min():.2f}, {soft_symbols.max():.2f}]")
    print(f"  sync_count: {demod._sync_count}")

    # Now trace with FRESH detector on THESE soft symbols
    print("\n" + "=" * 70)
    print("Fresh Detector Trace on Demodulator Soft Symbols")
    print("=" * 70)

    detector = _SoftSyncDetector()
    THRESHOLD = 130.0

    # Find all positions where score >= 200
    high_score_positions = []
    for i, sym in enumerate(soft_symbols):
        score = detector.process(sym)
        if score >= 200:
            high_score_positions.append((i, score))

    print(f"\nPositions with score >= 200: {len(high_score_positions)}")
    for pos, score in high_score_positions[:10]:
        print(f"  Position {pos}: score={score:.1f}")

    # Now check: at these positions, what does the demodulator's internal detector see?
    print("\n" + "=" * 70)
    print("Comparing with Demodulator's Sync Detector")
    print("=" * 70)

    print(f"\nDemodulator sync detector state after demodulate():")
    print(f"  pointer: {demod._sync_detector._pointer}")
    print(f"  buffer[0:10]: {demod._sync_detector._buffer[:10]}")

    # The demodulator's sync detector has processed all symbols too.
    # Let's see if its state matches the fresh detector at the same point.
    fresh_detector = _SoftSyncDetector()
    for sym in soft_symbols:
        fresh_detector.process(sym)

    print(f"\nFresh detector state after processing same symbols:")
    print(f"  pointer: {fresh_detector._pointer}")
    print(f"  buffer[0:10]: {fresh_detector._buffer[:10]}")

    # Compare scores at known positions
    print("\n" + "=" * 70)
    print("Score Comparison at Known Positions")
    print("=" * 70)

    test_positions = [19603, 20467, 21331, 22195]

    for test_pos in test_positions:
        # Fresh detector
        fresh = _SoftSyncDetector()
        for i in range(test_pos + 1):
            fresh_score = fresh.process(soft_symbols[i])

        print(f"\nPosition {test_pos}:")
        print(f"  Fresh detector score: {fresh_score:.1f}")

        # Also check what the 24 symbols at this position look like
        window = soft_symbols[test_pos-23:test_pos+1]
        if len(window) == 24:
            expected_sync = np.array([3, 3, 3, 3, 3, -3, 3, 3, -3, -3, 3, 3, -3, -3, -3, -3, 3, -3, 3, -3, -3, -3, -3, -3], dtype=np.float32)
            direct_corr = np.sum(window * expected_sync)
            print(f"  Direct correlation: {direct_corr:.1f}")
            print(f"  Window: {window[:8].round(1)} ... {window[-4:].round(1)}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("/Users/thw/SDRTrunk/recordings/20251227_121743_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav")
