#!/usr/bin/env python3
"""Verify the buffer size fix by checking symbol_indices from C4FMDemodulator."""

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
    print("C4FM Demodulator Fix Verification")
    print("=" * 70)

    iq, sample_rate = load_baseband(filepath, int(10 * 50000))
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Samples: {len(iq)}")

    from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator

    # Create demodulator - should use the new 65536 buffer
    demod = C4FMDemodulator(sample_rate=sample_rate)
    print(f"Buffer size: {len(demod._buffer)}")

    # Check sync count BEFORE demodulation
    print(f"\nBefore demodulate:")
    print(f"  sync_count: {demod._sync_count}")
    print(f"  buffer_pointer: {demod._buffer_pointer}")

    # Run demodulation
    print("\nRunning demodulate()...")
    dibits, soft_symbols = demod.demodulate(iq)

    print(f"\nAfter demodulate:")
    print(f"  Dibits: {len(dibits)}")
    print(f"  Soft symbols: {len(soft_symbols)}")
    print(f"  sync_count: {demod._sync_count}")
    print(f"  buffer_pointer: {demod._buffer_pointer}")

    # With 65536 buffer and 500000 samples:
    # - Buffer should shift about (500000 - 65536) / 32768 = 13 times
    # - Valid indices: last ~32768 samples = ~3140 symbols
    # This is much better than before!

    samples_per_symbol = sample_rate / 4800.0
    buffer_half = len(demod._buffer) // 2
    expected_valid_symbols = int(buffer_half / samples_per_symbol)
    print(f"\nExpected valid symbols (after shifts): ~{expected_valid_symbols}")

    # Look for sync patterns in the soft symbols
    print("\n" + "=" * 70)
    print("Sync Pattern Search")
    print("=" * 70)

    expected_sync = np.array([3, 3, 3, 3, 3, -3, 3, 3, -3, -3, 3, 3, -3, -3, -3, -3, 3, -3, 3, -3, -3, -3, -3, -3], dtype=np.float32)

    best_matches = []
    for i in range(len(soft_symbols) - 24):
        window = soft_symbols[i:i+24]
        corr = np.sum(window * expected_sync)
        if corr > 200:
            best_matches.append((i, corr))

    print(f"Positions with correlation > 200: {len(best_matches)}")
    for pos, corr in best_matches[:5]:
        print(f"  Position {pos}: correlation={corr:.1f}")

    if demod._sync_count == 0 and len(best_matches) > 0:
        print("\nWARNING: Sync patterns found but sync_count is 0!")
        print("         This means sync detection is still not triggering.")
    elif demod._sync_count > 0:
        print(f"\nSUCCESS: sync_count = {demod._sync_count}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("/Users/thw/SDRTrunk/recordings/20251227_121743_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav")
