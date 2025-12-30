#!/usr/bin/env python3
"""Test with realistic chunk sizes."""

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
    print("Chunked Processing Test")
    print("=" * 70)

    iq_full, sample_rate = load_baseband(filepath)
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Total samples: {len(iq_full)}")
    print(f"Duration: {len(iq_full)/sample_rate:.1f}s")

    from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator

    # Test with different chunk sizes
    chunk_sizes_ms = [50, 100, 200, 500, 1000]

    for chunk_ms in chunk_sizes_ms:
        chunk_samples = int(sample_rate * chunk_ms / 1000)

        # Create fresh demodulator
        demod = C4FMDemodulator(sample_rate=sample_rate)

        # Process in chunks
        total_dibits = 0
        all_soft = []

        for start in range(0, len(iq_full), chunk_samples):
            end = min(start + chunk_samples, len(iq_full))
            chunk = iq_full[start:end]

            dibits, soft = demod.demodulate(chunk)
            total_dibits += len(dibits)
            all_soft.extend(soft)

        print(f"\nChunk size: {chunk_ms}ms ({chunk_samples} samples)")
        print(f"  Total dibits: {total_dibits}")
        print(f"  sync_count: {demod._sync_count}")

        # Check if syncs were found
        if demod._sync_count > 0:
            print(f"  SUCCESS! Found {demod._sync_count} syncs")
        else:
            # Check manually for sync patterns
            soft_arr = np.array(all_soft)
            expected_sync = np.array([3, 3, 3, 3, 3, -3, 3, 3, -3, -3, 3, 3, -3, -3, -3, -3, 3, -3, 3, -3, -3, -3, -3, -3], dtype=np.float32)

            high_scores = 0
            for i in range(len(soft_arr) - 24):
                window = soft_arr[i:i+24]
                corr = np.sum(window * expected_sync)
                if corr > 200:
                    high_scores += 1

            print(f"  No syncs detected, but {high_scores} sync patterns exist in data")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("/Users/thw/SDRTrunk/recordings/20251227_121743_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav")
