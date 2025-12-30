#!/usr/bin/env python3
"""Track actual sync positions per chunk."""

import sys
import wave
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

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
    print("Sync Position Tracking")
    print("=" * 70)

    iq, sample_rate = load_baseband(filepath, int(5 * 50000))  # 5 seconds
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Samples: {len(iq)}")

    from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator

    # Process with chunked approach (100ms chunks)
    demod = C4FMDemodulator(sample_rate=sample_rate)
    chunk_samples = int(sample_rate * 0.1)  # 100ms

    all_dibits = []
    all_soft = []
    total_syncs = 0

    for chunk_idx, start in enumerate(range(0, len(iq), chunk_samples)):
        end = min(start + chunk_samples, len(iq))
        chunk = iq[start:end]

        before_syncs = demod._sync_count
        dibits, soft = demod.demodulate(chunk)
        after_syncs = demod._sync_count

        new_syncs = after_syncs - before_syncs
        if new_syncs > 0:
            print(f"Chunk {chunk_idx} (samples {start}-{end}): {new_syncs} new sync(s)")

        all_dibits.extend(dibits)
        all_soft.extend(soft)
        total_syncs = after_syncs

    dibits = np.array(all_dibits, dtype=np.uint8)
    soft = np.array(all_soft, dtype=np.float32)

    print(f"\nTotal syncs: {total_syncs}")
    print(f"Total dibits: {len(dibits)}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("/Users/thw/SDRTrunk/recordings/20251227_121743_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav")
