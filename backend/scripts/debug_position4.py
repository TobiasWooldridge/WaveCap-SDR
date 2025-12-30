#!/usr/bin/env python3
"""Deep dive into position 4 error."""

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
    print("Position 4 Deep Dive Analysis")
    print("=" * 70)

    iq, sample_rate = load_baseband(filepath, int(3 * 50000))  # 3 seconds
    print(f"Sample rate: {sample_rate} Hz")

    from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator

    demod = C4FMDemodulator(sample_rate=sample_rate)
    sps = demod.samples_per_symbol
    print(f"Samples per symbol: {sps:.3f}")

    # Process with 100ms chunks
    chunk_samples = int(sample_rate * 0.1)
    all_dibits = []
    all_soft = []

    for start in range(0, len(iq), chunk_samples):
        end = min(start + chunk_samples, len(iq))
        chunk = iq[start:end]
        dibits, soft = demod.demodulate(chunk)
        all_dibits.extend(dibits)
        all_soft.extend(soft)

    dibits = np.array(all_dibits, dtype=np.uint8)
    soft = np.array(all_soft, dtype=np.float32)

    print(f"\nTotal syncs: {demod._sync_count}")

    # Find sync patterns
    expected_sync = np.array([3, 3, 3, 3, 3, -3, 3, 3, -3, -3, 3, 3, -3, -3, -3, -3, 3, -3, 3, -3, -3, -3, -3, -3], dtype=np.float32)

    print("\nAnalyzing ALL positions for each sync:")
    print("-" * 70)

    # Find first 3 syncs
    sync_count = 0
    i = 0
    while i < len(soft) - 57 and sync_count < 3:
        window = soft[i:i+24]
        corr = np.sum(window * expected_sync)
        if corr > 200:
            nid_start = i + 24

            print(f"\n=== Sync at position {i}, correlation {corr:.1f} ===")

            # Expected values for SA-GRN
            # NAC 0x3DC = 001111011100
            # Breaking into dibits (2 bits each, MSB first):
            # 00=0, 11=3, 11=3, 01=1, 11=3, 00=0
            expected_nac = [0, 3, 3, 1, 3, 0]

            # DUID 7 = 0111 = 01|11 = dibits [1, 3]
            expected_duid = [1, 3]

            expected_all = expected_nac + expected_duid

            print(f"\n{'Pos':>4} {'Dibit':>6} {'Expect':>7} {'Soft':>8} {'ExpSoft':>8} {'Match':>6}")
            print("-" * 55)

            for j in range(8):
                pos = nid_start + j
                if pos < len(dibits):
                    dibit = dibits[pos]
                    s = soft[pos]
                    exp = expected_all[j]
                    exp_soft = {0: 1.0, 1: 3.0, 2: -1.0, 3: -3.0}[exp]
                    match = "ok" if dibit == exp else "ERR"
                    marker = " <--" if j == 4 else ""
                    print(f"{j:>4} {dibit:>6} {exp:>7} {s:>+8.2f} {exp_soft:>+8.1f} {match:>6}{marker}")

            # Now look at raw soft values AROUND position 4
            print(f"\n--- Soft values around position 4 (NID[4]) ---")
            nid4_pos = nid_start + 4
            for offset in range(-3, 4):
                p = nid4_pos + offset
                if 0 <= p < len(soft):
                    s = soft[p]
                    marker = " <-- POSITION 4" if offset == 0 else ""
                    print(f"  pos {p} (offset {offset:+d}): soft = {s:+.3f}{marker}")

            sync_count += 1
            i += 100  # Skip ahead to avoid duplicate detections
        else:
            i += 1

    # Statistical analysis of position 4 across all syncs
    print("\n" + "=" * 70)
    print("Statistical analysis of NID positions across all syncs")
    print("=" * 70)

    sync_positions = []
    i = 0
    while i < len(soft) - 57:
        window = soft[i:i+24]
        corr = np.sum(window * expected_sync)
        if corr > 200:
            sync_positions.append(i)
            i += 100
        else:
            i += 1

    print(f"Found {len(sync_positions)} syncs")

    # Collect soft values at each NID position
    for nid_pos in range(8):
        soft_values = []
        for sync_pos in sync_positions:
            p = sync_pos + 24 + nid_pos
            if p < len(soft):
                soft_values.append(soft[p])

        if soft_values:
            arr = np.array(soft_values)
            expected = expected_all[nid_pos]
            exp_soft = {0: 1.0, 1: 3.0, 2: -1.0, 3: -3.0}[expected]
            correct = sum(1 for s in arr if (s >= 0) == (exp_soft >= 0))  # Same sign
            print(f"  Position {nid_pos}: mean={arr.mean():+.3f}, std={arr.std():.3f}, expected={exp_soft:+.1f}, sign_match={100*correct/len(arr):.0f}%")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("/Users/thw/SDRTrunk/recordings/20251227_121743_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav")
