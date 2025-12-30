#!/usr/bin/env python3
"""Trace exact NID values through the demodulator."""

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
    print("NID Trace Through Actual Demodulator")
    print("=" * 70)

    iq, sample_rate = load_baseband(filepath, int(5 * 50000))  # 5 seconds
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Total samples: {len(iq)}")

    from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator

    # Process with 100ms chunks (realistic)
    demod = C4FMDemodulator(sample_rate=sample_rate)
    chunk_samples = int(sample_rate * 0.1)  # 100ms

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

    print(f"\nTotal dibits: {len(dibits)}")
    print(f"Total syncs detected: {demod._sync_count}")

    # Find sync patterns in the output
    print("\n" + "=" * 70)
    print("Finding sync patterns and analyzing NID")
    print("=" * 70)

    expected_sync = np.array([3, 3, 3, 3, 3, -3, 3, 3, -3, -3, 3, 3, -3, -3, -3, -3, 3, -3, 3, -3, -3, -3, -3, -3], dtype=np.float32)

    # Find all syncs
    sync_positions = []
    for i in range(len(soft) - 24):
        window = soft[i:i+24]
        corr = np.sum(window * expected_sync)
        if corr > 200:
            # Check it's not too close to a previous sync
            if not sync_positions or i - sync_positions[-1] > 100:
                sync_positions.append(i)

    print(f"Found {len(sync_positions)} sync patterns")

    # Analyze first 5 syncs
    expected_nac = [0, 3, 3, 1, 3, 0]  # 0x3DC
    expected_duid = [0, 1]  # DUID 7 for TSDU

    for sync_idx, sync_pos in enumerate(sync_positions[:5]):
        nid_start = sync_pos + 24  # NID starts after sync

        print(f"\n--- Sync #{sync_idx+1} at dibit position {sync_pos} ---")

        # Verify sync
        sync_dibits = dibits[sync_pos:sync_pos+24]
        sync_soft = soft[sync_pos:sync_pos+24]
        sync_corr = np.sum(sync_soft * expected_sync)
        print(f"Sync correlation: {sync_corr:.1f}")

        # Show first 8 NID symbols (NAC + partial DUID)
        print(f"\n{'Pos':>4} {'Dibit':>6} {'Expect':>7} {'Soft':>8} {'ExpSoft':>8} {'Delta':>8} {'Match':>6}")
        print("-" * 60)

        errors = 0
        for j in range(8):
            pos = nid_start + j
            if pos >= len(dibits):
                break

            dibit = dibits[pos]
            s = soft[pos]
            exp_dibit = (expected_nac + expected_duid)[j]

            # Expected soft values: dibit -> soft
            # 0 -> +1, 1 -> +3, 2 -> -1, 3 -> -3
            exp_soft = {0: 1.0, 1: 3.0, 2: -1.0, 3: -3.0}[exp_dibit]
            delta = s - exp_soft
            match = "ok" if dibit == exp_dibit else "ERR"
            if dibit != exp_dibit:
                errors += 1

            print(f"{j:>4} {dibit:>6} {exp_dibit:>7} {s:>+8.2f} {exp_soft:>+8.1f} {delta:>+8.2f} {match:>6}")

        # Compute NAC from dibits
        nac_dibits = [dibits[nid_start + k] for k in range(6) if nid_start + k < len(dibits)]
        if len(nac_dibits) == 6:
            nac = sum(d << (10 - 2*k) for k, d in enumerate(nac_dibits))
            print(f"\nDecoded NAC: 0x{nac:03x} (expected 0x3dc)")
            print(f"Errors: {errors}/8")

    # Specific analysis of position 4 across all syncs
    print("\n" + "=" * 70)
    print("Position 4 (NAC[4]) analysis across all syncs")
    print("=" * 70)

    pos4_dibits = []
    pos4_soft = []
    for sync_pos in sync_positions:
        nid_start = sync_pos + 24
        pos4 = nid_start + 4
        if pos4 < len(dibits):
            pos4_dibits.append(dibits[pos4])
            pos4_soft.append(soft[pos4])

    if pos4_dibits:
        print(f"Expected: dibit=3, soft=-3.0")
        print(f"\nDibit distribution at position 4:")
        for d in range(4):
            count = sum(1 for x in pos4_dibits if x == d)
            pct = 100 * count / len(pos4_dibits) if pos4_dibits else 0
            print(f"  dibit {d}: {count} ({pct:.1f}%)")

        soft_arr = np.array(pos4_soft)
        print(f"\nSoft value statistics at position 4:")
        print(f"  mean: {soft_arr.mean():+.3f}")
        print(f"  std:  {soft_arr.std():.3f}")
        print(f"  min:  {soft_arr.min():+.3f}")
        print(f"  max:  {soft_arr.max():+.3f}")

        # Check polarity
        if soft_arr.mean() > 0:
            print(f"\n*** POLARITY ERROR: Mean soft is POSITIVE but should be NEGATIVE (-3.0) ***")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("/Users/thw/SDRTrunk/recordings/20251227_121743_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav")
