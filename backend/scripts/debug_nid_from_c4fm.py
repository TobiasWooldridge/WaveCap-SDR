#!/usr/bin/env python3
"""Analyze NID symbols directly from C4FM demodulator output.

Look at the actual soft symbol values produced by the C4FM demodulator
for NID positions to understand what's happening.
"""

import sys
import wave
import numpy as np

sys.path.insert(0, '/Users/thw/Projects/WaveCap-SDR/backend')

from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator

# Frame sync as dibit array (24 dibits)
FRAME_SYNC_DIBITS = np.array([
    1, 1, 1, 1, 1, 3, 1, 1, 3, 3, 1, 1, 3, 3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3
], dtype=np.uint8)

# Expected SA-GRN NAC dibits + DUID
EXPECTED_NAC_DIBITS = [0, 3, 3, 1, 3, 0]  # NAC = 0x3DC
EXPECTED_DUID_DIBITS = [0, 1]  # DUID 7 (TSDU) = 0111 -> dibits [0,1]


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


def find_syncs(dibits: np.ndarray, threshold: int = 24) -> list[tuple[int, int]]:
    syncs = []
    sync_len = len(FRAME_SYNC_DIBITS)
    for i in range(len(dibits) - sync_len - 40):
        matches = np.sum(dibits[i:i+sync_len] == FRAME_SYNC_DIBITS)
        if matches >= threshold:
            syncs.append((i, matches))
    return syncs


def main(filepath: str):
    print(f"\n{'='*70}")
    print("C4FM Demodulator NID Analysis")
    print(f"{'='*70}")

    iq, sample_rate = load_baseband(filepath, int(10 * 50000))
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Samples: {len(iq)}")

    # Run C4FM demodulator
    demod = C4FMDemodulator(sample_rate=sample_rate)
    dibits, soft = demod.demodulate(iq)
    print(f"Dibits extracted: {len(dibits)}")
    print(f"Soft symbol range: [{soft.min():.2f}, {soft.max():.2f}]")
    print(f"Soft symbol std: {soft.std():.2f}")

    # Find syncs
    syncs = find_syncs(dibits, threshold=24)
    print(f"Perfect syncs (24/24): {len(syncs)}")

    if not syncs:
        syncs = find_syncs(dibits, threshold=22)
        print(f"Syncs (>=22/24): {len(syncs)}")

    print(f"\n{'='*70}")
    print("NID Symbol Analysis - What C4FM Demodulator Actually Produces")
    print(f"{'='*70}")

    for sync_idx, (sync_pos, score) in enumerate(syncs[:5]):
        print(f"\n--- Frame {sync_idx + 1}: Sync at symbol {sync_pos} ({score}/24) ---")

        # Show last 4 sync symbols to verify we're at right position
        print("\nLast 4 sync symbols:")
        for i in range(20, 24):
            pos = sync_pos + i
            d = dibits[pos]
            s = soft[pos]
            exp = FRAME_SYNC_DIBITS[i]
            exp_s = {0: 1.0, 1: 3.0, 2: -1.0, 3: -3.0}[exp]
            match = "ok" if d == exp else "ERR"
            print(f"  SYNC[{i:2d}] dibit={d} soft={s:+6.2f}  expected={exp} ({exp_s:+.0f}) {match}")

        # Show NID symbols (first 8 = NAC + DUID)
        print("\nNID symbols (first 8 = NAC[6] + DUID[2]):")
        nid_start = sync_pos + 24

        expected_all = EXPECTED_NAC_DIBITS + EXPECTED_DUID_DIBITS
        error_count = 0

        for i in range(8):
            pos = nid_start + i
            if pos >= len(dibits):
                break

            d = dibits[pos]
            s = soft[pos]
            exp = expected_all[i]
            exp_s = {0: 1.0, 1: 3.0, 2: -1.0, 3: -3.0}[exp]

            match = "ok" if d == exp else "ERR"
            if d != exp:
                error_count += 1

            region = "NAC" if i < 6 else "DUID"
            print(f"  {region}[{i}] dibit={d} soft={s:+6.2f}  expected={exp} ({exp_s:+.0f}) {match}")

        print(f"\nErrors in NAC+DUID: {error_count}/8")

        # Compute what NAC we would get from these dibits
        nac_dibits = [dibits[nid_start + i] for i in range(6)]
        nac = 0
        for d in nac_dibits:
            nac = (nac << 2) | d
        print(f"NAC from dibits: 0x{nac:03x} (expected 0x3dc)")

    # Summary statistics
    print(f"\n{'='*70}")
    print("Summary: Soft Symbol Analysis at Error Positions")
    print(f"{'='*70}")

    # For all syncs, analyze the soft values at positions that are wrong
    position_stats = {i: {'soft_values': [], 'errors': 0, 'total': 0} for i in range(8)}

    for sync_pos, _ in syncs:
        nid_start = sync_pos + 24
        expected_all = EXPECTED_NAC_DIBITS + EXPECTED_DUID_DIBITS

        for i in range(8):
            pos = nid_start + i
            if pos >= len(dibits):
                continue

            d = dibits[pos]
            s = soft[pos]
            exp = expected_all[i]

            position_stats[i]['soft_values'].append(s)
            position_stats[i]['total'] += 1
            if d != exp:
                position_stats[i]['errors'] += 1

    print(f"\n{'Pos':>3} {'Expect':>7} {'Errors':>8} {'ErrorRate':>10} {'SoftMean':>10} {'SoftStd':>9}")
    for i in range(8):
        stats = position_stats[i]
        if stats['total'] == 0:
            continue

        soft_arr = np.array(stats['soft_values'])
        exp = (EXPECTED_NAC_DIBITS + EXPECTED_DUID_DIBITS)[i]
        exp_s = {0: 1.0, 1: 3.0, 2: -1.0, 3: -3.0}[exp]
        error_rate = 100 * stats['errors'] / stats['total']

        print(f"{i:>3} {exp_s:>+7.0f} {stats['errors']:>5}/{stats['total']:<2} {error_rate:>9.1f}% "
              f"{soft_arr.mean():>+10.2f} {soft_arr.std():>9.2f}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("/Users/thw/SDRTrunk/recordings/20251227_121743_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav")
