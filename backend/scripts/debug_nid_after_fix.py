#!/usr/bin/env python3
"""Debug NID resampling to see if it fixes the position 4 error."""

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
    print("NID After Fix Debug")
    print("=" * 70)

    iq, sample_rate = load_baseband(filepath, int(10 * 50000))  # 10 seconds
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Samples: {len(iq)}")

    from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator

    # Process with chunked approach (100ms chunks)
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

    print(f"\nDemodulator output:")
    print(f"  Dibits: {len(dibits)}")
    print(f"  sync_count: {demod._sync_count}")

    # Find sync patterns in the output
    print("\n" + "=" * 70)
    print("Analyzing NID at first sync position")
    print("=" * 70)

    expected_sync = np.array([3, 3, 3, 3, 3, -3, 3, 3, -3, -3, 3, 3, -3, -3, -3, -3, 3, -3, 3, -3, -3, -3, -3, -3], dtype=np.float32)

    # Find first sync
    first_sync_pos = None
    for i in range(len(soft) - 24):
        window = soft[i:i+24]
        corr = np.sum(window * expected_sync)
        if corr > 200:
            first_sync_pos = i
            break

    if first_sync_pos is None:
        print("No sync found!")
        return

    print(f"First sync ends at position: {first_sync_pos + 23}")

    # NID starts at first_sync_pos + 24
    nid_start = first_sync_pos + 24
    nid_end = nid_start + 33

    print(f"NID positions: {nid_start} to {nid_end - 1}")

    # Expected NAC dibits for SA-GRN (0x3DC)
    expected_nac = [0, 3, 3, 1, 3, 0]
    expected_duid = [0, 1]  # DUID 7 for TSDU

    # Show NID symbols
    print("\nNID symbols (first 8 = NAC + DUID):")
    print(f"{'Pos':>4} {'Dibit':>6} {'Expect':>7} {'Soft':>8} {'Match':>6}")

    for i in range(8):
        pos = nid_start + i
        if pos >= len(dibits):
            break

        dibit = dibits[pos]
        s = soft[pos]
        exp = (expected_nac + expected_duid)[i]
        exp_soft = {0: 1.0, 1: 3.0, 2: -1.0, 3: -3.0}[exp]
        match = "ok" if dibit == exp else "ERR"

        print(f"{i:>4} {dibit:>6} {exp:>4}({exp_soft:+.0f}) {s:>+8.2f} {match:>6}")

    # Compute NAC from dibits
    nac_dibits = [dibits[nid_start + i] for i in range(6)]
    nac = sum(d << (10 - 2*i) for i, d in enumerate(nac_dibits))
    print(f"\nDecoded NAC: 0x{nac:03x} (expected 0x3dc)")

    # Check if position 4 error persists
    pos4_dibit = dibits[nid_start + 4]
    pos4_soft = soft[nid_start + 4]
    print(f"\nPosition 4 analysis:")
    print(f"  Dibit: {pos4_dibit} (expected 3)")
    print(f"  Soft: {pos4_soft:+.2f} (expected ~-3.0)")
    print(f"  Error: {'YES' if pos4_dibit != 3 else 'NO'}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("/Users/thw/SDRTrunk/recordings/20251227_121743_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav")
