#!/usr/bin/env python3
"""Debug TSBK extraction to find where CRC failures originate.

This script traces the exact dibit values through:
1. C4FMDemodulator output (after resampling)
2. ControlChannelMonitor buffer
3. TSBK data extraction
4. Trellis decoding

To find where the error_metric=25 is coming from.
"""

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
    print("TSBK Extraction Debug")
    print("=" * 70)

    iq, sample_rate = load_baseband(filepath, int(30 * 50000))  # 30 seconds
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Total samples: {len(iq)}")

    from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator
    from wavecapsdr.decoders.p25_frames import decode_nid, decode_tsdu, DUID, remove_status_symbols_with_offset

    # Demodulate with single call (no chunking for simplicity)
    demod = C4FMDemodulator(sample_rate=sample_rate)
    dibits, soft = demod.demodulate(iq)

    print(f"\nTotal dibits: {len(dibits)}")
    print(f"Total syncs: {demod._sync_count}")

    # Find sync patterns in the output
    expected_sync = np.array([3, 3, 3, 3, 3, -3, 3, 3, -3, -3, 3, 3, -3, -3, -3, -3, 3, -3, 3, -3, -3, -3, -3, -3], dtype=np.float32)

    sync_positions = []
    for i in range(len(soft) - 360):
        window = soft[i:i+24]
        corr = np.sum(window * expected_sync)
        if corr > 200:  # Strong sync correlation
            if not sync_positions or i - sync_positions[-1] > 100:
                sync_positions.append(i)

    print(f"Found {len(sync_positions)} sync patterns")

    # Analyze first 5 syncs
    for sync_idx, sync_pos in enumerate(sync_positions[:5]):
        print(f"\n{'='*70}")
        print(f"SYNC #{sync_idx+1} at dibit position {sync_pos}")
        print("="*70)

        # Extract full frame (360 dibits)
        frame_dibits = np.array(dibits[sync_pos:sync_pos+360], dtype=np.uint8)
        frame_soft = np.array(soft[sync_pos:sync_pos+360], dtype=np.float32)

        # Decode NID (33 dibits after sync)
        nid_dibits = frame_dibits[24:57]
        nid = decode_nid(nid_dibits, skip_status_at_10=True)

        if nid:
            print(f"NID: NAC=0x{nid.nac:03x}, DUID={nid.duid.name}, errors={nid.errors}")
        else:
            print("NID: decode failed")
            continue

        if nid.duid != DUID.TSDU:
            print(f"  Not a TSDU frame, skipping TSBK analysis")
            continue

        # Show TSBK data positions
        print(f"\nTSBK data analysis:")

        # TSDU layout: 24 sync + 33 NID (with status at 11) = 57 dibits header
        # Then TSBK data blocks with status symbols interspersed
        tsbk_start = 57

        # Show soft values at key positions
        print(f"\n  Frame positions 55-70 (around NID end / TSBK start):")
        print(f"  {'Pos':>4} {'Dibit':>6} {'Soft':>8} {'Note':>20}")
        print("  " + "-" * 50)
        for j in range(55, min(75, len(frame_dibits))):
            dibit = frame_dibits[j]
            s = frame_soft[j]
            note = ""
            if j == 56:
                note = "<-- Last NID dibit"
            elif j == 57:
                note = "<-- First TSBK data"
            elif (j + 1) % 36 == 0:
                note = "<-- Status symbol"
            print(f"  {j:>4} {dibit:>6} {s:>+8.2f} {note:>20}")

        # Decode TSDU to get TSBK blocks
        tsdu = decode_tsdu(frame_dibits, frame_soft)

        if tsdu and tsdu.tsbk_blocks:
            print(f"\n  TSDU decoded: {len(tsdu.tsbk_blocks)} TSBK blocks")
            for block_idx, block in enumerate(tsdu.tsbk_blocks):
                print(f"\n  Block {block_idx}:")
                print(f"    opcode=0x{block.opcode:02x}, mfid=0x{block.mfid:02x}")
                print(f"    crc_valid={block.crc_valid}")
                print(f"    trellis_error_metric={getattr(block, 'trellis_error_metric', 'N/A')}")

                # Show the raw trellis input (before decoding)
                # TSBK block is 196 dibits after removing status symbols
                # First find the raw dibits for this block
                if block_idx == 0:
                    # First TSBK starts at position 57 + 98*block_idx (approximately)
                    raw_start = 57
                    raw_end = raw_start + 98 + 20  # Include some extra for status
                    raw_dibits = frame_dibits[raw_start:raw_end]
                    raw_soft = frame_soft[raw_start:raw_end]
                    print(f"    Raw data positions {raw_start}-{raw_end}:")
                    print(f"    First 20 dibits: {list(raw_dibits[:20])}")
                    print(f"    First 20 soft:   {[f'{s:.1f}' for s in raw_soft[:20]]}")
        else:
            print(f"\n  TSDU decode failed or no TSBK blocks")

    # Statistics across all syncs
    print("\n" + "=" * 70)
    print("TSBK CRC statistics across all frames")
    print("=" * 70)

    crc_pass = 0
    crc_fail = 0
    error_metrics = []

    for sync_pos in sync_positions:
        if sync_pos + 360 > len(dibits):
            break
        frame_dibits = np.array(dibits[sync_pos:sync_pos+360], dtype=np.uint8)
        frame_soft = np.array(soft[sync_pos:sync_pos+360], dtype=np.float32)

        nid_dibits = frame_dibits[24:57]
        nid = decode_nid(nid_dibits, skip_status_at_10=True)

        if nid and nid.duid == DUID.TSDU:
            tsdu = decode_tsdu(frame_dibits, frame_soft)
            if tsdu and tsdu.tsbk_blocks:
                for block in tsdu.tsbk_blocks:
                    if block.crc_valid:
                        crc_pass += 1
                    else:
                        crc_fail += 1
                    if hasattr(block, 'trellis_error_metric'):
                        error_metrics.append(block.trellis_error_metric)

    total = crc_pass + crc_fail
    if total > 0:
        print(f"TSBK blocks: {total}")
        print(f"CRC pass: {crc_pass} ({100*crc_pass/total:.1f}%)")
        print(f"CRC fail: {crc_fail} ({100*crc_fail/total:.1f}%)")

    if error_metrics:
        em_arr = np.array(error_metrics)
        print(f"\nTrellis error metrics:")
        print(f"  min: {em_arr.min()}")
        print(f"  max: {em_arr.max()}")
        print(f"  mean: {em_arr.mean():.1f}")
        print(f"  median: {np.median(em_arr):.1f}")

        # Distribution
        print(f"\n  Distribution:")
        for threshold in [0, 5, 10, 15, 20, 25, 30]:
            count = np.sum(em_arr <= threshold)
            print(f"    <= {threshold}: {count} ({100*count/len(em_arr):.1f}%)")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("/Users/thw/SDRTrunk/recordings/20251227_121743_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav")
