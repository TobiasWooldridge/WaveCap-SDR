#!/usr/bin/env python3
"""Compare dibit outputs from SDRTrunk reference vs WaveCap C4FM.

This script helps identify systematic differences in symbol recovery.
"""

import sys
import wave
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, '/Users/thw/Projects/WaveCap-SDR/backend')

from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator
from scripts.sdrtrunk_reference import SDRTrunkC4FMDemodulator

# Frame sync as dibit array (24 dibits) - standard P25 sync
FRAME_SYNC_DIBITS = np.array([
    1, 1, 1, 1, 1, 3, 1, 1, 3, 3, 1, 1, 3, 3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3
], dtype=np.uint8)


def load_baseband(filepath: str, max_samples: int = None) -> tuple[np.ndarray, int]:
    """Load SDRTrunk baseband recording."""
    with wave.open(filepath, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        if max_samples:
            n_frames = min(n_frames, max_samples)
        raw = wf.readframes(n_frames)

        # SDRTrunk baseband: stereo int16
        data = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
        iq = (data[:, 0] + 1j * data[:, 1]).astype(np.complex64) / 32768.0
        return iq, sample_rate


def find_sync_positions(dibits: np.ndarray, threshold: int = 21) -> list[int]:
    """Find sync pattern positions with at least threshold matches."""
    positions = []
    sync_len = len(FRAME_SYNC_DIBITS)

    for i in range(len(dibits) - sync_len):
        # Check normal polarity
        matches = np.sum(dibits[i:i+sync_len] == FRAME_SYNC_DIBITS)
        if matches >= threshold:
            positions.append(i)

    return positions


def analyze_sync_regions(dibits: np.ndarray, sync_positions: list[int]) -> dict:
    """Analyze dibits around sync positions."""
    stats = {
        'sync_count': len(sync_positions),
        'sync_matches': [],
        'post_sync_dibits': [],
    }

    for pos in sync_positions[:10]:  # Analyze first 10
        # Count exact matches
        matches = np.sum(dibits[pos:pos+24] == FRAME_SYNC_DIBITS)
        stats['sync_matches'].append(matches)

        # Get 8 dibits after sync (start of NID)
        if pos + 24 + 8 <= len(dibits):
            post_sync = dibits[pos+24:pos+24+8]
            stats['post_sync_dibits'].append(list(post_sync))

    return stats


def main(filepath: str):
    print(f"\n{'='*70}")
    print(f"Comparing dibit outputs: WaveCap vs SDRTrunk Reference")
    print(f"{'='*70}")

    # Load first 5 seconds of data
    duration = 5.0
    iq, sample_rate = load_baseband(filepath, int(duration * 50000))
    print(f"File: {filepath}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Samples: {len(iq)} ({len(iq)/sample_rate:.2f}s)")

    # SDRTrunk reference demodulator
    print(f"\n--- SDRTrunk Reference C4FM ---")
    sdrtrunk = SDRTrunkC4FMDemodulator(sample_rate)
    sdr_dibits, sdr_soft = sdrtrunk.demodulate(iq)
    print(f"Symbols: {len(sdr_dibits)}")
    print(f"Soft std: {sdr_soft.std():.4f} (expected ~0.785)")
    print(f"First 50 dibits: {list(sdr_dibits[:50])}")

    # Dibit distribution
    sdr_unique, sdr_counts = np.unique(sdr_dibits, return_counts=True)
    sdr_dist = {int(u): int(c) for u, c in zip(sdr_unique, sdr_counts)}
    print(f"Distribution: {sdr_dist}")

    # Find syncs
    sdr_syncs = find_sync_positions(sdr_dibits)
    print(f"Sync patterns found: {len(sdr_syncs)}")
    if sdr_syncs:
        sdr_stats = analyze_sync_regions(sdr_dibits, sdr_syncs)
        print(f"First 5 sync matches: {sdr_stats['sync_matches'][:5]}")
        if sdr_stats['post_sync_dibits']:
            print(f"Post-sync dibits (first sync): {sdr_stats['post_sync_dibits'][0]}")

    # WaveCap C4FM demodulator
    print(f"\n--- WaveCap C4FM ---")
    wavecap = C4FMDemodulator(sample_rate=sample_rate)
    wc_dibits, wc_soft = wavecap.demodulate(iq)
    print(f"Symbols: {len(wc_dibits)}")
    print(f"Soft std: {wc_soft.std():.4f} (expected ~0.785)")
    print(f"First 50 dibits: {list(wc_dibits[:50])}")

    # Dibit distribution
    wc_unique, wc_counts = np.unique(wc_dibits, return_counts=True)
    wc_dist = {int(u): int(c) for u, c in zip(wc_unique, wc_counts)}
    print(f"Distribution: {wc_dist}")

    # Find syncs
    wc_syncs = find_sync_positions(wc_dibits)
    print(f"Sync patterns found: {len(wc_syncs)}")
    if wc_syncs:
        wc_stats = analyze_sync_regions(wc_dibits, wc_syncs)
        print(f"First 5 sync matches: {wc_stats['sync_matches'][:5]}")
        if wc_stats['post_sync_dibits']:
            print(f"Post-sync dibits (first sync): {wc_stats['post_sync_dibits'][0]}")

    # Compare dibits
    print(f"\n--- Comparison ---")
    min_len = min(len(sdr_dibits), len(wc_dibits))
    if min_len > 0:
        exact_matches = np.sum(sdr_dibits[:min_len] == wc_dibits[:min_len])
        match_rate = exact_matches / min_len * 100
        print(f"Exact match rate: {match_rate:.1f}% ({exact_matches}/{min_len})")

        # Check for systematic XOR patterns
        xor_result = sdr_dibits[:min_len] ^ wc_dibits[:min_len]
        xor_unique, xor_counts = np.unique(xor_result, return_counts=True)
        xor_dist = {int(u): int(c) for u, c in zip(xor_unique, xor_counts)}
        print(f"XOR distribution: {xor_dist}")

        # If XOR with 2 gives better match, polarity is inverted
        xor2_result = (wc_dibits[:min_len] ^ 2)
        xor2_matches = np.sum(sdr_dibits[:min_len] == xor2_result)
        xor2_rate = xor2_matches / min_len * 100
        print(f"Match rate with XOR 2 (polarity flip): {xor2_rate:.1f}%")

        # Check for XOR 1 (swap 0↔1 and 2↔3)
        xor1_result = wc_dibits[:min_len] ^ 1
        xor1_matches = np.sum(sdr_dibits[:min_len] == xor1_result)
        xor1_rate = xor1_matches / min_len * 100
        print(f"Match rate with XOR 1: {xor1_rate:.1f}%")

        # Check for XOR 3
        xor3_result = wc_dibits[:min_len] ^ 3
        xor3_matches = np.sum(sdr_dibits[:min_len] == xor3_result)
        xor3_rate = xor3_matches / min_len * 100
        print(f"Match rate with XOR 3: {xor3_rate:.1f}%")

    # Look at sync regions specifically
    if sdr_syncs and wc_syncs:
        print(f"\n--- Sync Region Analysis ---")
        # Compare first sync position
        print(f"SDRTrunk first sync at: {sdr_syncs[0] if sdr_syncs else 'N/A'}")
        print(f"WaveCap first sync at: {wc_syncs[0] if wc_syncs else 'N/A'}")

        # If syncs are at different positions, there's a timing offset
        if sdr_syncs and wc_syncs:
            offset = wc_syncs[0] - sdr_syncs[0]
            print(f"Position offset: {offset} dibits")

            # Compare sync pattern dibits themselves
            if len(sdr_dibits) > sdr_syncs[0] + 24 and len(wc_dibits) > wc_syncs[0] + 24:
                sdr_sync = sdr_dibits[sdr_syncs[0]:sdr_syncs[0]+24]
                wc_sync = wc_dibits[wc_syncs[0]:wc_syncs[0]+24]
                print(f"SDRTrunk sync dibits: {list(sdr_sync)}")
                print(f"WaveCap sync dibits:  {list(wc_sync)}")
                print(f"Expected sync dibits: {list(FRAME_SYNC_DIBITS)}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("/Users/thw/SDRTrunk/recordings/20251227_121743_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav")
