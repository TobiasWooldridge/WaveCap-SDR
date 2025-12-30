#!/usr/bin/env python3
"""Debug TSDU frame structure and TSBK extraction.

This traces through the exact byte/dibit positions to identify
where TSBK extraction might be going wrong.
"""

import sys
import wave
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, '/Users/thw/Projects/WaveCap-SDR/backend')

from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator

# Frame sync as dibit array (24 dibits)
FRAME_SYNC_DIBITS = np.array([
    1, 1, 1, 1, 1, 3, 1, 1, 3, 3, 1, 1, 3, 3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3
], dtype=np.uint8)

# P25 TSDU frame structure (positions are dibit counts from frame start):
# Sync: 0-23 (24 dibits)
# NID: 24-56 (33 dibits including 1 status symbol at pos 35)
# TSBK blocks: 57-... (up to 3 blocks of 98 dibits each, with status symbols)
# Status symbols at positions: 35, 71, 107, 143, ...

def load_baseband(filepath: str, max_samples: int = None) -> tuple[np.ndarray, int]:
    """Load SDRTrunk baseband recording."""
    with wave.open(filepath, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        if max_samples:
            n_frames = min(n_frames, max_samples)
        raw = wf.readframes(n_frames)
        data = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
        iq = (data[:, 0] + 1j * data[:, 1]).astype(np.complex64) / 32768.0
        return iq, sample_rate


def find_sync_positions(dibits: np.ndarray, threshold: int = 22) -> list[tuple[int, int]]:
    """Find sync pattern positions with at least threshold matches."""
    positions = []
    sync_len = len(FRAME_SYNC_DIBITS)

    for i in range(len(dibits) - sync_len):
        matches = np.sum(dibits[i:i+sync_len] == FRAME_SYNC_DIBITS)
        if matches >= threshold:
            positions.append((i, matches))

    return positions


def decode_nid_simple(dibits: np.ndarray) -> dict:
    """Simple NID extraction without BCH correction."""
    if len(dibits) < 33:
        return None

    # Skip status symbol at position 11 (frame position 35)
    clean = []
    for i in range(33):
        if i == 11:  # Status symbol
            continue
        clean.append(dibits[i])

    # NAC is first 6 dibits (12 bits)
    nac = 0
    for i in range(6):
        nac = (nac << 2) | clean[i]

    # DUID is next 2 dibits (4 bits)
    duid = (clean[6] << 2) | clean[7]

    return {
        'nac': nac,
        'duid': duid,
        'nac_hex': f'0x{nac:03x}',
        'duid_hex': f'0x{duid:x}',
        'raw_dibits': clean[:8],
    }


def analyze_tsdu_frame(dibits: np.ndarray, sync_pos: int) -> dict:
    """Analyze a complete TSDU frame."""
    result = {
        'sync_pos': sync_pos,
        'sync_dibits': list(dibits[sync_pos:sync_pos+24]),
    }

    # NID starts at sync_pos + 24
    nid_start = sync_pos + 24
    if len(dibits) < nid_start + 33:
        result['error'] = 'Not enough dibits for NID'
        return result

    nid_dibits = dibits[nid_start:nid_start+33]
    result['nid'] = decode_nid_simple(nid_dibits)
    result['nid_raw'] = list(nid_dibits[:12])  # First 12 including status

    # Check DUID - only continue for TSDU (0x7)
    if result['nid'] is None or result['nid']['duid'] != 0x7:
        result['frame_type'] = 'not_tsdu'
        return result

    result['frame_type'] = 'tsdu'

    # TSBK data starts after NID
    # NID is 33 dibits (32 data + 1 status at position 11)
    # So TSBK starts at sync_pos + 24 + 33 = sync_pos + 57
    tsbk_start = sync_pos + 57

    # But we also need to account for remaining status symbols
    # Status symbols are at frame positions 35, 71, 107, 143, 179, 215, 251, 287, 323, 359
    # Frame position = sync_pos + offset

    # For TSDU, we need to extract TSBK blocks
    # Each TSBK is 98 dibits (after interleaving)
    # Status symbols within TSBK regions must be removed

    # Calculate raw frame length needed for 3 TSBK blocks
    # 24 (sync) + 33 (NID with 1 status) + 98*3 (TSBK) + more status symbols
    # Status symbols at: 35, 71, 107, 143, 179, 215, 251, 287, 323

    # For now, just show the raw dibits at key positions
    result['frame_positions'] = {}

    # Show dibits at each key position
    positions = [
        ('sync_start', sync_pos),
        ('sync_end', sync_pos + 23),
        ('nid_start', sync_pos + 24),
        ('status_1', sync_pos + 35),  # First status symbol
        ('nid_end', sync_pos + 56),
        ('tsbk_start', sync_pos + 57),
        ('status_2', sync_pos + 71),  # Second status symbol
        ('status_3', sync_pos + 107),  # Third status symbol
    ]

    for name, pos in positions:
        if pos < len(dibits):
            result['frame_positions'][name] = {
                'pos': pos,
                'dibit': int(dibits[pos]) if pos < len(dibits) else None,
            }

    # Extract TSBK region raw dibits
    tsbk_region_end = sync_pos + 360  # Approximate end of TSDU
    if len(dibits) >= tsbk_region_end:
        tsbk_region = dibits[tsbk_start:tsbk_region_end]
        result['tsbk_region_raw'] = list(tsbk_region[:50])  # First 50 dibits of TSBK region
        result['tsbk_region_len'] = len(tsbk_region)

    return result


def main(filepath: str):
    print(f"\n{'='*70}")
    print(f"TSDU Frame Structure Analysis")
    print(f"{'='*70}")

    # Load first 10 seconds
    iq, sample_rate = load_baseband(filepath, int(10 * 50000))
    print(f"File: {filepath}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {len(iq)/sample_rate:.2f}s")

    # Demodulate
    demod = C4FMDemodulator(sample_rate=sample_rate)
    dibits, soft = demod.demodulate(iq)
    print(f"Dibits: {len(dibits)}")

    # Find syncs
    syncs = find_sync_positions(dibits)
    print(f"Sync patterns found: {len(syncs)}")

    # Analyze first few TSDU frames
    tsdu_count = 0
    for sync_pos, matches in syncs[:20]:
        frame = analyze_tsdu_frame(dibits, sync_pos)

        if frame.get('frame_type') == 'tsdu':
            tsdu_count += 1
            print(f"\n--- TSDU Frame {tsdu_count} ---")
            print(f"Sync position: {sync_pos} (matches: {matches}/24)")
            print(f"NID: NAC={frame['nid']['nac_hex']}, DUID=0x{frame['nid']['duid']:x}")
            print(f"NID raw dibits: {frame['nid_raw']}")

            if 'frame_positions' in frame:
                print("Key frame positions:")
                for name, info in frame['frame_positions'].items():
                    print(f"  {name}: pos={info['pos']}, dibit={info['dibit']}")

            if 'tsbk_region_raw' in frame:
                print(f"TSBK region first 50 dibits: {frame['tsbk_region_raw']}")

            if tsdu_count >= 3:
                break

    # Now let's trace through what the ControlChannelMonitor sees
    print(f"\n{'='*70}")
    print("TSDU Structure Reference:")
    print("="*70)
    print("""
    P25 TSDU Frame Layout:
    ----------------------
    Dibits  0-23:  Frame Sync (24 dibits)
    Dibits 24-56:  NID (33 dibits, includes 1 status at pos 35)
                   - Status at frame pos 35 = NID dibit 11
    Dibits 57+:    TSBK data (up to 3 blocks of 98 dibits each)
                   - Status symbols at frame pos 71, 107, 143, 179, 215, 251, 287

    Status symbol positions (every 36 dibits from frame start):
    35, 71, 107, 143, 179, 215, 251, 287, 323, 359...

    For a TSDU with 3 TSBK blocks:
    - Total frame length: ~360 dibits (includes ~10 status symbols)
    - Sync: 24 dibits
    - NID: 32 data + 1 status = 33 raw dibits
    - TSBK1: 98 data + status symbols interspersed
    - TSBK2: 98 data + status symbols interspersed
    - TSBK3: 98 data + status symbols interspersed
    """)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("/Users/thw/SDRTrunk/recordings/20251227_121743_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav")
