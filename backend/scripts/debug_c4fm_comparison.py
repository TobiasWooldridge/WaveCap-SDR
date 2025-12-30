#!/usr/bin/env python3
"""Compare C4FM demodulation approaches on SDRTrunk baseband.

This script helps identify where WaveCap diverges from expected behavior.
"""

import sys
import wave
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, '/Users/thw/Projects/WaveCap-SDR/backend')

from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator
from wavecapsdr.decoders.p25_frames import decode_nid, DUID

# Frame sync as dibit array (24 dibits) - standard P25 sync
FRAME_SYNC_DIBITS = np.array([
    1, 1, 1, 1, 1, 3, 1, 1, 3, 3, 1, 1, 3, 3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3
], dtype=np.uint8)


def load_baseband(filepath: str) -> tuple[np.ndarray, int]:
    """Load SDRTrunk baseband recording."""
    with wave.open(filepath, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

        # SDRTrunk baseband: stereo int16
        data = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
        iq = (data[:, 0] + 1j * data[:, 1]).astype(np.complex64) / 32768.0
        return iq, sample_rate


def simple_fm_demod(iq: np.ndarray) -> np.ndarray:
    """Simple FM demodulation: phase derivative."""
    phase = np.angle(iq)
    unwrapped = np.unwrap(phase)
    disc = np.diff(unwrapped)
    return disc.astype(np.float32)


def find_sync_correlation(dibits: np.ndarray) -> list[tuple[int, int]]:
    """Find sync pattern with correlation score.

    Returns: [(position, matches), ...]
    """
    results = []
    sync_len = len(FRAME_SYNC_DIBITS)

    for i in range(len(dibits) - sync_len):
        # Check exact match and with polarity reversal
        normal_matches = np.sum(dibits[i:i+sync_len] == FRAME_SYNC_DIBITS)

        # Reversed polarity: XOR with 2
        reversed_dibits = dibits[i:i+sync_len] ^ 2
        reversed_matches = np.sum(reversed_dibits == FRAME_SYNC_DIBITS)

        best_matches = max(normal_matches, reversed_matches)
        polarity = 'normal' if normal_matches >= reversed_matches else 'reversed'

        if best_matches >= 20:  # At least 20/24 matches
            results.append((i, best_matches, polarity))

    return results


def main(filepath: str):
    print(f"\n=== Loading: {filepath} ===")
    iq, sample_rate = load_baseband(filepath)
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {len(iq)/sample_rate:.2f} seconds")
    print(f"IQ samples: {len(iq)}")

    # Take first 30 seconds for faster analysis
    duration = 30.0
    samples = int(duration * sample_rate)
    iq = iq[:samples]
    print(f"Analyzing first {duration:.0f} seconds ({samples} samples)")

    # Simple FM demod
    print(f"\n=== Simple FM Demodulation ===")
    disc = simple_fm_demod(iq)
    print(f"Discriminator: min={disc.min():.4f}, max={disc.max():.4f}, std={disc.std():.4f}")

    # Expected for P25 C4FM:
    # - Deviation: ±600 Hz (inner) and ±1800 Hz (outer)
    # - At 50kHz sample rate, this gives ~±0.075 and ~±0.226 radians per sample
    # But discriminator is the phase DIFFERENCE, so:
    # - Symbol duration = 1/4800 = 208.33us
    # - At 50kHz, that's 10.42 samples per symbol
    # - Total phase shift per symbol = deviation * 2π / sample_rate
    print(f"Expected discriminator range for P25 C4FM: ~±0.23 radians")

    # C4FM Demodulator
    print(f"\n=== WaveCap C4FM Demodulator ===")
    demod = C4FMDemodulator(sample_rate=sample_rate)
    dibits, soft = demod.demodulate(iq)
    print(f"Symbols recovered: {len(dibits)}")
    print(f"Soft symbol std: {soft.std():.4f}")
    print(f"First 30 dibits: {dibits[:30]}")

    # Dibit distribution
    unique, counts = np.unique(dibits, return_counts=True)
    print(f"Dibit distribution: {dict(zip(unique, counts))}")

    # Check for sync patterns
    print(f"\n=== Frame Sync Detection ===")
    syncs = find_sync_correlation(dibits)
    print(f"Found {len(syncs)} potential sync patterns (>=20/24 matches)")

    if syncs:
        print("First 10 syncs:")
        for i, (pos, matches, polarity) in enumerate(syncs[:10]):
            print(f"  [{i}] pos={pos}, matches={matches}/24, polarity={polarity}")

            # Try to decode NID after sync
            nid_start = pos + 24
            if nid_start + 33 <= len(dibits):
                nid_dibits = dibits[nid_start:nid_start+33]
                nid = decode_nid(nid_dibits, skip_status_at_10=True)
                if nid:
                    print(f"       NID: NAC=0x{nid.nac:03x}, DUID={nid.duid.name if hasattr(nid.duid, 'name') else nid.duid}")

    # Test discriminator-based demodulation
    print(f"\n=== Discriminator-based C4FM ===")
    demod2 = C4FMDemodulator(sample_rate=sample_rate)
    dibits2, soft2 = demod2.demodulate_discriminator(disc)
    print(f"Symbols recovered: {len(dibits2)}")
    print(f"Soft symbol std: {soft2.std():.4f}")
    print(f"First 30 dibits: {dibits2[:30]}")

    # Dibit distribution
    unique2, counts2 = np.unique(dibits2, return_counts=True)
    print(f"Dibit distribution: {dict(zip(unique2, counts2))}")

    # Check for sync patterns
    syncs2 = find_sync_correlation(dibits2)
    print(f"Found {len(syncs2)} potential sync patterns (>=20/24 matches)")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("/Users/thw/SDRTrunk/recordings/20251227_121743_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav")
