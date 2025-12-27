#!/usr/bin/env python3
"""Debug NID extraction from actual signal to diagnose BCH failures."""

import sys
import importlib.util
import wave
import numpy as np
from pathlib import Path

# Direct file imports
backend_path = Path(__file__).parent.parent

def import_module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

c4fm_module = import_module_from_file(
    'c4fm',
    backend_path / 'wavecapsdr' / 'dsp' / 'p25' / 'c4fm.py'
)
C4FMDemodulator = c4fm_module.C4FMDemodulator

bch_module = import_module_from_file(
    'bch',
    backend_path / 'wavecapsdr' / 'dsp' / 'fec' / 'bch.py'
)
bch_decode = bch_module.bch_decode

# P25 sync pattern (48 bits = 24 dibits)
SYNC_PATTERN = 0x5575F5FF77FF
SYNC_DIBITS = np.array([
    (SYNC_PATTERN >> ((23 - i) * 2)) & 0x3
    for i in range(24)
], dtype=np.uint8)


def load_iq_wav(path: str) -> tuple[int, np.ndarray]:
    """Load stereo IQ WAV file."""
    with wave.open(path, 'rb') as wf:
        rate = wf.getframerate()
        n_frames = wf.getnframes()
        n_channels = wf.getnchannels()

        if n_channels != 2:
            raise ValueError(f"Expected stereo WAV, got {n_channels} channels")

        raw = wf.readframes(n_frames)
        samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
        iq = (samples[:, 0] + 1j * samples[:, 1]).astype(np.complex64) / 32768.0

    return rate, iq


def find_sync_positions(dibits: np.ndarray, threshold: int = 3) -> list[int]:
    """Find positions where sync pattern matches with few errors."""
    positions = []
    for i in range(len(dibits) - 24):
        errors = 0
        for j in range(24):
            if dibits[i + j] != SYNC_DIBITS[j]:
                errors += 1
                if errors > threshold:
                    break
        if errors <= threshold:
            positions.append(i)
    return positions


def extract_nid(dibits: np.ndarray, sync_pos: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract NID from dibits after sync position.

    Returns tuple of (nid_dibits, nid_bits).
    """
    # NID starts at sync_pos + 24 (after 24 dibit sync)
    nid_start = sync_pos + 24

    # Collect 33 dibits (32 NID + 1 status at position 11)
    if nid_start + 33 > len(dibits):
        return None, None

    nid_dibits_raw = dibits[nid_start:nid_start + 33]

    # Remove status symbol at position 11
    nid_dibits = np.concatenate([nid_dibits_raw[:11], nid_dibits_raw[12:33]])

    # Convert to bits
    nid_bits = np.zeros(64, dtype=np.uint8)
    for i in range(32):
        nid_bits[i * 2] = (nid_dibits[i] >> 1) & 1
        nid_bits[i * 2 + 1] = nid_dibits[i] & 1

    return nid_dibits, nid_bits


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='?', default=None, help='Input WAV file')
    args = parser.parse_args()

    if args.input is None:
        # Try to find a test file
        test_files = [
            '/tmp/sigid_C4FM_CC_IF.wav',
            'sigid_C4FM_CC_IF.wav',
        ]
        for f in test_files:
            if Path(f).exists():
                args.input = f
                break

    if args.input is None or not Path(args.input).exists():
        print("No input file found. Please provide a WAV file path.")
        return 1

    print(f"Loading {args.input}...")
    sample_rate, iq = load_iq_wav(args.input)
    print(f"Loaded {len(iq)} samples at {sample_rate} Hz")

    print("\nDemodulating...")
    demod = C4FMDemodulator(sample_rate=sample_rate)

    # Print demod config for debugging
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Symbol rate: {demod.symbol_rate} baud")
    print(f"  Samples per symbol: {demod.samples_per_symbol:.2f}")
    print(f"  FM demod symbol_delay: {demod._fm_demod.symbol_delay}")

    dibits, soft_symbols = demod.demodulate(iq)
    print(f"Produced {len(dibits)} dibits")

    print(f"\nSoft symbol stats: mean={np.mean(soft_symbols):.3f}, std={np.std(soft_symbols):.3f}")

    # Check soft symbol distribution by quadrant
    # Soft symbols are normalized to ±1, ±3 range (multiply by 4/π from radians)
    # Convert back to radians for analysis
    soft_rad = soft_symbols * (np.pi / 4.0)
    print(f"  Phase range: [{np.min(soft_rad):.3f}, {np.max(soft_rad):.3f}] rad")

    # Count symbols near each ideal position
    ideal_phases = {
        "+3 (3π/4)": 3 * np.pi / 4,
        "+1 (π/4)": np.pi / 4,
        "-1 (-π/4)": -np.pi / 4,
        "-3 (-3π/4)": -3 * np.pi / 4,
    }
    print("  Symbol distribution (within ±π/8 of ideal):")
    for name, ideal in ideal_phases.items():
        count = np.sum(np.abs(soft_rad - ideal) < np.pi / 8)
        pct = count / len(soft_rad) * 100
        print(f"    {name}: {count} ({pct:.1f}%)")

    dibit_counts = [np.sum(dibits == i) for i in range(4)]
    pcts = [c/len(dibits)*100 for c in dibit_counts]
    print(f"Dibit distribution: {dibit_counts} ({pcts[0]:.1f}%, {pcts[1]:.1f}%, {pcts[2]:.1f}%, {pcts[3]:.1f}%)")

    # Print the expected sync pattern for debugging
    print("\nExpected sync pattern dibits:")
    print(f"  {list(SYNC_DIBITS)}")

    # Look for any 24-dibit sequence that matches even partially
    print("\nSearching for sync patterns...")
    sync_positions = find_sync_positions(dibits, threshold=3)
    print(f"Found {len(sync_positions)} sync positions (threshold=3)")

    if len(sync_positions) == 0:
        print("\nNo syncs found! Trying with higher threshold...")
        sync_positions = find_sync_positions(dibits, threshold=6)
        print(f"Found {len(sync_positions)} sync positions (threshold=6)")

    if len(sync_positions) == 0:
        # Debug: check correlation at various positions
        print("\nDebug: checking dibit correlation across entire file...")
        matches = []
        for i in range(len(dibits) - 24):
            errors = sum(1 for j in range(24) if dibits[i + j] != SYNC_DIBITS[j])
            if errors <= 8:  # Keep matches with 8 or fewer errors
                matches.append((i, errors))

        print(f"  Found {len(matches)} potential matches with <=8 errors")
        if matches:
            # Show first 10
            for pos, errors in matches[:10]:
                print(f"    Position {pos}: {errors} errors")
            # Show best match
            best = min(matches, key=lambda x: x[1])
            print(f"  Best match: position {best[0]} with {best[1]} errors")
            print(f"  Dibits at best match: {list(dibits[best[0]:best[0]+24])}")

        # Check if maybe the dibits are inverted (bit swap within dibit)
        print("\nChecking for inverted dibits (bit swap)...")
        inverted_sync = np.array([((d & 1) << 1) | ((d >> 1) & 1) for d in SYNC_DIBITS], dtype=np.uint8)
        print(f"  Inverted sync pattern: {list(inverted_sync)}")
        best_pos_inv = -1
        best_errors_inv = 24
        for i in range(min(10000, len(dibits) - 24)):
            errors = sum(1 for j in range(24) if dibits[i + j] != inverted_sync[j])
            if errors < best_errors_inv:
                best_errors_inv = errors
                best_pos_inv = i
        print(f"  Best inverted match: position {best_pos_inv} with {best_errors_inv} errors")

        # Check for rotated dibits (90, 180, 270 degree rotation)
        print("\nChecking for rotated dibits...")
        rotations = [(1, 0, 3, 2), (3, 2, 1, 0), (2, 3, 0, 1)]  # +90, 180, -90
        for name, rot in [("90", (1, 0, 3, 2)), ("180", (3, 2, 1, 0)), ("270", (2, 3, 0, 1))]:
            rot_sync = np.array([rot[d] for d in SYNC_DIBITS], dtype=np.uint8)
            best_pos_rot = -1
            best_errors_rot = 24
            for i in range(min(10000, len(dibits) - 24)):
                errors = sum(1 for j in range(24) if dibits[i + j] != rot_sync[j])
                if errors < best_errors_rot:
                    best_errors_rot = errors
                    best_pos_rot = i
            if best_errors_rot <= 6:
                print(f"  {name}° rotation: position {best_pos_rot} with {best_errors_rot} errors")
                print(f"    Rotated sync: {list(rot_sync)}")
                print(f"    Dibits at match: {list(dibits[best_pos_rot:best_pos_rot+24])}")

    print("\n" + "=" * 60)
    print("NID Analysis")
    print("=" * 60)

    nid_count = 0
    bch_success = 0
    bch_fail = 0

    for sync_pos in sync_positions[:20]:  # Analyze first 20
        nid_dibits, nid_bits = extract_nid(dibits, sync_pos)

        if nid_dibits is None:
            continue

        nid_count += 1

        # Try BCH decode
        data, errors = bch_decode(nid_bits[:63])

        if errors >= 0:
            bch_success += 1
            nac = (data >> 4) & 0xFFF
            duid = data & 0xF
            print(f"\nSync at {sync_pos}: BCH SUCCESS (errors={errors})")
            print(f"  NAC=0x{nac:03X}, DUID=0x{duid:X}")
        else:
            bch_fail += 1
            print(f"\nSync at {sync_pos}: BCH FAILED")

            # Dump NID bits for analysis
            print(f"  NID dibits: {list(nid_dibits[:8])}...")
            print(f"  NID bits[0:16]: {list(nid_bits[:16])}")

            # Try to extract what would be NAC/DUID from bits
            raw_nac = 0
            for i in range(12):
                raw_nac = (raw_nac << 1) | nid_bits[i]
            raw_duid = 0
            for i in range(4):
                raw_duid = (raw_duid << 1) | nid_bits[12 + i]
            print(f"  Raw (uncorrected): NAC=0x{raw_nac:03X}, DUID=0x{raw_duid:X}")

    print("\n" + "=" * 60)
    print(f"Summary: {nid_count} NIDs analyzed")
    print(f"  BCH success: {bch_success}")
    print(f"  BCH fail: {bch_fail}")
    if nid_count > 0:
        print(f"  Success rate: {bch_success/nid_count*100:.1f}%")


if __name__ == '__main__':
    sys.exit(main() or 0)
