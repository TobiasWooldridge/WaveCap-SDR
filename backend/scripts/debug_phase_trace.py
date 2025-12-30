#!/usr/bin/env python3
"""Trace raw FM phase values to understand symbol errors.

Looking for why NAC position 4 shows +1.3 instead of -2.3.
"""

import sys
import wave
import numpy as np
import logging

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, '/Users/thw/Projects/WaveCap-SDR/backend')

# Frame sync as dibit array (24 dibits)
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
        data = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
        iq = (data[:, 0] + 1j * data[:, 1]).astype(np.complex64) / 32768.0
        return iq, sample_rate


def simple_fm_demod(iq: np.ndarray) -> np.ndarray:
    """Simple FM demodulation using arctan2."""
    # Differential phase: angle(iq[n] * conj(iq[n-1]))
    diff = iq[1:] * np.conj(iq[:-1])
    phase = np.arctan2(diff.imag, diff.real)
    return phase.astype(np.float32)


def simple_symbol_extract(phase: np.ndarray, samples_per_symbol: float) -> tuple[np.ndarray, np.ndarray, list]:
    """Simple symbol extraction without equalizer/PLL."""
    n_symbols = int(len(phase) / samples_per_symbol)
    dibits = []
    soft = []
    sample_indices = []

    for i in range(n_symbols):
        # Sample at symbol center
        idx = int(i * samples_per_symbol + samples_per_symbol / 2)
        if idx >= len(phase):
            break

        p = phase[idx]

        # Simple π/2 decision boundaries (no equalization)
        boundary = np.pi / 2.0
        if p >= boundary:
            d = 1  # +3
        elif p >= 0:
            d = 0  # +1
        elif p >= -boundary:
            d = 2  # -1
        else:
            d = 3  # -3

        dibits.append(d)
        # Normalize to ±1, ±3 scale
        soft.append(p * 4.0 / np.pi)
        sample_indices.append(idx)

    return np.array(dibits, dtype=np.uint8), np.array(soft, dtype=np.float32), sample_indices


def find_sync_positions(dibits: np.ndarray, threshold: int = 22) -> list[tuple[int, int]]:
    """Find sync pattern positions."""
    positions = []
    sync_len = len(FRAME_SYNC_DIBITS)

    for i in range(len(dibits) - sync_len):
        matches = np.sum(dibits[i:i+sync_len] == FRAME_SYNC_DIBITS)
        if matches >= threshold:
            positions.append((i, matches))

    return positions


def main(filepath: str):
    print(f"\n{'='*70}")
    print(f"Phase Value Trace for Symbol Errors")
    print(f"{'='*70}")

    # Load first 10 seconds
    iq, sample_rate = load_baseband(filepath, int(10 * 50000))
    print(f"File: {filepath}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Samples: {len(iq)}")

    samples_per_symbol = sample_rate / 4800.0
    print(f"Samples per symbol: {samples_per_symbol:.2f}")

    # Simple FM demod (no LPF, no RRC)
    print("\n--- Simple FM Demod (no filtering) ---")
    phase_raw = simple_fm_demod(iq)
    dibits_raw, soft_raw, indices_raw = simple_symbol_extract(phase_raw, samples_per_symbol)
    print(f"Symbols: {len(dibits_raw)}")
    print(f"Soft symbol std: {soft_raw.std():.4f}")
    print(f"Phase range: [{phase_raw.min():.4f}, {phase_raw.max():.4f}]")

    syncs_raw = find_sync_positions(dibits_raw, threshold=22)
    print(f"Syncs found (>=22/24): {len(syncs_raw)}")

    if syncs_raw:
        sync_pos, matches = syncs_raw[0]
        print(f"\nFirst sync at symbol {sync_pos} ({matches}/24)")

        # Show symbols around sync and into NID
        print("\n--- Symbols around first sync ---")
        print(f"{'Pos':>4} {'Dibit':>6} {'Soft':>8} {'Expected':>10}")

        # Last 4 of sync + first 12 of NID
        for rel_pos in range(-4, 12):
            pos = sync_pos + 24 + rel_pos  # Position relative to NID start
            if pos < 0 or pos >= len(dibits_raw):
                continue

            d = dibits_raw[pos]
            s = soft_raw[pos]

            if rel_pos < 0:
                # Last of sync
                expected_d = FRAME_SYNC_DIBITS[24 + rel_pos]
                region = "SYNC"
            else:
                # NID region
                region = "NID"
                # Expected based on NAC=0x3DC = dibits [0,3,3,1,3,0] + DUID=7 = dibits [0,1,...]
                if rel_pos < 6:
                    expected_dibits = [0, 3, 3, 1, 3, 0]  # NAC
                    expected_d = expected_dibits[rel_pos]
                elif rel_pos < 8:
                    expected_dibits = [0, 1]  # DUID 7 (LDU1)
                    expected_d = expected_dibits[rel_pos - 6]
                else:
                    expected_d = "?"

            expected_soft = {0: 1.0, 1: 3.0, 2: -1.0, 3: -3.0}.get(expected_d, "?")
            match = "✓" if d == expected_d else "✗"

            print(f"{pos:4d} {d:6d} {s:8.3f} {str(expected_soft):>10} {region:>6} {match}")

        # Now let's look at the raw IQ and phase at NAC position 4
        nid_start_symbol = sync_pos + 24
        nac_pos_4_symbol = nid_start_symbol + 4

        # Find sample index for NAC position 4
        if nac_pos_4_symbol < len(indices_raw):
            sample_idx = indices_raw[nac_pos_4_symbol]
            print(f"\n--- Raw data at NAC position 4 (sample {sample_idx}) ---")

            # Show a window of samples around this symbol
            window = int(samples_per_symbol * 2)
            start = max(0, sample_idx - window)
            end = min(len(iq), sample_idx + window)

            print(f"IQ magnitude at symbol center: {np.abs(iq[sample_idx]):.6f}")
            print(f"IQ phase at symbol center: {np.angle(iq[sample_idx]):.6f}")

            # Show phase values in a window around the symbol
            print(f"\nPhase values in ±1 symbol window:")
            for offset in range(-int(samples_per_symbol), int(samples_per_symbol)+1, 2):
                idx = sample_idx + offset
                if 0 <= idx < len(phase_raw):
                    p = phase_raw[idx]
                    soft = p * 4.0 / np.pi
                    marker = " <-- SYMBOL CENTER" if offset == 0 else ""
                    print(f"  offset {offset:+4d}: phase={p:+.4f} soft={soft:+.3f}{marker}")

    # Now try with the full C4FM demodulator for comparison
    print(f"\n{'='*70}")
    print("Comparison with Full C4FM Demodulator")
    print(f"{'='*70}")

    from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator

    demod = C4FMDemodulator(sample_rate=sample_rate)
    dibits_full, soft_full = demod.demodulate(iq)
    print(f"Symbols: {len(dibits_full)}")
    print(f"Soft symbol std: {soft_full.std():.4f}")

    syncs_full = find_sync_positions(dibits_full, threshold=22)
    print(f"Syncs found (>=22/24): {len(syncs_full)}")

    if syncs_full:
        sync_pos, matches = syncs_full[0]
        print(f"\nFirst sync at symbol {sync_pos} ({matches}/24)")

        print("\n--- NID dibits comparison ---")
        print(f"{'Pos':>4} {'Simple':>8} {'Full':>8} {'Expected':>10} {'Match':>6}")

        for rel_pos in range(12):
            pos_raw = syncs_raw[0][0] + 24 + rel_pos if syncs_raw else -1
            pos_full = sync_pos + 24 + rel_pos

            d_raw = dibits_raw[pos_raw] if 0 <= pos_raw < len(dibits_raw) else -1
            d_full = dibits_full[pos_full] if 0 <= pos_full < len(dibits_full) else -1

            if rel_pos < 6:
                expected = [0, 3, 3, 1, 3, 0][rel_pos]
            elif rel_pos < 8:
                expected = [0, 1][rel_pos - 6]  # DUID 7
            else:
                expected = -1

            match = "✓" if d_full == expected else "✗"
            print(f"{rel_pos:4d} {d_raw:8d} {d_full:8d} {expected:10d} {match:>6}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("/Users/thw/SDRTrunk/recordings/20251227_121743_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav")
