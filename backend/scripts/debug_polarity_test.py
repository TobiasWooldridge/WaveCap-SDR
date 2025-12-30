#!/usr/bin/env python3
"""Test different IQ polarities and orderings to find correct decoding."""

import sys
import wave
import numpy as np
import logging

logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, '/Users/thw/Projects/WaveCap-SDR/backend')

from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator

FRAME_SYNC_DIBITS = np.array([
    1, 1, 1, 1, 1, 3, 1, 1, 3, 3, 1, 1, 3, 3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3
], dtype=np.uint8)

# Expected SA-GRN NAC
EXPECTED_NAC = 0x3dc
EXPECTED_NAC_DIBITS = [0, 3, 3, 1, 3, 0]


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


def find_sync_positions(dibits: np.ndarray, threshold: int = 22) -> list[tuple[int, int]]:
    positions = []
    sync_len = len(FRAME_SYNC_DIBITS)
    for i in range(len(dibits) - sync_len):
        matches = np.sum(dibits[i:i+sync_len] == FRAME_SYNC_DIBITS)
        if matches >= threshold:
            positions.append((i, matches))
    return positions


def check_nac(dibits: np.ndarray, sync_pos: int) -> tuple[list, bool, int]:
    """Extract NAC dibits and check against expected."""
    nid_start = sync_pos + 24
    if len(dibits) < nid_start + 6:
        return [], False, 0

    nac_dibits = list(dibits[nid_start:nid_start+6])
    matches = sum(1 for a, b in zip(nac_dibits, EXPECTED_NAC_DIBITS) if a == b)
    return nac_dibits, nac_dibits == EXPECTED_NAC_DIBITS, matches


def test_configuration(iq: np.ndarray, sample_rate: int, name: str) -> dict:
    """Test a specific IQ configuration."""
    demod = C4FMDemodulator(sample_rate=sample_rate)
    dibits, soft = demod.demodulate(iq)

    syncs = find_sync_positions(dibits, threshold=22)

    results = {
        'name': name,
        'syncs': len(syncs),
        'perfect_syncs': sum(1 for _, m in syncs if m == 24),
        'nac_matches': 0,
        'nac_exact': 0,
        'sample_nacs': [],
    }

    for sync_pos, _ in syncs[:10]:
        nac_dibits, exact, matches = check_nac(dibits, sync_pos)
        results['sample_nacs'].append((nac_dibits, matches))
        if exact:
            results['nac_exact'] += 1
        results['nac_matches'] += matches

    if syncs:
        results['nac_avg_match'] = results['nac_matches'] / (len(syncs[:10]) * 6)
    else:
        results['nac_avg_match'] = 0

    return results


def main(filepath: str):
    print(f"\n{'='*70}")
    print(f"Polarity and IQ Order Test")
    print(f"{'='*70}")

    iq_orig, sample_rate = load_baseband(filepath, int(10 * 50000))
    print(f"File: {filepath}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Samples: {len(iq_orig)}")
    print(f"Expected NAC: 0x{EXPECTED_NAC:03x} = dibits {EXPECTED_NAC_DIBITS}")

    # Test different configurations
    configs = [
        ("Original (I+jQ)", iq_orig),
        ("Conjugate (I-jQ)", np.conj(iq_orig)),
        ("Swapped (Q+jI)", iq_orig.imag + 1j * iq_orig.real),
        ("Swapped+Conj (Q-jI)", iq_orig.imag - 1j * iq_orig.real),
        ("Negate I (-I+jQ)", -iq_orig.real + 1j * iq_orig.imag),
        ("Negate Q (I-jQ)", iq_orig.real - 1j * iq_orig.imag),
        ("Negate both (-I-jQ)", -iq_orig),
    ]

    print(f"\n{'Configuration':<25} {'Syncs':>8} {'Perfect':>8} {'NAC Match':>10} {'Best NAC':<20}")
    print("=" * 75)

    best_result = None
    best_score = 0

    for name, iq in configs:
        result = test_configuration(iq, sample_rate, name)

        # Show first NAC sample
        first_nac = result['sample_nacs'][0] if result['sample_nacs'] else ([], 0)

        print(f"{name:<25} {result['syncs']:>8} {result['perfect_syncs']:>8} "
              f"{result['nac_avg_match']*100:>9.1f}% {str(first_nac[0]):>20}")

        score = result['nac_avg_match'] + result['perfect_syncs'] * 0.1
        if score > best_score:
            best_score = score
            best_result = result

    print(f"\nBest configuration: {best_result['name']}")
    print(f"  Perfect syncs: {best_result['perfect_syncs']}")
    print(f"  NAC average match: {best_result['nac_avg_match']*100:.1f}%")
    print(f"  NAC exact matches: {best_result['nac_exact']}")

    # Try with frequency correction
    print(f"\n{'='*70}")
    print("Testing with frequency offset correction")
    print(f"{'='*70}")

    # Estimate frequency offset
    n = 10000
    phases = np.angle(iq_orig[:n])
    unwrapped = np.unwrap(phases)
    phase_rate = np.mean(np.diff(unwrapped))
    freq_offset = phase_rate * sample_rate / (2 * np.pi)
    print(f"Estimated frequency offset: {freq_offset:.1f} Hz")

    # Apply frequency correction
    t = np.arange(len(iq_orig)) / sample_rate
    for offset in [-100, -50, 0, 50, 100, freq_offset]:
        correction = np.exp(-2j * np.pi * offset * t).astype(np.complex64)
        iq_corrected = iq_orig * correction

        result = test_configuration(iq_corrected, sample_rate, f"Offset {offset:+.0f} Hz")
        first_nac = result['sample_nacs'][0] if result['sample_nacs'] else ([], 0)
        print(f"Offset {offset:+6.0f} Hz: syncs={result['syncs']:>3}, "
              f"perfect={result['perfect_syncs']:>3}, "
              f"NAC match={result['nac_avg_match']*100:>5.1f}%, "
              f"NAC={first_nac[0]}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("/Users/thw/SDRTrunk/recordings/20251227_121743_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav")
