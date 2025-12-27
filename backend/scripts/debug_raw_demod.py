#!/usr/bin/env python3
"""Debug raw differential demodulation on test signal without filters."""

import wave
import numpy as np
from pathlib import Path


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


def differential_demod(iq: np.ndarray, delay: int) -> np.ndarray:
    """Simple differential demodulation."""
    n = len(iq)
    phases = np.zeros(n, dtype=np.float32)

    for x in range(delay, n):
        s_curr = iq[x]
        s_prev = iq[x - delay]

        # s[n] * conj(s[n-delay])
        diff = s_curr * np.conj(s_prev)
        phases[x] = np.arctan2(diff.imag, diff.real)

    return phases


def dibit_from_phase(phase: float) -> int:
    """Map phase to dibit."""
    boundary = np.pi / 2.0
    if phase >= boundary:
        return 1
    elif phase >= 0:
        return 0
    elif phase >= -boundary:
        return 2
    else:
        return 3


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input WAV file')
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    sample_rate, iq = load_iq_wav(args.input)
    print(f"Loaded {len(iq)} samples at {sample_rate} Hz")

    symbol_rate = 4800
    samples_per_symbol = sample_rate // symbol_rate
    print(f"Samples per symbol: {samples_per_symbol}")

    # Raw differential demodulation (no filters)
    print("\n=== Raw differential demodulation ===")
    phases = differential_demod(iq, samples_per_symbol)

    # Sample at symbol centers
    symbol_phases = phases[samples_per_symbol // 2::samples_per_symbol]
    print(f"Extracted {len(symbol_phases)} symbols")

    # Check distribution
    print("\nPhase distribution:")
    near_p4 = np.sum(np.abs(symbol_phases - np.pi/4) < np.pi/8)
    near_3p4 = np.sum(np.abs(symbol_phases - 3*np.pi/4) < np.pi/8)
    near_m_p4 = np.sum(np.abs(symbol_phases + np.pi/4) < np.pi/8)
    near_m_3p4 = np.sum(np.abs(symbol_phases + 3*np.pi/4) < np.pi/8)
    print(f"  Near +π/4 (dibit 0): {near_p4} ({100*near_p4/len(symbol_phases):.1f}%)")
    print(f"  Near +3π/4 (dibit 1): {near_3p4} ({100*near_3p4/len(symbol_phases):.1f}%)")
    print(f"  Near -π/4 (dibit 2): {near_m_p4} ({100*near_m_p4/len(symbol_phases):.1f}%)")
    print(f"  Near -3π/4 (dibit 3): {near_m_3p4} ({100*near_m_3p4/len(symbol_phases):.1f}%)")

    # Convert to dibits
    dibits = np.array([dibit_from_phase(p) for p in symbol_phases], dtype=np.uint8)

    dibit_counts = [np.sum(dibits == i) for i in range(4)]
    print(f"\nDibit distribution: {dibit_counts}")
    print(f"  Dibit 0: {100*dibit_counts[0]/len(dibits):.1f}%")
    print(f"  Dibit 1: {100*dibit_counts[1]/len(dibits):.1f}%")
    print(f"  Dibit 2: {100*dibit_counts[2]/len(dibits):.1f}%")
    print(f"  Dibit 3: {100*dibit_counts[3]/len(dibits):.1f}%")

    # Search for sync
    print("\n=== Sync pattern search ===")
    print(f"Expected sync dibits: {list(SYNC_DIBITS)}")

    best_pos = -1
    best_errors = 24
    for i in range(len(dibits) - 24):
        errors = sum(1 for j in range(24) if dibits[i + j] != SYNC_DIBITS[j])
        if errors < best_errors:
            best_errors = errors
            best_pos = i

    print(f"\nBest sync match: position {best_pos} with {best_errors} errors")
    if best_pos >= 0:
        print(f"Dibits at match: {list(dibits[best_pos:best_pos+24])}")
        print(f"Phases at match: {[f'{p:.2f}' for p in symbol_phases[best_pos:best_pos+24]]}")

    # Also check first 50 symbols
    print("\n=== First 50 symbols ===")
    for i in range(min(50, len(dibits))):
        phase = symbol_phases[i]
        dibit = dibits[i]
        print(f"  Symbol {i:3d}: phase={phase:+.3f} dibit={dibit}")


if __name__ == '__main__':
    main()
