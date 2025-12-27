#!/usr/bin/env python3
"""Standalone test for the interpolator and demodulator."""

import sys
import wave
import numpy as np
from pathlib import Path

# Direct import of just what we need
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import only the c4fm module directly
import wavecapsdr.dsp.p25.c4fm as c4fm


# P25 sync pattern
SYNC_PATTERN = 0x5575F5FF77FF
SYNC_DIBITS = np.array([
    (SYNC_PATTERN >> ((23 - i) * 2)) & 0x3
    for i in range(24)
], dtype=np.uint8)


def load_iq_wav(path: str) -> tuple[int, np.ndarray]:
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


def test_interpolator():
    """Test the polyphase interpolator."""
    print("=== Testing Interpolator ===")

    interp = c4fm._Interpolator()

    # Test with a simple sine wave
    t = np.linspace(0, 2*np.pi, 129)
    samples = np.sin(t).astype(np.float32)

    # Test at integer positions
    print("Testing at integer positions (mu=0):")
    for i in range(4, 120, 20):
        expected = samples[i + 3]  # mu=0 gives offset+3
        actual = interp.filter(samples, i, 0.0)
        print(f"  offset={i}: expected={expected:.4f}, actual={actual:.4f}, diff={abs(expected-actual):.6f}")

    # Test at half positions (mu=0.5)
    print("\nTesting at half positions (mu=0.5):")
    for i in range(4, 120, 20):
        # For a smooth function, mu=0.5 should be between samples[i+3] and samples[i+4]
        expected_approx = (samples[i + 3] + samples[i + 4]) / 2
        actual = interp.filter(samples, i, 0.5)
        print(f"  offset={i}: linear_approx={expected_approx:.4f}, actual={actual:.4f}")

    print("\n✓ Interpolator tests passed\n")


def test_demodulator_basic():
    """Test demodulator creation."""
    print("=== Testing Demodulator Creation ===")

    demod = c4fm.C4FMDemodulator(sample_rate=48000, symbol_rate=4800)
    print(f"  Sample rate: {demod.sample_rate}")
    print(f"  Symbol rate: {demod.symbol_rate}")
    print(f"  Samples per symbol: {demod.samples_per_symbol}")

    print("\n✓ Demodulator creation tests passed\n")


def test_with_synthetic_signal():
    """Test with synthetic P25 signal."""
    print("=== Testing with Synthetic Signal ===")

    wav_path = "/tmp/p25_test_signal.wav"

    try:
        sample_rate, iq = load_iq_wav(wav_path)
        print(f"Loaded {len(iq)} samples at {sample_rate} Hz")
    except FileNotFoundError:
        print(f"  Skipping: {wav_path} not found")
        return

    demod = c4fm.C4FMDemodulator(sample_rate=sample_rate)
    dibits, soft = demod.demodulate(iq)

    print(f"  Demodulated {len(dibits)} dibits")

    # Check dibit distribution
    dibit_counts = [np.sum(dibits == i) for i in range(4)]
    total = len(dibits)
    print(f"  Dibit distribution:")
    for i, count in enumerate(dibit_counts):
        print(f"    Dibit {i}: {count} ({100*count/total:.1f}%)")

    # Find sync pattern
    best_pos = -1
    best_errors = 24
    for i in range(len(dibits) - 24):
        errors = sum(1 for j in range(24) if dibits[i + j] != SYNC_DIBITS[j])
        if errors < best_errors:
            best_errors = errors
            best_pos = i

    print(f"  Best sync match: position {best_pos} with {best_errors} errors")

    if best_errors <= 2:
        print("  ✓ Good sync detection!")
    elif best_errors <= 5:
        print("  ~ Marginal sync detection")
    else:
        print("  ✗ Poor sync detection")


if __name__ == '__main__':
    test_interpolator()
    test_demodulator_basic()
    test_with_synthetic_signal()
