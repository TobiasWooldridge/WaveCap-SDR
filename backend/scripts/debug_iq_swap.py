#!/usr/bin/env python3
"""Test if I/Q channels need to be swapped."""

import sys
import wave
import numpy as np
from pathlib import Path


def load_iq_wav(path: str) -> tuple[int, np.ndarray]:
    try:
        import soundfile as sf
        samples, rate = sf.read(path)
        print(f"  Loaded with soundfile: {samples.shape}, {samples.dtype}")
        iq = (samples[:, 0] + 1j * samples[:, 1]).astype(np.complex64)
        return rate, iq
    except ImportError:
        pass

    from scipy.io import wavfile
    rate, samples = wavfile.read(path)
    print(f"  Loaded with scipy: {samples.shape}, {samples.dtype}")

    if samples.dtype == np.int16:
        scale = 32768.0
    elif samples.dtype == np.int32:
        scale = 2147483648.0
    elif samples.dtype == np.float32 or samples.dtype == np.float64:
        scale = 1.0
    else:
        raise ValueError(f"Unsupported dtype: {samples.dtype}")

    iq = (samples[:, 0] + 1j * samples[:, 1]).astype(np.complex64) / scale
    return rate, iq


def test_differential(i, q, symbol_delay=5):
    """Test differential demodulation and return sync match errors."""
    phases = np.zeros(len(i), dtype=np.float32)
    for x in range(symbol_delay, len(i)):
        i_prev, q_prev = i[x - symbol_delay], q[x - symbol_delay]
        i_curr, q_curr = i[x], q[x]
        diff_i = i_curr * i_prev + q_curr * q_prev
        diff_q = q_curr * i_prev - i_curr * q_prev
        phases[x] = np.arctan2(diff_q, diff_i)

    # Decimate to symbol rate
    symbol_phases = phases[symbol_delay::symbol_delay]

    # Convert to dibits
    boundary = np.pi / 2
    dibits = np.zeros(len(symbol_phases), dtype=np.uint8)
    for i, phase in enumerate(symbol_phases):
        if phase >= boundary:
            dibits[i] = 1
        elif phase >= 0:
            dibits[i] = 0
        elif phase >= -boundary:
            dibits[i] = 2
        else:
            dibits[i] = 3

    # P25 sync pattern
    SYNC_PATTERN = 0x5575F5FF77FF
    sync_dibits = np.array([
        (SYNC_PATTERN >> ((23 - j) * 2)) & 0x3
        for j in range(24)
    ], dtype=np.uint8)

    # Find best sync match
    best_errors = 24
    for i in range(min(20000, len(dibits) - 24)):
        errors = sum(1 for j in range(24) if dibits[i + j] != sync_dibits[j])
        if errors < best_errors:
            best_errors = errors

    return best_errors


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    sample_rate, iq = load_iq_wav(args.input)
    iq = iq[:100000]  # First 100k samples
    print(f"Testing {len(iq)} samples at {sample_rate} Hz")

    # Test configurations
    configs = [
        ("Normal (I=L, Q=R)", iq.real, iq.imag),
        ("Swapped (I=R, Q=L)", iq.imag, iq.real),
        ("I inverted", -iq.real, iq.imag),
        ("Q inverted", iq.real, -iq.imag),
        ("Both inverted", -iq.real, -iq.imag),
        ("Swap + I inverted", -iq.imag, iq.real),
        ("Swap + Q inverted", iq.imag, -iq.real),
        ("Swap + both inverted", -iq.imag, -iq.real),
    ]

    print("\nConfiguration                  Best sync errors")
    print("-" * 50)
    for name, i, q in configs:
        errors = test_differential(i, q)
        marker = " <-- BEST" if errors <= 4 else ""
        print(f"{name:30s} {errors:3d}{marker}")


if __name__ == '__main__':
    main()
