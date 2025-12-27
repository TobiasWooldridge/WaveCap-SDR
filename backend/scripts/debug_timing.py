#!/usr/bin/env python3
"""Debug timing and group delay issues."""

import wave
import numpy as np
from scipy import signal as sig
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from wavecapsdr.dsp.p25.c4fm import design_baseband_lpf, design_rrc_filter


def load_iq_wav(path: str) -> tuple[int, np.ndarray]:
    with wave.open(path, 'rb') as wf:
        rate = wf.getframerate()
        n_frames = wf.getnframes()
        n_channels = wf.getnchannels()
        if n_channels != 2:
            raise ValueError(f"Expected stereo WAV")
        raw = wf.readframes(n_frames)
        samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
        iq = (samples[:, 0] + 1j * samples[:, 1]).astype(np.complex64) / 32768.0
    return rate, iq


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    sample_rate, iq = load_iq_wav(args.input)
    samples_per_symbol = sample_rate / 4800
    symbol_delay = int(np.ceil(samples_per_symbol))

    print(f"Sample rate: {sample_rate} Hz")
    print(f"Samples per symbol: {samples_per_symbol}")
    print(f"Symbol delay: {symbol_delay}")

    # Get filter specs
    lpf = design_baseband_lpf(sample_rate)
    rrc = design_rrc_filter(samples_per_symbol, num_taps=int(16 * samples_per_symbol) + 1, alpha=0.2)

    lpf_delay = (len(lpf) - 1) // 2
    rrc_delay = (len(rrc) - 1) // 2
    total_filter_delay = lpf_delay + rrc_delay

    print(f"\nFilter delays:")
    print(f"  LPF: {len(lpf)} taps, delay = {lpf_delay} samples")
    print(f"  RRC: {len(rrc)} taps, delay = {rrc_delay} samples")
    print(f"  Total: {total_filter_delay} samples = {total_filter_delay / samples_per_symbol:.2f} symbols")

    # Apply filters
    i = iq.real.astype(np.float32)
    q = iq.imag.astype(np.float32)

    i_lpf = sig.lfilter(lpf, 1.0, i)
    q_lpf = sig.lfilter(lpf, 1.0, q)
    i_rrc = sig.lfilter(rrc, 1.0, i_lpf)
    q_rrc = sig.lfilter(rrc, 1.0, q_lpf)

    # Differential demod - raw
    phases_raw = np.zeros(len(i), dtype=np.float32)
    for x in range(symbol_delay, len(i)):
        diff = (i[x] + 1j*q[x]) * np.conj(i[x-symbol_delay] + 1j*q[x-symbol_delay])
        phases_raw[x] = np.arctan2(diff.imag, diff.real)

    # Differential demod - filtered
    phases_filt = np.zeros(len(i_rrc), dtype=np.float32)
    for x in range(symbol_delay, len(i_rrc)):
        diff = (i_rrc[x] + 1j*q_rrc[x]) * np.conj(i_rrc[x-symbol_delay] + 1j*q_rrc[x-symbol_delay])
        phases_filt[x] = np.arctan2(diff.imag, diff.real)

    # Compare at known symbol positions (e.g., idle symbols at start)
    print("\n=== Symbol comparison at different offsets ===")

    # Try different sample offsets
    for offset in range(0, 20, 2):
        print(f"\nOffset = {offset} samples:")
        print("  Raw phases (symbols 5-15):")
        for sym in range(5, 15):
            idx = int(sym * samples_per_symbol + samples_per_symbol / 2) + offset
            if idx < len(phases_raw):
                phase = phases_raw[idx]
                dibit = 1 if phase >= np.pi/2 else (0 if phase >= 0 else (2 if phase >= -np.pi/2 else 3))
                print(f"    sym {sym}: {phase:+.3f} -> dibit {dibit}")

        print("  Filtered phases (symbols 5-15):")
        for sym in range(5, 15):
            idx = int(sym * samples_per_symbol + samples_per_symbol / 2) + offset
            if idx < len(phases_filt):
                phase = phases_filt[idx]
                dibit = 1 if phase >= np.pi/2 else (0 if phase >= 0 else (2 if phase >= -np.pi/2 else 3))
                print(f"    sym {sym}: {phase:+.3f} -> dibit {dibit}")


if __name__ == '__main__':
    main()
