#!/usr/bin/env python3
"""Debug filter phase response."""

import wave
import numpy as np
from scipy import signal as sig
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from wavecapsdr.dsp.p25.c4fm import design_baseband_lpf, design_rrc_filter


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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input WAV file')
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    sample_rate, iq = load_iq_wav(args.input)
    print(f"Loaded {len(iq)} samples at {sample_rate} Hz")

    samples_per_symbol = sample_rate / 4800

    # Get filter coefficients
    baseband_lpf = design_baseband_lpf(sample_rate)
    rrc_filter = design_rrc_filter(samples_per_symbol, num_taps=int(16 * samples_per_symbol) + 1, alpha=0.2)

    print(f"\nFilter lengths:")
    print(f"  Baseband LPF: {len(baseband_lpf)} taps")
    print(f"  RRC filter: {len(rrc_filter)} taps")

    # Check filter gains
    print(f"\nFilter gains:")
    print(f"  Baseband LPF sum: {np.sum(baseband_lpf):.4f}")
    print(f"  RRC filter sum: {np.sum(rrc_filter):.4f}")

    # Check filter symmetry
    print(f"\nFilter symmetry (first 5 vs last 5):")
    print(f"  Baseband LPF: {baseband_lpf[:5]} vs {baseband_lpf[-5:][::-1]}")
    print(f"  RRC filter: {rrc_filter[:5]} vs {rrc_filter[-5:][::-1]}")

    i = iq.real.astype(np.float32)
    q = iq.imag.astype(np.float32)

    # Test 1: Just baseband LPF
    print("\n=== Test 1: Just baseband LPF ===")
    i_lpf = sig.lfilter(baseband_lpf, 1.0, i)
    q_lpf = sig.lfilter(baseband_lpf, 1.0, q)

    # Check phase preservation at symbol center
    delay_lpf = (len(baseband_lpf) - 1) // 2
    print(f"  LPF group delay: {delay_lpf} samples")

    for sym in range(10, 20):
        idx_raw = int(sym * samples_per_symbol + samples_per_symbol / 2)
        idx_filt = idx_raw + delay_lpf
        if idx_filt < len(i_lpf):
            phase_raw = np.arctan2(q[idx_raw], i[idx_raw])
            phase_filt = np.arctan2(q_lpf[idx_filt], i_lpf[idx_filt])
            print(f"  Symbol {sym}: raw={phase_raw:+.3f} filt={phase_filt:+.3f} diff={phase_filt-phase_raw:+.3f}")

    # Test 2: LPF + RRC
    print("\n=== Test 2: LPF + RRC ===")
    i_rrc = sig.lfilter(rrc_filter, 1.0, i_lpf)
    q_rrc = sig.lfilter(rrc_filter, 1.0, q_lpf)

    delay_rrc = (len(rrc_filter) - 1) // 2
    total_delay = delay_lpf + delay_rrc
    print(f"  RRC group delay: {delay_rrc} samples")
    print(f"  Total delay: {total_delay} samples")

    for sym in range(20, 30):
        idx_raw = int(sym * samples_per_symbol + samples_per_symbol / 2)
        idx_filt = idx_raw + total_delay
        if idx_filt < len(i_rrc):
            phase_raw = np.arctan2(q[idx_raw], i[idx_raw])
            phase_filt = np.arctan2(q_rrc[idx_filt], i_rrc[idx_filt])
            # Also check magnitude
            mag_raw = np.sqrt(i[idx_raw]**2 + q[idx_raw]**2)
            mag_filt = np.sqrt(i_rrc[idx_filt]**2 + q_rrc[idx_filt]**2)
            print(f"  Symbol {sym}: raw={phase_raw:+.3f} filt={phase_filt:+.3f} mag_filt={mag_filt:.3f}")

    # Test 3: Differential demod after filtering
    print("\n=== Test 3: Differential demod comparison ===")
    symbol_delay = int(np.ceil(samples_per_symbol))

    # Raw differential
    phases_raw = np.zeros(len(i), dtype=np.float32)
    for x in range(symbol_delay, len(i)):
        diff = (i[x] + 1j*q[x]) * np.conj(i[x-symbol_delay] + 1j*q[x-symbol_delay])
        phases_raw[x] = np.arctan2(diff.imag, diff.real)

    # Filtered differential
    phases_filt = np.zeros(len(i_rrc), dtype=np.float32)
    for x in range(symbol_delay, len(i_rrc)):
        diff = (i_rrc[x] + 1j*q_rrc[x]) * np.conj(i_rrc[x-symbol_delay] + 1j*q_rrc[x-symbol_delay])
        phases_filt[x] = np.arctan2(diff.imag, diff.real)

    # Compare at symbol centers
    print("\nDifferential phase comparison (symbol centers):")
    for sym in range(50, 60):
        idx_raw = int(sym * samples_per_symbol + samples_per_symbol / 2)
        idx_filt = idx_raw + total_delay
        if idx_filt < len(phases_filt):
            print(f"  Symbol {sym}: raw={phases_raw[idx_raw]:+.3f} filt={phases_filt[idx_filt]:+.3f}")


if __name__ == '__main__':
    main()
