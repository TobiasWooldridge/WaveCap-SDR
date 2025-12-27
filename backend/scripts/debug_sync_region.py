#!/usr/bin/env python3
"""Debug phases around sync region."""

import wave
import numpy as np
from scipy import signal as sig
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from wavecapsdr.dsp.p25.c4fm import design_baseband_lpf, design_rrc_filter


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
        raw = wf.readframes(n_frames)
        samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
        iq = (samples[:, 0] + 1j * samples[:, 1]).astype(np.complex64) / 32768.0
    return rate, iq


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    args = parser.parse_args()

    sample_rate, iq = load_iq_wav(args.input)
    sps = sample_rate / 4800
    delay = int(np.ceil(sps))

    # Get filters
    lpf = design_baseband_lpf(sample_rate)
    rrc = design_rrc_filter(sps, num_taps=int(16 * sps) + 1, alpha=0.2)
    filter_delay = (len(lpf) - 1) // 2 + (len(rrc) - 1) // 2

    # Apply filters
    i, q = iq.real.astype(np.float32), iq.imag.astype(np.float32)
    i_filt = sig.lfilter(rrc, 1.0, sig.lfilter(lpf, 1.0, i))
    q_filt = sig.lfilter(rrc, 1.0, sig.lfilter(lpf, 1.0, q))

    # Demod raw
    phases_raw = np.zeros(len(i), dtype=np.float32)
    for x in range(delay, len(i)):
        diff = (i[x] + 1j*q[x]) * np.conj(i[x-delay] + 1j*q[x-delay])
        phases_raw[x] = np.arctan2(diff.imag, diff.real)

    # Demod filtered
    phases_filt = np.zeros(len(i_filt), dtype=np.float32)
    for x in range(delay, len(i_filt)):
        diff = (i_filt[x] + 1j*q_filt[x]) * np.conj(i_filt[x-delay] + 1j*q_filt[x-delay])
        phases_filt[x] = np.arctan2(diff.imag, diff.real)

    # Sample at symbol centers
    def sample_at_symbols(phases, start_sym, count):
        dibits = []
        for sym in range(start_sym, start_sym + count):
            idx = int(sym * sps + sps / 2)
            if idx < len(phases):
                phase = phases[idx]
                dibit = 1 if phase >= np.pi/2 else (0 if phase >= 0 else (2 if phase >= -np.pi/2 else 3))
                dibits.append((sym, phase, dibit))
        return dibits

    print(f"Expected sync: {list(SYNC_DIBITS)}")

    # Find sync in raw (we know it's around symbol 50)
    print("\n=== Raw demod around sync ===")
    for sym, phase, dibit in sample_at_symbols(phases_raw, 45, 35):
        expected = SYNC_DIBITS[sym - 50] if 50 <= sym < 74 else None
        match = "✓" if expected is not None and dibit == expected else (" " if expected is None else "✗")
        exp_str = f"exp={expected}" if expected is not None else "      "
        print(f"  sym {sym}: phase={phase:+.3f} dibit={dibit} {exp_str} {match}")

    print("\n=== Filtered demod around sync (without delay adjustment) ===")
    for sym, phase, dibit in sample_at_symbols(phases_filt, 45, 35):
        expected = SYNC_DIBITS[sym - 50] if 50 <= sym < 74 else None
        match = "✓" if expected is not None and dibit == expected else (" " if expected is None else "✗")
        exp_str = f"exp={expected}" if expected is not None else "      "
        print(f"  sym {sym}: phase={phase:+.3f} dibit={dibit} {exp_str} {match}")

    # With delay adjustment
    delay_symbols = filter_delay / sps
    adjusted_start = int(50 + delay_symbols)
    print(f"\n=== Filtered demod around sync (with delay = {filter_delay} samples = {delay_symbols:.1f} symbols) ===")
    for sym, phase, dibit in sample_at_symbols(phases_filt, adjusted_start - 5, 35):
        sync_sym = sym - adjusted_start
        expected = SYNC_DIBITS[sync_sym] if 0 <= sync_sym < 24 else None
        match = "✓" if expected is not None and dibit == expected else (" " if expected is None else "✗")
        exp_str = f"exp={expected}" if expected is not None else "      "
        print(f"  sym {sym} (sync+{sync_sym:+d}): phase={phase:+.3f} dibit={dibit} {exp_str} {match}")


if __name__ == '__main__':
    main()
