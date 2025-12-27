#!/usr/bin/env python3
"""Debug demodulator pipeline stage by stage."""

import sys
import importlib.util
import wave
import numpy as np
from pathlib import Path

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


def analyze_phase_distribution(phases: np.ndarray, name: str):
    """Analyze and print phase distribution."""
    print(f"\n{name}:")
    print(f"  Range: [{np.min(phases):.3f}, {np.max(phases):.3f}] rad")
    print(f"  Mean: {np.mean(phases):.3f}, Std: {np.std(phases):.3f}")

    # Check how many samples are near each ideal C4FM position
    ideal_phases = [np.pi/4, 3*np.pi/4, -np.pi/4, -3*np.pi/4]
    for ideal in ideal_phases:
        near = np.sum(np.abs(phases - ideal) < np.pi/8)
        pct = near / len(phases) * 100
        print(f"  Near {ideal:.2f} rad: {near} ({pct:.1f}%)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    sample_rate, iq = load_iq_wav(args.input)
    print(f"Loaded {len(iq)} samples at {sample_rate} Hz")

    # Take first 50000 samples for analysis
    iq = iq[:50000]

    # Test both I/Q orderings
    for swap in [False, True]:
        if swap:
            print("\n" + "=" * 60)
            print("TESTING WITH I/Q SWAPPED")
            print("=" * 60)
            i = iq.imag
            q = iq.real
        else:
            print("\n" + "=" * 60)
            print("TESTING NORMAL I/Q")
            print("=" * 60)
            i = iq.real
            q = iq.imag

        # Stage 1: Raw differential demod (no filtering)
    print("\n=== Stage 1: Raw Differential Demod (no filters) ===")
    symbol_delay = 5  # samples per symbol
    phases_raw = np.zeros(len(i), dtype=np.float32)
    for x in range(symbol_delay, len(i)):
        i_prev, q_prev = i[x - symbol_delay], q[x - symbol_delay]
        i_curr, q_curr = i[x], q[x]
        # s[n] * conj(s[n-delay])
        diff_i = i_curr * i_prev + q_curr * q_prev
        diff_q = q_curr * i_prev - i_curr * q_prev
        phases_raw[x] = np.arctan2(diff_q, diff_i)
    analyze_phase_distribution(phases_raw[symbol_delay:], "Raw differential")

    # Stage 2: With baseband LPF
    print("\n=== Stage 2: With Baseband LPF ===")
    from scipy import signal
    # Design 6kHz LPF at 24kHz sample rate
    nyq = sample_rate / 2
    lpf = signal.firwin(65, 6000 / nyq, window='hamming')
    i_filt = signal.lfilter(lpf, 1.0, i)
    q_filt = signal.lfilter(lpf, 1.0, q)

    phases_lpf = np.zeros(len(i_filt), dtype=np.float32)
    for x in range(symbol_delay, len(i_filt)):
        i_prev, q_prev = i_filt[x - symbol_delay], q_filt[x - symbol_delay]
        i_curr, q_curr = i_filt[x], q_filt[x]
        diff_i = i_curr * i_prev + q_curr * q_prev
        diff_q = q_curr * i_prev - i_curr * q_prev
        phases_lpf[x] = np.arctan2(diff_q, diff_i)
    analyze_phase_distribution(phases_lpf[symbol_delay:], "With baseband LPF")

    # Stage 3: With RRC filter
    print("\n=== Stage 3: With RRC Filter ===")
    sps = sample_rate / 4800
    rrc = c4fm_module.design_rrc_filter(sps, num_taps=int(16*sps)+1, alpha=0.2)
    i_rrc = signal.lfilter(rrc, 1.0, i_filt)
    q_rrc = signal.lfilter(rrc, 1.0, q_filt)

    phases_rrc = np.zeros(len(i_rrc), dtype=np.float32)
    for x in range(symbol_delay, len(i_rrc)):
        i_prev, q_prev = i_rrc[x - symbol_delay], q_rrc[x - symbol_delay]
        i_curr, q_curr = i_rrc[x], q_rrc[x]
        diff_i = i_curr * i_prev + q_curr * q_prev
        diff_q = q_curr * i_prev - i_curr * q_prev
        phases_rrc[x] = np.arctan2(diff_q, diff_i)
    analyze_phase_distribution(phases_rrc[symbol_delay:], "With RRC filter")

    # Stage 4: Sample at symbol rate (every 5th sample)
    print("\n=== Stage 4: Symbol-rate samples ===")
    symbol_phases = phases_rrc[symbol_delay::5]
    analyze_phase_distribution(symbol_phases, "Symbol-rate (every 5th)")

    # Stage 5: Check sync pattern correlation in symbols
    print("\n=== Stage 5: Sync Pattern Search ===")
    SYNC_PATTERN = 0x5575F5FF77FF
    sync_dibits = np.array([
        (SYNC_PATTERN >> ((23 - i) * 2)) & 0x3
        for i in range(24)
    ], dtype=np.uint8)
    sync_phases = np.array([
        3*np.pi/4 if d == 1 else -3*np.pi/4 for d in sync_dibits
    ])

    # Convert symbol_phases to dibits
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

    # Search for sync
    best_pos = -1
    best_errors = 24
    for i in range(len(dibits) - 24):
        errors = sum(1 for j in range(24) if dibits[i + j] != sync_dibits[j])
        if errors < best_errors:
            best_errors = errors
            best_pos = i

    print(f"Best sync match: position {best_pos} with {best_errors} errors")
    if best_pos >= 0:
        print(f"  Expected: {list(sync_dibits)}")
        print(f"  Got:      {list(dibits[best_pos:best_pos+24])}")
        print(f"  Phases:   {[f'{p:.2f}' for p in symbol_phases[best_pos:best_pos+24]]}")


if __name__ == '__main__':
    main()
