#!/usr/bin/env python3
"""Trace equalizer PLL/gain behavior around sync->NID transition.

Hypothesis: The equalizer PLL is building up an offset during the sync region
(which ends with 5 consecutive -3 symbols) that then incorrectly affects
the NID symbols.
"""

import sys
import wave
import numpy as np
from scipy import signal

sys.path.insert(0, '/Users/thw/Projects/WaveCap-SDR/backend')

# Frame sync as dibit array (24 dibits)
FRAME_SYNC_DIBITS = np.array([
    1, 1, 1, 1, 1, 3, 1, 1, 3, 3, 1, 1, 3, 3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3
], dtype=np.uint8)

# Expected SA-GRN NAC dibits + DUID
EXPECTED_NAC_DIBITS = [0, 3, 3, 1, 3, 0]
EXPECTED_DUID_DIBITS = [0, 1]


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


def design_lpf(sample_rate: float) -> np.ndarray:
    return signal.firwin(63, 5200, fs=sample_rate, window='hamming').astype(np.float32)


def design_rrc(sps: float, num_taps: int = 101, alpha: float = 0.2) -> np.ndarray:
    if num_taps % 2 == 0:
        num_taps += 1
    n = np.arange(num_taps) - (num_taps - 1) / 2
    t = n / sps
    h = np.zeros(num_taps, dtype=np.float64)
    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = 1 - alpha + 4 * alpha / np.pi
        elif abs(ti) == 1 / (4 * alpha):
            h[i] = (alpha / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
            )
        else:
            num = np.sin(np.pi * ti * (1 - alpha)) + 4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha))
            den = np.pi * ti * (1 - (4 * alpha * ti) ** 2)
            if abs(den) > 1e-10:
                h[i] = num / den
    h = h / np.sum(h)
    return h.astype(np.float32)


def differential_fm_demod(iq: np.ndarray, delay: int) -> np.ndarray:
    """FM demod with specific sample delay."""
    diff = iq[delay:] * np.conj(iq[:-delay])
    return np.arctan2(diff.imag, diff.real).astype(np.float32)


def extract_symbols_with_equalizer_trace(phases: np.ndarray, sps: float):
    """Extract symbols with full equalizer trace."""
    initial_pll = 0.0
    initial_gain = 1.219  # SDRTrunk's initial gain

    symbols = []
    sample_point = sps  # Start at first symbol

    for i in range(int(len(phases) / sps) - 10):
        idx = int(i * sps + sps / 2)  # Sample at symbol center
        if idx >= len(phases):
            break

        phase = phases[idx]
        soft_rad = (phase + initial_pll) * initial_gain
        soft_norm = soft_rad * 4.0 / np.pi

        # Decision
        boundary = np.pi / 2.0
        if soft_rad >= boundary:
            dibit = 1
        elif soft_rad >= 0:
            dibit = 0
        elif soft_rad >= -boundary:
            dibit = 2
        else:
            dibit = 3

        symbols.append({
            'idx': idx,
            'sym_idx': i,
            'phase': phase,
            'pll': initial_pll,
            'gain': initial_gain,
            'soft_rad': soft_rad,
            'soft_norm': soft_norm,
            'dibit': dibit,
        })

    return symbols


def find_sync(symbols: list) -> list[int]:
    """Find sync positions in symbol list."""
    syncs = []
    for i in range(len(symbols) - 32):
        matches = sum(1 for j, exp in enumerate(FRAME_SYNC_DIBITS)
                     if symbols[i + j]['dibit'] == exp)
        if matches >= 22:
            syncs.append(i)
    return syncs


def main(filepath: str):
    print(f"\n{'='*70}")
    print("Equalizer Trace at Sync->NID Transition")
    print(f"{'='*70}")

    iq, sample_rate = load_baseband(filepath, int(10 * 50000))
    print(f"Sample rate: {sample_rate} Hz")

    sps = sample_rate / 4800.0
    print(f"Samples per symbol: {sps:.3f}")

    # Process through filter chain
    lpf = design_lpf(sample_rate)
    rrc = design_rrc(sps)

    i_lpf = signal.lfilter(lpf, 1.0, iq.real)
    q_lpf = signal.lfilter(lpf, 1.0, iq.imag)
    i_rrc = signal.lfilter(rrc, 1.0, i_lpf)
    q_rrc = signal.lfilter(rrc, 1.0, q_lpf)
    iq_filt = (i_rrc + 1j * q_rrc).astype(np.complex64)

    # Try different FM demod delays
    print(f"\n{'='*70}")
    print("Testing different FM demod delays")
    print(f"{'='*70}")

    for delay in [1, int(sps), int(sps) + 1]:
        print(f"\n--- FM demod delay = {delay} samples ---")

        phases = differential_fm_demod(iq_filt, delay)
        symbols = extract_symbols_with_equalizer_trace(phases, sps)

        syncs = find_sync(symbols)
        print(f"Syncs found (>=22/24): {len(syncs)}")

        if syncs:
            sync_pos = syncs[0]
            print(f"\nFirst sync at symbol {sync_pos}")

            # Show symbols around sync->NID boundary
            print(f"\nSymbols at sync->NID boundary (last 4 sync + first 8 NID):")
            print(f"{'Rel':>4} {'Dibit':>6} {'Expect':>7} {'Phase':>8} {'SoftN':>8} {'Match':>6}")

            all_expected = list(FRAME_SYNC_DIBITS[-4:]) + EXPECTED_NAC_DIBITS + EXPECTED_DUID_DIBITS

            for i, exp in enumerate(all_expected):
                rel = i - 4  # Relative to NID start
                sym = symbols[sync_pos + 20 + i]

                region = "SYNC" if rel < 0 else "NID"
                match = "ok" if sym['dibit'] == exp else "ERR"
                exp_soft = {0: 1.0, 1: 3.0, 2: -1.0, 3: -3.0}[exp]

                print(f"{rel:>4} {sym['dibit']:>6} {exp:>4}({exp_soft:+.0f}) "
                      f"{sym['phase']:>+8.3f} {sym['soft_norm']:>+8.2f} {match:>6}")

    # Now test with adaptive PLL
    print(f"\n{'='*70}")
    print("Testing with Adaptive PLL During Sync Region")
    print(f"{'='*70}")

    phases = differential_fm_demod(iq_filt, int(sps))
    symbols = extract_symbols_with_equalizer_trace(phases, sps)

    syncs = find_sync(symbols)
    if syncs:
        sync_pos = syncs[0]

        # Simulate PLL adaptation during sync
        pll = 0.0
        gain = 1.219
        pll_gain = 0.15

        print(f"\nAdaptive PLL during sync region:")
        print(f"{'Pos':>4} {'Dibit':>6} {'Expect':>7} {'Phase':>8} {'PLL':>8} {'Corrected':>10}")

        for i, exp in enumerate(FRAME_SYNC_DIBITS):
            sym = symbols[sync_pos + i]
            expected_soft = {0: 1.0, 1: 3.0, 2: -1.0, 3: -3.0}[exp]

            # Compute corrected soft value
            corrected_rad = (sym['phase'] + pll) * gain
            corrected_norm = corrected_rad * 4.0 / np.pi

            # Update PLL based on error
            error = expected_soft - corrected_norm
            pll_update = error * (np.pi / 4.0) * pll_gain / gain

            if i >= 20:  # Show last 4
                print(f"{i:>4} {sym['dibit']:>6} {exp:>4}({expected_soft:+.0f}) "
                      f"{sym['phase']:>+8.3f} {pll:>+8.4f} {corrected_norm:>+10.2f}")

            pll += pll_update

        print(f"\nFinal PLL after sync: {pll:+.4f}")

        # Now apply this PLL to NID
        print(f"\nNID with accumulated PLL = {pll:+.4f}:")
        print(f"{'Pos':>4} {'Dibit':>6} {'Expect':>7} {'Phase':>8} {'Corrected':>10} {'NewDibit':>9}")

        for i, exp in enumerate(EXPECTED_NAC_DIBITS + EXPECTED_DUID_DIBITS):
            sym = symbols[sync_pos + 24 + i]
            expected_soft = {0: 1.0, 1: 3.0, 2: -1.0, 3: -3.0}[exp]

            corrected_rad = (sym['phase'] + pll) * gain
            corrected_norm = corrected_rad * 4.0 / np.pi

            # Decide new dibit
            boundary = np.pi / 2.0
            if corrected_rad >= boundary:
                new_dibit = 1
            elif corrected_rad >= 0:
                new_dibit = 0
            elif corrected_rad >= -boundary:
                new_dibit = 2
            else:
                new_dibit = 3

            match = "ok" if new_dibit == exp else "ERR"
            region = "NAC" if i < 6 else "DUID"
            print(f"{i:>4} {sym['dibit']:>6} {exp:>4}({expected_soft:+.0f}) "
                  f"{sym['phase']:>+8.3f} {corrected_norm:>+10.2f} {new_dibit:>6} {match}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("/Users/thw/SDRTrunk/recordings/20251227_121743_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav")
