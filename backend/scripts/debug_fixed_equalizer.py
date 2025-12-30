#!/usr/bin/env python3
"""Test C4FM with fixed equalizer to isolate the error source.

If the error persists with fixed PLL=0 and gain=1, it's in the FM demod.
If the error goes away, it's in the equalizer adaptation.
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


def design_rrc(sps, taps=101, alpha=0.2):
    """Design RRC filter."""
    n = np.arange(taps) - (taps-1)/2
    t = n / sps
    h = np.zeros(taps)
    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = 1 - alpha + 4*alpha/np.pi
        elif abs(abs(ti) - 1/(4*alpha)) < 1e-10:
            h[i] = (alpha/np.sqrt(2)) * ((1+2/np.pi)*np.sin(np.pi/(4*alpha)) + (1-2/np.pi)*np.cos(np.pi/(4*alpha)))
        else:
            denom = np.pi * ti * (1 - (4*alpha*ti)**2)
            if abs(denom) > 1e-10:
                h[i] = (np.sin(np.pi*ti*(1-alpha)) + 4*alpha*ti*np.cos(np.pi*ti*(1+alpha))) / denom
    return (h / np.sum(h)).astype(np.float32)


def simple_c4fm_demod(iq: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Simplified C4FM demodulator with FIXED equalizer (PLL=0, gain=1)."""
    samples_per_symbol = sample_rate / 4800.0

    # Design filters
    lpf = signal.firwin(63, 5200, fs=sample_rate, window='hamming')
    rrc = design_rrc(samples_per_symbol)

    # Apply LPF
    i_lpf = signal.lfilter(lpf, 1.0, iq.real)
    q_lpf = signal.lfilter(lpf, 1.0, iq.imag)

    # Apply RRC
    i_rrc = signal.lfilter(rrc, 1.0, i_lpf)
    q_rrc = signal.lfilter(rrc, 1.0, q_lpf)

    # Create complex
    iq_filt = i_rrc + 1j * q_rrc

    # Differential FM demod over ~1 symbol period
    delay = int(round(samples_per_symbol))
    diff = iq_filt[delay:] * np.conj(iq_filt[:-delay])
    phase = np.arctan2(diff.imag, diff.real)

    # Symbol extraction - sample at symbol rate with NO equalizer
    n_symbols = len(phase) // int(samples_per_symbol)
    dibits = []
    soft = []

    for i in range(n_symbols):
        # Sample at symbol center (no interpolation, no equalization)
        idx = int(i * samples_per_symbol + samples_per_symbol / 2)
        if idx >= len(phase):
            break

        p = phase[idx]

        # Fixed decision boundaries (no PLL, no gain)
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
        soft.append(p * 4.0 / np.pi)  # Normalize to ±1, ±3

    return np.array(dibits, dtype=np.uint8), np.array(soft, dtype=np.float32)


def find_sync_positions(dibits: np.ndarray, threshold: int = 22) -> list[tuple[int, int]]:
    positions = []
    sync_len = len(FRAME_SYNC_DIBITS)
    for i in range(len(dibits) - sync_len):
        matches = np.sum(dibits[i:i+sync_len] == FRAME_SYNC_DIBITS)
        if matches >= threshold:
            positions.append((i, matches))
    return positions


def main(filepath: str):
    print(f"\n{'='*70}")
    print(f"Fixed Equalizer Test")
    print(f"{'='*70}")

    iq, sample_rate = load_baseband(filepath, int(10 * 50000))
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Samples: {len(iq)}")

    # Demod with fixed equalizer
    print(f"\n--- Simple C4FM (fixed equalizer) ---")
    dibits, soft = simple_c4fm_demod(iq, sample_rate)
    print(f"Symbols: {len(dibits)}")
    print(f"Soft std: {soft.std():.4f}")

    syncs = find_sync_positions(dibits, threshold=22)
    print(f"Syncs found (>=22/24): {len(syncs)}")

    if syncs:
        sync_pos, matches = syncs[0]
        print(f"\nFirst sync at position {sync_pos} ({matches}/24)")

        # Show NID dibits
        print(f"\nNID region (first 8):")
        nac_expected = [0, 3, 3, 1, 3, 0, 1, 3]  # NAC + DUID
        for j in range(8):
            pos = sync_pos + 24 + j
            if pos < len(dibits):
                d = dibits[pos]
                s = soft[pos]
                exp_d = nac_expected[j]
                exp_s = {0: 1.0, 1: 3.0, 2: -1.0, 3: -3.0}[exp_d]
                match = "✓" if d == exp_d else "✗"
                print(f"  [{j}] dibit={d} soft={s:+.3f} expected_d={exp_d} expected_s={exp_s:+.1f} {match}")

    # Expected NAC + DUID dibits
    nac_expected = [0, 3, 3, 1, 3, 0, 1, 3]  # NAC 0x3DC + DUID 0x7

    # Compare with full C4FM demodulator
    print(f"\n--- Full C4FM Demodulator ---")
    from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator
    demod = C4FMDemodulator(sample_rate=sample_rate)
    dibits2, soft2 = demod.demodulate(iq)
    print(f"Symbols: {len(dibits2)}")
    print(f"Soft std: {soft2.std():.4f}")

    syncs2 = find_sync_positions(dibits2, threshold=22)
    print(f"Syncs found (>=22/24): {len(syncs2)}")

    if syncs2:
        sync_pos2, matches2 = syncs2[0]
        print(f"\nFirst sync at position {sync_pos2} ({matches2}/24)")

        print(f"\nNID region (first 8):")
        for j in range(8):
            pos = sync_pos2 + 24 + j
            if pos < len(dibits2):
                d = dibits2[pos]
                s = soft2[pos]
                exp_d = nac_expected[j]
                exp_s = {0: 1.0, 1: 3.0, 2: -1.0, 3: -3.0}[exp_d]
                match = "✓" if d == exp_d else "✗"
                print(f"  [{j}] dibit={d} soft={s:+.3f} expected_d={exp_d} expected_s={exp_s:+.1f} {match}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("/Users/thw/SDRTrunk/recordings/20251227_121743_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav")
