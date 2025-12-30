#!/usr/bin/env python3
"""Check how many symbol_indices are valid after demodulate()."""

import sys
import wave
import numpy as np

sys.path.insert(0, '/Users/thw/Projects/WaveCap-SDR/backend')


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


def main(filepath: str):
    print("=" * 70)
    print("Symbol Indices Validity Check")
    print("=" * 70)

    iq, sample_rate = load_baseband(filepath, int(10 * 50000))
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Samples: {len(iq)}")
    samples_per_symbol = sample_rate / 4800.0
    print(f"Samples per symbol: {samples_per_symbol:.3f}")

    # Use the symbol recovery function directly to see indices
    from wavecapsdr.dsp.p25.c4fm import _symbol_recovery_jit, _interpolator
    from scipy import signal

    # Create filters (simplified from C4FMDemodulator)
    lpf = signal.firwin(63, 5200, fs=sample_rate, window='hamming').astype(np.float32)

    # RRC filter design
    num_taps = 101
    alpha = 0.2
    n = np.arange(num_taps) - (num_taps - 1) / 2
    t = n / samples_per_symbol
    h = np.zeros(num_taps, dtype=np.float64)
    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = 1 - alpha + 4 * alpha / np.pi
        elif abs(ti) == 1 / (4 * alpha):
            h[i] = (alpha / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
            )
        else:
            num = np.sin(np.pi * ti * (1 - alpha)) + 4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha))
            den = np.pi * ti * (1 - (4 * alpha * ti) ** 2)
            if abs(den) > 1e-10:
                h[i] = num / den
    h = h / np.sum(h)
    rrc = h.astype(np.float32)

    # Process signal
    i_lpf = signal.lfilter(lpf, 1.0, iq.real)
    q_lpf = signal.lfilter(lpf, 1.0, iq.imag)
    i_rrc = signal.lfilter(rrc, 1.0, i_lpf)
    q_rrc = signal.lfilter(rrc, 1.0, q_lpf)
    iq_filt = (i_rrc + 1j * q_rrc).astype(np.complex64)

    # FM demod
    delay = 1
    diff = iq_filt[delay:] * np.conj(iq_filt[:-delay])
    phases = np.arctan2(diff.imag, diff.real).astype(np.float32)

    # Symbol recovery
    buffer = np.zeros(2048, dtype=np.float32)
    dibits, soft, symbol_indices, buf_ptr, samp_pt = _symbol_recovery_jit(
        phases,
        buffer,
        0,
        samples_per_symbol,
        samples_per_symbol,
        0.0,  # pll
        1.219,  # gain
        _interpolator.TAPS,
    )

    print(f"\nSymbol recovery results:")
    print(f"  Total symbols: {len(dibits)}")
    print(f"  Buffer pointer: {buf_ptr}")

    # Check symbol_indices validity
    valid_count = np.sum(symbol_indices >= 0)
    invalid_count = np.sum(symbol_indices < 0)
    print(f"\nSymbol indices:")
    print(f"  Valid (>= 0): {valid_count}")
    print(f"  Invalid (< 0): {invalid_count}")
    print(f"  Valid percentage: {100*valid_count/len(symbol_indices):.1f}%")

    if valid_count > 0:
        valid_indices = symbol_indices[symbol_indices >= 0]
        print(f"  Valid range: [{valid_indices.min()}, {valid_indices.max()}]")

    # Where are the valid indices?
    valid_positions = np.where(symbol_indices >= 0)[0]
    if len(valid_positions) > 0:
        print(f"  First valid at symbol position: {valid_positions[0]}")
        print(f"  Last valid at symbol position: {valid_positions[-1]}")

    # Check the sync positions
    print("\nSync position check:")
    expected_sync_positions = [19603, 20467, 21331, 22195]
    for pos in expected_sync_positions:
        if pos < len(symbol_indices):
            idx = symbol_indices[pos]
            status = "VALID" if idx >= 0 else "INVALID"
            print(f"  Position {pos}: symbol_indices={idx} -> {status}")

    # Calculate how many shifts occurred
    print("\nBuffer shift analysis:")
    half_buffer = 1024
    samples_processed = len(phases)
    approx_shifts = max(0, (samples_processed - 2048) // half_buffer)
    print(f"  Samples processed: {samples_processed}")
    print(f"  Approximate shifts: {approx_shifts}")
    print(f"  Symbols that remain valid after all shifts: ~{int(half_buffer / samples_per_symbol)}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("/Users/thw/SDRTrunk/recordings/20251227_121743_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav")
