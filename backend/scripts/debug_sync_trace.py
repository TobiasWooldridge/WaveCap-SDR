#!/usr/bin/env python3
"""Trace sync detection step by step."""

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
    print("Sync Detection Trace")
    print("=" * 70)

    iq, sample_rate = load_baseband(filepath, int(10 * 50000))
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Samples: {len(iq)}")

    from wavecapsdr.dsp.p25.c4fm import (
        C4FMDemodulator,
        _SoftSyncDetector,
        _symbol_recovery_jit,
        _interpolator,
    )
    from scipy import signal

    # Create demodulator
    demod = C4FMDemodulator(sample_rate=sample_rate)
    samples_per_symbol = demod.samples_per_symbol

    # Process through filters
    i = iq.real.astype(np.float32)
    q = iq.imag.astype(np.float32)

    i_lpf = signal.lfilter(demod._baseband_lpf, 1.0, i)
    q_lpf = signal.lfilter(demod._baseband_lpf, 1.0, q)
    i_rrc = signal.lfilter(demod._rrc_filter, 1.0, i_lpf)
    q_rrc = signal.lfilter(demod._rrc_filter, 1.0, q_lpf)

    # FM demod
    phases = demod._fm_demod.demodulate(i_rrc.astype(np.float32), q_rrc.astype(np.float32))

    print(f"\nPhases: {len(phases)}")
    print(f"Phase range: [{phases.min():.3f}, {phases.max():.3f}]")

    # Symbol recovery with demodulator's buffer
    buffer = demod._buffer.copy()
    print(f"Buffer size: {len(buffer)}")

    dibits, soft_symbols, symbol_indices, buf_ptr, samp_pt = _symbol_recovery_jit(
        phases.astype(np.float32),
        buffer,
        0,
        samples_per_symbol,
        samples_per_symbol,
        0.0,  # pll
        1.219,  # gain
        _interpolator.TAPS,
    )

    print(f"\nSymbol recovery:")
    print(f"  Symbols: {len(dibits)}")
    print(f"  Buffer pointer: {buf_ptr}")

    # Check symbol_indices validity
    valid_mask = symbol_indices >= 0
    invalid_count = np.sum(~valid_mask)
    print(f"  Invalid indices: {invalid_count}/{len(symbol_indices)} ({100*invalid_count/len(symbol_indices):.1f}%)")

    if invalid_count > 0:
        first_valid = np.argmax(valid_mask)
        print(f"  First valid at position: {first_valid}")

    # Now trace sync detection manually
    print("\n" + "=" * 70)
    print("Manual Sync Detection Trace")
    print("=" * 70)

    detector = _SoftSyncDetector()
    THRESHOLD = 130.0

    # Known sync position from previous tests
    test_positions = [19603, 20467, 21331, 22195]

    for test_pos in test_positions:
        # Reset detector and process up to test position
        detector.reset()
        for i in range(test_pos + 1):
            score = detector.process(soft_symbols[i])

        print(f"\nAt position {test_pos}:")
        print(f"  Sync score: {score:.1f}")
        print(f"  Threshold: {THRESHOLD}")
        print(f"  Triggered: {score >= THRESHOLD}")
        print(f"  symbol_indices[{test_pos}]: {symbol_indices[test_pos]}")
        print(f"  Valid: {symbol_indices[test_pos] >= 0}")

        if score >= THRESHOLD and symbol_indices[test_pos] >= 0:
            print(f"  -> Should trigger timing optimization!")
            # Check buffer value at that position
            buf_idx = symbol_indices[test_pos]
            if buf_idx < len(buffer):
                print(f"  Buffer value at {buf_idx}: {buffer[buf_idx]:.4f}")
        elif score >= THRESHOLD and symbol_indices[test_pos] < 0:
            print(f"  -> Would trigger but index is invalid!")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("/Users/thw/SDRTrunk/recordings/20251227_121743_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav")
