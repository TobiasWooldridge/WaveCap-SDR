#!/usr/bin/env python3
"""Check symbol_indices at correct sync positions."""

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
    print("Checking Correct Sync Positions")
    print("=" * 70)

    iq, sample_rate = load_baseband(filepath, int(10 * 50000))
    print(f"Sample rate: {sample_rate} Hz")

    from wavecapsdr.dsp.p25.c4fm import (
        C4FMDemodulator,
        _SoftSyncDetector,
        _symbol_recovery_jit,
        _interpolator,
    )
    from scipy import signal

    # Get soft_symbols and symbol_indices from the demodulator path
    demod = C4FMDemodulator(sample_rate=sample_rate)
    samples_per_symbol = demod.samples_per_symbol

    # Process through filters (matching demodulator)
    i = iq.real.astype(np.float32)
    q = iq.imag.astype(np.float32)

    i_lpf, _ = signal.lfilter(demod._baseband_lpf, 1.0, i, zi=np.zeros(len(demod._baseband_lpf)-1))
    q_lpf, _ = signal.lfilter(demod._baseband_lpf, 1.0, q, zi=np.zeros(len(demod._baseband_lpf)-1))
    i_rrc, _ = signal.lfilter(demod._rrc_filter, 1.0, i_lpf, zi=np.zeros(len(demod._rrc_filter)-1))
    q_rrc, _ = signal.lfilter(demod._rrc_filter, 1.0, q_lpf, zi=np.zeros(len(demod._rrc_filter)-1))

    phases = demod._fm_demod.demodulate(i_rrc.astype(np.float32), q_rrc.astype(np.float32))

    # Use demod's buffer (65536)
    buffer = demod._buffer.copy()

    dibits, soft_symbols, symbol_indices, buf_ptr, samp_pt = _symbol_recovery_jit(
        phases.astype(np.float32),
        buffer,
        0,
        samples_per_symbol,
        samples_per_symbol,
        0.0,
        1.219,
        _interpolator.TAPS,
    )

    print(f"\nSymbol recovery:")
    print(f"  Symbols: {len(dibits)}")
    print(f"  Buffer size: {len(buffer)}")
    print(f"  Buffer pointer: {buf_ptr}")

    # Check symbol_indices validity
    valid_mask = symbol_indices >= 0
    invalid_count = np.sum(~valid_mask)
    print(f"  Invalid indices: {invalid_count}/{len(symbol_indices)} ({100*invalid_count/len(symbol_indices):.1f}%)")

    # Correct sync positions
    correct_positions = [19626, 20490, 21354, 22218]

    print("\n" + "=" * 70)
    print("Sync Position Analysis (Correct Positions)")
    print("=" * 70)

    for pos in correct_positions:
        idx = symbol_indices[pos]
        valid = idx >= 0
        print(f"\nPosition {pos}:")
        print(f"  symbol_indices[{pos}]: {idx}")
        print(f"  Valid: {valid}")

        if valid:
            # Check sync score at this position
            detector = _SoftSyncDetector()
            for i in range(pos + 1):
                score = detector.process(soft_symbols[i])
            print(f"  Sync score: {score:.1f}")
            print(f"  Would trigger: {score >= 130.0}")

    # What's the range of valid indices?
    if np.any(valid_mask):
        valid_positions = np.where(valid_mask)[0]
        print(f"\nValid symbol positions: {valid_positions[0]} to {valid_positions[-1]}")
        print(f"  (symbols {valid_positions[0]} through {valid_positions[-1]} have valid buffer indices)")

    # Check: are sync positions BEFORE valid range?
    if np.any(valid_mask):
        first_valid = valid_positions[0]
        for pos in correct_positions:
            if pos < first_valid:
                print(f"\nPosition {pos} is BEFORE first valid position ({first_valid})")
                print("  -> This position will NEVER have a valid buffer index!")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("/Users/thw/SDRTrunk/recordings/20251227_121743_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav")
