#!/usr/bin/env python3
"""Detailed trace of sync detection including timing optimizer."""

import sys
import wave
import numpy as np
from scipy import signal

sys.path.insert(0, '/Users/thw/Projects/WaveCap-SDR/backend')

from wavecapsdr.dsp.p25.c4fm import (
    C4FMDemodulator,
    _SoftSyncDetector,
    _TimingOptimizer,
    _Equalizer,
    _symbol_recovery_jit,
    _interpolator,
)


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
    print("Detailed Sync Detection Trace")
    print("=" * 70)

    iq, sample_rate = load_baseband(filepath, int(10 * 50000))
    print(f"Sample rate: {sample_rate} Hz")
    samples_per_symbol = sample_rate / 4800.0
    print(f"Samples per symbol: {samples_per_symbol:.3f}")

    # Create demodulator to get access to its components
    demod = C4FMDemodulator(sample_rate=sample_rate)

    # Get the soft symbols and buffer by running demodulate
    print("\nRunning demodulate()...")
    dibits, soft_symbols = demod.demodulate(iq)
    print(f"Dibits: {len(dibits)}, Soft symbols: {len(soft_symbols)}")
    print(f"Sync count from demodulator: {demod._sync_count}")

    # Now manually trace through the sync detection process
    print("\n" + "=" * 70)
    print("Manual trace of sync detection with timing optimizer")
    print("=" * 70)

    # Get the internal buffer from demodulator
    buffer = demod._buffer.copy()
    print(f"Buffer length: {len(buffer)}")
    print(f"Buffer range: [{buffer.min():.2f}, {buffer.max():.2f}]")

    # Create fresh components for testing
    sync_detector = _SoftSyncDetector()
    timing_optimizer = _TimingOptimizer(samples_per_symbol)
    equalizer = _Equalizer()

    THRESHOLD = 130.0

    # Process symbols looking for sync
    sync_triggers = []
    for i, soft_sym in enumerate(soft_symbols):
        score = sync_detector.process(soft_sym)
        if score >= THRESHOLD:
            sync_triggers.append((i, score))

    print(f"\nSync triggers (score >= {THRESHOLD}): {len(sync_triggers)}")

    # For each trigger, check what timing optimizer would return
    print("\nAnalyzing first 20 sync triggers:")
    print(f"{'Pos':>6} {'Score':>8} {'OptScore':>10} {'TimingAdj':>10} {'PLL':>8} {'Gain':>8} {'Result':>10}")

    # We need to correlate symbol positions with buffer positions
    # The buffer was populated during demodulate()
    # Symbol i corresponds roughly to buffer position floor(i * samples_per_symbol)

    for i, (sym_pos, score) in enumerate(sync_triggers[:20]):
        # Estimate buffer position for this symbol
        # This is approximate - we'd need symbol_indices from demodulate
        buf_pos = int(sym_pos * samples_per_symbol) + int(samples_per_symbol / 2)

        # Ensure buffer position is valid
        if buf_pos < 30 or buf_pos >= len(buffer) - 30:
            print(f"{sym_pos:>6} {score:>8.1f} {'N/A':>10} {'(out of bounds)':>30}")
            continue

        # Run timing optimizer
        mu = 0.5
        try:
            timing_adj, opt_score, pll_adj, gain_adj = timing_optimizer.optimize(
                buffer, buf_pos + mu, equalizer, fine_sync=False
            )
            result = "PASS" if opt_score >= THRESHOLD else "FAIL"
            print(f"{sym_pos:>6} {score:>8.1f} {opt_score:>10.1f} {timing_adj:>+10.3f} {pll_adj:>+8.4f} {gain_adj:>8.3f} {result:>10}")
        except Exception as e:
            print(f"{sym_pos:>6} {score:>8.1f} {'ERROR':>10} {str(e)[:30]}")

    # Find the real sync positions (correlation > 200)
    print("\n" + "=" * 70)
    print("Analyzing high-correlation sync positions (> 200)")
    print("=" * 70)

    expected_sync = np.array([3, 3, 3, 3, 3, -3, 3, 3, -3, -3, 3, 3, -3, -3, -3, -3, 3, -3, 3, -3, -3, -3, -3, -3], dtype=np.float32)

    best_matches = []
    for i in range(len(soft_symbols) - 24):
        window = soft_symbols[i:i+24]
        corr = np.sum(window * expected_sync)
        if corr > 200:
            best_matches.append((i, corr))

    print(f"Positions with correlation > 200: {len(best_matches)}")

    for sym_pos, corr in best_matches[:10]:
        buf_pos = int(sym_pos * samples_per_symbol) + int(samples_per_symbol / 2)

        if buf_pos < 30 or buf_pos >= len(buffer) - 30:
            print(f"{sym_pos:>6} corr={corr:.1f} (out of bounds)")
            continue

        # Run timing optimizer
        mu = 0.5
        try:
            timing_adj, opt_score, pll_adj, gain_adj = timing_optimizer.optimize(
                buffer, buf_pos + mu, equalizer, fine_sync=False
            )
            result = "PASS" if opt_score >= THRESHOLD else "FAIL"
            print(f"Pos {sym_pos}: corr={corr:.1f}, opt_score={opt_score:.1f}, timing={timing_adj:+.3f} -> {result}")
        except Exception as e:
            print(f"Pos {sym_pos}: corr={corr:.1f}, ERROR: {e}")

    # Check the actual symbol_indices that demodulate() generates
    print("\n" + "=" * 70)
    print("Checking relationship between symbol index and buffer position")
    print("=" * 70)

    # Re-run symbol recovery to get symbol_indices
    # We need to replicate what demodulate() does

    # Apply filters (simplified - using demodulator's filters)
    i_sig = iq.real.astype(np.float32)
    q_sig = iq.imag.astype(np.float32)

    # LPF
    lpf = demod._baseband_lpf
    i_lpf = signal.lfilter(lpf, 1.0, i_sig)
    q_lpf = signal.lfilter(lpf, 1.0, q_sig)

    # RRC
    rrc = demod._rrc_filter
    i_rrc = signal.lfilter(rrc, 1.0, i_lpf)
    q_rrc = signal.lfilter(rrc, 1.0, q_lpf)

    # FM demod
    phases = demod._fm_demod.demodulate(i_rrc.astype(np.float32), q_rrc.astype(np.float32))

    # Symbol recovery
    buffer2 = np.zeros(2048, dtype=np.float32)
    dibits2, soft2, symbol_indices, buf_ptr2, samp_pt2 = _symbol_recovery_jit(
        phases.astype(np.float32),
        buffer2,
        0,
        samples_per_symbol,
        samples_per_symbol,
        0.0,  # pll
        1.219,  # gain
        _interpolator.TAPS,
    )

    print(f"Symbol recovery: {len(dibits2)} symbols")
    print(f"Symbol indices range: [{symbol_indices.min()}, {symbol_indices.max()}]")
    print(f"Buffer pointer: {buf_ptr2}")

    # For the best sync position, check what buffer index it corresponds to
    if best_matches:
        sym_pos, corr = best_matches[0]
        if sym_pos < len(symbol_indices):
            buf_idx = symbol_indices[sym_pos]
            print(f"\nBest sync at symbol {sym_pos}:")
            print(f"  Correlation: {corr:.1f}")
            print(f"  Buffer index from symbol_indices: {buf_idx}")

            # Show buffer values around this position
            if buf_idx >= 30 and buf_idx < len(buffer2) - 30:
                print(f"  Buffer values around sync end (indices {buf_idx-5} to {buf_idx+5}):")
                for j in range(buf_idx - 5, buf_idx + 6):
                    phase = buffer2[j]
                    soft_norm = phase * 4.0 / np.pi
                    print(f"    [{j}] phase={phase:+.3f}, soft_norm={soft_norm:+.2f}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("/Users/thw/SDRTrunk/recordings/20251227_121743_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav")
