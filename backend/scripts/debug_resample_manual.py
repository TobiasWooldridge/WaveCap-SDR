#!/usr/bin/env python3
"""Manually trace NID resampling calculation."""

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
    print("Manual NID Resample Trace")
    print("=" * 70)

    iq, sample_rate = load_baseband(filepath, int(2 * 50000))  # 2 seconds
    print(f"Sample rate: {sample_rate} Hz")

    from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator, _resample_nid_jit
    from scipy import signal

    # Create demodulator to get buffer
    demod = C4FMDemodulator(sample_rate=sample_rate)
    samples_per_symbol = demod.samples_per_symbol
    print(f"Samples per symbol: {samples_per_symbol:.3f}")

    # Process through filters
    i = iq.real.astype(np.float32)
    q = iq.imag.astype(np.float32)

    i_lpf, _ = signal.lfilter(demod._baseband_lpf, 1.0, i, zi=np.zeros(len(demod._baseband_lpf)-1))
    q_lpf, _ = signal.lfilter(demod._baseband_lpf, 1.0, q, zi=np.zeros(len(demod._baseband_lpf)-1))
    i_rrc, _ = signal.lfilter(demod._rrc_filter, 1.0, i_lpf, zi=np.zeros(len(demod._rrc_filter)-1))
    q_rrc, _ = signal.lfilter(demod._rrc_filter, 1.0, q_lpf, zi=np.zeros(len(demod._rrc_filter)-1))

    phases = demod._fm_demod.demodulate(i_rrc.astype(np.float32), q_rrc.astype(np.float32))
    print(f"Phases: {len(phases)}, range: [{phases.min():.3f}, {phases.max():.3f}]")

    # Copy phases to buffer (simulating what symbol recovery does)
    buffer = phases.copy()  # Use actual phases as buffer

    # Expected NAC for SA-GRN: 0x3DC = [0, 3, 3, 1, 3, 0]
    expected_nac = [0, 3, 3, 1, 3, 0]
    expected_soft = [1.0, -3.0, -3.0, 3.0, -3.0, 1.0]

    # Find sync pattern in phases (manually)
    # Sync pattern soft symbols
    sync_pattern = np.array([3, 3, 3, 3, 3, -3, 3, 3, -3, -3, 3, 3, -3, -3, -3, -3, 3, -3, 3, -3, -3, -3, -3, -3], dtype=np.float32)

    # Test with different PLL/gain values
    print("\n" + "=" * 70)
    print("Testing NID resample with different parameters")
    print("=" * 70)

    # Try finding a sync position in the first part of the buffer
    # Sample soft symbols at regular intervals
    test_start = 5000  # Start looking after 5000 samples (~480 symbols)

    best_corr = -1000
    best_pos = 0
    for pos in range(test_start, min(len(phases) - 30 * int(samples_per_symbol), 50000)):
        # Extract 24 soft symbols starting at pos
        soft_at_pos = []
        for j in range(24):
            idx = pos + int(j * samples_per_symbol + samples_per_symbol / 2)
            if idx < len(phases):
                soft_at_pos.append(phases[idx] * 4.0 / np.pi)  # Normalize

        if len(soft_at_pos) == 24:
            corr = np.sum(np.array(soft_at_pos) * sync_pattern)
            if corr > best_corr:
                best_corr = corr
                best_pos = pos

    print(f"Best sync position: {best_pos} with correlation {best_corr:.1f}")

    if best_corr > 150:
        # Found a sync, now test NID resampling
        sync_sample_start = float(best_pos)

        # Test with default PLL=0, gain=1
        print(f"\nNID resample at sync_start={sync_sample_start:.1f} with PLL=0, gain=1.0:")
        nid_dibits, nid_soft = _resample_nid_jit(buffer, sync_sample_start, samples_per_symbol, 0.0, 1.0)
        print(f"  NAC dibits: {list(nid_dibits[:6])}")
        print(f"  Expected:   {expected_nac}")
        print(f"  Soft:       {[f'{s:.2f}' for s in nid_soft[:6]]}")
        print(f"  Expected:   {expected_soft}")

        # Test with PLL=-0.3, gain=1.25 (typical values from debug)
        print(f"\nNID resample at sync_start={sync_sample_start:.1f} with PLL=-0.3, gain=1.25:")
        nid_dibits, nid_soft = _resample_nid_jit(buffer, sync_sample_start, samples_per_symbol, -0.3, 1.25)
        print(f"  NAC dibits: {list(nid_dibits[:6])}")
        print(f"  Expected:   {expected_nac}")
        print(f"  Soft:       {[f'{s:.2f}' for s in nid_soft[:6]]}")
        print(f"  Expected:   {expected_soft}")

        # Show raw phase values at NID positions
        print("\nRaw phase values at NID positions:")
        nid_start = sync_sample_start + 24 * samples_per_symbol
        for j in range(6):
            sample_pos = nid_start + j * samples_per_symbol + samples_per_symbol / 2
            idx = int(sample_pos)
            if idx < len(buffer):
                phase = buffer[idx]
                soft_norm = phase * 4.0 / np.pi
                expected_dibit = expected_nac[j]
                expected_s = expected_soft[j]
                print(f"  NID[{j}]: sample_pos={sample_pos:.1f}, phase={phase:+.4f}, soft_norm={soft_norm:+.2f}, expected={expected_s:+.1f}")
    else:
        print("No good sync found in data!")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("/Users/thw/SDRTrunk/recordings/20251227_121743_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav")
