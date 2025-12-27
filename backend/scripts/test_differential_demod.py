#!/usr/bin/env python3
"""Test differential demodulation directly with known input.

Creates a simple test signal and verifies demodulation output matches expected.
"""

import numpy as np


def create_test_signal(phases: list[float], samples_per_symbol: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Create I/Q test signal from differential phase values.

    Args:
        phases: List of differential phases (radians) for each symbol
        samples_per_symbol: Number of samples per symbol

    Returns:
        (I, Q) arrays
    """
    # Accumulate phases (differential encoding)
    accumulated = np.cumsum(phases)

    # Upsample to sample rate
    n_samples = len(phases) * samples_per_symbol
    phase_samples = np.zeros(n_samples, dtype=np.float32)

    for i, phase in enumerate(accumulated):
        start = i * samples_per_symbol
        end = (i + 1) * samples_per_symbol
        phase_samples[start:end] = phase

    # Create complex signal
    i = np.cos(phase_samples).astype(np.float32)
    q = np.sin(phase_samples).astype(np.float32)

    return i, q


def differential_demod_wavecap(i: np.ndarray, q: np.ndarray, delay: int = 10) -> np.ndarray:
    """WaveCap-style differential demodulation."""
    n = len(i)
    demodulated = np.zeros(n, dtype=np.float32)

    # Circular buffer for delayed samples
    i_buffer = np.zeros(delay, dtype=np.float32)
    q_buffer = np.zeros(delay, dtype=np.float32)
    buf_pos = 0

    for x in range(n):
        # Get delayed sample
        i_prev = i_buffer[buf_pos]
        q_prev = q_buffer[buf_pos]

        # Current sample
        i_curr = i[x]
        q_curr = q[x]

        # Store in buffer
        i_buffer[buf_pos] = i_curr
        q_buffer[buf_pos] = q_curr
        buf_pos = (buf_pos + 1) % delay

        # s[n] * conj(s[n-delay])
        demod_i = i_curr * i_prev + q_curr * q_prev
        demod_q = q_curr * i_prev - i_curr * q_prev

        demodulated[x] = np.arctan2(demod_q, demod_i)

    return demodulated


def differential_demod_sdrtrunk(i: np.ndarray, q: np.ndarray, delay: int = 10) -> np.ndarray:
    """SDRTrunk-style differential demodulation (compare current to previous).

    SDRTrunk computes: conj(s[n-delay]) * s[n]
    """
    n = len(i)
    demodulated = np.zeros(n, dtype=np.float32)

    for x in range(delay, n):
        i_prev = i[x - delay]
        q_prev_conj = -q[x - delay]  # conjugate

        i_curr = i[x]
        q_curr = q[x]

        # conj(s[n-delay]) * s[n] = (i_prev - j*q_prev) * (i_curr + j*q_curr)
        diff_i = i_prev * i_curr - q_prev_conj * q_curr
        diff_q = i_prev * q_curr + i_curr * q_prev_conj

        demodulated[x] = np.arctan2(diff_q, diff_i)

    return demodulated


def dibit_from_phase(phase: float) -> int:
    """Map phase to dibit (WaveCap/SDRTrunk convention)."""
    boundary = np.pi / 2.0

    if phase >= boundary:
        return 1  # +3
    elif phase >= 0:
        return 0  # +1
    elif phase >= -boundary:
        return 2  # -1
    else:
        return 3  # -3


def main():
    samples_per_symbol = 10

    # Define test phases (differential phase changes)
    # P25 dibit mapping:
    #   dibit 0 -> +π/4
    #   dibit 1 -> +3π/4
    #   dibit 2 -> -π/4
    #   dibit 3 -> -3π/4
    test_dibits = [0, 1, 2, 3, 0, 0, 1, 1, 2, 2, 3, 3]
    dibit_to_phase = {
        0: np.pi / 4,
        1: 3 * np.pi / 4,
        2: -np.pi / 4,
        3: -3 * np.pi / 4,
    }

    test_phases = [dibit_to_phase[d] for d in test_dibits]

    print("Test signal:")
    print(f"  Dibits: {test_dibits}")
    print(f"  Phases: {[f'{p:.3f}' for p in test_phases]}")

    # Create test signal
    i, q = create_test_signal(test_phases, samples_per_symbol)

    print(f"\nGenerated {len(i)} samples ({len(test_dibits)} symbols)")

    # Demodulate using both methods
    demod_wavecap = differential_demod_wavecap(i, q, samples_per_symbol)
    demod_sdrtrunk = differential_demod_sdrtrunk(i, q, samples_per_symbol)

    # Sample at symbol centers
    print(f"\nWaveCap demodulation (sampled at symbol center):")
    wavecap_dibits = []
    for sym in range(len(test_dibits)):
        sample_idx = sym * samples_per_symbol + samples_per_symbol // 2
        if sample_idx < len(demod_wavecap):
            phase = demod_wavecap[sample_idx]
            dibit = dibit_from_phase(phase)
            wavecap_dibits.append(dibit)
            match = "✓" if dibit == test_dibits[sym] else "✗"
            print(f"  Symbol {sym}: phase={phase:+.3f} -> dibit={dibit} (expected {test_dibits[sym]}) {match}")

    print(f"\nSDRTrunk demodulation (sampled at symbol center):")
    sdrtrunk_dibits = []
    for sym in range(len(test_dibits)):
        sample_idx = sym * samples_per_symbol + samples_per_symbol // 2
        if sample_idx < len(demod_sdrtrunk):
            phase = demod_sdrtrunk[sample_idx]
            dibit = dibit_from_phase(phase)
            sdrtrunk_dibits.append(dibit)
            match = "✓" if dibit == test_dibits[sym] else "✗"
            print(f"  Symbol {sym}: phase={phase:+.3f} -> dibit={dibit} (expected {test_dibits[sym]}) {match}")

    # Summary
    wavecap_correct = sum(1 for a, b in zip(wavecap_dibits, test_dibits) if a == b)
    sdrtrunk_correct = sum(1 for a, b in zip(sdrtrunk_dibits, test_dibits) if a == b)

    print(f"\nSummary:")
    print(f"  WaveCap:   {wavecap_correct}/{len(test_dibits)} correct")
    print(f"  SDRTrunk:  {sdrtrunk_correct}/{len(test_dibits)} correct")

    # Check symbol timing alignment
    print(f"\n--- Detailed phase trace (first 3 symbols) ---")
    for sym in range(min(3, len(test_dibits))):
        print(f"\nSymbol {sym} (expected dibit {test_dibits[sym]}, phase {test_phases[sym]:.3f}):")
        start = sym * samples_per_symbol
        end = (sym + 1) * samples_per_symbol
        for s in range(start, end):
            if s < len(demod_wavecap):
                print(f"  Sample {s}: wavecap={demod_wavecap[s]:+.3f}, sdrtrunk={demod_sdrtrunk[s]:+.3f}")


if __name__ == '__main__':
    main()
