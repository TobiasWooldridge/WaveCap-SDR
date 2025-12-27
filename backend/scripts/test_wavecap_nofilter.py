#!/usr/bin/env python3
"""Test WaveCap demodulator with filters disabled."""

import wave
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator, _FMDemodulator, Dibit


# P25 sync pattern
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
        if n_channels != 2:
            raise ValueError(f"Expected stereo WAV, got {n_channels} channels")
        raw = wf.readframes(n_frames)
        samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
        iq = (samples[:, 0] + 1j * samples[:, 1]).astype(np.complex64) / 32768.0
    return rate, iq


class C4FMDemodulatorNoFilter(C4FMDemodulator):
    """C4FM demodulator with filters disabled for testing."""

    def demodulate(self, iq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Demodulate without filters."""
        if len(iq) == 0:
            return (np.array([], dtype=np.uint8), np.array([], dtype=np.float32))

        # Skip filters - use raw I/Q
        i = iq.real.astype(np.float32)
        q = iq.imag.astype(np.float32)

        # FM demodulate
        phases = self._fm_demod.demodulate(i, q)

        # Extract symbols at symbol rate
        dibits = []
        soft_symbols = []
        sps = self.samples_per_symbol

        for sym in range(int(len(phases) / sps)):
            idx = int(sym * sps + sps / 2)
            if idx < len(phases):
                phase = phases[idx]
                soft_symbols.append(phase)
                dibit = Dibit.from_soft_symbol(phase)
                dibits.append(dibit.value)

        return np.array(dibits, dtype=np.uint8), np.array(soft_symbols, dtype=np.float32)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input WAV file')
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    sample_rate, iq = load_iq_wav(args.input)
    print(f"Loaded {len(iq)} samples at {sample_rate} Hz")

    print("\n=== Standard WaveCap demodulator (with filters) ===")
    demod_std = C4FMDemodulator(sample_rate=sample_rate)
    dibits_std, soft_std = demod_std.demodulate(iq)

    dibit_counts = [np.sum(dibits_std == i) for i in range(4)]
    print(f"Dibit distribution: {dibit_counts}")
    print(f"  Dibit 0: {100*dibit_counts[0]/len(dibits_std):.1f}%")
    print(f"  Dibit 1: {100*dibit_counts[1]/len(dibits_std):.1f}%")
    print(f"  Dibit 2: {100*dibit_counts[2]/len(dibits_std):.1f}%")
    print(f"  Dibit 3: {100*dibit_counts[3]/len(dibits_std):.1f}%")

    # Find sync
    best_pos = -1
    best_errors = 24
    for i in range(len(dibits_std) - 24):
        errors = sum(1 for j in range(24) if dibits_std[i + j] != SYNC_DIBITS[j])
        if errors < best_errors:
            best_errors = errors
            best_pos = i
    print(f"Best sync: position {best_pos} with {best_errors} errors")

    print("\n=== WaveCap demodulator WITHOUT filters ===")
    demod_nofilt = C4FMDemodulatorNoFilter(sample_rate=sample_rate)
    dibits_nofilt, soft_nofilt = demod_nofilt.demodulate(iq)

    dibit_counts = [np.sum(dibits_nofilt == i) for i in range(4)]
    print(f"Dibit distribution: {dibit_counts}")
    print(f"  Dibit 0: {100*dibit_counts[0]/len(dibits_nofilt):.1f}%")
    print(f"  Dibit 1: {100*dibit_counts[1]/len(dibits_nofilt):.1f}%")
    print(f"  Dibit 2: {100*dibit_counts[2]/len(dibits_nofilt):.1f}%")
    print(f"  Dibit 3: {100*dibit_counts[3]/len(dibits_nofilt):.1f}%")

    # Find sync
    best_pos = -1
    best_errors = 24
    for i in range(len(dibits_nofilt) - 24):
        errors = sum(1 for j in range(24) if dibits_nofilt[i + j] != SYNC_DIBITS[j])
        if errors < best_errors:
            best_errors = errors
            best_pos = i
    print(f"Best sync: position {best_pos} with {best_errors} errors")


if __name__ == '__main__':
    main()
