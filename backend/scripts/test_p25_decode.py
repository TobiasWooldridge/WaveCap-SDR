#!/usr/bin/env python3
"""Test P25 decoding pipeline against SDRTrunk baseband recording.

Standalone script - does not import from wavecapsdr to avoid dependencies.
Implements minimal C4FM and CQPSK demodulators for comparison testing.
"""

import sys
import wave
import numpy as np
from pathlib import Path


def firwin_simple(numtaps: int, cutoff: float) -> np.ndarray:
    """Simple lowpass FIR filter using sinc function with Hamming window."""
    n = np.arange(numtaps)
    center = (numtaps - 1) / 2

    # Sinc filter
    h = np.zeros(numtaps)
    for i in range(numtaps):
        t = i - center
        if abs(t) < 1e-10:
            h[i] = 2 * cutoff
        else:
            h[i] = np.sin(2 * np.pi * cutoff * t) / (np.pi * t)

    # Hamming window
    window = 0.54 - 0.46 * np.cos(2 * np.pi * n / (numtaps - 1))
    h = h * window

    # Normalize
    h = h / np.sum(h)
    return h


# P25 sync pattern: 0x5575F5FF77FF (48 bits = 24 dibits)
# As symbols: +3 +3 +3 +3 +3 -3 +3 +3 -3 -3 +3 +3 -3 -3 -3 -3 +3 -3 +3 -3 -3 -3 -3 -3
P25_SYNC_SYMBOLS = [3, 3, 3, 3, 3, -3, 3, 3, -3, -3, 3, 3, -3, -3, -3, -3, 3, -3, 3, -3, -3, -3, -3, -3]
# Dibits: symbol +3 = dibit 1, +1 = dibit 0, -1 = dibit 2, -3 = dibit 3
P25_SYNC_DIBITS = np.array([1, 1, 1, 1, 1, 3, 1, 1, 3, 3, 1, 1, 3, 3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3], dtype=np.uint8)


def load_sdrtrunk_baseband(wav_path: str, max_samples: int = 500000):
    """Load SDRTrunk baseband recording as complex IQ.

    Format: 16-bit stereo WAV at 50000 Hz (I=left, Q=right)
    """
    print(f"\n=== Loading {wav_path} ===")

    with wave.open(wav_path, 'rb') as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        frames = wf.getnframes()

        print(f"  Sample rate: {rate} Hz")
        print(f"  Channels: {channels}")
        print(f"  Sample width: {width} bytes")
        print(f"  Total frames: {frames}")

        n_frames = min(frames, max_samples)
        raw = wf.readframes(n_frames)

    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        raise ValueError(f"Unsupported sample width: {width}")

    if channels == 2:
        i_samples = data[::2]
        q_samples = data[1::2]
        iq = i_samples + 1j * q_samples
    else:
        raise ValueError(f"Expected stereo, got {channels} channels")

    print(f"  Loaded {len(iq)} complex samples")
    print(f"  IQ magnitude: mean={np.mean(np.abs(iq)):.4f}, max={np.max(np.abs(iq)):.4f}")

    return iq, rate


def c4fm_demodulate(iq: np.ndarray, sample_rate: int = 50000, symbol_rate: int = 4800):
    """Minimal C4FM (4-FSK) demodulator using FM discriminator."""
    print(f"\n=== C4FM Demodulator ===")
    print(f"  Sample rate: {sample_rate} Hz, Symbol rate: {symbol_rate} baud")
    print(f"  Samples per symbol: {sample_rate / symbol_rate:.2f}")

    # FM discriminator: instantaneous frequency from phase derivative
    phase = np.angle(iq)
    inst_freq = np.diff(np.unwrap(phase))

    print(f"  FM disc output: mean={np.mean(inst_freq):.4f}, std={np.std(inst_freq):.4f}")

    # Baseband low-pass filter (5200 Hz cutoff, matching SDRTrunk)
    nyquist = sample_rate / 2
    cutoff = 5200 / nyquist
    taps = firwin_simple(63, min(cutoff, 0.99) / 2)
    filtered = np.convolve(inst_freq, taps, mode='same')

    # DC removal
    dc_est = 0.0
    dc_alpha = 0.01
    dc_removed = np.zeros_like(filtered)
    for i, s in enumerate(filtered):
        dc_est = dc_est * (1 - dc_alpha) + s * dc_alpha
        dc_removed[i] = s - dc_est

    # Normalize to nominal ±3 range (based on deviation)
    # P25 uses ±1800 Hz deviation, at 50kHz sample rate this is ±0.226 rad/sample
    # Scale to get ±3 nominal output
    deviation_rad = 1800 * 2 * np.pi / sample_rate
    gain = 3.0 / deviation_rad
    scaled = dc_removed * gain

    print(f"  Scaled output: mean={np.mean(scaled):.4f}, std={np.std(scaled):.4f}")

    # Symbol timing recovery (simple peak detection)
    sps = sample_rate / symbol_rate
    symbols = []
    sample_point = sps / 2  # Start at middle

    while sample_point < len(scaled):
        idx = int(sample_point)
        if idx < len(scaled):
            symbols.append(scaled[idx])
        sample_point += sps

    symbols = np.array(symbols)
    print(f"  Extracted {len(symbols)} symbols")

    if len(symbols) > 0:
        print(f"  Symbol stats: mean={np.mean(symbols):.2f}, std={np.std(symbols):.2f}")
        print(f"  Symbol range: [{np.min(symbols):.2f}, {np.max(symbols):.2f}]")

    # 4-level slicing to dibits
    # Symbol +3 → dibit 1, +1 → dibit 0, -1 → dibit 2, -3 → dibit 3
    dibits = np.zeros(len(symbols), dtype=np.uint8)
    for i, s in enumerate(symbols):
        if s > 2:
            dibits[i] = 1  # +3
        elif s > 0:
            dibits[i] = 0  # +1
        elif s > -2:
            dibits[i] = 2  # -1
        else:
            dibits[i] = 3  # -3

    if len(dibits) > 0:
        unique, counts = np.unique(dibits, return_counts=True)
        print(f"  Dibit distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    return dibits, symbols


def cqpsk_demodulate(iq: np.ndarray, sample_rate: int = 50000, symbol_rate: int = 4800):
    """Minimal CQPSK (π/4-DQPSK) demodulator using differential phase."""
    print(f"\n=== CQPSK/LSM Demodulator ===")
    print(f"  Sample rate: {sample_rate} Hz, Symbol rate: {symbol_rate} baud")

    # Baseband low-pass filter (7250 Hz for LSM)
    nyquist = sample_rate / 2
    cutoff = 7250 / nyquist
    taps = firwin_simple(63, min(cutoff, 0.99) / 2)
    filtered_i = np.convolve(iq.real, taps, mode='same')
    filtered_q = np.convolve(iq.imag, taps, mode='same')
    filtered = filtered_i + 1j * filtered_q

    # Symbol timing (simple)
    sps = sample_rate / symbol_rate
    symbols = []
    sample_point = sps / 2

    prev_sample = filtered[0] if len(filtered) > 0 else 1+0j

    while sample_point < len(filtered):
        idx = int(sample_point)
        if idx < len(filtered):
            curr = filtered[idx]

            # Differential demodulation
            if abs(curr) > 1e-6 and abs(prev_sample) > 1e-6:
                diff = (curr / abs(curr)) * np.conj(prev_sample / abs(prev_sample))
            else:
                diff = curr * np.conj(prev_sample)
            phase = np.angle(diff)
            symbols.append(phase)
            prev_sample = curr

        sample_point += sps

    symbols = np.array(symbols)
    print(f"  Extracted {len(symbols)} phase changes")

    if len(symbols) > 0:
        print(f"  Phase stats: mean={np.mean(symbols):.4f}, std={np.std(symbols):.4f}")

    # π/4-DQPSK slicing
    # Phase change → dibit:
    # +π/4 (0 to π/2) → dibit 0
    # +3π/4 (π/2 to π) → dibit 1
    # -π/4 (-π/2 to 0) → dibit 2
    # -3π/4 (-π to -π/2) → dibit 3
    half_pi = np.pi / 2
    dibits = np.zeros(len(symbols), dtype=np.uint8)
    for i, phase in enumerate(symbols):
        if phase >= half_pi:
            dibits[i] = 1  # +3π/4
        elif phase >= 0:
            dibits[i] = 0  # +π/4
        elif phase >= -half_pi:
            dibits[i] = 2  # -π/4
        else:
            dibits[i] = 3  # -3π/4

    if len(dibits) > 0:
        unique, counts = np.unique(dibits, return_counts=True)
        print(f"  Dibit distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    return dibits, symbols


def find_sync_patterns(dibits: np.ndarray, min_matches: int = 18):
    """Search for P25 sync patterns in dibit stream."""
    print(f"\n  Searching for sync in {len(dibits)} dibits...")

    sync_len = len(P25_SYNC_DIBITS)
    matches = []

    for i in range(len(dibits) - sync_len):
        match_count = np.sum(dibits[i:i+sync_len] == P25_SYNC_DIBITS)
        if match_count >= min_matches:
            matches.append((i, match_count))

    print(f"  Found {len(matches)} sync candidates (>= {min_matches}/24 match)")

    if matches:
        matches.sort(key=lambda x: -x[1])
        print(f"  Top syncs:")
        for pos, count in matches[:5]:
            print(f"    Position {pos}: {count}/24 dibits match ({100*count/24:.0f}%)")

    return matches


def main():
    recordings_dir = Path.home() / "SDRTrunk" / "recordings"
    recordings = sorted(recordings_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not recordings:
        print("No SDRTrunk recordings found!")
        return 1

    wav_path = recordings[0]
    print(f"Using latest recording: {wav_path.name}")

    # Load IQ
    iq, sample_rate = load_sdrtrunk_baseband(str(wav_path), max_samples=300000)

    # Test C4FM
    print("\n" + "="*60)
    print("C4FM DEMODULATOR (what SDRTrunk uses for SA-GRN)")
    print("="*60)
    dibits_c4fm, symbols_c4fm = c4fm_demodulate(iq, sample_rate)
    matches_c4fm = find_sync_patterns(dibits_c4fm) if len(dibits_c4fm) > 0 else []

    # Test CQPSK
    print("\n" + "="*60)
    print("CQPSK/LSM DEMODULATOR (WaveCap default)")
    print("="*60)
    dibits_lsm, symbols_lsm = cqpsk_demodulate(iq, sample_rate)
    matches_lsm = find_sync_patterns(dibits_lsm) if len(dibits_lsm) > 0 else []

    # Summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"C4FM:  {len(matches_c4fm):3d} sync patterns found")
    print(f"CQPSK: {len(matches_lsm):3d} sync patterns found")

    if len(matches_c4fm) > len(matches_lsm):
        print("\n>>> C4FM finds more syncs - SA-GRN likely uses C4FM, not LSM!")
    elif len(matches_lsm) > len(matches_c4fm):
        print("\n>>> CQPSK finds more syncs - SA-GRN may use LSM/simulcast")
    else:
        print("\n>>> Both find similar syncs - need deeper analysis")

    return 0


if __name__ == "__main__":
    sys.exit(main())
