#!/usr/bin/env python3
"""Generate synthetic P25 C4FM test signal for decoder comparison.

Creates a 16-bit stereo WAV file at 48kHz containing:
- Valid P25 sync patterns
- Valid NIDs with BCH encoding
- Frame structure matching P25 Phase 1

This signal can be tested with both SDRTrunk and WaveCap for comparison.
"""

import sys
import wave
import struct
import numpy as np
from pathlib import Path


# P25 constants
SYMBOL_RATE = 4800  # baud
SYNC_PATTERN = 0x5575F5FF77FF  # 48 bits = 24 dibits

# C4FM symbol phases (in radians)
DIBIT_TO_PHASE = {
    0: np.pi / 4,       # +1
    1: 3 * np.pi / 4,   # +3
    2: -np.pi / 4,      # -1
    3: -3 * np.pi / 4,  # -3
}

# BCH(63,16,23) generator matrix rows (from SDRTrunk P25 NID)
P25_NID_GENERATOR = [
    int("6331141367235452", 8),  # bit 0
    int("5265521614723276", 8),  # bit 1
    int("6633610705374177", 8),  # bit 2
    int("3516644250573527", 8),  # bit 3
    int("7656173024272313", 8),  # bit 4
    int("3727035410151606", 8),  # bit 5
    int("5764366604062343", 8),  # bit 6
    int("6573032200427671", 8),  # bit 7
    int("3465365002200374", 8),  # bit 8
    int("5633531401104636", 8),  # bit 9
    int("6714614600446757", 8),  # bit 10
    int("3256156200237737", 8),  # bit 11
    int("5450026100113317", 8),  # bit 12
    int("6624062040041107", 8),  # bit 13
    int("3312034020024503", 8),  # bit 14
    int("7444016010016701", 8),  # bit 15
]


def encode_nid(nac: int, duid: int) -> np.ndarray:
    """Encode NAC and DUID into 64-bit NID with BCH parity."""
    data_word = ((nac & 0xFFF) << 4) | (duid & 0xF)

    codeword = 0
    for i in range(16):
        if (data_word >> (15 - i)) & 1:
            codeword ^= P25_NID_GENERATOR[i]

    # Convert to 64 bits (63 data + 1 unused)
    bits = np.zeros(64, dtype=np.uint8)
    for i in range(63):
        bits[i] = (codeword >> (62 - i)) & 1

    return bits


def bits_to_dibits(bits: np.ndarray) -> np.ndarray:
    """Convert bit array to dibit array."""
    n_dibits = len(bits) // 2
    dibits = np.zeros(n_dibits, dtype=np.uint8)
    for i in range(n_dibits):
        dibits[i] = (bits[i * 2] << 1) | bits[i * 2 + 1]
    return dibits


def get_sync_dibits() -> np.ndarray:
    """Get P25 sync pattern as dibit array."""
    dibits = np.zeros(24, dtype=np.uint8)
    for i in range(24):
        dibits[i] = (SYNC_PATTERN >> ((23 - i) * 2)) & 0x3
    return dibits


def modulate_dibits(dibits: np.ndarray, sample_rate: int, use_rrc: bool = False) -> np.ndarray:
    """Convert dibits to baseband I/Q using C4FM differential modulation.

    C4FM encodes data in the phase CHANGE between symbols, not absolute phase.
    The demodulator computes: s[n] * conj(s[n-delay]) to extract phase difference.

    For proper C4FM simulation, RRC pulse shaping should be applied to the
    frequency deviation (rate of phase change), not to the I/Q directly.

    Args:
        dibits: Array of dibit values (0-3)
        sample_rate: Output sample rate in Hz
        use_rrc: If True, apply RRC pulse shaping to frequency deviation
    """
    from scipy import signal as sig

    samples_per_symbol = sample_rate / SYMBOL_RATE
    n_samples = int(len(dibits) * samples_per_symbol)

    # Convert dibits to frequency deviations
    # C4FM uses ±600 Hz and ±1800 Hz deviations
    # Normalize to ±1, ±3 (will scale by 600 Hz)
    DIBIT_TO_FREQ = {
        0: 1.0,   # +600 Hz -> +1
        1: 3.0,   # +1800 Hz -> +3
        2: -1.0,  # -600 Hz -> -1
        3: -3.0,  # -1800 Hz -> -3
    }
    freq_symbols = np.array([DIBIT_TO_FREQ[d] for d in dibits], dtype=np.float32)

    # Upsample frequency deviations to sample rate (rectangular pulses)
    freq_samples = np.zeros(n_samples, dtype=np.float32)
    for i, freq in enumerate(freq_symbols):
        start = int(i * samples_per_symbol)
        end = int((i + 1) * samples_per_symbol)
        freq_samples[start:end] = freq

    if use_rrc:
        # Apply RRC pulse shaping to frequency deviation (like a real transmitter)
        alpha = 0.2
        span = 8  # symbols
        n_taps = int(span * samples_per_symbol) | 1  # ensure odd
        t = np.arange(-(n_taps - 1) // 2, (n_taps + 1) // 2) / samples_per_symbol

        rrc = np.zeros(n_taps, dtype=np.float32)
        for i, ti in enumerate(t):
            if ti == 0:
                rrc[i] = 1 + alpha * (4 / np.pi - 1)
            elif abs(ti) == 1 / (4 * alpha):
                rrc[i] = (alpha / np.sqrt(2)) * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) +
                    (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
                )
            else:
                num = np.sin(np.pi * ti * (1 - alpha)) + 4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha))
                den = np.pi * ti * (1 - (4 * alpha * ti) ** 2)
                if abs(den) > 1e-10:
                    rrc[i] = num / den
                else:
                    rrc[i] = 0

        rrc /= np.sum(rrc)  # Normalize to unit DC gain
        freq_samples = sig.lfilter(rrc, 1.0, freq_samples)

    # Convert frequency deviation to phase
    # freq (normalized to ±1, ±3) * 600 Hz = actual frequency deviation
    # Phase increment per sample = 2π * freq_dev * (1/sample_rate)
    # Since we want phase = ±π/4, ±3π/4 per symbol, scale appropriately
    # For differential demod with delay = samples_per_symbol:
    #   phase_change_per_symbol = freq_samples[n] * (π/4) * samples_per_symbol / samples_per_symbol
    #                           = freq_samples[n] * (π/4)
    # So integrate frequency to get phase, scaled by π/4 per sample_period
    phase_increment_per_sample = freq_samples * (np.pi / 4) / samples_per_symbol
    phase_samples = np.cumsum(phase_increment_per_sample)

    # Create complex signal from phase
    iq = np.exp(1j * phase_samples).astype(np.complex64)

    return iq


def generate_p25_frame(nac: int = 0x293, duid: int = 0x0) -> np.ndarray:
    """Generate a P25 frame with sync + NID.

    Args:
        nac: 12-bit Network Access Code
        duid: 4-bit Data Unit ID (0=HDU, 5=LDU1, 7=TSBK, etc.)

    Returns:
        Array of dibits for the frame header
    """
    frame = []

    # Sync pattern (24 dibits)
    sync = get_sync_dibits()
    frame.extend(sync)

    # NID (32 dibits = 64 bits, but we insert status at position 11)
    nid_bits = encode_nid(nac, duid)
    nid_dibits = bits_to_dibits(nid_bits)

    # Insert status symbol (dibit 0) at position 11
    # This mirrors P25 frame structure
    frame.extend(nid_dibits[:11])
    frame.append(0)  # Status symbol
    frame.extend(nid_dibits[11:])

    # Add some random voice data (for a complete frame)
    # Just using idle pattern for simplicity
    for _ in range(100):  # ~200 dibits of padding
        frame.append(0)

    return np.array(frame, dtype=np.uint8)


def generate_test_signal(
    sample_rate: int = 48000,
    duration_sec: float = 1.0,
    nac: int = 0x293,
    n_frames: int = 10
) -> np.ndarray:
    """Generate test signal with multiple P25 frames.

    Args:
        sample_rate: Output sample rate (Hz)
        duration_sec: Total duration (seconds)
        nac: Network Access Code
        n_frames: Number of P25 frames to include

    Returns:
        Complex IQ samples
    """
    # Generate frames
    all_dibits = []

    # Add some idle dibits at start
    for _ in range(50):
        all_dibits.append(0)

    # Add P25 frames
    for i in range(n_frames):
        # Alternate between TSBK (0x7) and other DUIDs
        duid = 0x7 if i % 2 == 0 else 0x5
        frame = generate_p25_frame(nac=nac, duid=duid)
        all_dibits.extend(frame)

        # Add inter-frame gap
        for _ in range(20):
            all_dibits.append(0)

    dibits = np.array(all_dibits, dtype=np.uint8)

    # Modulate to IQ
    iq = modulate_dibits(dibits, sample_rate)

    # Pad or truncate to exact duration
    n_samples = int(sample_rate * duration_sec)
    if len(iq) < n_samples:
        iq = np.concatenate([iq, np.zeros(n_samples - len(iq), dtype=np.complex64)])
    else:
        iq = iq[:n_samples]

    return iq


def save_wav(iq: np.ndarray, path: str, sample_rate: int):
    """Save complex IQ as 16-bit stereo WAV (SDRTrunk format)."""
    # Normalize to ±0.9 to leave headroom
    max_val = max(np.abs(iq.real).max(), np.abs(iq.imag).max())
    if max_val > 0:
        iq = iq * 0.9 / max_val

    # Convert to 16-bit signed
    i_16 = (iq.real * 32767).astype(np.int16)
    q_16 = (iq.imag * 32767).astype(np.int16)

    # Interleave I and Q
    samples = np.column_stack([i_16, q_16]).flatten()

    with wave.open(path, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate P25 C4FM test signal')
    parser.add_argument('--output', '-o', default='/tmp/p25_test_signal.wav',
                        help='Output WAV file path')
    parser.add_argument('--sample-rate', '-r', type=int, default=48000,
                        help='Sample rate (Hz)')
    parser.add_argument('--duration', '-d', type=float, default=2.0,
                        help='Duration (seconds)')
    parser.add_argument('--nac', type=lambda x: int(x, 0), default=0x293,
                        help='Network Access Code (hex)')
    parser.add_argument('--frames', '-n', type=int, default=20,
                        help='Number of P25 frames')
    args = parser.parse_args()

    print(f"Generating P25 test signal:")
    print(f"  Sample rate: {args.sample_rate} Hz")
    print(f"  Duration: {args.duration} sec")
    print(f"  NAC: 0x{args.nac:03X}")
    print(f"  Frames: {args.frames}")

    iq = generate_test_signal(
        sample_rate=args.sample_rate,
        duration_sec=args.duration,
        nac=args.nac,
        n_frames=args.frames
    )

    print(f"\nGenerated {len(iq)} samples")
    print(f"  I range: [{iq.real.min():.3f}, {iq.real.max():.3f}]")
    print(f"  Q range: [{iq.imag.min():.3f}, {iq.imag.max():.3f}]")

    save_wav(iq, args.output, args.sample_rate)
    print(f"\nSaved to: {args.output}")

    # Verify by reading back
    print("\nVerification:")
    with wave.open(args.output, 'rb') as wf:
        print(f"  Channels: {wf.getnchannels()}")
        print(f"  Sample width: {wf.getsampwidth()} bytes")
        print(f"  Frame rate: {wf.getframerate()} Hz")
        print(f"  Frames: {wf.getnframes()}")


if __name__ == '__main__':
    main()
