#!/usr/bin/env python3
"""Generate synthetic P25 C4FM test signal with known LDU frames.

This creates a test file for validating the native IMBE decoder pipeline.
"""

import logging
import numpy as np
import scipy.io.wavfile as wav

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# P25 Frame Sync pattern (48 bits as dibits)
# FS = 0x5575F5FF77FF (normal polarity)
FRAME_SYNC_DIBITS = np.array([
    1, 1, 1, 1, 0, 1, 1, 1,  # 0x55
    0, 1, 1, 1, 1, 1, 0, 1,  # 0x75
    3, 3, 1, 1, 3, 1, 0, 1,  # 0xF5
    3, 3, 1, 1, 3, 3, 1, 1,  # 0xFF
    0, 1, 1, 1, 0, 1, 1, 1,  # 0x77
    3, 3, 1, 1, 3, 3, 1, 1,  # 0xFF
], dtype=np.uint8)


def dibits_to_bytes(dibits: np.ndarray) -> bytes:
    """Convert dibits to bytes."""
    # Pack 4 dibits per byte
    n_bytes = (len(dibits) + 3) // 4
    result = []
    for i in range(n_bytes):
        byte = 0
        for j in range(4):
            idx = i * 4 + j
            if idx < len(dibits):
                byte |= (dibits[idx] & 0x03) << (6 - 2*j)
        result.append(byte)
    return bytes(result)


def generate_nid(duid: int = 0x05, nac: int = 0x3DC) -> np.ndarray:
    """Generate P25 Network ID (NID) as dibits.

    NID is 64 bits:
    - NAC: 12 bits (Network Access Code)
    - DUID: 4 bits (Data Unit ID)
    - Parity: 48 bits (BCH/Reed-Solomon)

    For testing, we'll use a simplified NID without full FEC.
    """
    # Simple NID: NAC (12 bits) + DUID (4 bits) + padding
    nid_bits = []

    # NAC (12 bits, MSB first)
    for i in range(11, -1, -1):
        nid_bits.append((nac >> i) & 1)

    # DUID (4 bits)
    for i in range(3, -1, -1):
        nid_bits.append((duid >> i) & 1)

    # Padding with zeros (48 bits for FEC placeholder)
    nid_bits.extend([0] * 48)

    # Convert bits to dibits
    nid_dibits = []
    for i in range(0, len(nid_bits), 2):
        dibit = (nid_bits[i] << 1) | nid_bits[i + 1]
        nid_dibits.append(dibit)

    return np.array(nid_dibits, dtype=np.uint8)


def generate_ldu1_frame() -> np.ndarray:
    """Generate a simplified LDU1 frame.

    LDU1 is 1728 bits = 864 dibits, containing:
    - Frame sync (48 bits = 24 dibits)
    - NID (64 bits = 32 dibits)
    - Link Control (72 bits = 36 dibits)
    - 9 IMBE voice frames (88 bits each = 44 dibits each)
    - Low Speed Data (96 bits = 48 dibits)

    Total structure is more complex with interleaving, but for testing
    we'll generate a recognizable pattern.
    """
    dibits = []

    # Frame sync
    dibits.extend(FRAME_SYNC_DIBITS.tolist())

    # NID (DUID=0x05 for LDU1)
    nid = generate_nid(duid=0x05, nac=0x3DC)
    dibits.extend(nid.tolist())

    # Fill rest with pseudo-random pattern for now
    # Real LDU1 would have IMBE voice data
    np.random.seed(42)
    remaining = 864 - len(dibits)
    dibits.extend(np.random.randint(0, 4, remaining).tolist())

    return np.array(dibits[:864], dtype=np.uint8)


def generate_ldu2_frame() -> np.ndarray:
    """Generate a simplified LDU2 frame (DUID=0x0A)."""
    dibits = []

    # Frame sync
    dibits.extend(FRAME_SYNC_DIBITS.tolist())

    # NID (DUID=0x0A for LDU2)
    nid = generate_nid(duid=0x0A, nac=0x3DC)
    dibits.extend(nid.tolist())

    # Fill rest with pseudo-random pattern
    np.random.seed(43)
    remaining = 864 - len(dibits)
    dibits.extend(np.random.randint(0, 4, remaining).tolist())

    return np.array(dibits[:864], dtype=np.uint8)


def dibits_to_c4fm_audio(dibits: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
    """Convert dibits to C4FM discriminator audio.

    C4FM uses 4 levels: -3, -1, +1, +3
    At 4800 baud with 48kHz sample rate = 10 samples/symbol
    Deviation is ±1800 Hz -> ±0.236 rad/sample at 48kHz
    """
    baud_rate = 4800
    samples_per_symbol = sample_rate // baud_rate

    # Map dibits to deviation levels
    # 00 -> +3, 01 -> +1, 10 -> -1, 11 -> -3
    level_map = {0: 3, 1: 1, 2: -1, 3: -3}
    levels = np.array([level_map[d] for d in dibits])

    # Scale to discriminator output (radians/sample)
    # Max deviation = 1800 Hz
    # At 48kHz: 1800/48000 * 2π ≈ 0.236 rad/sample
    max_dev_rad = 1800 / sample_rate * 2 * np.pi
    normalized = levels / 3.0  # -1 to +1
    disc_values = normalized * max_dev_rad

    # Create samples with raised cosine transitions
    audio = np.zeros(len(dibits) * samples_per_symbol, dtype=np.float32)

    for i, val in enumerate(disc_values):
        start = i * samples_per_symbol
        end = start + samples_per_symbol

        if i == 0:
            # First symbol - constant
            audio[start:end] = val
        else:
            # Smooth transition from previous symbol
            prev_val = disc_values[i - 1]
            # Raised cosine transition in first half
            half = samples_per_symbol // 2
            t = np.linspace(0, np.pi, half)
            transition = prev_val + (val - prev_val) * (1 - np.cos(t)) / 2
            audio[start:start + half] = transition
            audio[start + half:end] = val

    return audio


def generate_test_file(filename: str, duration: float = 5.0, sample_rate: int = 48000):
    """Generate a test WAV file with P25 LDU frames.

    Creates alternating LDU1 and LDU2 frames with gaps of silence.
    """
    logger.info(f"Generating P25 test signal: {filename}")
    logger.info(f"Duration: {duration}s, Sample rate: {sample_rate}Hz")

    # Calculate timing
    # LDU frame = 864 dibits at 4800 baud = 180ms
    frame_duration = 864 / 4800  # 0.18 seconds
    samples_per_frame = int(frame_duration * sample_rate)

    # Add gap between frames (typical P25 has continuous transmission)
    gap_samples = sample_rate // 10  # 100ms gap

    total_samples = int(duration * sample_rate)
    audio = np.zeros(total_samples, dtype=np.float32)

    frame_count = 0
    pos = 0

    while pos + samples_per_frame < total_samples:
        # Alternate LDU1 and LDU2
        if frame_count % 2 == 0:
            frame_dibits = generate_ldu1_frame()
            frame_type = "LDU1"
        else:
            frame_dibits = generate_ldu2_frame()
            frame_type = "LDU2"

        # Convert to audio
        frame_audio = dibits_to_c4fm_audio(frame_dibits, sample_rate)

        # Insert frame
        end_pos = min(pos + len(frame_audio), total_samples)
        audio[pos:end_pos] = frame_audio[:end_pos - pos]

        logger.info(f"  Frame {frame_count}: {frame_type} at sample {pos} ({pos/sample_rate:.3f}s)")

        pos = end_pos + gap_samples
        frame_count += 1

    # Add some noise
    noise = np.random.normal(0, 0.01, total_samples).astype(np.float32)
    audio = audio + noise

    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9

    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)

    # Save
    wav.write(filename, sample_rate, audio_int16)
    logger.info(f"Saved {len(audio)} samples ({duration:.1f}s) to {filename}")
    logger.info(f"Total frames: {frame_count}")

    return filename


def main():
    import sys

    # Generate test file
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "p25_test_ldu.wav"

    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0

    generate_test_file(filename, duration=duration)

    # Quick verification
    logger.info("\nVerifying generated file...")
    sr, audio = wav.read(filename)
    logger.info(f"Read: {len(audio)} samples at {sr}Hz")
    logger.info(f"Audio range: [{audio.min()}, {audio.max()}]")
    logger.info(f"RMS: {np.sqrt(np.mean(audio.astype(float)**2)):.1f}")


if __name__ == "__main__":
    main()
