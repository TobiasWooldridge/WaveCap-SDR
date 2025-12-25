"""Test CQPSK demodulator with known-good sample files."""

import wave
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from wavecapsdr.decoders.p25 import CQPSKDemodulator, P25FrameSync
from wavecapsdr.trunking.control_channel import ControlChannelMonitor, P25Modulation


def load_iq_wav(filepath: str) -> tuple[np.ndarray, int]:
    """Load IQ data from a stereo WAV file.

    Args:
        filepath: Path to WAV file (stereo, I in left channel, Q in right)

    Returns:
        Tuple of (complex IQ array, sample rate)
    """
    with wave.open(filepath, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()

        logger.info(f"Loading {filepath}: {n_frames} frames, {sample_rate} Hz, "
                   f"{n_channels} channels, {sample_width*8}-bit")

        # Read raw data
        raw_data = wf.readframes(n_frames)

        # Convert to numpy array based on sample width
        if sample_width == 3:  # 24-bit
            # Read as bytes, reshape to 3 bytes per sample
            data_bytes = np.frombuffer(raw_data, dtype=np.uint8)
            # Reshape to (samples, channels, 3 bytes)
            data_bytes = data_bytes.reshape(-1, n_channels, 3)

            # Convert 24-bit to 32-bit signed integers
            # Pad with sign extension
            samples = np.zeros((n_frames, n_channels), dtype=np.int32)
            for ch in range(n_channels):
                b0 = data_bytes[:, ch, 0].astype(np.int32)
                b1 = data_bytes[:, ch, 1].astype(np.int32)
                b2 = data_bytes[:, ch, 2].astype(np.int32)
                # Little-endian 24-bit to 32-bit with sign extension
                raw = b0 | (b1 << 8) | (b2 << 16)
                # Sign extend from 24-bit
                samples[:, ch] = np.where(raw >= 0x800000, raw - 0x1000000, raw)

            # Normalize to [-1, 1]
            samples = samples.astype(np.float32) / 8388608.0  # 2^23

        elif sample_width == 2:  # 16-bit
            samples = np.frombuffer(raw_data, dtype=np.int16).reshape(-1, n_channels)
            samples = samples.astype(np.float32) / 32768.0

        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        # Convert stereo (I, Q) to complex
        if n_channels == 2:
            iq = samples[:, 0] + 1j * samples[:, 1]
        else:
            raise ValueError(f"Expected stereo file, got {n_channels} channels")

        return iq, sample_rate


def test_cqpsk_control_channel():
    """Test CQPSK demodulator on control channel sample."""
    filepath = "tests/fixtures/p25_samples/P25_CQPSK-CC_IF.wav"

    # Load IQ data
    iq, sample_rate = load_iq_wav(filepath)
    logger.info(f"Loaded {len(iq)} samples at {sample_rate} Hz")
    logger.info(f"IQ magnitude: mean={np.mean(np.abs(iq)):.4f}, max={np.max(np.abs(iq)):.4f}")

    # Create demodulator at file's sample rate
    demod = CQPSKDemodulator(sample_rate=sample_rate, symbol_rate=4800)
    frame_sync = P25FrameSync()

    # Process in chunks
    chunk_size = sample_rate // 10  # 100ms chunks
    all_dibits = []
    tsbk_count = 0
    tsbk_valid = 0

    for i in range(0, len(iq), chunk_size):
        chunk = iq[i:i+chunk_size]
        if len(chunk) < 100:
            break

        # Demodulate
        dibits = demod.demodulate(chunk)
        all_dibits.extend(dibits)

        # Try to find frames
        if len(all_dibits) > 400:
            dibits_arr = np.array(all_dibits, dtype=np.uint8)
            pos, frame_type, nac, duid = frame_sync.find_sync(dibits_arr)

            if pos is not None:
                logger.info(f"Found sync at {pos}: {frame_type} NAC={nac:03X}")
                # Consume dibits up to after the frame
                all_dibits = all_dibits[pos + 360:]  # Approximate TSDU frame length

    logger.info(f"Processed {len(iq)} samples, {len(all_dibits)} residual dibits")


def test_with_control_monitor():
    """Test with full ControlChannelMonitor."""
    from wavecapsdr.trunking.config import TrunkingProtocol

    filepath = "tests/fixtures/p25_samples/P25_CQPSK-CC_IF.wav"

    # Load IQ data
    iq, sample_rate = load_iq_wav(filepath)
    logger.info(f"Loaded {len(iq)} samples at {sample_rate} Hz")

    # Create control channel monitor
    monitor = ControlChannelMonitor(
        protocol=TrunkingProtocol.P25_PHASE1,
        sample_rate=sample_rate,
        modulation=P25Modulation.LSM,
    )

    # Process in chunks
    chunk_size = sample_rate // 10  # 100ms chunks
    all_tsbks = []

    for i in range(0, len(iq), chunk_size):
        chunk = iq[i:i+chunk_size]
        if len(chunk) < 100:
            break

        # Process through monitor
        tsbks = monitor.process_iq(chunk.astype(np.complex64))
        all_tsbks.extend(tsbks)

    logger.info(f"Decoded {len(all_tsbks)} TSBKs")
    for tsbk in all_tsbks[:10]:  # Show first 10
        logger.info(f"  TSBK: {tsbk}")


if __name__ == "__main__":
    import sys
    import os

    # Change to backend directory
    os.chdir("/Users/thw/Projects/WaveCap-SDR/backend")

    print("=== Testing CQPSK with known-good sample ===\n")

    print("--- Test 1: Raw demodulator ---")
    test_cqpsk_control_channel()

    print("\n--- Test 2: Full control monitor ---")
    test_with_control_monitor()
