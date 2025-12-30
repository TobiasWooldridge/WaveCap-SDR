#!/usr/bin/env python3
"""Test TSBK decoding on SDRTrunk baseband recording.

This script tests our P25 decoding pipeline against a known-good
SDRTrunk recording to isolate radio issues from decoder issues.
"""

import sys
import wave
import numpy as np

sys.path.insert(0, "/Users/thw/Projects/WaveCap-SDR/backend")

from wavecapsdr.trunking.control_channel import ControlChannelMonitor
from wavecapsdr.trunking.config import TrunkingProtocol


def load_iq_from_wav(wav_path: str) -> tuple[np.ndarray, int]:
    """Load IQ samples from a WAV file (stereo = I/Q interleaved)."""
    with wave.open(wav_path, 'rb') as w:
        sample_rate = w.getframerate()
        n_frames = w.getnframes()
        n_channels = w.getnchannels()
        sample_width = w.getsampwidth()

        print(f"WAV file: {wav_path}")
        print(f"  Channels: {n_channels}")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Sample width: {sample_width} bytes")
        print(f"  Frames: {n_frames}")
        print(f"  Duration: {n_frames / sample_rate:.2f} seconds")

        # Read raw bytes
        raw_data = w.readframes(n_frames)

        # Convert to numpy array
        if sample_width == 2:
            samples = np.frombuffer(raw_data, dtype=np.int16)
        elif sample_width == 4:
            samples = np.frombuffer(raw_data, dtype=np.int32)
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        # Reshape to (n_frames, n_channels)
        samples = samples.reshape(-1, n_channels)

        # Convert to complex IQ
        if n_channels == 2:
            # Stereo = I/Q
            i = samples[:, 0].astype(np.float64)
            q = samples[:, 1].astype(np.float64)
            # Normalize to -1 to 1
            max_val = 2 ** (sample_width * 8 - 1)
            i = i / max_val
            q = q / max_val
            iq = i + 1j * q
        else:
            raise ValueError(f"Expected stereo file, got {n_channels} channels")

        return iq, sample_rate


def test_decode(wav_path: str):
    """Test decoding a WAV file."""
    # Load IQ samples
    iq, sample_rate = load_iq_from_wav(wav_path)

    print(f"\nLoaded {len(iq)} samples")
    print(f"IQ power: {np.mean(np.abs(iq)**2):.6f}")
    print(f"IQ peak: {np.max(np.abs(iq)):.6f}")

    # Create control channel monitor
    # The recording is already at baseband, no decimation needed
    # ControlChannelMonitor expects 48 kHz sample rate
    # We'll resample from 50 kHz to 48 kHz
    target_rate = 48000
    if sample_rate != target_rate:
        from scipy import signal as scipy_signal
        # Resample to 48 kHz
        new_length = int(len(iq) * target_rate / sample_rate)
        iq_resampled = scipy_signal.resample(iq, new_length)
        print(f"Resampled from {sample_rate} Hz to {target_rate} Hz: {len(iq_resampled)} samples")
        iq = iq_resampled
        sample_rate = target_rate

    monitor = ControlChannelMonitor(protocol=TrunkingProtocol.P25_PHASE1, sample_rate=sample_rate)

    # Process in chunks
    chunk_size = 10000  # Same as IQ_BUFFER_MIN_SAMPLES
    total_tsbks = 0
    total_tsbk_pass = 0
    total_frames = 0

    print(f"\nProcessing {len(iq)} samples in chunks of {chunk_size}...")

    for i in range(0, len(iq), chunk_size):
        chunk = iq[i:i+chunk_size]
        if len(chunk) < chunk_size:
            break

        results = monitor.process_iq(chunk)

        for tsbk_data in results:
            if tsbk_data:
                total_tsbks += 1
                opcode = tsbk_data.get("opcode_name", tsbk_data.get("opcode", "?"))
                print(f"  TSBK #{total_tsbks}: {opcode}")

    # Get final stats
    stats = monitor.get_stats()
    print(f"\n=== Final Stats ===")
    print(f"Sync state: {stats.get('sync_state', '?')}")
    print(f"Frames decoded: {stats.get('frames_decoded', 0)}")
    print(f"TSBK attempts: {stats.get('tsbk_attempts', 0)}")
    print(f"TSBK CRC pass: {stats.get('tsbk_crc_pass', 0)}")
    if stats.get('tsbk_attempts', 0) > 0:
        rate = 100 * stats.get('tsbk_crc_pass', 0) / stats.get('tsbk_attempts', 1)
        print(f"CRC pass rate: {rate:.1f}%")
    print(f"Sync losses: {stats.get('sync_losses', 0)}")
    print(f"Total TSBKs passed: {total_tsbks}")


if __name__ == "__main__":
    wav_path = sys.argv[1] if len(sys.argv) > 1 else "20251227_224220_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav"
    test_decode(wav_path)
