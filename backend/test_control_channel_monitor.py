#!/usr/bin/env python3
"""Test ControlChannelMonitor with SDRTrunk baseband recording.

This simulates the live SDR path by feeding IQ samples through
the ControlChannelMonitor in chunks (like the live system does).
"""

import sys
import wave
import numpy as np
import logging

# Enable detailed logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add wavecapsdr to path
sys.path.insert(0, '/Users/thw/Projects/WaveCap-SDR/backend')

from wavecapsdr.trunking.control_channel import (
    ControlChannelMonitor, create_control_monitor, P25Modulation
)
from wavecapsdr.trunking.config import TrunkingProtocol


def load_baseband_wav(filepath: str) -> tuple[np.ndarray, int]:
    """Load SDRTrunk baseband recording (complex float32 or int16 in WAV)."""
    with wave.open(filepath, 'rb') as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()

        print(f"WAV: {n_channels} ch, {sample_width} bytes/sample, {sample_rate} Hz, {n_frames} frames")

        raw = wf.readframes(n_frames)

        # SDRTrunk baseband: 2 channels (I/Q)
        if n_channels == 2 and sample_width == 4:
            # Interleaved I/Q as float32
            data = np.frombuffer(raw, dtype=np.float32)
            i = data[0::2]
            q = data[1::2]
            return (i + 1j * q).astype(np.complex64), sample_rate
        elif n_channels == 2 and sample_width == 2:
            # Interleaved I/Q as int16 (normalized to float)
            data = np.frombuffer(raw, dtype=np.int16)
            i = data[0::2].astype(np.float32) / 32768.0
            q = data[1::2].astype(np.float32) / 32768.0
            return (i + 1j * q).astype(np.complex64), sample_rate
        else:
            raise ValueError(f"Unexpected format: {n_channels} ch, {sample_width} bytes")


def test_control_monitor_chunked(recording_path: str, duration_sec: float = 30.0, chunk_samples: int = 4000):
    """Test ControlChannelMonitor by feeding IQ in chunks (simulating live SDR)."""
    print(f"\n=== Testing ControlChannelMonitor (chunked) on: {recording_path} ===")

    # Load recording
    iq, sample_rate = load_baseband_wav(recording_path)
    max_samples = int(duration_sec * sample_rate)
    if len(iq) > max_samples:
        iq = iq[:max_samples]

    print(f"Processing {len(iq)} samples ({len(iq)/sample_rate:.2f} sec) at {sample_rate} Hz")
    print(f"Chunk size: {chunk_samples} samples (~{chunk_samples/(sample_rate/4800):.0f} symbols)")

    # Create ControlChannelMonitor with same sample rate as recording
    monitor = create_control_monitor(
        protocol=TrunkingProtocol.P25_PHASE1,
        sample_rate=sample_rate,
        modulation=P25Modulation.C4FM,
    )
    print(f"Created ControlChannelMonitor at {sample_rate} Hz")

    # Process in chunks (simulating live SDR)
    all_results = []
    num_chunks = (len(iq) + chunk_samples - 1) // chunk_samples

    for i in range(num_chunks):
        start = i * chunk_samples
        end = min(start + chunk_samples, len(iq))
        chunk = iq[start:end]

        results = monitor.process_iq(chunk)
        all_results.extend(results)

        # Progress every 100 chunks
        if (i + 1) % 100 == 0:
            stats = monitor.get_stats()
            print(f"  Chunk {i+1}/{num_chunks}: state={stats['sync_state']}, "
                  f"tsbk_attempts={stats['tsbk_attempts']}, crc_pass_rate={stats['tsbk_crc_pass_rate']:.1f}%")

    # Final stats
    stats = monitor.get_stats()
    print(f"\n=== Final Results ===")
    print(f"Sync state: {stats['sync_state']}")
    print(f"Frames decoded: {stats['frames_decoded']}")
    print(f"TSBK attempts: {stats['tsbk_attempts']}")
    print(f"TSBK CRC pass: {stats['tsbk_crc_pass']}")
    print(f"TSBK CRC pass rate: {stats['tsbk_crc_pass_rate']:.1f}%")
    print(f"Sync losses: {stats['sync_losses']}")
    print(f"Parsed TSBK results: {len(all_results)}")

    # Show sample results
    if all_results:
        print(f"\nSample decoded TSBK messages:")
        for result in all_results[:5]:
            print(f"  {result.get('opcode_name', 'UNKNOWN')} - {result}")

    return stats['tsbk_crc_pass'] > 0


def test_control_monitor_batch(recording_path: str, duration_sec: float = 30.0):
    """Test ControlChannelMonitor by feeding all IQ at once (like test_baseband_decode.py)."""
    print(f"\n=== Testing ControlChannelMonitor (batch) on: {recording_path} ===")

    # Load recording
    iq, sample_rate = load_baseband_wav(recording_path)
    max_samples = int(duration_sec * sample_rate)
    if len(iq) > max_samples:
        iq = iq[:max_samples]

    print(f"Processing {len(iq)} samples ({len(iq)/sample_rate:.2f} sec) at {sample_rate} Hz")

    # Create ControlChannelMonitor
    monitor = create_control_monitor(
        protocol=TrunkingProtocol.P25_PHASE1,
        sample_rate=sample_rate,
        modulation=P25Modulation.C4FM,
    )

    # Process ALL at once
    results = monitor.process_iq(iq)

    # Stats
    stats = monitor.get_stats()
    print(f"\n=== Batch Results ===")
    print(f"Sync state: {stats['sync_state']}")
    print(f"Frames decoded: {stats['frames_decoded']}")
    print(f"TSBK attempts: {stats['tsbk_attempts']}")
    print(f"TSBK CRC pass: {stats['tsbk_crc_pass']}")
    print(f"TSBK CRC pass rate: {stats['tsbk_crc_pass_rate']:.1f}%")
    print(f"Sync losses: {stats['sync_losses']}")
    print(f"Parsed TSBK results: {len(results)}")

    return stats['tsbk_crc_pass'] > 0


if __name__ == "__main__":
    recording = "/Users/thw/SDRTrunk/recordings/20251227_121743_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav"

    # Test batch mode (all at once)
    batch_success = test_control_monitor_batch(recording, duration_sec=30.0)

    # Test chunked mode with different chunk sizes
    chunk_sizes = [4000, 10000, 25000, 50000]
    chunk_results = {}

    for chunk_size in chunk_sizes:
        print(f"\n{'='*60}")
        chunk_success = test_control_monitor_chunked(
            recording, duration_sec=30.0, chunk_samples=chunk_size
        )
        chunk_results[chunk_size] = chunk_success

    print(f"\n=== Summary ===")
    print(f"Batch mode:         {'PASS' if batch_success else 'FAIL'}")
    for chunk_size, success in chunk_results.items():
        print(f"Chunked {chunk_size:>5} smp: {'PASS' if success else 'FAIL'}")

    sys.exit(0 if (batch_success and all(chunk_results.values())) else 1)
