#!/usr/bin/env python3
"""Test native IMBE decoder with SDRTrunk baseband recordings.

SDRTrunk saves baseband as 50kHz IQ (2-channel WAV with I and Q).
We need to:
1. Load the IQ data
2. FM demodulate to get discriminator audio
3. Run through native IMBE decoder
"""

import logging
import time
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

from wavecapsdr.decoders.imbe_native import IMBEDecoderNative, check_native_imbe_available


def fm_demodulate(iq: np.ndarray) -> np.ndarray:
    """FM demodulate IQ signal to discriminator audio.

    Returns instantaneous frequency in radians per sample.
    """
    # Compute instantaneous phase
    phase = np.angle(iq)

    # Unwrap phase and take derivative
    phase_unwrapped = np.unwrap(phase)
    disc_audio = np.diff(phase_unwrapped)

    return disc_audio.astype(np.float32)


def test_with_iq_baseband(filename: str) -> None:
    """Test decoder with SDRTrunk IQ baseband recording."""

    # Check availability
    available, msg = check_native_imbe_available()
    print(f"Native IMBE available: {available}")
    print(f"Message: {msg}")

    if not available:
        print("Cannot test - mbelib-neo not available")
        return

    # Load IQ file
    print(f"\nLoading {filename}...")
    sample_rate, audio = wav.read(filename)
    print(f"Sample rate: {sample_rate}")
    print(f"Audio shape: {audio.shape}")
    print(f"Audio dtype: {audio.dtype}")
    print(f"Duration: {len(audio)/sample_rate:.1f}s")

    if audio.ndim != 2 or audio.shape[1] != 2:
        print("Error: Expected 2-channel IQ audio")
        return

    # Convert to complex IQ
    # Normalize int16 to float and combine I+jQ
    iq = (audio[:, 0] + 1j * audio[:, 1]).astype(np.complex64) / 32768.0
    print(f"IQ shape: {iq.shape}")
    print(f"IQ power: {np.mean(np.abs(iq)**2):.6f}")

    # FM demodulate
    print("\nFM demodulating...")
    disc_audio = fm_demodulate(iq)
    print(f"Discriminator audio shape: {disc_audio.shape}")
    print(f"Discriminator range: {disc_audio.min():.4f} to {disc_audio.max():.4f}")
    print(f"Discriminator RMS: {np.sqrt(np.mean(disc_audio**2)):.6f}")

    # Create decoder - SDRTrunk uses 50kHz, we need to resample or adjust
    # Actually, the native decoder handles resampling internally
    decoder = IMBEDecoderNative(output_rate=48000, input_rate=sample_rate)
    decoder.start()

    try:
        # Process in chunks
        chunk_size = sample_rate // 10  # 100ms chunks
        num_chunks = len(disc_audio) // chunk_size

        print(f"\nProcessing {num_chunks} chunks of {chunk_size} samples each...")

        for i in range(num_chunks):
            chunk = disc_audio[i * chunk_size : (i + 1) * chunk_size]
            decoder.decode(chunk)

            # Check for output periodically
            if i > 0 and i % 10 == 0:
                while True:
                    out_audio = decoder.get_audio()
                    if out_audio is None:
                        break
                    print(f"  Got audio output: {len(out_audio)} samples")

        # Wait for final processing
        time.sleep(0.5)

        # Collect any remaining output
        total_output = 0
        output_chunks = []
        while True:
            out_audio = decoder.get_audio()
            if out_audio is None:
                break
            total_output += len(out_audio)
            output_chunks.append(out_audio)
            print(f"Final output: {len(out_audio)} samples")

        print(f"\n=== Results ===")
        print(f"Bytes processed: {decoder.bytes_processed}")
        print(f"LDU frames detected: {decoder.ldu_frames}")
        print(f"IMBE frames processed: {decoder.imbe_frames}")
        print(f"Frames decoded: {decoder.frames_decoded}")
        print(f"Frames dropped: {decoder.frames_dropped}")
        print(f"Total output samples: {total_output}")

        # Save output if we got any
        if output_chunks:
            output_audio = np.concatenate(output_chunks)
            output_filename = filename.replace('.wav', '_decoded.wav')
            output_int16 = (output_audio * 32767).astype(np.int16)
            wav.write(output_filename, 48000, output_int16)
            print(f"Saved decoded audio to: {output_filename}")

    finally:
        decoder.stop()


def list_sdrtrunk_recordings():
    """List available SDRTrunk recordings."""
    import os

    recordings_dir = "/Users/thw/SDRTrunk/recordings"
    if not os.path.exists(recordings_dir):
        print(f"Recordings directory not found: {recordings_dir}")
        return []

    files = sorted([
        f for f in os.listdir(recordings_dir)
        if f.endswith('.wav')
    ])

    print(f"Found {len(files)} SDRTrunk recordings")
    for i, f in enumerate(files[:20]):
        size = os.path.getsize(os.path.join(recordings_dir, f))
        print(f"  {i+1}. {f} ({size/1024/1024:.1f} MB)")

    return files


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Test with provided file
        test_with_iq_baseband(sys.argv[1])
    else:
        # List recordings and use a small one
        print("Listing SDRTrunk recordings...")
        files = list_sdrtrunk_recordings()

        if files:
            # Pick a small T-Control-Channel (traffic/voice) recording
            traffic_files = [f for f in files if 'T-Control' in f]
            if traffic_files:
                # Use the first traffic channel recording
                filename = f"/Users/thw/SDRTrunk/recordings/{traffic_files[0]}"
                print(f"\nUsing traffic channel recording: {filename}")
                test_with_iq_baseband(filename)
            else:
                # Use a small control channel recording
                filename = f"/Users/thw/SDRTrunk/recordings/{files[0]}"
                print(f"\nUsing control channel recording: {filename}")
                test_with_iq_baseband(filename)
