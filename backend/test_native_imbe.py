#!/usr/bin/env python3
"""Test the native IMBE decoder directly with audio files."""

import logging
import time
import numpy as np
import scipy.io.wavfile as wav

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

from wavecapsdr.decoders.imbe_native import IMBEDecoderNative, check_native_imbe_available


def test_with_discriminator_audio(filename: str) -> None:
    """Test decoder with discriminator audio from a WAV file."""

    # Check availability
    available, msg = check_native_imbe_available()
    print(f"Native IMBE available: {available}")
    print(f"Message: {msg}")

    if not available:
        print("Cannot test - mbelib-neo not available")
        return

    # Load audio file
    print(f"\nLoading {filename}...")
    sample_rate, audio = wav.read(filename)
    print(f"Sample rate: {sample_rate}")
    print(f"Audio shape: {audio.shape}")
    print(f"Audio dtype: {audio.dtype}")

    # Handle stereo - take first channel
    if audio.ndim > 1:
        print(f"Stereo audio detected, taking first channel")
        audio = audio[:, 0]

    # Convert to float32 if needed
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.float64:
        audio = audio.astype(np.float32)

    print(f"Audio range: {audio.min():.4f} to {audio.max():.4f}")
    print(f"Audio RMS: {np.sqrt(np.mean(audio**2)):.6f}")

    # Create decoder
    decoder = IMBEDecoderNative(output_rate=48000, input_rate=sample_rate)
    decoder.start()

    try:
        # Process in chunks
        chunk_size = 4800  # 100ms at 48kHz
        num_chunks = len(audio) // chunk_size

        print(f"\nProcessing {num_chunks} chunks of {chunk_size} samples each...")

        for i in range(num_chunks):
            chunk = audio[i * chunk_size : (i + 1) * chunk_size]
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
        while True:
            out_audio = decoder.get_audio()
            if out_audio is None:
                break
            total_output += len(out_audio)
            print(f"Final output: {len(out_audio)} samples")

        print(f"\n=== Results ===")
        print(f"Bytes processed: {decoder.bytes_processed}")
        print(f"LDU frames detected: {decoder.ldu_frames}")
        print(f"IMBE frames processed: {decoder.imbe_frames}")
        print(f"Frames decoded: {decoder.frames_decoded}")
        print(f"Frames dropped: {decoder.frames_dropped}")
        print(f"Total output samples: {total_output}")

    finally:
        decoder.stop()


def test_with_synthetic_c4fm() -> None:
    """Test with synthetic C4FM signal."""
    print("\n=== Testing with synthetic C4FM ===")

    available, msg = check_native_imbe_available()
    if not available:
        print("Cannot test - mbelib-neo not available")
        return

    # Generate synthetic 4FSK signal at 4800 baud
    sample_rate = 48000
    baud_rate = 4800
    samples_per_symbol = sample_rate // baud_rate  # 10 samples/symbol

    # Create some random dibits
    np.random.seed(42)
    num_symbols = 4800  # 1 second
    dibits = np.random.randint(0, 4, num_symbols)

    # Map dibits to deviation levels
    deviation_map = {0: -3, 1: -1, 2: 1, 3: 3}  # 4FSK levels
    deviations = np.array([deviation_map[d] for d in dibits])

    # Create samples (FM deviation as instantaneous frequency)
    # For discriminator audio, this is already the instantaneous frequency
    # Scale to match expected P25 deviation (~1800 Hz)
    max_deviation = 1800  # Hz
    norm_deviation = deviations / 3.0  # Normalize to [-1, 1]

    # Create upsampled discriminator signal
    disc_audio = np.repeat(norm_deviation, samples_per_symbol)

    # Add some noise
    noise = np.random.normal(0, 0.1, len(disc_audio))
    disc_audio = (disc_audio + noise).astype(np.float32)

    print(f"Generated {len(disc_audio)} samples of synthetic C4FM")
    print(f"Audio range: {disc_audio.min():.4f} to {disc_audio.max():.4f}")

    # Create decoder
    decoder = IMBEDecoderNative(output_rate=48000, input_rate=sample_rate)
    decoder.start()

    try:
        # Process in chunks
        chunk_size = 4800
        num_chunks = len(disc_audio) // chunk_size

        for i in range(num_chunks):
            chunk = disc_audio[i * chunk_size : (i + 1) * chunk_size]
            decoder.decode(chunk)

        # Wait for processing
        time.sleep(0.5)

        print(f"\n=== Synthetic Signal Results ===")
        print(f"Bytes processed: {decoder.bytes_processed}")
        print(f"LDU frames detected: {decoder.ldu_frames}")
        print(f"IMBE frames processed: {decoder.imbe_frames}")
        print(f"Frames decoded: {decoder.frames_decoded}")

        # Note: Random dibits won't produce valid P25 frames,
        # but we can verify the pipeline is running
        if decoder.bytes_processed > 0:
            print("✓ Decoder is processing data correctly")
        else:
            print("✗ Decoder did not process any data")

    finally:
        decoder.stop()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Test with provided file
        test_with_discriminator_audio(sys.argv[1])
    else:
        # Test with synthetic signal first
        test_with_synthetic_c4fm()

        # Then test with a live capture file
        print("\n" + "="*60 + "\n")
        test_with_discriminator_audio("wavecap_413450_live.wav")
