#!/usr/bin/env python3
"""
Audio Quality Analyzer for WaveCap-SDR

Captures a sample of audio from a WaveCap-SDR channel and analyzes it to determine
if the channel is properly tuned and producing usable audio.
"""

import argparse
import sys
import numpy as np
import requests
from typing import Tuple, Dict
import struct


def capture_audio_stream(
    host: str, port: int, channel_id: str, duration: float, format: str = "pcm16"
) -> Tuple[np.ndarray, int]:
    """
    Capture audio from the WaveCap-SDR stream.

    Args:
        host: Server hostname/IP
        port: Server port
        channel_id: Channel ID to capture from
        duration: Duration in seconds to capture
        format: Audio format (pcm16 or f32)

    Returns:
        Tuple of (audio_data, sample_rate)
    """
    url = f"http://{host}:{port}/api/v1/stream/channels/{channel_id}.pcm"
    params = {"format": format}

    print(f"Connecting to {url}...")

    try:
        response = requests.get(url, params=params, stream=True, timeout=5)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to stream: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Is the server running on {host}:{port}?")
        print(f"  2. Does channel '{channel_id}' exist?")
        print(f"  3. Is the channel started?")
        sys.exit(1)

    # Extract sample rate from headers
    sample_rate = int(response.headers.get("X-Audio-Rate", 48000))
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Capturing {duration} seconds of audio...")

    # Calculate bytes to read
    if format == "pcm16":
        bytes_per_sample = 2
        dtype = np.int16
    elif format == "f32":
        bytes_per_sample = 4
        dtype = np.float32
    else:
        raise ValueError(f"Unsupported format: {format}")

    total_samples = int(sample_rate * duration)
    total_bytes = total_samples * bytes_per_sample

    # Read audio data
    audio_bytes = bytearray()
    bytes_read = 0

    for chunk in response.iter_content(chunk_size=4096):
        if not chunk:
            break
        audio_bytes.extend(chunk)
        bytes_read += len(chunk)

        if bytes_read >= total_bytes:
            break

    if bytes_read == 0:
        print("Error: No data received from stream")
        sys.exit(1)

    # Truncate to exact size needed
    audio_bytes = audio_bytes[:total_bytes]

    # Convert to numpy array
    audio_data = np.frombuffer(audio_bytes, dtype=dtype)

    # Normalize to [-1.0, 1.0] range
    if format == "pcm16":
        audio_data = audio_data.astype(np.float32) / 32768.0

    print(f"Captured {len(audio_data)} samples ({len(audio_data)/sample_rate:.2f} seconds)")

    return audio_data, sample_rate


def calculate_rms_db(signal: np.ndarray) -> float:
    """Calculate RMS level in dB."""
    rms = np.sqrt(np.mean(signal**2))
    if rms > 0:
        return 20 * np.log10(rms)
    else:
        return -np.inf


def calculate_peak_db(signal: np.ndarray) -> float:
    """Calculate peak level in dB."""
    peak = np.max(np.abs(signal))
    if peak > 0:
        return 20 * np.log10(peak)
    else:
        return -np.inf


def calculate_crest_factor(signal: np.ndarray) -> float:
    """Calculate crest factor (peak-to-RMS ratio)."""
    peak = np.max(np.abs(signal))
    rms = np.sqrt(np.mean(signal**2))
    if rms > 0:
        return peak / rms
    else:
        return 0.0


def calculate_zero_crossing_rate(signal: np.ndarray) -> float:
    """Calculate zero crossing rate (normalized)."""
    zero_crossings = np.sum(np.abs(np.diff(np.sign(signal))))
    return zero_crossings / (2 * len(signal))


def calculate_spectral_flatness(signal: np.ndarray) -> float:
    """
    Calculate spectral flatness (Wiener entropy).

    Value near 1.0 indicates noise-like signal (flat spectrum).
    Value near 0.0 indicates tonal signal (peaked spectrum).
    """
    # Compute power spectrum
    spectrum = np.abs(np.fft.rfft(signal))**2

    # Avoid log(0) by adding small epsilon
    epsilon = 1e-10
    spectrum = spectrum + epsilon

    # Geometric mean
    geometric_mean = np.exp(np.mean(np.log(spectrum)))

    # Arithmetic mean
    arithmetic_mean = np.mean(spectrum)

    # Spectral flatness
    if arithmetic_mean > 0:
        return geometric_mean / arithmetic_mean
    else:
        return 0.0


def calculate_spectral_centroid(signal: np.ndarray, sample_rate: int) -> float:
    """Calculate spectral centroid in Hz."""
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1/sample_rate)

    # Weighted average of frequencies
    if np.sum(spectrum) > 0:
        return np.sum(freqs * spectrum) / np.sum(spectrum)
    else:
        return 0.0


def analyze_audio(audio_data: np.ndarray, sample_rate: int) -> Dict:
    """
    Perform comprehensive audio analysis.

    Args:
        audio_data: Audio samples normalized to [-1.0, 1.0]
        sample_rate: Sample rate in Hz

    Returns:
        Dictionary of analysis metrics
    """
    results = {
        "rms_db": calculate_rms_db(audio_data),
        "peak_db": calculate_peak_db(audio_data),
        "crest_factor": calculate_crest_factor(audio_data),
        "zero_crossing_rate": calculate_zero_crossing_rate(audio_data),
        "spectral_flatness": calculate_spectral_flatness(audio_data),
        "spectral_centroid": calculate_spectral_centroid(audio_data, sample_rate),
    }

    return results


def classify_signal(results: Dict) -> Tuple[str, str, str]:
    """
    Classify the signal quality based on analysis metrics.

    Args:
        results: Dictionary of analysis metrics

    Returns:
        Tuple of (classification, emoji, description)
    """
    rms_db = results["rms_db"]
    peak_db = results["peak_db"]
    spectral_flatness = results["spectral_flatness"]
    crest_factor = results["crest_factor"]

    # Classification logic
    if rms_db < -60:
        return "SILENCE", "🔇", "No signal detected - channel may be stopped or broken"

    elif peak_db > -0.5:
        return "CLIPPED", "📊", "Signal is clipping - reduce gain or check for overmodulation"

    elif spectral_flatness > 0.7 and rms_db < -20:
        return "NOISE", "📡", "Noise-like signal - channel may be poorly tuned or no carrier present"

    elif rms_db > -40 and spectral_flatness < 0.6:
        return "GOOD AUDIO", "✅", "Structured audio signal detected - channel appears well-tuned"

    elif spectral_flatness < 0.5:
        return "WEAK AUDIO", "⚠️", "Audio signal present but weak - check tuning and gain"

    else:
        return "UNCERTAIN", "❓", "Signal characteristics unclear - may need manual verification"


def print_analysis_report(results: Dict, classification: Tuple[str, str, str]):
    """Print a formatted analysis report."""
    status, emoji, description = classification

    print("\n" + "="*70)
    print(f"  AUDIO QUALITY ANALYSIS REPORT")
    print("="*70)
    print()

    print(f"Overall Assessment: {emoji} {status}")
    print(f"  {description}")
    print()

    print("Signal Level Metrics:")
    print("-"*70)
    print(f"  RMS Level:        {results['rms_db']:>8.2f} dB")
    print(f"  Peak Level:       {results['peak_db']:>8.2f} dB")
    print(f"  Crest Factor:     {results['crest_factor']:>8.2f}")
    print()

    print("Spectral Analysis:")
    print("-"*70)
    print(f"  Spectral Flatness:   {results['spectral_flatness']:>6.4f}  (1.0=noise, 0.0=tonal)")
    print(f"  Spectral Centroid:   {results['spectral_centroid']:>6.0f} Hz")
    print(f"  Zero Crossing Rate:  {results['zero_crossing_rate']:>6.4f}")
    print()

    print("Interpretation Guide:")
    print("-"*70)

    # RMS interpretation
    rms_db = results['rms_db']
    if rms_db < -60:
        rms_status = "Silent - No signal"
    elif rms_db < -40:
        rms_status = "Very quiet - Possibly just noise"
    elif rms_db < -20:
        rms_status = "Moderate - Could be good signal"
    elif rms_db < -6:
        rms_status = "Good level - Normal audio"
    else:
        rms_status = "Hot - May be overdriven"
    print(f"  RMS Level: {rms_status}")

    # Spectral flatness interpretation
    sf = results['spectral_flatness']
    if sf > 0.8:
        sf_status = "Very noise-like (white noise/static)"
    elif sf > 0.6:
        sf_status = "Somewhat noisy (weak or no carrier)"
    elif sf > 0.3:
        sf_status = "Mixed (some tonal content)"
    else:
        sf_status = "Tonal (structured audio/speech/music)"
    print(f"  Spectral Character: {sf_status}")

    # Peak level interpretation
    peak_db = results['peak_db']
    if peak_db > -0.5:
        peak_status = "Clipping detected!"
    elif peak_db > -3:
        peak_status = "Very close to clipping"
    elif peak_db > -10:
        peak_status = "Good headroom"
    else:
        peak_status = "Lots of headroom"
    print(f"  Peak Level: {peak_status}")

    print()
    print("="*70)
    print()

    # Recommendations
    if status == "SILENCE":
        print("Recommendations:")
        print("  • Check if channel is started")
        print("  • Verify capture device is running")
        print("  • Check antenna/device connections")

    elif status == "NOISE":
        print("Recommendations:")
        print("  • Try adjusting channel frequency (offset_hz)")
        print("  • Verify you're tuned to an active FM station")
        print("  • Check antenna connection and positioning")
        print("  • Ensure correct modulation mode (WBFM)")

    elif status == "CLIPPED":
        print("Recommendations:")
        print("  • Reduce SDR device gain settings")
        print("  • Check for overmodulation from broadcaster")
        print("  • Review RF gain configuration")

    elif status == "WEAK AUDIO":
        print("Recommendations:")
        print("  • Fine-tune frequency offset for better signal")
        print("  • May need to increase gain slightly")
        print("  • Verify signal strength at this frequency")

    elif status == "GOOD AUDIO":
        print("Recommendations:")
        print("  • Channel is working well!")
        print("  • Optionally adjust squelch_db to reduce inter-signal noise")
        print("  • No changes needed")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze audio quality from WaveCap-SDR channel stream"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8087,
        help="Server port (default: 8087)"
    )
    parser.add_argument(
        "--channel",
        default="ch1",
        help="Channel ID to analyze (default: ch1)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Duration to capture in seconds (default: 3.0)"
    )
    parser.add_argument(
        "--format",
        choices=["pcm16", "f32"],
        default="pcm16",
        help="Audio format (default: pcm16)"
    )

    args = parser.parse_args()

    # Capture audio
    try:
        audio_data, sample_rate = capture_audio_stream(
            args.host, args.port, args.channel, args.duration, args.format
        )
    except KeyboardInterrupt:
        print("\nCapture cancelled by user")
        sys.exit(0)

    # Analyze audio
    print("\nAnalyzing audio...")
    results = analyze_audio(audio_data, sample_rate)

    # Classify signal
    classification = classify_signal(results)

    # Print report
    print_analysis_report(results, classification)


if __name__ == "__main__":
    main()
