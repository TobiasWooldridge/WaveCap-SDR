#!/usr/bin/env python3
"""Analyze IF spectrum from IQ WAV file to find carrier offset.

This script loads an IQ WAV file, computes the FFT to visualize the spectrum,
finds the peak frequency offset from DC (the carrier position), and determines
if frequency shifting is needed to bring the signal to baseband.

Usage:
    python scripts/analyze_if_spectrum.py
    python scripts/analyze_if_spectrum.py --input /path/to/file.wav
    python scripts/analyze_if_spectrum.py --input /path/to/file.wav --output /tmp/spectrum.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_iq_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load IQ data from stereo WAV file.

    Supports 16-bit and 24-bit WAV files using scipy for better compatibility.
    Falls back to wave module if scipy is not available.
    """
    try:
        from scipy.io import wavfile
        sample_rate, data = wavfile.read(str(path))

        if data.ndim == 1:
            raise ValueError("Expected stereo IQ WAV file, got mono")

        n_channels = data.shape[1]
        if n_channels != 2:
            raise ValueError(f"Expected stereo IQ WAV, got {n_channels} channels")

        # Normalize based on dtype
        if data.dtype == np.int16:
            samples_f = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            # scipy reads 24-bit as int32, values are in upper 24 bits
            # or it may read them scaled - check the range
            max_val = np.max(np.abs(data))
            if max_val > 8388608:
                # Scaled as 32-bit
                samples_f = data.astype(np.float32) / 2147483648.0
            else:
                # 24-bit values
                samples_f = data.astype(np.float32) / 8388608.0
        elif data.dtype == np.float32 or data.dtype == np.float64:
            samples_f = data.astype(np.float32)
        else:
            raise ValueError(f"Unsupported WAV dtype: {data.dtype}")

        iq = samples_f[:, 0] + 1j * samples_f[:, 1]
        logger.info(f"Loaded via scipy: {len(iq)} samples, dtype={data.dtype}")
        return iq, sample_rate

    except ImportError:
        logger.warning("scipy not available, falling back to wave module")

    # Fallback to wave module
    import wave
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

    if n_channels != 2:
        raise ValueError(f"Expected stereo IQ WAV, got {n_channels} channels")

    if sample_width == 2:
        samples = np.frombuffer(raw_data, dtype=np.int16).reshape(-1, n_channels)
        samples_f = samples.astype(np.float32) / 32768.0
    elif sample_width == 3:
        # 24-bit samples need manual unpacking
        data_bytes = np.frombuffer(raw_data, dtype=np.uint8).reshape(-1, n_channels, 3)
        samples = np.zeros((n_frames, n_channels), dtype=np.int32)
        for ch in range(n_channels):
            b0 = data_bytes[:, ch, 0].astype(np.int32)
            b1 = data_bytes[:, ch, 1].astype(np.int32)
            b2 = data_bytes[:, ch, 2].astype(np.int32)
            raw24 = b0 | (b1 << 8) | (b2 << 16)
            samples[:, ch] = np.where(raw24 >= 0x800000, raw24 - 0x1000000, raw24)
        samples_f = samples.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        samples = np.frombuffer(raw_data, dtype=np.int32).reshape(-1, n_channels)
        samples_f = samples.astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    iq = samples_f[:, 0] + 1j * samples_f[:, 1]
    logger.info(f"Loaded via wave: {len(iq)} samples, sample_width={sample_width}")
    return iq, sample_rate


def compute_spectrum(iq: np.ndarray, sample_rate: int, fft_size: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Compute averaged power spectrum using Welch's method (averaged periodograms).

    Returns:
        freqs: Frequency bins in Hz (centered on DC)
        power_db: Power spectral density in dB
    """
    # Use overlapping segments for better averaging
    segment_size = fft_size
    overlap = segment_size // 2
    step = segment_size - overlap

    n_segments = max(1, (len(iq) - overlap) // step)

    # Apply window function
    window = np.hanning(segment_size)

    # Accumulate power spectra
    power_sum = np.zeros(segment_size, dtype=np.float64)

    for i in range(n_segments):
        start = i * step
        end = start + segment_size
        if end > len(iq):
            break

        segment = iq[start:end] * window
        spectrum = np.fft.fft(segment)
        power_sum += np.abs(spectrum) ** 2

    # Average and shift to center DC
    power_avg = power_sum / n_segments
    power_avg_shifted = np.fft.fftshift(power_avg)

    # Convert to dB
    power_db = 10 * np.log10(power_avg_shifted + 1e-12)  # Add small value to avoid log(0)

    # Generate frequency axis
    freqs = np.fft.fftshift(np.fft.fftfreq(segment_size, 1.0 / sample_rate))

    logger.info(f"Computed spectrum: {n_segments} segments, FFT size={fft_size}")
    return freqs, power_db


def find_peak_offset(freqs: np.ndarray, power_db: np.ndarray,
                     min_freq: float = 100.0, max_freq: float = None) -> tuple[float, float]:
    """Find the peak frequency offset from DC.

    Args:
        freqs: Frequency bins in Hz
        power_db: Power in dB
        min_freq: Minimum frequency offset to consider (to avoid DC spike)
        max_freq: Maximum frequency offset to consider

    Returns:
        peak_freq: Peak frequency offset in Hz
        peak_power: Power at peak in dB
    """
    if max_freq is None:
        max_freq = np.max(np.abs(freqs)) * 0.9  # 90% of Nyquist

    # Mask frequencies outside the search range
    mask = (np.abs(freqs) >= min_freq) & (np.abs(freqs) <= max_freq)

    if not np.any(mask):
        logger.warning("No frequencies in search range")
        return 0.0, 0.0

    # Find peak
    masked_power = np.where(mask, power_db, -np.inf)
    peak_idx = np.argmax(masked_power)
    peak_freq = freqs[peak_idx]
    peak_power = power_db[peak_idx]

    # Refine peak using quadratic interpolation
    if 0 < peak_idx < len(freqs) - 1:
        y0, y1, y2 = power_db[peak_idx - 1], power_db[peak_idx], power_db[peak_idx + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-10:
            delta = 0.5 * (y0 - y2) / denom
            freq_step = freqs[1] - freqs[0]
            peak_freq = peak_freq + delta * freq_step

    return float(peak_freq), float(peak_power)


def plot_spectrum(freqs: np.ndarray, power_db: np.ndarray,
                  peak_freq: float, peak_power: float,
                  sample_rate: int, output_path: Path) -> None:
    """Plot the spectrum and save to file."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot spectrum
    ax.plot(freqs / 1000, power_db, 'b-', linewidth=0.5, alpha=0.8)

    # Mark peak
    ax.axvline(x=peak_freq / 1000, color='r', linestyle='--', linewidth=1.5,
               label=f'Peak: {peak_freq:+.1f} Hz')
    ax.plot(peak_freq / 1000, peak_power, 'ro', markersize=8)

    # Mark DC
    ax.axvline(x=0, color='g', linestyle=':', linewidth=1, alpha=0.5, label='DC')

    # Labels and title
    ax.set_xlabel('Frequency Offset (kHz)')
    ax.set_ylabel('Power (dB)')
    ax.set_title(f'IF Spectrum Analysis\nSample Rate: {sample_rate/1000:.1f} kHz, '
                 f'Peak Offset: {peak_freq:+.1f} Hz ({peak_freq/1000:+.2f} kHz)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Set reasonable y-axis limits
    y_max = np.max(power_db)
    y_min = max(np.min(power_db), y_max - 80)  # Show at most 80 dB dynamic range
    ax.set_ylim(y_min, y_max + 5)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()

    logger.info(f"Saved spectrum plot to {output_path}")


def analyze_spectrum(input_path: Path, output_path: Path, fft_size: int = 4096) -> dict:
    """Main analysis function.

    Returns dict with analysis results.
    """
    # Load IQ data
    logger.info(f"Loading IQ data from {input_path}")
    iq, sample_rate = load_iq_wav(input_path)

    duration = len(iq) / sample_rate
    logger.info(f"Sample rate: {sample_rate} Hz")
    logger.info(f"Duration: {duration:.3f} seconds ({len(iq)} samples)")

    # Compute spectrum
    freqs, power_db = compute_spectrum(iq, sample_rate, fft_size=fft_size)

    # Find peak
    # For P25 C4FM, the signal bandwidth is about 12.5 kHz, so look within that range
    max_search_freq = min(sample_rate / 2 * 0.9, 15000)  # Up to 15 kHz or 90% of Nyquist
    peak_freq, peak_power = find_peak_offset(freqs, power_db,
                                              min_freq=100,
                                              max_freq=max_search_freq)

    # Plot
    plot_spectrum(freqs, power_db, peak_freq, peak_power, sample_rate, output_path)

    # Calculate statistics
    noise_floor = np.median(power_db)
    snr = peak_power - noise_floor

    results = {
        "input_file": str(input_path),
        "sample_rate": sample_rate,
        "duration_seconds": duration,
        "num_samples": len(iq),
        "peak_offset_hz": peak_freq,
        "peak_power_db": peak_power,
        "noise_floor_db": noise_floor,
        "snr_db": snr,
        "output_plot": str(output_path),
    }

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze IF spectrum to find carrier offset"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("/tmp/sigid_C4FM_CC_IF.wav"),
        help="Input IQ WAV file (default: /tmp/sigid_C4FM_CC_IF.wav)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("/tmp/spectrum_analysis.png"),
        help="Output spectrum plot (default: /tmp/spectrum_analysis.png)"
    )
    parser.add_argument(
        "--fft-size",
        type=int,
        default=4096,
        help="FFT size for spectrum computation (default: 4096)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Check input file exists
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        print(f"\nError: Input file not found: {args.input}")
        print("Please provide a valid IQ WAV file.")
        return 1

    try:
        results = analyze_spectrum(args.input, args.output, fft_size=args.fft_size)
    except Exception as e:
        logger.exception(f"Analysis failed: {e}")
        return 1

    # Print results
    print("\n" + "=" * 60)
    print("IF SPECTRUM ANALYSIS RESULTS")
    print("=" * 60)
    print(f"  Input file:     {results['input_file']}")
    print(f"  Sample rate:    {results['sample_rate']} Hz")
    print(f"  Duration:       {results['duration_seconds']:.3f} seconds")
    print(f"  Samples:        {results['num_samples']:,}")
    print()
    print(f"  Peak offset:    {results['peak_offset_hz']:+.1f} Hz ({results['peak_offset_hz']/1000:+.3f} kHz)")
    print(f"  Peak power:     {results['peak_power_db']:.1f} dB")
    print(f"  Noise floor:    {results['noise_floor_db']:.1f} dB")
    print(f"  SNR:            {results['snr_db']:.1f} dB")
    print()
    print(f"  Spectrum plot:  {results['output_plot']}")
    print("=" * 60)

    # Provide recommendations
    print("\nRECOMMENDATION:")
    offset = results['peak_offset_hz']
    if abs(offset) < 500:
        print("  Signal appears to be near baseband (offset < 500 Hz).")
        print("  Frequency shifting is likely NOT needed.")
    else:
        print(f"  Carrier is offset by {offset:+.1f} Hz from DC.")
        print(f"  To bring to baseband, shift by {-offset:+.1f} Hz")
        print(f"  (multiply by exp(-j * 2 * pi * {-offset:.1f} * t))")

    if results['snr_db'] < 10:
        print(f"\n  WARNING: Low SNR ({results['snr_db']:.1f} dB). Signal may be weak or noisy.")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
