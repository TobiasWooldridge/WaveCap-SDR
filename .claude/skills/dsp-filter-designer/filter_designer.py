#!/usr/bin/env python3
"""
DSP Filter Designer for WaveCap-SDR

Design, visualize, and test digital filters for audio processing.
"""

import argparse
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def design_filter(
    filter_type: str,
    cutoff: Tuple[float, ...],
    sample_rate: int,
    order: int,
    filter_design: str,
    ripple: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a digital filter with specified parameters.

    Args:
        filter_type: 'lowpass', 'highpass', 'bandpass', or 'notch'
        cutoff: Cutoff frequency(ies) in Hz
        sample_rate: Sample rate in Hz
        order: Filter order
        filter_design: 'butterworth', 'chebyshev1', 'chebyshev2', 'elliptic'
        ripple: Passband ripple in dB (for Chebyshev/Elliptic)

    Returns:
        (b, a): Filter coefficients (numerator, denominator)
    """
    nyquist = sample_rate / 2

    # Normalize cutoff frequency(ies)
    if filter_type in ['bandpass', 'bandstop', 'notch']:
        if len(cutoff) != 2:
            raise ValueError(f"{filter_type} filter requires two cutoff frequencies")
        Wn = [cutoff[0] / nyquist, cutoff[1] / nyquist]
        btype = 'bandstop' if filter_type == 'notch' else 'bandpass'
    else:
        if len(cutoff) != 1:
            raise ValueError(f"{filter_type} filter requires one cutoff frequency")
        Wn = cutoff[0] / nyquist
        btype = filter_type

    # Design filter based on type
    if filter_design == 'butterworth':
        b, a = signal.butter(order, Wn, btype=btype, analog=False)
    elif filter_design == 'chebyshev1':
        b, a = signal.cheby1(order, ripple, Wn, btype=btype, analog=False)
    elif filter_design == 'chebyshev2':
        b, a = signal.cheby2(order, ripple, Wn, btype=btype, analog=False)
    elif filter_design == 'elliptic':
        b, a = signal.ellip(order, ripple, ripple * 10, Wn, btype=btype, analog=False)
    else:
        raise ValueError(f"Unknown filter design: {filter_design}")

    return b, a


def plot_frequency_response(b: np.ndarray, a: np.ndarray, sample_rate: int, ax=None):
    """Plot frequency response (magnitude and phase)"""
    w, h = signal.freqz(b, a, worN=8000, fs=sample_rate)

    if ax is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    else:
        ax1, ax2 = ax

    # Magnitude response
    ax1.plot(w, 20 * np.log10(abs(h)), 'b', linewidth=2)
    ax1.set_title('Frequency Response')
    ax1.set_ylabel('Magnitude [dB]')
    ax1.set_xlabel('Frequency [Hz]')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(-3, color='r', linestyle='--', alpha=0.5, label='-3 dB')
    ax1.legend()

    # Phase response
    angles = np.unwrap(np.angle(h))
    ax2.plot(w, np.degrees(angles), 'g', linewidth=2)
    ax2.set_ylabel('Phase [degrees]')
    ax2.set_xlabel('Frequency [Hz]')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Phase Response')

    plt.tight_layout()
    return ax1, ax2


def plot_impulse_response(b: np.ndarray, a: np.ndarray, sample_rate: int, ax=None):
    """Plot impulse response"""
    impulse = np.zeros(512)
    impulse[0] = 1.0
    response = signal.lfilter(b, a, impulse)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    time = np.arange(len(response)) / sample_rate * 1000  # Convert to ms
    ax.plot(time, response, 'b', linewidth=2)
    ax.set_title('Impulse Response')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return ax


def print_filter_info(
    filter_type: str,
    cutoff: Tuple[float, ...],
    sample_rate: int,
    order: int,
    filter_design: str,
    b: np.ndarray,
    a: np.ndarray,
):
    """Print filter specifications and coefficients"""
    print("\n" + "="*60)
    print("FILTER SPECIFICATIONS")
    print("="*60)
    print(f"Type:          {filter_type}")
    print(f"Design:        {filter_design}")
    print(f"Order:         {order}")
    print(f"Sample Rate:   {sample_rate} Hz")

    if filter_type in ['bandpass', 'notch']:
        print(f"Cutoff:        {cutoff[0]} - {cutoff[1]} Hz")
        print(f"Bandwidth:     {cutoff[1] - cutoff[0]} Hz")
    else:
        print(f"Cutoff:        {cutoff[0]} Hz")

    print("\n" + "="*60)
    print("FILTER COEFFICIENTS")
    print("="*60)
    print(f"Numerator (b):   {len(b)} coefficients")
    print(f"Denominator (a): {len(a)} coefficients")

    print("\nNumerator coefficients (b):")
    print(b)
    print("\nDenominator coefficients (a):")
    print(a)

    # Compute and display -3 dB point
    w, h = signal.freqz(b, a, worN=8000, fs=sample_rate)
    mag_db = 20 * np.log10(abs(h))

    # Find -3 dB points
    idx_3db = np.where(mag_db >= -3.1)[0]
    if len(idx_3db) > 0:
        if filter_type == 'lowpass':
            freq_3db = w[idx_3db[-1]]
            print(f"\nActual -3 dB point: {freq_3db:.1f} Hz")
        elif filter_type == 'highpass':
            freq_3db = w[idx_3db[0]]
            print(f"\nActual -3 dB point: {freq_3db:.1f} Hz")
        elif filter_type in ['bandpass', 'notch']:
            if len(idx_3db) >= 2:
                freq_low = w[idx_3db[0]]
                freq_high = w[idx_3db[-1]]
                print(f"\nActual -3 dB points: {freq_low:.1f} - {freq_high:.1f} Hz")

    print("\n" + "="*60)


def export_python_code(
    filter_type: str,
    cutoff: Tuple[float, ...],
    sample_rate: int,
    order: int,
    filter_design: str,
    ripple: float,
    b: np.ndarray,
    a: np.ndarray,
):
    """Generate Python code for the filter"""
    print("\n" + "="*60)
    print("PYTHON CODE (ready for wavecapsdr/dsp/filters.py)")
    print("="*60)

    # Generate function name
    if filter_type in ['bandpass', 'notch']:
        func_name = f"{filter_type}_{int(cutoff[0])}_{int(cutoff[1])}_hz"
    else:
        func_name = f"{filter_type}_{int(cutoff[0])}_hz"

    print(f"""
import numpy as np
from scipy.signal import {filter_design[0:5]}, lfilter

def {func_name}(signal: np.ndarray, sample_rate: int = {sample_rate}) -> np.ndarray:
    \"\"\"
    Apply {filter_design} {filter_type} filter.

    Parameters:
        - Order: {order}
        - Cutoff: {cutoff if len(cutoff) > 1 else cutoff[0]} Hz
        - Sample rate: {sample_rate} Hz
    \"\"\"
    nyquist = sample_rate / 2
    """)

    if filter_type in ['bandpass', 'notch']:
        print(f"    Wn = [{cutoff[0]} / nyquist, {cutoff[1]} / nyquist]")
        btype = 'bandstop' if filter_type == 'notch' else 'bandpass'
    else:
        print(f"    Wn = {cutoff[0]} / nyquist")
        btype = filter_type

    if filter_design == 'butterworth':
        print(f"    b, a = butter({order}, Wn, btype='{btype}', analog=False)")
    elif filter_design in ['chebyshev1', 'chebyshev2']:
        design_func = 'cheby1' if filter_design == 'chebyshev1' else 'cheby2'
        print(f"    b, a = {design_func}({order}, {ripple}, Wn, btype='{btype}', analog=False)")
    elif filter_design == 'elliptic':
        print(f"    b, a = ellip({order}, {ripple}, {ripple * 10}, Wn, btype='{btype}', analog=False)")

    print(f"""
    return lfilter(b, a, signal)


# Example usage:
# filtered_audio = {func_name}(audio_signal, sample_rate=48000)
""")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Design and visualize DSP filters for WaveCap-SDR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 15 kHz lowpass for FM de-emphasis
  %(prog)s --type lowpass --cutoff 15000 --order 5

  # 300-3000 Hz bandpass for SSB
  %(prog)s --type bandpass --cutoff 300,3000 --order 4

  # 60 Hz notch for AC hum removal
  %(prog)s --type notch --cutoff 55,65 --order 4
        """
    )

    parser.add_argument(
        '--type',
        choices=['lowpass', 'highpass', 'bandpass', 'notch'],
        required=True,
        help='Filter type'
    )
    parser.add_argument(
        '--cutoff',
        required=True,
        help='Cutoff frequency in Hz (single value for low/high, comma-separated for band/notch)'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=48000,
        help='Sample rate in Hz (default: 48000)'
    )
    parser.add_argument(
        '--order',
        type=int,
        default=5,
        help='Filter order (default: 5)'
    )
    parser.add_argument(
        '--filter-design',
        choices=['butterworth', 'chebyshev1', 'chebyshev2', 'elliptic'],
        default='butterworth',
        help='Filter design method (default: butterworth)'
    )
    parser.add_argument(
        '--ripple',
        type=float,
        default=0.5,
        help='Passband ripple in dB for Chebyshev/Elliptic (default: 0.5)'
    )
    parser.add_argument(
        '--output',
        help='Save plots to file instead of displaying'
    )
    parser.add_argument(
        '--export-code',
        action='store_true',
        help='Export Python code for the filter'
    )

    args = parser.parse_args()

    # Parse cutoff frequency(ies)
    try:
        cutoff = tuple(float(x) for x in args.cutoff.split(','))
    except ValueError:
        print(f"Error: Invalid cutoff frequency format: {args.cutoff}", file=sys.stderr)
        print("Use single value (e.g., 15000) or comma-separated (e.g., 300,3000)", file=sys.stderr)
        return 1

    # Design filter
    try:
        b, a = design_filter(
            args.type,
            cutoff,
            args.sample_rate,
            args.order,
            args.filter_design,
            args.ripple,
        )
    except Exception as e:
        print(f"Error designing filter: {e}", file=sys.stderr)
        return 1

    # Print filter information
    print_filter_info(args.type, cutoff, args.sample_rate, args.order, args.filter_design, b, a)

    # Export code if requested
    if args.export_code:
        export_python_code(args.type, cutoff, args.sample_rate, args.order, args.filter_design, args.ripple, b, a)

    # Plot responses
    fig = plt.figure(figsize=(12, 10))

    # Frequency response
    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2)
    plot_frequency_response(b, a, args.sample_rate, (ax1, ax2))

    # Impulse response
    ax3 = plt.subplot(3, 1, 3)
    plot_impulse_response(b, a, args.sample_rate, ax3)

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"\nPlots saved to: {args.output}")
    else:
        plt.show()

    return 0


if __name__ == '__main__':
    sys.exit(main())
