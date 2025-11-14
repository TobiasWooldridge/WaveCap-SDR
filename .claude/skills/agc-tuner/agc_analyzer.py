#!/usr/bin/env python3
"""
AGC Analyzer for WaveCap-SDR

Analyze audio dynamics and optimize AGC parameters.
"""

import argparse
import sys
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import requests


class AGCSimulator:
    """Simulate AGC behavior for analysis"""

    def __init__(self, attack_time: float, release_time: float, target_level: float, sample_rate: int = 48000):
        self.attack_alpha = 1.0 - np.exp(-1.0 / (attack_time * sample_rate))
        self.release_alpha = 1.0 - np.exp(-1.0 / (release_time * sample_rate))
        self.target_level = target_level
        self.sample_rate = sample_rate
        self.current_gain = 1.0
        self.envelope = 0.0

    def process(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process signal through AGC and return output + gain history.

        Returns:
            (output_signal, gain_history)
        """
        output = np.zeros_like(signal)
        gain_history = np.zeros(len(signal))

        for i, sample in enumerate(signal):
            # Update envelope (RMS-like)
            sample_energy = sample * sample
            self.envelope += 0.01 * (sample_energy - self.envelope)  # Smooth envelope

            # Compute desired gain
            envelope_rms = np.sqrt(max(self.envelope, 1e-10))
            desired_gain = self.target_level / envelope_rms

            # Clamp desired gain
            desired_gain = np.clip(desired_gain, 0.01, 100.0)

            # Update current gain with attack/release
            if desired_gain < self.current_gain:
                # Attack: signal got louder, reduce gain quickly
                self.current_gain += self.attack_alpha * (desired_gain - self.current_gain)
            else:
                # Release: signal got quieter, increase gain slowly
                self.current_gain += self.release_alpha * (desired_gain - self.current_gain)

            # Apply gain
            output[i] = sample * self.current_gain
            gain_history[i] = self.current_gain

        return output, gain_history


def fetch_audio_stream(host: str, port: int, channel: str, duration: float) -> Optional[np.ndarray]:
    """Fetch audio from WaveCap-SDR channel"""
    url = f"http://{host}:{port}/api/v1/stream/channels/{channel}.pcm?format=pcm16"

    try:
        response = requests.get(url, stream=True, timeout=5)
        response.raise_for_status()

        # Calculate bytes to read
        sample_rate = 48000  # Default audio rate
        bytes_per_sample = 2  # 16-bit PCM
        total_bytes = int(duration * sample_rate * bytes_per_sample)

        # Read audio data
        audio_bytes = b''
        for chunk in response.iter_content(chunk_size=4096):
            audio_bytes += chunk
            if len(audio_bytes) >= total_bytes:
                audio_bytes = audio_bytes[:total_bytes]
                break

        # Convert to numpy array
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        # Normalize to -1.0 to 1.0
        audio = audio.astype(np.float32) / 32768.0

        return audio

    except requests.exceptions.RequestException as e:
        print(f"Error fetching audio stream: {e}", file=sys.stderr)
        return None


def analyze_dynamics(signal: np.ndarray, sample_rate: int = 48000) -> dict:
    """Analyze signal dynamics"""
    # Compute RMS level
    rms = np.sqrt(np.mean(signal ** 2))
    rms_db = 20 * np.log10(rms + 1e-10)

    # Compute peak level
    peak = np.max(np.abs(signal))
    peak_db = 20 * np.log10(peak + 1e-10)

    # Compute crest factor
    crest_factor = peak / (rms + 1e-10)
    crest_factor_db = peak_db - rms_db

    # Compute RMS over time (1 second windows)
    window_size = sample_rate
    num_windows = len(signal) // window_size
    rms_over_time = []

    for i in range(num_windows):
        window = signal[i * window_size:(i + 1) * window_size]
        rms_over_time.append(np.sqrt(np.mean(window ** 2)))

    # Dynamic range
    if len(rms_over_time) > 0:
        rms_over_time = np.array(rms_over_time)
        dynamic_range_db = 20 * np.log10(np.max(rms_over_time) / (np.min(rms_over_time) + 1e-10))
    else:
        dynamic_range_db = 0.0

    return {
        'rms': rms,
        'rms_db': rms_db,
        'peak': peak,
        'peak_db': peak_db,
        'crest_factor': crest_factor,
        'crest_factor_db': crest_factor_db,
        'dynamic_range_db': dynamic_range_db,
        'rms_over_time': rms_over_time,
    }


def suggest_agc_parameters(dynamics: dict) -> dict:
    """Suggest optimal AGC parameters based on signal dynamics"""
    suggestions = {}

    # Based on crest factor
    if dynamics['crest_factor'] > 6:
        # Very dynamic (speech, classical music)
        suggestions['attack_time'] = 0.005  # 5ms
        suggestions['release_time'] = 0.300  # 300ms
        suggestions['target_level'] = 0.15  # Lower target for more headroom
        suggestions['reason'] = "Highly dynamic signal detected (speech/classical)"

    elif dynamics['crest_factor'] > 4:
        # Moderately dynamic (pop music, FM broadcast)
        suggestions['attack_time'] = 0.010  # 10ms
        suggestions['release_time'] = 0.500  # 500ms
        suggestions['target_level'] = 0.20
        suggestions['reason'] = "Moderately dynamic signal (broadcast/music)"

    elif dynamics['crest_factor'] > 2:
        # Compressed (modern music, digital modes)
        suggestions['attack_time'] = 0.002  # 2ms
        suggestions['release_time'] = 0.150  # 150ms
        suggestions['target_level'] = 0.25
        suggestions['reason'] = "Compressed signal detected (digital/modern music)"

    else:
        # Highly compressed or steady tone
        suggestions['attack_time'] = 0.020  # 20ms
        suggestions['release_time'] = 0.800  # 800ms
        suggestions['target_level'] = 0.15
        suggestions['reason'] = "Steady or highly compressed signal"

    # Adjust based on dynamic range
    if dynamics['dynamic_range_db'] > 20:
        # High dynamic range, need more aggressive AGC
        suggestions['release_time'] = max(0.3, suggestions['release_time'])
        suggestions['reason'] += " | High dynamic range detected"

    return suggestions


def plot_agc_analysis(
    original: np.ndarray,
    output: np.ndarray,
    gain_history: np.ndarray,
    sample_rate: int,
    dynamics_orig: dict,
    dynamics_agc: dict,
    output_file: Optional[str] = None,
):
    """Plot AGC analysis results"""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    time = np.arange(len(original)) / sample_rate

    # Plot 1: Original vs AGC Output
    ax = axes[0]
    # Plot only first 0.5 seconds for visibility
    plot_samples = min(len(original), int(0.5 * sample_rate))
    ax.plot(time[:plot_samples], original[:plot_samples], 'b', alpha=0.5, label='Original', linewidth=0.5)
    ax.plot(time[:plot_samples], output[:plot_samples], 'r', alpha=0.7, label='AGC Output', linewidth=0.5)
    ax.set_ylabel('Amplitude')
    ax.set_title('Waveform Comparison (First 0.5s)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Gain over time
    ax = axes[1]
    ax.plot(time, 20 * np.log10(gain_history + 1e-10), 'g', linewidth=1)
    ax.set_ylabel('Gain [dB]')
    ax.set_title('AGC Gain Over Time')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='r', linestyle='--', alpha=0.5)

    # Plot 3: RMS level over time
    ax = axes[2]
    window_size = sample_rate  # 1 second windows
    num_windows = len(original) // window_size

    time_windows = np.arange(num_windows)
    rms_orig = []
    rms_agc = []

    for i in range(num_windows):
        window_orig = original[i * window_size:(i + 1) * window_size]
        window_agc = output[i * window_size:(i + 1) * window_size]
        rms_orig.append(np.sqrt(np.mean(window_orig ** 2)))
        rms_agc.append(np.sqrt(np.mean(window_agc ** 2)))

    rms_orig_db = 20 * np.log10(np.array(rms_orig) + 1e-10)
    rms_agc_db = 20 * np.log10(np.array(rms_agc) + 1e-10)

    ax.plot(time_windows, rms_orig_db, 'b', marker='o', label='Original', linewidth=2)
    ax.plot(time_windows, rms_agc_db, 'r', marker='s', label='AGC Output', linewidth=2)
    ax.set_xlabel('Time [seconds]')
    ax.set_ylabel('RMS Level [dB]')
    ax.set_title('RMS Level Over Time (1-second windows)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Statistics summary (text)
    ax = axes[3]
    ax.axis('off')

    summary_text = f"""
    ORIGINAL SIGNAL STATISTICS:
      RMS Level:        {dynamics_orig['rms_db']:.1f} dB
      Peak Level:       {dynamics_orig['peak_db']:.1f} dB
      Crest Factor:     {dynamics_orig['crest_factor']:.2f} ({dynamics_orig['crest_factor_db']:.1f} dB)
      Dynamic Range:    {dynamics_orig['dynamic_range_db']:.1f} dB

    AGC OUTPUT STATISTICS:
      RMS Level:        {dynamics_agc['rms_db']:.1f} dB
      Peak Level:       {dynamics_agc['peak_db']:.1f} dB
      Crest Factor:     {dynamics_agc['crest_factor']:.2f} ({dynamics_agc['crest_factor_db']:.1f} dB)
      Dynamic Range:    {dynamics_agc['dynamic_range_db']:.1f} dB

    GAIN RANGE:
      Min Gain:         {20 * np.log10(np.min(gain_history)):.1f} dB
      Max Gain:         {20 * np.log10(np.max(gain_history)):.1f} dB
      Gain Variation:   {20 * np.log10(np.max(gain_history) / (np.min(gain_history) + 1e-10)):.1f} dB
    """

    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax.transAxes)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nPlots saved to: {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze audio dynamics and optimize AGC parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--channel', default='ch1', help='Channel ID to analyze (default: ch1)')
    parser.add_argument('--duration', type=float, default=10.0, help='Seconds of audio to capture (default: 10)')
    parser.add_argument('--host', default='127.0.0.1', help='Server host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8087, help='Server port (default: 8087)')
    parser.add_argument('--attack', type=float, help='Attack time in seconds (default: auto-suggest)')
    parser.add_argument('--release', type=float, help='Release time in seconds (default: auto-suggest)')
    parser.add_argument('--target', type=float, help='Target level 0.0-1.0 (default: auto-suggest)')
    parser.add_argument('--plot', action='store_true', help='Generate plots of AGC behavior')
    parser.add_argument('--output', help='Save plots to file')

    args = parser.parse_args()

    # Fetch audio
    print(f"Fetching {args.duration}s of audio from {args.channel}...")
    audio = fetch_audio_stream(args.host, args.port, args.channel, args.duration)

    if audio is None:
        return 1

    print(f"Captured {len(audio)} samples ({len(audio) / 48000:.2f}s)")

    # Analyze original signal
    print("\n" + "="*60)
    print("ORIGINAL SIGNAL ANALYSIS")
    print("="*60)
    dynamics_orig = analyze_dynamics(audio)

    print(f"RMS Level:        {dynamics_orig['rms_db']:>8.1f} dB  ({dynamics_orig['rms']:.4f})")
    print(f"Peak Level:       {dynamics_orig['peak_db']:>8.1f} dB  ({dynamics_orig['peak']:.4f})")
    print(f"Crest Factor:     {dynamics_orig['crest_factor']:>8.2f}     ({dynamics_orig['crest_factor_db']:.1f} dB)")
    print(f"Dynamic Range:    {dynamics_orig['dynamic_range_db']:>8.1f} dB")

    # Get suggested parameters or use provided
    suggestions = suggest_agc_parameters(dynamics_orig)

    attack_time = args.attack if args.attack else suggestions['attack_time']
    release_time = args.release if args.release else suggestions['release_time']
    target_level = args.target if args.target else suggestions['target_level']

    print("\n" + "="*60)
    print("AGC PARAMETERS")
    print("="*60)
    print(f"Attack Time:      {attack_time*1000:>8.1f} ms  ({attack_time:.4f} s)")
    print(f"Release Time:     {release_time*1000:>8.1f} ms  ({release_time:.4f} s)")
    print(f"Target Level:     {target_level:>8.2f}     ({20*np.log10(target_level):.1f} dB)")

    if not (args.attack and args.release and args.target):
        print(f"\nSuggestion: {suggestions['reason']}")

    # Simulate AGC
    print("\nSimulating AGC...")
    agc = AGCSimulator(attack_time, release_time, target_level)
    output, gain_history = agc.process(audio)

    # Analyze AGC output
    print("\n" + "="*60)
    print("AGC OUTPUT ANALYSIS")
    print("="*60)
    dynamics_agc = analyze_dynamics(output)

    print(f"RMS Level:        {dynamics_agc['rms_db']:>8.1f} dB  ({dynamics_agc['rms']:.4f})")
    print(f"Peak Level:       {dynamics_agc['peak_db']:>8.1f} dB  ({dynamics_agc['peak']:.4f})")
    print(f"Crest Factor:     {dynamics_agc['crest_factor']:>8.2f}     ({dynamics_agc['crest_factor_db']:.1f} dB)")
    print(f"Dynamic Range:    {dynamics_agc['dynamic_range_db']:>8.1f} dB")

    # Gain statistics
    print("\n" + "="*60)
    print("GAIN STATISTICS")
    print("="*60)
    min_gain_db = 20 * np.log10(np.min(gain_history))
    max_gain_db = 20 * np.log10(np.max(gain_history))
    mean_gain_db = 20 * np.log10(np.mean(gain_history))

    print(f"Min Gain:         {min_gain_db:>8.1f} dB")
    print(f"Max Gain:         {max_gain_db:>8.1f} dB")
    print(f"Mean Gain:        {mean_gain_db:>8.1f} dB")
    print(f"Gain Variation:   {max_gain_db - min_gain_db:>8.1f} dB")

    # Check for pumping
    gain_db = 20 * np.log10(gain_history + 1e-10)
    gain_changes = np.abs(np.diff(gain_db))
    rapid_changes = np.sum(gain_changes > 1.0)  # Changes > 1 dB per sample

    if rapid_changes > len(gain_history) * 0.01:
        print(f"\n⚠ WARNING: Detected {rapid_changes} rapid gain changes")
        print("  This may indicate pumping artifacts.")
        print("  Consider increasing release_time.")

    # Generate plots if requested
    if args.plot or args.output:
        plot_agc_analysis(
            audio, output, gain_history, 48000,
            dynamics_orig, dynamics_agc, args.output
        )

    print("\n" + "="*60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
