#!/usr/bin/env python3
"""Identify P25 control channels from spectrum data.

Control channels transmit continuously, so they appear as constant power
at a specific frequency. Voice channels only transmit during calls.

This script monitors spectrum data and identifies frequencies with:
1. Consistent high power (always active)
2. Narrow bandwidth typical of P25 (12.5 kHz)
"""
import argparse
import time
import json
import requests
import numpy as np
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Identify control channels from spectrum")
    parser.add_argument("--capture", default="c1", help="Capture ID")
    parser.add_argument("--duration", type=int, default=60, help="Monitoring duration (seconds)")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8087, help="Server port")
    parser.add_argument("--threshold-db", type=float, default=-60, help="Min power to consider active")
    parser.add_argument("--channel-width-hz", type=float, default=12500, help="P25 channel width")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}/api/v1"

    # Get capture info
    resp = requests.get(f"{base_url}/captures/{args.capture}")
    if resp.status_code != 200:
        print(f"Error: Capture {args.capture} not found")
        return

    capture = resp.json()
    center_hz = capture["centerHz"]
    sample_rate = capture["sampleRate"]
    fft_size = capture.get("fftSize", 2048)

    freq_span = sample_rate
    freq_start = center_hz - freq_span / 2
    freq_end = center_hz + freq_span / 2
    bin_width = freq_span / fft_size

    print(f"Monitoring {args.capture} for {args.duration} seconds")
    print(f"  Center: {center_hz/1e6:.3f} MHz")
    print(f"  Span: {freq_span/1e6:.3f} MHz ({freq_start/1e6:.3f} - {freq_end/1e6:.3f} MHz)")
    print(f"  FFT bins: {fft_size}, bin width: {bin_width:.1f} Hz")
    print(f"  Power threshold: {args.threshold_db} dB")
    print()

    # Track power per bin over time
    # Key: bin index, Value: list of power values
    power_history = defaultdict(list)
    sample_count = 0

    start_time = time.time()
    last_print = start_time

    print("Collecting spectrum samples...")
    while time.time() - start_time < args.duration:
        try:
            resp = requests.get(f"{base_url}/captures/{args.capture}/spectrum/snapshot", timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                power = data.get("power", [])
                if power:
                    for i, p in enumerate(power):
                        power_history[i].append(p)
                    sample_count += 1
        except Exception as e:
            pass

        # Print progress every 10 seconds
        now = time.time()
        if now - last_print >= 10:
            elapsed = now - start_time
            print(f"  {elapsed:.0f}s: collected {sample_count} samples")
            last_print = now

        time.sleep(0.2)  # 5 samples per second

    print(f"\nCollected {sample_count} samples over {args.duration} seconds")
    print()

    if sample_count < 10:
        print("Not enough samples collected")
        return

    # Analyze each bin
    # Control channels: high average power, low variance (always on)
    # Voice channels: variable power (on/off), higher variance

    bin_stats = []
    for bin_idx in range(len(power_history.get(0, []) or [1])):
        if bin_idx not in power_history or len(power_history[bin_idx]) < 10:
            continue

        powers = np.array(power_history[bin_idx])
        avg_power = np.mean(powers)
        std_power = np.std(powers)
        min_power = np.min(powers)
        max_power = np.max(powers)

        # What fraction of time was power above threshold?
        active_ratio = np.sum(powers > args.threshold_db) / len(powers)

        freq_hz = freq_start + bin_idx * bin_width

        bin_stats.append({
            "bin": bin_idx,
            "freq_hz": freq_hz,
            "freq_mhz": freq_hz / 1e6,
            "avg_power": avg_power,
            "std_power": std_power,
            "min_power": min_power,
            "max_power": max_power,
            "active_ratio": active_ratio,
        })

    # Sort by average power descending
    bin_stats.sort(key=lambda x: x["avg_power"], reverse=True)

    # Find peaks - bins that are local maxima
    # Group adjacent bins into "channels"
    channels = []
    bins_per_channel = int(args.channel_width_hz / bin_width)

    visited = set()
    for stat in bin_stats:
        if stat["avg_power"] < args.threshold_db:
            continue
        if stat["bin"] in visited:
            continue

        # Found a peak - mark nearby bins as visited
        center_bin = stat["bin"]
        for offset in range(-bins_per_channel, bins_per_channel + 1):
            visited.add(center_bin + offset)

        channels.append(stat)

    # Classify channels
    print("=" * 80)
    print("LIKELY CONTROL CHANNELS (constant power, low variance)")
    print("=" * 80)
    print(f"{'Frequency':>14} {'Avg dB':>8} {'Std':>6} {'Active%':>8} {'Classification':>15}")
    print("-" * 80)

    control_candidates = []
    voice_candidates = []

    for ch in channels[:30]:  # Top 30 by power
        # Control channel criteria:
        # - High average power (strong signal)
        # - Low variance (constant transmission)
        # - High active ratio (always above threshold)

        is_control = (
            ch["std_power"] < 3.0 and  # Low variance
            ch["active_ratio"] > 0.95  # Almost always active
        )

        is_voice = (
            ch["std_power"] >= 3.0 or  # Variable power
            ch["active_ratio"] < 0.8   # Not always active
        )

        if is_control:
            classification = "CONTROL"
            control_candidates.append(ch)
        elif is_voice:
            classification = "voice"
            voice_candidates.append(ch)
        else:
            classification = "unknown"

        print(f"{ch['freq_mhz']:>12.4f} MHz {ch['avg_power']:>8.1f} {ch['std_power']:>6.2f} {ch['active_ratio']*100:>7.1f}% {classification:>15}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if control_candidates:
        print(f"\nIdentified {len(control_candidates)} likely CONTROL channel(s):")
        for ch in sorted(control_candidates, key=lambda x: x["freq_mhz"]):
            print(f"  {ch['freq_mhz']:.4f} MHz (avg {ch['avg_power']:.1f} dB, std {ch['std_power']:.2f})")
    else:
        print("\nNo clear control channels identified")

    if voice_candidates:
        print(f"\nIdentified {len(voice_candidates)} likely VOICE channel(s) (intermittent activity):")
        for ch in sorted(voice_candidates, key=lambda x: x["freq_mhz"])[:10]:
            print(f"  {ch['freq_mhz']:.4f} MHz (avg {ch['avg_power']:.1f} dB, active {ch['active_ratio']*100:.0f}%)")

    # Check for 413.4577 specifically
    print()
    target_freq = 413.4577e6
    for stat in bin_stats:
        if abs(stat["freq_hz"] - target_freq) < bin_width * 2:
            print(f">>> 413.4577 MHz: avg {stat['avg_power']:.1f} dB, std {stat['std_power']:.2f}, "
                  f"active {stat['active_ratio']*100:.0f}%")
            if stat["std_power"] < 3.0 and stat["active_ratio"] > 0.95:
                print("    Classification: LIKELY CONTROL CHANNEL")
            break


if __name__ == "__main__":
    main()
