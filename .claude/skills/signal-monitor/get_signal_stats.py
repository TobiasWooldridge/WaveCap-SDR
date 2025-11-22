#!/usr/bin/env python3
"""
Signal Monitor - Get real-time signal quality metrics from WaveCap-SDR channels.

Usage:
    python get_signal_stats.py --channel ch1
    python get_signal_stats.py --channel ch1 --monitor --interval 1.0
    python get_signal_stats.py --capture c1 --spectrum
"""

import argparse
import json
import sys
import time
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError


def fetch_json(url: str) -> dict:
    """Fetch JSON from URL."""
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode())
    except HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        raise RuntimeError(f"HTTP {e.code}: {error_body}")
    except URLError as e:
        raise RuntimeError(f"Connection error: {e.reason}")


def get_extended_metrics(host: str, port: int, channel_id: str) -> dict:
    """Get extended metrics for a channel."""
    url = f"http://{host}:{port}/api/v1/channels/{channel_id}/metrics/extended"
    return fetch_json(url)


def get_spectrum_snapshot(host: str, port: int, capture_id: str) -> dict:
    """Get spectrum snapshot for a capture."""
    url = f"http://{host}:{port}/api/v1/captures/{capture_id}/spectrum/snapshot"
    return fetch_json(url)


def get_all_channels(host: str, port: int) -> list:
    """Get all channels from all captures."""
    captures = fetch_json(f"http://{host}:{port}/api/v1/captures")
    channels = []
    for cap in captures:
        cap_channels = fetch_json(f"http://{host}:{port}/api/v1/captures/{cap['id']}/channels")
        for ch in cap_channels:
            ch["capture"] = cap
        channels.extend(cap_channels)
    return channels


def format_s_meter(s_units: Optional[str]) -> str:
    """Format S-meter reading with visual bar."""
    if not s_units:
        return "---"

    # Parse S-units
    if s_units.startswith("S9+"):
        level = 9 + int(s_units[3:]) // 6
    elif s_units.startswith("S"):
        level = int(s_units[1:])
    else:
        level = 0

    # Create visual bar (max 15 chars for S9+60)
    bar_len = min(level, 15)
    bar = "█" * bar_len + "░" * (9 - min(bar_len, 9))

    return f"{s_units:>6} [{bar}]"


def format_db(value: Optional[float], width: int = 7) -> str:
    """Format dB value with proper alignment."""
    if value is None:
        return "-".center(width)
    return f"{value:>{width}.1f}"


def print_metrics(metrics: dict, show_header: bool = True) -> None:
    """Print metrics in a formatted table."""
    if show_header:
        print("\n╔═══════════════════════════════════════════════════════════════════╗")
        print("║                    WaveCap-SDR Signal Monitor                     ║")
        print("╠═══════════════════════════════════════════════════════════════════╣")
        print("║ Channel │  RSSI (dB) │  SNR (dB) │ Power (dB) │     S-Meter      ║")
        print("╠═════════╪════════════╪═══════════╪════════════╪══════════════════╣")

    ch_id = metrics.get("channelId", "?")[:7]
    rssi = format_db(metrics.get("rssiDb"))
    snr = format_db(metrics.get("snrDb"))
    power = format_db(metrics.get("signalPowerDb"))
    s_meter = format_s_meter(metrics.get("sUnits"))
    squelch = "🔊" if metrics.get("squelchOpen") else "🔇"

    print(f"║ {ch_id:<7} │ {rssi:>10} │ {snr:>9} │ {power:>10} │ {s_meter} {squelch} ║")


def print_footer(metrics: dict) -> None:
    """Print footer with additional info."""
    print("╠═══════════════════════════════════════════════════════════════════╣")
    state = metrics.get("captureState", "unknown")
    subs = metrics.get("streamSubscribers", 0)
    drops = metrics.get("streamDropsPerSec", 0)
    print(f"║ Capture: {state:<10} │ Subscribers: {subs:<3} │ Drops/sec: {drops:<6.2f}   ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")


def monitor_channel(host: str, port: int, channel_id: str, interval: float, duration: float) -> None:
    """Monitor channel metrics over time."""
    start_time = time.time()
    iteration = 0

    try:
        while True:
            elapsed = time.time() - start_time
            if duration > 0 and elapsed >= duration:
                break

            try:
                metrics = get_extended_metrics(host, port, channel_id)
                print_metrics(metrics, show_header=(iteration == 0))
                if iteration == 0:
                    print_footer(metrics)
                iteration += 1
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


def print_spectrum_summary(spectrum: dict) -> None:
    """Print summary of spectrum snapshot."""
    power = spectrum.get("power", [])
    freqs = spectrum.get("freqs", [])
    center = spectrum.get("centerHz", 0)
    sample_rate = spectrum.get("sampleRate", 0)

    if not power:
        print("No spectrum data available.")
        return

    import statistics

    min_power = min(power)
    max_power = max(power)
    avg_power = statistics.mean(power)

    # Find peak frequency
    peak_idx = power.index(max_power)
    peak_freq = freqs[peak_idx] if peak_idx < len(freqs) else center

    print("\n╔═══════════════════════════════════════════════════════════════════╗")
    print("║                   Spectrum Snapshot Summary                       ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print(f"║ Center Frequency: {center/1e6:>10.3f} MHz                              ║")
    print(f"║ Sample Rate:      {sample_rate/1e3:>10.1f} kHz                              ║")
    print(f"║ FFT Bins:         {len(power):>10}                                   ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print(f"║ Min Power:        {min_power:>10.1f} dB                               ║")
    print(f"║ Max Power:        {max_power:>10.1f} dB                               ║")
    print(f"║ Avg Power:        {avg_power:>10.1f} dB                               ║")
    print(f"║ Peak Frequency:   {peak_freq/1e6:>10.3f} MHz                              ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")


def main():
    parser = argparse.ArgumentParser(
        description="Get signal quality metrics from WaveCap-SDR channels"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8087, help="Server port (default: 8087)")
    parser.add_argument("--channel", "-c", help="Channel ID to monitor")
    parser.add_argument("--capture", help="Capture ID (for spectrum snapshot)")
    parser.add_argument("--spectrum", action="store_true", help="Get spectrum snapshot")
    parser.add_argument("--monitor", "-m", action="store_true", help="Monitor continuously")
    parser.add_argument("--interval", "-i", type=float, default=1.0, help="Monitor interval (seconds)")
    parser.add_argument("--duration", "-d", type=float, default=0, help="Monitor duration (0=forever)")
    parser.add_argument("--list", "-l", action="store_true", help="List all channels")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")

    args = parser.parse_args()

    try:
        # List all channels
        if args.list:
            channels = get_all_channels(args.host, args.port)
            if args.json:
                print(json.dumps(channels, indent=2))
            else:
                print("\nAvailable Channels:")
                print("-" * 60)
                for ch in channels:
                    cap = ch.get("capture", {})
                    print(f"  {ch['id']:<8} │ {ch['mode']:<6} │ {ch['state']:<8} │ capture: {cap.get('id', '?')}")
            return 0

        # Get spectrum snapshot
        if args.spectrum:
            capture_id = args.capture or "c1"
            spectrum = get_spectrum_snapshot(args.host, args.port, capture_id)
            if args.json:
                print(json.dumps(spectrum, indent=2))
            else:
                print_spectrum_summary(spectrum)
            return 0

        # Get channel metrics
        if not args.channel:
            # Auto-detect first channel
            channels = get_all_channels(args.host, args.port)
            if not channels:
                print("Error: No channels found. Create a capture and channel first.", file=sys.stderr)
                return 1
            args.channel = channels[0]["id"]
            print(f"Using channel: {args.channel}")

        if args.monitor:
            monitor_channel(args.host, args.port, args.channel, args.interval, args.duration)
        else:
            metrics = get_extended_metrics(args.host, args.port, args.channel)
            if args.json:
                print(json.dumps(metrics, indent=2))
            else:
                print_metrics(metrics)
                print_footer(metrics)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
