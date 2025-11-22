#!/usr/bin/env python3
"""
Radio Tuner - Adjust SDR radio settings in WaveCap-SDR.

Usage:
    python adjust_settings.py --capture c1 --show
    python adjust_settings.py --capture c1 --frequency 90.3 --gain 35
    python adjust_settings.py --channel ch1 --squelch -50 --mode nbfm
"""

import argparse
import json
import sys
from typing import Any, Optional
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


def patch_json(url: str, data: dict) -> dict:
    """Send PATCH request with JSON body."""
    try:
        body = json.dumps(data).encode("utf-8")
        req = Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="PATCH",
        )
        with urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode())
    except HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        raise RuntimeError(f"HTTP {e.code}: {error_body}")
    except URLError as e:
        raise RuntimeError(f"Connection error: {e.reason}")


def get_capture(host: str, port: int, capture_id: str) -> dict:
    """Get capture details."""
    url = f"http://{host}:{port}/api/v1/captures/{capture_id}"
    return fetch_json(url)


def get_channel(host: str, port: int, channel_id: str) -> dict:
    """Get channel details."""
    url = f"http://{host}:{port}/api/v1/channels/{channel_id}"
    return fetch_json(url)


def update_capture(host: str, port: int, capture_id: str, settings: dict) -> dict:
    """Update capture settings."""
    url = f"http://{host}:{port}/api/v1/captures/{capture_id}"
    return patch_json(url, settings)


def update_channel(host: str, port: int, channel_id: str, settings: dict) -> dict:
    """Update channel settings."""
    url = f"http://{host}:{port}/api/v1/channels/{channel_id}"
    return patch_json(url, settings)


def get_all_captures(host: str, port: int) -> list:
    """Get all captures."""
    return fetch_json(f"http://{host}:{port}/api/v1/captures")


def get_all_channels_for_capture(host: str, port: int, capture_id: str) -> list:
    """Get all channels for a capture."""
    return fetch_json(f"http://{host}:{port}/api/v1/captures/{capture_id}/channels")


def format_hz(hz: float) -> str:
    """Format frequency in Hz to human readable."""
    if hz >= 1e9:
        return f"{hz/1e9:.6f} GHz"
    elif hz >= 1e6:
        return f"{hz/1e6:.6f} MHz"
    elif hz >= 1e3:
        return f"{hz/1e3:.3f} kHz"
    return f"{hz:.1f} Hz"


def print_capture_settings(capture: dict) -> None:
    """Print capture settings in formatted table."""
    print("\n╔═══════════════════════════════════════════════════════════════════╗")
    print(f"║ Capture: {capture.get('id', '?'):<57} ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print(f"║ State:         {capture.get('state', '?'):<52} ║")
    print(f"║ Device:        {capture.get('deviceId', '?')[:52]:<52} ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")

    center_hz = capture.get("centerHz", 0)
    sample_rate = capture.get("sampleRate", 0)
    gain = capture.get("gain")
    bandwidth = capture.get("bandwidth")
    ppm = capture.get("ppm")
    antenna = capture.get("antenna")

    print(f"║ Center Freq:   {format_hz(center_hz):<52} ║")
    print(f"║ Sample Rate:   {format_hz(sample_rate):<52} ║")
    print(f"║ Gain:          {gain if gain is not None else 'auto'!s:<52} ║")
    print(f"║ Bandwidth:     {format_hz(bandwidth) if bandwidth else 'auto':<52} ║")
    print(f"║ PPM:           {ppm if ppm is not None else '0'!s:<52} ║")
    print(f"║ Antenna:       {antenna or 'default':<52} ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")


def print_channel_settings(channel: dict) -> None:
    """Print channel settings in formatted table."""
    print("\n╔═══════════════════════════════════════════════════════════════════╗")
    print(f"║ Channel: {channel.get('id', '?'):<58} ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print(f"║ State:         {channel.get('state', '?'):<52} ║")
    print(f"║ Capture:       {channel.get('captureId', '?'):<52} ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")

    mode = channel.get("mode", "?")
    offset = channel.get("offsetHz", 0)
    squelch = channel.get("squelchDb")
    audio_rate = channel.get("audioRate", 48000)

    print(f"║ Mode:          {mode:<52} ║")
    print(f"║ Offset:        {format_hz(offset):<52} ║")
    print(f"║ Squelch:       {squelch if squelch is not None else 'disabled'!s:<52} ║")
    print(f"║ Audio Rate:    {audio_rate} Hz{'':<44} ║")

    # AGC settings
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print("║ AGC Settings                                                      ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    agc_enabled = channel.get("enableAgc", False)
    agc_target = channel.get("agcTargetDb", -20)
    agc_attack = channel.get("agcAttackMs", 10)
    agc_release = channel.get("agcReleaseMs", 500)

    print(f"║ AGC Enabled:   {agc_enabled!s:<52} ║")
    print(f"║ AGC Target:    {agc_target} dB{'':<47} ║")
    print(f"║ AGC Attack:    {agc_attack} ms{'':<47} ║")
    print(f"║ AGC Release:   {agc_release} ms{'':<46} ║")

    # Filter settings
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print("║ Filter Settings                                                   ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    deemph = channel.get("enableDeemphasis", False)
    deemph_tau = channel.get("deemphasisTauUs", 75)
    notch = channel.get("notchFrequencies", [])

    print(f"║ De-emphasis:   {deemph!s} ({deemph_tau} µs){'':<40} ║")
    print(f"║ Notch Filters: {notch if notch else 'none'!s:<52} ║")

    print("╚═══════════════════════════════════════════════════════════════════╝")


def main():
    parser = argparse.ArgumentParser(
        description="Adjust SDR radio settings in WaveCap-SDR"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8087, help="Server port")

    # Target selection
    parser.add_argument("--capture", "-c", help="Capture ID to modify")
    parser.add_argument("--channel", help="Channel ID to modify")
    parser.add_argument("--show", action="store_true", help="Show current settings")
    parser.add_argument("--list", "-l", action="store_true", help="List all captures/channels")

    # Capture settings
    parser.add_argument("--frequency", "-f", type=float, help="Center frequency (MHz)")
    parser.add_argument("--sample-rate", type=int, help="Sample rate (Hz)")
    parser.add_argument("--gain", "-g", type=float, help="RF gain (dB)")
    parser.add_argument("--bandwidth", "-b", type=float, help="Bandwidth (Hz)")
    parser.add_argument("--ppm", type=float, help="Frequency correction (PPM)")
    parser.add_argument("--antenna", help="Antenna port")

    # Channel settings
    parser.add_argument("--mode", "-m", choices=["wbfm", "nbfm", "am", "ssb", "raw", "p25", "dmr"],
                        help="Demodulation mode")
    parser.add_argument("--offset", type=float, help="Offset from center (Hz)")
    parser.add_argument("--squelch", type=float, help="Squelch threshold (dB)")
    parser.add_argument("--audio-rate", type=int, help="Audio output rate (Hz)")

    # AGC settings
    parser.add_argument("--agc", type=str, choices=["on", "off"], help="Enable/disable AGC")
    parser.add_argument("--agc-target", type=float, help="AGC target level (dB)")
    parser.add_argument("--agc-attack", type=float, help="AGC attack time (ms)")
    parser.add_argument("--agc-release", type=float, help="AGC release time (ms)")

    # Filter settings
    parser.add_argument("--deemphasis", type=str, choices=["on", "off"], help="Enable/disable de-emphasis")
    parser.add_argument("--deemphasis-tau", type=float, help="De-emphasis time constant (µs)")
    parser.add_argument("--notch", type=float, nargs="+", help="Notch filter frequencies (Hz)")
    parser.add_argument("--clear-notch", action="store_true", help="Clear all notch filters")

    parser.add_argument("--json", action="store_true", help="Output raw JSON")

    args = parser.parse_args()

    try:
        # List all captures and channels
        if args.list:
            captures = get_all_captures(args.host, args.port)
            print("\nCaptures and Channels:")
            print("=" * 60)
            for cap in captures:
                print(f"\n  Capture: {cap['id']} ({cap['state']})")
                print(f"    Frequency: {format_hz(cap['centerHz'])}")
                print(f"    Gain: {cap.get('gain', 'auto')}")
                channels = get_all_channels_for_capture(args.host, args.port, cap["id"])
                for ch in channels:
                    print(f"      Channel: {ch['id']} ({ch['mode']}, {ch['state']})")
            return 0

        # Show capture settings
        if args.capture and args.show:
            capture = get_capture(args.host, args.port, args.capture)
            if args.json:
                print(json.dumps(capture, indent=2))
            else:
                print_capture_settings(capture)

            # Also show channels
            channels = get_all_channels_for_capture(args.host, args.port, args.capture)
            for ch in channels:
                if args.json:
                    print(json.dumps(ch, indent=2))
                else:
                    print_channel_settings(ch)
            return 0

        # Show channel settings
        if args.channel and args.show:
            channel = get_channel(args.host, args.port, args.channel)
            if args.json:
                print(json.dumps(channel, indent=2))
            else:
                print_channel_settings(channel)
            return 0

        # Build capture update payload
        capture_updates: dict[str, Any] = {}
        if args.frequency is not None:
            capture_updates["centerHz"] = args.frequency * 1e6
        if args.sample_rate is not None:
            capture_updates["sampleRate"] = args.sample_rate
        if args.gain is not None:
            capture_updates["gain"] = args.gain
        if args.bandwidth is not None:
            capture_updates["bandwidth"] = args.bandwidth
        if args.ppm is not None:
            capture_updates["ppm"] = args.ppm
        if args.antenna is not None:
            capture_updates["antenna"] = args.antenna

        # Build channel update payload
        channel_updates: dict[str, Any] = {}
        if args.mode is not None:
            channel_updates["mode"] = args.mode
        if args.offset is not None:
            channel_updates["offsetHz"] = args.offset
        if args.squelch is not None:
            channel_updates["squelchDb"] = args.squelch
        if args.audio_rate is not None:
            channel_updates["audioRate"] = args.audio_rate
        if args.agc is not None:
            channel_updates["enableAgc"] = args.agc == "on"
        if args.agc_target is not None:
            channel_updates["agcTargetDb"] = args.agc_target
        if args.agc_attack is not None:
            channel_updates["agcAttackMs"] = args.agc_attack
        if args.agc_release is not None:
            channel_updates["agcReleaseMs"] = args.agc_release
        if args.deemphasis is not None:
            channel_updates["enableDeemphasis"] = args.deemphasis == "on"
        if args.deemphasis_tau is not None:
            channel_updates["deemphasisTauUs"] = args.deemphasis_tau
        if args.notch is not None:
            channel_updates["notchFrequencies"] = args.notch
        if args.clear_notch:
            channel_updates["notchFrequencies"] = []

        # Apply capture updates
        if capture_updates and args.capture:
            print(f"Updating capture {args.capture}...")
            result = update_capture(args.host, args.port, args.capture, capture_updates)
            print(f"  Updated: {list(capture_updates.keys())}")
            if args.json:
                print(json.dumps(result, indent=2))

        # Apply channel updates
        if channel_updates and args.channel:
            print(f"Updating channel {args.channel}...")
            result = update_channel(args.host, args.port, args.channel, channel_updates)
            print(f"  Updated: {list(channel_updates.keys())}")
            if args.json:
                print(json.dumps(result, indent=2))

        # If nothing was updated, show help
        if not capture_updates and not channel_updates and not args.show and not args.list:
            print("No settings to update. Use --show to see current settings, or specify settings to change.")
            print("Examples:")
            print("  python adjust_settings.py --capture c1 --show")
            print("  python adjust_settings.py --capture c1 --frequency 90.3 --gain 35")
            print("  python adjust_settings.py --channel ch1 --squelch -50")
            return 1

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
