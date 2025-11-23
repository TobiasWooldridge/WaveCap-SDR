#!/usr/bin/env python3
"""
View and analyze WaveCap-SDR server logs.

Usage:
    python view_logs.py --all          # View all available log info
    python view_logs.py --errors       # View error logs only
    python view_logs.py --status       # View server status
    python view_logs.py --requests     # View recent API requests
    python view_logs.py --sdr          # View SDR device logs
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Default configuration
DEFAULT_PORT = 8087
DEFAULT_HOST = "127.0.0.1"
ERROR_LOG_PATH = "/tmp/wavecapsdr_error.log"


def get_server_status(host: str, port: int) -> dict:
    """Check if server is running and get basic status."""
    import urllib.request
    import urllib.error

    status = {
        "running": False,
        "pid": None,
        "uptime": None,
        "captures": 0,
        "channels": 0,
        "error": None,
    }

    # Find process
    try:
        result = subprocess.run(
            ["pgrep", "-f", "wavecapsdr"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            pids = result.stdout.strip().split("\n")
            status["pid"] = pids[0] if pids else None
    except Exception:
        pass

    # Check API
    try:
        url = f"http://{host}:{port}/api/v1/captures"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            status["running"] = True
            status["captures"] = len(data)

            # Count channels
            for cap in data:
                try:
                    chan_url = f"http://{host}:{port}/api/v1/captures/{cap['id']}/channels"
                    with urllib.request.urlopen(chan_url, timeout=5) as chan_resp:
                        channels = json.loads(chan_resp.read().decode())
                        status["channels"] += len(channels)
                except Exception:
                    pass
    except urllib.error.URLError as e:
        status["error"] = f"Cannot connect to server: {e.reason}"
    except Exception as e:
        status["error"] = str(e)

    return status


def view_error_log(minutes: Optional[int] = None) -> None:
    """View error log file."""
    print("\n=== Error Log ===")

    if not os.path.exists(ERROR_LOG_PATH):
        print(f"No error log found at {ERROR_LOG_PATH}")
        print("(This is normal if no errors have occurred)")
        return

    try:
        with open(ERROR_LOG_PATH, "r") as f:
            content = f.read()

        if not content.strip():
            print("Error log is empty (no errors recorded)")
            return

        # Optionally filter by time
        if minutes:
            cutoff = datetime.now() - timedelta(minutes=minutes)
            lines = []
            current_entry = []
            entry_time = None

            for line in content.split("\n"):
                if line.startswith("--- "):
                    if current_entry and entry_time and entry_time >= cutoff:
                        lines.extend(current_entry)
                    current_entry = [line]
                    try:
                        time_str = line.strip("- ").strip()
                        entry_time = datetime.fromisoformat(time_str)
                    except ValueError:
                        entry_time = None
                else:
                    current_entry.append(line)

            # Don't forget last entry
            if current_entry and entry_time and entry_time >= cutoff:
                lines.extend(current_entry)

            content = "\n".join(lines)

        if content.strip():
            print(content)
        else:
            print(f"No errors in the last {minutes} minutes")

    except Exception as e:
        print(f"Error reading log file: {e}")


def view_api_requests(host: str, port: int) -> None:
    """Show info about API capabilities (actual request logs require server access)."""
    import urllib.request

    print("\n=== API Status ===")

    try:
        # Get captures
        url = f"http://{host}:{port}/api/v1/captures"
        with urllib.request.urlopen(url, timeout=5) as response:
            captures = json.loads(response.read().decode())

        print(f"\nActive Captures: {len(captures)}")
        for cap in captures:
            print(f"  - {cap['id']}: {cap['state']} @ {cap.get('centerHz', 0)/1e6:.3f} MHz")
            if cap.get("errorMessage"):
                print(f"    ERROR: {cap['errorMessage']}")

        # Get channels for each capture
        total_channels = 0
        for cap in captures:
            chan_url = f"http://{host}:{port}/api/v1/captures/{cap['id']}/channels"
            try:
                with urllib.request.urlopen(chan_url, timeout=5) as chan_resp:
                    channels = json.loads(chan_resp.read().decode())
                    total_channels += len(channels)
                    for ch in channels:
                        print(f"    Channel {ch['id']}: {ch['mode']} {ch['state']}")
            except Exception:
                pass

        print(f"\nTotal Channels: {total_channels}")

    except Exception as e:
        print(f"Cannot fetch API status: {e}")


def view_sdr_logs() -> None:
    """Show SDR-related information."""
    print("\n=== SDR Device Information ===")

    # Try to run SoapySDRUtil
    try:
        result = subprocess.run(
            ["SoapySDRUtil", "--find"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        print("\nDetected SDR Devices:")
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
    except FileNotFoundError:
        print("SoapySDRUtil not found in PATH")
    except subprocess.TimeoutExpired:
        print("SoapySDRUtil timed out")
    except Exception as e:
        print(f"Error running SoapySDRUtil: {e}")


def print_status(status: dict) -> None:
    """Pretty print server status."""
    print("\n=== Server Status ===")

    if status["running"]:
        print(f"Status: RUNNING")
        print(f"PID: {status['pid'] or 'Unknown'}")
        print(f"Captures: {status['captures']}")
        print(f"Channels: {status['channels']}")
    else:
        print(f"Status: NOT RUNNING")
        if status["error"]:
            print(f"Error: {status['error']}")
        if status["pid"]:
            print(f"Process found (PID {status['pid']}) but API not responding")


def main():
    parser = argparse.ArgumentParser(
        description="View and analyze WaveCap-SDR logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--host", default=DEFAULT_HOST, help="Server host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
    parser.add_argument("--minutes", type=int, help="Filter logs to last N minutes")

    # View options
    parser.add_argument("--all", action="store_true", help="Show all log information")
    parser.add_argument("--errors", action="store_true", help="Show error logs")
    parser.add_argument("--status", action="store_true", help="Show server status")
    parser.add_argument("--requests", action="store_true", help="Show API request info")
    parser.add_argument("--sdr", action="store_true", help="Show SDR device info")

    args = parser.parse_args()

    # Default to --all if no options specified
    if not any([args.all, args.errors, args.status, args.requests, args.sdr]):
        args.all = True

    print(f"WaveCap-SDR Log Viewer")
    print(f"Server: {args.host}:{args.port}")
    print("=" * 50)

    if args.all or args.status:
        status = get_server_status(args.host, args.port)
        print_status(status)

    if args.all or args.errors:
        view_error_log(args.minutes)

    if args.all or args.requests:
        view_api_requests(args.host, args.port)

    if args.all or args.sdr:
        view_sdr_logs()

    print("\n" + "=" * 50)
    print("Log analysis complete")


if __name__ == "__main__":
    main()
