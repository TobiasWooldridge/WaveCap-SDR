#!/usr/bin/env python3
"""Monitor both SDRTrunk and WaveCap-SDR for voice activity.

This script watches SDRTrunk logs for voice channel activity and simultaneously
monitors WaveCap-SDR to compare behavior.

Usage:
    1. Start SDRTrunk (make sure it's configured for SA-GRN)
    2. Start WaveCap-SDR with trunking enabled
    3. Run this script: python scripts/dual_monitor.py

The script will alert when either system detects voice and log the comparison.
"""

import argparse
import asyncio
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import aiohttp

# Default settings
SDRTRUNK_LOG = Path.home() / "SDRTrunk/logs/sdrtrunk_app.log"
WAVECAP_HOST = "localhost"
WAVECAP_PORT = 8087
WAVECAP_SYSTEM = "sa_grn_2"

# Regex patterns for SDRTrunk log parsing
SDRTRUNK_LDU_PATTERN = re.compile(
    r'\[sdrtrunk channel \[(\d+)/P25-1\] (\d+) thread.*DUID: (LOGICAL_LINK_DATA_UNIT_\d|HEADER_DATA_UNIT|TERMINATOR)',
    re.IGNORECASE
)
SDRTRUNK_SYNC_PATTERN = re.compile(
    r'\[sdrtrunk channel \[(\d+)/P25-1\] (\d+) thread.*Sync detected',
    re.IGNORECASE
)


class SDRTrunkMonitor:
    """Monitor SDRTrunk logs for voice activity."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.last_position = 0
        self.voice_events = []

    def check_new_lines(self) -> list[dict]:
        """Check for new log lines and parse voice events."""
        if not self.log_path.exists():
            return []

        events = []
        try:
            with open(self.log_path, 'r') as f:
                f.seek(0, 2)  # Seek to end to get size
                current_size = f.tell()

                if current_size < self.last_position:
                    # Log was rotated, start from beginning
                    self.last_position = 0

                f.seek(self.last_position)
                new_lines = f.readlines()
                self.last_position = f.tell()

                for line in new_lines:
                    # Check for LDU/HDU frames (voice activity)
                    match = SDRTRUNK_LDU_PATTERN.search(line)
                    if match:
                        channel_num = int(match.group(1))
                        frequency_hz = int(match.group(2))
                        frame_type = match.group(3)
                        events.append({
                            'source': 'sdrtrunk',
                            'time': datetime.now(),
                            'type': 'voice_frame',
                            'channel': channel_num,
                            'frequency_hz': frequency_hz,
                            'frame': frame_type,
                        })

        except Exception as e:
            print(f"Error reading SDRTrunk log: {e}")

        return events


class WaveCapMonitor:
    """Monitor WaveCap-SDR for voice activity."""

    def __init__(self, host: str, port: int, system_id: str):
        self.base_url = f"http://{host}:{port}"
        self.system_id = system_id
        self.last_active_recorders = set()

    async def check_voice_activity(self, session: aiohttp.ClientSession) -> list[dict]:
        """Check WaveCap for active voice recorders."""
        events = []

        try:
            url = f"{self.base_url}/api/v1/trunking/systems/{self.system_id}"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                if resp.status != 200:
                    return events

                data = await resp.json()
                stats = data.get('stats', {})

                # Check for active recorders
                recorders_active = stats.get('recorders_active', 0)
                if recorders_active > 0:
                    # Try to get more details from voice recorder status
                    # (Note: the API might not expose this directly)
                    events.append({
                        'source': 'wavecap',
                        'time': datetime.now(),
                        'type': 'voice_active',
                        'recorders_active': recorders_active,
                        'recorders_idle': stats.get('recorders_idle', 0),
                    })

        except Exception as e:
            # Connection errors are expected when server is restarting
            pass

        return events


async def monitor_both(
    sdrtrunk_log: Path,
    wavecap_host: str,
    wavecap_port: int,
    wavecap_system: str,
    duration_s: float = 300,
):
    """Monitor both systems for voice activity."""
    print(f"Monitoring for up to {duration_s}s...")
    print(f"SDRTrunk log: {sdrtrunk_log}")
    print(f"WaveCap: http://{wavecap_host}:{wavecap_port}/api/v1/trunking/systems/{wavecap_system}")
    print()

    sdrtrunk = SDRTrunkMonitor(sdrtrunk_log)
    wavecap = WaveCapMonitor(wavecap_host, wavecap_port, wavecap_system)

    # Skip to end of existing log
    if sdrtrunk_log.exists():
        with open(sdrtrunk_log, 'r') as f:
            f.seek(0, 2)
            sdrtrunk.last_position = f.tell()
        print(f"Skipped to end of SDRTrunk log (position: {sdrtrunk.last_position})")

    start_time = time.time()
    voice_detected = False

    async with aiohttp.ClientSession() as session:
        while time.time() - start_time < duration_s:
            try:
                # Check SDRTrunk
                sdr_events = sdrtrunk.check_new_lines()
                for event in sdr_events:
                    voice_detected = True
                    print(f"[{event['time'].strftime('%H:%M:%S.%f')[:-3]}] SDRTrunk: "
                          f"{event['frame']} on {event['frequency_hz']/1e6:.3f} MHz (ch {event['channel']})")

                # Check WaveCap
                wc_events = await wavecap.check_voice_activity(session)
                for event in wc_events:
                    voice_detected = True
                    print(f"[{event['time'].strftime('%H:%M:%S.%f')[:-3]}] WaveCap: "
                          f"{event['recorders_active']} active recorder(s)")

                if voice_detected:
                    print("--- Voice activity detected! Monitoring for more... ---")
                    voice_detected = False

                await asyncio.sleep(0.5)

            except KeyboardInterrupt:
                print("\nStopping...")
                break

    print("\nMonitoring complete.")


def main():
    parser = argparse.ArgumentParser(description="Monitor SDRTrunk and WaveCap for voice activity")
    parser.add_argument("--sdrtrunk-log", type=Path, default=SDRTRUNK_LOG,
                        help=f"SDRTrunk log file (default: {SDRTRUNK_LOG})")
    parser.add_argument("--wavecap-host", default=WAVECAP_HOST)
    parser.add_argument("--wavecap-port", type=int, default=WAVECAP_PORT)
    parser.add_argument("--wavecap-system", default=WAVECAP_SYSTEM)
    parser.add_argument("--duration", "-d", type=float, default=300,
                        help="Monitoring duration in seconds (default: 300)")

    args = parser.parse_args()

    asyncio.run(monitor_both(
        args.sdrtrunk_log,
        args.wavecap_host,
        args.wavecap_port,
        args.wavecap_system,
        args.duration,
    ))


if __name__ == "__main__":
    main()
