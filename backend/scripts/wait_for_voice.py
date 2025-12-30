#!/usr/bin/env python3
"""Wait for voice activity on SDRTrunk and save the timing.

Run this script AFTER starting SDRTrunk. It will:
1. Monitor SDRTrunk log for LDU/HDU frames (voice activity)
2. Print alerts when voice is detected
3. Save voice events to a file for later analysis

Usage:
    1. Start SDRTrunk manually
    2. Run: python scripts/wait_for_voice.py
    3. Wait for voice activity (could be minutes to hours depending on traffic)
    4. Script will alert and log when voice is detected
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# SDRTrunk log location
SDRTRUNK_LOG = Path.home() / "SDRTrunk/logs/sdrtrunk_app.log"
VOICE_LOG = Path(__file__).parent.parent / "voice_events.log"

# Regex patterns for voice frames
VOICE_PATTERNS = [
    re.compile(r'\[sdrtrunk channel \[(\d+)/P25-1\] (\d+).*DUID: (LOGICAL_LINK_DATA_UNIT_\d)', re.I),
    re.compile(r'\[sdrtrunk channel \[(\d+)/P25-1\] (\d+).*DUID: (HEADER_DATA_UNIT)', re.I),
    re.compile(r'\[sdrtrunk channel \[(\d+)/P25-1\] (\d+).*DUID: (TERMINATOR)', re.I),
]


def beep():
    """Make an audible beep."""
    print('\a', end='', flush=True)  # Terminal bell


def monitor_log(log_path: Path, output_file: Path, verbose: bool = True):
    """Monitor SDRTrunk log for voice activity."""
    print(f"Monitoring: {log_path}")
    print(f"Saving events to: {output_file}")
    print(f"Waiting for voice activity... (Ctrl+C to stop)")
    print()

    # Skip to end of existing log
    last_pos = 0
    if log_path.exists():
        with open(log_path, 'r') as f:
            f.seek(0, 2)
            last_pos = f.tell()
        print(f"Skipped to end of log (position: {last_pos})")

    voice_events = []
    last_voice_time = 0
    calls_detected = 0

    with open(output_file, 'a') as out:
        out.write(f"\n--- Session started: {datetime.now().isoformat()} ---\n")

        while True:
            try:
                if not log_path.exists():
                    time.sleep(1)
                    continue

                with open(log_path, 'r') as f:
                    # Check if log was rotated
                    f.seek(0, 2)
                    current_size = f.tell()
                    if current_size < last_pos:
                        last_pos = 0
                        print("[Log rotated, restarting from beginning]")

                    f.seek(last_pos)
                    new_lines = f.readlines()
                    last_pos = f.tell()

                for line in new_lines:
                    for pattern in VOICE_PATTERNS:
                        match = pattern.search(line)
                        if match:
                            channel = int(match.group(1))
                            freq_hz = int(match.group(2))
                            frame_type = match.group(3)
                            now = time.time()

                            # Rate limit notifications (1 per second)
                            if now - last_voice_time > 1.0:
                                beep()
                                calls_detected += 1
                                freq_mhz = freq_hz / 1e6

                                msg = (
                                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                                    f"🎙️ VOICE #{calls_detected}: {frame_type} "
                                    f"on {freq_mhz:.3f} MHz (ch {channel})"
                                )
                                print(msg)

                                # Log to file
                                out.write(f"{datetime.now().isoformat()} {frame_type} {freq_hz} ch{channel}\n")
                                out.flush()

                                last_voice_time = now

                                voice_events.append({
                                    'time': datetime.now(),
                                    'frame': frame_type,
                                    'freq_hz': freq_hz,
                                    'channel': channel,
                                })

                time.sleep(0.2)  # Check 5x per second

            except KeyboardInterrupt:
                print(f"\n\nStopped. Detected {calls_detected} voice event(s).")
                if voice_events:
                    print("\nVoice events:")
                    for e in voice_events[-10:]:  # Last 10
                        print(f"  {e['time'].strftime('%H:%M:%S')} - {e['freq_hz']/1e6:.3f} MHz - {e['frame']}")
                break

            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="Wait for SDRTrunk voice activity")
    parser.add_argument("--log", type=Path, default=SDRTRUNK_LOG,
                        help=f"SDRTrunk log file (default: {SDRTRUNK_LOG})")
    parser.add_argument("--output", "-o", type=Path, default=VOICE_LOG,
                        help=f"Output file for voice events (default: {VOICE_LOG})")

    args = parser.parse_args()

    if not args.log.exists():
        print(f"SDRTrunk log not found: {args.log}")
        print("\nPlease start SDRTrunk first:")
        print("1. Open SDRTrunk application")
        print("2. Make sure SA-GRN system is configured and enabled")
        print("3. Wait for control channel lock")
        print("4. Then run this script again")
        sys.exit(1)

    monitor_log(args.log, args.output)


if __name__ == "__main__":
    main()
