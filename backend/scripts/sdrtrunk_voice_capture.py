#!/usr/bin/env python3
"""Efficiently capture voice basebands from SDRTrunk.

This script monitors SDRTrunk for traffic channel recordings with actual voice
signal (not just noise). When a valid voice recording is found, it copies it
to a test directory for analysis.

How it works:
1. Watches ~/SDRTrunk/recordings/ for new traffic baseband files
2. Waits for file to stop growing (call ended)
3. Analyzes signal to check if it's actual voice vs noise
4. Copies valid recordings for testing

Usage:
    1. Start SDRTrunk (manually)
    2. Run: python scripts/sdrtrunk_voice_capture.py
    3. Wait for voice activity - script will capture automatically
"""

import argparse
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

# Directories
SDRTRUNK_RECORDINGS = Path.home() / "SDRTrunk/recordings"
OUTPUT_DIR = Path(__file__).parent.parent / "voice_test_recordings"

# Detection settings
MIN_FILE_SIZE = 50_000  # Minimum 50KB to consider
MIN_SIGNAL_POWER = 1e-6  # Minimum IQ power to be considered signal
MAX_DISC_RANGE = 1.5  # Valid P25 discriminator range (radians)
FILE_STABLE_SECONDS = 3  # Wait for file to stop growing


def analyze_baseband(path: Path) -> dict:
    """Analyze a baseband recording for signal quality."""
    try:
        rate, data = wavfile.read(str(path))

        # Convert to complex IQ
        if len(data.shape) > 1:
            iq = data[:, 0].astype(np.float32) + 1j * data[:, 1].astype(np.float32)
        else:
            iq = data.astype(np.float32)

        iq = iq / 32768.0

        # Basic stats
        iq_power = float(np.mean(np.abs(iq) ** 2))
        duration = len(iq) / rate

        # FM discriminator
        phase = np.angle(iq)
        phase_unwrapped = np.unwrap(phase)
        disc = np.diff(phase_unwrapped).astype(np.float32)

        disc_range = float(disc.max() - disc.min())
        disc_rms = float(np.sqrt(np.mean(disc ** 2)))

        # Check for signal in chunks (voice may not fill entire recording)
        chunk_size = rate // 5  # 200ms chunks
        signal_chunks = 0
        total_chunks = 0

        for i in range(0, len(disc) - chunk_size, chunk_size):
            chunk = disc[i:i + chunk_size]
            chunk_range = float(chunk.max() - chunk.min())
            total_chunks += 1
            if chunk_range < MAX_DISC_RANGE:
                signal_chunks += 1

        signal_ratio = signal_chunks / max(total_chunks, 1)

        # Determine if valid voice
        has_signal = (
            iq_power > MIN_SIGNAL_POWER and
            (disc_range < 2.0 or signal_ratio > 0.1)
        )

        return {
            'valid': True,
            'rate': rate,
            'duration': duration,
            'iq_power': iq_power,
            'disc_range': disc_range,
            'disc_rms': disc_rms,
            'signal_ratio': signal_ratio,
            'has_signal': has_signal,
        }

    except Exception as e:
        return {'valid': False, 'error': str(e)}


def wait_for_file_stable(path: Path, timeout: float = 30) -> bool:
    """Wait for file to stop growing (recording finished)."""
    last_size = 0
    stable_count = 0
    start = time.time()

    while time.time() - start < timeout:
        try:
            current_size = path.stat().st_size
            if current_size == last_size:
                stable_count += 1
                if stable_count >= FILE_STABLE_SECONDS:
                    return True
            else:
                stable_count = 0
                last_size = current_size
        except FileNotFoundError:
            return False

        time.sleep(1)

    return True  # Timeout - assume stable


def monitor_recordings(
    recordings_dir: Path,
    output_dir: Path,
    max_captures: int = 5,
    verbose: bool = True,
):
    """Monitor for new traffic basebands with voice signal."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Monitoring: {recordings_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Will capture up to {max_captures} voice recordings")
    print()
    print("Start SDRTrunk now and wait for voice activity...")
    print("Press Ctrl+C to stop")
    print()

    # Track seen files
    seen_files = set()
    for f in recordings_dir.glob("*.wav"):
        seen_files.add(f.name)

    captures = 0
    check_count = 0

    while captures < max_captures:
        try:
            check_count += 1

            # Find new traffic baseband files
            for f in recordings_dir.glob("*T-*baseband*.wav"):
                if f.name in seen_files:
                    continue

                seen_files.add(f.name)

                # Check minimum size
                try:
                    size = f.stat().st_size
                except FileNotFoundError:
                    continue

                if size < MIN_FILE_SIZE:
                    if verbose:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Skipping small file: {f.name} ({size} bytes)")
                    continue

                print(f"[{datetime.now().strftime('%H:%M:%S')}] New traffic baseband: {f.name}")
                print(f"  Size: {size / 1024:.1f} KB - waiting for recording to finish...")

                # Wait for file to stop growing
                if not wait_for_file_stable(f):
                    print(f"  File disappeared or timeout")
                    continue

                # Analyze signal
                analysis = analyze_baseband(f)

                if not analysis.get('valid'):
                    print(f"  Analysis failed: {analysis.get('error')}")
                    continue

                print(f"  Duration: {analysis['duration']:.2f}s")
                print(f"  IQ power: {analysis['iq_power']:.2e}")
                print(f"  Disc range: {analysis['disc_range']:.2f} rad")
                print(f"  Signal ratio: {analysis['signal_ratio']:.1%}")

                if analysis['has_signal']:
                    # Copy to output directory
                    dest = output_dir / f.name
                    shutil.copy2(f, dest)
                    captures += 1

                    print(f"  ✅ VOICE DETECTED! Saved to: {dest.name}")
                    print(f"  Captures: {captures}/{max_captures}")
                    print()

                    # Beep
                    print('\a', end='', flush=True)
                else:
                    print(f"  ❌ No voice signal (noise only)")

                print()

            # Status every 30 seconds
            if check_count % 30 == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Still monitoring... ({captures} captures so far)")

            time.sleep(1)

        except KeyboardInterrupt:
            print("\nStopping...")
            break

    print()
    print(f"Captured {captures} voice recording(s)")

    if captures > 0:
        print(f"\nRecordings saved to: {output_dir}")
        print("\nTo test with WaveCap decoder:")
        print(f"  cd {output_dir.parent}")
        print(f"  source .venv/bin/activate")
        for f in output_dir.glob("*.wav"):
            print(f"  PYTHONPATH=. python test_sdrtrunk_baseband.py {f}")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Capture voice basebands from SDRTrunk"
    )
    parser.add_argument(
        "--recordings", "-r",
        type=Path,
        default=SDRTRUNK_RECORDINGS,
        help=f"SDRTrunk recordings directory (default: {SDRTRUNK_RECORDINGS})"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for voice captures (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--max", "-n",
        type=int,
        default=5,
        help="Maximum number of recordings to capture (default: 5)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Less verbose output"
    )

    args = parser.parse_args()

    if not args.recordings.exists():
        print(f"SDRTrunk recordings directory not found: {args.recordings}")
        print("\nMake sure SDRTrunk is installed and has been run at least once.")
        sys.exit(1)

    monitor_recordings(
        args.recordings,
        args.output,
        args.max,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
