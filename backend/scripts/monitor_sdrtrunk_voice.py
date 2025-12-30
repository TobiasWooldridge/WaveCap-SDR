#!/usr/bin/env python3
"""Monitor SDRTrunk recordings directory for new voice channel basebands.

This script watches for new traffic channel baseband recordings from SDRTrunk
and automatically tests them with the WaveCap voice decoder.

SDRTrunk Setup Required:
1. Open SDRTrunk Playlist Editor
2. For each traffic channel, set Record to "Baseband I/Q"
3. Recordings appear in ~/SDRTrunk/recordings/

Usage:
    python monitor_sdrtrunk_voice.py [--recordings-dir PATH] [--test-immediately]
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime
import subprocess
import shutil

# Default SDRTrunk recordings directory
DEFAULT_RECORDINGS_DIR = os.path.expanduser("~/SDRTrunk/recordings")

# Patterns for traffic channel basebands
TRAFFIC_PATTERNS = ["Traffic", "Voice", "traffic", "voice"]


def find_traffic_basebands(recordings_dir: Path, since_time: float = 0) -> list[Path]:
    """Find traffic channel baseband recordings newer than since_time."""
    basebands = []

    for pattern in ["*Traffic*baseband*.wav", "*Voice*baseband*.wav",
                    "*traffic*baseband*.wav", "*voice*baseband*.wav"]:
        for f in recordings_dir.glob(pattern):
            if f.stat().st_mtime > since_time:
                basebands.append(f)

    return sorted(basebands, key=lambda f: f.stat().st_mtime)


def test_voice_recording(wav_path: Path, output_dir: Path) -> dict:
    """Test a voice recording with WaveCap voice decoder."""
    print(f"\n{'='*60}")
    print(f"Testing: {wav_path.name}")
    print(f"Size: {wav_path.stat().st_size / 1024:.1f} KB")
    print(f"{'='*60}")

    # Get script directory
    script_dir = Path(__file__).parent
    backend_dir = script_dir.parent

    # Run test script
    test_script = script_dir / "test_sdrtrunk_voice.py"

    if not test_script.exists():
        print(f"Creating test script at {test_script}")
        create_voice_test_script(test_script)

    # Run the test
    result = subprocess.run(
        [sys.executable, str(test_script), str(wav_path)],
        cwd=str(backend_dir),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(backend_dir)}
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return {
        "file": str(wav_path),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def create_voice_test_script(path: Path):
    """Create the voice decoder test script."""
    script_content = '''#!/usr/bin/env python3
"""Test WaveCap voice decoder on SDRTrunk traffic channel baseband."""

import sys
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from wavecapsdr.decoders.p25 import P25Decoder
from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator


def test_voice_baseband(wav_path: str):
    """Test voice decoding on SDRTrunk baseband recording."""
    print(f"Loading: {wav_path}")

    # Load WAV file
    sample_rate, data = wavfile.read(wav_path)
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {len(data) / sample_rate:.2f}s")
    print(f"Channels: {data.shape[1] if len(data.shape) > 1 else 1}")

    # Convert to complex IQ
    if len(data.shape) > 1:
        # Stereo = I/Q
        iq = data[:, 0].astype(np.float32) + 1j * data[:, 1].astype(np.float32)
    else:
        # Mono - treat as real
        iq = data.astype(np.float32)

    # Normalize
    iq = iq / 32768.0

    print(f"IQ samples: {len(iq)}")
    print(f"IQ power: {np.mean(np.abs(iq)**2):.6f}")

    # Create P25 decoder with discriminator input
    decoder = P25Decoder(
        sample_rate=sample_rate,
        use_discriminator_input=True,
    )

    # FM discriminator
    phase = np.angle(iq)
    phase_unwrapped = np.unwrap(phase)
    disc_audio = np.diff(phase_unwrapped).astype(np.float32)

    print(f"\\nDiscriminator stats:")
    print(f"  Range: [{disc_audio.min():.3f}, {disc_audio.max():.3f}]")
    print(f"  RMS: {np.sqrt(np.mean(disc_audio**2)):.3f}")

    # Check if valid P25 range
    disc_range = disc_audio.max() - disc_audio.min()
    if disc_range > 2.0:
        print(f"  WARNING: Range {disc_range:.2f} > 2.0 suggests noise, not signal")
    else:
        print(f"  OK: Range {disc_range:.2f} looks like valid P25")

    # Process through decoder
    print(f"\\nDecoding...")

    # Track frames
    frame_count = 0
    ldu_count = 0
    imbe_count = 0

    def on_frame(frame_type, data):
        nonlocal frame_count, ldu_count, imbe_count
        frame_count += 1
        if "LDU" in frame_type:
            ldu_count += 1
        if "imbe" in str(data).lower():
            imbe_count += 1

    # Process in chunks
    chunk_size = sample_rate // 10  # 100ms chunks
    for i in range(0, len(disc_audio), chunk_size):
        chunk = disc_audio[i:i+chunk_size]
        decoder.process_iq(chunk)

    print(f"\\nResults:")
    print(f"  Frames detected: {frame_count}")
    print(f"  LDU frames: {ldu_count}")
    print(f"  IMBE frames: {imbe_count}")

    if ldu_count > 0:
        print(f"\\nSUCCESS: Voice frames detected!")
        return True
    else:
        print(f"\\nNo voice frames detected")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: test_sdrtrunk_voice.py <baseband.wav>")
        sys.exit(1)

    success = test_voice_baseband(sys.argv[1])
    sys.exit(0 if success else 1)
'''
    path.write_text(script_content)
    path.chmod(0o755)


def monitor_recordings(recordings_dir: Path, test_immediately: bool = True):
    """Monitor recordings directory for new voice basebands."""
    print(f"Monitoring: {recordings_dir}")
    print(f"Press Ctrl+C to stop")
    print()

    # Track what we've seen
    seen_files = set()
    last_check = time.time()

    # Find existing files
    existing = find_traffic_basebands(recordings_dir)
    for f in existing:
        seen_files.add(f)
    print(f"Found {len(existing)} existing traffic basebands")

    if test_immediately and existing:
        print(f"\nTesting most recent: {existing[-1].name}")
        test_voice_recording(existing[-1], recordings_dir)

    # Monitor loop
    while True:
        try:
            time.sleep(5)  # Check every 5 seconds

            # Find new files
            current = find_traffic_basebands(recordings_dir)
            new_files = [f for f in current if f not in seen_files]

            for f in new_files:
                seen_files.add(f)
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] New recording: {f.name}")

                # Wait a moment for file to finish writing
                time.sleep(2)

                # Test it
                test_voice_recording(f, recordings_dir)

        except KeyboardInterrupt:
            print("\nStopping monitor")
            break


def main():
    parser = argparse.ArgumentParser(description="Monitor SDRTrunk for voice recordings")
    parser.add_argument(
        "--recordings-dir", "-d",
        default=DEFAULT_RECORDINGS_DIR,
        help=f"SDRTrunk recordings directory (default: {DEFAULT_RECORDINGS_DIR})"
    )
    parser.add_argument(
        "--test-immediately", "-t",
        action="store_true",
        default=True,
        help="Test the most recent recording immediately"
    )
    parser.add_argument(
        "--list-only", "-l",
        action="store_true",
        help="Just list existing recordings and exit"
    )

    args = parser.parse_args()

    recordings_dir = Path(args.recordings_dir)
    if not recordings_dir.exists():
        print(f"Recordings directory not found: {recordings_dir}")
        print(f"\nTo set up SDRTrunk recording:")
        print(f"1. Open SDRTrunk")
        print(f"2. Go to Playlist Editor")
        print(f"3. Select your P25 system")
        print(f"4. For traffic channels, set Record to 'Baseband I/Q'")
        print(f"5. Recordings will appear in {recordings_dir}")
        sys.exit(1)

    if args.list_only:
        basebands = find_traffic_basebands(recordings_dir)
        print(f"Found {len(basebands)} traffic channel basebands:")
        for f in basebands[-10:]:  # Show last 10
            size_kb = f.stat().st_size / 1024
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            print(f"  {mtime.strftime('%Y-%m-%d %H:%M')} - {f.name} ({size_kb:.0f} KB)")
        return

    monitor_recordings(recordings_dir, args.test_immediately)


if __name__ == "__main__":
    main()
