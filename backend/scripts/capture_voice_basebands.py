#!/usr/bin/env python3
"""Capture voice channel PCM audio from live WaveCap-SDR trunking.

This script connects to a running WaveCap-SDR server with trunking enabled
and saves decoded voice PCM audio to WAV files for offline analysis.

Usage:
    1. Start WaveCap-SDR: ./start-app.sh
    2. Start trunking system: curl -X POST http://localhost:8087/api/v1/trunking/systems/sa_grn/start
    3. Run this script: python scripts/capture_voice_basebands.py
"""

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np
import scipy.io.wavfile as wavfile

# Default settings
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8087
DEFAULT_SYSTEM = "sa_grn"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "voice_captures"


async def get_trunking_status(session: aiohttp.ClientSession, base_url: str, system_id: str) -> dict:
    """Get current trunking system status."""
    url = f"{base_url}/api/v1/trunking/systems/{system_id}"
    try:
        async with session.get(url) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                print(f"Error getting status: {resp.status}")
                return {}
    except Exception as e:
        print(f"Connection error: {e}")
        return {}


async def get_voice_streams(session: aiohttp.ClientSession, base_url: str, system_id: str) -> list:
    """Get list of active voice streams."""
    url = f"{base_url}/api/v1/trunking/systems/{system_id}/voice-streams"
    try:
        async with session.get(url) as resp:
            if resp.status == 200:
                return await resp.json()
            return []
    except Exception:
        return []


async def stream_voice_pcm(
    session: aiohttp.ClientSession,
    host: str,
    port: int,
    system_id: str,
    stream_id: str,
    output_path: Path,
    duration_s: float = 10.0,
) -> bool:
    """Stream decoded PCM audio and save to WAV file."""
    url = f"http://{host}:{port}/api/v1/trunking/stream/{system_id}/voice/{stream_id}.pcm"

    samples = []
    start_time = time.time()
    sample_rate = 48000  # Voice channel sample rate

    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                print(f"  Error: HTTP {resp.status}")
                return False

            print(f"  Streaming from voice stream {stream_id}...")

            async for chunk in resp.content.iter_any():
                if chunk:
                    # PCM16 data as int16
                    pcm = np.frombuffer(chunk, dtype=np.int16)
                    samples.append(pcm)

                    elapsed = time.time() - start_time
                    if elapsed >= duration_s:
                        break

    except asyncio.TimeoutError:
        print(f"  Timeout after {duration_s}s")
    except Exception as e:
        print(f"  Stream error: {e}")
        return False

    if not samples:
        print(f"  No samples received")
        return False

    # Concatenate and save
    all_samples = np.concatenate(samples)

    # Save as WAV (mono PCM16)
    wavfile.write(str(output_path), sample_rate, all_samples)
    duration = len(all_samples) / sample_rate
    size_kb = output_path.stat().st_size / 1024
    print(f"  Saved: {output_path.name} ({duration:.1f}s, {size_kb:.0f} KB)")

    return True


async def monitor_and_capture(
    host: str,
    port: int,
    system_id: str,
    output_dir: Path,
    capture_duration: float = 10.0,
    poll_interval: float = 1.0,
):
    """Monitor for voice calls and capture basebands."""
    base_url = f"http://{host}:{port}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Monitoring trunking system: {system_id}")
    print(f"Output directory: {output_dir}")
    print(f"Press Ctrl+C to stop\n")

    seen_calls = set()  # Track calls we've already captured

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                # Get system status
                status = await get_trunking_status(session, base_url, system_id)
                if not status:
                    print("Waiting for trunking system...")
                    await asyncio.sleep(5)
                    continue

                # Check active voice streams
                voice_streams = await get_voice_streams(session, base_url, system_id)
                for stream in voice_streams:
                    if stream.get("state") not in ("active", "silent"):
                        continue

                    talkgroup = stream.get("talkgroupId")
                    stream_id = stream.get("id")
                    start_time = stream.get("startTime", 0)

                    # Create unique key for this call
                    call_key = f"{stream_id}_{talkgroup}_{start_time}"

                    if call_key not in seen_calls:
                        seen_calls.add(call_key)

                        # Generate filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{timestamp}_TG{talkgroup}_voice_pcm.wav"
                        output_path = output_dir / filename

                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Voice stream detected!")
                        print(f"  TG: {talkgroup}, Stream: {stream_id}")

                        # Capture in background
                        asyncio.create_task(
                            stream_voice_pcm(
                                session, host, port, system_id, stream_id,
                                output_path, capture_duration
                            )
                        )

                await asyncio.sleep(poll_interval)

            except KeyboardInterrupt:
                print("\nStopping...")
                break
            except Exception as e:
                print(f"Error: {e}")
                await asyncio.sleep(5)


def main():
    parser = argparse.ArgumentParser(description="Capture voice basebands from WaveCap-SDR trunking")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Server host (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Server port (default: {DEFAULT_PORT})")
    parser.add_argument("--system", default=DEFAULT_SYSTEM, help=f"Trunking system ID (default: {DEFAULT_SYSTEM})")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--duration", "-d", type=float, default=10.0, help="Capture duration per call (seconds)")

    args = parser.parse_args()

    asyncio.run(monitor_and_capture(
        args.host, args.port, args.system, args.output, args.duration
    ))


if __name__ == "__main__":
    main()
