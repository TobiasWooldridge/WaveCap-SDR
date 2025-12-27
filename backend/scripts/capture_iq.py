#!/usr/bin/env python3
"""Capture raw IQ data from WaveCap-SDR via WebSocket and save to file.

Usage:
    python scripts/capture_iq.py --capture c1 --duration 10 --output capture.rawiq
    python scripts/capture_iq.py --capture c1 --duration 10 --output capture.wav --format wav

The rawiq format is interleaved int16 I/Q samples (same as WaveCap WebSocket output).
The WAV format is stereo (I=left, Q=right) int16.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import struct
import sys
import wave
from pathlib import Path

import numpy as np

try:
    import websockets
except ImportError:
    print("Error: websockets library not installed. Run: pip install websockets")
    sys.exit(1)

logger = logging.getLogger(__name__)


async def capture_iq(
    host: str,
    port: int,
    capture_id: str,
    duration_seconds: float,
    output_path: Path,
    output_format: str,
) -> None:
    """Capture IQ data from WaveCap-SDR WebSocket."""

    url = f"ws://{host}:{port}/api/v1/stream/captures/{capture_id}/iq"
    logger.info(f"Connecting to {url}...")

    all_data = bytearray()
    start_time = None
    sample_rate = None

    # First get capture info for sample rate
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://{host}:{port}/api/v1/captures/{capture_id}") as resp:
            if resp.status != 200:
                raise RuntimeError(f"Failed to get capture info: {resp.status}")
            info = await resp.json()
            sample_rate = info.get("sampleRate", 48000)
            logger.info(f"Capture sample rate: {sample_rate} Hz")

    async with websockets.connect(url) as ws:
        logger.info("Connected, capturing IQ data...")
        start_time = asyncio.get_event_loop().time()

        try:
            while True:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= duration_seconds:
                    break

                try:
                    data = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    if isinstance(data, bytes):
                        all_data.extend(data)
                        # IQ samples are int16 interleaved, so 4 bytes per sample
                        sample_count = len(all_data) // 4
                        if sample_count % 10000 < 1000:
                            logger.info(f"Captured {sample_count} IQ samples ({elapsed:.1f}s / {duration_seconds}s)")
                except asyncio.TimeoutError:
                    continue
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            raise

    # Calculate total samples
    total_samples = len(all_data) // 4
    logger.info(f"Capture complete: {total_samples} IQ samples")

    # Save to file
    if output_format == "rawiq":
        output_path.write_bytes(all_data)
        logger.info(f"Saved to {output_path} (rawiq format, {len(all_data)} bytes)")

        # Also save metadata
        meta_path = output_path.with_suffix(".meta")
        meta_path.write_text(f"sample_rate={sample_rate}\nformat=int16_interleaved\n")
        logger.info(f"Metadata saved to {meta_path}")

    elif output_format == "wav":
        # Convert to WAV format (stereo: I=left, Q=right)
        samples = np.frombuffer(all_data, dtype=np.int16)
        if len(samples) % 2 != 0:
            samples = samples[:-1]

        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(2)  # Stereo (I, Q)
            wf.setsampwidth(2)  # int16
            wf.setframerate(sample_rate)
            wf.writeframes(samples.tobytes())

        logger.info(f"Saved to {output_path} (WAV stereo, {sample_rate} Hz)")

    else:
        raise ValueError(f"Unknown output format: {output_format}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture IQ data from WaveCap-SDR")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="WaveCap-SDR host")
    parser.add_argument("--port", type=int, default=8087, help="WaveCap-SDR port")
    parser.add_argument("--capture", type=str, required=True, help="Capture ID (e.g., c1)")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration in seconds")
    parser.add_argument("--output", type=Path, required=True, help="Output file path")
    parser.add_argument(
        "--format",
        type=str,
        choices=["rawiq", "wav"],
        default="rawiq",
        help="Output format (rawiq or wav)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    asyncio.run(
        capture_iq(
            host=args.host,
            port=args.port,
            capture_id=args.capture,
            duration_seconds=args.duration,
            output_path=args.output,
            output_format=args.format,
        )
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
