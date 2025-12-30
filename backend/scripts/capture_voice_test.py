#!/usr/bin/env python3
"""Capture P25 control and voice channel IQ for offline testing.

This script:
1. Connects to the WaveCap-SDR API
2. Monitors for voice grants
3. Records IQ data around voice transmissions
4. Saves both control channel and voice channel recordings
"""

import asyncio
import json
import logging
import os
import struct
import time
from datetime import datetime

import aiohttp
import numpy as np
import scipy.io.wavfile as wav

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

API_BASE = "http://localhost:8087/api/v1"

# Recording parameters
SAMPLE_RATE = 48000  # Audio sample rate
IQ_SAMPLE_RATE = 6000000  # Full capture sample rate
VOICE_BANDWIDTH = 12500  # P25 channel bandwidth
RECORD_DURATION = 30  # seconds per recording


async def get_trunking_status() -> dict:
    """Get current trunking system status."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_BASE}/trunking") as resp:
            if resp.status == 200:
                return await resp.json()
    return {}


async def stream_trunking_events():
    """Stream trunking events via WebSocket."""
    ws_url = "ws://localhost:8087/api/v1/stream/trunking/sa_grn_2/events"

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(ws_url) as ws:
            logger.info(f"Connected to trunking event stream")

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    event = json.loads(msg.data)
                    yield event
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
                    break


async def record_audio_stream(channel_id: str, duration: float, filename: str):
    """Record audio from a channel to a WAV file."""
    ws_url = f"ws://localhost:8087/api/v1/stream/channels/{channel_id}?format=pcm16"

    samples = []
    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        try:
            async with session.ws_connect(ws_url, timeout=5) as ws:
                logger.info(f"Recording channel {channel_id} to {filename}")

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        # PCM16 audio
                        audio = np.frombuffer(msg.data, dtype=np.int16)
                        samples.append(audio)

                        if time.time() - start_time > duration:
                            break

                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        break

        except Exception as e:
            logger.error(f"Recording error: {e}")

    if samples:
        audio = np.concatenate(samples)
        wav.write(filename, SAMPLE_RATE, audio)
        logger.info(f"Saved {len(audio)} samples to {filename}")
        return True
    return False


async def record_discriminator_audio(trunking_system: str, duration: float, filename: str):
    """Record discriminator audio (FM demod output) for voice channel testing."""

    # Get the voice recorder's channel
    status = await get_trunking_status()
    if not status:
        logger.error("Could not get trunking status")
        return False

    systems = status.get("systems", {})
    system = systems.get(trunking_system, {})
    voice_recorders = system.get("voice_recorders", [])

    if not voice_recorders:
        logger.warning(f"No voice recorders active for {trunking_system}")
        return False

    # Get first active voice recorder
    recorder = voice_recorders[0]
    channel_id = recorder.get("channel_id")
    if not channel_id:
        logger.warning("Voice recorder has no channel_id")
        return False

    logger.info(f"Recording voice channel {channel_id}")
    return await record_audio_stream(channel_id, duration, filename)


async def main():
    """Main recording loop."""
    output_dir = "recordings/test_captures"
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Starting P25 capture for testing...")
    logger.info(f"Output directory: {output_dir}")

    # Check trunking status
    status = await get_trunking_status()
    if not status:
        logger.error("Could not connect to WaveCap-SDR API")
        return

    systems = status.get("systems", {})
    logger.info(f"Found {len(systems)} trunking systems")

    for name, sys_status in systems.items():
        logger.info(f"  {name}: state={sys_status.get('state')}, "
                   f"cc_state={sys_status.get('control_channel_state')}")

    # Monitor for voice grants
    grant_count = 0
    recording_active = False

    try:
        async for event in stream_trunking_events():
            event_type = event.get("type")

            if event_type == "voice_grant":
                grant_count += 1
                tgid = event.get("talkgroup_id", "unknown")
                freq = event.get("frequency_hz", 0) / 1e6
                logger.info(f"Voice grant #{grant_count}: TG={tgid}, freq={freq:.4f} MHz")

                # Record if not already recording
                if not recording_active:
                    recording_active = True
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{output_dir}/voice_{tgid}_{timestamp}.wav"

                    # Record in background
                    asyncio.create_task(
                        record_and_mark_done(filename, event.get("system_id", "sa_grn_2"))
                    )

            elif event_type == "call_end":
                tgid = event.get("talkgroup_id", "unknown")
                duration = event.get("duration_seconds", 0)
                logger.info(f"Call ended: TG={tgid}, duration={duration:.1f}s")
                recording_active = False

            elif event_type == "cc_locked":
                freq = event.get("frequency_hz", 0) / 1e6
                logger.info(f"Control channel locked: {freq:.4f} MHz")

            elif event_type == "cc_lost":
                logger.warning("Control channel lost!")

    except KeyboardInterrupt:
        logger.info("Recording stopped by user")
    except Exception as e:
        logger.error(f"Error: {e}")


async def record_and_mark_done(filename: str, system_id: str):
    """Record and log when done."""
    success = await record_discriminator_audio(system_id, RECORD_DURATION, filename)
    if success:
        logger.info(f"Recording complete: {filename}")
    else:
        logger.warning(f"Recording failed: {filename}")


if __name__ == "__main__":
    asyncio.run(main())
