"""Voice channel for P25 trunking audio streaming.

Handles audio decoding via vocoder (DSD-FME) and streaming to subscribers
with attached metadata (talkgroup, source ID, GPS location).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import logging
import os
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from wavecapsdr.decoders.voice import VocoderType, VoiceDecoder, VoiceDecoderError

logger = logging.getLogger(__name__)


def pack_pcm16(samples: np.ndarray) -> bytes:
    """Convert float32 audio to 16-bit signed PCM bytes."""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


@dataclass
class RadioLocation:
    """GPS location report from a radio unit."""
    unit_id: int
    latitude: float
    longitude: float
    altitude_m: float | None = None
    speed_kmh: float | None = None
    heading_deg: float | None = None
    accuracy_m: float | None = None
    timestamp: float = 0.0
    source: str = "unknown"  # "lrrp", "elc", "gps_tsbk"

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def is_valid(self) -> bool:
        """Check if coordinates are valid."""
        return -90 <= self.latitude <= 90 and -180 <= self.longitude <= 180

    def age_seconds(self) -> float:
        """Get age of location report."""
        return time.time() - self.timestamp

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "unitId": self.unit_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude_m,
            "speed": self.speed_kmh,
            "heading": self.heading_deg,
            "accuracy": self.accuracy_m,
            "timestamp": self.timestamp,
            "ageSeconds": self.age_seconds(),
            "source": self.source,
        }


@dataclass
class VoiceChannelConfig:
    """Configuration for a voice channel."""
    id: str
    system_id: str
    call_id: str
    recorder_id: str
    audio_rate: int = 8000  # DSD-FME outputs 8kHz
    output_rate: int = 48000  # Resampled output rate
    audio_gain: float = 1.0  # Audio output gain multiplier
    should_record: bool = True  # Whether to record this call to file
    recording_path: str = "./recordings"  # Path for audio file storage
    min_call_duration: float = 1.0  # Minimum call duration to save (seconds)


@dataclass
class VoiceChannel:
    """Voice stream for a P25 trunking call.

    Handles vocoder decoding and audio streaming with metadata.
    Uses same pub/sub pattern as regular Channel for audio delivery.

    Usage:
        channel = VoiceChannel(cfg=VoiceChannelConfig(...))
        await channel.start(vocoder_type=VocoderType.IMBE)

        # Subscribe to audio
        queue = await channel.subscribe_audio("pcm16")

        # Feed discriminator audio from demodulator
        await channel.process_discriminator_audio(disc_audio)

        # Audio with metadata is delivered to queue
        message = await queue.get()  # JSON bytes with audio payload
    """

    cfg: VoiceChannelConfig
    state: str = "created"  # created, starting, active, silent, ended

    # Call metadata
    talkgroup_id: int = 0
    talkgroup_name: str = ""
    source_id: int | None = None
    encrypted: bool = False

    # Location from LRRP cache
    source_location: RadioLocation | None = None

    # Timing
    start_time: float = field(default_factory=time.time)
    last_audio_time: float = field(default_factory=time.time)
    silence_timeout: float = 60.0  # Seconds of silence before release

    # Statistics
    audio_frame_count: int = 0
    audio_bytes_sent: int = 0

    # Vocoder
    _voice_decoder: VoiceDecoder | None = field(default=None, repr=False)

    # Audio subscribers: (queue, event_loop, format)
    _audio_sinks: set[tuple[asyncio.Queue[bytes], asyncio.AbstractEventLoop, str]] = field(
        default_factory=set, repr=False
    )

    # Decoder output reader task
    _decoder_reader_task: asyncio.Task[None] | None = field(default=None, repr=False)

    # Recording buffer for WAV file output
    _recording_buffer: io.BytesIO | None = field(default=None, repr=False)
    _recording_samples: int = 0  # Total samples recorded

    def __post_init__(self) -> None:
        """Initialize timing."""
        self.start_time = time.time()
        self.last_audio_time = time.time()

    @property
    def id(self) -> str:
        """Get channel ID."""
        return self.cfg.id

    @property
    def duration_seconds(self) -> float:
        """Get call duration in seconds."""
        return time.time() - self.start_time

    @property
    def silence_seconds(self) -> float:
        """Get seconds since last audio."""
        return time.time() - self.last_audio_time

    @property
    def is_silent(self) -> bool:
        """Check if channel has exceeded silence timeout."""
        return self.silence_seconds > self.silence_timeout

    async def start(self, vocoder_type: VocoderType = VocoderType.IMBE) -> None:
        """Start the voice channel with vocoder."""
        if self.state not in ("created", "ended"):
            return

        self.state = "starting"

        try:
            # Create and start vocoder
            self._voice_decoder = VoiceDecoder(
                vocoder_type=vocoder_type,
                output_rate=self.cfg.output_rate,
                input_rate=48000,  # DSD-FME expects 48kHz discriminator audio
            )
            await self._voice_decoder.start()

            # Start decoder output reader
            self._decoder_reader_task = asyncio.create_task(self._read_decoder_output())

            # Initialize recording buffer if recording is enabled
            if self.cfg.should_record:
                self._recording_buffer = io.BytesIO()
                self._recording_samples = 0

            self.state = "active"
            self.start_time = time.time()
            self.last_audio_time = time.time()

            logger.info(
                f"VoiceChannel {self.id}: Started ({vocoder_type.value}, "
                f"TG={self.talkgroup_id}, src={self.source_id}"
                f"{', RECORDING' if self.cfg.should_record else ''})"
            )

        except VoiceDecoderError as e:
            self.state = "ended"
            logger.error(f"VoiceChannel {self.id}: Failed to start vocoder: {e}")
            raise

    async def stop(self) -> None:
        """Stop the voice channel."""
        if self.state == "ended":
            return

        self.state = "ended"

        # Cancel decoder reader
        if self._decoder_reader_task:
            self._decoder_reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._decoder_reader_task
            self._decoder_reader_task = None

        # Stop vocoder
        if self._voice_decoder:
            await self._voice_decoder.stop()
            self._voice_decoder = None

        # Save recording if enabled and meets minimum duration
        recording_saved = False
        if self._recording_buffer and self._recording_samples > 0:
            duration_s = self._recording_samples / self.cfg.output_rate
            if duration_s >= self.cfg.min_call_duration:
                try:
                    recording_saved = self._save_recording()
                except Exception as e:
                    logger.error(f"VoiceChannel {self.id}: Failed to save recording: {e}")
            else:
                logger.debug(
                    f"VoiceChannel {self.id}: Recording too short "
                    f"({duration_s:.1f}s < {self.cfg.min_call_duration}s), not saving"
                )

        # Clear recording buffer
        self._recording_buffer = None
        self._recording_samples = 0

        # Clear subscribers
        self._audio_sinks.clear()

        logger.info(
            f"VoiceChannel {self.id}: Stopped (frames={self.audio_frame_count}, "
            f"duration={self.duration_seconds:.1f}s"
            f"{', saved' if recording_saved else ''})"
        )

    async def subscribe_audio(self, format: str = "json") -> asyncio.Queue[bytes]:
        """Subscribe to audio stream.

        Args:
            format: "json" (audio + metadata), "pcm16", or "f32"

        Returns:
            Queue that receives audio messages.
        """
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=64)
        loop = asyncio.get_running_loop()
        self._audio_sinks.add((q, loop, format))

        logger.info(
            f"VoiceChannel {self.id}: Subscriber added (format={format}, "
            f"total={len(self._audio_sinks)})"
        )
        return q

    def unsubscribe(self, q: asyncio.Queue[bytes]) -> None:
        """Unsubscribe from audio stream."""
        for item in list(self._audio_sinks):
            if item[0] is q:
                self._audio_sinks.discard(item)
                logger.info(
                    f"VoiceChannel {self.id}: Subscriber removed "
                    f"(total={len(self._audio_sinks)})"
                )
                break

    async def process_discriminator_audio(self, disc_audio: np.ndarray) -> None:
        """Feed discriminator audio to vocoder for decoding.

        Args:
            disc_audio: Instantaneous frequency values from FM discriminator (48kHz)
        """
        if self.state != "active" or not self._voice_decoder:
            return

        await self._voice_decoder.decode(disc_audio)

    async def _read_decoder_output(self) -> None:
        """Read decoded audio from vocoder and broadcast to subscribers."""
        while self.state == "active" and self._voice_decoder:
            try:
                # Wait for decoded audio
                audio = await self._voice_decoder.get_audio_blocking(timeout=0.5)

                if audio is not None and len(audio) > 0:
                    self.last_audio_time = time.time()
                    self.audio_frame_count += 1
                    await self._broadcast(audio)

                # Check for silence timeout
                if self.is_silent and self.state == "active":
                    self.state = "silent"
                    logger.info(
                        f"VoiceChannel {self.id}: Silence timeout "
                        f"({self.silence_seconds:.1f}s)"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"VoiceChannel {self.id}: Decoder read error: {e}")
                await asyncio.sleep(0.1)

    async def _broadcast(self, audio: np.ndarray) -> None:
        """Broadcast audio to all subscribers with metadata."""
        # Apply audio gain
        if self.cfg.audio_gain != 1.0:
            audio = audio * self.cfg.audio_gain

        # Write to recording buffer if enabled
        if self._recording_buffer is not None:
            pcm16_for_recording = pack_pcm16(audio)
            self._recording_buffer.write(pcm16_for_recording)
            self._recording_samples += len(audio)

        if not self._audio_sinks:
            return

        # Build metadata
        metadata = self._build_metadata()

        # Pack audio formats
        pcm16_data = pack_pcm16(audio)
        self.audio_bytes_sent += len(pcm16_data)

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        for q, loop, fmt in list(self._audio_sinks):
            try:
                if fmt == "json":
                    # JSON format with base64 audio
                    payload = self._pack_json_message(pcm16_data, metadata)
                elif fmt == "f32":
                    # Raw float32 audio
                    payload = audio.astype(np.float32).tobytes()
                else:
                    # Default to pcm16
                    payload = pcm16_data

                self._queue_put(q, loop, payload, current_loop)

            except Exception as e:
                logger.error(f"VoiceChannel {self.id}: Broadcast error: {e}")

    def _build_metadata(self) -> dict[str, Any]:
        """Build metadata dictionary for audio message."""
        return {
            "streamId": self.cfg.id,
            "systemId": self.cfg.system_id,
            "callId": self.cfg.call_id,
            "recorderId": self.cfg.recorder_id,
            "talkgroupId": self.talkgroup_id,
            "talkgroupName": self.talkgroup_name,
            "sourceId": self.source_id,
            "sourceLocation": self.source_location.to_dict() if self.source_location else None,
            "timestamp": time.time(),
            "encrypted": self.encrypted,
            "sampleRate": self.cfg.output_rate,
            "frameNumber": self.audio_frame_count,
        }

    def _pack_json_message(self, pcm16_data: bytes, metadata: dict[str, Any]) -> bytes:
        """Pack audio and metadata into JSON message."""
        message = {
            "type": "audio",
            **metadata,
            "format": "pcm16",
            "audio": base64.b64encode(pcm16_data).decode("ascii"),
        }
        return json.dumps(message).encode("utf-8")

    def _queue_put(
        self,
        q: asyncio.Queue[bytes],
        loop: asyncio.AbstractEventLoop,
        payload: bytes,
        current_loop: asyncio.AbstractEventLoop | None,
    ) -> None:
        """Put payload on queue, handling cross-loop delivery."""
        if current_loop is loop:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                # Drop oldest and retry
                try:
                    q.get_nowait()
                    q.put_nowait(payload)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass
        else:
            # Cross-loop delivery
            def _try_put() -> None:
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    try:
                        q.get_nowait()
                        q.put_nowait(payload)
                    except (asyncio.QueueEmpty, asyncio.QueueFull):
                        pass

            try:
                loop.call_soon_threadsafe(_try_put)
            except Exception:
                # Loop closed, remove subscriber
                self._audio_sinks.discard((q, loop, ""))

    def _save_recording(self) -> bool:
        """Save recorded audio to WAV file.

        Returns True if recording was saved successfully.
        """
        if self._recording_buffer is None or self._recording_samples == 0:
            return False

        # Create recording directory if needed
        recording_dir = Path(self.cfg.recording_path)
        try:
            recording_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"VoiceChannel {self.id}: Cannot create recording dir: {e}")
            return False

        # Generate filename: {system}_{tg}_{timestamp}.wav
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.start_time))
        filename = f"{self.cfg.system_id}_TG{self.talkgroup_id}_{timestamp}.wav"
        filepath = recording_dir / filename

        # Get raw PCM data from buffer
        self._recording_buffer.seek(0)
        pcm_data = self._recording_buffer.read()

        # Write WAV file
        try:
            with wave.open(str(filepath), "wb") as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.cfg.output_rate)
                wav_file.writeframes(pcm_data)

            duration_s = self._recording_samples / self.cfg.output_rate
            logger.info(
                f"VoiceChannel {self.id}: Saved recording {filename} "
                f"({duration_s:.1f}s, {len(pcm_data)} bytes)"
            )
            return True

        except Exception as e:
            logger.error(f"VoiceChannel {self.id}: Failed to write WAV: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get channel statistics."""
        return {
            "id": self.cfg.id,
            "systemId": self.cfg.system_id,
            "callId": self.cfg.call_id,
            "recorderId": self.cfg.recorder_id,
            "state": self.state,
            "talkgroupId": self.talkgroup_id,
            "talkgroupName": self.talkgroup_name,
            "sourceId": self.source_id,
            "encrypted": self.encrypted,
            "startTime": self.start_time,
            "durationSeconds": self.duration_seconds,
            "silenceSeconds": self.silence_seconds,
            "audioFrameCount": self.audio_frame_count,
            "audioBytesSent": self.audio_bytes_sent,
            "subscriberCount": len(self._audio_sinks),
            "sourceLocation": self.source_location.to_dict() if self.source_location else None,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return self.get_stats()


@dataclass
class VoiceChannelPool:
    """Pool of voice channels for a trunking system.

    Manages dynamic assignment of channels to active calls with
    silence-based release and priority preemption.
    """

    system_id: str
    max_channels: int = 10
    silence_timeout: float = 60.0

    _channels: dict[str, VoiceChannel] = field(default_factory=dict, repr=False)
    _available_ids: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """Initialize available channel IDs."""
        self._available_ids = [f"{self.system_id}_vc{i}" for i in range(self.max_channels)]

    def get_available_channel_id(self) -> str | None:
        """Get an available channel ID from the pool."""
        if self._available_ids:
            return self._available_ids.pop(0)
        return None

    def return_channel_id(self, channel_id: str) -> None:
        """Return a channel ID to the pool."""
        if channel_id not in self._available_ids:
            self._available_ids.append(channel_id)

    def get_channel(self, channel_id: str) -> VoiceChannel | None:
        """Get a channel by ID."""
        return self._channels.get(channel_id)

    def add_channel(self, channel: VoiceChannel) -> None:
        """Add a channel to the pool."""
        self._channels[channel.id] = channel

    def remove_channel(self, channel_id: str) -> VoiceChannel | None:
        """Remove a channel from the pool."""
        channel = self._channels.pop(channel_id, None)
        if channel:
            self.return_channel_id(channel_id)
        return channel

    def list_channels(self) -> list[VoiceChannel]:
        """List all active channels."""
        return list(self._channels.values())

    def list_active_channels(self) -> list[VoiceChannel]:
        """List channels that are actively streaming."""
        return [c for c in self._channels.values() if c.state == "active"]

    def list_silent_channels(self) -> list[VoiceChannel]:
        """List channels that have exceeded silence timeout."""
        return [c for c in self._channels.values() if c.is_silent]

    async def cleanup_silent_channels(self) -> int:
        """Stop and remove channels that have exceeded silence timeout."""
        silent = self.list_silent_channels()
        for channel in silent:
            await channel.stop()
            self.remove_channel(channel.id)
        return len(silent)

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        return {
            "systemId": self.system_id,
            "maxChannels": self.max_channels,
            "activeChannels": len(self._channels),
            "availableIds": len(self._available_ids),
            "channels": [c.get_stats() for c in self._channels.values()],
        }
