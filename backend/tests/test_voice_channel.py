"""Tests for VoiceChannel class.

Tests voice channel lifecycle, audio subscription, metadata building,
and pool management for P25 trunking voice streams.

Reference: SDRTrunk (https://github.com/DSheirer/sdrtrunk)
"""

import asyncio
import base64
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from wavecapsdr.trunking.voice_channel import (
    VoiceChannel,
    VoiceChannelConfig,
    VoiceChannelPool,
    RadioLocation,
    pack_pcm16,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def voice_channel_config():
    """Sample voice channel configuration."""
    return VoiceChannelConfig(
        id="test_vc0",
        system_id="test_system",
        call_id="call_001",
        recorder_id="vr0",
        audio_rate=8000,
        output_rate=48000,
    )


@pytest.fixture
def sample_location():
    """Sample GPS location."""
    return RadioLocation(
        unit_id=12345678,
        latitude=47.6062,
        longitude=-122.3321,
        altitude_m=100.0,
        speed_kmh=45.0,
        heading_deg=270.0,
        source="elc",
    )


@pytest.fixture
def sample_audio():
    """Sample audio data as float32 array."""
    # 100ms of 1kHz tone at 48kHz
    t = np.linspace(0, 0.1, 4800, dtype=np.float32)
    return np.sin(2 * np.pi * 1000 * t).astype(np.float32)


# ============================================================================
# pack_pcm16 Tests
# ============================================================================

class TestPackPCM16:
    """Test PCM16 packing function."""

    def test_pack_silence(self):
        """Pack silence (zeros)."""
        audio = np.zeros(100, dtype=np.float32)
        pcm = pack_pcm16(audio)

        assert len(pcm) == 200  # 100 samples * 2 bytes
        # All bytes should be zero (or close due to float conversion)
        assert pcm == b'\x00' * 200

    def test_pack_max_amplitude(self):
        """Pack max amplitude signal."""
        audio = np.ones(10, dtype=np.float32)
        pcm = pack_pcm16(audio)

        # Max value should be close to 32767
        values = np.frombuffer(pcm, dtype=np.int16)
        assert np.all(values == 32767)

    def test_pack_min_amplitude(self):
        """Pack min amplitude signal."""
        audio = -np.ones(10, dtype=np.float32)
        pcm = pack_pcm16(audio)

        values = np.frombuffer(pcm, dtype=np.int16)
        assert np.all(values == -32767)

    def test_pack_clips_overflow(self):
        """Values > 1.0 are clipped."""
        audio = np.array([2.0, -2.0], dtype=np.float32)
        pcm = pack_pcm16(audio)

        values = np.frombuffer(pcm, dtype=np.int16)
        assert values[0] == 32767
        assert values[1] == -32767


# ============================================================================
# VoiceChannelConfig Tests
# ============================================================================

class TestVoiceChannelConfig:
    """Test VoiceChannelConfig dataclass."""

    def test_create_config(self, voice_channel_config):
        """Create a config with all fields."""
        cfg = voice_channel_config

        assert cfg.id == "test_vc0"
        assert cfg.system_id == "test_system"
        assert cfg.call_id == "call_001"
        assert cfg.recorder_id == "vr0"
        assert cfg.audio_rate == 8000
        assert cfg.output_rate == 48000

    def test_default_rates(self):
        """Default audio rates are set."""
        cfg = VoiceChannelConfig(
            id="vc",
            system_id="sys",
            call_id="call",
            recorder_id="vr",
        )

        assert cfg.audio_rate == 8000
        assert cfg.output_rate == 48000


# ============================================================================
# VoiceChannel Tests
# ============================================================================

class TestVoiceChannel:
    """Test VoiceChannel class."""

    def test_create_channel(self, voice_channel_config):
        """Create a voice channel."""
        channel = VoiceChannel(cfg=voice_channel_config)

        assert channel.id == "test_vc0"
        assert channel.state == "created"
        assert channel.talkgroup_id == 0
        assert channel.source_id is None
        assert not channel.encrypted

    def test_channel_properties(self, voice_channel_config):
        """Test computed properties."""
        channel = VoiceChannel(cfg=voice_channel_config)

        # Duration should be very small initially
        assert channel.duration_seconds >= 0
        assert channel.duration_seconds < 1

        # Silence should be very small initially
        assert channel.silence_seconds >= 0
        assert channel.silence_seconds < 1

        # Not silent initially
        assert not channel.is_silent

    def test_channel_with_metadata(self, voice_channel_config, sample_location):
        """Create channel with full metadata."""
        channel = VoiceChannel(
            cfg=voice_channel_config,
            talkgroup_id=1001,
            talkgroup_name="Test TG",
            source_id=12345678,
            encrypted=False,
            source_location=sample_location,
        )

        assert channel.talkgroup_id == 1001
        assert channel.talkgroup_name == "Test TG"
        assert channel.source_id == 12345678
        assert channel.source_location is not None
        assert channel.source_location.latitude == 47.6062

    def test_build_metadata(self, voice_channel_config, sample_location):
        """Test metadata dictionary building."""
        channel = VoiceChannel(
            cfg=voice_channel_config,
            talkgroup_id=1001,
            talkgroup_name="Test TG",
            source_id=12345678,
            source_location=sample_location,
        )

        metadata = channel._build_metadata()

        assert metadata["streamId"] == "test_vc0"
        assert metadata["systemId"] == "test_system"
        assert metadata["talkgroupId"] == 1001
        assert metadata["talkgroupName"] == "Test TG"
        assert metadata["sourceId"] == 12345678
        assert metadata["sourceLocation"] is not None
        assert metadata["sourceLocation"]["latitude"] == 47.6062
        assert "timestamp" in metadata

    def test_build_metadata_no_location(self, voice_channel_config):
        """Metadata without location."""
        channel = VoiceChannel(cfg=voice_channel_config)

        metadata = channel._build_metadata()

        assert metadata["sourceLocation"] is None

    def test_pack_json_message(self, voice_channel_config, sample_audio):
        """Test JSON message packing."""
        channel = VoiceChannel(
            cfg=voice_channel_config,
            talkgroup_id=1001,
            talkgroup_name="Test TG",
        )

        pcm_data = pack_pcm16(sample_audio[:100])  # 100 samples
        metadata = channel._build_metadata()
        message = channel._pack_json_message(pcm_data, metadata)

        # Parse and verify
        parsed = json.loads(message.decode("utf-8"))

        assert parsed["type"] == "audio"
        assert parsed["streamId"] == "test_vc0"
        assert parsed["talkgroupId"] == 1001
        assert parsed["format"] == "pcm16"
        assert "audio" in parsed

        # Verify base64 audio
        audio_bytes = base64.b64decode(parsed["audio"])
        assert len(audio_bytes) == len(pcm_data)

    def test_get_stats(self, voice_channel_config):
        """Test statistics retrieval."""
        channel = VoiceChannel(
            cfg=voice_channel_config,
            talkgroup_id=1001,
        )
        channel.audio_frame_count = 10
        channel.audio_bytes_sent = 1000

        stats = channel.get_stats()

        assert stats["id"] == "test_vc0"
        assert stats["state"] == "created"
        assert stats["talkgroupId"] == 1001
        assert stats["audioFrameCount"] == 10
        assert stats["audioBytesSent"] == 1000
        assert "durationSeconds" in stats
        assert "silenceSeconds" in stats

    def test_to_dict(self, voice_channel_config):
        """Test dict conversion matches get_stats structure."""
        channel = VoiceChannel(cfg=voice_channel_config)

        d = channel.to_dict()

        # Verify key fields are present (durationSeconds/silenceSeconds change dynamically)
        assert d["id"] == "test_vc0"
        assert d["systemId"] == "test_system"
        assert d["callId"] == "call_001"
        assert d["state"] == "created"
        assert "durationSeconds" in d
        assert "silenceSeconds" in d


# ============================================================================
# VoiceChannel Async Tests
# ============================================================================

@pytest.mark.anyio
class TestVoiceChannelAsync:
    """Async tests for VoiceChannel."""

    async def test_subscribe_audio(self, voice_channel_config):
        """Subscribe to audio stream."""
        channel = VoiceChannel(cfg=voice_channel_config)

        queue = await channel.subscribe_audio("json")

        assert queue is not None
        assert len(channel._audio_sinks) == 1

    async def test_subscribe_multiple(self, voice_channel_config):
        """Subscribe multiple listeners."""
        channel = VoiceChannel(cfg=voice_channel_config)

        q1 = await channel.subscribe_audio("json")
        q2 = await channel.subscribe_audio("pcm16")
        q3 = await channel.subscribe_audio("f32")

        assert len(channel._audio_sinks) == 3

    async def test_unsubscribe(self, voice_channel_config):
        """Unsubscribe from audio stream."""
        channel = VoiceChannel(cfg=voice_channel_config)

        queue = await channel.subscribe_audio("json")
        assert len(channel._audio_sinks) == 1

        channel.unsubscribe(queue)
        assert len(channel._audio_sinks) == 0

    async def test_unsubscribe_unknown_queue(self, voice_channel_config):
        """Unsubscribe unknown queue is safe."""
        channel = VoiceChannel(cfg=voice_channel_config)

        unknown_queue: asyncio.Queue = asyncio.Queue()
        channel.unsubscribe(unknown_queue)  # Should not raise

        assert len(channel._audio_sinks) == 0

    async def test_broadcast_json(self, voice_channel_config, sample_audio):
        """Broadcast audio to JSON subscriber."""
        channel = VoiceChannel(
            cfg=voice_channel_config,
            talkgroup_id=1001,
        )
        channel.state = "active"

        queue = await channel.subscribe_audio("json")

        # Broadcast audio
        await channel._broadcast(sample_audio[:100])

        # Check queue received message
        assert not queue.empty()
        message = await queue.get()

        parsed = json.loads(message.decode("utf-8"))
        assert parsed["type"] == "audio"
        assert parsed["talkgroupId"] == 1001

    async def test_broadcast_pcm16(self, voice_channel_config, sample_audio):
        """Broadcast audio to PCM16 subscriber."""
        channel = VoiceChannel(cfg=voice_channel_config)
        channel.state = "active"

        queue = await channel.subscribe_audio("pcm16")

        await channel._broadcast(sample_audio[:100])

        assert not queue.empty()
        data = await queue.get()

        # Should be raw PCM bytes
        assert len(data) == 200  # 100 samples * 2 bytes

    async def test_broadcast_f32(self, voice_channel_config, sample_audio):
        """Broadcast audio to float32 subscriber."""
        channel = VoiceChannel(cfg=voice_channel_config)
        channel.state = "active"

        queue = await channel.subscribe_audio("f32")

        await channel._broadcast(sample_audio[:100])

        assert not queue.empty()
        data = await queue.get()

        # Should be raw float32 bytes
        assert len(data) == 400  # 100 samples * 4 bytes

    async def test_broadcast_updates_stats(self, voice_channel_config, sample_audio):
        """Broadcast updates statistics."""
        channel = VoiceChannel(cfg=voice_channel_config)
        channel.state = "active"

        await channel.subscribe_audio("pcm16")

        initial_sent = channel.audio_bytes_sent
        await channel._broadcast(sample_audio[:100])

        assert channel.audio_bytes_sent > initial_sent

    async def test_broadcast_no_subscribers(self, voice_channel_config, sample_audio):
        """Broadcast with no subscribers is safe."""
        channel = VoiceChannel(cfg=voice_channel_config)
        channel.state = "active"

        # No subscribers
        await channel._broadcast(sample_audio[:100])  # Should not raise

    async def test_stop_clears_subscribers(self, voice_channel_config):
        """Stop clears all subscribers."""
        channel = VoiceChannel(cfg=voice_channel_config)

        await channel.subscribe_audio("json")
        await channel.subscribe_audio("pcm16")

        assert len(channel._audio_sinks) == 2

        await channel.stop()

        assert len(channel._audio_sinks) == 0
        assert channel.state == "ended"


# ============================================================================
# VoiceChannelPool Tests
# ============================================================================

class TestVoiceChannelPool:
    """Test VoiceChannelPool class."""

    def test_create_pool(self):
        """Create a pool with default settings."""
        pool = VoiceChannelPool(system_id="test_sys")

        assert pool.system_id == "test_sys"
        assert pool.max_channels == 10
        assert len(pool._available_ids) == 10

    def test_create_pool_custom_size(self):
        """Create pool with custom size."""
        pool = VoiceChannelPool(system_id="test_sys", max_channels=5)

        assert pool.max_channels == 5
        assert len(pool._available_ids) == 5

    def test_get_available_channel_id(self):
        """Get available channel ID."""
        pool = VoiceChannelPool(system_id="test_sys", max_channels=3)

        id1 = pool.get_available_channel_id()
        id2 = pool.get_available_channel_id()
        id3 = pool.get_available_channel_id()
        id4 = pool.get_available_channel_id()  # Should be None

        assert id1 == "test_sys_vc0"
        assert id2 == "test_sys_vc1"
        assert id3 == "test_sys_vc2"
        assert id4 is None  # Pool exhausted

    def test_return_channel_id(self):
        """Return channel ID to pool."""
        pool = VoiceChannelPool(system_id="test_sys", max_channels=2)

        id1 = pool.get_available_channel_id()
        id2 = pool.get_available_channel_id()

        assert len(pool._available_ids) == 0

        pool.return_channel_id(id1)

        assert len(pool._available_ids) == 1
        assert pool._available_ids[0] == id1

    def test_return_duplicate_id_ignored(self):
        """Returning same ID twice doesn't duplicate."""
        pool = VoiceChannelPool(system_id="test_sys", max_channels=2)

        id1 = pool.get_available_channel_id()
        pool.return_channel_id(id1)
        pool.return_channel_id(id1)  # Should be ignored

        assert pool._available_ids.count(id1) == 1

    def test_add_and_get_channel(self):
        """Add and retrieve channel."""
        pool = VoiceChannelPool(system_id="test_sys")

        cfg = VoiceChannelConfig(
            id="test_sys_vc0",
            system_id="test_sys",
            call_id="call_001",
            recorder_id="vr0",
        )
        channel = VoiceChannel(cfg=cfg)

        pool.add_channel(channel)

        retrieved = pool.get_channel("test_sys_vc0")
        assert retrieved is channel

    def test_get_nonexistent_channel(self):
        """Get nonexistent channel returns None."""
        pool = VoiceChannelPool(system_id="test_sys")

        assert pool.get_channel("nonexistent") is None

    def test_remove_channel(self):
        """Remove channel from pool."""
        pool = VoiceChannelPool(system_id="test_sys")

        cfg = VoiceChannelConfig(
            id="test_sys_vc0",
            system_id="test_sys",
            call_id="call_001",
            recorder_id="vr0",
        )
        channel = VoiceChannel(cfg=cfg)
        pool.add_channel(channel)

        # Remove it
        removed = pool.remove_channel("test_sys_vc0")

        assert removed is channel
        assert pool.get_channel("test_sys_vc0") is None
        # ID should be returned to pool
        assert "test_sys_vc0" in pool._available_ids

    def test_remove_nonexistent_channel(self):
        """Remove nonexistent channel returns None."""
        pool = VoiceChannelPool(system_id="test_sys")

        removed = pool.remove_channel("nonexistent")
        assert removed is None

    def test_list_channels(self):
        """List all channels in pool."""
        pool = VoiceChannelPool(system_id="test_sys")

        # Add two channels
        for i in range(2):
            cfg = VoiceChannelConfig(
                id=f"test_sys_vc{i}",
                system_id="test_sys",
                call_id=f"call_{i}",
                recorder_id=f"vr{i}",
            )
            channel = VoiceChannel(cfg=cfg)
            pool.add_channel(channel)

        channels = pool.list_channels()
        assert len(channels) == 2

    def test_list_active_channels(self):
        """List only active channels."""
        pool = VoiceChannelPool(system_id="test_sys")

        # Add two channels, one active, one not
        for i in range(2):
            cfg = VoiceChannelConfig(
                id=f"test_sys_vc{i}",
                system_id="test_sys",
                call_id=f"call_{i}",
                recorder_id=f"vr{i}",
            )
            channel = VoiceChannel(cfg=cfg)
            if i == 0:
                channel.state = "active"
            pool.add_channel(channel)

        active = pool.list_active_channels()
        assert len(active) == 1
        assert active[0].id == "test_sys_vc0"

    def test_list_silent_channels(self):
        """List channels that exceeded silence timeout."""
        pool = VoiceChannelPool(system_id="test_sys", silence_timeout=1.0)

        cfg = VoiceChannelConfig(
            id="test_sys_vc0",
            system_id="test_sys",
            call_id="call_0",
            recorder_id="vr0",
        )
        channel = VoiceChannel(cfg=cfg)
        channel.last_audio_time = time.time() - 10  # 10 seconds ago
        channel.silence_timeout = 1.0
        pool.add_channel(channel)

        silent = pool.list_silent_channels()
        assert len(silent) == 1

    def test_get_stats(self):
        """Get pool statistics."""
        pool = VoiceChannelPool(system_id="test_sys", max_channels=5)

        cfg = VoiceChannelConfig(
            id="test_sys_vc0",
            system_id="test_sys",
            call_id="call_0",
            recorder_id="vr0",
        )
        channel = VoiceChannel(cfg=cfg)
        pool.get_available_channel_id()  # Take one ID
        pool.add_channel(channel)

        stats = pool.get_stats()

        assert stats["systemId"] == "test_sys"
        assert stats["maxChannels"] == 5
        assert stats["activeChannels"] == 1
        assert stats["availableIds"] == 4
        assert len(stats["channels"]) == 1


@pytest.mark.anyio
class TestVoiceChannelPoolAsync:
    """Async tests for VoiceChannelPool."""

    async def test_cleanup_silent_channels(self):
        """Cleanup removes silent channels."""
        pool = VoiceChannelPool(system_id="test_sys", silence_timeout=1.0)

        # Add two channels - one silent, one not
        for i in range(2):
            cfg = VoiceChannelConfig(
                id=f"test_sys_vc{i}",
                system_id="test_sys",
                call_id=f"call_{i}",
                recorder_id=f"vr{i}",
            )
            channel = VoiceChannel(cfg=cfg)
            channel.silence_timeout = 1.0
            if i == 0:
                channel.last_audio_time = time.time() - 10  # Silent
            pool.add_channel(channel)

        removed_count = await pool.cleanup_silent_channels()

        assert removed_count == 1
        assert len(pool.list_channels()) == 1
        assert pool.list_channels()[0].id == "test_sys_vc1"


# ============================================================================
# RadioLocation Tests (from voice_channel module)
# ============================================================================

class TestVoiceChannelRadioLocation:
    """Test RadioLocation as used in voice_channel module."""

    def test_create_and_validate(self):
        """Create and validate location."""
        loc = RadioLocation(
            unit_id=12345,
            latitude=47.6,
            longitude=-122.3,
        )

        assert loc.is_valid()

    def test_to_dict_for_metadata(self):
        """Location serializes for metadata."""
        loc = RadioLocation(
            unit_id=12345,
            latitude=47.6,
            longitude=-122.3,
            source="elc",
        )

        d = loc.to_dict()

        assert d["unitId"] == 12345
        assert d["latitude"] == 47.6
        assert d["longitude"] == -122.3
        assert d["source"] == "elc"
