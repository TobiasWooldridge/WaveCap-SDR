"""Unit tests for P25 trunking system components.

Tests the trunking controller infrastructure:
- TrunkingSystemConfig and TalkgroupConfig
- TrunkingSystem state machine and call handling
- TrunkingManager lifecycle
- ControlChannelMonitor frame sync
"""

import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import pytest
import numpy as np

from wavecapsdr.trunking import (
    TrunkingManager,
    TrunkingSystem,
    TrunkingSystemState,
    TrunkingSystemConfig,
    TalkgroupConfig,
    TrunkingProtocol,
    ControlChannelState,
    CallState,
    ActiveCall,
    VoiceRecorder,
    ControlChannelMonitor,
    SyncState,
    create_control_monitor,
)
from wavecapsdr.trunking.config import load_talkgroups_csv


class MockCapture:
    """Mock Capture for testing."""

    def __init__(self, capture_id: str):
        self.cfg = MagicMock()
        self.cfg.id = capture_id

    def start(self) -> None:
        """Sync start like the real Capture."""
        pass

    def stop(self) -> None:
        """Sync stop like the real Capture."""
        pass


class MockChannel:
    """Mock Channel for testing."""

    def __init__(self, channel_id: str):
        self.cfg = MagicMock()
        self.cfg.id = channel_id
        self.cfg.offset_hz = 0.0
        self._p25_decoder = None
        self.state = "created"

    def start(self) -> None:
        """Start the channel (sets state to running)."""
        self.state = "running"

    async def process_iq_chunk(self, iq: np.ndarray, sample_rate: int) -> None:
        pass

    def process_iq_chunk_sync(self, iq: np.ndarray, sample_rate: int) -> None:
        """Sync IQ processing (for testing)."""
        pass


class MockCaptureManager:
    """Mock CaptureManager for testing."""

    def __init__(self):
        self._capture_counter = 0
        self._channel_counter = 0

    def create_capture(self, **kwargs) -> MockCapture:
        self._capture_counter += 1
        return MockCapture(f"c{self._capture_counter}")

    def create_channel(self, cid: str, mode: str, offset_hz: float = 0.0) -> MockChannel:
        self._channel_counter += 1
        return MockChannel(f"ch{self._channel_counter}")

    async def delete_capture(self, cid: str) -> None:
        pass


@pytest.fixture
def mock_capture_manager():
    """Fixture for mock CaptureManager."""
    return MockCaptureManager()


class TestTalkgroupConfig:
    """Test TalkgroupConfig dataclass."""

    def test_basic_creation(self):
        """Test basic talkgroup creation."""
        tg = TalkgroupConfig(
            tgid=1217,
            name="Kirkland PD",
            category="Police",
            priority=1,
        )
        assert tg.tgid == 1217
        assert tg.name == "Kirkland PD"
        assert tg.category == "Police"
        assert tg.priority == 1
        assert tg.record is True  # Default
        assert tg.monitor is True  # Default

    def test_auto_alpha_tag(self):
        """Test automatic alpha tag generation."""
        tg = TalkgroupConfig(tgid=1217, name="Kirkland Police")
        assert tg.alpha_tag == "KIRKLAND_P"

    def test_explicit_alpha_tag(self):
        """Test explicit alpha tag."""
        tg = TalkgroupConfig(tgid=1217, name="Kirkland PD", alpha_tag="KPD")
        assert tg.alpha_tag == "KPD"


class TestTrunkingSystemConfig:
    """Test TrunkingSystemConfig dataclass."""

    def test_basic_creation(self):
        """Test basic system config creation."""
        cfg = TrunkingSystemConfig(
            id="psern",
            name="PSERN",
            protocol=TrunkingProtocol.P25_PHASE2,
            control_channels=[851.4625e6, 851.5375e6],
            center_hz=855.5e6,
            sample_rate=8_000_000,
        )
        assert cfg.id == "psern"
        assert cfg.name == "PSERN"
        assert cfg.protocol == TrunkingProtocol.P25_PHASE2
        assert len(cfg.control_channels) == 2
        assert cfg.max_voice_recorders == 4  # Default

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "id": "sagrn",
            "name": "SA-GRN",
            "protocol": "p25_phase1",
            "control_channels": [851.0e6, 851.1e6],
            "center_hz": 855.0e6,
            "sample_rate": 4000000,
            "talkgroups": {
                "100": {"name": "Admin", "category": "Admin", "priority": 5},
                "200": "Fire Dispatch",  # Simple format
            },
        }
        cfg = TrunkingSystemConfig.from_dict(data)
        assert cfg.id == "sagrn"
        assert cfg.protocol == TrunkingProtocol.P25_PHASE1
        assert len(cfg.talkgroups) == 2
        assert cfg.talkgroups[100].name == "Admin"
        assert cfg.talkgroups[200].name == "Fire Dispatch"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        cfg = TrunkingSystemConfig(
            id="test",
            name="Test System",
            protocol=TrunkingProtocol.P25_PHASE1,
            control_channels=[851.0e6],
            talkgroups={
                100: TalkgroupConfig(tgid=100, name="TG 100"),
            },
        )
        d = cfg.to_dict()
        assert d["id"] == "test"
        assert d["protocol"] == "p25_phase1"
        assert 100 in d["talkgroups"]

    def test_talkgroup_methods(self):
        """Test talkgroup query methods."""
        cfg = TrunkingSystemConfig(
            id="test",
            name="Test",
            talkgroups={
                100: TalkgroupConfig(tgid=100, name="High Priority", priority=1, monitor=True, record=True),
                200: TalkgroupConfig(tgid=200, name="Low Priority", priority=10, monitor=False, record=False),
            },
            record_unknown=False,
        )

        # get_talkgroup
        assert cfg.get_talkgroup(100) is not None
        assert cfg.get_talkgroup(999) is None

        # is_talkgroup_monitored
        assert cfg.is_talkgroup_monitored(100) is True
        assert cfg.is_talkgroup_monitored(200) is False
        assert cfg.is_talkgroup_monitored(999) is False  # Unknown, record_unknown=False

        # is_talkgroup_recorded
        assert cfg.is_talkgroup_recorded(100) is True
        assert cfg.is_talkgroup_recorded(200) is False

        # get_talkgroup_priority
        assert cfg.get_talkgroup_priority(100) == 1
        assert cfg.get_talkgroup_priority(200) == 10
        assert cfg.get_talkgroup_priority(999) == 10  # Unknown gets lowest priority


class TestVoiceRecorder:
    """Test VoiceRecorder state machine."""

    def test_initial_state(self):
        """Test recorder initial state."""
        recorder = VoiceRecorder(id="vr0", system_id="test")
        assert recorder.state == "idle"
        assert recorder.is_available() is True

    def test_assign_and_release(self):
        """Test assigning and releasing recorder."""
        recorder = VoiceRecorder(id="vr0", system_id="test")

        # Assign
        recorder.assign(
            call_id="call1",
            frequency_hz=851_000_000,
            talkgroup_id=1217,
            center_hz=855_500_000,
        )

        assert recorder.state == "tuning"
        assert recorder.call_id == "call1"
        assert recorder.talkgroup_id == 1217
        assert recorder.offset_hz == 851_000_000 - 855_500_000
        assert recorder.is_available() is False

        # Release
        recorder.release()
        assert recorder.state == "idle"
        assert recorder.call_id is None
        assert recorder.is_available() is True


class TestTrunkingSystem:
    """Test TrunkingSystem state machine."""

    def test_initialization(self):
        """Test system initialization."""
        cfg = TrunkingSystemConfig(
            id="test",
            name="Test System",
            control_channels=[851.0e6, 851.1e6],
            max_voice_recorders=4,
        )
        system = TrunkingSystem(cfg=cfg)

        assert system.state == TrunkingSystemState.STOPPED
        assert system.control_channel_state == ControlChannelState.UNLOCKED
        assert len(system._voice_recorders) == 4

    @pytest.mark.anyio
    async def test_start_stop(self, mock_capture_manager):
        """Test system start and stop."""
        cfg = TrunkingSystemConfig(
            id="test",
            name="Test System",
            control_channels=[851.0e6],
            center_hz=851.0e6,
        )
        system = TrunkingSystem(cfg=cfg)

        # Start
        await system.start(mock_capture_manager)
        assert system.state == TrunkingSystemState.SEARCHING
        assert system.control_channel_state == ControlChannelState.SEARCHING
        assert system.control_channel_freq_hz == 851.0e6

        # Stop
        await system.stop()
        assert system.state == TrunkingSystemState.STOPPED

    @pytest.mark.anyio
    async def test_no_control_channels_fails(self, mock_capture_manager):
        """Test that missing control channels causes failure."""
        cfg = TrunkingSystemConfig(
            id="test",
            name="Test System",
            control_channels=[],
        )
        system = TrunkingSystem(cfg=cfg)

        await system.start(mock_capture_manager)
        assert system.state == TrunkingSystemState.FAILED

    def test_to_dict(self):
        """Test serialization to dictionary."""
        cfg = TrunkingSystemConfig(
            id="test",
            name="Test System",
            control_channels=[851.0e6],
        )
        system = TrunkingSystem(cfg=cfg)
        d = system.to_dict()

        assert d["id"] == "test"
        assert d["name"] == "Test System"
        assert d["state"] == "stopped"
        assert "stats" in d

    def test_get_stats(self):
        """Test getting statistics."""
        cfg = TrunkingSystemConfig(id="test", name="Test", max_voice_recorders=4)
        system = TrunkingSystem(cfg=cfg)

        stats = system.get_stats()
        assert stats["tsbk_count"] == 0
        assert stats["grant_count"] == 0
        assert stats["recorders_idle"] == 4


class TestTrunkingManager:
    """Test TrunkingManager lifecycle."""

    @pytest.mark.anyio
    async def test_add_remove_system(self):
        """Test adding and removing systems."""
        manager = TrunkingManager()

        cfg = TrunkingSystemConfig(
            id="test",
            name="Test System",
            control_channels=[851.0e6],
        )

        # Add system
        system = await manager.add_system(cfg)
        assert system.cfg.id == "test"
        assert manager.get_system("test") is not None
        assert len(manager.list_systems()) == 1

        # Remove system
        await manager.remove_system("test")
        assert manager.get_system("test") is None
        assert len(manager.list_systems()) == 0

    @pytest.mark.anyio
    async def test_duplicate_system_fails(self):
        """Test that adding duplicate system fails."""
        manager = TrunkingManager()

        cfg = TrunkingSystemConfig(id="test", name="Test")
        await manager.add_system(cfg)

        with pytest.raises(ValueError, match="already exists"):
            await manager.add_system(cfg)

    @pytest.mark.anyio
    async def test_start_stop(self, mock_capture_manager):
        """Test manager start and stop."""
        manager = TrunkingManager()
        manager.set_capture_manager(mock_capture_manager)

        cfg = TrunkingSystemConfig(
            id="test",
            name="Test System",
            control_channels=[851.0e6],
            center_hz=851.0e6,
        )
        await manager.add_system(cfg)

        # Start manager
        await manager.start()

        # Start system
        await manager.start_system("test")
        system = manager.get_system("test")
        assert system is not None
        assert system.state == TrunkingSystemState.SEARCHING

        # Stop manager (stops all systems)
        await manager.stop()
        assert system.state == TrunkingSystemState.STOPPED

    @pytest.mark.anyio
    async def test_event_subscription(self):
        """Test event subscription."""
        manager = TrunkingManager()

        cfg = TrunkingSystemConfig(id="test", name="Test")
        await manager.add_system(cfg)

        # Subscribe to events
        queue = await manager.subscribe_events()

        # Should receive snapshot immediately
        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert event["type"] == "snapshot"
        assert len(event["systems"]) == 1

        # Unsubscribe
        await manager.unsubscribe_events(queue)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        manager = TrunkingManager()
        d = manager.to_dict()

        assert d["systemCount"] == 0
        assert d["totalActiveCalls"] == 0


class TestControlChannelMonitor:
    """Test ControlChannelMonitor frame sync."""

    def test_initialization_phase1(self):
        """Test Phase I monitor initialization."""
        monitor = create_control_monitor(
            protocol=TrunkingProtocol.P25_PHASE1,
            sample_rate=48000,
        )
        assert monitor.protocol == TrunkingProtocol.P25_PHASE1
        assert monitor.sync_state == SyncState.SEARCHING
        # Control channels always use C4FM, even for Phase I
        assert monitor._c4fm_demod is not None

    def test_initialization_phase2(self):
        """Test Phase II monitor initialization - still uses C4FM for control channel."""
        monitor = create_control_monitor(
            protocol=TrunkingProtocol.P25_PHASE2,
            sample_rate=48000,
        )
        assert monitor.protocol == TrunkingProtocol.P25_PHASE2
        # Control channels ALWAYS use C4FM even for Phase II systems
        # Only voice channels use CQPSK in Phase II
        assert monitor._c4fm_demod is not None

    def test_reset(self):
        """Test monitor reset."""
        monitor = create_control_monitor(TrunkingProtocol.P25_PHASE1)

        # Add some state
        monitor._dibit_buffer = [1, 2, 3, 4]
        monitor.sync_state = SyncState.SYNCED

        # Reset
        monitor.reset()

        assert monitor.sync_state == SyncState.SEARCHING
        assert len(monitor._dibit_buffer) == 0

    def test_get_sync_dibits(self):
        """Test sync pattern conversion."""
        monitor = create_control_monitor(TrunkingProtocol.P25_PHASE1)

        dibits = monitor._get_sync_dibits()
        assert len(dibits) == 24  # 48 bits = 24 dibits

    def test_process_empty_iq(self):
        """Test processing empty IQ."""
        monitor = create_control_monitor(TrunkingProtocol.P25_PHASE1)

        results = monitor.process_iq(np.array([], dtype=np.complex64))
        assert results == []

    def test_get_stats(self):
        """Test getting statistics."""
        monitor = create_control_monitor(TrunkingProtocol.P25_PHASE1)

        stats = monitor.get_stats()
        assert stats["sync_state"] == "searching"
        assert stats["frames_decoded"] == 0
        assert stats["tsbk_decoded"] == 0


class TestActiveCall:
    """Test ActiveCall dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        import time

        call = ActiveCall(
            id="test123",
            talkgroup_id=1217,
            talkgroup_name="Kirkland PD",
            source_id=12345,
            frequency_hz=851_012_500,
            channel_id=0x1234,
            state=CallState.RECORDING,
            start_time=time.time(),
            last_activity_time=time.time(),
            encrypted=False,
        )

        d = call.to_dict()
        assert d["id"] == "test123"
        assert d["talkgroupId"] == 1217
        assert d["talkgroupName"] == "Kirkland PD"
        assert d["state"] == "recording"
        assert d["encrypted"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
