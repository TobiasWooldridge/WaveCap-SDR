"""Tests for SDRTrunk-inspired patterns.

Tests the new patterns adapted from SDRTrunk for WaveCap-SDR:
- IdentifierCollection: Flexible metadata management
- P25EventTracker: Call state machine with staleness detection
- P25NetworkConfigurationMonitor: System configuration tracking
- DuplicateCallDetector: Duplicate event suppression

Reference: https://github.com/DSheirer/sdrtrunk
"""

import time
import pytest

from wavecapsdr.trunking.identifiers import (
    Identifier,
    IdentifierCollection,
    MutableIdentifierCollection,
    IdentifierRole,
    IdentifierForm,
    TalkerAliasManager,
)
from wavecapsdr.trunking.event_tracker import (
    P25CallEvent,
    P25EventTracker,
    P25EventTrackerManager,
    CallEventType,
    CallEventState,
)
from wavecapsdr.trunking.network_config import (
    P25NetworkConfigurationMonitor,
    FrequencyBand,
    SiteStatus,
    NetworkStatus,
    AdjacentSite,
    SystemServices,
)
from wavecapsdr.trunking.duplicate_detector import (
    DuplicateCallDetector,
    FrequencyBasedDuplicateDetector,
    CallEventSignature,
)


# ============================================================================
# Identifier Tests
# ============================================================================

class TestIdentifier:
    """Test Identifier class."""

    def test_create_identifier(self):
        """Create identifier with value, role, form."""
        ident = Identifier(12345, IdentifierRole.FROM, IdentifierForm.RADIO)

        assert ident.value == 12345
        assert ident.role == IdentifierRole.FROM
        assert ident.form == IdentifierForm.RADIO

    def test_identifier_hash_equality(self):
        """Identifiers with same value/role/form are equal."""
        id1 = Identifier(12345, IdentifierRole.FROM, IdentifierForm.RADIO)
        id2 = Identifier(12345, IdentifierRole.FROM, IdentifierForm.RADIO)
        id3 = Identifier(12345, IdentifierRole.TO, IdentifierForm.RADIO)

        assert id1 == id2
        assert hash(id1) == hash(id2)
        assert id1 != id3

    def test_identifier_immutable(self):
        """Identifier is immutable (frozen dataclass)."""
        ident = Identifier(12345, IdentifierRole.FROM, IdentifierForm.RADIO)

        with pytest.raises(AttributeError):
            ident.value = 99999


class TestIdentifierCollection:
    """Test IdentifierCollection class."""

    def test_create_empty_collection(self):
        """Create empty collection."""
        ic = IdentifierCollection()

        assert len(ic) == 0
        assert ic.get_from_identifier() is None
        assert ic.get_to_identifier() is None

    def test_create_with_identifiers(self):
        """Create collection with identifiers."""
        identifiers = [
            Identifier(12345, IdentifierRole.FROM, IdentifierForm.RADIO),
            Identifier(1001, IdentifierRole.TO, IdentifierForm.TALKGROUP),
        ]
        ic = IdentifierCollection(identifiers)

        assert len(ic) == 2
        assert ic.get_from_identifier().value == 12345
        assert ic.get_to_identifier().value == 1001

    def test_get_radio_id(self):
        """Get radio ID from collection."""
        ic = IdentifierCollection([
            Identifier(12345, IdentifierRole.FROM, IdentifierForm.RADIO),
        ])

        assert ic.get_radio_id() == 12345

    def test_get_talkgroup_id(self):
        """Get talkgroup ID from collection."""
        ic = IdentifierCollection([
            Identifier(1001, IdentifierRole.TO, IdentifierForm.TALKGROUP),
        ])

        assert ic.get_talkgroup_id() == 1001

    def test_has_identifier(self):
        """Check if collection contains identifier."""
        ident = Identifier(12345, IdentifierRole.FROM, IdentifierForm.RADIO)
        ic = IdentifierCollection([ident])

        assert ic.has_identifier(ident)
        assert not ic.has_identifier(
            Identifier(99999, IdentifierRole.FROM, IdentifierForm.RADIO)
        )

    def test_has_encryption(self):
        """Check for encryption identifier."""
        ic_no_enc = IdentifierCollection([
            Identifier(12345, IdentifierRole.FROM, IdentifierForm.RADIO),
        ])
        ic_enc = IdentifierCollection([
            Identifier(12345, IdentifierRole.FROM, IdentifierForm.RADIO),
            Identifier(0x0001, IdentifierRole.ANY, IdentifierForm.ENCRYPTION),
        ])

        assert not ic_no_enc.has_encryption()
        assert ic_enc.has_encryption()

    def test_to_dict(self):
        """Convert collection to dict."""
        ic = IdentifierCollection([
            Identifier(12345, IdentifierRole.FROM, IdentifierForm.RADIO),
            Identifier(1001, IdentifierRole.TO, IdentifierForm.TALKGROUP),
        ], timeslot=1)

        d = ic.to_dict()

        assert d["timeslot"] == 1
        assert d["from"]["value"] == 12345
        assert d["to"]["value"] == 1001
        assert d["identifierCount"] == 2


class TestMutableIdentifierCollection:
    """Test MutableIdentifierCollection class."""

    def test_update_identifier(self):
        """Update adds identifier to collection."""
        mic = MutableIdentifierCollection()
        mic.update(Identifier(12345, IdentifierRole.FROM, IdentifierForm.RADIO))

        assert len(mic) == 1
        assert mic.get_radio_id() == 12345

    def test_update_replaces_same_role_form(self):
        """Update replaces existing identifier with same role+form."""
        mic = MutableIdentifierCollection()
        mic.update(Identifier(12345, IdentifierRole.FROM, IdentifierForm.RADIO))
        mic.update(Identifier(67890, IdentifierRole.FROM, IdentifierForm.RADIO))

        assert len(mic) == 1
        assert mic.get_radio_id() == 67890

    def test_remove_identifier(self):
        """Remove identifier from collection."""
        ident = Identifier(12345, IdentifierRole.FROM, IdentifierForm.RADIO)
        mic = MutableIdentifierCollection()
        mic.update(ident)

        assert mic.remove(ident)
        assert len(mic) == 0

    def test_clear_collection(self):
        """Clear all identifiers."""
        mic = MutableIdentifierCollection()
        mic.update(Identifier(12345, IdentifierRole.FROM, IdentifierForm.RADIO))
        mic.update(Identifier(1001, IdentifierRole.TO, IdentifierForm.TALKGROUP))

        mic.clear()
        assert len(mic) == 0

    def test_to_immutable(self):
        """Convert to immutable collection."""
        mic = MutableIdentifierCollection()
        mic.update(Identifier(12345, IdentifierRole.FROM, IdentifierForm.RADIO))

        ic = mic.to_immutable()

        assert isinstance(ic, IdentifierCollection)
        assert not isinstance(ic, MutableIdentifierCollection)
        assert ic.get_radio_id() == 12345


class TestTalkerAliasManager:
    """Test TalkerAliasManager class."""

    def test_update_and_get_alias(self):
        """Update and retrieve alias."""
        manager = TalkerAliasManager()
        manager.update_alias(12345, "Engine 1")

        assert manager.get_alias(12345) == "Engine 1"
        assert manager.get_alias(99999) is None

    def test_update_talkgroup_alias(self):
        """Update and retrieve talkgroup alias."""
        manager = TalkerAliasManager()
        manager.update_talkgroup_alias(1001, "Fire Dispatch")

        assert manager.get_talkgroup_alias(1001) == "Fire Dispatch"

    def test_enrich_collection(self):
        """Enrich collection with cached aliases."""
        manager = TalkerAliasManager()
        manager.update_alias(12345, "Engine 1")
        manager.update_talkgroup_alias(1001, "Fire Dispatch")

        ic = IdentifierCollection([
            Identifier(12345, IdentifierRole.FROM, IdentifierForm.RADIO),
            Identifier(1001, IdentifierRole.TO, IdentifierForm.TALKGROUP),
        ])

        enriched = manager.enrich(ic)

        assert enriched.get_alias(IdentifierRole.FROM) == "Engine 1"
        assert enriched.get_alias(IdentifierRole.TO) == "Fire Dispatch"

    def test_enrich_no_aliases(self):
        """Enrich returns original if no aliases found."""
        manager = TalkerAliasManager()

        ic = IdentifierCollection([
            Identifier(12345, IdentifierRole.FROM, IdentifierForm.RADIO),
        ])

        enriched = manager.enrich(ic)
        assert enriched is ic  # Same object

    def test_load_from_config(self):
        """Bulk load aliases from config."""
        manager = TalkerAliasManager()
        manager.load_from_config(
            radio_aliases={12345: "Engine 1", 12346: "Engine 2"},
            talkgroup_aliases={1001: "Fire Dispatch", 1002: "EMS"}
        )

        assert manager.get_alias(12345) == "Engine 1"
        assert manager.get_talkgroup_alias(1002) == "EMS"


# ============================================================================
# Event Tracker Tests
# ============================================================================

class TestP25CallEvent:
    """Test P25CallEvent class."""

    def test_create_event(self):
        """Create call event."""
        event = P25CallEvent(
            event_type=CallEventType.GROUP_VOICE,
            frequency_hz=851_000_000,
            channel=0x1234,
        )

        assert event.event_type == CallEventType.GROUP_VOICE
        assert event.frequency_hz == 851_000_000
        assert event.time_start > 0

    def test_event_duration(self):
        """Event tracks duration."""
        event = P25CallEvent(
            event_type=CallEventType.GROUP_VOICE,
            frequency_hz=851e6,
            channel=0x1234,
        )
        event.time_start = 1000
        event.update(2000)

        assert event.duration_ms == 1000

    def test_event_to_dict(self):
        """Convert event to dict."""
        identifiers = MutableIdentifierCollection()
        identifiers.update(Identifier(12345, IdentifierRole.FROM, IdentifierForm.RADIO))
        identifiers.update(Identifier(1001, IdentifierRole.TO, IdentifierForm.TALKGROUP))

        event = P25CallEvent(
            event_type=CallEventType.GROUP_VOICE,
            frequency_hz=851e6,
            channel=0x1234,
            identifiers=identifiers.to_immutable(),
            encrypted=True,
        )

        d = event.to_dict()

        assert d["eventType"] == "group_voice"
        assert d["talkgroupId"] == 1001
        assert d["sourceId"] == 12345
        assert d["encrypted"] is True


class TestP25EventTracker:
    """Test P25EventTracker class."""

    def test_tracker_initial_state(self):
        """Tracker starts in pending state."""
        event = P25CallEvent(
            event_type=CallEventType.GROUP_VOICE,
            frequency_hz=851e6,
            channel=0x1234,
        )
        tracker = P25EventTracker(event)

        assert not tracker.is_started
        assert not tracker.is_complete
        assert tracker.get_state() == CallEventState.PENDING

    def test_update_from_control_channel(self):
        """Control channel updates accepted before start."""
        event = P25CallEvent(
            event_type=CallEventType.GROUP_VOICE,
            frequency_hz=851e6,
            channel=0x1234,
        )
        tracker = P25EventTracker(event)

        assert tracker.update_from_control_channel(1000)
        assert event.time_end == 1000

    def test_traffic_channel_takes_priority(self):
        """Traffic channel updates take priority over control channel."""
        event = P25CallEvent(
            event_type=CallEventType.GROUP_VOICE,
            frequency_hz=851e6,
            channel=0x1234,
        )
        tracker = P25EventTracker(event)

        # Use current time for realistic test
        now_ms = time.time() * 1000

        # Traffic channel starts
        tracker.update_from_traffic_channel(now_ms)
        assert tracker.is_started
        assert tracker.get_state() == CallEventState.ACTIVE

        # Control channel update now rejected
        assert not tracker.update_from_control_channel(now_ms + 100)

    def test_staleness_detection(self):
        """Tracker detects stale events."""
        event = P25CallEvent(
            event_type=CallEventType.GROUP_VOICE,
            frequency_hz=851e6,
            channel=0x1234,
        )
        event.time_start = 1000
        event.time_end = 1000

        tracker = P25EventTracker(event)

        # Not stale within threshold
        assert not tracker.is_stale(2000)

        # Stale after threshold (2000ms)
        assert tracker.is_stale(4000)

    def test_complete_event(self):
        """Complete event marks it as finished."""
        event = P25CallEvent(
            event_type=CallEventType.GROUP_VOICE,
            frequency_hz=851e6,
            channel=0x1234,
        )
        tracker = P25EventTracker(event)

        assert tracker.complete(5000)
        assert tracker.is_complete
        assert tracker.get_state() == CallEventState.COMPLETE

        # Can't complete twice
        assert not tracker.complete(6000)

    def test_same_call_detection(self):
        """Detect same call based on talkgroup and staleness."""
        identifiers = MutableIdentifierCollection()
        identifiers.update(Identifier(1001, IdentifierRole.TO, IdentifierForm.TALKGROUP))

        event = P25CallEvent(
            event_type=CallEventType.GROUP_VOICE,
            frequency_hz=851e6,
            channel=0x1234,
            identifiers=identifiers.to_immutable(),
        )
        event.time_start = 1000
        event.time_end = 1000

        tracker = P25EventTracker(event)

        # Same talkgroup, not stale
        assert tracker.is_same_call(1001, 2000)

        # Different talkgroup
        assert not tracker.is_same_call(1002, 2000)

        # Same talkgroup but stale
        assert not tracker.is_same_call(1001, 5000)


class TestP25EventTrackerManager:
    """Test P25EventTrackerManager class."""

    def test_process_voice_grant(self):
        """Process voice grant creates tracker."""
        manager = P25EventTrackerManager()

        tracker = manager.process_voice_grant(
            frequency_hz=851e6,
            channel=0x1234,
            talkgroup_id=1001,
            source_id=12345,
        )

        assert tracker is not None
        assert tracker.event.talkgroup_id == 1001
        assert tracker.event.source_id == 12345

    def test_duplicate_grant_reuses_tracker(self):
        """Duplicate grants within window reuse tracker."""
        manager = P25EventTrackerManager()

        tracker1 = manager.process_voice_grant(
            frequency_hz=851e6,
            channel=0x1234,
            talkgroup_id=1001,
            source_id=12345,
        )

        # Same grant immediately after
        tracker2 = manager.process_voice_grant(
            frequency_hz=851e6,
            channel=0x1234,
            talkgroup_id=1001,
            source_id=12345,
        )

        assert tracker2 is tracker1  # Same tracker

    def test_traffic_update(self):
        """Traffic update marks tracker as started."""
        manager = P25EventTrackerManager()

        manager.process_voice_grant(
            frequency_hz=851e6,
            channel=0x1234,
            talkgroup_id=1001,
            source_id=12345,
        )

        tracker = manager.process_traffic_update(frequency_hz=851e6, frame_count=10)

        assert tracker is not None
        assert tracker.is_started
        assert tracker.event.frame_count == 10

    def test_call_termination(self):
        """Process call termination."""
        manager = P25EventTrackerManager()

        manager.process_voice_grant(
            frequency_hz=851e6,
            channel=0x1234,
            talkgroup_id=1001,
            source_id=12345,
        )

        event = manager.process_call_termination(frequency_hz=851e6)

        assert event is not None
        assert event.talkgroup_id == 1001

    def test_cleanup_stale(self):
        """Cleanup stale events."""
        manager = P25EventTrackerManager()

        # Create event with old timestamp
        tracker = manager.process_voice_grant(
            frequency_hz=851e6,
            channel=0x1234,
            talkgroup_id=1001,
            source_id=12345,
        )
        tracker.event.time_start = 1000
        tracker.event.time_end = 1000

        # Cleanup with current time
        completed = manager.cleanup_stale()

        assert len(completed) == 1
        assert completed[0].talkgroup_id == 1001


# ============================================================================
# Network Config Tests
# ============================================================================

class TestFrequencyBand:
    """Test FrequencyBand class."""

    def test_downlink_frequency(self):
        """Calculate downlink frequency."""
        band = FrequencyBand(
            identifier=1,
            base_frequency_hz=851_000_000,
            channel_spacing_hz=12_500,
        )

        # Channel 0 = base frequency
        assert band.get_downlink_frequency(0) == 851_000_000

        # Channel 100 = base + 100 * spacing
        assert band.get_downlink_frequency(100) == 852_250_000

    def test_uplink_frequency(self):
        """Calculate uplink frequency with offset."""
        band = FrequencyBand(
            identifier=1,
            base_frequency_hz=851_000_000,
            channel_spacing_hz=12_500,
            transmit_offset_hz=45_000_000,  # 45 MHz offset
        )

        assert band.get_uplink_frequency(0) == 896_000_000


class TestP25NetworkConfigurationMonitor:
    """Test P25NetworkConfigurationMonitor class."""

    def test_process_identifier_update(self):
        """Process IDEN_UP message."""
        monitor = P25NetworkConfigurationMonitor()

        band = monitor.process_identifier_update(
            identifier=1,
            base_freq_mhz=851.0,
            channel_spacing_khz=12.5,
        )

        assert band.identifier == 1
        assert band.base_frequency_hz == 851_000_000

    def test_get_frequency(self):
        """Get frequency from channel number."""
        monitor = P25NetworkConfigurationMonitor()

        monitor.process_identifier_update(
            identifier=1,
            base_freq_mhz=851.0,
            channel_spacing_khz=12.5,
        )

        # Channel format: IIII CCCC CCCC CCCC
        # ID=1, channel=100 -> 0x1064
        channel = (1 << 12) | 100

        freq = monitor.get_frequency(channel)
        assert freq == 852_250_000  # 851 MHz + 100 * 12.5 kHz

    def test_get_frequency_unknown_band(self):
        """Unknown band returns None."""
        monitor = P25NetworkConfigurationMonitor()

        freq = monitor.get_frequency((5 << 12) | 100)  # Unknown band 5
        assert freq is None

    def test_process_rfss_status(self):
        """Process RFSS status broadcast."""
        monitor = P25NetworkConfigurationMonitor()

        status = monitor.process_rfss_status(
            system_id=0x123,
            rfss_id=1,
            site_id=1,
            lra=10,
            channel=0x1000,
            service_class=0xFF,
        )

        assert status.system_id == 0x123
        assert status.site_key == "123-1-1"
        assert monitor.system_id == 0x123

    def test_process_adjacent_status(self):
        """Process adjacent status broadcast."""
        monitor = P25NetworkConfigurationMonitor()

        site = monitor.process_adjacent_status(
            system_id=0x123,
            rfss_id=1,
            site_id=2,
            lra=10,
            channel=0x1100,
            service_class=0xFF,
        )

        assert site.site_key == "123-1-2"
        assert len(monitor.get_adjacent_sites()) == 1

    def test_system_info(self):
        """Get comprehensive system info."""
        monitor = P25NetworkConfigurationMonitor()
        monitor.nac = 0x123

        monitor.process_identifier_update(
            identifier=1,
            base_freq_mhz=851.0,
            channel_spacing_khz=12.5,
        )
        monitor.process_rfss_status(
            system_id=0x456,
            rfss_id=1,
            site_id=1,
            lra=10,
            channel=0x1000,
            service_class=0xFF,
        )

        info = monitor.get_system_info()

        assert info["nac"] == 0x123
        assert info["systemId"] == 0x456
        assert info["frequencyBandCount"] == 1
        assert monitor.is_configured()


class TestSystemServices:
    """Test SystemServices class."""

    def test_service_flags(self):
        """Decode service availability flags."""
        services = SystemServices(
            services_available=0xE00000,  # Composite + Data + Voice
            services_supported=0xF80000,
        )

        assert services.has_composite_control
        assert services.has_data_services
        assert services.has_voice_services
        assert not services.has_registration
        assert not services.has_authentication


# ============================================================================
# Duplicate Detector Tests
# ============================================================================

class TestDuplicateCallDetector:
    """Test DuplicateCallDetector class."""

    def test_first_event_not_duplicate(self):
        """First event is never a duplicate."""
        detector = DuplicateCallDetector()

        is_dup = detector.is_duplicate(
            talkgroup_id=1001,
            source_id=12345,
            frequency_hz=851e6,
        )

        assert not is_dup

    def test_immediate_repeat_is_duplicate(self):
        """Same event immediately after is duplicate."""
        detector = DuplicateCallDetector()
        timestamp = time.time() * 1000

        detector.is_duplicate(
            talkgroup_id=1001,
            source_id=12345,
            frequency_hz=851e6,
            timestamp_ms=timestamp,
        )

        is_dup = detector.is_duplicate(
            talkgroup_id=1001,
            source_id=12345,
            frequency_hz=851e6,
            timestamp_ms=timestamp + 100,  # 100ms later
        )

        assert is_dup

    def test_different_talkgroup_not_duplicate(self):
        """Different talkgroup is not a duplicate."""
        detector = DuplicateCallDetector()

        detector.is_duplicate(talkgroup_id=1001, source_id=12345, frequency_hz=851e6)
        is_dup = detector.is_duplicate(talkgroup_id=1002, source_id=12345, frequency_hz=851e6)

        assert not is_dup

    def test_after_window_not_duplicate(self):
        """Event after window is not duplicate."""
        detector = DuplicateCallDetector(duplicate_window_ms=500)
        timestamp = time.time() * 1000

        detector.is_duplicate(
            talkgroup_id=1001,
            source_id=12345,
            frequency_hz=851e6,
            timestamp_ms=timestamp,
        )

        is_dup = detector.is_duplicate(
            talkgroup_id=1001,
            source_id=12345,
            frequency_hz=851e6,
            timestamp_ms=timestamp + 600,  # 600ms later
        )

        assert not is_dup

    def test_stats(self):
        """Get detector statistics."""
        detector = DuplicateCallDetector()

        detector.is_duplicate(talkgroup_id=1001, source_id=12345, frequency_hz=851e6)
        detector.is_duplicate(talkgroup_id=1001, source_id=12345, frequency_hz=851e6)
        detector.is_duplicate(talkgroup_id=1002, source_id=12345, frequency_hz=851e6)

        stats = detector.get_stats()

        assert stats["totalChecked"] == 3
        assert stats["duplicatesDetected"] == 1


class TestFrequencyBasedDuplicateDetector:
    """Test FrequencyBasedDuplicateDetector class."""

    def test_allocate_frequency(self):
        """Allocate frequency to call."""
        detector = FrequencyBasedDuplicateDetector()

        success = detector.allocate(851e6, talkgroup_id=1001, source_id=12345)
        assert success
        assert detector.is_allocated(851e6)

    def test_double_allocation_fails(self):
        """Can't allocate same frequency to different call."""
        detector = FrequencyBasedDuplicateDetector()

        detector.allocate(851e6, talkgroup_id=1001)
        success = detector.allocate(851e6, talkgroup_id=1002)

        assert not success

    def test_same_call_reallocation_succeeds(self):
        """Same call can reallocate (update timestamp)."""
        detector = FrequencyBasedDuplicateDetector()

        detector.allocate(851e6, talkgroup_id=1001)
        success = detector.allocate(851e6, talkgroup_id=1001)

        assert success

    def test_release_frequency(self):
        """Release frequency makes it available again."""
        detector = FrequencyBasedDuplicateDetector()

        detector.allocate(851e6, talkgroup_id=1001)
        detector.release(851e6)

        assert not detector.is_allocated(851e6)

    def test_get_allocation(self):
        """Get allocation info for frequency."""
        detector = FrequencyBasedDuplicateDetector()

        detector.allocate(851e6, talkgroup_id=1001, source_id=12345)

        alloc = detector.get_allocation(851e6)
        assert alloc == (1001, 12345)

    def test_active_allocations(self):
        """Get list of active allocations."""
        detector = FrequencyBasedDuplicateDetector()

        detector.allocate(851e6, talkgroup_id=1001)
        detector.allocate(852e6, talkgroup_id=1002)

        allocations = detector.get_active_allocations()
        assert len(allocations) == 2
