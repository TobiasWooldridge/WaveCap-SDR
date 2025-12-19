"""P25 call event tracker with state machine.

Implements SDRTrunk's P25TrafficChannelEventTracker pattern for managing
call state transitions with proper handling of:
- Stale event detection
- Dual-source timing (control channel vs. traffic channel)
- Talker change detection
- TDMA timeslot tracking

Reference: https://github.com/DSheirer/sdrtrunk
  - P25TrafficChannelEventTracker.java
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from wavecapsdr.trunking.identifiers import (
    Identifier,
    IdentifierCollection,
    IdentifierForm,
    IdentifierRole,
    MutableIdentifierCollection,
)

logger = logging.getLogger(__name__)


class CallEventType(Enum):
    """Type of call event."""
    GROUP_VOICE = "group_voice"
    UNIT_TO_UNIT = "unit_to_unit"
    TELEPHONE_INTERCONNECT = "telephone_interconnect"
    DATA_CALL = "data_call"
    EMERGENCY = "emergency"


class CallEventState(Enum):
    """State of a call event."""
    PENDING = "pending"    # Granted but not yet started on traffic channel
    ACTIVE = "active"      # Active call on traffic channel
    STALE = "stale"        # No recent updates (likely abandoned)
    COMPLETE = "complete"  # Call has ended


@dataclass
class P25CallEvent:
    """P25 call event with full metadata.

    Represents a complete call lifecycle from grant to termination.
    """
    event_type: CallEventType
    frequency_hz: float
    channel: int
    timeslot: int = 0  # 0 for Phase I, 1-2 for Phase II TDMA

    # Identifiers (built up during call)
    identifiers: IdentifierCollection = field(
        default_factory=lambda: IdentifierCollection()
    )

    # Timing
    time_start: float = 0.0
    time_end: float = 0.0

    # Flags
    encrypted: bool = False
    emergency: bool = False
    duplex: bool = False
    priority: int = 0

    # Statistics
    frame_count: int = 0
    error_count: int = 0

    def __post_init__(self) -> None:
        if self.time_start == 0.0:
            self.time_start = time.time() * 1000  # ms

    def update(self, timestamp_ms: float) -> None:
        """Update event with new timestamp."""
        self.time_end = timestamp_ms

    def end(self, timestamp_ms: float) -> None:
        """Mark event as ended."""
        self.time_end = timestamp_ms

    @property
    def duration_ms(self) -> float:
        """Get call duration in milliseconds."""
        if self.time_end > 0:
            return self.time_end - self.time_start
        return time.time() * 1000 - self.time_start

    @property
    def talkgroup_id(self) -> Optional[int]:
        """Get talkgroup ID from identifiers."""
        return self.identifiers.get_talkgroup_id()

    @property
    def source_id(self) -> Optional[int]:
        """Get source radio ID from identifiers."""
        return self.identifiers.get_radio_id()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "eventType": self.event_type.value,
            "frequencyHz": self.frequency_hz,
            "channel": self.channel,
            "timeslot": self.timeslot,
            "talkgroupId": self.talkgroup_id,
            "sourceId": self.source_id,
            "timeStart": self.time_start,
            "timeEnd": self.time_end,
            "durationMs": self.duration_ms,
            "encrypted": self.encrypted,
            "emergency": self.emergency,
            "frameCount": self.frame_count,
            "errorCount": self.error_count,
            "identifiers": self.identifiers.to_dict(),
        }


class P25EventTracker:
    """Tracks P25 call event lifecycle with state management.

    Handles the complexity of P25 call tracking where:
    - Control channel grants the call
    - Traffic channel carries the actual audio
    - Updates can come from either source
    - Calls can be abandoned before traffic channel activates
    - Multiple talkers can use the same call (push-to-talk handoffs)

    Implements SDRTrunk's event tracker pattern with 2-second staleness
    threshold and dual-source timing updates.

    Example:
        tracker = P25EventTracker(event)

        # Control channel sends updates
        tracker.update_from_control_channel(timestamp)

        # Traffic channel starts
        tracker.mark_started()
        tracker.update_from_traffic_channel(timestamp)

        # Check for staleness
        if tracker.is_stale(current_time):
            tracker.complete(current_time)
    """

    # Event is stale if no updates for 2 seconds
    STALE_THRESHOLD_MS = 2000

    # Data channel timeout (shorter due to bursty nature)
    DATA_STALE_THRESHOLD_MS = 500

    def __init__(self, event: P25CallEvent) -> None:
        self.event = event
        self._started = False
        self._complete = False
        self._last_update_ms = event.time_start

    @property
    def is_started(self) -> bool:
        """Check if traffic channel has started."""
        return self._started

    @property
    def is_complete(self) -> bool:
        """Check if call is complete."""
        return self._complete

    def get_state(self) -> CallEventState:
        """Get current event state."""
        if self._complete:
            return CallEventState.COMPLETE
        if self.is_stale(time.time() * 1000):
            return CallEventState.STALE
        if self._started:
            return CallEventState.ACTIVE
        return CallEventState.PENDING

    def is_stale(self, timestamp_ms: float) -> bool:
        """Check if event is stale relative to timestamp.

        An event becomes stale when no updates have been received
        within the threshold period.
        """
        threshold = (self.DATA_STALE_THRESHOLD_MS
                    if self.event.event_type == CallEventType.DATA_CALL
                    else self.STALE_THRESHOLD_MS)

        reference_time = (self.event.time_end if self.event.time_end > 0
                         else self.event.time_start)

        return (timestamp_ms - reference_time) > threshold

    def update_from_control_channel(self, timestamp_ms: float) -> bool:
        """Update event timing from control channel.

        Control channel updates are only accepted before the traffic
        channel starts. Once traffic channel takes over, it has
        priority for timing updates.

        Returns True if update was accepted.
        """
        if self._started:
            return False  # Traffic channel has priority

        if not self._complete:
            self.event.update(timestamp_ms)
            self._last_update_ms = timestamp_ms
            return True
        return False

    def update_from_traffic_channel(self, timestamp_ms: float) -> None:
        """Update event timing from traffic channel.

        Traffic channel updates always accepted (if not complete).
        First traffic channel update marks the event as started.
        """
        if self._complete:
            return

        if not self._started:
            self._started = True
            logger.debug(f"Event tracker: traffic channel started for "
                        f"TG={self.event.talkgroup_id}")

        self.event.update(timestamp_ms)
        self._last_update_ms = timestamp_ms

    def mark_started(self) -> None:
        """Mark traffic channel as started."""
        self._started = True

    def complete(self, timestamp_ms: float) -> bool:
        """Mark event as complete.

        Returns True if event was newly completed.
        """
        if self._complete:
            return False

        self._complete = True
        self.event.end(timestamp_ms)

        logger.debug(f"Event tracker: call complete for "
                    f"TG={self.event.talkgroup_id}, "
                    f"duration={self.event.duration_ms:.0f}ms")
        return True

    def update_identifiers(self, identifiers: IdentifierCollection) -> None:
        """Update event identifiers."""
        # Merge new identifiers
        mic = MutableIdentifierCollection(
            self.event.identifiers.get_identifiers(),
            identifiers.timeslot
        )
        for ident in identifiers.get_identifiers():
            mic.update(ident)
        self.event.identifiers = mic.to_immutable()

    def is_same_call(self, talkgroup_id: int, timestamp_ms: float) -> bool:
        """Check if talkgroup matches and event isn't stale.

        Used to determine if a new grant should reuse this tracker
        or create a new one.
        """
        current_tg = self.event.talkgroup_id
        if current_tg is None:
            return False

        return (current_tg == talkgroup_id and
                not self.is_stale(timestamp_ms) and
                not self._complete)

    def is_different_talker(self, source_id: int) -> bool:
        """Check if source ID differs from current talker.

        Used to detect talker changes within a call (PTT handoffs).
        """
        current_source = self.event.source_id
        if current_source is None:
            return False
        return current_source != source_id


class P25EventTrackerManager:
    """Manages event trackers for a trunking system.

    Handles:
    - Creating trackers for new grants
    - Finding existing trackers for updates
    - Detecting and cleaning up stale events
    - Deduplicating rapid-fire grants for same call
    """

    # Don't create duplicate events within this window
    DUPLICATE_WINDOW_MS = 500

    def __init__(self) -> None:
        # frequency -> tracker
        self._trackers: Dict[float, P25EventTracker] = {}

        # Recent events for deduplication
        # (tgid, freq, timestamp_ms)
        self._recent_grants: List[tuple] = []

        # Callbacks
        self.on_call_start: Optional[Callable[[P25CallEvent], None]] = None
        self.on_call_end: Optional[Callable[[P25CallEvent], None]] = None
        self.on_talker_change: Optional[Callable[[P25CallEvent, int], None]] = None

    def process_voice_grant(self, frequency_hz: float, channel: int,
                           talkgroup_id: int, source_id: int,
                           encrypted: bool = False, emergency: bool = False,
                           timeslot: int = 0) -> Optional[P25EventTracker]:
        """Process a voice channel grant from control channel.

        Creates a new tracker or updates existing one.
        Returns the tracker, or None if this is a duplicate grant.
        """
        timestamp_ms = time.time() * 1000

        # Check for duplicate grant
        if self._is_duplicate_grant(talkgroup_id, frequency_hz, timestamp_ms):
            # Update existing tracker timing
            existing = self._trackers.get(frequency_hz)
            if existing:
                existing.update_from_control_channel(timestamp_ms)
            return existing

        # Check for existing tracker on this frequency
        existing = self._trackers.get(frequency_hz)
        if existing and existing.is_same_call(talkgroup_id, timestamp_ms):
            # Same call, check for talker change
            if source_id and existing.is_different_talker(source_id):
                if self.on_talker_change:
                    self.on_talker_change(existing.event, source_id)
                # Update source ID
                self._update_source(existing, source_id)

            existing.update_from_control_channel(timestamp_ms)
            return existing

        # Complete any existing tracker on this frequency
        if existing and not existing.is_complete:
            existing.complete(timestamp_ms)
            if self.on_call_end:
                self.on_call_end(existing.event)

        # Create new event and tracker
        identifiers = MutableIdentifierCollection()
        identifiers.update(Identifier(talkgroup_id, IdentifierRole.TO,
                                      IdentifierForm.TALKGROUP))
        if source_id:
            identifiers.update(Identifier(source_id, IdentifierRole.FROM,
                                         IdentifierForm.RADIO))

        event = P25CallEvent(
            event_type=CallEventType.EMERGENCY if emergency else CallEventType.GROUP_VOICE,
            frequency_hz=frequency_hz,
            channel=channel,
            timeslot=timeslot,
            identifiers=identifiers.to_immutable(),
            encrypted=encrypted,
            emergency=emergency,
        )

        tracker = P25EventTracker(event)
        self._trackers[frequency_hz] = tracker

        # Record for deduplication
        self._recent_grants.append((talkgroup_id, frequency_hz, timestamp_ms))
        self._cleanup_recent_grants(timestamp_ms)

        if self.on_call_start:
            self.on_call_start(event)

        logger.info(f"New voice grant: TG={talkgroup_id} SRC={source_id} "
                   f"FREQ={frequency_hz/1e6:.4f} MHz")

        return tracker

    def process_traffic_update(self, frequency_hz: float,
                              frame_count: int = 0) -> Optional[P25EventTracker]:
        """Process update from traffic channel.

        Called when audio frames are received on a traffic channel.
        """
        tracker = self._trackers.get(frequency_hz)
        if tracker is None:
            return None

        timestamp_ms = time.time() * 1000
        tracker.update_from_traffic_channel(timestamp_ms)
        tracker.event.frame_count += frame_count

        return tracker

    def process_call_termination(self, frequency_hz: float) -> Optional[P25CallEvent]:
        """Process call termination from traffic channel.

        Returns the completed event if found.
        """
        tracker = self._trackers.get(frequency_hz)
        if tracker is None:
            return None

        timestamp_ms = time.time() * 1000
        if tracker.complete(timestamp_ms):
            if self.on_call_end:
                self.on_call_end(tracker.event)
            return tracker.event
        return None

    def get_tracker(self, frequency_hz: float) -> Optional[P25EventTracker]:
        """Get tracker for frequency."""
        return self._trackers.get(frequency_hz)

    def get_active_calls(self) -> List[P25CallEvent]:
        """Get all active (non-complete) calls."""
        return [
            t.event for t in self._trackers.values()
            if not t.is_complete
        ]

    def cleanup_stale(self) -> List[P25CallEvent]:
        """Clean up stale events.

        Call periodically to detect abandoned calls.
        Returns list of completed events.
        """
        timestamp_ms = time.time() * 1000
        completed = []

        for freq, tracker in list(self._trackers.items()):
            if tracker.is_stale(timestamp_ms) and not tracker.is_complete:
                tracker.complete(timestamp_ms)
                completed.append(tracker.event)

                if self.on_call_end:
                    self.on_call_end(tracker.event)

                logger.debug(f"Cleaned up stale call: TG={tracker.event.talkgroup_id}")

        return completed

    def _is_duplicate_grant(self, talkgroup_id: int, frequency_hz: float,
                           timestamp_ms: float) -> bool:
        """Check if this is a duplicate grant within the window."""
        for tgid, freq, ts in self._recent_grants:
            if (tgid == talkgroup_id and
                freq == frequency_hz and
                (timestamp_ms - ts) < self.DUPLICATE_WINDOW_MS):
                return True
        return False

    def _cleanup_recent_grants(self, timestamp_ms: float) -> None:
        """Remove old entries from recent grants."""
        cutoff = timestamp_ms - self.DUPLICATE_WINDOW_MS
        self._recent_grants = [
            (tgid, freq, ts) for tgid, freq, ts in self._recent_grants
            if ts > cutoff
        ]

    def _update_source(self, tracker: P25EventTracker, source_id: int) -> None:
        """Update source ID in tracker's identifiers."""
        mic = MutableIdentifierCollection(
            tracker.event.identifiers.get_identifiers(),
            tracker.event.identifiers.timeslot
        )
        mic.update(Identifier(source_id, IdentifierRole.FROM, IdentifierForm.RADIO))
        tracker.event.identifiers = mic.to_immutable()

    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        active = sum(1 for t in self._trackers.values() if not t.is_complete)
        pending = sum(1 for t in self._trackers.values()
                     if not t.is_started and not t.is_complete)
        stale = sum(1 for t in self._trackers.values()
                   if t.get_state() == CallEventState.STALE)

        return {
            "totalTrackers": len(self._trackers),
            "activeTrackers": active,
            "pendingTrackers": pending,
            "staleTrackers": stale,
            "recentGrantCount": len(self._recent_grants),
        }
