"""Duplicate call event detector.

Suppresses duplicate call events that occur within a time window.
P25 control channels often send multiple grants for the same call,
and this module prevents flooding downstream systems with duplicates.

Inspired by SDRTrunk's DecodeEventDuplicateDetector pattern.

Reference: https://github.com/DSheirer/sdrtrunk
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CallEventSignature:
    """Unique signature for a call event."""
    talkgroup_id: int
    source_id: Optional[int]
    frequency_hz: float
    event_type: str  # "voice", "data", etc.

    def __hash__(self) -> int:
        return hash((self.talkgroup_id, self.source_id,
                    int(self.frequency_hz), self.event_type))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CallEventSignature):
            return False
        return (self.talkgroup_id == other.talkgroup_id and
                self.source_id == other.source_id and
                int(self.frequency_hz) == int(other.frequency_hz) and
                self.event_type == other.event_type)


class DuplicateCallDetector:
    """Detects and suppresses duplicate call events.

    Call events are deduplicated based on:
    - Talkgroup ID
    - Source ID (optional)
    - Frequency
    - Event type

    Events within the duplicate window are suppressed.

    Example:
        detector = DuplicateCallDetector(duplicate_window_ms=500)

        # First event - not duplicate
        is_dup = detector.is_duplicate(tgid=1001, source=12345, freq=851e6, "voice")
        # is_dup = False

        # Same event immediately after - duplicate
        is_dup = detector.is_duplicate(tgid=1001, source=12345, freq=851e6, "voice")
        # is_dup = True

        # Different talkgroup - not duplicate
        is_dup = detector.is_duplicate(tgid=1002, source=12345, freq=851e6, "voice")
        # is_dup = False
    """

    def __init__(self, duplicate_window_ms: float = 500,
                 max_entries: int = 1000) -> None:
        """Initialize detector.

        Args:
            duplicate_window_ms: Time window for duplicate detection (ms)
            max_entries: Maximum number of entries to track
        """
        self.duplicate_window_ms = duplicate_window_ms
        self.max_entries = max_entries

        # signature -> timestamp_ms (ordered for LRU cleanup)
        self._recent_events: OrderedDict[CallEventSignature, float] = OrderedDict()
        self._lock = Lock()

        # Statistics
        self._total_checked = 0
        self._duplicates_detected = 0

    def is_duplicate(self, talkgroup_id: int, source_id: Optional[int],
                    frequency_hz: float, event_type: str = "voice",
                    timestamp_ms: Optional[float] = None) -> bool:
        """Check if event is a duplicate.

        Args:
            talkgroup_id: Talkgroup ID
            source_id: Source radio ID (optional)
            frequency_hz: Frequency in Hz
            event_type: Event type ("voice", "data", etc.)
            timestamp_ms: Event timestamp (default: current time)

        Returns:
            True if this is a duplicate event that should be suppressed
        """
        if timestamp_ms is None:
            timestamp_ms = time.time() * 1000

        sig = CallEventSignature(
            talkgroup_id=talkgroup_id,
            source_id=source_id,
            frequency_hz=frequency_hz,
            event_type=event_type,
        )

        with self._lock:
            self._total_checked += 1

            # Check if we've seen this recently
            if sig in self._recent_events:
                last_seen = self._recent_events[sig]
                if (timestamp_ms - last_seen) < self.duplicate_window_ms:
                    self._duplicates_detected += 1
                    # Update timestamp (extend the window)
                    self._recent_events[sig] = timestamp_ms
                    self._recent_events.move_to_end(sig)
                    return True

            # Not a duplicate - record it
            self._recent_events[sig] = timestamp_ms
            self._recent_events.move_to_end(sig)

            # Cleanup old entries
            self._cleanup(timestamp_ms)

            return False

    def check_duplicate_event(self, event: Dict[str, Any]) -> bool:
        """Check if event dict is a duplicate.

        Convenience method that extracts fields from event dict.
        """
        return self.is_duplicate(
            talkgroup_id=event.get("tgid", event.get("talkgroup_id", 0)),
            source_id=event.get("source_id"),
            frequency_hz=event.get("frequency_hz", 0),
            event_type=event.get("event_type", event.get("type", "voice")),
        )

    def _cleanup(self, timestamp_ms: float) -> None:
        """Remove old entries from cache."""
        cutoff = timestamp_ms - self.duplicate_window_ms

        # Remove entries older than window
        while self._recent_events:
            sig, ts = next(iter(self._recent_events.items()))
            if ts > cutoff:
                break
            self._recent_events.pop(sig)

        # Enforce max entries (LRU eviction)
        while len(self._recent_events) > self.max_entries:
            self._recent_events.popitem(last=False)

    def clear(self) -> None:
        """Clear all tracked events."""
        with self._lock:
            self._recent_events.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        with self._lock:
            return {
                "trackedEvents": len(self._recent_events),
                "totalChecked": self._total_checked,
                "duplicatesDetected": self._duplicates_detected,
                "duplicateRate": (self._duplicates_detected / self._total_checked
                                 if self._total_checked > 0 else 0),
                "duplicateWindowMs": self.duplicate_window_ms,
            }


class FrequencyBasedDuplicateDetector:
    """Detects duplicates based on frequency allocation.

    A more sophisticated detector that tracks which frequencies
    are currently allocated to calls, preventing double-allocation.

    Example:
        detector = FrequencyBasedDuplicateDetector()

        # Allocate frequency to a call
        success = detector.allocate(freq=851e6, tgid=1001, source=12345)
        # success = True

        # Same frequency already allocated - fail
        success = detector.allocate(freq=851e6, tgid=1002, source=67890)
        # success = False

        # Release frequency
        detector.release(freq=851e6)

        # Now available again
        success = detector.allocate(freq=851e6, tgid=1002, source=67890)
        # success = True
    """

    def __init__(self, allocation_timeout_ms: float = 10000) -> None:
        """Initialize detector.

        Args:
            allocation_timeout_ms: Auto-release after this timeout (ms)
        """
        self.allocation_timeout_ms = allocation_timeout_ms

        # frequency -> (tgid, source_id, timestamp_ms)
        self._allocations: Dict[float, Tuple[int, Optional[int], float]] = {}
        self._lock = Lock()

    def is_allocated(self, frequency_hz: float) -> bool:
        """Check if frequency is currently allocated."""
        with self._lock:
            self._cleanup_stale()
            return frequency_hz in self._allocations

    def allocate(self, frequency_hz: float, talkgroup_id: int,
                source_id: Optional[int] = None) -> bool:
        """Allocate frequency to a call.

        Returns True if allocation succeeded, False if already allocated.
        """
        timestamp_ms = time.time() * 1000

        with self._lock:
            self._cleanup_stale()

            if frequency_hz in self._allocations:
                # Already allocated - check if same call
                existing_tgid, existing_src, _ = self._allocations[frequency_hz]
                if existing_tgid == talkgroup_id:
                    # Same call - update timestamp
                    self._allocations[frequency_hz] = (talkgroup_id, source_id,
                                                        timestamp_ms)
                    return True
                return False

            self._allocations[frequency_hz] = (talkgroup_id, source_id, timestamp_ms)
            return True

    def release(self, frequency_hz: float) -> bool:
        """Release frequency allocation.

        Returns True if frequency was allocated and released.
        """
        with self._lock:
            if frequency_hz in self._allocations:
                del self._allocations[frequency_hz]
                return True
            return False

    def get_allocation(self, frequency_hz: float) -> Optional[Tuple[int, Optional[int]]]:
        """Get allocation info for frequency.

        Returns (talkgroup_id, source_id) or None if not allocated.
        """
        with self._lock:
            self._cleanup_stale()
            if frequency_hz in self._allocations:
                tgid, src, _ = self._allocations[frequency_hz]
                return (tgid, src)
            return None

    def update(self, frequency_hz: float) -> bool:
        """Update timestamp for existing allocation.

        Returns True if allocation exists and was updated.
        """
        timestamp_ms = time.time() * 1000

        with self._lock:
            if frequency_hz in self._allocations:
                tgid, src, _ = self._allocations[frequency_hz]
                self._allocations[frequency_hz] = (tgid, src, timestamp_ms)
                return True
            return False

    def _cleanup_stale(self) -> None:
        """Remove stale allocations."""
        timestamp_ms = time.time() * 1000
        cutoff = timestamp_ms - self.allocation_timeout_ms

        stale = [freq for freq, (_, _, ts) in self._allocations.items()
                if ts < cutoff]

        for freq in stale:
            del self._allocations[freq]
            logger.debug(f"Released stale allocation: {freq/1e6:.4f} MHz")

    def clear(self) -> None:
        """Clear all allocations."""
        with self._lock:
            self._allocations.clear()

    def get_active_allocations(self) -> List[Dict[str, Any]]:
        """Get list of active allocations."""
        with self._lock:
            self._cleanup_stale()
            return [
                {
                    "frequencyHz": freq,
                    "frequencyMhz": freq / 1e6,
                    "talkgroupId": tgid,
                    "sourceId": src,
                    "timestampMs": ts,
                }
                for freq, (tgid, src, ts) in self._allocations.items()
            ]

    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        with self._lock:
            self._cleanup_stale()
            return {
                "activeAllocations": len(self._allocations),
                "allocationTimeoutMs": self.allocation_timeout_ms,
            }
