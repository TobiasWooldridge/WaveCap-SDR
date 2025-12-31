"""Error tracking module for real-time error monitoring.

Provides a singleton ErrorTracker that aggregates errors from captures and channels,
computes rolling statistics, and notifies subscribers for real-time updates.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


@dataclass
class ErrorEvent:
    """Represents a single error event."""

    type: str  # "iq_overflow", "audio_drop", "device_retry"
    capture_id: str
    channel_id: str | None
    timestamp: float
    count: int = 1  # For batched events (e.g., multiple overflows in one report)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ErrorStats:
    """Rolling statistics for a specific error type."""

    total_count: int = 0
    events_last_minute: int = 0
    rate_per_second: float = 0.0
    last_event_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total": self.total_count,
            "lastMinute": self.events_last_minute,
            "rate": round(self.rate_per_second, 2),
        }


class ErrorTracker:
    """Singleton for tracking error events with rolling statistics.

    Thread-safe error tracking with:
    - Ring buffer of recent events (last 1000)
    - Rolling window counters (1s, 1m)
    - Subscriber notification for real-time updates
    """

    _instance: ErrorTracker | None = None
    _lock = threading.Lock()

    def __new__(cls) -> ErrorTracker:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init()
            return cls._instance

    def _init(self) -> None:
        """Initialize the tracker (called only once)."""
        self._events: deque[ErrorEvent] = deque(maxlen=1000)
        self._stats: dict[str, ErrorStats] = {}
        self._subscribers: list[Callable[[ErrorEvent], None]] = []
        self._events_lock = threading.Lock()
        self._subscribers_lock = threading.Lock()

        # Rate calculation window
        self._rate_window: dict[
            str, deque[tuple[float, int]]
        ] = {}  # type -> deque of (timestamp, count)
        self._rate_window_seconds = 10.0  # Calculate rate over 10 seconds

    def record(self, event: ErrorEvent) -> None:
        """Record an error event and notify subscribers."""
        with self._events_lock:
            self._events.append(event)

            # Update stats
            if event.type not in self._stats:
                self._stats[event.type] = ErrorStats()
                self._rate_window[event.type] = deque(maxlen=100)

            stats = self._stats[event.type]
            stats.total_count += event.count
            stats.last_event_time = event.timestamp

            # Add to rate window
            self._rate_window[event.type].append((event.timestamp, event.count))

            # Calculate rate
            self._update_rate(event.type)

        # Notify subscribers (outside lock to avoid deadlocks)
        with self._subscribers_lock:
            subscribers = list(self._subscribers)

        for callback in subscribers:
            try:
                callback(event)
            except Exception:
                pass  # Don't let subscriber errors affect tracking

    def _update_rate(self, error_type: str) -> None:
        """Update rate calculation for an error type."""
        now = time.time()
        window = self._rate_window.get(error_type)
        if not window:
            return

        # Remove old entries
        cutoff = now - self._rate_window_seconds
        while window and window[0][0] < cutoff:
            window.popleft()

        # Calculate rate
        if window:
            total_count = sum(count for _, count in window)
            elapsed = now - window[0][0] if len(window) > 1 else self._rate_window_seconds
            elapsed = max(elapsed, 1.0)  # Avoid division by zero
            self._stats[error_type].rate_per_second = total_count / elapsed

            # Count events in last minute
            minute_cutoff = now - 60.0
            self._stats[error_type].events_last_minute = sum(
                count for ts, count in window if ts >= minute_cutoff
            )
        else:
            self._stats[error_type].rate_per_second = 0.0
            self._stats[error_type].events_last_minute = 0

    def subscribe(self, callback: Callable[[ErrorEvent], None]) -> Callable[[], None]:
        """Subscribe to error events.

        Args:
            callback: Function to call when an error event occurs.

        Returns:
            Unsubscribe function.
        """
        with self._subscribers_lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._subscribers_lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def get_stats(self, capture_id: str | None = None) -> dict[str, ErrorStats]:
        """Get current error statistics.

        Args:
            capture_id: If provided, filter stats for this capture only.
                       Note: Currently returns global stats; per-capture filtering
                       would require additional tracking.

        Returns:
            Dictionary mapping error type to ErrorStats.
        """
        with self._events_lock:
            # Update rates before returning
            time.time()
            for error_type in list(self._stats.keys()):
                self._update_rate(error_type)

            # Return a copy to avoid race conditions
            return {k: ErrorStats(**asdict(v)) for k, v in self._stats.items()}

    def get_recent_events(self, since: float | None = None, limit: int = 50) -> list[ErrorEvent]:
        """Get recent error events.

        Args:
            since: Only return events after this Unix timestamp.
            limit: Maximum number of events to return.

        Returns:
            List of recent ErrorEvents, newest first.
        """
        with self._events_lock:
            events = list(self._events)

        # Filter by timestamp if requested
        if since is not None:
            events = [e for e in events if e.timestamp > since]

        # Return newest first, limited
        return list(reversed(events))[:limit]

    def clear(self) -> None:
        """Clear all tracked events and stats (for testing)."""
        with self._events_lock:
            self._events.clear()
            self._stats.clear()
            self._rate_window.clear()


def get_error_tracker() -> ErrorTracker:
    """Get the global ErrorTracker singleton."""
    return ErrorTracker()
