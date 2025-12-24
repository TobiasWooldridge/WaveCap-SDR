"""State broadcaster for real-time capture/channel state updates.

Provides a pub/sub mechanism for pushing state changes to WebSocket clients,
reducing the need for polling.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class StateChange:
    """Represents a state change event."""
    type: str  # "capture" | "channel" | "scanner" | "device"
    action: str  # "created" | "updated" | "deleted" | "started" | "stopped"
    id: str  # Entity ID
    data: dict[str, Any] | None = None  # Full entity data (for created/updated)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "action": self.action,
            "id": self.id,
            "data": self.data,
            "timestamp": self.timestamp,
        }


class StateBroadcaster:
    """Singleton broadcaster for state change events."""

    _instance: StateBroadcaster | None = None
    _lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls) -> StateBroadcaster:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._subscribers: set[Callable[[StateChange], None]] = set()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        logger.info("[STATE] StateBroadcaster initialized")

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the asyncio event loop for thread-safe callbacks."""
        self._loop = loop

    def subscribe(self, callback: Callable[[StateChange], None]) -> Callable[[], None]:
        """Subscribe to state changes. Returns unsubscribe function."""
        with self._lock:
            self._subscribers.add(callback)
            logger.debug(f"[STATE] Subscriber added, total: {len(self._subscribers)}")

        def unsubscribe() -> None:
            with self._lock:
                self._subscribers.discard(callback)
                logger.debug(f"[STATE] Subscriber removed, total: {len(self._subscribers)}")

        return unsubscribe

    def broadcast(self, change: StateChange) -> None:
        """Broadcast a state change to all subscribers.

        Thread-safe: can be called from any thread.
        """
        with self._lock:
            subscribers = list(self._subscribers)

        if not subscribers:
            return

        logger.debug(f"[STATE] Broadcasting {change.type}.{change.action} id={change.id}")

        for callback in subscribers:
            try:
                # If we have an event loop and we're not in it, schedule callback
                if self._loop is not None:
                    try:
                        if threading.current_thread() != threading.main_thread():
                            # Schedule callback on event loop thread
                            self._loop.call_soon_threadsafe(callback, change)
                        else:
                            callback(change)
                    except RuntimeError:
                        # Event loop might be closed
                        callback(change)
                else:
                    callback(change)
            except Exception as e:
                logger.error(f"[STATE] Error in subscriber callback: {e}")

    def emit_capture_change(
        self,
        action: str,
        capture_id: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Emit a capture state change."""
        self.broadcast(StateChange(
            type="capture",
            action=action,
            id=capture_id,
            data=data,
        ))

    def emit_channel_change(
        self,
        action: str,
        channel_id: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Emit a channel state change."""
        self.broadcast(StateChange(
            type="channel",
            action=action,
            id=channel_id,
            data=data,
        ))

    def emit_scanner_change(
        self,
        action: str,
        scanner_id: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Emit a scanner state change."""
        self.broadcast(StateChange(
            type="scanner",
            action=action,
            id=scanner_id,
            data=data,
        ))

    def emit_device_change(
        self,
        action: str,
        device_id: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Emit a device state change."""
        self.broadcast(StateChange(
            type="device",
            action=action,
            id=device_id,
            data=data,
        ))


def get_broadcaster() -> StateBroadcaster:
    """Get the singleton state broadcaster instance."""
    return StateBroadcaster()
