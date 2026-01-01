"""Real-time log streaming for WebSocket clients."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

from .utils.log_levels import parse_log_level
from .utils.log_sampling import LogSamplingFilter, LogSamplingRule

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """A single log entry."""

    timestamp: float
    level: str
    logger_name: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "level": self.level,
            "loggerName": self.logger_name,
            "message": self.message,
        }


class LogStreamHandler(logging.Handler):
    """Custom handler that forwards logs to the streamer."""

    def __init__(self, callback: Callable[[LogEntry], None]) -> None:
        super().__init__()
        self._callback = callback
        # Simple format - the log message without redundant timestamp/level
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = LogEntry(
                timestamp=record.created,
                level=record.levelname,
                logger_name=record.name,
                message=self.format(record),
            )
            self._callback(entry)
        except Exception:
            pass  # Don't let logging errors break the application


class LogStreamer:
    """Streams log entries to subscribers in real-time.

    This is a singleton that installs a custom logging handler to capture
    all log messages and forward them to subscribers.
    """

    _instance: LogStreamer | None = None
    _lock = threading.Lock()

    def __new__(cls) -> LogStreamer:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init()
            return cls._instance

    def _init(self) -> None:
        self._subscribers: list[Callable[[LogEntry], None]] = []
        self._buffer: deque[LogEntry] = deque(maxlen=1000)
        self._subscribers_lock = threading.Lock()
        self._initialized = False

    def install_handler(self) -> None:
        """Install the log handler. Call this once at startup."""
        if self._initialized:
            return

        # Install custom logging handler on the root logger
        self._handler = LogStreamHandler(self._on_log)
        stream_level = parse_log_level(os.getenv("WAVECAP_LOG_STREAM_LEVEL"), logging.INFO)
        sampling_level = parse_log_level(os.getenv("WAVECAP_LOG_SAMPLING_LEVEL"), logging.INFO)
        self._handler.setLevel(stream_level)
        sampling_rules = (
            LogSamplingRule(prefix="wavecapsdr.decoders.p25", max_per_interval=3, interval_s=1.0),
            LogSamplingRule(
                prefix="wavecapsdr.decoders.p25_tsbk", max_per_interval=3, interval_s=1.0
            ),
            LogSamplingRule(
                prefix="wavecapsdr.trunking.control_channel", max_per_interval=3, interval_s=1.0
            ),
        )
        self._handler.addFilter(LogSamplingFilter(sampling_rules, max_level=sampling_level))
        logging.getLogger().addHandler(self._handler)
        self._initialized = True
        logger.debug("LogStreamer handler installed")

    def _on_log(self, entry: LogEntry) -> None:
        """Called when a log entry is received."""
        self._buffer.append(entry)

        with self._subscribers_lock:
            subscribers = list(self._subscribers)

        for callback in subscribers:
            try:
                callback(entry)
            except Exception:
                pass

    def subscribe(self, callback: Callable[[LogEntry], None]) -> Callable[[], None]:
        """Subscribe to log entries. Returns unsubscribe function."""
        with self._subscribers_lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._subscribers_lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def get_recent(self, limit: int = 100) -> list[LogEntry]:
        """Get recent log entries from the buffer."""
        return list(self._buffer)[-limit:]

    def subscriber_count(self) -> int:
        """Get the number of active subscribers."""
        with self._subscribers_lock:
            return len(self._subscribers)


# Global singleton instance
_log_streamer: LogStreamer | None = None


def get_log_streamer() -> LogStreamer:
    """Get the global LogStreamer singleton."""
    global _log_streamer
    if _log_streamer is None:
        _log_streamer = LogStreamer()
    return _log_streamer
