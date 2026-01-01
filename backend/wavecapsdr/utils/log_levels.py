from __future__ import annotations

import logging

_LEVEL_ALIASES: dict[str, int] = {
    "CRITICAL": logging.CRITICAL,
    "FATAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "TRACE": logging.DEBUG,
}


def parse_log_level(value: str | None, default: int) -> int:
    """Parse a log level string into a numeric level."""
    if not value:
        return default
    raw = value.strip()
    if not raw:
        return default
    if raw.isdigit():
        return int(raw)
    key = raw.upper().replace("-", "_")
    return _LEVEL_ALIASES.get(key, default)


def log_level_name(value: str | None, default: int) -> str:
    """Return a lowercase log level name for libraries expecting strings."""
    level = parse_log_level(value, default)
    name = logging.getLevelName(level)
    if isinstance(name, str) and not name.startswith("Level "):
        return name.lower()
    return logging.getLevelName(default).lower()
