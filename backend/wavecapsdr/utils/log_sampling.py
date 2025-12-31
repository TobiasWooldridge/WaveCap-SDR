"""Log sampling utilities to reduce high-frequency logging overhead."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class LogSamplingRule:
    """Sampling rule for loggers matching a prefix."""

    prefix: str
    max_per_interval: int
    interval_s: float


class LogSamplingFilter(logging.Filter):
    """Rate-limit log records for noisy logger prefixes.

    Sampling only applies to records at or below max_level. Warnings and errors
    are always allowed through.
    """

    def __init__(
        self,
        rules: Iterable[LogSamplingRule],
        max_level: int = logging.INFO,
    ) -> None:
        super().__init__()
        self._rules = tuple(rules)
        self._max_level = max_level
        self._state: dict[tuple[str, int, str], tuple[float, int]] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno > self._max_level:
            return True

        rule = self._match_rule(record.name)
        if rule is None:
            return True

        now = time.monotonic()
        key = (record.name, record.levelno, rule.prefix)
        window_start, count = self._state.get(key, (now, 0))

        if now - window_start >= rule.interval_s:
            self._state[key] = (now, 1)
            return True

        if count < rule.max_per_interval:
            self._state[key] = (window_start, count + 1)
            return True

        return False

    def _match_rule(self, logger_name: str) -> LogSamplingRule | None:
        for rule in self._rules:
            if logger_name.startswith(rule.prefix):
                return rule
        return None
