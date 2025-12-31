"""Lightweight profiler for real-time DSP performance analysis.

Usage:
    from wavecapsdr.utils.profiler import Profiler

    profiler = Profiler("MyComponent")

    # In hot path:
    with profiler.measure("decimation"):
        result = decimate(samples)

    # Or manual:
    profiler.start("fft")
    result = fft(samples)
    profiler.stop("fft")

    # Periodically (e.g., every 5 seconds):
    profiler.report()  # Logs statistics and resets
"""

import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

logger = logging.getLogger(__name__)


@dataclass
class TimingStats:
    """Accumulated timing statistics for a single operation."""

    count: int = 0
    total_ns: int = 0
    min_ns: int = 0
    max_ns: int = 0

    def record(self, elapsed_ns: int) -> None:
        """Record a timing measurement."""
        self.count += 1
        self.total_ns += elapsed_ns
        if self.count == 1:
            self.min_ns = elapsed_ns
            self.max_ns = elapsed_ns
        else:
            self.min_ns = min(self.min_ns, elapsed_ns)
            self.max_ns = max(self.max_ns, elapsed_ns)

    @property
    def avg_ns(self) -> float:
        """Average time in nanoseconds."""
        return self.total_ns / self.count if self.count > 0 else 0.0

    @property
    def avg_us(self) -> float:
        """Average time in microseconds."""
        return self.avg_ns / 1000.0

    @property
    def avg_ms(self) -> float:
        """Average time in milliseconds."""
        return self.avg_ns / 1_000_000.0

    def reset(self) -> None:
        """Reset all statistics."""
        self.count = 0
        self.total_ns = 0
        self.min_ns = 0
        self.max_ns = 0


@dataclass
class Profiler:
    """Lightweight profiler for real-time performance analysis.

    Designed for minimal overhead in hot paths. Uses time.perf_counter_ns()
    for high-resolution timing with ~100ns overhead per measurement.
    """

    name: str
    enabled: bool = True
    report_interval_s: float = 5.0  # Auto-report interval

    # Internal state
    _stats: dict[str, TimingStats] = field(default_factory=lambda: defaultdict(TimingStats))
    _active: dict[str, int] = field(default_factory=dict)  # start times for manual timing
    _last_report: float = field(default_factory=time.time)
    _samples_processed: int = 0

    def start(self, operation: str) -> None:
        """Start timing an operation (for manual start/stop)."""
        if not self.enabled:
            return
        self._active[operation] = time.perf_counter_ns()

    def stop(self, operation: str) -> None:
        """Stop timing an operation and record the result."""
        if not self.enabled:
            return
        start = self._active.pop(operation, None)
        if start is not None:
            elapsed = time.perf_counter_ns() - start
            self._stats[operation].record(elapsed)

    @contextmanager
    def measure(self, operation: str) -> Generator[None, None, None]:
        """Context manager for timing a block of code."""
        if not self.enabled:
            yield
            return

        start = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed = time.perf_counter_ns() - start
            self._stats[operation].record(elapsed)

    def add_samples(self, count: int) -> None:
        """Track number of samples processed (for throughput calculation)."""
        self._samples_processed += count

    def should_report(self) -> bool:
        """Check if it's time to auto-report."""
        return time.time() - self._last_report >= self.report_interval_s

    def report(self, force: bool = False) -> str | None:
        """Generate and log a performance report.

        Returns the report string, or None if nothing to report.
        """
        if not self.enabled:
            return None

        if not force and not self.should_report():
            return None

        elapsed_s = time.time() - self._last_report
        if elapsed_s < 0.1:
            return None  # Avoid division by zero / too-frequent reports

        if not self._stats:
            self._last_report = time.time()
            return None

        # Build report
        lines = [f"[PROFILE] {self.name} - {elapsed_s:.1f}s window:"]

        # Calculate throughput
        if self._samples_processed > 0:
            throughput = self._samples_processed / elapsed_s
            lines.append(
                f"  Throughput: {throughput / 1e6:.2f} Msps ({self._samples_processed:,} samples)"
            )

        # Sort operations by total time (descending)
        sorted_ops = sorted(self._stats.items(), key=lambda x: x[1].total_ns, reverse=True)

        # Calculate total time for percentage
        total_time_ns = sum(s.total_ns for _, s in sorted_ops)

        for op, stats in sorted_ops:
            if stats.count == 0:
                continue

            pct = 100.0 * stats.total_ns / total_time_ns if total_time_ns > 0 else 0.0
            min_us = stats.min_ns / 1000.0
            max_us = stats.max_ns / 1000.0
            avg_us = stats.avg_us

            # Format based on magnitude
            if avg_us >= 1000:
                lines.append(
                    f"  {op}: {pct:5.1f}% | {stats.count:6d} calls | "
                    f"avg={avg_us / 1000:.2f}ms min={min_us / 1000:.2f}ms max={max_us / 1000:.2f}ms"
                )
            else:
                lines.append(
                    f"  {op}: {pct:5.1f}% | {stats.count:6d} calls | "
                    f"avg={avg_us:.1f}us min={min_us:.1f}us max={max_us:.1f}us"
                )

        report = "\n".join(lines)
        logger.info(report)

        # Reset for next window
        self.reset()
        return report

    def reset(self) -> None:
        """Reset all statistics for a new measurement window."""
        for stats in self._stats.values():
            stats.reset()
        self._samples_processed = 0
        self._last_report = time.time()


# Global profilers for key components
_profilers: dict[str, Profiler] = {}


def get_profiler(name: str, enabled: bool = True) -> Profiler:
    """Get or create a named profiler instance."""
    if name not in _profilers:
        _profilers[name] = Profiler(name=name, enabled=enabled)
    return _profilers[name]


def report_all() -> None:
    """Generate reports for all profilers."""
    for profiler in _profilers.values():
        profiler.report(force=True)
