"""System metrics collection using psutil."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .state import AppState

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """Snapshot of system-wide metrics."""

    timestamp: float
    cpu_percent: float  # System-wide CPU usage
    cpu_per_core: list[float]  # Per-core CPU usage
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    temperatures: dict[str, float] = field(default_factory=dict)  # sensor_name -> temp_celsius

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "cpuPercent": self.cpu_percent,
            "cpuPerCore": self.cpu_per_core,
            "memoryUsedMb": self.memory_used_mb,
            "memoryTotalMb": self.memory_total_mb,
            "memoryPercent": self.memory_percent,
            "temperatures": self.temperatures,
        }


@dataclass
class CaptureMetrics:
    """Per-capture metrics snapshot."""

    capture_id: str
    device_id: str
    state: str
    iq_overflow_count: int
    iq_overflow_rate: float
    channel_count: int
    total_subscribers: int
    total_drops: int
    perf_loop_ms: float
    perf_dsp_ms: float
    perf_fft_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "captureId": self.capture_id,
            "deviceId": self.device_id,
            "state": self.state,
            "iqOverflowCount": self.iq_overflow_count,
            "iqOverflowRate": self.iq_overflow_rate,
            "channelCount": self.channel_count,
            "totalSubscribers": self.total_subscribers,
            "totalDrops": self.total_drops,
            "perfLoopMs": self.perf_loop_ms,
            "perfDspMs": self.perf_dsp_ms,
            "perfFftMs": self.perf_fft_ms,
        }


def get_system_metrics() -> SystemMetrics:
    """Collect current system metrics using psutil."""
    try:
        import psutil
    except ImportError:
        logger.warning("psutil not installed, returning empty metrics")
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=0.0,
            cpu_per_core=[],
            memory_used_mb=0.0,
            memory_total_mb=0.0,
            memory_percent=0.0,
            temperatures={},
        )

    # CPU - use interval=None for non-blocking (returns since last call)
    cpu_percent = psutil.cpu_percent(interval=None)
    cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)

    # Memory
    mem = psutil.virtual_memory()

    # Temperature (may not be available on all systems)
    temperatures: dict[str, float] = {}
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            for sensor_name, entries in temps.items():
                for entry in entries:
                    label = entry.label or sensor_name
                    # Avoid duplicate keys by adding index
                    key = label
                    if key in temperatures:
                        key = f"{label}_{sensor_name}"
                    temperatures[key] = entry.current
    except (AttributeError, NotImplementedError, OSError):
        pass  # Not available on this platform (e.g., macOS)

    return SystemMetrics(
        timestamp=time.time(),
        cpu_percent=cpu_percent,
        cpu_per_core=cpu_per_core,
        memory_used_mb=mem.used / (1024 * 1024),
        memory_total_mb=mem.total / (1024 * 1024),
        memory_percent=mem.percent,
        temperatures=temperatures,
    )


def get_capture_metrics(state: AppState) -> list[CaptureMetrics]:
    """Collect metrics for all active captures."""
    from .error_tracker import get_error_tracker

    tracker = get_error_tracker()
    stats = tracker.get_stats()

    # Get IQ overflow rate from error tracker
    iq_stats = stats.get("iq_overflow")
    overflow_rate = iq_stats.rate_per_second if iq_stats else 0.0

    metrics: list[CaptureMetrics] = []
    for cap in state.captures.list_captures():
        # Get performance stats
        perf = cap.get_performance_stats()

        # Get channel stats
        total_subscribers = 0
        total_drops = 0
        channels = state.captures.list_channels(cap.cfg.id)
        for ch in channels:
            q_stats = ch.get_queue_stats()
            total_subscribers += q_stats.get("total_subscribers", 0)
            total_drops += q_stats.get("drops_since_last_log", 0)

        # Get IQ overflow count from capture
        iq_overflow_count = getattr(cap, "_iq_overflow_count", 0)

        metrics.append(
            CaptureMetrics(
                capture_id=cap.cfg.id,
                device_id=cap.cfg.device_id,
                state=cap.state,
                iq_overflow_count=iq_overflow_count,
                iq_overflow_rate=overflow_rate,
                channel_count=len(channels),
                total_subscribers=total_subscribers,
                total_drops=total_drops,
                perf_loop_ms=perf["loop"]["mean_ms"],
                perf_dsp_ms=perf["dsp"]["mean_ms"],
                perf_fft_ms=perf["fft"]["mean_ms"],
            )
        )

    return metrics
