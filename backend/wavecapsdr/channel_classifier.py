"""Channel classifier for identifying control vs voice channels from spectrum data.

Control channels transmit continuously with low power variance.
Voice channels transmit intermittently with high power variance.
"""

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


@dataclass
class BinStats:
    """Running statistics for a single frequency bin."""
    sum: float = 0.0
    sum_sq: float = 0.0
    count: int = 0
    min_val: float = float('inf')
    max_val: float = float('-inf')

    def update(self, value: float) -> None:
        self.sum += value
        self.sum_sq += value * value
        self.count += 1
        if value < self.min_val:
            self.min_val = value
        if value > self.max_val:
            self.max_val = value

    @property
    def mean(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        mean = self.mean
        return (self.sum_sq / self.count) - (mean * mean)

    @property
    def std_dev(self) -> float:
        return math.sqrt(max(0.0, self.variance))


@dataclass
class ClassifiedChannel:
    """A classified channel with frequency and type."""
    freq_hz: float
    power_db: float
    std_dev_db: float
    channel_type: "ChannelType"


ChannelType = Literal["control", "voice", "variable", "unknown"]


@dataclass
class ChannelClassifier:
    """Accumulates spectrum statistics and classifies channels.

    Call update() with each FFT frame's power data.
    Call classify() to get the current channel classifications.
    Call reset() when capture parameters change.
    """

    # Configuration
    min_collection_seconds: float = 60.0
    min_samples_per_bin: int = 50
    noise_threshold_db: float = -50.0
    control_variance_threshold: float = 4.0  # Low std dev = control
    voice_variance_threshold: float = 10.0   # High std dev = voice

    # State
    _bin_stats: dict[int, BinStats] = field(default_factory=dict)
    _sample_count: int = 0
    _start_time: float | None = None
    _center_hz: float = 0.0
    _sample_rate: float = 0.0
    _freqs: list[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _cached_channels: list[ClassifiedChannel] = field(default_factory=list)
    _last_classify_time: float = 0.0

    def reset(self) -> None:
        """Reset all accumulated statistics."""
        with self._lock:
            self._bin_stats.clear()
            self._sample_count = 0
            self._start_time = None
            self._cached_channels.clear()
            self._last_classify_time = 0.0

    def update(self, power_db: list[float], freqs: list[float], center_hz: float, sample_rate: float) -> None:
        """Update statistics with a new FFT frame."""
        with self._lock:
            # Check if parameters changed
            if center_hz != self._center_hz or sample_rate != self._sample_rate:
                self._bin_stats.clear()
                self._sample_count = 0
                self._start_time = None
                self._cached_channels.clear()
                self._center_hz = center_hz
                self._sample_rate = sample_rate

            self._freqs = freqs

            if self._start_time is None:
                self._start_time = time.time()

            # Update per-bin statistics
            for i, p in enumerate(power_db):
                if i not in self._bin_stats:
                    self._bin_stats[i] = BinStats()
                self._bin_stats[i].update(p)

            self._sample_count += 1

    @property
    def elapsed_seconds(self) -> float:
        """Seconds since collection started."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def is_ready(self) -> bool:
        """True if enough data has been collected for classification."""
        return self.elapsed_seconds >= self.min_collection_seconds

    @property
    def sample_count(self) -> int:
        """Number of FFT frames collected."""
        return self._sample_count

    def classify(self, force: bool = False) -> list[ClassifiedChannel]:
        """Classify channels based on accumulated statistics.

        Returns cached result if called within 1 second (unless force=True).
        """
        now = time.time()

        with self._lock:
            # Return cached if recent
            if not force and now - self._last_classify_time < 1.0 and self._cached_channels:
                return self._cached_channels.copy()

            if not self.is_ready or not self._bin_stats:
                return []

            # Calculate noise floor from 20th percentile
            averages = [s.mean for s in self._bin_stats.values() if s.count >= self.min_samples_per_bin]
            if not averages:
                return []

            averages.sort()
            noise_floor = averages[int(len(averages) * 0.2)]
            signal_threshold = noise_floor + 10

            # Find peaks and classify
            classified: list[ClassifiedChannel] = []
            visited: set[int] = set()

            # Sort bins by power for peak detection
            sorted_bins = sorted(
                [(i, s) for i, s in self._bin_stats.items() if s.count >= self.min_samples_per_bin],
                key=lambda x: x[1].mean,
                reverse=True
            )

            for bin_idx, stats in sorted_bins:
                if bin_idx in visited:
                    continue

                avg = stats.mean
                std_dev = stats.std_dev

                if avg < signal_threshold:
                    continue

                # Check if local peak
                prev_stats = self._bin_stats.get(bin_idx - 1)
                next_stats = self._bin_stats.get(bin_idx + 1)
                prev_avg = prev_stats.mean if prev_stats else float('-inf')
                next_avg = next_stats.mean if next_stats else float('-inf')

                if avg <= prev_avg or avg <= next_avg:
                    continue

                # Mark nearby bins as visited
                for offset in range(-3, 4):
                    visited.add(bin_idx + offset)

                # Calculate frequency
                freq_hz = self._center_hz + (self._freqs[bin_idx] if bin_idx < len(self._freqs) else 0)

                # Classify
                if avg < noise_floor + 5:
                    channel_type: ChannelType = "unknown"
                elif std_dev < self.control_variance_threshold:
                    channel_type = "control"
                elif std_dev > self.voice_variance_threshold:
                    channel_type = "voice"
                else:
                    channel_type = "variable"

                classified.append(ClassifiedChannel(
                    freq_hz=freq_hz,
                    power_db=avg,
                    std_dev_db=std_dev,
                    channel_type=channel_type,
                ))

            # Sort by power
            classified.sort(key=lambda c: c.power_db, reverse=True)

            self._cached_channels = classified
            self._last_classify_time = now

            return classified.copy()

    def get_status(self) -> dict[str, Any]:
        """Get current classifier status."""
        return {
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "sample_count": self._sample_count,
            "is_ready": self.is_ready,
            "remaining_seconds": max(0, round(self.min_collection_seconds - self.elapsed_seconds, 1)),
            "center_hz": self._center_hz,
            "sample_rate": self._sample_rate,
        }
