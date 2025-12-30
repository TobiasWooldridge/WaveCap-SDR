"""NAC (Network Access Code) tracker for P25.

Tracks the dominant NAC value observed on a channel to assist BCH error
correction. When BCH decode fails, the decoder can use the tracked NAC
to overwrite corrupted NAC bits and retry decoding.

Based on SDRTrunk's NACTracker:
https://github.com/DSheirer/sdrtrunk/blob/master/src/main/java/io/github/dsheirer/module/decode/p25/phase1/NACTracker.java
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class NACTracker:
    """Tracks the dominant NAC value for P25 NID decoding assistance.

    This tracker maintains a count of recently observed NAC values and
    provides the most frequently observed NAC for use in BCH error correction.

    Attributes:
        MAX_TRACKER_COUNT: Maximum number of distinct NACs to track (3)
        MIN_OBSERVATION_THRESHOLD: Minimum observations before NAC is "dominant" (3)
    """

    MAX_TRACKER_COUNT = 3
    MIN_OBSERVATION_THRESHOLD = 3

    def __init__(self) -> None:
        """Initialize NAC tracker."""
        self._trackers: dict[int, _NACObservation] = {}

    def reset(self) -> None:
        """Remove all tracked NAC values.

        Invoke this after extended loss of sync.
        """
        self._trackers.clear()
        logger.debug("NAC tracker reset")

    def track(self, nac: int) -> None:
        """Track an observed NAC value.

        Each observation increments the count and updates the timestamp.
        When the tracker count exceeds MAX_TRACKER_COUNT, the oldest
        tracker (by timestamp) is removed.

        Args:
            nac: NAC value (12-bit, 0x000-0xFFF)
        """
        if nac < 0 or nac > 0xFFF:
            logger.warning(f"Invalid NAC value: 0x{nac:x}")
            return

        if nac in self._trackers:
            self._trackers[nac].increment()
        else:
            self._trackers[nac] = _NACObservation(nac)

            # Prune oldest if we exceed max count
            if len(self._trackers) > self.MAX_TRACKER_COUNT:
                # Find oldest by timestamp
                oldest_nac = min(
                    self._trackers.keys(), key=lambda k: self._trackers[k].timestamp
                )
                del self._trackers[oldest_nac]
                logger.debug(f"Pruned oldest NAC tracker: 0x{oldest_nac:03x}")

    def get_tracked_nac(self) -> int:
        """Get the dominant tracked NAC value.

        The dominant NAC is the one with the highest observation count,
        but it must have at least MIN_OBSERVATION_THRESHOLD observations.

        Returns:
            Dominant NAC value, or 0 if no dominant NAC
        """
        if not self._trackers:
            return 0

        # Find tracker with highest count
        dominant = max(self._trackers.values(), key=lambda t: t.count)

        if dominant.count >= self.MIN_OBSERVATION_THRESHOLD:
            return dominant.nac

        return 0

    def get_statistics(self) -> list[dict[str, Any]]:
        """Get statistics for all tracked NACs.

        Returns:
            List of dicts with keys: nac, count, timestamp
        """
        return [
            {"nac": t.nac, "count": t.count, "timestamp": t.timestamp}
            for t in sorted(self._trackers.values(), key=lambda t: t.count, reverse=True)
        ]


class _NACObservation:
    """Single NAC value observation tracker.

    Tracks observation count and last seen timestamp for a NAC value.
    """

    def __init__(self, nac: int) -> None:
        """Initialize observation tracker.

        Args:
            nac: NAC value
        """
        self.nac = nac
        self.count = 1
        self.timestamp = time.time()

    def increment(self) -> None:
        """Increment observation count and update timestamp."""
        self.count += 1
        self.timestamp = time.time()

        # Prevent rollover
        if self.count < 0:
            self.count = 2**31 - 1

    def __repr__(self) -> str:
        return f"NAC(0x{self.nac:03x}, count={self.count}, age={time.time() - self.timestamp:.1f}s)"
