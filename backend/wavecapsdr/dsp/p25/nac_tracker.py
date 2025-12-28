"""NAC Tracker - Track Network Access Codes for intelligent NID recovery.

Port of SDRTrunk's NACTracker.java.

The NAC (Network Access Code) is a 12-bit value in the NID (Network ID)
that identifies the P25 system. When BCH decoding of the NID fails,
we can retry using a known NAC value to help recover the DUID.

This tracker maintains the last N observed NAC values and provides
the dominant (most frequently seen) NAC for retry attempts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class NACTracker:
    """Track observed NACs for intelligent NID recovery.

    Port of SDRTrunk NACTracker.java.

    When NID BCH decoding fails, this tracker provides the most likely
    NAC value based on recent observations. This allows retry of BCH
    decoding with the NAC bits fixed to a known value.

    Attributes:
        max_observations: Maximum number of NAC observations to track
        min_for_dominant: Minimum observations required to return a dominant NAC
    """

    max_observations: int = 3
    min_for_dominant: int = 3

    # Private: (NAC, count) pairs sorted by count descending
    _observations: list[tuple[int, int]] = field(default_factory=list)

    def observe(self, nac: int) -> None:
        """Record an observed NAC value.

        If the NAC has been seen before, increment its count.
        Otherwise, add it to the observation list.

        Args:
            nac: 12-bit Network Access Code (0-4095)
        """
        if nac < 0 or nac > 0xFFF:
            logger.warning(f"NACTracker: Invalid NAC value 0x{nac:03X}")
            return

        # Check if we've seen this NAC before
        for i, (obs_nac, count) in enumerate(self._observations):
            if obs_nac == nac:
                # Increment count for existing NAC
                self._observations[i] = (nac, count + 1)
                # Re-sort by count descending
                self._observations.sort(key=lambda x: x[1], reverse=True)
                return

        # New NAC - add to list
        self._observations.append((nac, 1))

        # Keep only top max_observations by count
        self._observations.sort(key=lambda x: x[1], reverse=True)
        if len(self._observations) > self.max_observations:
            self._observations = self._observations[: self.max_observations]

    def get_dominant_nac(self) -> int | None:
        """Get the NAC with the most observations.

        Only returns a NAC if it has been observed at least
        min_for_dominant times.

        Returns:
            Most frequently observed NAC, or None if insufficient observations
        """
        if not self._observations:
            return None

        top_nac, top_count = self._observations[0]
        if top_count >= self.min_for_dominant:
            return top_nac

        return None

    def get_observations(self) -> list[tuple[int, int]]:
        """Get all NAC observations.

        Returns:
            List of (NAC, count) tuples sorted by count descending
        """
        return list(self._observations)

    def reset(self) -> None:
        """Clear all observations."""
        self._observations.clear()

    def __str__(self) -> str:
        """String representation for debugging."""
        if not self._observations:
            return "NACTracker(empty)"
        obs_str = ", ".join(f"0x{nac:03X}:{count}" for nac, count in self._observations)
        dominant = self.get_dominant_nac()
        dominant_str = f" dominant=0x{dominant:03X}" if dominant is not None else ""
        return f"NACTracker({obs_str}{dominant_str})"
