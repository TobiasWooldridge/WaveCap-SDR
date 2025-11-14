"""
Scanner service for automated frequency scanning.

Supports multiple scan modes:
- Sequential: Cycle through frequencies in order
- Priority: Check priority frequencies at regular intervals
- Activity: Pause on active signals (squelch-based)
"""
import asyncio
import time
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field


class ScanMode(str, Enum):
    """Scan mode types."""
    SEQUENTIAL = "sequential"  # A→B→C→A
    PRIORITY = "priority"      # Check priority freq every N seconds
    ACTIVITY = "activity"      # Pause on active signal (squelch-based)


class ScanState(str, Enum):
    """Scanner state."""
    STOPPED = "stopped"
    SCANNING = "scanning"
    PAUSED = "paused"      # Paused on activity
    LOCKED = "locked"      # Manually locked on frequency


@dataclass
class ScanConfig:
    """Scanner configuration."""
    scan_list: list[float]  # Frequencies to scan (Hz)
    mode: ScanMode = ScanMode.SEQUENTIAL
    dwell_time_ms: int = 500  # Time per frequency
    priority_frequencies: list[float] = field(default_factory=list)
    priority_interval_s: int = 5
    squelch_threshold_db: float = -60.0  # For activity mode
    lockout_frequencies: list[float] = field(default_factory=list)  # Skip these
    pause_duration_ms: int = 3000  # How long to pause on activity


@dataclass
class ScanStatus:
    """Current scanner status."""
    state: ScanState
    current_frequency: float
    current_index: int
    hits: list[tuple[float, float]]  # (frequency, timestamp) of activity
    lockout_list: list[float]  # Currently locked out frequencies
    last_priority_check: float = 0.0  # Timestamp of last priority check


class ScannerService:
    """Service for managing frequency scanning."""

    def __init__(self, capture_id: str, config: ScanConfig):
        """
        Initialize scanner.

        Args:
            capture_id: ID of the capture to control
            config: Scan configuration
        """
        self.capture_id = capture_id
        self.config = config
        self.status = ScanStatus(
            state=ScanState.STOPPED,
            current_frequency=config.scan_list[0] if config.scan_list else 0.0,
            current_index=0,
            hits=[],
            lockout_list=list(config.lockout_frequencies),
        )
        self._task: Optional[asyncio.Task] = None
        self._update_callback: Optional[callable] = None
        self._rssi_callback: Optional[callable] = None  # Get current RSSI

    def set_update_callback(self, callback: callable):
        """Set callback for frequency updates: callback(frequency_hz)."""
        self._update_callback = callback

    def set_rssi_callback(self, callback: callable):
        """Set callback to get current RSSI: rssi = callback()."""
        self._rssi_callback = callback

    def start(self):
        """Start scanning."""
        if self._task is not None:
            return

        self.status.state = ScanState.SCANNING
        self._task = asyncio.create_task(self._scan_loop())

    def stop(self):
        """Stop scanning."""
        if self._task is None:
            return

        self._task.cancel()
        self._task = None
        self.status.state = ScanState.STOPPED

    def pause(self):
        """Pause scanning (manual pause)."""
        self.status.state = ScanState.PAUSED

    def resume(self):
        """Resume scanning from pause."""
        if self.status.state == ScanState.PAUSED:
            self.status.state = ScanState.SCANNING

    def lock(self):
        """Lock on current frequency (stop scanning)."""
        self.status.state = ScanState.LOCKED

    def unlock(self):
        """Unlock and resume scanning."""
        if self.status.state == ScanState.LOCKED:
            self.status.state = ScanState.SCANNING

    def lockout_current(self):
        """Add current frequency to lockout list."""
        if self.status.current_frequency not in self.status.lockout_list:
            self.status.lockout_list.append(self.status.current_frequency)

    def clear_lockout(self, frequency: float):
        """Remove frequency from lockout list."""
        if frequency in self.status.lockout_list:
            self.status.lockout_list.remove(frequency)

    def clear_all_lockouts(self):
        """Clear all lockouts."""
        self.status.lockout_list.clear()

    def _get_next_frequency(self) -> tuple[float, int]:
        """
        Get next frequency to scan.

        Returns:
            (frequency, index) tuple
        """
        # Build scan list excluding lockouts
        scan_list = [
            freq for freq in self.config.scan_list
            if freq not in self.status.lockout_list
        ]

        if not scan_list:
            # All locked out, use first non-locked frequency
            return self.config.scan_list[0], 0

        # Advance to next frequency
        self.status.current_index = (self.status.current_index + 1) % len(scan_list)
        freq = scan_list[self.status.current_index]

        return freq, self.status.current_index

    def _should_check_priority(self) -> bool:
        """Check if it's time to check priority frequencies."""
        if not self.config.priority_frequencies:
            return False

        now = time.time()
        elapsed = now - self.status.last_priority_check
        return elapsed >= self.config.priority_interval_s

    def _get_current_rssi(self) -> Optional[float]:
        """Get current RSSI from callback."""
        if self._rssi_callback is None:
            return None
        return self._rssi_callback()

    def _is_activity_detected(self) -> bool:
        """Check if activity is detected (above squelch)."""
        rssi = self._get_current_rssi()
        if rssi is None:
            return False
        return rssi > self.config.squelch_threshold_db

    async def _scan_loop(self):
        """Main scan loop."""
        try:
            while True:
                # Check if we should pause for activity
                if self.config.mode == ScanMode.ACTIVITY and self._is_activity_detected():
                    self.status.state = ScanState.PAUSED
                    # Record hit
                    self.status.hits.append((self.status.current_frequency, time.time()))
                    # Pause for configured duration
                    await asyncio.sleep(self.config.pause_duration_ms / 1000.0)
                    self.status.state = ScanState.SCANNING

                # Check if paused or locked
                if self.status.state != ScanState.SCANNING:
                    await asyncio.sleep(0.1)
                    continue

                # Check priority frequencies
                if (
                    self.config.mode == ScanMode.PRIORITY
                    and self._should_check_priority()
                ):
                    self.status.last_priority_check = time.time()
                    for priority_freq in self.config.priority_frequencies:
                        if priority_freq in self.status.lockout_list:
                            continue

                        # Tune to priority frequency
                        if self._update_callback:
                            await self._update_callback(priority_freq)
                        self.status.current_frequency = priority_freq

                        # Dwell on priority frequency
                        await asyncio.sleep(self.config.dwell_time_ms / 1000.0)

                        # Check for activity on priority
                        if self._is_activity_detected():
                            self.status.state = ScanState.PAUSED
                            self.status.hits.append((priority_freq, time.time()))
                            await asyncio.sleep(self.config.pause_duration_ms / 1000.0)
                            self.status.state = ScanState.SCANNING
                            break  # Return to scan list after priority hit

                # Get next frequency
                next_freq, next_idx = self._get_next_frequency()

                # Update frequency
                if self._update_callback:
                    await self._update_callback(next_freq)

                self.status.current_frequency = next_freq
                self.status.current_index = next_idx

                # Dwell on frequency
                await asyncio.sleep(self.config.dwell_time_ms / 1000.0)

        except asyncio.CancelledError:
            # Clean shutdown
            pass
