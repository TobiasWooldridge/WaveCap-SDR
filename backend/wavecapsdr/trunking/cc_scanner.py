"""Control Channel Scanner - Signal strength measurement for P25 control channels.

This module implements control channel scanning to find the strongest control
channel frequency for initial connection and roaming.

Features:
- Scans all configured control channel frequencies
- Measures signal power in the P25 bandwidth
- Optionally verifies P25 sync pattern presence
- Supports periodic background scanning for roaming
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

from wavecapsdr.capture import freq_shift

logger = logging.getLogger(__name__)


@dataclass
class ChannelMeasurement:
    """Signal measurement for a control channel frequency."""

    frequency_hz: float
    power_db: float  # Average signal power in dB
    peak_power_db: float  # Peak signal power in dB
    noise_floor_db: float  # Estimated noise floor in dB
    snr_db: float  # Estimated SNR
    sync_detected: bool  # True if P25 sync pattern was detected
    measurement_time: float  # Timestamp of measurement
    sample_count: int  # Number of samples used

    def __str__(self) -> str:
        sync_str = "SYNC" if self.sync_detected else "----"
        return (
            f"{self.frequency_hz/1e6:.4f} MHz: "
            f"power={self.power_db:.1f} dB, "
            f"SNR={self.snr_db:.1f} dB, "
            f"{sync_str}"
        )


@dataclass
class ControlChannelScanner:
    """Scans control channel frequencies to find the strongest signal.

    Used for:
    1. Initial startup - scan all frequencies and pick the best one
    2. Periodic roaming - check if a better channel is available
    3. Hunt recovery - find a new channel when current one is lost

    The scanner uses the SDR capture's wideband IQ stream to measure
    signal strength at each control channel offset without retuning.
    """

    # Configuration
    center_hz: float  # SDR capture center frequency
    sample_rate: int  # SDR capture sample rate
    control_channels: list[float]  # List of control channel frequencies (Hz)

    # Measurement settings
    measurement_samples: int = 48000 * 2  # 2 seconds at 48 kHz
    channel_bandwidth: float = 12500  # P25 channel bandwidth in Hz
    min_snr_db: float = 6.0  # Minimum SNR to consider a channel usable
    sync_check_enabled: bool = True  # Whether to check for P25 sync pattern

    # State
    _last_scan_time: float = 0.0
    _measurements: dict[float, ChannelMeasurement] = field(default_factory=dict)
    _current_channel_hz: float | None = None
    _iq_buffer: list[np.ndarray] = field(default_factory=list)
    _buffer_samples: int = 0
    _measurement_in_progress: bool = False
    _measurement_complete: asyncio.Event = field(default_factory=asyncio.Event)

    # P25 sync pattern for detection (48 bits as 24 dibits)
    _sync_pattern: np.ndarray = field(default=None)

    def __post_init__(self) -> None:
        """Initialize the scanner."""
        # P25 frame sync pattern as dibits (same as ControlChannelMonitor)
        # +3 +3 +3 +3 +3 -3 +3 +3 -3 -3 +3 +3 -3 -3 -3 -3 +3 -3 +3 -3 -3 -3 -3 -3
        # +3 -> dibit 1, -3 -> dibit 3
        self._sync_pattern = np.array([
            1, 1, 1, 1, 1, 3, 1, 1, 3, 3, 1, 1,
            3, 3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3
        ], dtype=np.uint8)

        logger.info(
            f"ControlChannelScanner initialized: "
            f"center={self.center_hz/1e6:.4f} MHz, "
            f"channels={len(self.control_channels)}"
        )

    def get_channel_offset(self, freq_hz: float) -> float:
        """Calculate offset from center frequency for a channel."""
        return freq_hz - self.center_hz

    def is_channel_in_range(self, freq_hz: float) -> bool:
        """Check if a frequency is within the capture bandwidth."""
        offset = abs(self.get_channel_offset(freq_hz))
        max_offset = self.sample_rate / 2 - self.channel_bandwidth
        return offset <= max_offset

    def scan_all(self, iq: np.ndarray) -> dict[float, ChannelMeasurement]:
        """Scan all control channels and return measurements.

        This performs a quick scan of all configured control channel
        frequencies using the provided IQ samples.

        Args:
            iq: Complex IQ samples from SDR capture

        Returns:
            Dict mapping frequency to measurement result
        """
        measurements = {}

        for freq_hz in self.control_channels:
            if not self.is_channel_in_range(freq_hz):
                logger.warning(
                    f"Control channel {freq_hz/1e6:.4f} MHz is outside "
                    f"capture bandwidth, skipping"
                )
                continue

            measurement = self._measure_channel(iq, freq_hz)
            measurements[freq_hz] = measurement
            self._measurements[freq_hz] = measurement

        self._last_scan_time = time.time()
        return measurements

    def _measure_channel(self, iq: np.ndarray, freq_hz: float) -> ChannelMeasurement:
        """Measure signal strength at a specific frequency.

        Args:
            iq: Wideband IQ samples
            freq_hz: Target frequency in Hz

        Returns:
            Channel measurement result
        """
        # Frequency shift to center on channel
        offset_hz = self.get_channel_offset(freq_hz)
        shifted_iq = freq_shift(iq, offset_hz, self.sample_rate)

        # Decimate to P25 rate (~48 kHz) for analysis
        target_rate = 48000
        decim_factor = max(1, self.sample_rate // target_rate)

        if decim_factor > 1:
            # Simple decimation (no filter needed for power measurement)
            decimated_iq = shifted_iq[::decim_factor]
        else:
            decimated_iq = shifted_iq

        # Calculate signal power
        signal_power = np.mean(np.abs(decimated_iq) ** 2)
        peak_power = np.max(np.abs(decimated_iq) ** 2)

        # Estimate noise floor from quietest 10% of samples
        sorted_power = np.sort(np.abs(decimated_iq) ** 2)
        noise_samples = sorted_power[:max(1, len(sorted_power) // 10)]
        noise_floor = np.mean(noise_samples)

        # Convert to dB
        eps = 1e-12  # Avoid log(0)
        power_db = 10 * np.log10(signal_power + eps)
        peak_power_db = 10 * np.log10(peak_power + eps)
        noise_floor_db = 10 * np.log10(noise_floor + eps)
        snr_db = power_db - noise_floor_db

        # Check for P25 sync pattern
        sync_detected = False
        if self.sync_check_enabled and len(decimated_iq) > 0:
            sync_detected = self._detect_sync_pattern(decimated_iq)

        return ChannelMeasurement(
            frequency_hz=freq_hz,
            power_db=power_db,
            peak_power_db=peak_power_db,
            noise_floor_db=noise_floor_db,
            snr_db=snr_db,
            sync_detected=sync_detected,
            measurement_time=time.time(),
            sample_count=len(decimated_iq),
        )

    def _detect_sync_pattern(self, iq: np.ndarray) -> bool:
        """Detect P25 frame sync pattern in IQ samples.

        Uses simplified detection - checks for correlation with
        the expected sync pattern in the demodulated symbols.

        Args:
            iq: Decimated IQ samples (~48 kHz rate)

        Returns:
            True if sync pattern was detected
        """
        # Demodulate to symbols using differential phase
        # P25 at 4800 symbols/sec, 48000 samples/sec = 10 samples/symbol
        samples_per_symbol = 10

        if len(iq) < samples_per_symbol * len(self._sync_pattern):
            return False

        # Get phase
        phase = np.angle(iq)

        # Differential phase (like CQPSK demodulation)
        phase_diff = np.diff(phase)
        phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi

        # Sample at symbol rate
        symbols_count = len(phase_diff) // samples_per_symbol
        if symbols_count < len(self._sync_pattern):
            return False

        # Take middle sample of each symbol
        symbol_phases = phase_diff[samples_per_symbol // 2::samples_per_symbol][:symbols_count]

        # Map to dibits (0-3) based on phase quadrant
        # CQPSK: +π/4 -> 0, +3π/4 -> 1, -3π/4 -> 3, -π/4 -> 2
        half_pi = np.pi / 2
        dibits = np.zeros(len(symbol_phases), dtype=np.uint8)
        dibits[(symbol_phases >= 0) & (symbol_phases < half_pi)] = 0
        dibits[symbol_phases >= half_pi] = 1
        dibits[(symbol_phases >= -half_pi) & (symbol_phases < 0)] = 2
        dibits[symbol_phases < -half_pi] = 3

        # Search for sync pattern with tolerance
        sync_len = len(self._sync_pattern)
        max_errors = 4  # Allow up to 4 dibit errors

        for i in range(len(dibits) - sync_len):
            errors = np.sum(dibits[i:i + sync_len] != self._sync_pattern)
            if errors <= max_errors:
                return True

        return False

    def get_best_channel(self) -> tuple[float, ChannelMeasurement] | None:
        """Get the best control channel based on measurements.

        Prioritizes:
        1. Channels with detected P25 sync pattern
        2. Channels with highest SNR
        3. Channels with highest power

        Returns:
            Tuple of (frequency_hz, measurement) or None if no usable channel
        """
        if not self._measurements:
            return None

        # Filter to usable channels (above minimum SNR)
        usable = [
            (freq, m) for freq, m in self._measurements.items()
            if m.snr_db >= self.min_snr_db
        ]

        if not usable:
            # Fall back to all channels if none meet SNR threshold
            usable = list(self._measurements.items())

        if not usable:
            return None

        # Sort by: sync_detected (True first), then SNR, then power
        usable.sort(
            key=lambda x: (
                x[1].sync_detected,  # True sorts after False, so negate
                x[1].snr_db,
                x[1].power_db
            ),
            reverse=True
        )

        return usable[0]

    def get_channel_ranking(self) -> list[tuple[float, ChannelMeasurement]]:
        """Get all channels ranked by signal quality.

        Returns:
            List of (frequency_hz, measurement) sorted best to worst
        """
        if not self._measurements:
            return []

        ranked = list(self._measurements.items())
        ranked.sort(
            key=lambda x: (
                x[1].sync_detected,
                x[1].snr_db,
                x[1].power_db
            ),
            reverse=True
        )

        return ranked

    def should_roam(
        self,
        current_freq_hz: float,
        roam_threshold_db: float = 6.0,
    ) -> float | None:
        """Check if we should roam to a better channel.

        Args:
            current_freq_hz: Current control channel frequency
            roam_threshold_db: Minimum SNR improvement to trigger roaming

        Returns:
            Frequency to roam to, or None if current channel is best
        """
        if current_freq_hz not in self._measurements:
            return None

        current = self._measurements[current_freq_hz]
        best = self.get_best_channel()

        if best is None:
            return None

        best_freq, best_measurement = best

        # Don't roam to same channel
        if best_freq == current_freq_hz:
            return None

        # Check if improvement is significant
        snr_improvement = best_measurement.snr_db - current.snr_db

        # Roam if:
        # 1. Current channel has no sync but better one does, OR
        # 2. SNR improvement exceeds threshold
        if not current.sync_detected and best_measurement.sync_detected:
            logger.info(
                f"Roaming to {best_freq/1e6:.4f} MHz: "
                f"has sync pattern (current doesn't)"
            )
            return best_freq

        if snr_improvement >= roam_threshold_db:
            logger.info(
                f"Roaming to {best_freq/1e6:.4f} MHz: "
                f"SNR improvement {snr_improvement:.1f} dB"
            )
            return best_freq

        return None

    def log_scan_results(self) -> None:
        """Log current scan results for debugging."""
        if not self._measurements:
            logger.info("No control channel measurements available")
            return

        logger.info("Control Channel Scan Results:")
        logger.info("-" * 60)

        ranked = self.get_channel_ranking()
        for i, (_freq, m) in enumerate(ranked):
            marker = " *" if i == 0 else ""
            logger.info(f"  {i+1}. {m}{marker}")

        logger.info("-" * 60)

    def get_stats(self) -> dict:
        """Get scanner statistics."""
        return {
            "channels_configured": len(self.control_channels),
            "channels_measured": len(self._measurements),
            "last_scan_time": float(self._last_scan_time),
            "current_channel_hz": float(self._current_channel_hz) if self._current_channel_hz else None,
            "measurements": {
                f"{freq/1e6:.4f}_MHz": {
                    "power_db": float(m.power_db),
                    "snr_db": float(m.snr_db),
                    "sync_detected": bool(m.sync_detected),
                }
                for freq, m in self._measurements.items()
            },
        }
