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
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

from scipy import signal as scipy_signal

from wavecapsdr.capture import freq_shift

logger = logging.getLogger(__name__)

# Pre-computed anti-aliasing filter taps for scanner decimation
# Design once for typical 6 MHz -> 48 kHz decimation (125:1)
# Using Kaiser window with beta=6.0 for 60 dB stopband attenuation
# Cutoff at 0.8 / 125 = 0.0064 (normalized)
_SCANNER_DECIM_FILTER_TAPS: np.ndarray | None = None
_SCANNER_DECIM_FACTOR: int = 0


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
    _sync_pattern: np.ndarray | None = None

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

        # Log all measurements for debugging
        logger.info(f"[SCANNER] Scanned {len(measurements)} channels:")
        for freq_hz in sorted(measurements.keys()):
            m = measurements[freq_hz]
            sync_str = "SYNC" if m.sync_detected else "----"
            logger.info(
                f"[SCANNER]   {freq_hz/1e6:.4f} MHz: "
                f"power={m.power_db:.1f} dB, SNR={m.snr_db:.1f} dB, {sync_str}"
            )

        return measurements

    def _measure_channel(self, iq: np.ndarray, freq_hz: float) -> ChannelMeasurement:
        """Measure signal strength at a specific frequency.

        Uses adjacent-channel noise measurement for accurate SNR estimation.
        P25 control channels transmit continuously with constant-envelope
        modulation, so we can't use "quiet periods" for noise estimation.

        Args:
            iq: Wideband IQ samples
            freq_hz: Target frequency in Hz

        Returns:
            Channel measurement result
        """
        # Frequency shift to center on channel
        offset_hz = self.get_channel_offset(freq_hz)
        shifted_iq = freq_shift(iq, offset_hz, self.sample_rate)

        # Decimate to P25 rate (~48 kHz) for analysis with anti-aliasing filter
        target_rate = 48000
        decim_factor = max(1, self.sample_rate // target_rate)

        if decim_factor > 1:
            # Apply anti-aliasing lowpass filter before decimation
            # This prevents aliasing which would cause all channels to measure same power
            global _SCANNER_DECIM_FILTER_TAPS, _SCANNER_DECIM_FACTOR
            if _SCANNER_DECIM_FILTER_TAPS is None or _SCANNER_DECIM_FACTOR != decim_factor:
                # Design filter for this decimation factor
                # Cutoff at 0.8 / decim_factor to leave margin before Nyquist
                # Use 65 taps for reasonable stopband attenuation with low computational cost
                cutoff = 0.8 / decim_factor
                _SCANNER_DECIM_FILTER_TAPS = scipy_signal.firwin(
                    65, cutoff, window=("kaiser", 6.0)
                )
                _SCANNER_DECIM_FACTOR = decim_factor
                logger.info(
                    f"ControlChannelScanner: Created decimation filter: "
                    f"factor={decim_factor}, cutoff={cutoff:.4f}, taps={len(_SCANNER_DECIM_FILTER_TAPS)}"
                )

            # Apply lowpass filter then decimate
            filtered = scipy_signal.lfilter(_SCANNER_DECIM_FILTER_TAPS, 1.0, shifted_iq)
            decimated_iq = filtered[::decim_factor]
        else:
            decimated_iq = shifted_iq

        # Calculate signal power in channel
        signal_power = np.mean(np.abs(decimated_iq) ** 2)
        peak_power = np.max(np.abs(decimated_iq) ** 2)

        # Estimate noise floor from EDGES of capture bandwidth
        # This is more accurate than adjacent channels which may have other signals
        # With a 6 MHz sample rate centered at 415 MHz, edges are at ~412-418 MHz
        # Use ±2.8 MHz to stay within valid bandwidth but away from trunking signals
        noise_powers = []
        max_offset = self.sample_rate / 2 - 15000  # Stay 15 kHz from Nyquist edge

        # Sample noise at the edges of the capture bandwidth (away from signals)
        edge_offsets = [-max_offset + 25000, max_offset - 25000]  # 25 kHz from edges
        for edge_offset in edge_offsets:
            noise_iq = freq_shift(iq, edge_offset, self.sample_rate)
            if decim_factor > 1:
                noise_filtered = scipy_signal.lfilter(
                    _SCANNER_DECIM_FILTER_TAPS, 1.0, noise_iq
                )
                noise_decimated = noise_filtered[::decim_factor]
            else:
                noise_decimated = noise_iq
            noise_powers.append(np.mean(np.abs(noise_decimated) ** 2))

        if noise_powers:
            # Use the lower of the two edge measurements as noise floor
            # (in case one edge still has some signal)
            noise_floor = min(noise_powers)
        else:
            # Fallback: use quietest 5% of samples (less reliable)
            sorted_power = np.sort(np.abs(decimated_iq) ** 2)
            noise_samples = sorted_power[:max(1, len(sorted_power) // 20)]
            noise_floor = np.mean(noise_samples)

        # Convert to dB
        eps = 1e-12  # Avoid log(0)
        power_db = 10 * np.log10(signal_power + eps)
        peak_power_db = 10 * np.log10(peak_power + eps)
        noise_floor_db = 10 * np.log10(noise_floor + eps)
        snr_db = power_db - noise_floor_db

        # Check for P25 sync pattern
        # Only check sync on channels with minimum SNR (reject obvious noise)
        sync_detected = False
        min_snr_for_sync = 8.0  # dB - require at least 8 dB SNR for sync detection
        if self.sync_check_enabled and len(decimated_iq) > 0 and snr_db >= min_snr_for_sync:
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

        Uses C4FM FM demodulation with SOFT CORRELATION to detect the 24-dibit
        sync pattern. This avoids false positives on noise that occur with
        hard threshold decisions.

        The key insight: random noise FM demod output is uniformly distributed,
        which biases hard decisions toward outer dibits (1, 3). Since P25 sync
        only uses outer dibits, hard thresholding gives false positives on noise.

        Instead, we use soft correlation which measures how well the signal
        matches the expected ±3 pattern of the sync sequence.

        Args:
            iq: Decimated IQ samples (~48 kHz rate)

        Returns:
            True if sync pattern was detected
        """
        # P25 at 4800 symbols/sec, 48000 samples/sec = 10 samples/symbol
        samples_per_symbol = 10
        sync_pattern = self._sync_pattern
        if sync_pattern is None:
            return False
        sync_len = len(sync_pattern)

        if len(iq) < samples_per_symbol * sync_len + samples_per_symbol:
            return False

        # C4FM: FM demodulate by taking angle of product with conjugate of previous sample
        # This gives instantaneous frequency deviation
        fm_demod = np.angle(iq[1:] * np.conj(iq[:-1]))

        # Sample at symbol rate (take middle sample of each symbol)
        symbols_count = len(fm_demod) // samples_per_symbol
        if symbols_count < sync_len:
            return False

        symbol_samples = fm_demod[samples_per_symbol // 2::samples_per_symbol][:symbols_count]

        # P25 C4FM deviation: ±1.8 kHz for ±3 symbols, ±0.6 kHz for ±1 symbols
        # At 48 kHz sample rate: phase/sample = 2π × freq / 48000
        #   +3 level: 2π × 1800 / 48000 = 0.2356 rad
        #   -3 level: -0.2356 rad
        expected_deviation = 0.2356

        # Build the expected sync waveform as FM deviation values
        # Sync pattern dibits: 1=+3, 3=-3 (only uses outer symbols)
        sync_waveform = np.array([
            expected_deviation if d == 1 else -expected_deviation
            for d in sync_pattern
        ])

        # Use soft correlation: correlate FM demod output with expected waveform
        # For real P25 signal, correlation peak will be high
        # For noise, correlation will be low (random phases don't correlate)

        # Normalize both signals for correlation
        # Use a window that's slightly longer than sync to find peak
        search_len = min(len(symbol_samples) - sync_len, symbols_count - sync_len)
        if search_len <= 0:
            return False

        # Compute correlation at each position
        best_correlation = 0.0
        for i in range(search_len):
            window = symbol_samples[i:i + sync_len]
            # Normalized cross-correlation
            # For perfect match: correlation = 1.0
            # For random noise: correlation ≈ 0.0
            dot_product = np.sum(window * sync_waveform)
            norm_window = np.sqrt(np.sum(window ** 2) + 1e-10)
            norm_sync = np.sqrt(np.sum(sync_waveform ** 2))
            correlation = dot_product / (norm_window * norm_sync)
            if abs(correlation) > abs(best_correlation):
                best_correlation = correlation

        # Threshold: require correlation > 0.6 for sync detection
        # Real P25 sync should give correlation > 0.7 for clean signal
        # Noise can give correlation up to ~0.5 due to FM demod artifacts
        # Inverted polarity gives correlation near -0.6 (also valid)
        # Use 0.6 threshold to eliminate false positives on noise
        sync_threshold = 0.6

        if abs(best_correlation) > sync_threshold:
            return True

        return False

    def get_best_channel(self) -> tuple[float, ChannelMeasurement] | None:
        """Get the best control channel based on measurements.

        Prioritizes:
        1. Channels with detected P25 sync pattern
        2. For sync channels: highest POWER (active CC has constant modulation = high power)
        3. For non-sync channels: highest SNR (to find potential signals)

        IMPORTANT: Active P25 control channels show LOW SNR because they're
        constantly transmitting modulated data with no quiet periods. So we
        prioritize POWER for channels with sync detected.

        Returns:
            Tuple of (frequency_hz, measurement) or None if no usable channel
        """
        if not self._measurements:
            return None

        usable = list(self._measurements.items())
        if not usable:
            return None

        # Separate channels with and without sync
        with_sync = [(freq, m) for freq, m in usable if m.sync_detected]
        without_sync = [(freq, m) for freq, m in usable if not m.sync_detected]

        if with_sync:
            # For channels with sync: prioritize HIGHEST SNR
            # SNR is more important than power for reliable decoding
            with_sync.sort(key=lambda x: x[1].snr_db, reverse=True)
            best = with_sync[0]
            logger.info(
                f"[SCANNER] Best channel (sync): {best[0]/1e6:.4f} MHz, "
                f"SNR={best[1].snr_db:.1f} dB, power={best[1].power_db:.1f} dB"
            )
            return best

        # No sync detected - fall back to highest SNR
        # This helps find potential signals that might have sync on next scan
        without_sync.sort(key=lambda x: x[1].snr_db, reverse=True)
        best = without_sync[0]
        logger.info(
            f"[SCANNER] Best channel (no sync): {best[0]/1e6:.4f} MHz, "
            f"SNR={best[1].snr_db:.1f} dB (highest SNR without sync)"
        )
        return best

    def get_channel_ranking(self) -> list[tuple[float, ChannelMeasurement]]:
        """Get all channels ranked by signal quality.

        Channels with sync are ranked first (by SNR), then without sync (by SNR).

        Returns:
            List of (frequency_hz, measurement) sorted best to worst
        """
        if not self._measurements:
            return []

        # Separate by sync status
        with_sync = [(freq, m) for freq, m in self._measurements.items() if m.sync_detected]
        without_sync = [(freq, m) for freq, m in self._measurements.items() if not m.sync_detected]

        # Sort all channels by SNR (descending) - SNR is the best quality indicator
        with_sync.sort(key=lambda x: x[1].snr_db, reverse=True)
        without_sync.sort(key=lambda x: x[1].snr_db, reverse=True)

        # Combine: sync channels first (by SNR), then non-sync (by SNR)
        return with_sync + without_sync

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

    def get_stats(self) -> dict[str, Any]:
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
