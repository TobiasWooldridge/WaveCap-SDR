"""Frequency band recognition and automatic channel naming.

This module provides intelligent frequency-to-name mapping for radio channels.
It uses a comprehensive frequency band database to recognize services and generate
contextual names like "Marine Channel 16" or "FM 90.3".
"""

import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class FrequencyInfo:
    """Information about a recognized frequency."""

    band_name: str
    frequency_hz: float
    suggested_name: str
    channel_number: Optional[str] = None
    description: Optional[str] = None
    service_type: Optional[str] = None


class FrequencyNamer:
    """Recognizes radio frequencies and generates contextual names."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the frequency namer with band database.

        Args:
            config_path: Path to frequency_bands.yaml, defaults to config/frequency_bands.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "frequency_bands.yaml"

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.bands = self.config.get('bands', {})

    def identify_frequency(self, frequency_hz: float, tolerance_hz: float = 5000) -> Optional[FrequencyInfo]:
        """Identify a frequency and generate a contextual name.

        Args:
            frequency_hz: Frequency in Hz
            tolerance_hz: Tolerance for matching fixed channels (default 5 kHz)

        Returns:
            FrequencyInfo object if frequency is recognized, None otherwise
        """
        # Search through all bands
        for band_id, band in self.bands.items():
            freq_min = band.get('freq_min')
            freq_max = band.get('freq_max')

            # Check if frequency is within band range
            if freq_min and freq_max and freq_min <= frequency_hz <= freq_max:
                naming_scheme = band.get('naming_scheme')

                if naming_scheme == 'fixed_channels':
                    # Look for exact channel match
                    return self._match_fixed_channel(frequency_hz, band, tolerance_hz)

                elif naming_scheme == 'calculated_channel':
                    # Calculate channel number mathematically
                    return self._calculate_channel_number(frequency_hz, band)

                elif naming_scheme == 'frequency_mhz':
                    # Use frequency as name
                    return self._format_frequency_mhz(frequency_hz, band)

                elif naming_scheme == 'frequency_khz':
                    # Use frequency in kHz
                    return self._format_frequency_khz(frequency_hz, band)

        return None

    def _match_fixed_channel(self, frequency_hz: float, band: Dict[str, Any], tolerance_hz: float) -> Optional[FrequencyInfo]:
        """Match against fixed channel definitions."""
        channels = band.get('channels', {})

        # Search for closest match within tolerance
        best_match = None
        min_diff = tolerance_hz

        for freq_str, channel_info in channels.items():
            channel_freq = float(freq_str)
            diff = abs(channel_freq - frequency_hz)

            if diff < min_diff:
                min_diff = diff
                best_match = (channel_freq, channel_info)

        if best_match:
            matched_freq, channel_info = best_match
            channel_num = channel_info.get('channel', '')
            channel_name = channel_info.get('name', '')

            # Build the suggested name
            band_name = band.get('name', '')
            if channel_name:
                suggested_name = f"{band_name} Ch {channel_num} - {channel_name}"
            else:
                suggested_name = f"{band_name} Ch {channel_num}"

            return FrequencyInfo(
                band_name=band_name,
                frequency_hz=matched_freq,
                suggested_name=suggested_name,
                channel_number=channel_num,
                description=band.get('description'),
                service_type=band.get('mode')
            )

        return None

    def _calculate_channel_number(self, frequency_hz: float, band: Dict[str, Any]) -> FrequencyInfo:
        """Calculate channel number from frequency using spacing."""
        base_freq = band.get('channel_base_freq')
        spacing = band.get('channel_spacing')
        channel_start = band.get('channel_start', 1)
        template = band.get('template', '{channel}')

        # Calculate channel number
        channel_offset = (frequency_hz - base_freq) / spacing
        channel_num = int(round(channel_start + channel_offset))

        # Format the name
        suggested_name = template.format(channel=channel_num)

        return FrequencyInfo(
            band_name=band.get('name', ''),
            frequency_hz=frequency_hz,
            suggested_name=suggested_name,
            channel_number=str(channel_num),
            description=band.get('description'),
            service_type=band.get('mode')
        )

    def _format_frequency_mhz(self, frequency_hz: float, band: Dict[str, Any]) -> FrequencyInfo:
        """Format frequency in MHz with appropriate precision."""
        freq_mhz = frequency_hz / 1e6
        template = band.get('template', '{freq_mhz:.3f} MHz')

        # Check for notable frequencies
        notable = band.get('notable_frequencies', {})
        for notable_freq, notable_name in notable.items():
            if abs(float(notable_freq) - frequency_hz) < 1000:  # Within 1 kHz
                suggested_name = f"{band.get('name', '')} {notable_name}"
                return FrequencyInfo(
                    band_name=band.get('name', ''),
                    frequency_hz=frequency_hz,
                    suggested_name=suggested_name,
                    description=band.get('description'),
                    service_type=band.get('mode')
                )

        # Format using template
        suggested_name = template.format(freq_mhz=freq_mhz)

        return FrequencyInfo(
            band_name=band.get('name', ''),
            frequency_hz=frequency_hz,
            suggested_name=suggested_name,
            description=band.get('description'),
            service_type=band.get('mode')
        )

    def _format_frequency_khz(self, frequency_hz: float, band: Dict[str, Any]) -> FrequencyInfo:
        """Format frequency in kHz."""
        freq_khz = int(frequency_hz / 1e3)
        template = band.get('template', '{freq_khz} kHz')

        suggested_name = template.format(freq_khz=freq_khz)

        return FrequencyInfo(
            band_name=band.get('name', ''),
            frequency_hz=frequency_hz,
            suggested_name=suggested_name,
            description=band.get('description'),
            service_type=band.get('mode')
        )

    def suggest_channel_name(self, capture_center_hz: float, offset_hz: float) -> Optional[str]:
        """Generate a suggested name for a channel given capture center and offset.

        Args:
            capture_center_hz: Capture center frequency in Hz
            offset_hz: Channel offset from center in Hz

        Returns:
            Suggested channel name, or None if frequency not recognized
        """
        channel_freq_hz = capture_center_hz + offset_hz
        freq_info = self.identify_frequency(channel_freq_hz)

        if freq_info:
            return freq_info.suggested_name

        return None


# Global instance for easy access
_frequency_namer: Optional[FrequencyNamer] = None


def get_frequency_namer() -> FrequencyNamer:
    """Get the global FrequencyNamer instance (lazy initialization)."""
    global _frequency_namer
    if _frequency_namer is None:
        _frequency_namer = FrequencyNamer()
    return _frequency_namer
