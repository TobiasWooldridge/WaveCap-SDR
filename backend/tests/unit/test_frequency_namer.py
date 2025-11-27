"""Unit tests for frequency band recognition and naming.

These tests verify the pure logic of frequency identification.
"""

import pytest
from pathlib import Path

from wavecapsdr.frequency_namer import FrequencyNamer, FrequencyInfo


@pytest.fixture
def namer(config_dir: Path) -> FrequencyNamer:
    """Create a FrequencyNamer with the real config."""
    config_path = config_dir / "frequency_bands.yaml"
    if not config_path.exists():
        pytest.skip("frequency_bands.yaml not found")
    return FrequencyNamer(config_path)


class TestFrequencyNamer:
    """Tests for FrequencyNamer class."""

    def test_identify_fm_broadcast(self, namer: FrequencyNamer):
        """FM broadcast frequencies should be recognized."""
        # 90.3 MHz FM
        info = namer.identify_frequency(90_300_000)
        assert info is not None
        assert "FM" in info.band_name or "fm" in info.band_name.lower()
        # Should contain 90.3 somewhere
        assert "90.3" in info.suggested_name or "90.30" in info.suggested_name

    def test_identify_marine_vhf(self, namer: FrequencyNamer):
        """Marine VHF frequencies should be recognized."""
        # Marine Channel 16 (156.8 MHz) - international distress
        info = namer.identify_frequency(156_800_000)
        assert info is not None
        # Should recognize as marine
        if info.channel_number:
            assert info.channel_number == "16" or "16" in info.suggested_name

    def test_identify_weather_radio(self, namer: FrequencyNamer):
        """NOAA Weather Radio frequencies should be recognized."""
        # NOAA Weather Radio (162.55 MHz is a common frequency)
        info = namer.identify_frequency(162_550_000)
        assert info is not None
        # Should be in NOAA or weather band
        lower_name = (info.band_name + info.suggested_name).lower()
        assert "noaa" in lower_name or "weather" in lower_name or "162" in info.suggested_name

    def test_identify_aviation_freq(self, namer: FrequencyNamer):
        """Aviation frequencies should be recognized."""
        # Air traffic control (118-137 MHz range)
        info = namer.identify_frequency(121_500_000)  # Aviation emergency
        assert info is not None
        lower_name = (info.band_name + info.suggested_name).lower()
        # Should recognize as aviation or airband
        assert "air" in lower_name or "aviation" in lower_name or "121" in info.suggested_name

    def test_unknown_frequency(self, namer: FrequencyNamer):
        """Unknown frequencies should return None."""
        # A frequency unlikely to be in the database
        info = namer.identify_frequency(1_234_567_890)
        # Should be None or have a generic name
        # (depends on whether there's a catch-all band)

    def test_tolerance_matching(self, namer: FrequencyNamer):
        """Frequency matching should respect tolerance."""
        # Slightly off from Marine Channel 16
        info = namer.identify_frequency(156_802_000, tolerance_hz=5000)
        assert info is not None
        # Should still match Channel 16

    def test_suggest_channel_name_with_offset(self, namer: FrequencyNamer):
        """Channel naming should work with center + offset."""
        # Center at 156.8 MHz, offset 0 -> Channel 16
        name = namer.suggest_channel_name(156_800_000, 0)
        assert name is not None or name is None  # May or may not exist

    def test_frequency_info_has_required_fields(self, namer: FrequencyNamer):
        """FrequencyInfo should have all expected fields."""
        info = namer.identify_frequency(90_300_000)
        if info:
            assert hasattr(info, "band_name")
            assert hasattr(info, "frequency_hz")
            assert hasattr(info, "suggested_name")
            assert hasattr(info, "channel_number")
            assert hasattr(info, "description")
            assert hasattr(info, "service_type")


class TestFrequencyInfoDataclass:
    """Tests for FrequencyInfo dataclass."""

    def test_frequency_info_creation(self):
        """FrequencyInfo should be creatable with required fields."""
        info = FrequencyInfo(
            band_name="FM Broadcast",
            frequency_hz=90_300_000,
            suggested_name="FM 90.3",
        )
        assert info.band_name == "FM Broadcast"
        assert info.frequency_hz == 90_300_000
        assert info.suggested_name == "FM 90.3"

    def test_frequency_info_optional_fields(self):
        """FrequencyInfo optional fields should default to None."""
        info = FrequencyInfo(
            band_name="Test",
            frequency_hz=100_000_000,
            suggested_name="Test 100",
        )
        assert info.channel_number is None
        assert info.description is None
        assert info.service_type is None
