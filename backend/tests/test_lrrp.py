"""Tests for LRRP (Location Registration Request Protocol) decoder.

Tests GPS coordinate decoding, location caching, and Extended Link Control
GPS extraction for P25 trunking systems.

Reference: SDRTrunk (https://github.com/DSheirer/sdrtrunk)
"""

import time
from unittest.mock import patch

import pytest

from wavecapsdr.decoders.lrrp import (
    LocationCache,
    RadioLocation,
    decode_lrrp_coordinates,
    decode_lrrp_altitude,
    decode_lrrp_velocity,
    decode_lrrp_accuracy,
    decode_elc_gps,
    decode_lrrp_packet,
    LRRPOpcode,
    LocInfoType,
)


class TestLRRPCoordinates:
    """Test LRRP coordinate encoding/decoding."""

    def test_decode_positive_coordinates(self):
        """Decode positive lat/lon (Seattle area)."""
        # Latitude 47.6062 = 47.6062 * (2^23 / 90) = 0x2B_B0_A3
        # Longitude -122.3321 = -122.3321 * (2^23 / 180) = 0xB4_D9_9A (as signed)
        # For testing, use simpler values

        # lat = 45.0 degrees -> 45/90 * 2^23 = 4194304 = 0x400000
        # lon = 90.0 degrees -> 90/180 * 2^23 = 4194304 = 0x400000
        data = bytes([0x40, 0x00, 0x00, 0x40, 0x00, 0x00])
        lat, lon = decode_lrrp_coordinates(data)

        assert abs(lat - 45.0) < 0.0001
        assert abs(lon - 90.0) < 0.0001

    def test_decode_negative_coordinates(self):
        """Decode negative lat/lon (Southern/Western hemisphere)."""
        # lat = -45.0 degrees -> -45/90 * 2^23 = -4194304 = 0xC00000
        # lon = -90.0 degrees -> -90/180 * 2^23 = -4194304 = 0xC00000
        data = bytes([0xC0, 0x00, 0x00, 0xC0, 0x00, 0x00])
        lat, lon = decode_lrrp_coordinates(data)

        assert abs(lat - (-45.0)) < 0.0001
        assert abs(lon - (-90.0)) < 0.0001

    def test_decode_zero_coordinates(self):
        """Decode zero coordinates (null island)."""
        data = bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        lat, lon = decode_lrrp_coordinates(data)

        assert lat == 0.0
        assert lon == 0.0

    def test_decode_max_positive(self):
        """Decode maximum positive coordinates."""
        # Max lat = 90 degrees -> 0x7FFFFF
        # Max lon = 180 degrees -> 0x7FFFFF (but scaled differently)
        data = bytes([0x7F, 0xFF, 0xFF, 0x7F, 0xFF, 0xFF])
        lat, lon = decode_lrrp_coordinates(data)

        # 0x7FFFFF / 2^23 * 90 = 89.99999...
        assert lat > 89.0
        assert lat <= 90.0
        assert lon > 179.0
        assert lon <= 180.0

    def test_decode_max_negative(self):
        """Decode maximum negative coordinates."""
        # Min lat = -90 degrees -> 0x800000
        # Min lon = -180 degrees -> 0x800000
        data = bytes([0x80, 0x00, 0x00, 0x80, 0x00, 0x00])
        lat, lon = decode_lrrp_coordinates(data)

        assert lat == -90.0
        assert lon == -180.0

    def test_decode_short_data(self):
        """Handle too-short data gracefully."""
        data = bytes([0x40, 0x00, 0x00])  # Only 3 bytes
        lat, lon = decode_lrrp_coordinates(data)

        assert lat == 0.0
        assert lon == 0.0

    def test_decode_empty_data(self):
        """Handle empty data."""
        lat, lon = decode_lrrp_coordinates(bytes())

        assert lat == 0.0
        assert lon == 0.0


class TestLRRPAltitude:
    """Test altitude decoding."""

    def test_decode_sea_level(self):
        """Decode sea level altitude (500m offset)."""
        # 500 meters raw = 0 altitude with 500m offset
        data = bytes([0x01, 0xF4])  # 500 in big-endian
        alt = decode_lrrp_altitude(data)

        assert alt == 0.0

    def test_decode_positive_altitude(self):
        """Decode positive altitude."""
        # 1500 raw - 500 offset = 1000m
        data = bytes([0x05, 0xDC])  # 1500 in big-endian
        alt = decode_lrrp_altitude(data)

        assert alt == 1000.0

    def test_decode_negative_altitude(self):
        """Decode negative altitude (below sea level)."""
        # 100 raw - 500 offset = -400m
        data = bytes([0x00, 0x64])  # 100 in big-endian
        alt = decode_lrrp_altitude(data)

        assert alt == -400.0

    def test_decode_short_data(self):
        """Handle too-short data."""
        alt = decode_lrrp_altitude(bytes([0x00]))
        assert alt == 0.0


class TestLRRPVelocity:
    """Test velocity (speed + heading) decoding."""

    def test_decode_zero_velocity(self):
        """Decode zero speed and heading."""
        data = bytes([0x00, 0x00, 0x00])
        speed, heading = decode_lrrp_velocity(data)

        assert speed == 0.0
        assert heading == 0.0

    def test_decode_typical_velocity(self):
        """Decode typical driving speed and heading."""
        # Speed: 30 * 2 = 60 km/h
        # Heading: 256 * (360/512) = 180 degrees (south)
        # Heading raw: 256 = 0x100 -> split as (0x80, 0x00)
        data = bytes([0x1E, 0x80, 0x00])  # 30 speed, 256 heading
        speed, heading = decode_lrrp_velocity(data)

        assert speed == 60.0
        assert abs(heading - 180.0) < 0.1

    def test_decode_max_speed(self):
        """Decode maximum speed."""
        # 255 * 2 = 510 km/h
        data = bytes([0xFF, 0x00, 0x00])
        speed, heading = decode_lrrp_velocity(data)

        assert speed == 510.0

    def test_decode_short_data(self):
        """Handle too-short data."""
        speed, heading = decode_lrrp_velocity(bytes([0x00]))
        assert speed == 0.0
        assert heading == 0.0


class TestLRRPAccuracy:
    """Test accuracy class decoding."""

    def test_decode_unknown_accuracy(self):
        """Class 0 = unknown accuracy."""
        acc = decode_lrrp_accuracy(bytes([0x00]))
        assert acc == 0.0

    def test_decode_accuracy_classes(self):
        """Test exponential accuracy scaling."""
        # Class 1 = 1m, class 2 = 2m, class 3 = 4m, etc.
        assert decode_lrrp_accuracy(bytes([0x01])) == 1.0
        assert decode_lrrp_accuracy(bytes([0x02])) == 2.0
        assert decode_lrrp_accuracy(bytes([0x03])) == 4.0
        assert decode_lrrp_accuracy(bytes([0x04])) == 8.0
        assert decode_lrrp_accuracy(bytes([0x05])) == 16.0

    def test_decode_empty_data(self):
        """Handle empty data."""
        acc = decode_lrrp_accuracy(bytes())
        assert acc == 0.0


class TestRadioLocation:
    """Test RadioLocation dataclass."""

    def test_create_location(self):
        """Create a valid location."""
        loc = RadioLocation(
            unit_id=12345678,
            latitude=47.6062,
            longitude=-122.3321,
            source="elc",
        )

        assert loc.unit_id == 12345678
        assert loc.latitude == 47.6062
        assert loc.longitude == -122.3321
        assert loc.source == "elc"
        assert loc.timestamp > 0

    def test_is_valid_good_coords(self):
        """Valid coordinates pass validation."""
        loc = RadioLocation(
            unit_id=1,
            latitude=47.6,
            longitude=-122.3,
        )
        assert loc.is_valid()

    def test_is_valid_null_island(self):
        """Null island (0, 0) is invalid."""
        loc = RadioLocation(
            unit_id=1,
            latitude=0.0,
            longitude=0.0,
        )
        assert not loc.is_valid()

    def test_is_valid_out_of_range_lat(self):
        """Latitude out of range is invalid."""
        loc = RadioLocation(
            unit_id=1,
            latitude=91.0,
            longitude=-122.3,
        )
        assert not loc.is_valid()

    def test_is_valid_out_of_range_lon(self):
        """Longitude out of range is invalid."""
        loc = RadioLocation(
            unit_id=1,
            latitude=47.6,
            longitude=-181.0,
        )
        assert not loc.is_valid()

    def test_age_seconds(self):
        """Age calculation works."""
        loc = RadioLocation(
            unit_id=1,
            latitude=47.6,
            longitude=-122.3,
            timestamp=time.time() - 60,  # 60 seconds ago
        )

        age = loc.age_seconds()
        assert 59 < age < 62  # Allow some tolerance

    def test_to_dict(self):
        """Serialization to dict works."""
        loc = RadioLocation(
            unit_id=12345,
            latitude=47.6,
            longitude=-122.3,
            altitude_m=100.0,
            speed_kmh=60.0,
            heading_deg=180.0,
            accuracy_m=5.0,
            source="lrrp",
        )

        d = loc.to_dict()

        assert d["unitId"] == 12345
        assert d["latitude"] == 47.6
        assert d["longitude"] == -122.3
        assert d["altitude"] == 100.0
        assert d["speed"] == 60.0
        assert d["heading"] == 180.0
        assert d["accuracy"] == 5.0
        assert d["source"] == "lrrp"
        assert "timestamp" in d
        assert "ageSeconds" in d


class TestLocationCache:
    """Test LocationCache for radio GPS data."""

    def test_update_and_get(self):
        """Store and retrieve location."""
        cache = LocationCache(max_age_seconds=300)

        loc = RadioLocation(
            unit_id=12345,
            latitude=47.6,
            longitude=-122.3,
        )

        cache.update(loc)
        retrieved = cache.get(12345)

        assert retrieved is not None
        assert retrieved.unit_id == 12345
        assert retrieved.latitude == 47.6

    def test_get_nonexistent(self):
        """Get returns None for unknown unit."""
        cache = LocationCache(max_age_seconds=300)

        assert cache.get(99999) is None

    def test_reject_invalid_location(self):
        """Invalid locations are not cached."""
        cache = LocationCache(max_age_seconds=300)

        # Null island is invalid
        loc = RadioLocation(unit_id=1, latitude=0.0, longitude=0.0)
        cache.update(loc)

        assert cache.get(1) is None

    def test_staleness_check(self):
        """Stale locations are not returned."""
        cache = LocationCache(max_age_seconds=1)  # 1 second max age

        loc = RadioLocation(
            unit_id=12345,
            latitude=47.6,
            longitude=-122.3,
            timestamp=time.time() - 10,  # 10 seconds old
        )
        cache._locations[12345] = loc  # Direct insert to bypass validation

        # Should return None because it's stale
        assert cache.get(12345) is None

    def test_get_all_locations(self):
        """Get all stored locations."""
        cache = LocationCache(max_age_seconds=300)

        for i in range(3):
            loc = RadioLocation(
                unit_id=i + 1,
                latitude=47.0 + i,
                longitude=-122.0,
            )
            cache.update(loc)

        all_locs = cache.get_all()
        assert len(all_locs) == 3

    def test_get_fresh_only(self):
        """Get only fresh locations."""
        cache = LocationCache(max_age_seconds=60)

        # Add fresh location
        fresh = RadioLocation(unit_id=1, latitude=47.0, longitude=-122.0)
        cache.update(fresh)

        # Add stale location directly
        stale = RadioLocation(
            unit_id=2,
            latitude=48.0,
            longitude=-122.0,
            timestamp=time.time() - 120,  # 2 minutes old
        )
        cache._locations[2] = stale

        fresh_only = cache.get_fresh()
        assert len(fresh_only) == 1
        assert fresh_only[0].unit_id == 1

    def test_cleanup(self):
        """Cleanup removes very stale entries."""
        cache = LocationCache(max_age_seconds=60)

        # Add entry that's 3x max age (should be cleaned)
        very_stale = RadioLocation(
            unit_id=1,
            latitude=47.0,
            longitude=-122.0,
            timestamp=time.time() - 200,  # 200 seconds old
        )
        cache._locations[1] = very_stale

        # Add fresh entry (should not be cleaned)
        fresh = RadioLocation(unit_id=2, latitude=48.0, longitude=-122.0)
        cache.update(fresh)

        removed = cache.cleanup()

        assert removed == 1
        assert cache.get(1) is None  # Cleaned
        assert cache.get(2) is not None  # Still there (fresh)

    def test_clear(self):
        """Clear removes all entries."""
        cache = LocationCache(max_age_seconds=300)

        for i in range(3):
            loc = RadioLocation(unit_id=i + 1, latitude=47.0, longitude=-122.0)
            cache.update(loc)

        cache.clear()

        assert len(cache.get_all()) == 0

    def test_to_dict(self):
        """Serialization to dict works."""
        cache = LocationCache(max_age_seconds=300)

        loc = RadioLocation(unit_id=1, latitude=47.0, longitude=-122.0)
        cache.update(loc)

        d = cache.to_dict()

        assert d["totalLocations"] == 1
        assert d["freshLocations"] == 1
        assert d["maxAgeSeconds"] == 300
        assert len(d["locations"]) == 1


class TestDecodeELCGPS:
    """Test Extended Link Control GPS extraction."""

    def test_decode_lcf_0x09_basic_gps(self):
        """Decode LCF 0x09 (basic GPS position)."""
        # lat = 45.0, lon = 90.0
        data = bytes([0x40, 0x00, 0x00, 0x40, 0x00, 0x00])

        loc = decode_elc_gps(0x09, data, unit_id=12345)

        assert loc is not None
        assert loc.unit_id == 12345
        assert abs(loc.latitude - 45.0) < 0.001
        assert abs(loc.longitude - 90.0) < 0.001
        assert loc.source == "elc"

    def test_decode_lcf_0x0a_with_altitude(self):
        """Decode LCF 0x0A (GPS with altitude)."""
        # lat = 45.0, lon = 90.0, alt = 1000m (raw 1500)
        data = bytes([0x40, 0x00, 0x00, 0x40, 0x00, 0x00, 0x05, 0xDC])

        loc = decode_elc_gps(0x0A, data, unit_id=12345)

        assert loc is not None
        assert abs(loc.latitude - 45.0) < 0.001
        assert loc.altitude_m == 1000.0

    def test_decode_lcf_0x0b_with_velocity(self):
        """Decode LCF 0x0B (GPS with velocity)."""
        # lat = 45.0, lon = 90.0, speed = 60 km/h, heading = 180 deg
        data = bytes([0x40, 0x00, 0x00, 0x40, 0x00, 0x00, 0x1E, 0x80, 0x00])

        loc = decode_elc_gps(0x0B, data, unit_id=12345)

        assert loc is not None
        assert abs(loc.latitude - 45.0) < 0.001
        assert loc.speed_kmh == 60.0
        assert abs(loc.heading_deg - 180.0) < 0.1

    def test_decode_non_gps_lcf(self):
        """Non-GPS LCF returns None."""
        data = bytes([0x00] * 10)

        loc = decode_elc_gps(0x00, data, unit_id=12345)  # LCF 0x00 is voice user

        assert loc is None

    def test_decode_short_data(self):
        """Too-short data returns None."""
        data = bytes([0x40, 0x00])  # Only 2 bytes

        loc = decode_elc_gps(0x09, data, unit_id=12345)

        assert loc is None


class TestDecodeLRRPPacket:
    """Test full LRRP packet decoding."""

    def test_decode_location_response(self):
        """Decode immediate location response packet."""
        # Build a minimal LRRP packet
        # Opcode 0x02 = IMMEDIATE_LOC_RESPONSE
        # Version 0, opcode 0x02 = 0x02
        # Unit ID (3 bytes)
        # Location IE: type 0x22, length 6, lat/lon
        packet = bytes([
            0x02,  # Version 0, opcode 0x02
            0x00, 0x00, 0x01,  # Unit ID = 1
            0x22, 0x06,  # LOC_2D, length 6
            0x40, 0x00, 0x00,  # lat = 45.0
            0x40, 0x00, 0x00,  # lon = 90.0
        ])

        loc = decode_lrrp_packet(packet, unit_id=0)

        assert loc is not None
        assert loc.unit_id == 1  # From packet
        assert abs(loc.latitude - 45.0) < 0.001
        assert loc.source == "lrrp"

    def test_decode_triggered_location_response(self):
        """Decode triggered location response packet."""
        packet = bytes([
            0x06,  # Version 0, opcode 0x06 (TRIGGERED_LOC_RESPONSE)
            0x00, 0x00, 0x02,  # Unit ID = 2
            0x22, 0x06,  # LOC_2D
            0x40, 0x00, 0x00, 0x40, 0x00, 0x00,
        ])

        loc = decode_lrrp_packet(packet, unit_id=0)

        assert loc is not None
        assert loc.unit_id == 2

    def test_decode_with_3d_location(self):
        """Decode packet with 3D location (includes altitude)."""
        packet = bytes([
            0x02,
            0x00, 0x00, 0x01,
            0x33, 0x08,  # LOC_3D, length 8
            0x40, 0x00, 0x00,  # lat
            0x40, 0x00, 0x00,  # lon
            0x05, 0xDC,  # alt = 1000m
        ])

        loc = decode_lrrp_packet(packet)

        assert loc is not None
        assert loc.altitude_m == 1000.0

    def test_decode_with_velocity(self):
        """Decode packet with velocity IE."""
        packet = bytes([
            0x02,
            0x00, 0x00, 0x01,
            0x22, 0x06,  # LOC_2D
            0x40, 0x00, 0x00, 0x40, 0x00, 0x00,
            0x42, 0x03,  # VELOCITY, length 3
            0x1E, 0x80, 0x00,  # 60 km/h, 180 deg
        ])

        loc = decode_lrrp_packet(packet)

        assert loc is not None
        assert loc.speed_kmh == 60.0
        assert abs(loc.heading_deg - 180.0) < 0.1

    def test_decode_non_location_opcode(self):
        """Non-location opcode returns None."""
        packet = bytes([
            0x01,  # IMMEDIATE_LOC_REQUEST - not a response
            0x00, 0x00, 0x01,
        ])

        loc = decode_lrrp_packet(packet)

        assert loc is None

    def test_decode_empty_packet(self):
        """Empty packet returns None."""
        loc = decode_lrrp_packet(bytes())
        assert loc is None

    def test_decode_short_packet(self):
        """Too-short packet returns None."""
        loc = decode_lrrp_packet(bytes([0x02, 0x00]))
        assert loc is None
