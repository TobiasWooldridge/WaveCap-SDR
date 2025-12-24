"""LRRP (Location Registration Request Protocol) Decoder for P25.

Extracts GPS location data from P25 radio transmissions:
- Extended Link Control (ELC) in LDU1 voice frames
- PDU frames with SNDCP/IP containing LRRP packets

Reference: TIA-102.BAHA (LRRP) specification
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

logger = logging.getLogger(__name__)


class LRRPOpcode(IntEnum):
    """LRRP message opcodes."""
    IMMEDIATE_LOC_REQUEST = 0x01
    IMMEDIATE_LOC_RESPONSE = 0x02
    TRIGGERED_LOC_REQUEST = 0x03
    TRIGGERED_LOC_START = 0x04
    TRIGGERED_LOC_STOP = 0x05
    TRIGGERED_LOC_RESPONSE = 0x06
    IMMEDIATE_INFO_REQUEST = 0x07
    IMMEDIATE_INFO_RESPONSE = 0x08


class LocInfoType(IntEnum):
    """Location information element types."""
    LOC_2D = 0x22  # Latitude/longitude
    LOC_3D = 0x33  # Latitude/longitude/altitude
    VELOCITY = 0x42  # Speed and heading
    ACCURACY = 0x52  # Horizontal position uncertainty


# P25 Extended Link Control formats with GPS
GPS_ELC_FORMATS = {
    0x09: "GPS Position",  # GPS location in ELC
    0x0A: "GPS Position Extended",  # Extended GPS with altitude
    0x0B: "GPS Position with Velocity",  # GPS + speed/heading
}


@dataclass
class RadioLocation:
    """GPS location report from a radio unit.

    Attributes:
        unit_id: Radio unit identifier
        latitude: Latitude in decimal degrees (-90 to 90)
        longitude: Longitude in decimal degrees (-180 to 180)
        altitude_m: Altitude in meters (if available)
        speed_kmh: Speed in km/h (if available)
        heading_deg: Heading in degrees (0-360, if available)
        accuracy_m: Horizontal position accuracy in meters (if available)
        timestamp: Unix timestamp when location was received
        source: Source of location data ("lrrp", "elc", "gps_tsbk")
    """
    unit_id: int
    latitude: float
    longitude: float
    altitude_m: float | None = None
    speed_kmh: float | None = None
    heading_deg: float | None = None
    accuracy_m: float | None = None
    timestamp: float = 0.0
    source: str = "unknown"

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def is_valid(self) -> bool:
        """Check if coordinates are valid."""
        return (
            -90 <= self.latitude <= 90 and
            -180 <= self.longitude <= 180 and
            not (self.latitude == 0.0 and self.longitude == 0.0)  # Null Island check
        )

    def age_seconds(self) -> float:
        """Get age of location report in seconds."""
        return time.time() - self.timestamp

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "unitId": self.unit_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude_m,
            "speed": self.speed_kmh,
            "heading": self.heading_deg,
            "accuracy": self.accuracy_m,
            "timestamp": self.timestamp,
            "ageSeconds": self.age_seconds(),
            "source": self.source,
        }


def decode_lrrp_coordinates(data: bytes) -> tuple[float, float]:
    """Decode LRRP latitude/longitude from binary format.

    LRRP uses a 24-bit signed representation for coordinates:
    - Latitude: -90 to +90 degrees (scale factor 90/2^23)
    - Longitude: -180 to +180 degrees (scale factor 180/2^23)

    Args:
        data: 6 bytes containing latitude (3 bytes) and longitude (3 bytes)

    Returns:
        Tuple of (latitude, longitude) in decimal degrees
    """
    if len(data) < 6:
        return (0.0, 0.0)

    # Latitude: 24-bit signed integer
    lat_raw = (data[0] << 16) | (data[1] << 8) | data[2]
    if lat_raw & 0x800000:  # Sign extend
        lat_raw -= 0x1000000
    latitude = lat_raw * 90.0 / (1 << 23)

    # Longitude: 24-bit signed integer
    lon_raw = (data[3] << 16) | (data[4] << 8) | data[5]
    if lon_raw & 0x800000:  # Sign extend
        lon_raw -= 0x1000000
    longitude = lon_raw * 180.0 / (1 << 23)

    return (latitude, longitude)


def decode_lrrp_altitude(data: bytes) -> float:
    """Decode LRRP altitude from binary format.

    Altitude is a 16-bit value representing meters with offset.

    Args:
        data: 2 bytes containing altitude

    Returns:
        Altitude in meters (can be negative)
    """
    if len(data) < 2:
        return 0.0

    # 16-bit unsigned with 500m offset (allows -500m to 65035m)
    alt_raw = (data[0] << 8) | data[1]
    return float(alt_raw) - 500.0


def decode_lrrp_velocity(data: bytes) -> tuple[float, float]:
    """Decode LRRP velocity (speed and heading) from binary format.

    Args:
        data: 3 bytes containing velocity information

    Returns:
        Tuple of (speed_kmh, heading_degrees)
    """
    if len(data) < 3:
        return (0.0, 0.0)

    # Speed: 8-bit value * 2 km/h
    speed_kmh = data[0] * 2.0

    # Heading: 9-bit value * (360/512) degrees
    heading_raw = (data[1] << 1) | (data[2] >> 7)
    heading_deg = heading_raw * 360.0 / 512.0

    return (speed_kmh, heading_deg)


def decode_lrrp_accuracy(data: bytes) -> float:
    """Decode LRRP horizontal position accuracy.

    Args:
        data: 1 byte containing accuracy class

    Returns:
        Accuracy in meters (approximate)
    """
    if len(data) < 1:
        return 0.0

    # Accuracy classes (exponential scale)
    accuracy_class = data[0] & 0x0F
    # Approximate meters: 2^(class-1) for class > 0
    if accuracy_class == 0:
        return 0.0  # Unknown
    return float(2 ** (accuracy_class - 1))


def decode_elc_gps(lcf: int, data: bytes, unit_id: int) -> RadioLocation | None:
    """Decode GPS from P25 Extended Link Control.

    Extended Link Control (ELC) can carry GPS data in voice LDU frames.
    LCF 0x09 is the standard GPS position format.

    Args:
        lcf: Link Control Format byte
        data: ELC data bytes (variable length based on LCF)
        unit_id: Source radio unit ID

    Returns:
        RadioLocation if GPS data was successfully decoded, None otherwise
    """
    if lcf not in GPS_ELC_FORMATS:
        return None

    try:
        if lcf == 0x09:  # Standard GPS Position
            # Format: 6 bytes lat/lon
            if len(data) < 6:
                return None
            lat, lon = decode_lrrp_coordinates(data[:6])

            return RadioLocation(
                unit_id=unit_id,
                latitude=lat,
                longitude=lon,
                source="elc"
            )

        elif lcf == 0x0A:  # Extended GPS with altitude
            # Format: 6 bytes lat/lon + 2 bytes altitude
            if len(data) < 8:
                return None
            lat, lon = decode_lrrp_coordinates(data[:6])
            alt = decode_lrrp_altitude(data[6:8])

            return RadioLocation(
                unit_id=unit_id,
                latitude=lat,
                longitude=lon,
                altitude_m=alt,
                source="elc"
            )

        elif lcf == 0x0B:  # GPS with velocity
            # Format: 6 bytes lat/lon + 3 bytes velocity
            if len(data) < 9:
                return None
            lat, lon = decode_lrrp_coordinates(data[:6])
            speed, heading = decode_lrrp_velocity(data[6:9])

            return RadioLocation(
                unit_id=unit_id,
                latitude=lat,
                longitude=lon,
                speed_kmh=speed,
                heading_deg=heading,
                source="elc"
            )

    except Exception as e:
        logger.warning(f"Failed to decode ELC GPS (LCF={lcf:02X}): {e}")

    return None


def decode_lrrp_packet(data: bytes, unit_id: int = 0) -> RadioLocation | None:
    """Decode a full LRRP packet from PDU data.

    LRRP packets are carried in P25 PDU frames over SNDCP/IP.

    Args:
        data: Raw LRRP packet data
        unit_id: Source radio unit ID (if known from call context)

    Returns:
        RadioLocation if successfully decoded, None otherwise
    """
    if len(data) < 4:
        return None

    try:
        # LRRP header
        (data[0] >> 6) & 0x03
        opcode = data[0] & 0x3F

        if opcode not in (LRRPOpcode.IMMEDIATE_LOC_RESPONSE, LRRPOpcode.TRIGGERED_LOC_RESPONSE):
            # Only process location responses
            return None

        # Source ID (if in packet)
        offset = 1
        if len(data) > offset + 3:
            # Some formats include unit ID in response
            packet_unit_id = (data[offset] << 16) | (data[offset + 1] << 8) | data[offset + 2]
            if packet_unit_id > 0:
                unit_id = packet_unit_id
            offset += 3

        # Parse location information elements
        lat = 0.0
        lon = 0.0
        alt = None
        speed = None
        heading = None
        accuracy = None

        while offset < len(data):
            if offset + 2 > len(data):
                break

            ie_type = data[offset]
            ie_len = data[offset + 1]
            offset += 2

            if offset + ie_len > len(data):
                break

            ie_data = data[offset:offset + ie_len]
            offset += ie_len

            if ie_type == LocInfoType.LOC_2D:
                if len(ie_data) >= 6:
                    lat, lon = decode_lrrp_coordinates(ie_data[:6])

            elif ie_type == LocInfoType.LOC_3D:
                if len(ie_data) >= 8:
                    lat, lon = decode_lrrp_coordinates(ie_data[:6])
                    alt = decode_lrrp_altitude(ie_data[6:8])

            elif ie_type == LocInfoType.VELOCITY:
                if len(ie_data) >= 3:
                    speed, heading = decode_lrrp_velocity(ie_data[:3])

            elif ie_type == LocInfoType.ACCURACY and len(ie_data) >= 1:
                accuracy = decode_lrrp_accuracy(ie_data[:1])

        if lat == 0.0 and lon == 0.0:
            return None

        return RadioLocation(
            unit_id=unit_id,
            latitude=lat,
            longitude=lon,
            altitude_m=alt,
            speed_kmh=speed,
            heading_deg=heading,
            accuracy_m=accuracy,
            source="lrrp"
        )

    except Exception as e:
        logger.warning(f"Failed to decode LRRP packet: {e}")

    return None


@dataclass
class LocationCache:
    """Cache for radio unit locations.

    Stores recent GPS locations from radios with configurable freshness.
    Used to attach location metadata to voice calls.
    """

    _locations: dict[int, RadioLocation] = None
    max_age_seconds: float = 300.0  # 5 minutes default

    def __post_init__(self) -> None:
        if self._locations is None:
            self._locations = {}

    def update(self, location: RadioLocation) -> None:
        """Update or add a location for a radio unit."""
        if not location.is_valid():
            return
        self._locations[location.unit_id] = location
        logger.info(
            f"Location updated: unit={location.unit_id} "
            f"lat={location.latitude:.6f} lon={location.longitude:.6f} "
            f"source={location.source}"
        )

    def get(self, unit_id: int) -> RadioLocation | None:
        """Get location for a unit if fresh enough."""
        location = self._locations.get(unit_id)
        if location is None:
            return None

        if location.age_seconds() > self.max_age_seconds:
            # Location too old
            return None

        return location

    def get_all(self) -> list[RadioLocation]:
        """Get all locations (including stale)."""
        return list(self._locations.values())

    def get_fresh(self) -> list[RadioLocation]:
        """Get all fresh locations."""
        return [
            loc for loc in self._locations.values()
            if loc.age_seconds() <= self.max_age_seconds
        ]

    def cleanup(self) -> int:
        """Remove stale locations. Returns count of removed entries."""
        stale_ids = [
            uid for uid, loc in self._locations.items()
            if loc.age_seconds() > self.max_age_seconds * 2  # Double the max age before cleanup
        ]
        for uid in stale_ids:
            del self._locations[uid]
        return len(stale_ids)

    def clear(self) -> None:
        """Clear all cached locations."""
        self._locations.clear()

    def to_dict(self) -> dict[str, Any]:
        """Export cache as dictionary."""
        return {
            "totalLocations": len(self._locations),
            "freshLocations": len(self.get_fresh()),
            "maxAgeSeconds": self.max_age_seconds,
            "locations": [loc.to_dict() for loc in self.get_fresh()],
        }
