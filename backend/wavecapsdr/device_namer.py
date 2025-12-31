"""Device naming utilities for generating human-friendly SDR device names.

This module provides functions to generate shorthand names from device IDs
and labels, making it easier to identify devices in the UI.
"""

import re
from typing import Callable, Optional, Union

# Type for replacement: either a string or a callable that takes a Match and returns a string
PatternReplacement = Union[str, Callable[[re.Match[str]], str]]

# Common SDR device patterns and their shorthand names
DEVICE_PATTERNS: list[tuple[str, PatternReplacement]] = [
    # RTL-SDR variants
    (r"rtl[-_]?sdr", "RTL-SDR"),
    (r"rtl2832", "RTL-SDR"),
    (r"blog\s*v4", "RTL-SDR V4"),
    (r"blog\s*v3", "RTL-SDR V3"),
    # SDRplay
    (r"sdrplay.*rsp(?:dx|1a|1b|2|duo|2pro)", lambda m: f"SDRplay {m.group(0).split()[-1].upper()}"),
    (r"sdrplay.*rsp\s*(\w+)", lambda m: f"SDRplay RSP{m.group(1).upper()}"),
    (r"sdrplay", "SDRplay"),
    # HackRF
    (r"hackrf\s+one", "HackRF One"),
    (r"hackrf", "HackRF"),
    # Airspy
    (r"airspy\s+mini", "Airspy Mini"),
    (r"airspy\s+hf\+", "Airspy HF+"),
    (r"airspy\s+r2", "Airspy R2"),
    (r"airspy", "Airspy"),
    # LimeSDR
    (r"lime\s?sdr[\s-]?mini", "LimeSDR Mini"),
    (r"lime\s?sdr", "LimeSDR"),
    # USRP
    (r"usrp\s+[bn]\d+", lambda m: m.group(0).upper()),
    (r"usrp", "USRP"),
    # PlutoSDR
    (r"plutosdr", "PlutoSDR"),
    (r"pluto", "PlutoSDR"),
    # BladeRF
    (r"bladerf\s+2\.0", "bladeRF 2.0"),
    (r"bladerf", "bladeRF"),
    # FunCube Dongle
    (r"funcube.*pro\+", "FunCube Pro+"),
    (r"funcube.*pro", "FunCube Pro"),
    (r"funcube", "FunCube"),
    # Generic driver fallbacks
    (r"driver=(\w+)", lambda m: m.group(1).upper()),
]


def get_device_shorthand(device_id: str, device_label: str) -> str:
    """Generate a shorthand name for an SDR device.

    Args:
        device_id: Device ID string (e.g., "driver=rtlsdr,serial=00000001")
        device_label: Device label from SoapySDR

    Returns:
        Shorthand name (e.g., "RTL-SDR", "SDRplay RSPdx")
    """
    # Combine ID and label for matching
    combined = f"{device_id} {device_label}".lower()

    # Try each pattern
    for pattern, replacement in DEVICE_PATTERNS:
        match = re.search(pattern, combined, re.IGNORECASE)
        if match:
            if callable(replacement):
                return replacement(match)
            else:
                return replacement

    # Fallback: Extract serial or device number from label
    # E.g., "Generic RTL2832U OEM :: 00000001" -> "SDR-00000001"
    serial_match = re.search(r"(?:serial[=:]?\s*)?([0-9A-F]{6,})", combined, re.IGNORECASE)
    if serial_match:
        serial = serial_match.group(1)
        # Shorten long serials
        if len(serial) > 8:
            serial = serial[-8:]
        return f"SDR-{serial}"

    # Last resort: Use driver name + index
    driver_match = re.search(r"driver=(\w+)", device_id, re.IGNORECASE)
    if driver_match:
        driver = driver_match.group(1).upper()
        # Try to extract device number from label
        dev_num_match = re.search(r"dev[ice]?\s*(\d+)", device_label, re.IGNORECASE)
        if dev_num_match:
            return f"{driver} #{dev_num_match.group(1)}"
        return driver

    # Ultimate fallback
    return "SDR Device"


def generate_capture_name(
    center_hz: float,
    device_id: str,
    device_label: str,
    recipe_name: Optional[str] = None,
    device_nickname: Optional[str] = None,
) -> str:
    """Generate a default name for a capture.

    Args:
        center_hz: Center frequency in Hz
        device_id: Device ID string
        device_label: Device label from SoapySDR
        recipe_name: Name of recipe used (if any)
        device_nickname: Custom nickname for device (if set)

    Returns:
        Generated capture name (e.g., "FM Radio - RTL-SDR", "90.3 MHz - SDRplay")
    """
    # Use custom nickname if available, otherwise generate shorthand
    device_name = device_nickname or get_device_shorthand(device_id, device_label)

    if recipe_name:
        # Use recipe name as base
        return f"{recipe_name} - {device_name}"

    # Format frequency
    freq_mhz = center_hz / 1e6

    # Common bands with nice formatting
    if 87.5e6 <= center_hz <= 108e6:
        # FM Broadcast
        return f"FM {freq_mhz:.1f} - {device_name}"
    elif 118e6 <= center_hz <= 137e6:
        # Airband
        return f"Air {freq_mhz:.3f} - {device_name}"
    elif 144e6 <= center_hz <= 148e6:
        # 2m Ham
        return f"2m {freq_mhz:.3f} - {device_name}"
    elif 156e6 <= center_hz <= 158e6:
        # Marine VHF
        return f"Marine {freq_mhz:.3f} - {device_name}"
    elif 420e6 <= center_hz <= 450e6:
        # 70cm Ham
        return f"70cm {freq_mhz:.3f} - {device_name}"
    elif 460e6 <= center_hz <= 470e6:
        # GMRS/FRS
        return f"GMRS {freq_mhz:.3f} - {device_name}"
    else:
        # Generic frequency
        if freq_mhz < 1:
            # Sub-MHz, use kHz
            freq_khz = center_hz / 1e3
            return f"{freq_khz:.0f} kHz - {device_name}"
        elif freq_mhz >= 1000:
            # GHz range
            freq_ghz = center_hz / 1e9
            return f"{freq_ghz:.3f} GHz - {device_name}"
        else:
            return f"{freq_mhz:.3f} MHz - {device_name}"


# Global device nicknames cache (device_id -> nickname)
_device_nicknames: dict[str, str] = {}


def set_device_nickname(device_id: str, nickname: str) -> None:
    """Set a custom nickname for a device.

    Args:
        device_id: Device ID string
        nickname: Custom nickname (or empty string to clear)
    """
    if nickname:
        _device_nicknames[device_id] = nickname
    elif device_id in _device_nicknames:
        del _device_nicknames[device_id]


def get_device_nickname(device_id: str) -> Optional[str]:
    """Get custom nickname for a device.

    Args:
        device_id: Device ID string

    Returns:
        Custom nickname if set, None otherwise
    """
    return _device_nicknames.get(device_id)


def load_device_nicknames(nicknames: dict[str, str]) -> None:
    """Load device nicknames from configuration.

    Args:
        nicknames: Dictionary mapping device_id -> nickname
    """
    global _device_nicknames
    _device_nicknames = dict(nicknames)
