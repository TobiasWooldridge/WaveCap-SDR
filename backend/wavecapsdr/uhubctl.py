"""USB hub power control via uhubctl.

Provides functionality to power cycle USB ports for SDR devices,
enabling hardware reset without physical intervention.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class USBDevice:
    """A USB device detected by uhubctl."""
    vendor_id: str
    product_id: str
    description: str
    serial: Optional[str] = None


@dataclass
class USBPort:
    """A USB port on a hub."""
    port_number: int
    powered: bool
    connected: bool
    device: Optional[USBDevice] = None


@dataclass
class USBHub:
    """A USB hub detected by uhubctl."""
    location: str
    vendor_id: str
    product_id: str
    description: str
    ports: list[USBPort]


def is_uhubctl_available() -> bool:
    """Check if uhubctl is installed and available."""
    return shutil.which("uhubctl") is not None


def get_hub_status() -> list[USBHub]:
    """Get status of all USB hubs and their ports.

    Returns:
        List of USBHub objects with port status.
    """
    if not is_uhubctl_available():
        logger.warning("uhubctl not found in PATH")
        return []

    try:
        result = subprocess.run(
            ["uhubctl"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        return _parse_uhubctl_output(result.stdout)
    except subprocess.TimeoutExpired:
        logger.error("uhubctl command timed out")
        return []
    except Exception as e:
        logger.error(f"Failed to get hub status: {e}")
        return []


def _parse_uhubctl_output(output: str) -> list[USBHub]:
    """Parse uhubctl output into structured data."""
    hubs: list[USBHub] = []
    current_hub: Optional[USBHub] = None

    # Pattern for hub line: "Current status for hub 0-1.4 [0bda:5411 Generic USB2.1 Hub, USB 2.10, 4 ports, ppps]"
    hub_pattern = re.compile(
        r"Current status for hub ([^\s]+) \[([0-9a-f]{4}):([0-9a-f]{4}) ([^,]+)"
    )

    # Pattern for port line: "  Port 2: 0503 power highspeed enable connect [0bda:2838 RTLSDRBlog Blog V4 00000001]"
    # More flexible pattern that handles various port states
    port_pattern = re.compile(
        r"^\s+Port (\d+): ([0-9a-f]+) (.+?)(?:\s*\[([0-9a-f]{4}):([0-9a-f]{4}) ([^\]]+)\])?$"
    )

    for line in output.splitlines():
        hub_match = hub_pattern.match(line)
        if hub_match:
            if current_hub:
                hubs.append(current_hub)
            current_hub = USBHub(
                location=hub_match.group(1),
                vendor_id=hub_match.group(2),
                product_id=hub_match.group(3),
                description=hub_match.group(4).strip(),
                ports=[],
            )
            continue

        port_match = port_pattern.match(line)
        if port_match and current_hub:
            port_num = int(port_match.group(1))
            status_text = port_match.group(3).lower()
            is_powered = "power" in status_text and "off" not in status_text
            has_device = port_match.group(4) is not None

            device = None
            if has_device:
                # Parse device info - description may contain serial at end
                desc = port_match.group(6).strip()
                serial = None
                # Check if last word looks like a serial number
                parts = desc.rsplit(" ", 1)
                if len(parts) == 2 and re.match(r"^[A-Za-z0-9]+$", parts[1]):
                    desc = parts[0]
                    serial = parts[1]

                device = USBDevice(
                    vendor_id=port_match.group(4),
                    product_id=port_match.group(5),
                    description=desc,
                    serial=serial,
                )

            current_hub.ports.append(USBPort(
                port_number=port_num,
                powered=is_powered,
                connected=has_device,
                device=device,
            ))

    if current_hub:
        hubs.append(current_hub)

    return hubs


def find_device_port(device_id: str) -> Optional[tuple[str, int]]:
    """Find the hub location and port number for a device.

    Args:
        device_id: SoapySDR device ID string (e.g., "driver=sdrplay,serial=240309F070")

    Returns:
        Tuple of (hub_location, port_number) or None if not found.
    """
    # Extract serial from device_id
    serial = None
    for part in device_id.split(","):
        if part.startswith("serial="):
            serial = part.split("=")[1]
            break

    if not serial:
        logger.warning(f"No serial found in device_id: {device_id}")
        return None

    hubs = get_hub_status()
    for hub in hubs:
        for port in hub.ports:
            if port.device and port.device.serial == serial:
                return (hub.location, port.port_number)

    # Also check RTL-SDR which may have serial in description
    for hub in hubs:
        for port in hub.ports:
            if port.device:
                # RTL-SDR serial may be in description
                if serial in (port.device.serial or "") or serial in port.device.description:
                    return (hub.location, port.port_number)

    logger.warning(f"Device with serial {serial} not found on any USB hub")
    return None


def power_cycle_port(hub_location: str, port_number: int, delay: float = 2.0) -> bool:
    """Power cycle a specific USB port.

    Args:
        hub_location: Hub location string (e.g., "0-1.4")
        port_number: Port number (1-based)
        delay: Delay in seconds between off and on

    Returns:
        True if successful, False otherwise.
    """
    if not is_uhubctl_available():
        logger.error("uhubctl not available")
        return False

    try:
        # Power cycle using uhubctl -a cycle
        result = subprocess.run(
            ["uhubctl", "-l", hub_location, "-p", str(port_number), "-a", "cycle", "-d", str(delay)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            logger.error(f"uhubctl power cycle failed: {result.stderr}")
            return False

        logger.info(f"Power cycled port {port_number} on hub {hub_location}")
        return True

    except subprocess.TimeoutExpired:
        logger.error("uhubctl power cycle timed out")
        return False
    except Exception as e:
        logger.error(f"Failed to power cycle port: {e}")
        return False


def power_cycle_device(device_id: str, delay: float = 2.0) -> tuple[bool, str]:
    """Power cycle the USB port for a specific device.

    Args:
        device_id: SoapySDR device ID string
        delay: Delay in seconds between off and on

    Returns:
        Tuple of (success, message)
    """
    if not is_uhubctl_available():
        return False, "uhubctl not installed. Install with: brew install uhubctl"

    port_info = find_device_port(device_id)
    if not port_info:
        return False, f"Device not found on any controllable USB hub"

    hub_location, port_number = port_info
    success = power_cycle_port(hub_location, port_number, delay)

    if success:
        return True, f"Power cycled port {port_number} on hub {hub_location}"
    else:
        return False, "Failed to power cycle USB port"


def power_cycle_all_ports(delay: float = 2.0) -> tuple[bool, str, int]:
    """Power cycle all ports on all controllable USB hubs.

    Args:
        delay: Delay in seconds between off and on

    Returns:
        Tuple of (success, message, ports_cycled)
    """
    if not is_uhubctl_available():
        return False, "uhubctl not installed. Install with: brew install uhubctl", 0

    hubs = get_hub_status()
    if not hubs:
        return False, "No controllable USB hubs found", 0

    ports_cycled = 0
    errors = []

    for hub in hubs:
        for port in hub.ports:
            if port.connected:  # Only cycle ports with connected devices
                success = power_cycle_port(hub.location, port.port_number, delay)
                if success:
                    ports_cycled += 1
                else:
                    errors.append(f"Port {port.port_number} on hub {hub.location}")

    if errors:
        return False, f"Failed to cycle some ports: {', '.join(errors)}", ports_cycled

    if ports_cycled == 0:
        return True, "No devices connected to controllable USB ports", 0

    return True, f"Power cycled {ports_cycled} USB port(s)", ports_cycled


def get_hub_status_dict() -> Dict[str, Any]:
    """Get hub status as a dictionary for API response."""
    hubs = get_hub_status()
    return {
        "available": is_uhubctl_available(),
        "hubs": [
            {
                "location": hub.location,
                "vendorId": hub.vendor_id,
                "productId": hub.product_id,
                "description": hub.description,
                "ports": [
                    {
                        "port": port.port_number,
                        "powered": port.powered,
                        "connected": port.connected,
                        "device": {
                            "vendorId": port.device.vendor_id,
                            "productId": port.device.product_id,
                            "description": port.device.description,
                            "serial": port.device.serial,
                        } if port.device else None,
                    }
                    for port in hub.ports
                ],
            }
            for hub in hubs
        ],
    }
