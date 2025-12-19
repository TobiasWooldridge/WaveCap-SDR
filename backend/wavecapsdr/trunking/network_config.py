"""P25 network configuration monitor.

Tracks system configuration learned from control channel broadcasts:
- Frequency band definitions (IDEN_UP)
- System/site status (RFSS_STS_BCAST, NET_STS_BCAST)
- Adjacent sites (ADJ_STS_BCAST)
- Service availability (SYS_SRV_BCAST)

Inspired by SDRTrunk's P25P1NetworkConfigurationMonitor.

Reference: https://github.com/DSheirer/sdrtrunk
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FrequencyBand:
    """Frequency band definition from IDEN_UP message.

    Used to calculate actual frequencies from channel numbers.
    P25 uses a 4-bit identifier + 12-bit channel number system
    where the identifier selects the frequency band.
    """
    identifier: int  # 4-bit band identifier (0-15)
    base_frequency_hz: int  # Base frequency in Hz
    channel_spacing_hz: int  # Channel spacing in Hz
    transmit_offset_hz: int = 0  # TX offset from RX (for repeater)
    bandwidth_hz: int = 12500  # Channel bandwidth
    is_tdma: bool = False  # Phase II TDMA
    slot_count: int = 1  # TDMA slots (1 for FDMA, 2 for TDMA)

    def get_downlink_frequency(self, channel_number: int) -> int:
        """Calculate downlink (receive) frequency for channel number."""
        return self.base_frequency_hz + (channel_number * self.channel_spacing_hz)

    def get_uplink_frequency(self, channel_number: int) -> int:
        """Calculate uplink (transmit) frequency for channel number."""
        return self.get_downlink_frequency(channel_number) + self.transmit_offset_hz

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "baseFrequencyMhz": self.base_frequency_hz / 1e6,
            "channelSpacingKhz": self.channel_spacing_hz / 1000,
            "transmitOffsetMhz": self.transmit_offset_hz / 1e6,
            "bandwidthKhz": self.bandwidth_hz / 1000,
            "isTdma": self.is_tdma,
            "slotCount": self.slot_count,
        }


@dataclass
class SiteStatus:
    """Site status from RFSS_STS_BCAST."""
    rfss_id: int  # RF Subsystem ID
    site_id: int  # Site ID within RFSS
    system_id: int  # System ID
    lra: int  # Location Registration Area
    channel: int  # Control channel number
    service_class: int  # Service class flags
    last_update: float = 0.0

    def __post_init__(self):
        if self.last_update == 0.0:
            self.last_update = time.time()

    @property
    def site_key(self) -> str:
        """Unique key for this site."""
        return f"{self.system_id:03X}-{self.rfss_id}-{self.site_id}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rfssId": self.rfss_id,
            "siteId": self.site_id,
            "systemId": self.system_id,
            "siteKey": self.site_key,
            "lra": self.lra,
            "channel": self.channel,
            "serviceClass": self.service_class,
            "lastUpdate": self.last_update,
        }


@dataclass
class NetworkStatus:
    """Network status from NET_STS_BCAST."""
    wacn: int  # Wide Area Communication Network (20-bit)
    system_id: int  # System ID (12-bit)
    lra: int  # Location Registration Area
    channel: int  # Network control channel
    last_update: float = 0.0

    def __post_init__(self):
        if self.last_update == 0.0:
            self.last_update = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wacn": self.wacn,
            "systemId": self.system_id,
            "lra": self.lra,
            "channel": self.channel,
            "lastUpdate": self.last_update,
        }


@dataclass
class AdjacentSite:
    """Adjacent site from ADJ_STS_BCAST."""
    system_id: int
    rfss_id: int
    site_id: int
    lra: int
    channel: int
    service_class: int
    last_update: float = 0.0

    def __post_init__(self):
        if self.last_update == 0.0:
            self.last_update = time.time()

    @property
    def site_key(self) -> str:
        return f"{self.system_id:03X}-{self.rfss_id}-{self.site_id}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "systemId": self.system_id,
            "rfssId": self.rfss_id,
            "siteId": self.site_id,
            "siteKey": self.site_key,
            "lra": self.lra,
            "channel": self.channel,
            "serviceClass": self.service_class,
            "lastUpdate": self.last_update,
        }


@dataclass
class SystemServices:
    """System services from SYS_SRV_BCAST."""
    services_available: int  # 24-bit flags
    services_supported: int  # 24-bit flags
    last_update: float = 0.0

    # Service flag bits
    COMPOSITE_CONTROL = 0x800000
    DATA_SERVICES = 0x400000
    VOICE_SERVICES = 0x200000
    REGISTRATION = 0x100000
    AUTHENTICATION = 0x080000

    def __post_init__(self):
        if self.last_update == 0.0:
            self.last_update = time.time()

    @property
    def has_composite_control(self) -> bool:
        return bool(self.services_available & self.COMPOSITE_CONTROL)

    @property
    def has_data_services(self) -> bool:
        return bool(self.services_available & self.DATA_SERVICES)

    @property
    def has_voice_services(self) -> bool:
        return bool(self.services_available & self.VOICE_SERVICES)

    @property
    def has_registration(self) -> bool:
        return bool(self.services_available & self.REGISTRATION)

    @property
    def has_authentication(self) -> bool:
        return bool(self.services_available & self.AUTHENTICATION)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "servicesAvailable": self.services_available,
            "servicesSupported": self.services_supported,
            "compositeControl": self.has_composite_control,
            "dataServices": self.has_data_services,
            "voiceServices": self.has_voice_services,
            "registration": self.has_registration,
            "authentication": self.has_authentication,
            "lastUpdate": self.last_update,
        }


class P25NetworkConfigurationMonitor:
    """Monitors and caches P25 network configuration.

    Builds a complete picture of the P25 system from control channel
    broadcasts. This information is essential for:
    - Converting channel numbers to actual frequencies
    - Understanding system topology
    - Roaming between sites
    - Feature availability

    Example:
        monitor = P25NetworkConfigurationMonitor()

        # Process IDEN_UP to learn frequency bands
        monitor.process_identifier_update(ident_data)

        # Now we can look up frequencies
        freq = monitor.get_frequency(channel_number)

        # Get system overview
        info = monitor.get_system_info()
    """

    def __init__(self) -> None:
        # Frequency bands (up to 16)
        self._frequency_bands: Dict[int, FrequencyBand] = {}

        # Current site status
        self._site_status: Optional[SiteStatus] = None

        # Network status
        self._network_status: Optional[NetworkStatus] = None

        # Adjacent sites
        self._adjacent_sites: Dict[str, AdjacentSite] = {}

        # System services
        self._system_services: Optional[SystemServices] = None

        # Secondary control channels
        self._secondary_control_channels: Dict[int, int] = {}  # channel -> freq_hz

        # NAC (Network Access Code) - 12 bits
        self._nac: int = 0

        # Callbacks
        self.on_site_update: Optional[Callable[[SiteStatus], None]] = None
        self.on_adjacent_site: Optional[Callable[[AdjacentSite], None]] = None
        self.on_frequency_band: Optional[Callable[[FrequencyBand], None]] = None

    @property
    def nac(self) -> int:
        """Get Network Access Code."""
        return self._nac

    @nac.setter
    def nac(self, value: int) -> None:
        """Set Network Access Code."""
        self._nac = value & 0xFFF

    @property
    def system_id(self) -> Optional[int]:
        """Get system ID from current site or network status."""
        if self._site_status:
            return self._site_status.system_id
        if self._network_status:
            return self._network_status.system_id
        return None

    @property
    def site_key(self) -> Optional[str]:
        """Get current site key."""
        if self._site_status:
            return self._site_status.site_key
        return None

    def process_identifier_update(self, identifier: int, base_freq_mhz: float,
                                  channel_spacing_khz: float, tx_offset_mhz: float = 0,
                                  bandwidth_khz: float = 12.5, is_tdma: bool = False,
                                  slot_count: int = 1) -> FrequencyBand:
        """Process IDEN_UP message to learn frequency band.

        Args:
            identifier: 4-bit band identifier
            base_freq_mhz: Base frequency in MHz
            channel_spacing_khz: Channel spacing in kHz
            tx_offset_mhz: TX offset in MHz
            bandwidth_khz: Channel bandwidth in kHz
            is_tdma: True if TDMA (Phase II)
            slot_count: Number of TDMA slots

        Returns:
            The created FrequencyBand
        """
        band = FrequencyBand(
            identifier=identifier & 0xF,
            base_frequency_hz=int(base_freq_mhz * 1e6),
            channel_spacing_hz=int(channel_spacing_khz * 1000),
            transmit_offset_hz=int(tx_offset_mhz * 1e6),
            bandwidth_hz=int(bandwidth_khz * 1000),
            is_tdma=is_tdma,
            slot_count=slot_count,
        )

        self._frequency_bands[band.identifier] = band

        logger.debug(f"Frequency band {identifier}: base={base_freq_mhz:.4f} MHz, "
                    f"spacing={channel_spacing_khz} kHz")

        if self.on_frequency_band:
            self.on_frequency_band(band)

        return band

    def process_rfss_status(self, system_id: int, rfss_id: int, site_id: int,
                           lra: int, channel: int, service_class: int) -> SiteStatus:
        """Process RFSS_STS_BCAST message."""
        status = SiteStatus(
            rfss_id=rfss_id,
            site_id=site_id,
            system_id=system_id,
            lra=lra,
            channel=channel,
            service_class=service_class,
        )
        self._site_status = status

        logger.info(f"Site status: {status.site_key}")

        if self.on_site_update:
            self.on_site_update(status)

        return status

    def process_network_status(self, wacn: int, system_id: int, lra: int,
                              channel: int) -> NetworkStatus:
        """Process NET_STS_BCAST message."""
        status = NetworkStatus(
            wacn=wacn,
            system_id=system_id,
            lra=lra,
            channel=channel,
        )
        self._network_status = status

        logger.debug(f"Network status: WACN={wacn:05X} SYS={system_id:03X}")

        return status

    def process_adjacent_status(self, system_id: int, rfss_id: int, site_id: int,
                               lra: int, channel: int, service_class: int) -> AdjacentSite:
        """Process ADJ_STS_BCAST message."""
        site = AdjacentSite(
            system_id=system_id,
            rfss_id=rfss_id,
            site_id=site_id,
            lra=lra,
            channel=channel,
            service_class=service_class,
        )

        self._adjacent_sites[site.site_key] = site

        logger.debug(f"Adjacent site: {site.site_key}")

        if self.on_adjacent_site:
            self.on_adjacent_site(site)

        return site

    def process_system_service(self, services_available: int,
                              services_supported: int) -> SystemServices:
        """Process SYS_SRV_BCAST message."""
        services = SystemServices(
            services_available=services_available,
            services_supported=services_supported,
        )
        self._system_services = services

        logger.debug(f"System services: voice={services.has_voice_services}, "
                    f"data={services.has_data_services}")

        return services

    def get_frequency(self, channel: int) -> Optional[int]:
        """Get frequency in Hz for a channel number.

        Channel format: IIII CCCC CCCC CCCC
        - IIII: 4-bit identifier (selects frequency band)
        - CCCC...: 12-bit channel number

        Returns None if frequency band is unknown.
        """
        identifier = (channel >> 12) & 0xF
        channel_number = channel & 0xFFF

        band = self._frequency_bands.get(identifier)
        if band is None:
            logger.warning(f"Unknown frequency band identifier: {identifier}")
            return None

        return band.get_downlink_frequency(channel_number)

    def get_frequency_mhz(self, channel: int) -> float:
        """Get frequency in MHz for a channel number."""
        freq_hz = self.get_frequency(channel)
        return freq_hz / 1e6 if freq_hz else 0.0

    def get_frequency_band(self, identifier: int) -> Optional[FrequencyBand]:
        """Get frequency band by identifier."""
        return self._frequency_bands.get(identifier & 0xF)

    def get_all_frequency_bands(self) -> List[FrequencyBand]:
        """Get all known frequency bands."""
        return list(self._frequency_bands.values())

    def get_adjacent_sites(self) -> List[AdjacentSite]:
        """Get all known adjacent sites."""
        return list(self._adjacent_sites.values())

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        return {
            "nac": self._nac,
            "systemId": self.system_id,
            "siteKey": self.site_key,
            "siteStatus": self._site_status.to_dict() if self._site_status else None,
            "networkStatus": self._network_status.to_dict() if self._network_status else None,
            "systemServices": self._system_services.to_dict() if self._system_services else None,
            "frequencyBandCount": len(self._frequency_bands),
            "frequencyBands": [b.to_dict() for b in self._frequency_bands.values()],
            "adjacentSiteCount": len(self._adjacent_sites),
            "adjacentSites": [s.to_dict() for s in self._adjacent_sites.values()],
        }

    def is_configured(self) -> bool:
        """Check if basic configuration is known.

        Returns True if we have at least one frequency band
        and site status.
        """
        return bool(self._frequency_bands and self._site_status)

    def clear(self) -> None:
        """Clear all configuration data."""
        self._frequency_bands.clear()
        self._site_status = None
        self._network_status = None
        self._adjacent_sites.clear()
        self._system_services = None
        self._secondary_control_channels.clear()
        self._nac = 0
