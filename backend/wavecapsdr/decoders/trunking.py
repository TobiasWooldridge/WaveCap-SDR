"""
Trunking manager for coordinating control channels and voice channels.

Handles automatic voice channel following for P25, DMR, and other trunked systems.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)


class TrunkingProtocol(Enum):
    """Supported trunking protocols"""

    P25_PHASE1 = "P25 Phase 1"
    P25_PHASE2 = "P25 Phase 2"
    DMR_TIER2 = "DMR Tier 2"
    DMR_TIER3 = "DMR Tier 3 (Capacity Plus/Connect Plus)"
    NXDN = "NXDN"
    DSTAR = "D-STAR"


@dataclass
class VoiceGrant:
    """Voice channel grant information"""

    talkgroup_id: int
    frequency_hz: float
    timestamp: float
    source_id: int | None = None
    expires_at: float | None = None  # For time-slot systems


@dataclass
class TalkgroupConfig:
    """Talkgroup monitoring configuration"""

    tgid: int
    name: str
    priority: int = 5  # 1-10, higher = more important
    record: bool = True
    alert: bool = False


@dataclass
class TrunkingSystem:
    """Trunking system configuration"""

    system_id: str
    name: str
    protocol: TrunkingProtocol
    control_channels: list[float]  # Control channel frequencies in Hz
    talkgroups: dict[int, TalkgroupConfig] = field(default_factory=dict)

    # Voice channel following
    max_voice_channels: int = 4  # Maximum simultaneous voice channels to follow
    hold_time: float = 2.0  # Seconds to hold voice channel after transmission ends


class TrunkingManager:
    """
    Manages trunked radio systems:
    - Monitors control channels
    - Tracks voice grants
    - Follows voice channels automatically
    - Records talkgroups
    """

    def __init__(self, system: TrunkingSystem):
        self.system = system
        self.active_grants: dict[int, VoiceGrant] = {}  # tgid -> grant
        self.monitored_talkgroups: set[int] = set()

        # Voice channel callbacks
        self.on_grant: Callable[[VoiceGrant], None] | None = None
        self.on_release: Callable[[int], None] | None = None  # tgid
        self.on_voice_data: Callable[[int, bytes], None] | None = None  # (tgid, audio)

        logger.info(f"Trunking manager initialized: {system.name} ({system.protocol.value})")

    def add_monitored_talkgroup(self, tgid: int, name: str = "", priority: int = 5) -> None:
        """Add a talkgroup to monitor"""
        self.monitored_talkgroups.add(tgid)
        if tgid not in self.system.talkgroups:
            self.system.talkgroups[tgid] = TalkgroupConfig(
                tgid=tgid, name=name or f"TG {tgid}", priority=priority
            )
        logger.info(f"Monitoring talkgroup {tgid}: {name}")

    def remove_monitored_talkgroup(self, tgid: int) -> None:
        """Remove talkgroup from monitoring"""
        self.monitored_talkgroups.discard(tgid)
        logger.info(f"Stopped monitoring talkgroup {tgid}")

    def handle_voice_grant(
        self, tgid: int, frequency_hz: float, source_id: int | None = None
    ) -> None:
        """
        Handle a voice channel grant from control channel.

        This is called by the P25/DMR decoder when a grant is detected.
        """
        import time

        # Check if we're monitoring this talkgroup
        if tgid not in self.monitored_talkgroups:
            logger.debug(f"Ignoring grant for unmonitored TG {tgid}")
            return

        # Check if we have capacity for another voice channel
        if len(self.active_grants) >= self.system.max_voice_channels:
            # Need to drop lowest priority grant
            self._drop_lowest_priority_grant()

        grant = VoiceGrant(
            talkgroup_id=tgid, frequency_hz=frequency_hz, timestamp=time.time(), source_id=source_id
        )

        self.active_grants[tgid] = grant

        tg_name = self.system.talkgroups.get(
            tgid, TalkgroupConfig(tgid=tgid, name=f"TG {tgid}")
        ).name
        logger.info(f"Voice grant: {tg_name} (TG {tgid}) on {frequency_hz / 1e6:.4f} MHz")

        if self.on_grant:
            self.on_grant(grant)

    def handle_voice_end(self, tgid: int) -> None:
        """Handle end of voice transmission"""
        if tgid in self.active_grants:
            self.active_grants.pop(tgid)
            logger.info(f"Voice ended: TG {tgid}")

            if self.on_release:
                self.on_release(tgid)

    def _drop_lowest_priority_grant(self) -> None:
        """Drop the lowest priority active grant to make room"""
        if not self.active_grants:
            return

        # Find lowest priority talkgroup
        lowest_tgid = None
        lowest_priority = 100

        for tgid in self.active_grants:
            tg_config = self.system.talkgroups.get(tgid)
            priority = tg_config.priority if tg_config else 5

            if priority < lowest_priority:
                lowest_priority = priority
                lowest_tgid = tgid

        if lowest_tgid:
            logger.info(f"Dropping lowest priority grant: TG {lowest_tgid}")
            self.handle_voice_end(lowest_tgid)

    def get_active_frequencies(self) -> list[float]:
        """Get list of frequencies currently in use for voice channels"""
        return [grant.frequency_hz for grant in self.active_grants.values()]

    def get_talkgroup_status(self, tgid: int) -> VoiceGrant | None:
        """Check if a talkgroup is currently active"""
        return self.active_grants.get(tgid)

    def is_monitored(self, tgid: int) -> bool:
        """Check if talkgroup is being monitored"""
        return tgid in self.monitored_talkgroups


# Example usage helper
def create_p25_system(name: str, control_freq: float, talkgroups: dict[int, str]) -> TrunkingSystem:
    """Create a P25 trunking system configuration"""
    system = TrunkingSystem(
        system_id=name.lower().replace(" ", "_"),
        name=name,
        protocol=TrunkingProtocol.P25_PHASE1,
        control_channels=[control_freq],
        max_voice_channels=4,
    )

    for tgid, tg_name in talkgroups.items():
        system.talkgroups[tgid] = TalkgroupConfig(tgid=tgid, name=tg_name, priority=5, record=True)

    return system


def create_dmr_system(
    name: str, control_freq: float, color_code: int, talkgroups: dict[int, str]
) -> TrunkingSystem:
    """Create a DMR trunking system configuration"""
    system = TrunkingSystem(
        system_id=name.lower().replace(" ", "_"),
        name=name,
        protocol=TrunkingProtocol.DMR_TIER3,
        control_channels=[control_freq],
        max_voice_channels=2,  # DMR has 2 time slots
    )

    for tgid, tg_name in talkgroups.items():
        system.talkgroups[tgid] = TalkgroupConfig(tgid=tgid, name=tg_name, priority=5, record=True)

    return system
