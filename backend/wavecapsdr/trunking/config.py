"""Configuration dataclasses for P25 trunking systems.

Defines the configuration schema for trunking systems including:
- System-level configuration (protocol, frequencies, device)
- Talkgroup definitions (ID, name, priority, recording options)
- Channel identification for frequency calculation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class TrunkingProtocol(str, Enum):
    """Supported trunking protocols."""
    P25_PHASE1 = "p25_phase1"  # C4FM, IMBE (SA-GRN, most systems)
    P25_PHASE2 = "p25_phase2"  # CQPSK/TDMA, AMBE+2 (PSERN, newer systems)


@dataclass
class TalkgroupConfig:
    """Configuration for a single talkgroup.

    Attributes:
        tgid: Talkgroup ID (decimal)
        name: Human-readable name (e.g., "Kirkland PD")
        alpha_tag: Short identifier for display
        category: Category for grouping (e.g., "Police", "Fire")
        priority: Recording priority (1-10, lower = higher priority)
        record: Whether to record calls
        monitor: Whether to stream audio live
    """
    tgid: int
    name: str
    alpha_tag: str = ""
    category: str = ""
    priority: int = 5
    record: bool = True
    monitor: bool = True

    def __post_init__(self) -> None:
        if not self.alpha_tag:
            # Generate alpha tag from name
            self.alpha_tag = self.name[:10].upper().replace(" ", "_")


@dataclass
class TrunkingSystemConfig:
    """Configuration for a P25 trunking system.

    Attributes:
        id: Unique system identifier
        name: Human-readable system name
        protocol: P25 Phase I or Phase II
        control_channels: List of control channel frequencies (Hz)
        center_hz: SDR center frequency (Hz)
        sample_rate: SDR sample rate (Hz)
        device_id: SoapySDR device string
        max_voice_recorders: Maximum concurrent voice channel recordings
        talkgroups: Talkgroup configurations keyed by TGID
        recording_path: Path for audio file storage
        record_unknown: Whether to record unknown talkgroups
        min_call_duration: Minimum call duration to save (seconds)
        squelch_db: Squelch level for voice channels
    """
    id: str
    name: str
    protocol: TrunkingProtocol = TrunkingProtocol.P25_PHASE1
    control_channels: List[float] = field(default_factory=list)
    center_hz: float = 851_000_000
    sample_rate: int = 8_000_000
    device_id: str = ""
    gain: Optional[float] = None  # RF gain (None = auto)
    antenna: Optional[str] = None  # SDR antenna port
    device_settings: Dict[str, str] = field(default_factory=dict)  # Device-specific settings (e.g., rfnotch_ctrl, dabnotch_ctrl)
    max_voice_recorders: int = 4
    talkgroups: Dict[int, TalkgroupConfig] = field(default_factory=dict)
    recording_path: str = "./recordings"
    record_unknown: bool = False
    min_call_duration: float = 1.0
    squelch_db: float = -50.0

    # Startup/shutdown
    auto_start: bool = True  # Start system automatically on server startup

    # Advanced settings
    control_channel_timeout: float = 10.0  # Seconds before trying next CC
    voice_hold_time: float = 2.0  # Seconds to hold voice channel after last audio
    audio_gain: float = 1.0  # Audio output gain multiplier

    def get_talkgroup(self, tgid: int) -> Optional[TalkgroupConfig]:
        """Get talkgroup config by ID."""
        return self.talkgroups.get(tgid)

    def is_talkgroup_monitored(self, tgid: int) -> bool:
        """Check if talkgroup should be monitored."""
        tg = self.talkgroups.get(tgid)
        if tg is not None:
            return tg.monitor
        return self.record_unknown

    def is_talkgroup_recorded(self, tgid: int) -> bool:
        """Check if talkgroup should be recorded."""
        tg = self.talkgroups.get(tgid)
        if tg is not None:
            return tg.record
        return self.record_unknown

    def get_talkgroup_priority(self, tgid: int) -> int:
        """Get talkgroup priority (lower = higher priority)."""
        tg = self.talkgroups.get(tgid)
        if tg is not None:
            return tg.priority
        return 10  # Lowest priority for unknown

    @classmethod
    def from_dict(cls, data: Dict) -> "TrunkingSystemConfig":
        """Create config from dictionary (e.g., from YAML)."""
        # Parse talkgroups
        talkgroups = {}
        for tgid, tg_data in data.get("talkgroups", {}).items():
            if isinstance(tg_data, dict):
                talkgroups[int(tgid)] = TalkgroupConfig(
                    tgid=int(tgid),
                    name=tg_data.get("name", f"TG {tgid}"),
                    alpha_tag=tg_data.get("alpha_tag", ""),
                    category=tg_data.get("category", ""),
                    priority=tg_data.get("priority", 5),
                    record=tg_data.get("record", True),
                    monitor=tg_data.get("monitor", True),
                )
            elif isinstance(tg_data, str):
                # Simple format: just the name
                talkgroups[int(tgid)] = TalkgroupConfig(
                    tgid=int(tgid),
                    name=tg_data,
                )

        # Parse protocol
        protocol_str = data.get("protocol", "p25_phase1")
        try:
            protocol = TrunkingProtocol(protocol_str)
        except ValueError:
            protocol = TrunkingProtocol.P25_PHASE1

        gain = data.get("gain")
        if gain is not None:
            gain = float(gain)

        return cls(
            id=data.get("id", "system"),
            name=data.get("name", "P25 System"),
            protocol=protocol,
            control_channels=[float(f) for f in data.get("control_channels", [])],
            center_hz=float(data.get("center_hz", 851_000_000)),
            sample_rate=int(data.get("sample_rate", 8_000_000)),
            device_id=data.get("device_id", ""),
            gain=gain,
            antenna=data.get("antenna"),
            max_voice_recorders=int(data.get("max_voice_recorders", 4)),
            talkgroups=talkgroups,
            recording_path=data.get("recording_path", "./recordings"),
            record_unknown=data.get("record_unknown", False),
            min_call_duration=float(data.get("min_call_duration", 1.0)),
            squelch_db=float(data.get("squelch_db", -50.0)),
            auto_start=data.get("auto_start", True),
        )

    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "protocol": self.protocol.value,
            "control_channels": self.control_channels,
            "center_hz": self.center_hz,
            "sample_rate": self.sample_rate,
            "device_id": self.device_id,
            "gain": self.gain,
            "max_voice_recorders": self.max_voice_recorders,
            "talkgroups": {
                tgid: {
                    "name": tg.name,
                    "alpha_tag": tg.alpha_tag,
                    "category": tg.category,
                    "priority": tg.priority,
                    "record": tg.record,
                    "monitor": tg.monitor,
                }
                for tgid, tg in self.talkgroups.items()
            },
            "recording_path": self.recording_path,
            "record_unknown": self.record_unknown,
            "min_call_duration": self.min_call_duration,
            "squelch_db": self.squelch_db,
            "auto_start": self.auto_start,
        }


def load_talkgroups_csv(csv_path: str) -> Dict[int, TalkgroupConfig]:
    """Load talkgroups from CSV file (trunk-recorder format).

    Expected CSV format:
    Decimal,Hex,Alpha Tag,Mode,Description,Tag,Category,Priority

    Args:
        csv_path: Path to CSV file

    Returns:
        Dict mapping TGID to TalkgroupConfig
    """
    import csv

    talkgroups = {}

    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    tgid = int(row.get("Decimal", 0))
                    if tgid == 0:
                        continue

                    talkgroups[tgid] = TalkgroupConfig(
                        tgid=tgid,
                        name=row.get("Description", f"TG {tgid}"),
                        alpha_tag=row.get("Alpha Tag", ""),
                        category=row.get("Category", row.get("Tag", "")),
                        priority=int(row.get("Priority", 5)),
                        record=True,
                        monitor=True,
                    )
                except (ValueError, KeyError) as e:
                    continue  # Skip invalid rows

    except FileNotFoundError:
        pass

    return talkgroups
