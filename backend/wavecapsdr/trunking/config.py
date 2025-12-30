"""Configuration dataclasses for P25 trunking systems.

Defines the configuration schema for trunking systems including:
- System-level configuration (protocol, frequencies, device)
- Talkgroup definitions (ID, name, priority, recording options)
- Channel identification for frequency calculation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def parse_frequency(value: str | int | float) -> float:
    """Parse a frequency value with optional unit suffix.

    Supports formats like:
    - 412950000 (raw Hz as int/float)
    - "412.95 MHz" or "412.95MHz"
    - "412950 kHz" or "412950kHz"
    - "412950000 Hz" or "412950000Hz"
    - "412.95" (assumed MHz if < 1000, otherwise Hz)

    Returns:
        Frequency in Hz as float
    """
    if isinstance(value, (int, float)):
        return float(value)

    # Parse string with optional unit
    value = str(value).strip()

    # Match number with optional unit
    match = re.match(r'^([\d.]+)\s*(MHz|mhz|MHZ|kHz|khz|KHZ|Hz|hz|HZ)?$', value)
    if not match:
        raise ValueError(f"Invalid frequency format: {value}")

    num = float(match.group(1))
    unit = (match.group(2) or '').lower()

    if unit == 'mhz':
        return num * 1_000_000
    elif unit == 'khz':
        return num * 1_000
    elif unit == 'hz':
        return num
    else:
        # No unit specified - assume MHz if small number, Hz if large
        if num < 1000:
            return num * 1_000_000  # Assume MHz
        else:
            return num  # Assume Hz


class TrunkingProtocol(str, Enum):
    """Supported trunking protocols."""
    P25_PHASE1 = "p25_phase1"  # C4FM, IMBE (SA-GRN, most systems)
    P25_PHASE2 = "p25_phase2"  # CQPSK/TDMA, AMBE+2 (PSERN, newer systems)


class P25Modulation(str, Enum):
    """P25 modulation types."""
    C4FM = "c4fm"    # Standard 4FSK modulation
    LSM = "lsm"      # Linear Simulcast Modulation (CQPSK)


class HuntMode(str, Enum):
    """Control channel hunting mode.

    Determines how the trunking system finds and maintains control channel lock.
    """
    AUTO = "auto"           # Default: hunt continuously, roam if better channel found
    MANUAL = "manual"       # Lock to specified channel, no hunting ever
    SCAN_ONCE = "scan_once" # Scan all channels once, lock to best, stay there


@dataclass
class ChannelIdentifierConfig:
    """Channel identifier configuration (IDEN_UP seed data).

    Uses MHz/kHz units to match IDEN_UP fields and avoid rounding.
    """
    identifier: int  # 4-bit band identifier (0-15)
    base_freq_mhz: float
    channel_spacing_khz: float
    bandwidth_khz: float = 12.5
    tx_offset_mhz: float = 0.0

    @classmethod
    def from_dict(cls, identifier: int, data: dict[str, Any]) -> "ChannelIdentifierConfig":
        """Parse a channel identifier from config data."""
        base_freq_mhz = data.get("base_freq_mhz", data.get("base_freq"))
        channel_spacing_khz = data.get("channel_spacing_khz", data.get("spacing_khz"))
        if base_freq_mhz is None or channel_spacing_khz is None:
            raise ValueError("channel_identifiers entry requires base_freq_mhz and channel_spacing_khz")

        tx_offset_mhz = data.get("tx_offset_mhz", data.get("tx_offset", 0.0))
        tx_offset_hz = data.get("tx_offset_hz")
        if tx_offset_hz is not None:
            tx_offset_mhz = float(tx_offset_hz) / 1e6

        return cls(
            identifier=int(identifier),
            base_freq_mhz=float(base_freq_mhz),
            channel_spacing_khz=float(channel_spacing_khz),
            bandwidth_khz=float(data.get("bandwidth_khz", data.get("bandwidth", 12.5))),
            tx_offset_mhz=float(tx_offset_mhz),
        )

    def to_dict(self) -> dict[str, float]:
        """Serialize to config-friendly dict."""
        return {
            "identifier": int(self.identifier),
            "base_freq_mhz": float(self.base_freq_mhz),
            "channel_spacing_khz": float(self.channel_spacing_khz),
            "bandwidth_khz": float(self.bandwidth_khz),
            "tx_offset_mhz": float(self.tx_offset_mhz),
        }


@dataclass
class ControlChannelConfig:
    """Configuration for a single control channel.

    Attributes:
        frequency_hz: Control channel frequency in Hz
        name: Optional human-readable name (e.g., "CC1 - Primary", "Mt Barker")
    """
    frequency_hz: float
    name: str = ""

    @classmethod
    def from_value(cls, value: str | int | float | dict) -> "ControlChannelConfig":
        """Parse a control channel from various formats.

        Supports:
        - Float/int: Raw frequency in Hz
        - String: Frequency with optional unit (e.g., "413.075 MHz")
        - Dict: {frequency: "413.075 MHz", name: "CC1 - Primary"}
        """
        if isinstance(value, dict):
            freq = value.get("frequency") or value.get("freq") or value.get("frequency_hz")
            if freq is None:
                raise ValueError(f"Control channel dict missing frequency: {value}")
            return cls(
                frequency_hz=parse_frequency(freq),
                name=value.get("name", ""),
            )
        else:
            return cls(frequency_hz=parse_frequency(value))


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
        control_channels: List of control channel configurations (frequency + optional name)
        center_hz: SDR center frequency (Hz)
        sample_rate: SDR sample rate (Hz)
        device_id: SoapySDR device string
        max_voice_recorders: Maximum concurrent voice channel recordings
        talkgroups: Talkgroup configurations keyed by TGID
        recording_path: Path for audio file storage
        record_unknown: Whether to record unknown talkgroups
        min_call_duration: Minimum call duration to save (seconds)
        squelch_db: Squelch level for voice channels
        channel_identifiers: Optional IDEN_UP seed data for channel-to-frequency lookup
    """
    id: str
    name: str
    protocol: TrunkingProtocol = TrunkingProtocol.P25_PHASE1
    modulation: P25Modulation | None = None  # None = auto-detect based on protocol
    control_channels: list[ControlChannelConfig] = field(default_factory=list)
    center_hz: float = 851_000_000
    sample_rate: int = 8_000_000
    device_id: str = ""
    gain: float | None = None  # RF gain (None = auto)
    antenna: str | None = None  # SDR antenna port
    device_settings: dict[str, str] = field(default_factory=dict)  # Device-specific settings (e.g., rfnotch_ctrl, dabnotch_ctrl)
    element_gains: dict[str, float] = field(default_factory=dict)  # Per-element gains (e.g., RFGR for SDRplay LNA)
    agc_enabled: bool = False  # Enable automatic gain control (SDRplay only)
    max_voice_recorders: int = 4
    talkgroups: dict[int, TalkgroupConfig] = field(default_factory=dict)
    talkgroups_file: str | None = None  # External YAML file for talkgroup definitions
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

    # Control channel scanning and roaming
    roam_check_interval: float = 30.0  # Seconds between roaming checks
    roam_threshold_db: float = 6.0  # SNR improvement required to trigger roaming
    initial_scan_enabled: bool = True  # Whether to scan all channels at startup
    default_hunt_mode: HuntMode = HuntMode.AUTO  # Default control channel hunting mode
    channel_identifiers: dict[int, ChannelIdentifierConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize control channel entries to ControlChannelConfig."""
        normalized: list[ControlChannelConfig] = []
        for cc in self.control_channels:
            if isinstance(cc, ControlChannelConfig):
                normalized.append(cc)
            else:
                normalized.append(ControlChannelConfig.from_value(cc))
        self.control_channels = normalized

    @property
    def control_channel_frequencies(self) -> list[float]:
        """Get list of control channel frequencies (Hz) for backward compatibility."""
        freqs: list[float] = []
        for cc in self.control_channels:
            if isinstance(cc, ControlChannelConfig):
                freqs.append(cc.frequency_hz)
            else:
                freqs.append(parse_frequency(cc))
        return freqs

    def get_control_channel_name(self, frequency_hz: float) -> str:
        """Get the name for a control channel frequency, or empty string if not named."""
        for cc in self.control_channels:
            if isinstance(cc, ControlChannelConfig):
                if abs(cc.frequency_hz - frequency_hz) < 1000:  # 1 kHz tolerance
                    return cc.name
        return ""

    def get_talkgroup(self, tgid: int) -> TalkgroupConfig | None:
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
    def from_dict(cls, data: dict, config_dir: str | None = None) -> TrunkingSystemConfig:
        """Create config from dictionary (e.g., from YAML).

        Args:
            data: Configuration dictionary
            config_dir: Base directory for resolving relative paths (e.g., talkgroups_file)
        """
        # Parse talkgroups from external file if specified
        talkgroups_file = data.get("talkgroups_file")
        talkgroups = {}

        if talkgroups_file:
            # Load from external YAML file
            talkgroups = load_talkgroups_yaml(talkgroups_file, config_dir)

        # Parse inline talkgroups (can supplement or override file-based ones)
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

        # Parse modulation
        modulation_str = data.get("modulation")
        modulation = None
        if modulation_str:
            try:
                modulation = P25Modulation(modulation_str.lower())
            except ValueError:
                pass  # Use default

        gain = data.get("gain")
        if gain is not None:
            gain = float(gain)

        channel_identifiers: dict[int, ChannelIdentifierConfig] = {}
        channel_raw = data.get("channel_identifiers", {})
        if isinstance(channel_raw, dict):
            for key, entry in channel_raw.items():
                if not isinstance(entry, dict):
                    continue
                ident = entry.get("identifier")
                if ident is None:
                    try:
                        ident = int(key)
                    except (TypeError, ValueError):
                        continue
                channel_identifiers[int(ident)] = ChannelIdentifierConfig.from_dict(int(ident), entry)
        elif isinstance(channel_raw, list):
            for entry in channel_raw:
                if not isinstance(entry, dict):
                    continue
                ident = entry.get("identifier")
                if ident is None:
                    continue
                channel_identifiers[int(ident)] = ChannelIdentifierConfig.from_dict(int(ident), entry)

        return cls(
            id=data.get("id", "system"),
            name=data.get("name", "P25 System"),
            protocol=protocol,
            modulation=modulation,
            control_channels=[ControlChannelConfig.from_value(f) for f in data.get("control_channels", [])],
            center_hz=parse_frequency(data.get("center_hz", 851_000_000)),
            sample_rate=int(data.get("sample_rate", 8_000_000)),
            device_id=data.get("device_id", ""),
            gain=gain,
            antenna=data.get("antenna"),
            device_settings=data.get("device_settings", {}),
            element_gains=data.get("element_gains", {}),
            agc_enabled=data.get("agc_enabled", False),
            max_voice_recorders=int(data.get("max_voice_recorders", 4)),
            talkgroups=talkgroups,
            talkgroups_file=talkgroups_file,
            recording_path=data.get("recording_path", "./recordings"),
            record_unknown=data.get("record_unknown", False),
            min_call_duration=float(data.get("min_call_duration", 1.0)),
            squelch_db=float(data.get("squelch_db", -50.0)),
            auto_start=data.get("auto_start", True),
            control_channel_timeout=float(data.get("control_channel_timeout", 10.0)),
            roam_check_interval=float(data.get("roam_check_interval", 30.0)),
            roam_threshold_db=float(data.get("roam_threshold_db", 6.0)),
            initial_scan_enabled=data.get("initial_scan_enabled", True),
            default_hunt_mode=HuntMode(data.get("default_hunt_mode", "auto")),
            channel_identifiers=channel_identifiers,
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "protocol": self.protocol.value,
            "control_channels": [
                {"frequency": cc.frequency_hz, "name": cc.name} if cc.name else cc.frequency_hz
                for cc in self.control_channels
            ],
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
            "default_hunt_mode": self.default_hunt_mode.value,
            "channel_identifiers": {
                str(ident): ident_cfg.to_dict()
                for ident, ident_cfg in self.channel_identifiers.items()
            },
        }


def load_talkgroups_csv(csv_path: str) -> dict[int, TalkgroupConfig]:
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
        with open(csv_path, newline="", encoding="utf-8") as f:
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
                except (ValueError, KeyError):
                    continue  # Skip invalid rows

    except FileNotFoundError:
        pass

    return talkgroups


def load_talkgroups_yaml(yaml_path: str, config_dir: str | None = None) -> dict[int, TalkgroupConfig]:
    """Load talkgroups from YAML file.

    Expected YAML format:
    ```yaml
    talkgroups:
      100:
        name: 'CFS State Dispatch'
        category: 'Fire'
        priority: 1
      101:
        name: 'CFS Adelaide Metro'
        category: 'Fire'
        priority: 2
    ```

    Args:
        yaml_path: Path to YAML file (absolute or relative to config_dir)
        config_dir: Base directory for resolving relative paths

    Returns:
        Dict mapping TGID to TalkgroupConfig
    """
    import os
    import yaml

    talkgroups = {}

    # Resolve path relative to config directory if not absolute
    if config_dir and not os.path.isabs(yaml_path):
        yaml_path = os.path.join(config_dir, yaml_path)

    try:
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        tg_data = data.get("talkgroups", data)  # Support both root-level and nested

        for tgid, tg in tg_data.items():
            try:
                tgid_int = int(tgid)
                if isinstance(tg, dict):
                    talkgroups[tgid_int] = TalkgroupConfig(
                        tgid=tgid_int,
                        name=tg.get("name", f"TG {tgid}"),
                        alpha_tag=tg.get("alpha_tag", ""),
                        category=tg.get("category", ""),
                        priority=int(tg.get("priority", 5)),
                        record=tg.get("record", True),
                        monitor=tg.get("monitor", True),
                    )
                elif isinstance(tg, str):
                    # Simple format: just the name
                    talkgroups[tgid_int] = TalkgroupConfig(
                        tgid=tgid_int,
                        name=tg,
                    )
            except (ValueError, KeyError):
                continue  # Skip invalid entries

    except FileNotFoundError:
        import logging
        logging.getLogger(__name__).warning(f"Talkgroups file not found: {yaml_path}")
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error loading talkgroups from {yaml_path}: {e}")

    return talkgroups
