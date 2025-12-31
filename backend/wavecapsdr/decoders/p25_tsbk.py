"""P25 TSBK (Trunking Signaling Block) Message Parser.

This module parses TSBK messages from P25 control channels. TSBK messages
control trunking operations including:
- Voice channel grants
- Channel updates
- System information
- Registration and authentication

Framing/CRC/FEC conventions (Phase I control channel):
- Control bursts arrive as 196 dibits carrying a 96-bit TSBK after convolutional
  (rate 1/2) trellis decoding in `p25_frames.decode_tsdu`. Interleaving is
  removed before the parser sees the payload.
- Bits are MSB-first. Header layout: [LB|Protect|Opcode(6b)] + MFID (8b) +
  64-bit payload + CRC-16-CCITT (80-bit syndrome, init=0xFFFF, big-endian
  append). CRC validation uses `CCITT_80_CHECKSUMS` to mirror SDRTrunk.
- Payload bytes are encoded big-endian following SDRTrunk bit positions; helper
  encoders here return the 8-byte payloads as well as full 12-byte frames via
  `encode_control_frame`.

Typed message structs emitted by the parser:
- GroupVoiceGrantMessage, GroupVoiceGrantUpdateMessage, GroupVoiceGrantUpdateExplicitMessage
- UnitToUnitGrantMessage, UnitToUnitGrantUpdateMessage
- IdentifierUpdateVUMessage, IdentifierUpdateTDMA, IdentifierUpdateMessage (alias for 0x3D)
- RFSSStatusMessage, NetworkStatusMessage, AdjacentStatusMessage
- SystemServiceMessage, GroupAffiliationResponseMessage, DenyResponseMessage
- OpaqueTSBKMessage (unknown/manufacturer) and ParseErrorMessage for failures

TSBK Opcodes are defined in TIA-102.AABB-A. Key opcodes for trunking:
- 0x00: Group Voice Channel Grant
- 0x02: Group Voice Channel Grant Update
- 0x03: Unit to Unit Voice Channel Grant
- 0x28: RFSS Status Broadcast
- 0x34: Identifier Update (channel info)
- 0x3A: Adjacent Status Broadcast
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)

from wavecapsdr.decoders.p25_frames import CCITT_80_CHECKSUMS

from wavecapsdr.validation import (
    BASE_FREQ_MAX_MHZ,
    BASE_FREQ_MIN_MHZ,
    CHANNEL_ID_MAX,
    CHANNEL_ID_MIN,
    CHANNEL_SPACING_MAX_KHZ,
    CHANNEL_SPACING_MIN_KHZ,
    IDENTIFIER_MAX,
    RFSS_ID_MAX,
    SITE_ID_MAX,
    SYSTEM_ID_MAX,
    TGID_MAX,
    TX_OFFSET_MAX_MHZ,
    UNIT_ID_MAX,
    WACN_MAX,
    validate_float_range,
    validate_frequency_hz,
    validate_int_range,
)


class TSBKOpcode(IntEnum):
    """TSBK Opcode values (6 bits) - per SDRTrunk Opcode.java.

    OSP = Outbound Signaling Packet (tower to mobile)
    ISP = Inbound Signaling Packet (mobile to tower)
    """
    # Voice grants (OSP) - 0x00-0x07
    GRP_V_CH_GRANT = 0x00           # Group Voice Channel Grant
    GRP_V_CH_GRANT_UPDT = 0x02      # Group Voice Channel Grant Update
    GRP_V_CH_GRANT_UPDT_EXP = 0x03  # Group Voice Channel Grant Update Explicit
    UU_V_CH_GRANT = 0x04            # Unit to Unit Voice Channel Grant
    UU_ANS_REQ = 0x05               # Unit to Unit Answer Request
    UU_V_CH_GRANT_UPDT = 0x06       # Unit to Unit Voice Channel Grant Update

    # Telephone interconnect - 0x08-0x0A
    TEL_INT_CH_GRANT = 0x08         # Telephone Interconnect Voice Channel Grant
    TEL_INT_CH_GRANT_UPDT = 0x09    # Telephone Interconnect Voice Channel Grant Update
    TEL_INT_ANS_REQ = 0x0A          # Telephone Interconnect Answer Request

    # Data grants (obsolete) - 0x10-0x13
    IND_DATA_CH_GRANT = 0x10        # Individual Data Channel Grant (obsolete)
    GRP_DATA_CH_GRANT = 0x11        # Group Data Channel Grant (obsolete)
    GRP_DATA_CH_ANN = 0x12          # Group Data Channel Announcement (obsolete)
    GRP_DATA_CH_ANN_EXP = 0x13      # Group Data Channel Announcement Explicit (obsolete)

    # SNDCP data - 0x14-0x16
    SNDCP_CH_GNT = 0x14             # SNDCP Data Channel Grant
    SNDCP_PAGE_REQ = 0x15           # SNDCP Data Page Request
    SNDCP_CH_ANN_EXP = 0x16         # SNDCP Data Channel Announcement Explicit

    # Status/Message - 0x18-0x1F
    STATUS_UPDT = 0x18              # Status Update
    STATUS_QUERY = 0x1A             # Status Query
    MSG_UPDT = 0x1C                 # Message Update
    RADIO_MON_CMD = 0x1D            # Radio Unit Monitor Command
    RADIO_MON_ENH_CMD = 0x1E        # Radio Unit Monitor Enhanced Command
    CALL_ALRT = 0x1F                # Call Alert

    # Control responses - 0x20-0x27
    ACK_RSP = 0x20                  # Acknowledge Response
    QUE_RSP = 0x21                  # Queued Response
    EXT_FNCT_CMD = 0x24             # Extended Function Command
    DENY_RSP = 0x27                 # Deny Response

    # Affiliation/Registration - 0x28-0x2F
    GRP_AFF_RSP = 0x28              # Group Affiliation Response
    SCCB_EXP = 0x29                 # Secondary Control Channel Broadcast Explicit
    GRP_AFF_QUERY = 0x2A            # Group Affiliation Query
    LOC_REG_RSP = 0x2B              # Location Registration Response
    UNIT_REG_RSP = 0x2C             # Unit Registration Response
    UNIT_REG_CMD = 0x2D             # Unit Registration Command
    AUTH_CMD = 0x2E                 # Authentication Command
    UNIT_DEREG_ACK = 0x2F           # Unit De-registration Acknowledge

    # Synchronization/Authentication - 0x30-0x32
    TDMA_SYNC = 0x30                # TDMA Sync Broadcast
    AUTH_DMAN = 0x31                # Authentication Demand
    AUTH_FNE_RSP = 0x32             # Authentication FNE Response

    # Channel identification - 0x33-0x35
    IDEN_UP_TDMA = 0x33             # Identifier Update TDMA
    IDEN_UP_VU = 0x34               # Identifier Update VHF/UHF
    TIME_DATE_ANN = 0x35            # Time and Date Announcement

    # Roaming - 0x36-0x37
    ROAM_ADDR_CMD = 0x36            # Roaming Address Command
    ROAM_ADDR_UPDT = 0x37           # Roaming Address Update

    # System status broadcasts - 0x38-0x3D
    SYS_SRV_BCAST = 0x38            # System Service Broadcast
    SCCB = 0x39                     # Secondary Control Channel Broadcast
    RFSS_STS_BCAST = 0x3A           # RFSS Status Broadcast
    NET_STS_BCAST = 0x3B            # Network Status Broadcast
    ADJ_STS_BCAST = 0x3C            # Adjacent Status Broadcast
    IDEN_UP = 0x3D                  # Identifier Update


@dataclass
class ChannelIdentifier:
    """Channel identifier from IDEN_UP messages.

    Used to calculate actual frequencies from channel numbers.
    """
    identifier: int  # 4-bit channel ID
    bw: float  # Bandwidth (kHz)
    tx_offset: float  # TX offset (MHz)
    channel_spacing: float  # Channel spacing (kHz)
    base_freq: float  # Base frequency (MHz)

    def get_frequency(self, channel_number: int) -> float:
        """Calculate actual frequency from channel number.

        Returns frequency in Hz.
        """
        freq_mhz = self.base_freq + (channel_number * self.channel_spacing / 1000.0)
        return freq_mhz * 1e6


@dataclass
class VoiceGrant:
    """Voice channel grant information."""
    tgid: int  # Talkgroup ID
    source_id: int  # Source unit ID (caller)
    channel: int  # Channel number
    frequency_hz: float  # Actual frequency (if known)
    emergency: bool = False
    encrypted: bool = False


@dataclass
class SystemStatus:
    """System status information from control channel."""
    nac: int = 0  # Network Access Code
    system_id: int = 0  # System ID
    rfss_id: int = 0  # RF Subsystem ID
    site_id: int = 0  # Site ID
    channel: int = 0  # Control channel number
    services: int = 0  # Available services bitmap


@dataclass
class ChannelGrant:
    """Grant entry for group or unit-to-unit updates."""
    channel: int
    frequency_band: int
    channel_number: int
    tgid: int | None = None
    target_id: int | None = None
    frequency_hz: float | None = None


@dataclass
class TSBKMessage:
    """Typed representation of a decoded TSBK."""
    opcode: int
    opcode_name: str
    mfid: int
    message_type: str
    raw_data: bytes
    lb: int | None = field(default=None, init=False)
    protect: int | None = field(default=None, init=False)
    trellis_errors: int | None = field(default=None, init=False)
    last_block: int | None = field(default=None, init=False)

    def payload_bytes(self) -> bytes:
        """Return the raw 8-byte payload if available."""
        return self.raw_data

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the legacy dict format expected by the trunking stack."""
        payload = asdict(self)
        payload["type"] = payload.pop("message_type")
        payload["data"] = payload.pop("raw_data").hex()
        return payload


@dataclass
class GroupVoiceGrantMessage(TSBKMessage):
    """Group voice channel grant."""
    tgid: int
    source_id: int
    channel: int
    frequency_band: int
    channel_number: int
    frequency_hz: float
    emergency: bool
    encrypted: bool
    duplex: bool
    priority: int

    def payload_bytes(self) -> bytes:
        svc_opts = (
            (0x80 if self.emergency else 0)
            | (0x40 if self.encrypted else 0)
            | (0x20 if self.duplex else 0)
            | (self.priority & 0x07)
        )
        data = bytearray(8)
        data[0] = svc_opts
        data[1] = ((self.channel >> 12) & 0x0F) << 4 | ((self.channel >> 8) & 0x0F)
        data[2] = self.channel & 0xFF
        data[3] = (self.tgid >> 8) & 0xFF
        data[4] = self.tgid & 0xFF
        data[5] = (self.source_id >> 16) & 0xFF
        data[6] = (self.source_id >> 8) & 0xFF
        data[7] = self.source_id & 0xFF
        return bytes(data)


@dataclass
class GroupVoiceGrantUpdateMessage(TSBKMessage):
    """Grant update with one or two group calls."""
    grant1: ChannelGrant
    grant2: ChannelGrant | None = None

    def payload_bytes(self) -> bytes:
        data = bytearray(8)
        data[0] = ((self.grant1.frequency_band & 0x0F) << 4) | ((self.grant1.channel_number >> 8) & 0x0F)
        data[1] = self.grant1.channel_number & 0xFF
        if self.grant1.tgid is not None:
            data[2] = (self.grant1.tgid >> 8) & 0xFF
            data[3] = self.grant1.tgid & 0xFF

        if self.grant2:
            data[4] = ((self.grant2.frequency_band & 0x0F) << 4) | ((self.grant2.channel_number >> 8) & 0x0F)
            data[5] = self.grant2.channel_number & 0xFF
            if self.grant2.tgid is not None:
                data[6] = (self.grant2.tgid >> 8) & 0xFF
                data[7] = self.grant2.tgid & 0xFF
        return bytes(data)


@dataclass
class GroupVoiceGrantUpdateExplicitMessage(TSBKMessage):
    """Grant update with explicit downlink/uplink channels."""
    downlink_channel: int
    downlink_frequency_band: int
    downlink_channel_number: int
    downlink_frequency_hz: float
    uplink_channel: int
    uplink_frequency_band: int
    uplink_channel_number: int
    uplink_frequency_hz: float
    tgid: int
    emergency: bool
    encrypted: bool
    duplex: bool
    priority: int

    def payload_bytes(self) -> bytes:
        svc_opts = (
            (0x80 if self.emergency else 0)
            | (0x40 if self.encrypted else 0)
            | (0x20 if self.duplex else 0)
            | (self.priority & 0x07)
        )
        data = bytearray(8)
        data[0] = svc_opts
        data[2] = ((self.downlink_frequency_band & 0x0F) << 4) | ((self.downlink_channel_number >> 8) & 0x0F)
        data[3] = self.downlink_channel_number & 0xFF
        data[4] = ((self.uplink_frequency_band & 0x0F) << 4) | ((self.uplink_channel_number >> 8) & 0x0F)
        data[5] = self.uplink_channel_number & 0xFF
        data[6] = (self.tgid >> 8) & 0xFF
        data[7] = self.tgid & 0xFF
        return bytes(data)


@dataclass
class UnitToUnitGrantMessage(TSBKMessage):
    """Unit-to-unit voice grant."""
    target_id: int
    source_id: int
    channel: int
    frequency_band: int
    channel_number: int
    frequency_hz: float

    def payload_bytes(self) -> bytes:
        data = bytearray(8)
        data[0] = ((self.frequency_band & 0x0F) << 4) | ((self.channel_number >> 8) & 0x0F)
        data[1] = self.channel_number & 0xFF
        data[2] = (self.target_id >> 16) & 0xFF
        data[3] = (self.target_id >> 8) & 0xFF
        data[4] = self.target_id & 0xFF
        data[5] = (self.source_id >> 16) & 0xFF
        data[6] = (self.source_id >> 8) & 0xFF
        data[7] = self.source_id & 0xFF
        return bytes(data)


@dataclass
class UnitToUnitGrantUpdateMessage(TSBKMessage):
    """Unit-to-unit voice grant update."""
    grant1: ChannelGrant
    grant2: ChannelGrant | None = None

    def payload_bytes(self) -> bytes:
        data = bytearray(8)
        data[0] = ((self.grant1.frequency_band & 0x0F) << 4) | ((self.grant1.channel_number >> 8) & 0x0F)
        data[1] = self.grant1.channel_number & 0xFF
        if self.grant1.target_id is not None:
            data[2] = (self.grant1.target_id >> 16) & 0xFF
            data[3] = (self.grant1.target_id >> 8) & 0xFF
            data[4] = self.grant1.target_id & 0xFF

        if self.grant2:
            data[5] = ((self.grant2.frequency_band & 0x0F) << 4) | ((self.grant2.channel_number >> 8) & 0x0F)
            data[6] = self.grant2.channel_number & 0xFF
            data[7] = (self.grant2.target_id or 0) & 0xFF
        return bytes(data)


@dataclass
class IdentifierUpdateVUMessage(TSBKMessage):
    """Identifier update for VHF/UHF FDMA."""
    identifier: int
    bandwidth_khz: float
    bandwidth_code: int
    tx_offset_sign: bool
    tx_offset_hz: float
    tx_offset_mhz: float
    channel_spacing_khz: float
    base_freq_mhz: float


@dataclass
class IdentifierUpdateTDMA(TSBKMessage):
    """Identifier update for TDMA."""
    identifier: int
    channel_type: int
    access_type: str
    slot_count: int
    tx_offset_sign: bool
    tx_offset_hz: float
    tx_offset_mhz: float
    channel_spacing_khz: float
    base_freq_mhz: float


@dataclass
class SystemServiceMessage(TSBKMessage):
    """System service broadcast message."""
    services_available: int
    services_supported: int
    composite_control: bool
    data_services: bool
    voice_services: bool
    registration: bool
    authentication: bool

    def payload_bytes(self) -> bytes:
        data = bytearray(8)
        data[0] = (self.services_available >> 16) & 0xFF
        data[1] = (self.services_available >> 8) & 0xFF
        data[2] = self.services_available & 0xFF
        data[3] = (self.services_supported >> 16) & 0xFF
        data[4] = (self.services_supported >> 8) & 0xFF
        data[5] = self.services_supported & 0xFF
        return bytes(data)


@dataclass
class GroupAffiliationResponseMessage(TSBKMessage):
    """Affiliation response."""
    response: int
    success: bool
    tgid: int
    announcement_group: int
    target_id: int

    def payload_bytes(self) -> bytes:
        data = bytearray(8)
        data[0] = self.response & 0xFF
        data[1] = (self.tgid >> 8) & 0xFF
        data[2] = self.tgid & 0xFF
        data[3] = (self.announcement_group >> 8) & 0xFF
        data[4] = self.announcement_group & 0xFF
        data[5] = (self.target_id >> 8) & 0xFF
        data[6] = self.target_id & 0xFF
        return bytes(data)


@dataclass
class DenyResponseMessage(TSBKMessage):
    """Deny response."""
    service_type: int
    reason: int
    reason_text: str
    target_address: int

    def payload_bytes(self) -> bytes:
        data = bytearray(8)
        data[0] = (self.service_type & 0x3F) << 2
        data[1] = self.reason & 0xFF
        data[2] = (self.target_address >> 16) & 0xFF
        data[3] = (self.target_address >> 8) & 0xFF
        data[4] = self.target_address & 0xFF
        return bytes(data)


@dataclass
class RFSSStatusMessage(TSBKMessage):
    """RFSS status broadcast."""
    lra: int
    active_network_connection: bool
    system_id: int
    rfss_id: int
    site_id: int
    channel: int
    frequency_band: int
    channel_number: int
    service_class: int


@dataclass
class AdjacentStatusMessage(TSBKMessage):
    """Adjacent status broadcast."""
    lra: int
    active_network_connection: bool
    system_id: int
    rfss_id: int
    site_id: int
    channel: int
    frequency_band: int
    channel_number: int
    service_class: int


@dataclass
class NetworkStatusMessage(TSBKMessage):
    """Network status broadcast."""
    lra: int
    wacn: int
    system_id: int
    channel: int
    frequency_band: int
    channel_number: int
    service_class: int


@dataclass
class IdentifierUpdateMessage(IdentifierUpdateVUMessage):
    """Identifier update using opcode 0x3D (alias of IDEN_UP_VU)."""


@dataclass
class OpaqueTSBKMessage(TSBKMessage):
    """Catch-all message when the opcode is not parsed."""
    data: bytes = field(default_factory=bytes)

    def payload_bytes(self) -> bytes:
        return self.data or self.raw_data

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["data"] = (self.data or self.raw_data).hex()
        return payload


@dataclass
class ParseErrorMessage(TSBKMessage):
    """Parsing failed due to validation."""
    error: str


SUPPORTED_TSBK_STRUCTS: tuple[type[TSBKMessage], ...] = (
    GroupVoiceGrantMessage,
    GroupVoiceGrantUpdateMessage,
    GroupVoiceGrantUpdateExplicitMessage,
    UnitToUnitGrantMessage,
    UnitToUnitGrantUpdateMessage,
    IdentifierUpdateVUMessage,
    IdentifierUpdateTDMA,
    IdentifierUpdateMessage,
    SystemServiceMessage,
    GroupAffiliationResponseMessage,
    DenyResponseMessage,
    RFSSStatusMessage,
    AdjacentStatusMessage,
    NetworkStatusMessage,
)


class TSBKParser:
    """Parser for TSBK messages.

    Maintains channel identifier table and parses TSBK opcodes
    into structured data for the trunking controller.
    """

    def __init__(self) -> None:
        # Channel identifier table (from IDEN_UP messages)
        self._channel_ids: dict[int, ChannelIdentifier] = {}

        # System status
        self.system_status = SystemStatus()

        # Callbacks
        self.on_voice_grant: Callable[[VoiceGrant], None] | None = None
        self.on_system_update: Callable[[SystemStatus], None] | None = None

    @staticmethod
    def supported_structs() -> Sequence[type[TSBKMessage]]:
        """Return the typed message classes emitted by the parser."""
        return SUPPORTED_TSBK_STRUCTS

    def add_channel_id(self, ident: ChannelIdentifier) -> None:
        """Add or update channel identifier."""
        self._channel_ids[ident.identifier] = ident
        logger.debug(f"Channel ID {ident.identifier}: base={ident.base_freq} MHz, "
                    f"spacing={ident.channel_spacing} kHz")

    def get_frequency(self, channel: int) -> float:
        """Get frequency for channel number.

        Channel format: IIII CCCC CCCC CCCC (4-bit ID + 12-bit channel)

        Returns frequency in Hz, or 0 if channel ID unknown.
        """
        ident_id = (channel >> 12) & 0xF
        channel_num = channel & 0xFFF

        ident = self._channel_ids.get(ident_id)
        if ident is None:
            logger.warning(f"Unknown channel identifier: {ident_id}")
            return 0.0

        return ident.get_frequency(channel_num)

    def parse(self, opcode: int, mfid: int, data: bytes) -> TSBKMessage:
        """Parse TSBK message into a typed model.

        Args:
            opcode: 6-bit opcode
            mfid: Manufacturer ID (0 = standard)
            data: 8-byte data payload (64 bits per TIA-102.AABB-A)

        Returns:
            Typed TSBKMessage describing the decoded payload.
        """
        if len(data) != 8:
            raise ValueError(f"TSBK data must be 8 bytes (got {len(data)})")

        result: dict[str, Any] = {
            'opcode': opcode,
            'opcode_name': self._opcode_name(opcode),
            'mfid': mfid,
            'raw_data': data,
        }

        if mfid != 0:
            # Non-standard (manufacturer-specific) message
            return OpaqueTSBKMessage(
                opcode=opcode,
                opcode_name=self._opcode_name(opcode),
                mfid=mfid,
                message_type='MANUFACTURER_SPECIFIC',
                raw_data=data,
                data=data,
            )

        # Parse based on opcode
        try:
            if opcode == TSBKOpcode.GRP_V_CH_GRANT:
                self._parse_grp_v_ch_grant(data, result)
            elif opcode == TSBKOpcode.GRP_V_CH_GRANT_UPDT:
                self._parse_grp_v_ch_grant_updt(data, result)
            elif opcode == TSBKOpcode.GRP_V_CH_GRANT_UPDT_EXP:
                self._parse_grp_v_ch_grant_updt_exp(data, result)
            elif opcode == TSBKOpcode.UU_V_CH_GRANT:
                self._parse_uu_v_ch_grant(data, result)
            elif opcode == TSBKOpcode.RFSS_STS_BCAST:
                self._parse_rfss_sts_bcast(data, result)
            elif opcode == TSBKOpcode.NET_STS_BCAST:
                self._parse_net_sts_bcast(data, result)
            elif opcode == TSBKOpcode.ADJ_STS_BCAST:
                self._parse_adj_sts_bcast(data, result)
            elif opcode == TSBKOpcode.IDEN_UP_VU:
                self._parse_iden_up_vu(data, result)
            elif opcode == TSBKOpcode.IDEN_UP_TDMA:
                self._parse_iden_up_tdma(data, result)
            elif opcode == TSBKOpcode.SYS_SRV_BCAST:
                self._parse_sys_srv_bcast(data, result)
            elif opcode == TSBKOpcode.GRP_AFF_RSP:
                self._parse_grp_aff_rsp(data, result)
            elif opcode == TSBKOpcode.DENY_RSP:
                self._parse_deny_rsp(data, result)
            elif opcode == TSBKOpcode.IDEN_UP:
                # IDEN_UP (0x3D) uses same format as IDEN_UP_VU (0x34)
                self._parse_iden_up_vu(data, result)
                result['type'] = 'IDENTIFIER_UPDATE'
            elif opcode == TSBKOpcode.UU_V_CH_GRANT_UPDT:
                self._parse_uu_v_ch_grant_updt(data, result)
            elif opcode == TSBKOpcode.ACK_RSP:
                result['type'] = 'ACKNOWLEDGE_RESPONSE'
                result['data'] = data.hex()
            elif opcode == TSBKOpcode.QUE_RSP:
                result['type'] = 'QUEUED_RESPONSE'
                result['data'] = data.hex()
            elif opcode == TSBKOpcode.EXT_FNCT_CMD:
                result['type'] = 'EXTENDED_FUNCTION_COMMAND'
                result['data'] = data.hex()
            elif opcode == TSBKOpcode.SCCB:
                result['type'] = 'SECONDARY_CONTROL_CHANNEL'
                result['data'] = data.hex()
            else:
                result['type'] = 'UNKNOWN'
                result['data'] = data.hex()
                logger.debug(f"Unknown TSBK opcode 0x{opcode:02X}: {data.hex()}")

            self._validate_result(result)
            return self._build_message(result, data)
        except Exception as e:
            logger.warning(f"Error parsing TSBK opcode {opcode:02X}: {e}")
            return ParseErrorMessage(
                opcode=opcode,
                opcode_name=self._opcode_name(opcode),
                mfid=mfid,
                message_type='PARSE_ERROR',
                raw_data=data,
                error=str(e),
            )

    def _opcode_name(self, opcode: int) -> str:
        """Get human-readable opcode name."""
        try:
            return TSBKOpcode(opcode).name
        except ValueError:
            return f"UNKNOWN_0x{opcode:02X}"

    def _require_int_range(
        self,
        value: Any,
        min_value: int,
        max_value: int,
        label: str,
    ) -> int:
        ok, reason = validate_int_range(value, min_value, max_value, label)
        if not ok:
            raise ValueError(reason)
        return int(value)

    def _require_float_range(
        self,
        value: Any,
        min_value: float,
        max_value: float,
        label: str,
    ) -> float:
        ok, reason = validate_float_range(value, min_value, max_value, label)
        if not ok:
            raise ValueError(reason)
        return float(value)

    def _require_frequency(self, value: Any, label: str) -> float:
        value_f = float(value)
        ok, reason = validate_frequency_hz(value_f)
        if not ok:
            raise ValueError(f"{label}: {reason}")
        return value_f

    def _validate_grant_fields(self, grant: dict[str, Any], id_label: str) -> None:
        self._require_int_range(grant.get("channel"), CHANNEL_ID_MIN, CHANNEL_ID_MAX, "channel")
        self._require_int_range(grant.get(id_label), 1, TGID_MAX, id_label)
        freq = grant.get("frequency_hz")
        if freq is not None and float(freq) > 0:
            self._require_frequency(freq, "frequency_hz")

    def _validate_result(self, result: dict[str, Any]) -> None:
        msg_type = result.get("type")
        if msg_type in ("UNKNOWN", "MANUFACTURER_SPECIFIC"):
            return

        if msg_type == "GROUP_VOICE_GRANT":
            self._require_int_range(result.get("tgid"), 1, TGID_MAX, "tgid")
            self._require_int_range(result.get("source_id"), 0, UNIT_ID_MAX, "source_id")
            self._require_int_range(result.get("channel"), CHANNEL_ID_MIN, CHANNEL_ID_MAX, "channel")
            freq = result.get("frequency_hz")
            if freq is not None and float(freq) > 0:
                self._require_frequency(freq, "frequency_hz")
            return

        if msg_type == "GROUP_VOICE_GRANT_UPDATE_EXPLICIT":
            self._require_int_range(result.get("tgid"), 1, TGID_MAX, "tgid")
            dl = result.get("downlink_channel")
            if dl is not None:
                self._require_int_range(dl, CHANNEL_ID_MIN, CHANNEL_ID_MAX, "downlink_channel")
            ul = result.get("uplink_channel")
            if ul is not None:
                self._require_int_range(ul, CHANNEL_ID_MIN, CHANNEL_ID_MAX, "uplink_channel")
            return

        if msg_type in ("GROUP_VOICE_GRANT_UPDATE", "UNIT_TO_UNIT_GRANT_UPDATE"):
            grant1 = result.get("grant1")
            if isinstance(grant1, dict):
                id_label = "target_id" if msg_type == "UNIT_TO_UNIT_GRANT_UPDATE" else "tgid"
                self._validate_grant_fields(grant1, id_label)
            grant2 = result.get("grant2")
            if isinstance(grant2, dict):
                id_label = "target_id" if msg_type == "UNIT_TO_UNIT_GRANT_UPDATE" else "tgid"
                self._validate_grant_fields(grant2, id_label)
            return

        if msg_type == "UNIT_TO_UNIT_GRANT":
            self._require_int_range(result.get("target_id"), 0, UNIT_ID_MAX, "target_id")
            self._require_int_range(result.get("source_id"), 0, UNIT_ID_MAX, "source_id")
            self._require_int_range(result.get("channel"), CHANNEL_ID_MIN, CHANNEL_ID_MAX, "channel")
            freq = result.get("frequency_hz")
            if freq is not None and float(freq) > 0:
                self._require_frequency(freq, "frequency_hz")
            return

        if msg_type in ("RFSS_STATUS", "ADJACENT_STATUS"):
            self._require_int_range(result.get("system_id"), 0, SYSTEM_ID_MAX, "system_id")
            self._require_int_range(result.get("rfss_id"), 0, RFSS_ID_MAX, "rfss_id")
            self._require_int_range(result.get("site_id"), 0, SITE_ID_MAX, "site_id")
            self._require_int_range(result.get("channel"), 0, CHANNEL_ID_MAX, "channel")
            return

        if msg_type == "NETWORK_STATUS":
            self._require_int_range(result.get("wacn"), 0, WACN_MAX, "wacn")
            self._require_int_range(result.get("system_id"), 0, SYSTEM_ID_MAX, "system_id")
            self._require_int_range(result.get("channel"), 0, CHANNEL_ID_MAX, "channel")
            return

        if msg_type in ("IDENTIFIER_UPDATE_VU", "IDENTIFIER_UPDATE_TDMA", "IDENTIFIER_UPDATE"):
            self._require_int_range(result.get("identifier"), 0, IDENTIFIER_MAX, "identifier")
            self._require_float_range(
                result.get("channel_spacing_khz"),
                CHANNEL_SPACING_MIN_KHZ,
                CHANNEL_SPACING_MAX_KHZ,
                "channel_spacing_khz",
            )
            self._require_float_range(
                result.get("base_freq_mhz"),
                BASE_FREQ_MIN_MHZ,
                BASE_FREQ_MAX_MHZ,
                "base_freq_mhz",
            )
            tx_offset_mhz = result.get("tx_offset_mhz")
            if tx_offset_mhz is not None:
                self._require_float_range(
                    tx_offset_mhz,
                    -TX_OFFSET_MAX_MHZ,
                    TX_OFFSET_MAX_MHZ,
                    "tx_offset_mhz",
                )
            return

    def _build_message(self, result: dict[str, Any], raw_data: bytes) -> TSBKMessage:
        """Convert validated dict payload to a typed TSBKMessage."""
        msg_type = result.get("type", "UNKNOWN")
        opcode = int(result["opcode"])
        opcode_name = str(result.get("opcode_name", ""))
        mfid = int(result.get("mfid", 0))
        common_args = dict(
            opcode=opcode,
            opcode_name=opcode_name,
            mfid=mfid,
            message_type=msg_type,
            raw_data=raw_data,
        )

        if msg_type == "GROUP_VOICE_GRANT":
            return GroupVoiceGrantMessage(
                **common_args,
                tgid=int(result["tgid"]),
                source_id=int(result["source_id"]),
                channel=int(result["channel"]),
                frequency_band=int(result["frequency_band"]),
                channel_number=int(result["channel_number"]),
                frequency_hz=float(result.get("frequency_hz", 0.0)),
                emergency=bool(result.get("emergency", False)),
                encrypted=bool(result.get("encrypted", False)),
                duplex=bool(result.get("duplex", False)),
                priority=int(result.get("priority", 0)),
            )

        if msg_type == "GROUP_VOICE_GRANT_UPDATE":
            grant1 = result.get("grant1") or {}
            grant2 = result.get("grant2") or None
            return GroupVoiceGrantUpdateMessage(
                **common_args,
                grant1=self._grant_from_dict(grant1, False),
                grant2=self._grant_from_dict(grant2, False) if isinstance(grant2, dict) else None,
            )

        if msg_type == "GROUP_VOICE_GRANT_UPDATE_EXPLICIT":
            return GroupVoiceGrantUpdateExplicitMessage(
                **common_args,
                downlink_channel=int(result["downlink_channel"]),
                downlink_frequency_band=int(result["downlink_frequency_band"]),
                downlink_channel_number=int(result["downlink_channel_number"]),
                downlink_frequency_hz=float(result.get("downlink_frequency_hz", 0.0)),
                uplink_channel=int(result["uplink_channel"]),
                uplink_frequency_band=int(result["uplink_frequency_band"]),
                uplink_channel_number=int(result["uplink_channel_number"]),
                uplink_frequency_hz=float(result.get("uplink_frequency_hz", 0.0)),
                tgid=int(result["tgid"]),
                emergency=bool(result.get("emergency", False)),
                encrypted=bool(result.get("encrypted", False)),
                duplex=bool(result.get("duplex", False)),
                priority=int(result.get("priority", 0)),
            )

        if msg_type == "UNIT_TO_UNIT_GRANT":
            return UnitToUnitGrantMessage(
                **common_args,
                target_id=int(result["target_id"]),
                source_id=int(result["source_id"]),
                channel=int(result["channel"]),
                frequency_band=int(result["frequency_band"]),
                channel_number=int(result["channel_number"]),
                frequency_hz=float(result.get("frequency_hz", 0.0)),
            )

        if msg_type == "UNIT_TO_UNIT_GRANT_UPDATE":
            grant1 = self._grant_from_dict(result.get("grant1") or {}, True)
            grant2_dict = result.get("grant2")
            grant2 = self._grant_from_dict(grant2_dict, True) if isinstance(grant2_dict, dict) else None
            return UnitToUnitGrantUpdateMessage(
                **common_args,
                grant1=grant1,
                grant2=grant2,
            )

        if msg_type == "IDENTIFIER_UPDATE_VU":
            return IdentifierUpdateVUMessage(
                **common_args,
                identifier=int(result["identifier"]),
                bandwidth_khz=float(result["bandwidth_khz"]),
                bandwidth_code=int(result["bandwidth_code"]),
                tx_offset_sign=bool(result["tx_offset_sign"]),
                tx_offset_hz=float(result["tx_offset_hz"]),
                tx_offset_mhz=float(result["tx_offset_mhz"]),
                channel_spacing_khz=float(result["channel_spacing_khz"]),
                base_freq_mhz=float(result["base_freq_mhz"]),
            )

        if msg_type == "IDENTIFIER_UPDATE_TDMA":
            return IdentifierUpdateTDMA(
                **common_args,
                identifier=int(result["identifier"]),
                channel_type=int(result["channel_type"]),
                access_type=str(result["access_type"]),
                slot_count=int(result["slot_count"]),
                tx_offset_sign=bool(result["tx_offset_sign"]),
                tx_offset_hz=float(result["tx_offset_hz"]),
                tx_offset_mhz=float(result["tx_offset_mhz"]),
                channel_spacing_khz=float(result["channel_spacing_khz"]),
                base_freq_mhz=float(result["base_freq_mhz"]),
            )

        if msg_type == "IDENTIFIER_UPDATE":
            return IdentifierUpdateMessage(
                **common_args,
                identifier=int(result["identifier"]),
                bandwidth_khz=float(result["bandwidth_khz"]),
                bandwidth_code=int(result["bandwidth_code"]),
                tx_offset_sign=bool(result["tx_offset_sign"]),
                tx_offset_hz=float(result["tx_offset_hz"]),
                tx_offset_mhz=float(result["tx_offset_mhz"]),
                channel_spacing_khz=float(result["channel_spacing_khz"]),
                base_freq_mhz=float(result["base_freq_mhz"]),
            )

        if msg_type == "SYSTEM_SERVICE":
            return SystemServiceMessage(
                **common_args,
                services_available=int(result["services_available"]),
                services_supported=int(result["services_supported"]),
                composite_control=bool(result.get("composite_control", False)),
                data_services=bool(result.get("data_services", False)),
                voice_services=bool(result.get("voice_services", False)),
                registration=bool(result.get("registration", False)),
                authentication=bool(result.get("authentication", False)),
            )

        if msg_type == "GROUP_AFFILIATION_RESPONSE":
            return GroupAffiliationResponseMessage(
                **common_args,
                response=int(result["response"]),
                success=bool(result.get("success", False)),
                tgid=int(result["tgid"]),
                announcement_group=int(result["announcement_group"]),
                target_id=int(result["target_id"]),
            )

        if msg_type == "DENY_RESPONSE":
            return DenyResponseMessage(
                **common_args,
                service_type=int(result["service_type"]),
                reason=int(result["reason"]),
                reason_text=str(result["reason_text"]),
                target_address=int(result["target_address"]),
            )

        if msg_type == "RFSS_STATUS":
            return RFSSStatusMessage(
                **common_args,
                lra=int(result["lra"]),
                active_network_connection=bool(result.get("active_network_connection", False)),
                system_id=int(result["system_id"]),
                rfss_id=int(result["rfss_id"]),
                site_id=int(result["site_id"]),
                channel=int(result["channel"]),
                frequency_band=int(result["frequency_band"]),
                channel_number=int(result["channel_number"]),
                service_class=int(result["service_class"]),
            )

        if msg_type == "ADJACENT_STATUS":
            return AdjacentStatusMessage(
                **common_args,
                lra=int(result["lra"]),
                active_network_connection=bool(result.get("active_network_connection", False)),
                system_id=int(result["system_id"]),
                rfss_id=int(result["rfss_id"]),
                site_id=int(result["site_id"]),
                channel=int(result["channel"]),
                frequency_band=int(result["frequency_band"]),
                channel_number=int(result["channel_number"]),
                service_class=int(result["service_class"]),
            )

        if msg_type == "NETWORK_STATUS":
            return NetworkStatusMessage(
                **common_args,
                lra=int(result["lra"]),
                wacn=int(result["wacn"]),
                system_id=int(result["system_id"]),
                channel=int(result["channel"]),
                frequency_band=int(result["frequency_band"]),
                channel_number=int(result["channel_number"]),
                service_class=int(result["service_class"]),
            )

        if msg_type == "PARSE_ERROR":
            return ParseErrorMessage(
                **common_args,
                error=str(result.get("error", "failed to parse message")),
            )

        # Fallback to opaque raw message
        return OpaqueTSBKMessage(
            **common_args,
            data=raw_data,
        )

    def _grant_from_dict(self, grant: dict[str, Any], use_target: bool) -> ChannelGrant:
        """Convert dict grant payloads to ChannelGrant dataclass."""
        if not grant:
            return ChannelGrant(channel=0, frequency_band=0, channel_number=0)
        return ChannelGrant(
            channel=int(grant.get("channel", 0)),
            frequency_band=int(grant.get("frequency_band", 0)),
            channel_number=int(grant.get("channel_number", 0)),
            tgid=None if use_target else int(grant.get("tgid", 0)),
            target_id=int(grant.get("target_id", 0)) if use_target else None,
            frequency_hz=float(grant.get("frequency_hz", 0.0)) if grant.get("frequency_hz") is not None else None,
        )

    def _parse_grp_v_ch_grant(self, data: bytes, result: dict[str, Any]) -> None:
        """Parse Group Voice Channel Grant.

        Data format (64 bits per TIA-102.AABB-A, SDRTrunk bit positions):
        - Bits 16-23: Service Options (8 bits)
        - Bits 24-27: Frequency Band (4 bits)
        - Bits 28-39: Channel Number (12 bits)
        - Bits 40-55: Group Address (16 bits)
        - Bits 56-79: Source Address (24 bits)
        """
        result['type'] = 'GROUP_VOICE_GRANT'

        # Service options (byte 0 = bits 16-23)
        svc_opts = data[0]

        # Frequency band (upper 4 bits of byte 1 = bits 24-27)
        freq_band = (data[1] >> 4) & 0x0F

        # Channel number (lower 4 bits of byte 1 + byte 2 = bits 28-39)
        channel_num = ((data[1] & 0x0F) << 8) | data[2]

        # Combine into 16-bit channel ID (4-bit band + 12-bit channel)
        channel = (freq_band << 12) | channel_num

        # Group address (bytes 3-4 = bits 40-55)
        tgid = (data[3] << 8) | data[4]

        # Source address (bytes 5-7 = bits 56-79)
        source = (data[5] << 16) | (data[6] << 8) | data[7]

        result['emergency'] = bool(svc_opts & 0x80)
        result['encrypted'] = bool(svc_opts & 0x40)
        result['duplex'] = bool(svc_opts & 0x20)
        result['priority'] = (svc_opts >> 0) & 0x07

        result['channel'] = channel
        result['frequency_band'] = freq_band
        result['channel_number'] = channel_num
        result['tgid'] = tgid
        result['source_id'] = source

        # Calculate frequency
        freq = self.get_frequency(channel)
        result['frequency_hz'] = freq
        result['frequency_mhz'] = freq / 1e6 if freq else 0

        logger.info(f"Voice Grant: TGID={tgid} BAND={freq_band} CH={channel_num} "
                   f"FREQ={result['frequency_mhz']:.4f} MHz "
                   f"SRC={source}")

        # Fire callback
        if self.on_voice_grant:
            grant = VoiceGrant(
                tgid=tgid,
                source_id=source,
                channel=channel,
                frequency_hz=freq,
                emergency=result['emergency'],
                encrypted=result['encrypted']
            )
            self.on_voice_grant(grant)

    def _parse_grp_v_ch_grant_updt_exp(self, data: bytes, result: dict[str, Any]) -> None:
        """Parse Group Voice Channel Grant Update Explicit.

        Contains explicit downlink/uplink frequencies instead of channel identifiers.

        Data format per SDRTrunk:
        - Bits 16-23: Service Options (8 bits)
        - Bits 24-31: Reserved (8 bits)
        - Bits 32-35: Downlink Frequency Band (4 bits)
        - Bits 36-47: Downlink Channel Number (12 bits)
        - Bits 48-51: Uplink Frequency Band (4 bits)
        - Bits 52-63: Uplink Channel Number (12 bits)
        - Bits 64-79: Group Address (16 bits)
        """
        result['type'] = 'GROUP_VOICE_GRANT_UPDATE_EXPLICIT'

        # Service options (byte 0)
        svc_opts = data[0]

        # Reserved (byte 1)
        # Skip

        # Downlink: Frequency band (upper 4 bits of byte 2)
        dl_freq_band = (data[2] >> 4) & 0x0F
        # Downlink: Channel number (lower 4 bits of byte 2 + byte 3)
        dl_channel_num = ((data[2] & 0x0F) << 8) | data[3]
        dl_channel = (dl_freq_band << 12) | dl_channel_num

        # Uplink: Frequency band (upper 4 bits of byte 4)
        ul_freq_band = (data[4] >> 4) & 0x0F
        # Uplink: Channel number (lower 4 bits of byte 4 + byte 5)
        ul_channel_num = ((data[4] & 0x0F) << 8) | data[5]
        ul_channel = (ul_freq_band << 12) | ul_channel_num

        # Group address (bytes 6-7)
        tgid = (data[6] << 8) | data[7]

        result['emergency'] = bool(svc_opts & 0x80)
        result['encrypted'] = bool(svc_opts & 0x40)
        result['duplex'] = bool(svc_opts & 0x20)
        result['priority'] = (svc_opts >> 0) & 0x07

        result['downlink_channel'] = dl_channel
        result['downlink_frequency_band'] = dl_freq_band
        result['downlink_channel_number'] = dl_channel_num
        result['downlink_frequency_hz'] = self.get_frequency(dl_channel)

        result['uplink_channel'] = ul_channel
        result['uplink_frequency_band'] = ul_freq_band
        result['uplink_channel_number'] = ul_channel_num
        result['uplink_frequency_hz'] = self.get_frequency(ul_channel)

        result['tgid'] = tgid

        logger.info(f"Voice Grant Update Explicit: TGID={tgid} "
                   f"DL_BAND={dl_freq_band} DL_CH={dl_channel_num} "
                   f"UL_BAND={ul_freq_band} UL_CH={ul_channel_num}")

    def _parse_grp_v_ch_grant_updt(self, data: bytes, result: dict[str, Any]) -> None:
        """Parse Group Voice Channel Grant Update.

        Contains two channel/group pairs for active calls.

        Data format per SDRTrunk:
        - Bits 16-19: Frequency Band A (4 bits)
        - Bits 20-31: Channel Number A (12 bits)
        - Bits 32-47: Group Address A (16 bits)
        - Bits 48-51: Frequency Band B (4 bits)
        - Bits 52-63: Channel Number B (12 bits)
        - Bits 64-79: Group Address B (16 bits)
        """
        result['type'] = 'GROUP_VOICE_GRANT_UPDATE'

        # Grant A: Frequency band (upper 4 bits of byte 0)
        freq_band_a = (data[0] >> 4) & 0x0F
        # Channel number (lower 4 bits of byte 0 + byte 1)
        channel_num_a = ((data[0] & 0x0F) << 8) | data[1]
        channel_a = (freq_band_a << 12) | channel_num_a
        # Group address (bytes 2-3)
        tgid_a = (data[2] << 8) | data[3]

        result['grant1'] = {
            'channel': channel_a,
            'frequency_band': freq_band_a,
            'channel_number': channel_num_a,
            'tgid': tgid_a,
            'frequency_hz': self.get_frequency(channel_a)
        }

        # Grant B: Frequency band (upper 4 bits of byte 4)
        freq_band_b = (data[4] >> 4) & 0x0F
        # Channel number (lower 4 bits of byte 4 + byte 5)
        channel_num_b = ((data[4] & 0x0F) << 8) | data[5]
        channel_b = (freq_band_b << 12) | channel_num_b
        # Group address (bytes 6-7)
        tgid_b = (data[6] << 8) | data[7]

        # Only include grant B if it's valid (non-zero and different from A)
        if tgid_b != 0 and tgid_b != tgid_a:
            result['grant2'] = {
                'channel': channel_b,
                'frequency_band': freq_band_b,
                'channel_number': channel_num_b,
                'tgid': tgid_b,
                'frequency_hz': self.get_frequency(channel_b)
            }

    def _parse_uu_v_ch_grant(self, data: bytes, result: dict[str, Any]) -> None:
        """Parse Unit to Unit Voice Channel Grant.

        Data format per SDRTrunk (NO service options):
        - Bits 16-19: Frequency Band (4 bits)
        - Bits 20-31: Channel Number (12 bits)
        - Bits 32-55: Target Address (24 bits)
        - Bits 56-79: Source Address (24 bits)

        Note: This opcode has NO service options field - channel starts at bit 16.
        """
        result['type'] = 'UNIT_TO_UNIT_GRANT'

        # Frequency band (upper 4 bits of byte 0 = bits 16-19)
        freq_band = (data[0] >> 4) & 0x0F
        # Channel number (lower 4 bits of byte 0 + byte 1 = bits 20-31)
        channel_num = ((data[0] & 0x0F) << 8) | data[1]
        channel = (freq_band << 12) | channel_num

        # Target address (bytes 2-4 = bits 32-55)
        target = (data[2] << 16) | (data[3] << 8) | data[4]
        # Source address (bytes 5-7 = bits 56-79)
        source = (data[5] << 16) | (data[6] << 8) | data[7]

        result['channel'] = channel
        result['frequency_band'] = freq_band
        result['channel_number'] = channel_num
        result['target_id'] = target
        result['source_id'] = source
        result['frequency_hz'] = self.get_frequency(channel)
        result['emergency'] = False  # No service options in this opcode
        result['encrypted'] = False

        logger.info(f"Unit-to-Unit Grant: TARGET={target} SRC={source} CH={channel} "
                   f"FREQ={result['frequency_hz']/1e6:.4f} MHz")

    def _parse_uu_v_ch_grant_updt(self, data: bytes, result: dict[str, Any]) -> None:
        """Parse Unit to Unit Voice Channel Grant Update.

        Data format per SDRTrunk:
        - Bits 16-19: Frequency Band A (4 bits)
        - Bits 20-31: Channel Number A (12 bits)
        - Bits 32-55: Target Address A (24 bits)
        - Bits 56-59: Frequency Band B (4 bits)
        - Bits 60-71: Channel Number B (12 bits)
        - Bits 72-95: Target Address B (24 bits)
        """
        result['type'] = 'UNIT_TO_UNIT_GRANT_UPDATE'

        # Grant A
        freq_band_a = (data[0] >> 4) & 0x0F
        channel_num_a = ((data[0] & 0x0F) << 8) | data[1]
        channel_a = (freq_band_a << 12) | channel_num_a
        target_a = (data[2] << 16) | (data[3] << 8) | data[4]

        result['grant1'] = {
            'channel': channel_a,
            'frequency_band': freq_band_a,
            'channel_number': channel_num_a,
            'target_id': target_a,
            'frequency_hz': self.get_frequency(channel_a)
        }

        # Grant B (if present - check if non-zero)
        # Note: UU grant update only has 6 data bytes, remaining 2 might be padding
        if len(data) >= 8:
            freq_band_b = (data[5] >> 4) & 0x0F
            channel_num_b = ((data[5] & 0x0F) << 8) | data[6]
            if channel_num_b != 0:
                channel_b = (freq_band_b << 12) | channel_num_b
                target_b = data[7] if len(data) > 7 else 0

                result['grant2'] = {
                    'channel': channel_b,
                    'frequency_band': freq_band_b,
                    'channel_number': channel_num_b,
                    'target_id': target_b,
                    'frequency_hz': self.get_frequency(channel_b)
                }

    def _parse_rfss_sts_bcast(self, data: bytes, result: dict[str, Any]) -> None:
        """Parse RFSS Status Broadcast.

        Contains RFSS and site identification.

        Data format per SDRTrunk:
        - Bits 16-23: LRA (8 bits)
        - Bit 27: Active Network Connection Flag
        - Bits 28-39: System ID (12 bits)
        - Bits 40-47: RFSS ID (8 bits)
        - Bits 48-55: Site ID (8 bits)
        - Bits 56-59: Frequency Band (4 bits)
        - Bits 60-71: Channel Number (12 bits)
        - Bits 72-79: System Service Class (8 bits)
        """
        result['type'] = 'RFSS_STATUS'

        # LRA (byte 0 = bits 16-23)
        lra = data[0]

        # System ID (12 bits spanning bytes 1-2):
        # Upper 8 bits from low 4 bits of byte 1 + upper 4 bits of byte 2
        # Bits 28-39 = bits 4-7 of byte 1 (after bit 27) + bits 0-7 of byte 2
        sys_id = ((data[1] & 0x0F) << 8) | data[2]

        # Active network connection flag (bit 27 = bit 3 of byte 1)
        active_network = bool(data[1] & 0x08)

        # RFSS ID (byte 3 upper 4 bits would be wrong - let me recalculate)
        # Bits 40-47 = all of byte 3 (after system ID which ends at bit 39)
        rfss_id = data[3]

        # Site ID (byte 4 = bits 48-55)
        site_id = data[4]

        # Frequency band (upper 4 bits of byte 5 = bits 56-59)
        freq_band = (data[5] >> 4) & 0x0F

        # Channel number (lower 4 bits of byte 5 + byte 6 = bits 60-71)
        channel_num = ((data[5] & 0x0F) << 8) | data[6]
        channel = (freq_band << 12) | channel_num

        # Service class (byte 7 = bits 72-79)
        svc_class = data[7]

        result['lra'] = lra
        result['active_network_connection'] = active_network
        result['system_id'] = sys_id
        result['rfss_id'] = rfss_id
        result['site_id'] = site_id
        result['channel'] = channel
        result['frequency_band'] = freq_band
        result['channel_number'] = channel_num
        result['service_class'] = svc_class

        # Update system status
        self.system_status.system_id = sys_id
        self.system_status.rfss_id = rfss_id
        self.system_status.site_id = site_id
        self.system_status.channel = channel

        logger.info(f"RFSS Status: SysID={sys_id:03X} RFSS={rfss_id} Site={site_id} "
                   f"BAND={freq_band} CH={channel_num}")

        if self.on_system_update:
            self.on_system_update(self.system_status)

    def _parse_net_sts_bcast(self, data: bytes, result: dict[str, Any]) -> None:
        """Parse Network Status Broadcast.

        Data format per SDRTrunk:
        - Bits 16-23: LRA (8 bits)
        - Bits 24-43: WACN (20 bits)
        - Bits 44-55: System ID (12 bits)
        - Bits 56-59: Frequency Band (4 bits)
        - Bits 60-71: Channel Number (12 bits)
        - Bits 72-79: System Service Class (8 bits)
        """
        result['type'] = 'NETWORK_STATUS'

        # LRA (byte 0 = bits 16-23)
        lra = data[0]

        # WACN (20 bits spanning bytes 1-3):
        # Byte 1 (8 bits) + byte 2 (8 bits) + upper 4 bits of byte 3
        wacn = (data[1] << 12) | (data[2] << 4) | ((data[3] >> 4) & 0x0F)

        # System ID (12 bits spanning bytes 3-4):
        # Lower 4 bits of byte 3 + byte 4
        sys_id = ((data[3] & 0x0F) << 8) | data[4]

        # Frequency band (upper 4 bits of byte 5)
        freq_band = (data[5] >> 4) & 0x0F

        # Channel number (lower 4 bits of byte 5 + byte 6)
        channel_num = ((data[5] & 0x0F) << 8) | data[6]
        channel = (freq_band << 12) | channel_num

        # System service class (byte 7)
        svc_class = data[7]

        result['lra'] = lra
        result['wacn'] = wacn  # Wide Area Communication Network
        result['system_id'] = sys_id
        result['channel'] = channel
        result['frequency_band'] = freq_band
        result['channel_number'] = channel_num
        result['service_class'] = svc_class

        self.system_status.system_id = sys_id
        self.system_status.services = svc_class

    def _parse_adj_sts_bcast(self, data: bytes, result: dict[str, Any]) -> None:
        """Parse Adjacent Status Broadcast.

        Information about neighboring sites for roaming.

        Data format (same structure as RFSS Status):
        - Bits 16-23: LRA (8 bits)
        - Bit 27: Active Network Connection Flag
        - Bits 28-39: System ID (12 bits)
        - Bits 40-47: RFSS ID (8 bits)
        - Bits 48-55: Site ID (8 bits)
        - Bits 56-59: Frequency Band (4 bits)
        - Bits 60-71: Channel Number (12 bits)
        - Bits 72-79: System Service Class (8 bits)
        """
        result['type'] = 'ADJACENT_STATUS'

        # LRA (byte 0)
        lra = data[0]

        # System ID (12 bits spanning bytes 1-2)
        sys_id = ((data[1] & 0x0F) << 8) | data[2]

        # Active network connection flag (bit 27 = bit 3 of byte 1)
        active_network = bool(data[1] & 0x08)

        # RFSS ID (byte 3)
        rfss_id = data[3]

        # Site ID (byte 4)
        site_id = data[4]

        # Frequency band (upper 4 bits of byte 5)
        freq_band = (data[5] >> 4) & 0x0F

        # Channel number (lower 4 bits of byte 5 + byte 6)
        channel_num = ((data[5] & 0x0F) << 8) | data[6]
        channel = (freq_band << 12) | channel_num

        # Service class (byte 7)
        svc_class = data[7]

        result['lra'] = lra
        result['active_network_connection'] = active_network
        result['system_id'] = sys_id
        result['rfss_id'] = rfss_id
        result['site_id'] = site_id
        result['channel'] = channel
        result['frequency_band'] = freq_band
        result['channel_number'] = channel_num
        result['service_class'] = svc_class

    def _parse_iden_up_vu(self, data: bytes, result: dict[str, Any]) -> None:
        """Parse Identifier Update for VHF/UHF.

        Defines channel numbering for a frequency band.

        Data format per SDRTrunk:
        - Bits 16-19: Identifier (4 bits)
        - Bits 20-23: Bandwidth (4 bits) - NOTE: SDRTrunk uses 4 bits, not 9!
        - Bit 24: TX Offset Sign (1 bit) - 1=positive, 0=negative
        - Bits 25-37: TX Offset (13 bits)
        - Bits 38-47: Channel Spacing (10 bits)
        - Bits 48-79: Base Frequency (32 bits)
        """
        result['type'] = 'IDENTIFIER_UPDATE_VU'

        # Identifier (upper 4 bits of byte 0 = bits 16-19)
        ident = (data[0] >> 4) & 0x0F

        # Bandwidth (lower 4 bits of byte 0 = bits 20-23)
        bw = data[0] & 0x0F

        # TX offset sign (bit 24 = bit 0 of byte 1)
        tx_offset_sign = bool(data[1] & 0x80)

        # TX offset (13 bits spanning bytes 1-2):
        # Lower 7 bits of byte 1 + upper 6 bits of byte 2
        tx_offset = ((data[1] & 0x7F) << 6) | ((data[2] >> 2) & 0x3F)

        # Channel spacing (10 bits spanning bytes 2-3):
        # Lower 2 bits of byte 2 + all of byte 3
        spacing = ((data[2] & 0x03) << 8) | data[3]

        # Base frequency (32 bits = bytes 4-7)
        base_freq_raw = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7]
        # Convert to MHz: multiply by 5 Hz per SDRTrunk
        base_freq = base_freq_raw * 0.000005

        # Apply sign to TX offset (multiply by channel spacing per SDRTrunk)
        tx_offset_hz = tx_offset * (spacing * 125)
        if not tx_offset_sign:
            tx_offset_hz *= -1

        # Convert bandwidth code to kHz (per SDRTrunk)
        if bw == 0x4:
            bw_khz = 6.25
        elif bw == 0x5:
            bw_khz = 12.5
        else:
            bw_khz = 0

        if spacing == 0 or base_freq <= 0:
            raise ValueError(
                f"IDEN_UP_VU invalid values: ident={ident} base={base_freq} "
                f"spacing={spacing} bw_khz={bw_khz}"
            )

        result['identifier'] = ident
        result['bandwidth_khz'] = bw_khz
        result['bandwidth_code'] = bw
        result['tx_offset_sign'] = tx_offset_sign
        result['tx_offset_hz'] = tx_offset_hz
        result['tx_offset_mhz'] = tx_offset_hz / 1e6
        result['channel_spacing_khz'] = spacing * 0.125
        result['base_freq_mhz'] = base_freq

        # Store channel identifier
        chan_id = ChannelIdentifier(
            identifier=ident,
            bw=int(bw_khz),
            tx_offset=int(tx_offset_hz / 1e6),
            channel_spacing=int(spacing * 0.125),
            base_freq=base_freq
        )
        self.add_channel_id(chan_id)

        logger.info(f"Channel ID {ident}: base={base_freq:.4f} MHz, "
                   f"spacing={spacing * 0.125} kHz, bw={bw_khz} kHz, "
                   f"tx_offset={tx_offset_hz/1e6:.4f} MHz")

    def _parse_iden_up_tdma(self, data: bytes, result: dict[str, Any]) -> None:
        """Parse Identifier Update for TDMA (Phase II).

        Data format per SDRTrunk:
        - Bits 16-19: Identifier (4 bits)
        - Bits 20-23: Channel Type (4 bits) - encodes access type and slot count
        - Bit 24: TX Offset Sign (1 bit) - 1=positive, 0=negative
        - Bits 25-37: TX Offset (13 bits)
        - Bits 38-47: Channel Spacing (10 bits)
        - Bits 48-79: Base Frequency (32 bits)
        """
        result['type'] = 'IDENTIFIER_UPDATE_TDMA'

        # Identifier (upper 4 bits of byte 0 = bits 16-19)
        ident = (data[0] >> 4) & 0x0F

        # Channel Type (lower 4 bits of byte 0 = bits 20-23)
        # Per SDRTrunk ChannelType enum: encodes both access type and slot count
        channel_type = data[0] & 0x0F

        # TX offset sign (bit 24 = MSB of byte 1)
        tx_offset_sign = bool(data[1] & 0x80)

        # TX offset (13 bits spanning bytes 1-2):
        # Lower 7 bits of byte 1 + upper 6 bits of byte 2
        tx_offset = ((data[1] & 0x7F) << 6) | ((data[2] >> 2) & 0x3F)

        # Channel spacing (10 bits spanning bytes 2-3):
        # Lower 2 bits of byte 2 + all of byte 3
        spacing = ((data[2] & 0x03) << 8) | data[3]

        # Base frequency (32 bits = bytes 4-7)
        base_freq_raw = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7]
        base_freq = base_freq_raw * 0.000005

        # Apply sign to TX offset (multiply by channel spacing per SDRTrunk)
        tx_offset_hz = tx_offset * (spacing * 125)
        if not tx_offset_sign:
            tx_offset_hz *= -1

        if spacing == 0 or base_freq <= 0:
            raise ValueError(
                f"IDEN_UP_TDMA invalid values: ident={ident} base={base_freq} spacing={spacing}"
            )

        # Decode channel type (per SDRTrunk ChannelType.java)
        # Common values: 0x0=FDMA, 0x2=TDMA_2SLOT, 0x3=TDMA_4SLOT, 0x6=TDMA_6SLOT
        if channel_type == 0x02:
            access_type = 'TDMA'
            slot_count = 2
        elif channel_type == 0x03:
            access_type = 'TDMA'
            slot_count = 4
        elif channel_type == 0x06:
            access_type = 'TDMA'
            slot_count = 6
        else:
            access_type = 'FDMA'
            slot_count = 1

        result['identifier'] = ident
        result['channel_type'] = channel_type
        result['access_type'] = access_type
        result['slot_count'] = slot_count
        result['tx_offset_sign'] = tx_offset_sign
        result['tx_offset_hz'] = tx_offset_hz
        result['tx_offset_mhz'] = tx_offset_hz / 1e6
        result['channel_spacing_khz'] = spacing * 0.125
        result['base_freq_mhz'] = base_freq

        # Store channel identifier
        chan_id = ChannelIdentifier(
            identifier=ident,
            bw=0,  # Not specified in TDMA IDEN
            tx_offset=int(tx_offset_hz / 1e6),
            channel_spacing=int(spacing * 0.125),
            base_freq=base_freq
        )
        self.add_channel_id(chan_id)

        logger.info(f"Channel ID {ident} ({access_type} {slot_count}-slot): base={base_freq:.4f} MHz, "
                   f"spacing={spacing * 0.125} kHz, tx_offset={tx_offset_hz/1e6:.4f} MHz")

    def _parse_sys_srv_bcast(self, data: bytes, result: dict[str, Any]) -> None:
        """Parse System Service Broadcast.

        Available services bitmap.
        """
        result['type'] = 'SYSTEM_SERVICE'

        svc_available = (data[0] << 16) | (data[1] << 8) | data[2]
        svc_supported = (data[3] << 16) | (data[4] << 8) | data[5]

        result['services_available'] = svc_available
        result['services_supported'] = svc_supported

        # Decode service flags
        result['composite_control'] = bool(svc_available & 0x800000)
        result['data_services'] = bool(svc_available & 0x400000)
        result['voice_services'] = bool(svc_available & 0x200000)
        result['registration'] = bool(svc_available & 0x100000)
        result['authentication'] = bool(svc_available & 0x080000)

        self.system_status.services = svc_available

    def _parse_grp_aff_rsp(self, data: bytes, result: dict[str, Any]) -> None:
        """Parse Group Affiliation Response."""
        result['type'] = 'GROUP_AFFILIATION_RESPONSE'

        # Response code
        result['response'] = data[0]
        result['success'] = (data[0] & 0x03) == 0

        tgid = (data[1] << 8) | data[2]
        announcement_group = (data[3] << 8) | data[4]
        target = (data[5] << 8) | data[6]

        result['tgid'] = tgid
        result['announcement_group'] = announcement_group
        result['target_id'] = target

    def _parse_deny_rsp(self, data: bytes, result: dict[str, Any]) -> None:
        """Parse Deny Response."""
        result['type'] = 'DENY_RESPONSE'

        # Deny reason
        result['service_type'] = (data[0] >> 2) & 0x3F
        result['reason'] = data[1]

        # Reason codes
        reason_names = {
            0x00: 'No reason',
            0x10: 'Unit not valid',
            0x11: 'Unit not authorized',
            0x12: 'Target not valid',
            0x20: 'Group not valid',
            0x21: 'Group not authorized',
            0x2F: 'Unit busy',
            0x30: 'Target unit busy',
            0x40: 'Site access denied',
        }
        result['reason_text'] = reason_names.get(data[1], f'Unknown (0x{data[1]:02X})')

        target = (data[2] << 16) | (data[3] << 8) | data[4]
        result['target_address'] = target

        logger.info(f"Deny: {result['reason_text']} for target {target}")


# ============================================================================
# Encoder helpers
# ============================================================================

def _bytes_to_bits_be(data: Sequence[int]) -> list[int]:
    """Convert bytes to MSB-first bit list."""
    bits: list[int] = []
    for byte in data:
        for shift in range(7, -1, -1):
            bits.append((int(byte) >> shift) & 0x1)
    return bits


def _compute_crc16_ccitt(bits: Sequence[int]) -> int:
    """Compute CCITT-16 CRC across the first 80 bits of a TSBK frame."""
    calculated = 0xFFFF
    for i in range(min(80, len(bits))):
        if bits[i]:
            calculated ^= CCITT_80_CHECKSUMS[i]
    return calculated & 0xFFFF


def encode_control_frame(message: TSBKMessage, lb: int = 1, protect: int = 0) -> bytes:
    """Encode a full 96-bit control frame (header + MFID + payload + CRC).

    Args:
        message: Typed message to encode. The payload is taken from payload_bytes().
        lb: Last block bit (0 or 1).
        protect: Protect flag bit (0 or 1).

    Returns:
        12-byte control frame suitable for dibit mapping (MSB-first).
    """
    payload = message.payload_bytes()
    if len(payload) != 8:
        raise ValueError(f"TSBK payload must be 8 bytes, got {len(payload)}")

    header = ((lb & 0x1) << 7) | ((protect & 0x1) << 6) | (message.opcode & 0x3F)
    frame = bytearray()
    frame.append(header)
    frame.append(message.mfid & 0xFF)
    frame.extend(payload)

    bits = _bytes_to_bits_be(frame)
    crc = _compute_crc16_ccitt(bits)
    frame.extend(crc.to_bytes(2, "big"))
    return bytes(frame)


def encode_voice_assignment(grant: GroupVoiceGrantMessage) -> bytes:
    """Encode a group voice grant payload (8 bytes)."""
    return grant.payload_bytes()


def encode_voice_assignment_update(update: GroupVoiceGrantUpdateMessage | GroupVoiceGrantUpdateExplicitMessage) -> bytes:
    """Encode a group voice grant update payload (8 bytes)."""
    return update.payload_bytes()


def encode_unit_assignment_update(update: UnitToUnitGrantUpdateMessage) -> bytes:
    """Encode a unit-to-unit voice grant update payload (8 bytes)."""
    return update.payload_bytes()


def encode_affiliation_response(message: GroupAffiliationResponseMessage) -> bytes:
    """Encode a group affiliation response payload (8 bytes)."""
    return message.payload_bytes()
