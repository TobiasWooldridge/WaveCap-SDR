"""P25 TSBK (Trunking Signaling Block) Message Parser.

This module parses TSBK messages from P25 control channels. TSBK messages
control trunking operations including:
- Voice channel grants
- Channel updates
- System information
- Registration and authentication

TSBK Opcodes are defined in TIA-102.AABB-A.

Key opcodes for trunking:
- 0x00: Group Voice Channel Grant
- 0x02: Group Voice Channel Grant Update
- 0x03: Unit to Unit Voice Channel Grant
- 0x28: RFSS Status Broadcast
- 0x34: Identifier Update (channel info)
- 0x3A: Adjacent Status Broadcast
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable

logger = logging.getLogger(__name__)

from wavecapsdr.decoders.tsbk_utils import payload_to_bits, read_field
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
    UNIT_ID_MIN,
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
    slot_id: int | None = None


@dataclass
class SystemStatus:
    """System status information from control channel."""
    nac: int = 0  # Network Access Code
    system_id: int = 0  # System ID
    rfss_id: int = 0  # RF Subsystem ID
    site_id: int = 0  # Site ID
    channel: int = 0  # Control channel number
    services: int = 0  # Available services bitmap


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

    def parse(self, opcode: int, mfid: int, data: bytes) -> dict[str, Any]:
        """Parse TSBK message.

        Args:
            opcode: 6-bit opcode
            mfid: Manufacturer ID (0 = standard)
            data: 8-byte data payload (64 bits per TIA-102.AABB-A)

        Returns:
            Dict with parsed message fields
        """
        if len(data) != 8:
            raise ValueError(f"TSBK data must be 8 bytes (got {len(data)})")

        result: dict[str, Any] = {
            'opcode': opcode,
            'opcode_name': self._opcode_name(opcode),
            'mfid': mfid,
        }

        if mfid != 0:
            # Non-standard (manufacturer-specific) message
            result['type'] = 'MANUFACTURER_SPECIFIC'
            result['data'] = data.hex()
            return result

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
            elif opcode == TSBKOpcode.UU_ANS_REQ:
                self._parse_unit_service_request(data, result)
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
            elif opcode == TSBKOpcode.STATUS_UPDT:
                self._parse_status_update(data, result, msg_type="STATUS_UPDATE")
            elif opcode == TSBKOpcode.STATUS_QUERY:
                self._parse_status_update(data, result, msg_type="STATUS_UPDATE_REQUEST")
            elif opcode == TSBKOpcode.UNIT_REG_RSP:
                self._parse_unit_reg_rsp(data, result)
            elif opcode == TSBKOpcode.UNIT_DEREG_ACK:
                self._parse_unit_dereg_ack(data, result)
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
        except Exception as e:
            logger.warning(f"Error parsing TSBK opcode {opcode:02X}: {e}")
            result['type'] = 'PARSE_ERROR'
            result['error'] = str(e)
            result['data'] = data.hex()

        return result

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

        if msg_type == "GROUP_AFFILIATION_RESPONSE":
            self._require_int_range(result.get("tgid"), 1, TGID_MAX, "tgid")
            self._require_int_range(result.get("announcement_group"), 1, TGID_MAX, "announcement_group")
            self._require_int_range(result.get("target_id"), UNIT_ID_MIN, UNIT_ID_MAX, "target_id")
            return

        if msg_type in ("STATUS_UPDATE", "STATUS_UPDATE_REQUEST"):
            self._require_int_range(result.get("unit_status"), 0, 0xFF, "unit_status")
            self._require_int_range(result.get("user_status"), 0, 0xFF, "user_status")
            self._require_int_range(result.get("target_id"), UNIT_ID_MIN, UNIT_ID_MAX, "target_id")
            self._require_int_range(result.get("source_id"), UNIT_ID_MIN, UNIT_ID_MAX, "source_id")
            return

        if msg_type == "UNIT_REGISTRATION_RESPONSE":
            self._require_int_range(result.get("response"), 0, 3, "response")
            self._require_int_range(result.get("system_id"), 0, SYSTEM_ID_MAX, "system_id")
            self._require_int_range(result.get("source_id"), UNIT_ID_MIN, UNIT_ID_MAX, "source_id")
            self._require_int_range(result.get("registered_address"), UNIT_ID_MIN, UNIT_ID_MAX, "registered_address")
            return

        if msg_type == "UNIT_DEREGISTRATION_ACK":
            self._require_int_range(result.get("wacn"), 0, WACN_MAX, "wacn")
            self._require_int_range(result.get("system_id"), 0, SYSTEM_ID_MAX, "system_id")
            self._require_int_range(result.get("target_id"), UNIT_ID_MIN, UNIT_ID_MAX, "target_id")
            return

        if msg_type == "UNIT_SERVICE_REQUEST":
            self._require_int_range(result.get("service_options"), 0, 0xFF, "service_options")
            self._require_int_range(result.get("target_id"), UNIT_ID_MIN, UNIT_ID_MAX, "target_id")
            self._require_int_range(result.get("source_id"), UNIT_ID_MIN, UNIT_ID_MAX, "source_id")
            return

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
        slot_id = (svc_opts >> 3) & 0x01
        result['slot_id'] = slot_id

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
                encrypted=result['encrypted'],
                slot_id=slot_id,
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
        result['slot_id'] = (svc_opts >> 3) & 0x01

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
        bits = payload_to_bits(data)

        result['global'] = bool(read_field(bits, 0, 1))
        response_code = read_field(bits, 6, 2)
        result['response'] = response_code
        result['success'] = response_code == 0

        announcement_group = read_field(bits, 8, 16)
        tgid = read_field(bits, 24, 16)
        target = read_field(bits, 40, 24)

        result['tgid'] = tgid
        result['announcement_group'] = announcement_group
        result['target_id'] = target
        result['source_id'] = target

    def _parse_status_update(self, data: bytes, result: dict[str, Any], msg_type: str) -> None:
        """Parse Status Update (OSP) or Status Update Request (ISP)."""
        bits = payload_to_bits(data)
        result['type'] = msg_type
        result['unit_status'] = read_field(bits, 0, 8)
        result['user_status'] = read_field(bits, 8, 8)
        result['target_id'] = read_field(bits, 16, 24)
        result['source_id'] = read_field(bits, 40, 24)

    def _parse_unit_reg_rsp(self, data: bytes, result: dict[str, Any]) -> None:
        """Parse Unit Registration Response."""
        bits = payload_to_bits(data)
        response = read_field(bits, 2, 2)
        system_id = read_field(bits, 4, 12)
        source_id = read_field(bits, 16, 24)
        registered_address = read_field(bits, 40, 24)

        result['type'] = 'UNIT_REGISTRATION_RESPONSE'
        result['response'] = response
        result['success'] = response == 0
        result['system_id'] = system_id
        result['source_id'] = source_id
        result['registered_address'] = registered_address

    def _parse_unit_dereg_ack(self, data: bytes, result: dict[str, Any]) -> None:
        """Parse Unit De-registration Acknowledge."""
        bits = payload_to_bits(data)
        wacn = read_field(bits, 8, 20)
        system_id = read_field(bits, 28, 12)
        target_id = read_field(bits, 40, 24)

        result['type'] = 'UNIT_DEREGISTRATION_ACK'
        result['wacn'] = wacn
        result['system_id'] = system_id
        result['target_id'] = target_id

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

    def _parse_unit_service_request(self, data: bytes, result: dict[str, Any]) -> None:
        """Parse Unit-to-Unit Voice Service Request (opcode 0x05)."""
        bits = payload_to_bits(data)
        result['type'] = 'UNIT_SERVICE_REQUEST'
        result['service_options'] = read_field(bits, 0, 8)
        result['target_id'] = read_field(bits, 16, 24)
        result['source_id'] = read_field(bits, 40, 24)
