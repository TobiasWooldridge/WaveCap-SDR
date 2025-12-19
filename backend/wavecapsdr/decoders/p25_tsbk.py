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
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


class TSBKOpcode(IntEnum):
    """TSBK Opcode values (6 bits)."""
    # Voice grants
    GRP_V_CH_GRANT = 0x00  # Group Voice Channel Grant
    GRP_V_CH_GRANT_UPDT = 0x02  # Group Voice Channel Grant Update
    UU_V_CH_GRANT = 0x03  # Unit to Unit Voice Channel Grant
    UU_ANS_REQ = 0x05  # Unit to Unit Answer Request
    UU_V_CH_GRANT_UPDT = 0x06  # Unit to Unit Voice Channel Grant Update

    # Data grants
    GRP_DATA_CH_GRANT = 0x25  # Group Data Channel Grant
    IND_DATA_CH_GRANT = 0x26  # Individual Data Channel Grant

    # Call control
    TEL_INT_CH_GRANT = 0x08  # Telephone Interconnect Voice Channel Grant
    TEL_INT_ANS_REQ = 0x09  # Telephone Interconnect Answer Request

    # System information
    NET_STS_BCAST = 0x20  # Network Status Broadcast
    RFSS_STS_BCAST = 0x28  # RFSS Status Broadcast
    ADJ_STS_BCAST = 0x3A  # Adjacent Status Broadcast
    SYS_SRV_BCAST = 0x2A  # System Service Broadcast

    # Channel identification
    IDEN_UP_VU = 0x34  # Identifier Update for VHF/UHF
    IDEN_UP_TDMA = 0x33  # Identifier Update for TDMA

    # Control
    ACK_RSP_U = 0x21  # Acknowledge Response - Unit
    QUE_RSP = 0x22  # Queued Response
    DENY_RSP = 0x27  # Deny Response
    GRP_AFF_RSP = 0x29  # Group Affiliation Response
    LOC_REG_RSP = 0x2B  # Location Registration Response

    # Authentication
    AUTH_CMD = 0x2C  # Authentication Command
    AUTH_DMAN = 0x2D  # Authentication Demand
    AUTH_FNE_RST = 0x2E  # Authentication FNE Result

    # SNDCP (data)
    SNDCP_CH_GNT = 0x31  # SNDCP Data Channel Grant
    SNDCP_CH_ANN_UPDT = 0x32  # SNDCP Data Channel Announcement Update

    # Miscellaneous
    CALL_ALRT = 0x1F  # Call Alert
    EXT_FNCT_CMD = 0x24  # Extended Function Command
    ROAM_ADDR_CMD = 0x2F  # Roaming Address Command


@dataclass
class ChannelIdentifier:
    """Channel identifier from IDEN_UP messages.

    Used to calculate actual frequencies from channel numbers.
    """
    identifier: int  # 4-bit channel ID
    bw: int  # Bandwidth (kHz)
    tx_offset: int  # TX offset (MHz)
    channel_spacing: int  # Channel spacing (kHz)
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


class TSBKParser:
    """Parser for TSBK messages.

    Maintains channel identifier table and parses TSBK opcodes
    into structured data for the trunking controller.
    """

    def __init__(self) -> None:
        # Channel identifier table (from IDEN_UP messages)
        self._channel_ids: Dict[int, ChannelIdentifier] = {}

        # System status
        self.system_status = SystemStatus()

        # Callbacks
        self.on_voice_grant: Optional[Callable[[VoiceGrant], None]] = None
        self.on_system_update: Optional[Callable[[SystemStatus], None]] = None

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

    def parse(self, opcode: int, mfid: int, data: bytes) -> Dict[str, Any]:
        """Parse TSBK message.

        Args:
            opcode: 6-bit opcode
            mfid: Manufacturer ID (0 = standard)
            data: 8-byte data payload (64 bits per TIA-102.AABB-A)

        Returns:
            Dict with parsed message fields
        """
        result: Dict[str, Any] = {
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
            else:
                result['type'] = 'UNKNOWN'
                result['data'] = data.hex()
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

    def _parse_grp_v_ch_grant(self, data: bytes, result: Dict[str, Any]) -> None:
        """Parse Group Voice Channel Grant.

        Data format (64 bits per TIA-102.AABB-A):
        - Service Options (8 bits)
        - Channel (16 bits: 4-bit IDEN + 12-bit channel number)
        - Group Address (16 bits)
        - Source Address (24 bits)
        """
        result['type'] = 'GROUP_VOICE_GRANT'

        svc_opts = data[0]
        channel = (data[1] << 8) | data[2]
        tgid = (data[3] << 8) | data[4]
        # Source is 24 bits (3 bytes) per P25 spec
        source = (data[5] << 16) | (data[6] << 8) | data[7]

        result['emergency'] = bool(svc_opts & 0x80)
        result['encrypted'] = bool(svc_opts & 0x40)
        result['duplex'] = bool(svc_opts & 0x20)
        result['priority'] = (svc_opts >> 0) & 0x07

        result['channel'] = channel
        result['tgid'] = tgid
        result['source_id'] = source

        # Calculate frequency
        freq = self.get_frequency(channel)
        result['frequency_hz'] = freq
        result['frequency_mhz'] = freq / 1e6 if freq else 0

        logger.info(f"Voice Grant: TGID={tgid} CH={channel} "
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

    def _parse_grp_v_ch_grant_updt(self, data: bytes, result: Dict[str, Any]) -> None:
        """Parse Group Voice Channel Grant Update.

        Contains two channel/group pairs for active calls.
        """
        result['type'] = 'GROUP_VOICE_GRANT_UPDATE'

        # First grant
        channel1 = (data[0] << 8) | data[1]
        tgid1 = (data[2] << 8) | data[3]

        result['grant1'] = {
            'channel': channel1,
            'tgid': tgid1,
            'frequency_hz': self.get_frequency(channel1)
        }

        # Second grant (if present)
        if len(data) >= 7:
            channel2 = (data[4] << 8) | data[5]
            tgid2 = (data[6] << 8) if len(data) > 6 else 0

            if channel2 != 0:
                result['grant2'] = {
                    'channel': channel2,
                    'tgid': tgid2,
                    'frequency_hz': self.get_frequency(channel2)
                }

    def _parse_uu_v_ch_grant(self, data: bytes, result: Dict[str, Any]) -> None:
        """Parse Unit to Unit Voice Channel Grant.

        Data format (64 bits per TIA-102.AABB-A):
        - Channel (16 bits: 4-bit IDEN + 12-bit channel number)
        - Target Address (24 bits)
        - Source Address (24 bits)

        Note: This opcode has NO service options field - channel starts at bit 0.
        """
        result['type'] = 'UNIT_TO_UNIT_GRANT'

        # Channel is first (no service options in this opcode)
        channel = (data[0] << 8) | data[1]
        # Target is 24 bits (bytes 2-4)
        target = (data[2] << 16) | (data[3] << 8) | data[4]
        # Source is 24 bits (bytes 5-7)
        source = (data[5] << 16) | (data[6] << 8) | data[7]

        result['channel'] = channel
        result['target_id'] = target
        result['source_id'] = source
        result['frequency_hz'] = self.get_frequency(channel)
        result['emergency'] = False  # No service options in this opcode
        result['encrypted'] = False

    def _parse_rfss_sts_bcast(self, data: bytes, result: Dict[str, Any]) -> None:
        """Parse RFSS Status Broadcast.

        Contains RFSS and site identification.
        """
        result['type'] = 'RFSS_STATUS'

        lra = data[0]  # Location Registration Area
        sys_id = (data[1] << 4) | (data[2] >> 4)
        rfss_id = data[2] & 0x0F
        site_id = data[3]
        channel = (data[4] << 8) | data[5]
        svc_class = data[6]

        result['lra'] = lra
        result['system_id'] = sys_id
        result['rfss_id'] = rfss_id
        result['site_id'] = site_id
        result['channel'] = channel
        result['service_class'] = svc_class

        # Update system status
        self.system_status.system_id = sys_id
        self.system_status.rfss_id = rfss_id
        self.system_status.site_id = site_id
        self.system_status.channel = channel

        logger.info(f"RFSS Status: SysID={sys_id:03X} RFSS={rfss_id} Site={site_id}")

        if self.on_system_update:
            self.on_system_update(self.system_status)

    def _parse_net_sts_bcast(self, data: bytes, result: Dict[str, Any]) -> None:
        """Parse Network Status Broadcast."""
        result['type'] = 'NETWORK_STATUS'

        lra = data[0]
        wacn = (data[1] << 12) | (data[2] << 4) | (data[3] >> 4)
        sys_id = ((data[3] & 0x0F) << 8) | data[4]
        channel = (data[5] << 8) | data[6]

        result['lra'] = lra
        result['wacn'] = wacn  # Wide Area Communication Network
        result['system_id'] = sys_id
        result['channel'] = channel

        self.system_status.system_id = sys_id

    def _parse_adj_sts_bcast(self, data: bytes, result: Dict[str, Any]) -> None:
        """Parse Adjacent Status Broadcast.

        Information about neighboring sites for roaming.
        """
        result['type'] = 'ADJACENT_STATUS'

        lra = data[0]
        sys_id = (data[1] << 4) | (data[2] >> 4)
        rfss_id = data[2] & 0x0F
        site_id = data[3]
        channel = (data[4] << 8) | data[5]
        svc_class = data[6]

        result['lra'] = lra
        result['system_id'] = sys_id
        result['rfss_id'] = rfss_id
        result['site_id'] = site_id
        result['channel'] = channel
        result['service_class'] = svc_class

    def _parse_iden_up_vu(self, data: bytes, result: Dict[str, Any]) -> None:
        """Parse Identifier Update for VHF/UHF.

        Defines channel numbering for a frequency band.

        Data format (64 bits per TIA-102.AABB-A):
        - Identifier (4 bits)
        - Bandwidth (9 bits)
        - TX Offset Sign (1 bit) - 1=positive, 0=negative
        - TX Offset (8 bits)
        - Channel Spacing (10 bits)
        - Base Frequency (32 bits)
        """
        result['type'] = 'IDENTIFIER_UPDATE_VU'

        ident = (data[0] >> 4) & 0x0F
        bw = ((data[0] & 0x0F) << 5) | (data[1] >> 3)
        # TX offset sign is bit 2 of data[1] (1=positive, 0=negative per SDRTrunk)
        tx_offset_sign = bool((data[1] >> 2) & 1)
        # TX offset is 8 bits (2 bits from data[1] + 6 bits from data[2])
        tx_offset = ((data[1] & 0x03) << 6) | (data[2] >> 2)
        spacing = ((data[2] & 0x03) << 8) | data[3]
        # Base frequency is 32 bits (was incorrectly 24 bits before)
        base_freq = ((data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7]) * 0.000005

        # Apply sign to TX offset
        tx_offset_val = tx_offset if tx_offset_sign else -tx_offset

        result['identifier'] = ident
        result['bandwidth_khz'] = bw * 0.125
        result['tx_offset_sign'] = tx_offset_sign
        result['tx_offset_mhz'] = tx_offset_val * 0.25
        result['channel_spacing_khz'] = spacing * 0.125
        result['base_freq_mhz'] = base_freq

        # Store channel identifier
        chan_id = ChannelIdentifier(
            identifier=ident,
            bw=int(bw * 0.125),
            tx_offset=int(tx_offset_val * 0.25),
            channel_spacing=int(spacing * 0.125),
            base_freq=base_freq
        )
        self.add_channel_id(chan_id)

        logger.info(f"Channel ID {ident}: base={base_freq:.4f} MHz, "
                   f"spacing={spacing * 0.125} kHz, tx_offset={tx_offset_val * 0.25} MHz")

    def _parse_iden_up_tdma(self, data: bytes, result: Dict[str, Any]) -> None:
        """Parse Identifier Update for TDMA (Phase II).

        Data format (64 bits per TIA-102.AABB-A):
        - Identifier (4 bits)
        - Channel Type (4 bits): access type + slot count
        - TX Offset Sign (1 bit) - 1=positive, 0=negative
        - TX Offset (13 bits)
        - Channel Spacing (10 bits)
        - Base Frequency (32 bits)
        """
        result['type'] = 'IDENTIFIER_UPDATE_TDMA'

        ident = (data[0] >> 4) & 0x0F
        # TDMA-specific fields: Channel Type (4 bits)
        access_type = (data[0] >> 2) & 0x03  # 0=FDMA, 1=TDMA, etc.
        slot_count = (data[0] & 0x03) + 1

        # TX offset sign is bit 0 of data[1] (bit 8 of data field)
        # Per SDRTrunk FrequencyBandUpdateTDMA: TRANSMIT_OFFSET_SIGN = 24 (bit 8 of data)
        tx_offset_sign = bool((data[1] >> 7) & 1)
        # TX offset is 13 bits (bits 25-37)
        tx_offset = ((data[1] & 0x7F) << 6) | (data[2] >> 2)
        # Channel spacing is 10 bits
        spacing = ((data[2] & 0x03) << 8) | data[3]
        # Base frequency is 32 bits
        base_freq = ((data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7]) * 0.000005

        # Apply sign to TX offset
        tx_offset_val = tx_offset if tx_offset_sign else -tx_offset

        result['identifier'] = ident
        result['access_type'] = 'TDMA' if access_type == 1 else 'FDMA'
        result['slot_count'] = slot_count
        result['tx_offset_sign'] = tx_offset_sign
        result['tx_offset_mhz'] = tx_offset_val * 0.25
        result['channel_spacing_khz'] = spacing * 0.125
        result['base_freq_mhz'] = base_freq

        # Store channel identifier
        chan_id = ChannelIdentifier(
            identifier=ident,
            bw=0,  # Not specified in TDMA IDEN
            tx_offset=int(tx_offset_val * 0.25),
            channel_spacing=int(spacing * 0.125),
            base_freq=base_freq
        )
        self.add_channel_id(chan_id)

        logger.info(f"Channel ID {ident} (TDMA): base={base_freq:.4f} MHz, "
                   f"spacing={spacing * 0.125} kHz, tx_offset={tx_offset_val * 0.25} MHz")

    def _parse_sys_srv_bcast(self, data: bytes, result: Dict[str, Any]) -> None:
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

    def _parse_grp_aff_rsp(self, data: bytes, result: Dict[str, Any]) -> None:
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

    def _parse_deny_rsp(self, data: bytes, result: Dict[str, Any]) -> None:
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
