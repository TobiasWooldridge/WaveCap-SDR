"""Encoding helpers for trunked traffic/voice PDUs and burst payloads.

These helpers mirror the decoder expectations used by :mod:`wavecapsdr.decoders.p25_tsbk`
for voice channel grants and provide a simple header/payload wrapper for voice bursts.
They are intentionally typed and deterministic so tests can round-trip structures
through existing decoders.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from wavecapsdr.decoders.p25_tsbk import TSBKOpcode, TSBKParser


class VoicePayloadType(Enum):
    """Supported codec payload identifiers."""

    AMBE = 0
    IMBE = 1
    PCM = 2


@dataclass
class TrafficChannelGrant:
    """Voice channel assignment parameters for packing into a PDU."""

    channel_id: int  # 0-15
    channel_number: int  # 0-4095
    tgid: int  # 0-65535
    source_id: int  # 0-16_777_215 (24 bits)
    timeslot: int = 0  # 0 or 1
    emergency: bool = False
    encrypted: bool = False
    duplex: bool = False
    priority: int = 0  # 0-7

    def __post_init__(self) -> None:
        """Validate field ranges."""
        if not 0 <= self.channel_id <= 0x0F:
            raise ValueError(f"channel_id must be 0-15, got {self.channel_id}")
        if not 0 <= self.channel_number <= 0x0FFF:
            raise ValueError(f"channel_number must be 0-4095, got {self.channel_number}")
        if not 0 <= self.tgid <= 0xFFFF:
            raise ValueError(f"tgid must be 0-65535, got {self.tgid}")
        if not 0 <= self.source_id <= 0xFFFFFF:
            raise ValueError(f"source_id must be 24-bit, got {self.source_id}")
        if self.timeslot not in (0, 1):
            raise ValueError(f"timeslot must be 0 or 1, got {self.timeslot}")
        if not 0 <= self.priority <= 0x07:
            raise ValueError(f"priority must be 0-7, got {self.priority}")


@dataclass
class VoiceBurstHeader:
    """Minimal voice burst header used for test framing."""

    timeslot: int
    tgid: int
    source_id: int
    channel_ref: int
    payload_type: VoicePayloadType = VoicePayloadType.AMBE
    encrypted: bool = False
    emergency: bool = False

    def to_bytes(self) -> bytes:
        """Serialize the header into an 8-byte structure.

        Layout:
        - Byte 0: [timeslot:1][emergency:1][encrypted:1][payload_type:5]
        - Bytes 1-2: Talkgroup ID (big endian)
        - Bytes 3-5: Source ID (24-bit, big endian)
        - Bytes 6-7: Channel reference (12-bit, big endian nibble + byte)
        """
        if self.timeslot not in (0, 1):
            raise ValueError(f"timeslot must be 0 or 1, got {self.timeslot}")
        if not 0 <= self.tgid <= 0xFFFF:
            raise ValueError(f"tgid must be 0-65535, got {self.tgid}")
        if not 0 <= self.source_id <= 0xFFFFFF:
            raise ValueError(f"source_id must be 24-bit, got {self.source_id}")
        if not 0 <= self.channel_ref <= 0x0FFF:
            raise ValueError(f"channel_ref must be 0-4095, got {self.channel_ref}")

        first_byte = (
            (self.timeslot & 0x01) << 7
            | (int(self.emergency) << 6)
            | (int(self.encrypted) << 5)
            | (self.payload_type.value & 0x1F)
        )

        header = bytearray(8)
        header[0] = first_byte
        header[1] = (self.tgid >> 8) & 0xFF
        header[2] = self.tgid & 0xFF
        header[3] = (self.source_id >> 16) & 0xFF
        header[4] = (self.source_id >> 8) & 0xFF
        header[5] = self.source_id & 0xFF
        header[6] = (self.channel_ref >> 8) & 0x0F
        header[7] = self.channel_ref & 0xFF
        return bytes(header)

    @staticmethod
    def from_bytes(data: bytes) -> VoiceBurstHeader:
        """Parse a serialized voice burst header."""
        if len(data) < 8:
            raise ValueError(f"voice burst header must be 8 bytes, got {len(data)}")

        first = data[0]
        payload_type_val = first & 0x1F
        try:
            payload_type = VoicePayloadType(payload_type_val)
        except ValueError as exc:
            raise ValueError(f"unsupported payload type {payload_type_val}") from exc

        timeslot = (first >> 7) & 0x01
        emergency = bool((first >> 6) & 0x01)
        encrypted = bool((first >> 5) & 0x01)

        tgid = (data[1] << 8) | data[2]
        source_id = (data[3] << 16) | (data[4] << 8) | data[5]
        channel_ref = ((data[6] & 0x0F) << 8) | data[7]

        return VoiceBurstHeader(
            timeslot=timeslot,
            tgid=tgid,
            source_id=source_id,
            channel_ref=channel_ref,
            payload_type=payload_type,
            encrypted=encrypted,
            emergency=emergency,
        )


def encode_group_voice_grant_pdu(grant: TrafficChannelGrant) -> bytes:
    """Encode a Group Voice Channel Grant PDU payload (8 bytes).

    Field layout follows :func:`TSBKParser._parse_grp_v_ch_grant`:
    - Byte 0: Service options (emergency/encrypted/duplex/slot/priority)
    - Bytes 1-2: Channel identifier (4-bit ID + 12-bit channel number)
    - Bytes 3-4: Talkgroup ID
    - Bytes 5-7: Source unit ID
    """
    svc_opts = (
        (0x80 if grant.emergency else 0)
        | (0x40 if grant.encrypted else 0)
        | (0x20 if grant.duplex else 0)
        | ((grant.timeslot & 0x01) << 3)
        | (grant.priority & 0x07)
    )

    data = bytearray(8)
    data[0] = svc_opts
    data[1] = (grant.channel_id << 4) | ((grant.channel_number >> 8) & 0x0F)
    data[2] = grant.channel_number & 0xFF
    data[3] = (grant.tgid >> 8) & 0xFF
    data[4] = grant.tgid & 0xFF
    data[5] = (grant.source_id >> 16) & 0xFF
    data[6] = (grant.source_id >> 8) & 0xFF
    data[7] = grant.source_id & 0xFF
    return bytes(data)


def encode_explicit_voice_grant_pdu(
    grant: TrafficChannelGrant, uplink_channel: tuple[int, int] | None = None
) -> bytes:
    """Encode an explicit Group Voice Channel Grant Update PDU (8 bytes)."""
    uplink_id, uplink_num = (
        uplink_channel if uplink_channel else (grant.channel_id, grant.channel_number)
    )

    svc_opts = (
        (0x80 if grant.emergency else 0)
        | (0x40 if grant.encrypted else 0)
        | (0x20 if grant.duplex else 0)
        | ((grant.timeslot & 0x01) << 3)
        | (grant.priority & 0x07)
    )

    data = bytearray(8)
    data[0] = svc_opts
    data[1] = 0x00  # reserved
    data[2] = (grant.channel_id << 4) | ((grant.channel_number >> 8) & 0x0F)
    data[3] = grant.channel_number & 0xFF
    data[4] = (uplink_id << 4) | ((uplink_num >> 8) & 0x0F)
    data[5] = uplink_num & 0xFF
    data[6] = (grant.tgid >> 8) & 0xFF
    data[7] = grant.tgid & 0xFF
    return bytes(data)


def decode_voice_grant(data: bytes, parser: TSBKParser | None = None) -> dict[str, Any]:
    """Decode a voice grant PDU using the standard TSBK parser."""
    tsbk_parser = parser or TSBKParser()
    return tsbk_parser.parse(TSBKOpcode.GRP_V_CH_GRANT, 0, data)


def frame_codec_payload(header: VoiceBurstHeader, payload: bytes) -> bytes:
    """Wrap a codec frame with a burst header and 16-bit length prefix."""
    if len(payload) > 0xFFFF:
        raise ValueError(f"payload too large for framing ({len(payload)} bytes)")
    header_bytes = header.to_bytes()
    length_bytes = len(payload).to_bytes(2, "big")
    return header_bytes + length_bytes + payload


def unframe_codec_payload(data: bytes) -> tuple[VoiceBurstHeader, bytes]:
    """Extract header and payload from a framed burst."""
    if len(data) < 10:
        raise ValueError("framed payload too short to contain header and length")
    header = VoiceBurstHeader.from_bytes(data[:8])
    length = int.from_bytes(data[8:10], "big")
    payload = data[10:]
    if length != len(payload):
        raise ValueError(f"length prefix {length} does not match payload size {len(payload)}")
    return header, payload
