"""Round-trip coverage for traffic/voice PDU encoders."""

import pytest

from wavecapsdr.decoders.p25_tsbk import TSBKOpcode, TSBKParser
from wavecapsdr.decoders.traffic_voice import (
    TrafficChannelGrant,
    VoiceBurstHeader,
    VoicePayloadType,
    decode_voice_grant,
    encode_explicit_voice_grant_pdu,
    encode_group_voice_grant_pdu,
    frame_codec_payload,
    unframe_codec_payload,
)


def test_group_voice_grant_pdu_round_trip() -> None:
    grant = TrafficChannelGrant(
        channel_id=0xA,
        channel_number=0x2BC,
        tgid=0x1234,
        source_id=0x0ABCDE,
        timeslot=1,
        emergency=True,
        encrypted=True,
        duplex=True,
        priority=5,
    )

    encoded = encode_group_voice_grant_pdu(grant)

    # Service opts: emergency/encrypted/duplex/slot1 + priority=5
    expected = bytes([0xED, 0xA2, 0xBC, 0x12, 0x34, 0x0A, 0xBC, 0xDE])
    assert encoded == expected

    decoded = decode_voice_grant(encoded, parser=TSBKParser())
    assert decoded["type"] == "GROUP_VOICE_GRANT"
    assert decoded["slot_id"] == grant.timeslot
    assert decoded["tgid"] == grant.tgid
    assert decoded["source_id"] == grant.source_id
    assert decoded["channel"] == (grant.channel_id << 12) | grant.channel_number
    assert decoded["emergency"] is True
    assert decoded["encrypted"] is True
    assert decoded["priority"] == grant.priority


def test_explicit_voice_grant_encodes_slot_and_channels() -> None:
    grant = TrafficChannelGrant(
        channel_id=0x2,
        channel_number=0x345,
        tgid=0xBEEF,
        source_id=0x010203,
        timeslot=0,
        emergency=False,
        encrypted=True,
        duplex=False,
        priority=1,
    )

    encoded = encode_explicit_voice_grant_pdu(grant, uplink_channel=(0xC, 0x678))

    assert encoded[:2] == bytes([0x41, 0x00])  # encrypted flag + priority + reserved byte
    assert encoded[2:] == bytes([0x23, 0x45, 0xC6, 0x78, 0xBE, 0xEF])

    parser = TSBKParser()
    decoded = parser.parse(TSBKOpcode.GRP_V_CH_GRANT_UPDT_EXP, 0, encoded)
    assert decoded["slot_id"] == 0
    assert decoded["downlink_channel"] == (grant.channel_id << 12) | grant.channel_number
    assert decoded["uplink_channel"] == (0xC << 12) | 0x678
    assert decoded["tgid"] == grant.tgid


def test_voice_burst_framing_preserves_header_and_payload() -> None:
    header = VoiceBurstHeader(
        timeslot=1,
        tgid=0x4567,
        source_id=0x010203,
        channel_ref=0x0A5,
        payload_type=VoicePayloadType.IMBE,
        encrypted=True,
        emergency=False,
    )
    payload = bytes.fromhex("11223344556677889900")

    framed = frame_codec_payload(header, payload)
    parsed_header, parsed_payload = unframe_codec_payload(framed)

    assert parsed_header == header
    assert parsed_payload == payload


def test_unframe_validates_length_prefix() -> None:
    header = VoiceBurstHeader(
        timeslot=0,
        tgid=1,
        source_id=1,
        channel_ref=1,
    )
    framed = frame_codec_payload(header, b"\x01\x02")

    with pytest.raises(ValueError):
        # Drop one byte to force mismatch
        unframe_codec_payload(framed[:-1])
