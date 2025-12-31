import numpy as np
import pytest

from wavecapsdr.decoders.p25_frames import (
    CCITT_80_CHECKSUMS,
    DATA_DEINTERLEAVE,
    dibits_to_bits,
    extract_tsbk_blocks,
)
from wavecapsdr.decoders.p25_tsbk import TSBKOpcode, TSBKParser
from wavecapsdr.decoders.p25_tsbk_encoders import encode_status_update
from wavecapsdr.decoders.p25_tsbk_encoders import (
    encode_group_affiliation_response,
    encode_unit_deregistration_ack,
    encode_unit_registration_response,
    encode_unit_service_request,
)
from wavecapsdr.decoders.traffic_voice import (
    TrafficChannelGrant,
    encode_explicit_voice_grant_pdu,
    encode_group_voice_grant_pdu,
)
from wavecapsdr.utils.packing import int_to_bits
from wavecapsdr.dsp.fec.trellis import trellis_encode
from wavecapsdr.decoders.tsbk_utils import payload_to_bits


def _compute_crc_ccitt_80(bits: list[int]) -> int:
    if len(bits) != 80:
        raise ValueError(f"Expected 80 bits, got {len(bits)}")
    checksum = 0xFFFF
    for idx, bit in enumerate(bits):
        if bit:
            checksum ^= CCITT_80_CHECKSUMS[idx]
    return checksum


def _bits_to_dibits(bits: list[int]) -> np.ndarray:
    if len(bits) % 2 != 0:
        raise ValueError("Bit length must be even to convert to dibits")
    dibits = np.zeros(len(bits) // 2, dtype=np.uint8)
    for i in range(len(dibits)):
        dibits[i] = (bits[i * 2] << 1) | bits[i * 2 + 1]
    return dibits


def _interleave_bits(bits: np.ndarray) -> np.ndarray:
    if len(bits) != 196:
        raise ValueError(f"Expected 196 bits, got {len(bits)}")
    interleaved = np.zeros(196, dtype=np.uint8)
    for i in range(196):
        interleaved[i] = bits[DATA_DEINTERLEAVE[i]]
    return interleaved


def _encode_tsbk_block(opcode: int, mfid: int, payload: bytes) -> np.ndarray:
    header_bits = [1, 0]  # last_block=1, protect=0
    header_bits += int_to_bits(opcode, 6)
    header_bits += int_to_bits(mfid, 8)
    payload_bits = payload_to_bits(payload)
    bits_80 = header_bits + payload_bits
    crc = _compute_crc_ccitt_80(bits_80)
    tsbk_bits = bits_80 + int_to_bits(crc, 16)

    input_dibits = _bits_to_dibits(tsbk_bits)
    encoded_dibits = trellis_encode(input_dibits)

    encoded_with_flush = np.zeros(98, dtype=np.uint8)
    encoded_with_flush[:len(encoded_dibits)] = encoded_dibits
    interleaved_bits = _interleave_bits(dibits_to_bits(encoded_with_flush))

    return _bits_to_dibits(interleaved_bits.tolist())


def _encode_iden_up_vu_payload(
    *,
    identifier: int,
    bandwidth_code: int,
    tx_offset_sign: bool,
    tx_offset_raw: int,
    spacing_raw: int,
    base_freq_mhz: float,
) -> bytes:
    if identifier < 0 or identifier > 0x0F:
        raise ValueError("identifier must be 0-15")
    if bandwidth_code < 0 or bandwidth_code > 0x0F:
        raise ValueError("bandwidth_code must be 0-15")
    if tx_offset_raw < 0 or tx_offset_raw > 0x1FFF:
        raise ValueError("tx_offset_raw must be 13 bits")
    if spacing_raw < 0 or spacing_raw > 0x3FF:
        raise ValueError("spacing_raw must be 10 bits")

    base_freq_raw = int(base_freq_mhz / 0.000005)
    data = bytearray(8)
    data[0] = ((identifier & 0x0F) << 4) | (bandwidth_code & 0x0F)
    data[1] = ((1 if tx_offset_sign else 0) << 7) | ((tx_offset_raw >> 6) & 0x7F)
    data[2] = ((tx_offset_raw & 0x3F) << 2) | ((spacing_raw >> 8) & 0x03)
    data[3] = spacing_raw & 0xFF
    data[4:8] = base_freq_raw.to_bytes(4, "big")
    return bytes(data)


def _encode_rfss_status_payload(
    *,
    lra: int,
    system_id: int,
    rfss_id: int,
    site_id: int,
    frequency_band: int,
    channel_number: int,
    service_class: int,
    active_network: bool = False,
) -> bytes:
    data = bytearray(8)
    data[0] = lra & 0xFF
    data[1] = ((0x08 if active_network else 0) | ((system_id >> 8) & 0x0F))
    data[2] = system_id & 0xFF
    data[3] = rfss_id & 0xFF
    data[4] = site_id & 0xFF
    data[5] = ((frequency_band & 0x0F) << 4) | ((channel_number >> 8) & 0x0F)
    data[6] = channel_number & 0xFF
    data[7] = service_class & 0xFF
    return bytes(data)


def _encode_adjacent_status_payload(
    *,
    lra: int,
    system_id: int,
    rfss_id: int,
    site_id: int,
    frequency_band: int,
    channel_number: int,
    service_class: int,
    active_network: bool = False,
) -> bytes:
    return _encode_rfss_status_payload(
        lra=lra,
        system_id=system_id,
        rfss_id=rfss_id,
        site_id=site_id,
        frequency_band=frequency_band,
        channel_number=channel_number,
        service_class=service_class,
        active_network=active_network,
    )


def _encode_net_status_payload(
    *,
    lra: int,
    wacn: int,
    system_id: int,
    frequency_band: int,
    channel_number: int,
    service_class: int,
) -> bytes:
    data = bytearray(8)
    data[0] = lra & 0xFF
    data[1] = (wacn >> 12) & 0xFF
    data[2] = (wacn >> 4) & 0xFF
    data[3] = ((wacn & 0x0F) << 4) | ((system_id >> 8) & 0x0F)
    data[4] = system_id & 0xFF
    data[5] = ((frequency_band & 0x0F) << 4) | ((channel_number >> 8) & 0x0F)
    data[6] = channel_number & 0xFF
    data[7] = service_class & 0xFF
    return bytes(data)


def _encode_sys_srv_payload(*, services_available: int, services_supported: int) -> bytes:
    data = bytearray(8)
    data[0] = (services_available >> 16) & 0xFF
    data[1] = (services_available >> 8) & 0xFF
    data[2] = services_available & 0xFF
    data[3] = (services_supported >> 16) & 0xFF
    data[4] = (services_supported >> 8) & 0xFF
    data[5] = services_supported & 0xFF
    return bytes(data)


def _encode_deny_response_payload(
    *,
    service_type: int,
    reason: int,
    target_address: int,
) -> bytes:
    data = bytearray(8)
    data[0] = (service_type & 0x3F) << 2
    data[1] = reason & 0xFF
    data[2] = (target_address >> 16) & 0xFF
    data[3] = (target_address >> 8) & 0xFF
    data[4] = target_address & 0xFF
    return bytes(data)


def _encode_iden_up_tdma_payload(
    *,
    identifier: int,
    channel_type: int,
    tx_offset_sign: bool,
    tx_offset_raw: int,
    spacing_raw: int,
    base_freq_mhz: float,
) -> bytes:
    if identifier < 0 or identifier > 0x0F:
        raise ValueError("identifier must be 0-15")
    if channel_type < 0 or channel_type > 0x0F:
        raise ValueError("channel_type must be 0-15")
    if tx_offset_raw < 0 or tx_offset_raw > 0x1FFF:
        raise ValueError("tx_offset_raw must be 13 bits")
    if spacing_raw < 0 or spacing_raw > 0x3FF:
        raise ValueError("spacing_raw must be 10 bits")

    base_freq_raw = int(base_freq_mhz / 0.000005)
    data = bytearray(8)
    data[0] = ((identifier & 0x0F) << 4) | (channel_type & 0x0F)
    data[1] = ((1 if tx_offset_sign else 0) << 7) | ((tx_offset_raw >> 6) & 0x7F)
    data[2] = ((tx_offset_raw & 0x3F) << 2) | ((spacing_raw >> 8) & 0x03)
    data[3] = spacing_raw & 0xFF
    data[4:8] = base_freq_raw.to_bytes(4, "big")
    return bytes(data)


def test_status_update_decodes_from_encoded_tsbk_block() -> None:
    payload = encode_status_update(
        unit_status=0x12,
        user_status=0x34,
        target_id=0x00ABCD,
        source_id=0x010203,
    )
    dibits = _encode_tsbk_block(TSBKOpcode.STATUS_UPDT, 0, payload)

    blocks = extract_tsbk_blocks(dibits)
    assert len(blocks) == 1
    block = blocks[0]
    assert block.crc_valid is True
    assert block.opcode == TSBKOpcode.STATUS_UPDT
    assert block.mfid == 0
    assert block.data == payload

    parser = TSBKParser()
    parsed = parser.parse(block.opcode, block.mfid, block.data)
    assert parsed["type"] == "STATUS_UPDATE"
    assert parsed["unit_status"] == 0x12
    assert parsed["user_status"] == 0x34
    assert parsed["target_id"] == 0x00ABCD
    assert parsed["source_id"] == 0x010203


def test_group_voice_grant_decodes_from_encoded_tsbk_block() -> None:
    grant = TrafficChannelGrant(
        channel_id=0x2,
        channel_number=0x123,
        tgid=0x4567,
        source_id=0x010203,
        timeslot=1,
        emergency=False,
        encrypted=True,
        duplex=False,
        priority=3,
    )
    payload = encode_group_voice_grant_pdu(grant)
    dibits = _encode_tsbk_block(TSBKOpcode.GRP_V_CH_GRANT, 0, payload)

    blocks = extract_tsbk_blocks(dibits)
    assert len(blocks) == 1
    block = blocks[0]
    assert block.crc_valid is True
    assert block.opcode == TSBKOpcode.GRP_V_CH_GRANT
    assert block.data == payload

    parser = TSBKParser()
    parsed = parser.parse(block.opcode, block.mfid, block.data)
    assert parsed["type"] == "GROUP_VOICE_GRANT"
    assert parsed["tgid"] == grant.tgid
    assert parsed["source_id"] == grant.source_id
    assert parsed["channel"] == (grant.channel_id << 12) | grant.channel_number


def test_group_voice_grant_update_explicit_decodes_from_encoded_tsbk_block() -> None:
    grant = TrafficChannelGrant(
        channel_id=0x4,
        channel_number=0x456,
        tgid=0x2222,
        source_id=0x010203,
        timeslot=0,
        emergency=True,
        encrypted=False,
        duplex=True,
        priority=5,
    )
    payload = encode_explicit_voice_grant_pdu(grant, uplink_channel=(0x7, 0x123))
    dibits = _encode_tsbk_block(TSBKOpcode.GRP_V_CH_GRANT_UPDT_EXP, 0, payload)

    blocks = extract_tsbk_blocks(dibits)
    assert len(blocks) == 1
    block = blocks[0]
    assert block.crc_valid is True
    assert block.opcode == TSBKOpcode.GRP_V_CH_GRANT_UPDT_EXP

    parsed = TSBKParser().parse(block.opcode, block.mfid, block.data)
    assert parsed["type"] == "GROUP_VOICE_GRANT_UPDATE_EXPLICIT"
    assert parsed["tgid"] == grant.tgid
    assert parsed["downlink_channel"] == (grant.channel_id << 12) | grant.channel_number
    assert parsed["uplink_channel"] == (0x7 << 12) | 0x123


def test_group_affiliation_response_decodes_from_encoded_tsbk_block() -> None:
    payload = encode_group_affiliation_response(
        response_code=1,
        announcement_group=0x1111,
        group_address=0x2222,
        target_id=0x00ABCDE,
        global_affiliation=True,
    )
    dibits = _encode_tsbk_block(TSBKOpcode.GRP_AFF_RSP, 0, payload)

    blocks = extract_tsbk_blocks(dibits)
    assert len(blocks) == 1
    block = blocks[0]
    assert block.crc_valid is True
    assert block.opcode == TSBKOpcode.GRP_AFF_RSP

    parsed = TSBKParser().parse(block.opcode, block.mfid, block.data)
    assert parsed["type"] == "GROUP_AFFILIATION_RESPONSE"
    assert parsed["response"] == 1
    assert parsed["announcement_group"] == 0x1111
    assert parsed["tgid"] == 0x2222
    assert parsed["target_id"] == 0x00ABCDE
    assert parsed["global"] is True


def test_unit_registration_response_decodes_from_encoded_tsbk_block() -> None:
    payload = encode_unit_registration_response(
        response_code=0,
        system_id=0x321,
        source_id=0x010203,
        registered_address=0x00FEDC,
    )
    dibits = _encode_tsbk_block(TSBKOpcode.UNIT_REG_RSP, 0, payload)

    blocks = extract_tsbk_blocks(dibits)
    assert len(blocks) == 1
    block = blocks[0]
    assert block.crc_valid is True
    assert block.opcode == TSBKOpcode.UNIT_REG_RSP

    parsed = TSBKParser().parse(block.opcode, block.mfid, block.data)
    assert parsed["type"] == "UNIT_REGISTRATION_RESPONSE"
    assert parsed["response"] == 0
    assert parsed["system_id"] == 0x321
    assert parsed["source_id"] == 0x010203
    assert parsed["registered_address"] == 0x00FEDC


def test_unit_deregistration_ack_decodes_from_encoded_tsbk_block() -> None:
    payload = encode_unit_deregistration_ack(
        wacn=0xABCDE,
        system_id=0x321,
        target_id=0x00DADA,
    )
    dibits = _encode_tsbk_block(TSBKOpcode.UNIT_DEREG_ACK, 0, payload)

    blocks = extract_tsbk_blocks(dibits)
    assert len(blocks) == 1
    block = blocks[0]
    assert block.crc_valid is True
    assert block.opcode == TSBKOpcode.UNIT_DEREG_ACK

    parsed = TSBKParser().parse(block.opcode, block.mfid, block.data)
    assert parsed["type"] == "UNIT_DEREGISTRATION_ACK"
    assert parsed["wacn"] == 0xABCDE
    assert parsed["system_id"] == 0x321
    assert parsed["target_id"] == 0x00DADA


def test_unit_service_request_decodes_from_encoded_tsbk_block() -> None:
    payload = encode_unit_service_request(
        service_options=0xA5,
        target_id=0x00F001,
        source_id=0x001122,
    )
    dibits = _encode_tsbk_block(TSBKOpcode.UU_ANS_REQ, 0, payload)

    blocks = extract_tsbk_blocks(dibits)
    assert len(blocks) == 1
    block = blocks[0]
    assert block.crc_valid is True
    assert block.opcode == TSBKOpcode.UU_ANS_REQ

    parsed = TSBKParser().parse(block.opcode, block.mfid, block.data)
    assert parsed["type"] == "UNIT_SERVICE_REQUEST"
    assert parsed["service_options"] == 0xA5
    assert parsed["target_id"] == 0x00F001
    assert parsed["source_id"] == 0x001122


def test_iden_up_vu_decodes_from_encoded_tsbk_block() -> None:
    payload = _encode_iden_up_vu_payload(
        identifier=3,
        bandwidth_code=0x5,
        tx_offset_sign=True,
        tx_offset_raw=45,
        spacing_raw=100,
        base_freq_mhz=851.00625,
    )
    dibits = _encode_tsbk_block(TSBKOpcode.IDEN_UP_VU, 0, payload)

    blocks = extract_tsbk_blocks(dibits)
    assert len(blocks) == 1
    block = blocks[0]
    assert block.crc_valid is True
    assert block.opcode == TSBKOpcode.IDEN_UP_VU

    parsed = TSBKParser().parse(block.opcode, block.mfid, block.data)
    assert parsed["type"] == "IDENTIFIER_UPDATE_VU"
    assert parsed["identifier"] == 3
    assert parsed["bandwidth_code"] == 0x5
    assert parsed["bandwidth_khz"] == 12.5
    assert parsed["channel_spacing_khz"] == 12.5
    assert parsed["tx_offset_sign"] is True
    assert parsed["base_freq_mhz"] == pytest.approx(851.00625, rel=1e-6)


def test_iden_up_decodes_from_encoded_tsbk_block() -> None:
    payload = _encode_iden_up_vu_payload(
        identifier=9,
        bandwidth_code=0x3,
        tx_offset_sign=False,
        tx_offset_raw=80,
        spacing_raw=200,
        base_freq_mhz=762.125,
    )
    dibits = _encode_tsbk_block(TSBKOpcode.IDEN_UP, 0, payload)

    blocks = extract_tsbk_blocks(dibits)
    assert len(blocks) == 1
    block = blocks[0]
    assert block.crc_valid is True
    assert block.opcode == TSBKOpcode.IDEN_UP

    parsed = TSBKParser().parse(block.opcode, block.mfid, block.data)
    assert parsed["type"] == "IDENTIFIER_UPDATE"
    assert parsed["identifier"] == 9
    assert parsed["bandwidth_code"] == 0x3
    assert parsed["channel_spacing_khz"] == 25.0
    assert parsed["tx_offset_sign"] is False
    assert parsed["base_freq_mhz"] == pytest.approx(762.125, rel=1e-6)


def test_rfss_status_decodes_from_encoded_tsbk_block() -> None:
    payload = _encode_rfss_status_payload(
        lra=0x12,
        system_id=0x321,
        rfss_id=0x45,
        site_id=0x67,
        frequency_band=2,
        channel_number=0x234,
        service_class=0xAA,
        active_network=False,
    )
    dibits = _encode_tsbk_block(TSBKOpcode.RFSS_STS_BCAST, 0, payload)

    blocks = extract_tsbk_blocks(dibits)
    assert len(blocks) == 1
    block = blocks[0]
    assert block.crc_valid is True
    assert block.opcode == TSBKOpcode.RFSS_STS_BCAST

    parsed = TSBKParser().parse(block.opcode, block.mfid, block.data)
    assert parsed["type"] == "RFSS_STATUS"
    assert parsed["lra"] == 0x12
    assert parsed["system_id"] == 0x321
    assert parsed["rfss_id"] == 0x45
    assert parsed["site_id"] == 0x67
    assert parsed["frequency_band"] == 2
    assert parsed["channel_number"] == 0x234
    assert parsed["service_class"] == 0xAA


def test_net_status_decodes_from_encoded_tsbk_block() -> None:
    payload = _encode_net_status_payload(
        lra=0x10,
        wacn=0xABCDE,
        system_id=0x321,
        frequency_band=3,
        channel_number=0x145,
        service_class=0x55,
    )
    dibits = _encode_tsbk_block(TSBKOpcode.NET_STS_BCAST, 0, payload)

    blocks = extract_tsbk_blocks(dibits)
    assert len(blocks) == 1
    block = blocks[0]
    assert block.crc_valid is True
    assert block.opcode == TSBKOpcode.NET_STS_BCAST

    parsed = TSBKParser().parse(block.opcode, block.mfid, block.data)
    assert parsed["type"] == "NETWORK_STATUS"
    assert parsed["lra"] == 0x10
    assert parsed["wacn"] == 0xABCDE
    assert parsed["system_id"] == 0x321
    assert parsed["frequency_band"] == 3
    assert parsed["channel_number"] == 0x145
    assert parsed["service_class"] == 0x55


def test_status_update_request_decodes_from_encoded_tsbk_block() -> None:
    payload = encode_status_update(
        unit_status=0xFE,
        user_status=0xED,
        target_id=0x00ABCD,
        source_id=0x000123,
    )
    dibits = _encode_tsbk_block(TSBKOpcode.STATUS_QUERY, 0, payload)

    blocks = extract_tsbk_blocks(dibits)
    assert len(blocks) == 1
    block = blocks[0]
    assert block.crc_valid is True
    assert block.opcode == TSBKOpcode.STATUS_QUERY

    parsed = TSBKParser().parse(block.opcode, block.mfid, block.data)
    assert parsed["type"] == "STATUS_UPDATE_REQUEST"
    assert parsed["unit_status"] == 0xFE
    assert parsed["user_status"] == 0xED
    assert parsed["target_id"] == 0x00ABCD
    assert parsed["source_id"] == 0x000123


def test_sys_service_broadcast_decodes_from_encoded_tsbk_block() -> None:
    payload = _encode_sys_srv_payload(
        services_available=0xA1B2C3,
        services_supported=0x0F0E0D,
    )
    dibits = _encode_tsbk_block(TSBKOpcode.SYS_SRV_BCAST, 0, payload)

    blocks = extract_tsbk_blocks(dibits)
    assert len(blocks) == 1
    block = blocks[0]
    assert block.crc_valid is True
    assert block.opcode == TSBKOpcode.SYS_SRV_BCAST

    parsed = TSBKParser().parse(block.opcode, block.mfid, block.data)
    assert parsed["type"] == "SYSTEM_SERVICE"
    assert parsed["services_available"] == 0xA1B2C3
    assert parsed["services_supported"] == 0x0F0E0D
    assert parsed["composite_control"] is True
    assert parsed["data_services"] is False


def test_deny_response_decodes_from_encoded_tsbk_block() -> None:
    payload = _encode_deny_response_payload(
        service_type=0x15,
        reason=0x20,
        target_address=0x123456,
    )
    dibits = _encode_tsbk_block(TSBKOpcode.DENY_RSP, 0, payload)

    blocks = extract_tsbk_blocks(dibits)
    assert len(blocks) == 1
    block = blocks[0]
    assert block.crc_valid is True
    assert block.opcode == TSBKOpcode.DENY_RSP

    parsed = TSBKParser().parse(block.opcode, block.mfid, block.data)
    assert parsed["type"] == "DENY_RESPONSE"
    assert parsed["service_type"] == 0x15
    assert parsed["reason"] == 0x20
    assert parsed["reason_text"] == "Group not valid"
    assert parsed["target_address"] == 0x123456


def test_adjacent_status_decodes_from_encoded_tsbk_block() -> None:
    payload = _encode_adjacent_status_payload(
        lra=0x10,
        system_id=0x123,
        rfss_id=0x01,
        site_id=0x02,
        frequency_band=5,
        channel_number=0x045,
        service_class=0x0F,
        active_network=False,
    )
    dibits = _encode_tsbk_block(TSBKOpcode.ADJ_STS_BCAST, 0, payload)

    blocks = extract_tsbk_blocks(dibits)
    assert len(blocks) == 1
    block = blocks[0]
    assert block.crc_valid is True
    assert block.opcode == TSBKOpcode.ADJ_STS_BCAST

    parsed = TSBKParser().parse(block.opcode, block.mfid, block.data)
    assert parsed["type"] == "ADJACENT_STATUS"
    assert parsed["lra"] == 0x10
    assert parsed["system_id"] == 0x123
    assert parsed["rfss_id"] == 0x01
    assert parsed["site_id"] == 0x02
    assert parsed["frequency_band"] == 5
    assert parsed["channel_number"] == 0x045
    assert parsed["service_class"] == 0x0F


def test_iden_up_tdma_decodes_from_encoded_tsbk_block() -> None:
    payload = _encode_iden_up_tdma_payload(
        identifier=4,
        channel_type=0x02,
        tx_offset_sign=False,
        tx_offset_raw=25,
        spacing_raw=200,
        base_freq_mhz=762.00000,
    )
    dibits = _encode_tsbk_block(TSBKOpcode.IDEN_UP_TDMA, 0, payload)

    blocks = extract_tsbk_blocks(dibits)
    assert len(blocks) == 1
    block = blocks[0]
    assert block.crc_valid is True
    assert block.opcode == TSBKOpcode.IDEN_UP_TDMA

    parsed = TSBKParser().parse(block.opcode, block.mfid, block.data)
    assert parsed["type"] == "IDENTIFIER_UPDATE_TDMA"
    assert parsed["identifier"] == 4
    assert parsed["channel_type"] == 0x02
    assert parsed["access_type"] == "TDMA"
    assert parsed["slot_count"] == 2
    assert parsed["channel_spacing_khz"] == 25.0
    assert parsed["tx_offset_sign"] is False
    assert parsed["base_freq_mhz"] == pytest.approx(762.0, rel=1e-6)
