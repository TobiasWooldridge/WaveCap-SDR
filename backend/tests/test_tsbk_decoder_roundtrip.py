import numpy as np

from wavecapsdr.decoders.p25_frames import (
    CCITT_80_CHECKSUMS,
    DATA_DEINTERLEAVE,
    dibits_to_bits,
    extract_tsbk_blocks,
)
from wavecapsdr.decoders.p25_tsbk import TSBKOpcode, TSBKParser
from wavecapsdr.decoders.p25_tsbk_encoders import encode_status_update
from wavecapsdr.decoders.traffic_voice import (
    TrafficChannelGrant,
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
