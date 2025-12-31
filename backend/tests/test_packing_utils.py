from __future__ import annotations

import pytest

from wavecapsdr.utils.packing import (
    BitFieldSpec,
    append_parity_bit,
    bits_to_bytes,
    bits_to_int,
    bytes_to_bits,
    decode_variable_length_id,
    encode_variable_length_id,
    insert_crc16,
    pack_bit_fields,
    pad_bits,
    unpack_bit_fields,
    verify_parity_bit,
)


def test_pack_and_unpack_bit_fields_round_trip() -> None:
    fields = (
        BitFieldSpec("a", 3),
        BitFieldSpec("b", 5, min_value=1),
    )
    packed_bits = pack_bit_fields(fields, {"a": 5, "b": 0x1F}, pad_to=8)
    assert packed_bits == [1, 0, 1, 1, 1, 1, 1, 1]

    decoded = unpack_bit_fields(bits_to_bytes(packed_bits), fields)
    assert decoded == {"a": 5, "b": 0x1F}


def test_pack_bit_fields_rejects_out_of_range() -> None:
    fields = (BitFieldSpec("a", 3), BitFieldSpec("b", 5))
    with pytest.raises(ValueError):
        pack_bit_fields(fields, {"a": 8, "b": 0})


@pytest.mark.parametrize(
    ("value", "expected_width", "selector"),
    [
        (0x0A, 4, 0),
        (0xAA, 8, 1),
        (0xABCDE, 20, 2),
    ],
)
def test_variable_length_id_round_trip(
    value: int,
    expected_width: int,
    selector: int,
) -> None:
    widths = (4, 8, 20)
    encoded = encode_variable_length_id(value, widths)
    assert encoded.width == expected_width
    assert encoded.selector == selector

    decoded = decode_variable_length_id(encoded.bits, widths, encoded.selector)
    assert decoded.value == value
    assert decoded.width == expected_width
    assert decoded.selector == selector


def test_variable_length_selector_validation() -> None:
    with pytest.raises(ValueError):
        decode_variable_length_id([0, 1, 0, 1], (4, 8), 3)


@pytest.mark.parametrize("even_parity", [True, False])
def test_parity_helpers(even_parity: bool) -> None:
    payload = [1, 0, 1, 1]
    with_parity = append_parity_bit(payload, even=even_parity)
    assert verify_parity_bit(with_parity, even=even_parity)

    flipped = list(with_parity)
    flipped[-1] ^= 1
    assert not verify_parity_bit(flipped, even=even_parity)


def test_padding_respects_block_size_and_fill() -> None:
    padded = pad_bits([1, 0, 1], block_size=8, pad_bit=1)
    assert len(padded) == 8
    assert padded[:3] == [1, 0, 1]
    assert padded[3:] == [1] * 5


def test_crc_insertion_matches_reference() -> None:
    payload_bits = bytes_to_bits(b"123456789")
    with_crc = insert_crc16(payload_bits)
    crc_value = bits_to_int(with_crc, len(with_crc) - 16, 16)
    assert crc_value == 0x29B1
