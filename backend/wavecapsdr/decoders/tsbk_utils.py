"""Helpers for packing and unpacking P25 TSBK payload fields.

Field bit positions in TSBK specs are defined relative to the full 80-bit
message (including the 16-bit opcode/mfid header). These helpers operate on
the 64-bit data payload starting at bit 16.
"""
from __future__ import annotations

from typing import Iterable

PAYLOAD_BITS = 64


def payload_to_bits(payload: bytes) -> list[int]:
    """Convert an 8-byte TSBK payload to a list of bits (MSB first)."""
    if len(payload) != 8:
        raise ValueError(f"TSBK payload must be 8 bytes, got {len(payload)}")

    bits: list[int] = [0] * PAYLOAD_BITS
    for i, byte in enumerate(payload):
        for bit in range(8):
            bits[i * 8 + bit] = (byte >> (7 - bit)) & 0x1
    return bits


def bits_to_payload(bits: Iterable[int]) -> bytes:
    """Convert a 64-bit iterable (MSB first) back to an 8-byte payload."""
    bit_list = list(bits)
    if len(bit_list) != PAYLOAD_BITS:
        raise ValueError(f"TSBK payload must be {PAYLOAD_BITS} bits")

    out = bytearray(8)
    for i in range(PAYLOAD_BITS):
        byte_idx = i // 8
        bit_pos = 7 - (i % 8)
        out[byte_idx] |= (bit_list[i] & 0x1) << bit_pos
    return bytes(out)


def read_field(bits: list[int], start: int, length: int) -> int:
    """Read a contiguous field starting at bit offset `start`."""
    if start < 0 or start + length > PAYLOAD_BITS:
        raise ValueError(f"Field {start}+{length} exceeds payload length")

    value = 0
    for idx in range(start, start + length):
        value = (value << 1) | (bits[idx] & 0x1)
    return value


def write_field(bits: list[int], start: int, length: int, value: int) -> None:
    """Write a contiguous field into the bit array (MSB first)."""
    if start < 0 or start + length > PAYLOAD_BITS:
        raise ValueError(f"Field {start}+{length} exceeds payload length")

    max_val = (1 << length) - 1
    if value < 0 or value > max_val:
        raise ValueError(f"Value {value} does not fit in {length} bits")

    for offset in range(length):
        bit_val = (value >> (length - offset - 1)) & 0x1
        bits[start + offset] = bit_val
