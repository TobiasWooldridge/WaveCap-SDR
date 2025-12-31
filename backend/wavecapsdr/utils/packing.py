from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class BitFieldSpec:
    """Specification for a named bitfield."""

    name: str
    width: int
    min_value: int = 0
    max_value: int | None = None

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError(f"{self.name} width must be positive (got {self.width})")
        max_for_width = (1 << self.width) - 1
        max_value = max_for_width if self.max_value is None else int(self.max_value)
        if max_value > max_for_width:
            raise ValueError(
                f"{self.name} max_value {max_value} exceeds width {self.width}"
            )
        object.__setattr__(self, "max_value", max_value)
        if self.min_value < 0 or self.min_value > max_value:
            raise ValueError(
                f"{self.name} min_value {self.min_value} out of range 0-{max_value}"
            )

    def validate(self, value: int) -> int:
        try:
            int_value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{self.name} is not an int-like value") from exc
        if int_value < self.min_value or int_value > int(self.max_value):
            raise ValueError(
                f"{self.name} out of range {self.min_value}-{self.max_value} "
                f"(got {int_value})"
            )
        return int_value


def int_to_bits(value: int, width: int) -> list[int]:
    """Convert int to big-endian bit list of fixed width."""
    if width <= 0:
        raise ValueError("width must be positive")
    if value < 0 or value >= (1 << width):
        raise ValueError(f"value {value} does not fit in {width} bits")
    return [(value >> (width - 1 - i)) & 1 for i in range(width)]


def bits_to_int(bits: Sequence[int], start: int, length: int) -> int:
    """Extract an integer from a bit sequence."""
    if length <= 0:
        raise ValueError("length must be positive")
    if start < 0 or start + length > len(bits):
        raise ValueError(
            f"cannot read {length} bits from offset {start} (len={len(bits)})"
        )
    value = 0
    for i in range(length):
        value = (value << 1) | (int(bits[start + i]) & 1)
    return value


def bytes_to_bits(data: bytes | bytearray) -> list[int]:
    """Expand bytes into a big-endian bit list."""
    bits: list[int] = []
    for byte in data:
        bits.extend(int_to_bits(byte, 8))
    return bits


def bits_to_bytes(bits: Sequence[int]) -> bytes:
    """Pack a sequence of bits into bytes (MSB first)."""
    if len(bits) % 8 != 0:
        raise ValueError("bit length must be a multiple of 8 to convert to bytes")
    out = bytearray(len(bits) // 8)
    for i in range(len(out)):
        out[i] = bits_to_int(bits, i * 8, 8)
    return bytes(out)


def pad_bits(bits: Sequence[int], block_size: int, pad_bit: int = 0) -> list[int]:
    """Pad bits to a multiple of block_size using pad_bit."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if pad_bit not in (0, 1):
        raise ValueError("pad_bit must be 0 or 1")
    padded = list(bits)
    remainder = len(padded) % block_size
    if remainder:
        padded.extend([pad_bit] * (block_size - remainder))
    return padded


def pack_bit_fields(
    fields: Sequence[BitFieldSpec],
    values: Mapping[str, int],
    *,
    pad_to: int | None = 8,
    pad_bit: int = 0,
) -> list[int]:
    """Pack named bitfields into a contiguous bit list."""
    bits: list[int] = []
    for field in fields:
        if field.name not in values:
            raise ValueError(f"missing value for {field.name}")
        validated = field.validate(values[field.name])
        bits.extend(int_to_bits(validated, field.width))
    if pad_to:
        bits = pad_bits(bits, pad_to, pad_bit)
    return bits


@dataclass(frozen=True)
class VariableLengthEncoding:
    """Result of encoding a variable-length integer."""

    value: int
    width: int
    selector: int
    bits: list[int]


def encode_variable_length_id(
    value: int,
    bit_lengths: Sequence[int],
) -> VariableLengthEncoding:
    """Encode an ID using the smallest fitting bit-length."""
    if not bit_lengths:
        raise ValueError("bit_lengths cannot be empty")
    for selector, width in enumerate(bit_lengths):
        if value < (1 << width):
            return VariableLengthEncoding(
                value=int(value),
                width=width,
                selector=selector,
                bits=int_to_bits(int(value), width),
            )
    raise ValueError(f"value {value} does not fit in provided bit lengths {bit_lengths}")


def decode_variable_length_id(
    bit_buffer: Sequence[int] | bytes | bytearray,
    bit_lengths: Sequence[int],
    selector: int,
) -> VariableLengthEncoding:
    """Decode an ID given a selector into bit_lengths."""
    if selector < 0 or selector >= len(bit_lengths):
        raise ValueError(f"selector {selector} out of range for {len(bit_lengths)}")
    width = bit_lengths[selector]
    bits = _coerce_bits(bit_buffer)
    if len(bits) < width:
        raise ValueError(f"need {width} bits to decode ID (got {len(bits)})")
    value = bits_to_int(bits, 0, width)
    return VariableLengthEncoding(
        value=value,
        width=width,
        selector=selector,
        bits=bits[:width],
    )


def append_parity_bit(bits: Sequence[int], *, even: bool = True) -> list[int]:
    """Append a parity bit (even or odd) to the end of the bit sequence."""
    parity = sum(int(b) & 1 for b in bits) & 1
    parity_bit = parity if even else parity ^ 1
    return list(bits) + [parity_bit]


def verify_parity_bit(bits: Sequence[int], *, even: bool = True) -> bool:
    """Verify the trailing parity bit of a sequence."""
    if not bits:
        return False
    return (sum(int(b) & 1 for b in bits) & 1) == (0 if even else 1)


def insert_crc16(
    bits: Sequence[int],
    *,
    polynomial: int = 0x1021,
    init: int = 0xFFFF,
) -> list[int]:
    """Append CRC16-CCITT bits to the payload."""
    crc = _crc16_ccitt(bits, polynomial=polynomial, init=init)
    return list(bits) + int_to_bits(crc, 16)


def unpack_bit_fields(
    bit_buffer: Sequence[int] | bytes | bytearray,
    fields: Sequence[BitFieldSpec],
    *,
    offset: int = 0,
) -> dict[str, int]:
    """Extract named bitfields from a buffer."""
    bits = _coerce_bits(bit_buffer)
    results: dict[str, int] = {}
    cursor = offset
    for field in fields:
        if cursor + field.width > len(bits):
            raise ValueError(
                f"not enough bits to read {field.name} ({field.width} bits at {cursor})"
            )
        value = bits_to_int(bits, cursor, field.width)
        results[field.name] = field.validate(value)
        cursor += field.width
    return results


def _crc16_ccitt(
    bits: Sequence[int],
    *,
    polynomial: int = 0x1021,
    init: int = 0xFFFF,
) -> int:
    """Calculate CRC16-CCITT over a bit sequence."""
    crc = init & 0xFFFF
    for bit in bits:
        bit_val = int(bit) & 1
        xor = ((crc >> 15) & 1) ^ bit_val
        crc = (crc << 1) & 0xFFFF
        if xor:
            crc ^= polynomial
    return crc & 0xFFFF


def _coerce_bits(bit_buffer: Sequence[int] | bytes | bytearray) -> list[int]:
    """Ensure we operate on a list of bits."""
    if isinstance(bit_buffer, (bytes, bytearray)):
        return bytes_to_bits(bit_buffer)
    return [int(b) & 1 for b in bit_buffer]


__all__ = [
    "BitFieldSpec",
    "VariableLengthEncoding",
    "append_parity_bit",
    "bits_to_bytes",
    "bits_to_int",
    "bytes_to_bits",
    "decode_variable_length_id",
    "encode_variable_length_id",
    "insert_crc16",
    "int_to_bits",
    "pack_bit_fields",
    "pad_bits",
    "unpack_bit_fields",
    "verify_parity_bit",
]
