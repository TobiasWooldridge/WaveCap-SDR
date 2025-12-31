"""Utility modules for WaveCap-SDR."""

from wavecapsdr.utils.profiler import Profiler, get_profiler, report_all
from wavecapsdr.utils.packing import (
    BitFieldSpec,
    VariableLengthEncoding,
    append_parity_bit,
    bits_to_bytes,
    bits_to_int,
    bytes_to_bits,
    decode_variable_length_id,
    encode_variable_length_id,
    insert_crc16,
    int_to_bits,
    pack_bit_fields,
    pad_bits,
    unpack_bit_fields,
    verify_parity_bit,
)

__all__ = [
    "Profiler",
    "get_profiler",
    "report_all",
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
