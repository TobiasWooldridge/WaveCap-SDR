"""Test TSBK encoding/decoding chain to find systematic errors."""

import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from wavecapsdr.dsp.fec.trellis import trellis_encode, trellis_decode
from wavecapsdr.decoders.p25_frames import (
    DATA_DEINTERLEAVE,
    dibits_to_bits,
    bits_to_int,
    deinterleave_data,
    crc16_ccitt_p25,
)


def compute_crc16_ccitt(data_bits: np.ndarray) -> int:
    """Compute CRC-16 CCITT over data bits."""
    poly = 0x1021
    crc = 0xFFFF

    for bit in data_bits:
        msb = (crc >> 15) & 1
        crc = ((crc << 1) | int(bit)) & 0xFFFF
        if msb:
            crc ^= poly

    # Final shift
    for _ in range(16):
        msb = (crc >> 15) & 1
        crc = (crc << 1) & 0xFFFF
        if msb:
            crc ^= poly

    return crc


def bits_to_dibits(bits: np.ndarray) -> np.ndarray:
    """Convert bit array to dibit array (MSB first)."""
    dibits = np.zeros(len(bits) // 2, dtype=np.uint8)
    for i in range(len(dibits)):
        dibits[i] = (bits[i*2] << 1) | bits[i*2 + 1]
    return dibits


def interleave_data(bits: np.ndarray) -> np.ndarray:
    """Interleave 196-bit block (inverse of deinterleave)."""
    if len(bits) != 196:
        raise ValueError(f"Expected 196 bits, got {len(bits)}")

    # Build inverse mapping: for each deinterleaved position, where does it go in interleaved?
    interleaved = np.zeros(196, dtype=np.uint8)
    for out_pos in range(196):
        in_pos = DATA_DEINTERLEAVE[out_pos]
        interleaved[in_pos] = bits[out_pos]

    return interleaved


def test_trellis_roundtrip():
    """Test basic trellis encode/decode roundtrip."""
    print("\n=== Test 1: Trellis Roundtrip ===")

    # 48 dibits input -> 96 dibits output (with padding)
    original_dibits = np.array([i % 4 for i in range(48)], dtype=np.uint8)

    # Encode
    encoded = trellis_encode(original_dibits)
    print(f"Original: {len(original_dibits)} dibits -> Encoded: {len(encoded)} dibits")

    # Decode
    decoded, errors = trellis_decode(encoded)
    print(f"Decoded: {len(decoded)} dibits, errors={errors}")
    print(f"Original[0:10]: {list(original_dibits[:10])}")
    print(f"Decoded[0:10]:  {list(decoded[:10])}")

    # Check match
    match_len = min(len(original_dibits), len(decoded))
    match = np.array_equal(original_dibits[:match_len], decoded[:match_len])
    print(f"Match (first {match_len}): {match}")

    return match


def test_interleave_roundtrip():
    """Test interleave/deinterleave roundtrip."""
    print("\n=== Test 2: Interleave Roundtrip ===")

    # Create test data
    original_bits = np.array([i % 2 for i in range(196)], dtype=np.uint8)

    # Interleave
    interleaved = interleave_data(original_bits)
    print(f"Original[0:20]: {list(original_bits[:20])}")
    print(f"Interleaved[0:20]: {list(interleaved[:20])}")

    # Deinterleave
    deinterleaved = deinterleave_data(interleaved)
    print(f"Deinterleaved[0:20]: {list(deinterleaved[:20])}")

    # Check match
    match = np.array_equal(original_bits, deinterleaved)
    print(f"Match: {match}")

    return match


def test_full_tsbk_chain():
    """Test full TSBK encoding and decoding chain."""
    print("\n=== Test 3: Full TSBK Chain ===")

    # Create a test TSBK message (96 bits = 80 data + 16 CRC)
    # Structure: LB(1) + Protect(1) + Opcode(6) + MFID(8) + Data(64) + CRC(16)
    tsbk_bits = np.zeros(96, dtype=np.uint8)

    # LB = 1 (last block)
    tsbk_bits[0] = 1
    # Protect = 0
    tsbk_bits[1] = 0
    # Opcode = 0x3C (IDEN_UP_VU - known opcode)
    opcode = 0x3C
    for i in range(6):
        tsbk_bits[2 + i] = (opcode >> (5 - i)) & 1
    # MFID = 0x00 (standard)
    mfid = 0x00
    for i in range(8):
        tsbk_bits[8 + i] = (mfid >> (7 - i)) & 1
    # Data = test pattern
    for i in range(64):
        tsbk_bits[16 + i] = i % 2

    # Compute and append CRC
    crc = compute_crc16_ccitt(tsbk_bits[:80])
    print(f"Computed CRC: 0x{crc:04X}")
    for i in range(16):
        tsbk_bits[80 + i] = (crc >> (15 - i)) & 1

    print(f"TSBK message: LB=1, opcode=0x{opcode:02X}, mfid=0x{mfid:02X}")
    print(f"TSBK bits[0:20]: {list(tsbk_bits[:20])}")

    # Verify CRC before encoding
    crc_ok, _ = crc16_ccitt_p25(tsbk_bits)
    print(f"CRC valid before encoding: {crc_ok}")

    # Step 1: Convert 96 bits to 48 dibits for trellis encoding
    input_dibits = bits_to_dibits(tsbk_bits)
    print(f"Input dibits (first 10): {list(input_dibits[:10])}")

    # Step 2: Trellis encode (48 dibits -> 96 dibits)
    encoded_dibits = trellis_encode(input_dibits)
    print(f"Trellis encoded: {len(encoded_dibits)} dibits")

    # Step 3: Convert to bits for interleaving (96 dibits -> 192 bits)
    # But we need 196 bits for interleaving - P25 adds 2 "flush" dibits
    encoded_dibits_with_flush = np.zeros(98, dtype=np.uint8)
    encoded_dibits_with_flush[:len(encoded_dibits)] = encoded_dibits

    encoded_bits = dibits_to_bits(encoded_dibits_with_flush)
    print(f"Encoded bits for interleave: {len(encoded_bits)}")

    # Step 4: Interleave 196 bits
    interleaved_bits = interleave_data(encoded_bits)
    print(f"Interleaved bits: {len(interleaved_bits)}")

    # Now simulate receiving: convert back to dibits
    received_dibits = bits_to_dibits(interleaved_bits)
    print(f"Received dibits: {len(received_dibits)}")

    # === DECODING (our code path) ===
    print("\n--- Decoding ---")

    # Step 1: Convert received dibits to bits
    received_bits = dibits_to_bits(received_dibits)
    print(f"Received bits: {len(received_bits)}")

    # Step 2: Deinterleave
    deinterleaved_bits = deinterleave_data(received_bits)
    print(f"Deinterleaved bits: {len(deinterleaved_bits)}")

    # Step 3: Convert to dibits for trellis decoder
    trellis_input = bits_to_dibits(deinterleaved_bits)
    print(f"Trellis input: {len(trellis_input)} dibits")

    # Step 4: Trellis decode
    decoded_dibits, error_metric = trellis_decode(trellis_input)
    print(f"Trellis decoded: {len(decoded_dibits)} dibits, error_metric={error_metric}")

    # Step 5: Convert decoded dibits to bits (need 96 bits from 48 dibits)
    decoded_bits = np.zeros(96, dtype=np.uint8)
    for i in range(min(48, len(decoded_dibits))):
        decoded_bits[i*2] = (decoded_dibits[i] >> 1) & 1
        decoded_bits[i*2 + 1] = decoded_dibits[i] & 1

    print(f"Decoded TSBK bits[0:20]: {list(decoded_bits[:20])}")
    print(f"Original bits[0:20]:     {list(tsbk_bits[:20])}")

    # Check CRC
    crc_valid, crc_errors = crc16_ccitt_p25(decoded_bits)
    print(f"CRC valid: {crc_valid}, errors: {crc_errors}")

    # Extract fields
    lb = bool(decoded_bits[0])
    opcode_decoded = bits_to_int(decoded_bits, 2, 6)
    mfid_decoded = bits_to_int(decoded_bits, 8, 8)

    print(f"Decoded: LB={lb}, opcode=0x{opcode_decoded:02X}, mfid=0x{mfid_decoded:02X}")
    print(f"Expected: LB=True, opcode=0x{opcode:02X}, mfid=0x{mfid:02X}")

    # Compare original vs decoded
    bit_errors = np.sum(tsbk_bits != decoded_bits)
    print(f"Bit errors: {bit_errors} / 96")

    return crc_valid


def test_dibit_mapping():
    """Test dibit to bit and back mapping."""
    print("\n=== Test 4: Dibit Mapping ===")

    # Test all 4 dibits
    for dibit in range(4):
        bits = np.zeros(2, dtype=np.uint8)
        bits[0] = (dibit >> 1) & 1
        bits[1] = dibit & 1

        # Convert back
        recovered = (bits[0] << 1) | bits[1]

        print(f"Dibit {dibit} -> bits [{bits[0]}, {bits[1]}] -> recovered {recovered}")
        assert dibit == recovered

    return True


if __name__ == "__main__":
    print("=== TSBK Chain Diagnostics ===")

    test_dibit_mapping()
    test_trellis_roundtrip()
    test_interleave_roundtrip()
    test_full_tsbk_chain()
