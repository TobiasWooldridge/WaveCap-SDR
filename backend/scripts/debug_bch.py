#!/usr/bin/env python3
"""Debug BCH decoder with known-good test vectors and actual signal data."""

import sys
import importlib.util
import numpy as np
from pathlib import Path

# Direct file import to avoid FastAPI dependency
backend_path = Path(__file__).parent.parent

def import_module_from_file(name: str, path: Path):
    """Import a module directly from file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

bch_module = import_module_from_file(
    'bch',
    backend_path / 'wavecapsdr' / 'dsp' / 'fec' / 'bch.py'
)
BCH_63_16_23 = bch_module.BCH_63_16_23
bch_decode = bch_module.bch_decode


def test_known_good_nid():
    """Test BCH with a known-good NID from P25 spec."""
    # Create BCH decoder
    bch = BCH_63_16_23()

    # P25 NID BCH(63,16) generator matrix from SDRTrunk
    # Each row is a 63-bit codeword for the corresponding data bit
    P25_NID_GENERATOR = [
        int("6331141367235452", 8),  # bit 0
        int("5265521614723276", 8),  # bit 1
        int("4603711461164164", 8),  # bit 2
        int("2301744630472072", 8),  # bit 3
        int("7271623073000466", 8),  # bit 4
        int("5605650752635660", 8),  # bit 5
        int("2702724365316730", 8),  # bit 6
        int("1341352172547354", 8),  # bit 7
        int("0560565075263566", 8),  # bit 8
        int("6141333751704220", 8),  # bit 9
        int("3060555764742110", 8),  # bit 10
        int("1430266772361044", 8),  # bit 11
        int("0614133375170422", 8),  # bit 12
        int("6037114611641642", 8),  # bit 13
        int("5326507063515373", 8),  # bit 14
        int("4662302756473127", 8),  # bit 15
    ]

    def create_valid_nid(nac: int, duid: int) -> np.ndarray:
        """Create a valid P25 NID codeword with correct parity."""
        data = (nac << 4) | (duid & 0xF)

        # Compute parity using generator matrix
        # Generator matrix produces 48-bit parity, we use bits 0-46 (47 bits)
        parity = 0
        for i in range(16):
            if (data >> (15 - i)) & 1:
                parity ^= P25_NID_GENERATOR[i]

        # Build 63-bit codeword: 16 data bits + 47 parity bits
        # Parity is stored MSB first, and we skip the LSB (bit 63 of 64-bit NID)
        codeword = np.zeros(63, dtype=np.uint8)
        for i in range(16):
            codeword[i] = (data >> (15 - i)) & 1
        for i in range(47):
            # Parity bits are in positions 47-1 of the 48-bit parity field (skip bit 0)
            codeword[16 + i] = (parity >> (47 - i)) & 1

        return codeword

    print("\n=== Testing with valid P25 NID codewords ===")

    test_cases = [
        (0x293, 0x0, "HDU"),    # NAC=0x293, DUID=HDU
        (0x293, 0x3, "TDU"),    # NAC=0x293, DUID=TDU
        (0x293, 0x5, "LDU1"),   # NAC=0x293, DUID=LDU1
        (0x293, 0x7, "TSBK"),   # NAC=0x293, DUID=TSBK
        (0x293, 0xA, "LDU2"),   # NAC=0x293, DUID=LDU2
        (0x001, 0x7, "Default NAC"),  # NAC=1, DUID=TSBK
    ]

    for nac, duid, name in test_cases:
        codeword = create_valid_nid(nac, duid)

        # Test with no errors
        syn = bch._compute_syndromes(codeword)
        all_zero = np.all(syn == 0)
        print(f"\n{name} (NAC=0x{nac:03X}, DUID=0x{duid:X}):")
        print(f"  Zero syndromes (no errors): {all_zero}")

        if all_zero:
            # Test decode
            data, errors = bch.decode(codeword.copy())
            decoded_nac = (data >> 4) & 0xFFF
            decoded_duid = data & 0xF
            print(f"  Decoded: NAC=0x{decoded_nac:03X}, DUID=0x{decoded_duid:X}, errors={errors}")

        # Test with 5 bit errors
        codeword_with_errors = codeword.copy()
        for i in [0, 10, 20, 30, 40]:
            codeword_with_errors[i] ^= 1

        data, errors = bch.decode(codeword_with_errors.copy())
        if errors >= 0:
            decoded_nac = (data >> 4) & 0xFFF
            decoded_duid = data & 0xF
            print(f"  With 5 errors: NAC=0x{decoded_nac:03X}, DUID=0x{decoded_duid:X}, corrected={errors}")
        else:
            print(f"  With 5 errors: DECODE FAILED (errors={errors})")

    return True


def test_syndrome_calculation():
    """Test syndrome calculation matches SDRTrunk's approach."""
    bch = BCH_63_16_23()

    # Test with a single bit set
    for bit_pos in [0, 1, 15, 31, 62]:
        msg = np.zeros(63, dtype=np.uint8)
        msg[bit_pos] = 1

        syn = bch._compute_syndromes(msg)

        # Expected: S_j = alpha^(j * (N-1-bit_pos)) for each syndrome j=1..2T
        expected = []
        for j in range(22):  # 2*T = 22
            exp = bch._a_pow((j + 1) * (62 - bit_pos))
            expected.append(exp)

        match = np.array_equal(syn, expected)
        print(f"Bit {bit_pos:2d}: syndromes match expected = {match}")
        if not match:
            print(f"  Got:      {syn[:6]}...")
            print(f"  Expected: {expected[:6]}...")


def test_berlekamp_massey():
    """Test Berlekamp-Massey with known error pattern."""
    bch = BCH_63_16_23()

    # Create message with known single-bit error
    msg = np.zeros(63, dtype=np.uint8)
    msg[10] = 1  # Single bit error at position 10

    syn = bch._compute_syndromes(msg)
    print(f"\nSingle error at position 10:")
    print(f"  Syndromes[0:6]: {syn[:6]}")

    poly, degree = bch._find_error_locator_poly(syn)
    print(f"  Error locator poly degree: {degree}")
    print(f"  Poly coefficients: {poly[:degree+1]}")

    # For single error, degree should be 1
    if degree == 1:
        roots = bch._find_roots_chien_search(poly, degree)
        print(f"  Roots found: {roots}")
        # Root should correspond to position 10
        expected_root = (63 - 1 - 10) % 63  # SDRTrunk inverts positions
        print(f"  Expected root (before inversion): ~{expected_root}")


def test_with_zeros():
    """Test that all-zero message has zero syndromes."""
    bch = BCH_63_16_23()

    msg = np.zeros(63, dtype=np.uint8)
    syn = bch._compute_syndromes(msg)

    all_zero = np.all(syn == 0)
    print(f"\nAll-zero message syndromes all zero: {all_zero}")
    if not all_zero:
        print(f"  Non-zero syndromes: {syn}")


def test_multi_error():
    """Test BCH with multiple errors."""
    bch = BCH_63_16_23()

    print("\n=== Testing multiple errors ===")

    for num_errors in [2, 3, 5, 11]:
        # Create message with known errors
        msg = np.zeros(63, dtype=np.uint8)
        error_positions = list(range(num_errors))  # Errors at positions 0, 1, 2, ...
        for pos in error_positions:
            msg[pos] = 1

        syn = bch._compute_syndromes(msg)
        poly, degree = bch._find_error_locator_poly(syn)

        print(f"\n{num_errors} errors at positions {error_positions}:")
        print(f"  ELP degree: {degree}")

        if degree > 0:
            roots = bch._find_roots_chien_search(poly, degree)
            print(f"  Roots found: {len(roots)} (expected {num_errors})")
            if len(roots) == degree:
                # Invert roots to get actual positions
                corrected_positions = [(63 - 1 - r) % 63 for r in roots]
                print(f"  Corrected positions: {sorted(corrected_positions)}")
            else:
                print(f"  Root count mismatch! Roots: {roots}")


def analyze_captured_nid():
    """Analyze NID from captured signal (if available)."""
    print("\n=== Analyzing actual NID patterns ===")

    # Simulate what we might see from demod
    # A typical P25 NID has structure:
    # - 12 bits NAC (Network Access Code)
    # - 4 bits DUID (Data Unit ID)
    # - 47 bits parity

    # Common NAC values: 0x293 (default), etc.
    # DUID values: 0x0 (HDU), 0x3 (TDU), 0x5 (LDU1), 0x7 (TSBK), etc.

    print("If dibits are being converted incorrectly, the NAC/DUID extraction will fail.")
    print("The BCH decoder should find 0-11 errors for valid messages.")
    print("Root count mismatch means either:")
    print("  1. Too many bit errors (>11)")
    print("  2. Bit ordering mismatch")
    print("  3. Wrong polynomial/algorithm")


def main():
    print("=" * 60)
    print("BCH Decoder Debug")
    print("=" * 60)

    test_with_zeros()
    test_syndrome_calculation()
    test_berlekamp_massey()
    test_multi_error()
    test_known_good_nid()
    analyze_captured_nid()


if __name__ == '__main__':
    main()
