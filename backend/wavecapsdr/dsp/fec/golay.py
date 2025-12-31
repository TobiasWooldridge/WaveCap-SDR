"""Golay(24,12) Forward Error Correction decoder.

The extended Golay code G(24,12) is used in P25 for protecting critical
header information. It encodes 12 data bits into 24 bits and can:
- Detect up to 4 bit errors
- Correct up to 3 bit errors

The code structure:
- 12 data bits (d0-d11)
- 12 parity bits (p0-p11)
- Generator matrix based on perfect Golay code G(23,12)

P25 uses Golay for:
- NAC (Network Access Code) in NID
- Talkgroup ID in LC
- Message Indicator fields

Reference: TIA-102.BAAA-A Annex A
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
from wavecapsdr.typing import NDArrayAny

logger = logging.getLogger(__name__)

# Golay(24,12) generator matrix (lower triangular parity portion)
# Each row is the parity bits for a single data bit
GOLAY_GENERATOR = np.array(
    [
        [1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1],  # d0
        [0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1],  # d1
        [1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0],  # d2
        [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0],  # d3
        [0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0],  # d4
        [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1],  # d5
        [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1],  # d6
        [1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0],  # d7
        [0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1],  # d8
        [1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0],  # d9
        [0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0],  # d10
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0],  # d11
    ],
    dtype=np.uint8,
)

# Parity check matrix H = [I_12 | P^T]
# Used for syndrome calculation


def golay_encode(data: int) -> int:
    """Encode 12 bits of data using Golay(24,12).

    Args:
        data: 12-bit data value (0-4095)

    Returns:
        24-bit codeword with data in upper 12 bits, parity in lower 12
    """
    data = data & 0xFFF  # Ensure 12 bits

    # Extract data bits
    data_bits = np.array([(data >> (11 - i)) & 1 for i in range(12)], dtype=np.uint8)

    # Calculate parity bits using generator matrix
    parity_bits = np.mod(np.dot(data_bits, GOLAY_GENERATOR), 2)

    # Construct codeword: data || parity
    codeword = data << 12
    for i, p in enumerate(parity_bits):
        codeword |= int(p) << (11 - i)

    return codeword


def golay_syndrome(codeword: int) -> int:
    """Calculate syndrome for Golay(24,12) codeword.

    The syndrome indicates the error pattern. Zero syndrome means no errors
    (or undetectable error pattern).

    Args:
        codeword: 24-bit received codeword

    Returns:
        12-bit syndrome
    """
    # Extract data and parity portions
    data = (codeword >> 12) & 0xFFF
    received_parity = codeword & 0xFFF

    # Recalculate expected parity
    data_bits = np.array([(data >> (11 - i)) & 1 for i in range(12)], dtype=np.uint8)
    expected_parity_bits = np.mod(np.dot(data_bits, GOLAY_GENERATOR), 2)

    # Convert to integer
    expected_parity = 0
    for i, p in enumerate(expected_parity_bits):
        expected_parity |= int(p) << (11 - i)

    # Syndrome is XOR of received and expected parity
    syndrome = received_parity ^ expected_parity

    return syndrome


def _popcount(x: int) -> int:
    """Count number of 1 bits (Hamming weight)."""
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count


def _rotate_right_12(x: int) -> int:
    """Rotate 12-bit value right by 1."""
    return ((x >> 1) | ((x & 1) << 11)) & 0xFFF


def golay_decode(codeword: int) -> tuple[int, int]:
    """Decode Golay(24,12) codeword with error correction.

    Uses the standard Golay decoding algorithm that can correct up to
    3 bit errors. Returns the decoded data and number of errors corrected.

    Algorithm:
    1. Calculate syndrome
    2. If weight(syndrome) <= 3, errors are in parity bits only
    3. Otherwise, search for minimum-weight error pattern in data bits

    The decoder prioritizes finding the minimum total error weight to ensure
    correct decoding when multiple error patterns could produce similar syndromes.

    Args:
        codeword: 24-bit received codeword

    Returns:
        Tuple of (decoded_data, num_errors)
        - decoded_data: 12-bit corrected data (-1 if uncorrectable)
        - num_errors: Number of bit errors corrected (-1 if uncorrectable)
    """
    codeword = codeword & 0xFFFFFF  # Ensure 24 bits

    # Calculate initial syndrome
    syndrome = golay_syndrome(codeword)

    if syndrome == 0:
        # No errors detected
        return (codeword >> 12), 0

    # Check if errors are only in parity bits (weight <= 3)
    syndrome_weight = _popcount(syndrome)
    if syndrome_weight <= 3:
        # All errors in parity portion, data is correct
        return (codeword >> 12), syndrome_weight

    # Pre-compute all generator syndromes for efficiency
    data = (codeword >> 12) & 0xFFF
    gen_syndromes = []
    for i in range(12):
        data_bit_syndrome = 0
        for j, g in enumerate(GOLAY_GENERATOR[i]):
            data_bit_syndrome |= int(g) << (11 - j)
        gen_syndromes.append(data_bit_syndrome)

    # Try single data bit errors - find minimum total weight
    # First pass: look for exact match (syndrome == generator row)
    for i in range(12):
        if syndrome == gen_syndromes[i]:
            # Single data bit error, no parity errors
            corrected_data = data ^ (1 << (11 - i))
            return corrected_data, 1

    # Second pass: look for 1 data + 1 parity (weight 1 residual)
    for i in range(12):
        test_syndrome = syndrome ^ gen_syndromes[i]
        if _popcount(test_syndrome) == 1:
            corrected_data = data ^ (1 << (11 - i))
            return corrected_data, 2

    # Third pass: look for 1 data + 2 parity (weight 2 residual)
    for i in range(12):
        test_syndrome = syndrome ^ gen_syndromes[i]
        if _popcount(test_syndrome) == 2:
            corrected_data = data ^ (1 << (11 - i))
            return corrected_data, 3

    # Try two data bit errors
    # First: look for 2 data + 0 parity
    for i in range(12):
        for j in range(i + 1, 12):
            test_syndrome = syndrome ^ gen_syndromes[i] ^ gen_syndromes[j]
            if test_syndrome == 0:
                corrected_data = data ^ (1 << (11 - i)) ^ (1 << (11 - j))
                return corrected_data, 2

    # Then: look for 2 data + 1 parity
    for i in range(12):
        for j in range(i + 1, 12):
            test_syndrome = syndrome ^ gen_syndromes[i] ^ gen_syndromes[j]
            if _popcount(test_syndrome) == 1:
                corrected_data = data ^ (1 << (11 - i)) ^ (1 << (11 - j))
                return corrected_data, 3

    # Try three data bit errors (no parity errors)
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                test_syndrome = syndrome ^ gen_syndromes[i] ^ gen_syndromes[j] ^ gen_syndromes[k]
                if test_syndrome == 0:
                    corrected_data = data ^ (1 << (11 - i)) ^ (1 << (11 - j)) ^ (1 << (11 - k))
                    return corrected_data, 3

    # Uncorrectable error (more than 3 bit errors)
    logger.warning(f"Golay decode: uncorrectable error, syndrome={syndrome:03x}")
    return -1, -1


def golay_decode_soft(codeword: int, soft_bits: NDArrayAny | None = None) -> tuple[int, int, float]:
    """Decode Golay(24,12) with soft decision information.

    When soft bit reliabilities are available, uses them to improve
    error correction by preferring to flip less reliable bits.

    Args:
        codeword: 24-bit hard decision codeword
        soft_bits: Optional 24-element array of bit reliabilities (0-1)

    Returns:
        Tuple of (decoded_data, num_errors, confidence)
        - confidence: Reliability metric (0-1)
    """
    # First try hard decision decode
    data, errors = golay_decode(codeword)

    if errors >= 0:
        # Hard decode succeeded
        confidence = 1.0 if errors == 0 else max(0.5, 1.0 - errors * 0.2)
        return data, errors, confidence

    # Hard decode failed - try soft decode if reliabilities available
    if soft_bits is None:
        return data, errors, 0.0

    # Find least reliable bits and try flipping them
    # Sort bit positions by reliability (ascending)
    sorted_positions = np.argsort(soft_bits)

    # Try flipping combinations of unreliable bits
    for num_flips in range(1, 5):
        for positions in _combinations(sorted_positions[:8], num_flips):
            test_codeword = codeword
            for pos in positions:
                test_codeword ^= 1 << (23 - pos)

            test_data, test_errors = golay_decode(test_codeword)
            if test_errors >= 0:
                # Found valid codeword
                total_errors = num_flips + test_errors
                confidence = max(0.3, 1.0 - total_errors * 0.15)
                return test_data, total_errors, confidence

    return -1, -1, 0.0


def _combinations(items: NDArrayAny, r: int) -> Iterator[tuple[int, ...]]:
    """Generate r-combinations of items."""
    n = len(items)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(int(items[i]) for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield tuple(int(items[i]) for i in indices)
