"""BCH(63,16,23) Forward Error Correction decoder for P25.

BCH (Bose-Chaudhuri-Hocquenghem) codes are cyclic error-correcting codes.
The P25 standard uses BCH(63,16,23) for protecting the Network ID (NID).

Code parameters:
- n = 63 (codeword length)
- k = 16 (message length: 12-bit NAC + 4-bit DUID)
- t = 11 (error correction capacity)

Based on the Linux kernel BCH implementation and SDRTrunk's Java port:
https://github.com/DSheirer/sdrtrunk/blob/master/src/main/java/io/github/dsheirer/edac/bch/

Reference: TIA-102.BAAA-A Network ID specification
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class BCH_63_16_23:
    """BCH(63,16,23) decoder for P25 Network ID.

    This decoder can correct up to 11 bit errors in a 63-bit codeword.
    The codeword structure is:
    - 16 data bits (12-bit NAC + 4-bit DUID)
    - 47 parity bits (63 - 16)

    Attributes:
        M: Galois Field size (2^M - 1 = 63)
        K: Message size in bits (16)
        T: Error correction capacity (11)
    """

    # BCH code parameters
    M = 6  # GF(2^6)
    N = 63  # Codeword length (2^M - 1)
    K = 16  # Message length
    T = 11  # Error correction capacity

    # Primitive polynomial for GF(2^6): x^6 + x + 1 = 0x43
    PRIMITIVE_POLYNOMIAL = 0x43

    MESSAGE_NOT_CORRECTED = -1

    def __init__(self):
        """Initialize BCH decoder with lookup tables."""
        self._init_galois_tables()
        self._init_degree2_table()

    def _init_galois_tables(self):
        """Initialize Galois Field logarithm and antilog tables."""
        self.a_pow_tab = np.zeros(self.N + 1, dtype=np.int32)
        self.a_log_tab = np.zeros(self.N + 1, dtype=np.int32)

        # Generate power and log tables for GF(2^M)
        x = 1
        for i in range(self.N):
            self.a_pow_tab[i] = x
            self.a_log_tab[x] = i
            x <<= 1
            if x & (1 << self.M):
                x ^= self.PRIMITIVE_POLYNOMIAL

        # Wrap-around for convenience
        self.a_pow_tab[self.N] = 1
        self.a_log_tab[0] = 0

    def _init_degree2_table(self):
        """Initialize lookup table for degree-2 polynomial roots."""
        self.xi_tab = np.zeros(self.M, dtype=np.int32)

        # Find k such that Tr(a^k) = 1 and 0 <= k < M
        ak = 0
        for i in range(self.M):
            trace = 0
            for j in range(self.M):
                trace ^= self._a_pow(i * (1 << j))
            if trace != 0:
                ak = self.a_pow_tab[i]
                break

        # Find xi such that xi^2 + xi = a^i + Tr(a^i) * a^k
        xi_found = [False] * self.M
        remaining = self.M

        for x in range(self.N + 1):
            if remaining == 0:
                break

            y = self._gf_sqr(x) ^ x
            for _ in range(2):
                if y != 0:
                    r = self._a_log(y)
                    if r < self.M and not xi_found[r]:
                        self.xi_tab[r] = x
                        xi_found[r] = True
                        remaining -= 1
                        break
                y ^= ak

    def _a_pow(self, i: int) -> int:
        """Get alpha^i from power table."""
        return self.a_pow_tab[i % self.N]

    def _a_log(self, x: int) -> int:
        """Get log_alpha(x) from log table."""
        if x == 0:
            return 0
        return self.a_log_tab[x]

    def _gf_sqr(self, x: int) -> int:
        """Square x in GF(2^M)."""
        if x == 0:
            return 0
        return self.a_pow_tab[(2 * self.a_log_tab[x]) % self.N]

    def _gf_mul(self, a: int, b: int) -> int:
        """Multiply two GF elements."""
        if a == 0 or b == 0:
            return 0
        return self.a_pow_tab[(self.a_log_tab[a] + self.a_log_tab[b]) % self.N]

    def _compute_syndromes(self, msg: np.ndarray) -> np.ndarray:
        """Compute syndromes for error detection.

        Args:
            msg: Binary message (63 bits)

        Returns:
            Syndrome array (2*T syndromes)
        """
        syndromes = np.zeros(2 * self.T, dtype=np.int32)

        for i in range(2 * self.T):
            s = 0
            for j in range(self.N):
                if j < len(msg) and msg[j]:
                    s ^= self._a_pow((i + 1) * j)
            syndromes[i] = s

        return syndromes

    def _find_error_locator_poly(self, syndromes: np.ndarray) -> tuple[np.ndarray, int]:
        """Find error locator polynomial using Berlekamp-Massey algorithm.

        Args:
            syndromes: Syndrome array

        Returns:
            Tuple of (error locator polynomial coefficients, degree)
        """
        # Berlekamp-Massey algorithm
        C = np.zeros(self.T + 1, dtype=np.int32)
        B = np.zeros(self.T + 1, dtype=np.int32)
        C[0] = 1
        B[0] = 1
        L = 0
        m = 1
        b = 1

        for n in range(2 * self.T):
            # Compute discrepancy
            d = syndromes[n]
            for i in range(1, L + 1):
                if C[i] != 0 and syndromes[n - i] != 0:
                    d ^= self._gf_mul(C[i], syndromes[n - i])

            if d == 0:
                m += 1
            else:
                T_poly = C.copy()

                # C(x) = C(x) - (d/b) * x^m * B(x)
                db = self._gf_mul(d, self._gf_inv(b))
                for i in range(self.T + 1 - m):
                    if B[i] != 0:
                        C[i + m] ^= self._gf_mul(db, B[i])

                if n >= 2 * L:
                    L = n + 1 - L
                    B = T_poly
                    b = d
                    m = 1
                else:
                    m += 1

        return C, L

    def _gf_inv(self, x: int) -> int:
        """Compute multiplicative inverse in GF."""
        if x == 0:
            return 0
        return self.a_pow_tab[(self.N - self.a_log_tab[x]) % self.N]

    def _find_roots_chien_search(self, poly: np.ndarray, degree: int) -> np.ndarray:
        """Find roots of error locator polynomial using Chien search.

        Args:
            poly: Error locator polynomial coefficients
            degree: Polynomial degree

        Returns:
            Array of error positions
        """
        roots = []

        # Chien search: evaluate polynomial at alpha^(-i) for i=0..N-1
        for i in range(self.N):
            val = 0
            for j in range(degree + 1):
                if poly[j] != 0:
                    val ^= self._a_pow((self.a_log_tab[poly[j]] + i * j) % self.N)

            if val == 0:
                # Root found at alpha^(-i), error at position (N-1-i)
                roots.append((self.N - 1 - i) % self.N)

        return np.array(roots, dtype=np.int32)

    def decode(self, codeword: np.ndarray, tracked_nac: int | None = None) -> tuple[int, int]:
        """Decode BCH codeword with error correction.

        This implements a two-pass decoding strategy:
        1. Attempt to decode the codeword as-is
        2. If that fails and a tracked NAC is provided, overwrite the NAC field
           and try again (helps when NAC bits are corrupted)

        Args:
            codeword: Binary array of 63 bits (16 data + 47 parity)
            tracked_nac: Optional tracked NAC value from previous successful decodes

        Returns:
            Tuple of (corrected_data, num_errors_corrected)
            - corrected_data: 16-bit corrected data (NAC + DUID)
            - num_errors: Number of errors corrected, or MESSAGE_NOT_CORRECTED (-1)
        """
        # First pass: try to decode as-is
        data, errors = self._decode_internal(codeword)

        if errors != self.MESSAGE_NOT_CORRECTED:
            return data, errors

        # Second pass: if we have a tracked NAC and first pass failed,
        # overwrite NAC field and try again
        if tracked_nac is not None and tracked_nac > 0:
            # Extract current NAC from first 12 bits
            current_nac = 0
            for i in range(12):
                current_nac = (current_nac << 1) | int(codeword[i])

            if current_nac != tracked_nac:
                # Overwrite NAC field with tracked value
                codeword_copy = codeword.copy()
                for i in range(12):
                    codeword_copy[i] = (tracked_nac >> (11 - i)) & 1

                data, errors = self._decode_internal(codeword_copy)
                return data, errors

        return data, errors

    def _decode_internal(self, codeword: np.ndarray) -> tuple[int, int]:
        """Internal decode implementation.

        Args:
            codeword: 63-bit binary array

        Returns:
            Tuple of (corrected_data, num_errors)
        """
        if len(codeword) < self.N:
            logger.warning(f"BCH decode: codeword too short ({len(codeword)} < {self.N})")
            return 0, self.MESSAGE_NOT_CORRECTED

        # Compute syndromes
        syndromes = self._compute_syndromes(codeword[:self.N])

        # Check if all syndromes are zero (no errors)
        if np.all(syndromes == 0):
            # Extract data bits (first K bits)
            data = 0
            for i in range(self.K):
                data = (data << 1) | int(codeword[i])
            return data, 0

        # Find error locator polynomial
        err_loc_poly, degree = self._find_error_locator_poly(syndromes)

        if degree == 0 or degree > self.T:
            # Too many errors or polynomial not found
            logger.debug(f"BCH decode: uncorrectable (degree={degree})")
            return 0, self.MESSAGE_NOT_CORRECTED

        # Find error positions
        error_positions = self._find_roots_chien_search(err_loc_poly, degree)

        if len(error_positions) != degree:
            # Could not find all roots
            logger.debug(f"BCH decode: root count mismatch ({len(error_positions)} != {degree})")
            return 0, self.MESSAGE_NOT_CORRECTED

        # Correct errors
        corrected = codeword.copy()
        for pos in error_positions:
            if pos < self.N:
                corrected[pos] ^= 1

        # Verify correction by recomputing syndromes
        verify_syndromes = self._compute_syndromes(corrected[:self.N])
        if not np.all(verify_syndromes == 0):
            logger.debug("BCH decode: correction verification failed")
            return 0, self.MESSAGE_NOT_CORRECTED

        # Extract corrected data bits
        data = 0
        for i in range(self.K):
            data = (data << 1) | int(corrected[i])

        return data, len(error_positions)


# Global decoder instance (reused for efficiency)
_bch_decoder: BCH_63_16_23 | None = None


def bch_decode(codeword: np.ndarray, tracked_nac: int | None = None) -> tuple[int, int]:
    """Decode P25 NID using BCH(63,16,23) error correction.

    Args:
        codeword: 63-bit binary array (16 data + 47 parity)
        tracked_nac: Optional previously observed NAC for second-pass correction

    Returns:
        Tuple of (decoded_16bit_data, num_errors_corrected)
    """
    global _bch_decoder
    if _bch_decoder is None:
        _bch_decoder = BCH_63_16_23()

    return _bch_decoder.decode(codeword, tracked_nac)
