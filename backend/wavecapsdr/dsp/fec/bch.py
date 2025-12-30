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
from typing import Any, Callable, TypeVar, cast

import numpy as np

F = TypeVar("F", bound=Callable[..., Any])

def _fallback_jit(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        return func
    return decorator

# Try to import numba for JIT compilation
try:
    from numba import jit as _numba_jit
    NUMBA_AVAILABLE = True
except ImportError:
    _numba_jit = cast(Callable[..., Any], _fallback_jit)
    NUMBA_AVAILABLE = False

jit = _numba_jit

logger = logging.getLogger(__name__)


# JIT-compiled Berlekamp-Massey algorithm for BCH decoding
# This is the main performance bottleneck in BCH decode
@jit(nopython=True, cache=True)
def _berlekamp_massey_jit(
    syndromes: np.ndarray,
    a_pow_tab: np.ndarray,
    a_log_tab: np.ndarray,
    T: int,
    N: int,
) -> tuple[np.ndarray, int]:
    """JIT-compiled Berlekamp-Massey algorithm.

    Finds the error locator polynomial from syndromes.

    Args:
        syndromes: Syndrome array (2*T elements)
        a_pow_tab: GF power table (alpha^i)
        a_log_tab: GF log table (log_alpha(x))
        T: Error correction capacity (11 for BCH(63,16,23))
        N: Field size minus 1 (63 for GF(2^6))

    Returns:
        Tuple of (error locator polynomial coefficients, degree)
    """
    # Initialize polynomials
    C = np.zeros(T + 1, dtype=np.int32)
    B = np.zeros(T + 1, dtype=np.int32)
    C[0] = 1
    B[0] = 1
    L = 0
    m = 1
    b = 1
    log_b = 0

    for n in range(2 * T):
        # Compute discrepancy
        d = syndromes[n]

        # Add contribution from existing terms
        upper = L + 1
        if upper > T + 1:
            upper = T + 1

        for i in range(1, upper):
            c_val = C[i]
            if c_val != 0 and n >= i:
                s_val = syndromes[n - i]
                if s_val != 0:
                    d ^= a_pow_tab[(a_log_tab[c_val] + a_log_tab[s_val]) % N]

        if d == 0:
            m += 1
        else:
            # Save C for potential swap
            T_poly = C.copy()

            # C(x) = C(x) - (d/b) * x^m * B(x)
            log_d = a_log_tab[d]
            log_db = (log_d + N - log_b) % N

            # Update C polynomial
            limit = T + 1 - m
            for i in range(limit):
                b_val = B[i]
                if b_val != 0:
                    product = a_pow_tab[(a_log_tab[b_val] + log_db) % N]
                    C[i + m] ^= product

            if n >= 2 * L:
                L = n + 1 - L
                B = T_poly
                b = d
                log_b = log_d
                m = 1
            else:
                m += 1

    return C, L


# JIT-compiled Chien search for finding polynomial roots
@jit(nopython=True, cache=True)
def _chien_search_jit(
    poly: np.ndarray,
    degree: int,
    a_pow_tab: np.ndarray,
    a_log_tab: np.ndarray,
    N: int,
) -> np.ndarray:
    """JIT-compiled Chien search for finding polynomial roots.

    Evaluates the polynomial at all field elements to find roots.

    Args:
        poly: Error locator polynomial coefficients
        degree: Polynomial degree
        a_pow_tab: GF power table
        a_log_tab: GF log table
        N: Field size minus 1 (63)

    Returns:
        Array of error positions
    """
    # Collect non-zero coefficients
    nonzero_count = 0
    for i in range(degree + 1):
        if poly[i] != 0:
            nonzero_count += 1

    if nonzero_count == 0:
        return np.empty(0, dtype=np.int32)

    # Store non-zero indices and their log values
    nonzero_indices = np.empty(nonzero_count, dtype=np.int32)
    log_coeffs = np.empty(nonzero_count, dtype=np.int32)
    idx = 0
    for i in range(degree + 1):
        if poly[i] != 0:
            nonzero_indices[idx] = i
            log_coeffs[idx] = a_log_tab[poly[i]]
            idx += 1

    # Find roots by evaluating at all field elements
    roots = np.empty(degree, dtype=np.int32)
    root_count = 0

    for i in range(N):
        # Evaluate polynomial at alpha^i
        val = 0
        for j in range(nonzero_count):
            power_idx = (log_coeffs[j] + i * nonzero_indices[j]) % N
            val ^= a_pow_tab[power_idx]

        if val == 0:
            # Found a root - convert to error position
            error_pos = (N - i) % N
            roots[root_count] = error_pos
            root_count += 1
            if root_count >= degree:
                break

    return roots[:root_count]


# JIT-compiled syndrome computation
@jit(nopython=True, cache=True)
def _compute_syndromes_jit(
    msg: np.ndarray,
    a_pow_tab: np.ndarray,
    T: int,
    N: int,
) -> np.ndarray:
    """JIT-compiled syndrome computation.

    Args:
        msg: Binary message (N bits)
        a_pow_tab: GF power table
        T: Error correction capacity
        N: Field size minus 1

    Returns:
        Syndrome array (2*T syndromes)
    """
    n_syndromes = 2 * T
    syndromes = np.zeros(n_syndromes, dtype=np.int32)
    n_minus_1 = N - 1

    for bit_pos in range(N):
        if msg[bit_pos] != 0:
            for s in range(n_syndromes):
                power_idx = ((s + 1) * (n_minus_1 - bit_pos)) % N
                syndromes[s] ^= a_pow_tab[power_idx]

    return syndromes


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

    def __init__(self) -> None:
        """Initialize BCH decoder with lookup tables."""
        self._init_galois_tables()
        self._init_degree2_table()

    def _init_galois_tables(self) -> None:
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

    def _init_degree2_table(self) -> None:
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
        return int(self.a_pow_tab[i % self.N])

    def _a_log(self, x: int) -> int:
        """Get log_alpha(x) from log table."""
        if x == 0:
            return 0
        return int(self.a_log_tab[x])

    def _gf_sqr(self, x: int) -> int:
        """Square x in GF(2^M)."""
        if x == 0:
            return 0
        return int(self.a_pow_tab[(2 * self.a_log_tab[x]) % self.N])

    def _gf_mul(self, a: int, b: int) -> int:
        """Multiply two GF elements."""
        if a == 0 or b == 0:
            return 0
        return int(self.a_pow_tab[(self.a_log_tab[a] + self.a_log_tab[b]) % self.N])

    def _gf_mul_vec(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Vectorized GF multiply for arrays."""
        result = np.zeros_like(a)
        nonzero_mask = (a != 0) & (b != 0)
        if np.any(nonzero_mask):
            log_sum = (self.a_log_tab[a[nonzero_mask]] + self.a_log_tab[b[nonzero_mask]]) % self.N
            result[nonzero_mask] = self.a_pow_tab[log_sum]
        return result

    def _compute_syndromes(self, msg: np.ndarray) -> np.ndarray:
        """Compute syndromes for error detection.

        OPTIMIZED: Uses JIT-compiled function for ~10x speedup.

        Args:
            msg: Binary message (63 bits)

        Returns:
            Syndrome array (2*T syndromes)
        """
        return _compute_syndromes_jit(
            msg[:self.N].astype(np.int32),
            self.a_pow_tab,
            self.T,
            self.N,
        )

    def _compute_syndromes_vectorized(self, msg: np.ndarray) -> np.ndarray:
        """Vectorized syndrome computation using precomputed power matrix.

        This is ~10x faster than the scalar version by:
        1. Precomputing all power indices
        2. Using numpy advanced indexing
        3. Using numpy reduce for XOR accumulation
        """
        # Find non-zero bit positions (only these contribute to syndromes)
        nonzero_positions = np.nonzero(msg[:self.N])[0]

        if len(nonzero_positions) == 0:
            # No set bits = no syndromes
            return np.zeros(2 * self.T, dtype=np.int32)

        n_syndromes = 2 * self.T
        n_minus_1 = self.N - 1

        # Precompute power indices for all syndrome/position combinations
        # power_idx[i, j] = (i + 1) * (n_minus_1 - positions[j]) % N
        syndrome_multipliers = np.arange(1, n_syndromes + 1, dtype=np.int32)[:, np.newaxis]
        position_values = (n_minus_1 - nonzero_positions).astype(np.int32)[np.newaxis, :]

        power_indices = (syndrome_multipliers * position_values) % self.N

        # Look up power values and XOR reduce
        powers = self.a_pow_tab[power_indices]

        # XOR reduce along axis 1 (positions)
        syndromes = np.bitwise_xor.reduce(powers, axis=1)

        return np.asarray(syndromes, dtype=np.int32)

    def _find_error_locator_poly(self, syndromes: np.ndarray) -> tuple[np.ndarray, int]:
        """Find error locator polynomial using Berlekamp-Massey algorithm.

        OPTIMIZED: Uses JIT-compiled function for ~20x speedup.

        Args:
            syndromes: Syndrome array

        Returns:
            Tuple of (error locator polynomial coefficients, degree)
        """
        return _berlekamp_massey_jit(
            syndromes.astype(np.int32),
            self.a_pow_tab,
            self.a_log_tab,
            self.T,
            self.N,
        )

    def _berlekamp_massey_optimized(self, syndromes: np.ndarray) -> tuple[np.ndarray, int]:
        """Optimized Berlekamp-Massey with minimal Python/numpy overhead.

        Uses direct array operations and avoids numpy.any() which has
        significant overhead for small arrays.
        """
        T = self.T
        N = self.N
        a_pow = self.a_pow_tab
        a_log = self.a_log_tab

        # Initialize polynomials
        C = np.zeros(T + 1, dtype=np.int32)
        B = np.zeros(T + 1, dtype=np.int32)
        C[0] = 1
        B[0] = 1
        L = 0
        m = 1
        b = 1
        log_b = 0  # log of b

        for n in range(2 * T):
            # Compute discrepancy
            d = int(syndromes[n])

            # Compute discrepancy contribution from existing terms
            # Avoid numpy operations for small arrays
            for i in range(1, min(L + 1, T + 1)):
                c_val = int(C[i])
                if c_val != 0 and n >= i:
                    s_val = int(syndromes[n - i])
                    if s_val != 0:
                        d ^= a_pow[(a_log[c_val] + a_log[s_val]) % N]

            if d == 0:
                m += 1
            else:
                T_poly = C.copy()

                # C(x) = C(x) - (d/b) * x^m * B(x)
                log_d = a_log[d]
                log_db = (log_d + N - log_b) % N

                # Update C polynomial
                limit = T + 1 - m
                for i in range(limit):
                    b_val = int(B[i])
                    if b_val != 0:
                        product = a_pow[(a_log[b_val] + log_db) % N]
                        C[i + m] ^= product

                if n >= 2 * L:
                    L = n + 1 - L
                    B = T_poly
                    b = d
                    log_b = log_d
                    m = 1
                else:
                    m += 1

        return C, L

    def _gf_inv(self, x: int) -> int:
        """Compute multiplicative inverse in GF."""
        if x == 0:
            return 0
        return int(self.a_pow_tab[(self.N - self.a_log_tab[x]) % self.N])

    def _find_roots_chien_search(self, poly: np.ndarray, degree: int) -> np.ndarray:
        """Find roots of error locator polynomial using Chien search.

        OPTIMIZED: Uses JIT-compiled function for ~10x speedup.

        Args:
            poly: Error locator polynomial coefficients
            degree: Polynomial degree

        Returns:
            Array of error positions
        """
        return _chien_search_jit(
            poly.astype(np.int32),
            degree,
            self.a_pow_tab,
            self.a_log_tab,
            self.N,
        )

    def _find_roots_chien_search_vectorized(self, poly: np.ndarray, degree: int) -> np.ndarray:
        """Vectorized Chien search using numpy.

        Evaluates the polynomial at all field elements simultaneously.
        """
        # Get non-zero coefficient indices and their log values
        nonzero_mask = poly[:degree + 1] != 0
        nonzero_indices = np.where(nonzero_mask)[0]

        if len(nonzero_indices) == 0:
            return np.array([], dtype=np.int32)

        nonzero_coeffs = poly[nonzero_indices]
        log_coeffs = self.a_log_tab[nonzero_coeffs]

        # Evaluate at all field elements: i = 0..N-1
        # For each i, compute sum(alpha^(log[coeff[j]] + i*j) for j in nonzero_indices)
        i_vals = np.arange(self.N, dtype=np.int32)[:, np.newaxis]  # (N, 1)
        j_vals = nonzero_indices[np.newaxis, :]  # (1, num_nonzero)

        # power_indices[i, j] = (log_coeffs[j] + i * j_vals[j]) % N
        power_indices = (log_coeffs[np.newaxis, :] + i_vals * j_vals) % self.N

        # Look up power values
        powers = self.a_pow_tab[power_indices]

        # XOR reduce along axis 1 to get polynomial value at each field element
        poly_vals = np.bitwise_xor.reduce(powers, axis=1)

        # Find roots (where polynomial value is 0)
        root_indices = np.where(poly_vals == 0)[0]

        # Convert to error positions
        # Root at alpha^i corresponds to error at position (N-i) mod N
        error_positions = (self.N - root_indices) % self.N

        return error_positions.astype(np.int32)

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

        # Invert error positions to match SDRTrunk's coordinate system
        # Because we use reversed bit indexing in syndrome computation,
        # the error positions need to be inverted to reference correct message indices
        error_positions = np.array([(self.N - 1 - pos) % self.N for pos in error_positions])

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
