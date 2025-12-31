"""Trellis (Viterbi) decoder for P25.

P25 uses a 1/2 rate trellis-coded modulation (TCM) for voice frame data.
The trellis encoder uses a 4-state trellis with constraint length K=5.

The encoder structure:
- Input: 2 bits at a time (dibits)
- Output: 4 bits (encoded dibits that map to C4FM symbols)
- 4 states (2-bit state register)

The Viterbi decoder finds the maximum likelihood path through the trellis
using soft or hard decisions.

P25 uses trellis coding in:
- Voice LDU frames (IMBE data protection)
- Link Control data
- Some packet data

Reference: TIA-102.BAAA-A Annex E
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from wavecapsdr.typing import NDArrayAny

logger = logging.getLogger(__name__)

# P25 trellis encoder state transition table (from SDRTrunk P25_1_2_Node.java)
# For each state (0-3) and input dibit (0-3), gives (next_state, output_dibit_pair)
# Output is the 4-bit constellation point selection (2 dibits)
#
# SDRTrunk transition matrix (nibbles):
#   State 0: {2,12,1,15}   -> {(0,2),(3,0),(0,1),(3,3)}
#   State 1: {14,0,13,3}   -> {(3,2),(0,0),(3,1),(0,3)}
#   State 2: {9,7,10,4}    -> {(2,1),(1,3),(2,2),(1,0)}
#   State 3: {5,11,6,8}    -> {(1,1),(2,3),(1,2),(2,0)}
#
# Next state calculation: state transitions follow standard 2-bit shift register
TRELLIS_ENCODER = {
    # State 0: inputs 0,1,2,3 produce outputs (0,2),(3,0),(0,1),(3,3)
    (0, 0): (0, (0, 2)),
    (0, 1): (1, (3, 0)),
    (0, 2): (2, (0, 1)),
    (0, 3): (3, (3, 3)),
    # State 1: inputs 0,1,2,3 produce outputs (3,2),(0,0),(3,1),(0,3)
    (1, 0): (0, (3, 2)),
    (1, 1): (1, (0, 0)),
    (1, 2): (2, (3, 1)),
    (1, 3): (3, (0, 3)),
    # State 2: inputs 0,1,2,3 produce outputs (2,1),(1,3),(2,2),(1,0)
    (2, 0): (0, (2, 1)),
    (2, 1): (1, (1, 3)),
    (2, 2): (2, (2, 2)),
    (2, 3): (3, (1, 0)),
    # State 3: inputs 0,1,2,3 produce outputs (1,1),(2,3),(1,2),(2,0)
    (3, 0): (0, (1, 1)),
    (3, 1): (1, (2, 3)),
    (3, 2): (2, (1, 2)),
    (3, 3): (3, (2, 0)),
}

# Build reverse lookup: (state, output_pair) -> (prev_state, input_dibit)
TRELLIS_DECODER_TABLE = {}
for (state, inp), (next_state, output) in TRELLIS_ENCODER.items():
    TRELLIS_DECODER_TABLE[(next_state, output)] = (state, inp)

# Constellation mapping (C4FM dibit to soft value)
CONSTELLATION = {
    0: 1.0,  # +1
    1: 3.0,  # +3
    2: -1.0,  # -1
    3: -3.0,  # -3
}


@dataclass
class TrellisPath:
    """A path through the trellis for Viterbi decoding."""

    state: int
    metric: float
    decoded_bits: list[int]


class TrellisDecoder:
    """Viterbi decoder for P25 1/2 rate trellis code.

    This implements soft-decision Viterbi decoding for the P25 trellis code.
    The decoder maintains 4 survivor paths (one per state) and updates them
    as new symbols arrive.

    Features:
    - Soft-decision decoding for improved performance
    - Hard-decision fallback when soft info unavailable
    - Traceback for decoded bit recovery

    Example usage:
        decoder = TrellisDecoder()
        decoded = decoder.decode(dibits)
    """

    NUM_STATES = 4
    TRACEBACK_DEPTH = 12  # Number of symbols before making decision

    def __init__(self) -> None:
        """Initialize trellis decoder."""
        self.reset()

    def reset(self) -> None:
        """Reset decoder state."""
        # Initialize paths - start in state 0
        self._paths = [
            TrellisPath(state=i, metric=0.0 if i == 0 else float("inf"), decoded_bits=[])
            for i in range(self.NUM_STATES)
        ]
        self._symbol_count = 0

    def _branch_metric(
        self,
        received: tuple[int, int],
        expected: tuple[int, int],
        soft: tuple[float, float] | None = None,
    ) -> float:
        """Calculate branch metric between received and expected symbol pair.

        Args:
            received: Received dibit pair (hard decision)
            expected: Expected dibit pair from trellis
            soft: Optional soft values for received symbols

        Returns:
            Branch metric (lower = better match)
        """
        if soft is not None:
            # Soft-decision: Euclidean distance
            exp_soft = (CONSTELLATION[expected[0]], CONSTELLATION[expected[1]])
            metric = (soft[0] - exp_soft[0]) ** 2 + (soft[1] - exp_soft[1]) ** 2
        else:
            # Hard-decision: Hamming distance counting actual bit errors
            # Each dibit is 2 bits, so XOR and count set bits (0-2 per dibit, 0-4 total)
            # This matches OP25's count_bits(codeword ^ next_words[state][j])
            xor0 = received[0] ^ expected[0]
            xor1 = received[1] ^ expected[1]
            metric = float(bin(xor0).count("1") + bin(xor1).count("1"))

        return metric

    def decode_step(
        self,
        received: tuple[int, int],
        soft: tuple[float, float] | None = None,
    ) -> int | None:
        """Process one trellis step (2 dibits in, 1 dibit out).

        Args:
            received: Received dibit pair
            soft: Optional soft symbol values

        Returns:
            Decoded dibit (or None if still filling traceback)
        """
        # Calculate metrics for all transitions to each state
        new_paths = []

        for next_state in range(self.NUM_STATES):
            best_metric = float("inf")
            best_prev_state = 0
            best_input = 0

            # Check all possible transitions to this state
            for input_dibit in range(4):
                for prev_state in range(self.NUM_STATES):
                    key = (prev_state, input_dibit)
                    if key not in TRELLIS_ENCODER:
                        continue

                    ns, output = TRELLIS_ENCODER[key]
                    if ns != next_state:
                        continue

                    # Calculate branch metric
                    branch = self._branch_metric(received, output, soft)
                    path_metric = self._paths[prev_state].metric + branch

                    if path_metric < best_metric:
                        best_metric = path_metric
                        best_prev_state = prev_state
                        best_input = input_dibit

            # Create new path for this state
            prev_path = self._paths[best_prev_state]
            new_bits = [*prev_path.decoded_bits, best_input]
            new_paths.append(
                TrellisPath(state=next_state, metric=best_metric, decoded_bits=new_bits)
            )

        self._paths = new_paths
        self._symbol_count += 1

        # Output decoded bit if we have enough traceback depth
        if self._symbol_count >= self.TRACEBACK_DEPTH:
            # Find best path
            best_path = min(self._paths, key=lambda p: p.metric)
            output_idx = len(best_path.decoded_bits) - self.TRACEBACK_DEPTH
            if output_idx >= 0:
                return best_path.decoded_bits[output_idx]

        return None

    def decode(
        self,
        dibits: NDArrayAny,
        soft_values: NDArrayAny | None = None,
        debug: bool = False,
    ) -> tuple[NDArrayAny, int]:
        """Decode a block of trellis-encoded dibits.

        Args:
            dibits: Array of dibits (must be even length)
            soft_values: Optional soft values for each dibit
            debug: If True, log detailed debug information

        Returns:
            Tuple of (decoded_dibits, num_errors)
        """
        if len(dibits) % 2 != 0:
            logger.warning("Trellis decode: odd number of dibits, truncating")
            dibits = dibits[:-1]
            if soft_values is not None:
                soft_values = soft_values[:-1]

        if debug:
            dibit_str = " ".join(str(d) for d in dibits[:20])
            logger.info(f"Trellis decode: input {len(dibits)} dibits, first 20: {dibit_str}")
            if soft_values is not None:
                soft_str = " ".join(f"{s:.1f}" for s in soft_values[:20])
                logger.info(f"Trellis decode: soft values first 20: {soft_str}")

        self.reset()
        decoded = []

        # Process pairs of dibits
        for i in range(0, len(dibits), 2):
            received = (int(dibits[i]), int(dibits[i + 1]))

            if soft_values is not None:
                soft = (float(soft_values[i]), float(soft_values[i + 1]))
            else:
                soft = None

            output = self.decode_step(received, soft)
            if output is not None:
                decoded.append(output)

        # Flush remaining bits from traceback
        remaining = self._flush()
        decoded.extend(remaining)

        # Calculate error metric (from best path)
        best_path = min(self._paths, key=lambda p: p.metric)
        error_metric = int(best_path.metric)

        if debug:
            decoded_str = " ".join(str(d) for d in decoded[:20])
            logger.info(f"Trellis decode: output {len(decoded)} dibits, first 20: {decoded_str}")
            logger.info(f"Trellis decode: error_metric={error_metric}")

        return np.array(decoded, dtype=np.uint8), error_metric

    def _flush(self) -> list[int]:
        """Flush remaining decoded bits from traceback buffer.

        Returns:
            Remaining decoded bits
        """
        if not self._paths:
            return []

        best_path = min(self._paths, key=lambda p: p.metric)
        # During decode_step, we already output up to index (len - TRACEBACK_DEPTH)
        # So remaining starts at (len - TRACEBACK_DEPTH + 1)
        already_output = len(best_path.decoded_bits) - self.TRACEBACK_DEPTH
        output_start = max(0, already_output + 1)
        return best_path.decoded_bits[output_start:]


def trellis_encode(dibits: NDArrayAny) -> NDArrayAny:
    """Encode dibits using P25 trellis code.

    Args:
        dibits: Input dibits

    Returns:
        Encoded dibits (2x input length)
    """
    state = 0
    output: list[int] = []

    for dibit in dibits:
        dibit = int(dibit) & 0x3
        next_state, out_pair = TRELLIS_ENCODER[(state, dibit)]
        output.extend(out_pair)
        state = next_state

    return np.array(output, dtype=np.uint8)


def trellis_decode(
    dibits: NDArrayAny,
    soft_values: NDArrayAny | None = None,
    debug: bool = False,
) -> tuple[NDArrayAny, int]:
    """Decode trellis-encoded dibits (convenience function).

    Args:
        dibits: Encoded dibits (even length)
        soft_values: Optional soft decision values
        debug: If True, log detailed debug information

    Returns:
        Tuple of (decoded_dibits, error_metric)
    """
    decoder = TrellisDecoder()
    return decoder.decode(dibits, soft_values, debug=debug)


def trellis_interleave(dibits: NDArrayAny, block_size: int = 98) -> NDArrayAny:
    """Apply P25 interleaving to trellis-encoded data.

    P25 uses block interleaving to spread burst errors across multiple
    codewords, improving error correction performance.

    Args:
        dibits: Input dibits
        block_size: Interleave block size

    Returns:
        Interleaved dibits
    """
    if len(dibits) < block_size:
        return dibits

    # P25 uses a convolutional interleaver
    # For simplicity, using block interleaving here
    rows = len(dibits) // block_size
    padded = np.zeros(rows * block_size, dtype=np.uint8)
    padded[: len(dibits)] = dibits[: rows * block_size]

    matrix = padded.reshape(rows, block_size)
    interleaved = matrix.T.flatten()

    return np.asarray(interleaved[: len(dibits)], dtype=np.uint8)


def trellis_deinterleave(dibits: NDArrayAny, block_size: int = 98) -> NDArrayAny:
    """Remove P25 interleaving from received data.

    Args:
        dibits: Interleaved dibits
        block_size: Interleave block size

    Returns:
        De-interleaved dibits
    """
    if len(dibits) < block_size:
        return dibits

    rows = len(dibits) // block_size
    padded = np.zeros(rows * block_size, dtype=np.uint8)
    padded[: len(dibits)] = dibits[: rows * block_size]

    matrix = padded.reshape(block_size, rows)
    deinterleaved = matrix.T.flatten()

    return np.asarray(deinterleaved[: len(dibits)], dtype=np.uint8)


# =============================================================================
# P25 3/4 Rate Trellis Decoder (for TSBK/PDU)
# =============================================================================

# P25 3/4 rate trellis encoder state transition table
# From SDRTrunk P25_3_4_Node.java TRANSITION_MATRIX
# Row = current state (0-7), Column = input tribit (0-7), Value = output nibble (0-15)
# The output nibble represents the 4-bit transmitted symbol
TRELLIS_3_4_TRANSITION = np.array(
    [
        [2, 13, 14, 1, 7, 8, 11, 4],  # state 0
        [14, 1, 7, 8, 11, 4, 2, 13],  # state 1
        [10, 5, 6, 9, 15, 0, 3, 12],  # state 2
        [6, 9, 15, 0, 3, 12, 10, 5],  # state 3
        [15, 0, 3, 12, 10, 5, 6, 9],  # state 4
        [3, 12, 10, 5, 6, 9, 15, 0],  # state 5
        [7, 8, 11, 4, 2, 13, 14, 1],  # state 6
        [11, 4, 2, 13, 14, 1, 7, 8],  # state 7
    ],
    dtype=np.int32,
)

# Hamming weight lookup for 0-15 (used for branch metric calculation)
HAMMING_WEIGHT = np.array([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4], dtype=np.int32)


@dataclass
class TrellisPath34:
    """A path through the 3/4 rate trellis for Viterbi decoding."""

    state: int
    metric: int  # Cumulative error metric
    input_values: list[int]  # Sequence of input tribits (0-7)


class TrellisDecoder34:
    """Viterbi decoder for P25 3/4 rate trellis code (TSBK/PDU).

    P25 TSBK uses 3/4 rate trellis coded modulation:
    - 3 input bits (tribit, 0-7)
    - 4 output bits (nibble, 0-15)
    - 8 states

    The decoder uses the Viterbi algorithm to find the maximum likelihood
    path through the trellis based on Hamming distance.

    Reference: TIA-102.BAAA-A Annex E, SDRTrunk ViterbiDecoder_3_4_P25.java
    """

    NUM_STATES = 8
    MAX_METRIC = 99999

    def __init__(self) -> None:
        """Initialize 3/4 rate trellis decoder."""
        pass

    def decode(self, symbols: NDArrayAny, debug: bool = False) -> tuple[NDArrayAny, int]:
        """Decode 3/4 rate trellis encoded symbols.

        Args:
            symbols: Array of 4-bit received symbols (nibbles, 0-15)
            debug: If True, log detailed debug information

        Returns:
            Tuple of (decoded_bits, error_metric)
            - decoded_bits: Decoded data as bit array
            - error_metric: Total Hamming distance errors
        """
        n_symbols = len(symbols)
        if n_symbols < 2:
            return np.array([], dtype=np.uint8), 0

        if debug:
            sym_str = " ".join(f"{s:x}" for s in symbols[:20])
            logger.info(f"Trellis34 decode: {n_symbols} symbols, first 20: {sym_str}")

        # Initialize paths - all start with infinite metric except state 0
        paths: list[TrellisPath34 | None] = [
            TrellisPath34(state=i, metric=0 if i == 0 else self.MAX_METRIC, input_values=[])
            for i in range(self.NUM_STATES)
        ]

        # Process symbols 0 to n-2 with normal add (all inputs allowed)
        for sym_idx in range(n_symbols - 1):
            received_sym = int(symbols[sym_idx]) & 0xF  # Ensure 4-bit nibble
            new_paths: list[TrellisPath34 | None] = [None] * self.NUM_STATES

            # For each possible next state, find best predecessor
            for next_state in range(self.NUM_STATES):
                best_metric = self.MAX_METRIC
                best_prev_state = 0
                best_input = 0

                # Check all possible transitions to this next_state
                for prev_state in range(self.NUM_STATES):
                    prev_path = paths[prev_state]
                    if prev_path is None or prev_path.metric >= self.MAX_METRIC:
                        continue

                    # For 3/4 rate, next state = input tribit
                    input_tribit = next_state

                    # Get expected output for this transition
                    expected_sym = TRELLIS_3_4_TRANSITION[prev_state, input_tribit]

                    # Calculate branch metric (Hamming distance)
                    error_mask = received_sym ^ expected_sym
                    branch_metric = int(HAMMING_WEIGHT[error_mask])

                    # Total path metric
                    path_metric = prev_path.metric + branch_metric

                    if path_metric < best_metric:
                        best_metric = path_metric
                        best_prev_state = prev_state
                        best_input = input_tribit

                # Create new path for this state
                if best_metric < self.MAX_METRIC:
                    prev_path = paths[best_prev_state]
                    if prev_path is None:
                        continue
                    new_paths[next_state] = TrellisPath34(
                        state=next_state,
                        metric=best_metric,
                        input_values=[*prev_path.input_values, best_input],
                    )
                else:
                    new_paths[next_state] = TrellisPath34(
                        state=next_state,
                        metric=self.MAX_METRIC,
                        input_values=[],
                    )

            paths = new_paths

        # Process last symbol with flush (only input=0 allowed, forcing convergence to state 0)
        received_sym = int(symbols[n_symbols - 1]) & 0xF
        flush_input = 0  # P25 encoder flushes with zeros
        best_path = None
        best_metric = self.MAX_METRIC

        for prev_state in range(self.NUM_STATES):
            prev_path = paths[prev_state]
            if prev_path is None or prev_path.metric >= self.MAX_METRIC:
                continue

            # Expected output for flushing transition (prev_state -> input 0)
            expected_sym = TRELLIS_3_4_TRANSITION[prev_state, flush_input]

            # Calculate branch metric
            error_mask = received_sym ^ expected_sym
            branch_metric = int(HAMMING_WEIGHT[error_mask])

            # Total path metric
            path_metric = prev_path.metric + branch_metric

            if path_metric < best_metric:
                best_metric = path_metric
                best_path = TrellisPath34(
                    state=flush_input,
                    metric=path_metric,
                    input_values=[*prev_path.input_values, flush_input],
                )

        # If flush didn't find a valid path, fallback to best survivor
        if best_path is None:
            best_path = paths[0]
            assert best_path is not None
            for p in paths:
                if p is not None and p.metric < best_path.metric:
                    best_path = p
        assert best_path is not None

        # Extract decoded bits from input values (tribits → bits)
        # Each input tribit represents 3 bits
        # Last tribit is the flushing input (zeros), skip it
        n_tribits = len(best_path.input_values)
        if n_tribits < 2:
            return np.array([], dtype=np.uint8), best_path.metric

        # Skip only the last (flushing) node
        # The starting state is implicit in the initialization (not in input_values)
        # SDRTrunk skips first (starting node) and last (flushing), but our input_values
        # doesn't include the starting node, so we only skip the last
        decoded_bits = []
        for i in range(n_tribits - 1):  # All except last (flushing)
            tribit = best_path.input_values[i]
            # Extract 3 bits from tribit (MSB first)
            decoded_bits.append((tribit >> 2) & 1)
            decoded_bits.append((tribit >> 1) & 1)
            decoded_bits.append(tribit & 1)

        error_metric = best_path.metric

        if debug:
            logger.info(
                f"Trellis34 decode: output {len(decoded_bits)} bits, error_metric={error_metric}"
            )
            if decoded_bits:
                bit_str = "".join(str(b) for b in decoded_bits[:24])
                logger.info(f"Trellis34 decode: first 24 bits: {bit_str}")

        return np.array(decoded_bits, dtype=np.uint8), error_metric


def trellis_decode_3_4(
    symbols: NDArrayAny,
    debug: bool = False,
) -> tuple[NDArrayAny, int]:
    """Decode 3/4 rate trellis encoded symbols (convenience function).

    This is the correct decoder for P25 TSBK (control channel) messages.

    Args:
        symbols: Array of 4-bit received symbols (nibbles, 0-15)
        debug: If True, log detailed debug information

    Returns:
        Tuple of (decoded_bits, error_metric)
    """
    decoder = TrellisDecoder34()
    return decoder.decode(symbols, debug=debug)
