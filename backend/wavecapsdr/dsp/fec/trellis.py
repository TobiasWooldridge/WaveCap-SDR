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
    0: 1.0,   # +1
    1: 3.0,   # +3
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

    def __init__(self):
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
        self, received: tuple[int, int], expected: tuple[int, int], soft: tuple[float, float] | None = None
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
            metric = float(bin(xor0).count('1') + bin(xor1).count('1'))

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
        dibits: np.ndarray,
        soft_values: np.ndarray | None = None,
    ) -> tuple[np.ndarray, int]:
        """Decode a block of trellis-encoded dibits.

        Args:
            dibits: Array of dibits (must be even length)
            soft_values: Optional soft values for each dibit

        Returns:
            Tuple of (decoded_dibits, num_errors)
        """
        if len(dibits) % 2 != 0:
            logger.warning("Trellis decode: odd number of dibits, truncating")
            dibits = dibits[:-1]
            if soft_values is not None:
                soft_values = soft_values[:-1]

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


def trellis_encode(dibits: np.ndarray) -> np.ndarray:
    """Encode dibits using P25 trellis code.

    Args:
        dibits: Input dibits

    Returns:
        Encoded dibits (2x input length)
    """
    state = 0
    output = []

    for dibit in dibits:
        dibit = int(dibit) & 0x3
        next_state, out_pair = TRELLIS_ENCODER[(state, dibit)]
        output.extend(out_pair)
        state = next_state

    return np.array(output, dtype=np.uint8)


def trellis_decode(
    dibits: np.ndarray,
    soft_values: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Decode trellis-encoded dibits (convenience function).

    Args:
        dibits: Encoded dibits (even length)
        soft_values: Optional soft decision values

    Returns:
        Tuple of (decoded_dibits, error_metric)
    """
    decoder = TrellisDecoder()
    return decoder.decode(dibits, soft_values)


def trellis_interleave(dibits: np.ndarray, block_size: int = 98) -> np.ndarray:
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

    return interleaved[: len(dibits)]


def trellis_deinterleave(dibits: np.ndarray, block_size: int = 98) -> np.ndarray:
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

    return deinterleaved[: len(dibits)]
