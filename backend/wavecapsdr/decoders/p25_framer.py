"""
P25 Phase 1 Message Framer - SDRTrunk-compatible streaming architecture.

This module provides streaming symbol processing with proper status symbol
stripping and message assembly, matching SDRTrunk's P25P1MessageFramer.java.

Key features:
- Symbol-by-symbol processing (not batch)
- Continuous status symbol stripping every 36 dibits
- Soft sync correlation with 24-symbol sliding window
- BCH error correction for NID (NAC + DUID)
- Multi-block message assembly for TSBK chains and PDU sequences
- Message dispatch callbacks

Reference: SDRTrunk P25P1MessageFramer.java, P25P1MessageAssembler.java
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, cast

import numpy as np
from wavecapsdr.typing import NDArrayFloat, NDArrayInt

from wavecapsdr.dsp.fec.bch import bch_decode

logger = logging.getLogger(__name__)


class P25P1DataUnitID(IntEnum):
    """P25 Phase 1 Data Unit IDs (DUID)."""
    HEADER_DATA_UNIT = 0x0  # HDU
    TERMINATOR_DATA_UNIT = 0x3  # TDU (without LC)
    LOGICAL_LINK_DATA_UNIT_1 = 0x5  # LDU1
    TRUNKING_SIGNALING_BLOCK_1 = 0x7  # TSDU / TSBK
    LOGICAL_LINK_DATA_UNIT_2 = 0xA  # LDU2
    PACKET_DATA_UNIT = 0xC  # PDU
    TERMINATOR_DATA_UNIT_LINK_CONTROL = 0xF  # TDULC

    # Internal use
    UNKNOWN = 0xE
    PLACE_HOLDER = 0xD

    # Multi-block continuation markers
    TRUNKING_SIGNALING_BLOCK_2 = 0x17
    TRUNKING_SIGNALING_BLOCK_3 = 0x27
    PACKET_DATA_UNIT_BLOCK_1 = 0x1C
    PACKET_DATA_UNIT_BLOCK_2 = 0x2C
    PACKET_DATA_UNIT_BLOCK_3 = 0x3C
    PACKET_DATA_UNIT_BLOCK_4 = 0x4C
    PACKET_DATA_UNIT_BLOCK_5 = 0x5C

    @classmethod
    def from_value(cls, value: int) -> 'P25P1DataUnitID':
        """Get DUID from 4-bit value."""
        try:
            return cls(value)
        except ValueError:
            return cls.UNKNOWN

    def get_message_length(self) -> int:
        """Get expected message length in bits (after status symbol removal)."""
        lengths = {
            self.HEADER_DATA_UNIT: 648,
            self.TERMINATOR_DATA_UNIT: 28,
            self.LOGICAL_LINK_DATA_UNIT_1: 1568,
            self.LOGICAL_LINK_DATA_UNIT_2: 1568,
            self.TRUNKING_SIGNALING_BLOCK_1: 196,
            self.TRUNKING_SIGNALING_BLOCK_2: 392,
            self.TRUNKING_SIGNALING_BLOCK_3: 588,
            self.PACKET_DATA_UNIT: 196,
            self.PACKET_DATA_UNIT_BLOCK_1: 392,
            self.PACKET_DATA_UNIT_BLOCK_2: 588,
            self.PACKET_DATA_UNIT_BLOCK_3: 784,
            self.PACKET_DATA_UNIT_BLOCK_4: 980,
            self.PACKET_DATA_UNIT_BLOCK_5: 1176,
            self.TERMINATOR_DATA_UNIT_LINK_CONTROL: 168,
            self.PLACE_HOLDER: 2000,  # Max placeholder length
        }
        return lengths.get(self, 196)

    def get_elapsed_dibit_length(self) -> int:
        """Get expected elapsed dibits including sync and NID."""
        # SYNC (24 dibits) + NID (32 dibits) + message dibits
        return 57 + (self.get_message_length() // 2)


@dataclass
class Dibit:
    """P25 dibit (2-bit symbol) with ideal phase for correlation."""
    value: int  # 0, 1, 2, or 3

    # Dibit to ideal phase mapping (radians)
    # dibit 0 -> +1 symbol -> +π/4
    # dibit 1 -> +3 symbol -> +3π/4
    # dibit 2 -> -1 symbol -> -π/4
    # dibit 3 -> -3 symbol -> -3π/4
    IDEAL_PHASES = {
        0: np.pi / 4,      # +1
        1: 3 * np.pi / 4,  # +3
        2: -np.pi / 4,     # -1
        3: -3 * np.pi / 4, # -3
    }

    @property
    def ideal_phase(self) -> float:
        return self.IDEAL_PHASES.get(self.value, 0.0)

    @property
    def bit1(self) -> int:
        """High bit of dibit."""
        return (self.value >> 1) & 1

    @property
    def bit2(self) -> int:
        """Low bit of dibit."""
        return self.value & 1


class P25P1SoftSyncDetector:
    """
    Soft sync pattern detector for P25 Phase 1.

    Uses sliding window correlation against the 24-symbol sync pattern.
    Returns correlation score at each symbol position.

    Reference: SDRTrunk P25P1SoftSyncDetectorScalar.java
    """

    # Sync pattern: 0x5575F5FF77FF (48 bits = 24 dibits)
    SYNC_PATTERN = 0x5575F5FF77FF

    # Sync pattern as ideal phase values (for soft correlation)
    # +3 = dibit 1, -3 = dibit 3
    SYNC_PATTERN_SYMBOLS: NDArrayFloat = field(default_factory=lambda: np.zeros(24))

    def __init__(self) -> None:
        # Convert sync pattern to symbol phases
        self.SYNC_PATTERN_SYMBOLS = self._pattern_to_symbols()

        # Circular buffer for 24 symbols (doubled for easy access)
        self._symbols = np.zeros(48, dtype=np.float32)
        self._pointer = 0

    def _pattern_to_symbols(self) -> NDArrayFloat:
        """Convert sync pattern to ideal symbol values."""
        symbols = np.zeros(24, dtype=np.float32)
        pattern = self.SYNC_PATTERN

        for i in range(24):
            # Extract 2 bits from position (23-i)*2
            dibit = (pattern >> ((23 - i) * 2)) & 0x3
            # dibit 1 -> +3, dibit 3 -> -3 (only these appear in sync)
            symbols[i] = 3.0 if dibit == 1 else -3.0

        return cast(NDArrayFloat, symbols)

    def reset(self) -> None:
        """Reset detector state."""
        self._symbols.fill(0.0)
        self._pointer = 0

    def process(self, soft_symbol: float) -> float:
        """
        Process one soft symbol and return correlation score.

        Args:
            soft_symbol: Demodulated symbol value (normalized to ±1, ±3)

        Returns:
            Correlation score (max 216 for perfect match)
        """
        # Store in circular buffer (both halves for easy access)
        self._symbols[self._pointer] = soft_symbol
        self._symbols[self._pointer + 24] = soft_symbol
        self._pointer = (self._pointer + 1) % 24

        # Compute correlation
        return self._calculate()

    def _calculate(self) -> float:
        """Calculate correlation score against sync pattern."""
        # Vectorized dot product instead of Python loop
        return float(np.dot(self.SYNC_PATTERN_SYMBOLS, self._symbols[self._pointer:self._pointer + 24]))

    def process_batch(self, soft_symbols: NDArrayFloat) -> NDArrayFloat:
        """
        Process batch of soft symbols and return correlation scores.

        Uses vectorized numpy operations for ~100x speedup over per-symbol loop.

        Args:
            soft_symbols: Array of demodulated symbol values (normalized to ±1, ±3)

        Returns:
            Array of correlation scores, one per input symbol
        """
        n = len(soft_symbols)
        if n == 0:
            return np.array([], dtype=np.float32)

        # Extend symbol buffer with new symbols
        # We need 24 symbols of history for correlation
        extended = np.concatenate([self._symbols[self._pointer:self._pointer + 24], soft_symbols])

        # Use numpy correlate for sliding window correlation
        # mode='valid' gives output length = len(extended) - len(pattern) + 1 = n + 1
        # We take the last n scores (one per input symbol, after it's been added)
        scores = np.correlate(extended, self.SYNC_PATTERN_SYMBOLS, mode='valid')
        scores = scores[-n:]  # Align with input: scores[i] = correlation after symbol[i]

        # Update circular buffer with last 24 symbols
        if n >= 24:
            self._symbols[:24] = soft_symbols[-24:]
            self._symbols[24:48] = soft_symbols[-24:]
            self._pointer = 0
        else:
            # Shift existing buffer and append new symbols
            for sym in soft_symbols:
                self._symbols[self._pointer] = sym
                self._symbols[self._pointer + 24] = sym
                self._pointer = (self._pointer + 1) % 24

        return cast(NDArrayFloat, scores.astype(np.float32))


class P25P1MessageAssembler:
    """
    Assembles P25 Phase 1 messages from dibit stream.

    Tracks expected message length and signals completion.
    Supports DUID reassignment for placeholder messages.

    Reference: SDRTrunk P25P1MessageAssembler.java
    """

    def __init__(self, nac: int, duid: P25P1DataUnitID) -> None:
        self.nac = nac
        self.duid = duid
        self._bits: list[int] = []
        self._target_length = duid.get_message_length()
        self._force_completed = False

    def receive(self, dibit: int) -> None:
        """Add a dibit (2 bits) to the message."""
        if dibit < 0 or dibit > 3:
            raise AssertionError(f"Invalid dibit {dibit} for DUID {self.duid.name}")

        if len(self._bits) < self._target_length:
            self._bits.append((dibit >> 1) & 1)
            if len(self._bits) < self._target_length:
                self._bits.append(dibit & 1)

    def is_complete(self) -> bool:
        """Check if message has reached expected length."""
        return len(self._bits) >= self._target_length

    def get_message_bits(self) -> NDArrayInt:
        """Get assembled message as bit array."""
        return np.array(self._bits, dtype=np.uint8)

    def current_size(self) -> int:
        """Get current message size in bits."""
        return len(self._bits)

    def reconfigure(self, duid: P25P1DataUnitID) -> None:
        """Reconfigure for continuation block assembly."""
        self.duid = duid
        self._target_length = duid.get_message_length()

    def set_duid(self, duid: P25P1DataUnitID) -> None:
        """Set DUID and update target length."""
        self.duid = duid
        self._target_length = duid.get_message_length()

    def was_force_completed(self) -> bool:
        """Return True when the message was force-completed after a resync."""
        return self._force_completed

    def force_completion(
        self,
        previous_duid: P25P1DataUnitID,
        next_duid: P25P1DataUnitID
    ) -> int:
        """
        Force message completion and attempt DUID inference.

        Returns number of dropped bits.
        """
        current_size = len(self._bits)
        self._force_completed = True

        # Try to infer DUID from context if we have a placeholder
        if self.duid == P25P1DataUnitID.PLACE_HOLDER:
            if current_size <= 28:
                self.duid = P25P1DataUnitID.TERMINATOR_DATA_UNIT
            elif next_duid == P25P1DataUnitID.LOGICAL_LINK_DATA_UNIT_1:
                if current_size <= 770:
                    self.duid = P25P1DataUnitID.HEADER_DATA_UNIT
                elif current_size >= 1500:
                    self.duid = P25P1DataUnitID.LOGICAL_LINK_DATA_UNIT_2
            elif next_duid == P25P1DataUnitID.LOGICAL_LINK_DATA_UNIT_2:
                if current_size >= 1500:
                    self.duid = P25P1DataUnitID.LOGICAL_LINK_DATA_UNIT_1
            elif next_duid == P25P1DataUnitID.TRUNKING_SIGNALING_BLOCK_1:
                if current_size >= 195:
                    self.duid = P25P1DataUnitID.TRUNKING_SIGNALING_BLOCK_1

        # Default to TDU if still unknown
        if self.duid == P25P1DataUnitID.PLACE_HOLDER:
            self.duid = P25P1DataUnitID.TERMINATOR_DATA_UNIT

        self._target_length = self.duid.get_message_length()
        return max(0, self._target_length - current_size)


class NACTracker:
    """
    Tracks observed NAC values for BCH decode assistance.

    Reference: SDRTrunk NACTracker.java
    """

    def __init__(self, min_observations: int = 3) -> None:
        self._observations: dict[int, int] = {}
        self._min_observations = min_observations
        self._tracked_nac: int | None = None

    def track(self, nac: int) -> None:
        """Record a NAC observation."""
        if 0x001 <= nac <= 0xFFE:  # Valid NAC range
            self._observations[nac] = self._observations.get(nac, 0) + 1

            # Update tracked NAC if this one has enough observations
            if self._observations[nac] >= self._min_observations:
                self._tracked_nac = nac

    def get_tracked_nac(self) -> int:
        """Get the most frequently observed NAC (0 if not enough data)."""
        return self._tracked_nac or 0

    def reset(self) -> None:
        """Reset tracker state."""
        self._observations.clear()
        self._tracked_nac = None


@dataclass
class P25P1Message:
    """Assembled P25 Phase 1 message."""
    duid: P25P1DataUnitID
    nac: int
    timestamp: int
    bits: NDArrayInt
    corrected_bit_count: int = 0
    valid: bool = True


class P25P1MessageFramer:
    """
    P25 Phase 1 Message Framer - SDRTrunk-compatible streaming architecture.

    Processes demodulated symbols one at a time, handling:
    - Sync pattern detection (soft correlation)
    - Status symbol stripping (every 36 dibits)
    - NID decoding with BCH error correction
    - Message assembly with DUID-based length tracking
    - Multi-block TSBK and PDU sequence handling

    Reference: SDRTrunk P25P1MessageFramer.java
    """

    DIBIT_LENGTH_NID = 33  # 32 dibits + 1 status
    SYNC_DETECTION_THRESHOLD = 60.0

    def __init__(self) -> None:
        # Sync detection
        self._soft_sync_detector = P25P1SoftSyncDetector()
        self._sync_detected = False

        # NID buffer
        self._nid_buffer: list[int] = []
        self._nid_pointer = 0

        # NAC tracking
        self._nac_tracker = NACTracker()

        # Counters
        self._dibit_counter = 58  # Start > 57 to avoid triggering on startup
        self._status_symbol_counter = 36  # Start > 35 to avoid triggering
        self._dibit_since_timestamp = 0
        self._debug_symbol_count = 0

        # Message assembly
        self._message_assembler: P25P1MessageAssembler | None = None
        self._message_assembly_required = False
        self._previous_duid = P25P1DataUnitID.PLACE_HOLDER
        self._detected_duid = P25P1DataUnitID.PLACE_HOLDER
        self._detected_nac = 0
        self._detected_sync_bit_errors = 0

        # Timestamp
        self._reference_timestamp = 0

        # Callback for assembled messages
        self._message_listener: Callable[[P25P1Message], None] | None = None

        # Running state
        self._running = False

    def start(self) -> None:
        """Start message dispatching."""
        self._running = True

    def stop(self) -> None:
        """Stop message dispatching."""
        self._running = False

    def set_listener(self, listener: Callable[[P25P1Message], None]) -> None:
        """Set callback for assembled messages."""
        self._message_listener = listener

    def set_timestamp(self, timestamp: int) -> None:
        """Set reference timestamp for message timing."""
        self._reference_timestamp = timestamp
        self._dibit_since_timestamp = 0

    def _get_timestamp(self) -> int:
        """Calculate timestamp for current position."""
        if self._reference_timestamp > 0:
            return self._reference_timestamp + int(1000.0 * self._dibit_since_timestamp / 4800)
        return 0

    def process_with_soft_sync(self, soft_symbol: float, dibit: int) -> bool:
        """
        Process one symbol with soft sync detection.

        This is the main entry point for streaming symbol processing.

        Args:
            soft_symbol: Soft symbol value (normalized to ±1, ±3)
            dibit: Hard decision dibit (0-3)

        Returns:
            True if a valid NID was detected
        """
        valid_nid_detected = self._process(dibit)

        # Check for sync in soft symbol stream
        if self._soft_sync_detector.process(soft_symbol) > self.SYNC_DETECTION_THRESHOLD:
            self._sync_detected_callback()

        return valid_nid_detected

    def process(self, dibit: int) -> bool:
        """
        Process one dibit without sync detection (for external sync).

        Args:
            dibit: Hard decision dibit (0-3)

        Returns:
            True if a valid NID was detected
        """
        return self._process(dibit)

    def process_batch(self, soft_symbols: NDArrayFloat, dibits: NDArrayInt) -> int:
        """
        Process batch of symbols with vectorized sync detection.

        This is ~50-100x faster than calling process_with_soft_sync() per symbol
        because sync correlation is done vectorized with numpy.

        Args:
            soft_symbols: Array of soft symbol values (normalized to ±1, ±3)
            dibits: Array of hard decision dibits (0-3)

        Returns:
            Number of valid NIDs detected
        """
        n = len(dibits)
        if n == 0 or len(soft_symbols) != n:
            return 0

        # Vectorized sync detection - single numpy operation for entire batch
        scores = self._soft_sync_detector.process_batch(soft_symbols)

        # Find sync positions where correlation exceeds threshold
        sync_positions = np.where(scores > self.SYNC_DETECTION_THRESHOLD)[0]
        sync_set = set(sync_positions)

        # Process dibits through state machine
        # The state machine must still run per-symbol, but we've removed
        # the expensive per-symbol correlation
        nid_count = 0
        for i in range(n):
            # Trigger sync callback at detected positions
            if i in sync_set:
                self._sync_detected_callback()

            # Process through state machine (lightweight without correlation)
            if self._process(int(dibits[i])):
                nid_count += 1

        return nid_count

    def _sync_detected_callback(self) -> None:
        """Called when sync pattern is detected."""
        self._sync_detected = True
        self._nid_pointer = 0
        self._nid_buffer = []

    def _process(self, dibit: int) -> bool:
        """
        Internal symbol processing.

        Handles status symbol stripping, NID collection, and message assembly.
        """
        valid_nid_detected = False

        self._debug_symbol_count += 1
        self._dibit_since_timestamp += 1

        # Status symbol counter runs continuously
        self._status_symbol_counter += 1

        # Collect NID after sync detection
        if self._sync_detected:
            self._nid_buffer.append(dibit)
            self._nid_pointer += 1

            if self._nid_pointer >= self.DIBIT_LENGTH_NID:
                valid_nid_detected = self._check_nid()
                self._sync_detected = False

        # Strip status symbol at position 36
        if self._status_symbol_counter == 36:
            # Status symbol - don't feed to message assembler
            self._status_symbol_counter = 0
            self._dibit_counter += 1
            return False

        # Feed to message assembler if active
        if self._message_assembler is not None:
            # Check completion before adding new dibit
            if self._message_assembler.is_complete():
                self._dispatch_message()

                # If still have assembler (continuation), feed current dibit
                if self._message_assembler is not None:
                    self._message_assembler.receive(dibit)
            else:
                self._message_assembler.receive(dibit)

        # Start assembler at dibit 57 (after sync + NID + 1 status)
        elif self._dibit_counter == 57:
            if self._message_assembly_required:
                self._message_assembler = P25P1MessageAssembler(
                    self._detected_nac,
                    self._detected_duid
                )
                self._message_assembly_required = False
            elif self._detected_nac > 0:
                # Start placeholder assembly
                self._detected_duid = P25P1DataUnitID.PLACE_HOLDER
                self._message_assembler = P25P1MessageAssembler(
                    self._detected_nac,
                    self._detected_duid
                )

        # Sync loss detection (1 second without sync)
        elif self._dibit_counter >= 4800:
            self._dibit_counter -= 4800
            logger.debug("P25 sync loss (1 second)")

        self._dibit_counter += 1
        return valid_nid_detected

    def _check_nid(self) -> bool:
        """
        Decode NID from collected dibits using BCH error correction.

        Returns True if NID is valid.
        """
        if len(self._nid_buffer) < self.DIBIT_LENGTH_NID:
            return False

        # Remove status symbol at position 11
        nid_dibits = self._nid_buffer[:11] + self._nid_buffer[12:33]

        # Convert to bits
        nid_bits = np.zeros(64, dtype=np.uint8)
        for i, d in enumerate(nid_dibits[:32]):
            nid_bits[i * 2] = (d >> 1) & 1
            nid_bits[i * 2 + 1] = d & 1

        # BCH(63,16,23) decode
        tracked_nac = self._nac_tracker.get_tracked_nac()
        decoded_data, errors = bch_decode(nid_bits[:63], tracked_nac if tracked_nac else None)

        if errors < 0:
            return False

        # Extract NAC and DUID
        nac = (decoded_data >> 4) & 0xFFF
        duid_value = decoded_data & 0xF
        duid = P25P1DataUnitID.from_value(duid_value)

        # Track NAC
        self._nac_tracker.track(nac)

        # Trigger NID detected
        self._nid_detected(nac, duid, errors)

        return True

    def _nid_detected(self, nac: int, duid: P25P1DataUnitID, bit_errors: int) -> None:
        """Handle validated NID detection."""
        self._detected_duid = duid
        self._detected_nac = nac
        self._detected_sync_bit_errors = bit_errors

        # Handle UNKNOWN DUID
        if duid == P25P1DataUnitID.UNKNOWN:
            self._detected_duid = P25P1DataUnitID.PLACE_HOLDER

        # Force completion of current message if active
        if self._message_assembler is not None:
            if self._message_assembler.is_complete():
                if self._message_assembler.duid != P25P1DataUnitID.PLACE_HOLDER:
                    self._dispatch_message()
            else:
                dropped = self._message_assembler.force_completion(
                    self._previous_duid,
                    self._detected_duid
                )
                if dropped > 0:
                    logger.debug(f"P25 dropped {dropped} bits")
                self._dispatch_message()

        # Reset counters for new frame
        self._dibit_counter -= 57
        if self._dibit_counter > 0:
            logger.debug(f"P25 sync loss: {self._dibit_counter * 2} bits")

        self._message_assembly_required = True
        self._dibit_counter = 57
        self._status_symbol_counter = 21  # SDRTrunk value

    def _assert_message_length(
        self,
        bits: np.ndarray,
        duid: P25P1DataUnitID,
        allow_truncated: bool = False
    ) -> None:
        """Validate message length to fast-fail on malformed frames."""
        if duid == P25P1DataUnitID.PLACE_HOLDER:
            raise AssertionError("Cannot dispatch placeholder message")

        message_length = int(bits.size)
        expected_length = duid.get_message_length()

        if allow_truncated and message_length < expected_length:
            return

        if duid in (
            P25P1DataUnitID.TRUNKING_SIGNALING_BLOCK_1,
            P25P1DataUnitID.TRUNKING_SIGNALING_BLOCK_2,
            P25P1DataUnitID.TRUNKING_SIGNALING_BLOCK_3,
            P25P1DataUnitID.PACKET_DATA_UNIT,
            P25P1DataUnitID.PACKET_DATA_UNIT_BLOCK_1,
            P25P1DataUnitID.PACKET_DATA_UNIT_BLOCK_2,
            P25P1DataUnitID.PACKET_DATA_UNIT_BLOCK_3,
            P25P1DataUnitID.PACKET_DATA_UNIT_BLOCK_4,
            P25P1DataUnitID.PACKET_DATA_UNIT_BLOCK_5,
        ):
            if message_length < expected_length:
                raise AssertionError(
                    f"P25 {duid.name} length {message_length} below minimum {expected_length}"
                )
            if message_length % 196 != 0:
                raise AssertionError(
                    f"P25 {duid.name} length {message_length} is not aligned to 196-bit blocks"
                )
            return

        if message_length != expected_length:
            raise AssertionError(
                f"P25 {duid.name} length {message_length} did not match expected {expected_length}"
            )

    def _dispatch_message(self) -> None:
        """Dispatch assembled message to listener."""
        if self._message_assembler is None:
            return

        self._previous_duid = self._message_assembler.duid

        if not self._running or self._message_listener is None:
            self._message_assembler = None
            return

        duid = self._message_assembler.duid
        allow_truncated = self._message_assembler.was_force_completed()

        # Handle different message types
        if duid in (
            P25P1DataUnitID.TRUNKING_SIGNALING_BLOCK_1,
            P25P1DataUnitID.TRUNKING_SIGNALING_BLOCK_2,
            P25P1DataUnitID.TRUNKING_SIGNALING_BLOCK_3,
        ):
            self._dispatch_tsbk(allow_truncated=allow_truncated)
        elif duid in (
            P25P1DataUnitID.PACKET_DATA_UNIT,
            P25P1DataUnitID.PACKET_DATA_UNIT_BLOCK_1,
            P25P1DataUnitID.PACKET_DATA_UNIT_BLOCK_2,
            P25P1DataUnitID.PACKET_DATA_UNIT_BLOCK_3,
            P25P1DataUnitID.PACKET_DATA_UNIT_BLOCK_4,
            P25P1DataUnitID.PACKET_DATA_UNIT_BLOCK_5,
        ):
            self._dispatch_pdu(allow_truncated=allow_truncated)
        elif duid == P25P1DataUnitID.PLACE_HOLDER:
            self._message_assembler = None
        else:
            self._dispatch_other(allow_truncated=allow_truncated)

    def _dispatch_other(self, allow_truncated: bool = False) -> None:
        """Dispatch non-TSBK/PDU message."""
        if self._message_assembler is None:
            return

        bits = self._message_assembler.get_message_bits()
        self._assert_message_length(
            bits,
            self._message_assembler.duid,
            allow_truncated=allow_truncated
        )

        message = P25P1Message(
            duid=self._message_assembler.duid,
            nac=self._message_assembler.nac,
            timestamp=self._get_timestamp(),
            bits=bits,
            corrected_bit_count=self._detected_sync_bit_errors,
        )

        self._broadcast(message)
        self._message_assembler = None

    def _dispatch_tsbk(self, allow_truncated: bool = False) -> None:
        """Dispatch TSBK message with multi-block handling."""
        if self._message_assembler is None:
            return

        duid = self._message_assembler.duid
        bits = self._message_assembler.get_message_bits()
        self._assert_message_length(bits, duid, allow_truncated=allow_truncated)

        if duid == P25P1DataUnitID.TRUNKING_SIGNALING_BLOCK_1:
            # First TSBK block
            if len(bits) >= 196:
                block1_bits = bits[:196]
                message = P25P1Message(
                    duid=duid,
                    nac=self._message_assembler.nac,
                    timestamp=self._get_timestamp(),
                    bits=block1_bits,
                    corrected_bit_count=self._detected_sync_bit_errors,
                )
                self._broadcast(message)

                # Check for continuation
                if len(bits) >= 392:
                    self._message_assembler.set_duid(P25P1DataUnitID.TRUNKING_SIGNALING_BLOCK_2)
                    self._dispatch_tsbk(allow_truncated=allow_truncated)
                else:
                    # Check last block flag (would need trellis decode to check)
                    # For now, prepare for potential continuation
                    self._message_assembler.reconfigure(P25P1DataUnitID.TRUNKING_SIGNALING_BLOCK_2)

        elif duid == P25P1DataUnitID.TRUNKING_SIGNALING_BLOCK_2:
            if len(bits) >= 392:
                block2_bits = bits[196:392]
                message = P25P1Message(
                    duid=duid,
                    nac=self._message_assembler.nac,
                    timestamp=self._get_timestamp(),
                    bits=block2_bits,
                    corrected_bit_count=0,
                )
                self._broadcast(message)

                if len(bits) >= 588:
                    self._message_assembler.set_duid(P25P1DataUnitID.TRUNKING_SIGNALING_BLOCK_3)
                    self._dispatch_tsbk(allow_truncated=allow_truncated)
                else:
                    self._message_assembler.reconfigure(P25P1DataUnitID.TRUNKING_SIGNALING_BLOCK_3)

        elif duid == P25P1DataUnitID.TRUNKING_SIGNALING_BLOCK_3:
            if len(bits) >= 588:
                block3_bits = bits[392:588]
                message = P25P1Message(
                    duid=duid,
                    nac=self._message_assembler.nac,
                    timestamp=self._get_timestamp(),
                    bits=block3_bits,
                    corrected_bit_count=0,
                )
                self._broadcast(message)
            self._message_assembler = None

    def _dispatch_pdu(self, allow_truncated: bool = False) -> None:
        """Dispatch PDU message with multi-block handling."""
        # Similar structure to TSBK but for PDU blocks
        if self._message_assembler is None:
            return

        bits = self._message_assembler.get_message_bits()
        self._assert_message_length(
            bits,
            self._message_assembler.duid,
            allow_truncated=allow_truncated
        )

        message = P25P1Message(
            duid=self._message_assembler.duid,
            nac=self._message_assembler.nac,
            timestamp=self._get_timestamp(),
            bits=bits,
            corrected_bit_count=self._detected_sync_bit_errors,
        )
        self._broadcast(message)
        self._message_assembler = None

    def _broadcast(self, message: P25P1Message) -> None:
        """Send message to listener."""
        if self._running and self._message_listener is not None:
            try:
                self._message_listener(message)
            except Exception as e:
                logger.error(f"Error in message listener: {e}")

    def reset(self) -> None:
        """Reset framer to initial state."""
        self._soft_sync_detector.reset()
        self._sync_detected = False
        self._nid_buffer = []
        self._nid_pointer = 0
        self._dibit_counter = 58
        self._status_symbol_counter = 36
        self._message_assembler = None
        self._message_assembly_required = False
        self._previous_duid = P25P1DataUnitID.PLACE_HOLDER
        self._detected_duid = P25P1DataUnitID.PLACE_HOLDER
        self._detected_nac = 0
