"""
P25 Phase 2 (TDMA) Message Framer - SDRTrunk-compatible implementation.

This module provides streaming TDMA processing with proper SuperFrame detection
and timeslot demultiplexing, matching SDRTrunk's P25P2SuperFrameDetector.java.

P25 Phase 2 uses TDMA with 2 timeslots per channel at 6000 symbols/second
(12000 baud with CQPSK/HDQPSK modulation).

SuperFrame Fragment Structure (720 dibits = 1440 bits):
    I-ISCH1 (20 dibits) + TIMESLOT1 (160 dibits) = 180 dibits
    I-ISCH2 (20 dibits) + TIMESLOT2 (160 dibits) = 180 dibits
    S-SISCH1/Sync1 (20 dibits) + TIMESLOT3 (160 dibits) = 180 dibits
    S-SISCH2/Sync2 (20 dibits) + TIMESLOT4 (160 dibits) = 180 dibits

Sync patterns are at positions 360 and 540 dibits within the fragment.

Reference: SDRTrunk P25P2SuperFrameDetector.java, P25P2SyncPattern.java
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable

import numpy as np
from wavecapsdr.typing import NDArrayAny

logger = logging.getLogger(__name__)


# P25 Phase 2 Sync Pattern: 0x575D57F7FF (40 bits = 20 dibits)
# This pattern appears at positions 360 and 540 within each 720-dibit SuperFrame fragment
P25_PHASE2_SYNC_PATTERN = 0x575D57F7FF
P25_PHASE2_SYNC_PATTERN_90_CCW = 0xFEFBFEAEAA
P25_PHASE2_SYNC_PATTERN_90_CW = 0x0104015155
P25_PHASE2_SYNC_PATTERN_180 = 0xA8A2A80800

# Sync pattern as dibit array (20 dibits)
P25_PHASE2_SYNC_DIBITS = np.array(
    [(P25_PHASE2_SYNC_PATTERN >> (38 - i * 2)) & 0x3 for i in range(20)], dtype=np.uint8
)


class P25P2TimeslotType(IntEnum):
    """P25 Phase 2 timeslot types."""

    VOICE = 0
    SACCH = 1  # Slow Associated Control Channel
    FACCH = 2  # Fast Associated Control Channel


@dataclass
class P25P2Timeslot:
    """Decoded P25 Phase 2 timeslot."""

    timeslot_number: int  # 0 or 1
    slot_type: P25P2TimeslotType
    dibits: NDArrayAny
    isch_dibits: NDArrayAny  # Info ISCH for this slot
    timestamp: int = 0
    valid: bool = True


@dataclass
class P25P2SuperFrameFragment:
    """A 720-dibit P25 Phase 2 SuperFrame fragment containing 4 timeslots."""

    dibits: NDArrayAny
    timestamp: int
    sync_bit_errors: int = 0

    def get_timeslot(self, index: int) -> tuple[NDArrayAny, NDArrayAny]:
        """Get timeslot data and preceding ISCH.

        Args:
            index: Timeslot index (0-3)

        Returns:
            Tuple of (isch_dibits, timeslot_dibits)
        """
        if index < 0 or index > 3:
            raise ValueError(f"Invalid timeslot index: {index}")

        # Each timeslot segment is 180 dibits: 20 ISCH + 160 timeslot
        start = index * 180
        isch = self.dibits[start : start + 20]
        timeslot = self.dibits[start + 20 : start + 180]
        return isch, timeslot


class P25P2SyncPattern:
    """P25 Phase 2 sync pattern utilities."""

    @staticmethod
    def get_bit_error_count(dibits: NDArrayAny) -> int:
        """Calculate bit error count (hamming distance) against sync pattern.

        Args:
            dibits: 20-dibit sequence to compare

        Returns:
            Number of bit errors (0-40)
        """
        if len(dibits) < 20:
            return 40  # Max errors

        errors = 0
        for i in range(20):
            expected = P25_PHASE2_SYNC_DIBITS[i]
            actual = dibits[i] & 0x3

            # Count differing bits
            diff = expected ^ actual
            errors += (diff & 1) + ((diff >> 1) & 1)

        return errors

    @staticmethod
    def detect_phase_error(dibits: NDArrayAny) -> int:
        """Detect phase rotation error in sync pattern.

        Args:
            dibits: 20-dibit sequence

        Returns:
            Phase error in degrees (0, 90, 180, 270) or -1 if no match
        """
        if len(dibits) < 20:
            return -1

        # Convert dibits to 40-bit value
        value = 0
        for i in range(20):
            value = (value << 2) | (dibits[i] & 0x3)

        # Check against each pattern
        patterns = [
            (0, P25_PHASE2_SYNC_PATTERN),
            (90, P25_PHASE2_SYNC_PATTERN_90_CW),
            (180, P25_PHASE2_SYNC_PATTERN_180),
            (270, P25_PHASE2_SYNC_PATTERN_90_CCW),
        ]

        best_error = 40
        best_phase = -1

        for phase, pattern in patterns:
            errors = bin(value ^ pattern).count("1")
            if errors < best_error:
                best_error = errors
                best_phase = phase

        return best_phase if best_error <= 7 else -1


class P25P2SyncDetector:
    """
    P25 Phase 2 sync pattern detector.

    Implements soft correlation against the 20-dibit sync pattern
    with support for phase error detection.
    """

    # Sync pattern as soft symbols for correlation
    # dibit 0 -> +1, 1 -> +3, 2 -> -1, 3 -> -3
    DIBIT_TO_SYMBOL = np.array([1.0, 3.0, -1.0, -3.0], dtype=np.float32)
    SYNC_PATTERN_SYMBOLS = DIBIT_TO_SYMBOL[P25_PHASE2_SYNC_DIBITS]

    def __init__(self, threshold: int = 4) -> None:
        """Initialize sync detector.

        Args:
            threshold: Maximum bit errors to consider a valid sync
        """
        self.threshold = threshold

        # Circular buffer for 20 dibits (doubled for easy access)
        self._dibits = np.zeros(40, dtype=np.uint8)
        self._pointer = 0
        self._dibits_processed = 0

    def reset(self) -> None:
        """Reset detector state."""
        self._dibits.fill(0)
        self._pointer = 0
        self._dibits_processed = 0

    def process(self, dibit: int) -> tuple[bool, int]:
        """Process one dibit and check for sync.

        Args:
            dibit: Input dibit (0-3)

        Returns:
            Tuple of (sync_detected, bit_errors)
        """
        # Store in circular buffer
        self._dibits[self._pointer] = dibit & 0x3
        self._dibits[self._pointer + 20] = dibit & 0x3
        self._pointer = (self._pointer + 1) % 20
        self._dibits_processed += 1

        # Calculate bit errors
        bit_errors = self._get_bit_errors()

        return bit_errors <= self.threshold, bit_errors

    def _get_bit_errors(self) -> int:
        """Calculate bit errors for current buffer position."""
        errors = 0
        for i in range(20):
            expected = P25_PHASE2_SYNC_DIBITS[i]
            actual = self._dibits[self._pointer + i]
            diff = expected ^ actual
            errors += (diff & 1) + ((diff >> 1) & 1)
        return errors


class DibitDelayBuffer:
    """Circular delay buffer for dibits."""

    def __init__(self, size: int) -> None:
        self._buffer = np.zeros(size, dtype=np.uint8)
        self._size = size
        self._pointer = 0
        self._filled = False

    def put(self, dibit: int) -> None:
        """Add a dibit to the buffer."""
        self._buffer[self._pointer] = dibit & 0x3
        self._pointer = (self._pointer + 1) % self._size
        if self._pointer == 0:
            self._filled = True

    def get_and_put(self, dibit: int) -> int:
        """Get oldest dibit and replace with new one."""
        oldest = self._buffer[self._pointer]
        self._buffer[self._pointer] = dibit & 0x3
        self._pointer = (self._pointer + 1) % self._size
        if self._pointer == 0:
            self._filled = True
        return int(oldest)

    def get_buffer(self, start: int, length: int) -> NDArrayAny:
        """Get a contiguous section of the buffer."""
        result = np.zeros(length, dtype=np.uint8)
        for i in range(length):
            idx = (self._pointer + start + i) % self._size
            result[i] = self._buffer[idx]
        return result

    def get_message(self, start: int, length: int) -> NDArrayAny:
        """Get buffer section as bit array (2 bits per dibit)."""
        dibits = self.get_buffer(start, length)
        bits = np.zeros(length * 2, dtype=np.uint8)
        for i, d in enumerate(dibits):
            bits[i * 2] = (d >> 1) & 1
            bits[i * 2 + 1] = d & 1
        return bits


class P25P2SuperFrameDetector:
    """
    P25 Phase 2 SuperFrame fragment detector.

    Detects sync patterns and correctly frames 720-dibit SuperFrame fragments
    containing 4 timeslots and 4 ISCH segments.

    Reference: SDRTrunk P25P2SuperFrameDetector.java
    """

    # SuperFrame fragment length in dibits
    FRAGMENT_LENGTH = 720

    # Buffer oversize for alignment correction
    BUFFER_OVERSIZE = 2

    # Sync pattern positions within fragment
    SYNC1_POSITION = 360 + BUFFER_OVERSIZE
    SYNC2_POSITION = 540 + BUFFER_OVERSIZE

    # Thresholds
    SYNC_THRESHOLD_SYNCHRONIZED = 7
    SYNC_THRESHOLD_UNSYNCHRONIZED = 4

    # Misalignment recovery
    MISALIGNED_SYNC_DIBIT_COUNT = FRAGMENT_LENGTH - 180

    # Sync loss broadcast interval (dibits per second at 6000 baud)
    SYNC_LOSS_INTERVAL = 3720

    def __init__(self) -> None:
        # Fragment buffer (oversized for alignment)
        buffer_size = self.FRAGMENT_LENGTH + 2 * self.BUFFER_OVERSIZE
        self._fragment_buffer = DibitDelayBuffer(buffer_size)

        # Sync detection delay buffer
        self._sync_delay_buffer = DibitDelayBuffer(160 + self.BUFFER_OVERSIZE)

        # Sync detector
        self._sync_detector = P25P2SyncDetector(self.SYNC_THRESHOLD_UNSYNCHRONIZED)

        # State
        self._dibits_processed = 0
        self._synchronized = False

        # Callbacks
        self._fragment_listener: Callable[[P25P2SuperFrameFragment], None] | None = None
        self._sync_listener: Callable[[bool, int], None] | None = None

        # Timestamp tracking
        self._reference_timestamp = 0

    def set_fragment_listener(self, listener: Callable[[P25P2SuperFrameFragment], None]) -> None:
        """Set callback for assembled SuperFrame fragments."""
        self._fragment_listener = listener

    def set_sync_listener(self, listener: Callable[[bool, int], None]) -> None:
        """Set callback for sync status changes."""
        self._sync_listener = listener

    def set_timestamp(self, timestamp: int) -> None:
        """Set reference timestamp."""
        self._reference_timestamp = timestamp

    def _get_timestamp(self) -> int:
        """Calculate current timestamp."""
        if self._reference_timestamp > 0:
            return self._reference_timestamp + int(1000.0 * self._dibits_processed / 6000)
        return 0

    def process(self, dibit: int) -> None:
        """Process one dibit through the SuperFrame detector."""
        self._dibits_processed += 1

        # Add to fragment buffer
        self._fragment_buffer.put(dibit)

        if self._synchronized:
            # When synchronized, use counter-based triggering
            self._sync_delay_buffer.put(dibit)

            if self._dibits_processed >= self.FRAGMENT_LENGTH:
                self._check_fragment_sync(0)
        else:
            # When not synchronized, use sync detector
            delayed = self._sync_delay_buffer.get_and_put(dibit)
            sync_detected, bit_errors = self._sync_detector.process(delayed)

            if sync_detected:
                self._on_sync_detected(bit_errors)

        # Periodic sync loss notification
        if self._dibits_processed > self.SYNC_LOSS_INTERVAL:
            self._dibits_processed -= 3000
            if self._sync_listener:
                self._sync_listener(False, 3000)

    def _on_sync_detected(self, bit_errors: int) -> None:
        """Handle sync pattern detection."""
        self._check_fragment_sync(bit_errors)

        if self._sync_listener:
            self._sync_listener(True, bit_errors)

    def _check_fragment_sync(self, sync_detector_errors: int) -> None:
        """Check and broadcast SuperFrame fragment."""
        if self._dibits_processed <= 0:
            return

        if self._synchronized:
            # Counter-based trigger - check both sync positions
            sync1_errors = self._get_sync_bit_errors(self.SYNC1_POSITION)

            if sync1_errors <= self.SYNC_THRESHOLD_SYNCHRONIZED:
                sync2_errors = self._get_sync_bit_errors(self.SYNC2_POSITION)

                if sync2_errors <= self.SYNC_THRESHOLD_SYNCHRONIZED:
                    self._broadcast_fragment(sync1_errors + sync2_errors, 0)
                    return

            # Lost sync
            self._synchronized = False
        else:
            # Sync detector trigger - check sync 1
            sync1_errors = self._get_sync_bit_errors(self.SYNC1_POSITION)

            if sync1_errors <= self.SYNC_THRESHOLD_UNSYNCHRONIZED:
                self._synchronized = True
                self._broadcast_fragment(sync1_errors + sync_detector_errors, 0)
            else:
                # Probably misaligned - adjust and wait
                self._synchronized = True
                if self._dibits_processed > self.MISALIGNED_SYNC_DIBIT_COUNT:
                    logger.debug(f"P25P2 sync misaligned, adjusting")
                self._dibits_processed = self.MISALIGNED_SYNC_DIBIT_COUNT

    def _get_sync_bit_errors(self, position: int) -> int:
        """Calculate bit errors at a sync position."""
        dibits = self._fragment_buffer.get_buffer(position, 20)
        return P25P2SyncPattern.get_bit_error_count(dibits)

    def _broadcast_fragment(self, bit_errors: int, offset: int) -> None:
        """Broadcast a complete SuperFrame fragment."""
        self._dibits_processed = offset

        # Extract fragment dibits
        dibits = self._fragment_buffer.get_buffer(
            self.BUFFER_OVERSIZE + offset, self.FRAGMENT_LENGTH
        )

        fragment = P25P2SuperFrameFragment(
            dibits=dibits,
            timestamp=self._get_timestamp(),
            sync_bit_errors=bit_errors,
        )

        if self._fragment_listener:
            try:
                self._fragment_listener(fragment)
            except Exception as e:
                logger.error(f"Error in fragment listener: {e}")

    def reset(self) -> None:
        """Reset detector to initial state."""
        self._dibits_processed = 0
        self._synchronized = False
        self._sync_detector.reset()


class P25P2Decoder:
    """
    P25 Phase 2 Decoder with TDMA timeslot demultiplexing.

    Processes CQPSK-demodulated dibits through SuperFrame detection
    and outputs individual timeslot data.
    """

    def __init__(self) -> None:
        self._superframe_detector = P25P2SuperFrameDetector()
        self._superframe_detector.set_fragment_listener(self._on_fragment)

        # Collected timeslots
        self._timeslots: list[P25P2Timeslot] = []

        # Callbacks
        self.on_timeslot: Callable[[P25P2Timeslot], None] | None = None

    def process_dibits(
        self, dibits: NDArrayAny, soft_symbols: NDArrayAny | None = None
    ) -> list[P25P2Timeslot]:
        """Process demodulated dibits.

        Args:
            dibits: Hard decision dibits
            soft_symbols: Optional soft symbols for enhanced processing
        """
        self._timeslots.clear()

        # Feed dibits through SuperFrame detector
        for dibit in dibits:
            self._superframe_detector.process(int(dibit))

        return self._timeslots.copy()

    def _on_fragment(self, fragment: P25P2SuperFrameFragment) -> None:
        """Handle complete SuperFrame fragment."""
        # Extract 4 timeslots from fragment
        for i in range(4):
            isch, ts_dibits = fragment.get_timeslot(i)

            # Determine logical timeslot (0 or 1) based on position
            # Slots 0,2 -> timeslot 0; Slots 1,3 -> timeslot 1
            logical_slot = i % 2

            timeslot = P25P2Timeslot(
                timeslot_number=logical_slot,
                slot_type=P25P2TimeslotType.VOICE,  # Default, would need ISCH decode
                dibits=ts_dibits,
                isch_dibits=isch,
                timestamp=fragment.timestamp,
            )

            self._timeslots.append(timeslot)

            if self.on_timeslot:
                self.on_timeslot(timeslot)

    def reset(self) -> None:
        """Reset decoder state."""
        self._superframe_detector.reset()
        self._timeslots.clear()
