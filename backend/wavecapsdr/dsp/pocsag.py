"""POCSAG (Post Office Code Standardisation Advisory Group) decoder.

POCSAG is a paging protocol used for one-way numeric and alphanumeric messaging.
- Data rate: 512, 1200, or 2400 baud (most common: 1200 baud)
- FSK modulation with +/- 4.5 kHz deviation
- 32-bit codewords in batches of 16 (2 frames of 8 codewords)

Key data types decoded:
- Address codewords: 21-bit address + 2-bit function code
- Message codewords: Numeric (BCD) or alphanumeric (7-bit ASCII)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np
from wavecapsdr.typing import NDArrayAny

# POCSAG Constants
POCSAG_SYNC_CODEWORD = 0x7CD215D8  # Sync pattern
POCSAG_IDLE_CODEWORD = 0x7A89C197  # Idle pattern

# BCH(31,21) generator polynomial: x^10 + x^9 + x^8 + x^6 + x^5 + x^3 + 1
BCH_POLY = 0x769

# Numeric character table for BCD encoding
NUMERIC_CHARS = "0123456789*U -)(  "


class MessageType(IntEnum):
    """POCSAG message type based on function code."""
    NUMERIC = 0
    ALPHA = 1
    ALERT_ONLY = 2
    ALPHA_2 = 3


@dataclass
class POCSAGMessage:
    """A decoded POCSAG message."""

    address: int  # 21-bit address (capcode)
    function: int  # 2-bit function code (0-3)
    message_type: MessageType
    message: str  # Decoded message content
    timestamp: float = field(default_factory=time.time)
    baud_rate: int = 1200

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "address": self.address,
            "function": self.function,
            "messageType": self.message_type.name.lower(),
            "message": self.message,
            "timestamp": self.timestamp,
            "baudRate": self.baud_rate,
        }


def _bch_check(codeword: int) -> bool:
    """Check BCH(31,21) error correction code.

    Returns True if codeword is valid (or correctable).
    """
    # Simple syndrome check (no error correction implemented)
    syndrome = 0
    for i in range(31):
        if (codeword >> (30 - i)) & 1:
            syndrome ^= BCH_POLY << (20 - i) if i < 21 else BCH_POLY >> (i - 20)
    syndrome &= 0x3FF  # 10-bit syndrome

    # Also verify even parity
    parity = bin(codeword).count('1') & 1

    return syndrome == 0 and parity == 0


def _decode_numeric(data_bits: list[int]) -> str:
    """Decode BCD numeric message."""
    result: list[str] = []
    # Process 4 bits at a time (BCD)
    for i in range(0, len(data_bits) - 3, 4):
        value = (data_bits[i] << 3) | (data_bits[i+1] << 2) | (data_bits[i+2] << 1) | data_bits[i+3]
        if value < len(NUMERIC_CHARS):
            char = NUMERIC_CHARS[value]
            if char != ' ' or (result and result[-1] != ' '):
                result.append(char)

    return ''.join(result).strip()


def _decode_alpha(data_bits: list[int]) -> str:
    """Decode 7-bit ASCII alphanumeric message."""
    result = []
    # Process 7 bits at a time
    for i in range(0, len(data_bits) - 6, 7):
        value = 0
        for j in range(7):
            value |= data_bits[i + j] << (6 - j)

        # Valid printable ASCII
        if 32 <= value <= 126:
            result.append(chr(value))
        elif value == 0:
            break  # End of message

    return ''.join(result).strip()


class POCSAGDecoder:
    """Real-time POCSAG decoder.

    Takes FSK-demodulated audio and extracts POCSAG messages.
    The input should be from an NBFM receiver tuned to a paging frequency.
    """

    def __init__(self, sample_rate: int = 48000, baud_rate: int = 1200):
        """Initialize POCSAG decoder.

        Args:
            sample_rate: Audio sample rate in Hz
            baud_rate: POCSAG baud rate (512, 1200, or 2400)
        """
        self.sample_rate = sample_rate
        self.baud_rate = baud_rate
        self.samples_per_bit = sample_rate / baud_rate

        # Bit recovery state
        self._bit_buffer: list[int] = []
        self._sample_buffer = np.array([], dtype=np.float32)
        self._last_zero_crossing = 0
        self._phase = 0.0

        # Frame sync state
        self._synced = False
        self._sync_word_buffer = 0
        self._codeword_buffer: list[int] = []

        # Current message state
        self._current_address: int | None = None
        self._current_function: int = 0
        self._data_bits: list[int] = []

        # Decoded messages queue
        self.messages: list[POCSAGMessage] = []
        self._max_messages = 100

    def process(self, audio: NDArrayAny) -> list[POCSAGMessage]:
        """Process audio samples and extract POCSAG messages.

        Args:
            audio: Audio samples (float32, mono)

        Returns:
            List of newly decoded messages (may be empty)
        """
        if audio.size == 0:
            return []

        new_messages: list[POCSAGMessage] = []

        # Append to sample buffer
        self._sample_buffer = np.concatenate([self._sample_buffer, audio.astype(np.float32)])

        # Process samples into bits using zero-crossing detection
        while len(self._sample_buffer) >= self.samples_per_bit * 2:
            # Look for zero crossings
            bit_samples = int(self.samples_per_bit)
            chunk = self._sample_buffer[:bit_samples * 2]

            # Simple bit slicer: compare average of first half vs second half
            avg = np.mean(chunk)
            bit = 1 if avg > 0 else 0

            self._bit_buffer.append(bit)
            self._sample_buffer = self._sample_buffer[bit_samples:]

            # Try to sync and decode
            if len(self._bit_buffer) >= 32:
                msgs = self._try_sync_and_decode()
                new_messages.extend(msgs)

            # Prevent buffer from growing unbounded
            if len(self._bit_buffer) > 1024:
                self._bit_buffer = self._bit_buffer[-512:]

        # Limit sample buffer size
        if len(self._sample_buffer) > self.sample_rate * 2:
            self._sample_buffer = self._sample_buffer[-self.sample_rate:]

        # Add to message history
        for msg in new_messages:
            self.messages.append(msg)
            if len(self.messages) > self._max_messages:
                self.messages.pop(0)

        return new_messages

    def _try_sync_and_decode(self) -> list[POCSAGMessage]:
        """Try to sync to POCSAG stream and decode codewords."""
        messages: list[POCSAGMessage] = []

        # Build 32-bit word from buffer
        if len(self._bit_buffer) < 32:
            return messages

        word = 0
        for i in range(32):
            word = (word << 1) | self._bit_buffer[i]

        # Look for sync word
        if not self._synced:
            # Check for sync word (allow bit inversions)
            if word == POCSAG_SYNC_CODEWORD or word == (~POCSAG_SYNC_CODEWORD & 0xFFFFFFFF):
                self._synced = True
                self._codeword_buffer = []
                self._bit_buffer = self._bit_buffer[32:]
                return messages

            # Shift by one bit
            self._bit_buffer = self._bit_buffer[1:]
            return messages

        # We're synced - decode codewords
        if len(self._bit_buffer) >= 32:
            codeword = 0
            for i in range(32):
                codeword = (codeword << 1) | self._bit_buffer[i]

            self._bit_buffer = self._bit_buffer[32:]

            # Check for sync word (start of new batch)
            if codeword == POCSAG_SYNC_CODEWORD or codeword == (~POCSAG_SYNC_CODEWORD & 0xFFFFFFFF):
                # Flush any pending message
                if self._current_address is not None and self._data_bits:
                    msg = self._flush_message()
                    if msg:
                        messages.append(msg)
                return messages

            # Check for idle
            if codeword == POCSAG_IDLE_CODEWORD or codeword == (~POCSAG_IDLE_CODEWORD & 0xFFFFFFFF):
                # Flush any pending message
                if self._current_address is not None and self._data_bits:
                    msg = self._flush_message()
                    if msg:
                        messages.append(msg)
                return messages

            # Validate BCH
            if not _bch_check(codeword):
                # Bad codeword - might have lost sync
                self._synced = False
                if self._current_address is not None and self._data_bits:
                    msg = self._flush_message()
                    if msg:
                        messages.append(msg)
                return messages

            # Decode codeword
            is_address = not (codeword & 0x80000000)

            if is_address:
                # Flush any pending message first
                if self._current_address is not None and self._data_bits:
                    msg = self._flush_message()
                    if msg:
                        messages.append(msg)

                # Extract address (bits 30-10) and function (bits 9-8)
                self._current_address = (codeword >> 10) & 0x1FFFFF
                self._current_function = (codeword >> 8) & 0x3
                self._data_bits = []
            else:
                # Message codeword - extract 20 data bits (bits 30-11)
                if self._current_address is not None:
                    for i in range(20):
                        self._data_bits.append((codeword >> (30 - i)) & 1)

        return messages

    def _flush_message(self) -> POCSAGMessage | None:
        """Flush the current message buffer and create a message object."""
        if self._current_address is None:
            return None

        msg_type = MessageType(self._current_function)

        # Decode based on function code
        if not self._data_bits:
            message = ""
        elif msg_type == MessageType.NUMERIC:
            message = _decode_numeric(self._data_bits)
        else:
            message = _decode_alpha(self._data_bits)

        result = POCSAGMessage(
            address=self._current_address,
            function=self._current_function,
            message_type=msg_type,
            message=message,
            baud_rate=self.baud_rate,
        )

        # Reset state
        self._current_address = None
        self._current_function = 0
        self._data_bits = []

        return result

    def reset(self) -> None:
        """Reset decoder state."""
        self._bit_buffer = []
        self._sample_buffer = np.array([], dtype=np.float32)
        self._synced = False
        self._sync_word_buffer = 0
        self._codeword_buffer = []
        self._current_address = None
        self._current_function = 0
        self._data_bits = []

    def get_messages(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent decoded messages as dicts.

        Args:
            limit: Maximum number of messages to return

        Returns:
            List of message dictionaries (most recent first)
        """
        return [msg.to_dict() for msg in reversed(self.messages[-limit:])]
