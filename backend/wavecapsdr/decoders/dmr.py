"""DMR (Digital Mobile Radio) decoder with trunking support."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, cast

import numpy as np
from wavecapsdr.typing import NDArrayComplex, NDArrayFloat, NDArrayInt

logger = logging.getLogger(__name__)

from wavecapsdr.validation import validate_finite_array


class DMRSlot(Enum):
    """DMR time slot"""

    SLOT1 = 1
    SLOT2 = 2


class DMRFrameType(Enum):
    """DMR frame types"""

    VOICE = "Voice"
    DATA = "Data"
    CSBK = "Control Signaling Block"
    UNKNOWN = "Unknown"


@dataclass
class DMRFrame:
    """Decoded DMR frame"""

    frame_type: DMRFrameType
    slot: DMRSlot
    color_code: int  # 0-15
    src_id: int | None = None  # Source radio ID
    dst_id: int | None = None  # Destination (talkgroup) ID
    voice_data: bytes | None = None  # AMBE voice frames
    csbk_opcode: int | None = None
    csbk_data: dict[str, Any] | None = None


class DMR4FSKDemodulator:
    """4-FSK demodulator for DMR"""

    def __init__(self, sample_rate: int = 48000, symbol_rate: int = 4800):
        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.samples_per_symbol = sample_rate // symbol_rate

        # DMR deviation: ±1944 Hz
        self.deviations = np.array([-1944, -648, 648, 1944])

    def demodulate(self, iq: NDArrayComplex) -> NDArrayInt:
        """Demodulate to dibits"""
        if iq.size == 0:
            return np.array([], dtype=np.uint8)

        x: NDArrayComplex = iq.astype(np.complex64, copy=False)
        prod = x[1:] * np.conj(x[:-1])
        inst_freq = cast(NDArrayFloat, np.angle(prod)) * self.sample_rate / (2 * np.pi)

        num_symbols = len(inst_freq) // self.samples_per_symbol
        symbols = np.zeros(num_symbols, dtype=np.uint8)

        for i in range(num_symbols):
            start_idx = i * self.samples_per_symbol
            end_idx = start_idx + self.samples_per_symbol
            symbol_freq = float(np.mean(inst_freq[start_idx:end_idx]))
            distances = np.abs(self.deviations - symbol_freq)
            symbols[i] = int(np.argmin(distances))

        return cast(NDArrayInt, symbols)


class DMRDecoder:
    """DMR Tier 2/3 decoder with trunking support"""

    # DMR sync patterns
    BS_SOURCED_VOICE = 0x755FD7DF75F7  # Base station voice
    MS_SOURCED_VOICE = 0xDFF57D75DF5D  # Mobile station voice
    BS_SOURCED_DATA = 0xDFD55D7FDD57  # Base station data
    MS_SOURCED_DATA = 0x57F757DD5DFF  # Mobile station data

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.demodulator = DMR4FSKDemodulator(sample_rate)

        # Callbacks
        self.on_voice_frame: Callable[[int, int, bytes], None] | None = None  # (slot, tgid, voice)
        self.on_csbk_message: Callable[[dict[str, Any]], None] | None = None

        logger.info(f"DMR decoder initialized (sample_rate={sample_rate})")

    def process_iq(self, iq: NDArrayComplex) -> list[DMRFrame]:
        """Process IQ and decode DMR frames"""
        if iq.size == 0:
            return []
        if not validate_finite_array(iq):
            logger.warning("DMR: non-finite IQ samples, dropping")
            return []
        dibits = self.demodulator.demodulate(iq)

        if len(dibits) == 0:
            return []

        # Find DMR sync (simplified)
        sync_pos = self._find_sync(dibits)
        if sync_pos is None:
            return []

        # Extract frame
        frame_dibits = dibits[sync_pos:]
        return self._decode_frame(frame_dibits)

    def _find_sync(self, dibits: NDArrayInt) -> int | None:
        """Find DMR sync pattern"""
        # Simplified sync search
        if len(dibits) < 24:  # 48 bits sync
            return None
        # Would implement proper sync detection
        return 0  # Placeholder

    def _decode_frame(self, dibits: NDArrayInt) -> list[DMRFrame]:
        """Decode DMR frame"""
        if len(dibits) < 144:  # DMR burst is 288 bits
            return []

        # Extract slot info, color code, etc.
        # This is highly simplified

        color_code = dibits[0] & 0x0F
        slot = DMRSlot.SLOT1 if (dibits[1] & 1) == 0 else DMRSlot.SLOT2

        # Extract IDs (simplified)
        src_id = 0
        dst_id = 0
        for i in range(12):
            src_id = (src_id << 2) | dibits[10 + i]
            dst_id = (dst_id << 2) | dibits[22 + i]

        logger.debug(f"DMR: Slot={slot.value} CC={color_code} SRC={src_id} DST={dst_id}")

        frame = DMRFrame(
            frame_type=DMRFrameType.VOICE,
            slot=slot,
            color_code=color_code,
            src_id=src_id,
            dst_id=dst_id,
        )

        return [frame]
