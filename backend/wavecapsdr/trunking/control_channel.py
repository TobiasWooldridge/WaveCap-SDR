"""ControlChannelMonitor - P25 control channel decoder.

This module implements the control channel monitor that:
- Demodulates P25 C4FM (Phase I) or CQPSK (Phase II) signal
- Synchronizes to frame boundaries
- Decodes TSDU (Trunking Signaling Data Unit) frames
- Extracts TSBK messages and passes them to TrunkingSystem

The control channel carries trunking signaling:
- Voice channel grants (GRP_V_CH_GRANT, UU_V_CH_GRANT)
- Channel identifiers (IDEN_UP) for frequency calculation
- System status (RFSS_STS_BCAST, NET_STS_BCAST)
- Adjacent site info (ADJ_STS_BCAST)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from wavecapsdr.trunking.config import TrunkingProtocol
from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator
from wavecapsdr.dsp.p25.cqpsk import CQPSKDemodulator
from wavecapsdr.decoders.p25_frames import (
    DUID,
    NID,
    TSDUFrame,
    FRAME_SYNC_PATTERNS,
    decode_nid,
    decode_tsdu,
)
from wavecapsdr.decoders.p25_tsbk import TSBKParser

logger = logging.getLogger(__name__)


class SyncState(str, Enum):
    """Frame synchronization state."""
    SEARCHING = "searching"  # Looking for frame sync pattern
    SYNCED = "synced"        # Locked to frame timing


@dataclass
class ControlChannelMonitor:
    """P25 control channel monitor.

    Demodulates and decodes P25 control channel to extract TSBK messages.
    Supports both Phase I (C4FM) and Phase II (CQPSK) systems.
    """

    protocol: TrunkingProtocol
    sample_rate: int = 48000  # Input IQ sample rate

    # Demodulators
    _c4fm_demod: Optional[C4FMDemodulator] = None
    _cqpsk_demod: Optional[CQPSKDemodulator] = None

    # TSBK parser
    _tsbk_parser: Optional[TSBKParser] = None

    # Frame sync state
    sync_state: SyncState = SyncState.SEARCHING
    _dibit_buffer: List[int] = field(default_factory=list)
    _sync_pattern_idx: int = 0

    # Statistics
    frames_decoded: int = 0
    tsbk_decoded: int = 0
    sync_losses: int = 0
    _last_sync_time: float = 0.0

    # Callbacks
    on_tsbk: Optional[Callable[[bytes], None]] = None
    on_sync_acquired: Optional[Callable[[], None]] = None
    on_sync_lost: Optional[Callable[[], None]] = None

    # Constants
    FRAME_SYNC_DIBITS: int = 24  # 48-bit sync word = 24 dibits
    TSDU_FRAME_DIBITS: int = 360  # Total TSDU frame length in dibits
    SYNC_TIMEOUT: float = 2.0  # Seconds before declaring sync lost

    def __post_init__(self) -> None:
        """Initialize demodulators and parser."""
        # Create appropriate demodulator
        if self.protocol == TrunkingProtocol.P25_PHASE1:
            self._c4fm_demod = C4FMDemodulator(
                sample_rate=self.sample_rate,
                symbol_rate=4800,  # P25 Phase I: 4800 baud
            )
            logger.info(f"ControlChannelMonitor: Using C4FM demodulator @ {self.sample_rate} Hz")
        else:
            self._cqpsk_demod = CQPSKDemodulator(
                sample_rate=self.sample_rate,
                symbol_rate=6000,  # P25 Phase II: 6000 symbols/sec (12000 baud TDMA)
            )
            logger.info(f"ControlChannelMonitor: Using CQPSK demodulator @ {self.sample_rate} Hz")

        # Create TSBK parser
        self._tsbk_parser = TSBKParser()

        # Initialize dibit buffer
        self._dibit_buffer = []

    def reset(self) -> None:
        """Reset monitor state."""
        self.sync_state = SyncState.SEARCHING
        self._dibit_buffer.clear()
        self._sync_pattern_idx = 0

        if self._c4fm_demod:
            self._c4fm_demod.reset()
        if self._cqpsk_demod:
            self._cqpsk_demod.reset()

    def process_iq(self, iq: np.ndarray) -> List[Dict[str, Any]]:
        """Process IQ samples and extract TSBK messages.

        Args:
            iq: Complex IQ samples (float32 or complex64)

        Returns:
            List of parsed TSBK results
        """
        if iq.size == 0:
            return []

        # Demodulate to dibits
        if self._c4fm_demod:
            dibits, soft = self._c4fm_demod.demodulate(iq.astype(np.complex64))
        elif self._cqpsk_demod:
            dibits = self._cqpsk_demod.demodulate(iq.astype(np.complex64))
        else:
            return []

        if len(dibits) == 0:
            return []

        # Process dibits
        return self._process_dibits(dibits)

    def _process_dibits(self, dibits: np.ndarray) -> List[Dict[str, Any]]:
        """Process demodulated dibits and extract TSBK messages.

        Args:
            dibits: Array of dibits (0-3)

        Returns:
            List of parsed TSBK results
        """
        results: List[Dict[str, Any]] = []

        # Add to buffer
        self._dibit_buffer.extend(dibits.tolist())

        # Process buffer
        while True:
            if self.sync_state == SyncState.SEARCHING:
                # Look for frame sync pattern
                sync_idx = self._find_sync_in_buffer()
                if sync_idx < 0:
                    # Not found, keep last (FRAME_SYNC_DIBITS - 1) dibits
                    if len(self._dibit_buffer) > self.FRAME_SYNC_DIBITS:
                        self._dibit_buffer = self._dibit_buffer[-(self.FRAME_SYNC_DIBITS - 1):]
                    break

                # Found sync pattern
                self._dibit_buffer = self._dibit_buffer[sync_idx:]
                self.sync_state = SyncState.SYNCED
                self._last_sync_time = time.time()

                logger.info("ControlChannelMonitor: Frame sync acquired")
                if self.on_sync_acquired:
                    self.on_sync_acquired()

            # In synced state, try to decode frame
            if len(self._dibit_buffer) < self.TSDU_FRAME_DIBITS:
                # Need more dibits
                break

            # Extract frame
            frame_dibits = np.array(self._dibit_buffer[:self.TSDU_FRAME_DIBITS], dtype=np.uint8)

            # Verify sync pattern at start of frame
            if not self._verify_sync(frame_dibits[:self.FRAME_SYNC_DIBITS]):
                # Lost sync
                self.sync_state = SyncState.SEARCHING
                self.sync_losses += 1
                logger.warning("ControlChannelMonitor: Lost frame sync")
                if self.on_sync_lost:
                    self.on_sync_lost()
                # Don't consume dibits, let search find next sync
                continue

            # Decode NID (Network ID) after sync
            nid_dibits = frame_dibits[self.FRAME_SYNC_DIBITS:self.FRAME_SYNC_DIBITS + 32]
            nid = decode_nid(nid_dibits)

            if nid is not None and nid.duid == DUID.TSDU:
                # This is a TSDU frame - decode it
                tsdu = decode_tsdu(frame_dibits)
                if tsdu and tsdu.tsbk_blocks:
                    self.frames_decoded += 1

                    # Parse each TSBK block in the frame
                    for tsbk_block in tsdu.tsbk_blocks:
                        if not tsbk_block.crc_valid:
                            continue  # Skip blocks with bad CRC

                        tsbk_bytes = tsbk_block.data
                        result = self._parse_tsbk(tsbk_bytes)
                        if result:
                            results.append(result)
                            self.tsbk_decoded += 1

                        # Call TSBK callback
                        if self.on_tsbk:
                            self.on_tsbk(tsbk_bytes)

            # Consume frame from buffer
            self._dibit_buffer = self._dibit_buffer[self.TSDU_FRAME_DIBITS:]
            self._last_sync_time = time.time()

        # Check for sync timeout
        if self.sync_state == SyncState.SYNCED:
            if time.time() - self._last_sync_time > self.SYNC_TIMEOUT:
                self.sync_state = SyncState.SEARCHING
                self.sync_losses += 1
                logger.warning("ControlChannelMonitor: Sync timeout")
                if self.on_sync_lost:
                    self.on_sync_lost()

        return results

    def _find_sync_in_buffer(self) -> int:
        """Find frame sync pattern in dibit buffer.

        Returns:
            Index of sync pattern start, or -1 if not found
        """
        if len(self._dibit_buffer) < self.FRAME_SYNC_DIBITS:
            return -1

        # Convert expected sync pattern to dibits
        # P25 frame sync: 0x5575F5FF77FF (48 bits = 24 dibits)
        sync_dibits = self._get_sync_dibits()

        # Search for pattern
        for i in range(len(self._dibit_buffer) - self.FRAME_SYNC_DIBITS + 1):
            match = True
            errors = 0
            max_errors = 2  # Allow up to 2 dibit errors

            for j in range(self.FRAME_SYNC_DIBITS):
                if self._dibit_buffer[i + j] != sync_dibits[j]:
                    errors += 1
                    if errors > max_errors:
                        match = False
                        break

            if match:
                return i

        return -1

    def _verify_sync(self, dibits: np.ndarray) -> bool:
        """Verify that dibits match sync pattern.

        Args:
            dibits: First 24 dibits of frame

        Returns:
            True if sync pattern matches (with error tolerance)
        """
        sync_dibits = self._get_sync_dibits()
        errors = np.sum(dibits != sync_dibits)
        return errors <= 3  # Allow up to 3 dibit errors

    def _get_sync_dibits(self) -> np.ndarray:
        """Get P25 frame sync pattern as dibits.

        The P25 frame sync is: 0x5575F5FF77FF (48 bits)
        This maps to specific frequency deviations in C4FM.
        """
        # Frame sync word: 0x5575F5FF77FF
        # Bit pairs (MSB first) -> dibits
        sync_hex = 0x5575F5FF77FF
        dibits = []
        for i in range(23, -1, -1):
            dibit = (sync_hex >> (i * 2)) & 0x3
            dibits.append(dibit)
        return np.array(dibits, dtype=np.uint8)

    def _parse_tsbk(self, tsbk_data: bytes) -> Optional[Dict[str, Any]]:
        """Parse TSBK message.

        Args:
            tsbk_data: Raw TSBK data (12 bytes)

        Returns:
            Parsed TSBK dictionary or None
        """
        if self._tsbk_parser is None:
            return None

        try:
            return self._tsbk_parser.parse(tsbk_data)
        except Exception as e:
            logger.error(f"ControlChannelMonitor: TSBK parse error: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get monitor statistics."""
        return {
            "sync_state": self.sync_state.value,
            "frames_decoded": self.frames_decoded,
            "tsbk_decoded": self.tsbk_decoded,
            "sync_losses": self.sync_losses,
            "buffer_dibits": len(self._dibit_buffer),
        }


def create_control_monitor(
    protocol: TrunkingProtocol,
    sample_rate: int = 48000,
) -> ControlChannelMonitor:
    """Create a control channel monitor.

    Args:
        protocol: P25 protocol (Phase I or II)
        sample_rate: Input sample rate in Hz

    Returns:
        Configured ControlChannelMonitor
    """
    return ControlChannelMonitor(
        protocol=protocol,
        sample_rate=sample_rate,
    )
