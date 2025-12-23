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
# Use demodulators from decoders/p25.py
from wavecapsdr.decoders.p25 import C4FMDemodulator as P25C4FMDemodulator
from wavecapsdr.decoders.p25 import CQPSKDemodulator as P25CQPSKDemodulator
from wavecapsdr.decoders.p25_frames import (
    DUID,
    NID,
    TSDUFrame,
    FRAME_SYNC_PATTERN,
    FRAME_SYNC_DIBITS,
    decode_nid,
    decode_tsdu,
)
from wavecapsdr.decoders.p25_tsbk import TSBKParser

logger = logging.getLogger(__name__)


class SyncState(str, Enum):
    """Frame synchronization state."""
    SEARCHING = "searching"  # Looking for frame sync pattern
    SYNCED = "synced"        # Locked to frame timing


class P25Modulation(str, Enum):
    """P25 modulation types."""
    C4FM = "c4fm"    # Standard non-simulcast
    LSM = "lsm"      # Linear Simulcast Modulation (CQPSK)


@dataclass
class ControlChannelMonitor:
    """P25 control channel monitor.

    Demodulates and decodes P25 control channel to extract TSBK messages.
    Supports both C4FM (standard) and LSM/CQPSK (simulcast) modulation.
    """

    protocol: TrunkingProtocol
    sample_rate: int = 48000  # Input IQ sample rate
    modulation: P25Modulation = P25Modulation.LSM  # Default to LSM for simulcast systems

    # Demodulator - either C4FM or CQPSK based on modulation
    _demod: Optional[Any] = None  # P25C4FMDemodulator or P25CQPSKDemodulator

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
        # P25 control channels can use either C4FM or LSM (CQPSK) modulation.
        # - C4FM: Standard non-simulcast systems
        # - LSM/CQPSK: Simulcast systems (like SA-GRN with 240+ sites)
        #
        # For simulcast, the same signal is transmitted from multiple sites
        # simultaneously. CQPSK/LSM is more robust to the multipath interference
        # that occurs when signals from different sites combine.
        #
        if self.modulation == P25Modulation.LSM:
            # Use CQPSK demodulator for simulcast systems
            self._demod = P25CQPSKDemodulator(
                sample_rate=self.sample_rate,
                symbol_rate=4800,
            )
            logger.info(
                f"ControlChannelMonitor: Using P25CQPSKDemodulator @ {self.sample_rate} Hz "
                f"(LSM/CQPSK for simulcast systems)"
            )
        else:
            # Use C4FM demodulator for standard systems
            self._demod = P25C4FMDemodulator(
                sample_rate=self.sample_rate,
                symbol_rate=4800,
            )
            logger.info(
                f"ControlChannelMonitor: Using P25C4FMDemodulator @ {self.sample_rate} Hz "
                f"(C4FM for standard systems)"
            )

        # Create TSBK parser
        self._tsbk_parser = TSBKParser()

        # Initialize dibit buffer
        self._dibit_buffer = []

    def reset(self) -> None:
        """Reset monitor state."""
        self.sync_state = SyncState.SEARCHING
        self._dibit_buffer.clear()
        self._sync_pattern_idx = 0
        # P25C4FMDemodulator doesn't have a reset method; it maintains
        # minimal state that resets naturally with new IQ blocks

    def process_iq(self, iq: np.ndarray) -> List[Dict[str, Any]]:
        """Process IQ samples and extract TSBK messages.

        Args:
            iq: Complex IQ samples (float32 or complex64)

        Returns:
            List of parsed TSBK results
        """
        import time as _time_mod

        if iq.size == 0:
            return []

        # DEBUG: Track call count
        if not hasattr(self, '_process_iq_calls'):
            self._process_iq_calls = 0
        self._process_iq_calls += 1
        _verbose = self._process_iq_calls <= 5

        if _verbose:
            logger.info(f"ControlChannelMonitor.process_iq: ENTRY call #{self._process_iq_calls}, iq.size={iq.size}")

        # Demodulate to dibits using C4FM (control channels always use C4FM)
        if self._demod:
            _start = _time_mod.perf_counter()
            if _verbose:
                logger.info(f"ControlChannelMonitor.process_iq: calling demodulate")
            # P25C4FMDemodulator.demodulate returns just dibits (no soft symbols)
            dibits = self._demod.demodulate(iq.astype(np.complex64))
            _elapsed = (_time_mod.perf_counter() - _start) * 1000
            if _verbose:
                logger.info(f"ControlChannelMonitor.process_iq: demodulate returned {len(dibits)} dibits in {_elapsed:.1f}ms")
        else:
            return []

        if len(dibits) == 0:
            return []

        # Debug: log dibit count periodically
        if hasattr(self, '_dibit_debug_count'):
            self._dibit_debug_count += len(dibits)
            if self._dibit_debug_count >= 10000:
                logger.info(
                    f"ControlChannelMonitor: Processed {self._dibit_debug_count} dibits, "
                    f"buffer={len(self._dibit_buffer)}, sync_state={self.sync_state.value}"
                )
                self._dibit_debug_count = 0
        else:
            self._dibit_debug_count = len(dibits)

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
            # Debug: track frame decode attempts
            if not hasattr(self, '_decode_attempts'):
                self._decode_attempts = 0
            self._decode_attempts += 1
            if self._decode_attempts % 100 == 1:
                logger.info(f"ControlChannelMonitor: Frame decode attempt {self._decode_attempts}, buffer={len(self._dibit_buffer)}")

            if len(self._dibit_buffer) < self.TSDU_FRAME_DIBITS:
                # Need more dibits
                break

            # Extract frame
            frame_dibits = np.array(self._dibit_buffer[:self.TSDU_FRAME_DIBITS], dtype=np.uint8)

            # Verify sync pattern at start of frame
            sync_dibits = self._get_sync_dibits()
            frame_sync = frame_dibits[:self.FRAME_SYNC_DIBITS]
            sync_errors = np.sum(frame_sync != sync_dibits)
            if sync_errors > 4:  # Match search tolerance
                # Lost sync
                self.sync_state = SyncState.SEARCHING
                self.sync_losses += 1
                logger.warning(f"ControlChannelMonitor: Lost frame sync (errors={sync_errors})")
                if self.on_sync_lost:
                    self.on_sync_lost()
                # Don't consume dibits, let search find next sync
                continue
            # Debug: log successful sync verification (every 50 frames)
            if not hasattr(self, '_verified_frames'):
                self._verified_frames = 0
            self._verified_frames += 1
            if self._verified_frames % 50 == 1:
                logger.info(f"ControlChannelMonitor: Sync verified (frame {self._verified_frames}, errors={sync_errors})")

            # Decode NID (Network ID) after sync
            # NID is 32 dibits of data, but there's a status symbol at position 11
            # (35 dibits from frame start = 24 sync + 11 into NID), so we need 33 dibits
            nid_dibits = frame_dibits[self.FRAME_SYNC_DIBITS:self.FRAME_SYNC_DIBITS + 33]

            # Debug: log NID dibits for first few frames
            if not hasattr(self, '_frame_count'):
                self._frame_count = 0
            self._frame_count += 1

            if self._frame_count <= 10:
                # Log DUID position dibits specifically
                duid_pos_in_frame = self.FRAME_SYNC_DIBITS + 6  # positions 30-31
                logger.info(
                    f"ControlChannelMonitor: Frame {self._frame_count}, "
                    f"frame_dibits[30:32]={list(frame_dibits[30:32])}, "
                    f"nid_dibits[6:8]={list(nid_dibits[6:8])}, "
                    f"NID dibits (33): {list(nid_dibits[:15])}..."
                )

            nid = decode_nid(nid_dibits, skip_status_at_11=True)

            # Debug: log every 10th frame for now (to see DUID distribution)
            if self._frame_count <= 5 or self._frame_count % 10 == 1:
                logger.info(
                    f"ControlChannelMonitor: Frame {self._frame_count}, NID={nid}, "
                    f"DUID={nid.duid if nid else 'None'}, TSDU={DUID.TSDU}"
                )

            if nid is not None and nid.duid == DUID.TSDU:
                # This is a TSDU frame - decode it
                logger.info(f"ControlChannelMonitor: Calling decode_tsdu on TSDU frame, frame_len={len(frame_dibits)}")
                tsdu = decode_tsdu(frame_dibits)
                if tsdu:
                    logger.info(f"ControlChannelMonitor: decode_tsdu returned {len(tsdu.tsbk_blocks) if tsdu.tsbk_blocks else 0} TSBK blocks")
                else:
                    logger.info("ControlChannelMonitor: decode_tsdu returned None")
                if tsdu and tsdu.tsbk_blocks:
                    self.frames_decoded += 1

                    # Parse each TSBK block in the frame
                    for tsbk_block in tsdu.tsbk_blocks:
                        if not tsbk_block.crc_valid:
                            continue  # Skip blocks with bad CRC

                        # Pass opcode and mfid from TSBKBlock to parser
                        result = self._parse_tsbk(tsbk_block.opcode, tsbk_block.mfid, tsbk_block.data)
                        if result:
                            results.append(result)
                            self.tsbk_decoded += 1

                        # Call TSBK callback
                        if self.on_tsbk:
                            self.on_tsbk(tsbk_block.data)

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
        sync_dibits = self._get_sync_dibits()

        # Debug: Log buffer sample every 1000 calls
        if not hasattr(self, '_find_sync_calls'):
            self._find_sync_calls = 0
        self._find_sync_calls += 1
        if self._find_sync_calls <= 3 or self._find_sync_calls % 500 == 0:
            sample = self._dibit_buffer[:min(30, len(self._dibit_buffer))]
            logger.info(
                f"ControlChannelMonitor._find_sync_in_buffer: call #{self._find_sync_calls}, "
                f"buffer_len={len(self._dibit_buffer)}, "
                f"first_30_dibits={list(sample)}, "
                f"expected_sync={list(sync_dibits[:12])}..."
            )

        # Search for pattern (allow 4 errors like P25FrameSync)
        for i in range(len(self._dibit_buffer) - self.FRAME_SYNC_DIBITS + 1):
            match = True
            errors = 0
            max_errors = 4  # Allow up to 4 dibit errors (matches P25FrameSync.sync_threshold)

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
        return errors <= 4  # Allow up to 4 dibit errors (matches P25FrameSync)

    def _get_sync_dibits(self) -> np.ndarray:
        """Get P25 frame sync pattern as dibits.

        P25 uses the SAME 48-bit sync pattern for ALL frame types (HDU, TSDU, LDU, etc).
        The frame type (DUID) is in the NID that follows the sync pattern.

        Per TIA-102.BAAA constellation mapping:
        C4FM symbols: +3 +3 +3 +3 +3 -3 +3 +3 -3 -3 +3 +3 -3 -3 -3 -3 +3 -3 +3 -3 -3 -3 -3 -3

        Correct dibit encoding:
        +3 symbol -> dibit 1 (binary 01)
        -3 symbol -> dibit 3 (binary 11)

        This matches SDRTrunk's pattern: 0x5575F5FF77FF
        """
        # Updated to match correct P25 constellation mapping
        return np.array([1, 1, 1, 1, 1, 3, 1, 1, 3, 3, 1, 1,
                         3, 3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3], dtype=np.uint8)

    def _parse_tsbk(self, opcode: int, mfid: int, data: bytes) -> Optional[Dict[str, Any]]:
        """Parse TSBK message.

        Args:
            opcode: TSBK opcode (6 bits)
            mfid: Manufacturer ID (8 bits, 0 = standard)
            data: TSBK data payload (8 bytes)

        Returns:
            Parsed TSBK dictionary or None
        """
        if self._tsbk_parser is None:
            return None

        try:
            return self._tsbk_parser.parse(opcode, mfid, data)
        except Exception as e:
            logger.error(f"ControlChannelMonitor: TSBK parse error (opcode=0x{opcode:02X}): {e}")
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
    modulation: Optional[P25Modulation] = None,
) -> ControlChannelMonitor:
    """Create a control channel monitor.

    Args:
        protocol: P25 protocol (Phase I or II)
        sample_rate: Input sample rate in Hz
        modulation: Override modulation type (None = use default for protocol)

    Returns:
        Configured ControlChannelMonitor
    """
    # Determine modulation based on protocol if not specified
    if modulation is None:
        # Phase 2 uses CQPSK/LSM, Phase 1 typically uses C4FM
        # But many simulcast systems use LSM even for Phase 1
        modulation = P25Modulation.LSM  # Default to LSM for compatibility

    return ControlChannelMonitor(
        protocol=protocol,
        sample_rate=sample_rate,
        modulation=modulation,
    )
