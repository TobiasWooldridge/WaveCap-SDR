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
from typing import Any, Callable

import numpy as np
from wavecapsdr.typing import NDArrayComplex, NDArrayFloat, NDArrayInt

# Use demodulators from decoders/p25.py
from wavecapsdr.decoders.p25 import CQPSKDemodulator as P25CQPSKDemodulator
from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator as DSPC4FMDemodulator
from wavecapsdr.decoders.nac_tracker import NACTracker
from wavecapsdr.decoders.p25_frames import (
    DUID,
    decode_nid,
    decode_tsdu,
)
from wavecapsdr.decoders.p25_tsbk import TSBKParser
from wavecapsdr.utils.profiler import get_profiler

# Profiler for control channel processing (disabled by default to reduce overhead)
_cc_profiler = get_profiler("ControlChannel", enabled=False)
from wavecapsdr.trunking.config import TrunkingProtocol

logger = logging.getLogger(__name__)


class SyncState(str, Enum):
    """Frame synchronization state."""

    SEARCHING = "searching"  # Looking for frame sync pattern
    SYNCED = "synced"  # Locked to frame timing


class P25Modulation(str, Enum):
    """P25 modulation types."""

    C4FM = "c4fm"  # Standard non-simulcast
    LSM = "lsm"  # Linear Simulcast Modulation (CQPSK)
    C4FM_WIDE = "c4fm_wide"  # Wide-pulse C4FM (older Motorola ASTRO simulcast)


@dataclass
class ControlChannelMonitor:
    """P25 control channel monitor.

    Demodulates and decodes P25 control channel to extract TSBK messages.
    Supports both C4FM (standard) and LSM/CQPSK (simulcast) modulation.
    """

    protocol: TrunkingProtocol
    sample_rate: int = 19200  # Input IQ sample rate (~4 SPS, matches SDRTrunk)
    modulation: P25Modulation = P25Modulation.C4FM  # Default to C4FM (most common)

    # Demodulator - either C4FM or CQPSK based on modulation
    _demod: Any | None = None  # P25C4FMDemodulator or P25CQPSKDemodulator
    _soft_buffer: list[float] = field(default_factory=list)

    # TSBK parser
    _tsbk_parser: TSBKParser | None = None

    # NAC tracker for intelligent NID recovery (port of SDRTrunk NACTracker)
    _nac_tracker: NACTracker = field(default_factory=NACTracker)

    # Frame sync state
    sync_state: SyncState = SyncState.SEARCHING
    _dibit_buffer: list[int] = field(default_factory=list)
    _sync_pattern_idx: int = 0

    # Statistics
    frames_decoded: int = 0
    tsbk_decoded: int = 0
    sync_losses: int = 0
    _last_sync_time: float = 0.0

    # TSBK decode statistics (for debugging FEC issues)
    tsbk_attempts: int = 0  # Number of TSBK blocks we tried to decode
    tsbk_crc_pass: int = 0  # Number that passed CRC
    tsbk_error_sum: int = 0  # Sum of bit errors for failed CRCs (for averaging)
    _dibit_debug_count: int = 0
    tsbk_rejected: int = 0  # TSBKs rejected due to invalid fields

    # Detailed diagnostics for debugging decode chain
    _diag_duid_histogram: dict[int, int] = field(default_factory=dict)  # DUID value -> count
    _diag_bch_errors: list[int] = field(default_factory=list)  # BCH error counts
    _diag_trellis_metrics: list[int] = field(default_factory=list)  # Trellis error metrics

    # Callbacks
    on_tsbk: Callable[[bytes], None] | None = None
    on_sync_acquired: Callable[[], None] | None = None
    on_sync_lost: Callable[[], None] | None = None

    # Constants
    FRAME_SYNC_DIBITS: int = 24  # 48-bit sync word = 24 dibits
    TSDU_FRAME_DIBITS: int = 360  # Total TSDU frame length in dibits
    SYNC_TIMEOUT: float = 1.0  # Seconds before declaring sync lost (SDRTrunk: 4800 dibits = 1 sec)

    def __post_init__(self) -> None:
        """Initialize demodulators and parser."""
        if self.sample_rate <= 0:
            raise ValueError(
                f"ControlChannelMonitor: sample_rate must be > 0 (got {self.sample_rate})"
            )

        # P25 control channels can use either C4FM or LSM (CQPSK) modulation.
        # - C4FM: Standard non-simulcast systems
        # - C4FM_WIDE: Wide-pulse C4FM (older Motorola ASTRO simulcast, not P25 compliant)
        # - LSM/CQPSK: Simulcast systems (like SA-GRN with 240+ sites)
        #
        # For simulcast, the same signal is transmitted from multiple sites
        # simultaneously. CQPSK/LSM is more robust to the multipath interference
        # that occurs when signals from different sites combine.
        #
        # Wide-pulse C4FM was used by older Motorola ASTRO equipment (~2000 era)
        # and uses a different pulse-shaping filter that spreads energy wider.
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
        elif self.modulation == P25Modulation.C4FM_WIDE:
            # Use DSP C4FM demodulator with wide-pulse mode for older Motorola ASTRO
            self._demod = DSPC4FMDemodulator(
                sample_rate=self.sample_rate,
                symbol_rate=4800,
                wide_pulse=True,
            )
            logger.info(
                f"ControlChannelMonitor: Using DSP C4FMDemodulator @ {self.sample_rate} Hz "
                f"(wide-pulse C4FM for older Motorola ASTRO simulcast)"
            )
        else:
            # Use DSP C4FM demodulator with soft symbols for trellis decoding
            self._demod = DSPC4FMDemodulator(
                sample_rate=self.sample_rate,
                symbol_rate=4800,
            )
            logger.info(
                f"ControlChannelMonitor: Using DSP C4FMDemodulator @ {self.sample_rate} Hz "
                f"(C4FM for standard systems, soft symbols enabled)"
            )

        # Create TSBK parser
        self._tsbk_parser = TSBKParser()

        # Initialize dibit buffer
        self._dibit_buffer = []
        self._soft_buffer = []

        # Polarity reversal tracking (OP25-style)
        # 0 = normal polarity, 2 = reversed (XOR mask to apply)
        self._reverse_p: int = 0

    def reset(self, preserve_polarity: bool = False) -> None:
        """Reset monitor state.

        Args:
            preserve_polarity: If True, keep the latched polarity setting.
                              Use this when hunting/roaming within same system.
                              If False (default), fully reset polarity detection.
        """
        self.sync_state = SyncState.SEARCHING
        self._dibit_buffer.clear()
        self._soft_buffer.clear()
        self._sync_pattern_idx = 0
        if not preserve_polarity:
            self._reverse_p = 0
            self._polarity_latched = False
        self._nac_tracker.reset()  # Clear NAC tracking on reset
        # P25C4FMDemodulator doesn't have a reset method; it maintains
        # minimal state that resets naturally with new IQ blocks

    def process_iq(self, iq: NDArrayComplex) -> list[dict[str, Any]]:
        """Process IQ samples and extract TSBK messages.

        Args:
            iq: Complex IQ samples (float32 or complex64)

        Returns:
            List of parsed TSBK results
        """
        import time as _time_mod

        if iq.size == 0:
            return []
        if not np.isfinite(iq).all():
            raise ValueError("ControlChannelMonitor.process_iq: non-finite IQ samples")

        # DEBUG: Track call count
        if not hasattr(self, "_process_iq_calls"):
            self._process_iq_calls = 0
        self._process_iq_calls += 1
        _verbose = self._process_iq_calls <= 5

        if _verbose:
            logger.debug(
                f"ControlChannelMonitor.process_iq: ENTRY call #{self._process_iq_calls}, iq.size={iq.size}"
            )

        # Demodulate to dibits using C4FM (control channels always use C4FM)
        soft: NDArrayFloat | None = None
        if self._demod:
            if _verbose:
                logger.debug("ControlChannelMonitor.process_iq: calling demodulate")
            with _cc_profiler.measure("demodulate"):
                if isinstance(self._demod, DSPC4FMDemodulator):
                    dibits, soft = self._demod.demodulate(iq.astype(np.complex64))
                else:
                    # CQPSK demodulator returns dibits only
                    dibits = self._demod.demodulate(iq.astype(np.complex64))
            if _verbose:
                logger.debug(
                    f"ControlChannelMonitor.process_iq: demodulate returned {len(dibits)} dibits"
                )
        else:
            return []

        if len(dibits) == 0:
            return []

        # Debug: log dibit count periodically (DEBUG level, less frequent)
        if hasattr(self, "_dibit_debug_count"):
            self._dibit_debug_count += len(dibits)
            if self._dibit_debug_count >= 50000:  # Reduced frequency
                logger.debug(
                    f"ControlChannelMonitor: Processed {self._dibit_debug_count} dibits, "
                    f"buffer={len(self._dibit_buffer)}, sync_state={self.sync_state.value}"
                )
                self._dibit_debug_count = 0
        else:
            self._dibit_debug_count = len(dibits)

        # Process dibits
        with _cc_profiler.measure("process_dibits"):
            results = self._process_dibits(dibits, soft)

        # Report profiling periodically
        _cc_profiler.report()
        return results

    def _process_dibits(
        self, dibits: NDArrayInt, soft: NDArrayFloat | None
    ) -> list[dict[str, Any]]:
        """Process demodulated dibits and extract TSBK messages.

        Args:
            dibits: Array of dibits (0-3)
            soft: Optional soft symbol values aligned to dibits (C4FM only)

        Returns:
            List of parsed TSBK results
        """
        results: list[dict[str, Any]] = []

        if dibits.size > 0 and ((dibits < 0).any() or (dibits > 3).any()):
            raise ValueError("ControlChannelMonitor._process_dibits: dibits out of range")
        if soft is not None and len(soft) != len(dibits):
            raise ValueError("ControlChannelMonitor._process_dibits: soft length mismatch")

        # Apply polarity correction if needed (OP25-style)
        if self._reverse_p:
            dibits = dibits ^ self._reverse_p
            if soft is not None and self._reverse_p == 2:
                soft = -soft

        # [DIAG-POLARITY] Log polarity state periodically
        if not hasattr(self, "_polarity_diag_count"):
            self._polarity_diag_count = 0
        self._polarity_diag_count += 1
        if self._polarity_diag_count % 500 == 1:
            sample = dibits[:8].tolist() if len(dibits) >= 8 else dibits.tolist()
            logger.info(
                f"[DIAG-POLARITY] reverse_p={self._reverse_p}, latched={getattr(self, '_polarity_latched', False)}, "
                f"first_8_dibits_after_correction={sample}"
            )

        # [DIAG-HISTOGRAM] Track dibit distribution from slicer
        if not hasattr(self, "_dibit_histogram"):
            self._dibit_histogram = {0: 0, 1: 0, 2: 0, 3: 0}
            self._dibit_histogram_count = 0
        for d in dibits:
            if 0 <= d <= 3:
                self._dibit_histogram[int(d)] += 1
        self._dibit_histogram_count += len(dibits)
        if self._dibit_histogram_count >= 50000 and self._polarity_diag_count % 500 == 1:
            total = sum(self._dibit_histogram.values())
            pcts = {k: 100 * v / total if total > 0 else 0 for k, v in self._dibit_histogram.items()}
            logger.info(
                f"[DIAG-HISTOGRAM] Dibit distribution (n={total}): "
                f"0(+1)={pcts[0]:.1f}%, 1(+3)={pcts[1]:.1f}%, "
                f"2(-1)={pcts[2]:.1f}%, 3(-3)={pcts[3]:.1f}%"
            )

        # [DIAG-HZ-HISTOGRAM] Track discriminator output in Hz at decision instants
        # P25 C4FM ideal deviation points: -1800, -600, +600, +1800 Hz
        # Conversion: Hz ≈ soft_symbol_norm * 600 (since boundary at 2.0 = 1200 Hz)
        if soft is not None and len(soft) > 0:
            if not hasattr(self, "_hz_histogram_bins"):
                # Bins from -2400 to +2400 Hz in 200 Hz steps
                self._hz_histogram_bins = np.linspace(-2400.0, 2400.0, 25)
                self._hz_histogram_counts = np.zeros(24, dtype=np.int64)
                self._hz_symbol_total = 0
                self._hz_min = float('inf')
                self._hz_max = float('-inf')
                self._hz_sum = 0.0
                self._hz_sum_sq = 0.0  # For standard deviation

            # Convert soft symbols to Hz (approximate)
            # Soft symbol scale: ±3 → ±1800 Hz, so Hz = soft * 600
            soft_hz = soft * 600.0

            # Update histogram
            hist, _ = np.histogram(soft_hz, bins=self._hz_histogram_bins)
            self._hz_histogram_counts += hist
            self._hz_symbol_total += len(soft_hz)
            self._hz_min = min(self._hz_min, float(np.min(soft_hz)))
            self._hz_max = max(self._hz_max, float(np.max(soft_hz)))
            self._hz_sum += float(np.sum(soft_hz))
            self._hz_sum_sq += float(np.sum(soft_hz ** 2))

            # Log periodically (every ~10000 symbols after initial 5000)
            if self._hz_symbol_total >= 5000 and self._hz_symbol_total % 10000 < 500:
                mean_hz = self._hz_sum / self._hz_symbol_total if self._hz_symbol_total > 0 else 0
                var_hz = (self._hz_sum_sq / self._hz_symbol_total) - (mean_hz ** 2) if self._hz_symbol_total > 0 else 0
                std_hz = np.sqrt(max(0, var_hz))

                total_counts = self._hz_histogram_counts.sum()
                if total_counts > 0:
                    pcts = 100.0 * self._hz_histogram_counts / total_counts
                    bin_centers = (self._hz_histogram_bins[:-1] + self._hz_histogram_bins[1:]) / 2

                    # Show distribution as ASCII histogram
                    hist_str = ""
                    for center, pct in zip(bin_centers, pcts):
                        bar = "#" * int(pct / 2)  # Scale to reasonable width
                        # Mark expected peak positions
                        marker = ""
                        if abs(center - (-1800)) < 150:
                            marker = " ← -1800"
                        elif abs(center - (-600)) < 150:
                            marker = " ← -600"
                        elif abs(center - 600) < 150:
                            marker = " ← +600"
                        elif abs(center - 1800) < 150:
                            marker = " ← +1800"
                        hist_str += f"\n    {center:+5.0f} Hz: {pct:5.1f}% {bar}{marker}"

                    # Assess quality: check if peaks are near expected positions
                    # Find the 4 highest bins
                    top_4_indices = np.argsort(self._hz_histogram_counts)[-4:]
                    top_4_centers = [bin_centers[i] for i in top_4_indices]

                    logger.info(
                        f"[DIAG-HZ-HISTOGRAM] Discriminator Hz at decision instants (n={self._hz_symbol_total}):\n"
                        f"    min={self._hz_min:.0f} Hz, max={self._hz_max:.0f} Hz\n"
                        f"    DC offset (mean)={mean_hz:.0f} Hz, std={std_hz:.0f} Hz\n"
                        f"    Top 4 peaks at: {sorted(top_4_centers)}\n"
                        f"    (Expected: 4 peaks near -1800, -600, +600, +1800 Hz){hist_str}"
                    )

        # Track total dibits for sync delta calculation
        if not hasattr(self, "_total_dibits_seen"):
            self._total_dibits_seen = 0
        self._total_dibits_seen += len(dibits)

        # Add to buffer
        self._dibit_buffer.extend(dibits.tolist())
        if soft is not None:
            self._soft_buffer.extend(soft.tolist())

        # Process buffer
        while True:
            if self.sync_state == SyncState.SEARCHING:
                # Look for frame sync pattern
                sync_idx = self._find_sync_in_buffer()
                if sync_idx < 0:
                    # Not found, keep last (FRAME_SYNC_DIBITS - 1) dibits
                    if len(self._dibit_buffer) > self.FRAME_SYNC_DIBITS:
                        self._dibit_buffer = self._dibit_buffer[-(self.FRAME_SYNC_DIBITS - 1) :]
                        if self._soft_buffer:
                            self._soft_buffer = self._soft_buffer[-(self.FRAME_SYNC_DIBITS - 1) :]
                    break

                # Found sync pattern
                self._dibit_buffer = self._dibit_buffer[sync_idx:]
                if self._soft_buffer:
                    self._soft_buffer = self._soft_buffer[sync_idx:]
                self.sync_state = SyncState.SYNCED
                self._last_sync_time = time.time()

                logger.debug("ControlChannelMonitor: Frame sync acquired")
                if self.on_sync_acquired:
                    self.on_sync_acquired()

            # In synced state, try to decode frame
            # Debug: track frame decode attempts
            if not hasattr(self, "_decode_attempts"):
                self._decode_attempts = 0
            self._decode_attempts += 1
            if self._decode_attempts % 500 == 1:  # Reduced frequency
                logger.debug(
                    f"ControlChannelMonitor: Frame decode attempt {self._decode_attempts}, buffer={len(self._dibit_buffer)}"
                )

            if len(self._dibit_buffer) < self.TSDU_FRAME_DIBITS:
                # Need more dibits
                break

            # Extract frame
            frame_dibits = np.array(self._dibit_buffer[: self.TSDU_FRAME_DIBITS], dtype=np.uint8)
            frame_soft = None
            if self._soft_buffer and len(self._soft_buffer) >= self.TSDU_FRAME_DIBITS:
                frame_soft = np.array(self._soft_buffer[: self.TSDU_FRAME_DIBITS], dtype=np.float32)

            # Verify sync pattern at start of frame using soft correlation
            frame_sync = frame_dibits[: self.FRAME_SYNC_DIBITS]
            sync_score, _ = self._soft_correlation(frame_sync.tolist())
            if sync_score < self.SOFT_SYNC_THRESHOLD:  # Match search tolerance
                # Lost sync
                self.sync_state = SyncState.SEARCHING
                self.sync_losses += 1
                logger.warning(
                    f"ControlChannelMonitor: Lost frame sync (score={sync_score:.1f} < {self.SOFT_SYNC_THRESHOLD})"
                )
                if self.on_sync_lost:
                    self.on_sync_lost()
                # Don't consume dibits, let search find next sync
                continue
            # Debug: log successful sync verification (every 50 frames)
            if not hasattr(self, "_verified_frames"):
                self._verified_frames = 0
            self._verified_frames += 1
            if self._verified_frames % 500 == 1:  # Reduced frequency
                logger.debug(
                    f"ControlChannelMonitor: Sync verified (frame {self._verified_frames}, score={sync_score:.1f})"
                )

            # Decode NID (Network ID) after sync
            # NID is 32 dibits of data, but there's a status symbol at position 12
            # (36 dibits from frame start = 24 sync + 12 into NID), so we need 33 dibits
            nid_dibits = frame_dibits[self.FRAME_SYNC_DIBITS : self.FRAME_SYNC_DIBITS + 33]

            # Debug: log NID dibits for first few frames
            if not hasattr(self, "_frame_count"):
                self._frame_count = 0
            self._frame_count += 1

            if self._frame_count <= 3:  # Only first 3 frames at DEBUG
                # Log DUID position dibits specifically
                self.FRAME_SYNC_DIBITS + 6  # positions 30-31
                logger.debug(
                    f"ControlChannelMonitor: Frame {self._frame_count}, "
                    f"frame_dibits[30:32]={list(frame_dibits[30:32])}, "
                    f"nid_dibits[6:8]={list(nid_dibits[6:8])}, "
                    f"NID dibits (33): {list(nid_dibits[:15])}..."
                )

            # Decode NID with NAC tracking for intelligent BCH retry
            nid = decode_nid(nid_dibits, skip_status_at_10=True, nac_tracker=self._nac_tracker)

            # [DIAG-STAGE7] NID decode statistics
            if not hasattr(self, "_diag_nid_count"):
                self._diag_nid_count = 0
                self._diag_nid_valid = 0
                self._diag_last_nacs = []
            self._diag_nid_count += 1
            if nid is not None:
                self._diag_nid_valid += 1
                self._diag_last_nacs.append(nid.nac)
                if len(self._diag_last_nacs) > 20:
                    self._diag_last_nacs = self._diag_last_nacs[-20:]

            # Track DUID histogram and BCH errors
            if nid is not None:
                duid_val = nid.duid.value if hasattr(nid.duid, 'value') else int(nid.duid)
                self._diag_duid_histogram[duid_val] = self._diag_duid_histogram.get(duid_val, 0) + 1
                if nid.errors >= 0:  # BCH succeeded
                    self._diag_bch_errors.append(nid.errors)
                    if len(self._diag_bch_errors) > 100:
                        self._diag_bch_errors = self._diag_bch_errors[-100:]

            if self._diag_nid_count % 20 == 0:
                valid_rate = (
                    100.0 * self._diag_nid_valid / self._diag_nid_count
                    if self._diag_nid_count > 0
                    else 0.0
                )
                # Check NAC consistency - all same = good, all different = bad
                unique_nacs = set(self._diag_last_nacs)
                nac_str = ",".join([f"0x{n:03x}" for n in list(unique_nacs)[:5]])

                # DUID histogram for diagnosis
                duid_str = ", ".join([f"{k}:{v}" for k, v in sorted(self._diag_duid_histogram.items())])

                # BCH error stats
                bch_stats = ""
                if self._diag_bch_errors:
                    bch_pass = len([e for e in self._diag_bch_errors if e >= 0])
                    bch_avg = sum(self._diag_bch_errors) / len(self._diag_bch_errors) if self._diag_bch_errors else 0
                    bch_stats = f", bch_avg_errors={bch_avg:.1f}"

                logger.info(
                    f"[DIAG-STAGE7] NID: count={self._diag_nid_count}, "
                    f"valid={self._diag_nid_valid}, rate={valid_rate:.1f}%, "
                    f"unique_nacs={len(unique_nacs)}, recent_nacs=[{nac_str}]"
                )
                logger.info(
                    f"[DIAG-DUID] DUID histogram: {{{duid_str}}}{bch_stats}"
                )

            # Debug: log every 100th frame (less frequent)
            if self._frame_count <= 3 or self._frame_count % 100 == 1:
                logger.debug(
                    f"ControlChannelMonitor: Frame {self._frame_count}, NID={nid}, "
                    f"DUID={nid.duid if nid else 'None'}, TSDU={DUID.TSDU}"
                )

            # Calculate actual frame length based on DUID
            # Per P25P1DataUnitID.java:
            #   TSBK1: 56 + 98 + 5 + 21 = 180 dibits
            #   TSBK2: 56 + 196 + 8 + 28 = 288 dibits (2 blocks)
            #   TSBK3: 56 + 294 + 10 + 0 = 360 dibits (3 blocks)
            # We use a sync-search approach instead of fixed consumption
            frame_consumed = 0

            if nid is not None and nid.duid == DUID.TSDU:
                # This is a TSDU frame - decode it
                logger.debug(
                    f"ControlChannelMonitor: Calling decode_tsdu on TSDU frame, frame_len={len(frame_dibits)}"
                )
                tsdu = decode_tsdu(frame_dibits, frame_soft)
                if tsdu:
                    logger.debug(
                        f"ControlChannelMonitor: decode_tsdu returned {len(tsdu.tsbk_blocks) if tsdu.tsbk_blocks else 0} TSBK blocks"
                    )
                else:
                    logger.debug("ControlChannelMonitor: decode_tsdu returned None")
                if tsdu and tsdu.tsbk_blocks:
                    self.frames_decoded += 1
                    num_blocks = len(tsdu.tsbk_blocks)

                    # Calculate frame length based on number of TSBK blocks
                    # sync(24) + nid(33 incl status) + data + status + null
                    if num_blocks == 1:
                        frame_consumed = 180
                    elif num_blocks == 2:
                        frame_consumed = 288
                    else:
                        frame_consumed = 360

                    # Parse each TSBK block in the frame
                    for tsbk_block in tsdu.tsbk_blocks:
                        # Track all TSBK decode attempts for stats
                        self.tsbk_attempts += 1

                        # Track trellis error metrics
                        self._diag_trellis_metrics.append(tsbk_block.error_metric)
                        if len(self._diag_trellis_metrics) > 100:
                            self._diag_trellis_metrics = self._diag_trellis_metrics[-100:]

                        if not tsbk_block.crc_valid:
                            # CRC failed - log periodically for debugging
                            if self.tsbk_attempts <= 10 or self.tsbk_attempts % 50 == 0:
                                logger.warning(
                                    f"TSBK CRC failed: attempts={self.tsbk_attempts}, "
                                    f"pass_rate={100 * self.tsbk_crc_pass / self.tsbk_attempts:.1f}%"
                                )
                            continue  # Skip blocks with bad CRC

                        # CRC passed
                        self.tsbk_crc_pass += 1

                        # [DIAG-STAGE7b] TSBK CRC statistics (every 20 attempts)
                        if self.tsbk_attempts % 20 == 0:
                            crc_rate = 100.0 * self.tsbk_crc_pass / self.tsbk_attempts
                            # Trellis error metric stats
                            trellis_stats = ""
                            if self._diag_trellis_metrics:
                                avg_metric = sum(self._diag_trellis_metrics) / len(self._diag_trellis_metrics)
                                min_metric = min(self._diag_trellis_metrics)
                                max_metric = max(self._diag_trellis_metrics)
                                trellis_stats = f", trellis_err: avg={avg_metric:.1f} min={min_metric} max={max_metric}"
                            logger.info(
                                f"[DIAG-STAGE7b] TSBK: attempts={self.tsbk_attempts}, "
                                f"crc_pass={self.tsbk_crc_pass}, rate={crc_rate:.1f}%{trellis_stats}"
                            )

                        # Pass opcode and mfid from TSBKBlock to parser
                        result = self._parse_tsbk(
                            tsbk_block.opcode, tsbk_block.mfid, tsbk_block.data
                        )
                        if result:
                            results.append(result)
                            self.tsbk_decoded += 1

                        # Call TSBK callback
                        if self.on_tsbk:
                            self.on_tsbk(tsbk_block.data)

            # If we couldn't determine frame length (e.g., non-TSDU or decode failed),
            # use minimum skip to avoid re-finding same sync
            if frame_consumed == 0:
                # Skip sync + NID to avoid re-detecting same frame
                frame_consumed = 60  # 24 sync + 33 NID + 3 buffer

            # Consume actual frame length from buffer
            self._dibit_buffer = self._dibit_buffer[frame_consumed:]
            if self._soft_buffer:
                self._soft_buffer = self._soft_buffer[frame_consumed:]
            self._last_sync_time = time.time()

            # After processing a frame, go back to searching for next sync
            # This is more robust than assuming fixed frame spacing
            self.sync_state = SyncState.SEARCHING

        # Check for sync timeout
        if self.sync_state == SyncState.SYNCED:
            if time.time() - self._last_sync_time > self.SYNC_TIMEOUT:
                self.sync_state = SyncState.SEARCHING
                self.sync_losses += 1
                logger.warning("ControlChannelMonitor: Sync timeout")
                if self.on_sync_lost:
                    self.on_sync_lost()

        return results

    @property
    def has_sync(self) -> bool:
        """Check if we currently have sync (for lock decisions).

        Returns True if sync was detected within the timeout period.
        SDRTrunk uses sync + NID validation for lock, not TSBK CRC.
        """
        if self._last_sync_time == 0.0:
            return False
        return (time.time() - self._last_sync_time) < self.SYNC_TIMEOUT

    @property
    def last_sync_age(self) -> float:
        """Get seconds since last sync detection."""
        if self._last_sync_time == 0.0:
            return float("inf")
        return time.time() - self._last_sync_time

    # Soft correlation constants (same as P25FrameSync)
    SYNC_PATTERN_SYMBOLS = np.array(
        [
            +3,
            +3,
            +3,
            +3,
            +3,
            -3,
            +3,
            +3,
            -3,
            -3,
            +3,
            +3,
            -3,
            -3,
            -3,
            -3,
            +3,
            -3,
            +3,
            -3,
            -3,
            -3,
            -3,
            -3,
        ],
        dtype=np.float32,
    )
    # Reversed polarity sync pattern (negated symbols)
    SYNC_PATTERN_SYMBOLS_REV = np.array(
        [
            -3,
            -3,
            -3,
            -3,
            -3,
            +3,
            -3,
            -3,
            +3,
            +3,
            -3,
            -3,
            +3,
            +3,
            +3,
            +3,
            -3,
            +3,
            -3,
            +3,
            +3,
            +3,
            +3,
            +3,
        ],
        dtype=np.float32,
    )
    DIBIT_TO_SYMBOL = np.array([+1.0, +3.0, -1.0, -3.0], dtype=np.float32)
    # SDRTrunk uses 80 with radian-scale symbols (max ~133), which is 60% of max
    # WaveCap uses ±3 normalized symbols (max 216), so 60% = 130
    # Lowered to 100 (46%) to allow marginal signals to be decoded
    SOFT_SYNC_THRESHOLD = 100  # Lowered for weak signals (was 130)

    def _soft_correlation(
        self, dibits: list[int], detect_polarity: bool = False
    ) -> tuple[float, bool]:
        """Compute soft correlation score between dibits and sync pattern.

        SDRTrunk-style: dot product of received symbols with ideal sync symbols.
        Max score is 24 * 9 = 216 when all symbols match perfectly at ±3.
        Checks both normal and reversed polarity, returns best absolute score.

        Args:
            dibits: 24 dibits to correlate
            detect_polarity: If True, also return whether reversed polarity is better

        Returns:
            If detect_polarity=False: best score
            If detect_polarity=True: tuple of (best_score, is_reversed)
        """
        # Convert dibits to symbols: 0->+1, 1->+3, 2->-1, 3->-3
        symbols = self.DIBIT_TO_SYMBOL[np.clip(dibits[:24], 0, 3)]
        # Check both normal and reversed polarity
        normal_score = float(np.dot(symbols, self.SYNC_PATTERN_SYMBOLS))
        rev_score = float(np.dot(symbols, self.SYNC_PATTERN_SYMBOLS_REV))

        if detect_polarity and rev_score > normal_score:
            return rev_score, True
        return max(normal_score, rev_score), False

    def _find_sync_in_buffer(self) -> int:
        """Find frame sync pattern in dibit buffer using soft correlation.

        Uses SDRTrunk-style soft correlation for robust detection even
        with noisy dibits. This is more tolerant than hard dibit matching.
        Detects and latches polarity reversal (OP25-style).

        Returns:
            Index of sync pattern start, or -1 if not found
        """
        if len(self._dibit_buffer) < self.FRAME_SYNC_DIBITS:
            return -1

        # Debug: Log buffer sample periodically
        if not hasattr(self, "_find_sync_calls"):
            self._find_sync_calls = 0
        self._find_sync_calls += 1
        verbose = self._find_sync_calls <= 3 or self._find_sync_calls % 500 == 0

        best_pos = -1
        best_score = 0.0
        best_is_reversed = False

        # Search for sync using soft correlation with polarity detection
        for i in range(len(self._dibit_buffer) - self.FRAME_SYNC_DIBITS + 1):
            window = self._dibit_buffer[i : i + self.FRAME_SYNC_DIBITS]
            score, is_reversed = self._soft_correlation(window, detect_polarity=True)

            if score > best_score:
                best_score = score
                best_pos = i
                best_is_reversed = is_reversed

        # [DIAG-STAGE6] Sync detection statistics
        if not hasattr(self, "_diag_sync_attempts"):
            self._diag_sync_attempts = 0
            self._diag_sync_found = 0
            self._diag_exact_sync_found = 0
        self._diag_sync_attempts += 1
        if best_score >= self.SOFT_SYNC_THRESHOLD:
            self._diag_sync_found += 1

        # [DIAG-SYNC-SCORE-HISTOGRAM] Track distribution of sync correlation scores
        # This helps identify:
        # 1. Whether there's a bimodal distribution (noise vs real syncs)
        # 2. Where the optimal threshold should be set
        # 3. False positive rate at current threshold
        # Expected: Peak near 0 (noise) and peak near 180-216 (real syncs with good signal)
        if not hasattr(self, "_sync_score_histogram"):
            # Bins: 0-20, 20-40, ..., 200-220 (11 bins covering 0-220)
            self._sync_score_histogram: dict[int, int] = {}
            self._sync_score_total = 0
            self._sync_score_sum = 0.0
            self._sync_score_max = 0.0

        # Track this score
        bucket = int(best_score // 20) * 20  # 20-point buckets
        self._sync_score_histogram[bucket] = self._sync_score_histogram.get(bucket, 0) + 1
        self._sync_score_total += 1
        self._sync_score_sum += best_score
        if best_score > self._sync_score_max:
            self._sync_score_max = best_score

        # Log histogram periodically
        if self._sync_score_total > 0 and self._sync_score_total % 100 == 0:
            # Build ASCII histogram
            sorted_buckets = sorted(self._sync_score_histogram.keys())
            max_count = max(self._sync_score_histogram.values()) if self._sync_score_histogram else 1
            hist_lines = []
            for b in range(0, 220, 20):
                count = self._sync_score_histogram.get(b, 0)
                bar_len = int(40 * count / max_count) if max_count > 0 else 0
                bar = "#" * bar_len
                pct = 100.0 * count / self._sync_score_total if self._sync_score_total > 0 else 0
                # Mark threshold bucket
                marker = " <-- THRESHOLD" if b <= self.SOFT_SYNC_THRESHOLD < b + 20 else ""
                hist_lines.append(f"    {b:3d}-{b+19:3d}: {bar:<40} {count:5d} ({pct:5.1f}%){marker}")

            mean_score = self._sync_score_sum / self._sync_score_total if self._sync_score_total > 0 else 0
            above_threshold = sum(c for b, c in self._sync_score_histogram.items() if b >= self.SOFT_SYNC_THRESHOLD)
            false_alarm_rate = 100.0 * above_threshold / self._sync_score_total if self._sync_score_total > 0 else 0

            logger.info(
                f"[DIAG-SYNC-SCORE] Sync score histogram (n={self._sync_score_total}):\n"
                + "\n".join(hist_lines) + "\n"
                f"    Mean={mean_score:.1f}, Max={self._sync_score_max:.1f}, "
                f"Above threshold={above_threshold} ({false_alarm_rate:.1f}%)"
            )

        # [DIAG-EXACT-SYNC] Also search for EXACT sync pattern match (hard decision)
        # Expected: 1 1 1 1 1 3 1 1 3 3 1 1 3 3 3 3 1 3 1 3 3 3 3 3
        # Reversed (XOR 2): 3 3 3 3 3 1 3 3 1 1 3 3 1 1 1 1 3 1 3 1 1 1 1 1
        EXACT_SYNC = [1, 1, 1, 1, 1, 3, 1, 1, 3, 3, 1, 1, 3, 3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3]
        EXACT_SYNC_REV = [3, 3, 3, 3, 3, 1, 3, 3, 1, 1, 3, 3, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1]
        exact_pos = -1
        exact_rev = False
        for i in range(len(self._dibit_buffer) - 24 + 1):
            window = self._dibit_buffer[i:i+24]
            if window == EXACT_SYNC:
                exact_pos = i
                exact_rev = False
                self._diag_exact_sync_found += 1
                break
            elif window == EXACT_SYNC_REV:
                exact_pos = i
                exact_rev = True
                self._diag_exact_sync_found += 1
                break

        if self._diag_sync_attempts % 50 == 0:
            sync_rate = (
                100.0 * self._diag_sync_found / self._diag_sync_attempts
                if self._diag_sync_attempts > 0
                else 0.0
            )
            logger.info(
                f"[DIAG-STAGE6] Sync: attempts={self._diag_sync_attempts}, "
                f"soft_found={self._diag_sync_found}, exact_found={self._diag_exact_sync_found}, "
                f"rate={sync_rate:.1f}%, best_score={best_score:.1f}, threshold={self.SOFT_SYNC_THRESHOLD}, "
                f"polarity={'reversed' if best_is_reversed else 'normal'}, "
                f"exact_pos={exact_pos}, exact_rev={exact_rev if exact_pos >= 0 else 'N/A'}"
            )

        if verbose:
            sample = self._dibit_buffer[: min(30, len(self._dibit_buffer))]
            logger.info(
                f"ControlChannelMonitor._find_sync_in_buffer: call #{self._find_sync_calls}, "
                f"buffer_len={len(self._dibit_buffer)}, best_score={best_score:.1f}, "
                f"threshold={self.SOFT_SYNC_THRESHOLD}, best_pos={best_pos}, reversed={best_is_reversed}, "
                f"first_30_dibits={list(sample)}"
            )

        # Check if best score exceeds threshold
        if best_score >= self.SOFT_SYNC_THRESHOLD:
            # [DIAG-SYNC-DELTA] Track sync-to-sync delta for periodicity check
            # On control channel, real syncs should be ~180 dibits apart (single-block TSBK)
            # Flat/random distribution = false positives
            if not hasattr(self, "_last_sync_dibit_pos"):
                self._last_sync_dibit_pos = 0
                self._sync_delta_histogram: dict[int, int] = {}
                self._total_syncs_for_delta = 0

            current_total_dibits = getattr(self, "_total_dibits_seen", 0) + best_pos
            delta = current_total_dibits - self._last_sync_dibit_pos
            self._last_sync_dibit_pos = current_total_dibits

            if delta > 0 and delta < 1000:  # Ignore first sync and very large gaps
                # Bucket into 10-dibit ranges for histogram
                bucket = (delta // 10) * 10
                self._sync_delta_histogram[bucket] = self._sync_delta_histogram.get(bucket, 0) + 1
                self._total_syncs_for_delta += 1

            # Log periodically
            if self._total_syncs_for_delta > 0 and self._total_syncs_for_delta % 50 == 0:
                # Sort by bucket and show top 10
                delta_top_buckets = sorted(self._sync_delta_histogram.items(), key=lambda x: -x[1])[:10]
                hist_str = ", ".join([f"{b}:{c}" for b, c in delta_top_buckets])
                # Check if there's a dominant peak (should be near 180 for control channel)
                if delta_top_buckets:
                    dominant_bucket, dominant_count = delta_top_buckets[0]
                    total_count = sum(self._sync_delta_histogram.values())
                    dominant_pct = 100.0 * dominant_count / total_count if total_count > 0 else 0
                    is_periodic = 150 <= dominant_bucket <= 210 and dominant_pct > 30
                    logger.info(
                        f"[DIAG-SYNC-DELTA] Sync-to-sync delta histogram (n={self._total_syncs_for_delta}):\n"
                        f"    Top buckets: {hist_str}\n"
                        f"    Dominant: {dominant_bucket} dibits ({dominant_pct:.1f}%), "
                        f"periodic={'YES' if is_periodic else 'NO (expect peak at ~180)'}"
                    )

            # Latch polarity on FIRST successful sync only (don't flip-flop)
            # This follows OP25 approach: detect once and maintain
            if not hasattr(self, "_polarity_latched"):
                self._polarity_latched = False

            if not self._polarity_latched:
                if best_is_reversed:
                    self._reverse_p = 2
                    logger.info(f"ControlChannelMonitor: Latching reversed polarity (reverse_p=2)")
                    # Apply polarity correction to entire buffer (dibits AND soft symbols)
                    self._dibit_buffer = [d ^ 2 for d in self._dibit_buffer]
                    if self._soft_buffer:
                        self._soft_buffer = [-s for s in self._soft_buffer]
                else:
                    logger.info(f"ControlChannelMonitor: Latching normal polarity (reverse_p=0)")
                self._polarity_latched = True

            # [DIAG-SYNC-DUMP] Dump raw dibits at sync for expert analysis
            # This helps verify:
            # 1. Sync pattern matches 0x5575F5FF77FF (first 24 dibits)
            # 2. Raw frame structure for microslot/status alignment analysis
            if not hasattr(self, "_sync_dump_count"):
                self._sync_dump_count = 0
            self._sync_dump_count += 1
            if self._sync_dump_count <= 5:
                # Dump 180 dibits (5 microslots) starting at sync position
                end_pos = min(best_pos + 180, len(self._dibit_buffer))
                raw_dibits = self._dibit_buffer[best_pos:end_pos]

                # Verify sync pattern: expected 0x5575F5FF77FF
                # Dibits: 1 1 1 1 1 3 1 1 3 3 1 1 3 3 3 3 1 3 1 3 3 3 3 3
                sync_dibits = raw_dibits[:24] if len(raw_dibits) >= 24 else raw_dibits
                expected_sync = [1, 1, 1, 1, 1, 3, 1, 1, 3, 3, 1, 1, 3, 3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3]
                sync_match = sync_dibits == expected_sync

                # Convert sync dibits to hex for comparison with 0x5575F5FF77FF
                sync_hex = 0
                for d in sync_dibits:
                    sync_hex = (sync_hex << 2) | d

                logger.info(
                    f"[DIAG-SYNC-DUMP] Frame #{self._sync_dump_count}: "
                    f"sync_hex=0x{sync_hex:012x} (expected 0x5575f5ff77ff), match={sync_match}"
                )

                # Dump first 60 dibits (sync + NID region)
                dibit_str = " ".join(f"{d}" for d in raw_dibits[:60])
                logger.info(f"[DIAG-SYNC-DUMP] raw_dibits[0:60] = {dibit_str}")

                # Dump full 180 dibits for expert analysis (split into 3 lines)
                if len(raw_dibits) >= 180:
                    line1 = " ".join(f"{d}" for d in raw_dibits[0:60])
                    line2 = " ".join(f"{d}" for d in raw_dibits[60:120])
                    line3 = " ".join(f"{d}" for d in raw_dibits[120:180])
                    logger.info(f"[DIAG-SYNC-DUMP] raw_dibits[0:60]   = {line1}")
                    logger.info(f"[DIAG-SYNC-DUMP] raw_dibits[60:120] = {line2}")
                    logger.info(f"[DIAG-SYNC-DUMP] raw_dibits[120:180]= {line3}")

                    # Also show dibit positions where status symbols should be
                    # Status at frame positions 35, 71, 107, 143, 179...
                    status_positions = [35, 71, 107, 143, 179]
                    status_dibits = [raw_dibits[p] if p < len(raw_dibits) else -1 for p in status_positions]
                    logger.info(
                        f"[DIAG-SYNC-DUMP] Status symbol positions {status_positions}: "
                        f"dibits={status_dibits}"
                    )

            return best_pos

        return -1

    def _verify_sync(self, dibits: NDArrayInt) -> bool:
        """Verify that dibits match sync pattern using soft correlation.

        Args:
            dibits: First 24 dibits of frame

        Returns:
            True if sync pattern matches (above soft correlation threshold)
        """
        score, _ = self._soft_correlation(dibits.tolist())
        # Use same threshold as search for consistency
        return score >= self.SOFT_SYNC_THRESHOLD

    def _get_sync_dibits(self) -> NDArrayInt:
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
        return np.array(
            [1, 1, 1, 1, 1, 3, 1, 1, 3, 3, 1, 1, 3, 3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3], dtype=np.uint8
        )

    def _parse_tsbk(self, opcode: int, mfid: int, data: bytes) -> dict[str, Any] | None:
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
            result = self._tsbk_parser.parse(opcode, mfid, data)
            if result.get("type") == "PARSE_ERROR":
                self.tsbk_rejected += 1
                return None
            return result
        except Exception as e:
            logger.error(f"ControlChannelMonitor: TSBK parse error (opcode=0x{opcode:02X}): {e}")
            return None

    def get_stats(self) -> dict[str, Any]:
        """Get monitor statistics."""
        # Calculate CRC pass rate
        crc_pass_rate = (
            100.0 * self.tsbk_crc_pass / self.tsbk_attempts if self.tsbk_attempts > 0 else 0.0
        )
        return {
            "sync_state": self.sync_state.value,
            "modulation": self.modulation.value,
            "frames_decoded": self.frames_decoded,
            "tsbk_decoded": self.tsbk_decoded,
            "tsbk_rejected": self.tsbk_rejected,
            "sync_losses": self.sync_losses,
            "buffer_dibits": len(self._dibit_buffer),
            # TSBK FEC debug stats
            "tsbk_attempts": self.tsbk_attempts,
            "tsbk_crc_pass": self.tsbk_crc_pass,
            "tsbk_crc_pass_rate": round(crc_pass_rate, 1),
        }


def create_control_monitor(
    protocol: TrunkingProtocol,
    sample_rate: int = 19200,  # ~4 SPS like SDRTrunk
    modulation: P25Modulation | None = None,
) -> ControlChannelMonitor:
    """Create a control channel monitor.

    Args:
        protocol: P25 protocol (Phase I or II)
        sample_rate: Input sample rate in Hz (~19200 for 4 SPS like SDRTrunk)
        modulation: Override modulation type (None = use default for protocol)

    Returns:
        Configured ControlChannelMonitor
    """
    # Determine modulation based on protocol if not specified
    if modulation is None:
        # Phase 1 typically uses C4FM (including SA-GRN)
        # Phase 2 uses CQPSK/LSM
        modulation = P25Modulation.C4FM  # Default to C4FM (most common for Phase 1)

    return ControlChannelMonitor(
        protocol=protocol,
        sample_rate=sample_rate,
        modulation=modulation,
    )
