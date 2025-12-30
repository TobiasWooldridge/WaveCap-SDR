#!/usr/bin/env python3
"""Full integration test of WaveCap P25 decoder on live SDR.

Tests the complete pipeline:
1. SDR capture at 6 MHz
2. Two-stage decimation to 50 kHz
3. C4FM demodulation with sync detection
4. P25 frame decoding with PROPER SYNC ALIGNMENT
5. CRC validation

Success criteria: TSBK CRC pass rate > 30%
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.append('/opt/homebrew/lib/python3.14/site-packages')

import numpy as np
from scipy import signal as scipy_signal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import SoapySDR
except ImportError:
    print("Error: SoapySDR not installed")
    sys.exit(1)

# Add backend path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Import WaveCap modules (using venv with all dependencies)
from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator
from wavecapsdr.decoders import p25_frames

# P25 sync pattern constants
SYNC_PATTERN = 0x5575F5FF77FF
FRAME_SYNC_DIBITS = 24
TSDU_FRAME_DIBITS = 360  # Max TSDU frame size
SOFT_SYNC_THRESHOLD = 80  # SDRTrunk-style threshold

# Sync pattern as dibits: dibit 1 -> +3, dibit 3 -> -3
SYNC_DIBITS = np.array([1, 1, 1, 1, 1, 3, 1, 1, 3, 3, 1, 1,
                        3, 3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3], dtype=np.uint8)

# Sync pattern as symbols for correlation (+3 or -3)
SYNC_SYMBOLS = np.array([3, 3, 3, 3, 3, -3, 3, 3, -3, -3, 3, 3,
                         -3, -3, -3, -3, 3, -3, 3, -3, -3, -3, -3, -3], dtype=np.float32)

# Dibit to symbol mapping: 0->+1, 1->+3, 2->-1, 3->-3
DIBIT_TO_SYMBOL = np.array([1.0, 3.0, -1.0, -3.0], dtype=np.float32)


def soft_correlation(dibits: np.ndarray) -> tuple[float, bool]:
    """Compute soft correlation with sync pattern.

    Returns (score, is_reversed) where:
    - score: correlation score (max 216 for perfect match)
    - is_reversed: True if reversed polarity gives better match
    """
    if len(dibits) < FRAME_SYNC_DIBITS:
        return 0.0, False

    # Convert dibits to symbols
    symbols = DIBIT_TO_SYMBOL[np.clip(dibits[:FRAME_SYNC_DIBITS], 0, 3)]

    # Check both normal and reversed polarity
    normal_score = float(np.dot(symbols, SYNC_SYMBOLS))
    rev_score = float(np.dot(symbols, -SYNC_SYMBOLS))

    if rev_score > normal_score:
        return rev_score, True
    return normal_score, False


def find_sync_in_buffer(dibit_buffer: np.ndarray) -> tuple[int, bool]:
    """Find sync pattern in dibit buffer using soft correlation.

    Returns (position, is_reversed) where:
    - position: index of sync start, or -1 if not found
    - is_reversed: True if polarity is reversed
    """
    if len(dibit_buffer) < FRAME_SYNC_DIBITS:
        return -1, False

    best_pos = -1
    best_score = 0.0
    best_is_reversed = False

    for i in range(len(dibit_buffer) - FRAME_SYNC_DIBITS + 1):
        score, is_reversed = soft_correlation(dibit_buffer[i:i + FRAME_SYNC_DIBITS])
        if score > best_score:
            best_score = score
            best_pos = i
            best_is_reversed = is_reversed

    if best_score >= SOFT_SYNC_THRESHOLD:
        return best_pos, best_is_reversed
    return -1, False


def test_full_integration(
    freq_hz: int = 413_450_000,
    sample_rate: int = 6_000_000,
    duration_sec: float = 60.0,
) -> None:
    """Full integration test of P25 decoder with proper sync alignment."""

    logger.info(f"=== Full Integration Test (Sync-Aligned) ===")
    logger.info(f"Frequency: {freq_hz/1e6:.4f} MHz")
    logger.info(f"Duration: {duration_sec}s")

    # Open SDR
    logger.info("Opening SDR...")
    sdr = SoapySDR.Device("driver=sdrplay")
    sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, sample_rate)
    sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, freq_hz)
    sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, False)
    sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, "IFGR", 40)
    sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, "RFGR", 0)
    sdr.setAntenna(SoapySDR.SOAPY_SDR_RX, 0, "Antenna B")

    stream = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
    sdr.activateStream(stream)

    # Decimation filters (6 MHz → 50 kHz)
    stage1_factor = 30
    stage2_factor = 4
    stage1_taps = scipy_signal.firwin(157, 0.8/stage1_factor, window=("kaiser", 7.857))
    stage1_zi_template = scipy_signal.lfilter_zi(stage1_taps, 1.0).astype(np.complex128)
    stage1_zi = None
    stage2_taps = scipy_signal.firwin(73, 0.8/stage2_factor, window=("kaiser", 7.857))
    stage2_zi_template = scipy_signal.lfilter_zi(stage2_taps, 1.0).astype(np.complex128)
    stage2_zi = None

    # Create C4FM demodulator at 50 kHz
    baseband_rate = 50000
    demod = C4FMDemodulator(sample_rate=baseband_rate)

    # Stats
    stats = {
        "chunks": 0,
        "samples": 0,
        "syncs": 0,
        "sync_found": 0,
        "frames_decoded": 0,
        "nid_attempts": 0,
        "nid_decoded": 0,
        "tsbk_attempts": 0,
        "tsbk_crc_pass": 0,
        "nacs": set(),
        "opcodes": {},
    }

    # Dibit buffer for frame assembly
    dibit_buffer = np.array([], dtype=np.uint8)

    # Polarity tracking (latched on first sync)
    polarity_latched = False
    reverse_polarity = False

    chunk_size = sample_rate // 4
    buffer = np.zeros(chunk_size, dtype=np.complex64)

    logger.info("Starting decode loop...")
    start_time = time.time()
    last_report = start_time

    while time.time() - start_time < duration_sec:
        sr = sdr.readStream(stream, [buffer], chunk_size)
        if sr.ret <= 0:
            continue

        iq = buffer[:sr.ret].astype(np.complex128)
        stats["chunks"] += 1
        stats["samples"] += len(iq)

        # Initialize stage1_zi with first sample
        if stage1_zi is None:
            stage1_zi = stage1_zi_template * iq[0]

        filtered1, stage1_zi = scipy_signal.lfilter(stage1_taps, 1.0, iq, zi=stage1_zi)
        decimated1 = filtered1[::stage1_factor]

        if stage2_zi is None:
            stage2_zi = stage2_zi_template * decimated1[0]

        filtered2, stage2_zi = scipy_signal.lfilter(stage2_taps, 1.0, decimated1, zi=stage2_zi)
        baseband = filtered2[::stage2_factor].astype(np.complex64)

        # Demodulate
        try:
            dibits, soft_symbols = demod.demodulate(baseband)
            stats["syncs"] = demod._sync_count

            if len(dibits) > 0:
                # Apply polarity correction if needed
                if reverse_polarity:
                    dibits = dibits ^ 2  # XOR with 2 to flip polarity

                # Accumulate dibits
                dibit_buffer = np.concatenate([dibit_buffer, dibits.astype(np.uint8)])

                # Process buffer - look for sync patterns
                while len(dibit_buffer) >= TSDU_FRAME_DIBITS:
                    # Search for sync pattern
                    sync_pos, is_reversed = find_sync_in_buffer(dibit_buffer)

                    if sync_pos < 0:
                        # No sync found, keep last (FRAME_SYNC_DIBITS - 1) dibits
                        if len(dibit_buffer) > FRAME_SYNC_DIBITS:
                            dibit_buffer = dibit_buffer[-(FRAME_SYNC_DIBITS - 1):]
                        break

                    stats["sync_found"] += 1

                    # Latch polarity on first sync
                    if not polarity_latched:
                        reverse_polarity = is_reversed
                        polarity_latched = True
                        if is_reversed:
                            logger.info("Polarity reversed detected, applying correction")
                            # Correct all dibits in buffer
                            dibit_buffer = dibit_buffer ^ 2

                    # Skip to sync position
                    if sync_pos > 0:
                        dibit_buffer = dibit_buffer[sync_pos:]

                    # Need enough for full frame
                    if len(dibit_buffer) < TSDU_FRAME_DIBITS:
                        break

                    # Extract frame (sync + NID + data)
                    frame_dibits = dibit_buffer[:TSDU_FRAME_DIBITS]
                    stats["frames_decoded"] += 1

                    # Decode NID (after 24-dibit sync, with status symbol at position 10)
                    # NID spans 33 dibits to account for the status symbol
                    nid_dibits = frame_dibits[FRAME_SYNC_DIBITS:FRAME_SYNC_DIBITS + 33]

                    stats["nid_attempts"] += 1
                    nid = p25_frames.decode_nid(nid_dibits, skip_status_at_10=True)

                    if nid is not None:
                        stats["nid_decoded"] += 1
                        stats["nacs"].add(nid.nac)

                        # Only decode TSDU frames
                        if nid.duid == p25_frames.DUID.TSDU:
                            # decode_tsdu expects full frame including sync
                            tsdu = p25_frames.decode_tsdu(frame_dibits)
                            if tsdu is not None and hasattr(tsdu, 'tsbk_blocks') and tsdu.tsbk_blocks:
                                for block in tsdu.tsbk_blocks:
                                    stats["tsbk_attempts"] += 1
                                    if block.crc_valid:
                                        stats["tsbk_crc_pass"] += 1
                                        opcode = block.opcode
                                        stats["opcodes"][opcode] = stats["opcodes"].get(opcode, 0) + 1
                                        if stats["tsbk_crc_pass"] <= 10:
                                            logger.info(f"TSBK CRC OK: NAC=0x{nid.nac:03x}, opcode=0x{opcode:02x}")
                            elif tsdu is not None and hasattr(tsdu, 'blocks') and tsdu.blocks:
                                # Alternative attribute name
                                for block in tsdu.blocks:
                                    stats["tsbk_attempts"] += 1
                                    if hasattr(block, 'crc_valid') and block.crc_valid:
                                        stats["tsbk_crc_pass"] += 1
                                        opcode = block.opcode
                                        stats["opcodes"][opcode] = stats["opcodes"].get(opcode, 0) + 1
                                        if stats["tsbk_crc_pass"] <= 10:
                                            logger.info(f"TSBK CRC OK: NAC=0x{nid.nac:03x}, opcode=0x{opcode:02x}")
                                    elif hasattr(block, 'crc_ok') and block.crc_ok:
                                        stats["tsbk_crc_pass"] += 1
                                        opcode = block.opcode
                                        stats["opcodes"][opcode] = stats["opcodes"].get(opcode, 0) + 1
                                        if stats["tsbk_crc_pass"] <= 10:
                                            logger.info(f"TSBK CRC OK: NAC=0x{nid.nac:03x}, opcode=0x{opcode:02x}")

                    # Consume sync pattern and let the loop search for next sync
                    dibit_buffer = dibit_buffer[FRAME_SYNC_DIBITS:]

        except Exception as e:
            logger.debug(f"Decode error: {e}")

        # Report every 10 seconds
        elapsed = time.time() - start_time
        if elapsed - (last_report - start_time) >= 10.0:
            crc_rate = (stats["tsbk_crc_pass"] / stats["tsbk_attempts"] * 100) if stats["tsbk_attempts"] > 0 else 0
            nid_rate = (stats["nid_decoded"] / stats["nid_attempts"] * 100) if stats["nid_attempts"] > 0 else 0
            logger.info(
                f"{elapsed:.0f}s: C4FM_syncs={stats['syncs']}, frame_syncs={stats['sync_found']}, "
                f"NID={stats['nid_decoded']}/{stats['nid_attempts']} ({nid_rate:.1f}%), "
                f"TSBK={stats['tsbk_crc_pass']}/{stats['tsbk_attempts']} ({crc_rate:.1f}%), "
                f"NACs={len(stats['nacs'])}"
            )
            last_report = time.time()

    sdr.deactivateStream(stream)
    sdr.closeStream(stream)

    # Final report
    elapsed = time.time() - start_time
    crc_rate = (stats["tsbk_crc_pass"] / stats["tsbk_attempts"] * 100) if stats["tsbk_attempts"] > 0 else 0
    nid_rate = (stats["nid_decoded"] / stats["nid_attempts"] * 100) if stats["nid_attempts"] > 0 else 0

    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS")
    logger.info("="*60)
    logger.info(f"Duration: {elapsed:.1f}s")
    logger.info(f"Samples processed: {stats['samples']/1e6:.1f}M")
    logger.info(f"C4FM sync detections: {stats['syncs']}")
    logger.info(f"Frame syncs found: {stats['sync_found']}")
    logger.info(f"Frames decoded: {stats['frames_decoded']}")
    logger.info(f"NID decode rate: {stats['nid_decoded']}/{stats['nid_attempts']} ({nid_rate:.1f}%)")
    logger.info(f"TSBK CRC pass rate: {stats['tsbk_crc_pass']}/{stats['tsbk_attempts']} ({crc_rate:.1f}%)")
    logger.info(f"NACs seen: {stats['nacs']}")

    if stats['opcodes']:
        logger.info(f"TSBK opcodes: {dict(sorted(stats['opcodes'].items()))}")

    logger.info("="*60)
    if crc_rate >= 30:
        logger.info(f"SUCCESS: TSBK CRC pass rate {crc_rate:.1f}% >= 30%")
    elif crc_rate > 0:
        logger.warning(f"PARTIAL: TSBK CRC pass rate {crc_rate:.1f}% < 30%")
    else:
        logger.error(f"FAILED: No TSBK CRC passes")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Full P25 integration test")
    parser.add_argument("--freq", type=int, default=413_450_000, help="Frequency in Hz")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration in seconds")
    args = parser.parse_args()

    test_full_integration(freq_hz=args.freq, duration_sec=args.duration)
