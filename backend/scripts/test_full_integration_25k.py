#!/usr/bin/env python3
"""Full integration test using 25 kHz baseband rate (matching SDRTrunk).

SDRTrunk uses 8 MHz → 25 kHz via polyphase channelizer.
We'll use 6 MHz → 25 kHz via multi-stage decimation.
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

# Import WaveCap modules
from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator
from wavecapsdr.decoders import p25_frames

# P25 sync pattern constants
FRAME_SYNC_DIBITS = 24
TSDU_FRAME_DIBITS = 360
# Sync threshold for dibit-based correlation
# Perfect match: 24 × 9 = 216, using 50% threshold = 108
SOFT_SYNC_THRESHOLD = 108

# Sync symbols for correlation
SYNC_SYMBOLS = np.array([3, 3, 3, 3, 3, -3, 3, 3, -3, -3, 3, 3,
                         -3, -3, -3, -3, 3, -3, 3, -3, -3, -3, -3, -3], dtype=np.float32)
DIBIT_TO_SYMBOL = np.array([1.0, 3.0, -1.0, -3.0], dtype=np.float32)


def soft_correlation(dibits: np.ndarray) -> tuple[float, bool]:
    if len(dibits) < FRAME_SYNC_DIBITS:
        return 0.0, False
    symbols = DIBIT_TO_SYMBOL[np.clip(dibits[:FRAME_SYNC_DIBITS], 0, 3)]
    normal_score = float(np.dot(symbols, SYNC_SYMBOLS))
    rev_score = float(np.dot(symbols, -SYNC_SYMBOLS))
    if rev_score > normal_score:
        return rev_score, True
    return normal_score, False


def find_sync_in_buffer(dibit_buffer: np.ndarray) -> tuple[int, bool]:
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


def test_full_integration_25k(
    freq_hz: int = 413_450_000,
    sample_rate: int = 6_000_000,
    duration_sec: float = 60.0,
) -> None:
    """Full integration test using 25 kHz baseband (like SDRTrunk)."""

    logger.info(f"=== Full Integration Test (25 kHz baseband) ===")
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

    # Three-stage decimation: 6 MHz → 200 kHz → 50 kHz → 25 kHz
    # Stage 1: 6 MHz → 200 kHz (30:1)
    stage1_factor = 30
    stage1_taps = scipy_signal.firwin(157, 0.8/stage1_factor, window=("kaiser", 7.857))
    stage1_zi_template = scipy_signal.lfilter_zi(stage1_taps, 1.0).astype(np.complex128)
    stage1_zi = None

    # Stage 2: 200 kHz → 50 kHz (4:1)
    stage2_factor = 4
    stage2_taps = scipy_signal.firwin(73, 0.8/stage2_factor, window=("kaiser", 7.857))
    stage2_zi_template = scipy_signal.lfilter_zi(stage2_taps, 1.0).astype(np.complex128)
    stage2_zi = None

    # Stage 3: 50 kHz → 25 kHz (2:1)
    stage3_factor = 2
    stage3_taps = scipy_signal.firwin(41, 0.8/stage3_factor, window=("kaiser", 7.857))
    stage3_zi_template = scipy_signal.lfilter_zi(stage3_taps, 1.0).astype(np.complex128)
    stage3_zi = None

    # Create C4FM demodulator at 25 kHz (matching SDRTrunk)
    baseband_rate = 25000
    demod = C4FMDemodulator(sample_rate=baseband_rate)

    logger.info(f"Baseband rate: {baseband_rate} Hz (5.2 samples/symbol)")

    # Stats
    stats = {
        "chunks": 0,
        "samples": 0,
        "syncs": 0,
        "sync_found": 0,
        "frames_decoded": 0,
        "nid_decoded": 0,
        "tsbk_attempts": 0,
        "tsbk_crc_pass": 0,
        "nacs": set(),
        "opcodes": {},
    }

    dibit_buffer = np.array([], dtype=np.uint8)
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

        # Stage 1: 6 MHz → 200 kHz
        if stage1_zi is None:
            stage1_zi = stage1_zi_template * iq[0]
        filtered1, stage1_zi = scipy_signal.lfilter(stage1_taps, 1.0, iq, zi=stage1_zi)
        decimated1 = filtered1[::stage1_factor]

        # Stage 2: 200 kHz → 50 kHz
        if stage2_zi is None:
            stage2_zi = stage2_zi_template * decimated1[0]
        filtered2, stage2_zi = scipy_signal.lfilter(stage2_taps, 1.0, decimated1, zi=stage2_zi)
        decimated2 = filtered2[::stage2_factor]

        # Stage 3: 50 kHz → 25 kHz
        if stage3_zi is None:
            stage3_zi = stage3_zi_template * decimated2[0]
        filtered3, stage3_zi = scipy_signal.lfilter(stage3_taps, 1.0, decimated2, zi=stage3_zi)
        baseband = filtered3[::stage3_factor].astype(np.complex64)

        # Demodulate
        try:
            dibits, soft_symbols = demod.demodulate(baseband)
            stats["syncs"] = demod._sync_count

            if len(dibits) > 0:
                if reverse_polarity:
                    dibits = dibits ^ 2

                dibit_buffer = np.concatenate([dibit_buffer, dibits.astype(np.uint8)])

                while len(dibit_buffer) >= TSDU_FRAME_DIBITS:
                    sync_pos, is_reversed = find_sync_in_buffer(dibit_buffer)

                    if sync_pos < 0:
                        if len(dibit_buffer) > FRAME_SYNC_DIBITS:
                            dibit_buffer = dibit_buffer[-(FRAME_SYNC_DIBITS - 1):]
                        break

                    stats["sync_found"] += 1

                    if not polarity_latched:
                        reverse_polarity = is_reversed
                        polarity_latched = True
                        if is_reversed:
                            logger.info("Polarity reversed detected")
                            dibit_buffer = dibit_buffer ^ 2

                    if sync_pos > 0:
                        dibit_buffer = dibit_buffer[sync_pos:]

                    if len(dibit_buffer) < TSDU_FRAME_DIBITS:
                        break

                    frame_dibits = dibit_buffer[:TSDU_FRAME_DIBITS]
                    stats["frames_decoded"] += 1

                    nid_dibits = frame_dibits[FRAME_SYNC_DIBITS:FRAME_SYNC_DIBITS + 33]
                    nid = p25_frames.decode_nid(nid_dibits, skip_status_at_10=True)

                    if nid is not None:
                        stats["nid_decoded"] += 1
                        stats["nacs"].add(nid.nac)

                        if nid.duid == p25_frames.DUID.TSDU:
                            tsdu = p25_frames.decode_tsdu(frame_dibits)
                            if tsdu is not None:
                                blocks = getattr(tsdu, 'tsbk_blocks', None) or getattr(tsdu, 'blocks', None)
                                if blocks:
                                    for block in blocks:
                                        stats["tsbk_attempts"] += 1
                                        crc_ok = getattr(block, 'crc_valid', False) or getattr(block, 'crc_ok', False)
                                        if crc_ok:
                                            stats["tsbk_crc_pass"] += 1
                                            opcode = block.opcode
                                            stats["opcodes"][opcode] = stats["opcodes"].get(opcode, 0) + 1
                                            if stats["tsbk_crc_pass"] <= 10:
                                                logger.info(f"TSBK CRC OK: NAC=0x{nid.nac:03x}, opcode=0x{opcode:02x}")
                                    # Successfully decoded TSDU - advance by full frame length
                                    dibit_buffer = dibit_buffer[TSDU_FRAME_DIBITS:]
                                    continue

                    # For failed decodes or non-TSDU, advance by sync length to re-search
                    dibit_buffer = dibit_buffer[FRAME_SYNC_DIBITS:]

        except Exception as e:
            logger.debug(f"Decode error: {e}")

        elapsed = time.time() - start_time
        if elapsed - (last_report - start_time) >= 10.0:
            crc_rate = (stats["tsbk_crc_pass"] / stats["tsbk_attempts"] * 100) if stats["tsbk_attempts"] > 0 else 0
            logger.info(
                f"{elapsed:.0f}s: syncs={stats['syncs']}, frame_syncs={stats['sync_found']}, "
                f"TSBK={stats['tsbk_crc_pass']}/{stats['tsbk_attempts']} ({crc_rate:.1f}%), "
                f"NACs={len(stats['nacs'])}"
            )
            last_report = time.time()

    sdr.deactivateStream(stream)
    sdr.closeStream(stream)

    elapsed = time.time() - start_time
    crc_rate = (stats["tsbk_crc_pass"] / stats["tsbk_attempts"] * 100) if stats["tsbk_attempts"] > 0 else 0

    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS")
    logger.info("="*60)
    logger.info(f"Duration: {elapsed:.1f}s")
    logger.info(f"C4FM sync detections: {stats['syncs']}")
    logger.info(f"Frame syncs found: {stats['sync_found']}")
    logger.info(f"Frames decoded: {stats['frames_decoded']}")
    logger.info(f"NID decoded: {stats['nid_decoded']}")
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
    parser = argparse.ArgumentParser(description="Full P25 integration test (25 kHz)")
    parser.add_argument("--freq", type=int, default=413_450_000, help="Frequency in Hz")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration in seconds")
    args = parser.parse_args()

    test_full_integration_25k(freq_hz=args.freq, duration_sec=args.duration)
