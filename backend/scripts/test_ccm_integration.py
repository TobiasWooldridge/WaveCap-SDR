#!/usr/bin/env python3
"""Integration test using ControlChannelMonitor directly.

Uses the production ControlChannelMonitor class for proper frame handling.
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
from wavecapsdr.trunking.config import TrunkingProtocol
from wavecapsdr.trunking.control_channel import ControlChannelMonitor


def test_ccm_integration(
    freq_hz: int = 413_450_000,
    sample_rate: int = 6_000_000,
    duration_sec: float = 60.0,
) -> None:
    """Test ControlChannelMonitor on live SDR."""

    logger.info(f"=== ControlChannelMonitor Integration Test ===")
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

    # Create ControlChannelMonitor at 50 kHz
    baseband_rate = 50000
    ccm = ControlChannelMonitor(
        protocol=TrunkingProtocol.P25_PHASE1,
        sample_rate=baseband_rate,
    )

    # Stats
    stats = {
        "chunks": 0,
        "samples": 0,
        "tsbk_messages": 0,
    }

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

        # Process through ControlChannelMonitor
        try:
            results = ccm.process_iq(baseband)
            stats["tsbk_messages"] += len(results)

            for result in results:
                if stats["tsbk_messages"] <= 10:
                    logger.info(f"TSBK: {result}")

        except Exception as e:
            logger.debug(f"CCM error: {e}")

        # Report every 10 seconds
        elapsed = time.time() - start_time
        if elapsed - (last_report - start_time) >= 10.0:
            crc_rate = (ccm.tsbk_crc_pass / ccm.tsbk_attempts * 100) if ccm.tsbk_attempts > 0 else 0
            logger.info(
                f"{elapsed:.0f}s: frames={ccm.frames_decoded}, "
                f"TSBK={ccm.tsbk_crc_pass}/{ccm.tsbk_attempts} ({crc_rate:.1f}%), "
                f"sync_losses={ccm.sync_losses}"
            )
            last_report = time.time()

    sdr.deactivateStream(stream)
    sdr.closeStream(stream)

    # Final report
    elapsed = time.time() - start_time
    crc_rate = (ccm.tsbk_crc_pass / ccm.tsbk_attempts * 100) if ccm.tsbk_attempts > 0 else 0

    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS")
    logger.info("="*60)
    logger.info(f"Duration: {elapsed:.1f}s")
    logger.info(f"Samples processed: {stats['samples']/1e6:.1f}M")
    logger.info(f"Frames decoded: {ccm.frames_decoded}")
    logger.info(f"TSBK CRC pass rate: {ccm.tsbk_crc_pass}/{ccm.tsbk_attempts} ({crc_rate:.1f}%)")
    logger.info(f"Sync losses: {ccm.sync_losses}")
    logger.info(f"TSBK messages extracted: {stats['tsbk_messages']}")

    logger.info("="*60)
    if crc_rate >= 30:
        logger.info(f"SUCCESS: TSBK CRC pass rate {crc_rate:.1f}% >= 30%")
    elif crc_rate > 0:
        logger.warning(f"PARTIAL: TSBK CRC pass rate {crc_rate:.1f}% < 30%")
    else:
        logger.error(f"FAILED: No TSBK CRC passes")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CCM integration test")
    parser.add_argument("--freq", type=int, default=413_450_000, help="Frequency in Hz")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration in seconds")
    args = parser.parse_args()

    test_ccm_integration(freq_hz=args.freq, duration_sec=args.duration)
