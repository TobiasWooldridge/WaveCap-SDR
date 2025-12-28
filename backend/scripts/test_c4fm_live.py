#!/usr/bin/env python3
"""Test C4FMDemodulator sync detection on live SDR signal.

This is a minimal test that only tests the C4FMDemodulator's ability
to detect sync patterns on a live P25 control channel.
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

# Add backend path and mock dependencies
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from types import ModuleType
import contextlib

class MockProfiler:
    def __init__(self, name, enabled=False): pass
    @contextlib.contextmanager
    def measure(self, name): yield
    def start(self, name): pass
    def stop(self, name): pass
    def report(self): pass

mock_profiler = ModuleType("wavecapsdr.utils.profiler")
mock_profiler.get_profiler = lambda name, enabled=False: MockProfiler(name, enabled)
sys.modules["wavecapsdr"] = ModuleType("wavecapsdr")
sys.modules["wavecapsdr.utils"] = ModuleType("wavecapsdr.utils")
sys.modules["wavecapsdr.utils.profiler"] = mock_profiler

# Import C4FMDemodulator
import importlib.util
spec = importlib.util.spec_from_file_location("c4fm", str(backend_path / "wavecapsdr" / "dsp" / "p25" / "c4fm.py"))
c4fm = importlib.util.module_from_spec(spec)
sys.modules["c4fm"] = c4fm
spec.loader.exec_module(c4fm)

C4FMDemodulator = c4fm.C4FMDemodulator


def test_c4fm_live(
    freq_hz: int = 413_450_000,
    sample_rate: int = 6_000_000,
    duration_sec: float = 30.0,
) -> None:
    """Test C4FMDemodulator sync detection on live SDR."""

    logger.info(f"Testing C4FMDemodulator on {freq_hz/1e6:.4f} MHz for {duration_sec}s")

    # Open SDR
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
        "dibits": 0,
        "syncs": 0,
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

        # Demodulate
        try:
            dibits, soft_symbols = demod.demodulate(baseband)
            stats["syncs"] = demod._sync_count
            stats["dibits"] += len(dibits)
        except Exception as e:
            logger.error(f"Demod error: {e}")

        # Report every 5 seconds
        elapsed = time.time() - start_time
        if elapsed - (last_report - start_time) >= 5.0:
            logger.info(
                f"{elapsed:.0f}s: syncs={stats['syncs']}, dibits={stats['dibits']}, "
                f"chunks={stats['chunks']}, samples={stats['samples']/1e6:.1f}M"
            )
            if len(soft_symbols) > 0:
                soft = np.array(soft_symbols)
                logger.info(
                    f"  Symbols: mean={np.mean(soft):.3f}, std={np.std(soft):.3f}, "
                    f"range=[{np.min(soft):.3f}, {np.max(soft):.3f}]"
                )
            last_report = time.time()

    sdr.deactivateStream(stream)
    sdr.closeStream(stream)

    # Final report
    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS")
    logger.info("="*60)
    logger.info(f"Duration: {time.time() - start_time:.1f}s")
    logger.info(f"Syncs detected: {stats['syncs']}")
    logger.info(f"Dibits produced: {stats['dibits']}")
    logger.info(f"Samples processed: {stats['samples']/1e6:.1f}M")

    if stats['syncs'] > 0:
        logger.info(f"✓ SUCCESS: C4FMDemodulator detected {stats['syncs']} syncs")
    else:
        logger.warning("✗ FAILED: No syncs detected")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test C4FMDemodulator on live SDR")
    parser.add_argument("--freq", type=int, default=413_450_000, help="Frequency in Hz")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration in seconds")
    args = parser.parse_args()

    test_c4fm_live(freq_hz=args.freq, duration_sec=args.duration)
