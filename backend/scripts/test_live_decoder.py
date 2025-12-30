#!/usr/bin/env python3
"""Test WaveCap P25 decoder on live SDR signal.

This script bypasses the web app to test the core decoding logic.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Add system site-packages for SoapySDR
sys.path.append('/opt/homebrew/lib/python3.14/site-packages')

import numpy as np

try:
    import SoapySDR
except ImportError:
    print("Error: SoapySDR not installed")
    sys.exit(1)

from scipy import signal as scipy_signal

# Add backend to path - but avoid importing wavecapsdr package (needs fastapi)
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Mock the profiler to avoid fastapi dependency chain
from types import ModuleType
import contextlib

class MockProfiler:
    """Stub profiler for standalone testing."""
    def __init__(self, name, enabled=False):
        self.name = name
        self.enabled = enabled
    @contextlib.contextmanager
    def measure(self, name):
        yield
    def start(self, name):
        pass
    def stop(self, name):
        pass
    def report(self):
        pass

def get_profiler(name, enabled=False):
    return MockProfiler(name, enabled)

# Create mock wavecapsdr.utils.profiler module
mock_profiler_module = ModuleType("wavecapsdr.utils.profiler")
mock_profiler_module.get_profiler = get_profiler
mock_profiler_module.MockProfiler = MockProfiler

# Create mock wavecapsdr package hierarchy
mock_wavecapsdr = ModuleType("wavecapsdr")
mock_utils = ModuleType("wavecapsdr.utils")
sys.modules["wavecapsdr"] = mock_wavecapsdr
sys.modules["wavecapsdr.utils"] = mock_utils
sys.modules["wavecapsdr.utils.profiler"] = mock_profiler_module

# Import decoder components directly to avoid fastapi dependency
# We import the modules directly rather than through the package
import importlib.util

def import_module_directly(module_name: str, file_path: str):
    """Import a module directly from file path to avoid package __init__."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import c4fm module directly
c4fm = import_module_directly(
    "c4fm",
    str(backend_path / "wavecapsdr" / "dsp" / "p25" / "c4fm.py")
)
C4FMDemodulator = c4fm.C4FMDemodulator

# Import p25_frames module directly
p25_frames = import_module_directly(
    "p25_frames",
    str(backend_path / "wavecapsdr" / "decoders" / "p25_frames.py")
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def test_live_decoder(
    freq_hz: int = 413_450_000,
    sample_rate: int = 6_000_000,
    duration_sec: float = 30.0,
) -> None:
    """Test P25 decoder on live SDR signal."""

    logger.info(f"Testing P25 decoder on {freq_hz/1e6:.4f} MHz for {duration_sec}s")

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

    # Create demodulator
    baseband_rate = 50000
    demod = C4FMDemodulator(sample_rate=baseband_rate)

    # Stats
    stats = {
        "chunks": 0,
        "samples": 0,
        "syncs": 0,
        "frames": 0,
        "tsbk_attempts": 0,
        "tsbk_crc_pass": 0,
        "nacs": set(),
    }

    # Accumulated dibits buffer
    dibit_buffer = np.array([], dtype=np.int8)

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

        # Initialize stage2_zi with first sample
        if stage2_zi is None:
            stage2_zi = stage2_zi_template * decimated1[0]

        filtered2, stage2_zi = scipy_signal.lfilter(stage2_taps, 1.0, decimated1, zi=stage2_zi)
        baseband = filtered2[::stage2_factor].astype(np.complex64)

        # Demodulate - returns tuple (dibits, soft_symbols)
        try:
            dibits, soft_symbols = demod.demodulate(baseband)

            # Track sync count from demodulator (internal counter)
            stats["syncs"] = demod._sync_count

            if dibits is not None and len(dibits) > 0:
                # Accumulate dibits
                dibit_buffer = np.concatenate([dibit_buffer, dibits])

                # Try to decode TSDU frames (TSBK)
                # TSDU is 432 dibits (864 bits)
                while len(dibit_buffer) >= 432:
                    frame_dibits = dibit_buffer[:432]
                    dibit_buffer = dibit_buffer[432:]

                    # Try to decode NID first
                    nid = p25_frames.decode_nid(frame_dibits[:48])
                    if nid is not None:
                        stats["nacs"].add(nid.nac)
                        stats["frames"] += 1

                        # Try TSDU decode
                        if nid.duid == p25_frames.DUID.TSDU:
                            tsdu = p25_frames.decode_tsdu(frame_dibits)
                            if tsdu is not None:
                                for block in tsdu.blocks:
                                    stats["tsbk_attempts"] += 1
                                    if block.crc_ok:
                                        stats["tsbk_crc_pass"] += 1
                                        logger.info(f"TSBK CRC OK: NAC=0x{nid.nac:03x}, opcode={block.opcode}")

        except Exception as e:
            pass  # Ignore decode errors

        # Report every 5 seconds
        elapsed = time.time() - start_time
        if elapsed - (last_report - start_time) >= 5.0:
            crc_rate = (stats["tsbk_crc_pass"] / stats["tsbk_attempts"] * 100) if stats["tsbk_attempts"] > 0 else 0
            logger.info(
                f"{elapsed:.0f}s: syncs={stats['syncs']}, frames={stats['frames']}, "
                f"TSBK={stats['tsbk_attempts']}, CRC={stats['tsbk_crc_pass']} ({crc_rate:.1f}%), "
                f"NACs={stats['nacs']}"
            )
            last_report = time.time()

    # Stop SDR
    sdr.deactivateStream(stream)
    sdr.closeStream(stream)

    # Final report
    elapsed = time.time() - start_time
    crc_rate = (stats["tsbk_crc_pass"] / stats["tsbk_attempts"] * 100) if stats["tsbk_attempts"] > 0 else 0

    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS")
    logger.info("="*60)
    logger.info(f"Duration: {elapsed:.1f}s")
    logger.info(f"Chunks processed: {stats['chunks']}")
    logger.info(f"Samples processed: {stats['samples']/1e6:.1f}M")
    logger.info(f"Sync detections: {stats['syncs']}")
    logger.info(f"Frames decoded: {stats['frames']}")
    logger.info(f"TSBK attempts: {stats['tsbk_attempts']}")
    logger.info(f"TSBK CRC pass: {stats['tsbk_crc_pass']} ({crc_rate:.1f}%)")
    logger.info(f"NACs seen: {stats['nacs']}")

    if crc_rate >= 30:
        logger.info("✓ SUCCESS: CRC pass rate >= 30%")
    else:
        logger.warning(f"✗ LOW CRC: Pass rate {crc_rate:.1f}% (expected >= 30%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test P25 decoder on live SDR")
    parser.add_argument("--freq", type=int, default=413_450_000, help="Frequency in Hz")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration in seconds")
    args = parser.parse_args()

    test_live_decoder(freq_hz=args.freq, duration_sec=args.duration)
