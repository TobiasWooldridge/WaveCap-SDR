#!/usr/bin/env python3
"""Test C4FM sync detection on live SDR signal.

Compares two approaches:
1. C4FMDemodulator (complex symbol-spaced differential)
2. Simple sample-by-sample FM demod with direct sync correlation

This helps identify where sync detection is failing.
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

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# P25 sync pattern: 0x5575F5FF77FF (48 bits = 24 dibits)
SYNC_PATTERN = 0x5575F5FF77FF
SYNC_THRESHOLD = 60.0

def get_sync_symbols() -> np.ndarray:
    """Convert sync pattern to ideal symbol values."""
    symbols = np.zeros(24, dtype=np.float32)
    pattern = SYNC_PATTERN
    for i in range(24):
        dibit = (pattern >> ((23 - i) * 2)) & 0x3
        # Sync pattern only contains dibits 1 (+3) and 3 (-3)
        symbols[i] = 3.0 if dibit == 1 else -3.0
    return symbols

SYNC_SYMBOLS = get_sync_symbols()


def correlate_sync(soft_symbols: np.ndarray) -> np.ndarray:
    """Correlate soft symbols against P25 sync pattern."""
    # Use cross-correlation with sync pattern (reversed for correlation)
    sync_rev = SYNC_SYMBOLS[::-1]
    correlation = np.correlate(soft_symbols, sync_rev, mode='valid')
    return correlation


def simple_fm_demod(iq: np.ndarray) -> np.ndarray:
    """Simple sample-by-sample FM demodulation."""
    # Differential phase: angle(iq[n] * conj(iq[n-1]))
    diff = iq[1:] * np.conj(iq[:-1])
    phase = np.angle(diff)
    return phase.astype(np.float32)


def symbol_spaced_fm_demod(iq: np.ndarray, symbol_delay: int = 10) -> np.ndarray:
    """Symbol-spaced differential FM demodulation (like C4FMDemodulator)."""
    # Delay by symbol_delay samples
    if len(iq) <= symbol_delay:
        return np.array([], dtype=np.float32)

    # Compare samples that are symbol_delay apart
    i = iq.real.astype(np.float32)
    q = iq.imag.astype(np.float32)

    i_delayed = i[:-symbol_delay]
    q_delayed = q[:-symbol_delay]
    i_current = i[symbol_delay:]
    q_current = q[symbol_delay:]

    # Differential demodulation
    demod_i = i_current * i_delayed + q_current * q_delayed
    demod_q = q_current * i_delayed - i_current * q_delayed

    phase = np.arctan2(demod_q, demod_i).astype(np.float32)
    return phase


def phases_to_symbols(phases: np.ndarray, samples_per_symbol: float) -> np.ndarray:
    """Extract symbols from phase samples at symbol rate."""
    n_symbols = int(len(phases) / samples_per_symbol)
    symbols = np.zeros(n_symbols, dtype=np.float32)

    for i in range(n_symbols):
        idx = int((i + 0.5) * samples_per_symbol)
        if idx < len(phases):
            # Normalize to ±3 range (4/π scaling)
            symbols[i] = phases[idx] * (4.0 / np.pi)

    return symbols


def test_sync_detection(
    freq_hz: int = 413_450_000,
    sample_rate: int = 6_000_000,
    duration_sec: float = 30.0,
) -> None:
    """Test sync detection approaches on live SDR signal."""

    logger.info(f"Testing sync detection on {freq_hz/1e6:.4f} MHz for {duration_sec}s")

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

    baseband_rate = 50000
    samples_per_symbol = baseband_rate / 4800  # 10.416...
    symbol_delay = int(np.ceil(samples_per_symbol))  # 11

    # Stats
    stats = {
        "chunks": 0,
        "simple_syncs": 0,
        "symbol_spaced_syncs": 0,
        "max_simple_score": 0.0,
        "max_symbol_score": 0.0,
    }

    chunk_size = sample_rate // 4
    buffer = np.zeros(chunk_size, dtype=np.complex64)

    logger.info("Starting decode loop...")
    logger.info(f"samples_per_symbol={samples_per_symbol:.3f}, symbol_delay={symbol_delay}")
    start_time = time.time()
    last_report = start_time

    while time.time() - start_time < duration_sec:
        sr = sdr.readStream(stream, [buffer], chunk_size)

        if sr.ret <= 0:
            continue

        iq = buffer[:sr.ret].astype(np.complex128)
        stats["chunks"] += 1

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

        # === Method 1: Simple sample-by-sample FM demod ===
        simple_phases = simple_fm_demod(baseband)
        simple_symbols = phases_to_symbols(simple_phases, samples_per_symbol)

        if len(simple_symbols) >= 24:
            simple_corr = correlate_sync(simple_symbols)
            simple_max = float(np.max(np.abs(simple_corr)))
            if simple_max > stats["max_simple_score"]:
                stats["max_simple_score"] = simple_max
            simple_syncs = np.sum(np.abs(simple_corr) >= SYNC_THRESHOLD)
            stats["simple_syncs"] += int(simple_syncs)

        # === Method 2: Symbol-spaced differential FM demod ===
        symbol_phases = symbol_spaced_fm_demod(baseband, symbol_delay)
        symbol_symbols = phases_to_symbols(symbol_phases, samples_per_symbol)

        if len(symbol_symbols) >= 24:
            symbol_corr = correlate_sync(symbol_symbols)
            symbol_max = float(np.max(np.abs(symbol_corr)))
            if symbol_max > stats["max_symbol_score"]:
                stats["max_symbol_score"] = symbol_max
            symbol_syncs = np.sum(np.abs(symbol_corr) >= SYNC_THRESHOLD)
            stats["symbol_spaced_syncs"] += int(symbol_syncs)

        # Report every 5 seconds
        elapsed = time.time() - start_time
        if elapsed - (last_report - start_time) >= 5.0:
            logger.info(
                f"{elapsed:.0f}s: simple_syncs={stats['simple_syncs']} (max={stats['max_simple_score']:.1f}), "
                f"symbol_syncs={stats['symbol_spaced_syncs']} (max={stats['max_symbol_score']:.1f})"
            )

            # Debug: print symbol statistics
            if len(simple_symbols) > 0:
                logger.debug(
                    f"  simple_symbols: mean={np.mean(simple_symbols):.3f}, "
                    f"std={np.std(simple_symbols):.3f}, "
                    f"range=[{np.min(simple_symbols):.3f}, {np.max(simple_symbols):.3f}]"
                )
            if len(symbol_symbols) > 0:
                logger.debug(
                    f"  symbol_symbols: mean={np.mean(symbol_symbols):.3f}, "
                    f"std={np.std(symbol_symbols):.3f}, "
                    f"range=[{np.min(symbol_symbols):.3f}, {np.max(symbol_symbols):.3f}]"
                )

            last_report = time.time()

    # Stop SDR
    sdr.deactivateStream(stream)
    sdr.closeStream(stream)

    # Final report
    elapsed = time.time() - start_time

    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS")
    logger.info("="*60)
    logger.info(f"Duration: {elapsed:.1f}s")
    logger.info(f"Chunks processed: {stats['chunks']}")
    logger.info(f"Simple FM syncs: {stats['simple_syncs']} (max score: {stats['max_simple_score']:.1f})")
    logger.info(f"Symbol-spaced syncs: {stats['symbol_spaced_syncs']} (max score: {stats['max_symbol_score']:.1f})")

    if stats['simple_syncs'] > 0 and stats['symbol_spaced_syncs'] == 0:
        logger.warning("Simple FM finds syncs but symbol-spaced doesn't - investigate symbol delay")
    elif stats['simple_syncs'] == 0 and stats['symbol_spaced_syncs'] > 0:
        logger.info("Symbol-spaced works better than simple FM")
    elif stats['simple_syncs'] == 0 and stats['symbol_spaced_syncs'] == 0:
        logger.warning("Neither method finds syncs - signal issue or wrong frequency")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test sync detection on live SDR")
    parser.add_argument("--freq", type=int, default=413_450_000, help="Frequency in Hz")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration in seconds")
    args = parser.parse_args()

    test_sync_detection(freq_hz=args.freq, duration_sec=args.duration)
