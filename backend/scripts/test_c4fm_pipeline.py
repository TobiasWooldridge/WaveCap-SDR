#!/usr/bin/env python3
"""Test C4FM demodulation pipeline components.

Tests different combinations of:
1. Baseband LPF + FM demod (no RRC)
2. Baseband LPF + RRC + FM demod (current C4FMDemodulator approach)
3. FM demod + RRC post-filter (SDRTrunk approach)

This helps identify which pipeline produces the best sync detection.
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
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# P25 sync pattern: 0x5575F5FF77FF
SYNC_PATTERN = 0x5575F5FF77FF
SYNC_THRESHOLD = 60.0
SYMBOL_RATE = 4800


def get_sync_symbols() -> np.ndarray:
    """Convert sync pattern to ideal symbol values."""
    symbols = np.zeros(24, dtype=np.float32)
    pattern = SYNC_PATTERN
    for i in range(24):
        dibit = (pattern >> ((23 - i) * 2)) & 0x3
        symbols[i] = 3.0 if dibit == 1 else -3.0
    return symbols

SYNC_SYMBOLS = get_sync_symbols()


def design_baseband_lpf(sample_rate: float) -> np.ndarray:
    """Design baseband LPF (5.2kHz passband, 6.5kHz stopband)."""
    from scipy.signal import remez
    nyquist = sample_rate / 2.0
    bands = [0, 5200, 6500, nyquist]
    desired = [1, 0]
    try:
        h = remez(63, bands, desired, Hz=sample_rate)
    except Exception:
        h = scipy_signal.firwin(63, 5200, fs=sample_rate, window='hamming')
    return h.astype(np.float32)


def design_rrc_filter(samples_per_symbol: float, num_taps: int = 101, alpha: float = 0.2) -> np.ndarray:
    """Design Root-Raised Cosine filter."""
    if num_taps % 2 == 0:
        num_taps += 1
    n = np.arange(num_taps) - (num_taps - 1) / 2
    t = n / samples_per_symbol
    h = np.zeros(num_taps, dtype=np.float64)

    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = (1 - alpha + 4 * alpha / np.pi)
        elif abs(ti) == 1 / (4 * alpha):
            h[i] = (alpha / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
            )
        else:
            num = np.sin(np.pi * ti * (1 - alpha)) + 4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha))
            den = np.pi * ti * (1 - (4 * alpha * ti) ** 2)
            if den != 0:
                h[i] = num / den

    h = h / np.sum(h)
    return h.astype(np.float32)


def symbol_spaced_fm_demod(i: np.ndarray, q: np.ndarray, symbol_delay: int) -> np.ndarray:
    """Symbol-spaced differential FM demodulation."""
    n = len(i)
    if n <= symbol_delay:
        return np.array([], dtype=np.float32)

    i_delayed = np.zeros(n, dtype=np.float32)
    q_delayed = np.zeros(n, dtype=np.float32)
    i_delayed[symbol_delay:] = i[:-symbol_delay]
    q_delayed[symbol_delay:] = q[:-symbol_delay]

    # Differential demodulation
    demod_i = i * i_delayed + q * q_delayed
    demod_q = q * i_delayed - i * q_delayed

    phase = np.arctan2(demod_q, demod_i).astype(np.float32)
    return phase


def extract_symbols(phases: np.ndarray, samples_per_symbol: float) -> np.ndarray:
    """Extract symbols at symbol rate."""
    n_symbols = int(len(phases) / samples_per_symbol)
    if n_symbols == 0:
        return np.array([], dtype=np.float32)

    symbols = np.zeros(n_symbols, dtype=np.float32)
    for i in range(n_symbols):
        idx = int((i + 0.5) * samples_per_symbol)
        if idx < len(phases):
            symbols[i] = phases[idx] * (4.0 / np.pi)
    return symbols


def correlate_sync(symbols: np.ndarray) -> int:
    """Count sync detections above threshold."""
    if len(symbols) < 24:
        return 0, 0.0
    sync_rev = SYNC_SYMBOLS[::-1]
    corr = np.correlate(symbols, sync_rev, mode='valid')
    max_score = float(np.max(np.abs(corr)))
    syncs = int(np.sum(np.abs(corr) >= SYNC_THRESHOLD))
    return syncs, max_score


def test_pipeline(
    freq_hz: int = 413_450_000,
    sample_rate: int = 6_000_000,
    duration_sec: float = 30.0,
) -> None:
    """Test different C4FM demodulation pipelines."""

    logger.info(f"Testing C4FM pipelines on {freq_hz/1e6:.4f} MHz for {duration_sec}s")

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

    baseband_rate = 50000
    samples_per_symbol = baseband_rate / SYMBOL_RATE
    symbol_delay = int(np.ceil(samples_per_symbol))

    # Design filters
    bb_lpf = design_baseband_lpf(baseband_rate)
    rrc = design_rrc_filter(samples_per_symbol, num_taps=int(16 * samples_per_symbol) + 1, alpha=0.2)

    # Filter states (persistent across chunks)
    bb_zi_i = scipy_signal.lfilter_zi(bb_lpf, 1.0).astype(np.float32)
    bb_zi_q = scipy_signal.lfilter_zi(bb_lpf, 1.0).astype(np.float32)
    rrc_zi = scipy_signal.lfilter_zi(rrc, 1.0).astype(np.float32)

    # FM demod delay buffers
    fm_i_buf = np.zeros(symbol_delay, dtype=np.float32)
    fm_q_buf = np.zeros(symbol_delay, dtype=np.float32)

    # Stats for each pipeline
    pipelines = {
        "raw_fm": {"syncs": 0, "max": 0.0},           # Raw FM demod, no filtering
        "bblpf_fm": {"syncs": 0, "max": 0.0},         # Baseband LPF + FM demod
        "bblpf_rrc_fm": {"syncs": 0, "max": 0.0},     # Baseband LPF + RRC pre + FM demod (current)
        "bblpf_fm_rrc": {"syncs": 0, "max": 0.0},     # Baseband LPF + FM demod + RRC post
    }

    chunk_size = sample_rate // 4
    buffer = np.zeros(chunk_size, dtype=np.complex64)

    logger.info(f"samples_per_symbol={samples_per_symbol:.3f}, symbol_delay={symbol_delay}")
    start_time = time.time()
    last_report = start_time

    while time.time() - start_time < duration_sec:
        sr = sdr.readStream(stream, [buffer], chunk_size)
        if sr.ret <= 0:
            continue

        iq = buffer[:sr.ret].astype(np.complex128)

        # Decimation
        if stage1_zi is None:
            stage1_zi = stage1_zi_template * iq[0]
        filtered1, stage1_zi = scipy_signal.lfilter(stage1_taps, 1.0, iq, zi=stage1_zi)
        decimated1 = filtered1[::stage1_factor]

        if stage2_zi is None:
            stage2_zi = stage2_zi_template * decimated1[0]
        filtered2, stage2_zi = scipy_signal.lfilter(stage2_taps, 1.0, decimated1, zi=stage2_zi)
        baseband = filtered2[::stage2_factor].astype(np.complex64)

        i_raw = baseband.real.astype(np.float32)
        q_raw = baseband.imag.astype(np.float32)

        # Pipeline 1: Raw FM demod (no baseband filtering)
        phases_raw = symbol_spaced_fm_demod(i_raw, q_raw, symbol_delay)
        symbols_raw = extract_symbols(phases_raw, samples_per_symbol)
        syncs, max_score = correlate_sync(symbols_raw)
        pipelines["raw_fm"]["syncs"] += syncs
        pipelines["raw_fm"]["max"] = max(pipelines["raw_fm"]["max"], max_score)

        # Pipeline 2: Baseband LPF + FM demod
        i_lpf, bb_zi_i = scipy_signal.lfilter(bb_lpf, 1.0, i_raw, zi=bb_zi_i * (i_raw[0] if bb_zi_i[0] == 0 else 1))
        q_lpf, bb_zi_q = scipy_signal.lfilter(bb_lpf, 1.0, q_raw, zi=bb_zi_q * (q_raw[0] if bb_zi_q[0] == 0 else 1))
        phases_lpf = symbol_spaced_fm_demod(i_lpf.astype(np.float32), q_lpf.astype(np.float32), symbol_delay)
        symbols_lpf = extract_symbols(phases_lpf, samples_per_symbol)
        syncs, max_score = correlate_sync(symbols_lpf)
        pipelines["bblpf_fm"]["syncs"] += syncs
        pipelines["bblpf_fm"]["max"] = max(pipelines["bblpf_fm"]["max"], max_score)

        # Pipeline 3: Baseband LPF + RRC pre-filter + FM demod (CURRENT C4FMDemodulator)
        # Apply RRC to filtered I/Q before FM demod
        i_rrc = scipy_signal.lfilter(rrc, 1.0, i_lpf)
        q_rrc = scipy_signal.lfilter(rrc, 1.0, q_lpf)
        phases_pre_rrc = symbol_spaced_fm_demod(i_rrc.astype(np.float32), q_rrc.astype(np.float32), symbol_delay)
        symbols_pre_rrc = extract_symbols(phases_pre_rrc, samples_per_symbol)
        syncs, max_score = correlate_sync(symbols_pre_rrc)
        pipelines["bblpf_rrc_fm"]["syncs"] += syncs
        pipelines["bblpf_rrc_fm"]["max"] = max(pipelines["bblpf_rrc_fm"]["max"], max_score)

        # Pipeline 4: Baseband LPF + FM demod + RRC post-filter (SDRTrunk-style)
        phases_post_rrc, rrc_zi = scipy_signal.lfilter(rrc, 1.0, phases_lpf, zi=rrc_zi * (phases_lpf[0] if rrc_zi[0] == 0 else 1))
        symbols_post_rrc = extract_symbols(phases_post_rrc.astype(np.float32), samples_per_symbol)
        syncs, max_score = correlate_sync(symbols_post_rrc)
        pipelines["bblpf_fm_rrc"]["syncs"] += syncs
        pipelines["bblpf_fm_rrc"]["max"] = max(pipelines["bblpf_fm_rrc"]["max"], max_score)

        # Report every 5 seconds
        elapsed = time.time() - start_time
        if elapsed - (last_report - start_time) >= 5.0:
            logger.info(f"\n{elapsed:.0f}s results:")
            for name, stats in pipelines.items():
                logger.info(f"  {name:15s}: syncs={stats['syncs']:5d}, max={stats['max']:.1f}")
            last_report = time.time()

    sdr.deactivateStream(stream)
    sdr.closeStream(stream)

    # Final report
    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS")
    logger.info("="*60)
    for name, stats in pipelines.items():
        logger.info(f"  {name:15s}: syncs={stats['syncs']:5d}, max_score={stats['max']:.1f}")

    # Identify winner
    best = max(pipelines.keys(), key=lambda k: pipelines[k]["syncs"])
    logger.info(f"\nBest pipeline: {best} with {pipelines[best]['syncs']} syncs")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test C4FM demod pipelines")
    parser.add_argument("--freq", type=int, default=413_450_000, help="Frequency in Hz")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration in seconds")
    args = parser.parse_args()

    test_pipeline(freq_hz=args.freq, duration_sec=args.duration)
