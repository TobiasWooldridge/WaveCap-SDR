#!/usr/bin/env python3
"""Debug C4FMDemodulator internals.

Compares each stage of C4FMDemodulator against simple implementations
to identify where the pipeline is failing.
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
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Add backend path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Mock profiler to avoid fastapi
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

# Now import C4FMDemodulator components
import importlib.util
spec = importlib.util.spec_from_file_location("c4fm", str(backend_path / "wavecapsdr" / "dsp" / "p25" / "c4fm.py"))
c4fm = importlib.util.module_from_spec(spec)
sys.modules["c4fm"] = c4fm
spec.loader.exec_module(c4fm)

C4FMDemodulator = c4fm.C4FMDemodulator
_FMDemodulator = c4fm._FMDemodulator
_SoftSyncDetector = c4fm._SoftSyncDetector
design_baseband_lpf = c4fm.design_baseband_lpf
design_rrc_filter = c4fm.design_rrc_filter

# P25 sync pattern
SYNC_PATTERN = 0x5575F5FF77FF
SYNC_THRESHOLD = 60.0
SYMBOL_RATE = 4800

def get_sync_symbols() -> np.ndarray:
    symbols = np.zeros(24, dtype=np.float32)
    for i in range(24):
        dibit = (SYNC_PATTERN >> ((23 - i) * 2)) & 0x3
        symbols[i] = 3.0 if dibit == 1 else -3.0
    return symbols

SYNC_SYMBOLS = get_sync_symbols()

def simple_symbol_spaced_fm(i: np.ndarray, q: np.ndarray, delay: int) -> np.ndarray:
    """Reference implementation of symbol-spaced FM demod."""
    n = len(i)
    if n <= delay:
        return np.array([], dtype=np.float32)

    i_delayed = np.zeros(n, dtype=np.float32)
    q_delayed = np.zeros(n, dtype=np.float32)
    i_delayed[delay:] = i[:-delay]
    q_delayed[delay:] = q[:-delay]

    demod_i = i * i_delayed + q * q_delayed
    demod_q = q * i_delayed - i * q_delayed
    return np.arctan2(demod_q, demod_i).astype(np.float32)


def test_fm_demodulator():
    """Test _FMDemodulator against reference implementation."""
    logger.info("=== Testing _FMDemodulator ===")

    sample_rate = 50000
    symbol_delay = 11

    # Create test signal - C4FM-like with symbols at ±π/4, ±3π/4
    n_samples = 5000
    t = np.arange(n_samples) / sample_rate

    # Generate phase modulated signal
    symbols = np.random.choice([1, 3, -1, -3], size=n_samples // 10)
    symbol_phases = np.repeat(symbols * np.pi / 4.0, 10)
    if len(symbol_phases) > n_samples:
        symbol_phases = symbol_phases[:n_samples]
    elif len(symbol_phases) < n_samples:
        symbol_phases = np.concatenate([symbol_phases, np.zeros(n_samples - len(symbol_phases))])

    # Create IQ from phase
    iq = np.exp(1j * symbol_phases).astype(np.complex64)
    i = iq.real.astype(np.float32)
    q = iq.imag.astype(np.float32)

    # Reference implementation
    ref_phases = simple_symbol_spaced_fm(i, q, symbol_delay)

    # C4FMDemodulator's _FMDemodulator
    fm_demod = _FMDemodulator(symbol_delay=symbol_delay)
    demod_phases = fm_demod.demodulate(i, q)

    # Compare
    # Skip first symbol_delay samples where delay buffer isn't filled
    start_idx = symbol_delay * 2
    ref_trimmed = ref_phases[start_idx:]
    demod_trimmed = demod_phases[start_idx:]

    min_len = min(len(ref_trimmed), len(demod_trimmed))
    if min_len > 0:
        diff = np.abs(ref_trimmed[:min_len] - demod_trimmed[:min_len])
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)
        logger.info(f"  Reference phases: mean={np.mean(ref_trimmed[:min_len]):.4f}, std={np.std(ref_trimmed[:min_len]):.4f}")
        logger.info(f"  Demod phases:     mean={np.mean(demod_trimmed[:min_len]):.4f}, std={np.std(demod_trimmed[:min_len]):.4f}")
        logger.info(f"  Difference: mean={mean_diff:.6f}, max={max_diff:.6f}")

        if max_diff < 0.01:
            logger.info("  ✓ _FMDemodulator matches reference")
        else:
            logger.warning(f"  ✗ _FMDemodulator differs from reference by up to {max_diff:.6f}")


def test_sync_detector():
    """Test _SoftSyncDetector."""
    logger.info("\n=== Testing _SoftSyncDetector ===")

    detector = _SoftSyncDetector()

    # Feed perfect sync pattern
    perfect_symbols = SYNC_SYMBOLS.copy()
    scores = []
    for symbol in perfect_symbols:
        score = detector.process(symbol)
        scores.append(score)

    final_score = scores[-1]
    max_possible = 24 * 9  # 24 symbols * (3*3) max correlation
    logger.info(f"  Perfect sync score: {final_score:.1f} (max possible: {max_possible})")

    if final_score >= SYNC_THRESHOLD:
        logger.info(f"  ✓ Sync detected (threshold={SYNC_THRESHOLD})")
    else:
        logger.warning(f"  ✗ Sync NOT detected (score={final_score:.1f} < threshold={SYNC_THRESHOLD})")

    # Test with noise
    detector.reset()
    noisy_symbols = perfect_symbols + np.random.randn(24).astype(np.float32) * 0.5
    for symbol in noisy_symbols:
        score = detector.process(symbol)
    logger.info(f"  Noisy sync score: {score:.1f}")


def test_full_demodulator_on_synthetic():
    """Test C4FMDemodulator on synthetic P25 signal."""
    logger.info("\n=== Testing C4FMDemodulator on synthetic signal ===")

    sample_rate = 50000
    samples_per_symbol = sample_rate / SYMBOL_RATE

    # Create C4FMDemodulator
    demod = C4FMDemodulator(sample_rate=sample_rate)

    # Generate synthetic P25 frame with sync pattern
    n_symbols = 200  # ~200 symbols
    symbols = []

    # Add sync pattern (24 dibits)
    for i in range(24):
        dibit = (SYNC_PATTERN >> ((23 - i) * 2)) & 0x3
        if dibit == 0:
            symbols.append(1)
        elif dibit == 1:
            symbols.append(3)
        elif dibit == 2:
            symbols.append(-1)
        else:
            symbols.append(-3)

    # Add random data
    for _ in range(n_symbols - 24):
        symbols.append(np.random.choice([1, 3, -1, -3]))

    symbols = np.array(symbols, dtype=np.float32)

    # Expand to sample rate
    n_samples = int(len(symbols) * samples_per_symbol) + 100
    phase = np.zeros(n_samples, dtype=np.float32)
    for i, sym in enumerate(symbols):
        start_idx = int(i * samples_per_symbol)
        end_idx = int((i + 1) * samples_per_symbol)
        if end_idx <= n_samples:
            phase[start_idx:end_idx] = sym * np.pi / 4.0

    # Create IQ from phase (FM modulated)
    # For symbol-spaced demod, we need phase difference = symbol phase
    # So cumulative phase should be integral of symbol phases
    cumulative_phase = np.cumsum(phase) / samples_per_symbol
    iq = np.exp(1j * cumulative_phase).astype(np.complex64)

    # Demodulate
    dibits, soft_symbols = demod.demodulate(iq)

    logger.info(f"  Input: {len(iq)} samples ({len(iq)/sample_rate:.3f} sec)")
    logger.info(f"  Output: {len(dibits)} dibits, {len(soft_symbols)} soft symbols")
    logger.info(f"  Sync count: {demod._sync_count}")

    if len(soft_symbols) > 0:
        soft_arr = np.array(soft_symbols)
        logger.info(f"  Soft symbols: mean={np.mean(soft_arr):.3f}, std={np.std(soft_arr):.3f}")
        logger.info(f"  Soft symbols range: [{np.min(soft_arr):.3f}, {np.max(soft_arr):.3f}]")

    if demod._sync_count > 0:
        logger.info("  ✓ Sync detected on synthetic signal")
    else:
        logger.warning("  ✗ No sync on synthetic signal - demodulator issue")


def test_demodulator_output_stats():
    """Test C4FMDemodulator output statistics with various inputs."""
    logger.info("\n=== Testing C4FMDemodulator output stats ===")

    sample_rate = 50000
    duration = 0.1  # 100ms
    n_samples = int(sample_rate * duration)

    demod = C4FMDemodulator(sample_rate=sample_rate)

    # Test 1: White noise
    noise = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)).astype(np.complex64) * 0.1
    dibits, soft = demod.demodulate(noise)
    if len(soft) > 0:
        logger.info(f"  White noise: {len(soft)} symbols, std={np.std(soft):.3f}, syncs={demod._sync_count}")

    # Test 2: Pure tone (DC)
    demod.reset()
    dc = np.ones(n_samples, dtype=np.complex64) * 0.5
    dibits, soft = demod.demodulate(dc)
    if len(soft) > 0:
        logger.info(f"  DC signal: {len(soft)} symbols, std={np.std(soft):.3f}, syncs={demod._sync_count}")

    # Test 3: FM tone (frequency offset)
    demod.reset()
    freq_offset = 1000  # 1 kHz
    t = np.arange(n_samples) / sample_rate
    fm_tone = np.exp(2j * np.pi * freq_offset * t).astype(np.complex64) * 0.5
    dibits, soft = demod.demodulate(fm_tone)
    if len(soft) > 0:
        logger.info(f"  FM tone (1kHz): {len(soft)} symbols, std={np.std(soft):.3f}, syncs={demod._sync_count}")


if __name__ == "__main__":
    test_fm_demodulator()
    test_sync_detector()
    test_full_demodulator_on_synthetic()
    test_demodulator_output_stats()

    logger.info("\n" + "="*60)
    logger.info("SUMMARY: Check above for ✗ marks indicating failures")
