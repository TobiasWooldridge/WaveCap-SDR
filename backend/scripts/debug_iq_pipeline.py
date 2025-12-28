#!/usr/bin/env python3
"""Debug IQ pipeline to find where signal gets corrupted.

Records IQ at each stage:
1. Raw SDR (6 MHz)
2. After frequency shift
3. After stage 1 decimation (200 kHz)
4. After stage 2 decimation (50 kHz)
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

try:
    import SoapySDR
except ImportError:
    print("Error: SoapySDR not installed")
    sys.exit(1)

from scipy import signal as scipy_signal

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_iq(iq: np.ndarray, name: str) -> None:
    """Analyze and print IQ statistics."""
    logger.info(f"\n=== {name} ===")
    logger.info(f"  Shape: {iq.shape}, dtype: {iq.dtype}")
    logger.info(f"  I: mean={np.mean(iq.real):.6f}, std={np.std(iq.real):.6f}, "
                f"range=[{np.min(iq.real):.6f}, {np.max(iq.real):.6f}]")
    logger.info(f"  Q: mean={np.mean(iq.imag):.6f}, std={np.std(iq.imag):.6f}, "
                f"range=[{np.min(iq.imag):.6f}, {np.max(iq.imag):.6f}]")
    logger.info(f"  Power: {np.mean(np.abs(iq)**2):.9f}")
    logger.info(f"  DC offset: {np.abs(np.mean(iq)):.6f}")

    # Check for NaN/Inf
    if np.any(np.isnan(iq)):
        logger.warning(f"  *** Contains NaN! ***")
    if np.any(np.isinf(iq)):
        logger.warning(f"  *** Contains Inf! ***")


def debug_pipeline(
    center_freq_hz: int = 413_075_000,
    sample_rate: int = 6_000_000,
    offset_hz: float = 0.0,
    duration_sec: float = 2.0,
) -> None:
    """Debug IQ pipeline with detailed logging."""

    logger.info("Opening SDR...")
    sdr = SoapySDR.Device("driver=sdrplay")
    sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, sample_rate)
    sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, center_freq_hz)
    sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, 40)
    sdr.setAntenna(SoapySDR.SOAPY_SDR_RX, 0, "Antenna B")

    stream = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
    sdr.activateStream(stream)

    # Collect raw samples
    logger.info(f"\nCollecting {duration_sec}s of data at {sample_rate/1e6:.1f} MHz...")
    total_samples = int(sample_rate * duration_sec)
    chunk_size = sample_rate // 4
    raw_iq = []

    samples_read = 0
    while samples_read < total_samples:
        buffer = np.zeros(chunk_size, dtype=np.complex64)
        sr = sdr.readStream(stream, [buffer], chunk_size)
        if sr.ret > 0:
            raw_iq.append(buffer[:sr.ret].copy())
            samples_read += sr.ret

    sdr.deactivateStream(stream)
    sdr.closeStream(stream)

    raw_iq = np.concatenate(raw_iq)
    analyze_iq(raw_iq, f"STAGE 0: Raw SDR ({sample_rate/1e6:.1f} MHz)")

    # Stage 1: Frequency shift
    logger.info(f"\nApplying frequency shift: {offset_hz} Hz")
    sample_idx = np.arange(len(raw_iq))
    shift = np.exp(-1j * 2 * np.pi * offset_hz * sample_idx / sample_rate)
    shifted_iq = (raw_iq * shift).astype(np.complex128)
    analyze_iq(shifted_iq, f"STAGE 1: After freq shift ({offset_hz} Hz)")

    # Stage 2: First decimation (6 MHz -> 200 kHz, factor 30)
    stage1_factor = 30
    stage1_rate = sample_rate // stage1_factor
    stage1_cutoff = 0.8 / stage1_factor
    stage1_taps = scipy_signal.firwin(157, stage1_cutoff, window=("kaiser", 7.857))

    logger.info(f"\nStage 1 filter: {len(stage1_taps)} taps, cutoff={stage1_cutoff:.4f}")
    logger.info(f"  Filter taps[0:10]: {stage1_taps[:10]}")
    logger.info(f"  Filter taps sum: {np.sum(stage1_taps):.6f}")

    stage1_zi_template = scipy_signal.lfilter_zi(stage1_taps, 1.0).astype(np.complex128)
    logger.info(f"  Initial zi template shape: {stage1_zi_template.shape}")
    logger.info(f"  Initial zi template[0:5]: {stage1_zi_template[:5]}")

    # FIXED: Initialize zi with first sample to prevent transient
    stage1_zi = stage1_zi_template * shifted_iq[0]
    logger.info(f"  Scaled zi[0:5]: {stage1_zi[:5]}")

    filtered1, _ = scipy_signal.lfilter(stage1_taps, 1.0, shifted_iq, zi=stage1_zi)
    analyze_iq(filtered1, f"STAGE 2a: After filter 1 (before decim)")

    decimated1 = filtered1[::stage1_factor]
    analyze_iq(decimated1, f"STAGE 2b: After decim 1 ({stage1_rate/1e3:.0f} kHz)")

    # Stage 3: Second decimation (200 kHz -> 50 kHz, factor 4)
    stage2_factor = 4
    stage2_rate = stage1_rate // stage2_factor
    stage2_cutoff = 0.8 / stage2_factor
    stage2_taps = scipy_signal.firwin(73, stage2_cutoff, window=("kaiser", 7.857))

    logger.info(f"\nStage 2 filter: {len(stage2_taps)} taps, cutoff={stage2_cutoff:.4f}")

    stage2_zi_template = scipy_signal.lfilter_zi(stage2_taps, 1.0).astype(np.complex128)

    # FIXED: Initialize zi with first sample to prevent transient
    stage2_zi = stage2_zi_template * decimated1[0]

    filtered2, _ = scipy_signal.lfilter(stage2_taps, 1.0, decimated1, zi=stage2_zi)
    analyze_iq(filtered2, f"STAGE 3a: After filter 2 (before decim)")

    decimated2 = filtered2[::stage2_factor]
    analyze_iq(decimated2, f"STAGE 3b: After decim 2 ({stage2_rate/1e3:.0f} kHz) - FINAL")

    # Compare with just using scipy.signal.decimate
    logger.info("\n=== Alternative: scipy.signal.decimate ===")
    alt_decim = scipy_signal.decimate(raw_iq.astype(np.complex128), 30, ftype='fir', zero_phase=False)
    alt_decim = scipy_signal.decimate(alt_decim, 4, ftype='fir', zero_phase=False)
    analyze_iq(alt_decim, "scipy.decimate (30x then 4x)")


if __name__ == "__main__":
    debug_pipeline()
