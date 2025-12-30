#!/usr/bin/env python3
"""Record baseband IQ from live SDR for comparison with SDRTrunk.

This script captures wideband IQ from the SDR and extracts a single control
channel using the same processing as TrunkingSystem, saving the baseband IQ
in the same WAV format as SDRTrunk's baseband recordings.

Usage:
    python scripts/record_baseband_iq.py --freq 413075000 --duration 30

Output:
    baseband_<freq>_<timestamp>.wav - 50 kHz stereo int16 WAV (I=left, Q=right)

This recording can then be:
1. Fed through WaveCap's decoder to verify it works
2. Compared with SDRTrunk's baseband recording from the same frequency
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import SoapySDR
except ImportError:
    print("Error: SoapySDR Python bindings not installed")
    sys.exit(1)

from scipy import signal as scipy_signal

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def record_baseband(
    center_freq_hz: int,
    target_freq_hz: int,
    duration_sec: float,
    output_path: Path,
    sample_rate: int = 6_000_000,
    gain_db: float = 40.0,
    antenna: str = "Antenna B",
) -> None:
    """Record baseband IQ from SDR.

    Uses the same two-stage decimation as TrunkingSystem:
    - Stage 1: 6 MHz → 200 kHz (30:1)
    - Stage 2: 200 kHz → 50 kHz (4:1)
    """

    # Calculate frequency offset
    offset_hz = target_freq_hz - center_freq_hz
    logger.info(f"Center: {center_freq_hz/1e6:.4f} MHz")
    logger.info(f"Target: {target_freq_hz/1e6:.4f} MHz")
    logger.info(f"Offset: {offset_hz/1e3:.1f} kHz")

    # Open SDR
    logger.info("Opening SDR...")
    sdr = SoapySDR.Device("driver=sdrplay")
    sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, sample_rate)
    sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, center_freq_hz)
    sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, gain_db)
    sdr.setAntenna(SoapySDR.SOAPY_SDR_RX, 0, antenna)

    stream = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
    sdr.activateStream(stream)

    # ================================================================
    # TWO-STAGE DECIMATION (matching TrunkingSystem exactly)
    # ================================================================
    stage1_factor = 30
    stage2_factor = 4
    stage1_rate = sample_rate // stage1_factor  # 200 kHz
    stage2_rate = stage1_rate // stage2_factor   # 50 kHz
    total_decim = stage1_factor * stage2_factor  # 120

    logger.info(f"Decimation: {sample_rate/1e6:.1f} MHz → {stage1_rate/1e3:.0f} kHz → {stage2_rate/1e3:.0f} kHz")

    # Stage 1 filter (Kaiser beta=7.857 for 80 dB stopband, 157 taps)
    stage1_cutoff = 0.8 / stage1_factor
    stage1_taps = scipy_signal.firwin(157, stage1_cutoff, window=("kaiser", 7.857))
    stage1_zi_template = scipy_signal.lfilter_zi(stage1_taps, 1.0).astype(np.complex128)
    stage1_zi = None  # Will be initialized with first sample

    # Stage 2 filter (Kaiser beta=7.857 for 80 dB stopband, 73 taps)
    stage2_cutoff = 0.8 / stage2_factor
    stage2_taps = scipy_signal.firwin(73, stage2_cutoff, window=("kaiser", 7.857))
    stage2_zi_template = scipy_signal.lfilter_zi(stage2_taps, 1.0).astype(np.complex128)
    stage2_zi = None  # Will be initialized with first sample

    # Phase-continuous frequency shift state
    sample_idx = 0
    first_chunk = True

    # Output buffer
    baseband_samples = []

    # Collect samples
    logger.info(f"Recording {duration_sec:.1f} seconds...")
    start_time = time.time()
    chunk_size = sample_rate // 4  # 250ms chunks
    buffer = np.zeros(chunk_size, dtype=np.complex64)

    samples_read = 0
    last_progress = 0

    while time.time() - start_time < duration_sec:
        sr = sdr.readStream(stream, [buffer], chunk_size)

        if sr.ret <= 0:
            continue

        iq = buffer[:sr.ret]
        samples_read += len(iq)

        # Phase-continuous frequency shift
        n = len(iq)
        indices = np.arange(sample_idx, sample_idx + n)
        shift = np.exp(-1j * 2 * np.pi * offset_hz * indices / sample_rate)
        centered_iq = (iq * shift).astype(np.complex128)
        sample_idx += n

        # Initialize filter states with first sample (prevents huge transient)
        if first_chunk:
            stage1_zi = stage1_zi_template * centered_iq[0]
            first_chunk = False

        # Stage 1 decimation
        filtered1, stage1_zi = scipy_signal.lfilter(
            stage1_taps, 1.0, centered_iq, zi=stage1_zi
        )
        decimated1 = filtered1[::stage1_factor]

        # Initialize stage 2 filter with first sample of decimated output
        if stage2_zi is None:
            stage2_zi = stage2_zi_template * decimated1[0]

        # Stage 2 decimation
        filtered2, stage2_zi = scipy_signal.lfilter(
            stage2_taps, 1.0, decimated1, zi=stage2_zi
        )
        decimated2 = filtered2[::stage2_factor]

        baseband_samples.append(decimated2.astype(np.complex64))

        # Progress
        elapsed = time.time() - start_time
        if elapsed - last_progress >= 5.0:
            logger.info(f"  {elapsed:.0f}s / {duration_sec:.0f}s - {samples_read/1e6:.1f}M samples read")
            last_progress = elapsed

    # Stop SDR
    sdr.deactivateStream(stream)
    sdr.closeStream(stream)

    # Combine all samples
    baseband_iq = np.concatenate(baseband_samples)
    logger.info(f"Recorded {len(baseband_iq)} baseband samples ({len(baseband_iq)/stage2_rate:.2f} sec)")

    # Save as WAV (matching SDRTrunk format: stereo int16, I=left, Q=right)
    save_baseband_wav(baseband_iq, stage2_rate, output_path)
    logger.info(f"Saved to: {output_path}")

    # Verify by running through decoder
    verify_recording(output_path)


def save_baseband_wav(iq: np.ndarray, sample_rate: int, path: Path) -> None:
    """Save baseband IQ as stereo WAV (SDRTrunk format)."""

    # Scale to int16 range
    # Find max amplitude and normalize
    max_amp = max(np.max(np.abs(iq.real)), np.max(np.abs(iq.imag)))
    if max_amp > 0:
        scale = 32000.0 / max_amp  # Leave some headroom
    else:
        scale = 1.0

    i_samples = (iq.real * scale).astype(np.int16)
    q_samples = (iq.imag * scale).astype(np.int16)

    # Interleave I and Q
    interleaved = np.empty(len(iq) * 2, dtype=np.int16)
    interleaved[0::2] = i_samples
    interleaved[1::2] = q_samples

    # Write WAV
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(interleaved.tobytes())

    logger.info(f"WAV: {len(iq)} frames, {sample_rate} Hz, stereo int16")


def verify_recording(wav_path: Path) -> None:
    """Verify the recording by running through WaveCap decoder."""

    from wavecapsdr.trunking.control_channel import create_control_monitor, P25Modulation
    from wavecapsdr.trunking.config import TrunkingProtocol

    # Load WAV
    with wave.open(str(wav_path), 'rb') as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    # Parse IQ
    data = np.frombuffer(raw, dtype=np.int16)
    i = data[0::2].astype(np.float32) / 32768.0
    q = data[1::2].astype(np.float32) / 32768.0
    iq = (i + 1j * q).astype(np.complex64)

    logger.info(f"\nVerifying recording: {len(iq)} samples ({len(iq)/sample_rate:.2f} sec)")

    # Create monitor
    monitor = create_control_monitor(
        protocol=TrunkingProtocol.P25_PHASE1,
        sample_rate=sample_rate,
        modulation=P25Modulation.C4FM,
    )

    # Process
    results = monitor.process_iq(iq)

    # Stats
    stats = monitor.get_stats()
    logger.info(f"=== Verification Results ===")
    logger.info(f"Sync state: {stats['sync_state']}")
    logger.info(f"Frames decoded: {stats['frames_decoded']}")
    logger.info(f"TSBK attempts: {stats['tsbk_attempts']}")
    logger.info(f"TSBK CRC pass: {stats['tsbk_crc_pass']}")
    logger.info(f"TSBK CRC pass rate: {stats['tsbk_crc_pass_rate']:.1f}%")

    if stats['tsbk_crc_pass_rate'] >= 50:
        logger.info("✓ Recording verified: CRC pass rate >= 50%")
    else:
        logger.warning(f"✗ Recording verification failed: CRC pass rate {stats['tsbk_crc_pass_rate']:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Record baseband IQ for comparison with SDRTrunk")
    parser.add_argument("--freq", type=int, required=True, help="Control channel frequency in Hz")
    parser.add_argument("--center", type=int, default=None, help="SDR center frequency (default: same as --freq)")
    parser.add_argument("--duration", type=float, default=30.0, help="Recording duration in seconds")
    parser.add_argument("--output", type=str, default=None, help="Output WAV path")
    parser.add_argument("--sample-rate", type=int, default=6_000_000, help="SDR sample rate")
    parser.add_argument("--gain", type=float, default=40.0, help="SDR gain in dB")
    parser.add_argument("--antenna", type=str, default="Antenna B", help="SDR antenna")

    args = parser.parse_args()

    # Default center to target frequency
    center_freq = args.center if args.center else args.freq

    # Default output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"baseband_{args.freq}_{timestamp}.wav")

    record_baseband(
        center_freq_hz=center_freq,
        target_freq_hz=args.freq,
        duration_sec=args.duration,
        output_path=output_path,
        sample_rate=args.sample_rate,
        gain_db=args.gain,
        antenna=args.antenna,
    )


if __name__ == "__main__":
    main()
