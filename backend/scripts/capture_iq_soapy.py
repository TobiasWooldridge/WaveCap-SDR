#!/usr/bin/env python3
"""Capture raw IQ data directly from an SDR device using SoapySDR.

This creates an IQ recording that can be played back through both WaveCap-SDR
and SDRTrunk for comparison.

Usage:
    # List available devices
    python scripts/capture_iq_soapy.py --list-devices

    # Capture 10 seconds at 6 MHz sample rate, centered on 415.5 MHz (SA-GRN control)
    python scripts/capture_iq_soapy.py \\
        --device "driver=sdrplay" \\
        --frequency 415500000 \\
        --sample-rate 6000000 \\
        --gain 40 \\
        --duration 10 \\
        --output sagrn_415.5mhz_6msps.wav

Output is WAV stereo (I=left, Q=right) int16 format that both WaveCap and SDRTrunk can read.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import wave
from pathlib import Path

import numpy as np

try:
    import SoapySDR
except ImportError:
    print("Error: SoapySDR Python bindings not installed")
    print("Install with: pip install soapysdr")
    sys.exit(1)

logger = logging.getLogger(__name__)


def list_devices() -> None:
    """List all available SoapySDR devices."""
    print("Available SoapySDR devices:")
    print()

    devices = SoapySDR.Device.enumerate()
    if not devices:
        print("  No devices found!")
        return

    for i, dev in enumerate(devices):
        print(f"  [{i}] {dict(dev)}")


def capture_iq(
    device_args: str,
    frequency_hz: int,
    sample_rate: int,
    gain_db: float,
    duration_seconds: float,
    output_path: Path,
    antenna: str | None,
    bandwidth_hz: int | None,
) -> None:
    """Capture IQ data from SDR device."""

    logger.info(f"Opening device: {device_args}")
    sdr = SoapySDR.Device(device_args)

    # List available settings
    logger.info(f"Device: {sdr.getHardwareKey()}")
    logger.info(f"Available antennas: {sdr.listAntennas(SoapySDR.SOAPY_SDR_RX, 0)}")
    logger.info(f"Gain range: {sdr.getGainRange(SoapySDR.SOAPY_SDR_RX, 0)}")

    # Configure receiver
    if antenna:
        logger.info(f"Setting antenna: {antenna}")
        sdr.setAntenna(SoapySDR.SOAPY_SDR_RX, 0, antenna)

    logger.info(f"Setting frequency: {frequency_hz / 1e6:.3f} MHz")
    sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, frequency_hz)

    logger.info(f"Setting sample rate: {sample_rate / 1e6:.3f} MSPS")
    sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, sample_rate)

    actual_rate = sdr.getSampleRate(SoapySDR.SOAPY_SDR_RX, 0)
    logger.info(f"Actual sample rate: {actual_rate / 1e6:.3f} MSPS")

    if bandwidth_hz:
        logger.info(f"Setting bandwidth: {bandwidth_hz / 1e6:.3f} MHz")
        sdr.setBandwidth(SoapySDR.SOAPY_SDR_RX, 0, bandwidth_hz)

    logger.info(f"Setting gain: {gain_db} dB")
    sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, gain_db)

    # Setup stream
    rx_stream = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
    sdr.activateStream(rx_stream)

    # Calculate buffer sizes
    chunk_size = 65536  # Samples per read
    total_samples = int(duration_seconds * actual_rate)
    collected = 0

    logger.info(f"Capturing {total_samples} samples ({duration_seconds} seconds)...")

    # Pre-allocate output buffer
    all_samples = np.zeros(total_samples, dtype=np.complex64)
    buff = np.zeros(chunk_size, dtype=np.complex64)

    start_time = time.time()

    try:
        while collected < total_samples:
            remaining = total_samples - collected
            to_read = min(chunk_size, remaining)

            sr = sdr.readStream(rx_stream, [buff], to_read, timeoutUs=1000000)

            if sr.ret > 0:
                all_samples[collected : collected + sr.ret] = buff[: sr.ret]
                collected += sr.ret

                # Progress update
                elapsed = time.time() - start_time
                if collected % (chunk_size * 10) < chunk_size:
                    pct = 100.0 * collected / total_samples
                    logger.info(f"  {pct:.1f}% ({collected}/{total_samples} samples, {elapsed:.1f}s)")

            elif sr.ret < 0:
                logger.warning(f"Stream error: {sr.ret}")
                break

    finally:
        sdr.deactivateStream(rx_stream)
        sdr.closeStream(rx_stream)

    elapsed = time.time() - start_time
    logger.info(f"Capture complete: {collected} samples in {elapsed:.1f}s")

    # Trim to actual collected samples
    all_samples = all_samples[:collected]

    # Convert to int16 stereo WAV (I=left, Q=right)
    # Scale complex64 [-1, 1] to int16 [-32768, 32767]
    i_samples = np.clip(all_samples.real * 32767, -32768, 32767).astype(np.int16)
    q_samples = np.clip(all_samples.imag * 32767, -32768, 32767).astype(np.int16)

    # Interleave as stereo
    stereo = np.zeros(len(all_samples) * 2, dtype=np.int16)
    stereo[0::2] = i_samples  # Left = I
    stereo[1::2] = q_samples  # Right = Q

    # Write WAV file
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(2)  # Stereo
        wf.setsampwidth(2)  # int16
        wf.setframerate(int(actual_rate))
        wf.writeframes(stereo.tobytes())

    logger.info(f"Saved to {output_path}")
    logger.info(f"  Format: Stereo WAV, int16, {int(actual_rate)} Hz")
    logger.info(f"  Size: {output_path.stat().st_size / 1e6:.1f} MB")

    # Also save rawiq format for WaveCap compatibility
    rawiq_path = output_path.with_suffix(".rawiq")
    rawiq_path.write_bytes(stereo.tobytes())
    logger.info(f"Also saved to {rawiq_path} (raw IQ format)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture IQ data from SDR device")
    parser.add_argument("--list-devices", action="store_true", help="List available devices")
    parser.add_argument("--device", type=str, default="driver=sdrplay", help="SoapySDR device args")
    parser.add_argument("--frequency", type=float, required=False, help="Center frequency in Hz")
    parser.add_argument("--sample-rate", type=int, default=6000000, help="Sample rate in Hz")
    parser.add_argument("--bandwidth", type=int, default=None, help="IF bandwidth in Hz")
    parser.add_argument("--gain", type=float, default=40.0, help="RF gain in dB")
    parser.add_argument("--antenna", type=str, default=None, help="Antenna port name")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration in seconds")
    parser.add_argument("--output", type=Path, default=None, help="Output file path (.wav)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.list_devices:
        list_devices()
        return 0

    if not args.frequency:
        parser.error("--frequency is required for capture")

    if not args.output:
        # Generate default filename
        freq_mhz = args.frequency / 1e6
        rate_msps = args.sample_rate / 1e6
        args.output = Path(f"capture_{freq_mhz:.3f}mhz_{rate_msps:.1f}msps.wav")

    capture_iq(
        device_args=args.device,
        frequency_hz=int(args.frequency),
        sample_rate=args.sample_rate,
        gain_db=args.gain,
        duration_seconds=args.duration,
        output_path=args.output,
        antenna=args.antenna,
        bandwidth_hz=args.bandwidth,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
