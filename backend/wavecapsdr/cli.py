#!/usr/bin/env python3
"""WaveCap-SDR Command Line Interface.

Provides standalone CLI utilities for SDR operations without running the full server.

Usage:
    python -m wavecapsdr.cli list-devices
    python -m wavecapsdr.cli capture-iq --device 240309F070 --antenna B --duration 60
    python -m wavecapsdr.cli decode-iq --file capture.wav --modulation c4fm
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def get_rspdx_max_lna(frequency_hz: float) -> int:
    """Get maximum LNA setting for RSPdx based on frequency.

    From SDRTrunk's API section 5 Gain Reduction Table values.
    Returns the maximum valid LNA index for the given frequency.
    """
    if frequency_hz < 12_000_000:
        return 21
    elif frequency_hz < 50_000_000:
        return 19
    elif frequency_hz < 60_000_000:
        return 24
    elif frequency_hz < 250_000_000:
        return 26
    elif frequency_hz < 420_000_000:
        return 27
    elif frequency_hz < 1_000_000_000:
        return 20
    else:
        return 18


def cmd_list_devices(args: argparse.Namespace) -> int:
    """List available SDR devices."""
    try:
        import SoapySDR
    except ImportError:
        print("Error: SoapySDR not available")
        return 1

    print("Discovering SDR devices...\n")

    # Check SDRplay devices
    rsp_devices = SoapySDR.Device.enumerate("driver=sdrplay")
    if rsp_devices:
        print("SDRplay RSP Devices:")
        for dev in rsp_devices:
            d = dict(dev)
            print(f"  Serial: {d.get('serial', 'unknown')}")
            print(f"    Label: {d.get('label', 'unknown')}")
            if args.verbose:
                # Open device to get more info
                try:
                    sdr = SoapySDR.Device(dev)
                    antennas = sdr.listAntennas(SoapySDR.SOAPY_SDR_RX, 0)
                    print(f"    Antennas: {', '.join(antennas)}")

                    print("    Gain Elements:")
                    for elem in sdr.listGains(SoapySDR.SOAPY_SDR_RX, 0):
                        rng = sdr.getGainRange(SoapySDR.SOAPY_SDR_RX, 0, elem)
                        print(f"      {elem}: {rng.minimum():.0f} to {rng.maximum():.0f} dB")

                    rng = sdr.getGainRange(SoapySDR.SOAPY_SDR_RX, 0)
                    print(f"    Overall Gain: {rng.minimum():.0f} to {rng.maximum():.0f} dB")
                    sdr = None
                except Exception as e:
                    print(f"    (Could not get details: {e})")
            print()
    else:
        print("No SDRplay devices found.\n")

    # Check RTL-SDR devices
    rtl_devices = SoapySDR.Device.enumerate("driver=rtlsdr")
    if rtl_devices:
        print("RTL-SDR Devices:")
        for dev in rtl_devices:
            d = dict(dev)
            print(f"  Serial: {d.get('serial', 'unknown')}")
            print(f"    Label: {d.get('label', 'unknown')}")
            print()
    else:
        print("No RTL-SDR devices found.\n")

    return 0


def cmd_capture_iq(args: argparse.Namespace) -> int:
    """Capture raw IQ samples to a WAV file."""
    try:
        import SoapySDR
    except ImportError:
        print("Error: SoapySDR not available")
        return 1

    # Wideband mode settings
    WIDEBAND_SAMPLE_RATE = 8_000_000  # 8 MHz like SDRTrunk
    CHANNEL_BANDWIDTH = 25_000  # 25 kHz channels

    # Determine actual capture sample rate
    if args.wideband:
        capture_sample_rate = WIDEBAND_SAMPLE_RATE
        output_sample_rate = 50_000  # After channelizer: 2x oversampled 25kHz channel
        print(f"Wideband capture mode:")
        print(f"  Capture rate: {capture_sample_rate/1e6:.1f} MHz")
        print(f"  Output rate: {output_sample_rate} Hz (after channelizer)")
        if args.channel_freq:
            print(f"  Extract channel: {args.channel_freq/1e6:.6f} MHz")
        else:
            print(f"  Extract channel: {args.frequency/1e6:.6f} MHz (center)")
    else:
        capture_sample_rate = args.sample_rate
        output_sample_rate = args.sample_rate

    print(f"Capture IQ to: {args.output}")
    print(f"  Frequency: {args.frequency / 1e6:.4f} MHz")
    print(f"  Sample rate: {capture_sample_rate}")
    print(f"  Duration: {args.duration} seconds")

    # Find device
    if args.device:
        results = SoapySDR.Device.enumerate(f"driver=sdrplay,serial={args.device}")
        if not results:
            # Try RTL-SDR
            results = SoapySDR.Device.enumerate(f"driver=rtlsdr,serial={args.device}")
        if not results:
            print(f"Error: Device not found: {args.device}")
            return 1
    else:
        # Use first available RSP
        results = SoapySDR.Device.enumerate("driver=sdrplay")
        if not results:
            results = SoapySDR.Device.enumerate("driver=rtlsdr")
        if not results:
            print("Error: No SDR devices found")
            return 1

    dev_info = dict(results[0])
    print(f"  Device: {dev_info.get('label', dev_info.get('serial', 'unknown'))}")

    # Open device
    sdr = SoapySDR.Device(results[0])

    # Configure antenna for RSP devices
    if args.antenna:
        antenna_map = {"A": "Antenna A", "B": "Antenna B", "C": "Antenna C"}
        antenna_name = antenna_map.get(args.antenna.upper(), args.antenna)
        sdr.setAntenna(SoapySDR.SOAPY_SDR_RX, 0, antenna_name)
        print(f"  Antenna: {antenna_name}")

    # Configure frequency and sample rate
    sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, capture_sample_rate)
    sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, args.frequency)

    # SDRplay-specific settings (Zero-IF is default for sample rates ≥2 MHz)
    is_sdrplay = "sdrplay" in dev_info.get("driver", "").lower()
    if is_sdrplay:
        # Enable IQ correction for DC offset and IQ imbalance
        try:
            sdr.writeSetting("iqcorr_ctrl", "true")
        except Exception:
            pass  # Setting may not be available

        # Enable RF notch for FM broadcast interference (if applicable)
        # Only enable if frequency is below 1 GHz (where FM interference matters)
        if args.frequency < 1_000_000_000:
            try:
                sdr.writeSetting("rfnotch_ctrl", "true")
            except Exception:
                pass

    # Configure gain
    # For RSP: IFGR = GR (20-59), RFGR = 27-LNA (0-27)
    # Detect if this is an RSPdx device for frequency-dependent LNA limits
    is_rspdx = "RSPdx" in dev_info.get("label", "")

    if args.lna is not None or args.gr is not None:
        # Use SDRTrunk-style gain parameters
        lna = args.lna if args.lna is not None else 0
        gr = args.gr if args.gr is not None else 40

        # Apply frequency-dependent LNA limit for RSPdx
        if is_rspdx:
            max_lna = get_rspdx_max_lna(args.frequency)
            if lna > max_lna:
                print(f"  Warning: LNA={lna} exceeds max for {args.frequency/1e6:.1f} MHz, clamping to {max_lna}")
                lna = max_lna

        # Map to SoapySDR element gains
        rfgr = 27 - lna  # Invert LNA to RFGR
        ifgr = gr

        # Clamp to valid ranges
        rfgr = max(0, min(27, rfgr))
        ifgr = max(20, min(59, ifgr))

        sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, False)  # Manual gain
        sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, "RFGR", rfgr)
        sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, "IFGR", ifgr)
        print(f"  Gain: LNA={lna} (RFGR={rfgr}), GR={gr} (IFGR={ifgr})")
    elif args.gain is not None:
        sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, False)
        sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, args.gain)
        print(f"  Gain: {args.gain} dB (overall)")
    else:
        sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, True)  # AGC
        print("  Gain: AGC (automatic)")

    # Start stream
    stream = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
    sdr.activateStream(stream)

    # Calculate samples needed
    total_samples = int(capture_sample_rate * args.duration)
    buffer_size = 65536
    samples_captured = 0
    all_samples = []

    print(f"\nCapturing {total_samples} samples...")
    start_time = time.time()
    last_update = 0
    error_count = 0
    max_errors = 10

    try:
        while samples_captured < total_samples:
            buff = np.zeros(buffer_size, dtype=np.complex64)
            # Use 100ms timeout
            sr = sdr.readStream(stream, [buff], buffer_size, timeoutUs=100000)
            if sr.ret > 0:
                all_samples.append(buff[:sr.ret].copy())
                samples_captured += sr.ret
                error_count = 0  # Reset on success

                # Progress update every second
                elapsed = time.time() - start_time
                if elapsed - last_update >= 1.0:
                    pct = 100 * samples_captured / total_samples
                    print(f"  {pct:.0f}% ({samples_captured}/{total_samples} samples, {elapsed:.1f}s)")
                    last_update = elapsed
            elif sr.ret == SoapySDR.SOAPY_SDR_TIMEOUT:
                # Timeout is normal, just retry
                error_count += 1
                if error_count > max_errors:
                    print(f"Too many timeouts, stopping capture")
                    break
            elif sr.ret == SoapySDR.SOAPY_SDR_OVERFLOW:
                # Overflow - data was lost, continue
                print("  (overflow - samples dropped)")
                error_count += 1
            elif sr.ret < 0:
                print(f"Stream error: {sr.ret}")
                error_count += 1
                if error_count > max_errors:
                    break
    except KeyboardInterrupt:
        print("\nCapture interrupted.")
    finally:
        sdr.deactivateStream(stream)
        sdr.closeStream(stream)
        sdr = None

    # Combine all samples
    if not all_samples:
        print("No samples captured!")
        return 1

    iq_data = np.concatenate(all_samples)
    print(f"\nCaptured {len(iq_data)} samples in {time.time() - start_time:.1f}s")
    print(f"  IQ magnitude: mean={np.mean(np.abs(iq_data)):.4f}, max={np.max(np.abs(iq_data)):.4f}")

    # Wideband mode: run polyphase channelizer to extract target channel
    if args.wideband:
        from wavecapsdr.dsp.channelizer import PolyphaseChannelizer, ChannelCalculator

        target_freq = args.channel_freq if args.channel_freq else args.frequency
        print(f"\n=== Polyphase Channelizer ===")
        print(f"  Input: {len(iq_data)} samples at {capture_sample_rate/1e6:.1f} MHz")

        # Create channelizer
        channelizer = PolyphaseChannelizer(capture_sample_rate, CHANNEL_BANDWIDTH)
        print(f"  Channels: {channelizer.channel_count} x {CHANNEL_BANDWIDTH/1000:.0f} kHz")
        print(f"  Output sample rate: {channelizer.channel_sample_rate:.0f} Hz")

        # Calculate target channel index
        calculator = ChannelCalculator(args.frequency, capture_sample_rate, CHANNEL_BANDWIDTH)
        channel_idx = calculator.get_channel_index(target_freq)
        channel_center = calculator.get_channel_center_frequency(channel_idx)
        print(f"  Target: {target_freq/1e6:.6f} MHz → channel {channel_idx}")
        print(f"  Channel center: {channel_center/1e6:.6f} MHz")

        # Process through channelizer
        print("  Processing...")
        channel_results = channelizer.process(iq_data)
        print(f"  Output: {len(channel_results)} time samples")

        # Extract target channel
        iq_data = channelizer.extract_channel(channel_results, channel_idx)
        output_sample_rate = int(channelizer.channel_sample_rate)

        print(f"  Extracted {len(iq_data)} samples for channel {channel_idx}")
        print(f"  Output magnitude: mean={np.mean(np.abs(iq_data)):.4f}, max={np.max(np.abs(iq_data)):.4f}")

    # Save as WAV (stereo: I=left, Q=right, int16)
    output_path = Path(args.output)

    # Use fixed scale to preserve signal levels (matches SDRTrunk's channelizer output)
    # SDRTrunk's polyphase channelizer has ~10x gain from decimation
    # SoapySDR CF32 samples are typically ±1.0 range, but actual signal << 1.0
    # Use scale = 32767 so samples in range ±0.1 become ±3276 in int16
    scale = 32767.0

    i_samples = np.clip(iq_data.real * scale, -32768, 32767).astype(np.int16)
    q_samples = np.clip(iq_data.imag * scale, -32768, 32767).astype(np.int16)

    # Interleave I and Q
    stereo = np.empty(len(i_samples) * 2, dtype=np.int16)
    stereo[0::2] = i_samples
    stereo[1::2] = q_samples

    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(output_sample_rate)
        wf.writeframes(stereo.tobytes())

    print(f"Saved to {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")

    return 0


def estimate_freq_offset(iq: np.ndarray, sample_rate: int) -> float:
    """Estimate frequency offset by finding signal peak in spectrum."""
    from scipy import signal as sig

    # Use a few seconds of data
    chunk = iq[:min(len(iq), sample_rate * 3)]

    # Compute power spectrum using Welch method
    f, psd = sig.welch(chunk, sample_rate, nperseg=4096, return_onesided=False)
    f = np.fft.fftshift(f)
    psd = np.fft.fftshift(psd)

    # Find peak within P25 signal region (±15 kHz)
    p25_mask = np.abs(f) < 15000
    if not np.any(psd[p25_mask] > 0):
        return 0.0

    peak_idx = np.argmax(psd[p25_mask])
    offset = f[p25_mask][peak_idx]

    return float(offset)


def apply_freq_correction(iq: np.ndarray, offset_hz: float, sample_rate: int) -> np.ndarray:
    """Apply frequency correction by mixing with complex exponential."""
    t = np.arange(len(iq)) / sample_rate
    correction = np.exp(-1j * 2 * np.pi * offset_hz * t)
    return iq * correction


def cmd_decode_iq(args: argparse.Namespace) -> int:
    """Decode IQ file through P25 pipeline."""
    import wave as wave_mod

    input_path = Path(args.file)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return 1

    print(f"Decoding IQ file: {input_path}")

    # Load WAV file
    with wave_mod.open(str(input_path), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        n_frames = wf.getnframes()

        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Channels: {channels}")
        print(f"  Duration: {n_frames / sample_rate:.1f} seconds")

        raw = wf.readframes(n_frames)

    # Parse to complex IQ
    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        print(f"Error: Unsupported sample width: {width}")
        return 1

    if channels == 2:
        iq = data[::2] + 1j * data[1::2]
    elif channels == 1:
        # Assume real samples, create analytic signal
        from scipy.signal import hilbert
        iq = hilbert(data).astype(np.complex64)
    else:
        print(f"Error: Unsupported channel count: {channels}")
        return 1

    print(f"  Loaded {len(iq)} complex samples")
    print(f"  IQ magnitude: mean={np.mean(np.abs(iq)):.4f}, max={np.max(np.abs(iq)):.4f}")

    # Frequency offset detection and correction
    if args.freq_offset is not None:
        offset_hz = args.freq_offset
        print(f"\n=== Manual Frequency Offset: {offset_hz:.1f} Hz ===")
    else:
        offset_hz = estimate_freq_offset(iq, sample_rate)
        print(f"\n=== Auto-detected Frequency Offset: {offset_hz:.1f} Hz ===")

    if abs(offset_hz) > 100:
        print(f"  Applying correction of {-offset_hz:.1f} Hz")
        iq = apply_freq_correction(iq, offset_hz, sample_rate)
        # Verify correction
        new_offset = estimate_freq_offset(iq, sample_rate)
        print(f"  Residual offset after correction: {new_offset:.1f} Hz")

    # Choose demodulator
    modulation = args.modulation.lower()

    if modulation == "c4fm":
        print("\n=== C4FM Demodulation ===")
        dibits, symbols = demod_c4fm(iq, sample_rate)
    elif modulation in ("lsm", "cqpsk"):
        print("\n=== CQPSK/LSM Demodulation ===")
        dibits, symbols = demod_cqpsk(iq, sample_rate)
    else:
        print(f"Error: Unknown modulation: {modulation}")
        return 1

    # Use proper P25 framer for message extraction
    print("\n=== P25 Message Framing ===")
    from wavecapsdr.decoders.p25_framer import P25P1MessageFramer

    messages = []
    def on_message(msg):
        messages.append({
            'nac': msg.nac,
            'duid': str(msg.duid.name if hasattr(msg.duid, 'name') else msg.duid),
            'bits': len(msg._bits) if hasattr(msg, '_bits') else 0,
        })

    framer = P25P1MessageFramer()
    framer.set_listener(on_message)
    framer.start()

    # Use batch processing for speed
    soft_symbols_array = np.array(symbols, dtype=np.float32)
    dibits_array = np.array(dibits, dtype=np.uint8)

    nid_count = framer.process_batch(soft_symbols_array, dibits_array)
    print(f"  NIDs detected: {nid_count}")

    # Print summary
    print(f"  Messages decoded: {len(messages)}")

    if messages:
        print("\n=== Decoded Messages ===")
        # Group by DUID
        duid_counts = {}
        for msg in messages:
            duid = msg.get('duid', 'UNKNOWN')
            duid_counts[duid] = duid_counts.get(duid, 0) + 1

        for duid, count in sorted(duid_counts.items()):
            print(f"  {duid}: {count} messages")

        # Show sample messages
        print("\n  Sample messages:")
        for msg in messages[:10]:
            nac = msg.get('nac', 0)
            duid = msg.get('duid', 'UNKNOWN')
            print(f"    NAC=0x{nac:03X} DUID={duid}")

    # Also run simple sync search for comparison
    syncs = find_p25_syncs(dibits)
    print(f"\n  Simple sync search found {len(syncs)} patterns (for comparison)")

    return 0


def demod_c4fm(iq: np.ndarray, sample_rate: int, symbol_rate: int = 4800) -> tuple:
    """C4FM demodulation using proper SDRTrunk-style demodulator."""
    from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator

    # Use the proper C4FM demodulator with equalizer, PLL, and timing recovery
    demod = C4FMDemodulator(sample_rate=sample_rate, symbol_rate=symbol_rate)
    dibits, soft_symbols = demod.demodulate(iq)

    print(f"  Extracted {len(dibits)} symbols")
    print(f"  Symbol stats: mean={np.mean(soft_symbols):.2f}, std={np.std(soft_symbols):.2f}")
    print(f"  Sync count: {demod._sync_count}")
    print(f"  Equalizer: PLL={demod._equalizer.pll:.4f}, gain={demod._equalizer.gain:.3f}")

    unique, counts = np.unique(dibits, return_counts=True)
    print(f"  Dibit distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    return dibits, soft_symbols


def demod_cqpsk(iq: np.ndarray, sample_rate: int, symbol_rate: int = 4800) -> tuple:
    """CQPSK/LSM demodulation (π/4-DQPSK)."""
    from scipy.signal import firwin, lfilter

    # Lowpass filter
    cutoff = 7250 / (sample_rate / 2)
    taps = firwin(63, min(cutoff, 0.99))
    filtered = lfilter(taps, 1, iq)

    # Symbol timing with differential phase
    sps = sample_rate / symbol_rate
    symbols = []
    pos = sps / 2
    prev = filtered[0] if len(filtered) > 0 else 1+0j

    while pos < len(filtered):
        curr = filtered[int(pos)]
        if abs(curr) > 1e-6 and abs(prev) > 1e-6:
            diff = (curr / abs(curr)) * np.conj(prev / abs(prev))
        else:
            diff = curr * np.conj(prev)
        symbols.append(np.angle(diff))
        prev = curr
        pos += sps

    symbols = np.array(symbols)
    print(f"  Extracted {len(symbols)} phase changes")
    print(f"  Phase stats: mean={np.mean(symbols):.4f}, std={np.std(symbols):.4f}")

    # π/4-DQPSK slicing
    half_pi = np.pi / 2
    dibits = np.zeros(len(symbols), dtype=np.uint8)
    dibits[symbols >= half_pi] = 1
    dibits[(symbols >= 0) & (symbols < half_pi)] = 0
    dibits[(symbols >= -half_pi) & (symbols < 0)] = 2
    dibits[symbols < -half_pi] = 3

    unique, counts = np.unique(dibits, return_counts=True)
    print(f"  Dibit distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    return dibits, symbols


# P25 sync pattern (48 bits = 24 dibits)
P25_SYNC_DIBITS = np.array([1,1,1,1,1,3,1,1,3,3,1,1,3,3,3,3,1,3,1,3,3,3,3,3], dtype=np.uint8)


def find_p25_syncs(dibits: np.ndarray, min_match: int = 18) -> list:
    """Find P25 sync patterns in dibit stream."""
    sync_len = len(P25_SYNC_DIBITS)
    matches = []

    for i in range(len(dibits) - sync_len):
        match_count = np.sum(dibits[i:i+sync_len] == P25_SYNC_DIBITS)
        if match_count >= min_match:
            matches.append((i, match_count))

    matches.sort(key=lambda x: -x[1])
    return matches


def decode_tsbks(dibits: np.ndarray, syncs: list) -> None:
    """Attempt to decode TSBK messages after sync patterns."""
    # NID is 64 bits (32 dibits) after sync
    # TSBK is after NID

    for pos, score in syncs[:5]:
        nid_start = pos + 24  # After sync
        if nid_start + 32 > len(dibits):
            continue

        nid_dibits = dibits[nid_start:nid_start + 32]

        # Extract DUID from NID (simplified - last 4 bits after deinterleave)
        # This is a simplified check
        duid_approx = (nid_dibits[-2] << 2) | nid_dibits[-1]

        duid_names = {
            0: "HDU (Header)",
            3: "TDU (Terminator)",
            5: "LDU1 (Voice 1)",
            7: "TSBK",
            10: "LDU2 (Voice 2)",
            12: "PDU",
            15: "TDULC"
        }

        print(f"  Sync at {pos} ({score}/24): DUID≈{duid_approx} ({duid_names.get(duid_approx, 'Unknown')})")


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="wavecapsdr-cli",
        description="WaveCap-SDR Command Line Interface"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # list-devices
    p_list = subparsers.add_parser("list-devices", help="List available SDR devices")
    p_list.set_defaults(func=cmd_list_devices)

    # capture-iq
    p_capture = subparsers.add_parser("capture-iq", help="Capture IQ samples to WAV file")
    p_capture.add_argument("-d", "--device", help="Device serial (partial match)")
    p_capture.add_argument("-a", "--antenna", help="Antenna: A, B, or C (RSP only)")
    p_capture.add_argument("-f", "--frequency", type=float, required=True,
                          help="Center frequency in Hz")
    p_capture.add_argument("-r", "--sample-rate", type=int, default=50000,
                          help="Sample rate (default: 50000)")
    p_capture.add_argument("-t", "--duration", type=float, default=60,
                          help="Duration in seconds (default: 60)")
    p_capture.add_argument("-o", "--output", required=True, help="Output WAV file")
    p_capture.add_argument("--gain", type=float, help="Overall gain in dB")
    p_capture.add_argument("--lna", type=int, help="LNA state 0-27 (RSP, SDRTrunk-style)")
    p_capture.add_argument("--gr", type=int, help="Gain reduction 20-59 dB (RSP, SDRTrunk-style)")
    p_capture.add_argument("--wideband", action="store_true",
                          help="Use wideband capture (8 MHz) with polyphase channelizer")
    p_capture.add_argument("--channel-freq", type=float,
                          help="Extract specific channel frequency (requires --wideband)")
    p_capture.set_defaults(func=cmd_capture_iq)

    # decode-iq
    p_decode = subparsers.add_parser("decode-iq", help="Decode IQ file through P25 pipeline")
    p_decode.add_argument("-f", "--file", required=True, help="Input WAV file")
    p_decode.add_argument("-m", "--modulation", default="c4fm",
                         choices=["c4fm", "cqpsk", "lsm"],
                         help="Modulation type (default: c4fm)")
    p_decode.add_argument("--freq-offset", type=float, default=None,
                         help="Manual frequency offset in Hz (auto-detected if not specified)")
    p_decode.set_defaults(func=cmd_decode_iq)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
