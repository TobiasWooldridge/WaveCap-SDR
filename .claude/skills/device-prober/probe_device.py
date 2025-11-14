#!/usr/bin/env python3
"""
Device Prober for WaveCap-SDR

Probe SDR hardware capabilities and test configurations.
"""

import argparse
import json
import sys
from typing import Dict, Any, List, Optional

try:
    import SoapySDR
except ImportError:
    print("Error: SoapySDR Python bindings not installed", file=sys.stderr)
    print("Install with: pip install SoapySDR", file=sys.stderr)
    sys.exit(1)

import numpy as np


def list_devices() -> List[Dict[str, str]]:
    """Enumerate all SDR devices"""
    results = SoapySDR.Device.enumerate()
    return [dict(r) for r in results]


def probe_device(device_args: str) -> Dict[str, Any]:
    """Probe device capabilities"""
    # Parse device args
    args_dict = {}
    for arg in device_args.split(','):
        if '=' in arg:
            key, val = arg.split('=', 1)
            args_dict[key] = val

    # Open device
    try:
        sdr = SoapySDR.Device(args_dict)
    except Exception as e:
        return {'error': f"Failed to open device: {e}"}

    info = {}

    try:
        # Hardware info
        info['driver'] = sdr.getDriverKey()
        info['hardware'] = sdr.getHardwareKey()

        hardware_info = {}
        for key in ['serial', 'product', 'manufacturer', 'revision']:
            try:
                val = sdr.getHardwareInfo().get(key, 'N/A')
                hardware_info[key] = val
            except:
                pass
        info['hardware_info'] = hardware_info

        # Channels
        num_rx = sdr.getNumChannels(SoapySDR.SOAPY_SDR_RX)
        info['num_rx_channels'] = num_rx

        # Per-channel info (use channel 0)
        channel = 0

        # Frequency ranges
        freq_ranges = []
        for r in sdr.getFrequencyRange(SoapySDR.SOAPY_SDR_RX, channel):
            freq_ranges.append({
                'min_hz': r.minimum(),
                'max_hz': r.maximum(),
                'step_hz': r.step() if r.step() > 0 else None,
            })
        info['frequency_ranges'] = freq_ranges

        # Sample rates
        sample_rates = []
        for rate in sdr.listSampleRates(SoapySDR.SOAPY_SDR_RX, channel):
            sample_rates.append(rate)

        # Check for sample rate ranges
        sample_rate_ranges = []
        try:
            for r in sdr.getSampleRateRange(SoapySDR.SOAPY_SDR_RX, channel):
                sample_rate_ranges.append({
                    'min': r.minimum(),
                    'max': r.maximum(),
                })
        except:
            pass

        info['sample_rates'] = sorted(sample_rates) if sample_rates else None
        info['sample_rate_ranges'] = sample_rate_ranges if sample_rate_ranges else None

        # Bandwidths
        bandwidths = []
        try:
            for bw in sdr.listBandwidths(SoapySDR.SOAPY_SDR_RX, channel):
                bandwidths.append(bw)
            info['bandwidths'] = sorted(bandwidths) if bandwidths else None
        except:
            info['bandwidths'] = None

        # Gain elements
        gains = {}
        for gain_name in sdr.listGains(SoapySDR.SOAPY_SDR_RX, channel):
            gain_range = sdr.getGainRange(SoapySDR.SOAPY_SDR_RX, channel, gain_name)
            gains[gain_name] = {
                'min': gain_range.minimum(),
                'max': gain_range.maximum(),
                'step': gain_range.step() if gain_range.step() > 0 else None,
            }

        # Overall gain range
        overall_range = sdr.getGainRange(SoapySDR.SOAPY_SDR_RX, channel)
        gains['overall'] = {
            'min': overall_range.minimum(),
            'max': overall_range.maximum(),
        }

        info['gains'] = gains

        # AGC support
        try:
            has_agc = sdr.hasGainMode(SoapySDR.SOAPY_SDR_RX, channel)
            info['has_automatic_gain'] = has_agc
        except:
            info['has_automatic_gain'] = False

        # Antennas
        antennas = sdr.listAntennas(SoapySDR.SOAPY_SDR_RX, channel)
        info['antennas'] = antennas

        # Sensors
        sensors = {}
        for sensor_name in sdr.listSensors():
            try:
                sensor_info = sdr.getSensorInfo(sensor_name)
                sensors[sensor_name] = {
                    'description': sensor_info.description,
                    'key': sensor_info.key,
                    'type': sensor_info.type,
                }
            except:
                sensors[sensor_name] = {}

        info['sensors'] = sensors if sensors else None

        # Frontend settings
        try:
            has_dc_offset = sdr.hasDCOffsetMode(SoapySDR.SOAPY_SDR_RX, channel)
            info['has_dc_offset_mode'] = has_dc_offset
        except:
            info['has_dc_offset_mode'] = False

        try:
            has_freq_correction = sdr.hasFrequencyCorrection(SoapySDR.SOAPY_SDR_RX, channel)
            info['has_frequency_correction'] = has_freq_correction
        except:
            info['has_frequency_correction'] = False

    finally:
        SoapySDR.Device.unmake(sdr)

    return info


def test_capture(device_args: str, duration: float = 2.0) -> Dict[str, Any]:
    """Test device by performing a short capture"""
    args_dict = {}
    for arg in device_args.split(','):
        if '=' in arg:
            key, val = arg.split('=', 1)
            args_dict[key] = val

    result = {'success': False, 'error': None}

    try:
        sdr = SoapySDR.Device(args_dict)

        # Configure device
        sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, 2.048e6)
        sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, 100e6)

        # Setup stream
        stream = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
        sdr.activateStream(stream)

        # Capture samples
        samples_to_read = int(2.048e6 * duration)
        buffer = np.zeros(4096, dtype=np.complex64)
        total_samples = 0

        while total_samples < samples_to_read:
            sr = sdr.readStream(stream, [buffer], len(buffer), timeoutUs=1000000)
            if sr.ret > 0:
                total_samples += sr.ret
            elif sr.ret == SoapySDR.SOAPY_SDR_TIMEOUT:
                result['error'] = "Stream timeout"
                break
            else:
                result['error'] = f"Stream error: {sr.ret}"
                break

        # Deactivate stream
        sdr.deactivateStream(stream)
        sdr.closeStream(stream)
        SoapySDR.Device.unmake(sdr)

        if total_samples > 0:
            result['success'] = True
            result['samples_captured'] = total_samples
            result['duration_seconds'] = total_samples / 2.048e6

    except Exception as e:
        result['error'] = str(e)

    return result


def print_device_info(info: Dict[str, Any]):
    """Print device information in human-readable format"""
    if 'error' in info:
        print(f"ERROR: {info['error']}")
        return

    print("\n" + "="*60)
    print("SDR DEVICE CAPABILITIES")
    print("="*60)

    print(f"\nDriver:      {info.get('driver', 'Unknown')}")
    print(f"Hardware:    {info.get('hardware', 'Unknown')}")

    hw_info = info.get('hardware_info', {})
    if hw_info:
        print(f"Serial:      {hw_info.get('serial', 'N/A')}")
        print(f"Product:     {hw_info.get('product', 'N/A')}")
        print(f"Manufacturer: {hw_info.get('manufacturer', 'N/A')}")

    print(f"\nRX Channels: {info.get('num_rx_channels', 0)}")

    # Frequency ranges
    print("\n" + "-"*60)
    print("FREQUENCY RANGES")
    print("-"*60)
    for i, fr in enumerate(info.get('frequency_ranges', []), 1):
        min_mhz = fr['min_hz'] / 1e6
        max_mhz = fr['max_hz'] / 1e6
        print(f"  Range {i}: {min_mhz:.3f} MHz - {max_mhz:.3f} MHz")

    # Sample rates
    print("\n" + "-"*60)
    print("SAMPLE RATES")
    print("-"*60)

    rates = info.get('sample_rates')
    if rates:
        print("  Discrete rates:")
        for rate in rates[:10]:  # Show first 10
            print(f"    {rate/1e6:.3f} MHz ({rate} Hz)")
        if len(rates) > 10:
            print(f"    ... and {len(rates)-10} more")

    rate_ranges = info.get('sample_rate_ranges')
    if rate_ranges:
        print("  Rate ranges:")
        for rr in rate_ranges:
            print(f"    {rr['min']/1e6:.3f} - {rr['max']/1e6:.3f} MHz")

    # Gains
    print("\n" + "-"*60)
    print("GAIN SETTINGS")
    print("-"*60)
    gains = info.get('gains', {})

    if 'overall' in gains:
        overall = gains['overall']
        print(f"  Overall: {overall['min']:.1f} - {overall['max']:.1f} dB")

    for name, gr in gains.items():
        if name != 'overall':
            step_str = f", step {gr['step']:.1f}" if gr.get('step') else ""
            print(f"  {name}: {gr['min']:.1f} - {gr['max']:.1f} dB{step_str}")

    if info.get('has_automatic_gain'):
        print("  ✓ Automatic Gain Control supported")

    # Antennas
    antennas = info.get('antennas', [])
    if antennas:
        print("\n" + "-"*60)
        print("ANTENNAS")
        print("-"*60)
        for ant in antennas:
            print(f"  {ant}")

    # Sensors
    sensors = info.get('sensors')
    if sensors:
        print("\n" + "-"*60)
        print("SENSORS")
        print("-"*60)
        for name, sinfo in sensors.items():
            desc = sinfo.get('description', 'N/A')
            print(f"  {name}: {desc}")

    # Features
    features = []
    if info.get('has_dc_offset_mode'):
        features.append("DC offset correction")
    if info.get('has_frequency_correction'):
        features.append("Frequency correction")

    if features:
        print("\n" + "-"*60)
        print("FEATURES")
        print("-"*60)
        for feat in features:
            print(f"  ✓ {feat}")

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Probe SDR device capabilities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--list', action='store_true', help='List all devices')
    parser.add_argument('--device', help='Device args (e.g., "driver=rtlsdr")')
    parser.add_argument('--output', help='Save probe results to JSON file')
    parser.add_argument('--test-capture', action='store_true', help='Test device with short capture')
    parser.add_argument('--test-duration', type=float, default=2.0, help='Test capture duration (default: 2s)')

    args = parser.parse_args()

    if args.list:
        print("\nEnumerating SDR devices...")
        devices = list_devices()

        if not devices:
            print("No devices found.")
            return 1

        print(f"\nFound {len(devices)} device(s):\n")
        for i, dev in enumerate(devices):
            print(f"  [{i}] {', '.join(f'{k}={v}' for k, v in dev.items())}")

        print()
        return 0

    if not args.device:
        print("Error: --device required (or use --list to enumerate)", file=sys.stderr)
        return 1

    # Probe device
    print(f"\nProbing device: {args.device}")
    info = probe_device(args.device)

    # Print results
    print_device_info(info)

    # Save to JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"\n✓ Results saved to: {args.output}")

    # Test capture if requested
    if args.test_capture and 'error' not in info:
        print("\nTesting device with short capture...")
        test_result = test_capture(args.device, args.test_duration)

        if test_result['success']:
            print(f"✓ Capture successful!")
            print(f"  Samples: {test_result['samples_captured']}")
            print(f"  Duration: {test_result['duration_seconds']:.2f}s")
        else:
            print(f"✗ Capture failed: {test_result['error']}")
            return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
