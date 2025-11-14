#!/usr/bin/env python3
"""Channel Optimizer for WaveCap-SDR"""
import argparse
import sys
import time
import requests
import numpy as np

def measure_audio_quality(host, port, channel_id, duration=2.0):
    """Capture audio and measure RMS level"""
    url = f"http://{host}:{port}/api/v1/stream/channels/{channel_id}.pcm?format=pcm16"

    try:
        response = requests.get(url, stream=True, timeout=5)
        response.raise_for_status()

        audio_bytes = b''
        sample_rate = 48000
        bytes_needed = int(duration * sample_rate * 2)

        for chunk in response.iter_content(chunk_size=4096):
            audio_bytes += chunk
            if len(audio_bytes) >= bytes_needed:
                audio_bytes = audio_bytes[:bytes_needed]
                break

        # Convert to numpy
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Measure RMS
        rms = np.sqrt(np.mean(audio ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)

        return {'rms': rms, 'rms_db': rms_db, 'success': True}

    except Exception as e:
        return {'success': False, 'error': str(e)}

def optimize_offset(host, port, capture_id, initial_offset, offsets_to_test):
    """Find optimal frequency offset"""
    print(f"\nOptimizing offset for capture {capture_id}")
    print(f"Testing offsets: {offsets_to_test}\n")

    results = []

    for offset in offsets_to_test:
        print(f"Testing offset {offset} Hz...")

        # Create test channel
        channel_data = {
            'name': f'Test {offset}',
            'offset_hz': offset,
            'mode': 'fm',
        }

        try:
            # Create channel
            resp = requests.post(
                f"http://{host}:{port}/api/v1/captures/{capture_id}/channels",
                json=channel_data
            )
            resp.raise_for_status()
            channel_id = resp.json()['id']

            # Start channel
            requests.post(f"http://{host}:{port}/api/v1/channels/{channel_id}/start")
            time.sleep(1)  # Wait for startup

            # Measure quality
            quality = measure_audio_quality(host, port, channel_id)

            if quality['success']:
                print(f"  RMS: {quality['rms_db']:.1f} dB")
                results.append({'offset': offset, 'rms_db': quality['rms_db']})
            else:
                print(f"  Failed: {quality.get('error')}")

            # Delete channel
            requests.delete(f"http://{host}:{port}/api/v1/channels/{channel_id}")

        except Exception as e:
            print(f"  Error: {e}")

    # Find best offset
    if results:
        best = max(results, key=lambda x: x['rms_db'])
        print(f"\n✓ Best offset: {best['offset']} Hz (RMS: {best['rms_db']:.1f} dB)")
        return best['offset']
    else:
        print("\n✗ Optimization failed")
        return None

def main():
    parser = argparse.ArgumentParser(description='Optimize channel parameters')
    parser.add_argument('--capture', required=True)
    parser.add_argument('--frequency', type=float)
    parser.add_argument('--offset', type=float, default=0)
    parser.add_argument('--optimize', choices=['offset', 'squelch', 'agc'], required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8087)

    args = parser.parse_args()

    if args.optimize == 'offset':
        # Test offsets around initial
        offsets = [args.offset + x for x in [-10000, -5000, 0, 5000, 10000]]
        best_offset = optimize_offset(args.host, args.port, args.capture, args.offset, offsets)
        return 0 if best_offset is not None else 1

    else:
        print(f"Optimization type '{args.optimize}' not yet implemented")
        return 1

if __name__ == '__main__':
    sys.exit(main())
