#!/usr/bin/env python3
"""Stream Validator for WaveCap-SDR"""
import argparse
import sys
import time
import numpy as np
import requests

def validate_audio_stream(host, port, channel, duration):
    """Validate audio stream health"""
    url = f"http://{host}:{port}/api/v1/stream/channels/{channel}.pcm?format=pcm16"

    print(f"Validating audio stream: {url}")
    start_time = time.time()

    try:
        response = requests.get(url, stream=True, timeout=5)
        response.raise_for_status()

        bytes_received = 0
        sample_rate = 48000
        bytes_per_sample = 2

        for chunk in response.iter_content(chunk_size=4096):
            bytes_received += len(chunk)
            if time.time() - start_time >= duration:
                break

        elapsed = time.time() - start_time
        throughput = bytes_received / elapsed
        samples = bytes_received // bytes_per_sample

        print(f"\n✓ Stream Validation Results:")
        print(f"  Duration: {elapsed:.2f}s")
        print(f"  Bytes: {bytes_received}")
        print(f"  Throughput: {throughput/1024:.2f} KB/s")
        print(f"  Samples: {samples}")
        print(f"  Expected: {sample_rate * bytes_per_sample / 1024:.2f} KB/s")

        if throughput > 90000:  # ~90 KB/s for 48 kHz
            print("  Status: HEALTHY ✓")
            return 0
        else:
            print("  Status: DEGRADED ⚠")
            return 1

    except Exception as e:
        print(f"✗ Stream validation failed: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(description='Validate WaveCap-SDR streams')
    parser.add_argument('--type', choices=['audio', 'spectrum', 'iq'], default='audio')
    parser.add_argument('--channel', default='ch1')
    parser.add_argument('--capture')
    parser.add_argument('--duration', type=float, default=10.0)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8087)
    parser.add_argument('--report', action='store_true')

    args = parser.parse_args()

    if args.type == 'audio':
        return validate_audio_stream(args.host, args.port, args.channel, args.duration)
    else:
        print(f"Stream type '{args.type}' not yet implemented")
        return 1

if __name__ == '__main__':
    sys.exit(main())
