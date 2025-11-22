#!/usr/bin/env python3
"""
Stream Validator for WaveCap-SDR

Validates WebSocket and HTTP streaming health, measures metrics,
and diagnoses streaming issues.
"""
import argparse
import asyncio
import json
import sys
import time
from typing import Dict, Any, Optional

import numpy as np

try:
    import requests
except ImportError:
    print("Error: requests library required. Install with: pip install requests", file=sys.stderr)
    sys.exit(1)

try:
    import websockets
except ImportError:
    websockets = None


def validate_audio_stream(host: str, port: int, channel: str, duration: float, verbose: bool = False) -> Dict[str, Any]:
    """Validate HTTP audio stream health"""
    url = f"http://{host}:{port}/api/v1/stream/channels/{channel}.pcm?format=pcm16"

    result = {
        "type": "audio",
        "channel": channel,
        "url": url,
        "success": False,
        "status": "UNKNOWN",
    }

    print(f"Validating audio stream: {url}")
    start_time = time.time()
    connect_time = None

    try:
        response = requests.get(url, stream=True, timeout=10)
        connect_time = time.time() - start_time
        response.raise_for_status()

        bytes_received = 0
        sample_rate = 48000
        bytes_per_sample = 2
        audio_chunks = []

        for chunk in response.iter_content(chunk_size=4096):
            bytes_received += len(chunk)
            if verbose and len(audio_chunks) < 50:  # Collect first ~200KB for analysis
                audio_chunks.append(chunk)
            if time.time() - start_time >= duration:
                break

        elapsed = time.time() - start_time
        throughput = bytes_received / elapsed if elapsed > 0 else 0
        samples = bytes_received // bytes_per_sample
        expected_throughput = sample_rate * bytes_per_sample

        result.update({
            "success": True,
            "connect_time_ms": connect_time * 1000,
            "duration_seconds": elapsed,
            "bytes_received": bytes_received,
            "throughput_kbps": throughput / 1024,
            "expected_throughput_kbps": expected_throughput / 1024,
            "samples_received": samples,
            "throughput_ratio": throughput / expected_throughput if expected_throughput > 0 else 0,
        })

        # Audio quality analysis
        if audio_chunks:
            audio_bytes = b''.join(audio_chunks)
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            if len(audio) > 0:
                rms = np.sqrt(np.mean(audio ** 2))
                rms_db = 20 * np.log10(rms + 1e-10)
                peak = np.max(np.abs(audio))
                peak_db = 20 * np.log10(peak + 1e-10)

                # Detect silence (very low RMS)
                is_silent = rms_db < -50
                # Detect clipping (peak very close to 1.0)
                is_clipping = peak > 0.99

                result["audio_analysis"] = {
                    "rms_db": round(rms_db, 2),
                    "peak_db": round(peak_db, 2),
                    "is_silent": is_silent,
                    "is_clipping": is_clipping,
                }

        # Determine health status
        if throughput_ratio := result.get("throughput_ratio", 0):
            if throughput_ratio >= 0.95:
                result["status"] = "HEALTHY"
            elif throughput_ratio >= 0.80:
                result["status"] = "DEGRADED"
            else:
                result["status"] = "UNHEALTHY"

    except requests.exceptions.ConnectionError as e:
        result["error"] = f"Connection failed: {e}"
        result["status"] = "UNHEALTHY"
    except requests.exceptions.Timeout:
        result["error"] = "Connection timeout"
        result["status"] = "UNHEALTHY"
    except Exception as e:
        result["error"] = str(e)
        result["status"] = "UNHEALTHY"

    return result


async def validate_spectrum_stream_async(host: str, port: int, capture: str, duration: float) -> Dict[str, Any]:
    """Validate WebSocket spectrum stream health"""
    url = f"ws://{host}:{port}/api/v1/stream/captures/{capture}/spectrum"

    result = {
        "type": "spectrum",
        "capture": capture,
        "url": url,
        "success": False,
        "status": "UNKNOWN",
    }

    print(f"Validating spectrum stream: {url}")
    start_time = time.time()

    try:
        async with websockets.connect(url, ping_interval=None) as ws:
            connect_time = time.time() - start_time
            result["connect_time_ms"] = connect_time * 1000

            messages_received = 0
            bytes_received = 0
            fft_sizes = []
            latencies = []

            while time.time() - start_time < duration:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    recv_time = time.time()

                    messages_received += 1
                    bytes_received += len(msg)

                    # Parse spectrum data
                    try:
                        data = json.loads(msg)

                        # Track FFT size
                        if "bins" in data:
                            fft_sizes.append(len(data["bins"]))
                        elif "spectrum" in data:
                            fft_sizes.append(len(data["spectrum"]))

                        # Track latency if timestamp provided
                        if "timestamp" in data:
                            latency = recv_time - data["timestamp"]
                            if 0 < latency < 5:  # Reasonable latency range
                                latencies.append(latency * 1000)  # Convert to ms

                    except json.JSONDecodeError:
                        pass

                except asyncio.TimeoutError:
                    result["warning"] = "Some messages timed out"
                    break

            elapsed = time.time() - start_time

            result.update({
                "success": True,
                "duration_seconds": elapsed,
                "messages_received": messages_received,
                "bytes_received": bytes_received,
                "messages_per_second": messages_received / elapsed if elapsed > 0 else 0,
                "throughput_kbps": (bytes_received / elapsed / 1024) if elapsed > 0 else 0,
            })

            if fft_sizes:
                result["fft_size_avg"] = sum(fft_sizes) / len(fft_sizes)
                result["fft_size_consistent"] = len(set(fft_sizes)) == 1

            if latencies:
                result["latency_ms"] = {
                    "min": round(min(latencies), 2),
                    "max": round(max(latencies), 2),
                    "avg": round(sum(latencies) / len(latencies), 2),
                }

            # Determine health status
            fps = result.get("messages_per_second", 0)
            if fps >= 8:
                result["status"] = "HEALTHY"
            elif fps >= 4:
                result["status"] = "DEGRADED"
            else:
                result["status"] = "UNHEALTHY"

    except Exception as e:
        result["error"] = str(e)
        result["status"] = "UNHEALTHY"

    return result


def validate_spectrum_stream(host: str, port: int, capture: str, duration: float) -> Dict[str, Any]:
    """Sync wrapper for spectrum validation"""
    if websockets is None:
        return {
            "type": "spectrum",
            "capture": capture,
            "success": False,
            "status": "UNHEALTHY",
            "error": "websockets library not installed. Install with: pip install websockets"
        }
    return asyncio.run(validate_spectrum_stream_async(host, port, capture, duration))


async def validate_ws_audio_stream_async(host: str, port: int, channel: str, duration: float) -> Dict[str, Any]:
    """Validate WebSocket audio stream health"""
    url = f"ws://{host}:{port}/api/v1/stream/channels/{channel}"

    result = {
        "type": "audio_ws",
        "channel": channel,
        "url": url,
        "success": False,
        "status": "UNKNOWN",
    }

    print(f"Validating WebSocket audio stream: {url}")
    start_time = time.time()

    try:
        async with websockets.connect(url, ping_interval=None) as ws:
            connect_time = time.time() - start_time
            result["connect_time_ms"] = connect_time * 1000

            bytes_received = 0
            messages_received = 0
            audio_chunks = []

            while time.time() - start_time < duration:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    messages_received += 1
                    bytes_received += len(msg)

                    if len(audio_chunks) < 50:  # Collect some audio for analysis
                        audio_chunks.append(msg)

                except asyncio.TimeoutError:
                    result["warning"] = "Some messages timed out"
                    break

            elapsed = time.time() - start_time
            sample_rate = 48000
            bytes_per_sample = 2
            expected_throughput = sample_rate * bytes_per_sample
            throughput = bytes_received / elapsed if elapsed > 0 else 0

            result.update({
                "success": True,
                "duration_seconds": elapsed,
                "messages_received": messages_received,
                "bytes_received": bytes_received,
                "throughput_kbps": throughput / 1024,
                "expected_throughput_kbps": expected_throughput / 1024,
                "throughput_ratio": throughput / expected_throughput if expected_throughput > 0 else 0,
            })

            # Audio quality analysis
            if audio_chunks:
                audio_bytes = b''.join(audio_chunks)
                if len(audio_bytes) >= 2:
                    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                    if len(audio) > 0:
                        rms = np.sqrt(np.mean(audio ** 2))
                        rms_db = 20 * np.log10(rms + 1e-10)
                        peak = np.max(np.abs(audio))
                        peak_db = 20 * np.log10(peak + 1e-10)

                        result["audio_analysis"] = {
                            "rms_db": round(rms_db, 2),
                            "peak_db": round(peak_db, 2),
                            "is_silent": rms_db < -50,
                            "is_clipping": peak > 0.99,
                        }

            # Determine health status
            if throughput_ratio := result.get("throughput_ratio", 0):
                if throughput_ratio >= 0.95:
                    result["status"] = "HEALTHY"
                elif throughput_ratio >= 0.80:
                    result["status"] = "DEGRADED"
                else:
                    result["status"] = "UNHEALTHY"

    except Exception as e:
        result["error"] = str(e)
        result["status"] = "UNHEALTHY"

    return result


def validate_ws_audio_stream(host: str, port: int, channel: str, duration: float) -> Dict[str, Any]:
    """Sync wrapper for WebSocket audio validation"""
    if websockets is None:
        return {
            "type": "audio_ws",
            "channel": channel,
            "success": False,
            "status": "UNHEALTHY",
            "error": "websockets library not installed. Install with: pip install websockets"
        }
    return asyncio.run(validate_ws_audio_stream_async(host, port, channel, duration))


def get_server_health(host: str, port: int) -> Dict[str, Any]:
    """Query server health endpoint"""
    url = f"http://{host}:{port}/api/v1/health"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


def print_results(result: Dict[str, Any]):
    """Print validation results in a human-readable format"""
    print("\n" + "=" * 60)
    print(f"STREAM VALIDATION RESULTS: {result.get('type', 'unknown').upper()}")
    print("=" * 60)

    # Status with color indicator
    status = result.get("status", "UNKNOWN")
    status_icon = {"HEALTHY": "✓", "DEGRADED": "⚠", "UNHEALTHY": "✗"}.get(status, "?")
    print(f"\nStatus: {status_icon} {status}")

    if result.get("error"):
        print(f"Error: {result['error']}")
        return

    print(f"\nConnection:")
    if "connect_time_ms" in result:
        print(f"  Connect time: {result['connect_time_ms']:.1f} ms")
    print(f"  Duration: {result.get('duration_seconds', 0):.2f}s")

    print(f"\nThroughput:")
    if "bytes_received" in result:
        print(f"  Bytes received: {result['bytes_received']:,}")
    if "throughput_kbps" in result:
        print(f"  Throughput: {result['throughput_kbps']:.2f} KB/s")
    if "expected_throughput_kbps" in result:
        print(f"  Expected: {result['expected_throughput_kbps']:.2f} KB/s")
    if "throughput_ratio" in result:
        print(f"  Ratio: {result['throughput_ratio'] * 100:.1f}%")
    if "messages_per_second" in result:
        print(f"  Messages/sec: {result['messages_per_second']:.1f}")

    if "latency_ms" in result:
        lat = result["latency_ms"]
        print(f"\nLatency:")
        print(f"  Min: {lat['min']:.1f} ms")
        print(f"  Avg: {lat['avg']:.1f} ms")
        print(f"  Max: {lat['max']:.1f} ms")

    if "audio_analysis" in result:
        audio = result["audio_analysis"]
        print(f"\nAudio Analysis:")
        print(f"  RMS level: {audio['rms_db']:.1f} dB")
        print(f"  Peak level: {audio['peak_db']:.1f} dB")
        if audio.get("is_silent"):
            print(f"  ⚠ Audio appears silent")
        if audio.get("is_clipping"):
            print(f"  ⚠ Audio is clipping")

    if "fft_size_avg" in result:
        print(f"\nSpectrum:")
        print(f"  FFT size: {result['fft_size_avg']:.0f} bins")
        print(f"  Consistent: {'Yes' if result.get('fft_size_consistent') else 'No'}")

    if result.get("warning"):
        print(f"\n⚠ Warning: {result['warning']}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Validate WaveCap-SDR streams',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate HTTP audio stream
  %(prog)s --type audio --channel ch1 --duration 10

  # Validate WebSocket audio stream
  %(prog)s --type audio_ws --channel ch1 --duration 10

  # Validate spectrum stream
  %(prog)s --type spectrum --capture cap_abc123 --duration 10

  # Check server health first
  %(prog)s --health-check
        """
    )

    parser.add_argument('--type', choices=['audio', 'audio_ws', 'spectrum', 'iq'], default='audio',
                       help='Stream type to validate (default: audio)')
    parser.add_argument('--channel', default='ch1',
                       help='Channel ID for audio streams (default: ch1)')
    parser.add_argument('--capture',
                       help='Capture ID for spectrum/IQ streams')
    parser.add_argument('--duration', type=float, default=10.0,
                       help='Seconds to monitor (default: 10)')
    parser.add_argument('--host', default='127.0.0.1',
                       help='Server host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8087,
                       help='Server port (default: 8087)')
    parser.add_argument('--report', action='store_true',
                       help='Output JSON report')
    parser.add_argument('--health-check', action='store_true',
                       help='Check server health first')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output with audio analysis')

    args = parser.parse_args()

    # Check server health if requested
    if args.health_check:
        print("Checking server health...")
        health = get_server_health(args.host, args.port)
        print(f"Server status: {health.get('status', 'unknown')}")

        if health.get("status") == "error":
            print(f"Error: {health.get('error')}")
            return 1

        if "checks" in health:
            for check_name, check_data in health["checks"].items():
                status = check_data.get("status", "unknown")
                count = check_data.get("count", "N/A")
                print(f"  {check_name}: {status} (count: {count})")
        print()

    # Run validation based on type
    if args.type == 'audio':
        result = validate_audio_stream(args.host, args.port, args.channel, args.duration, args.verbose)
    elif args.type == 'audio_ws':
        result = validate_ws_audio_stream(args.host, args.port, args.channel, args.duration)
    elif args.type == 'spectrum':
        if not args.capture:
            print("Error: --capture required for spectrum streams", file=sys.stderr)
            return 1
        result = validate_spectrum_stream(args.host, args.port, args.capture, args.duration)
    elif args.type == 'iq':
        print(f"Stream type '{args.type}' not yet implemented")
        return 1
    else:
        print(f"Unknown stream type: {args.type}", file=sys.stderr)
        return 1

    # Output results
    if args.report:
        print(json.dumps(result, indent=2))
    else:
        print_results(result)

    # Exit code based on status
    if result.get("status") == "HEALTHY":
        return 0
    elif result.get("status") == "DEGRADED":
        return 0  # Degraded is still acceptable
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())
