#!/usr/bin/env python3
"""
Harness Runner for WaveCap-SDR

Automated test harness execution with parameter sweeps and reporting.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


def run_harness_test(
    preset: str,
    duration: float,
    driver: str,
    device_args: Optional[str],
    gain: Optional[float],
    bandwidth: Optional[float],
    output_dir: Path,
    test_name: str,
) -> Dict[str, Any]:
    """
    Run a single harness test and return results.

    Returns:
        {
            'test_name': str,
            'exit_code': int,
            'stdout': str,
            'stderr': str,
            'results': dict or None,
        }
    """
    # Build harness command
    cmd = [
        sys.executable,
        "-m", "wavecapsdr.harness",
        "--start-server",
        "--driver", driver,
        "--preset", preset,
        "--duration", str(duration),
        "--out", str(output_dir / test_name),
    ]

    if device_args:
        cmd.extend(["--device-args", device_args])

    if gain is not None:
        cmd.extend(["--gain", str(gain)])

    if bandwidth is not None:
        cmd.extend(["--bandwidth", str(bandwidth)])

    # Run harness
    print(f"\n{'='*60}")
    print(f"Running test: {test_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    env = {"PYTHONPATH": "backend"}
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parents[3],  # Root of repo
        env={**subprocess.os.environ, **env}
    )

    # Parse JSON output from stdout
    results_json = None
    try:
        # Look for JSON in stdout
        for line in result.stdout.split('\n'):
            if line.strip().startswith('{'):
                results_json = json.loads(line)
                break
    except json.JSONDecodeError:
        pass

    return {
        'test_name': test_name,
        'exit_code': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'results': results_json,
        'params': {
            'preset': preset,
            'duration': duration,
            'driver': driver,
            'device_args': device_args,
            'gain': gain,
            'bandwidth': bandwidth,
        }
    }


def run_gain_sweep(
    preset: str,
    duration: float,
    driver: str,
    device_args: Optional[str],
    gain_values: List[float],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """Run harness tests with different gain values"""
    results = []

    for gain in gain_values:
        test_name = f"gain_{gain:.1f}db"
        result = run_harness_test(
            preset, duration, driver, device_args,
            gain, None, output_dir, test_name
        )
        results.append(result)

    return results


def run_bandwidth_sweep(
    preset: str,
    duration: float,
    driver: str,
    device_args: Optional[str],
    gain: Optional[float],
    bandwidth_values: List[float],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """Run harness tests with different bandwidth values"""
    results = []

    for bw in bandwidth_values:
        test_name = f"bw_{bw/1e6:.2f}mhz"
        result = run_harness_test(
            preset, duration, driver, device_args,
            gain, bw, output_dir, test_name
        )
        results.append(result)

    return results


def generate_report(results: List[Dict[str, Any]], output_path: Path):
    """Generate JSON and text reports"""
    # JSON report
    json_path = output_path / "report.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ JSON report saved to: {json_path}")

    # Text summary
    txt_path = output_path / "report.txt"
    with open(txt_path, 'w') as f:
        f.write(f"WaveCap-SDR Harness Test Report\n")
        f.write(f"{'='*60}\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total tests: {len(results)}\n\n")

        passed = sum(1 for r in results if r['exit_code'] == 0)
        failed = len(results) - passed

        f.write(f"Summary: {passed} PASSED, {failed} FAILED\n")
        f.write(f"{'='*60}\n\n")

        for i, result in enumerate(results, 1):
            f.write(f"\nTest {i}: {result['test_name']}\n")
            f.write(f"{'-'*60}\n")
            f.write(f"Status: {'PASS' if result['exit_code'] == 0 else 'FAIL'}\n")
            f.write(f"Exit code: {result['exit_code']}\n")

            if result['params']:
                f.write(f"Parameters:\n")
                for key, val in result['params'].items():
                    if val is not None:
                        f.write(f"  {key}: {val}\n")

            if result['results']:
                f.write(f"\nChannel Results:\n")
                for channel in result['results'].get('channels', []):
                    status = channel.get('status', 'UNKNOWN')
                    rms = channel.get('rmsDb', -100)
                    peak = channel.get('peakDb', -100)
                    label = channel.get('label', 'Unknown')
                    f.write(f"  [{status}] {label}: RMS={rms:.1f} dB, Peak={peak:.1f} dB\n")

            if result['stderr']:
                f.write(f"\nErrors:\n{result['stderr']}\n")

    print(f"✓ Text report saved to: {txt_path}")

    # Print summary to console
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests:  {len(results)}")
    print(f"Passed:       {passed}")
    print(f"Failed:       {failed}")
    print(f"Success rate: {passed/len(results)*100:.1f}%")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run WaveCap-SDR harness with parameter sweeps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--preset', default='kexp', help='Preset name (default: kexp)')
    parser.add_argument('--duration', type=float, default=3.0, help='Seconds to capture (default: 3)')
    parser.add_argument('--driver', default='fake', help='Driver: fake, soapy, rtl (default: fake)')
    parser.add_argument('--device-args', help='Device selector string')
    parser.add_argument('--output-dir', default='harness_results', help='Output directory')

    # Sweep options
    parser.add_argument('--sweep', choices=['gain', 'bandwidth', 'none'], default='none',
                       help='Parameter to sweep (default: none)')
    parser.add_argument('--gain', type=float, help='Fixed gain in dB (if not sweeping)')
    parser.add_argument('--gain-values', type=float, nargs='+',
                       help='Gain values to sweep (e.g., 10 20 30 40)')
    parser.add_argument('--bandwidth-values', type=float, nargs='+',
                       help='Bandwidth values to sweep in Hz (e.g., 200000 500000 1000000)')

    # Reporting
    parser.add_argument('--report', action='store_true', help='Generate detailed report')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nWaveCap-SDR Harness Runner")
    print(f"Output directory: {output_dir.absolute()}\n")

    # Run tests based on sweep type
    results = []

    if args.sweep == 'gain':
        if not args.gain_values:
            print("Error: --gain-values required for gain sweep", file=sys.stderr)
            return 1

        results = run_gain_sweep(
            args.preset, args.duration, args.driver, args.device_args,
            args.gain_values, output_dir
        )

    elif args.sweep == 'bandwidth':
        if not args.bandwidth_values:
            print("Error: --bandwidth-values required for bandwidth sweep", file=sys.stderr)
            return 1

        results = run_bandwidth_sweep(
            args.preset, args.duration, args.driver, args.device_args,
            args.gain, args.bandwidth_values, output_dir
        )

    else:
        # Single test
        test_name = f"{args.preset}_{args.driver}"
        result = run_harness_test(
            args.preset, args.duration, args.driver, args.device_args,
            args.gain, None, output_dir, test_name
        )
        results = [result]

    # Generate report if requested
    if args.report:
        generate_report(results, output_dir)

    # Exit with non-zero if any tests failed
    failed = sum(1 for r in results if r['exit_code'] != 0)
    if failed > 0:
        print(f"\n⚠ {failed} test(s) failed")
        return 1

    print(f"\n✓ All tests passed")
    return 0


if __name__ == '__main__':
    sys.exit(main())
