#!/usr/bin/env python3
"""SDRplay stress test - exercises multiple SDRplay devices with complex operations.

Tests the global lock serialization to ensure no sdrplay_api_Fail errors occur
when multiple devices are operated simultaneously.

Usage:
    # Start server first:
    cd backend && PYTHONPATH=. python -m wavecapsdr --bind 0.0.0.0 --port 8087

    # Run stress test (from project root):
    cd backend && PYTHONPATH=. python ../scripts/sdrplay_stress_test.py

    # With options:
    PYTHONPATH=. python ../scripts/sdrplay_stress_test.py --host 192.168.2.98 --port 8087 --iterations 5

Test scenarios:
    1. Simultaneous startup of all available SDRplay devices
    2. Rapid antenna switching on each device
    3. Concurrent reconfiguration (frequency, gain, bandwidth changes)
    4. Stop/start cycles while other devices are streaming
    5. Rapid fire API calls from multiple threads
    6. Long-running stability (continuous operation)

Success criteria:
    - Zero failures during test
    - All captures remain in "running" state
    - No "sdrplay_api_Fail" errors
    - No IQ watchdog triggers
    - Consistent spectrum data from all devices
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    duration_ms: float
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StressTestStats:
    """Accumulated stats for the stress test run."""
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    errors: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def record(self, result: TestResult) -> None:
        self.total_tests += 1
        if result.passed:
            self.passed += 1
            logger.info(f"  PASS: {result.name} ({result.duration_ms:.0f}ms)")
        else:
            self.failed += 1
            self.errors.append(f"{result.name}: {result.error}")
            logger.error(f"  FAIL: {result.name} - {result.error}")

    def summary(self) -> str:
        elapsed = time.time() - self.start_time
        return (
            f"\n{'='*60}\n"
            f"STRESS TEST SUMMARY\n"
            f"{'='*60}\n"
            f"Total tests:  {self.total_tests}\n"
            f"Passed:       {self.passed}\n"
            f"Failed:       {self.failed}\n"
            f"Duration:     {elapsed:.1f}s\n"
            f"{'='*60}\n"
            + ("\n".join(f"  - {e}" for e in self.errors) if self.errors else "All tests passed!")
        )


class SDRplayStressTester:
    """Stress test runner for SDRplay devices via WaveCap-SDR API."""

    def __init__(self, host: str, port: int):
        self.base_url = f"http://{host}:{port}/api/v1"
        self.host = host
        self.port = port
        self.stats = StressTestStats()
        self.capture_ids: List[str] = []
        self.sdrplay_devices: List[Dict[str, Any]] = []

    async def _request(
        self,
        method: str,
        path: str,
        json: Optional[Dict] = None,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """Make HTTP request to API."""
        async with httpx.AsyncClient(timeout=timeout) as client:
            url = f"{self.base_url}{path}"
            response = await client.request(method, url, json=json)
            response.raise_for_status()
            return response.json() if response.content else {}

    async def get_sdrplay_devices(self) -> List[Dict[str, Any]]:
        """Get all available SDRplay devices."""
        devices = await self._request("GET", "/devices")
        sdrplay = [d for d in devices if "sdrplay" in d.get("driver", "").lower()]
        return sdrplay

    async def create_capture(self, device_id: str, name: str) -> str:
        """Create a new capture on the specified device."""
        result = await self._request("POST", "/captures", json={
            "deviceId": device_id,
            "name": name,
            "centerHz": 90_300_000,  # KEXP 90.3 FM
            "sampleRate": 2_000_000,
            "gain": 30,
        })
        return result["id"]

    async def start_capture(self, capture_id: str) -> Dict[str, Any]:
        """Start a capture."""
        return await self._request("POST", f"/captures/{capture_id}/start")

    async def stop_capture(self, capture_id: str) -> Dict[str, Any]:
        """Stop a capture."""
        return await self._request("POST", f"/captures/{capture_id}/stop")

    async def delete_capture(self, capture_id: str) -> None:
        """Delete a capture."""
        await self._request("DELETE", f"/captures/{capture_id}")

    async def get_capture_status(self, capture_id: str) -> Dict[str, Any]:
        """Get capture status."""
        captures = await self._request("GET", "/captures")
        for c in captures:
            if c["id"] == capture_id:
                return c
        raise ValueError(f"Capture {capture_id} not found")

    async def reconfigure_capture(
        self,
        capture_id: str,
        center_hz: Optional[float] = None,
        gain: Optional[float] = None,
        antenna: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Reconfigure a running capture."""
        params = {}
        if center_hz is not None:
            params["centerHz"] = center_hz
        if gain is not None:
            params["gain"] = gain
        if antenna is not None:
            params["antenna"] = antenna
        return await self._request("PATCH", f"/captures/{capture_id}", json=params)

    async def wait_for_running(
        self,
        capture_id: str,
        timeout: float = 60.0,
        check_interval: float = 0.5,
    ) -> bool:
        """Wait for capture to reach 'running' state."""
        start = time.time()
        while time.time() - start < timeout:
            status = await self.get_capture_status(capture_id)
            state = status.get("state", "")
            if state == "running":
                return True
            elif state in ("error", "failed"):
                raise RuntimeError(f"Capture {capture_id} failed: {status.get('error', 'Unknown')}")
            await asyncio.sleep(check_interval)
        return False

    # =========================================================================
    # Test Scenarios
    # =========================================================================

    async def test_simultaneous_startup(self) -> TestResult:
        """Test 1: Start all SDRplay devices simultaneously."""
        name = "Simultaneous Startup"
        start = time.time()

        try:
            # Create captures for all devices
            capture_ids = []
            for i, device in enumerate(self.sdrplay_devices):
                device_id = device["id"]
                cid = await self.create_capture(device_id, f"stress_test_{i}")
                capture_ids.append(cid)
                logger.info(f"    Created capture {cid} for device {device_id}")

            # Start all captures concurrently
            logger.info(f"    Starting {len(capture_ids)} captures simultaneously...")
            start_tasks = [self.start_capture(cid) for cid in capture_ids]
            await asyncio.gather(*start_tasks)

            # Wait for all to reach running state
            logger.info("    Waiting for all captures to reach 'running' state...")
            wait_tasks = [self.wait_for_running(cid, timeout=120.0) for cid in capture_ids]
            results = await asyncio.gather(*wait_tasks, return_exceptions=True)

            # Check results
            failed = []
            for i, (cid, result) in enumerate(zip(capture_ids, results)):
                if isinstance(result, Exception):
                    failed.append(f"{cid}: {result}")
                elif not result:
                    failed.append(f"{cid}: Timed out waiting for running state")

            if failed:
                return TestResult(
                    name=name,
                    passed=False,
                    duration_ms=(time.time() - start) * 1000,
                    error=f"Failed captures: {', '.join(failed)}",
                )

            self.capture_ids = capture_ids
            return TestResult(
                name=name,
                passed=True,
                duration_ms=(time.time() - start) * 1000,
                details={"captures_started": len(capture_ids)},
            )

        except Exception as e:
            return TestResult(
                name=name,
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                error=str(e),
            )

    async def test_rapid_antenna_switching(self, iterations: int = 5) -> TestResult:
        """Test 2: Rapidly switch antennas on each device."""
        name = "Rapid Antenna Switching"
        start = time.time()

        if not self.capture_ids:
            return TestResult(
                name=name,
                passed=False,
                duration_ms=0,
                error="No captures available for testing",
            )

        try:
            # Get available antennas for each capture
            antennas_map: Dict[str, List[str]] = {}
            for cid in self.capture_ids:
                status = await self.get_capture_status(cid)
                antennas = status.get("available_antennas", ["Antenna A", "Antenna B", "Antenna C"])
                antennas_map[cid] = list(antennas) if antennas else ["Antenna A", "Antenna B"]

            errors = []
            for iteration in range(iterations):
                logger.info(f"    Antenna switch iteration {iteration + 1}/{iterations}")

                # Switch all antennas concurrently
                tasks = []
                for cid in self.capture_ids:
                    antennas = antennas_map[cid]
                    new_antenna = antennas[iteration % len(antennas)]
                    tasks.append(self.reconfigure_capture(cid, antenna=new_antenna))

                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        errors.append(f"Iteration {iteration}, capture {self.capture_ids[i]}: {result}")

                # Verify all still running (longer wait due to 1s cooldown)
                await asyncio.sleep(1.5)
                for cid in self.capture_ids:
                    status = await self.get_capture_status(cid)
                    if status.get("state") != "running":
                        errors.append(f"Capture {cid} not running after antenna switch: {status.get('state')}")

            if errors:
                return TestResult(
                    name=name,
                    passed=False,
                    duration_ms=(time.time() - start) * 1000,
                    error=f"{len(errors)} errors: {errors[0]}{'...' if len(errors) > 1 else ''}",
                )

            return TestResult(
                name=name,
                passed=True,
                duration_ms=(time.time() - start) * 1000,
                details={"iterations": iterations, "captures": len(self.capture_ids)},
            )

        except Exception as e:
            return TestResult(
                name=name,
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                error=str(e),
            )

    async def test_concurrent_reconfiguration(self, iterations: int = 5) -> TestResult:
        """Test 3: Concurrent reconfiguration (frequency, gain changes)."""
        name = "Concurrent Reconfiguration"
        start = time.time()

        if not self.capture_ids:
            return TestResult(
                name=name,
                passed=False,
                duration_ms=0,
                error="No captures available for testing",
            )

        try:
            # FM broadcast frequencies to switch between
            frequencies = [
                88_500_000,  # 88.5 MHz
                90_300_000,  # 90.3 MHz (KEXP)
                94_900_000,  # 94.9 MHz
                97_300_000,  # 97.3 MHz
                101_500_000, # 101.5 MHz
            ]

            errors = []
            for iteration in range(iterations):
                logger.info(f"    Reconfiguration iteration {iteration + 1}/{iterations}")

                # Reconfigure all captures with different frequencies
                tasks = []
                for i, cid in enumerate(self.capture_ids):
                    freq = frequencies[(iteration + i) % len(frequencies)]
                    gain = 20 + (iteration % 4) * 5  # Vary gain 20-35
                    tasks.append(self.reconfigure_capture(cid, center_hz=freq, gain=gain))

                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        errors.append(f"Iteration {iteration}, capture {self.capture_ids[i]}: {result}")

                # Brief pause and verify still running (longer due to 1s cooldown)
                await asyncio.sleep(1.5)
                for cid in self.capture_ids:
                    status = await self.get_capture_status(cid)
                    if status.get("state") != "running":
                        errors.append(f"Capture {cid} not running after reconfig: {status.get('state')}")

            if errors:
                return TestResult(
                    name=name,
                    passed=False,
                    duration_ms=(time.time() - start) * 1000,
                    error=f"{len(errors)} errors: {errors[0]}{'...' if len(errors) > 1 else ''}",
                )

            return TestResult(
                name=name,
                passed=True,
                duration_ms=(time.time() - start) * 1000,
                details={"iterations": iterations},
            )

        except Exception as e:
            return TestResult(
                name=name,
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                error=str(e),
            )

    async def test_stop_start_cycles(self, iterations: int = 3) -> TestResult:
        """Test 4: Stop/start cycles while other devices are streaming."""
        name = "Stop/Start Cycles"
        start = time.time()

        if len(self.capture_ids) < 2:
            return TestResult(
                name=name,
                passed=False,
                duration_ms=0,
                error="Need at least 2 captures for this test",
            )

        try:
            errors = []
            # Pick one capture to cycle, others stay running
            cycle_cid = self.capture_ids[0]
            other_cids = self.capture_ids[1:]

            for iteration in range(iterations):
                logger.info(f"    Stop/start cycle {iteration + 1}/{iterations}")

                # Stop the cycling capture
                await self.stop_capture(cycle_cid)
                await asyncio.sleep(0.5)

                # Verify it stopped and others still running
                status = await self.get_capture_status(cycle_cid)
                if status.get("state") not in ("stopped", "idle"):
                    errors.append(f"Cycle {iteration}: capture didn't stop properly: {status.get('status')}")

                for cid in other_cids:
                    status = await self.get_capture_status(cid)
                    if status.get("state") != "running":
                        errors.append(f"Cycle {iteration}: other capture {cid} stopped unexpectedly")

                # Restart the cycling capture
                await self.start_capture(cycle_cid)
                running = await self.wait_for_running(cycle_cid, timeout=60.0)
                if not running:
                    errors.append(f"Cycle {iteration}: capture didn't restart")

                # Brief pause for stability
                await asyncio.sleep(1.0)

            if errors:
                return TestResult(
                    name=name,
                    passed=False,
                    duration_ms=(time.time() - start) * 1000,
                    error=f"{len(errors)} errors: {errors[0]}{'...' if len(errors) > 1 else ''}",
                )

            return TestResult(
                name=name,
                passed=True,
                duration_ms=(time.time() - start) * 1000,
                details={"iterations": iterations},
            )

        except Exception as e:
            return TestResult(
                name=name,
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                error=str(e),
            )

    async def test_rapid_api_calls(self, iterations: int = 20) -> TestResult:
        """Test 5: Rapid fire API calls from multiple coroutines."""
        name = "Rapid API Calls"
        start = time.time()

        if not self.capture_ids:
            return TestResult(
                name=name,
                passed=False,
                duration_ms=0,
                error="No captures available for testing",
            )

        try:
            errors = []

            async def rapid_status_checks(cid: str, count: int):
                """Rapidly check status of a capture."""
                for _ in range(count):
                    try:
                        status = await self.get_capture_status(cid)
                        if status.get("state") != "running":
                            return f"Capture {cid} not running"
                    except Exception as e:
                        return f"Status check failed for {cid}: {e}"
                    await asyncio.sleep(0.05)  # 50ms between calls
                return None

            # Fire off rapid status checks for all captures concurrently
            logger.info(f"    Running {iterations} rapid API calls per capture...")
            tasks = [rapid_status_checks(cid, iterations) for cid in self.capture_ids]
            results = await asyncio.gather(*tasks)

            for result in results:
                if result:
                    errors.append(result)

            if errors:
                return TestResult(
                    name=name,
                    passed=False,
                    duration_ms=(time.time() - start) * 1000,
                    error=f"{len(errors)} errors: {errors[0]}{'...' if len(errors) > 1 else ''}",
                )

            return TestResult(
                name=name,
                passed=True,
                duration_ms=(time.time() - start) * 1000,
                details={"calls_per_capture": iterations, "total_calls": iterations * len(self.capture_ids)},
            )

        except Exception as e:
            return TestResult(
                name=name,
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                error=str(e),
            )

    async def test_stability_soak(self, duration_seconds: int = 30) -> TestResult:
        """Test 6: Long-running stability (continuous operation)."""
        name = f"Stability Soak ({duration_seconds}s)"
        start = time.time()

        if not self.capture_ids:
            return TestResult(
                name=name,
                passed=False,
                duration_ms=0,
                error="No captures available for testing",
            )

        try:
            errors = []
            check_interval = 2.0
            checks_performed = 0

            logger.info(f"    Running stability soak for {duration_seconds} seconds...")
            end_time = time.time() + duration_seconds

            while time.time() < end_time:
                checks_performed += 1

                # Check all captures are still running
                for cid in self.capture_ids:
                    try:
                        status = await self.get_capture_status(cid)
                        state = status.get("state", "")
                        if state != "running":
                            errors.append(f"Check {checks_performed}: Capture {cid} state={state}")
                        elif status.get("error"):
                            errors.append(f"Check {checks_performed}: Capture {cid} has error: {status.get('error')}")
                    except Exception as e:
                        errors.append(f"Check {checks_performed}: Failed to get status for {cid}: {e}")

                # Random reconfiguration during soak
                if random.random() < 0.2:  # 20% chance each check
                    cid = random.choice(self.capture_ids)
                    freq = random.choice([88_500_000, 90_300_000, 94_900_000])
                    try:
                        await self.reconfigure_capture(cid, center_hz=freq)
                    except Exception as e:
                        errors.append(f"Check {checks_performed}: Reconfig failed: {e}")

                await asyncio.sleep(check_interval)

            if errors:
                return TestResult(
                    name=name,
                    passed=False,
                    duration_ms=(time.time() - start) * 1000,
                    error=f"{len(errors)} errors: {errors[0]}{'...' if len(errors) > 1 else ''}",
                )

            return TestResult(
                name=name,
                passed=True,
                duration_ms=(time.time() - start) * 1000,
                details={"checks_performed": checks_performed, "duration_actual": time.time() - start},
            )

        except Exception as e:
            return TestResult(
                name=name,
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                error=str(e),
            )

    async def cleanup(self) -> None:
        """Clean up all test captures."""
        logger.info("Cleaning up test captures...")
        for cid in self.capture_ids:
            try:
                await self.stop_capture(cid)
            except Exception:
                pass
            try:
                await self.delete_capture(cid)
                logger.info(f"  Deleted capture {cid}")
            except Exception as e:
                logger.warning(f"  Failed to delete capture {cid}: {e}")
        self.capture_ids = []

    async def run(self, iterations: int = 5, soak_duration: int = 30) -> bool:
        """Run all stress tests."""
        logger.info(f"\n{'='*60}")
        logger.info("SDRplay Stress Test")
        logger.info(f"Server: {self.base_url}")
        logger.info(f"{'='*60}\n")

        # Discover SDRplay devices
        logger.info("Discovering SDRplay devices...")
        try:
            self.sdrplay_devices = await self.get_sdrplay_devices()
        except Exception as e:
            logger.error(f"Failed to discover devices: {e}")
            return False

        if not self.sdrplay_devices:
            logger.error("No SDRplay devices found! Ensure devices are connected and server is running.")
            return False

        logger.info(f"Found {len(self.sdrplay_devices)} SDRplay device(s):")
        for d in self.sdrplay_devices:
            logger.info(f"  - {d['id']}: {d.get('label', d.get('hardware', 'Unknown'))}")

        if len(self.sdrplay_devices) < 2:
            logger.warning("Only one SDRplay device found. Some tests require 2+ devices.")

        try:
            # Test 1: Simultaneous Startup
            logger.info("\nTest 1: Simultaneous Startup")
            result = await self.test_simultaneous_startup()
            self.stats.record(result)

            if not result.passed:
                logger.error("Simultaneous startup failed - cannot continue tests")
                return False

            # Test 2: Rapid Antenna Switching
            logger.info("\nTest 2: Rapid Antenna Switching")
            result = await self.test_rapid_antenna_switching(iterations=iterations)
            self.stats.record(result)

            # Test 3: Concurrent Reconfiguration
            logger.info("\nTest 3: Concurrent Reconfiguration")
            result = await self.test_concurrent_reconfiguration(iterations=iterations)
            self.stats.record(result)

            # Test 4: Stop/Start Cycles
            logger.info("\nTest 4: Stop/Start Cycles")
            result = await self.test_stop_start_cycles(iterations=min(iterations, 3))
            self.stats.record(result)

            # Test 5: Rapid API Calls
            logger.info("\nTest 5: Rapid API Calls")
            result = await self.test_rapid_api_calls(iterations=iterations * 4)
            self.stats.record(result)

            # Test 6: Stability Soak
            logger.info("\nTest 6: Stability Soak")
            result = await self.test_stability_soak(duration_seconds=soak_duration)
            self.stats.record(result)

        finally:
            await self.cleanup()

        # Print summary
        print(self.stats.summary())

        return self.stats.failed == 0


async def main():
    parser = argparse.ArgumentParser(description="SDRplay stress test for WaveCap-SDR")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8087, help="Server port (default: 8087)")
    parser.add_argument("--iterations", type=int, default=5, help="Iterations per test (default: 5)")
    parser.add_argument("--soak", type=int, default=30, help="Stability soak duration in seconds (default: 30)")
    args = parser.parse_args()

    tester = SDRplayStressTester(args.host, args.port)
    success = await tester.run(iterations=args.iterations, soak_duration=args.soak)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
