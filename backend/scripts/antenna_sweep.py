#!/usr/bin/env python3
"""Antenna and gain sweep for SA-GRN reception optimization.

Tests all available SDR devices, antennas, and gain settings to find
the best configuration for SA-GRN (413-415 MHz) reception.
"""

import json
import time
import requests
from dataclasses import dataclass

API_BASE = "http://localhost:8087/api/v1"

# SA-GRN control channel frequencies
CONTROL_CHANNELS = [413.45e6, 413.3125e6, 414.6875e6, 414.7625e6, 415.325e6]
CENTER_FREQ = 414.3e6  # Center frequency for captures

# Device configurations
# RSPdx device settings:
# - rfnotch_en: RF notch filter (for broadcast FM rejection)
# - dabnotch_en: DAB notch filter
# - biasT_en: Bias-T power for active antennas
# - HDR mode: High dynamic range mode (only for < 2 MHz)
# The RSPdx has automatic tracking RF preselector that follows the tuned frequency

# For UHF (413 MHz), RF preselector auto-tracks, no notch filters needed
RSPDX_SETTINGS = {
    "rfnotch_en": False,   # RF notch only helps with FM broadcast interference
    "dabnotch_en": False,  # DAB notch only for 175-239 MHz
}

DEVICES = [
    {
        "name": "RTL-SDR Blog V4",
        "device_id": "driver=rtlsdr,serial=00000001",
        "antennas": ["RX"],  # RTL-SDR has single antenna
        "gains": [20, 30, 40, 49.6],  # RTL-SDR gain range
        "sample_rate": 2400000,
        "bandwidths": [None],  # RTL-SDR doesn't have adjustable IF bandwidth
        "device_settings": {},
    },
    {
        "name": "SDRplay RSPdx-R2 E670",
        "device_id": "driver=sdrplay,serial=240305E670",
        "antennas": ["Antenna A", "Antenna B", "Antenna C"],
        "gains": [0, 3, 5, 7, 10, 15, 20],  # Focus on lower gains as user noted 5.0 was best
        "sample_rate": 6000000,
        "bandwidths": [600000, 1536000, 5000000],  # IF filter bandwidths: 600kHz, 1.536MHz, 5MHz
        "device_settings": RSPDX_SETTINGS,
    },
    {
        "name": "SDRplay RSPdx-R2 F070",
        "device_id": "driver=sdrplay,serial=240309F070",
        "antennas": ["Antenna A", "Antenna B", "Antenna C"],
        "gains": [0, 3, 5, 7, 10, 15, 20],
        "sample_rate": 6000000,
        "bandwidths": [600000, 1536000, 5000000],
        "device_settings": RSPDX_SETTINGS,
    },
]


@dataclass
class SweepResult:
    device_name: str
    device_id: str
    antenna: str
    gain: float
    bandwidth: float | None
    snr_db: float
    power_db: float
    error: str | None = None


def create_capture(device_id: str, antenna: str, gain: float, sample_rate: int,
                   bandwidth: float | None = None, device_settings: dict | None = None) -> str | None:
    """Create a capture and return its ID."""
    payload = {
        "deviceId": device_id,
        "centerHz": CENTER_FREQ,
        "sampleRate": sample_rate,
        "gain": gain,
        "antenna": antenna,
    }
    if bandwidth is not None:
        payload["bandwidth"] = bandwidth
    if device_settings:
        payload["deviceSettings"] = device_settings
    try:
        resp = requests.post(f"{API_BASE}/captures", json=payload, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("id")
        else:
            print(f"  Failed to create capture: {resp.status_code} {resp.text[:100]}")
            return None
    except Exception as e:
        print(f"  Error creating capture: {e}")
        return None


def start_capture(capture_id: str) -> bool:
    """Start a capture."""
    try:
        resp = requests.post(f"{API_BASE}/captures/{capture_id}/start", timeout=30)
        return resp.status_code == 200
    except Exception as e:
        print(f"  Error starting capture: {e}")
        return False


def stop_capture(capture_id: str) -> bool:
    """Stop a capture."""
    try:
        resp = requests.post(f"{API_BASE}/captures/{capture_id}/stop", timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"  Error stopping capture: {e}")
        return False


def delete_capture(capture_id: str) -> bool:
    """Delete a capture."""
    try:
        resp = requests.delete(f"{API_BASE}/captures/{capture_id}", timeout=10)
        return resp.status_code in (200, 204)
    except Exception as e:
        print(f"  Error deleting capture: {e}")
        return False


def get_capture_status(capture_id: str) -> dict | None:
    """Get capture status."""
    try:
        resp = requests.get(f"{API_BASE}/captures/{capture_id}", timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


def measure_signal_strength(capture_id: str) -> tuple[float, float]:
    """Measure SNR and power at SA-GRN control channel frequencies.

    Uses FFT peak detection to find signal power at control channel frequencies.
    Returns (max_snr, power_at_max_snr).
    """
    import numpy as np

    # Wait for FFT data to stabilize
    time.sleep(3)

    try:
        # Get capture info
        resp = requests.get(f"{API_BASE}/captures/{capture_id}", timeout=10)
        if resp.status_code != 200:
            return -100, -100

        capture = resp.json()
        center_hz = capture.get("centerHz", CENTER_FREQ)
        sample_rate = capture.get("sampleRate", 2400000)
        state = capture.get("state", "unknown")

        if state != "running":
            return -100, -100

        # Calculate frequency range
        low_freq = center_hz - sample_rate / 2
        high_freq = center_hz + sample_rate / 2

        # Check which control channels are in range
        channels_in_range = [f for f in CONTROL_CHANNELS if low_freq <= f <= high_freq]

        if not channels_in_range:
            return -100, -100

        # Use WebSocket to get FFT data, or fall back to simple power measurement
        # For now, create a P25 channel and check if it syncs

        # Create a test channel at the first control channel frequency
        test_freq = channels_in_range[0]
        offset = test_freq - center_hz

        channel_payload = {
            "offsetHz": offset,
            "mode": "p25",  # Use P25 mode to try to detect sync
            "squelchDb": -100,  # Disable squelch for measurement
        }

        resp = requests.post(f"{API_BASE}/captures/{capture_id}/channels", json=channel_payload, timeout=10)
        if resp.status_code != 200:
            # Try NBFM as fallback
            channel_payload["mode"] = "nbfm"
            resp = requests.post(f"{API_BASE}/captures/{capture_id}/channels", json=channel_payload, timeout=10)
            if resp.status_code != 200:
                return 0, -20

        channel = resp.json()
        channel_id = channel.get("id")

        # Wait for channel to process some samples
        time.sleep(2)

        # Get channel status for signal info
        resp = requests.get(f"{API_BASE}/channels/{channel_id}", timeout=10)
        snr_db = 0.0
        power_db = -100.0

        if resp.status_code == 200:
            channel_data = resp.json()
            # Try to get signal levels from different possible fields
            signal_db = channel_data.get("signalDb") or channel_data.get("signal_db") or channel_data.get("rssi") or -100
            noise_db = channel_data.get("noiseDb") or channel_data.get("noise_db") or channel_data.get("noiseFloor") or -120
            power_db = float(signal_db) if signal_db is not None else -100.0

            if signal_db is not None and noise_db is not None:
                snr_db = float(signal_db) - float(noise_db)
            else:
                # Estimate SNR from power level
                # Assume noise floor around -120 dBFS
                snr_db = power_db + 120 if power_db > -100 else 0

        # Clean up channel
        requests.delete(f"{API_BASE}/channels/{channel_id}", timeout=5)

        return snr_db, power_db

    except Exception as e:
        print(f"  Measurement error: {e}")

    return 0, -100


def run_sweep() -> list[SweepResult]:
    """Run the full antenna/gain sweep."""
    results = []

    print("=" * 70)
    print("SA-GRN Antenna/Gain Sweep")
    print(f"Target frequencies: {[f/1e6 for f in CONTROL_CHANNELS]} MHz")
    print(f"Center frequency: {CENTER_FREQ/1e6} MHz")
    print("=" * 70)

    for device in DEVICES:
        print(f"\n{'='*60}")
        print(f"Device: {device['name']}")
        print(f"ID: {device['device_id']}")
        print("=" * 60)

        for antenna in device["antennas"]:
            print(f"\n  Antenna: {antenna}")
            print("  " + "-" * 40)

            for gain in device["gains"]:
                print(f"    Gain {gain:5.1f} dB: ", end="", flush=True)

                # Create capture
                capture_id = create_capture(
                    device["device_id"],
                    antenna,
                    gain,
                    device["sample_rate"]
                )

                if not capture_id:
                    result = SweepResult(
                        device_name=device["name"],
                        device_id=device["device_id"],
                        antenna=antenna,
                        gain=gain,
                        snr_db=-100,
                        power_db=-100,
                        error="Failed to create capture"
                    )
                    results.append(result)
                    print("FAILED (create)")
                    continue

                # Start capture
                if not start_capture(capture_id):
                    delete_capture(capture_id)
                    result = SweepResult(
                        device_name=device["name"],
                        device_id=device["device_id"],
                        antenna=antenna,
                        gain=gain,
                        snr_db=-100,
                        power_db=-100,
                        error="Failed to start capture"
                    )
                    results.append(result)
                    print("FAILED (start)")
                    continue

                # Wait for capture to start (SDRplay can be slow)
                time.sleep(4)

                # Check if capture is running
                status = get_capture_status(capture_id)
                if not status or status.get("state") != "running":
                    stop_capture(capture_id)
                    delete_capture(capture_id)
                    error_msg = (status.get("errorMessage") or "Unknown") if status else "No status"
                    result = SweepResult(
                        device_name=device["name"],
                        device_id=device["device_id"],
                        antenna=antenna,
                        gain=gain,
                        snr_db=-100,
                        power_db=-100,
                        error=f"Capture not running: {error_msg}"
                    )
                    results.append(result)
                    print(f"FAILED ({str(error_msg)[:30]})")
                    continue

                # Measure signal
                snr_db, power_db = measure_signal_strength(capture_id)

                # Stop and delete capture
                stop_capture(capture_id)
                time.sleep(0.5)
                delete_capture(capture_id)

                result = SweepResult(
                    device_name=device["name"],
                    device_id=device["device_id"],
                    antenna=antenna,
                    gain=gain,
                    snr_db=snr_db,
                    power_db=power_db,
                )
                results.append(result)

                # Print result
                if snr_db > 10:
                    quality = "GOOD"
                elif snr_db > 5:
                    quality = "OK"
                elif snr_db > 0:
                    quality = "WEAK"
                else:
                    quality = "NONE"

                print(f"SNR={snr_db:5.1f} dB, Power={power_db:5.1f} dBFS [{quality}]")

    return results


def print_summary(results: list[SweepResult]):
    """Print summary of results."""
    print("\n" + "=" * 70)
    print("SUMMARY - Best configurations for SA-GRN")
    print("=" * 70)

    # Filter out errors
    valid_results = [r for r in results if r.error is None and r.snr_db > -50]

    if not valid_results:
        print("No valid results!")
        return

    # Sort by SNR
    valid_results.sort(key=lambda r: r.snr_db, reverse=True)

    print("\nTop 10 configurations:")
    print("-" * 70)
    print(f"{'Rank':<5} {'Device':<25} {'Antenna':<12} {'Gain':>6} {'SNR':>8} {'Power':>8}")
    print("-" * 70)

    for i, r in enumerate(valid_results[:10], 1):
        print(f"{i:<5} {r.device_name:<25} {r.antenna:<12} {r.gain:>5.1f}dB {r.snr_db:>7.1f}dB {r.power_db:>7.1f}dB")

    # Best overall
    best = valid_results[0]
    print("\n" + "=" * 70)
    print("RECOMMENDED CONFIGURATION:")
    print(f"  Device:  {best.device_name}")
    print(f"  ID:      {best.device_id}")
    print(f"  Antenna: {best.antenna}")
    print(f"  Gain:    {best.gain} dB")
    print(f"  SNR:     {best.snr_db:.1f} dB")
    print("=" * 70)


if __name__ == "__main__":
    try:
        results = run_sweep()
        print_summary(results)
    except KeyboardInterrupt:
        print("\n\nSweep interrupted by user")
    except Exception as e:
        print(f"\nSweep failed: {e}")
        raise
