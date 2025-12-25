#!/usr/bin/env python3
"""Simple antenna sweep using trunking system's control channel scanner.

Tests all available SDR devices, antennas, and gain settings to find
the best configuration for SA-GRN (413-415 MHz) reception.
"""

import json
import time
import requests
from dataclasses import dataclass
import yaml

API_BASE = "http://localhost:8087/api/v1"
CONFIG_PATH = "/Users/thw/Projects/WaveCap-SDR/backend/config/wavecapsdr.yaml"

# Test configurations
CONFIGS = [
    # RTL-SDR tests
    {"name": "RTL-SDR RX g20", "device": "driver=rtlsdr,serial=00000001", "antenna": "RX", "gain": 20, "sample_rate": 2400000, "bandwidth": None},
    {"name": "RTL-SDR RX g30", "device": "driver=rtlsdr,serial=00000001", "antenna": "RX", "gain": 30, "sample_rate": 2400000, "bandwidth": None},
    {"name": "RTL-SDR RX g40", "device": "driver=rtlsdr,serial=00000001", "antenna": "RX", "gain": 40, "sample_rate": 2400000, "bandwidth": None},
    {"name": "RTL-SDR RX g50", "device": "driver=rtlsdr,serial=00000001", "antenna": "RX", "gain": 49.6, "sample_rate": 2400000, "bandwidth": None},

    # SDRplay E670 - Antenna A
    {"name": "E670 Ant-A g0", "device": "driver=sdrplay,serial=240305E670", "antenna": "Antenna A", "gain": 0, "sample_rate": 6000000, "bandwidth": 5000000},
    {"name": "E670 Ant-A g3", "device": "driver=sdrplay,serial=240305E670", "antenna": "Antenna A", "gain": 3, "sample_rate": 6000000, "bandwidth": 5000000},
    {"name": "E670 Ant-A g5", "device": "driver=sdrplay,serial=240305E670", "antenna": "Antenna A", "gain": 5, "sample_rate": 6000000, "bandwidth": 5000000},
    {"name": "E670 Ant-A g7", "device": "driver=sdrplay,serial=240305E670", "antenna": "Antenna A", "gain": 7, "sample_rate": 6000000, "bandwidth": 5000000},
    {"name": "E670 Ant-A g10", "device": "driver=sdrplay,serial=240305E670", "antenna": "Antenna A", "gain": 10, "sample_rate": 6000000, "bandwidth": 5000000},

    # SDRplay E670 - Antenna B
    {"name": "E670 Ant-B g0", "device": "driver=sdrplay,serial=240305E670", "antenna": "Antenna B", "gain": 0, "sample_rate": 6000000, "bandwidth": 5000000},
    {"name": "E670 Ant-B g3", "device": "driver=sdrplay,serial=240305E670", "antenna": "Antenna B", "gain": 3, "sample_rate": 6000000, "bandwidth": 5000000},
    {"name": "E670 Ant-B g5", "device": "driver=sdrplay,serial=240305E670", "antenna": "Antenna B", "gain": 5, "sample_rate": 6000000, "bandwidth": 5000000},
    {"name": "E670 Ant-B g7", "device": "driver=sdrplay,serial=240305E670", "antenna": "Antenna B", "gain": 7, "sample_rate": 6000000, "bandwidth": 5000000},
    {"name": "E670 Ant-B g10", "device": "driver=sdrplay,serial=240305E670", "antenna": "Antenna B", "gain": 10, "sample_rate": 6000000, "bandwidth": 5000000},

    # SDRplay E670 - Antenna C
    {"name": "E670 Ant-C g0", "device": "driver=sdrplay,serial=240305E670", "antenna": "Antenna C", "gain": 0, "sample_rate": 6000000, "bandwidth": 5000000},
    {"name": "E670 Ant-C g3", "device": "driver=sdrplay,serial=240305E670", "antenna": "Antenna C", "gain": 3, "sample_rate": 6000000, "bandwidth": 5000000},
    {"name": "E670 Ant-C g5", "device": "driver=sdrplay,serial=240305E670", "antenna": "Antenna C", "gain": 5, "sample_rate": 6000000, "bandwidth": 5000000},
    {"name": "E670 Ant-C g7", "device": "driver=sdrplay,serial=240305E670", "antenna": "Antenna C", "gain": 7, "sample_rate": 6000000, "bandwidth": 5000000},
    {"name": "E670 Ant-C g10", "device": "driver=sdrplay,serial=240305E670", "antenna": "Antenna C", "gain": 10, "sample_rate": 6000000, "bandwidth": 5000000},

    # SDRplay F070 - Antenna A
    {"name": "F070 Ant-A g0", "device": "driver=sdrplay,serial=240309F070", "antenna": "Antenna A", "gain": 0, "sample_rate": 6000000, "bandwidth": 5000000},
    {"name": "F070 Ant-A g3", "device": "driver=sdrplay,serial=240309F070", "antenna": "Antenna A", "gain": 3, "sample_rate": 6000000, "bandwidth": 5000000},
    {"name": "F070 Ant-A g5", "device": "driver=sdrplay,serial=240309F070", "antenna": "Antenna A", "gain": 5, "sample_rate": 6000000, "bandwidth": 5000000},
    {"name": "F070 Ant-A g7", "device": "driver=sdrplay,serial=240309F070", "antenna": "Antenna A", "gain": 7, "sample_rate": 6000000, "bandwidth": 5000000},
    {"name": "F070 Ant-A g10", "device": "driver=sdrplay,serial=240309F070", "antenna": "Antenna A", "gain": 10, "sample_rate": 6000000, "bandwidth": 5000000},

    # SDRplay F070 - Antenna B
    {"name": "F070 Ant-B g0", "device": "driver=sdrplay,serial=240309F070", "antenna": "Antenna B", "gain": 0, "sample_rate": 6000000, "bandwidth": 5000000},
    {"name": "F070 Ant-B g3", "device": "driver=sdrplay,serial=240309F070", "antenna": "Antenna B", "gain": 3, "sample_rate": 6000000, "bandwidth": 5000000},
    {"name": "F070 Ant-B g5", "device": "driver=sdrplay,serial=240309F070", "antenna": "Antenna B", "gain": 5, "sample_rate": 6000000, "bandwidth": 5000000},
    {"name": "F070 Ant-B g7", "device": "driver=sdrplay,serial=240309F070", "antenna": "Antenna B", "gain": 7, "sample_rate": 6000000, "bandwidth": 5000000},
    {"name": "F070 Ant-B g10", "device": "driver=sdrplay,serial=240309F070", "antenna": "Antenna B", "gain": 10, "sample_rate": 6000000, "bandwidth": 5000000},

    # SDRplay F070 - Antenna C
    {"name": "F070 Ant-C g0", "device": "driver=sdrplay,serial=240309F070", "antenna": "Antenna C", "gain": 0, "sample_rate": 6000000, "bandwidth": 5000000},
    {"name": "F070 Ant-C g3", "device": "driver=sdrplay,serial=240309F070", "antenna": "Antenna C", "gain": 3, "sample_rate": 6000000, "bandwidth": 5000000},
    {"name": "F070 Ant-C g5", "device": "driver=sdrplay,serial=240309F070", "antenna": "Antenna C", "gain": 5, "sample_rate": 6000000, "bandwidth": 5000000},
    {"name": "F070 Ant-C g7", "device": "driver=sdrplay,serial=240309F070", "antenna": "Antenna C", "gain": 7, "sample_rate": 6000000, "bandwidth": 5000000},
    {"name": "F070 Ant-C g10", "device": "driver=sdrplay,serial=240309F070", "antenna": "Antenna C", "gain": 10, "sample_rate": 6000000, "bandwidth": 5000000},
]


@dataclass
class SweepResult:
    name: str
    snr_db: float
    power_db: float
    sync_detected: bool
    tsbk_count: int
    error: str | None = None


def update_config(device: str, antenna: str, gain: float, sample_rate: int, bandwidth: float | None) -> bool:
    """Update the wavecapsdr.yaml config for SA-GRN system."""
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)

        # Update SA-GRN trunking config
        sa_grn = config.get('trunking', {}).get('systems', {}).get('sa_grn', {})
        sa_grn['device_id'] = device
        sa_grn['antenna'] = antenna
        sa_grn['gain'] = gain
        sa_grn['sample_rate'] = sample_rate
        if bandwidth:
            sa_grn['bandwidth'] = bandwidth
        else:
            sa_grn.pop('bandwidth', None)

        # Make sure auto_start is off
        sa_grn['auto_start'] = False

        config['trunking']['systems']['sa_grn'] = sa_grn

        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        return True
    except Exception as e:
        print(f"  Config update failed: {e}")
        return False


def test_config(name: str, timeout_sec: int = 15) -> SweepResult:
    """Test the current config by starting the trunking system and measuring SNR."""

    # Stop any running system
    requests.post(f"{API_BASE}/trunking/systems/sa_grn/stop", timeout=5)
    time.sleep(1)

    # Start the system
    resp = requests.post(f"{API_BASE}/trunking/systems/sa_grn/start", timeout=30)
    if resp.status_code != 200:
        return SweepResult(name=name, snr_db=-100, power_db=-100, sync_detected=False, tsbk_count=0,
                          error=f"Start failed: {resp.status_code}")

    # Wait for control channel scanner to complete
    time.sleep(timeout_sec)

    # Get system status
    resp = requests.get(f"{API_BASE}/trunking/systems/sa_grn", timeout=10)
    if resp.status_code != 200:
        return SweepResult(name=name, snr_db=-100, power_db=-100, sync_detected=False, tsbk_count=0,
                          error="Failed to get status")

    data = resp.json()
    stats = data.get("stats", {})
    cc_scanner = stats.get("cc_scanner", {})
    control_monitor = stats.get("control_monitor", {})

    # Find best SNR across all measured channels
    measurements = cc_scanner.get("measurements", {})
    best_snr = -100
    best_power = -100
    sync_detected = False

    for channel_name, meas in measurements.items():
        snr = meas.get("snr_db", -100)
        power = meas.get("power_db", -100)
        sync = meas.get("sync_detected", False)

        if snr > best_snr:
            best_snr = snr
            best_power = power
        if sync:
            sync_detected = True

    tsbk_count = stats.get("tsbk_count", 0)
    control_sync = control_monitor.get("sync_state", "unknown") == "synced"

    # Stop system
    requests.post(f"{API_BASE}/trunking/systems/sa_grn/stop", timeout=5)

    return SweepResult(
        name=name,
        snr_db=best_snr,
        power_db=best_power,
        sync_detected=sync_detected or control_sync,
        tsbk_count=tsbk_count
    )


def run_sweep() -> list[SweepResult]:
    """Run the antenna/gain sweep."""
    results = []

    print("=" * 70)
    print("SA-GRN Antenna/Gain Sweep")
    print("Testing", len(CONFIGS), "configurations")
    print("=" * 70)

    for i, cfg in enumerate(CONFIGS, 1):
        name = cfg["name"]
        print(f"\n[{i}/{len(CONFIGS)}] {name}: ", end="", flush=True)

        # Update config file
        if not update_config(cfg["device"], cfg["antenna"], cfg["gain"],
                           cfg["sample_rate"], cfg.get("bandwidth")):
            result = SweepResult(name=name, snr_db=-100, power_db=-100,
                               sync_detected=False, tsbk_count=0, error="Config update failed")
            results.append(result)
            print("FAILED (config)")
            continue

        # Test this config
        result = test_config(name)
        results.append(result)

        if result.error:
            print(f"FAILED ({result.error[:30]})")
        else:
            sync_status = "SYNC" if result.sync_detected else "no sync"
            tsbk_status = f"TSBK={result.tsbk_count}" if result.tsbk_count > 0 else ""
            quality = "EXCELLENT" if result.snr_db > 20 else "GOOD" if result.snr_db > 15 else "OK" if result.snr_db > 10 else "WEAK" if result.snr_db > 5 else "POOR"
            print(f"SNR={result.snr_db:5.1f}dB Power={result.power_db:5.1f}dB [{quality}] {sync_status} {tsbk_status}")

    return results


def print_summary(results: list[SweepResult]):
    """Print summary of results."""
    print("\n" + "=" * 70)
    print("SUMMARY - Best configurations for SA-GRN")
    print("=" * 70)

    # Filter out errors
    valid_results = [r for r in results if r.error is None]

    if not valid_results:
        print("No valid results!")
        return

    # Sort by SNR
    valid_results.sort(key=lambda r: r.snr_db, reverse=True)

    print("\nTop 10 configurations by SNR:")
    print("-" * 70)
    print(f"{'Rank':<5} {'Configuration':<25} {'SNR':>8} {'Power':>8} {'Sync':<6} {'TSBK':<6}")
    print("-" * 70)

    for i, r in enumerate(valid_results[:10], 1):
        sync = "YES" if r.sync_detected else "no"
        tsbk = str(r.tsbk_count) if r.tsbk_count > 0 else "-"
        print(f"{i:<5} {r.name:<25} {r.snr_db:>7.1f}dB {r.power_db:>7.1f}dB {sync:<6} {tsbk:<6}")

    # Best with TSBK decode
    tsbk_results = [r for r in valid_results if r.tsbk_count > 0]
    if tsbk_results:
        best_tsbk = max(tsbk_results, key=lambda r: r.tsbk_count)
        print("\n" + "-" * 70)
        print(f"BEST FOR TSBK DECODE: {best_tsbk.name}")
        print(f"  SNR:  {best_tsbk.snr_db:.1f} dB")
        print(f"  TSBK: {best_tsbk.tsbk_count} decoded")

    # Best SNR
    best = valid_results[0]
    print("\n" + "=" * 70)
    print("RECOMMENDED CONFIGURATION (highest SNR):")
    print(f"  Config: {best.name}")
    print(f"  SNR:    {best.snr_db:.1f} dB")
    print(f"  Power:  {best.power_db:.1f} dB")
    print(f"  Sync:   {'YES' if best.sync_detected else 'NO'}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        results = run_sweep()
        print_summary(results)
    except KeyboardInterrupt:
        print("\n\nSweep interrupted by user")
        # Stop trunking system
        requests.post(f"{API_BASE}/trunking/systems/sa_grn/stop", timeout=5)
    except Exception as e:
        print(f"\nSweep failed: {e}")
        import traceback
        traceback.print_exc()
