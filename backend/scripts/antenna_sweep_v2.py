#!/usr/bin/env python3
"""Enhanced antenna sweep with RSPdx settings and CSV output.

Tests all available SDR devices, antennas, gain settings, IF bandwidths,
and RF notch filter configurations to find the best setup for SA-GRN reception.
"""

import csv
import json
import time
import requests
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import yaml

API_BASE = "http://localhost:8087/api/v1"
CONFIG_PATH = "/Users/thw/Projects/WaveCap-SDR/backend/config/wavecapsdr.yaml"

# Device IDs
RTL_SDR = "driver=rtlsdr,serial=00000001"
SDRPLAY_F070 = "driver=sdrplay,serial=240309F070"
SDRPLAY_E670 = "driver=sdrplay,serial=240305E670"


@dataclass
class SweepResult:
    timestamp: str
    name: str
    device: str
    antenna: str
    gain: float
    bandwidth_hz: int | None
    rf_notch: bool | None
    sample_rate: int
    snr_db: float
    power_db: float
    sync_detected: bool
    tsbk_count: int
    frames_decoded: int
    sync_losses: int
    error: str | None = None


def generate_configs() -> list[dict]:
    """Generate all test configurations."""
    configs = []

    # RTL-SDR tests (no special settings, just gain)
    for gain in [30, 40, 49.6]:
        configs.append({
            "name": f"RTL-SDR_RX_g{int(gain)}",
            "device": RTL_SDR,
            "antenna": "RX",
            "gain": gain,
            "sample_rate": 2400000,
            "bandwidth": None,
            "rf_notch": None,
            "device_settings": {},
        })

    # SDRplay F070 - Antenna A and B (best performers from prior sweep)
    for antenna in ["Antenna A", "Antenna B"]:
        ant_short = "AntA" if antenna == "Antenna A" else "AntB"
        for gain in [5, 7, 10]:
            for bw in [600000, 1536000, 5000000]:
                bw_short = f"{bw//1000}k"
                for rf_notch in [False, True]:
                    notch_short = "notch1" if rf_notch else "notch0"
                    configs.append({
                        "name": f"F070_{ant_short}_g{gain}_bw{bw_short}_{notch_short}",
                        "device": SDRPLAY_F070,
                        "antenna": antenna,
                        "gain": gain,
                        "sample_rate": 6000000,
                        "bandwidth": bw,
                        "rf_notch": rf_notch,
                        "device_settings": {
                            "rfnotch_en": "true" if rf_notch else "false",
                        },
                    })

    # SDRplay E670 - Antenna A only (best performer from prior sweep)
    for gain in [5, 7, 10]:
        for bw in [600000, 1536000, 5000000]:
            bw_short = f"{bw//1000}k"
            for rf_notch in [False, True]:
                notch_short = "notch1" if rf_notch else "notch0"
                configs.append({
                    "name": f"E670_AntA_g{gain}_bw{bw_short}_{notch_short}",
                    "device": SDRPLAY_E670,
                    "antenna": "Antenna A",
                    "gain": gain,
                    "sample_rate": 6000000,
                    "bandwidth": bw,
                    "rf_notch": rf_notch,
                    "device_settings": {
                        "rfnotch_en": "true" if rf_notch else "false",
                    },
                })

    return configs


def update_config(device: str, antenna: str, gain: float, sample_rate: int,
                  bandwidth: int | None, device_settings: dict) -> bool:
    """Update the wavecapsdr.yaml config for SA-GRN system."""
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)

        sa_grn = config.get('trunking', {}).get('systems', {}).get('sa_grn', {})
        sa_grn['device_id'] = device
        sa_grn['antenna'] = antenna
        sa_grn['gain'] = gain
        sa_grn['sample_rate'] = sample_rate
        sa_grn['auto_start'] = False

        if bandwidth:
            sa_grn['bandwidth'] = bandwidth
        else:
            sa_grn.pop('bandwidth', None)

        if device_settings:
            sa_grn['device_settings'] = device_settings
        else:
            sa_grn.pop('device_settings', None)

        config['trunking']['systems']['sa_grn'] = sa_grn

        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        return True
    except Exception as e:
        print(f"  Config update failed: {e}")
        return False


def test_config(cfg: dict, timeout_sec: int = 10) -> SweepResult:
    """Test a configuration by starting the trunking system and measuring metrics."""
    name = cfg["name"]
    timestamp = datetime.now().isoformat()

    # Stop any running system
    try:
        requests.post(f"{API_BASE}/trunking/systems/sa_grn/stop", timeout=5)
    except Exception:
        pass
    time.sleep(1)

    # Start the system
    try:
        resp = requests.post(f"{API_BASE}/trunking/systems/sa_grn/start", timeout=30)
        if resp.status_code != 200:
            return SweepResult(
                timestamp=timestamp, name=name, device=cfg["device"], antenna=cfg["antenna"],
                gain=cfg["gain"], bandwidth_hz=cfg.get("bandwidth"), rf_notch=cfg.get("rf_notch"),
                sample_rate=cfg["sample_rate"], snr_db=-100, power_db=-100, sync_detected=False,
                tsbk_count=0, frames_decoded=0, sync_losses=0, error=f"Start failed: {resp.status_code}"
            )
    except Exception as e:
        return SweepResult(
            timestamp=timestamp, name=name, device=cfg["device"], antenna=cfg["antenna"],
            gain=cfg["gain"], bandwidth_hz=cfg.get("bandwidth"), rf_notch=cfg.get("rf_notch"),
            sample_rate=cfg["sample_rate"], snr_db=-100, power_db=-100, sync_detected=False,
            tsbk_count=0, frames_decoded=0, sync_losses=0, error=f"Start exception: {e}"
        )

    # Wait for measurements
    time.sleep(timeout_sec)

    # Get system status
    try:
        resp = requests.get(f"{API_BASE}/trunking/systems/sa_grn", timeout=10)
        if resp.status_code != 200:
            return SweepResult(
                timestamp=timestamp, name=name, device=cfg["device"], antenna=cfg["antenna"],
                gain=cfg["gain"], bandwidth_hz=cfg.get("bandwidth"), rf_notch=cfg.get("rf_notch"),
                sample_rate=cfg["sample_rate"], snr_db=-100, power_db=-100, sync_detected=False,
                tsbk_count=0, frames_decoded=0, sync_losses=0, error="Failed to get status"
            )

        data = resp.json()
    except Exception as e:
        return SweepResult(
            timestamp=timestamp, name=name, device=cfg["device"], antenna=cfg["antenna"],
            gain=cfg["gain"], bandwidth_hz=cfg.get("bandwidth"), rf_notch=cfg.get("rf_notch"),
            sample_rate=cfg["sample_rate"], snr_db=-100, power_db=-100, sync_detected=False,
            tsbk_count=0, frames_decoded=0, sync_losses=0, error=f"Status exception: {e}"
        )

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
    frames_decoded = control_monitor.get("frames_decoded", 0)
    sync_losses = control_monitor.get("sync_losses", 0)
    control_sync = control_monitor.get("sync_state", "unknown") == "synced"

    # Stop system
    try:
        requests.post(f"{API_BASE}/trunking/systems/sa_grn/stop", timeout=5)
    except Exception:
        pass

    return SweepResult(
        timestamp=timestamp,
        name=name,
        device=cfg["device"],
        antenna=cfg["antenna"],
        gain=cfg["gain"],
        bandwidth_hz=cfg.get("bandwidth"),
        rf_notch=cfg.get("rf_notch"),
        sample_rate=cfg["sample_rate"],
        snr_db=round(best_snr, 2),
        power_db=round(best_power, 2),
        sync_detected=sync_detected or control_sync,
        tsbk_count=tsbk_count,
        frames_decoded=frames_decoded,
        sync_losses=sync_losses,
    )


def save_results_csv(results: list[SweepResult], filepath: str):
    """Save sweep results to CSV file."""
    with open(filepath, 'w', newline='') as f:
        if not results:
            return

        fieldnames = list(asdict(results[0]).keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            writer.writerow(asdict(r))

    print(f"\nResults saved to: {filepath}")


def run_sweep() -> list[SweepResult]:
    """Run the antenna/gain/settings sweep."""
    configs = generate_configs()
    results = []

    print("=" * 80)
    print("SA-GRN Enhanced Antenna Sweep")
    print(f"Testing {len(configs)} configurations")
    print("Parameters: device, antenna, gain, IF bandwidth, RF notch")
    print("=" * 80)

    for i, cfg in enumerate(configs, 1):
        name = cfg["name"]
        print(f"\n[{i}/{len(configs)}] {name}: ", end="", flush=True)

        # Update config file
        if not update_config(
            cfg["device"], cfg["antenna"], cfg["gain"],
            cfg["sample_rate"], cfg.get("bandwidth"), cfg.get("device_settings", {})
        ):
            result = SweepResult(
                timestamp=datetime.now().isoformat(), name=name, device=cfg["device"],
                antenna=cfg["antenna"], gain=cfg["gain"], bandwidth_hz=cfg.get("bandwidth"),
                rf_notch=cfg.get("rf_notch"), sample_rate=cfg["sample_rate"],
                snr_db=-100, power_db=-100, sync_detected=False, tsbk_count=0,
                frames_decoded=0, sync_losses=0, error="Config update failed"
            )
            results.append(result)
            print("FAILED (config)")
            continue

        # Test this config
        result = test_config(cfg)
        results.append(result)

        if result.error:
            print(f"FAILED ({result.error[:30]})")
        else:
            sync_status = "SYNC" if result.sync_detected else "no sync"
            tsbk_status = f"TSBK={result.tsbk_count}" if result.tsbk_count > 0 else ""
            frames_status = f"frames={result.frames_decoded}" if result.frames_decoded > 0 else ""

            if result.snr_db > 20:
                quality = "EXCELLENT"
            elif result.snr_db > 15:
                quality = "GOOD"
            elif result.snr_db > 10:
                quality = "OK"
            elif result.snr_db > 5:
                quality = "WEAK"
            else:
                quality = "POOR"

            print(f"SNR={result.snr_db:5.1f}dB [{quality}] {sync_status} {frames_status} {tsbk_status}")

    return results


def print_summary(results: list[SweepResult]):
    """Print summary of results."""
    print("\n" + "=" * 80)
    print("SUMMARY - Best configurations for SA-GRN")
    print("=" * 80)

    # Filter out errors
    valid_results = [r for r in results if r.error is None]

    if not valid_results:
        print("No valid results!")
        return

    # Sort by SNR
    valid_results.sort(key=lambda r: r.snr_db, reverse=True)

    print("\nTop 15 configurations by SNR:")
    print("-" * 80)
    print(f"{'Rank':<4} {'Configuration':<35} {'SNR':>7} {'BW':>8} {'Notch':<6} {'Sync':<5} {'Frames':<7}")
    print("-" * 80)

    for i, r in enumerate(valid_results[:15], 1):
        bw_str = f"{r.bandwidth_hz//1000}k" if r.bandwidth_hz else "N/A"
        notch_str = "ON" if r.rf_notch else "OFF" if r.rf_notch is not None else "N/A"
        sync_str = "YES" if r.sync_detected else "no"
        print(f"{i:<4} {r.name:<35} {r.snr_db:>6.1f}dB {bw_str:>8} {notch_str:<6} {sync_str:<5} {r.frames_decoded:<7}")

    # Analyze bandwidth impact
    print("\n" + "-" * 80)
    print("Analysis by IF Bandwidth:")
    for bw in [600000, 1536000, 5000000]:
        bw_results = [r for r in valid_results if r.bandwidth_hz == bw]
        if bw_results:
            avg_snr = sum(r.snr_db for r in bw_results) / len(bw_results)
            print(f"  {bw//1000:>4}kHz: avg SNR = {avg_snr:.1f} dB ({len(bw_results)} samples)")

    # Analyze RF notch impact
    print("\nAnalysis by RF Notch:")
    notch_on = [r for r in valid_results if r.rf_notch is True]
    notch_off = [r for r in valid_results if r.rf_notch is False]
    if notch_on:
        avg_on = sum(r.snr_db for r in notch_on) / len(notch_on)
        print(f"  Notch ON:  avg SNR = {avg_on:.1f} dB ({len(notch_on)} samples)")
    if notch_off:
        avg_off = sum(r.snr_db for r in notch_off) / len(notch_off)
        print(f"  Notch OFF: avg SNR = {avg_off:.1f} dB ({len(notch_off)} samples)")

    # Best overall
    best = valid_results[0]
    print("\n" + "=" * 80)
    print("RECOMMENDED CONFIGURATION:")
    print(f"  Name:      {best.name}")
    print(f"  Device:    {best.device}")
    print(f"  Antenna:   {best.antenna}")
    print(f"  Gain:      {best.gain} dB")
    print(f"  Bandwidth: {best.bandwidth_hz//1000 if best.bandwidth_hz else 'N/A'} kHz")
    print(f"  RF Notch:  {'ON' if best.rf_notch else 'OFF' if best.rf_notch is not None else 'N/A'}")
    print(f"  SNR:       {best.snr_db:.1f} dB")
    print(f"  Frames:    {best.frames_decoded}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SA-GRN antenna sweep with RSPdx settings")
    parser.add_argument("--output", "-o", type=str, help="Output CSV file path")
    args = parser.parse_args()

    # Generate default output filename with timestamp
    if args.output:
        csv_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"antenna_sweep_{timestamp}.csv"

    try:
        results = run_sweep()
        print_summary(results)
        save_results_csv(results, csv_path)
    except KeyboardInterrupt:
        print("\n\nSweep interrupted by user")
        requests.post(f"{API_BASE}/trunking/systems/sa_grn/stop", timeout=5)
    except Exception as e:
        print(f"\nSweep failed: {e}")
        import traceback
        traceback.print_exc()
