from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import httpx
except Exception as e:  # pragma: no cover
    print("Missing dependency httpx. Install with: pip install httpx", file=sys.stderr)
    raise

try:
    import websockets
except Exception as e:  # pragma: no cover
    print("Missing dependency websockets. Install with: pip install websockets", file=sys.stderr)
    raise

import uvicorn

from .app import create_app
from .config import AppConfig, DeviceConfig, ServerConfig, StreamConfig, LimitsConfig


@dataclass
class HarnessReport:
    capture_id: str
    channel_reports: List[Dict[str, Any]]

    def to_json(self) -> str:
        return json.dumps({
            "captureId": self.capture_id,
            "channels": self.channel_reports,
        }, indent=2)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WaveCap-SDR test harness")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8087)
    p.add_argument("--token", default=None, help="Bearer token if server requires auth")
    p.add_argument("--start-server", action="store_true", help="Start an in-process server")
    p.add_argument("--driver", choices=["soapy", "rtl", "fake"], default="soapy")
    p.add_argument(
        "--device-args",
        action="append",
        default=None,
        help="Device selector/args. For Soapy: 'driver=rtlsdr' or 'driver=sdrplay,serial=...'. Repeatable.",
    )
    p.add_argument(
        "--all-devices",
        action="store_true",
        help="When using Soapy driver, enumerate all devices and run the preset on each.",
    )
    p.add_argument("--config", default=None, help="Path to YAML config (for presets). Defaults to repo config.")
    p.add_argument("--preset", default="kexp", help="Preset name (e.g., kexp, marine, tone)")
    p.add_argument("--center-hz", type=float, default=None)
    p.add_argument("--sample-rate", type=int, default=None)
    p.add_argument("--offset", action="append", type=float, help="Channel offset Hz (repeatable)")
    p.add_argument("--duration", type=float, default=10.0, help="Seconds to capture per channel")
    p.add_argument("--audio-rate", type=int, default=48_000)
    p.add_argument("--gain", type=float, default=None, help="Optional receiver gain in dB")
    p.add_argument("--bandwidth", type=float, default=None, help="Optional RF bandwidth in Hz")
    p.add_argument("--out", default="harness_out", help="Output directory for WAV dumps")
    p.add_argument("--auto-gain", action="store_true", help="Auto-select gain per device based on audio level")
    p.add_argument("--probe-seconds", type=float, default=2.0, help="Seconds for auto-gain probe")
    return p.parse_args(argv)


def build_config_for_server(args: argparse.Namespace) -> AppConfig:
    cfg = AppConfig(
        server=ServerConfig(bind_address=args.host, port=args.port),
        stream=StreamConfig(default_transport="ws", default_format="iq16", default_audio_rate=args.audio_rate),
        limits=LimitsConfig(),
        # Do not pin a specific device at the app level; we pass deviceId per capture.
        device=DeviceConfig(driver=args.driver, device_args=None),
    )
    return cfg


def _default_config_path() -> Path:
    # backend/wavecapsdr/harness.py -> backend/config/wavecapsdr.yaml
    return Path(__file__).resolve().parents[1] / "config" / "wavecapsdr.yaml"


def preset_channels(preset: str, cfg_path: Optional[str]) -> Tuple[float, int, List[float]]:
    # Try config-defined presets first
    path = Path(cfg_path) if cfg_path else _default_config_path()
    if path.exists():
        try:
            import yaml

            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            presets = (data or {}).get("presets", {}) or {}
            if preset in presets:
                spec = presets[preset] or {}
                center = float(spec.get("center_hz"))
                sr = int(spec.get("sample_rate", 1_000_000))
                offsets = [float(x) for x in (spec.get("offsets") or [0.0])]
                return center, sr, offsets
        except Exception:
            pass

    if preset == "marine":
        # Small subset of US VHF Marine band (MHz)
        freqs = [156.650e6, 156.700e6, 156.800e6]  # Ch 13, 14, 16
        center = float(np.mean(freqs))
        span = max(freqs) - min(freqs)
        sr = max(1_000_000, int(span * 2))  # at least 1 MS/s
        offsets = [f - center for f in freqs]
        return center, sr, offsets
    if preset.lower() in {"kexp", "fm", "fm_kexp"}:
        # 90.3 MHz is KEXP (Seattle). Single channel; center tuned to station.
        return 90.3e6, 2_000_000, [0.0]
    # default tone preset (fake driver emits ~5 kHz tone relative to center)
    center = 100_000_000.0
    sr = 1_000_000
    offsets = [5_000.0]
    return center, sr, offsets


async def ensure_server_running(args: argparse.Namespace) -> Optional[asyncio.AbstractServer]:
    if not args.start_server:
        return None
    cfg = build_config_for_server(args)
    app = create_app(cfg)
    config = uvicorn.Config(app, host=cfg.server.bind_address, port=cfg.server.port, log_level="info")
    server = uvicorn.Server(config)

    loop = asyncio.get_running_loop()
    task = loop.create_task(server.serve())
    # Give it a moment to bind
    await asyncio.sleep(0.3)
    return None


async def http_post_json(client: httpx.AsyncClient, path: str, body: Dict[str, Any], token: Optional[str]) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = await client.post(path, json=body, headers=headers)
    r.raise_for_status()
    return r.json()


async def run_harness(args: argparse.Namespace) -> int:
    await ensure_server_running(args)
    base = f"http://{args.host}:{args.port}/api/v1"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Derive scenario
    if args.center_hz is not None and args.sample_rate is not None:
        center, sr = args.center_hz, args.sample_rate
        offsets = args.offset or []
        if not offsets:
            offsets = [0.0]
    else:
        center, sr, offsets = preset_channels(args.preset, args.config)

    # Decide which devices to use
    device_args_list: List[Optional[str]] = []
    if args.device_args:
        # Already a list due to action='append'
        device_args_list = list(args.device_args)
    elif args.all_devices and args.driver == "soapy":
        # Discover via API
        async with httpx.AsyncClient(base_url=base, timeout=10.0) as client:
            r = await client.get("/devices")
            r.raise_for_status()
            devs = r.json() or []
            for d in devs:
                # Each device's id is a Soapy args string when using SoapyDriver
                device_args_list.append(d.get("id"))
    else:
        device_args_list = [None]

    chan_ids: List[str] = []
    capture_ids: List[str] = []
    device_best_gain: Dict[Optional[str], Optional[float]] = {}

    async with httpx.AsyncClient(base_url=base, timeout=10.0) as client:
        # Auto-gain sweep per device (if requested)
        if args.auto_gain:
            candidates: List[Optional[float]]
            if args.gain is not None:
                candidates = [args.gain]
            else:
                # Reasonable sweep for RTL-SDR and SDRplay
                candidates = [None, 0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
            for dev_args in device_args_list:
                try:
                    best = await _probe_best_gain(
                        client=client,
                        token=args.token,
                        device_id=dev_args,
                        center=center,
                        sample_rate=sr,
                        offset=offsets[0] if offsets else 0.0,
                        audio_rate=args.audio_rate,
                        candidates=candidates,
                        seconds=args.probe_seconds,
                        host=args.host,
                        port=args.port,
                    )
                except Exception:
                    best = args.gain
                device_best_gain[dev_args] = best

        for dev_args in device_args_list:
            cap = await http_post_json(
                client,
                "/captures",
                {
                    "deviceId": dev_args,
                    "centerHz": center,
                    "sampleRate": sr,
                    **({"gain": device_best_gain.get(dev_args)} if args.auto_gain else ({} if args.gain is None else {"gain": args.gain})),
                    **({"bandwidth": args.bandwidth} if args.bandwidth is not None else {}),
                },
                args.token,
            )
            cid = cap["id"]
            capture_ids.append(cid)
            await http_post_json(client, f"/captures/{cid}/start", {}, args.token)

            for off in offsets:
                ch = await http_post_json(
                    client,
                    f"/captures/{cid}/channels",
                    {"mode": "wbfm", "offsetHz": float(off), "audioRate": args.audio_rate},
                    args.token,
                )
                chan_ids.append(ch["id"])
                await http_post_json(client, f"/channels/{ch['id']}/start", {}, args.token)

    # Collect audio for each channel
    async def collect_channel(chan_id: str) -> Dict[str, Any]:
        url = f"ws://{args.host}:{args.port}/api/v1/stream/channels/{chan_id}"
        if args.token:
            url += f"?token={args.token}"
        total = bytearray()
        start = time.time()
        async with websockets.connect(url, max_size=None) as ws:
            while time.time() - start < args.duration:
                msg = await ws.recv()
                if isinstance(msg, (bytes, bytearray)):
                    total.extend(msg)
        # Analyze
        if total:
            audio = np.frombuffer(total, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(np.square(audio))))
            peak = float(np.max(np.abs(audio)))
            # Simple pass/fail heuristic: minimum RMS and peak thresholds
            ok = (rms > 0.003) and (peak > 0.05)
        else:
            rms = 0.0
            peak = 0.0
            ok = False
        # Dump wav
        wav_path = out_dir / f"channel_{chan_id}.wav"
        try:
            import wave

            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(args.audio_rate)
                wf.writeframes(bytes(total))
        except Exception:
            pass
        return {"channelId": chan_id, "rms": rms, "peak": peak, "wav": str(wav_path), "ok": ok}

    reports: List[Dict[str, Any]] = []
    for chan_id in chan_ids:
        reports.append(await collect_channel(chan_id))

    # Cleanup
    async with httpx.AsyncClient(base_url=base, timeout=10.0) as client:
        for cid in capture_ids:
            try:
                await http_post_json(client, f"/captures/{cid}/stop", {}, args.token)
            except Exception:
                pass

    # Backwards compatibility: if multiple captures were used, show the first id; include all channels
    print(HarnessReport(capture_id=capture_ids[0] if capture_ids else "c0", channel_reports=reports).to_json())
    # Fail (non-zero) if any channel did not pass heuristic
    return 0 if all(r.get("ok") for r in reports) else 2


async def _probe_best_gain(
    *,
    client: httpx.AsyncClient,
    token: Optional[str],
    device_id: Optional[str],
    center: float,
    sample_rate: int,
    offset: float,
    audio_rate: int,
    candidates: List[Optional[float]],
    seconds: float,
    host: str,
    port: int,
) -> Optional[float]:
    """Quickly tests several gains and returns the one with strong audio without clipping.

    Heuristic:
      - score = rms if peak < 0.98 else rms * 0.1 (penalize clipping)
      - choose highest score; if all are very low (rms < 0.002), return None to let device AGC decide.
    """
    results: List[Tuple[Optional[float], float, float]] = []  # (gain, rms, peak)

    async def one(g: Optional[float]) -> Tuple[Optional[float], float, float]:
        cap = await http_post_json(
            client,
            "/captures",
            {
                "deviceId": device_id,
                "centerHz": center,
                "sampleRate": sample_rate,
                **({"gain": g} if g is not None else {}),
            },
            token,
        )
        cid = cap["id"]
        await http_post_json(client, f"/captures/{cid}/start", {}, token)
        ch = await http_post_json(
            client,
            f"/captures/{cid}/channels",
            {"mode": "wbfm", "offsetHz": float(offset), "audioRate": audio_rate},
            token,
        )
        chan_id = ch["id"]

        url = f"ws://{host}:{port}/api/v1/stream/channels/{chan_id}"
        total = bytearray()
        start = time.time()
        try:
            async with websockets.connect(url, max_size=None) as ws:
                while time.time() - start < seconds:
                    msg = await ws.recv()
                    if isinstance(msg, (bytes, bytearray)):
                        total.extend(msg)
        finally:
            try:
                await http_post_json(client, f"/captures/{cid}/stop", {}, token)
            except Exception:
                pass

        if total:
            audio = np.frombuffer(total, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(np.square(audio))))
            peak = float(np.max(np.abs(audio)))
        else:
            rms = 0.0
            peak = 0.0
        return g, rms, peak

    for g in candidates:
        try:
            results.append(await one(g))
        except Exception:
            # Ignore failures for unsupported gains
            continue

    if not results:
        return None

    # Select best by heuristic
    best_gain: Optional[float] = None
    best_score = -1.0
    for g, rms, peak in results:
        score = rms if peak < 0.98 else (rms * 0.1)
        if score > best_score:
            best_score = score
            best_gain = g

    if best_score < 0.002:
        return None
    return best_gain


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    try:
        asyncio.run(run_harness(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":  # pragma: no cover - manual invocation path
    main()
