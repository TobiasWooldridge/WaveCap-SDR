from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml

DriverName = Literal["soapy", "fake"]


@dataclass
class ServerConfig:
    bind_address: str = "127.0.0.1"
    port: int = 8087
    auth_token: Optional[str] = None


@dataclass
class StreamConfig:
    # Default output: IQ streaming over websocket, little-endian int16 IQ interleaved
    default_transport: Literal["ws", "http"] = "ws"
    default_format: Literal["iq16", "f32", "pcm16"] = "iq16"
    default_audio_rate: int = 48_000


@dataclass
class LimitsConfig:
    max_concurrent_captures: int = 2
    max_channels_per_capture: int = 8
    max_sample_rate: Optional[int] = None


@dataclass
class DeviceConfig:
    driver: DriverName = "soapy"
    # Optional SoapySDR device args string, e.g., "driver=rtlsdr" or specific serial.
    device_args: Optional[str] = None


@dataclass
class PresetConfig:
    """A preset configuration for a capture."""
    center_hz: float
    sample_rate: int
    offsets: List[float] = field(default_factory=list)
    gain: Optional[float] = None
    bandwidth: Optional[float] = None
    ppm: Optional[float] = None
    antenna: Optional[str] = None
    squelch_db: Optional[float] = None


@dataclass
class CaptureStartConfig:
    """Configuration for a capture to auto-start."""
    preset: str  # Name of preset to use
    device_id: Optional[str] = None  # If None, use any available device


@dataclass
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    presets: Dict[str, PresetConfig] = field(default_factory=dict)
    captures: List[CaptureStartConfig] = field(default_factory=list)


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("Config root must be a mapping")
        return data


def _overlay(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _overlay(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config(path_str: str) -> AppConfig:
    path = Path(path_str)
    raw: Dict[str, Any] = _read_yaml(path)

    # Environment overrides (prefix WAVECAPSDR__SECTION__KEY)
    # Example: WAVECAPSDR__SERVER__PORT=8089
    prefix = "WAVECAPSDR__"
    for k, v in os_environ_items():
        if not k.startswith(prefix):
            continue
        parts = k[len(prefix) :].split("__")
        if len(parts) != 2:
            continue
        section, key = parts
        section = section.lower()
        key = key.lower()
        raw.setdefault(section, {})
        if section in ("server", "stream", "limits", "device") and isinstance(raw[section], dict):
            raw[section][key] = coerce_env_value(v)

    # Construct typed config
    server = ServerConfig(**raw.get("server", {}))
    stream = StreamConfig(**raw.get("stream", {}))
    limits = LimitsConfig(**raw.get("limits", {}))
    device = DeviceConfig(**raw.get("device", {}))

    # Parse presets
    presets: Dict[str, PresetConfig] = {}
    presets_raw = raw.get("presets", {})
    if isinstance(presets_raw, dict):
        for name, preset_data in presets_raw.items():
            if isinstance(preset_data, dict):
                presets[name] = PresetConfig(**preset_data)

    # Parse captures list
    captures: List[CaptureStartConfig] = []
    captures_raw = raw.get("captures", [])
    if isinstance(captures_raw, list):
        for cap_data in captures_raw:
            if isinstance(cap_data, dict):
                captures.append(CaptureStartConfig(**cap_data))

    return AppConfig(
        server=server,
        stream=stream,
        limits=limits,
        device=device,
        presets=presets,
        captures=captures,
    )


def coerce_env_value(val: str) -> Any:
    # Basic bool/int coercion for convenience
    lower = val.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        return int(val)
    except ValueError:
        return val


def os_environ_items() -> list[tuple[str, str]]:
    # Wrapped for testability
    from os import environ

    return [(k, v) for k, v in environ.items()]
