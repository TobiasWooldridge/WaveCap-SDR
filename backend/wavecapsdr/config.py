from __future__ import annotations

import shutil
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
    # SoapySDR device settings (key-value pairs passed to writeSetting)
    device_settings: Dict[str, Any] = field(default_factory=dict)
    # Per-element gains (e.g., {"LNA": 20, "VGA": 15})
    element_gains: Dict[str, float] = field(default_factory=dict)
    # Stream format preference ("CF32", "CS16", "CS8")
    stream_format: Optional[str] = None
    # Enable automatic DC offset correction
    dc_offset_auto: bool = True
    # Enable automatic IQ balance correction
    iq_balance_auto: bool = True


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


def save_config(config: AppConfig, path_str: str) -> None:
    """Save the AppConfig back to a YAML file, preserving structure and comments where possible."""
    path = Path(path_str)

    # Read existing file to preserve comments and structure
    existing_data: Dict[str, Any] = _read_yaml(path) if path.exists() else {}
    if path.exists():
        backup_path = path.with_suffix(path.suffix + ".bak")
        try:
            shutil.copy2(path, backup_path)
        except Exception as exc:
            print(f"Warning: Failed to write config backup to {backup_path}: {exc}", flush=True)

    # Update the config data
    # Server config
    server_data = {
        "bind_address": config.server.bind_address,
        "port": config.server.port,
    }
    if config.server.auth_token is not None:
        server_data["auth_token"] = config.server.auth_token
    existing_data["server"] = server_data

    # Stream config
    existing_data["stream"] = {
        "default_transport": config.stream.default_transport,
        "default_format": config.stream.default_format,
        "default_audio_rate": config.stream.default_audio_rate,
    }

    # Limits config
    limits_data = {
        "max_concurrent_captures": config.limits.max_concurrent_captures,
    }
    limits_data["max_channels_per_capture"] = config.limits.max_channels_per_capture
    if config.limits.max_sample_rate is not None:
        limits_data["max_sample_rate"] = config.limits.max_sample_rate
    existing_data["limits"] = limits_data

    # Device config
    device_data = {"driver": config.device.driver}
    if config.device.device_args is not None:
        device_data["device_args"] = config.device.device_args
    existing_data["device"] = device_data

    # Presets
    presets_data = {}
    for name, preset in config.presets.items():
        preset_dict = {
            "center_hz": int(preset.center_hz),
            "sample_rate": preset.sample_rate,
            "offsets": preset.offsets,
        }
        if preset.gain is not None:
            preset_dict["gain"] = preset.gain
        if preset.bandwidth is not None:
            preset_dict["bandwidth"] = preset.bandwidth
        if preset.ppm is not None:
            preset_dict["ppm"] = preset.ppm
        if preset.antenna is not None:
            preset_dict["antenna"] = preset.antenna
        if preset.squelch_db is not None:
            preset_dict["squelch_db"] = preset.squelch_db
        if preset.device_settings:
            preset_dict["device_settings"] = preset.device_settings
        if preset.element_gains:
            preset_dict["element_gains"] = preset.element_gains
        if preset.stream_format is not None:
            preset_dict["stream_format"] = preset.stream_format
        if not preset.dc_offset_auto:
            preset_dict["dc_offset_auto"] = preset.dc_offset_auto
        if not preset.iq_balance_auto:
            preset_dict["iq_balance_auto"] = preset.iq_balance_auto
        presets_data[name] = preset_dict
    existing_data["presets"] = presets_data

    # Captures list
    captures_data = []
    for cap in config.captures:
        cap_dict = {"preset": cap.preset}
        if cap.device_id is not None:
            cap_dict["device_id"] = cap.device_id
        captures_data.append(cap_dict)
    if captures_data:
        existing_data["captures"] = captures_data
    else:
        original = existing_data.get("captures")
        if original:
            print(
                "Warning: Config save skipped overwriting existing captures to avoid accidental deletion.",
                flush=True,
            )
        else:
            existing_data["captures"] = []

    # Write to file with nice formatting
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(existing_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
