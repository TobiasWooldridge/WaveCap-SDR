from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

if TYPE_CHECKING:
    pass

DriverName = Literal["soapy", "fake", "rtl"]


@dataclass
class ServerConfig:
    bind_address: str = "127.0.0.1"
    port: int = 8087
    auth_token: str | None = None


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
    max_sample_rate: int | None = None


@dataclass
class DeviceConfig:
    driver: DriverName = "soapy"
    # Optional SoapySDR device args string, e.g., "driver=rtlsdr" or specific serial.
    device_args: str | None = None
    # Show fake/test device even when real devices are available (for development)
    show_fake_device: bool = False


@dataclass
class RecoveryConfig:
    """Configuration for automatic device recovery."""

    # Enable automatic SDRplay service restart on failure
    sdrplay_service_restart_enabled: bool = True
    # Minimum seconds between service restart attempts
    sdrplay_service_restart_cooldown: float = 60.0
    # Maximum service restarts allowed per hour
    max_service_restarts_per_hour: int = 5
    # Minimum seconds between SDRplay device operations (configure, close)
    # Prevents rapid reconfiguration from overwhelming the SDRplay API service
    sdrplay_operation_cooldown: float = 0.5
    # Enable IQ sample watchdog (restart capture if no samples for this many seconds)
    iq_watchdog_enabled: bool = True
    iq_watchdog_timeout: float = 30.0


@dataclass
class RecipeChannel:
    """Definition of a channel to create from a recipe."""
    offset_hz: float
    name: str  # Display name like "Channel 16 - Emergency"
    mode: str = "wbfm"
    squelch_db: float = -60
    # POCSAG decoding settings (NBFM only)
    enable_pocsag: bool = False
    pocsag_baud: int = 1200


@dataclass
class RecipeConfig:
    """A recipe/wizard template for creating captures."""
    name: str  # Display name
    description: str  # Help text
    category: str  # "Marine", "Aviation", "Broadcast", etc.
    # Template values
    center_hz: float
    sample_rate: int
    gain: float | None = None
    bandwidth: float | None = None
    # Channels to create
    channels: list[RecipeChannel] = field(default_factory=list)
    # Whether user can customize frequency
    allow_frequency_input: bool = False
    # Frequency input label (e.g., "Station Frequency")
    frequency_label: str | None = None


@dataclass
class PresetConfig:
    """A preset configuration for a capture."""
    center_hz: float
    sample_rate: int
    offsets: list[float] = field(default_factory=list)
    gain: float | None = None
    bandwidth: float | None = None
    ppm: float | None = None
    antenna: str | None = None
    squelch_db: float | None = None
    # SoapySDR device settings (key-value pairs passed to writeSetting)
    device_settings: dict[str, Any] = field(default_factory=dict)
    # Per-element gains (e.g., {"LNA": 20, "VGA": 15})
    element_gains: dict[str, float] = field(default_factory=dict)
    # Stream format preference ("CF32", "CS16", "CS8")
    stream_format: str | None = None
    # Enable automatic DC offset correction
    dc_offset_auto: bool = True
    # Enable automatic IQ balance correction
    iq_balance_auto: bool = True


@dataclass
class CaptureStartConfig:
    """Configuration for a capture to auto-start."""
    preset: str  # Name of preset to use
    device_id: str | None = None  # If None, use any available device


@dataclass
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    presets: dict[str, PresetConfig] = field(default_factory=dict)
    recipes: dict[str, RecipeConfig] = field(default_factory=dict)
    captures: list[CaptureStartConfig] = field(default_factory=list)
    device_names: dict[str, str] = field(default_factory=dict)  # device_id -> custom name
    # Raw trunking system configs (parsed into TrunkingSystemConfig objects in state.py)
    trunking_systems: dict[str, dict[str, Any]] = field(default_factory=dict)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("Config root must be a mapping")
        return data


def _overlay(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _overlay(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config(path_str: str) -> AppConfig:
    path = Path(path_str)
    raw: dict[str, Any] = _read_yaml(path)

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
    recovery = RecoveryConfig(**raw.get("recovery", {}))

    # Parse presets
    presets: dict[str, PresetConfig] = {}
    presets_raw = raw.get("presets", {})
    if isinstance(presets_raw, dict):
        for name, preset_data in presets_raw.items():
            if isinstance(preset_data, dict):
                presets[name] = PresetConfig(**preset_data)

    # Parse recipes
    recipes: dict[str, RecipeConfig] = {}
    recipes_raw = raw.get("recipes", {})
    if isinstance(recipes_raw, dict):
        for name, recipe_data in recipes_raw.items():
            if isinstance(recipe_data, dict):
                # Parse channels if present
                channels = []
                channels_raw = recipe_data.get("channels", [])
                if isinstance(channels_raw, list):
                    for ch_data in channels_raw:
                        if isinstance(ch_data, dict):
                            channels.append(RecipeChannel(**ch_data))

                # Create recipe with parsed channels
                recipe_dict = dict(recipe_data)
                recipe_dict["channels"] = channels
                recipes[name] = RecipeConfig(**recipe_dict)

    # Parse captures list
    captures: list[CaptureStartConfig] = []
    captures_raw = raw.get("captures", [])
    if isinstance(captures_raw, list):
        for cap_data in captures_raw:
            if isinstance(cap_data, dict):
                captures.append(CaptureStartConfig(**cap_data))

    # Parse device_names mapping
    device_names: dict[str, str] = {}
    device_names_raw = raw.get("device_names", {})
    if isinstance(device_names_raw, dict):
        device_names = {k: str(v) for k, v in device_names_raw.items()}

    # Parse trunking systems (raw dicts, converted to TrunkingSystemConfig in state.py)
    trunking_systems: dict[str, dict[str, Any]] = {}
    trunking_raw = raw.get("trunking", {})
    if isinstance(trunking_raw, dict):
        systems_raw = trunking_raw.get("systems", {})
        if isinstance(systems_raw, dict):
            for sys_id, sys_data in systems_raw.items():
                if isinstance(sys_data, dict):
                    trunking_systems[sys_id] = sys_data

    return AppConfig(
        server=server,
        stream=stream,
        limits=limits,
        device=device,
        recovery=recovery,
        presets=presets,
        recipes=recipes,
        captures=captures,
        device_names=device_names,
        trunking_systems=trunking_systems,
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
    existing_data: dict[str, Any] = _read_yaml(path) if path.exists() else {}
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
    device_data: dict[str, str] = {"driver": config.device.driver}
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

    # Recipes
    recipes_data = {}
    for name, recipe in config.recipes.items():
        # Serialize channels
        channels_data = []
        for ch in recipe.channels:
            ch_dict = {
                "offset_hz": ch.offset_hz,
                "name": ch.name,
                "mode": ch.mode,
                "squelch_db": ch.squelch_db,
            }
            channels_data.append(ch_dict)

        recipe_dict = {
            "name": recipe.name,
            "description": recipe.description,
            "category": recipe.category,
            "center_hz": int(recipe.center_hz),
            "sample_rate": recipe.sample_rate,
            "channels": channels_data,
            "allow_frequency_input": recipe.allow_frequency_input,
        }
        if recipe.gain is not None:
            recipe_dict["gain"] = recipe.gain
        if recipe.bandwidth is not None:
            recipe_dict["bandwidth"] = recipe.bandwidth
        if recipe.frequency_label is not None:
            recipe_dict["frequency_label"] = recipe.frequency_label

        recipes_data[name] = recipe_dict
    existing_data["recipes"] = recipes_data

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

    # Device names mapping
    if config.device_names:
        existing_data["device_names"] = config.device_names

    # Trunking systems
    if config.trunking_systems:
        trunking_data = existing_data.setdefault("trunking", {})
        trunking_data["systems"] = config.trunking_systems

    # Write to file with nice formatting
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(existing_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def update_trunking_system_state(
    path_str: str, system_id: str, auto_start: bool
) -> None:
    """Update a specific trunking system's auto_start state in the config file.

    This function reads the existing config, updates just the auto_start field
    for the specified system, and writes it back. Used to persist state changes
    across server restarts.

    Args:
        path_str: Path to the config file
        system_id: ID of the trunking system to update
        auto_start: New auto_start value
    """
    path = Path(path_str)
    if not path.exists():
        return

    raw = _read_yaml(path)

    # Navigate to trunking.systems.{system_id}
    trunking = raw.setdefault("trunking", {})
    systems = trunking.setdefault("systems", {})

    if system_id not in systems:
        return

    # Update the auto_start field
    systems[system_id]["auto_start"] = auto_start

    # Write back
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
