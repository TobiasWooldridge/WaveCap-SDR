from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .config import AppConfig, default_config_path

if TYPE_CHECKING:
    from .scanner import ScannerService
from .devices.base import DeviceDriver
from .devices.composite import CompositeDriver
from .devices.fake import FakeDriver
from .devices.rtl import RtlDriver
from .trunking import TrunkingManager
from .trunking.config import TrunkingSystemConfig
from .trunking.manager_types import TrunkingManagerLike
from .trunking.process_manager import TrunkingProcessManager

try:  # Optional import; only when using Soapy driver
    from .devices.soapy import SoapyDriver
except Exception:  # pragma: no cover
    SoapyDriver = None  # type: ignore

from .capture import CaptureManager
from .device_namer import load_device_nicknames


def create_device_driver(cfg: AppConfig) -> DeviceDriver:
    driver: DeviceDriver
    if cfg.device.driver == "fake":
        # Explicit fake driver mode - only show fake device
        driver = FakeDriver()
    elif cfg.device.driver == "rtl":
        try:
            driver = RtlDriver()
        except Exception:
            driver = FakeDriver()
    else:
        # Soapy driver mode - use CompositeDriver to manage real + fake devices
        if SoapyDriver is None:
            # SoapySDR not available, fall back to fake driver
            driver = FakeDriver()  # type: ignore[unreachable]
        else:
            try:
                soapy_driver = SoapyDriver(cfg.device)
                # Wrap SoapyDriver in CompositeDriver to handle fake device visibility
                # CompositeDriver only shows fake device when show_fake_device=True
                driver = CompositeDriver(soapy_driver, cfg.device)
            except Exception:
                # If Soapy Python bindings are missing or initialization fails,
                # fall back to the fake driver so the app remains operable.
                driver = FakeDriver()
    return driver


@dataclass
class AppState:
    config: AppConfig
    driver: DeviceDriver
    captures: CaptureManager
    trunking_manager: TrunkingManagerLike
    config_path: str | None = None
    # Map capture_id -> preset_name for persistence
    capture_presets: dict[str, str] = field(default_factory=dict)
    # Map scanner_id -> ScannerService for active scanners
    scanners: dict[str, ScannerService] = field(default_factory=dict)
    # Snapshot of devices (device_id -> model dict) for state stream diffs
    device_snapshot: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: AppConfig, config_path: str | None = None) -> AppState:
        driver = create_device_driver(cfg)

        captures = CaptureManager(cfg, driver)

        resolved_config_path = config_path or default_config_path()
        trunking_manager: TrunkingManagerLike
        if cfg.trunking_workers.mode == "per_device":
            trunking_manager = TrunkingProcessManager(
                config_path=resolved_config_path,
                rpc_timeout_s=cfg.trunking_workers.rpc_timeout_s,
                event_queue_size=cfg.trunking_workers.event_queue_size,
            )
        else:
            trunking_manager = TrunkingManager()
        trunking_manager.set_capture_manager(captures)

        # Set config path for state persistence
        trunking_manager.set_config_path(resolved_config_path)

        # Get config directory for resolving relative paths (e.g., talkgroups_file)
        config_dir = os.path.dirname(resolved_config_path) if resolved_config_path else None

        # Register trunking systems from config (loaded during manager.start())
        for sys_id, sys_data in cfg.trunking_systems.items():
            try:
                # Ensure the id is set in the data
                sys_data_with_id = dict(sys_data)
                if "id" not in sys_data_with_id:
                    sys_data_with_id["id"] = sys_id
                trunking_config = TrunkingSystemConfig.from_dict(
                    sys_data_with_id,
                    config_dir=config_dir,
                    rr_config=cfg.radioreference,
                )
                trunking_manager.register_config(trunking_config)
            except Exception as e:
                print(f"Warning: Failed to parse trunking system '{sys_id}': {e}", flush=True)

        # Load device nicknames from config
        load_device_nicknames(cfg.device_names)

        return cls(
            config=cfg,
            driver=driver,
            captures=captures,
            trunking_manager=trunking_manager,
            config_path=config_path,
        )
