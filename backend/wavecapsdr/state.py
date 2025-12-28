from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .config import AppConfig

if TYPE_CHECKING:
    from .scanner import ScannerService
from .devices.base import DeviceDriver
from .devices.composite import CompositeDriver
from .devices.fake import FakeDriver
from .devices.rtl import RtlDriver
from .trunking import TrunkingManager
from .trunking.config import TrunkingSystemConfig

try:  # Optional import; only when using Soapy driver
    from .devices.soapy import SoapyDriver
except Exception:  # pragma: no cover
    SoapyDriver = None  # type: ignore

from .capture import CaptureManager
from .device_namer import load_device_nicknames


@dataclass
class AppState:
    config: AppConfig
    driver: DeviceDriver
    captures: CaptureManager
    trunking_manager: TrunkingManager
    config_path: str | None = None
    # Map capture_id -> preset_name for persistence
    capture_presets: dict[str, str] = field(default_factory=dict)
    # Map scanner_id -> ScannerService for active scanners
    scanners: dict[str, ScannerService] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: AppConfig, config_path: str | None = None) -> AppState:
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

        captures = CaptureManager(cfg, driver)

        # Initialize P25 trunking manager with CaptureManager reference
        trunking_manager = TrunkingManager()
        trunking_manager.set_capture_manager(captures)

        # Set config path for state persistence
        if config_path:
            trunking_manager.set_config_path(config_path)

        # Register trunking systems from config (loaded during manager.start())
        for sys_id, sys_data in cfg.trunking_systems.items():
            try:
                # Ensure the id is set in the data
                sys_data_with_id = dict(sys_data)
                if "id" not in sys_data_with_id:
                    sys_data_with_id["id"] = sys_id
                trunking_config = TrunkingSystemConfig.from_dict(sys_data_with_id)
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
