from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional

from .config import AppConfig

if TYPE_CHECKING:
    from .scanner import ScannerService
from .devices.base import DeviceDriver
from .devices.fake import FakeDriver
from .devices.rtl import RtlDriver
from .devices.composite import CompositeDriver

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
    config_path: Optional[str] = None
    # Map capture_id -> preset_name for persistence
    capture_presets: Dict[str, str] = field(default_factory=dict)
    # Map scanner_id -> ScannerService for active scanners
    scanners: Dict[str, ScannerService] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: AppConfig, config_path: Optional[str] = None) -> "AppState":
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
                    # CompositeDriver will:
                    # - Hide fake device when real devices exist (unless show_fake_device=True)
                    # - Show fake device when no real devices are available (development fallback)
                    driver = CompositeDriver(soapy_driver, cfg.device)
                except Exception:
                    # If Soapy Python bindings are missing or initialization fails,
                    # fall back to the fake driver so the app remains operable.
                    driver = FakeDriver()

        captures = CaptureManager(cfg, driver)

        # Load device nicknames from config
        load_device_nicknames(cfg.device_names)

        return cls(config=cfg, driver=driver, captures=captures, config_path=config_path)
