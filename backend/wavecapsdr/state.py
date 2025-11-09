from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .config import AppConfig
from .devices.base import DeviceDriver
from .devices.fake import FakeDriver
from .devices.rtl import RtlDriver

try:  # Optional import; only when using Soapy driver
    from .devices.soapy import SoapyDriver
except Exception:  # pragma: no cover
    SoapyDriver = None  # type: ignore

from .capture import CaptureManager


@dataclass
class AppState:
    config: AppConfig
    driver: DeviceDriver
    captures: CaptureManager
    config_path: Optional[str] = None
    # Map capture_id -> preset_name for persistence
    capture_presets: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: AppConfig, config_path: Optional[str] = None) -> "AppState":
        driver: DeviceDriver
        if cfg.device.driver == "fake":
            driver = FakeDriver()
        elif cfg.device.driver == "rtl":
            try:
                driver = RtlDriver()
            except Exception:
                driver = FakeDriver()
        else:
            if SoapyDriver is None:
                driver = FakeDriver()
            else:
                try:
                    driver = SoapyDriver(cfg.device)
                    # If Soapy is available but no devices are found, fall back to Fake
                    try:
                        devices = list(driver.enumerate())
                        if not devices:
                            print("[INFO] No SoapySDR devices found; falling back to Fake driver for development.", flush=True)
                            driver = FakeDriver()
                    except Exception:
                        # Enumeration failed; use Fake to keep the app usable
                        driver = FakeDriver()
                except Exception:
                    # If Soapy Python bindings are missing or initialization fails,
                    # fall back to the fake driver so the app remains operable.
                    driver = FakeDriver()

        captures = CaptureManager(cfg, driver)
        return cls(config=cfg, driver=driver, captures=captures, config_path=config_path)
