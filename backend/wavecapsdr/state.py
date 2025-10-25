from __future__ import annotations

from dataclasses import dataclass

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

    @classmethod
    def from_config(cls, cfg: AppConfig) -> "AppState":
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
                except Exception:
                    # If Soapy Python bindings are missing or initialization fails,
                    # fall back to the fake driver so the app remains operable.
                    driver = FakeDriver()

        captures = CaptureManager(cfg, driver)
        return cls(config=cfg, driver=driver, captures=captures)
