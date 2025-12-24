"""Composite device driver that combines multiple drivers with conditional filtering.

This driver wraps a primary driver (e.g., SoapyDriver) and optionally includes
fake/test devices for development purposes. The fake device is automatically
hidden when real devices are available, unless explicitly enabled via config.
"""
from __future__ import annotations

from collections.abc import Iterable

from ..config import DeviceConfig
from .base import Device, DeviceDriver, DeviceInfo
from .fake import FakeDriver


class CompositeDriver(DeviceDriver):
    """Combines multiple device drivers with conditional fake device visibility.

    Behavior:
    - Always enumerate real devices from the primary driver
    - Include fake device only if:
      1. show_fake_device is True in config (explicit enable), OR
      2. No real devices are available (fallback for development)
    """

    name = "composite"

    def __init__(self, primary_driver: DeviceDriver, cfg: DeviceConfig):
        self._primary = primary_driver
        self._fake = FakeDriver()
        self._show_fake_device = cfg.show_fake_device

    def enumerate(self) -> Iterable[DeviceInfo]:
        """Enumerate devices from all drivers with conditional fake device."""
        # Get real devices from primary driver
        real_devices = list(self._primary.enumerate())

        # Decide whether to include fake device
        include_fake = self._show_fake_device or len(real_devices) == 0

        if include_fake:
            # Include fake device(s)
            yield from self._fake.enumerate()

        # Yield all real devices
        yield from real_devices

    def open(self, id_or_args: str | None = None) -> Device:
        """Open a device by ID, routing to the appropriate driver."""
        # Check if this is a fake device ID
        if id_or_args is not None and id_or_args.startswith("fake"):
            return self._fake.open(id_or_args)

        # Otherwise, use primary driver
        return self._primary.open(id_or_args)
