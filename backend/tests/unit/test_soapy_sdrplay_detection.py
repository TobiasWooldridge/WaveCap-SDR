from __future__ import annotations

from types import SimpleNamespace

from wavecapsdr.config import DeviceConfig
from wavecapsdr.devices.base import DeviceInfo
from wavecapsdr.devices.sdrplay_proxy import SDRplayProxyDevice
from wavecapsdr.devices import soapy


def test_open_uses_proxy_when_cached_driver_matches(monkeypatch) -> None:
    fake_soapy = SimpleNamespace()
    monkeypatch.setattr(soapy, "_import_soapy", lambda: fake_soapy)

    driver = soapy.SoapyDriver(DeviceConfig())
    device_info = DeviceInfo(
        id="device0",
        driver="sdrplay",
        label="SDRplay",
        freq_min_hz=0.0,
        freq_max_hz=1.0,
        sample_rates=(1,),
        gains=(),
    )
    driver._enumerate_cache = (0.0, [device_info])

    device = driver.open("device0")

    assert isinstance(device, SDRplayProxyDevice)
