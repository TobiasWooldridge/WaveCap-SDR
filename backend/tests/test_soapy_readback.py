from __future__ import annotations

import sys
from types import SimpleNamespace

from wavecapsdr.devices.base import DeviceInfo
from wavecapsdr.devices.soapy import _SoapyDevice


def test_soapy_device_readback_reports_mismatches(monkeypatch, capsys) -> None:
    fake_soapy = SimpleNamespace(SOAPY_SDR_RX=0)
    monkeypatch.setitem(sys.modules, "SoapySDR", fake_soapy)

    class FakeSDR:
        def __init__(self) -> None:
            self.read_setting_calls: list[str] = []

        def getSampleRate(self, *_: object) -> float:
            return 1_200_000.0

        def getFrequency(self, *_: object) -> float:
            return 162_400_000.0

        def getGain(self, *args: object) -> float:
            if len(args) == 2:
                return 33.0
            return 12.0

        def getBandwidth(self, *_: object) -> float:
            return 1_500_000.0

        def getFrequencyCorrection(self, *_: object) -> float:
            return 0.75

        def getAntenna(self, *_: object) -> str:
            return "RX"

        def readSetting(self, key: str) -> str:
            self.read_setting_calls.append(key)
            return "off" if key == "bias_tee" else "unknown"

    device_info = DeviceInfo(
        id="device0",
        driver="rtlsdr",
        label="Fake",
        freq_min_hz=0.0,
        freq_max_hz=1e9,
        sample_rates=(1,),
        gains=(),
    )
    device = _SoapyDevice(info=device_info, sdr=FakeSDR())

    device._verify_settings_applied(  # type: ignore[attr-defined]
        center_hz=162_000_000.0,
        sample_rate=1_000_000,
        gain=20.0,
        bandwidth=1_000_000.0,
        ppm=0.5,
        antenna="Antenna A",
        device_settings={"bias_tee": "on"},
        element_gains={"LNA": 15.0},
    )

    output = capsys.readouterr().out
    assert "[SOAPY] Readback:" in output
    assert "sample_rate=1200000Hz" in output
    assert "setting[bias_tee]=off" in output
    assert "gain[LNA]=12.00dB" in output
    assert "WARNING: Setting mismatches detected:" in output
