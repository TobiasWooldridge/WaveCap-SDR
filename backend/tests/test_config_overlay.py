"""Tests for config overlay behavior with local overrides."""
from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from wavecapsdr.config import load_config


def test_load_config_overlays_local_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "wavecapsdr.yaml"
        local_path = Path(tmpdir) / "wavecapsdr.local.yaml"

        base_config = {
            "device": {"driver": "soapy"},
            "presets": {
                "base": {
                    "center_hz": 100_000_000,
                    "sample_rate": 1_000_000,
                }
            },
        }
        local_config = {
            "device": {"device_args": "driver=sdrplay,serial=123"},
            "device_names": {"driver=sdrplay,serial=123": "RSP"},
        }

        base_path.write_text(yaml.safe_dump(base_config), encoding="utf-8")
        local_path.write_text(yaml.safe_dump(local_config), encoding="utf-8")

        config = load_config(str(base_path))

        assert config.device.driver == "soapy"
        assert config.device.device_args == "driver=sdrplay,serial=123"
        assert config.device_names["driver=sdrplay,serial=123"] == "RSP"
