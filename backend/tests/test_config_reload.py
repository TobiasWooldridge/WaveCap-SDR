"""Tests for the config reload API endpoint."""
from __future__ import annotations

import tempfile
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from wavecapsdr.app import create_app
from wavecapsdr.config import AppConfig, DeviceConfig, PresetConfig


def test_config_reload_no_config_path():
    """Test reload fails gracefully when no config path is set."""
    cfg = AppConfig()
    cfg.device = DeviceConfig(driver="fake")
    # Create app without config_path
    app = create_app(cfg, config_path=None)
    client = TestClient(app)

    r = client.post("/api/v1/config/reload")
    assert r.status_code == 400
    assert "No config path configured" in r.json()["detail"]


def test_config_reload_file_not_found():
    """Test reload fails when config file doesn't exist."""
    cfg = AppConfig()
    cfg.device = DeviceConfig(driver="fake")
    app = create_app(cfg, config_path="/nonexistent/path/config.yaml")
    client = TestClient(app)

    r = client.post("/api/v1/config/reload")
    assert r.status_code == 404
    assert "Config file not found" in r.json()["detail"]


def test_config_reload_updates_presets():
    """Test that reloading config updates presets."""
    # Create a temp config file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        initial_config = {
            "device": {"driver": "fake"},
            "presets": {
                "test1": {
                    "center_hz": 100_000_000,
                    "sample_rate": 1_000_000,
                }
            },
        }
        yaml.dump(initial_config, f)
        config_path = f.name

    try:
        cfg = AppConfig()
        cfg.device = DeviceConfig(driver="fake")
        cfg.presets = {
            "test1": PresetConfig(center_hz=100_000_000, sample_rate=1_000_000)
        }
        app = create_app(cfg, config_path=config_path)
        client = TestClient(app)

        # Verify initial state
        assert len(app.state.app_state.config.presets) == 1

        # Modify the config file
        updated_config = {
            "device": {"driver": "fake"},
            "presets": {
                "test1": {
                    "center_hz": 100_000_000,
                    "sample_rate": 1_000_000,
                },
                "test2": {
                    "center_hz": 200_000_000,
                    "sample_rate": 2_000_000,
                },
            },
        }
        with open(config_path, "w") as f:
            yaml.dump(updated_config, f)

        # Reload config
        r = client.post("/api/v1/config/reload")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "presets" in str(data["updated"])

        # Verify presets were updated
        assert len(app.state.app_state.config.presets) == 2
        assert "test2" in app.state.app_state.config.presets

    finally:
        Path(config_path).unlink(missing_ok=True)


def test_config_reload_no_changes():
    """Test that reloading unchanged config reports no changes."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config = {
            "device": {"driver": "fake"},
            "presets": {},
        }
        yaml.dump(config, f)
        config_path = f.name

    try:
        cfg = AppConfig()
        cfg.device = DeviceConfig(driver="fake")
        app = create_app(cfg, config_path=config_path)
        client = TestClient(app)

        # Reload without changes
        r = client.post("/api/v1/config/reload")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["updated"] == []

    finally:
        Path(config_path).unlink(missing_ok=True)
