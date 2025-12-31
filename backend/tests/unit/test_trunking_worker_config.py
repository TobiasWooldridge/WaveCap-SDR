from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from wavecapsdr.config import load_config


def test_load_trunking_worker_config() -> None:
    config_data = {
        "trunking": {
            "worker_mode": "per_device",
            "worker_rpc_timeout_s": 12.5,
            "worker_event_queue_size": 321,
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "wavecapsdr.yaml"
        config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

        config = load_config(str(config_path))

    assert config.trunking_workers.mode == "per_device"
    assert config.trunking_workers.rpc_timeout_s == 12.5
    assert config.trunking_workers.event_queue_size == 321
