from __future__ import annotations

import concurrent.futures
import time

from fastapi.testclient import TestClient

from wavecapsdr.app import create_app
from wavecapsdr.config import AppConfig, DeviceConfig


def make_client():
    cfg = AppConfig()
    cfg.device = DeviceConfig(driver="fake")
    app = create_app(cfg)
    return TestClient(app)


def _receive_bytes_with_timeout(ws, timeout_s: float = 2.0) -> bytes:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(ws.receive_bytes)
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError as exc:
            raise AssertionError(
                f"Timed out waiting for WebSocket data after {timeout_s:.1f}s"
            ) from exc


def test_capture_iq_stream():
    with make_client() as client:
        # Create capture
        req = {"centerHz": 100_000_000, "sampleRate": 1_000_000}
        r = client.post("/api/v1/captures", json=req)
        assert r.status_code == 200
        cid = r.json()["id"]
        # Start capture
        r = client.post(f"/api/v1/captures/{cid}/start")
        assert r.status_code == 200
        # WS stream IQ
        with client.websocket_connect(f"/api/v1/stream/captures/{cid}/iq") as ws:
            time.sleep(0.05)
            data = _receive_bytes_with_timeout(ws, timeout_s=2.0)
            assert isinstance(data, (bytes, bytearray))
            assert len(data) > 0
        # Stop
        r = client.post(f"/api/v1/captures/{cid}/stop")
        assert r.status_code == 200


def test_channel_audio_stream():
    with make_client() as client:
        req = {"centerHz": 100_000_000, "sampleRate": 1_000_000}
        r = client.post("/api/v1/captures", json=req)
        assert r.status_code == 200
        cid = r.json()["id"]
        r = client.post(f"/api/v1/captures/{cid}/start")
        assert r.status_code == 200
        # Create channel with 5 kHz offset matching fake driver signal
        r = client.post(f"/api/v1/captures/{cid}/channels", json={"mode": "wbfm", "offsetHz": 5000.0})
        assert r.status_code == 200
        chan_id = r.json()["id"]
        # Start channel
        r = client.post(f"/api/v1/channels/{chan_id}/start")
        assert r.status_code == 200
        # WS audio
        with client.websocket_connect(f"/api/v1/stream/channels/{chan_id}") as ws:
            time.sleep(0.05)
            data = _receive_bytes_with_timeout(ws, timeout_s=2.0)
            assert isinstance(data, (bytes, bytearray))
            assert len(data) > 0
        # Stop channel and capture
        r = client.post(f"/api/v1/channels/{chan_id}/stop")
        assert r.status_code == 200
        r = client.post(f"/api/v1/captures/{cid}/stop")
        assert r.status_code == 200
