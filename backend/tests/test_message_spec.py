from __future__ import annotations

import base64
import json
from pathlib import Path

from wavecapsdr.message_spec import encode_message, load_message_spec


def test_load_message_spec_accepts_json(tmp_path: Path) -> None:
    payload = {
        "codec": "imbe",
        "frames": [
            {"base64": base64.b64encode(b"\x01\x02\x03").decode("ascii"), "repeat": 1},
        ],
        "outputRate": 8000,
    }
    path = tmp_path / "message.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    spec = load_message_spec(path)
    assert spec.codec == "imbe"
    assert spec.frames[0].data.startswith(b"\x01\x02\x03")
    assert spec.output_rate == 8000


def test_encode_message_produces_audio_and_bytes(tmp_path: Path) -> None:
    spec_data = {
        "codec": "imbe",
        "frameDurationMs": 20.0,
        "outputRate": 8000,
        "frames": [
            {
                "hex": "000102030405060708090a0b0c0d0e0f1011",
                "repeat": 2,
                "silence_ms": 5.0,
            }
        ],
    }
    path = tmp_path / "message.yaml"
    path.write_text(json.dumps(spec_data), encoding="utf-8")

    result = encode_message(path)

    # 18-byte frame repeated twice
    assert len(result.payload_bytes) == 36

    # 20 ms of audio per frame plus 5 ms of silence after each frame
    expected_samples = int(0.02 * 8000) * 2 + int(0.005 * 8000) * 2
    assert len(result.audio) == expected_samples
    assert result.sample_rate == 8000
