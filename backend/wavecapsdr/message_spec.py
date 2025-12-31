from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from wavecapsdr.decoders.mbelib_neo import IMBEDecoderNeo

logger = logging.getLogger(__name__)


CodecType = Literal["imbe"]


@dataclass(frozen=True)
class VoiceFrameSpec:
    """Single vocoder frame entry."""

    data: bytes
    repeat: int = 1
    pad_silence_ms: float = 0.0

    @staticmethod
    def from_mapping(raw: dict[str, Any]) -> "VoiceFrameSpec":
        """Create a frame from a mapping loaded from YAML/JSON."""
        data_field = raw.get("hex") or raw.get("base64") or raw.get("bytes")
        if data_field is None:
            raise ValueError("Frame is missing required hex/base64/bytes field")

        data = _decode_frame_bytes(data_field)
        if not data:
            raise ValueError("Frame data resolved to empty bytes")

        repeat = int(raw.get("repeat", 1))
        pad_silence_ms = float(raw.get("silence_ms", raw.get("pad_silence_ms", 0.0)))
        return VoiceFrameSpec(data=data, repeat=max(1, repeat), pad_silence_ms=max(0.0, pad_silence_ms))


@dataclass(frozen=True)
class MessageSpec:
    """Voice message specification loaded from JSON/YAML."""

    codec: CodecType
    frames: list[VoiceFrameSpec]
    frame_duration_ms: float = 20.0
    output_rate: int = 48_000

    @staticmethod
    def from_mapping(raw: dict[str, Any]) -> "MessageSpec":
        """Create a spec from a parsed mapping."""
        codec = str(raw.get("codec", "imbe")).lower()
        if codec != "imbe":
            raise ValueError(f"Unsupported codec '{codec}'. Only 'imbe' is currently supported.")

        frame_list = raw.get("frames")
        if not isinstance(frame_list, list) or not frame_list:
            raise ValueError("Message spec must include a non-empty 'frames' array")

        frames = [VoiceFrameSpec.from_mapping(entry) for entry in frame_list if isinstance(entry, dict)]
        if not frames:
            raise ValueError("No valid frames found in message spec")

        frame_duration_ms = float(raw.get("frame_duration_ms", raw.get("frameDurationMs", 20.0)))
        output_rate = int(raw.get("output_rate", raw.get("outputRate", 48_000)))
        return MessageSpec(
            codec="imbe",
            frames=frames,
            frame_duration_ms=max(1.0, frame_duration_ms),
            output_rate=max(8000, output_rate),
        )

    def encoded_bytes(self) -> bytes:
        """Return concatenated encoded frames."""
        chunks: list[bytes] = []
        for frame in self.frames:
            for _ in range(frame.repeat):
                chunks.append(frame.data)
        return b"".join(chunks)

    def render_audio(self) -> tuple[np.ndarray, bool]:
        """Render PCM audio for the spec, returning audio and whether a decoder was used."""
        total_frames = sum(frame.repeat for frame in self.frames)
        frame_samples = int(self.output_rate * (self.frame_duration_ms / 1000.0))

        decoder_available = IMBEDecoderNeo.is_available()
        if not decoder_available:
            logger.warning("mbelib-neo not available; rendering silence for message spec audio")

        decoder: IMBEDecoderNeo | None = None
        audio_chunks: list[np.ndarray] = []

        try:
            if decoder_available:
                decoder = IMBEDecoderNeo(output_rate=self.output_rate)
                decoder.start()

            for frame in self.frames:
                for _ in range(frame.repeat):
                    if decoder is not None:
                        audio = decoder.decode_frame(frame.data)
                        if audio is None:
                            audio = decoder.decode_silence()
                    else:
                        audio = np.zeros(frame_samples, dtype=np.float32)
                    audio_chunks.append(audio)

                    if frame.pad_silence_ms > 0:
                        pad_samples = int(self.output_rate * (frame.pad_silence_ms / 1000.0))
                        if pad_samples > 0:
                            audio_chunks.append(np.zeros(pad_samples, dtype=np.float32))
        finally:
            if decoder is not None:
                decoder.stop()

        if not audio_chunks:
            return np.zeros(0, dtype=np.float32), decoder_available

        return np.concatenate(audio_chunks), decoder_available


@dataclass(frozen=True)
class EncodedMessage:
    """Result of rendering a message spec."""

    payload_bytes: bytes
    audio: np.ndarray
    sample_rate: int
    used_decoder: bool


def load_message_spec(path: Path) -> MessageSpec:
    """Load a message spec from a JSON or YAML file."""
    if not path.exists():
        raise FileNotFoundError(path)

    content = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError:
        data = json.loads(content)

    if not isinstance(data, dict):
        raise ValueError("Message spec root must be an object")

    return MessageSpec.from_mapping(data)


def encode_message(path: Path) -> EncodedMessage:
    """Load and render a message spec."""
    spec = load_message_spec(path)
    payload_bytes = spec.encoded_bytes()
    audio, used_decoder = spec.render_audio()
    return EncodedMessage(payload_bytes=payload_bytes, audio=audio, sample_rate=spec.output_rate, used_decoder=used_decoder)


def pcm16le_bytes(audio: np.ndarray) -> bytes:
    """Convert float PCM [-1,1] to 16-bit little endian bytes."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write PCM audio to a mono WAV file."""
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    pcm_bytes = pcm16le_bytes(audio)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def _decode_frame_bytes(data_field: Any) -> bytes:
    """Decode a frame from supported serialized forms."""
    if isinstance(data_field, (bytes, bytearray)):
        return bytes(data_field)
    if isinstance(data_field, str):
        stripped = data_field.strip()
        try:
            return bytes.fromhex(stripped)
        except ValueError:
            pass
        try:
            return base64.b64decode(stripped, validate=True)
        except Exception:
            raise ValueError("Frame string must be valid hex or base64") from None
    if isinstance(data_field, list):
        return bytes(int(v) & 0xFF for v in data_field)
    if isinstance(data_field, int):
        return bytes([data_field & 0xFF])
    raise ValueError(f"Unsupported frame encoding type: {type(data_field)}")
