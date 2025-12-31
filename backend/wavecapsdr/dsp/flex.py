"""FLEX pager decoding via multimon-ng.

Uses an external multimon-ng process (FLEX demod) and feeds it 22.05 kHz
PCM audio, matching the rtl_fm + multimon-ng reference flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import contextlib
import json
import logging
import queue
import shutil
import subprocess
import threading
import time
from typing import Any

import numpy as np

from wavecapsdr.typing import NDArrayFloat
from .fm import resample_poly


logger = logging.getLogger(__name__)

FLEX_SAMPLE_RATE = 22050


@dataclass
class FlexMessage:
    """Decoded FLEX pager message."""

    capcode: int
    message_type: str
    message: str
    timestamp: float = field(default_factory=time.time)
    baud_rate: int | None = None
    levels: int | None = None
    phase: str | None = None
    cycle_number: int | None = None
    frame_number: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "capcode": self.capcode,
            "messageType": self.message_type,
            "message": self.message,
            "timestamp": self.timestamp,
            "baudRate": self.baud_rate,
            "levels": self.levels,
            "phase": self.phase,
            "cycleNumber": self.cycle_number,
            "frameNumber": self.frame_number,
        }


def parse_flex_json_message(payload: dict[str, Any]) -> FlexMessage | None:
    """Parse a multimon-ng FLEX JSON payload into a FlexMessage."""
    demod_name = str(payload.get("demod_name", ""))
    if not demod_name.startswith("flex_"):
        return None

    message_type = "unknown"
    if demod_name == "flex_alphanumeric":
        message_type = "alpha"
    elif demod_name == "flex_numeric":
        message_type = "numeric"
    elif demod_name == "flex_tone_only":
        message_type = "tone"

    message = ""
    if message_type == "tone":
        message = str(payload.get("tone_long") or payload.get("tone") or "")
    else:
        message = str(payload.get("message") or "")

    capcode_raw = payload.get("capcode")
    if capcode_raw is None:
        return None
    try:
        capcode = int(capcode_raw)
    except (TypeError, ValueError):
        return None

    def _to_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    baud_rate = _to_int(payload.get("sync_baud"))
    levels = _to_int(payload.get("sync_level"))
    cycle_number = _to_int(payload.get("cycle_number"))
    frame_number = _to_int(payload.get("frame_number"))
    phase = payload.get("phase_number")
    phase_str = str(phase) if phase is not None else None

    return FlexMessage(
        capcode=capcode,
        message_type=message_type,
        message=message,
        baud_rate=baud_rate,
        levels=levels,
        phase=phase_str,
        cycle_number=cycle_number,
        frame_number=frame_number,
    )


class FlexDecoder:
    """Decode FLEX pager messages using multimon-ng."""

    def __init__(
        self,
        multimon_path: str = "multimon-ng",
        target_rate: int = FLEX_SAMPLE_RATE,
        audio_queue_size: int = 8,
        message_queue_size: int = 200,
    ) -> None:
        self._multimon_path = multimon_path
        self._target_rate = target_rate
        self._audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=audio_queue_size)
        self._message_queue: queue.Queue[FlexMessage] = queue.Queue(maxsize=message_queue_size)
        self._process: subprocess.Popen[bytes] | None = None
        self._stop_event = threading.Event()
        self._writer_thread: threading.Thread | None = None
        self._reader_thread: threading.Thread | None = None
        self._failed = False

    @property
    def target_rate(self) -> int:
        return self._target_rate

    def start(self) -> None:
        if self._process is not None or self._failed:
            return

        if shutil.which(self._multimon_path) is None:
            logger.warning("FLEX decoder disabled: multimon-ng not found in PATH")
            self._failed = True
            return

        try:
            self._process = subprocess.Popen(
                [
                    self._multimon_path,
                    "--json",
                    "-a",
                    "FLEX",
                    "-t",
                    "raw",
                    "/dev/stdin",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except Exception as exc:
            logger.error("Failed to start multimon-ng for FLEX decoding: %s", exc)
            self._failed = True
            self._process = None
            return

        self._stop_event.clear()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._writer_thread.start()
        self._reader_thread.start()

    def stop(self) -> None:
        if self._process is None:
            return

        self._stop_event.set()
        with contextlib.suppress(queue.Full):
            self._audio_queue.put_nowait(b"")

        proc = self._process
        self._process = None

        with contextlib.suppress(Exception):
            if proc.stdin:
                proc.stdin.close()
        with contextlib.suppress(Exception):
            proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=1.0)

    def feed(self, audio: NDArrayFloat, input_rate: int) -> None:
        """Feed demodulated audio into the FLEX decoder."""
        if self._failed:
            return
        if self._process is None:
            self.start()
        if self._process is None:
            return
        if audio.size == 0:
            return

        samples = audio
        if input_rate != self._target_rate:
            samples = resample_poly(samples, input_rate, self._target_rate)
        if samples.size == 0:
            return

        pcm = np.clip(samples, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype(np.int16)
        data = pcm16.tobytes()
        with contextlib.suppress(queue.Full):
            self._audio_queue.put_nowait(data)

    def drain_messages(self) -> list[FlexMessage]:
        messages: list[FlexMessage] = []
        while True:
            try:
                messages.append(self._message_queue.get_nowait())
            except queue.Empty:
                break
        return messages

    def _writer_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                data = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if not data:
                continue
            proc = self._process
            if proc is None or proc.stdin is None:
                return
            try:
                proc.stdin.write(data)
                proc.stdin.flush()
            except Exception:
                return

    def _reader_loop(self) -> None:
        proc = self._process
        if proc is None or proc.stdout is None:
            return
        while not self._stop_event.is_set():
            line = proc.stdout.readline()
            if not line:
                return
            try:
                decoded = line.decode("utf-8", errors="replace").strip()
            except Exception:
                continue
            if not decoded:
                continue
            try:
                payload = json.loads(decoded)
            except json.JSONDecodeError:
                continue
            msg = parse_flex_json_message(payload)
            if msg is None:
                continue
            with contextlib.suppress(queue.Full):
                self._message_queue.put_nowait(msg)
