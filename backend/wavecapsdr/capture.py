from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

import numpy as np

from .config import AppConfig
from .devices.base import Device, DeviceDriver, StreamHandle
from .dsp.fm import wbfm_demod


def pack_iq16(samples: np.ndarray) -> bytes:
    if samples.size == 0:
        return b""
    x = samples.astype(np.complex64, copy=False)
    re = np.clip(np.real(x), -1.0, 1.0)
    im = np.clip(np.imag(x), -1.0, 1.0)
    interleaved = np.empty(re.size * 2, dtype=np.int16)
    interleaved[0::2] = (re * 32767.0).astype(np.int16)
    interleaved[1::2] = (im * 32767.0).astype(np.int16)
    return interleaved.tobytes()


def pack_pcm16(samples: np.ndarray) -> bytes:
    if samples.size == 0:
        return b""
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def freq_shift(iq: np.ndarray, offset_hz: float, sample_rate: int) -> np.ndarray:
    if offset_hz == 0.0 or iq.size == 0:
        return iq
    n = np.arange(iq.shape[0], dtype=np.float32)
    ph = np.exp(-1j * 2.0 * np.pi * (offset_hz / float(sample_rate)) * n).astype(np.complex64)
    return (iq.astype(np.complex64, copy=False) * ph).astype(np.complex64)


@dataclass
class ChannelConfig:
    id: str
    capture_id: str
    mode: str  # "wbfm" | "am" | "ssb" (initially wbfm only)
    offset_hz: float = 0.0
    audio_rate: int = 48_000
    squelch_db: Optional[float] = None


@dataclass
class Channel:
    cfg: ChannelConfig
    state: str = "created"
    _audio_sinks: Set[asyncio.Queue[bytes]] = field(default_factory=set)

    def start(self) -> None:
        self.state = "running"

    def stop(self) -> None:
        self.state = "stopped"

    async def subscribe_audio(self) -> asyncio.Queue[bytes]:
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=8)
        self._audio_sinks.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[bytes]) -> None:
        self._audio_sinks.discard(q)

    async def _broadcast(self, payload: bytes) -> None:
        if not self._audio_sinks:
            return
        for q in list(self._audio_sinks):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                try:
                    _ = q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    pass

    async def process_iq_chunk(self, iq: np.ndarray, sample_rate: int) -> None:
        if self.state != "running":
            return
        if self.cfg.mode == "wbfm":
            base = freq_shift(iq, self.cfg.offset_hz, sample_rate)
            audio = wbfm_demod(base, sample_rate, self.cfg.audio_rate)
            await self._broadcast(pack_pcm16(audio))
        else:
            # Unknown mode: ignore for now
            return


@dataclass
class CaptureConfig:
    id: str
    device_id: str
    center_hz: float
    sample_rate: int
    gain: Optional[float] = None
    bandwidth: Optional[float] = None
    ppm: Optional[float] = None


@dataclass
class Capture:
    cfg: CaptureConfig
    device: Device
    state: str = "created"
    _stream: Optional[StreamHandle] = None
    _task: Optional[asyncio.Task[None]] = None
    _iq_sinks: Set[asyncio.Queue[bytes]] = field(default_factory=set)
    _stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    _channels: Dict[str, Channel] = field(default_factory=dict)

    def create_channel(self, chan: Channel) -> None:
        self._channels[chan.cfg.id] = chan

    def remove_channel(self, chan_id: str) -> None:
        self._channels.pop(chan_id, None)

    def start(self) -> None:
        if self._task is not None:
            return
        self.state = "running"
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            await self._task
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        self.device.close()
        self.state = "stopped"

    async def subscribe_iq(self) -> asyncio.Queue[bytes]:
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=8)
        self._iq_sinks.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[bytes]) -> None:
        self._iq_sinks.discard(q)

    async def _broadcast_iq(self, payload: bytes) -> None:
        if not self._iq_sinks:
            return
        for q in list(self._iq_sinks):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                try:
                    _ = q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    pass

    async def _run_loop(self) -> None:
        # Configure device and start streaming
        self.device.configure(
            center_hz=self.cfg.center_hz,
            sample_rate=self.cfg.sample_rate,
            gain=self.cfg.gain,
            bandwidth=self.cfg.bandwidth,
            ppm=self.cfg.ppm,
        )
        self._stream = self.device.start_stream()
        chunk = 4096
        while not self._stop_event.is_set():
            # Offload blocking reads to a thread
            samples, _ov = await asyncio.to_thread(self._stream.read, chunk)
            if samples.size == 0:
                await asyncio.sleep(0.001)
                continue
            await self._broadcast_iq(pack_iq16(samples))
            # Dispatch to channels
            chans = list(self._channels.values())
            for ch in chans:
                await ch.process_iq_chunk(samples, self.cfg.sample_rate)


class CaptureManager:
    def __init__(self, cfg: AppConfig, driver: DeviceDriver):
        self._cfg = cfg
        self._driver = driver
        self._captures: Dict[str, Capture] = {}
        self._channels: Dict[str, Channel] = {}
        self._next_cap_id = 1
        self._next_chan_id = 1

    def list_devices(self) -> list[dict]:
        return [d.__dict__ for d in self._driver.enumerate()]

    def list_captures(self) -> list[Capture]:
        return list(self._captures.values())

    def get_capture(self, cid: str) -> Optional[Capture]:
        return self._captures.get(cid)

    def create_capture(
        self,
        device_id: Optional[str],
        center_hz: float,
        sample_rate: int,
        gain: Optional[float] = None,
        bandwidth: Optional[float] = None,
        ppm: Optional[float] = None,
    ) -> Capture:
        cid = f"c{self._next_cap_id}"
        self._next_cap_id += 1
        dev = self._driver.open(device_id)
        cfg = CaptureConfig(
            id=cid,
            device_id=device_id or "device0",
            center_hz=center_hz,
            sample_rate=sample_rate,
            gain=gain,
            bandwidth=bandwidth,
            ppm=ppm,
        )
        cap = Capture(cfg=cfg, device=dev)
        self._captures[cid] = cap
        return cap

    async def delete_capture(self, cid: str) -> None:
        cap = self._captures.pop(cid, None)
        if cap is not None:
            await cap.stop()
        # Remove channels owned by this capture
        for k in list(self._channels.keys()):
            if self._channels[k].cfg.capture_id == cid:
                self._channels.pop(k, None)

    def list_channels(self, cid: Optional[str] = None) -> list[Channel]:
        if cid is None:
            return list(self._channels.values())
        return [ch for ch in self._channels.values() if ch.cfg.capture_id == cid]

    def get_channel(self, chan_id: str) -> Optional[Channel]:
        return self._channels.get(chan_id)

    def create_channel(
        self,
        cid: str,
        mode: str,
        offset_hz: float = 0.0,
        audio_rate: Optional[int] = None,
        squelch_db: Optional[float] = None,
    ) -> Channel:
        cap = self.get_capture(cid)
        if cap is None:
            raise ValueError("Capture not found")
        chan_id = f"ch{self._next_chan_id}"
        self._next_chan_id += 1
        cfg = ChannelConfig(
            id=chan_id,
            capture_id=cid,
            mode=mode,
            offset_hz=offset_hz,
            audio_rate=audio_rate or self._cfg.stream.default_audio_rate,
            squelch_db=squelch_db,
        )
        ch = Channel(cfg=cfg)
        cap.create_channel(ch)
        self._channels[chan_id] = ch
        return ch

    def delete_channel(self, chan_id: str) -> None:
        ch = self._channels.pop(chan_id, None)
        if ch is None:
            return
        cap = self.get_capture(ch.cfg.capture_id)
        if cap is not None:
            cap.remove_channel(chan_id)

