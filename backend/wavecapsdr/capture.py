from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

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
    # Store (queue, loop) to support cross-event-loop broadcasting safely
    _audio_sinks: Set[Tuple[asyncio.Queue[bytes], asyncio.AbstractEventLoop]] = field(
        default_factory=set
    )

    def start(self) -> None:
        self.state = "running"

    def stop(self) -> None:
        self.state = "stopped"

    async def subscribe_audio(self) -> asyncio.Queue[bytes]:
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=8)
        loop = asyncio.get_running_loop()
        self._audio_sinks.add((q, loop))
        return q

    def unsubscribe(self, q: asyncio.Queue[bytes]) -> None:
        for item in list(self._audio_sinks):
            if item[0] is q:
                self._audio_sinks.discard(item)

    async def _broadcast(self, payload: bytes) -> None:
        if not self._audio_sinks:
            return
        current_loop = None
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        for (q, loop) in list(self._audio_sinks):
            if current_loop is loop:
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
            else:
                # Schedule put on the queue's owning loop to avoid cross-loop errors
                def _try_put() -> None:
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

                try:
                    loop.call_soon_threadsafe(_try_put)
                except Exception:
                    # If loop is closed or unavailable, drop sink
                    try:
                        self._audio_sinks.discard((q, loop))
                    except Exception:
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
    _thread: Optional[threading.Thread] = None
    _iq_sinks: Set[Tuple[asyncio.Queue[bytes], asyncio.AbstractEventLoop]] = field(
        default_factory=set
    )
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _channels: Dict[str, Channel] = field(default_factory=dict)

    def create_channel(self, chan: Channel) -> None:
        self._channels[chan.cfg.id] = chan

    def remove_channel(self, chan_id: str) -> None:
        self._channels.pop(chan_id, None)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.state = "running"
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_thread, name=f"Capture-{self.cfg.id}", daemon=True)
        self._thread.start()

    async def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        try:
            self.device.close()
        except Exception:
            pass
        self.state = "stopped"

    async def subscribe_iq(self) -> asyncio.Queue[bytes]:
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=8)
        loop = asyncio.get_running_loop()
        self._iq_sinks.add((q, loop))
        return q

    def unsubscribe(self, q: asyncio.Queue[bytes]) -> None:
        for item in list(self._iq_sinks):
            if item[0] is q:
                self._iq_sinks.discard(item)

    async def _broadcast_iq(self, payload: bytes) -> None:
        if not self._iq_sinks:
            return
        current_loop = None
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        for (q, loop) in list(self._iq_sinks):
            if current_loop is loop:
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
            else:
                def _try_put() -> None:
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

                try:
                    loop.call_soon_threadsafe(_try_put)
                except Exception:
                    try:
                        self._iq_sinks.discard((q, loop))
                    except Exception:
                        pass

    def _run_thread(self) -> None:
        # Configure device and start streaming
        try:
            self.device.configure(
                center_hz=self.cfg.center_hz,
                sample_rate=self.cfg.sample_rate,
                gain=self.cfg.gain,
                bandwidth=self.cfg.bandwidth,
                ppm=self.cfg.ppm,
            )
            self._stream = self.device.start_stream()
        except Exception:
            return
        chunk = 4096
        while not self._stop_event.is_set():
            try:
                samples, _ov = self._stream.read(chunk)
            except Exception:
                break
            if samples.size == 0:
                # Light backoff to avoid busy-spin
                try:
                    threading.Event().wait(0.001)
                except Exception:
                    pass
                continue
            # Broadcast IQ to subscribers (schedule on their loops)
            # Use asyncio.run() is not allowed here; rely on _broadcast_iq scheduling
            try:
                # Schedule broadcast onto any event loop via call_soon_threadsafe
                asyncio.get_event_loop
            except Exception:
                pass
            # Reuse asyncio to schedule coroutine execution in a thread-safe manner
            # by using the same logic inside _broadcast_iq (which uses call_soon_threadsafe)
            # Invoke it in a synchronous context using asyncio.run in a dedicated loop is heavy.
            # Instead, inline the same logic here to avoid requiring a loop in this thread.
            #
            # Duplicate minimal logic of _broadcast_iq without awaiting.
            payload = pack_iq16(samples)
            for (q, loop) in list(self._iq_sinks):
                def _try_put() -> None:
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
                try:
                    loop.call_soon_threadsafe(_try_put)
                except Exception:
                    try:
                        self._iq_sinks.discard((q, loop))
                    except Exception:
                        pass
            # Dispatch to channels by scheduling their processing on their loops
            chans = list(self._channels.values())
            for ch in chans:
                # Capture channel reference vars for closure
                def _process(ch=ch, samples=samples):
                    asyncio.create_task(ch.process_iq_chunk(samples, self.cfg.sample_rate))
                # Try to schedule on any one sink loop if available, otherwise skip
                target_loop = None
                try:
                    # Prefer a sink loop if exists; else, try any loop from audio sinks
                    if self._iq_sinks:
                        target_loop = next(iter(self._iq_sinks))[1]
                    elif ch._audio_sinks:
                        target_loop = next(iter(ch._audio_sinks))[1]
                except Exception:
                    target_loop = None
                if target_loop is not None:
                    try:
                        target_loop.call_soon_threadsafe(_process)
                    except Exception:
                        pass


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
