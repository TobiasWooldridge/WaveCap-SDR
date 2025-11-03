from __future__ import annotations

import asyncio
from enum import Enum
import functools
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set, Tuple, TypeVar

import numpy as np

from .config import AppConfig
from .devices.base import Device, DeviceDriver, StreamHandle
from .dsp.fm import wbfm_demod


F = TypeVar('F', bound=Callable[..., Any])


class CaptureState(Enum):
    """Capture lifecycle states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"


def with_retry(max_attempts: int = 3, backoff_factor: float = 2.0) -> Callable[[F], F]:
    """Decorator to add retry logic with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts (default: 3)
        backoff_factor: Multiplier for delay between attempts (default: 2.0)

    Returns:
        Decorated function that will retry on failure
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            delay = 0.5  # Start with 0.5 second delay

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        print(
                            f"[RETRY] {func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}",
                            flush=True
                        )
                        time.sleep(delay)
                        delay *= backoff_factor

            # All attempts failed
            print(
                f"[ERROR] {func.__name__} failed after {max_attempts} attempts",
                flush=True
            )
            raise last_exception  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]
    return decorator


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


def pack_f32(samples: np.ndarray) -> bytes:
    """Pack samples as 32-bit float (little-endian)."""
    if samples.size == 0:
        return b""
    return np.clip(samples, -1.0, 1.0).astype(np.float32).tobytes()


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
    # Store (queue, loop, format) to support cross-event-loop broadcasting safely
    _audio_sinks: Set[Tuple[asyncio.Queue[bytes], asyncio.AbstractEventLoop, str]] = field(
        default_factory=set
    )

    def start(self) -> None:
        self.state = "running"

    def stop(self) -> None:
        self.state = "stopped"

    async def subscribe_audio(self, format: str = "pcm16") -> asyncio.Queue[bytes]:
        """Subscribe to audio stream with specified format.

        Args:
            format: Audio format - "pcm16" (16-bit signed PCM) or "f32" (32-bit float)
        """
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=8)
        loop = asyncio.get_running_loop()
        self._audio_sinks.add((q, loop, format))
        return q

    def unsubscribe(self, q: asyncio.Queue[bytes]) -> None:
        for item in list(self._audio_sinks):
            if item[0] is q:
                self._audio_sinks.discard(item)

    async def _broadcast(self, audio: np.ndarray) -> None:
        """Broadcast audio to all subscribers, converting to their requested format."""
        if not self._audio_sinks:
            return
        current_loop = None
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        for (q, loop, fmt) in list(self._audio_sinks):
            # Convert audio to requested format
            if fmt == "f32":
                payload = pack_f32(audio)
            else:  # Default to pcm16
                payload = pack_pcm16(audio)

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
                        self._audio_sinks.discard((q, loop, fmt))
                    except Exception:
                        pass

    async def process_iq_chunk(self, iq: np.ndarray, sample_rate: int) -> None:
        if self.state != "running":
            return
        if self.cfg.mode == "wbfm":
            base = freq_shift(iq, self.cfg.offset_hz, sample_rate)
            audio = wbfm_demod(base, sample_rate, self.cfg.audio_rate)

            # Apply squelch if configured
            if self.cfg.squelch_db is not None and audio.size > 0:
                # Calculate signal power in dB
                power = np.mean(audio ** 2)
                power_db = 10.0 * np.log10(power + 1e-10)  # Add small value to avoid log(0)

                # Mute audio if below threshold
                if power_db < self.cfg.squelch_db:
                    audio = np.zeros_like(audio)

            await self._broadcast(audio)
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
    antenna: Optional[str] = None
    # SoapySDR advanced features
    device_settings: Dict[str, Any] = field(default_factory=dict)
    element_gains: Dict[str, float] = field(default_factory=dict)
    stream_format: Optional[str] = None
    dc_offset_auto: bool = True
    iq_balance_auto: bool = True


@dataclass
class Capture:
    cfg: CaptureConfig
    device: Device
    state: str = "stopped"  # Use string for backwards compatibility with API
    antenna: Optional[str] = None  # Actual antenna in use
    error_message: Optional[str] = None  # Error message if state is "failed"
    _stream: Optional[StreamHandle] = None
    _thread: Optional[threading.Thread] = None
    _health_monitor: Optional[threading.Thread] = None
    _iq_sinks: Set[Tuple[asyncio.Queue[bytes], asyncio.AbstractEventLoop]] = field(
        default_factory=set
    )
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _channels: Dict[str, Channel] = field(default_factory=dict)
    # Retry tracking (inspired by OpenWebRX)
    _retry_count: int = 0
    _max_retries: int = 10  # OpenWebRX uses 10
    _retry_delay: float = 15.0  # OpenWebRX uses 15 seconds
    _retry_timer: Optional[threading.Timer] = None
    _auto_restart_enabled: bool = True  # Enable automatic restart on failure
    _last_run_time: float = 0.0  # Track when the thread was last running

    def create_channel(self, chan: Channel) -> None:
        self._channels[chan.cfg.id] = chan

    def remove_channel(self, chan_id: str) -> None:
        self._channels.pop(chan_id, None)

    def _cancel_retry_timer(self) -> None:
        """Cancel any pending retry timer."""
        if self._retry_timer is not None:
            self._retry_timer.cancel()
            self._retry_timer = None

    def _schedule_restart(self) -> None:
        """Schedule automatic restart after delay (OpenWebRX pattern)."""
        if not self._auto_restart_enabled:
            return

        self._retry_count += 1
        if self._retry_count > self._max_retries:
            print(
                f"[ERROR] Capture {self.cfg.id} failed after {self._max_retries} retries, giving up",
                flush=True
            )
            self.state = "failed"
            return

        print(
            f"[RETRY] Capture {self.cfg.id} will retry in {self._retry_delay}s (attempt {self._retry_count}/{self._max_retries})",
            flush=True
        )
        self._cancel_retry_timer()
        self._retry_timer = threading.Timer(self._retry_delay, self.start)
        self._retry_timer.daemon = True
        self._retry_timer.start()

    def _health_monitor_thread(self) -> None:
        """Monitor capture thread health and restart if crashed (OpenWebRX pattern)."""
        while not self._stop_event.is_set():
            try:
                # Check if thread is alive
                if self._thread is not None and not self._thread.is_alive():
                    # Thread died unexpectedly
                    if self.state == "running":
                        print(
                            f"[WARNING] Capture {self.cfg.id} thread died unexpectedly, scheduling restart",
                            flush=True
                        )
                        self._schedule_restart()
                        return  # Health monitor exits, will be restarted by new start()
                # Update last run time
                if self._thread is not None and self._thread.is_alive():
                    self._last_run_time = time.time()
                # Sleep for a bit before next check
                self._stop_event.wait(1.0)  # Check every second
            except Exception as e:
                print(f"[ERROR] Health monitor error: {e}", flush=True)
                break

    def start(self) -> None:
        """Start capture with error handling and automatic retry."""
        if self._thread is not None and self._thread.is_alive():
            return

        # Cancel any pending retry timer
        self._cancel_retry_timer()

        self.state = "starting"
        self.error_message = None
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_thread, name=f"Capture-{self.cfg.id}", daemon=True)
        self._thread.start()

        # Start health monitor if not already running
        if self._health_monitor is None or not self._health_monitor.is_alive():
            self._health_monitor = threading.Thread(
                target=self._health_monitor_thread,
                name=f"HealthMon-{self.cfg.id}",
                daemon=True
            )
            self._health_monitor.start()

    async def stop(self) -> None:
        """Stop capture with graceful shutdown (OpenWebRX pattern)."""
        # Disable auto-restart while stopping
        self._auto_restart_enabled = False
        self._cancel_retry_timer()

        self.state = "stopping"
        self._stop_event.set()

        # Wait for threads to finish gracefully
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                print(f"[WARNING] Capture {self.cfg.id} thread did not stop gracefully", flush=True)
            self._thread = None

        if self._health_monitor is not None:
            self._health_monitor.join(timeout=1.0)
            self._health_monitor = None

        # Close stream
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        # Note: We do NOT close the device here - the device should stay open
        # for the lifetime of the Capture. We only close the stream.
        # The device will be closed when the Capture is deleted.
        self.state = "stopped"

        # Re-enable auto-restart for next start()
        self._auto_restart_enabled = True

    async def reconfigure(
        self,
        center_hz: Optional[float] = None,
        sample_rate: Optional[int] = None,
        gain: Optional[float] = None,
        bandwidth: Optional[float] = None,
        ppm: Optional[float] = None,
        antenna: Optional[str] = None,
    ) -> None:
        """Reconfigure capture, using hot reconfiguration when possible.

        If only hot-reconfigurable parameters are changed (center_hz, gain,
        bandwidth, ppm), the capture will be updated without restarting.
        If sample_rate or antenna changes, a full restart is required.
        """
        # Update config
        if center_hz is not None:
            self.cfg.center_hz = center_hz
        if sample_rate is not None:
            self.cfg.sample_rate = sample_rate
        if gain is not None:
            self.cfg.gain = gain
        if bandwidth is not None:
            self.cfg.bandwidth = bandwidth
        if ppm is not None:
            self.cfg.ppm = ppm
        if antenna is not None:
            self.cfg.antenna = antenna

        # Check if we need full restart
        needs_restart = sample_rate is not None or antenna is not None

        was_running = self.state == "running"

        if needs_restart and was_running:
            # Full restart required
            await self.stop()
            self.start()
        elif was_running:
            # Try hot reconfiguration
            try:
                self.device.reconfigure_running(
                    center_hz=center_hz,
                    gain=gain,
                    bandwidth=bandwidth,
                    ppm=ppm,
                )
                self.error_message = None
            except Exception as e:
                # Hot reconfiguration failed, do full restart
                print(f"[WARNING] Hot reconfiguration failed: {e}, restarting capture", flush=True)
                await self.stop()
                self.start()

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
        # Configure device and start streaming with retry logic
        @with_retry(max_attempts=3, backoff_factor=2.0)
        def _configure_and_start() -> StreamHandle:
            self.device.configure(
                center_hz=self.cfg.center_hz,
                sample_rate=self.cfg.sample_rate,
                gain=self.cfg.gain,
                bandwidth=self.cfg.bandwidth,
                ppm=self.cfg.ppm,
                antenna=self.cfg.antenna,
                device_settings=self.cfg.device_settings,
                element_gains=self.cfg.element_gains,
                stream_format=self.cfg.stream_format,
                dc_offset_auto=self.cfg.dc_offset_auto,
                iq_balance_auto=self.cfg.iq_balance_auto,
            )
            stream = self.device.start_stream()
            # Store the actual antenna being used
            self.antenna = self.device.get_antenna()
            return stream

        try:
            self._stream = _configure_and_start()
            # Successfully started!
            self.state = "running"
            self._retry_count = 0  # Reset retry counter on success
        except Exception as e:
            # Failed to start - schedule automatic restart
            self.error_message = f"Failed to start capture: {str(e)}"
            print(f"[ERROR] Capture {self.cfg.id} failed to start: {e}", flush=True)
            self._schedule_restart()
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
                # Skip if channel has no audio sinks
                if not ch._audio_sinks:
                    continue

                # Get event loop from an audio sink
                target_loop = None
                try:
                    # Get loop from first audio sink (tuple is (q, loop, format))
                    target_loop = next(iter(ch._audio_sinks))[1]
                except (StopIteration, IndexError):
                    # Try to get from IQ sinks as fallback
                    try:
                        if self._iq_sinks:
                            target_loop = next(iter(self._iq_sinks))[1]
                    except Exception:
                        pass

                if target_loop is not None:
                    # Capture channel reference vars for closure
                    coro = ch.process_iq_chunk(samples.copy(), self.cfg.sample_rate)
                    try:
                        fut = asyncio.run_coroutine_threadsafe(coro, target_loop)
                        # Don't wait for result, just schedule it
                    except Exception as e:
                        import sys
                        print(f"Error scheduling audio processing: {e}", file=sys.stderr, flush=True)


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
        antenna: Optional[str] = None,
        device_settings: Optional[Dict[str, Any]] = None,
        element_gains: Optional[Dict[str, float]] = None,
        stream_format: Optional[str] = None,
        dc_offset_auto: bool = True,
        iq_balance_auto: bool = True,
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
            antenna=antenna,
            device_settings=device_settings or {},
            element_gains=element_gains or {},
            stream_format=stream_format,
            dc_offset_auto=dc_offset_auto,
            iq_balance_auto=iq_balance_auto,
        )
        cap = Capture(cfg=cfg, device=dev)
        self._captures[cid] = cap
        return cap

    async def delete_capture(self, cid: str) -> None:
        cap = self._captures.pop(cid, None)
        if cap is not None:
            await cap.stop()
            # Close the device when deleting the capture
            try:
                cap.device.close()
            except Exception:
                pass
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
