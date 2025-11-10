from __future__ import annotations

import asyncio
from enum import Enum
import functools
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set, Tuple, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

from .config import AppConfig
from .devices.base import Device, DeviceDriver, StreamHandle
from .dsp.fm import wbfm_demod, nbfm_demod
from .encoders import create_encoder, AudioEncoder
from .decoders.p25 import P25Decoder
from .decoders.dmr import DMRDecoder


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
    name: Optional[str] = None  # User-provided name
    auto_name: Optional[str] = None  # Auto-generated contextual name


@dataclass
class Channel:
    cfg: ChannelConfig
    state: str = "created"
    # Store (queue, loop, format) to support cross-event-loop broadcasting safely
    _audio_sinks: Set[Tuple[asyncio.Queue[bytes], asyncio.AbstractEventLoop, str]] = field(
        default_factory=set
    )
    # Encoders for compressed formats (mp3, opus, aac) - created on demand
    _encoders: Dict[str, AudioEncoder] = field(default_factory=dict)
    # Subscriber counts for encoded formats
    _encoder_subscribers: Dict[str, int] = field(default_factory=dict)
    # Signal strength metrics (calculated server-side)
    signal_power_db: Optional[float] = None  # Current signal power in dB
    rssi_db: Optional[float] = None  # Received Signal Strength Indicator from IQ
    snr_db: Optional[float] = None  # Signal-to-Noise Ratio estimate
    # Digital voice decoders (lazily initialized)
    _p25_decoder: Optional[P25Decoder] = None
    _dmr_decoder: Optional[DMRDecoder] = None

    def start(self) -> None:
        self.state = "running"

    def stop(self) -> None:
        self.state = "stopped"

    def update_signal_metrics(self, iq: np.ndarray, sample_rate: int) -> None:
        """Calculate signal metrics from IQ samples (server-side, no audio needed)."""
        if self.state != "running" or iq.size == 0:
            return

        # Frequency shift to channel offset
        shifted_iq = freq_shift(iq, self.cfg.offset_hz, sample_rate)

        # Calculate RSSI (power of IQ samples in dB)
        power = np.mean(np.abs(shifted_iq) ** 2)
        self.rssi_db = float(10.0 * np.log10(power + 1e-10))

        # Estimate SNR using percentile-based noise floor estimation
        # Sort by magnitude and use lower percentile as noise estimate
        magnitudes = np.abs(shifted_iq)
        noise_floor = np.percentile(magnitudes, 10)  # Bottom 10% as noise
        signal_peak = np.percentile(magnitudes, 90)  # Top 10% as signal

        noise_power = noise_floor ** 2
        signal_power = signal_peak ** 2

        if noise_power > 1e-10:
            self.snr_db = float(10.0 * np.log10(signal_power / noise_power))
        else:
            self.snr_db = None

    async def subscribe_audio(self, format: str = "pcm16") -> asyncio.Queue[bytes]:
        """Subscribe to audio stream with specified format.

        Args:
            format: Audio format - "pcm16", "f32", "mp3", "opus", or "aac"
        """
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=32)
        loop = asyncio.get_running_loop()

        # Handle encoded formats (mp3, opus, aac)
        if format in ("mp3", "opus", "aac"):
            # Start encoder if this is the first subscriber
            if format not in self._encoders:
                logger.info(f"Channel {self.cfg.id}: Starting {format} encoder")
                encoder = create_encoder(format, sample_rate=self.cfg.audio_rate)
                self._encoders[format] = encoder
                await encoder.start()
                # Start encoder reader task
                asyncio.create_task(self._read_encoder_output(format))

            # Increment subscriber count
            self._encoder_subscribers[format] = self._encoder_subscribers.get(format, 0) + 1
            logger.info(f"Channel {self.cfg.id}: Encoder subscriber added, format={format}, subscribers={self._encoder_subscribers[format]}")

        self._audio_sinks.add((q, loop, format))
        logger.info(f"Channel {self.cfg.id}: Audio subscriber added, format={format}, total_subscribers={len(self._audio_sinks)}")
        return q

    def unsubscribe(self, q: asyncio.Queue[bytes]) -> None:
        for item in list(self._audio_sinks):
            if item[0] is q:
                fmt = item[2]
                self._audio_sinks.discard(item)
                logger.info(f"Channel {self.cfg.id}: Audio subscriber removed, format={fmt}, total_subscribers={len(self._audio_sinks)}")

                # Stop encoder if this was the last subscriber for an encoded format
                if fmt in ("mp3", "opus", "aac"):
                    self._encoder_subscribers[fmt] = max(0, self._encoder_subscribers.get(fmt, 1) - 1)
                    if self._encoder_subscribers[fmt] == 0 and fmt in self._encoders:
                        logger.info(f"Channel {self.cfg.id}: Stopping {fmt} encoder (no more subscribers)")
                        encoder = self._encoders.pop(fmt)
                        asyncio.create_task(encoder.stop())
                        del self._encoder_subscribers[fmt]

    async def _read_encoder_output(self, format: str) -> None:
        """Read encoded output and broadcast to subscribers of this format."""
        encoder = self._encoders.get(format)
        if not encoder:
            return

        try:
            while format in self._encoders:
                # Get encoded data from encoder
                data = await encoder.get_encoded()

                # Broadcast to all subscribers of this format
                for (q, loop, fmt) in list(self._audio_sinks):
                    if fmt != format:
                        continue

                    try:
                        current_loop = asyncio.get_running_loop()
                        if current_loop is loop:
                            q.put_nowait(data)
                        else:
                            loop.call_soon_threadsafe(q.put_nowait, data)
                    except asyncio.QueueFull:
                        # Drop packet if queue is full
                        pass
                    except Exception as e:
                        logger.error(f"Error broadcasting {format} to subscriber: {e}")
        except Exception as e:
            logger.error(f"Error reading {format} encoder output: {e}")

    async def _broadcast(self, audio: np.ndarray) -> None:
        """Broadcast audio to all subscribers, converting to their requested format."""
        if not self._audio_sinks:
            return
        current_loop = None
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        # Feed PCM audio to active encoders
        pcm_data = pack_pcm16(audio)
        for format, encoder in self._encoders.items():
            await encoder.encode(pcm_data)

        for (q, loop, fmt) in list(self._audio_sinks):
            # Skip encoded formats - they get data from _read_encoder_output
            if fmt in ("mp3", "opus", "aac"):
                continue

            # Convert audio to requested format
            if fmt == "f32":
                payload = pack_f32(audio)
            else:  # Default to pcm16
                payload = pcm_data

            if current_loop is loop:
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    # Queue full - try to drop oldest and retry
                    logger.warning(f"Channel {self.cfg.id}: Audio queue full for format={fmt}, dropping oldest packet")
                    try:
                        _ = q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    try:
                        q.put_nowait(payload)
                    except asyncio.QueueFull:
                        logger.warning(f"Channel {self.cfg.id}: Audio queue still full for format={fmt}, dropping packet")
            else:
                # Schedule put on the queue's owning loop to avoid cross-loop errors
                def _try_put() -> None:
                    try:
                        q.put_nowait(payload)
                    except asyncio.QueueFull:
                        # Queue full - try to drop oldest and retry
                        logger.warning(f"Channel {self.cfg.id}: Audio queue full (cross-loop) for format={fmt}, dropping oldest packet")
                        try:
                            _ = q.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        try:
                            q.put_nowait(payload)
                        except asyncio.QueueFull:
                            logger.warning(f"Channel {self.cfg.id}: Audio queue still full (cross-loop) for format={fmt}, dropping packet")

                try:
                    loop.call_soon_threadsafe(_try_put)
                except Exception as e:
                    # If loop is closed or unavailable, drop sink
                    logger.warning(f"Channel {self.cfg.id}: Removing audio sink for format={fmt} due to loop error: {type(e).__name__}: {e}")
                    try:
                        self._audio_sinks.discard((q, loop, fmt))
                    except Exception:
                        pass

    async def process_iq_chunk(self, iq: np.ndarray, sample_rate: int) -> None:
        if self.state != "running":
            return

        if self.cfg.mode in ("wbfm", "nbfm"):
            # FM demodulation (wide or narrow band)
            base = freq_shift(iq, self.cfg.offset_hz, sample_rate)
            if self.cfg.mode == "wbfm":
                audio = wbfm_demod(base, sample_rate, self.cfg.audio_rate)
            else:  # nbfm
                audio = nbfm_demod(base, sample_rate, self.cfg.audio_rate)

            # Calculate signal power in dB (always, for metrics)
            if audio.size > 0:
                power = np.mean(audio ** 2)
                power_db = 10.0 * np.log10(power + 1e-10)  # Add small value to avoid log(0)
                self.signal_power_db = float(power_db)
                logger.debug(f"Channel {self.cfg.id}: signal_power_db={power_db:.2f}")
            else:
                self.signal_power_db = None

            # Apply squelch if configured
            if self.cfg.squelch_db is not None and audio.size > 0:
                # Mute audio if below threshold
                if self.signal_power_db is not None and self.signal_power_db < self.cfg.squelch_db:
                    audio = np.zeros_like(audio)

            await self._broadcast(audio)

        elif self.cfg.mode == "p25":
            # P25 digital voice decoding with trunking support
            # Initialize decoder if needed
            if self._p25_decoder is None:
                self._p25_decoder = P25Decoder(sample_rate)
                self._p25_decoder.on_voice_frame = lambda voice_data: self._handle_p25_voice(voice_data)
                self._p25_decoder.on_grant = lambda tgid, freq: self._handle_trunking_grant(tgid, freq)
                logger.info(f"Channel {self.cfg.id}: P25 decoder initialized")

            # Frequency shift to channel offset
            base = freq_shift(iq, self.cfg.offset_hz, sample_rate)

            # Calculate signal power for metrics
            if base.size > 0:
                power = np.mean(np.abs(base) ** 2)
                power_db = 10.0 * np.log10(power + 1e-10)
                self.signal_power_db = float(power_db)
            else:
                self.signal_power_db = None

            # Decode P25 frames
            try:
                frames = self._p25_decoder.process_iq(base)

                # Log decoded frames (for debugging)
                for frame in frames:
                    if frame.tgid is not None:
                        logger.debug(f"Channel {self.cfg.id}: P25 frame type={frame.frame_type.value} TGID={frame.tgid}")
                    elif frame.tsbk_data:
                        logger.debug(f"Channel {self.cfg.id}: P25 TSBK: {frame.tsbk_data}")

                # Note: Voice audio will be handled by the on_voice_frame callback
                # For now, we don't have IMBE decoder, so voice remains silent
                # Future: Add IMBE/codec2 decoder to convert voice_data to PCM
            except Exception as e:
                logger.error(f"Channel {self.cfg.id}: P25 decoding error: {e}")

        elif self.cfg.mode == "dmr":
            # DMR digital voice decoding with trunking support
            # Initialize decoder if needed
            if self._dmr_decoder is None:
                self._dmr_decoder = DMRDecoder(sample_rate)
                self._dmr_decoder.on_voice_frame = lambda slot, tgid, voice_data: self._handle_dmr_voice(slot, tgid, voice_data)
                self._dmr_decoder.on_csbk_message = lambda msg: self._handle_dmr_csbk(msg)
                logger.info(f"Channel {self.cfg.id}: DMR decoder initialized")

            # Frequency shift to channel offset
            base = freq_shift(iq, self.cfg.offset_hz, sample_rate)

            # Calculate signal power for metrics
            if base.size > 0:
                power = np.mean(np.abs(base) ** 2)
                power_db = 10.0 * np.log10(power + 1e-10)
                self.signal_power_db = float(power_db)
            else:
                self.signal_power_db = None

            # Decode DMR frames
            try:
                frames = self._dmr_decoder.process_iq(base)

                # Log decoded frames
                for frame in frames:
                    logger.debug(f"Channel {self.cfg.id}: DMR frame type={frame.frame_type.value} slot={frame.slot.value} dst={frame.dst_id}")

                # Note: Voice audio will be handled by the on_voice_frame callback
                # Future: Add AMBE decoder to convert voice_data to PCM
            except Exception as e:
                logger.error(f"Channel {self.cfg.id}: DMR decoding error: {e}")

        elif self.cfg.mode == "raw":
            # Raw IQ output (for external decoding)
            # Frequency shift to extract the desired channel
            base = freq_shift(iq, self.cfg.offset_hz, sample_rate)

            # Calculate signal power for metrics
            if base.size > 0:
                power = np.mean(np.abs(base) ** 2)
                power_db = 10.0 * np.log10(power + 1e-10)
                self.signal_power_db = float(power_db)
            else:
                self.signal_power_db = None

            # Convert IQ to interleaved I/Q float32 for streaming
            # Format: [I0, Q0, I1, Q1, ...]
            iq_interleaved = np.empty(base.size * 2, dtype=np.float32)
            iq_interleaved[0::2] = base.real
            iq_interleaved[1::2] = base.imag

            await self._broadcast(iq_interleaved)

        else:
            # Unknown mode: ignore
            return

    def _handle_p25_voice(self, voice_data: bytes) -> None:
        """Handle decoded P25 voice frames (IMBE codec).

        Currently logs voice activity. Future implementation will:
        - Decode IMBE voice frames to PCM audio
        - Use mbelib or codec2 for voice synthesis
        - Broadcast audio to subscribers
        """
        logger.debug(f"Channel {self.cfg.id}: P25 voice frame received ({len(voice_data)} bytes)")
        # TODO: Implement IMBE decoder
        # For now, voice is not output as audio (awaiting codec integration)

    def _handle_dmr_voice(self, slot: int, tgid: int, voice_data: bytes) -> None:
        """Handle decoded DMR voice frames (AMBE codec).

        Currently logs voice activity. Future implementation will:
        - Decode AMBE voice frames to PCM audio
        - Use mbelib for voice synthesis
        - Broadcast audio to subscribers
        """
        logger.debug(f"Channel {self.cfg.id}: DMR voice frame slot={slot} TGID={tgid} ({len(voice_data)} bytes)")
        # TODO: Implement AMBE decoder

    def _handle_trunking_grant(self, tgid: int, freq_hz: float) -> None:
        """Handle P25 trunking voice channel grant.

        This is called when a control channel broadcasts a voice grant.
        Future implementation will:
        - Automatically create voice channel following the grant
        - Track talkgroup activity
        - Integrate with TrunkingManager for priority-based following
        """
        logger.info(f"Channel {self.cfg.id}: P25 voice grant - TGID {tgid} on {freq_hz/1e6:.4f} MHz")
        # TODO: Implement automatic voice channel following

    def _handle_dmr_csbk(self, msg: dict) -> None:
        """Handle DMR Control Signaling Block messages.

        CSBK messages contain control information similar to P25 TSBK.
        """
        logger.debug(f"Channel {self.cfg.id}: DMR CSBK message: {msg}")
        # TODO: Implement DMR trunking logic


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
    driver: DeviceDriver
    requested_device_id: Optional[str] = None
    state: str = "stopped"  # Use string for backwards compatibility with API
    device: Optional[Device] = None
    antenna: Optional[str] = None  # Actual antenna in use
    error_message: Optional[str] = None  # Error message if state is "failed"
    _stream: Optional[StreamHandle] = None
    _thread: Optional[threading.Thread] = None
    _health_monitor: Optional[threading.Thread] = None
    _iq_sinks: Set[Tuple[asyncio.Queue[bytes], asyncio.AbstractEventLoop]] = field(
        default_factory=set
    )
    _fft_sinks: Set[Tuple[asyncio.Queue[dict], asyncio.AbstractEventLoop]] = field(
        default_factory=set
    )  # Spectrum/FFT subscribers (only calculated when needed for efficiency)
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _channels: Dict[str, Channel] = field(default_factory=dict)
    # Retry tracking (inspired by OpenWebRX)
    _retry_count: int = 0
    _max_retries: int = 10  # OpenWebRX uses 10
    _retry_delay: float = 15.0  # OpenWebRX uses 15 seconds
    _retry_timer: Optional[threading.Timer] = None
    _auto_restart_enabled: bool = True  # Enable automatic restart on failure
    _last_run_time: float = 0.0  # Track when the thread was last running
    # FFT data (server-side spectrum analyzer)
    _fft_power: Optional[np.ndarray] = None  # Power spectrum in dB
    _fft_freqs: Optional[np.ndarray] = None  # Frequency bins in Hz

    def _ensure_device(self) -> Device:
        """Lazily open the device when the capture actually starts."""
        if self.device is None:
            dev = self.driver.open(self.requested_device_id)
            self.device = dev
            # Update config with the concrete device id so the UI shows it
            try:
                self.cfg.device_id = dev.info.id
            except Exception:
                pass
        return self.device

    def release_device(self) -> None:
        """Close and drop the device reference (used on deletion/failure)."""
        if self.device is None:
            return
        try:
            self.device.close()
        except Exception:
            pass
        finally:
            self.device = None
            self.antenna = None

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
        device_settings: Optional[Dict[str, Any]] = None,
        element_gains: Optional[Dict[str, float]] = None,
        stream_format: Optional[str] = None,
        dc_offset_auto: Optional[bool] = None,
        iq_balance_auto: Optional[bool] = None,
    ) -> None:
        """Reconfigure capture, using hot reconfiguration when possible.

        If only hot-reconfigurable parameters are changed (center_hz, gain,
        bandwidth, ppm), the capture will be updated without restarting.
        If sample_rate, antenna, or advanced settings change, a full restart is required.
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
        if device_settings is not None:
            self.cfg.device_settings = device_settings
        if element_gains is not None:
            self.cfg.element_gains = element_gains
        if stream_format is not None:
            self.cfg.stream_format = stream_format
        if dc_offset_auto is not None:
            self.cfg.dc_offset_auto = dc_offset_auto
        if iq_balance_auto is not None:
            self.cfg.iq_balance_auto = iq_balance_auto

        # Check if we need full restart
        # Advanced settings require restart as they affect device initialization
        needs_restart = (
            sample_rate is not None
            or antenna is not None
            or device_settings is not None
            or element_gains is not None
            or stream_format is not None
            or dc_offset_auto is not None
            or iq_balance_auto is not None
        )

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

    async def subscribe_fft(self) -> asyncio.Queue[dict]:
        """Subscribe to FFT/spectrum data. Only calculated when there are active subscribers."""
        q: asyncio.Queue[dict] = asyncio.Queue(maxsize=4)
        loop = asyncio.get_running_loop()
        self._fft_sinks.add((q, loop))
        return q

    def unsubscribe_fft(self, q: asyncio.Queue[dict]) -> None:
        """Unsubscribe from FFT/spectrum data."""
        for item in list(self._fft_sinks):
            if item[0] is q:
                self._fft_sinks.discard(item)

    def _calculate_fft(self, samples: np.ndarray, sample_rate: int, fft_size: int = 2048) -> None:
        """Calculate FFT for spectrum display. Only called when there are active subscribers."""
        if samples.size < fft_size:
            return

        # Take a chunk of samples for FFT
        chunk = samples[:fft_size]

        # Apply Hanning window to reduce spectral leakage
        window = np.hanning(fft_size)
        windowed = chunk * window

        # Perform FFT
        fft_result = np.fft.fft(windowed)
        fft_shifted = np.fft.fftshift(fft_result)

        # Calculate power spectrum in dB
        magnitude = np.abs(fft_shifted)
        power_db = 20.0 * np.log10(magnitude + 1e-10)

        # Calculate frequency bins
        freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, 1.0 / sample_rate))

        # Store results
        self._fft_power = power_db
        self._fft_freqs = freqs

    async def _broadcast_fft(self) -> None:
        """Broadcast FFT data to all subscribers."""
        if not self._fft_sinks or self._fft_power is None or self._fft_freqs is None:
            return

        # Create payload with FFT data
        payload = {
            "power": self._fft_power.tolist(),
            "freqs": self._fft_freqs.tolist(),
            "centerHz": self.cfg.center_hz,
            "sampleRate": self.cfg.sample_rate,
        }

        current_loop = None
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        for (q, loop) in list(self._fft_sinks):
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
                        self._fft_sinks.discard((q, loop))
                    except Exception:
                        pass

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
            device = self._ensure_device()
            device.configure(
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
            stream = device.start_stream()
            # Store the actual antenna being used
            self.antenna = device.get_antenna()
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
            self.release_device()
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
            # Calculate server-side metrics for all running channels (no async needed)
            chans = list(self._channels.values())
            for ch in chans:
                if ch.state == "running":
                    # Update signal metrics synchronously (RSSI, SNR)
                    ch.update_signal_metrics(samples.copy(), self.cfg.sample_rate)

            # Calculate FFT for spectrum display (only if there are active subscribers)
            if self._fft_sinks:
                self._calculate_fft(samples.copy(), self.cfg.sample_rate)
                # Broadcast FFT to subscribers
                for (q, loop) in list(self._fft_sinks):
                    if self._fft_power is not None and self._fft_freqs is not None:
                        payload_fft = {
                            "power": self._fft_power.tolist(),
                            "freqs": self._fft_freqs.tolist(),
                            "centerHz": self.cfg.center_hz,
                            "sampleRate": self.cfg.sample_rate,
                        }
                        def _try_put_fft() -> None:
                            try:
                                q.put_nowait(payload_fft)
                            except asyncio.QueueFull:
                                try:
                                    _ = q.get_nowait()
                                except asyncio.QueueEmpty:
                                    pass
                                try:
                                    q.put_nowait(payload_fft)
                                except asyncio.QueueFull:
                                    pass
                        try:
                            loop.call_soon_threadsafe(_try_put_fft)
                        except Exception:
                            try:
                                self._fft_sinks.discard((q, loop))
                            except Exception:
                                pass

            # Dispatch to channels for audio processing (requires event loop)
            for ch in chans:
                # Get event loop - try audio sinks first, then IQ sinks
                target_loop = None
                try:
                    # Get loop from first audio sink (tuple is (q, loop, format))
                    if ch._audio_sinks:
                        target_loop = next(iter(ch._audio_sinks))[1]
                except (StopIteration, IndexError):
                    pass

                # Fallback to IQ sinks if no audio sinks
                if target_loop is None:
                    try:
                        if self._iq_sinks:
                            target_loop = next(iter(self._iq_sinks))[1]
                    except Exception:
                        pass

                # Process channel if we have an event loop (always process for metrics)
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
        cfg = CaptureConfig(
            id=cid,
            device_id=device_id or "auto",
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
        cap = Capture(
            cfg=cfg,
            driver=self._driver,
            requested_device_id=device_id,
        )
        self._captures[cid] = cap
        return cap

    async def delete_capture(self, cid: str) -> None:
        cap = self._captures.pop(cid, None)
        if cap is not None:
            await cap.stop()
            # Close the device when deleting the capture
            cap.release_device()
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
