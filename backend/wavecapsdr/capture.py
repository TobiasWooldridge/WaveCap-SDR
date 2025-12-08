from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
import functools
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, cast

import numpy as np

# Import scipy.fft at module level (not inside hot loops)
try:
    from scipy.fft import fft, fftshift, fftfreq
    SCIPY_FFT_AVAILABLE = True
except ImportError:
    SCIPY_FFT_AVAILABLE = False

logger = logging.getLogger(__name__)

from .config import AppConfig
from .devices.base import Device, DeviceDriver, StreamHandle
# Disabled: automatic recovery tends to cause thrashing
# from .sdrplay_recovery import attempt_recovery
from .dsp.fm import wbfm_demod, nbfm_demod, quadrature_demod
from .dsp.am import am_demod, ssb_demod
from .dsp.rds import RDSDecoder, RDSData
from .dsp.pocsag import POCSAGDecoder, POCSAGMessage
from .encoders import create_encoder, AudioEncoder
from .decoders.p25 import P25Decoder
from .decoders.dmr import DMRDecoder
from .decoders.imbe import IMBEDecoder, check_imbe_available


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
    """Pack complex IQ samples as interleaved 16-bit integers.

    Performance: Uses numpy view for zero-copy interleaving.
    """
    if samples.size == 0:
        return b""
    # View complex64 as pairs of float32 (already interleaved I,Q,I,Q,...)
    x: np.ndarray = samples.astype(np.complex64, copy=False)
    # Use view to get interleaved float32 without copying
    interleaved_f32 = x.view(np.float32)
    # Clip and scale in one operation, then convert to int16
    np.clip(interleaved_f32, -1.0, 1.0, out=interleaved_f32)
    result: bytes = (interleaved_f32 * 32767.0).astype(np.int16).tobytes()
    return result


def pack_pcm16(samples: np.ndarray) -> bytes:
    """Pack audio samples as 16-bit PCM.

    Performance: Avoids intermediate array allocation.
    """
    if samples.size == 0:
        return b""
    # Use contiguous array for efficient conversion
    x: np.ndarray = np.ascontiguousarray(samples, dtype=np.float32)
    np.clip(x, -1.0, 1.0, out=x)
    x *= 32767.0
    result: bytes = x.astype(np.int16).tobytes()
    return result


def pack_f32(samples: np.ndarray) -> bytes:
    """Pack samples as 32-bit float (little-endian).

    Performance: Uses in-place clip when possible.
    """
    if samples.size == 0:
        return b""
    x: np.ndarray = np.ascontiguousarray(samples, dtype=np.float32)
    np.clip(x, -1.0, 1.0, out=x)
    result: bytes = x.tobytes()
    return result


# Cache for frequency shift exponentials using LRU cache (proper eviction)
@functools.lru_cache(maxsize=64)
def _get_freq_shift_exp(size: int, offset_hz: int, sample_rate: int) -> np.ndarray:
    """Get cached complex exponential for frequency shifting.

    Uses LRU cache for proper eviction semantics (vs manual dict with arbitrary removal).
    Note: offset_hz is int (rounded from float) to ensure consistent cache keys.
    """
    n = np.arange(size, dtype=np.float32)
    result: np.ndarray = np.exp(-1j * 2.0 * np.pi * (offset_hz / float(sample_rate)) * n).astype(np.complex64)
    return result


def freq_shift(iq: np.ndarray, offset_hz: float, sample_rate: int) -> np.ndarray:
    """Frequency shift IQ samples by mixing with complex exponential.

    Performance: Caches the complex exponential using LRU cache for repeated
    offset/sample_rate/size combinations, providing 10-15% speedup.
    """
    if offset_hz == 0.0 or iq.size == 0:
        return iq

    # Round offset_hz to int for consistent cache keys (prevents unbounded cache growth
    # from floating-point variations like 1000.0 vs 1000.0000001)
    ph = _get_freq_shift_exp(iq.shape[0], round(offset_hz), sample_rate)
    result: np.ndarray = (iq.astype(np.complex64, copy=False) * ph).astype(np.complex64)
    return result


def _process_channel_dsp_stateless(
    samples: np.ndarray,
    sample_rate: int,
    cfg: "ChannelConfig",
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Stateless DSP processing for a single channel.

    Thread-safe: operates only on input arrays and immutable config.
    Does NOT handle stateful decoders (RDS, POCSAG, P25, DMR).

    This function is designed to run in a ThreadPoolExecutor for parallel
    channel processing. NumPy/SciPy release the GIL during heavy computation,
    enabling ~2.7x speedup with multiple workers.

    Args:
        samples: IQ samples from capture device
        sample_rate: Sample rate in Hz
        cfg: Channel configuration (immutable during processing)

    Returns:
        Tuple of (audio_samples or None, metrics_dict)
    """
    metrics: Dict[str, Any] = {}

    # Frequency shift to channel offset
    if cfg.offset_hz == 0.0:
        base = samples
    else:
        base = freq_shift(samples, cfg.offset_hz, sample_rate)

    # Calculate RSSI from shifted IQ
    if base.size > 0:
        magnitudes = np.abs(base)
        power = np.mean(magnitudes ** 2)
        metrics['rssi_db'] = float(10.0 * np.log10(power + 1e-10))

    audio: Optional[np.ndarray] = None

    # Mode-specific demodulation (stateless operations only)
    if cfg.mode == "wbfm":
        audio = wbfm_demod(
            base,
            sample_rate,
            cfg.audio_rate,
            enable_deemphasis=cfg.enable_deemphasis,
            deemphasis_tau=cfg.deemphasis_tau_us * 1e-6,
            enable_mpx_filter=cfg.enable_mpx_filter,
            mpx_cutoff_hz=cfg.mpx_cutoff_hz,
            enable_highpass=cfg.enable_fm_highpass,
            highpass_hz=cfg.fm_highpass_hz,
            notch_frequencies=cfg.notch_frequencies if cfg.notch_frequencies else None,
            enable_noise_reduction=cfg.enable_noise_reduction,
            noise_reduction_db=cfg.noise_reduction_db,
        )

    elif cfg.mode == "nbfm":
        audio = nbfm_demod(
            base,
            sample_rate,
            cfg.audio_rate,
            enable_deemphasis=cfg.enable_deemphasis,
            deemphasis_tau=cfg.deemphasis_tau_us * 1e-6,
            enable_highpass=cfg.enable_fm_highpass,
            highpass_hz=cfg.fm_highpass_hz,
            enable_lowpass=cfg.enable_fm_lowpass,
            lowpass_hz=cfg.fm_lowpass_hz,
            notch_frequencies=cfg.notch_frequencies if cfg.notch_frequencies else None,
            enable_noise_reduction=cfg.enable_noise_reduction,
            noise_reduction_db=cfg.noise_reduction_db,
        )

    elif cfg.mode == "am":
        audio = am_demod(
            base,
            sample_rate,
            audio_rate=cfg.audio_rate,
            enable_agc=cfg.enable_agc,
            enable_highpass=cfg.enable_am_highpass,
            highpass_hz=cfg.am_highpass_hz,
            enable_lowpass=cfg.enable_am_lowpass,
            lowpass_hz=cfg.am_lowpass_hz,
            agc_target_db=cfg.agc_target_db,
            notch_frequencies=cfg.notch_frequencies if cfg.notch_frequencies else None,
        )

    elif cfg.mode == "ssb":
        audio = ssb_demod(
            base,
            sample_rate,
            audio_rate=cfg.audio_rate,
            mode=cfg.ssb_mode,
            enable_agc=cfg.enable_agc,
            enable_bandpass=cfg.enable_ssb_bandpass,
            bandpass_low=cfg.ssb_bandpass_low_hz,
            bandpass_high=cfg.ssb_bandpass_high_hz,
            agc_target_db=cfg.agc_target_db,
            notch_frequencies=cfg.notch_frequencies if cfg.notch_frequencies else None,
        )

    elif cfg.mode == "raw":
        # Raw IQ output (no demodulation)
        if base.size > 0:
            iq_interleaved = np.empty(base.size * 2, dtype=np.float32)
            iq_interleaved[0::2] = base.real
            iq_interleaved[1::2] = base.imag
            audio = iq_interleaved

    # P25 and DMR require stateful decoders - handled separately
    elif cfg.mode in ("p25", "dmr"):
        # Just return base IQ for stateful processing later
        if base.size > 0:
            power = np.mean(np.abs(base) ** 2)
            metrics['signal_power_db'] = float(10.0 * np.log10(power + 1e-10))
        return None, metrics

    # Calculate signal power from audio
    if audio is not None and audio.size > 0:
        power = np.mean(audio ** 2)
        metrics['signal_power_db'] = float(10.0 * np.log10(power + 1e-10))

    return audio, metrics


@dataclass
class ChannelConfig:
    id: str
    capture_id: str
    mode: str  # "wbfm" | "nbfm" | "am" | "ssb"
    offset_hz: float = 0.0
    audio_rate: int = 48_000
    squelch_db: Optional[float] = None
    name: Optional[str] = None  # User-provided name
    auto_name: Optional[str] = None  # Auto-generated contextual name

    # Filter configuration (mode-specific defaults applied at creation)
    # FM filters
    enable_deemphasis: bool = True
    deemphasis_tau_us: float = 75.0  # 75µs (US) or 50µs (EU)
    enable_mpx_filter: bool = True  # WBFM only: removes 19 kHz pilot tone
    mpx_cutoff_hz: float = 15_000
    enable_fm_highpass: bool = False
    fm_highpass_hz: float = 100
    enable_fm_lowpass: bool = False
    fm_lowpass_hz: float = 3_000  # NBFM voice bandwidth

    # AM/SSB filters
    enable_am_highpass: bool = True
    am_highpass_hz: float = 100
    enable_am_lowpass: bool = True
    am_lowpass_hz: float = 5_000  # AM broadcast bandwidth
    enable_ssb_bandpass: bool = True
    ssb_bandpass_low_hz: float = 300
    ssb_bandpass_high_hz: float = 3_000
    ssb_mode: str = "usb"  # "usb" or "lsb"

    # AGC (Automatic Gain Control)
    enable_agc: bool = False  # Default off for FM, enabled for AM/SSB
    agc_target_db: float = -20.0
    agc_attack_ms: float = 5.0
    agc_release_ms: float = 50.0

    # Noise blanker (impulse noise suppression)
    enable_noise_blanker: bool = False  # Default off
    noise_blanker_threshold_db: float = 10.0  # 10 dB above median level

    # Notch filters (interference rejection)
    notch_frequencies: list[float] = field(default_factory=list)  # List of frequencies to notch out

    # Spectral noise reduction (hiss/static suppression)
    enable_noise_reduction: bool = False  # Default off
    noise_reduction_db: float = 12.0  # 12 dB reduction

    # RDS decoding (WBFM only)
    enable_rds: bool = True  # Enabled by default for WBFM

    # POCSAG decoding (NBFM only)
    enable_pocsag: bool = False  # Disabled by default
    pocsag_baud: int = 1200  # 512, 1200, or 2400


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
    # Audio output level metering
    audio_rms_db: Optional[float] = None  # Output audio RMS level in dB
    audio_peak_db: Optional[float] = None  # Output audio peak level in dB
    audio_clipping_count: int = 0  # Number of samples that would have clipped
    # Digital voice decoders (lazily initialized)
    _p25_decoder: Optional[P25Decoder] = None
    _dmr_decoder: Optional[DMRDecoder] = None
    # IMBE voice codec decoder for P25 (lazily initialized)
    _imbe_decoder: Optional[IMBEDecoder] = None
    # RDS decoder for WBFM (lazily initialized)
    _rds_decoder: Optional[RDSDecoder] = None
    rds_data: Optional[RDSData] = None  # Current RDS data (exposed via API)
    # POCSAG decoder for NBFM pager feeds (lazily initialized)
    _pocsag_decoder: Optional[POCSAGDecoder] = None
    _pocsag_messages: list[POCSAGMessage] = field(default_factory=list)
    _pocsag_max_messages: int = 100  # Ring buffer size
    # Drop tracking for rate-limited logging
    _drop_count: int = 0
    _last_drop_log_time: float = 0.0
    # Metrics counter for throttled signal metrics calculation
    _metrics_counter: int = 0

    def start(self) -> None:
        self.state = "running"
        self._drop_count = 0
        self._last_drop_log_time = 0.0
        self.audio_clipping_count = 0

    def stop(self) -> None:
        self.state = "stopped"
        # Clean up IMBE decoder if running
        if self._imbe_decoder is not None:
            asyncio.create_task(self._imbe_decoder.stop())

    def get_pocsag_messages(self, limit: int = 50, since_timestamp: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get recent POCSAG messages.

        Args:
            limit: Maximum number of messages to return
            since_timestamp: Only return messages after this timestamp (for SSE)

        Returns:
            List of message dictionaries (most recent first)
        """
        msgs = self._pocsag_messages
        if since_timestamp is not None:
            msgs = [m for m in msgs if m.timestamp > since_timestamp]
        return [m.to_dict() for m in reversed(msgs[-limit:])]

    def _update_audio_metrics(self, audio: np.ndarray) -> None:
        """Calculate and update audio output level metrics.

        Args:
            audio: Audio samples (float32, expected range [-1, 1])
        """
        if audio.size == 0:
            self.audio_rms_db = None
            self.audio_peak_db = None
            return

        # Calculate RMS level in dB
        rms: float = float(np.sqrt(np.mean(audio ** 2)))
        if rms > 1e-10:
            self.audio_rms_db = float(20.0 * np.log10(rms))
        else:
            self.audio_rms_db = -100.0  # Effectively silence

        # Calculate peak level in dB
        peak: float = float(np.max(np.abs(audio)))
        if peak > 1e-10:
            self.audio_peak_db = float(20.0 * np.log10(peak))
        else:
            self.audio_peak_db = -100.0

        # Count samples that would have clipped (before soft clipping)
        # Threshold of 0.95 to catch samples that are near clipping
        clipping_samples: int = int(np.sum(np.abs(audio) > 0.95))
        self.audio_clipping_count += clipping_samples


    def _log_drop_warning(self, fmt: str) -> None:
        """Rate-limited logging for queue drops (once per 10 seconds)."""
        self._drop_count += 1
        now = time.time()

        # Report to error tracker (once per second)
        if now - self._last_drop_log_time >= 1.0:
            from .error_tracker import get_error_tracker, ErrorEvent
            get_error_tracker().record(ErrorEvent(
                type="audio_drop",
                capture_id=self.cfg.capture_id,
                channel_id=self.cfg.id,
                timestamp=now,
                count=self._drop_count,
                details={"format": fmt},
            ))

        # Log warning (once per 10 seconds)
        if now - self._last_drop_log_time >= 10.0:
            logger.warning(
                f"Channel {self.cfg.id}: Audio queue full for format={fmt}, "
                f"dropped {self._drop_count} packets in last 10s"
            )
            self._drop_count = 0
            self._last_drop_log_time = now

    def cleanup_zombie_subscribers(self) -> int:
        """Remove subscribers whose event loops are closed or unavailable.

        Returns:
            Number of zombie subscribers removed
        """
        zombies = []
        for item in list(self._audio_sinks):
            q, loop, fmt = item
            try:
                # Check if loop is closed or not running
                if loop.is_closed():
                    zombies.append(item)
            except Exception:
                # If we can't check the loop, it's probably dead
                zombies.append(item)

        for item in zombies:
            fmt = item[2]
            self._audio_sinks.discard(item)
            logger.info(f"Channel {self.cfg.id}: Removed zombie subscriber, format={fmt}")

            # Clean up encoder subscriber count
            if fmt in ("mp3", "opus", "aac"):
                self._encoder_subscribers[fmt] = max(0, self._encoder_subscribers.get(fmt, 1) - 1)
                if self._encoder_subscribers[fmt] == 0 and fmt in self._encoders:
                    logger.info(f"Channel {self.cfg.id}: Stopping {fmt} encoder (zombie cleanup)")
                    encoder = self._encoders.pop(fmt)
                    asyncio.create_task(encoder.stop())
                    del self._encoder_subscribers[fmt]

        return len(zombies)

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics for monitoring and health checks.

        Returns:
            Dictionary with subscriber counts, queue depths, and format breakdown
        """
        format_stats: Dict[str, Dict[str, int]] = {}

        for q, loop, fmt in list(self._audio_sinks):
            if fmt not in format_stats:
                format_stats[fmt] = {"subscribers": 0, "total_queue_depth": 0}
            format_stats[fmt]["subscribers"] += 1
            try:
                format_stats[fmt]["total_queue_depth"] += q.qsize()
            except Exception:
                pass  # Queue might be in invalid state

        return {
            "total_subscribers": len(self._audio_sinks),
            "formats": format_stats,
            "active_encoders": list(self._encoders.keys()),
            "drops_since_last_log": self._drop_count,
        }

    def update_signal_metrics(self, iq: np.ndarray, sample_rate: int) -> None:
        """Calculate signal metrics from IQ samples (server-side, no audio needed).

        Optimized to:
        - Skip freq_shift when offset is 0 (common for centered channels)
        - Throttle expensive SNR calculation (np.partition) to every 10th call
        """
        if self.state != "running" or iq.size == 0:
            return

        # OPTIMIZATION: Skip freq_shift if offset is 0 (common for centered channels)
        # Saves ~10-15% CPU for channels tuned to capture center frequency
        if self.cfg.offset_hz == 0.0:
            shifted_iq = iq
        else:
            shifted_iq = freq_shift(iq, self.cfg.offset_hz, sample_rate)

        # Calculate magnitudes once (reuse for RSSI and optionally SNR)
        magnitudes = np.abs(shifted_iq)

        # Calculate RSSI (power of IQ samples in dB) - always needed for squelch
        power = np.mean(magnitudes ** 2)
        self.rssi_db = float(10.0 * np.log10(power + 1e-10))

        # OPTIMIZATION: Throttle SNR calculation to every 10th call
        # np.partition is O(n) but still expensive - SNR doesn't need real-time updates
        self._snr_counter = getattr(self, '_snr_counter', 0) + 1
        if self._snr_counter % 10 != 0:
            return  # Skip SNR calculation this time

        # Estimate SNR using partition-based noise floor estimation (O(n) vs O(n log n) percentile)
        n = magnitudes.size
        k_noise = n // 10  # 10th percentile index
        k_signal = n - n // 10 - 1  # 90th percentile index

        if k_noise > 0 and k_signal > k_noise:
            # np.partition is O(n) - much faster than percentile's O(n log n)
            partitioned = np.partition(magnitudes, [k_noise, k_signal])
            noise_floor = partitioned[k_noise]
            signal_peak = partitioned[k_signal]

            noise_power = noise_floor ** 2
            signal_power = signal_peak ** 2

            if noise_power > 1e-10:
                self.snr_db = float(10.0 * np.log10(signal_power / noise_power))
            else:
                self.snr_db = None
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
                    try:
                        _ = q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    try:
                        q.put_nowait(payload)
                    except asyncio.QueueFull:
                        pass  # Still full, will be counted in drop warning
                    self._log_drop_warning(fmt)
            else:
                # Schedule put on the queue's owning loop to avoid cross-loop errors
                # Capture self reference for closure
                channel_ref = self

                def _try_put() -> None:
                    try:
                        q.put_nowait(payload)
                    except asyncio.QueueFull:
                        # Queue full - try to drop oldest and retry
                        try:
                            _ = q.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        try:
                            q.put_nowait(payload)
                        except asyncio.QueueFull:
                            pass  # Still full
                        channel_ref._log_drop_warning(fmt)

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
                # RDS decoding: Process FM baseband BEFORE MPX filter removes 57 kHz subcarrier
                if self.cfg.enable_rds and sample_rate >= 114000:
                    # Initialize RDS decoder if needed
                    if self._rds_decoder is None:
                        self._rds_decoder = RDSDecoder(sample_rate)
                        logger.info(f"Channel {self.cfg.id}: RDS decoder initialized")

                    # Get raw FM baseband (before MPX filter)
                    fm_baseband = quadrature_demod(base, sample_rate)

                    # Process RDS (extracts 57 kHz subcarrier and decodes)
                    try:
                        rds_result = self._rds_decoder.process(fm_baseband)
                        if rds_result:
                            self.rds_data = rds_result
                            if rds_result.ps_name.strip():
                                logger.debug(f"Channel {self.cfg.id}: RDS PS={rds_result.ps_name.strip()}")
                    except Exception as e:
                        logger.error(f"Channel {self.cfg.id}: RDS decoding error: {e}")

                audio = wbfm_demod(
                    base,
                    sample_rate,
                    self.cfg.audio_rate,
                    enable_deemphasis=self.cfg.enable_deemphasis,
                    deemphasis_tau=self.cfg.deemphasis_tau_us * 1e-6,  # Convert µs to seconds
                    enable_mpx_filter=self.cfg.enable_mpx_filter,
                    mpx_cutoff_hz=self.cfg.mpx_cutoff_hz,
                    enable_highpass=self.cfg.enable_fm_highpass,
                    highpass_hz=self.cfg.fm_highpass_hz,
                    notch_frequencies=self.cfg.notch_frequencies if self.cfg.notch_frequencies else None,
                    enable_noise_reduction=self.cfg.enable_noise_reduction,
                    noise_reduction_db=self.cfg.noise_reduction_db,
                )
            else:  # nbfm
                audio = nbfm_demod(
                    base,
                    sample_rate,
                    self.cfg.audio_rate,
                    enable_deemphasis=self.cfg.enable_deemphasis,
                    deemphasis_tau=self.cfg.deemphasis_tau_us * 1e-6,  # Convert µs to seconds
                    enable_highpass=self.cfg.enable_fm_highpass,
                    highpass_hz=self.cfg.fm_highpass_hz,
                    enable_lowpass=self.cfg.enable_fm_lowpass,
                    lowpass_hz=self.cfg.fm_lowpass_hz,
                    notch_frequencies=self.cfg.notch_frequencies if self.cfg.notch_frequencies else None,
                    enable_noise_reduction=self.cfg.enable_noise_reduction,
                    noise_reduction_db=self.cfg.noise_reduction_db,
                )

                # POCSAG decoding: Process demodulated audio for pager messages
                if self.cfg.enable_pocsag and audio.size > 0:
                    # Initialize POCSAG decoder if needed
                    if self._pocsag_decoder is None:
                        self._pocsag_decoder = POCSAGDecoder(
                            sample_rate=self.cfg.audio_rate,
                            baud_rate=self.cfg.pocsag_baud
                        )
                        logger.info(f"Channel {self.cfg.id}: POCSAG decoder initialized (baud={self.cfg.pocsag_baud})")

                    # Process audio and extract messages
                    try:
                        new_msgs = self._pocsag_decoder.process(audio)
                        for msg in new_msgs:
                            self._pocsag_messages.append(msg)
                            # Keep ring buffer bounded
                            if len(self._pocsag_messages) > self._pocsag_max_messages:
                                self._pocsag_messages.pop(0)
                            logger.info(f"Channel {self.cfg.id}: POCSAG msg addr={msg.address} func={msg.function}: {msg.message[:50] if msg.message else '(empty)'}")
                    except Exception as e:
                        logger.error(f"Channel {self.cfg.id}: POCSAG decoding error: {e}")

            # Calculate signal power in dB (always, for metrics)
            if audio.size > 0:
                power = np.mean(audio ** 2)
                power_db = 10.0 * np.log10(power + 1e-10)  # Add small value to avoid log(0)
                self.signal_power_db = float(power_db)
                logger.debug(f"Channel {self.cfg.id}: signal_power_db={power_db:.2f}")
            else:
                self.signal_power_db = None

            # Update audio output level metrics
            try:
                self._update_audio_metrics(audio)
            except Exception as e:
                logger.error(f"Channel {self.cfg.id}: Error in _update_audio_metrics: {e}")

            # Apply squelch if configured (use RSSI for proper RF-level squelch)
            if self.cfg.squelch_db is not None and audio.size > 0:
                # Mute audio if below threshold
                if self.rssi_db is not None and self.rssi_db < self.cfg.squelch_db:
                    audio = np.zeros_like(audio)

            await self._broadcast(audio)

        elif self.cfg.mode == "am":
            # AM demodulation
            base = freq_shift(iq, self.cfg.offset_hz, sample_rate)
            audio = am_demod(
                base,
                sample_rate,
                audio_rate=self.cfg.audio_rate,
                enable_agc=self.cfg.enable_agc,
                enable_highpass=self.cfg.enable_am_highpass,
                highpass_hz=self.cfg.am_highpass_hz,
                enable_lowpass=self.cfg.enable_am_lowpass,
                lowpass_hz=self.cfg.am_lowpass_hz,
                agc_target_db=self.cfg.agc_target_db,
                notch_frequencies=self.cfg.notch_frequencies if self.cfg.notch_frequencies else None,
            )

            # Calculate signal power in dB (always, for metrics)
            if audio.size > 0:
                power = np.mean(audio ** 2)
                power_db = 10.0 * np.log10(power + 1e-10)
                self.signal_power_db = float(power_db)
            else:
                self.signal_power_db = None

            # Update audio output level metrics
            self._update_audio_metrics(audio)

            # Apply squelch if configured (use RSSI for proper RF-level squelch)
            if self.cfg.squelch_db is not None and audio.size > 0:
                if self.rssi_db is not None and self.rssi_db < self.cfg.squelch_db:
                    audio = np.zeros_like(audio)

            await self._broadcast(audio)

        elif self.cfg.mode == "ssb":
            # SSB demodulation (USB or LSB)
            base = freq_shift(iq, self.cfg.offset_hz, sample_rate)
            audio = ssb_demod(
                base,
                sample_rate,
                audio_rate=self.cfg.audio_rate,
                mode=self.cfg.ssb_mode,
                enable_agc=self.cfg.enable_agc,
                enable_bandpass=self.cfg.enable_ssb_bandpass,
                bandpass_low=self.cfg.ssb_bandpass_low_hz,
                bandpass_high=self.cfg.ssb_bandpass_high_hz,
                agc_target_db=self.cfg.agc_target_db,
                notch_frequencies=self.cfg.notch_frequencies if self.cfg.notch_frequencies else None,
            )

            # Calculate signal power in dB (always, for metrics)
            if audio.size > 0:
                power = np.mean(audio ** 2)
                power_db = 10.0 * np.log10(power + 1e-10)
                self.signal_power_db = float(power_db)
            else:
                self.signal_power_db = None

            # Update audio output level metrics
            self._update_audio_metrics(audio)

            # Apply squelch if configured (use RSSI for proper RF-level squelch)
            if self.cfg.squelch_db is not None and audio.size > 0:
                if self.rssi_db is not None and self.rssi_db < self.cfg.squelch_db:
                    audio = np.zeros_like(audio)

            await self._broadcast(audio)

        elif self.cfg.mode == "p25":
            # P25 digital voice decoding with trunking support
            # Initialize P25 decoder for TSBK (control channel) parsing
            if self._p25_decoder is None:
                self._p25_decoder = P25Decoder(sample_rate)
                self._p25_decoder.on_voice_frame = lambda voice_data: self._handle_p25_voice(voice_data)
                self._p25_decoder.on_grant = lambda tgid, freq: self._handle_trunking_grant(tgid, freq)
                logger.info(f"Channel {self.cfg.id}: P25 decoder initialized")

            # Initialize IMBE decoder for voice audio (if available)
            # Use a class-level flag to avoid repeated availability checks/warnings
            if not hasattr(self, '_imbe_checked'):
                self._imbe_checked = True
                if IMBEDecoder.is_available():
                    self._imbe_decoder = IMBEDecoder(output_rate=self.cfg.audio_rate, input_rate=sample_rate)
                    asyncio.create_task(self._imbe_decoder.start())
                    logger.info(f"Channel {self.cfg.id}: IMBE decoder initialized (DSD-FME)")
                else:
                    logger.warning(
                        f"Channel {self.cfg.id}: IMBE decoder not available - "
                        "P25 voice will not be decoded. Install DSD-FME for voice support."
                    )

            # Frequency shift to channel offset
            base = freq_shift(iq, self.cfg.offset_hz, sample_rate)

            # Calculate signal power for metrics
            if base.size > 0:
                power = np.mean(np.abs(base) ** 2)
                power_db = 10.0 * np.log10(power + 1e-10)
                self.signal_power_db = float(power_db)
            else:
                self.signal_power_db = None

            # Decode P25 frames for trunking/TSBK
            try:
                frames = self._p25_decoder.process_iq(base)

                # Log decoded frames (for debugging)
                for frame in frames:
                    if frame.tgid is not None:
                        logger.debug(f"Channel {self.cfg.id}: P25 frame type={frame.frame_type.value} TGID={frame.tgid}")
                    elif frame.tsbk_data:
                        logger.debug(f"Channel {self.cfg.id}: P25 TSBK: {frame.tsbk_data}")
            except Exception as e:
                logger.error(f"Channel {self.cfg.id}: P25 decoding error: {e}")

            # Pass discriminator audio to IMBE decoder for voice decoding
            if self._imbe_decoder is not None and base.size > 0:
                try:
                    # Compute FM discriminator output (instantaneous frequency)
                    # This is the same computation used in C4FM demodulation
                    iq_c64: np.ndarray = base.astype(np.complex64, copy=False)
                    prod = iq_c64[1:] * np.conj(iq_c64[:-1])
                    discriminator = np.angle(prod) * sample_rate / (2 * np.pi)

                    # Queue discriminator audio for IMBE decoding
                    await self._imbe_decoder.decode(discriminator)

                    # Get decoded audio if available
                    decoded_audio = await self._imbe_decoder.get_audio()
                    if decoded_audio is not None and decoded_audio.size > 0:
                        self._update_audio_metrics(decoded_audio)
                        await self._broadcast(decoded_audio)
                except Exception as e:
                    logger.error(f"Channel {self.cfg.id}: IMBE decoding error: {e}")

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
                dmr_frames = self._dmr_decoder.process_iq(base)

                # Log decoded frames
                for dmr_frame in dmr_frames:
                    logger.debug(f"Channel {self.cfg.id}: DMR frame type={dmr_frame.frame_type.value} slot={dmr_frame.slot.value} dst={dmr_frame.dst_id}")

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

    def process_iq_chunk_sync(self, iq: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
        """Synchronous DSP processing - returns audio/IQ data for broadcast.

        This method performs all CPU-intensive DSP work (demodulation, filtering, etc.)
        and returns the processed audio data. It does NOT broadcast the data - that
        should be done separately on the event loop to avoid blocking HTTP requests.

        Returns:
            Processed audio data (np.ndarray) or None if channel not running/no output.
        """
        if self.state != "running":
            return None

        audio: Optional[np.ndarray] = None

        if self.cfg.mode in ("wbfm", "nbfm"):
            # FM demodulation (wide or narrow band)
            base = freq_shift(iq, self.cfg.offset_hz, sample_rate)
            if self.cfg.mode == "wbfm":
                # RDS decoding: Process FM baseband BEFORE MPX filter removes 57 kHz subcarrier
                if self.cfg.enable_rds and sample_rate >= 114000:
                    # Initialize RDS decoder if needed
                    if self._rds_decoder is None:
                        self._rds_decoder = RDSDecoder(sample_rate)
                        logger.info(f"Channel {self.cfg.id}: RDS decoder initialized")

                    # Get raw FM baseband (before MPX filter)
                    fm_baseband = quadrature_demod(base, sample_rate)

                    # Process RDS (extracts 57 kHz subcarrier and decodes)
                    try:
                        rds_result = self._rds_decoder.process(fm_baseband)
                        if rds_result:
                            self.rds_data = rds_result
                            if rds_result.ps_name.strip():
                                logger.debug(f"Channel {self.cfg.id}: RDS PS={rds_result.ps_name.strip()}")
                    except Exception as e:
                        logger.error(f"Channel {self.cfg.id}: RDS decoding error: {e}")

                audio = wbfm_demod(
                    base,
                    sample_rate,
                    self.cfg.audio_rate,
                    enable_deemphasis=self.cfg.enable_deemphasis,
                    deemphasis_tau=self.cfg.deemphasis_tau_us * 1e-6,
                    enable_mpx_filter=self.cfg.enable_mpx_filter,
                    mpx_cutoff_hz=self.cfg.mpx_cutoff_hz,
                    enable_highpass=self.cfg.enable_fm_highpass,
                    highpass_hz=self.cfg.fm_highpass_hz,
                    notch_frequencies=self.cfg.notch_frequencies if self.cfg.notch_frequencies else None,
                    enable_noise_reduction=self.cfg.enable_noise_reduction,
                    noise_reduction_db=self.cfg.noise_reduction_db,
                )
            else:  # nbfm
                audio = nbfm_demod(
                    base,
                    sample_rate,
                    self.cfg.audio_rate,
                    enable_deemphasis=self.cfg.enable_deemphasis,
                    deemphasis_tau=self.cfg.deemphasis_tau_us * 1e-6,
                    enable_highpass=self.cfg.enable_fm_highpass,
                    highpass_hz=self.cfg.fm_highpass_hz,
                    enable_lowpass=self.cfg.enable_fm_lowpass,
                    lowpass_hz=self.cfg.fm_lowpass_hz,
                    notch_frequencies=self.cfg.notch_frequencies if self.cfg.notch_frequencies else None,
                    enable_noise_reduction=self.cfg.enable_noise_reduction,
                    noise_reduction_db=self.cfg.noise_reduction_db,
                )

                # POCSAG decoding
                if self.cfg.enable_pocsag and audio.size > 0:
                    if self._pocsag_decoder is None:
                        self._pocsag_decoder = POCSAGDecoder(
                            sample_rate=self.cfg.audio_rate,
                            baud_rate=self.cfg.pocsag_baud
                        )
                        logger.info(f"Channel {self.cfg.id}: POCSAG decoder initialized (baud={self.cfg.pocsag_baud})")
                    try:
                        new_msgs = self._pocsag_decoder.process(audio)
                        for msg in new_msgs:
                            self._pocsag_messages.append(msg)
                            if len(self._pocsag_messages) > self._pocsag_max_messages:
                                self._pocsag_messages.pop(0)
                            logger.info(f"Channel {self.cfg.id}: POCSAG msg addr={msg.address} func={msg.function}: {msg.message[:50] if msg.message else '(empty)'}")
                    except Exception as e:
                        logger.error(f"Channel {self.cfg.id}: POCSAG decoding error: {e}")

            # Calculate signal power in dB
            if audio is not None and audio.size > 0:
                power = np.mean(audio ** 2)
                power_db = 10.0 * np.log10(power + 1e-10)
                self.signal_power_db = float(power_db)
                self._update_audio_metrics(audio)
            else:
                self.signal_power_db = None

            # Apply squelch
            if audio is not None and self.cfg.squelch_db is not None and audio.size > 0:
                if self.rssi_db is not None and self.rssi_db < self.cfg.squelch_db:
                    audio = np.zeros_like(audio)

        elif self.cfg.mode == "am":
            base = freq_shift(iq, self.cfg.offset_hz, sample_rate)
            audio = am_demod(
                base,
                sample_rate,
                audio_rate=self.cfg.audio_rate,
                enable_agc=self.cfg.enable_agc,
                enable_highpass=self.cfg.enable_am_highpass,
                highpass_hz=self.cfg.am_highpass_hz,
                enable_lowpass=self.cfg.enable_am_lowpass,
                lowpass_hz=self.cfg.am_lowpass_hz,
                agc_target_db=self.cfg.agc_target_db,
                notch_frequencies=self.cfg.notch_frequencies if self.cfg.notch_frequencies else None,
            )

            if audio is not None and audio.size > 0:
                power = np.mean(audio ** 2)
                power_db = 10.0 * np.log10(power + 1e-10)
                self.signal_power_db = float(power_db)
                self._update_audio_metrics(audio)
            else:
                self.signal_power_db = None

            if audio is not None and self.cfg.squelch_db is not None and audio.size > 0:
                if self.rssi_db is not None and self.rssi_db < self.cfg.squelch_db:
                    audio = np.zeros_like(audio)

        elif self.cfg.mode == "ssb":
            base = freq_shift(iq, self.cfg.offset_hz, sample_rate)
            audio = ssb_demod(
                base,
                sample_rate,
                audio_rate=self.cfg.audio_rate,
                mode=self.cfg.ssb_mode,
                enable_agc=self.cfg.enable_agc,
                enable_bandpass=self.cfg.enable_ssb_bandpass,
                bandpass_low=self.cfg.ssb_bandpass_low_hz,
                bandpass_high=self.cfg.ssb_bandpass_high_hz,
                agc_target_db=self.cfg.agc_target_db,
                notch_frequencies=self.cfg.notch_frequencies if self.cfg.notch_frequencies else None,
            )

            if audio is not None and audio.size > 0:
                power = np.mean(audio ** 2)
                power_db = 10.0 * np.log10(power + 1e-10)
                self.signal_power_db = float(power_db)
                self._update_audio_metrics(audio)
            else:
                self.signal_power_db = None

            if audio is not None and self.cfg.squelch_db is not None and audio.size > 0:
                if self.rssi_db is not None and self.rssi_db < self.cfg.squelch_db:
                    audio = np.zeros_like(audio)

        elif self.cfg.mode == "p25":
            # P25 digital voice - metrics only for now
            if self._p25_decoder is None:
                self._p25_decoder = P25Decoder(sample_rate)
                self._p25_decoder.on_voice_frame = lambda voice_data: self._handle_p25_voice(voice_data)
                self._p25_decoder.on_grant = lambda tgid, freq: self._handle_trunking_grant(tgid, freq)
                logger.info(f"Channel {self.cfg.id}: P25 decoder initialized")

            base = freq_shift(iq, self.cfg.offset_hz, sample_rate)
            if base.size > 0:
                power = np.mean(np.abs(base) ** 2)
                power_db = 10.0 * np.log10(power + 1e-10)
                self.signal_power_db = float(power_db)
            else:
                self.signal_power_db = None

            try:
                frames = self._p25_decoder.process_iq(base)
                for frame in frames:
                    if frame.tgid is not None:
                        logger.debug(f"Channel {self.cfg.id}: P25 frame type={frame.frame_type.value} TGID={frame.tgid}")
                    elif frame.tsbk_data:
                        logger.debug(f"Channel {self.cfg.id}: P25 TSBK: {frame.tsbk_data}")
            except Exception as e:
                logger.error(f"Channel {self.cfg.id}: P25 decoding error: {e}")
            return None  # P25 doesn't output audio yet

        elif self.cfg.mode == "dmr":
            # DMR digital voice - metrics only for now
            if self._dmr_decoder is None:
                self._dmr_decoder = DMRDecoder(sample_rate)
                self._dmr_decoder.on_voice_frame = lambda slot, tgid, voice_data: self._handle_dmr_voice(slot, tgid, voice_data)
                self._dmr_decoder.on_csbk_message = lambda msg: self._handle_dmr_csbk(msg)
                logger.info(f"Channel {self.cfg.id}: DMR decoder initialized")

            base = freq_shift(iq, self.cfg.offset_hz, sample_rate)
            if base.size > 0:
                power = np.mean(np.abs(base) ** 2)
                power_db = 10.0 * np.log10(power + 1e-10)
                self.signal_power_db = float(power_db)
            else:
                self.signal_power_db = None

            try:
                dmr_frames = self._dmr_decoder.process_iq(base)
                for dmr_frame in dmr_frames:
                    logger.debug(f"Channel {self.cfg.id}: DMR frame type={dmr_frame.frame_type.value} slot={dmr_frame.slot.value} dst={dmr_frame.dst_id}")
            except Exception as e:
                logger.error(f"Channel {self.cfg.id}: DMR decoding error: {e}")
            return None  # DMR doesn't output audio yet

        elif self.cfg.mode == "raw":
            base = freq_shift(iq, self.cfg.offset_hz, sample_rate)
            if base.size > 0:
                power = np.mean(np.abs(base) ** 2)
                power_db = 10.0 * np.log10(power + 1e-10)
                self.signal_power_db = float(power_db)
            else:
                self.signal_power_db = None

            # Convert IQ to interleaved float32
            iq_interleaved = np.empty(base.size * 2, dtype=np.float32)
            iq_interleaved[0::2] = base.real
            iq_interleaved[1::2] = base.imag
            audio = iq_interleaved

        return audio

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

    def _handle_dmr_csbk(self, msg: Dict[str, Any]) -> None:
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
    name: Optional[str] = None  # User-provided name (optional)
    auto_name: Optional[str] = None  # Auto-generated name (e.g., "FM 90.3 - RTL-SDR")
    # SoapySDR advanced features
    device_settings: Dict[str, Any] = field(default_factory=dict)
    element_gains: Dict[str, float] = field(default_factory=dict)
    stream_format: Optional[str] = None
    dc_offset_auto: bool = True
    iq_balance_auto: bool = True
    # FFT/Spectrum settings
    fft_fps: int = 15  # Target FFT frames per second (1-60)
    fft_size: int = 2048  # FFT bin count (512, 1024, 2048, 4096)


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
    _fft_sinks: Set[Tuple[asyncio.Queue[Dict[str, Any]], asyncio.AbstractEventLoop]] = field(
        default_factory=set
    )  # Spectrum/FFT subscribers (only calculated when needed for efficiency)
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _channels: Dict[str, Channel] = field(default_factory=dict)
    # Retry tracking (inspired by OpenWebRX)
    _retry_count: int = 0
    _max_retries: int = 10  # OpenWebRX uses 10
    _retry_delay: float = 15.0  # OpenWebRX uses 15 seconds
    _retry_timer: Optional[threading.Timer] = None
    _auto_restart_enabled: bool = False  # Disabled: auto-restart tends to cause thrashing
    _last_run_time: float = 0.0  # Track when the thread was last running
    # IQ watchdog - detect stuck devices (no samples being received)
    _last_iq_time: float = 0.0  # Track when IQ samples were last received
    _iq_watchdog_timeout: float = 30.0  # Trigger recovery if no IQ for this long
    _iq_watchdog_enabled: bool = True
    # Startup watchdog - detect stuck device initialization
    _startup_time: float = 0.0  # When capture entered "starting" state
    _startup_timeout: float = 45.0  # Max time allowed in "starting" state (fallback)
    _startup_watchdog_enabled: bool = True
    # FFT data (server-side spectrum analyzer)
    _fft_power: Optional[np.ndarray] = None  # Power spectrum in dB
    _fft_freqs: Optional[np.ndarray] = None  # Frequency bins in Hz
    _fft_power_list: Optional[List[float]] = None  # Cached Python list (avoids repeated .tolist())
    _fft_freqs_list: Optional[List[float]] = None  # Cached Python list (avoids repeated .tolist())
    _fft_counter: int = 0  # Frame counter for adaptive FFT throttling
    _fft_window_cache: Dict[int, np.ndarray] = field(default_factory=dict)  # Cached FFT windows by size
    _fft_last_time: float = 0.0  # Last FFT timestamp for FPS calculation
    _fft_actual_fps: float = 0.0  # Actual measured FFT FPS
    # Main event loop for scheduling audio processing when no subscribers
    _main_loop: Optional[asyncio.AbstractEventLoop] = None
    # IQ overflow tracking for error indicator UI
    _iq_overflow_count: int = 0
    _iq_overflow_batch: int = 0  # Batched count for rate-limited reporting
    _iq_overflow_last_report: float = 0.0
    # Performance timing metrics
    _perf_loop_times: List[float] = field(default_factory=list)  # Recent loop times in ms
    _perf_dsp_times: List[float] = field(default_factory=list)   # Recent DSP times in ms
    _perf_fft_times: List[float] = field(default_factory=list)   # Recent FFT times in ms
    _perf_max_samples: int = 100  # Keep last 100 samples for rolling average
    _perf_loop_counter: int = 0   # Counter for periodic logging

    def get_perf_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this capture."""
        def _stats(times: List[float]) -> Dict[str, Any]:
            if not times:
                return {"mean_ms": 0.0, "max_ms": 0.0, "samples": 0}
            return {
                "mean_ms": round(sum(times) / len(times), 2),
                "max_ms": round(max(times), 2),
                "samples": len(times),
            }
        return {
            "loop": _stats(self._perf_loop_times),
            "dsp": _stats(self._perf_dsp_times),
            "fft": _stats(self._perf_fft_times),
            "iq_overflow_count": self._iq_overflow_count,
            "channel_count": len([ch for ch in self._channels.values() if ch.state == "running"]),
        }

    def _record_perf_time(self, times_list: List[float], time_ms: float) -> None:
        """Record a timing sample, maintaining rolling window."""
        times_list.append(time_ms)
        if len(times_list) > self._perf_max_samples:
            times_list.pop(0)

    def _report_iq_overflow(self) -> None:
        """Rate-limited overflow reporting (max once per second)."""
        now = time.time()
        if now - self._iq_overflow_last_report >= 1.0 and self._iq_overflow_batch > 0:
            from .error_tracker import get_error_tracker, ErrorEvent
            get_error_tracker().record(ErrorEvent(
                type="iq_overflow",
                capture_id=self.cfg.id,
                channel_id=None,
                timestamp=now,
                count=self._iq_overflow_batch,
            ))
            self._iq_overflow_batch = 0
            self._iq_overflow_last_report = now

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
        """Monitor capture thread health and restart if crashed (OpenWebRX pattern).

        Also implements IQ watchdog: if no IQ samples received for _iq_watchdog_timeout
        seconds while state is 'running', attempts SDRplay service recovery.
        """
        while not self._stop_event.is_set():
            try:
                now = time.time()

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
                    self._last_run_time = now

                # Startup watchdog: detect stuck device initialization
                if (
                    self._startup_watchdog_enabled
                    and self.state == "starting"
                    and self._startup_time > 0
                    and (now - self._startup_time) > self._startup_timeout
                ):
                    elapsed = now - self._startup_time
                    driver = "unknown"
                    if self.requested_device_id:
                        driver = "sdrplay" if "sdrplay" in self.requested_device_id.lower() else "other"

                    print(
                        f"[WARNING] Capture {self.cfg.id} startup timeout: "
                        f"stuck in 'starting' for {elapsed:.1f}s (driver: {driver})",
                        flush=True
                    )

                    # Set state to failed with descriptive error
                    # Only set generic timeout message if there isn't already a specific error
                    self.state = "failed"
                    if not self.error_message:
                        if driver == "sdrplay":
                            self.error_message = (
                                f"Startup timeout: device initialization hung for {elapsed:.0f}s. "
                                "The SDRplay service may be stuck. "
                                "Try: POST /api/v1/devices/sdrplay/restart-service"
                            )
                            # Invalidate caches so stale devices aren't shown
                            from .devices.soapy import invalidate_sdrplay_caches
                            invalidate_sdrplay_caches()
                        else:
                            self.error_message = (
                                f"Startup timeout: device initialization hung for {elapsed:.0f}s. "
                                "Check device connection and ensure no other application is using it."
                            )
                    self._startup_time = 0.0  # Reset to prevent repeated triggers
                    continue  # Skip IQ watchdog check

                # IQ watchdog: detect stuck device (no samples for too long)
                if (
                    self._iq_watchdog_enabled
                    and self.state == "running"
                    and self._last_iq_time > 0
                    and (now - self._last_iq_time) > self._iq_watchdog_timeout
                ):
                    driver = "unknown"
                    if self.device and hasattr(self.device, "info"):
                        driver = getattr(self.device.info, "driver", "unknown")

                    print(
                        f"[WARNING] Capture {self.cfg.id} IQ watchdog triggered: "
                        f"no samples for {now - self._last_iq_time:.1f}s (driver: {driver})",
                        flush=True
                    )

                    # Don't attempt automatic recovery - it tends to make things worse.
                    # Just log the warning and set state to failed so user can see the issue.
                    # Manual restart of the service/WaveCap is more reliable.
                    self.state = "failed"
                    self._last_iq_time = now  # Reset to avoid log spam
                    # Don't auto-restart - let it stay in error state

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

        # Try to capture the main event loop for audio processing
        # (needed when there are no audio subscribers but we still want metrics)
        try:
            self._main_loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - will try again when audio subscribers connect
            pass

        self.state = "starting"
        self.error_message = None
        self._startup_time = time.time()  # Record startup time for watchdog
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

        # Keep auto-restart disabled - manual restart is more reliable
        # self._auto_restart_enabled = True

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
    ) -> list[str]:
        """Reconfigure capture, using hot reconfiguration when possible.

        If only hot-reconfigurable parameters are changed (center_hz, gain,
        bandwidth, ppm), the capture will be updated without restarting.
        If sample_rate, antenna, or advanced settings change, a full restart is required.

        Returns:
            List of channel IDs that were removed because they fell outside
            the new capture bandwidth.
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
        elif was_running and self.device is not None:
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

        # Clean up channels that are now outside the capture bandwidth
        # A channel is out-of-band if |offset_hz| > sample_rate / 2
        removed_channels: list[str] = []
        max_offset = self.cfg.sample_rate / 2.0
        for ch_id, ch in list(self._channels.items()):
            if abs(ch.cfg.offset_hz) > max_offset:
                print(
                    f"[INFO] Removing channel {ch_id}: offset {ch.cfg.offset_hz/1e3:.0f} kHz "
                    f"exceeds capture bandwidth ({max_offset/1e3:.0f} kHz)",
                    flush=True
                )
                ch.stop()
                del self._channels[ch_id]
                removed_channels.append(ch_id)

        return removed_channels

    async def subscribe_iq(self) -> asyncio.Queue[bytes]:
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=8)
        loop = asyncio.get_running_loop()
        self._iq_sinks.add((q, loop))
        return q

    def unsubscribe(self, q: asyncio.Queue[bytes]) -> None:
        for item in list(self._iq_sinks):
            if item[0] is q:
                self._iq_sinks.discard(item)

    async def subscribe_fft(self) -> asyncio.Queue[Dict[str, Any]]:
        """Subscribe to FFT/spectrum data. Only calculated when there are active subscribers."""
        q: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=4)
        loop = asyncio.get_running_loop()
        self._fft_sinks.add((q, loop))
        logger.info(f"FFT subscriber added for capture {self.cfg.id}, total subs: {len(self._fft_sinks)}")
        return q

    def unsubscribe_fft(self, q: asyncio.Queue[Dict[str, Any]]) -> None:
        """Unsubscribe from FFT/spectrum data."""
        for item in list(self._fft_sinks):
            if item[0] is q:
                self._fft_sinks.discard(item)

    def _calculate_fft(self, samples: np.ndarray, sample_rate: int, fft_size: int = 2048) -> None:
        """Calculate FFT for spectrum display. Only called when there are active subscribers.

        Performance: Uses scipy.fft which is 2-3x faster than numpy.fft.
        """
        if samples.size < fft_size:
            return

        # Take a chunk of samples for FFT
        chunk = samples[:fft_size]

        # Get cached Hanning window or create and cache it
        if fft_size not in self._fft_window_cache:
            self._fft_window_cache[fft_size] = np.hanning(fft_size).astype(np.float32)
        window = self._fft_window_cache[fft_size]

        # Apply window to reduce spectral leakage
        windowed = chunk * window

        # Perform FFT using scipy.fft (faster than numpy.fft)
        # Uses module-level import for performance (no import overhead in hot loop)
        if SCIPY_FFT_AVAILABLE:
            fft_result = fft(windowed)
            fft_shifted = fftshift(fft_result)
            freqs = fftshift(fftfreq(fft_size, 1.0 / sample_rate))
        else:
            # Fallback to numpy.fft
            fft_result = np.fft.fft(windowed)
            fft_shifted = np.fft.fftshift(fft_result)
            freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, 1.0 / sample_rate))

        # Calculate power spectrum in dB
        magnitude = np.abs(fft_shifted)
        power_db = 20.0 * np.log10(magnitude + 1e-10)

        # Store results and pre-convert to Python lists (avoids repeated .tolist())
        self._fft_power = power_db
        self._fft_freqs = freqs
        self._fft_power_list = power_db.tolist()
        self._fft_freqs_list = freqs.tolist()

    async def _broadcast_fft(self) -> None:
        """Broadcast FFT data to all subscribers."""
        if not self._fft_sinks or self._fft_power is None or self._fft_freqs is None:
            return

        # Create payload with FFT data (use cached lists to avoid repeated .tolist())
        payload = {
            "power": self._fft_power_list,
            "freqs": self._fft_freqs_list,
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

    def _process_channels_parallel(
        self,
        samples: np.ndarray,
        executor: ThreadPoolExecutor,
    ) -> List[Tuple["Channel", Optional[np.ndarray]]]:
        """Process all running channels in parallel using ThreadPoolExecutor.

        NumPy/SciPy release the GIL during heavy computation, enabling true
        parallelism in thread pools. This provides ~2.7x speedup for multi-channel
        scenarios.

        For single channel, skips executor overhead and processes directly.
        Stateful decoders (RDS, POCSAG, P25, DMR) are handled after parallel DSP.

        Args:
            samples: IQ samples from device
            executor: ThreadPoolExecutor for DSP work

        Returns:
            List of (channel, processed_audio) tuples
        """
        channels = [ch for ch in self._channels.values() if ch.state == "running"]

        if not channels:
            return []

        # For single channel, skip executor overhead - process directly
        if len(channels) == 1:
            ch = channels[0]
            audio = ch.process_iq_chunk_sync(samples, self.cfg.sample_rate)
            return [(ch, audio)]

        # Submit DSP work to executor (parallel)
        futures: Dict[Future[Tuple[Optional[np.ndarray], Dict[str, Any]]], "Channel"] = {}
        for ch in channels:
            future = executor.submit(
                _process_channel_dsp_stateless,
                samples,
                self.cfg.sample_rate,
                ch.cfg,
            )
            futures[future] = ch

        # Collect results and apply stateful processing
        results: List[Tuple["Channel", Optional[np.ndarray]]] = []
        for future, ch in futures.items():
            try:
                audio, metrics = future.result(timeout=2.0)

                # Update channel metrics from parallel processing
                if 'rssi_db' in metrics:
                    ch.rssi_db = metrics['rssi_db']
                if 'signal_power_db' in metrics:
                    ch.signal_power_db = metrics['signal_power_db']

                # Apply stateful processing in capture thread (RDS, POCSAG, squelch)
                audio = self._apply_stateful_processing(ch, audio, samples)

                # Update audio metrics
                if audio is not None and audio.size > 0:
                    ch._update_audio_metrics(audio)

                results.append((ch, audio))

            except TimeoutError:
                logger.warning(f"Channel {ch.cfg.id} DSP timeout")
                results.append((ch, None))
            except Exception as e:
                logger.error(f"Channel {ch.cfg.id} DSP error: {e}")
                results.append((ch, None))

        return results

    def _apply_stateful_processing(
        self,
        ch: "Channel",
        audio: Optional[np.ndarray],
        iq: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Apply stateful processing that must run in capture thread.

        Handles: RDS, POCSAG, squelch. These require state that cannot be
        shared across threads safely.

        Args:
            ch: Channel to process
            audio: Audio from stateless DSP (may be None for digital modes)
            iq: Original IQ samples (needed for RDS pre-MPX processing)

        Returns:
            Processed audio (may be zeroed if squelched)
        """
        if audio is None:
            return None

        # RDS decoding (WBFM only, needs pre-MPX baseband)
        if ch.cfg.mode == "wbfm" and ch.cfg.enable_rds:
            if self.cfg.sample_rate >= 114000:
                if ch._rds_decoder is None:
                    ch._rds_decoder = RDSDecoder(self.cfg.sample_rate)
                    logger.info(f"Channel {ch.cfg.id}: RDS decoder initialized")

                # Get raw FM baseband for RDS (before MPX filter)
                base = freq_shift(iq, ch.cfg.offset_hz, self.cfg.sample_rate) if ch.cfg.offset_hz != 0.0 else iq
                fm_baseband = quadrature_demod(base, self.cfg.sample_rate)

                try:
                    rds_result = ch._rds_decoder.process(fm_baseband)
                    if rds_result:
                        ch.rds_data = rds_result
                        if rds_result.ps_name.strip():
                            logger.debug(f"Channel {ch.cfg.id}: RDS PS={rds_result.ps_name.strip()}")
                except Exception as e:
                    logger.error(f"Channel {ch.cfg.id}: RDS decoding error: {e}")

        # POCSAG decoding (NBFM only)
        if ch.cfg.mode == "nbfm" and ch.cfg.enable_pocsag and audio.size > 0:
            if ch._pocsag_decoder is None:
                ch._pocsag_decoder = POCSAGDecoder(
                    sample_rate=ch.cfg.audio_rate,
                    baud_rate=ch.cfg.pocsag_baud
                )
                logger.info(f"Channel {ch.cfg.id}: POCSAG decoder initialized (baud={ch.cfg.pocsag_baud})")

            try:
                new_msgs = ch._pocsag_decoder.process(audio)
                for msg in new_msgs:
                    ch._pocsag_messages.append(msg)
                    if len(ch._pocsag_messages) > ch._pocsag_max_messages:
                        ch._pocsag_messages.pop(0)
                    logger.info(f"Channel {ch.cfg.id}: POCSAG msg addr={msg.address} func={msg.function}: {msg.message[:50] if msg.message else '(empty)'}")
            except Exception as e:
                logger.error(f"Channel {ch.cfg.id}: POCSAG decoding error: {e}")

        # Apply squelch
        if ch.cfg.squelch_db is not None and audio.size > 0:
            if ch.rssi_db is not None and ch.rssi_db < ch.cfg.squelch_db:
                audio = np.zeros_like(audio)

        return audio

    def _run_thread(self) -> None:
        # Configure device and start streaming with retry logic
        @with_retry(max_attempts=3, backoff_factor=2.0)
        def _configure_and_start() -> StreamHandle:
            device = self._ensure_device()

            # Use atomic configure_and_start for devices that support it (SDRplay)
            # This holds the global lock for the entire configure+start sequence
            if hasattr(device, 'configure_and_start'):
                stream = device.configure_and_start(
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
            else:
                # Standard two-phase startup for other devices
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
            return cast(StreamHandle, stream)

        try:
            self._stream = _configure_and_start()
            # Successfully started!
            self.state = "running"
            self._startup_time = 0.0  # Clear startup time - we're running now
            self._retry_count = 0  # Reset retry counter on success
            # Reset overflow counters on fresh start
            self._iq_overflow_count = 0
            self._iq_overflow_batch = 0
        except Exception as e:
            # Import SDRplay-specific exceptions for special handling
            from .devices.soapy import SDRplayServiceError, TimeoutError as SoapyTimeoutError, invalidate_sdrplay_caches

            # Handle SDRplay service errors specially - don't auto-retry, need manual intervention
            if isinstance(e, SDRplayServiceError):
                self.error_message = str(e)
                self.state = "failed"
                print(f"[ERROR] Capture {self.cfg.id} SDRplay service error: {e}", flush=True)
                # Invalidate caches so stale devices aren't shown
                invalidate_sdrplay_caches()
                self.release_device()
                # Don't schedule restart - service needs manual intervention
                return

            # Handle timeout errors - may succeed after recovery
            if isinstance(e, SoapyTimeoutError):
                self.error_message = str(e)
                self.state = "failed"
                print(f"[ERROR] Capture {self.cfg.id} device timeout: {e}", flush=True)
                # Invalidate caches to trigger fresh enumeration
                if "sdrplay" in str(self.cfg.device_id or "").lower():
                    invalidate_sdrplay_caches()
                self.release_device()
                # Schedule restart - recovery may have fixed the issue
                self._schedule_restart()
                return

            # Generic error - schedule automatic restart
            self.error_message = f"Failed to start capture: {str(e)}"
            print(f"[ERROR] Capture {self.cfg.id} failed to start: {e}", flush=True)
            # Report device retry event
            from .error_tracker import get_error_tracker, ErrorEvent
            get_error_tracker().record(ErrorEvent(
                type="device_retry",
                capture_id=self.cfg.id,
                channel_id=None,
                timestamp=time.time(),
                details={
                    "attempt": self._retry_count + 1,
                    "max_attempts": self._max_retries,
                    "delay_seconds": self._retry_delay,
                    "error": str(e),
                },
            ))
            self.release_device()
            self._schedule_restart()
            return
        # Scale chunk size with sample rate for ~50ms chunks
        # Larger chunks = fewer loop iterations = less overhead = fewer overflows
        # At 1MHz: 50,000 samples = 50ms (20 chunks/sec)
        # At 6MHz: 300,000 samples = 50ms (20 chunks/sec)
        # This gives more time for DSP processing between reads
        chunk = max(8192, self.cfg.sample_rate // 20)
        # Initialize IQ watchdog timestamp
        self._last_iq_time = time.time()
        import time as time_module  # Explicit import for perf_counter
        while not self._stop_event.is_set():
            loop_start = time_module.perf_counter()
            try:
                samples, overflow = self._stream.read(chunk)
                if overflow:
                    self._iq_overflow_count += 1
                    self._iq_overflow_batch += 1
                    self._report_iq_overflow()
            except Exception:
                break
            if samples.size == 0:
                # Light backoff to avoid busy-spin
                try:
                    threading.Event().wait(0.001)
                except Exception:
                    pass
                continue
            # Update IQ watchdog - we received samples
            self._last_iq_time = time.time()
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
                # Use default args to capture loop variables (avoids closure issues)
                def _try_put(q: asyncio.Queue[bytes] = q, payload: bytes = payload) -> None:
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
            # OPTIMIZATION: Only compute metrics if channel has subscribers OR every 10th chunk
            # This reduces CPU load by ~90% when channels are idle (no audio listeners)
            chans = list(self._channels.values())
            for ch in chans:
                if ch.state == "running":
                    ch._metrics_counter = getattr(ch, '_metrics_counter', 0) + 1
                    # Compute metrics if: has audio subscribers, OR every 10th chunk for API polling
                    if ch._audio_sinks or ch._metrics_counter % 10 == 0:
                        # Update signal metrics synchronously (RSSI, SNR)
                        # No copy needed - synchronous read-only operation
                        ch.update_signal_metrics(samples, self.cfg.sample_rate)

            # Calculate FFT for spectrum display (only if there are active subscribers)
            fft_time_ms = 0.0
            if self._fft_sinks:
                # Debug: log that we have FFT subscribers
                if self._fft_counter % 100 == 0:
                    logger.debug(f"FFT processing for {self.cfg.id}: {len(self._fft_sinks)} subscribers, counter={self._fft_counter}")

                # Adaptive FFT FPS based on subscriber count
                # - No viewers: low FPS (5) to save CPU
                # - 1 viewer: configured FPS (default 15)
                # - 2+ viewers: boost FPS for better responsiveness (up to 30)
                base_fps = self.cfg.fft_fps or 15
                subscriber_count = len(self._fft_sinks)
                if subscriber_count >= 2:
                    target_fps = min(30, base_fps * 2)
                elif subscriber_count == 1:
                    target_fps = base_fps
                else:
                    target_fps = 5  # Minimal FPS when no active viewers

                # Calculate FFT rate using ACTUAL chunk size (not hardcoded 8192)
                # chunk = max(8192, sample_rate // 20) from line 2123
                current_fft_rate = self.cfg.sample_rate / chunk

                # Calculate skip interval to achieve target FPS
                skip_interval = max(1, int(current_fft_rate / target_fps))

                # Increment frame counter
                self._fft_counter += 1

                # Only calculate FFT every Nth frame
                if self._fft_counter % skip_interval == 0:
                    # No copy needed - synchronous read-only operation
                    fft_start = time_module.perf_counter()
                    fft_size = self.cfg.fft_size or 2048
                    self._calculate_fft(samples, self.cfg.sample_rate, fft_size)
                    fft_time_ms = (time_module.perf_counter() - fft_start) * 1000
                    self._record_perf_time(self._perf_fft_times, fft_time_ms)

                    # Calculate actual FPS (exponential moving average)
                    now = time_module.perf_counter()
                    if self._fft_last_time > 0:
                        delta = now - self._fft_last_time
                        if delta > 0:
                            instant_fps = 1.0 / delta
                            # EMA with alpha=0.1 for smooth FPS display
                            self._fft_actual_fps = 0.9 * self._fft_actual_fps + 0.1 * instant_fps
                    self._fft_last_time = now

                    # Broadcast FFT to subscribers (use cached lists to avoid repeated .tolist())
                    if self._fft_power_list is not None and self._fft_freqs_list is not None:
                        payload_fft = {
                            "power": self._fft_power_list,
                            "freqs": self._fft_freqs_list,
                            "centerHz": self.cfg.center_hz,
                            "sampleRate": self.cfg.sample_rate,
                            "fftSize": fft_size,
                            "actualFps": round(self._fft_actual_fps, 1),
                        }
                        for (fft_q, fft_loop) in list(self._fft_sinks):
                            # Use default args to capture loop variables (avoids closure issues)
                            def _try_put_fft(q: asyncio.Queue[Dict[str, Any]] = fft_q, payload: Dict[str, Any] = payload_fft) -> None:
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
                                fft_loop.call_soon_threadsafe(_try_put_fft)
                            except Exception:
                                try:
                                    self._fft_sinks.discard((fft_q, fft_loop))
                                except Exception:
                                    pass

            # Dispatch to channels for audio processing
            # PARALLEL DSP: Use ThreadPoolExecutor for ~2.7x speedup with multiple channels
            # NumPy/SciPy release the GIL during heavy computation, enabling true parallelism
            # Only broadcast is scheduled on the event loop (lightweight queue operations)
            dsp_time_ms = 0.0
            if chans:
                from .app import get_dsp_executor
                dsp_executor = get_dsp_executor()
                samples_for_channels = samples.copy()

                # Process all channels in parallel (or sequentially for single channel)
                dsp_start = time_module.perf_counter()
                channel_results = self._process_channels_parallel(samples_for_channels, dsp_executor)
                dsp_time_ms = (time_module.perf_counter() - dsp_start) * 1000
                self._record_perf_time(self._perf_dsp_times, dsp_time_ms)

                # Broadcast audio to subscribers
                for ch, audio in channel_results:
                    # Only schedule broadcast if we have audio data
                    if audio is None:
                        continue

                    # Get event loop for broadcast - try audio sinks first, then IQ sinks, then main loop
                    target_loop = None
                    try:
                        if ch._audio_sinks:
                            target_loop = next(iter(ch._audio_sinks))[1]
                    except (StopIteration, IndexError):
                        pass

                    if target_loop is None:
                        try:
                            if self._iq_sinks:
                                target_loop = next(iter(self._iq_sinks))[1]
                        except Exception:
                            pass

                    if target_loop is None:
                        target_loop = self._main_loop

                    # Schedule only the lightweight broadcast on the event loop
                    if target_loop is not None:
                        try:
                            coro = ch._broadcast(audio)
                            asyncio.run_coroutine_threadsafe(coro, target_loop)
                        except Exception as e:
                            import sys
                            print(f"Error scheduling broadcast: {e}", file=sys.stderr, flush=True)

            # Record loop time and periodic logging
            loop_time_ms = (time_module.perf_counter() - loop_start) * 1000
            self._record_perf_time(self._perf_loop_times, loop_time_ms)
            self._perf_loop_counter += 1

            # Log performance metrics every 500 iterations (~25 seconds at 20 Hz)
            if self._perf_loop_counter % 500 == 0:
                stats = self.get_perf_stats()
                logger.debug(
                    f"Capture {self.cfg.id} perf: loop={stats['loop']['mean_ms']:.1f}ms, "
                    f"dsp={stats['dsp']['mean_ms']:.1f}ms, fft={stats['fft']['mean_ms']:.1f}ms, "
                    f"channels={stats['channel_count']}, overflows={stats['iq_overflow_count']}"
                )


class CaptureManager:
    def __init__(self, cfg: AppConfig, driver: DeviceDriver):
        self._cfg = cfg
        self._driver = driver
        self._captures: Dict[str, Capture] = {}
        self._channels: Dict[str, Channel] = {}
        self._next_cap_id = 1
        self._next_chan_id = 1

    def list_devices(self) -> List[Dict[str, Any]]:
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

    def _apply_mode_defaults(self, mode: str, cfg: ChannelConfig) -> None:
        """Apply mode-specific default filter settings.

        Mode-specific defaults:
        - WBFM: Deemphasis ON (75µs), MPX filter ON (15 kHz), AGC OFF
        - NBFM: All filters OFF by default (often used for voice comms without pre-emphasis)
        - AM: HPF ON (100 Hz), LPF ON (5 kHz), AGC ON (-20 dB)
        - SSB: BPF ON (300-3000 Hz), AGC ON (-20 dB), USB mode
        """
        if mode == "wbfm":
            # WBFM broadcast defaults
            cfg.enable_deemphasis = True
            cfg.deemphasis_tau_us = 75.0  # 75µs for US, 50µs for EU
            cfg.enable_mpx_filter = True  # Remove 19 kHz pilot tone
            cfg.mpx_cutoff_hz = 15_000
            cfg.enable_fm_highpass = False
            cfg.enable_fm_lowpass = False
            cfg.enable_agc = False  # FM typically doesn't need AGC

        elif mode == "nbfm":
            # NBFM voice comms defaults
            cfg.enable_deemphasis = False  # NBFM often doesn't use deemphasis
            cfg.enable_mpx_filter = False  # No stereo in NBFM
            cfg.enable_fm_highpass = False  # User can enable if needed
            cfg.fm_highpass_hz = 300
            cfg.enable_fm_lowpass = False  # User can enable if needed
            cfg.fm_lowpass_hz = 3_000
            cfg.enable_agc = False

        elif mode == "am":
            # AM broadcast/aviation defaults
            cfg.enable_am_highpass = True  # Remove DC offset
            cfg.am_highpass_hz = 100
            cfg.enable_am_lowpass = True  # Broadcast bandwidth
            cfg.am_lowpass_hz = 5_000
            cfg.enable_agc = True  # AM needs AGC
            cfg.agc_target_db = -20.0
            cfg.agc_attack_ms = 5.0
            cfg.agc_release_ms = 50.0

        elif mode == "ssb":
            # SSB ham/marine defaults
            cfg.enable_ssb_bandpass = True  # Voice bandwidth
            cfg.ssb_bandpass_low_hz = 300
            cfg.ssb_bandpass_high_hz = 3_000
            cfg.ssb_mode = "usb"  # Default to upper sideband
            cfg.enable_agc = True  # SSB needs AGC
            cfg.agc_target_db = -20.0
            cfg.agc_attack_ms = 5.0
            cfg.agc_release_ms = 50.0

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
        # Apply mode-specific filter defaults
        self._apply_mode_defaults(mode, cfg)
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
