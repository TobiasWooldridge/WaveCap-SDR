from __future__ import annotations

import asyncio
import functools
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar, cast

import numpy as np
from wavecapsdr.typing import NDArrayComplex, NDArrayFloat

from .dsp.fft import FFTBackend
from .dsp.fft import available_backends as available_fft_backends

# Import FFT backend (pluggable: scipy, fftw, mlx, cuda)
from .dsp.fft import get_backend as get_fft_backend

logger = logging.getLogger(__name__)

_decimate_debug_count = 0
_invalid_audio_log_times: dict[str, float] = {}
_invalid_audio_log_counts: dict[str, int] = {}
_invalid_audio_log_lock = threading.Lock()

import contextlib

from .channel_classifier import ChannelClassifier
from .config import AppConfig
from .decoders.ambe import DMRVoiceDecoder
from .decoders.dmr import DMRDecoder
from .decoders.imbe import IMBEDecoder
from .decoders.p25 import P25Decoder
from .devices.base import Device, DeviceDriver, StreamHandle
from .dsp.am import am_demod, ssb_demod

# Disabled: automatic recovery tends to cause thrashing
# from .sdrplay_recovery import attempt_recovery
from .dsp.fm import nbfm_demod, quadrature_demod, wbfm_demod
from .dsp.flex import FlexDecoder, FlexMessage
from .dsp.pocsag import POCSAGDecoder, POCSAGMessage
from .dsp.rds import RDSData, RDSDecoder
from .dsp.sam import sam_demod_simple
from .encoders import AudioEncoder, create_encoder
from .trunking.config import P25Modulation
from .validation import validate_audio_samples, validate_finite_array

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


def pack_iq16(samples: NDArrayComplex) -> bytes:
    """Pack complex IQ samples as interleaved 16-bit integers.

    Performance: Uses numpy view for zero-copy interleaving.
    """
    if samples.size == 0:
        return b""
    # View complex64 as pairs of float32 (already interleaved I,Q,I,Q,...)
    x: NDArrayComplex = samples.astype(np.complex64, copy=False)
    # Use view to get interleaved float32 without copying
    interleaved_f32 = x.view(np.float32)
    # Clip and scale in one operation, then convert to int16
    np.clip(interleaved_f32, -1.0, 1.0, out=interleaved_f32)
    result: bytes = (interleaved_f32 * 32767.0).astype(np.int16).tobytes()
    return result


def pack_pcm16(samples: NDArrayFloat) -> bytes:
    """Pack audio samples as 16-bit PCM.

    Performance: Avoids intermediate array allocation.
    """
    if samples.size == 0:
        return b""
    # Use contiguous array for efficient conversion
    x: NDArrayFloat = np.ascontiguousarray(samples, dtype=np.float32)
    np.clip(x, -1.0, 1.0, out=x)
    x *= 32767.0
    result: bytes = x.astype(np.int16).tobytes()
    return result


def pack_f32(samples: NDArrayFloat) -> bytes:
    """Pack samples as 32-bit float (little-endian).

    Performance: Uses in-place clip when possible.
    """
    if samples.size == 0:
        return b""
    x: NDArrayFloat = np.ascontiguousarray(samples, dtype=np.float32)
    np.clip(x, -1.0, 1.0, out=x)
    result: bytes = x.tobytes()
    return result


def _validate_audio_output(audio: NDArrayFloat, context: str) -> bool:
    ok, reason = validate_audio_samples(audio)
    if not ok:
        now = time.time()
        with _invalid_audio_log_lock:
            last = _invalid_audio_log_times.get(context, 0.0)
            _invalid_audio_log_counts[context] = _invalid_audio_log_counts.get(context, 0) + 1
            if now - last >= 5.0:
                count = _invalid_audio_log_counts[context]
                logger.warning(
                    f"{context}: invalid audio samples ({reason}), "
                    f"dropped {count} chunks"
                )
                _invalid_audio_log_times[context] = now
                _invalid_audio_log_counts[context] = 0
        return False
    return True


# Cache for frequency shift exponentials using LRU cache (proper eviction)
@functools.lru_cache(maxsize=64)
def _get_freq_shift_exp(size: int, offset_hz: int, sample_rate: int) -> NDArrayComplex:
    """Get cached complex exponential for frequency shifting.

    Uses LRU cache for proper eviction semantics (vs manual dict with arbitrary removal).
    Note: offset_hz is int (rounded from float) to ensure consistent cache keys.
    """
    n = np.arange(size, dtype=np.float32)
    result: NDArrayComplex = np.exp(-1j * 2.0 * np.pi * (offset_hz / float(sample_rate)) * n).astype(np.complex64)
    return result


def freq_shift(iq: NDArrayComplex, offset_hz: float, sample_rate: int) -> NDArrayComplex:
    """Frequency shift IQ samples by mixing with complex exponential.

    Performance: Caches the complex exponential using LRU cache for repeated
    offset/sample_rate/size combinations, providing 10-15% speedup.
    """
    if offset_hz == 0.0 or iq.size == 0:
        return iq

    # Round offset_hz to int for consistent cache keys (prevents unbounded cache growth
    # from floating-point variations like 1000.0 vs 1000.0000001)
    ph = _get_freq_shift_exp(iq.shape[0], round(offset_hz), sample_rate)
    result: NDArrayComplex = (iq.astype(np.complex64, copy=False) * ph).astype(np.complex64)
    return result


# P25 processing constants
# SDRTrunk targets ~19.2 kHz (4 samples per symbol at 4800 baud)
# Using power-of-2 decimation to get close to 19.2 kHz
P25_TARGET_SAMPLE_RATE = 19200  # 4 samples per symbol (matches SDRTrunk)
P25_CHANNEL_BANDWIDTH = 12500  # P25 uses 12.5 kHz channel bandwidth


def decimate_iq_for_p25(iq: NDArrayComplex, sample_rate: int) -> tuple[NDArrayComplex, int]:
    """Decimate complex IQ samples for P25 processing.

    P25 uses 12.5 kHz channel bandwidth with 4800 baud C4FM.
    Target sample rate is ~48 kHz (10 samples per symbol) for good timing recovery.

    Uses scipy.signal.resample_poly for high-quality rational resampling that
    preserves signal magnitude.

    Args:
        iq: Complex IQ samples (already frequency-shifted to channel)
        sample_rate: Input sample rate in Hz

    Returns:
        Tuple of (decimated IQ, effective sample rate)
    """
    # Target 48 kHz for P25 (10 samples per symbol at 4800 baud)
    TARGET_RATE = 48000

    if sample_rate <= TARGET_RATE:
        return iq, sample_rate

    if iq.size == 0:
        return iq, sample_rate

    try:
        import math

        from scipy import signal as scipy_signal

        # Debug: log before/after magnitudes periodically
        global _decimate_debug_count
        _decimate_debug_count += 1
        _debug = _decimate_debug_count <= 10 or _decimate_debug_count % 100 == 0

        before_mean = np.mean(np.abs(iq)) if _debug else 0

        # Find rational approximation for resampling ratio
        # up/down where output_rate = sample_rate * up / down
        gcd = math.gcd(sample_rate, TARGET_RATE)
        up = TARGET_RATE // gcd
        down = sample_rate // gcd

        # For P25 with high decimation ratios, simple subsampling works because:
        # 1. The channel has already been frequency-shifted to baseband
        # 2. The P25 signal (12.5 kHz BW) is narrow compared to output rate (48 kHz)
        # 3. Any aliasing from out-of-band signals is filtered by the demodulator
        # Using simple subsample avoids filter-induced signal loss.
        if down > 50:
            # Simple integer decimation for very high ratios
            decim_factor = sample_rate // TARGET_RATE
            output_rate = sample_rate // decim_factor

            # Just subsample - the channel shift already filtered to P25 bandwidth
            result = iq[::decim_factor].astype(np.complex64)

            if _debug:
                after_mean = np.mean(np.abs(result))
                ratio = before_mean / after_mean if after_mean > 1e-10 else 0
                print(
                    f"[DECIM] #{_decimate_debug_count}: "
                    f"before={before_mean:.4f}, after={after_mean:.4f}, "
                    f"ratio={ratio:.1f}x, decim={decim_factor}x (simple), "
                    f"output_rate={output_rate}",
                    flush=True
                )
            return result, output_rate

        # Use resample_poly for proper anti-aliasing resampling
        # Process I and Q separately
        result_i = scipy_signal.resample_poly(iq.real, up, down)
        result_q = scipy_signal.resample_poly(iq.imag, up, down)
        result = (result_i + 1j * result_q).astype(np.complex64)

        output_rate = (sample_rate * up) // down

        if _debug:
            after_mean = np.mean(np.abs(result))
            ratio = before_mean / after_mean if after_mean > 1e-10 else 0
            print(
                f"[DECIM] #{_decimate_debug_count}: "
                f"before={before_mean:.4f}, after={after_mean:.4f}, "
                f"ratio={ratio:.1f}x, resample={up}/{down}, "
                f"output_rate={output_rate}",
                flush=True
            )

        return result, output_rate

    except ImportError:
        # Fallback: simple decimation without filtering
        decim_factor = max(1, sample_rate // TARGET_RATE)
        return iq[::decim_factor].astype(np.complex64), sample_rate // decim_factor


def _process_channel_dsp_stateless(
    samples: NDArrayComplex,
    sample_rate: int,
    cfg: ChannelConfig,
) -> tuple[NDArrayFloat | None, dict[str, Any]]:
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
    metrics: dict[str, Any] = {}
    if samples.size == 0:
        return None, metrics
    if not validate_finite_array(samples):
        logger.warning(f"Channel {cfg.id}: non-finite IQ samples, dropping DSP chunk")
        return None, metrics

    # Frequency shift to channel offset
    base = samples if cfg.offset_hz == 0.0 else freq_shift(samples, cfg.offset_hz, sample_rate)

    # Calculate RSSI from shifted IQ
    if base.size > 0:
        magnitudes = np.abs(base)
        power = np.mean(magnitudes ** 2)
        metrics['rssi_db'] = float(10.0 * np.log10(power + 1e-10))

    audio: NDArrayFloat | None = None

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

    elif cfg.mode == "sam":
        # Synchronous AM with PLL carrier recovery
        audio = sam_demod_simple(
            base,
            sample_rate,
            audio_rate=cfg.audio_rate,
            sideband=cfg.sam_sideband,
            pll_bandwidth=cfg.sam_pll_bandwidth_hz,
            enable_agc=cfg.enable_agc,
            enable_highpass=cfg.enable_am_highpass,
            highpass_hz=cfg.am_highpass_hz,
            enable_lowpass=cfg.enable_am_lowpass,
            lowpass_hz=cfg.am_lowpass_hz,
            agc_target_db=cfg.agc_target_db,
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
            bfo_offset_hz=cfg.ssb_bfo_offset_hz,
        )

    elif cfg.mode == "raw":
        # Raw IQ output (no demodulation)
        if base.size > 0:
            iq_interleaved = np.empty(base.size * 2, dtype=np.float32)
            iq_interleaved[0::2] = base.real
            iq_interleaved[1::2] = base.imag
            audio = iq_interleaved

    # Digital voice modes require stateful decoders - handled separately
    elif cfg.mode in ("p25", "dmr", "nxdn", "dstar", "ysf"):
        # Just return base IQ for stateful processing later
        if base.size > 0:
            power = np.mean(np.abs(base) ** 2)
            metrics['signal_power_db'] = float(10.0 * np.log10(power + 1e-10))
        return None, metrics

    # Calculate signal power from audio
    if audio is not None and audio.size > 0:
        if not _validate_audio_output(audio, f"Channel {cfg.id}"):
            return None, metrics
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
    squelch_db: float | None = None
    name: str | None = None  # User-provided name
    auto_name: str | None = None  # Auto-generated contextual name

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
    ssb_bfo_offset_hz: float = 1500.0  # BFO offset for centering voice passband

    # SAM (Synchronous AM) settings
    sam_sideband: str = "dsb"  # "dsb", "usb", or "lsb"
    sam_pll_bandwidth_hz: float = 50.0  # PLL loop bandwidth (10-200 Hz)

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
    # FLEX decoding (NBFM only, via multimon-ng)
    enable_flex: bool = False  # Disabled by default


@dataclass
class Channel:
    cfg: ChannelConfig
    state: str = "created"
    # Store (queue, loop, format) to support cross-event-loop broadcasting safely
    _audio_sinks: set[tuple[asyncio.Queue[bytes], asyncio.AbstractEventLoop, str]] = field(
        default_factory=set
    )
    # Encoders for compressed formats (mp3, opus, aac) - created on demand
    _encoders: dict[str, AudioEncoder] = field(default_factory=dict)
    # Subscriber counts for encoded formats
    _encoder_subscribers: dict[str, int] = field(default_factory=dict)
    # Signal strength metrics (calculated server-side)
    signal_power_db: float | None = None  # Current signal power in dB
    rssi_db: float | None = None  # Received Signal Strength Indicator from IQ
    snr_db: float | None = None  # Signal-to-Noise Ratio estimate
    # Audio output level metering
    audio_rms_db: float | None = None  # Output audio RMS level in dB
    audio_peak_db: float | None = None  # Output audio peak level in dB
    audio_clipping_count: int = 0  # Number of samples that would have clipped
    # Digital voice decoders (lazily initialized)
    _p25_decoder: P25Decoder | None = None
    _dmr_decoder: DMRDecoder | None = None
    # P25 modulation type (set by TrunkingSystem for control channels)
    p25_modulation: P25Modulation | None = None  # from trunking.config
    # IMBE voice codec decoder for P25 (lazily initialized)
    _imbe_decoder: IMBEDecoder | None = None
    _imbe_loop: asyncio.AbstractEventLoop | None = None
    # DMR voice codec decoder (lazily initialized)
    _dmr_voice_decoder: DMRVoiceDecoder | None = None
    _dmr_voice_loop: asyncio.AbstractEventLoop | None = None
    # RDS decoder for WBFM (lazily initialized)
    _rds_decoder: RDSDecoder | None = None
    rds_data: RDSData | None = None  # Current RDS data (exposed via API)
    # POCSAG decoder for NBFM pager feeds (lazily initialized)
    _pocsag_decoder: POCSAGDecoder | None = None
    _pocsag_messages: list[POCSAGMessage] = field(default_factory=list)
    _pocsag_max_messages: int = 100  # Ring buffer size
    # FLEX decoder for NBFM pager feeds (lazily initialized, via multimon-ng)
    _flex_decoder: FlexDecoder | None = None
    _flex_messages: list[FlexMessage] = field(default_factory=list)
    _flex_max_messages: int = 100  # Ring buffer size
    # TSBK callback for trunking integration (called when P25 TSBK is decoded)
    on_tsbk: Callable[[dict[str, Any]], None] | None = None
    # Raw IQ callback for trunking/scanning (called with wideband IQ before P25 processing)
    # Signature: (iq: NDArrayComplex, sample_rate: int, overflow: bool) -> None
    # overflow=True indicates ring buffer overrun - caller should reset filter/demod state
    on_raw_iq: Callable[[NDArrayComplex, int, bool], None] | None = None
    # Voice channel factory for automatic voice following (called on P25 grants)
    # Signature: (tgid: int, freq_hz: float, source_id: Optional[int]) -> Optional[str]
    # Returns: channel_id of created voice channel, or None if not created
    _voice_channel_factory: Callable[[int, float, int | None], str | None] | None = None
    # Delete callback for cleaning up voice channels
    # Signature: (channel_id: str) -> None
    _voice_channel_delete_callback: Callable[[str], None] | None = None
    # Active voice channels created by this control channel (channel_id -> info)
    _voice_channels: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Voice channel timeout (auto-delete after silence)
    _voice_channel_timeout: float = 3.0  # seconds
    # Parent control channel ID (set on voice channels created by following)
    _parent_control_channel_id: str | None = None
    # Drop tracking for rate-limited logging
    _drop_count: int = 0
    _last_drop_log_time: float = 0.0
    # Metrics counter for throttled signal metrics calculation
    _metrics_counter: int = 0
    _p25_diag_count: int = 0
    _nxdn_warned: bool = False
    _dstar_warned: bool = False
    _ysf_warned: bool = False

    def start(self) -> None:
        self.state = "running"
        self._drop_count = 0
        self._last_drop_log_time = 0.0
        self.audio_clipping_count = 0

    def stop(self) -> None:
        self.state = "stopped"

        def _schedule_stop(decoder_stop_coro: Any, stored_loop: asyncio.AbstractEventLoop | None) -> None:
            """Schedule decoder stop on an available event loop."""
            # Try stored loop first
            if stored_loop is not None and not stored_loop.is_closed():
                with contextlib.suppress(Exception):
                    asyncio.run_coroutine_threadsafe(decoder_stop_coro, stored_loop)
                return

            # Try to get currently running loop (works if called from async context)
            try:
                running_loop = asyncio.get_running_loop()
                if not running_loop.is_closed():
                    running_loop.create_task(decoder_stop_coro)
                    return
            except RuntimeError:
                pass

            # Last resort: try asyncio.run (only works if no loop is running)
            try:
                asyncio.run(decoder_stop_coro)
            except RuntimeError:
                # No available loop; best-effort cleanup failed
                pass

        # Clean up IMBE decoder if running
        if self._imbe_decoder is not None:
            _schedule_stop(self._imbe_decoder.stop(), self._imbe_loop)
            self._imbe_loop = None

        # Clean up DMR voice decoder if running
        if self._dmr_voice_decoder is not None:
            _schedule_stop(self._dmr_voice_decoder.stop(), self._dmr_voice_loop)
            self._dmr_voice_loop = None

        # Clean up FLEX decoder process if running
        if self._flex_decoder is not None:
            with contextlib.suppress(Exception):
                self._flex_decoder.stop()
            self._flex_decoder = None

    def get_pocsag_messages(self, limit: int = 50, since_timestamp: float | None = None) -> list[dict[str, Any]]:
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

    def get_flex_messages(self, limit: int = 50, since_timestamp: float | None = None) -> list[dict[str, Any]]:
        """Get recent FLEX messages.

        Args:
            limit: Maximum number of messages to return
            since_timestamp: Only return messages after this timestamp (for polling)

        Returns:
            List of message dictionaries (most recent first)
        """
        msgs = self._flex_messages
        if since_timestamp is not None:
            msgs = [m for m in msgs if m.timestamp > since_timestamp]
        return [m.to_dict() for m in reversed(msgs[-limit:])]

    def _update_audio_metrics(self, audio: NDArrayFloat) -> None:
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
            from .error_tracker import ErrorEvent, get_error_tracker
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
            _q, loop, fmt = item
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

    def get_queue_stats(self) -> dict[str, Any]:
        """Get queue statistics for monitoring and health checks.

        Returns:
            Dictionary with subscriber counts, queue depths, and format breakdown
        """
        format_stats: dict[str, dict[str, int]] = {}

        for q, _loop, fmt in list(self._audio_sinks):
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

    def update_signal_metrics(self, iq: NDArrayComplex, sample_rate: int) -> None:
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

    async def _broadcast(self, audio: NDArrayFloat) -> None:
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
        for _format, encoder in self._encoders.items():
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
                    with contextlib.suppress(asyncio.QueueEmpty):
                        _ = q.get_nowait()
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
                        with contextlib.suppress(asyncio.QueueEmpty):
                            _ = q.get_nowait()
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
                    with contextlib.suppress(Exception):
                        self._audio_sinks.discard((q, loop, fmt))

    async def process_iq_chunk(self, iq: NDArrayComplex, sample_rate: int) -> None:
        if self.state != "running":
            return
        if iq.size == 0:
            return
        if not validate_finite_array(iq):
            logger.warning(f"Channel {self.cfg.id}: non-finite IQ samples, dropping chunk")
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
            if not _validate_audio_output(audio, f"Channel {self.cfg.id}"):
                return
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
            if not _validate_audio_output(audio, f"Channel {self.cfg.id}"):
                return
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

        elif self.cfg.mode == "sam":
            # Synchronous AM with PLL carrier recovery
            base = freq_shift(iq, self.cfg.offset_hz, sample_rate)
            audio = sam_demod_simple(
                base,
                sample_rate,
                audio_rate=self.cfg.audio_rate,
                sideband=self.cfg.sam_sideband,
                pll_bandwidth=self.cfg.sam_pll_bandwidth_hz,
                enable_agc=self.cfg.enable_agc,
                enable_highpass=self.cfg.enable_am_highpass,
                highpass_hz=self.cfg.am_highpass_hz,
                enable_lowpass=self.cfg.enable_am_lowpass,
                lowpass_hz=self.cfg.am_lowpass_hz,
                agc_target_db=self.cfg.agc_target_db,
            )

            # Calculate signal power in dB (always, for metrics)
            if not _validate_audio_output(audio, f"Channel {self.cfg.id}"):
                return
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
                bfo_offset_hz=self.cfg.ssb_bfo_offset_hz,
            )

            # Calculate signal power in dB (always, for metrics)
            if not _validate_audio_output(audio, f"Channel {self.cfg.id}"):
                return
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
                # NOTE: on_grant is not wired here - trunking systems use their own
                # VoiceRecorder pool for voice following (see TrunkingSystem)
                logger.info(f"Channel {self.cfg.id}: P25 decoder initialized")

            # Initialize IMBE decoder for voice audio (if available)
            # Use a class-level flag to avoid repeated availability checks/warnings
            if not hasattr(self, '_imbe_checked'):
                self._imbe_checked = True
                if IMBEDecoder.is_available():
                    self._imbe_decoder = IMBEDecoder(output_rate=self.cfg.audio_rate, input_rate=sample_rate)
                    self._imbe_loop = asyncio.get_running_loop()
                    asyncio.create_task(self._imbe_decoder.start())
                    logger.info(f"Channel {self.cfg.id}: IMBE decoder initialized (DSD-FME)")
                else:
                    logger.warning(
                        f"Channel {self.cfg.id}: IMBE decoder not available - "
                        "P25 voice will not be decoded. Install DSD-FME for voice support."
                    )

            # Frequency shift to channel offset
            base = freq_shift(iq, self.cfg.offset_hz, sample_rate)
            if base.size > 0 and not validate_finite_array(base):
                logger.warning(f"Channel {self.cfg.id}: non-finite P25 baseband, dropping chunk")
                return

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

                # Log decoded frames and invoke callbacks
                for frame in frames:
                    if frame.tgid is not None:
                        logger.debug(f"Channel {self.cfg.id}: P25 frame type={frame.frame_type.value} TGID={frame.tgid}")
                    elif frame.tsbk_data:
                        logger.debug(f"Channel {self.cfg.id}: P25 TSBK: {frame.tsbk_data}")
                        # Invoke TSBK callback for trunking integration
                        if self.on_tsbk is not None:
                            try:
                                self.on_tsbk(frame.tsbk_data)
                            except Exception as cb_err:
                                logger.error(f"Channel {self.cfg.id}: TSBK callback error: {cb_err}")
            except Exception as e:
                logger.error(f"Channel {self.cfg.id}: P25 decoding error: {e}")

            # Pass discriminator audio to IMBE decoder for voice decoding
            if self._imbe_decoder is not None and base.size > 0:
                try:
                    # Compute FM discriminator output (instantaneous frequency)
                    # This is the same computation used in C4FM demodulation
                    iq_c64: NDArrayComplex = base.astype(np.complex64, copy=False)
                    prod = iq_c64[1:] * np.conj(iq_c64[:-1])
                    discriminator = np.angle(prod) * sample_rate / (2 * np.pi)

                    # Queue discriminator audio for IMBE decoding
                    await self._imbe_decoder.decode(discriminator)

                    # Get decoded audio if available
                    decoded_audio = await self._imbe_decoder.get_audio()
                    if decoded_audio is not None and decoded_audio.size > 0:
                        if _validate_audio_output(decoded_audio, f"Channel {self.cfg.id}"):
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
            if base.size > 0 and not validate_finite_array(base):
                logger.warning(f"Channel {self.cfg.id}: non-finite DMR baseband, dropping chunk")
                return

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
            if not _validate_audio_output(iq_interleaved, f"Channel {self.cfg.id}"):
                return

            await self._broadcast(iq_interleaved)

        else:
            # Unknown mode: ignore
            return

    def process_iq_chunk_sync(self, iq: NDArrayComplex, sample_rate: int) -> NDArrayFloat | None:
        """Synchronous DSP processing - returns audio/IQ data for broadcast.

        This method performs all CPU-intensive DSP work (demodulation, filtering, etc.)
        and returns the processed audio data. It does NOT broadcast the data - that
        should be done separately on the event loop to avoid blocking HTTP requests.

        Returns:
            Processed audio data (NDArrayFloat) or None if channel not running/no output.
        """
        if self.state != "running":
            return None
        if iq.size == 0:
            return None
        if not validate_finite_array(iq):
            logger.warning(f"Channel {self.cfg.id}: non-finite IQ samples, dropping chunk")
            return None

        audio: NDArrayFloat | None = None

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
                if not _validate_audio_output(audio, f"Channel {self.cfg.id}"):
                    return None
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
                if not _validate_audio_output(audio, f"Channel {self.cfg.id}"):
                    return None
                power = np.mean(audio ** 2)
                power_db = 10.0 * np.log10(power + 1e-10)
                self.signal_power_db = float(power_db)
                self._update_audio_metrics(audio)
            else:
                self.signal_power_db = None

            if audio is not None and self.cfg.squelch_db is not None and audio.size > 0:
                if self.rssi_db is not None and self.rssi_db < self.cfg.squelch_db:
                    audio = np.zeros_like(audio)

        elif self.cfg.mode == "sam":
            base = freq_shift(iq, self.cfg.offset_hz, sample_rate)
            audio = sam_demod_simple(
                base,
                sample_rate,
                audio_rate=self.cfg.audio_rate,
                sideband=self.cfg.sam_sideband,
                pll_bandwidth=self.cfg.sam_pll_bandwidth_hz,
                enable_agc=self.cfg.enable_agc,
                enable_highpass=self.cfg.enable_am_highpass,
                highpass_hz=self.cfg.am_highpass_hz,
                enable_lowpass=self.cfg.enable_am_lowpass,
                lowpass_hz=self.cfg.am_lowpass_hz,
                agc_target_db=self.cfg.agc_target_db,
            )

            if audio is not None and audio.size > 0:
                if not _validate_audio_output(audio, f"Channel {self.cfg.id}"):
                    return None
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
                bfo_offset_hz=self.cfg.ssb_bfo_offset_hz,
            )

            if audio is not None and audio.size > 0:
                if not _validate_audio_output(audio, f"Channel {self.cfg.id}"):
                    return None
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
                # NOTE: on_grant is not wired - trunking uses VoiceRecorder pool
                logger.info(f"Channel {self.cfg.id}: P25 decoder initialized")

            base = freq_shift(iq, self.cfg.offset_hz, sample_rate)
            if base.size > 0 and not validate_finite_array(base):
                logger.warning(f"Channel {self.cfg.id}: non-finite P25 baseband, dropping chunk")
                return None
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
                        # Invoke TSBK callback for trunking integration
                        if self.on_tsbk is not None:
                            try:
                                self.on_tsbk(frame.tsbk_data)
                            except Exception as cb_err:
                                logger.error(f"Channel {self.cfg.id}: TSBK callback error: {cb_err}")
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
            if base.size > 0 and not validate_finite_array(base):
                logger.warning(f"Channel {self.cfg.id}: non-finite DMR baseband, dropping chunk")
                return None
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
            if not _validate_audio_output(iq_interleaved, f"Channel {self.cfg.id}"):
                return None
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

    def _handle_trunking_grant(
        self, tgid: int, freq_hz: float, source_id: int | None = None
    ) -> None:
        """Handle P25 trunking voice channel grant.

        When a control channel broadcasts a voice grant, this method:
        1. Checks if we already have a voice channel for this call
        2. If not, uses the voice_channel_factory to create one
        3. Tracks the voice channel for lifecycle management

        Args:
            tgid: Talkgroup ID receiving the grant
            freq_hz: Voice channel frequency in Hz
            source_id: Source radio ID (if known from grant message)
        """
        import time

        # Check if we already have a voice channel for this frequency
        for existing_chan_id, info in list(self._voice_channels.items()):
            if info.get("freq_hz") == freq_hz:
                # Update last grant time (call is continuing)
                info["last_grant_time"] = time.time()
                if source_id and info.get("source_id") != source_id:
                    # Talker change on same frequency
                    logger.info(
                        f"Channel {self.cfg.id}: Talker change on TGID {tgid} - "
                        f"new source {source_id}"
                    )
                    info["source_id"] = source_id
                return

        # No voice channel factory configured - just log
        if self._voice_channel_factory is None:
            logger.info(
                f"Channel {self.cfg.id}: P25 voice grant - TGID {tgid} on "
                f"{freq_hz/1e6:.4f} MHz (no voice following configured)"
            )
            return

        # Create a new voice channel via factory
        logger.info(
            f"Channel {self.cfg.id}: Creating voice channel for TGID {tgid} on "
            f"{freq_hz/1e6:.4f} MHz (source={source_id})"
        )

        try:
            new_chan_id = self._voice_channel_factory(tgid, freq_hz, source_id)
            if new_chan_id:
                self._voice_channels[new_chan_id] = {
                    "tgid": tgid,
                    "freq_hz": freq_hz,
                    "source_id": source_id,
                    "created_time": time.time(),
                    "last_grant_time": time.time(),
                }
                logger.info(
                    f"Channel {self.cfg.id}: Voice channel {new_chan_id} created for "
                    f"TGID {tgid}"
                )
        except Exception as e:
            logger.error(
                f"Channel {self.cfg.id}: Failed to create voice channel: {e}"
            )

    def cleanup_stale_voice_channels(
        self, delete_callback: Callable[[str], None] | None = None
    ) -> list[str]:
        """Clean up voice channels that haven't received grants recently.

        Should be called periodically (e.g., every second) to detect
        calls that have ended (no more grants being received).

        Args:
            delete_callback: Optional callback to delete the channel.
                             If not provided, only removes from tracking.

        Returns:
            List of channel IDs that were cleaned up.
        """
        import time

        stale_ids = []
        now = time.time()

        for chan_id, info in list(self._voice_channels.items()):
            last_grant = info.get("last_grant_time", 0)
            if now - last_grant > self._voice_channel_timeout:
                stale_ids.append(chan_id)
                tgid = info.get("tgid", "?")
                logger.info(
                    f"Channel {self.cfg.id}: Voice channel {chan_id} (TGID {tgid}) "
                    f"ended - no grants for {self._voice_channel_timeout:.1f}s"
                )
                del self._voice_channels[chan_id]
                if delete_callback:
                    try:
                        delete_callback(chan_id)
                    except Exception as e:
                        logger.error(
                            f"Channel {self.cfg.id}: Failed to delete voice "
                            f"channel {chan_id}: {e}"
                        )

        return stale_ids

    def remove_voice_channel(self, chan_id: str) -> None:
        """Remove a voice channel from tracking (called when channel is deleted)."""
        if chan_id in self._voice_channels:
            info = self._voice_channels.pop(chan_id)
            logger.debug(
                f"Channel {self.cfg.id}: Voice channel {chan_id} removed from "
                f"tracking (TGID {info.get('tgid', '?')})"
            )

    def _handle_dmr_csbk(self, msg: dict[str, Any]) -> None:
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
    gain: float | None = None
    bandwidth: float | None = None
    ppm: float | None = None
    antenna: str | None = None
    name: str | None = None  # User-provided name (optional)
    auto_name: str | None = None  # Auto-generated name (e.g., "FM 90.3 - RTL-SDR")
    # SoapySDR advanced features
    device_settings: dict[str, Any] = field(default_factory=dict)
    element_gains: dict[str, float] = field(default_factory=dict)
    agc_enabled: bool = False  # Enable automatic gain control
    stream_format: str | None = None
    dc_offset_auto: bool = True
    iq_balance_auto: bool = True
    # FFT/Spectrum settings
    fft_fps: int = 15  # Target FFT frames per second (1-60)
    fft_max_fps: int = 60  # Maximum FFT frames per second (hard cap, 1-120)
    fft_size: int = 2048  # FFT bin count (512, 1024, 2048, 4096)
    fft_accelerator: str = "auto"  # FFT backend: auto, scipy, fftw, mlx, cuda


@dataclass
class Capture:
    cfg: CaptureConfig
    driver: DeviceDriver
    requested_device_id: str | None = None
    state: str = "stopped"  # Use string for backwards compatibility with API
    device: Device | None = None
    antenna: str | None = None  # Actual antenna in use
    error_message: str | None = None  # Error message if state is "failed"
    trunking_system_id: str | None = None  # If set, capture is owned by a trunking system
    _stream: StreamHandle | None = None
    _thread: threading.Thread | None = None
    _health_monitor: threading.Thread | None = None
    _iq_sinks: set[tuple[asyncio.Queue[bytes], asyncio.AbstractEventLoop]] = field(
        default_factory=set
    )
    _fft_sinks: set[tuple[asyncio.Queue[dict[str, Any]], asyncio.AbstractEventLoop]] = field(
        default_factory=set
    )  # Spectrum/FFT subscribers (only calculated when needed for efficiency)
    _iq_sinks_lock: threading.Lock = field(default_factory=threading.Lock)
    _fft_sinks_lock: threading.Lock = field(default_factory=threading.Lock)
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _channels: dict[str, Channel] = field(default_factory=dict)
    # Retry tracking (inspired by OpenWebRX)
    _retry_count: int = 0
    _max_retries: int = 10  # OpenWebRX uses 10
    _retry_delay: float = 15.0  # OpenWebRX uses 15 seconds
    _retry_timer: threading.Timer | None = None
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
    _fft_power: NDArrayFloat | None = None  # Power spectrum in dB
    _fft_freqs: NDArrayFloat | None = None  # Frequency bins in Hz
    _fft_power_list: list[float] | None = None  # Cached Python list (avoids repeated .tolist())
    _fft_freqs_list: list[float] | None = None  # Cached Python list (avoids repeated .tolist())
    _fft_counter: int = 0  # Frame counter for adaptive FFT throttling
    _fft_window_cache: dict[int, NDArrayFloat] = field(default_factory=dict)  # Cached FFT windows by size (legacy)
    _fft_last_time: float = 0.0  # Last FFT timestamp for FPS calculation
    _fft_actual_fps: float = 0.0  # Actual measured FFT FPS
    _fft_backend: FFTBackend | None = None  # Pluggable FFT backend (scipy/fftw/mlx/cuda)
    # Main event loop for scheduling audio processing when no subscribers
    _main_loop: asyncio.AbstractEventLoop | None = None
    # IQ overflow tracking for error indicator UI
    _iq_overflow_count: int = 0
    _iq_overflow_batch: int = 0  # Batched count for rate-limited reporting
    _iq_overflow_last_report: float = 0.0
    _iq_overflow_current: bool = False  # True if current read had overflow (for callbacks)
    # Performance timing metrics
    _perf_loop_times: list[float] = field(default_factory=list)  # Recent loop times in ms
    _perf_dsp_times: list[float] = field(default_factory=list)   # Recent DSP times in ms
    _perf_fft_times: list[float] = field(default_factory=list)   # Recent FFT times in ms
    _perf_max_samples: int = 100  # Keep last 100 samples for rolling average
    _perf_loop_counter: int = 0   # Counter for periodic logging
    _dsp_inflight: int = 0
    _dsp_inflight_lock: threading.Lock = field(default_factory=threading.Lock)
    _dsp_drop_last_log: float = 0.0
    # Per-capture DSP executor for CPU isolation between captures
    _dsp_executor: ThreadPoolExecutor | None = None
    # Channel classifier for identifying control vs voice channels
    _channel_classifier: ChannelClassifier = field(default_factory=ChannelClassifier)

    def get_perf_stats(self) -> dict[str, Any]:
        """Get performance statistics for this capture."""
        def _stats(times: list[float]) -> dict[str, Any]:
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

    def _record_perf_time(self, times_list: list[float], time_ms: float) -> None:
        """Record a timing sample, maintaining rolling window."""
        times_list.append(time_ms)
        if len(times_list) > self._perf_max_samples:
            times_list.pop(0)

    def _get_dsp_executor(self) -> ThreadPoolExecutor:
        """Get or create per-capture DSP executor.

        Each capture gets its own executor for CPU isolation. This prevents
        captures from competing for DSP threads, ensuring each radio/trunking
        system has dedicated resources.

        Workers sized based on expected channel count (2-4 per capture).
        """
        if self._dsp_executor is None:
            # 3 workers handles typical use (1 control + voice recorders)
            # For heavy multi-channel, the executor will queue work
            max_workers = 3
            self._dsp_executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix=f"DSP-{self.cfg.id}-"
            )
            logger.debug(f"Capture {self.cfg.id}: Created per-capture DSP executor with {max_workers} workers")
        return self._dsp_executor

    def _shutdown_dsp_executor(self) -> None:
        """Shutdown per-capture DSP executor."""
        if self._dsp_executor is not None:
            logger.debug(f"Capture {self.cfg.id}: Shutting down per-capture DSP executor")
            try:
                self._dsp_executor.shutdown(wait=False, cancel_futures=True)
            except Exception as e:
                logger.warning(f"Capture {self.cfg.id}: Error shutting down DSP executor: {e}")
            self._dsp_executor = None

    def _report_iq_overflow(self) -> None:
        """Rate-limited overflow reporting (max once per second)."""
        now = time.time()
        if now - self._iq_overflow_last_report >= 1.0 and self._iq_overflow_batch > 0:
            from .error_tracker import ErrorEvent, get_error_tracker
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
            with contextlib.suppress(Exception):
                self.cfg.device_id = dev.info.id
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

    def _emit_state_change(self, action: str) -> None:
        """Notify subscribers of a capture state transition."""
        from .state_broadcaster import get_broadcaster

        get_broadcaster().emit_capture_change(
            action,
            self.cfg.id,
            {
                "id": self.cfg.id,
                "state": self.state,
                "errorMessage": self.error_message,
            },
        )

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
            self._emit_state_change("updated")
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
                    if self.state in ("running", "starting"):
                        print(
                            f"[WARNING] Capture {self.cfg.id} thread died unexpectedly",
                            flush=True
                        )
                        self.state = "failed"
                        if not self.error_message:
                            self.error_message = "Capture thread died unexpectedly"
                        self._emit_state_change("updated")
                        if self._auto_restart_enabled:
                            self._schedule_restart()
                    continue

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
                    self._emit_state_change("updated")
                    self._startup_time = 0.0  # Reset to prevent repeated triggers
                    continue  # Skip IQ watchdog check

                # IQ watchdog: detect stuck device (no samples for too long)
                if (
                    self._iq_watchdog_enabled
                    and self.state == "running"
                    and self._last_iq_time > 0
                    and (now - self._last_iq_time) > self._iq_watchdog_timeout
                ):
                    # Check if stream is ready yet (SDRplay proxy streams need startup time)
                    stream = self._stream
                    is_ready = False
                    if stream is not None and hasattr(stream, 'is_ready'):
                        is_ready = stream.is_ready()
                        logger.info(f"Capture {self.cfg.id}: IQ watchdog check - is_ready={is_ready}")

                    # If stream is ready but IQ thread hasn't received samples, there may be
                    # a shared memory synchronization issue. Give more time and don't fail yet.
                    if is_ready and not getattr(self, '_watchdog_ready_seen', False):
                        # First time seeing ready - give extra time
                        self._watchdog_ready_seen = True
                        logger.warning(
                            f"Capture {self.cfg.id}: IQ watchdog deferred - stream ready but IQ thread not receiving. "
                            f"Extending timeout for shared memory sync."
                        )
                        self._last_iq_time = now
                        continue

                    if stream is not None and hasattr(stream, 'is_ready') and not is_ready:
                        # Stream not ready yet - defer watchdog, reset timer
                        logger.debug(
                            f"Capture {self.cfg.id}: IQ watchdog deferred - stream not ready yet"
                        )
                        self._last_iq_time = now
                        continue

                    driver = "unknown"
                    if self.device and hasattr(self.device, "info"):
                        driver = getattr(self.device.info, "driver", "unknown")

                    print(
                        f"[WARNING] Capture {self.cfg.id} IQ watchdog triggered: "
                        f"no samples for {now - self._last_iq_time:.1f}s (driver: {driver})",
                        flush=True
                    )
                    logger.warning(
                        f"Capture {self.cfg.id} IQ watchdog: setting state to failed. "
                        f"No samples for {now - self._last_iq_time:.1f}s"
                    )

                    # Don't attempt automatic recovery - it tends to make things worse.
                    # Just log the warning and set state to failed so user can see the issue.
                    # Manual restart of the service/WaveCap is more reliable.
                    self.state = "failed"
                    self.error_message = f"No IQ samples received for {now - self._last_iq_time:.0f}s"
                    self._emit_state_change("updated")
                    self._last_iq_time = now  # Reset to avoid log spam
                    self._stop_event.set()
                    if self._stream is not None:
                        with contextlib.suppress(Exception):
                            self._stream.close()
                        self._stream = None
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
        self._emit_state_change("updated")
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
        self._emit_state_change("updated")
        self._stop_event.set()

        # Close stream early to unblock reads
        if self._stream is not None:
            stream = self._stream
            self._stream = None
            with contextlib.suppress(Exception):
                stream.close()

        # Wait for threads to finish gracefully
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                print(f"[WARNING] Capture {self.cfg.id} thread did not stop gracefully", flush=True)
            self._thread = None

        if self._health_monitor is not None:
            self._health_monitor.join(timeout=1.0)
            self._health_monitor = None

        # Shutdown per-capture DSP executor
        self._shutdown_dsp_executor()

        # Note: We do NOT close the device here - the device should stay open
        # for the lifetime of the Capture. We only close the stream.
        # The device will be closed when the Capture is deleted.
        self.state = "stopped"
        self._emit_state_change("stopped")

        # Keep auto-restart disabled - manual restart is more reliable
        # self._auto_restart_enabled = True

    async def reconfigure(
        self,
        center_hz: float | None = None,
        sample_rate: int | None = None,
        gain: float | None = None,
        bandwidth: float | None = None,
        ppm: float | None = None,
        antenna: str | None = None,
        device_settings: dict[str, Any] | None = None,
        element_gains: dict[str, float] | None = None,
        stream_format: str | None = None,
        dc_offset_auto: bool | None = None,
        iq_balance_auto: bool | None = None,
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
        with self._iq_sinks_lock:
            self._iq_sinks.add((q, loop))
        return q

    def unsubscribe(self, q: asyncio.Queue[bytes]) -> None:
        with self._iq_sinks_lock:
            for item in list(self._iq_sinks):
                if item[0] is q:
                    self._iq_sinks.discard(item)

    async def subscribe_fft(self) -> asyncio.Queue[dict[str, Any]]:
        """Subscribe to FFT/spectrum data. Only calculated when there are active subscribers."""
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=4)
        loop = asyncio.get_running_loop()
        with self._fft_sinks_lock:
            self._fft_sinks.add((q, loop))
        logger.info(f"FFT subscriber added for capture {self.cfg.id}, total subs: {len(self._fft_sinks)}")
        return q

    def unsubscribe_fft(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        """Unsubscribe from FFT/spectrum data."""
        with self._fft_sinks_lock:
            for item in list(self._fft_sinks):
                if item[0] is q:
                    self._fft_sinks.discard(item)

    def _get_fft_backend(self, fft_size: int = 2048) -> FFTBackend:
        """Get or create FFT backend instance.

        Lazily initializes the FFT backend on first use.
        Backend is selected automatically based on available hardware:
        - macOS: MLX (Metal GPU) if available, else scipy
        - Linux/Windows: CuPy (CUDA) if available, else scipy
        - All platforms: pyFFTW (SIMD CPU) as intermediate option
        """
        if self._fft_backend is None or self._fft_backend.fft_size != fft_size:
            # Get accelerator preference from config if available
            accelerator = getattr(self.cfg, 'fft_accelerator', 'auto')
            self._fft_backend = get_fft_backend(accelerator=accelerator, fft_size=fft_size)
            logger.info(
                f"Capture {self.cfg.id}: FFT backend initialized: {self._fft_backend.name} "
                f"(fft_size={fft_size}, available={available_fft_backends()})"
            )
        return self._fft_backend

    def _calculate_fft(self, samples: NDArrayComplex, sample_rate: int, fft_size: int = 2048) -> None:
        """Calculate FFT for spectrum display using pluggable backend.

        Uses hardware-accelerated FFT when available:
        - MLX (Apple Metal): 5-10x faster on Apple Silicon
        - CuPy (CUDA): 10-20x faster on NVIDIA GPUs
        - pyFFTW: 2-3x faster with SIMD on any CPU
        - scipy: Default fallback (always available)
        """
        if samples.size < fft_size:
            return

        # Get or create FFT backend
        backend = self._get_fft_backend(fft_size)

        # Execute FFT using backend
        result = backend.execute(samples, sample_rate)

        # Store results and pre-convert to Python lists (avoids repeated .tolist())
        self._fft_power = result.power_db
        self._fft_freqs = result.freqs
        self._fft_power_list = result.power_db.tolist()
        self._fft_freqs_list = result.freqs.tolist()

        # Feed spectrum data to channel classifier
        self._channel_classifier.update(
            self._fft_power_list,
            self._fft_freqs_list,
            self.cfg.center_hz,
            self.cfg.sample_rate,
        )

    async def _broadcast_fft(self) -> None:
        """Broadcast FFT data to all subscribers."""
        with self._fft_sinks_lock:
            fft_sinks = list(self._fft_sinks)
        if not fft_sinks or self._fft_power is None or self._fft_freqs is None:
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

        for (q, loop) in fft_sinks:
            if current_loop is loop:
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    with contextlib.suppress(asyncio.QueueEmpty):
                        _ = q.get_nowait()
                    with contextlib.suppress(asyncio.QueueFull):
                        q.put_nowait(payload)
            else:
                def _try_put() -> None:
                    try:
                        q.put_nowait(payload)
                    except asyncio.QueueFull:
                        with contextlib.suppress(asyncio.QueueEmpty):
                            _ = q.get_nowait()
                        with contextlib.suppress(asyncio.QueueFull):
                            q.put_nowait(payload)

                try:
                    loop.call_soon_threadsafe(_try_put)
                except Exception:
                    with self._fft_sinks_lock, contextlib.suppress(Exception):
                        self._fft_sinks.discard((q, loop))

    async def _broadcast_iq(self, payload: bytes) -> None:
        with self._iq_sinks_lock:
            iq_sinks = list(self._iq_sinks)
        if not iq_sinks:
            return
        current_loop = None
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        for (q, loop) in iq_sinks:
            if current_loop is loop:
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    with contextlib.suppress(asyncio.QueueEmpty):
                        _ = q.get_nowait()
                    with contextlib.suppress(asyncio.QueueFull):
                        q.put_nowait(payload)
            else:
                def _try_put() -> None:
                    try:
                        q.put_nowait(payload)
                    except asyncio.QueueFull:
                        with contextlib.suppress(asyncio.QueueEmpty):
                            _ = q.get_nowait()
                        with contextlib.suppress(asyncio.QueueFull):
                            q.put_nowait(payload)

                try:
                    loop.call_soon_threadsafe(_try_put)
                except Exception:
                    with self._iq_sinks_lock, contextlib.suppress(Exception):
                        self._iq_sinks.discard((q, loop))

    def _process_channels_parallel(
        self,
        samples: NDArrayComplex,
        executor: ThreadPoolExecutor,
    ) -> list[tuple[Channel, NDArrayFloat | None]]:
        """Process all running channels using ThreadPoolExecutor.

        NumPy/SciPy release the GIL during heavy computation, enabling true
        parallelism in thread pools. Always uses executor to keep capture thread
        responsive - this prevents CPU-intensive modes (P25, DMR, high sample rates)
        from blocking device I/O and causing buffer overflows.

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

        # Always use ThreadPoolExecutor for DSP to keep capture thread responsive
        # This prevents CPU-intensive modes (P25, DMR) from blocking device I/O
        # The ~100-200μs executor overhead is acceptable tradeoff for stability

        # Submit DSP work to executor (parallel), with backpressure to avoid unbounded queueing
        max_workers = getattr(executor, "_max_workers", 4)
        max_inflight = max(4, max_workers * 2)
        with self._dsp_inflight_lock:
            inflight = self._dsp_inflight
        if inflight >= max_inflight:
            now = time.time()
            if now - self._dsp_drop_last_log >= 2.0:
                logger.warning(
                    f"Capture {self.cfg.id}: DSP backlog {inflight}/{max_inflight}, skipping cycle to recover"
                )
                self._dsp_drop_last_log = now
            return []

        futures: dict[Future[tuple[NDArrayFloat | None, dict[str, Any]]], Channel] = {}
        for ch in channels:
            with self._dsp_inflight_lock:
                if self._dsp_inflight >= max_inflight:
                    break
                self._dsp_inflight += 1
            future = executor.submit(
                _process_channel_dsp_stateless,
                samples,
                self.cfg.sample_rate,
                ch.cfg,
            )
            def _done(_fut: Future[Any]) -> None:
                with self._dsp_inflight_lock:
                    self._dsp_inflight = max(0, self._dsp_inflight - 1)
            future.add_done_callback(_done)
            futures[future] = ch

        # Wait for ALL futures simultaneously (not sequentially!)
        # This is critical for multi-channel performance - waiting sequentially
        # on N channels taking T ms each = N*T total blocking time
        # Waiting in parallel = max(T) blocking time
        from concurrent.futures import wait, FIRST_EXCEPTION
        done, not_done = wait(futures.keys(), timeout=0.5, return_when=FIRST_EXCEPTION)

        # Cancel any timed-out futures
        for future in not_done:
            ch = futures[future]
            logger.warning(f"Channel {ch.cfg.id} DSP timeout (parallel wait)")
            future.cancel()

        # Collect results and apply stateful processing
        results: list[tuple[Channel, NDArrayFloat | None]] = []
        for future, ch in futures.items():
            if future in not_done:
                results.append((ch, None))
                continue
            try:
                audio, metrics = future.result(timeout=0)  # Already done, no wait

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

            except Exception as e:
                logger.error(f"Channel {ch.cfg.id} DSP error: {e}")
                results.append((ch, None))

        return results

    def _apply_stateful_processing(
        self,
        ch: Channel,
        audio: NDArrayFloat | None,
        iq: NDArrayComplex,
    ) -> NDArrayFloat | None:
        """Apply stateful processing that must run in capture thread.

        Handles: RDS, POCSAG, P25, DMR, squelch. These require state that cannot be
        shared across threads safely.

        Args:
            ch: Channel to process
            audio: Audio from stateless DSP (may be None for digital modes)
            iq: Original IQ samples (needed for RDS pre-MPX processing)

        Returns:
            Processed audio (may be zeroed if squelched)
        """
        # P25 decoding (requires stateful decoder, processes IQ not audio)
        if ch.cfg.mode == "p25":
            if iq.size == 0:
                return None
            if not validate_finite_array(iq):
                logger.warning(f"Channel {ch.cfg.id}: non-finite IQ samples, dropping P25 chunk")
                return None
            # Call raw IQ callback for trunking/scanning integration
            # This is called with wideband IQ samples BEFORE frequency shifting
            # Pass overflow flag so callback can reset filter/demod state after sample loss
            if ch.on_raw_iq is not None:
                try:
                    ch.on_raw_iq(iq, self.cfg.sample_rate, self._iq_overflow_current)
                except Exception as e:
                    logger.error(f"Channel {ch.cfg.id}: on_raw_iq callback error: {e}")
                # OPTIMIZATION: If TrunkingSystem handles IQ via on_raw_iq,
                # skip duplicate P25 decoding here. TrunkingSystem has its own
                # control channel monitor and voice recorders.
                return None

            # Get frequency-shifted IQ for P25 decoding
            base = freq_shift(iq, ch.cfg.offset_hz, self.cfg.sample_rate) if ch.cfg.offset_hz != 0.0 else iq
            if base.size > 0 and not validate_finite_array(base):
                logger.warning(f"Channel {ch.cfg.id}: non-finite P25 baseband, dropping chunk")
                return None
            p25_decoded_audio: NDArrayFloat | None = None

            if base.size > 0:
                # Diagnostic: Log raw and frequency-shifted IQ magnitude
                if not hasattr(ch, '_p25_diag_count'):
                    ch._p25_diag_count = 0
                ch._p25_diag_count += 1
                if ch._p25_diag_count <= 10 or ch._p25_diag_count % 100 == 0:
                    raw_mag = np.abs(iq)
                    base_mag = np.abs(base)
                    print(
                        f"[P25_IQ] Channel {ch.cfg.id} call #{ch._p25_diag_count}: "
                        f"raw_mean={np.mean(raw_mag):.4f}, raw_max={np.max(raw_mag):.4f}, "
                        f"after_shift_mean={np.mean(base_mag):.4f}, offset={ch.cfg.offset_hz/1e3:.1f}kHz",
                        flush=True
                    )

                # Decimate to P25 processing rate (~48 kHz) if needed
                # This is critical for wideband captures (e.g., 8 MHz trunking)
                p25_iq, p25_rate = decimate_iq_for_p25(base, self.cfg.sample_rate)

                # Initialize P25 frame decoder with the decimated sample rate
                if ch._p25_decoder is None:
                    # Use modulation from channel if set (e.g., by TrunkingSystem)
                    from wavecapsdr.decoders.p25 import P25Modulation as DecoderMod
                    from wavecapsdr.trunking.config import P25Modulation as TrunkingMod

                    modulation = ch.p25_modulation if ch.p25_modulation else TrunkingMod.LSM
                    decoder_mod = DecoderMod(modulation.value)
                    ch._p25_decoder = P25Decoder(p25_rate, modulation=decoder_mod)
                    ch._p25_decoder.on_voice_frame = lambda voice_data: ch._handle_p25_voice(voice_data)
                    # NOTE: on_grant is not wired - trunking uses VoiceRecorder pool
                    logger.info(
                        f"Channel {ch.cfg.id}: P25 decoder initialized "
                        f"(sample_rate={p25_rate}, decimation={self.cfg.sample_rate // p25_rate}x, "
                        f"modulation={decoder_mod.value})"
                    )

                # Initialize IMBE voice decoder for P25 voice decoding via DSD-FME
                if ch._imbe_decoder is None and IMBEDecoder.is_available():
                    ch._imbe_decoder = IMBEDecoder(output_rate=ch.cfg.audio_rate, input_rate=p25_rate)
                    ch._imbe_loop = self._main_loop
                    if self._main_loop is not None:
                        ch._imbe_decoder.start_in_loop(self._main_loop)
                        logger.info(f"Channel {ch.cfg.id}: IMBE decoder initialized (input={p25_rate}Hz, output={ch.cfg.audio_rate}Hz)")

                # Process P25 frames for TSBK/trunking
                try:
                    frames = ch._p25_decoder.process_iq(p25_iq)
                    for frame in frames:
                        if frame.tgid is not None:
                            logger.debug(f"Channel {ch.cfg.id}: P25 frame type={frame.frame_type.value} TGID={frame.tgid}")
                        elif frame.tsbk_data:
                            logger.debug(f"Channel {ch.cfg.id}: P25 TSBK: {frame.tsbk_data}")
                            # Invoke TSBK callback for trunking integration
                            if ch.on_tsbk is not None:
                                try:
                                    ch.on_tsbk(frame.tsbk_data)
                                except Exception as cb_err:
                                    logger.error(f"Channel {ch.cfg.id}: TSBK callback error: {cb_err}")
                except Exception as e:
                    import traceback
                    logger.error(f"Channel {ch.cfg.id}: P25 decoding error: {e}\n{traceback.format_exc()}")

                # P25 IMBE voice decoding via DSD-FME discriminator approach
                # Compute FM discriminator output (instantaneous frequency) and feed to DSD-FME
                if ch._imbe_decoder is not None and ch._imbe_decoder.running and p25_iq.size > 0:
                    try:
                        # Compute FM discriminator (same as C4FM demodulation)
                        p25_iq_c64: NDArrayComplex = p25_iq.astype(np.complex64, copy=False)
                        prod = p25_iq_c64[1:] * np.conj(p25_iq_c64[:-1])
                        discriminator = np.angle(prod) * p25_rate / (2 * np.pi)

                        # Queue discriminator audio for IMBE decoding (non-blocking)
                        ch._imbe_decoder.decode_sync(discriminator)

                        # Get any decoded audio available from previous frames
                        p25_decoded_audio = ch._imbe_decoder.get_audio_sync()
                        if p25_decoded_audio is not None and p25_decoded_audio.size > 0:
                            if not _validate_audio_output(p25_decoded_audio, f"Channel {ch.cfg.id}"):
                                p25_decoded_audio = None
                            else:
                                ch._update_audio_metrics(p25_decoded_audio)
                    except Exception as e:
                        logger.error(f"Channel {ch.cfg.id}: IMBE decoding error: {e}")

            return p25_decoded_audio

        # DMR decoding (requires stateful decoder, processes IQ not audio)
        if ch.cfg.mode == "dmr":
            dmr_decoded_audio: NDArrayFloat | None = None

            if iq.size == 0:
                return None
            if not validate_finite_array(iq):
                logger.warning(f"Channel {ch.cfg.id}: non-finite IQ samples, dropping DMR chunk")
                return None
            # Get frequency-shifted IQ for DMR decoding
            base = freq_shift(iq, ch.cfg.offset_hz, self.cfg.sample_rate) if ch.cfg.offset_hz != 0.0 else iq
            if base.size > 0 and not validate_finite_array(base):
                logger.warning(f"Channel {ch.cfg.id}: non-finite DMR baseband, dropping chunk")
                return None

            if base.size > 0:
                # Initialize DMR frame decoder
                if ch._dmr_decoder is None:
                    ch._dmr_decoder = DMRDecoder(self.cfg.sample_rate)
                    ch._dmr_decoder.on_voice_frame = lambda slot, tgid, voice_data: ch._handle_dmr_voice(slot, tgid, voice_data)
                    ch._dmr_decoder.on_csbk_message = lambda msg: ch._handle_dmr_csbk(msg)
                    logger.info(f"Channel {ch.cfg.id}: DMR decoder initialized")

                # Initialize DMR voice decoder for AMBE+2 voice decoding via DSD-FME
                if ch._dmr_voice_decoder is None and DMRVoiceDecoder.is_available():
                    ch._dmr_voice_decoder = DMRVoiceDecoder(output_rate=ch.cfg.audio_rate, input_rate=self.cfg.sample_rate)
                    ch._dmr_voice_loop = self._main_loop
                    if self._main_loop is not None:
                        ch._dmr_voice_decoder.start_in_loop(self._main_loop)
                        logger.info(f"Channel {ch.cfg.id}: DMR voice decoder initialized (input={self.cfg.sample_rate}Hz, output={ch.cfg.audio_rate}Hz)")

                # Process DMR frames for trunking/CSBK
                try:
                    dmr_frames = ch._dmr_decoder.process_iq(base)
                    for dmr_frame in dmr_frames:
                        logger.debug(f"Channel {ch.cfg.id}: DMR frame type={dmr_frame.frame_type.value} slot={dmr_frame.slot.value} dst={dmr_frame.dst_id}")
                except Exception as e:
                    logger.error(f"Channel {ch.cfg.id}: DMR decoding error: {e}")

                # DMR AMBE+2 voice decoding via DSD-FME discriminator approach
                # Compute FM discriminator output (instantaneous frequency) and feed to DSD-FME
                if ch._dmr_voice_decoder is not None and ch._dmr_voice_decoder.running and base.size > 0:
                    try:
                        # Compute FM discriminator (same as DMR 4FSK demodulation)
                        dmr_iq_c64: NDArrayComplex = base.astype(np.complex64, copy=False)
                        prod = dmr_iq_c64[1:] * np.conj(dmr_iq_c64[:-1])
                        discriminator = np.angle(prod) * self.cfg.sample_rate / (2 * np.pi)

                        # Queue discriminator audio for AMBE decoding (non-blocking)
                        ch._dmr_voice_decoder.decode_sync(discriminator)

                        # Get any decoded audio available from previous frames
                        dmr_decoded_audio = ch._dmr_voice_decoder.get_audio_sync()
                        if dmr_decoded_audio is not None and dmr_decoded_audio.size > 0:
                            if not _validate_audio_output(dmr_decoded_audio, f"Channel {ch.cfg.id}"):
                                dmr_decoded_audio = None
                            else:
                                ch._update_audio_metrics(dmr_decoded_audio)
                    except Exception as e:
                        logger.error(f"Channel {ch.cfg.id}: DMR voice decoding error: {e}")

            return dmr_decoded_audio

        # NXDN decoding (stub - not yet implemented)
        if ch.cfg.mode == "nxdn":
            # NXDN uses 4FSK modulation and AMBE+2 codec (similar to DMR)
            # Sample rate: 4800/9600 baud
            # TODO: Implement NXDNDecoder with AMBE+2 integration
            if not getattr(ch, '_nxdn_warned', False):
                logger.warning(f"Channel {ch.cfg.id}: NXDN mode not yet implemented - no audio output")
                ch._nxdn_warned = True
            return None

        # D-Star decoding (stub - not yet implemented)
        if ch.cfg.mode == "dstar":
            # D-Star uses GMSK modulation and AMBE codec
            # Data rate: 4800 bps (voice + data)
            # TODO: Implement DStarDecoder with AMBE integration
            if not getattr(ch, '_dstar_warned', False):
                logger.warning(f"Channel {ch.cfg.id}: D-Star mode not yet implemented - no audio output")
                ch._dstar_warned = True
            return None

        # YSF/Fusion decoding (stub - not yet implemented)
        if ch.cfg.mode == "ysf":
            # YSF (Yaesu System Fusion) uses C4FM modulation
            # Multiple codec modes: DN (digital narrow), VW (voice wide), FR (full rate)
            # Codecs: AMBE+2 (DN), IMBE (VW), or mixed
            # TODO: Implement YSFDecoder with AMBE+2/IMBE integration
            if not getattr(ch, '_ysf_warned', False):
                logger.warning(f"Channel {ch.cfg.id}: YSF (Fusion) mode not yet implemented - no audio output")
                ch._ysf_warned = True
            return None

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

        # FLEX decoding (NBFM only, via multimon-ng)
        if ch.cfg.mode == "nbfm" and ch.cfg.enable_flex and audio.size > 0:
            if ch._flex_decoder is None:
                ch._flex_decoder = FlexDecoder()
                logger.info(f"Channel {ch.cfg.id}: FLEX decoder initialized (multimon-ng)")

            ch._flex_decoder.feed(audio, ch.cfg.audio_rate)
            for flex_msg in ch._flex_decoder.drain_messages():
                ch._flex_messages.append(flex_msg)
                if len(ch._flex_messages) > ch._flex_max_messages:
                    ch._flex_messages.pop(0)

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
                    agc_enabled=self.cfg.agc_enabled,
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
                    agc_enabled=self.cfg.agc_enabled,
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
            self._emit_state_change("started")
            self._startup_time = 0.0  # Clear startup time - we're running now
            self._retry_count = 0  # Reset retry counter on success
            # Reset overflow counters on fresh start
            self._iq_overflow_count = 0
            self._iq_overflow_batch = 0
        except Exception as e:
            # Import SDRplay-specific exceptions for special handling
            from .devices.soapy import SDRplayServiceError, invalidate_sdrplay_caches
            from .devices.soapy import TimeoutError as SoapyTimeoutError

            # Handle SDRplay service errors specially - don't auto-retry, need manual intervention
            if isinstance(e, SDRplayServiceError):
                self.error_message = str(e)
                self.state = "failed"
                self._emit_state_change("updated")
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
                self._emit_state_change("updated")
                print(f"[ERROR] Capture {self.cfg.id} device timeout: {e}", flush=True)
                # Invalidate caches to trigger fresh enumeration
                if "sdrplay" in str(self.cfg.device_id or "").lower():
                    invalidate_sdrplay_caches()
                self.release_device()
                # Schedule restart - recovery may have fixed the issue
                self._schedule_restart()
                return

            # Generic error - schedule automatic restart
            self.error_message = f"Failed to start capture: {e!s}"
            print(f"[ERROR] Capture {self.cfg.id} failed to start: {e}", flush=True)
            # Report device retry event
            from .error_tracker import ErrorEvent, get_error_tracker
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
        _capture_loop_counter = 0
        stream = self._stream
        while not self._stop_event.is_set():
            _capture_loop_counter += 1
            if _capture_loop_counter <= 5 or _capture_loop_counter % 100000 == 0:
                logger.debug(f"Capture {self.cfg.id}: loop iteration {_capture_loop_counter}, calling read()")
            loop_start = time_module.perf_counter()
            try:
                samples, overflow = stream.read(chunk)
                # Track overflow for this cycle (passed to callbacks for state reset)
                self._iq_overflow_current = overflow
                if overflow:
                    self._iq_overflow_count += 1
                    self._iq_overflow_batch += 1
                    self._report_iq_overflow()
            except Exception as e:
                logger.error(f"Capture {self.cfg.id}: stream.read() exception: {e}")
                if not self._stop_event.is_set():
                    self.state = "failed"
                    self.error_message = f"Stream read failed: {e}"
                    self._emit_state_change("updated")
                break
            if samples.size == 0:
                # Light backoff to avoid busy-spin
                with contextlib.suppress(Exception):
                    threading.Event().wait(0.001)
                continue
            # Update IQ watchdog - we received samples
            self._last_iq_time = time.time()

            # Debug logging every 100 reads with data
            if not hasattr(self, '_iq_debug_counter'):
                self._iq_debug_counter = 0
            self._iq_debug_counter += 1
            # Verbose debug for first few iterations to trace where loop stops
            _verbose_debug = _capture_loop_counter <= 5 or self._iq_debug_counter % 10000 == 1
            if _verbose_debug:
                logger.debug(f"Capture {self.cfg.id}: iter={_capture_loop_counter} GOT DATA: {samples.size} IQ samples, chunk={chunk}")
            # Broadcast IQ to subscribers (schedule on their loops)
            # Reuse asyncio to schedule coroutine execution in a thread-safe manner
            # by using the same logic inside _broadcast_iq (which uses call_soon_threadsafe)
            # Invoke it in a synchronous context using asyncio.run in a dedicated loop is heavy.
            # Instead, inline the same logic here to avoid requiring a loop in this thread.
            #
            # Duplicate minimal logic of _broadcast_iq without awaiting.
            with self._iq_sinks_lock:
                iq_sinks = list(self._iq_sinks)
            if iq_sinks:
                payload = pack_iq16(samples)
                for (q, loop) in iq_sinks:
                    # Use default args to capture loop variables (avoids closure issues)
                    def _try_put(q: asyncio.Queue[bytes] = q, payload: bytes = payload) -> None:
                        try:
                            q.put_nowait(payload)
                        except asyncio.QueueFull:
                            with contextlib.suppress(asyncio.QueueEmpty):
                                _ = q.get_nowait()
                            with contextlib.suppress(asyncio.QueueFull):
                                q.put_nowait(payload)
                    try:
                        loop.call_soon_threadsafe(_try_put)
                    except Exception:
                        with self._iq_sinks_lock, contextlib.suppress(Exception):
                            self._iq_sinks.discard((q, loop))
            # Checkpoint A: After IQ broadcast
            if _verbose_debug:
                logger.debug(f"Capture {self.cfg.id}: iter={_capture_loop_counter} checkpoint A - after IQ broadcast")
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

            # Checkpoint B: After channel metrics
            if _verbose_debug:
                logger.debug(f"Capture {self.cfg.id}: iter={_capture_loop_counter} checkpoint B - after channel metrics ({len(chans)} channels)")
            # Calculate FFT for spectrum display and channel classifier
            # Always runs (for classifier), but rate adapts based on viewer count
            fft_time_ms = 0.0
            with self._fft_sinks_lock:
                fft_sinks = list(self._fft_sinks)

            # Debug: log FFT processing periodically
            if self._fft_counter % 100 == 0 and fft_sinks:
                logger.debug(f"FFT processing for {self.cfg.id}: {len(self._fft_sinks)} subscribers, counter={self._fft_counter}")

            # Adaptive FFT FPS based on subscriber count
            # - No viewers: 1 FPS (minimal, just for classifier)
            # - 1 viewer: configured FPS (default 15)
            # - 2+ viewers: boost FPS for better responsiveness (up to 2x target)
            # Final FPS is capped at fft_max_fps (hard limit)
            base_fps = self.cfg.fft_fps or 15
            max_fps = self.cfg.fft_max_fps or 60
            subscriber_count = len(fft_sinks)
            if subscriber_count >= 2:
                target_fps = min(max_fps, base_fps * 2)
            elif subscriber_count == 1:
                target_fps = min(max_fps, base_fps)
            else:
                target_fps = 1  # Minimal FPS for classifier when no viewers

            # Calculate FFT rate using ACTUAL received samples (not requested chunk)
            # The SDR may return fewer samples than requested
            actual_samples = samples.size
            current_fft_rate = self.cfg.sample_rate / max(1, actual_samples)

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

                # Broadcast FFT to subscribers (only if there are any)
                if fft_sinks and self._fft_power_list is not None and self._fft_freqs_list is not None:
                    payload_fft = {
                        "power": self._fft_power_list,
                        "freqs": self._fft_freqs_list,
                        "centerHz": self.cfg.center_hz,
                        "sampleRate": self.cfg.sample_rate,
                        "fftSize": fft_size,
                        "actualFps": round(self._fft_actual_fps, 1),
                    }
                    for (fft_q, fft_loop) in fft_sinks:
                        # Use default args to capture loop variables (avoids closure issues)
                        def _try_put_fft(q: asyncio.Queue[dict[str, Any]] = fft_q, payload: dict[str, Any] = payload_fft) -> None:
                            try:
                                q.put_nowait(payload)
                            except asyncio.QueueFull:
                                with contextlib.suppress(asyncio.QueueEmpty):
                                    _ = q.get_nowait()
                                with contextlib.suppress(asyncio.QueueFull):
                                    q.put_nowait(payload)
                        try:
                            fft_loop.call_soon_threadsafe(_try_put_fft)
                        except Exception:
                            with self._fft_sinks_lock:
                                with contextlib.suppress(Exception):
                                    self._fft_sinks.discard((fft_q, fft_loop))

            # Checkpoint C: After FFT processing
            if _verbose_debug:
                logger.debug(f"Capture {self.cfg.id}: iter={_capture_loop_counter} checkpoint C - after FFT ({len(self._fft_sinks)} sinks)")
            # Dispatch to channels for audio processing
            # PARALLEL DSP: Use per-capture ThreadPoolExecutor for CPU isolation
            # Each capture gets its own DSP threads, preventing cross-capture starvation
            # NumPy/SciPy release the GIL during heavy computation, enabling true parallelism
            dsp_time_ms = 0.0
            if chans:
                dsp_executor = self._get_dsp_executor()
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

            # Checkpoint D: After DSP/channel processing
            if _verbose_debug:
                logger.debug(f"Capture {self.cfg.id}: iter={_capture_loop_counter} checkpoint D - after DSP, dsp_time={dsp_time_ms:.1f}ms")

            # Voice channel cleanup: Run every ~1 second (20 iterations at 20 Hz)
            # This removes voice channels that haven't received grants recently
            if _capture_loop_counter % 20 == 0:
                for ch in chans:
                    if ch.cfg.mode == "p25" and ch._voice_channel_factory is not None:
                        stale = ch.cleanup_stale_voice_channels(
                            ch._voice_channel_delete_callback
                        )
                        if stale:
                            logger.debug(
                                f"Capture {self.cfg.id}: Cleaned up {len(stale)} "
                                f"stale voice channels"
                            )

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
            # Checkpoint E: End of loop iteration
            if _verbose_debug:
                logger.debug(f"Capture {self.cfg.id}: iter={_capture_loop_counter} checkpoint E - loop iteration complete, loop_time={loop_time_ms:.1f}ms")

        # Ensure stream is closed when the capture thread exits
        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.close()
            self._stream = None


class CaptureManager:
    def __init__(self, cfg: AppConfig, driver: DeviceDriver):
        self._cfg = cfg
        self._driver = driver
        self._captures: dict[str, Capture] = {}
        self._channels: dict[str, Channel] = {}
        self._next_cap_id = 1
        self._next_chan_id = 1

    def list_devices(self) -> list[dict[str, Any]]:
        return [d.__dict__ for d in self._driver.enumerate()]

    def list_captures(self) -> list[Capture]:
        return list(self._captures.values())

    def get_capture(self, cid: str) -> Capture | None:
        return self._captures.get(cid)

    async def stop_capture(self, cid: str) -> None:
        """Stop a capture by id without removing it from the manager."""
        cap = self._captures.get(cid)
        if cap is not None:
            await cap.stop()

    def create_capture(
        self,
        device_id: str | None,
        center_hz: float,
        sample_rate: int,
        gain: float | None = None,
        bandwidth: float | None = None,
        ppm: float | None = None,
        antenna: str | None = None,
        device_settings: dict[str, Any] | None = None,
        element_gains: dict[str, float] | None = None,
        agc_enabled: bool = False,
        stream_format: str | None = None,
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
            agc_enabled=agc_enabled,
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

    def list_channels(self, cid: str | None = None) -> list[Channel]:
        if cid is None:
            return list(self._channels.values())
        return [ch for ch in self._channels.values() if ch.cfg.capture_id == cid]

    def get_channel(self, chan_id: str) -> Channel | None:
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

        elif mode == "sam":
            # SAM (Synchronous AM) defaults - same filters as AM
            cfg.enable_am_highpass = True  # Remove DC offset
            cfg.am_highpass_hz = 100
            cfg.enable_am_lowpass = True  # Broadcast bandwidth
            cfg.am_lowpass_hz = 5_000
            cfg.sam_sideband = "dsb"  # Double sideband (like normal AM)
            cfg.sam_pll_bandwidth_hz = 50.0  # 50 Hz PLL bandwidth
            cfg.enable_agc = True  # SAM needs AGC
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

        # Digital voice modes - handled by stateful decoders
        # These modes use C4FM/GMSK/4FSK modulation and voice codecs
        elif mode in ("p25", "dmr", "nxdn", "dstar", "ysf"):
            # Digital voice modes use their own audio path via codec decoders
            # No analog filtering needed - decoded audio bypasses DSP chain
            cfg.enable_agc = False  # Voice codecs have their own gain control
            cfg.enable_fm_highpass = False
            cfg.enable_fm_lowpass = False

    def create_channel(
        self,
        cid: str,
        mode: str,
        offset_hz: float = 0.0,
        audio_rate: int | None = None,
        squelch_db: float | None = None,
        enable_voice_following: bool = True,
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

        # For P25 control channels, set up voice channel following
        if mode == "p25" and enable_voice_following:
            ch._voice_channel_factory = self._create_voice_channel_factory(
                cid, cap, chan_id
            )
            ch._voice_channel_delete_callback = self.delete_channel

        cap.create_channel(ch)
        self._channels[chan_id] = ch
        return ch

    def _create_voice_channel_factory(
        self, capture_id: str, capture: Capture, control_channel_id: str
    ) -> Callable[[int, float, int | None], str | None]:
        """Create a factory function for voice channels.

        The factory creates P25 voice channels on the same capture,
        tuned to the granted frequency.

        Args:
            capture_id: ID of the capture to create channels on
            capture: Capture instance for frequency calculations
            control_channel_id: ID of the control channel creating voice channels

        Returns:
            Factory function: (tgid, freq_hz, source_id) -> channel_id
        """

        def factory(
            tgid: int, freq_hz: float, source_id: int | None
        ) -> str | None:
            # Calculate offset from capture center frequency
            offset_hz = freq_hz - capture.cfg.center_hz

            # Check if frequency is within capture bandwidth
            max_offset = capture.cfg.sample_rate / 2
            if abs(offset_hz) > max_offset:
                logger.warning(
                    f"Voice channel at {freq_hz/1e6:.4f} MHz is outside capture "
                    f"bandwidth (center={capture.cfg.center_hz/1e6:.4f} MHz, "
                    f"BW={capture.cfg.sample_rate/1e6:.2f} MHz)"
                )
                return None

            # Create voice channel with auto-generated name
            name = f"TGID {tgid}"
            if source_id:
                name += f" (SRC {source_id})"

            try:
                voice_ch = self.create_channel(
                    cid=capture_id,
                    mode="p25",
                    offset_hz=offset_hz,
                    enable_voice_following=False,  # Don't nest voice following
                )
                voice_ch.cfg.name = name
                voice_ch.cfg.auto_name = f"P25 Voice - {name}"
                # Track parent control channel for cleanup on deletion
                voice_ch._parent_control_channel_id = control_channel_id
                voice_ch.start()
                return voice_ch.cfg.id
            except Exception as e:
                logger.error(f"Failed to create voice channel: {e}")
                return None

        return factory

    def delete_channel(self, chan_id: str) -> None:
        ch = self._channels.pop(chan_id, None)
        if ch is None:
            return
        ch.stop()

        # If this is a voice channel, notify the parent control channel
        if ch._parent_control_channel_id:
            parent_ch = self._channels.get(ch._parent_control_channel_id)
            if parent_ch is not None:
                parent_ch.remove_voice_channel(chan_id)

        cap = self.get_capture(ch.cfg.capture_id)
        if cap is not None:
            cap.remove_channel(chan_id)
