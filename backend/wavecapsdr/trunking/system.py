"""TrunkingSystem - P25 trunked radio system controller.

This module implements the core trunking system that:
- Owns an SDR device/Capture for wideband sampling
- Monitors control channel for voice grants (TSBK)
- Manages a pool of voice recorders that follow calls
- Routes audio to subscribers based on talkgroup subscriptions

Architecture:
    TrunkingSystem
        ├── ControlChannelMonitor (decodes P25 TSBK)
        │   └── Callbacks for voice grants, channel IDs, system info
        └── VoiceRecorderPool
            └── VoiceRecorder[] (follows voice grants, records/streams audio)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Coroutine
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from scipy import signal as scipy_signal

from wavecapsdr.decoders.lrrp import LocationCache
from wavecapsdr.decoders.p25_tsbk import ChannelIdentifier, TSBKParser
from wavecapsdr.decoders.voice import VocoderType
from wavecapsdr.trunking.cc_scanner import ControlChannelScanner
from wavecapsdr.trunking.config import TrunkingProtocol, TrunkingSystemConfig
from wavecapsdr.trunking.control_channel import ControlChannelMonitor, create_control_monitor
from wavecapsdr.trunking.voice_channel import RadioLocation, VoiceChannel, VoiceChannelConfig

if TYPE_CHECKING:
    from wavecapsdr.capture import Capture, CaptureManager, Channel

# Import freq_shift from capture module
import contextlib

from wavecapsdr.capture import freq_shift

logger = logging.getLogger(__name__)


class TrunkingSystemState(str, Enum):
    """Trunking system lifecycle states."""
    STOPPED = "stopped"
    STARTING = "starting"
    SEARCHING = "searching"  # Looking for control channel
    SYNCED = "synced"        # Locked to control channel
    RUNNING = "running"      # Full operation, voice channels active
    STOPPING = "stopping"
    FAILED = "failed"


class ControlChannelState(str, Enum):
    """Control channel lock state."""
    UNLOCKED = "unlocked"    # Not receiving valid TSBK
    SEARCHING = "searching"  # Trying to find control channel
    LOCKED = "locked"        # Receiving valid TSBK
    LOST = "lost"            # Was locked, lost signal


class CallState(str, Enum):
    """Voice call state machine."""
    IDLE = "idle"
    TUNING = "tuning"        # Voice recorder tuning to frequency
    RECORDING = "recording"  # Active voice
    HOLD = "hold"            # Voice ended, holding for continuation
    ENDED = "ended"          # Call complete


@dataclass
class ActiveCall:
    """Represents an active voice call on a trunking system."""
    id: str
    talkgroup_id: int
    talkgroup_name: str
    source_id: int | None
    frequency_hz: float
    channel_id: int
    state: CallState
    start_time: float
    last_activity_time: float
    encrypted: bool = False
    recorder_id: str | None = None

    # Audio stats
    audio_frames: int = 0
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API serialization."""
        return {
            "id": self.id,
            "talkgroupId": self.talkgroup_id,
            "talkgroupName": self.talkgroup_name,
            "sourceId": self.source_id,
            "frequencyHz": self.frequency_hz,
            "channelId": self.channel_id,
            "state": self.state.value,
            "startTime": self.start_time,
            "lastActivityTime": self.last_activity_time,
            "encrypted": self.encrypted,
            "audioFrames": self.audio_frames,
            "durationSeconds": self.duration_seconds,
        }


@dataclass
class VoiceRecorder:
    """A voice channel recorder that follows trunking calls.

    Each recorder can handle one active call at a time. When a voice
    grant is received, an available recorder tunes to the voice frequency
    and records/streams the audio via VoiceChannel.
    """
    id: str
    system_id: str
    state: str = "idle"  # idle, tuning, recording, hold

    # Current assignment
    call_id: str | None = None
    frequency_hz: float = 0.0
    talkgroup_id: int = 0
    talkgroup_name: str = ""
    source_id: int | None = None
    encrypted: bool = False

    # Frequency shift from capture center
    offset_hz: float = 0.0

    # Timing
    last_activity: float = 0.0
    hold_timeout: float = 2.0  # Seconds to hold after voice ends

    # Voice channel for audio streaming
    _voice_channel: VoiceChannel | None = field(default=None, repr=False)

    # Protocol for vocoder selection
    _protocol: TrunkingProtocol = TrunkingProtocol.P25_PHASE1

    # Decimation filter state for IQ processing
    _decim_filter_taps: np.ndarray | None = field(default=None, repr=False)
    _decim_filter_zi: np.ndarray | None = field(default=None, repr=False)
    _decim_factor: int = 1

    # FM demodulator state
    _last_phase: float = 0.0

    # Event loop for thread-safe audio scheduling
    _event_loop: asyncio.AbstractEventLoop | None = field(default=None, repr=False)

    def set_event_loop(self, loop: asyncio.AbstractEventLoop | None) -> None:
        """Set the event loop used for scheduling audio decoding."""
        self._event_loop = loop

    def assign(
        self,
        call_id: str,
        frequency_hz: float,
        talkgroup_id: int,
        talkgroup_name: str,
        center_hz: float,
        protocol: TrunkingProtocol = TrunkingProtocol.P25_PHASE1,
        source_id: int | None = None,
        encrypted: bool = False,
    ) -> None:
        """Assign recorder to a call (sync part - sets up state)."""
        self.state = "tuning"
        self.call_id = call_id
        self.frequency_hz = frequency_hz
        self.talkgroup_id = talkgroup_id
        self.talkgroup_name = talkgroup_name
        self.source_id = source_id
        self.encrypted = encrypted
        self.offset_hz = frequency_hz - center_hz
        self.last_activity = time.time()
        self._protocol = protocol
        self._last_phase = 0.0

        logger.info(
            f"VoiceRecorder {self.id}: Assigned to TG {talkgroup_id} ({talkgroup_name}) "
            f"at {frequency_hz/1e6:.4f} MHz (offset {self.offset_hz/1e3:.1f} kHz)"
        )

    async def start_voice_channel(self) -> None:
        """Start the voice channel for audio streaming (async part)."""
        if self._voice_channel is not None:
            await self._voice_channel.stop()

        # Create voice channel config
        cfg = VoiceChannelConfig(
            id=f"{self.id}_{self.call_id}",
            system_id=self.system_id,
            call_id=self.call_id or "",
            recorder_id=self.id,
            output_rate=8000,  # P25 voice is 8kHz
        )

        # Create and configure voice channel
        self._voice_channel = VoiceChannel(
            cfg=cfg,
            talkgroup_id=self.talkgroup_id,
            talkgroup_name=self.talkgroup_name,
            source_id=self.source_id,
            encrypted=self.encrypted,
        )

        # Select vocoder based on protocol
        vocoder_type = VocoderType.IMBE
        if self._protocol == TrunkingProtocol.P25_PHASE2:
            vocoder_type = VocoderType.AMBE2

        # Start the voice channel
        await self._voice_channel.start(vocoder_type=vocoder_type)
        self.state = "recording"

        logger.info(f"VoiceRecorder {self.id}: Voice channel started")

    def setup_decimation_filter(self, sample_rate: int, target_rate: int = 48000) -> None:
        """Set up decimation filter for IQ processing."""
        self._decim_factor = max(1, sample_rate // target_rate)

        if self._decim_factor > 1:
            # Design lowpass filter: cutoff at 0.8 * (target_rate/2) / (sample_rate/2)
            normalized_cutoff = 0.8 * target_rate / sample_rate
            num_taps = 65
            self._decim_filter_taps = scipy_signal.firwin(num_taps, normalized_cutoff)
            zi = scipy_signal.lfilter_zi(self._decim_filter_taps, 1.0)
            self._decim_filter_zi = zi.astype(np.complex128)
        else:
            self._decim_filter_taps = None
            self._decim_filter_zi = None

    def process_iq(self, iq: np.ndarray, sample_rate: int) -> None:
        """Process IQ samples for this voice channel.

        Performs frequency shift, decimation, FM demodulation,
        and feeds discriminator audio to vocoder.
        """
        if self.state != "recording" or self._voice_channel is None:
            return

        # Frequency shift to center on voice channel
        centered_iq = freq_shift(iq, self.offset_hz, sample_rate)

        # Apply anti-aliasing filter and decimate
        if self._decim_factor > 1 and self._decim_filter_taps is not None:
            if self._decim_filter_zi is not None:
                filtered_iq, self._decim_filter_zi = scipy_signal.lfilter(
                    self._decim_filter_taps, 1.0, centered_iq,
                    zi=self._decim_filter_zi * centered_iq[0]
                )
            else:
                filtered_iq = scipy_signal.lfilter(self._decim_filter_taps, 1.0, centered_iq)
            decimated_iq = filtered_iq[::self._decim_factor]
        else:
            decimated_iq = centered_iq

        # FM discriminator - extract instantaneous frequency
        # Phase = angle(iq), frequency = d(phase)/dt
        phase = np.angle(decimated_iq)
        # Unwrap phase to avoid discontinuities
        phase_unwrapped = np.unwrap(phase)
        # Prepend last phase for continuity
        phase_with_last = np.concatenate([[self._last_phase], phase_unwrapped])
        self._last_phase = phase_unwrapped[-1] if len(phase_unwrapped) > 0 else self._last_phase
        # Differentiate to get frequency
        disc_audio = np.diff(phase_with_last)

        # Update activity time
        self.last_activity = time.time()

        # Feed to voice channel (async, schedule on event loop)
        loop = self._event_loop
        if loop is None or not loop.is_running():
            return

        loop.call_soon_threadsafe(
            lambda: loop.create_task(
                self._voice_channel.process_discriminator_audio(disc_audio.astype(np.float32))
            )
        )

    async def release(self) -> None:
        """Release recorder from current call."""
        if self._voice_channel is not None:
            await self._voice_channel.stop()
            self._voice_channel = None

        if self.call_id:
            logger.info(
                f"VoiceRecorder {self.id}: Released from TG {self.talkgroup_id}"
            )

        self.state = "idle"
        self.call_id = None
        self.frequency_hz = 0.0
        self.talkgroup_id = 0
        self.talkgroup_name = ""
        self.source_id = None
        self.encrypted = False
        self.offset_hz = 0.0
        self._last_phase = 0.0

    def is_available(self) -> bool:
        """Check if recorder is available for new assignment."""
        if self.state == "idle":
            return True
        if self.state == "hold":
            # Can preempt if hold timeout expired
            return time.time() - self.last_activity > self.hold_timeout
        return False

    def get_voice_channel(self) -> VoiceChannel | None:
        """Get the voice channel if active."""
        return self._voice_channel


@dataclass
class TrunkingSystem:
    """P25 trunked radio system controller.

    Manages control channel monitoring and voice channel recording for
    a single P25 system (e.g., PSERN or SA-GRN).
    """
    cfg: TrunkingSystemConfig
    state: TrunkingSystemState = TrunkingSystemState.STOPPED

    # Control channel state
    control_channel_state: ControlChannelState = ControlChannelState.UNLOCKED
    control_channel_freq_hz: float | None = None
    control_channel_index: int = 0  # Index into cfg.control_channels

    # System info from RFSS/NET status broadcasts
    nac: int | None = None  # Network Access Code (12-bit)
    system_id: int | None = None
    rfss_id: int | None = None
    site_id: int | None = None

    # TSBK parser
    _tsbk_parser: TSBKParser | None = None

    # Channel identifiers from IDEN_UP messages
    _channel_identifiers: dict[int, ChannelIdentifier] = field(default_factory=dict)

    # SDR capture and channel references (created on start)
    _capture_manager: CaptureManager | None = None
    _capture: Capture | None = None
    _control_channel: Channel | None = None

    # Control channel monitor (uses correct Phase I/II demodulator)
    _control_monitor: ControlChannelMonitor | None = None

    # Event loop for thread-safe async scheduling
    _event_loop: asyncio.AbstractEventLoop | None = field(default=None, repr=False)

    # Voice recorders
    _voice_recorders: list[VoiceRecorder] = field(default_factory=list)

    # Active calls
    _active_calls: dict[str, ActiveCall] = field(default_factory=dict)
    _calls_by_talkgroup: dict[int, str] = field(default_factory=dict)

    # Stats
    _tsbk_count: int = 0
    _grant_count: int = 0
    _calls_total: int = 0  # Total calls started (cumulative)
    _last_tsbk_time: float = 0.0
    decode_rate: float = 0.0  # TSBK per second

    # Radio location cache (LRRP GPS data)
    _location_cache: LocationCache | None = None
    _location_max_age: float = 300.0  # 5 minutes

    # Event callbacks
    on_call_start: Callable[[ActiveCall], None] | None = None
    on_call_update: Callable[[ActiveCall], None] | None = None
    on_call_end: Callable[[ActiveCall], None] | None = None
    on_system_update: Callable[[TrunkingSystem], None] | None = None

    # Timing
    _state_change_time: float = field(default_factory=time.time)
    _control_channel_timeout: float = 10.0  # Seconds before trying next CC
    _hunt_check_task: asyncio.Task[None] | None = None

    # Control channel scanner for signal strength measurement
    _cc_scanner: ControlChannelScanner | None = None
    _roam_check_interval: float = 30.0  # Check for better channel every 30 seconds (from config)
    _last_roam_check: float = 0.0
    _roam_threshold_db: float = 6.0  # SNR improvement required to roam (from config)
    _scan_iq_buffer: list[np.ndarray] = field(default_factory=list)
    _scan_buffer_samples: int = 0
    _initial_scan_complete: bool = False
    _initial_scan_enabled: bool = True  # Whether to do initial scan (from config)

    def __post_init__(self) -> None:
        """Initialize after dataclass creation."""
        # Use config timeout if specified
        self._control_channel_timeout = self.cfg.control_channel_timeout

        # Use scanner config values
        self._roam_check_interval = self.cfg.roam_check_interval
        self._roam_threshold_db = self.cfg.roam_threshold_db
        self._initial_scan_enabled = self.cfg.initial_scan_enabled

        # Create TSBK parser
        self._tsbk_parser = TSBKParser()

        # Create location cache for radio GPS data
        self._location_cache = LocationCache(max_age_seconds=self._location_max_age)

        # Create control channel scanner
        self._cc_scanner = ControlChannelScanner(
            center_hz=self.cfg.center_hz,
            sample_rate=self.cfg.sample_rate,
            control_channels=self.cfg.control_channels,
        )

        # Create voice recorder pool
        for i in range(self.cfg.max_voice_recorders):
            recorder = VoiceRecorder(
                id=f"{self.cfg.id}_vr{i}",
                system_id=self.cfg.id,
                hold_timeout=self.cfg.voice_hold_time,
            )
            self._voice_recorders.append(recorder)

        logger.info(
            f"TrunkingSystem {self.cfg.id} initialized: "
            f"protocol={self.cfg.protocol.value}, "
            f"control_channels={len(self.cfg.control_channels)}, "
            f"voice_recorders={len(self._voice_recorders)}"
        )

    async def start(self, capture_manager: CaptureManager) -> None:
        """Start the trunking system.

        This creates an SDR capture, sets up the control channel decoder,
        and begins searching for a valid control channel.

        Args:
            capture_manager: CaptureManager for creating SDR captures
        """
        if self.state not in (TrunkingSystemState.STOPPED, TrunkingSystemState.FAILED):
            logger.warning(f"TrunkingSystem {self.cfg.id}: Cannot start from state {self.state}")
            return

        logger.info(f"TrunkingSystem {self.cfg.id}: Starting...")
        self._set_state(TrunkingSystemState.STARTING)
        self._capture_manager = capture_manager
        try:
            self._event_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._event_loop = None
            logger.warning(
                "TrunkingSystem %s: No running event loop; async operations may be skipped",
                self.cfg.id,
            )
        for recorder in self._voice_recorders:
            recorder.set_event_loop(self._event_loop)

        # Validate config
        if not self.cfg.control_channels:
            logger.error(f"TrunkingSystem {self.cfg.id}: No control channels configured")
            self._set_state(TrunkingSystemState.FAILED)
            return

        # Initialize control channel (will be updated after initial scan if enabled)
        self.control_channel_index = 0
        self.control_channel_freq_hz = self.cfg.control_channels[0]
        # If initial scan is disabled, mark as complete so we start immediately
        self._initial_scan_complete = not self._initial_scan_enabled
        if not self._initial_scan_enabled:
            logger.info(f"TrunkingSystem {self.cfg.id}: Initial scan disabled, starting on first channel")

        try:
            # Create wideband capture for this trunking system
            self._capture = capture_manager.create_capture(
                device_id=self.cfg.device_id,
                center_hz=self.cfg.center_hz,
                sample_rate=self.cfg.sample_rate,
                gain=self.cfg.gain,
                antenna=self.cfg.antenna,
                device_settings=self.cfg.device_settings if self.cfg.device_settings else None,
            )

            logger.info(
                f"TrunkingSystem {self.cfg.id}: Created capture {self._capture.cfg.id} "
                f"at {self.cfg.center_hz/1e6:.4f} MHz, {self.cfg.sample_rate/1e6:.1f} Msps"
            )

            # Calculate control channel offset from capture center
            cc_offset_hz = self.control_channel_freq_hz - self.cfg.center_hz

            # Create P25 control channel
            self._control_channel = capture_manager.create_channel(
                cid=self._capture.cfg.id,
                mode="p25",
                offset_hz=cc_offset_hz,
            )

            # Set modulation for the channel's P25 decoder
            if self.cfg.modulation:
                self._control_channel.p25_modulation = self.cfg.modulation

            # Wire up TSBK callback from P25 decoder in the channel
            # This routes TSBKs decoded by the channel's P25 decoder to the trunking system
            self._control_channel.on_tsbk = self._handle_tsbk_callback

            # Start the control channel (sets state to "running" so it processes IQ)
            self._control_channel.start()

            # Also set up ControlChannelMonitor as backup/alternative decoder
            self._setup_tsbk_callback()

            logger.info(
                f"TrunkingSystem {self.cfg.id}: Created P25 control channel {self._control_channel.cfg.id} "
                f"at offset {cc_offset_hz/1e3:.1f} kHz"
            )

            # Start the capture (sync method, runs in background thread)
            self._capture.start()

            logger.info(f"TrunkingSystem {self.cfg.id}: Capture started, searching for control channel")

        except Exception as e:
            logger.error(f"TrunkingSystem {self.cfg.id}: Failed to start: {e}")
            self._set_state(TrunkingSystemState.FAILED)
            await self._cleanup_capture()
            return

        self._set_state(TrunkingSystemState.SEARCHING)
        self.control_channel_state = ControlChannelState.SEARCHING

        logger.info(
            f"TrunkingSystem {self.cfg.id}: Searching for control channel "
            f"at {self.control_channel_freq_hz/1e6:.4f} MHz"
        )

        # Start background hunt check loop
        if self._event_loop is not None:
            self._hunt_check_task = self._event_loop.create_task(self._hunt_check_loop())

    def _setup_tsbk_callback(self) -> None:
        """Set up control channel decoding using ControlChannelMonitor.

        We use ControlChannelMonitor instead of Channel's P25 decoder because:
        - ControlChannelMonitor supports both Phase I (C4FM) and Phase II (CQPSK)
        - The Channel's P25 decoder only supports C4FM (Phase I)

        We intercept IQ samples going to the control channel, do frequency shift,
        and feed them to our ControlChannelMonitor with the correct protocol.
        """
        if self._control_channel is None or self._capture is None:
            return

        # Create ControlChannelMonitor with correct protocol and modulation
        # Sample rate 48000 is standard for P25 decoding
        self._control_monitor = create_control_monitor(
            protocol=self.cfg.protocol,
            sample_rate=48000,
            modulation=self.cfg.modulation,
        )

        mod_str = self.cfg.modulation.value if self.cfg.modulation else "auto"
        logger.info(
            f"TrunkingSystem {self.cfg.id}: Created ControlChannelMonitor "
            f"for {self.cfg.protocol.value} (modulation: {mod_str})"
        )

        # Store reference to self for dynamic offset access in closure
        system = self

        # Design anti-aliasing filter for decimation
        # Target output rate is 48kHz, P25 signal bandwidth is ~12.5kHz
        # Use a lowpass FIR filter with cutoff at 20kHz to preserve P25 signal
        target_rate = 48000
        decim_factor = max(1, self.cfg.sample_rate // target_rate)
        if decim_factor > 1:
            # Design lowpass filter: cutoff at 0.8 * (target_rate/2) / (sample_rate/2)
            # = 0.8 * target_rate / sample_rate
            normalized_cutoff = 0.8 * target_rate / self.cfg.sample_rate
            num_taps = 65  # Good tradeoff between quality and performance
            anti_alias_taps = scipy_signal.firwin(num_taps, normalized_cutoff)
            anti_alias_zi = scipy_signal.lfilter_zi(anti_alias_taps, 1.0)

            logger.info(
                f"TrunkingSystem {self.cfg.id}: Decimation filter: "
                f"{self.cfg.sample_rate} -> {self.cfg.sample_rate // decim_factor} Hz "
                f"(factor {decim_factor})"
            )
        else:
            anti_alias_taps = None
            anti_alias_zi = None

        # Store filter state for streaming
        filter_state = [anti_alias_zi.astype(np.complex128) if anti_alias_zi is not None else None]

        # Debug counter for IQ flow monitoring
        iq_debug_state = {"samples": 0, "calls": 0}

        # Initial scan state - collect samples for 2 seconds then scan
        # Use capture sample rate from config
        capture_sample_rate = self.cfg.sample_rate
        INITIAL_SCAN_SAMPLES = capture_sample_rate * 2  # 2 seconds of samples
        ROAM_SCAN_SAMPLES = capture_sample_rate  # 1 second of samples for roaming check

        def on_raw_iq_callback(iq: np.ndarray, sample_rate: int) -> None:
            """IQ callback for trunking system processing.

            This receives raw wideband IQ samples, handles initial scanning,
            periodic roaming checks, and feeds decimated IQ to ControlChannelMonitor.
            """
            # Debug: Track IQ flow
            iq_debug_state["samples"] += len(iq)
            iq_debug_state["calls"] += 1
            _verbose = iq_debug_state["calls"] <= 20 or iq_debug_state["calls"] % 50 == 0
            if _verbose:
                # Log raw IQ magnitude
                raw_mag = np.abs(iq)
                # Use print for guaranteed output
                print(
                    f"[RAW_IQ] TrunkingSystem call #{iq_debug_state['calls']}: "
                    f"samples={len(iq)}, raw_mean={np.mean(raw_mag):.4f}, raw_max={np.max(raw_mag):.4f}",
                    flush=True
                )

            # ============================================================
            # INITIAL SCAN: Collect samples and scan all control channels
            # ============================================================
            if not system._initial_scan_complete and system._initial_scan_enabled:
                system._scan_iq_buffer.append(iq.copy())
                system._scan_buffer_samples += len(iq)

                if system._scan_buffer_samples >= INITIAL_SCAN_SAMPLES:
                    # Concatenate all collected samples
                    all_iq = np.concatenate(system._scan_iq_buffer)
                    system._scan_iq_buffer.clear()
                    system._scan_buffer_samples = 0

                    # Run the scanner
                    if system._cc_scanner is not None:
                        logger.info(
                            f"TrunkingSystem {self.cfg.id}: Running initial control channel scan..."
                        )
                        system._cc_scanner.scan_all(all_iq)
                        system._cc_scanner.log_scan_results()

                        # Get the best channel
                        best = system._cc_scanner.get_best_channel()
                        if best is not None:
                            best_freq, best_measurement = best

                            # Update scanner's current channel tracking
                            system._cc_scanner._current_channel_hz = best_freq

                            # Update to best channel
                            system.control_channel_freq_hz = best_freq
                            system.control_channel_index = (
                                system.cfg.control_channels.index(best_freq)
                                if best_freq in system.cfg.control_channels
                                else 0
                            )

                            # Update control channel offset
                            if system._control_channel is not None:
                                new_offset = best_freq - system.cfg.center_hz
                                system._control_channel.cfg.offset_hz = new_offset
                                logger.info(
                                    f"TrunkingSystem {self.cfg.id}: Selected best control channel: "
                                    f"{best_freq/1e6:.4f} MHz (SNR={best_measurement.snr_db:.1f} dB, "
                                    f"sync={'YES' if best_measurement.sync_detected else 'NO'})"
                                )
                        else:
                            logger.warning(
                                f"TrunkingSystem {self.cfg.id}: No control channels detected, "
                                f"staying on {system.control_channel_freq_hz/1e6:.4f} MHz"
                            )

                    system._initial_scan_complete = True
                    system._last_roam_check = time.time()

                # During initial scan, just collect samples - don't process yet
                # The channel's P25 decoder will still run, but we won't decode with ControlChannelMonitor
                return

            # ============================================================
            # PERIODIC ROAMING CHECK: Check if a better channel is available
            # ============================================================
            now = time.time()
            if now - system._last_roam_check >= system._roam_check_interval:
                # Collect samples for roaming check
                system._scan_iq_buffer.append(iq.copy())
                system._scan_buffer_samples += len(iq)

                if system._scan_buffer_samples >= ROAM_SCAN_SAMPLES:
                    all_iq = np.concatenate(system._scan_iq_buffer)
                    system._scan_iq_buffer.clear()
                    system._scan_buffer_samples = 0

                    if system._cc_scanner is not None:
                        # Scan all channels
                        system._cc_scanner.scan_all(all_iq)

                        # Check if we should roam
                        current_freq = system.control_channel_freq_hz
                        roam_to = system._cc_scanner.should_roam(
                            current_freq,
                            roam_threshold_db=system._roam_threshold_db,
                        )

                        if roam_to is not None:
                            logger.info(
                                f"TrunkingSystem {self.cfg.id}: Roaming from "
                                f"{current_freq/1e6:.4f} MHz to {roam_to/1e6:.4f} MHz"
                            )
                            # Update scanner's current channel tracking
                            system._cc_scanner._current_channel_hz = roam_to

                            system.control_channel_freq_hz = roam_to
                            system.control_channel_index = (
                                system.cfg.control_channels.index(roam_to)
                                if roam_to in system.cfg.control_channels
                                else 0
                            )

                            # Update control channel offset
                            if system._control_channel is not None:
                                new_offset = roam_to - system.cfg.center_hz
                                system._control_channel.cfg.offset_hz = new_offset

                            # Reset control monitor for new channel
                            if system._control_monitor is not None:
                                system._control_monitor.reset()

                    system._last_roam_check = now

            # Shift frequency to center on control channel (dynamic offset for hunting)
            cc_offset_hz = system._control_channel.cfg.offset_hz if system._control_channel else 0
            centered_iq = freq_shift(iq, cc_offset_hz, sample_rate)
            if _verbose:
                centered_mag = np.abs(centered_iq)
                logger.info(
                    f"TrunkingSystem {self.cfg.id}: after freq_shift offset={cc_offset_hz/1e3:.1f}kHz, "
                    f"centered_mean={np.mean(centered_mag):.4f}"
                )

            # Apply anti-aliasing filter and decimate
            if decim_factor > 1 and anti_alias_taps is not None:
                # Apply lowpass anti-aliasing filter
                if filter_state[0] is not None:
                    filtered_iq, filter_state[0] = scipy_signal.lfilter(
                        anti_alias_taps, 1.0, centered_iq,
                        zi=filter_state[0] * centered_iq[0]
                    )
                else:
                    filtered_iq = scipy_signal.lfilter(anti_alias_taps, 1.0, centered_iq)
                # Decimate
                decimated_iq = filtered_iq[::decim_factor]
            else:
                decimated_iq = centered_iq
            if _verbose:
                decim_mag = np.abs(decimated_iq)
                logger.info(
                    f"TrunkingSystem {self.cfg.id}: after decim factor={decim_factor}, "
                    f"size={len(decimated_iq)}, decim_mean={np.mean(decim_mag):.4f}"
                )

            # Feed to ControlChannelMonitor
            if self._control_monitor is not None:
                try:
                    if _verbose:
                        logger.info(f"TrunkingSystem {self.cfg.id}: calling process_iq on control_monitor")
                    tsbk_results = self._control_monitor.process_iq(decimated_iq)
                    if _verbose:
                        logger.info(f"TrunkingSystem {self.cfg.id}: process_iq returned {len(tsbk_results)} results")

                    # Handle each TSBK result
                    for tsbk_data in tsbk_results:
                        if tsbk_data:
                            self._handle_parsed_tsbk(tsbk_data)

                except Exception as e:
                    import traceback
                    logger.error(f"TrunkingSystem {self.cfg.id}: Control monitor error: {e}\n{traceback.format_exc()}")

            # Check for control channel hunting (no TSBK received for too long)
            self._check_control_channel_hunt()

            # Route IQ to active voice recorders
            for recorder in self._voice_recorders:
                if recorder.state == "recording":
                    try:
                        recorder.process_iq(iq, sample_rate)
                    except Exception as e:
                        logger.error(
                            f"TrunkingSystem {self.cfg.id}: Voice recorder {recorder.id} error: {e}"
                        )

        # Register the IQ callback on the control channel
        self._control_channel.on_raw_iq = on_raw_iq_callback
        logger.info(f"TrunkingSystem {self.cfg.id}: Registered on_raw_iq callback for scanning and decoding")

    def _schedule_coroutine(self, coro: Coroutine[Any, Any, Any], context: str) -> None:
        """Schedule a coroutine on the system event loop."""
        loop = self._event_loop
        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, loop)
            return

        try:
            running_loop = asyncio.get_running_loop()
            running_loop.create_task(coro)
        except RuntimeError:
            logger.warning(
                "TrunkingSystem %s: No event loop to %s",
                self.cfg.id,
                context,
            )

    def _handle_tsbk_callback(self, tsbk_data: dict[str, Any]) -> None:
        """Handle TSBK message from P25 decoder.

        This is called by the P25 decoder when a TSBK is decoded.
        We convert the dict to bytes and pass to process_tsbk().
        """
        # The P25 decoder gives us parsed TSBK data as a dict
        # We need to convert back or handle directly
        # For now, let's handle the parsed data directly
        self._handle_parsed_tsbk(tsbk_data)

    def _handle_parsed_tsbk(self, tsbk_data: dict[str, Any]) -> None:
        """Handle parsed TSBK data from P25 decoder."""
        # Update stats
        self._tsbk_count += 1
        now = time.time()
        if self._last_tsbk_time > 0:
            elapsed = now - self._last_tsbk_time
            if elapsed > 0:
                instant_rate = 1.0 / elapsed
                self.decode_rate = 0.9 * self.decode_rate + 0.1 * instant_rate
        self._last_tsbk_time = now

        # If we were searching, we're now synced
        if self.control_channel_state == ControlChannelState.SEARCHING:
            self.control_channel_state = ControlChannelState.LOCKED
            self._set_state(TrunkingSystemState.SYNCED)
            logger.info(
                f"TrunkingSystem {self.cfg.id}: Locked to control channel "
                f"at {self.control_channel_freq_hz/1e6:.4f} MHz"
            )

        # Handle the TSBK based on opcode name (parser returns numeric in 'opcode', string in 'opcode_name')
        opcode_name = tsbk_data.get("opcode_name", "")

        # Voice grants
        if opcode_name in ("GRP_V_CH_GRANT", "GRP_V_CH_GRANT_UPDT", "GRP_V_CH_GRANT_UPDT_EXP", "UU_V_CH_GRANT"):
            self._handle_voice_grant(tsbk_data)

        # Channel identifiers
        elif opcode_name in ("IDEN_UP", "IDEN_UP_VU", "IDEN_UP_TDMA"):
            self._handle_channel_identifier(tsbk_data)

        # System status
        elif opcode_name == "RFSS_STS_BCAST":
            self._handle_rfss_status(tsbk_data)
        elif opcode_name == "NET_STS_BCAST":
            self._handle_net_status(tsbk_data)

    async def _cleanup_capture(self) -> None:
        """Clean up capture and channel resources."""
        if self._capture_manager and self._capture:
            try:
                await self._capture_manager.delete_capture(self._capture.cfg.id)
            except Exception as e:
                logger.error(f"TrunkingSystem {self.cfg.id}: Error cleaning up capture: {e}")
        self._capture = None
        self._control_channel = None

    async def stop(self) -> None:
        """Stop the trunking system."""
        if self.state == TrunkingSystemState.STOPPED:
            return

        logger.info(f"TrunkingSystem {self.cfg.id}: Stopping...")
        self._set_state(TrunkingSystemState.STOPPING)

        # Cancel hunt check loop
        if self._hunt_check_task is not None:
            self._hunt_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._hunt_check_task
            self._hunt_check_task = None

        # End all active calls
        for call_id in list(self._active_calls.keys()):
            self._end_call(call_id, "system_stop")

        # Release all recorders
        for recorder in self._voice_recorders:
            await recorder.release()

        # Clean up SDR capture and channel
        await self._cleanup_capture()

        self.control_channel_state = ControlChannelState.UNLOCKED
        self._set_state(TrunkingSystemState.STOPPED)

        logger.info(f"TrunkingSystem {self.cfg.id}: Stopped")

    def _set_state(self, new_state: TrunkingSystemState) -> None:
        """Update system state and notify callbacks."""
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            self._state_change_time = time.time()
            logger.info(f"TrunkingSystem {self.cfg.id}: State {old_state.value} -> {new_state.value}")

            if self.on_system_update:
                try:
                    self.on_system_update(self)
                except Exception as e:
                    logger.error(f"Error in on_system_update callback: {e}")

    def process_tsbk(self, tsbk_data: bytes) -> None:
        """Process a TSBK (Trunking Signaling Block).

        Called by the control channel decoder when a valid TSBK is received.

        Args:
            tsbk_data: Raw TSBK data (12 bytes)
        """
        if self._tsbk_parser is None:
            return

        try:
            result = self._tsbk_parser.parse(tsbk_data)
            if result is None:
                return

            # Update stats
            self._tsbk_count += 1
            now = time.time()
            if self._last_tsbk_time > 0:
                elapsed = now - self._last_tsbk_time
                if elapsed > 0:
                    # Exponential moving average of decode rate
                    instant_rate = 1.0 / elapsed
                    self.decode_rate = 0.9 * self.decode_rate + 0.1 * instant_rate
            self._last_tsbk_time = now

            # If we were searching, we're now synced
            if self.control_channel_state == ControlChannelState.SEARCHING:
                self.control_channel_state = ControlChannelState.LOCKED
                self._set_state(TrunkingSystemState.SYNCED)
                logger.info(
                    f"TrunkingSystem {self.cfg.id}: Locked to control channel "
                    f"at {self.control_channel_freq_hz/1e6:.4f} MHz"
                )

            # Handle different TSBK types
            opcode = result.get("opcode")

            # Voice grants
            if opcode in ("GRP_V_CH_GRANT", "GRP_V_CH_GRANT_UPDT", "UU_V_CH_GRANT"):
                self._handle_voice_grant(result)

            # Channel identifiers
            elif opcode in ("IDEN_UP", "IDEN_UP_VU"):
                self._handle_channel_identifier(result)

            # System status
            elif opcode == "RFSS_STS_BCAST":
                self._handle_rfss_status(result)
            elif opcode == "NET_STS_BCAST":
                self._handle_net_status(result)

        except Exception as e:
            logger.error(f"TrunkingSystem {self.cfg.id}: Error processing TSBK: {e}")

    def _handle_voice_grant(self, result: dict[str, Any]) -> None:
        """Handle a voice channel grant TSBK.

        Parser result contains:
        - tgid: Talkgroup ID
        - source_id: Source radio address
        - channel: 16-bit channel ID (4-bit band + 12-bit channel number)
        - frequency_hz: Calculated frequency (if channel ID known)
        - encrypted: bool
        - emergency: bool
        """
        # Extract fields from parser result
        tgid = result.get("tgid")
        if tgid is None:
            logger.warning(f"TrunkingSystem {self.cfg.id}: Voice grant missing tgid")
            return

        self._grant_count += 1

        source_id = result.get("source_id", 0)
        channel_id = result.get("channel", 0)
        encrypted = result.get("encrypted", False)

        # Get talkgroup config
        tg_config = self.cfg.get_talkgroup(tgid)

        # Check if talkgroup is monitored
        if not self.cfg.is_talkgroup_monitored(tgid):
            logger.debug(f"TrunkingSystem {self.cfg.id}: Ignoring grant for unmonitored TG {tgid}")
            return

        # Calculate voice channel frequency from channel ID
        freq_hz = self._calculate_frequency(channel_id)
        if freq_hz is None:
            logger.warning(
                f"TrunkingSystem {self.cfg.id}: Cannot calculate frequency for "
                f"channel ID 0x{channel_id:04X} (no IDEN_UP received for band {(channel_id >> 12) & 0xF})"
            )
            return

        # Check if we already have a call for this talkgroup
        existing_call_id = self._calls_by_talkgroup.get(tgid)
        if existing_call_id and existing_call_id in self._active_calls:
            call = self._active_calls[existing_call_id]
            now = time.time()
            time_since_activity = now - call.last_activity_time

            # SDRTrunk-style call identity: TALKGROUP + SOURCE_RADIO
            # A different source_id means a different talker, which should be a new call
            # UNLESS the source_id is 0 (some systems don't send source on every grant)
            source_changed = (
                source_id != 0 and
                call.source_id != 0 and
                source_id != call.source_id
            )

            # Hard 2-second staleness threshold (per SDRTrunk)
            # If activity is stale, treat as new call even with same source
            STALENESS_THRESHOLD_S = 2.0
            is_stale = time_since_activity > STALENESS_THRESHOLD_S

            if source_changed:
                # Different talker - end old call and start new one
                logger.info(
                    f"TrunkingSystem {self.cfg.id}: Talker change on TG {tgid}: "
                    f"source {call.source_id} -> {source_id}"
                )
                self._end_call(existing_call_id, "talker_change")
                # Fall through to create new call below

            elif is_stale:
                # Same talker but activity is stale - could be new call
                logger.debug(
                    f"TrunkingSystem {self.cfg.id}: Stale activity on TG {tgid} "
                    f"({time_since_activity:.1f}s), treating as new call"
                )
                self._end_call(existing_call_id, "stale")
                # Fall through to create new call below

            else:
                # Same talker, not stale - update existing call
                call.last_activity_time = now
                # Update source_id if we got a non-zero value
                if source_id != 0:
                    call.source_id = source_id

                # Check if frequency changed (call continuing on different channel)
                if abs(call.frequency_hz - freq_hz) > 1000:
                    logger.info(
                        f"TrunkingSystem {self.cfg.id}: TG {tgid} moving from "
                        f"{call.frequency_hz/1e6:.4f} to {freq_hz/1e6:.4f} MHz"
                    )
                    call.frequency_hz = freq_hz
                    call.channel_id = channel_id

                    # Retune recorder if assigned
                    if call.recorder_id:
                        self._retune_recorder(call.recorder_id, freq_hz)

                if call.state == CallState.HOLD:
                    call.state = CallState.RECORDING
                    logger.debug(f"TrunkingSystem {self.cfg.id}: Call {call.id} resumed")

                if self.on_call_update:
                    self.on_call_update(call)
                return

        # Start new call
        tg_name = tg_config.name if tg_config else f"TG {tgid}"
        call = ActiveCall(
            id=str(uuid.uuid4())[:8],
            talkgroup_id=tgid,
            talkgroup_name=tg_name,
            source_id=source_id,
            frequency_hz=freq_hz,
            channel_id=channel_id,
            state=CallState.TUNING,
            start_time=time.time(),
            last_activity_time=time.time(),
            encrypted=encrypted,
        )

        # Try to assign a voice recorder
        recorder = self._get_available_recorder(tgid)
        if recorder:
            recorder.assign(
                call_id=call.id,
                frequency_hz=freq_hz,
                talkgroup_id=tgid,
                talkgroup_name=tg_name,
                center_hz=self.cfg.center_hz,
                protocol=self.cfg.protocol,
                source_id=source_id,
                encrypted=encrypted,
            )
            # Set up decimation filter for IQ processing
            recorder.setup_decimation_filter(self.cfg.sample_rate)

            # Start voice channel (async, schedule on event loop)
            self._schedule_coroutine(recorder.start_voice_channel(), "start voice channel")

            call.recorder_id = recorder.id
            call.state = CallState.RECORDING

            logger.info(
                f"TrunkingSystem {self.cfg.id}: Call started - "
                f"TG {tgid} ({tg_name}) at {freq_hz/1e6:.4f} MHz"
            )
        else:
            logger.warning(
                f"TrunkingSystem {self.cfg.id}: No recorder available for "
                f"TG {tgid} ({tg_name})"
            )

        self._active_calls[call.id] = call
        self._calls_by_talkgroup[tgid] = call.id
        self._calls_total += 1

        if self.on_call_start:
            self.on_call_start(call)

        # Transition to running state if not already
        if self.state == TrunkingSystemState.SYNCED:
            self._set_state(TrunkingSystemState.RUNNING)

    def _handle_channel_identifier(self, result: dict[str, Any]) -> None:
        """Handle a channel identifier TSBK (IDEN_UP, IDEN_UP_VU, IDEN_UP_TDMA).

        Parser result contains:
        - identifier: 4-bit channel band ID
        - base_freq_mhz: Base frequency in MHz
        - channel_spacing_khz: Channel spacing in kHz
        - bandwidth_khz: Channel bandwidth in kHz
        - tx_offset_hz: TX offset in Hz
        """
        ident = result.get("identifier")
        if ident is None:
            logger.warning(f"TrunkingSystem {self.cfg.id}: IDEN_UP missing identifier")
            return

        # Create ChannelIdentifier from parser fields
        chan_id = ChannelIdentifier(
            identifier=ident,
            bw=int(result.get("bandwidth_khz", 12.5)),
            tx_offset=int(result.get("tx_offset_hz", 0) / 1e6),
            channel_spacing=int(result.get("channel_spacing_khz", 12.5)),
            base_freq=result.get("base_freq_mhz", 0.0),
        )

        # Store in our map
        self._channel_identifiers[chan_id.identifier] = chan_id

        logger.info(
            f"TrunkingSystem {self.cfg.id}: Channel ID {chan_id.identifier}: "
            f"base={chan_id.base_freq:.4f} MHz, "
            f"spacing={chan_id.channel_spacing} kHz"
        )

    def _check_control_channel_hunt(self) -> None:
        """Check if we need to hunt for a different control channel.

        If no TSBK has been received for too long, try the next control channel
        in the configured list.
        """
        # Only hunt if we're in searching state
        if self.control_channel_state == ControlChannelState.LOCKED:
            return

        # Check timeout
        now = time.time()
        last_tsbk = self._last_tsbk_time if self._last_tsbk_time > 0 else self._state_change_time
        elapsed = now - last_tsbk

        # Debug: Log hunting status periodically
        if not hasattr(self, '_hunt_log_count'):
            self._hunt_log_count = 0
        self._hunt_log_count += 1
        if self._hunt_log_count % 100 == 1:
            logger.info(
                f"TrunkingSystem {self.cfg.id}: Hunt check - "
                f"elapsed={elapsed:.1f}s, timeout={self._control_channel_timeout}s, "
                f"num_channels={len(self.cfg.control_channels)}"
            )

        if elapsed < self._control_channel_timeout:
            return

        # Time to try the next control channel
        num_channels = len(self.cfg.control_channels)
        if num_channels <= 1:
            return

        # Advance to next control channel
        self.control_channel_index = (self.control_channel_index + 1) % num_channels
        self.control_channel_freq_hz = self.cfg.control_channels[self.control_channel_index]

        logger.info(
            f"TrunkingSystem {self.cfg.id}: Control channel hunt - "
            f"trying {self.control_channel_freq_hz/1e6:.4f} MHz "
            f"(channel {self.control_channel_index + 1}/{num_channels})"
        )

        # Update the control channel offset
        if self._control_channel is not None:
            new_offset = self.control_channel_freq_hz - self.cfg.center_hz
            self._control_channel.cfg.offset_hz = new_offset

        # Reset state tracking
        self._state_change_time = now
        self._last_tsbk_time = 0.0

        # Reset the control channel monitor
        if self._control_monitor is not None:
            self._control_monitor.reset()

    async def _hunt_check_loop(self) -> None:
        """Background loop to periodically check for control channel hunting.

        This runs independently of IQ processing to ensure hunting happens
        even when no data is being received.
        """
        logger.info(f"TrunkingSystem {self.cfg.id}: Starting hunt check loop")
        try:
            while True:
                await asyncio.sleep(1.0)  # Check every second

                # Only hunt when in searching state
                if self.state not in (TrunkingSystemState.SEARCHING, TrunkingSystemState.RUNNING):
                    continue

                # Check for hunting
                self._check_control_channel_hunt()

        except asyncio.CancelledError:
            logger.info(f"TrunkingSystem {self.cfg.id}: Hunt check loop cancelled")
        except Exception as e:
            logger.error(f"TrunkingSystem {self.cfg.id}: Hunt check loop error: {e}")

    def _handle_rfss_status(self, result: dict[str, Any]) -> None:
        """Handle RFSS Status Broadcast TSBK."""
        if "nac" in result:
            self.nac = result["nac"]
        if "system_id" in result:
            self.system_id = result["system_id"]
        if "rfss_id" in result:
            self.rfss_id = result["rfss_id"]
        if "site_id" in result:
            self.site_id = result["site_id"]

        logger.debug(
            f"TrunkingSystem {self.cfg.id}: RFSS status - "
            f"NAC={self.nac}, SysID={self.system_id}, "
            f"RFSS={self.rfss_id}, Site={self.site_id}"
        )

    def _handle_net_status(self, result: dict[str, Any]) -> None:
        """Handle Network Status Broadcast TSBK."""
        if "nac" in result:
            self.nac = result["nac"]
        if "system_id" in result:
            self.system_id = result["system_id"]

        logger.debug(
            f"TrunkingSystem {self.cfg.id}: Network status - "
            f"NAC={self.nac}, SysID={self.system_id}"
        )

    def _calculate_frequency(self, channel_id: int) -> float | None:
        """Calculate frequency from channel ID.

        The channel ID format is: IDEN (4 bits) | CHANNEL (12 bits)

        Returns frequency in Hz.
        """
        iden = (channel_id >> 12) & 0xF
        channel = channel_id & 0xFFF

        chan_info = self._channel_identifiers.get(iden)
        if chan_info is None:
            # No IDEN_UP received yet for this band
            logger.warning(
                f"TrunkingSystem {self.cfg.id}: No channel identifier for IDEN {iden}"
            )
            return None

        # ChannelIdentifier has base_freq in MHz and channel_spacing in kHz
        # Convert to Hz for calculation
        base_freq_hz = chan_info.base_freq * 1e6
        channel_spacing_hz = chan_info.channel_spacing * 1e3
        freq_hz = base_freq_hz + (channel * channel_spacing_hz)
        return freq_hz

    def _get_available_recorder(self, talkgroup_id: int) -> VoiceRecorder | None:
        """Get an available voice recorder for a call.

        Prioritizes:
        1. Idle recorders
        2. Recorders in hold state past timeout
        3. Recorders with lower-priority talkgroups (preemption)
        """
        # First pass: find idle or expired-hold recorder
        for recorder in self._voice_recorders:
            if recorder.is_available():
                return recorder

        # Second pass: preempt lower-priority call
        new_priority = self.cfg.get_talkgroup_priority(talkgroup_id)

        for recorder in self._voice_recorders:
            if recorder.state in ("recording", "hold") and recorder.talkgroup_id:
                existing_priority = self.cfg.get_talkgroup_priority(recorder.talkgroup_id)
                if new_priority < existing_priority:
                    # New call has higher priority (lower number)
                    logger.info(
                        f"TrunkingSystem {self.cfg.id}: Preempting TG {recorder.talkgroup_id} "
                        f"(priority {existing_priority}) for TG {talkgroup_id} "
                        f"(priority {new_priority})"
                    )

                    # End the existing call
                    if recorder.call_id:
                        self._end_call(recorder.call_id, "preempted")

                    # Release is async, schedule it
                    self._schedule_coroutine(recorder.release(), "release preempted recorder")
                    return recorder

        return None

    def _retune_recorder(self, recorder_id: str, new_freq_hz: float) -> None:
        """Retune a voice recorder to a new frequency."""
        for recorder in self._voice_recorders:
            if recorder.id == recorder_id:
                recorder.frequency_hz = new_freq_hz
                recorder.offset_hz = new_freq_hz - self.cfg.center_hz
                recorder.last_activity = time.time()
                logger.debug(
                    f"VoiceRecorder {recorder_id}: Retuned to "
                    f"{new_freq_hz/1e6:.4f} MHz (offset {recorder.offset_hz/1e3:.1f} kHz)"
                )
                break

    def _end_call(self, call_id: str, reason: str = "normal") -> None:
        """End an active call."""
        call = self._active_calls.get(call_id)
        if not call:
            return

        call.state = CallState.ENDED
        call.duration_seconds = time.time() - call.start_time

        logger.info(
            f"TrunkingSystem {self.cfg.id}: Call ended - "
            f"TG {call.talkgroup_id} ({call.talkgroup_name}), "
            f"duration={call.duration_seconds:.1f}s, reason={reason}"
        )

        # Release recorder (async, schedule on event loop)
        if call.recorder_id:
            for recorder in self._voice_recorders:
                if recorder.id == call.recorder_id:
                    self._schedule_coroutine(recorder.release(), "release recorder")
                    break

        # Notify callback
        if self.on_call_end:
            self.on_call_end(call)

        # Clean up
        del self._active_calls[call_id]
        if call.talkgroup_id in self._calls_by_talkgroup:
            if self._calls_by_talkgroup[call.talkgroup_id] == call_id:
                del self._calls_by_talkgroup[call.talkgroup_id]

    def check_call_timeouts(self) -> None:
        """Check for calls that have timed out and end them.

        Called periodically from the main loop.
        """
        now = time.time()
        hold_timeout = self.cfg.voice_hold_time

        for call_id, call in list(self._active_calls.items()):
            # Check for calls in hold state past timeout
            if call.state == CallState.HOLD:
                if now - call.last_activity_time > hold_timeout:
                    self._end_call(call_id, "timeout")

            # Check for calls that haven't had activity for a while
            elif call.state == CallState.RECORDING:
                if now - call.last_activity_time > hold_timeout * 2:
                    # Transition to hold state
                    call.state = CallState.HOLD
                    logger.debug(
                        f"TrunkingSystem {self.cfg.id}: Call {call_id} entering hold state"
                    )

    def get_active_calls(self) -> list[ActiveCall]:
        """Get list of active calls."""
        return list(self._active_calls.values())

    def get_stats(self) -> dict[str, Any]:
        """Get system statistics."""
        stats = {
            "tsbk_count": int(self._tsbk_count),
            "grant_count": int(self._grant_count),
            "calls_total": int(self._calls_total),
            "decode_rate": float(round(self.decode_rate, 2)),
            "active_calls": len(self._active_calls),
            "recorders_idle": sum(1 for r in self._voice_recorders if r.state == "idle"),
            "recorders_active": sum(1 for r in self._voice_recorders if r.state == "recording"),
            "channel_identifiers": len(self._channel_identifiers),
            "initial_scan_complete": bool(self._initial_scan_complete),
        }
        # Add control channel scanner stats if available
        if self._cc_scanner:
            stats["cc_scanner"] = self._cc_scanner.get_stats()
        # Add control monitor stats if available
        if self._control_monitor:
            cm_stats = self._control_monitor.get_stats()
            stats["control_monitor"] = cm_stats
        return stats

    def to_dict(self) -> dict[str, Any]:
        """Convert system state to dictionary for API serialization."""
        return {
            "id": self.cfg.id,
            "name": self.cfg.name,
            "protocol": self.cfg.protocol.value,
            "deviceId": self.cfg.device_id or None,
            "state": self.state.value,
            "controlChannelState": self.control_channel_state.value,
            "controlChannelFreqHz": self.control_channel_freq_hz,
            "nac": self.nac,
            "systemId": self.system_id,
            "rfssId": self.rfss_id,
            "siteId": self.site_id,
            "decodeRate": round(self.decode_rate, 2),
            "activeCalls": len(self._active_calls),
            "stats": self.get_stats(),
        }

    def get_voice_channels(self) -> list[VoiceChannel]:
        """Get all active voice channels."""
        channels = []
        for recorder in self._voice_recorders:
            if recorder.state == "recording" and recorder._voice_channel is not None:
                channels.append(recorder._voice_channel)
        return channels

    def get_voice_channel(self, channel_id: str) -> VoiceChannel | None:
        """Get a specific voice channel by ID."""
        for recorder in self._voice_recorders:
            if recorder._voice_channel and recorder._voice_channel.id == channel_id:
                return recorder._voice_channel
        return None

    def get_voice_recorder(self, recorder_id: str) -> VoiceRecorder | None:
        """Get a voice recorder by ID."""
        for recorder in self._voice_recorders:
            if recorder.id == recorder_id:
                return recorder
        return None

    # =========================================================================
    # Location Cache (LRRP GPS data)
    # =========================================================================

    def update_radio_location(self, location: RadioLocation) -> None:
        """Update the location cache with a new GPS report.

        Called when GPS data is extracted from:
        - Extended Link Control in voice LDU frames
        - LRRP packets in PDU frames

        Args:
            location: RadioLocation object with GPS coordinates
        """
        if self._location_cache is None:
            return

        self._location_cache.update(location)

        # If this radio is currently in a call, attach location to the voice channel
        for recorder in self._voice_recorders:
            if recorder.state == "recording" and recorder.source_id == location.unit_id:
                if recorder._voice_channel:
                    recorder._voice_channel.source_location = location
                    logger.debug(
                        f"Attached location to voice channel: "
                        f"unit={location.unit_id} recorder={recorder.id}"
                    )

    def get_radio_location(self, unit_id: int) -> RadioLocation | None:
        """Get cached location for a radio unit.

        Returns None if no location is cached or location is stale.

        Args:
            unit_id: Radio unit identifier

        Returns:
            RadioLocation if fresh location exists, None otherwise
        """
        if self._location_cache is None:
            return None
        return self._location_cache.get(unit_id)

    def get_all_locations(self) -> list[RadioLocation]:
        """Get all fresh cached locations."""
        if self._location_cache is None:
            return []
        return self._location_cache.get_fresh()

    def get_location_cache_stats(self) -> dict[str, Any]:
        """Get location cache statistics."""
        if self._location_cache is None:
            return {"enabled": False}
        return {
            "enabled": True,
            **self._location_cache.to_dict()
        }
