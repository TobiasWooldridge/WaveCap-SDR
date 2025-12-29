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
from wavecapsdr.decoders.p25 import P25Decoder
from wavecapsdr.decoders.p25_tsbk import ChannelIdentifier, TSBKParser
from wavecapsdr.decoders.voice import VocoderType
from wavecapsdr.trunking.cc_scanner import ControlChannelScanner
from wavecapsdr.trunking.config import HuntMode, TrunkingProtocol, TrunkingSystemConfig
from wavecapsdr.trunking.control_channel import ControlChannelMonitor, create_control_monitor
from wavecapsdr.trunking.voice_channel import RadioLocation, VoiceChannel, VoiceChannelConfig

if TYPE_CHECKING:
    from wavecapsdr.capture import Capture, CaptureManager, Channel

# Import freq_shift from capture module
import contextlib

from wavecapsdr.capture import freq_shift
from wavecapsdr.utils.profiler import get_profiler

logger = logging.getLogger(__name__)

# Profiler for IQ processing performance analysis
_iq_profiler = get_profiler("TrunkingIQ", enabled=True)


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

    # Recording config (set from TrunkingSystemConfig)
    should_record: bool = True
    audio_gain: float = 1.0
    recording_path: str = "./recordings"
    min_call_duration: float = 1.0

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

    # P25 frame decoder for extracting GPS from LDU frames
    _p25_decoder: P25Decoder | None = field(default=None, repr=False)

    # Callback when GPS location is found in LDU1 link control
    on_location: Callable[[dict[str, Any]], None] | None = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop | None) -> None:
        """Set the event loop used for scheduling audio decoding."""
        self._event_loop = loop

    def update_center_frequency(self, new_center_hz: float) -> None:
        """Update offset when capture center frequency changes.

        Called when roaming moves the capture to a new center frequency.
        This keeps the voice channel tuned to the correct absolute frequency.
        """
        if self.frequency_hz != 0.0:
            old_offset = self.offset_hz
            self.offset_hz = self.frequency_hz - new_center_hz
            if abs(old_offset - self.offset_hz) > 100:  # Only log if significant change
                logger.debug(
                    f"VoiceRecorder {self.id}: Updated offset from {old_offset/1e3:.1f} kHz "
                    f"to {self.offset_hz/1e3:.1f} kHz (new center={new_center_hz/1e6:.4f} MHz)"
                )

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
        should_record: bool = True,
        audio_gain: float = 1.0,
        recording_path: str = "./recordings",
        min_call_duration: float = 1.0,
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

        # Recording config
        self.should_record = should_record
        self.audio_gain = audio_gain
        self.recording_path = recording_path
        self.min_call_duration = min_call_duration

        logger.info(
            f"VoiceRecorder {self.id}: Assigned to TG {talkgroup_id} ({talkgroup_name}) "
            f"at {frequency_hz/1e6:.4f} MHz (offset {self.offset_hz/1e3:.1f} kHz)"
            f"{' [RECORDING]' if should_record else ''}"
        )

    async def start_voice_channel(self) -> None:
        """Start the voice channel for audio streaming (async part)."""
        if self._voice_channel is not None:
            await self._voice_channel.stop()

        # Create voice channel config with recording settings
        cfg = VoiceChannelConfig(
            id=f"{self.id}_{self.call_id}",
            system_id=self.system_id,
            call_id=self.call_id or "",
            recorder_id=self.id,
            output_rate=8000,  # P25 voice is 8kHz
            audio_gain=self.audio_gain,
            should_record=self.should_record,
            recording_path=self.recording_path,
            min_call_duration=self.min_call_duration,
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

        # Create P25 decoder for extracting GPS from LDU frames
        # Uses discriminator input mode (48kHz discriminator audio)
        self._p25_decoder = P25Decoder(
            sample_rate=48000,
            use_discriminator_input=True,
        )
        # Wire the on_location callback if set
        if self.on_location:
            self._p25_decoder.on_location = self.on_location

        logger.info(f"VoiceRecorder {self.id}: Voice channel started")

    def setup_decimation_filter(self, sample_rate: int, target_rate: int = 48000) -> None:
        """Set up decimation filter for IQ processing."""
        self._decim_factor = max(1, sample_rate // target_rate)

        if self._decim_factor > 1:
            # Design lowpass filter: cutoff at 0.8 * (target_rate/2) / (sample_rate/2)
            normalized_cutoff = 0.8 * target_rate / sample_rate
            num_taps = 65
            self._decim_filter_taps = scipy_signal.firwin(num_taps, normalized_cutoff)
            # Store lfilter_zi as template - will be scaled by first sample
            self._decim_filter_zi_template = scipy_signal.lfilter_zi(self._decim_filter_taps, 1.0).astype(np.complex128)
            self._decim_filter_zi = None  # Initialized on first sample
        else:
            self._decim_filter_taps = None
            self._decim_filter_zi_template = None
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
            # Initialize filter state with first sample to prevent transient
            if self._decim_filter_zi is None and self._decim_filter_zi_template is not None:
                self._decim_filter_zi = self._decim_filter_zi_template * centered_iq[0]
            if self._decim_filter_zi is not None:
                filtered_iq, self._decim_filter_zi = scipy_signal.lfilter(
                    self._decim_filter_taps, 1.0, centered_iq,
                    zi=self._decim_filter_zi
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

        # Feed discriminator audio to P25 decoder for GPS extraction
        # The decoder uses DiscriminatorDemodulator which expects float discriminator samples
        if self._p25_decoder is not None:
            # process_iq with discriminator input will extract LDU frames and call on_location
            # when GPS data is found in Extended Link Control
            self._p25_decoder.process_iq(disc_audio.astype(np.float32))

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

        # Clean up P25 decoder
        self._p25_decoder = None

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

    # Message log buffer (ring buffer for decoded messages like SDRTrunk)
    _message_log: list[dict[str, Any]] = field(default_factory=list)
    _message_log_max_size: int = 500  # Keep last 500 messages
    on_message: Callable[[dict[str, Any]], None] | None = None

    # Call history buffer (ring buffer for ended calls)
    _call_history: list[dict[str, Any]] = field(default_factory=list)
    _call_history_max_size: int = 100  # Keep last 100 calls

    # Event callbacks
    on_call_start: Callable[[ActiveCall], None] | None = None
    on_call_update: Callable[[ActiveCall], None] | None = None
    on_call_end: Callable[[ActiveCall], None] | None = None
    on_system_update: Callable[[TrunkingSystem], None] | None = None

    # Timing
    _state_change_time: float = field(default_factory=time.time)
    _control_channel_timeout: float = 10.0  # Seconds before trying next CC
    _hunt_check_task: asyncio.Task[None] | None = None
    _has_ever_locked: bool = False  # Track if we've ever achieved lock (for dynamic timeouts)

    # IQ buffer for control channel - accumulates decimated samples before processing
    _cc_iq_buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.complex128))

    # Control channel scanner for signal strength measurement
    _cc_scanner: ControlChannelScanner | None = None
    _roam_check_interval: float = 30.0  # Check for better channel every 30 seconds (from config)
    _last_roam_check: float = 0.0
    _roam_threshold_db: float = 6.0  # SNR improvement required to roam (from config)
    _scan_iq_buffer: list[np.ndarray] = field(default_factory=list)
    _scan_buffer_samples: int = 0
    _initial_scan_complete: bool = False
    _initial_scan_enabled: bool = True  # Whether to do initial scan (from config)

    # Hunt mode control (for fixed stations)
    _hunt_mode: HuntMode = HuntMode.AUTO
    _enabled_channels: set[float] | None = None  # None = all enabled
    _locked_frequency: float | None = None  # For manual mode
    _scan_once_complete: bool = False  # Track if scan_once has completed

    def __post_init__(self) -> None:
        """Initialize after dataclass creation."""
        # Use config timeout if specified
        self._control_channel_timeout = self.cfg.control_channel_timeout

        # Use scanner config values
        self._roam_check_interval = self.cfg.roam_check_interval
        self._roam_threshold_db = self.cfg.roam_threshold_db
        self._initial_scan_enabled = self.cfg.initial_scan_enabled

        # Use hunt mode from config
        self._hunt_mode = self.cfg.default_hunt_mode

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
            # Wire GPS location callback from P25 decoder to location cache
            recorder.on_location = self._handle_location_from_ldu
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

            # Mark capture as owned by this trunking system
            self._capture.trunking_system_id = self.cfg.id

            logger.info(
                f"TrunkingSystem {self.cfg.id}: Created capture {self._capture.cfg.id} "
                f"at {self.cfg.center_hz/1e6:.4f} MHz, {self.cfg.sample_rate/1e6:.1f} Msps"
            )

            # Calculate control channel offset from capture center
            cc_offset_hz = self.control_channel_freq_hz - self.cfg.center_hz

            # Create P25 control channel
            # Disable capture-level voice following since trunking system
            # manages voice channels via its own VoiceRecorder pool
            self._control_channel = capture_manager.create_channel(
                cid=self._capture.cfg.id,
                mode="p25",
                offset_hz=cc_offset_hz,
                enable_voice_following=False,
            )

            # Set modulation for the channel's P25 decoder
            if self.cfg.modulation:
                self._control_channel.p25_modulation = self.cfg.modulation

            # NOTE: We intentionally do NOT wire up the channel's on_tsbk callback.
            # TSBK decoding is handled by ControlChannelMonitor in _setup_tsbk_callback(),
            # which supports both Phase I (C4FM) and Phase II (CQPSK) modulation.
            # Having both decoders active would create duplicate TSBK processing.

            # Start the control channel (sets state to "running" so it processes IQ)
            self._control_channel.start()

            # Set up ControlChannelMonitor as the primary TSBK decoder
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

        # Store reference to self for dynamic offset access in closure
        system = self

        # ================================================================
        # THREE-STAGE DECIMATION to match SDRTrunk's ~25 kHz (~5.2 SPS)
        # ================================================================
        # SDRTrunk's P25P1DecoderC4FM does power-of-2 decimation from polyphase
        # output (~50 kHz) to reach ~25 kHz (5 samples per symbol).
        #
        # From 6 MHz: need 240x total decimation to get 25 kHz
        # Three-stage approach: 30:1 → 4:1 → 2:1 = 240:1 total
        #
        # Stage 1: 6 MHz → 200 kHz (30:1 decimation)
        # Stage 2: 200 kHz → 50 kHz (4:1 decimation)
        # Stage 3: 50 kHz → 25 kHz (2:1 decimation) - matches SDRTrunk
        # ================================================================

        input_rate = self.cfg.sample_rate  # e.g., 6 MHz

        # Stage 1: Coarse decimation to 200 kHz
        stage1_factor = 30
        stage1_rate = input_rate // stage1_factor  # 200 kHz

        # Stage 2: Intermediate decimation to 50 kHz
        stage2_factor = 4
        stage2_rate = stage1_rate // stage2_factor  # 50 kHz

        # Stage 3: Final decimation to 25 kHz (5.2 SPS, matches SDRTrunk)
        stage3_factor = 2
        stage3_rate = stage2_rate // stage3_factor  # 25 kHz

        # Final output rate
        actual_sample_rate = stage3_rate
        total_decim = stage1_factor * stage2_factor * stage3_factor

        # Create ControlChannelMonitor with actual sample rate for correct timing
        self._control_monitor = create_control_monitor(
            protocol=self.cfg.protocol,
            sample_rate=actual_sample_rate,  # Use actual ~19.2 kHz
            modulation=self.cfg.modulation,
        )

        # Wire up sync callbacks for SDRTrunk-compatible lock behavior
        # Lock when sync is detected, not when TSBK is received
        def on_sync_acquired():
            if self.control_channel_state == ControlChannelState.SEARCHING:
                self.control_channel_state = ControlChannelState.LOCKED
                self._has_ever_locked = True  # Track for dynamic hunt timeouts
                self._set_state(TrunkingSystemState.SYNCED)
                logger.info(
                    f"TrunkingSystem {self.cfg.id}: Sync acquired - "
                    f"locked to {self.control_channel_freq_hz/1e6:.4f} MHz"
                )

        def on_sync_lost():
            if self.control_channel_state == ControlChannelState.LOCKED:
                logger.warning(
                    f"TrunkingSystem {self.cfg.id}: Sync lost on "
                    f"{self.control_channel_freq_hz/1e6:.4f} MHz"
                )
                # Don't immediately unlock - let _check_control_channel_hunt handle it
                # This allows for brief sync drops without losing lock

        self._control_monitor.on_sync_acquired = on_sync_acquired
        self._control_monitor.on_sync_lost = on_sync_lost

        mod_str = self.cfg.modulation.value if self.cfg.modulation else "auto"
        logger.info(
            f"TrunkingSystem {self.cfg.id}: Created ControlChannelMonitor "
            f"for {self.cfg.protocol.value} (modulation: {mod_str}, rate={actual_sample_rate}Hz, "
            f"SPS={actual_sample_rate/4800:.1f})"
        )

        # Design Stage 1 anti-aliasing filter
        # Cutoff at 0.8 * (stage1_rate/2) / (input_rate/2) = 0.8 / stage1_factor
        # Use Kaiser window with beta=7.857 for 80 dB stopband attenuation (matches SDRTrunk)
        stage1_normalized_cutoff = 0.8 / stage1_factor
        stage1_taps = scipy_signal.firwin(
            157, stage1_normalized_cutoff, window=("kaiser", 7.857)
        )
        # Store lfilter_zi as template - will be scaled by first sample
        stage1_zi_template = scipy_signal.lfilter_zi(stage1_taps, 1.0).astype(np.complex128)

        # Design Stage 2 anti-aliasing filter
        # Cutoff at 0.8 * (stage2_rate/2) / (stage1_rate/2) = 0.8 / stage2_factor
        # Use Kaiser window with beta=7.857 for 80 dB stopband attenuation (matches SDRTrunk)
        stage2_normalized_cutoff = 0.8 / stage2_factor
        stage2_taps = scipy_signal.firwin(
            73, stage2_normalized_cutoff, window=("kaiser", 7.857)
        )
        # Store lfilter_zi as template - will be scaled by first sample
        stage2_zi_template = scipy_signal.lfilter_zi(stage2_taps, 1.0).astype(np.complex128)

        # Design Stage 3 anti-aliasing filter (50 kHz → 25 kHz)
        # Cutoff at 0.8 * (stage3_rate/2) / (stage2_rate/2) = 0.8 / stage3_factor
        # Use Kaiser window with beta=7.857 for 80 dB stopband attenuation (matches SDRTrunk)
        stage3_normalized_cutoff = 0.8 / stage3_factor
        stage3_taps = scipy_signal.firwin(
            41, stage3_normalized_cutoff, window=("kaiser", 7.857)
        )
        # Store lfilter_zi as template - will be scaled by first sample
        stage3_zi_template = scipy_signal.lfilter_zi(stage3_taps, 1.0).astype(np.complex128)

        logger.info(
            f"TrunkingSystem {self.cfg.id}: Three-stage decimation: "
            f"{input_rate/1e6:.1f} MHz → {stage1_rate/1e3:.1f} kHz ({stage1_factor}:1, {len(stage1_taps)} taps) → "
            f"{stage2_rate/1e3:.1f} kHz ({stage2_factor}:1, {len(stage2_taps)} taps) → "
            f"{stage3_rate/1e3:.1f} kHz ({stage3_factor}:1, {len(stage3_taps)} taps), "
            f"total {total_decim}:1"
        )

        # Store filter states for streaming (mutable dict to update in closure)
        # zi values start as None - will be initialized with first sample to prevent transient
        # lfilter_zi returns step response initial conditions (~1.0), but signal is ~0.004
        # Multiplying by first sample scales zi to match signal level
        filter_state = {
            "stage1_zi": None,  # Initialized on first chunk
            "stage2_zi": None,  # Initialized on first chunk
            "stage3_zi": None,  # Initialized on first chunk
        }

        # Debug counter for IQ flow monitoring
        iq_debug_state = {"samples": 0, "calls": 0}

        # Phase-continuous frequency shift state
        # Track sample index across calls to maintain phase continuity
        # Phase discontinuities at chunk boundaries corrupt the narrowband signal
        freq_shift_state = {"sample_idx": 0, "last_offset_hz": 0.0}

        def phase_continuous_freq_shift(iq: np.ndarray, offset_hz: float, sample_rate: int) -> np.ndarray:
            """Frequency shift with phase continuity across calls.

            Unlike the regular freq_shift which starts at phase=0 each call,
            this maintains the sample index across calls so the phase continues
            smoothly. This is critical for extracting narrowband signals from
            wideband captures.
            """
            if offset_hz == 0.0 or iq.size == 0:
                return iq

            # Reset phase if offset changed (e.g., during hunting)
            if offset_hz != freq_shift_state["last_offset_hz"]:
                freq_shift_state["sample_idx"] = 0
                freq_shift_state["last_offset_hz"] = offset_hz

            # Generate complex exponential starting from current sample index
            n = np.arange(iq.size, dtype=np.float64) + freq_shift_state["sample_idx"]
            phase = -2.0 * np.pi * offset_hz * n / sample_rate
            shift = np.exp(1j * phase).astype(np.complex64)

            # Update sample index for next call
            freq_shift_state["sample_idx"] += iq.size

            # Prevent sample_idx from growing too large (wrap at 1 second worth of samples)
            # Phase is periodic so this doesn't cause discontinuities
            wrap_samples = sample_rate
            if freq_shift_state["sample_idx"] >= wrap_samples:
                freq_shift_state["sample_idx"] %= wrap_samples

            # [DIAG-STAGE2] Frequency shift diagnostics
            if "call_count" not in freq_shift_state:
                freq_shift_state["call_count"] = 0
            freq_shift_state["call_count"] += 1

            if freq_shift_state["call_count"] % 100 == 0:
                # Check phase continuity
                actual_start_phase = float(phase[0]) if len(phase) > 0 else 0.0
                iq_power_before = float(np.mean(np.abs(iq)**2))
                shifted_iq = (iq.astype(np.complex64, copy=False) * shift).astype(np.complex64)
                iq_power_after = float(np.mean(np.abs(shifted_iq)**2))
                logger.info(
                    f"[DIAG-STAGE2] calls={freq_shift_state['call_count']}, "
                    f"offset={offset_hz/1e3:.1f}kHz, sample_idx={freq_shift_state['sample_idx']}, "
                    f"start_phase={actual_start_phase:.4f}rad, "
                    f"power_before={iq_power_before:.6f}, power_after={iq_power_after:.6f}"
                )
                # [DIAG-STAGE2b] Check where signal power is in spectrum BEFORE and AFTER shift
                if len(shifted_iq) >= 4096 and freq_shift_state["call_count"] == 100:
                    freq_resolution = sample_rate / 4096

                    # FFT BEFORE shift
                    fft_before = np.fft.fft(iq[:4096])
                    fft_power_before = np.abs(fft_before)**2
                    peak_bin_before = np.argmax(fft_power_before)
                    peak_freq_before = peak_bin_before * freq_resolution
                    if peak_bin_before > 2048:
                        peak_freq_before = peak_freq_before - sample_rate

                    # FFT AFTER shift
                    fft_after = np.fft.fft(shifted_iq[:4096])
                    fft_power_after = np.abs(fft_after)**2
                    peak_bin_after = np.argmax(fft_power_after)
                    peak_freq_after = peak_bin_after * freq_resolution
                    if peak_bin_after > 2048:
                        peak_freq_after = peak_freq_after - sample_rate

                    # Check power at expected signal location BEFORE shift (-2300 kHz = offset)
                    expected_bin = int(offset_hz / freq_resolution)
                    if expected_bin < 0:
                        expected_bin += 4096
                    expected_power = fft_power_before[expected_bin]
                    noise_floor = np.median(fft_power_before)
                    snr_at_expected = 10 * np.log10(expected_power / noise_floor) if noise_floor > 0 else 0

                    # Check power in baseband (0 Hz ± 50 kHz) AFTER shift
                    baseband_bins = int(50000 / freq_resolution)
                    baseband_power = np.sum(fft_power_after[:baseband_bins]) + np.sum(fft_power_after[-baseband_bins:])
                    total_power = np.sum(fft_power_after)
                    baseband_ratio = baseband_power / total_power if total_power > 0 else 0

                    logger.info(
                        f"[DIAG-STAGE2b] BEFORE shift: peak_bin={peak_bin_before}, peak_freq={peak_freq_before/1e3:.1f}kHz, "
                        f"SNR_at_{offset_hz/1e3:.0f}kHz={snr_at_expected:.1f}dB"
                    )
                    logger.info(
                        f"[DIAG-STAGE2b] AFTER shift: peak_bin={peak_bin_after}, peak_freq={peak_freq_after/1e3:.1f}kHz, "
                        f"baseband_power_ratio={baseband_ratio:.4f} (should be >0.8 if signal centered)"
                    )
                return shifted_iq

            return (iq.astype(np.complex64, copy=False) * shift).astype(np.complex64)

        # IQ buffer for control channel - accumulate decimated samples before processing
        # SDRplay returns small chunks (~8192 samples), after decimation we get only ~26 samples
        # C4FM demodulation needs sufficient samples for reliable sync detection and timing recovery.
        # At 19.2kHz: 10000 samples = ~520ms = ~2083 symbols = ~12 P25 frames
        # (Same time window as previous 25000 at 48kHz, proportionally scaled)
        IQ_BUFFER_MIN_SAMPLES = 10000
        # Buffer is stored as self._cc_iq_buffer (instance variable) so it can be reset from hunting

        # Initial scan state - collect samples for 2 seconds then scan
        # Use capture sample rate from config
        capture_sample_rate = self.cfg.sample_rate
        INITIAL_SCAN_SAMPLES = capture_sample_rate * 2  # 2 seconds of samples
        ROAM_SCAN_SAMPLES = capture_sample_rate  # 1 second of samples for roaming check

        def on_raw_iq_callback(iq: np.ndarray, sample_rate: int, overflow: bool = False) -> None:
            """IQ callback for trunking system processing.

            This receives raw wideband IQ samples, handles initial scanning,
            periodic roaming checks, and feeds decimated IQ to ControlChannelMonitor.

            Args:
                iq: Raw IQ samples from SDR
                sample_rate: Sample rate in Hz
                overflow: True if ring buffer overrun occurred (samples were lost)
            """
            # Handle overflow: reset all stateful processing when samples are lost
            # Without this, the decimation filters, frequency shift phase, and demodulator
            # will produce corrupted output because their state is now invalid
            if overflow:
                logger.warning(
                    f"TrunkingSystem {self.cfg.id}: Ring buffer overflow detected, resetting filter states"
                )
                # Reset filter states to None - will be re-initialized with first sample
                filter_state["stage1_zi"] = None
                filter_state["stage2_zi"] = None
                filter_state["stage3_zi"] = None
                # Reset frequency shift phase (sample index back to 0)
                freq_shift_state["sample_idx"] = 0
                # Clear the IQ buffer (contains corrupted partial data)
                system._cc_iq_buffer = np.array([], dtype=np.complex128)
                # Reset the control channel monitor (demodulator, sync state)
                if system._control_monitor is not None:
                    system._control_monitor.reset()
                # Skip processing this batch - let state stabilize
                return

            # Debug: Track IQ flow (reduced frequency, use logger instead of print)
            iq_debug_state["samples"] += len(iq)
            iq_debug_state["calls"] += 1
            _verbose = iq_debug_state["calls"] <= 5 or iq_debug_state["calls"] % 500 == 0
            if _verbose:
                # Log raw IQ magnitude at DEBUG level
                raw_mag = np.abs(iq)
                logger.debug(
                    f"[RAW_IQ] TrunkingSystem call #{iq_debug_state['calls']}: "
                    f"samples={len(iq)}, raw_mean={np.mean(raw_mag):.4f}, raw_max={np.max(raw_mag):.4f}"
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

                            # Set the control channel offset based on configured center
                            # DON'T recenter the capture - the configured center (e.g. 415.375 MHz)
                            # is chosen to be in the middle of the control channel range so all
                            # channels are within the capture bandwidth. Recentering would cause
                            # some control channels to fall outside the bandwidth during hunting.
                            if system._control_channel is not None:
                                new_offset = best_freq - system.cfg.center_hz
                                system._control_channel.cfg.offset_hz = new_offset
                                logger.info(
                                    f"TrunkingSystem {self.cfg.id}: Set control channel offset to "
                                    f"{new_offset/1e3:.1f} kHz for {best_freq/1e6:.4f} MHz"
                                )
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

                            # Set the control channel offset based on configured center
                            # DON'T recenter the capture - keep it at the configured center
                            if system._control_channel is not None:
                                new_offset = roam_to - system.cfg.center_hz
                                system._control_channel.cfg.offset_hz = new_offset
                                logger.info(
                                    f"TrunkingSystem {self.cfg.id}: Set control channel offset to "
                                    f"{new_offset/1e3:.1f} kHz for {roam_to/1e6:.4f} MHz"
                                )

                            # Reset control monitor and IQ buffer for new channel
                            # Preserve polarity: same system = same polarity
                            if system._control_monitor is not None:
                                system._control_monitor.reset(preserve_polarity=True)
                            system._cc_iq_buffer = np.array([], dtype=np.complex128)

                    system._last_roam_check = now

            # Shift frequency to center on control channel (dynamic offset for hunting)
            # Use phase-continuous freq shift to avoid phase jumps at chunk boundaries
            cc_offset_hz = system._control_channel.cfg.offset_hz if system._control_channel else 0
            with _iq_profiler.measure("freq_shift"):
                centered_iq = phase_continuous_freq_shift(iq, cc_offset_hz, sample_rate)
            _iq_profiler.add_samples(len(iq))
            if _verbose:
                centered_mag = np.abs(centered_iq)
                logger.debug(
                    f"TrunkingSystem {self.cfg.id}: after freq_shift offset={cc_offset_hz/1e3:.1f}kHz, "
                    f"centered_mean={np.mean(centered_mag):.4f}"
                )

            # ================================================================
            # THREE-STAGE DECIMATION: 6 MHz → 200 kHz → 50 kHz → 25 kHz
            # ================================================================
            # Stage 1: Decimate by 30 (6 MHz → 200 kHz)
            if len(centered_iq) > 0:
                with _iq_profiler.measure("decim_stage1"):
                    # Initialize stage1_zi with first sample to prevent transient
                    if filter_state["stage1_zi"] is None:
                        filter_state["stage1_zi"] = stage1_zi_template * centered_iq[0]
                    filtered1, filter_state["stage1_zi"] = scipy_signal.lfilter(
                        stage1_taps, 1.0, centered_iq,
                        zi=filter_state["stage1_zi"]
                    )
                    decimated1 = filtered1[::stage1_factor]

                # Stage 2: Decimate by 4 (200 kHz → 50 kHz)
                with _iq_profiler.measure("decim_stage2"):
                    # Initialize stage2_zi with first sample to prevent transient
                    if filter_state["stage2_zi"] is None:
                        filter_state["stage2_zi"] = stage2_zi_template * decimated1[0]
                    filtered2, filter_state["stage2_zi"] = scipy_signal.lfilter(
                        stage2_taps, 1.0, decimated1,
                        zi=filter_state["stage2_zi"]
                    )
                    decimated2 = filtered2[::stage2_factor]

                # Stage 3: Decimate by 2 (50 kHz → 25 kHz) - matches SDRTrunk
                with _iq_profiler.measure("decim_stage3"):
                    # Initialize stage3_zi with first sample to prevent transient
                    if filter_state["stage3_zi"] is None:
                        filter_state["stage3_zi"] = stage3_zi_template * decimated2[0]
                    filtered3, filter_state["stage3_zi"] = scipy_signal.lfilter(
                        stage3_taps, 1.0, decimated2,
                        zi=filter_state["stage3_zi"]
                    )
                    decimated_iq = filtered3[::stage3_factor]

                # [DIAG-DECIM] Decimation diagnostics
                if iq_debug_state["calls"] % 100 == 0:
                    power_in = float(np.mean(np.abs(centered_iq)**2))
                    power_filt1 = float(np.mean(np.abs(filtered1)**2))  # After filter, before decim
                    power_stage1 = float(np.mean(np.abs(decimated1)**2))
                    power_filt2 = float(np.mean(np.abs(filtered2)**2))  # After filter, before decim
                    power_stage2 = float(np.mean(np.abs(decimated2)**2))
                    power_filt3 = float(np.mean(np.abs(filtered3)**2))  # After filter, before decim
                    power_out = float(np.mean(np.abs(decimated_iq)**2))
                    # Use scientific notation to see actual values - show filter vs decim power loss
                    logger.info(
                        f"[DIAG-DECIM] Stage1: in={power_in:.2e}, filtered={power_filt1:.2e}, decim={power_stage1:.2e}"
                    )
                    logger.info(
                        f"[DIAG-DECIM] Stage2: in={power_stage1:.2e}, filtered={power_filt2:.2e}, decim={power_stage2:.2e}"
                    )
                    logger.info(
                        f"[DIAG-DECIM] Stage3: in={power_stage2:.2e}, filtered={power_filt3:.2e}, decim={power_out:.2e}"
                    )
            else:
                decimated_iq = centered_iq
            if _verbose:
                decim_mag = np.abs(decimated_iq)
                logger.debug(
                    f"TrunkingSystem {self.cfg.id}: after decim factor={total_decim}, "
                    f"size={len(decimated_iq)}, decim_mean={np.mean(decim_mag):.4f}"
                )

            # Buffer decimated IQ samples until we have enough for reliable demodulation
            # Small chunks (65 samples) cause timing recovery to fail
            system._cc_iq_buffer = np.concatenate([system._cc_iq_buffer, decimated_iq])

            # Only process when we have enough samples
            if len(system._cc_iq_buffer) >= IQ_BUFFER_MIN_SAMPLES:
                buffered_iq = system._cc_iq_buffer
                system._cc_iq_buffer = np.array([], dtype=np.complex128)

                if _verbose:
                    logger.debug(
                        f"TrunkingSystem {self.cfg.id}: Processing buffered IQ: "
                        f"{len(buffered_iq)} samples ({len(buffered_iq)/10:.0f} symbols)"
                    )

                # Feed to ControlChannelMonitor
                if self._control_monitor is not None:
                    try:
                        with _iq_profiler.measure("control_monitor"):
                            tsbk_results = self._control_monitor.process_iq(buffered_iq)
                        if _verbose:
                            logger.debug(f"TrunkingSystem {self.cfg.id}: process_iq returned {len(tsbk_results)} results")

                        # Handle each TSBK result and update decode rate at batch level
                        valid_tsbk_count = 0
                        for tsbk_data in tsbk_results:
                            if tsbk_data:
                                self._handle_parsed_tsbk(tsbk_data, update_rate=False)
                                valid_tsbk_count += 1

                        # Update decode rate based on batch timing (not individual TSBKs)
                        if valid_tsbk_count > 0:
                            now = time.time()
                            if self._last_tsbk_time > 0:
                                elapsed = now - self._last_tsbk_time
                                if elapsed > 0.001:  # Minimum 1ms between rate updates
                                    # Rate = TSBKs per second
                                    instant_rate = valid_tsbk_count / elapsed
                                    self.decode_rate = 0.9 * self.decode_rate + 0.1 * instant_rate
                            self._last_tsbk_time = now

                    except Exception as e:
                        import traceback
                        logger.error(f"TrunkingSystem {self.cfg.id}: Control monitor error: {e}\n{traceback.format_exc()}")

            # Check for control channel hunting (no TSBK received for too long)
            self._check_control_channel_hunt()

            # Report profiling statistics periodically
            _iq_profiler.report()

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

    def _handle_parsed_tsbk(self, tsbk_data: dict[str, Any], update_rate: bool = True) -> None:
        """Handle parsed TSBK data from P25 decoder.

        Args:
            tsbk_data: Parsed TSBK dictionary
            update_rate: If True, update decode rate (set False for batch processing)
        """
        # Log the message for UI display
        self._log_message(tsbk_data)

        # Update stats
        self._tsbk_count += 1

        # Only update rate if requested (batch processing handles this externally)
        if update_rate:
            now = time.time()
            if self._last_tsbk_time > 0:
                elapsed = now - self._last_tsbk_time
                if elapsed > 0.001:  # Minimum 1ms between rate updates
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

        # Voice grants - normalize different formats before handling
        if opcode_name == "GRP_V_CH_GRANT":
            # Standard grant: already has tgid, channel, source_id at top level
            self._handle_voice_grant(tsbk_data)

        elif opcode_name == "GRP_V_CH_GRANT_UPDT":
            # Grant update: contains grant1 and optionally grant2 sub-dicts
            self._handle_grant_update(tsbk_data)

        elif opcode_name == "GRP_V_CH_GRANT_UPDT_EXP":
            # Explicit grant update: has tgid at top level, downlink_channel instead of channel
            normalized = dict(tsbk_data)
            normalized["channel"] = tsbk_data.get("downlink_channel", 0)
            self._handle_voice_grant(normalized)

        elif opcode_name == "UU_V_CH_GRANT":
            # Unit-to-unit grant: uses target_id instead of tgid
            normalized = dict(tsbk_data)
            normalized["tgid"] = tsbk_data.get("target_id", 0)
            normalized["is_unit_to_unit"] = True
            self._handle_voice_grant(normalized)

        elif opcode_name == "UU_V_CH_GRANT_UPDT":
            # Unit-to-unit grant update: grant1/grant2 with target_id
            self._handle_uu_grant_update(tsbk_data)

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

            # Voice grants - normalize different formats before handling
            if opcode == "GRP_V_CH_GRANT":
                self._handle_voice_grant(result)

            elif opcode == "GRP_V_CH_GRANT_UPDT":
                self._handle_grant_update(result)

            elif opcode == "GRP_V_CH_GRANT_UPDT_EXP":
                normalized = dict(result)
                normalized["channel"] = result.get("downlink_channel", 0)
                self._handle_voice_grant(normalized)

            elif opcode == "UU_V_CH_GRANT":
                normalized = dict(result)
                normalized["tgid"] = result.get("target_id", 0)
                normalized["is_unit_to_unit"] = True
                self._handle_voice_grant(normalized)

            elif opcode == "UU_V_CH_GRANT_UPDT":
                self._handle_uu_grant_update(result)

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
                # Recording config from system settings
                should_record=self.cfg.is_talkgroup_recorded(tgid),
                audio_gain=self.cfg.audio_gain,
                recording_path=self.cfg.recording_path,
                min_call_duration=self.cfg.min_call_duration,
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

    def _handle_grant_update(self, tsbk_data: dict[str, Any]) -> None:
        """Handle a Group Voice Channel Grant Update TSBK.

        These contain 1-2 grant updates in grant1/grant2 sub-dicts.
        Each grant has tgid, channel, frequency_hz.
        """
        # Process grant1
        grant1 = tsbk_data.get("grant1")
        if grant1:
            normalized = {
                "tgid": grant1.get("tgid"),
                "channel": grant1.get("channel", 0),
                "frequency_hz": grant1.get("frequency_hz"),
                "source_id": 0,  # Grant updates don't include source
                "encrypted": tsbk_data.get("encrypted", False),
                "emergency": tsbk_data.get("emergency", False),
            }
            self._handle_voice_grant(normalized)

        # Process grant2 if present
        grant2 = tsbk_data.get("grant2")
        if grant2:
            normalized = {
                "tgid": grant2.get("tgid"),
                "channel": grant2.get("channel", 0),
                "frequency_hz": grant2.get("frequency_hz"),
                "source_id": 0,
                "encrypted": tsbk_data.get("encrypted", False),
                "emergency": tsbk_data.get("emergency", False),
            }
            self._handle_voice_grant(normalized)

    def _handle_uu_grant_update(self, tsbk_data: dict[str, Any]) -> None:
        """Handle a Unit-to-Unit Voice Channel Grant Update TSBK.

        Similar to group grant updates but uses target_id instead of tgid.
        """
        # Process grant1
        grant1 = tsbk_data.get("grant1")
        if grant1:
            normalized = {
                "tgid": grant1.get("target_id"),  # Map target_id to tgid
                "channel": grant1.get("channel", 0),
                "frequency_hz": grant1.get("frequency_hz"),
                "source_id": 0,
                "is_unit_to_unit": True,
                "encrypted": False,
                "emergency": False,
            }
            self._handle_voice_grant(normalized)

        # Process grant2 if present
        grant2 = tsbk_data.get("grant2")
        if grant2:
            normalized = {
                "tgid": grant2.get("target_id"),
                "channel": grant2.get("channel", 0),
                "frequency_hz": grant2.get("frequency_hz"),
                "source_id": 0,
                "is_unit_to_unit": True,
                "encrypted": False,
                "emergency": False,
            }
            self._handle_voice_grant(normalized)

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

        Respects hunt mode settings:
        - MANUAL: Never hunt, stay on locked frequency
        - SCAN_ONCE: Scan once to find best channel, then lock permanently
        - AUTO: Hunt continuously, respecting enabled/disabled channels

        Uses SDRTrunk-compatible lock criteria:
        - Lock is based on sync detection, not TSBK CRC
        - Stay locked as long as sync is maintained
        - Transition to SEARCHING when sync is lost (1 second without sync)
        """
        now = time.time()

        # MANUAL mode: Never hunt, stay on current/locked channel
        if self._hunt_mode == HuntMode.MANUAL:
            return

        # SCAN_ONCE mode: If scan is complete, don't hunt anymore
        if self._hunt_mode == HuntMode.SCAN_ONCE and self._scan_once_complete:
            return

        # Get sync state from control monitor (SDRTrunk-style lock criteria)
        has_sync = False
        sync_age = float('inf')
        if self._control_monitor is not None:
            has_sync = self._control_monitor.has_sync
            sync_age = self._control_monitor.last_sync_age

        # Also track time on this channel (for hunt timing)
        time_on_channel = now - self._state_change_time

        # Dynamic hunt timeout based on state (WaveCap improvement over SDRTrunk)
        # SDRTrunk doesn't hunt at all - it requires pre-configured frequency
        # WaveCap scans to find the active control channel
        if self.control_channel_state == ControlChannelState.SEARCHING:
            if not self._has_ever_locked:
                hunt_timeout = 3.0  # Fast scan during initial search
            else:
                hunt_timeout = 5.0  # Moderate for re-acquisition after lock loss
        else:
            hunt_timeout = 10.0  # Conservative when locked (shouldn't reach this path normally)

        # Debug: Log hunting status periodically
        if not hasattr(self, '_hunt_log_count'):
            self._hunt_log_count = 0
        self._hunt_log_count += 1
        if self._hunt_log_count % 100 == 1:
            logger.info(
                f"TrunkingSystem {self.cfg.id}: Hunt check - "
                f"state={self.control_channel_state.value}, "
                f"has_sync={has_sync}, sync_age={sync_age:.1f}s, "
                f"time_on_channel={time_on_channel:.1f}s, hunt_timeout={hunt_timeout:.1f}s, "
                f"has_ever_locked={self._has_ever_locked}, "
                f"num_channels={len(self.cfg.control_channels)}, "
                f"hunt_mode={self._hunt_mode.value}"
            )

        # SDRTrunk-style lock: based on sync, not TSBK
        if self.control_channel_state == ControlChannelState.LOCKED:
            if not has_sync:
                # Sync lost - transition to SEARCHING (SDRTrunk: 1 second without sync)
                logger.warning(
                    f"TrunkingSystem {self.cfg.id}: Lost control channel lock "
                    f"(sync lost, age={sync_age:.1f}s), starting hunt"
                )
                self.control_channel_state = ControlChannelState.SEARCHING
            else:
                # Still have sync, stay locked (even if TSBK CRC is failing)
                # For SCAN_ONCE mode, mark scan as complete when we achieve lock
                if self._hunt_mode == HuntMode.SCAN_ONCE and not self._scan_once_complete:
                    self._scan_once_complete = True
                    self._locked_frequency = self.control_channel_freq_hz
                    logger.info(
                        f"TrunkingSystem {self.cfg.id}: SCAN_ONCE complete, "
                        f"locked to {self.control_channel_freq_hz/1e6:.4f} MHz"
                    )
                return

        # When searching, use dynamic timeout before rotating
        if time_on_channel < hunt_timeout:
            return

        # Get list of enabled channels only
        enabled_channels = self.get_enabled_channels()
        num_enabled = len(enabled_channels)
        if num_enabled == 0:
            logger.warning(f"TrunkingSystem {self.cfg.id}: No enabled control channels!")
            return
        if num_enabled == 1:
            # Only one enabled channel, no hunting needed
            return

        # Find next enabled channel
        current_freq = self.control_channel_freq_hz or self.cfg.control_channels[0]

        # Sort enabled channels and find the next one after current
        sorted_channels = sorted(enabled_channels)
        try:
            current_idx = next(
                i for i, f in enumerate(sorted_channels)
                if abs(f - current_freq) < 1000
            )
            next_idx = (current_idx + 1) % len(sorted_channels)
        except StopIteration:
            # Current frequency not in enabled list, use first enabled
            next_idx = 0

        next_freq = sorted_channels[next_idx]

        # Find the index in the original control_channels list
        try:
            self.control_channel_index = self.cfg.control_channels.index(next_freq)
        except ValueError:
            self.control_channel_index = 0

        self.control_channel_freq_hz = next_freq

        # Calculate offset from SDR center to the control channel
        # NOTE: Do NOT retune the SDR - use frequency shifting instead.
        # Retuning causes glitches and race conditions. The SDR stays at the
        # configured center frequency, and we use offset-based frequency shifting
        # to extract each control channel within the capture bandwidth.
        new_offset_hz = self.control_channel_freq_hz - self.cfg.center_hz

        logger.info(
            f"TrunkingSystem {self.cfg.id}: Control channel hunt - "
            f"trying {self.control_channel_freq_hz/1e6:.4f} MHz "
            f"(channel {next_idx + 1}/{num_enabled} enabled, "
            f"offset={new_offset_hz/1e3:.1f} kHz)"
        )

        # Update the control channel offset for frequency shifting
        if self._control_channel is not None:
            self._control_channel.cfg.offset_hz = new_offset_hz

        # Reset state tracking
        self._state_change_time = now
        self._last_tsbk_time = 0.0

        # Reset the control channel monitor and IQ buffer
        # Preserve polarity: same system = same polarity
        if self._control_monitor is not None:
            self._control_monitor.reset(preserve_polarity=True)
        self._cc_iq_buffer = np.array([], dtype=np.complex128)

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

        # Add to call history buffer
        call_record = call.to_dict()
        call_record["endReason"] = reason
        call_record["endTime"] = time.time()
        self._call_history.append(call_record)
        if len(self._call_history) > self._call_history_max_size:
            self._call_history = self._call_history[-self._call_history_max_size:]

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
        # Get channel measurements from scanner if available
        channel_measurements = {}
        if self._cc_scanner:
            scanner_stats = self._cc_scanner.get_stats()
            channel_measurements = scanner_stats.get("measurements", {})

        return {
            "id": self.cfg.id,
            "name": self.cfg.name,
            "protocol": self.cfg.protocol.value,
            "deviceId": self.cfg.device_id or None,
            "state": self.state.value,
            "controlChannelState": self.control_channel_state.value,
            "controlChannelFreqHz": self.control_channel_freq_hz,
            "centerHz": self.cfg.center_hz,  # SDR center frequency (auto-managed)
            "nac": self.nac,
            "systemId": self.system_id,
            "rfssId": self.rfss_id,
            "siteId": self.site_id,
            "decodeRate": round(self.decode_rate, 2),
            "activeCalls": len(self._active_calls),
            "stats": self.get_stats(),
            "channelMeasurements": channel_measurements,  # WaveCap improvement: show all channel strengths
            # Hunt mode control
            "huntMode": self._hunt_mode.value,
            "lockedFrequencyHz": self._locked_frequency,
            "controlChannels": self.get_control_channels_info(),
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
    # Hunt Mode Control (for fixed stations)
    # =========================================================================

    def get_hunt_mode(self) -> HuntMode:
        """Get the current control channel hunting mode."""
        return self._hunt_mode

    def set_hunt_mode(
        self,
        mode: HuntMode,
        locked_freq: float | None = None,
    ) -> None:
        """Set the control channel hunting mode.

        Args:
            mode: The hunting mode to set
            locked_freq: For MANUAL mode, the frequency to lock to (optional)

        Raises:
            ValueError: If locked_freq is not a valid control channel
        """
        # Validate locked_freq if provided
        if locked_freq is not None and locked_freq not in self.cfg.control_channels:
            raise ValueError(
                f"Frequency {locked_freq/1e6:.4f} MHz is not a configured control channel"
            )

        old_mode = self._hunt_mode
        self._hunt_mode = mode
        self._locked_frequency = locked_freq

        # If switching to MANUAL with a specific frequency, tune to it now
        if mode == HuntMode.MANUAL and locked_freq is not None:
            self._tune_to_frequency(locked_freq)
            logger.info(
                f"TrunkingSystem {self.cfg.id}: Hunt mode set to MANUAL, "
                f"locked to {locked_freq/1e6:.4f} MHz"
            )
        elif mode == HuntMode.SCAN_ONCE:
            # Reset scan_once state to trigger a new scan
            self._scan_once_complete = False
            logger.info(
                f"TrunkingSystem {self.cfg.id}: Hunt mode set to SCAN_ONCE, "
                f"will scan and lock to best channel"
            )
        else:
            logger.info(
                f"TrunkingSystem {self.cfg.id}: Hunt mode changed from "
                f"{old_mode.value} to {mode.value}"
            )

        # Notify observers
        if self.on_system_update:
            self.on_system_update(self)

    def get_locked_frequency(self) -> float | None:
        """Get the currently locked frequency (for MANUAL mode)."""
        return self._locked_frequency

    def get_enabled_channels(self) -> list[float]:
        """Get list of enabled control channel frequencies.

        Returns:
            List of enabled frequencies, or all channels if none are disabled.
        """
        if self._enabled_channels is None:
            return list(self.cfg.control_channels)
        return [f for f in self.cfg.control_channels if f in self._enabled_channels]

    def set_channel_enabled(self, freq_hz: float, enabled: bool) -> None:
        """Enable or disable a specific control channel.

        Args:
            freq_hz: The control channel frequency in Hz
            enabled: Whether to enable or disable the channel

        Raises:
            ValueError: If freq_hz is not a configured control channel
        """
        if freq_hz not in self.cfg.control_channels:
            raise ValueError(
                f"Frequency {freq_hz/1e6:.4f} MHz is not a configured control channel"
            )

        # Initialize enabled_channels set if needed
        if self._enabled_channels is None:
            self._enabled_channels = set(self.cfg.control_channels)

        if enabled:
            self._enabled_channels.add(freq_hz)
            logger.info(f"TrunkingSystem {self.cfg.id}: Enabled channel {freq_hz/1e6:.4f} MHz")
        else:
            self._enabled_channels.discard(freq_hz)
            logger.info(f"TrunkingSystem {self.cfg.id}: Disabled channel {freq_hz/1e6:.4f} MHz")

        # Notify observers
        if self.on_system_update:
            self.on_system_update(self)

    def is_channel_enabled(self, freq_hz: float) -> bool:
        """Check if a control channel is enabled."""
        if self._enabled_channels is None:
            return True
        return freq_hz in self._enabled_channels

    def trigger_scan(self) -> dict[float, dict]:
        """Trigger an immediate scan of all control channels.

        Returns:
            Dict mapping frequency to measurement results.
        """
        if self._cc_scanner is None:
            logger.warning(f"TrunkingSystem {self.cfg.id}: No scanner available for trigger_scan")
            return {}

        # Get latest measurements from scanner
        measurements = self._cc_scanner._measurements
        result = {}

        for freq, m in measurements.items():
            result[freq] = {
                "frequencyHz": freq,
                "powerDb": m.power_db,
                "snrDb": m.snr_db,
                "syncDetected": m.sync_detected,
                "measurementTime": m.measurement_time,
            }

        logger.info(f"TrunkingSystem {self.cfg.id}: Returning {len(result)} channel measurements")
        return result

    def get_control_channels_info(self) -> list[dict]:
        """Get detailed info about all control channels.

        Returns:
            List of dicts with channel info including enable state, measurements,
            and whether it's the current channel.
        """
        result = []
        measurements = {}

        if self._cc_scanner:
            measurements = self._cc_scanner._measurements

        for freq in self.cfg.control_channels:
            m = measurements.get(freq)
            is_current = (
                self.control_channel_freq_hz is not None
                and abs(freq - self.control_channel_freq_hz) < 1000
            )

            info = {
                "frequencyHz": freq,
                "enabled": self.is_channel_enabled(freq),
                "isCurrent": is_current,
                "isLocked": self._locked_frequency is not None and abs(freq - self._locked_frequency) < 1000,
                "snrDb": m.snr_db if m else None,
                "powerDb": m.power_db if m else None,
                "syncDetected": m.sync_detected if m else False,
                "measurementTime": m.measurement_time if m else None,
            }
            result.append(info)

        return result

    def _tune_to_frequency(self, freq_hz: float) -> None:
        """Tune to a specific control channel frequency.

        Internal method used by hunt mode control.
        """
        if freq_hz not in self.cfg.control_channels:
            return

        # Find the index of this frequency
        try:
            idx = self.cfg.control_channels.index(freq_hz)
        except ValueError:
            return

        self.control_channel_index = idx
        self.control_channel_freq_hz = freq_hz

        # Calculate new offset
        new_offset_hz = freq_hz - self.cfg.center_hz

        logger.info(
            f"TrunkingSystem {self.cfg.id}: Tuning to {freq_hz/1e6:.4f} MHz "
            f"(offset={new_offset_hz/1e3:.1f} kHz)"
        )

        # Update the control channel offset
        if self._control_channel is not None:
            self._control_channel.cfg.offset_hz = new_offset_hz

        # Reset state tracking
        self._state_change_time = time.time()
        self._last_tsbk_time = 0.0

        # Reset the control channel monitor and IQ buffer
        if self._control_monitor is not None:
            self._control_monitor.reset(preserve_polarity=True)
        self._cc_iq_buffer = np.array([], dtype=np.complex128)

    # =========================================================================
    # Location Cache (LRRP GPS data)
    # =========================================================================

    def _handle_location_from_ldu(self, location_data: dict[str, Any]) -> None:
        """Handle GPS location extracted from LDU1 Extended Link Control.

        This callback is invoked by VoiceRecorder.P25Decoder when GPS data
        is found in the link control portion of LDU1 voice frames.

        Args:
            location_data: Dict with keys: source_id, latitude, longitude,
                          altitude_m, speed_kmh, heading_deg, lcf
        """
        source_id = location_data.get("source_id", 0)
        if source_id == 0:
            return

        # Map LCF to source type
        lcf = location_data.get("lcf", 0x09)
        if lcf == 0x09:
            source_type = "elc_gps"
        elif lcf == 0x0A:
            source_type = "elc_gps_alt"
        elif lcf == 0x0B:
            source_type = "elc_gps_vel"
        else:
            source_type = "elc_unknown"

        location = RadioLocation(
            unit_id=source_id,
            latitude=location_data.get("latitude", 0.0),
            longitude=location_data.get("longitude", 0.0),
            altitude_m=location_data.get("altitude_m"),
            speed_kmh=location_data.get("speed_kmh"),
            heading_deg=location_data.get("heading_deg"),
            source=source_type,
        )

        if location.is_valid():
            logger.info(
                f"GPS from LDU1 ELC: unit={source_id} "
                f"lat={location.latitude:.6f} lon={location.longitude:.6f}"
            )
            self.update_radio_location(location)

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

    # =========================================================================
    # Message Log (decoded TSBK messages for UI display)
    # =========================================================================

    def _log_message(self, tsbk_data: dict[str, Any]) -> None:
        """Add a decoded message to the log buffer.

        Creates a timestamped, formatted message entry for UI display
        similar to SDRTrunk's message window.

        Args:
            tsbk_data: Parsed TSBK data from the decoder
        """
        now = time.time()
        opcode_name = tsbk_data.get("opcode_name", "UNKNOWN")

        # Format message summary based on opcode type
        summary = self._format_message_summary(tsbk_data, opcode_name)

        message = {
            "timestamp": now,
            "opcode": tsbk_data.get("opcode", 0),
            "opcode_name": opcode_name,
            "nac": tsbk_data.get("nac"),
            "summary": summary,
            "raw": tsbk_data,
        }

        # Add to ring buffer
        self._message_log.append(message)
        if len(self._message_log) > self._message_log_max_size:
            self._message_log = self._message_log[-self._message_log_max_size:]

        # Notify callback (for WebSocket broadcast)
        if self.on_message:
            try:
                self.on_message(message)
            except Exception as e:
                logger.error(f"Error in on_message callback: {e}")

    def _format_message_summary(self, tsbk_data: dict[str, Any], opcode_name: str) -> str:
        """Format a human-readable summary of a TSBK message.

        Args:
            tsbk_data: Parsed TSBK data
            opcode_name: TSBK opcode name

        Returns:
            Human-readable summary string
        """
        # Voice grants
        if opcode_name == "GRP_V_CH_GRANT":
            tgid = tsbk_data.get("tgid", 0)
            source = tsbk_data.get("source_id", 0)
            channel = tsbk_data.get("channel", 0)
            freq = tsbk_data.get("frequency_hz")
            freq_str = f" ({freq/1e6:.4f} MHz)" if freq else ""
            return f"Voice Grant TG:{tgid} SRC:{source} CH:0x{channel:04X}{freq_str}"

        elif opcode_name == "GRP_V_CH_GRANT_UPDT":
            parts = []
            for key in ("grant1", "grant2"):
                grant = tsbk_data.get(key)
                if grant:
                    tgid = grant.get("tgid", 0)
                    channel = grant.get("channel", 0)
                    parts.append(f"TG:{tgid} CH:0x{channel:04X}")
            return f"Grant Update {', '.join(parts)}"

        elif opcode_name == "GRP_V_CH_GRANT_UPDT_EXP":
            tgid = tsbk_data.get("tgid", 0)
            channel = tsbk_data.get("downlink_channel", 0)
            return f"Grant Update (Exp) TG:{tgid} CH:0x{channel:04X}"

        elif opcode_name == "UU_V_CH_GRANT":
            target = tsbk_data.get("target_id", 0)
            source = tsbk_data.get("source_id", 0)
            channel = tsbk_data.get("channel", 0)
            return f"Unit-Unit Grant SRC:{source} TGT:{target} CH:0x{channel:04X}"

        # Channel identifiers
        elif opcode_name in ("IDEN_UP", "IDEN_UP_VU", "IDEN_UP_TDMA"):
            ident = tsbk_data.get("identifier", 0)
            base = tsbk_data.get("base_freq_mhz", 0)
            spacing = tsbk_data.get("channel_spacing_khz", 0)
            return f"Channel ID:{ident} Base:{base:.4f} MHz Spacing:{spacing} kHz"

        # System status
        elif opcode_name == "RFSS_STS_BCAST":
            sys_id = tsbk_data.get("system_id", 0)
            rfss = tsbk_data.get("rfss_id", 0)
            site = tsbk_data.get("site_id", 0)
            return f"RFSS Status SYS:{sys_id} RFSS:{rfss} SITE:{site}"

        elif opcode_name == "NET_STS_BCAST":
            sys_id = tsbk_data.get("system_id", 0)
            wacn = tsbk_data.get("wacn", 0)
            return f"Network Status SYS:{sys_id} WACN:{wacn}"

        elif opcode_name == "ADJ_STS_BCAST":
            sys_id = tsbk_data.get("system_id", 0)
            rfss = tsbk_data.get("rfss_id", 0)
            site = tsbk_data.get("site_id", 0)
            return f"Adjacent Site SYS:{sys_id} RFSS:{rfss} SITE:{site}"

        # Affiliation/registration
        elif opcode_name == "GRP_AFF_RSP":
            tgid = tsbk_data.get("tgid", 0)
            source = tsbk_data.get("source_id", 0)
            return f"Group Affiliation TG:{tgid} SRC:{source}"

        elif opcode_name == "U_REG_RSP":
            source = tsbk_data.get("source_id", 0)
            sys_id = tsbk_data.get("system_id", 0)
            return f"Unit Registration SRC:{source} SYS:{sys_id}"

        elif opcode_name == "U_DE_REG_ACK":
            source = tsbk_data.get("source_id", 0)
            return f"Unit Deregistration SRC:{source}"

        # Location
        elif opcode_name == "LOC_REG_RSP":
            source = tsbk_data.get("source_id", 0)
            rfss = tsbk_data.get("rfss_id", 0)
            site = tsbk_data.get("site_id", 0)
            return f"Location Registration SRC:{source} RFSS:{rfss} SITE:{site}"

        # Default
        else:
            # Try to include any IDs we can find
            parts = [opcode_name]
            if "tgid" in tsbk_data:
                parts.append(f"TG:{tsbk_data['tgid']}")
            if "source_id" in tsbk_data:
                parts.append(f"SRC:{tsbk_data['source_id']}")
            return " ".join(parts)

    def get_messages(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """Get recent messages from the log buffer.

        Args:
            limit: Maximum number of messages to return
            offset: Number of messages to skip from the end

        Returns:
            List of message dictionaries, newest first
        """
        # Return messages in reverse order (newest first)
        messages = list(reversed(self._message_log))
        return messages[offset:offset + limit]

    def clear_messages(self) -> int:
        """Clear the message log.

        Returns:
            Number of messages cleared
        """
        count = len(self._message_log)
        self._message_log.clear()
        return count

    def get_call_history(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """Get recent call history from the buffer.

        Args:
            limit: Maximum number of calls to return
            offset: Number of calls to skip from the end

        Returns:
            List of call dictionaries, newest first
        """
        # Return calls in reverse order (newest first)
        calls = list(reversed(self._call_history))
        return calls[offset:offset + limit]

    def clear_call_history(self) -> int:
        """Clear the call history buffer.

        Returns:
            Number of calls cleared
        """
        count = len(self._call_history)
        self._call_history.clear()
        return count
