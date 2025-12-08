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
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from wavecapsdr.trunking.config import TrunkingSystemConfig, TrunkingProtocol
from wavecapsdr.decoders.p25_tsbk import TSBKParser, VoiceGrant, ChannelIdentifier

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
    source_id: Optional[int]
    frequency_hz: float
    channel_id: int
    state: CallState
    start_time: float
    last_activity_time: float
    encrypted: bool = False
    recorder_id: Optional[str] = None

    # Audio stats
    audio_frames: int = 0
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
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
    and records/streams the audio.
    """
    id: str
    system_id: str
    state: str = "idle"  # idle, tuning, recording, hold

    # Current assignment
    call_id: Optional[str] = None
    frequency_hz: float = 0.0
    talkgroup_id: int = 0

    # Frequency shift from capture center
    offset_hz: float = 0.0

    # Timing
    last_activity: float = 0.0
    hold_timeout: float = 2.0  # Seconds to hold after voice ends

    # Audio buffer for this recorder
    audio_buffer: List[np.ndarray] = field(default_factory=list)

    def assign(
        self,
        call_id: str,
        frequency_hz: float,
        talkgroup_id: int,
        center_hz: float,
    ) -> None:
        """Assign recorder to a call."""
        self.state = "tuning"
        self.call_id = call_id
        self.frequency_hz = frequency_hz
        self.talkgroup_id = talkgroup_id
        self.offset_hz = frequency_hz - center_hz
        self.last_activity = time.time()
        self.audio_buffer.clear()
        logger.info(
            f"VoiceRecorder {self.id}: Assigned to TG {talkgroup_id} "
            f"at {frequency_hz/1e6:.4f} MHz (offset {self.offset_hz/1e3:.1f} kHz)"
        )

    def release(self) -> None:
        """Release recorder from current call."""
        if self.call_id:
            logger.info(
                f"VoiceRecorder {self.id}: Released from TG {self.talkgroup_id}"
            )
        self.state = "idle"
        self.call_id = None
        self.frequency_hz = 0.0
        self.talkgroup_id = 0
        self.offset_hz = 0.0
        self.audio_buffer.clear()

    def is_available(self) -> bool:
        """Check if recorder is available for new assignment."""
        if self.state == "idle":
            return True
        if self.state == "hold":
            # Can preempt if hold timeout expired
            return time.time() - self.last_activity > self.hold_timeout
        return False


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
    control_channel_freq_hz: Optional[float] = None
    control_channel_index: int = 0  # Index into cfg.control_channels

    # System info from RFSS/NET status broadcasts
    nac: Optional[int] = None  # Network Access Code (12-bit)
    system_id: Optional[int] = None
    rfss_id: Optional[int] = None
    site_id: Optional[int] = None

    # TSBK parser
    _tsbk_parser: Optional[TSBKParser] = None

    # Channel identifiers from IDEN_UP messages
    _channel_identifiers: Dict[int, ChannelIdentifier] = field(default_factory=dict)

    # Voice recorders
    _voice_recorders: List[VoiceRecorder] = field(default_factory=list)

    # Active calls
    _active_calls: Dict[str, ActiveCall] = field(default_factory=dict)
    _calls_by_talkgroup: Dict[int, str] = field(default_factory=dict)

    # Stats
    _tsbk_count: int = 0
    _grant_count: int = 0
    _last_tsbk_time: float = 0.0
    decode_rate: float = 0.0  # TSBK per second

    # Event callbacks
    on_call_start: Optional[Callable[[ActiveCall], None]] = None
    on_call_update: Optional[Callable[[ActiveCall], None]] = None
    on_call_end: Optional[Callable[[ActiveCall], None]] = None
    on_system_update: Optional[Callable[[TrunkingSystem], None]] = None

    # Timing
    _state_change_time: float = field(default_factory=time.time)
    _control_channel_timeout: float = 10.0  # Seconds before trying next CC

    def __post_init__(self) -> None:
        """Initialize after dataclass creation."""
        # Create TSBK parser
        self._tsbk_parser = TSBKParser()

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

    async def start(self) -> None:
        """Start the trunking system.

        This initializes the control channel monitor and begins
        searching for a valid control channel.
        """
        if self.state not in (TrunkingSystemState.STOPPED, TrunkingSystemState.FAILED):
            logger.warning(f"TrunkingSystem {self.cfg.id}: Cannot start from state {self.state}")
            return

        logger.info(f"TrunkingSystem {self.cfg.id}: Starting...")
        self._set_state(TrunkingSystemState.STARTING)

        # Validate config
        if not self.cfg.control_channels:
            logger.error(f"TrunkingSystem {self.cfg.id}: No control channels configured")
            self._set_state(TrunkingSystemState.FAILED)
            return

        # Start with first control channel
        self.control_channel_index = 0
        self.control_channel_freq_hz = self.cfg.control_channels[0]

        self._set_state(TrunkingSystemState.SEARCHING)
        self.control_channel_state = ControlChannelState.SEARCHING

        logger.info(
            f"TrunkingSystem {self.cfg.id}: Searching for control channel "
            f"at {self.control_channel_freq_hz/1e6:.4f} MHz"
        )

    async def stop(self) -> None:
        """Stop the trunking system."""
        if self.state == TrunkingSystemState.STOPPED:
            return

        logger.info(f"TrunkingSystem {self.cfg.id}: Stopping...")
        self._set_state(TrunkingSystemState.STOPPING)

        # End all active calls
        for call_id in list(self._active_calls.keys()):
            self._end_call(call_id, "system_stop")

        # Release all recorders
        for recorder in self._voice_recorders:
            recorder.release()

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

    def _handle_voice_grant(self, result: Dict[str, Any]) -> None:
        """Handle a voice channel grant TSBK."""
        grant: Optional[VoiceGrant] = result.get("grant")
        if grant is None:
            return

        self._grant_count += 1

        # Get talkgroup info
        tgid = grant.talkgroup_id
        tg_config = self.cfg.get_talkgroup(tgid)

        # Check if talkgroup is monitored
        if not self.cfg.is_talkgroup_monitored(tgid):
            logger.debug(f"TrunkingSystem {self.cfg.id}: Ignoring grant for unmonitored TG {tgid}")
            return

        # Calculate voice channel frequency
        freq_hz = self._calculate_frequency(grant.channel_id)
        if freq_hz is None:
            logger.warning(
                f"TrunkingSystem {self.cfg.id}: Cannot calculate frequency for "
                f"channel ID {grant.channel_id}"
            )
            return

        # Check if we already have a call for this talkgroup
        existing_call_id = self._calls_by_talkgroup.get(tgid)
        if existing_call_id and existing_call_id in self._active_calls:
            # Update existing call
            call = self._active_calls[existing_call_id]
            call.last_activity_time = time.time()
            call.source_id = grant.source_id

            # Check if frequency changed (call continuing on different channel)
            if abs(call.frequency_hz - freq_hz) > 1000:
                logger.info(
                    f"TrunkingSystem {self.cfg.id}: TG {tgid} moving from "
                    f"{call.frequency_hz/1e6:.4f} to {freq_hz/1e6:.4f} MHz"
                )
                call.frequency_hz = freq_hz
                call.channel_id = grant.channel_id

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
            source_id=grant.source_id,
            frequency_hz=freq_hz,
            channel_id=grant.channel_id,
            state=CallState.TUNING,
            start_time=time.time(),
            last_activity_time=time.time(),
            encrypted=grant.encrypted,
        )

        # Try to assign a voice recorder
        recorder = self._get_available_recorder(tgid)
        if recorder:
            recorder.assign(
                call_id=call.id,
                frequency_hz=freq_hz,
                talkgroup_id=tgid,
                center_hz=self.cfg.center_hz,
            )
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

        if self.on_call_start:
            self.on_call_start(call)

        # Transition to running state if not already
        if self.state == TrunkingSystemState.SYNCED:
            self._set_state(TrunkingSystemState.RUNNING)

    def _handle_channel_identifier(self, result: Dict[str, Any]) -> None:
        """Handle a channel identifier TSBK (IDEN_UP, IDEN_UP_VU)."""
        chan_id: Optional[ChannelIdentifier] = result.get("channel_id")
        if chan_id is None:
            return

        # Store in our map
        self._channel_identifiers[chan_id.identifier] = chan_id

        logger.debug(
            f"TrunkingSystem {self.cfg.id}: Channel ID {chan_id.identifier}: "
            f"base={chan_id.base_freq_hz/1e6:.4f} MHz, "
            f"spacing={chan_id.channel_spacing_hz/1e3:.1f} kHz"
        )

    def _handle_rfss_status(self, result: Dict[str, Any]) -> None:
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

    def _handle_net_status(self, result: Dict[str, Any]) -> None:
        """Handle Network Status Broadcast TSBK."""
        if "nac" in result:
            self.nac = result["nac"]
        if "system_id" in result:
            self.system_id = result["system_id"]

        logger.debug(
            f"TrunkingSystem {self.cfg.id}: Network status - "
            f"NAC={self.nac}, SysID={self.system_id}"
        )

    def _calculate_frequency(self, channel_id: int) -> Optional[float]:
        """Calculate frequency from channel ID.

        The channel ID format is: IDEN (4 bits) | CHANNEL (12 bits)
        """
        iden = (channel_id >> 12) & 0xF
        channel = channel_id & 0xFFF

        chan_info = self._channel_identifiers.get(iden)
        if chan_info is None:
            # Try common P25 defaults if we haven't received IDEN_UP yet
            # 700 MHz band: 12.5 kHz spacing, base ~769 MHz
            # 800 MHz band: 12.5 kHz spacing, base ~851 MHz
            logger.warning(
                f"TrunkingSystem {self.cfg.id}: No channel identifier for IDEN {iden}"
            )
            return None

        freq_hz = chan_info.base_freq_hz + (channel * chan_info.channel_spacing_hz)
        return freq_hz

    def _get_available_recorder(self, talkgroup_id: int) -> Optional[VoiceRecorder]:
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

                    recorder.release()
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

        # Release recorder
        if call.recorder_id:
            for recorder in self._voice_recorders:
                if recorder.id == call.recorder_id:
                    recorder.release()
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

    def get_active_calls(self) -> List[ActiveCall]:
        """Get list of active calls."""
        return list(self._active_calls.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "tsbk_count": self._tsbk_count,
            "grant_count": self._grant_count,
            "decode_rate": round(self.decode_rate, 2),
            "active_calls": len(self._active_calls),
            "recorders_idle": sum(1 for r in self._voice_recorders if r.state == "idle"),
            "recorders_active": sum(1 for r in self._voice_recorders if r.state == "recording"),
            "channel_identifiers": len(self._channel_identifiers),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert system state to dictionary for API serialization."""
        return {
            "id": self.cfg.id,
            "name": self.cfg.name,
            "protocol": self.cfg.protocol.value,
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
