"""Decoder state machine tests.

Tests state transitions, timeouts, error recovery, and event generation
for P25 decoders. Adapted from SDRTrunk's decoder state patterns.

Reference: https://github.com/DSheirer/sdrtrunk
"""

import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

import pytest


# ============================================================================
# State Machine Infrastructure (for testing)
# ============================================================================

class DecoderState(Enum):
    """P25 decoder states."""
    IDLE = auto()
    SEARCHING = auto()
    SYNC_ACQUIRED = auto()
    DECODING = auto()
    VOICE_ACTIVE = auto()
    DATA_ACTIVE = auto()
    TERMINATING = auto()
    ERROR = auto()


class DecoderEvent(Enum):
    """Events that trigger state transitions."""
    FRAME_SYNC_DETECTED = auto()
    FRAME_SYNC_LOST = auto()
    HDU_RECEIVED = auto()
    LDU1_RECEIVED = auto()
    LDU2_RECEIVED = auto()
    TDU_RECEIVED = auto()
    PDU_RECEIVED = auto()
    TIMEOUT = auto()
    ERROR_THRESHOLD = auto()
    RESET = auto()


@dataclass
class StateTransition:
    """Defines a valid state transition."""
    from_state: DecoderState
    event: DecoderEvent
    to_state: DecoderState
    action: Optional[Callable[[], None]] = None


class DecoderStateMachine:
    """P25 decoder state machine (simplified for testing)."""

    # Valid transitions
    TRANSITIONS = [
        # From IDLE
        StateTransition(DecoderState.IDLE, DecoderEvent.FRAME_SYNC_DETECTED, DecoderState.SYNC_ACQUIRED),
        StateTransition(DecoderState.IDLE, DecoderEvent.RESET, DecoderState.IDLE),

        # From SEARCHING
        StateTransition(DecoderState.SEARCHING, DecoderEvent.FRAME_SYNC_DETECTED, DecoderState.SYNC_ACQUIRED),
        StateTransition(DecoderState.SEARCHING, DecoderEvent.TIMEOUT, DecoderState.IDLE),

        # From SYNC_ACQUIRED
        StateTransition(DecoderState.SYNC_ACQUIRED, DecoderEvent.HDU_RECEIVED, DecoderState.DECODING),
        StateTransition(DecoderState.SYNC_ACQUIRED, DecoderEvent.LDU1_RECEIVED, DecoderState.VOICE_ACTIVE),
        StateTransition(DecoderState.SYNC_ACQUIRED, DecoderEvent.PDU_RECEIVED, DecoderState.DATA_ACTIVE),
        StateTransition(DecoderState.SYNC_ACQUIRED, DecoderEvent.FRAME_SYNC_LOST, DecoderState.SEARCHING),
        StateTransition(DecoderState.SYNC_ACQUIRED, DecoderEvent.TIMEOUT, DecoderState.IDLE),

        # From DECODING
        StateTransition(DecoderState.DECODING, DecoderEvent.LDU1_RECEIVED, DecoderState.VOICE_ACTIVE),
        StateTransition(DecoderState.DECODING, DecoderEvent.PDU_RECEIVED, DecoderState.DATA_ACTIVE),
        StateTransition(DecoderState.DECODING, DecoderEvent.TDU_RECEIVED, DecoderState.TERMINATING),
        StateTransition(DecoderState.DECODING, DecoderEvent.FRAME_SYNC_LOST, DecoderState.SEARCHING),
        StateTransition(DecoderState.DECODING, DecoderEvent.ERROR_THRESHOLD, DecoderState.ERROR),

        # From VOICE_ACTIVE
        StateTransition(DecoderState.VOICE_ACTIVE, DecoderEvent.LDU1_RECEIVED, DecoderState.VOICE_ACTIVE),
        StateTransition(DecoderState.VOICE_ACTIVE, DecoderEvent.LDU2_RECEIVED, DecoderState.VOICE_ACTIVE),
        StateTransition(DecoderState.VOICE_ACTIVE, DecoderEvent.TDU_RECEIVED, DecoderState.TERMINATING),
        StateTransition(DecoderState.VOICE_ACTIVE, DecoderEvent.FRAME_SYNC_LOST, DecoderState.SEARCHING),
        StateTransition(DecoderState.VOICE_ACTIVE, DecoderEvent.TIMEOUT, DecoderState.IDLE),

        # From DATA_ACTIVE
        StateTransition(DecoderState.DATA_ACTIVE, DecoderEvent.PDU_RECEIVED, DecoderState.DATA_ACTIVE),
        StateTransition(DecoderState.DATA_ACTIVE, DecoderEvent.TDU_RECEIVED, DecoderState.TERMINATING),
        StateTransition(DecoderState.DATA_ACTIVE, DecoderEvent.FRAME_SYNC_LOST, DecoderState.SEARCHING),
        StateTransition(DecoderState.DATA_ACTIVE, DecoderEvent.TIMEOUT, DecoderState.IDLE),

        # From TERMINATING
        StateTransition(DecoderState.TERMINATING, DecoderEvent.RESET, DecoderState.IDLE),
        StateTransition(DecoderState.TERMINATING, DecoderEvent.FRAME_SYNC_DETECTED, DecoderState.SYNC_ACQUIRED),
        StateTransition(DecoderState.TERMINATING, DecoderEvent.TIMEOUT, DecoderState.IDLE),

        # From ERROR
        StateTransition(DecoderState.ERROR, DecoderEvent.RESET, DecoderState.IDLE),
        StateTransition(DecoderState.ERROR, DecoderEvent.TIMEOUT, DecoderState.IDLE),
    ]

    def __init__(self):
        self.state = DecoderState.IDLE
        self.transition_history: List[tuple] = []
        self.error_count = 0
        self.last_event_time = time.time()

        # Build transition map
        self._transitions: Dict[tuple, DecoderState] = {}
        for t in self.TRANSITIONS:
            self._transitions[(t.from_state, t.event)] = t.to_state

    def can_transition(self, event: DecoderEvent) -> bool:
        """Check if event can trigger a transition from current state."""
        return (self.state, event) in self._transitions

    def process_event(self, event: DecoderEvent) -> bool:
        """Process event and transition if valid.

        Returns True if transition occurred.
        """
        key = (self.state, event)
        if key not in self._transitions:
            return False

        old_state = self.state
        self.state = self._transitions[key]
        self.transition_history.append((old_state, event, self.state))
        self.last_event_time = time.time()

        return True

    def reset(self):
        """Reset to idle state."""
        self.process_event(DecoderEvent.RESET)

    def get_valid_events(self) -> Set[DecoderEvent]:
        """Get events valid from current state."""
        return {event for (state, event) in self._transitions.keys()
                if state == self.state}


# ============================================================================
# State Transition Tests
# ============================================================================

class TestValidStateTransitions:
    """Test valid state transitions."""

    def test_idle_to_sync_acquired(self):
        """IDLE -> SYNC_ACQUIRED on frame sync."""
        sm = DecoderStateMachine()
        assert sm.state == DecoderState.IDLE

        assert sm.process_event(DecoderEvent.FRAME_SYNC_DETECTED)
        assert sm.state == DecoderState.SYNC_ACQUIRED

    def test_sync_to_voice_active(self):
        """SYNC_ACQUIRED -> VOICE_ACTIVE on LDU1."""
        sm = DecoderStateMachine()
        sm.process_event(DecoderEvent.FRAME_SYNC_DETECTED)

        assert sm.process_event(DecoderEvent.LDU1_RECEIVED)
        assert sm.state == DecoderState.VOICE_ACTIVE

    def test_voice_continuation(self):
        """VOICE_ACTIVE stays active on LDU1/LDU2."""
        sm = DecoderStateMachine()
        sm.process_event(DecoderEvent.FRAME_SYNC_DETECTED)
        sm.process_event(DecoderEvent.LDU1_RECEIVED)

        # Multiple LDU frames maintain voice state
        assert sm.process_event(DecoderEvent.LDU2_RECEIVED)
        assert sm.state == DecoderState.VOICE_ACTIVE

        assert sm.process_event(DecoderEvent.LDU1_RECEIVED)
        assert sm.state == DecoderState.VOICE_ACTIVE

    def test_voice_to_terminating(self):
        """VOICE_ACTIVE -> TERMINATING on TDU."""
        sm = DecoderStateMachine()
        sm.process_event(DecoderEvent.FRAME_SYNC_DETECTED)
        sm.process_event(DecoderEvent.LDU1_RECEIVED)

        assert sm.process_event(DecoderEvent.TDU_RECEIVED)
        assert sm.state == DecoderState.TERMINATING

    def test_terminating_to_idle(self):
        """TERMINATING -> IDLE on reset/timeout."""
        sm = DecoderStateMachine()
        sm.process_event(DecoderEvent.FRAME_SYNC_DETECTED)
        sm.process_event(DecoderEvent.LDU1_RECEIVED)
        sm.process_event(DecoderEvent.TDU_RECEIVED)

        assert sm.process_event(DecoderEvent.RESET)
        assert sm.state == DecoderState.IDLE


class TestInvalidStateTransitions:
    """Test invalid state transitions are rejected."""

    def test_idle_rejects_ldu(self):
        """IDLE state rejects LDU without sync."""
        sm = DecoderStateMachine()

        assert not sm.process_event(DecoderEvent.LDU1_RECEIVED)
        assert sm.state == DecoderState.IDLE

    def test_sync_rejects_tdu(self):
        """SYNC_ACQUIRED rejects TDU without decode."""
        sm = DecoderStateMachine()
        sm.process_event(DecoderEvent.FRAME_SYNC_DETECTED)

        assert not sm.process_event(DecoderEvent.TDU_RECEIVED)
        assert sm.state == DecoderState.SYNC_ACQUIRED

    def test_voice_rejects_pdu(self):
        """VOICE_ACTIVE rejects PDU (voice/data are separate)."""
        sm = DecoderStateMachine()
        sm.process_event(DecoderEvent.FRAME_SYNC_DETECTED)
        sm.process_event(DecoderEvent.LDU1_RECEIVED)

        assert not sm.process_event(DecoderEvent.PDU_RECEIVED)
        assert sm.state == DecoderState.VOICE_ACTIVE


# ============================================================================
# Complete Sequence Tests
# ============================================================================

class TestCompleteSequences:
    """Test complete decoder sequences."""

    def test_complete_voice_call(self):
        """Complete voice call: IDLE -> SYNC -> VOICE -> TERM -> IDLE."""
        sm = DecoderStateMachine()

        # Sync acquired
        sm.process_event(DecoderEvent.FRAME_SYNC_DETECTED)
        assert sm.state == DecoderState.SYNC_ACQUIRED

        # Voice starts
        sm.process_event(DecoderEvent.LDU1_RECEIVED)
        assert sm.state == DecoderState.VOICE_ACTIVE

        # Voice continues
        for _ in range(5):
            sm.process_event(DecoderEvent.LDU2_RECEIVED)
            sm.process_event(DecoderEvent.LDU1_RECEIVED)
        assert sm.state == DecoderState.VOICE_ACTIVE

        # Voice ends
        sm.process_event(DecoderEvent.TDU_RECEIVED)
        assert sm.state == DecoderState.TERMINATING

        # Reset to idle
        sm.process_event(DecoderEvent.RESET)
        assert sm.state == DecoderState.IDLE

    def test_complete_data_call(self):
        """Complete data call: IDLE -> SYNC -> DATA -> TERM -> IDLE."""
        sm = DecoderStateMachine()

        sm.process_event(DecoderEvent.FRAME_SYNC_DETECTED)
        sm.process_event(DecoderEvent.PDU_RECEIVED)
        assert sm.state == DecoderState.DATA_ACTIVE

        # Data continues
        for _ in range(3):
            sm.process_event(DecoderEvent.PDU_RECEIVED)
        assert sm.state == DecoderState.DATA_ACTIVE

        # Data ends
        sm.process_event(DecoderEvent.TDU_RECEIVED)
        sm.process_event(DecoderEvent.RESET)
        assert sm.state == DecoderState.IDLE

    def test_voice_with_hdu(self):
        """Voice call with HDU header."""
        sm = DecoderStateMachine()

        sm.process_event(DecoderEvent.FRAME_SYNC_DETECTED)
        sm.process_event(DecoderEvent.HDU_RECEIVED)
        assert sm.state == DecoderState.DECODING

        sm.process_event(DecoderEvent.LDU1_RECEIVED)
        assert sm.state == DecoderState.VOICE_ACTIVE


# ============================================================================
# Error Recovery Tests
# ============================================================================

class TestErrorRecovery:
    """Test decoder error recovery."""

    def test_sync_loss_during_voice(self):
        """Sync loss during voice triggers search."""
        sm = DecoderStateMachine()
        sm.process_event(DecoderEvent.FRAME_SYNC_DETECTED)
        sm.process_event(DecoderEvent.LDU1_RECEIVED)

        sm.process_event(DecoderEvent.FRAME_SYNC_LOST)
        assert sm.state == DecoderState.SEARCHING

    def test_recover_from_sync_loss(self):
        """Recover from sync loss."""
        sm = DecoderStateMachine()
        sm.process_event(DecoderEvent.FRAME_SYNC_DETECTED)
        sm.process_event(DecoderEvent.LDU1_RECEIVED)
        sm.process_event(DecoderEvent.FRAME_SYNC_LOST)

        # Reacquire sync
        sm.process_event(DecoderEvent.FRAME_SYNC_DETECTED)
        assert sm.state == DecoderState.SYNC_ACQUIRED

    def test_error_state_recovery(self):
        """Recover from error state via reset."""
        sm = DecoderStateMachine()
        sm.process_event(DecoderEvent.FRAME_SYNC_DETECTED)
        sm.process_event(DecoderEvent.HDU_RECEIVED)
        sm.process_event(DecoderEvent.ERROR_THRESHOLD)
        assert sm.state == DecoderState.ERROR

        sm.process_event(DecoderEvent.RESET)
        assert sm.state == DecoderState.IDLE

    def test_timeout_from_voice(self):
        """Voice timeout returns to idle."""
        sm = DecoderStateMachine()
        sm.process_event(DecoderEvent.FRAME_SYNC_DETECTED)
        sm.process_event(DecoderEvent.LDU1_RECEIVED)

        sm.process_event(DecoderEvent.TIMEOUT)
        assert sm.state == DecoderState.IDLE


# ============================================================================
# Transition History Tests
# ============================================================================

class TestTransitionHistory:
    """Test state transition history tracking."""

    def test_history_recorded(self):
        """Transitions are recorded in history."""
        sm = DecoderStateMachine()
        sm.process_event(DecoderEvent.FRAME_SYNC_DETECTED)
        sm.process_event(DecoderEvent.LDU1_RECEIVED)

        assert len(sm.transition_history) == 2
        assert sm.transition_history[0] == (
            DecoderState.IDLE,
            DecoderEvent.FRAME_SYNC_DETECTED,
            DecoderState.SYNC_ACQUIRED
        )
        assert sm.transition_history[1] == (
            DecoderState.SYNC_ACQUIRED,
            DecoderEvent.LDU1_RECEIVED,
            DecoderState.VOICE_ACTIVE
        )

    def test_invalid_transition_not_recorded(self):
        """Invalid transitions are not recorded."""
        sm = DecoderStateMachine()
        sm.process_event(DecoderEvent.LDU1_RECEIVED)  # Invalid from IDLE

        assert len(sm.transition_history) == 0


# ============================================================================
# Valid Events Query Tests
# ============================================================================

class TestValidEventsQuery:
    """Test querying valid events for current state."""

    def test_idle_valid_events(self):
        """Valid events from IDLE state."""
        sm = DecoderStateMachine()
        valid = sm.get_valid_events()

        assert DecoderEvent.FRAME_SYNC_DETECTED in valid
        assert DecoderEvent.RESET in valid
        assert DecoderEvent.LDU1_RECEIVED not in valid

    def test_voice_valid_events(self):
        """Valid events from VOICE_ACTIVE state."""
        sm = DecoderStateMachine()
        sm.process_event(DecoderEvent.FRAME_SYNC_DETECTED)
        sm.process_event(DecoderEvent.LDU1_RECEIVED)

        valid = sm.get_valid_events()

        assert DecoderEvent.LDU1_RECEIVED in valid
        assert DecoderEvent.LDU2_RECEIVED in valid
        assert DecoderEvent.TDU_RECEIVED in valid
        assert DecoderEvent.FRAME_SYNC_LOST in valid
        assert DecoderEvent.TIMEOUT in valid


# ============================================================================
# Frame Sync State Tests
# ============================================================================

class SyncState(Enum):
    """Frame synchronization states."""
    SEARCHING = auto()
    TENTATIVE = auto()  # Found once, looking for confirmation
    LOCKED = auto()
    LOST = auto()


class FrameSyncStateMachine:
    """Frame sync state machine with confirmation requirement."""

    SYNC_CONFIRM_COUNT = 3  # Require 3 consecutive syncs to lock
    SYNC_LOSS_COUNT = 2     # 2 consecutive failures to lose lock

    def __init__(self):
        self.state = SyncState.SEARCHING
        self.consecutive_syncs = 0
        self.consecutive_failures = 0

    def process_frame(self, sync_found: bool) -> SyncState:
        """Process frame and update sync state."""
        if sync_found:
            self.consecutive_syncs += 1
            self.consecutive_failures = 0

            if self.state == SyncState.SEARCHING:
                self.state = SyncState.TENTATIVE
            elif self.state == SyncState.TENTATIVE:
                if self.consecutive_syncs >= self.SYNC_CONFIRM_COUNT:
                    self.state = SyncState.LOCKED
            elif self.state == SyncState.LOST:
                self.state = SyncState.TENTATIVE
                self.consecutive_syncs = 1
        else:
            self.consecutive_failures += 1
            self.consecutive_syncs = 0

            if self.state in (SyncState.LOCKED, SyncState.TENTATIVE):
                if self.consecutive_failures >= self.SYNC_LOSS_COUNT:
                    self.state = SyncState.LOST

        return self.state


class TestFrameSyncStateMachine:
    """Test frame sync state machine."""

    def test_initial_state(self):
        """Initial state is SEARCHING."""
        sm = FrameSyncStateMachine()
        assert sm.state == SyncState.SEARCHING

    def test_first_sync_tentative(self):
        """First sync moves to TENTATIVE."""
        sm = FrameSyncStateMachine()
        sm.process_frame(sync_found=True)
        assert sm.state == SyncState.TENTATIVE

    def test_lock_after_confirms(self):
        """Lock after consecutive sync confirmations."""
        sm = FrameSyncStateMachine()

        for _ in range(3):
            sm.process_frame(sync_found=True)

        assert sm.state == SyncState.LOCKED

    def test_stay_locked_with_syncs(self):
        """Stay locked with continued syncs."""
        sm = FrameSyncStateMachine()
        for _ in range(3):
            sm.process_frame(sync_found=True)

        # More syncs keep us locked
        for _ in range(10):
            sm.process_frame(sync_found=True)

        assert sm.state == SyncState.LOCKED

    def test_lose_lock_on_failures(self):
        """Lose lock after consecutive failures."""
        sm = FrameSyncStateMachine()
        for _ in range(3):
            sm.process_frame(sync_found=True)
        assert sm.state == SyncState.LOCKED

        # Consecutive failures
        sm.process_frame(sync_found=False)
        sm.process_frame(sync_found=False)

        assert sm.state == SyncState.LOST

    def test_single_failure_tolerated(self):
        """Single failure doesn't lose lock."""
        sm = FrameSyncStateMachine()
        for _ in range(3):
            sm.process_frame(sync_found=True)

        sm.process_frame(sync_found=False)
        assert sm.state == SyncState.LOCKED

        sm.process_frame(sync_found=True)  # Recovery
        assert sm.state == SyncState.LOCKED

    def test_reacquire_after_loss(self):
        """Reacquire sync after loss."""
        sm = FrameSyncStateMachine()
        for _ in range(3):
            sm.process_frame(sync_found=True)

        # Lose sync
        sm.process_frame(sync_found=False)
        sm.process_frame(sync_found=False)
        assert sm.state == SyncState.LOST

        # Reacquire
        sm.process_frame(sync_found=True)
        assert sm.state == SyncState.TENTATIVE

        for _ in range(2):
            sm.process_frame(sync_found=True)
        assert sm.state == SyncState.LOCKED


# ============================================================================
# Timeout Handling Tests
# ============================================================================

class TestTimeoutHandling:
    """Test timeout-based state transitions."""

    def test_last_event_time_updated(self):
        """Last event time is updated on transitions."""
        sm = DecoderStateMachine()
        before = sm.last_event_time

        time.sleep(0.01)
        sm.process_event(DecoderEvent.FRAME_SYNC_DETECTED)

        assert sm.last_event_time > before

    def test_timeout_detection(self):
        """Detect when state has timed out."""
        sm = DecoderStateMachine()
        sm.process_event(DecoderEvent.FRAME_SYNC_DETECTED)
        sm.process_event(DecoderEvent.LDU1_RECEIVED)

        timeout_threshold = 0.01  # 10ms for testing

        # Immediately after - not timed out
        elapsed = time.time() - sm.last_event_time
        assert elapsed < timeout_threshold

        # After waiting - timed out
        time.sleep(0.02)
        elapsed = time.time() - sm.last_event_time
        assert elapsed > timeout_threshold
