"""P25 Trunking System Module.

This module provides the trunking controller infrastructure for P25 systems:
- TrunkingManager: Top-level manager for multiple trunking systems
- TrunkingSystem: Single P25 system controller
- ControlChannelMonitor: Monitors control channel for voice grants
- VoiceRecorder: Records voice channel calls

Architecture:
    TrunkingManager
        └── TrunkingSystem (per system: PSERN, SA-GRN, etc.)
            ├── ControlChannelMonitor (one per system)
            └── VoiceRecorderPool
                └── VoiceRecorder (multiple, follows voice grants)

Supports:
- P25 Phase I (C4FM, IMBE)
- P25 Phase II (CQPSK/TDMA, AMBE+2)
"""

from wavecapsdr.trunking.config import (
    TrunkingSystemConfig,
    TalkgroupConfig,
    TrunkingProtocol,
)
from wavecapsdr.trunking.system import (
    TrunkingSystem,
    TrunkingSystemState,
    ControlChannelState,
    CallState,
    ActiveCall,
    VoiceRecorder,
)
from wavecapsdr.trunking.manager import TrunkingManager
from wavecapsdr.trunking.control_channel import (
    ControlChannelMonitor,
    SyncState,
    create_control_monitor,
)

__all__ = [
    # Manager
    "TrunkingManager",
    # System
    "TrunkingSystem",
    "TrunkingSystemState",
    "ControlChannelState",
    "CallState",
    "ActiveCall",
    "VoiceRecorder",
    # Control Channel
    "ControlChannelMonitor",
    "SyncState",
    "create_control_monitor",
    # Config
    "TrunkingSystemConfig",
    "TalkgroupConfig",
    "TrunkingProtocol",
]
