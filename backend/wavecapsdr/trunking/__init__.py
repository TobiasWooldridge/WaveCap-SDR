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
    TalkgroupConfig,
    TrunkingProtocol,
    TrunkingSystemConfig,
)
from wavecapsdr.trunking.control_channel import (
    ControlChannelMonitor,
    SyncState,
    create_control_monitor,
)
from wavecapsdr.trunking.duplicate_detector import (
    CallEventSignature,
    DuplicateCallDetector,
    FrequencyBasedDuplicateDetector,
)
from wavecapsdr.trunking.event_tracker import (
    CallEventState,
    CallEventType,
    P25CallEvent,
    P25EventTracker,
    P25EventTrackerManager,
)
from wavecapsdr.trunking.identifiers import (
    Identifier,
    IdentifierCollection,
    IdentifierForm,
    IdentifierRole,
    MutableIdentifierCollection,
    TalkerAliasManager,
)
from wavecapsdr.trunking.manager import TrunkingManager
from wavecapsdr.trunking.network_config import (
    AdjacentSite,
    FrequencyBand,
    NetworkStatus,
    P25NetworkConfigurationMonitor,
    SiteStatus,
    SystemServices,
)
from wavecapsdr.trunking.system import (
    ActiveCall,
    CallState,
    ControlChannelState,
    TrunkingSystem,
    TrunkingSystemState,
    VoiceRecorder,
)

__all__ = [
    "ActiveCall",
    "AdjacentSite",
    "CallEventSignature",
    "CallEventState",
    "CallEventType",
    "CallState",
    # Control Channel
    "ControlChannelMonitor",
    "ControlChannelState",
    # Duplicate Detection (SDRTrunk pattern)
    "DuplicateCallDetector",
    "FrequencyBand",
    "FrequencyBasedDuplicateDetector",
    # Identifiers (SDRTrunk pattern)
    "Identifier",
    "IdentifierCollection",
    "IdentifierForm",
    "IdentifierRole",
    "MutableIdentifierCollection",
    "NetworkStatus",
    # Event Tracker (SDRTrunk pattern)
    "P25CallEvent",
    "P25EventTracker",
    "P25EventTrackerManager",
    # Network Config (SDRTrunk pattern)
    "P25NetworkConfigurationMonitor",
    "SiteStatus",
    "SyncState",
    "SystemServices",
    "TalkerAliasManager",
    "TalkgroupConfig",
    # Manager
    "TrunkingManager",
    "TrunkingProtocol",
    # System
    "TrunkingSystem",
    # Config
    "TrunkingSystemConfig",
    "TrunkingSystemState",
    "VoiceRecorder",
    "create_control_monitor",
]
