"""Trunking REST API endpoints.

Provides REST and WebSocket endpoints for P25 trunking systems:

REST Endpoints:
- GET    /trunking/systems             - List all trunking systems
- POST   /trunking/systems             - Create a new system
- GET    /trunking/systems/{id}        - Get system details
- DELETE /trunking/systems/{id}        - Remove a system
- POST   /trunking/systems/{id}/start  - Start a system
- POST   /trunking/systems/{id}/stop   - Stop a system
- GET    /trunking/systems/{id}/talkgroups     - Get talkgroups
- POST   /trunking/systems/{id}/talkgroups     - Add/import talkgroups
- GET    /trunking/systems/{id}/calls/active   - Get active calls
- GET    /trunking/calls               - Get all active calls
- GET    /trunking/vocoders            - Check vocoder availability

WebSocket Endpoints:
- /stream/trunking/{systemId}  - Real-time trunking events
- /stream/trunking             - Events for all systems
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from wavecapsdr.decoders.voice import VoiceDecoder
from wavecapsdr.trunking import (
    HuntMode,
    TalkgroupConfig,
    TrunkingManager,
    TrunkingProtocol,
    TrunkingSystem,
    TrunkingSystemConfig,
)
from wavecapsdr.trunking.config import P25Modulation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trunking", tags=["trunking"])


# ============================================================================
# Pydantic Models for API
# ============================================================================

class TalkgroupRequest(BaseModel):
    """Request to create/update a talkgroup."""
    tgid: int = Field(..., description="Talkgroup ID (decimal)")
    name: str = Field(..., description="Human-readable name")
    alpha_tag: str | None = Field(None, description="Short identifier")
    category: str | None = Field(None, description="Category for grouping")
    priority: int = Field(5, ge=1, le=10, description="Priority (1=highest, 10=lowest)")
    record: bool = Field(True, description="Whether to record calls")
    monitor: bool = Field(True, description="Whether to stream audio live")


class CreateSystemRequest(BaseModel):
    """Request to create a new trunking system."""
    id: str = Field(..., description="Unique system identifier")
    name: str = Field(..., description="Human-readable system name")
    protocol: str = Field("p25_phase1", description="Protocol: p25_phase1 or p25_phase2")
    modulation: str | None = Field(None, description="Modulation: c4fm (standard) or lsm (simulcast)")
    control_channels: list[float] = Field(..., description="Control channel frequencies (Hz)")
    center_hz: float = Field(..., description="SDR center frequency (Hz)")
    sample_rate: int = Field(8_000_000, description="SDR sample rate (Hz)")
    device_id: str | None = Field(None, description="SoapySDR device string")
    gain: float | None = Field(None, description="RF gain (None = auto)")
    antenna: str | None = Field(None, description="SDR antenna port")
    max_voice_recorders: int = Field(4, ge=1, le=16, description="Maximum concurrent recordings")
    recording_path: str | None = Field(None, description="Path for audio file storage")
    record_unknown: bool = Field(False, description="Record unknown talkgroups")
    squelch_db: float = Field(-50.0, description="Squelch level for voice channels (dB)")
    auto_start: bool = Field(True, description="Start system automatically after creation")
    talkgroups: dict[str, TalkgroupRequest] | None = Field(
        None, description="Initial talkgroup configuration"
    )


class ControlChannelResponse(BaseModel):
    """Response containing control channel details."""
    frequencyHz: float
    enabled: bool
    isCurrent: bool
    isLocked: bool
    snrDb: float | None
    powerDb: float | None
    syncDetected: bool
    measurementTime: float | None


class SystemResponse(BaseModel):
    """Response containing system details."""
    id: str
    name: str
    protocol: str
    deviceId: str | None
    state: str
    controlChannelState: str
    controlChannelFreqHz: float | None
    centerHz: float  # SDR center frequency (auto-managed by trunking)
    nac: int | None
    systemId: int | None
    rfssId: int | None
    siteId: int | None
    decodeRate: float
    activeCalls: int
    stats: dict[str, Any]
    # Hunt mode fields
    huntMode: str
    lockedFrequencyHz: float | None
    controlChannels: list[ControlChannelResponse]


class ActiveCallResponse(BaseModel):
    """Response containing active call details."""
    id: str
    talkgroupId: int
    talkgroupName: str
    sourceId: int | None
    frequencyHz: float
    channelId: int
    state: str
    startTime: float
    lastActivityTime: float
    encrypted: bool
    audioFrames: int
    durationSeconds: float
    recorderId: str | None = None


class TalkgroupResponse(BaseModel):
    """Response containing talkgroup details."""
    tgid: int
    name: str
    alphaTag: str
    category: str
    priority: int
    record: bool
    monitor: bool


class VocoderStatusResponse(BaseModel):
    """Response containing vocoder availability."""
    imbe: dict[str, Any]
    ambe2: dict[str, Any]
    anyAvailable: bool


class LocationResponse(BaseModel):
    """Response containing GPS location."""
    unitId: int
    latitude: float
    longitude: float
    altitude: float | None = None
    speed: float | None = None
    heading: float | None = None
    accuracy: float | None = None
    timestamp: float
    ageSeconds: float
    source: str = "unknown"


class VoiceStreamResponse(BaseModel):
    """Response containing voice stream details."""
    id: str
    systemId: str
    callId: str
    recorderId: str
    state: str
    talkgroupId: int
    talkgroupName: str
    sourceId: int | None = None
    encrypted: bool = False
    startTime: float
    durationSeconds: float
    silenceSeconds: float
    audioFrameCount: int
    audioBytesSent: int
    subscriberCount: int
    sourceLocation: LocationResponse | None = None


class TrunkingRecipeResponse(BaseModel):
    """Response containing a trunking system recipe/template."""
    id: str
    name: str
    description: str | None = None
    category: str = "P25 Trunking"
    protocol: str
    controlChannels: list[float]
    centerHz: float
    sampleRate: int
    gain: float | None = None
    talkgroupCount: int = 0


# ============================================================================
# Helper Functions
# ============================================================================

def get_trunking_manager(request: Request) -> TrunkingManager:
    """Get the TrunkingManager from app state."""
    state = getattr(request.app.state, "app_state", None)
    if state is None:
        raise HTTPException(status_code=500, detail="AppState not initialized")

    manager = getattr(state, "trunking_manager", None)
    if manager is None:
        raise HTTPException(status_code=500, detail="TrunkingManager not initialized")

    return manager


def system_to_response(system: TrunkingSystem) -> SystemResponse:
    """Convert TrunkingSystem to API response."""
    d = system.to_dict()
    return SystemResponse(
        id=d["id"],
        name=d["name"],
        protocol=d["protocol"],
        deviceId=d.get("deviceId"),
        state=d["state"],
        controlChannelState=d["controlChannelState"],
        controlChannelFreqHz=d["controlChannelFreqHz"],
        centerHz=system.cfg.center_hz,  # SDR center frequency (auto-managed)
        nac=d["nac"],
        systemId=d["systemId"],
        rfssId=d["rfssId"],
        siteId=d["siteId"],
        decodeRate=d["decodeRate"],
        activeCalls=d["activeCalls"],
        stats=d["stats"],
        # Hunt mode fields
        huntMode=d["huntMode"],
        lockedFrequencyHz=d["lockedFrequencyHz"],
        controlChannels=d["controlChannels"],
    )


# ============================================================================
# REST Endpoints
# ============================================================================

@router.get("/systems", response_model=list[SystemResponse])
async def list_systems(request: Request) -> list[SystemResponse]:
    """List all trunking systems."""
    manager = get_trunking_manager(request)
    systems = manager.list_systems()
    return [system_to_response(s) for s in systems]


@router.post("/systems", response_model=SystemResponse)
async def create_system(request: Request, req: CreateSystemRequest) -> SystemResponse:
    """Create a new trunking system."""
    manager = get_trunking_manager(request)

    # Parse protocol
    try:
        protocol = TrunkingProtocol(req.protocol)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid protocol: {req.protocol}. Use 'p25_phase1' or 'p25_phase2'"
        )

    # Parse modulation
    modulation: P25Modulation | None = None
    if req.modulation:
        try:
            modulation = P25Modulation(req.modulation)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid modulation: {req.modulation}. Use 'c4fm' or 'lsm'"
            )

    # Build talkgroups dict
    talkgroups: dict[int, TalkgroupConfig] = {}
    if req.talkgroups:
        for tgid_str, tg_req in req.talkgroups.items():
            tgid = int(tgid_str) if tgid_str.isdigit() else tg_req.tgid
            talkgroups[tgid] = TalkgroupConfig(
                tgid=tgid,
                name=tg_req.name,
                alpha_tag=tg_req.alpha_tag or "",
                category=tg_req.category or "",
                priority=tg_req.priority,
                record=tg_req.record,
                monitor=tg_req.monitor,
            )

    # Create config
    config = TrunkingSystemConfig(
        id=req.id,
        name=req.name,
        protocol=protocol,
        modulation=modulation,
        control_channels=req.control_channels,
        center_hz=req.center_hz,
        sample_rate=req.sample_rate,
        device_id=req.device_id or "",
        gain=req.gain,
        antenna=req.antenna,
        max_voice_recorders=req.max_voice_recorders,
        talkgroups=talkgroups,
        recording_path=req.recording_path or "./recordings",
        record_unknown=req.record_unknown,
        squelch_db=req.squelch_db,
        auto_start=req.auto_start,
    )

    try:
        system = await manager.add_system(config)
        return system_to_response(system)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/systems/{system_id}", response_model=SystemResponse)
async def get_system(request: Request, system_id: str) -> SystemResponse:
    """Get details of a specific system."""
    manager = get_trunking_manager(request)
    system = manager.get_system(system_id)

    if system is None:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found")

    return system_to_response(system)


@router.delete("/systems/{system_id}")
async def delete_system(request: Request, system_id: str) -> dict[str, str]:
    """Remove a trunking system."""
    manager = get_trunking_manager(request)

    try:
        await manager.remove_system(system_id)
        return {"status": "ok", "message": f"System '{system_id}' removed"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/systems/{system_id}/start")
async def start_system(request: Request, system_id: str) -> dict[str, str]:
    """Start a trunking system."""
    manager = get_trunking_manager(request)

    try:
        await manager.start_system(system_id)
        return {"status": "ok", "message": f"System '{system_id}' started"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/systems/{system_id}/stop")
async def stop_system(request: Request, system_id: str) -> dict[str, str]:
    """Stop a trunking system."""
    manager = get_trunking_manager(request)

    try:
        await manager.stop_system(system_id)
        return {"status": "ok", "message": f"System '{system_id}' stopped"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/systems/{system_id}/talkgroups", response_model=list[TalkgroupResponse])
async def get_talkgroups(request: Request, system_id: str) -> list[TalkgroupResponse]:
    """Get talkgroups for a system."""
    manager = get_trunking_manager(request)
    system = manager.get_system(system_id)

    if system is None:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found")

    return [
        TalkgroupResponse(
            tgid=tg.tgid,
            name=tg.name,
            alphaTag=tg.alpha_tag,
            category=tg.category,
            priority=tg.priority,
            record=tg.record,
            monitor=tg.monitor,
        )
        for tg in system.cfg.talkgroups.values()
    ]


@router.post("/systems/{system_id}/talkgroups")
async def add_talkgroups(
    request: Request,
    system_id: str,
    talkgroups: list[TalkgroupRequest],
) -> dict[str, Any]:
    """Add or update talkgroups for a system."""
    manager = get_trunking_manager(request)
    system = manager.get_system(system_id)

    if system is None:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found")

    added = 0
    updated = 0

    for tg_req in talkgroups:
        tg = TalkgroupConfig(
            tgid=tg_req.tgid,
            name=tg_req.name,
            alpha_tag=tg_req.alpha_tag or "",
            category=tg_req.category or "",
            priority=tg_req.priority,
            record=tg_req.record,
            monitor=tg_req.monitor,
        )

        if tg_req.tgid in system.cfg.talkgroups:
            updated += 1
        else:
            added += 1

        system.cfg.talkgroups[tg_req.tgid] = tg

    return {
        "status": "ok",
        "added": added,
        "updated": updated,
        "total": len(system.cfg.talkgroups),
    }


@router.get("/systems/{system_id}/calls/active", response_model=list[ActiveCallResponse])
async def get_system_active_calls(request: Request, system_id: str) -> list[ActiveCallResponse]:
    """Get active calls for a specific system."""
    manager = get_trunking_manager(request)
    system = manager.get_system(system_id)

    if system is None:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found")

    calls = system.get_active_calls()
    return [
        ActiveCallResponse(
            id=c.id,
            talkgroupId=c.talkgroup_id,
            talkgroupName=c.talkgroup_name,
            sourceId=c.source_id,
            frequencyHz=c.frequency_hz,
            channelId=c.channel_id,
            state=c.state.value,
            startTime=c.start_time,
            lastActivityTime=c.last_activity_time,
            encrypted=c.encrypted,
            audioFrames=c.audio_frames,
            durationSeconds=c.duration_seconds,
            recorderId=c.recorder_id,
        )
        for c in calls
    ]


@router.get("/calls", response_model=list[ActiveCallResponse])
async def get_all_active_calls(request: Request) -> list[ActiveCallResponse]:
    """Get all active calls across all systems."""
    manager = get_trunking_manager(request)
    calls = manager.get_active_calls()

    return [
        ActiveCallResponse(
            id=c.id,
            talkgroupId=c.talkgroup_id,
            talkgroupName=c.talkgroup_name,
            sourceId=c.source_id,
            frequencyHz=c.frequency_hz,
            channelId=c.channel_id,
            state=c.state.value,
            startTime=c.start_time,
            lastActivityTime=c.last_activity_time,
            encrypted=c.encrypted,
            audioFrames=c.audio_frames,
            durationSeconds=c.duration_seconds,
            recorderId=c.recorder_id,
        )
        for c in calls
    ]


@router.get("/vocoders", response_model=VocoderStatusResponse)
async def get_vocoder_status() -> VocoderStatusResponse:
    """Check vocoder availability."""
    availability = VoiceDecoder.check_availability()
    return VocoderStatusResponse(
        imbe=availability["imbe"],
        ambe2=availability["ambe2"],
        anyAvailable=availability["any_available"],
    )


@router.get("/systems/{system_id}/locations", response_model=list[LocationResponse])
async def get_radio_locations(system_id: str, request: Request) -> list[LocationResponse]:
    """Get cached GPS locations for radio units in a trunking system.

    Returns all fresh (non-stale) radio locations from the LRRP cache.
    Locations are gathered from:
    - Extended Link Control in voice frames (LCF 0x09, 0x0A, 0x0B)
    - LRRP packets in PDU frames
    """
    manager = get_trunking_manager(request)
    system = manager.get_system(system_id)
    if system is None:
        raise HTTPException(status_code=404, detail=f"System {system_id} not found")

    locations = system.get_all_locations()
    return [
        LocationResponse(
            unitId=loc.unit_id,
            latitude=loc.latitude,
            longitude=loc.longitude,
            altitude=loc.altitude_m,
            speed=loc.speed_kmh,
            heading=loc.heading_deg,
            accuracy=loc.accuracy_m,
            timestamp=loc.timestamp,
            ageSeconds=loc.age_seconds(),
            source=loc.source,
        )
        for loc in locations
    ]


class MessageResponse(BaseModel):
    """Response containing a decoded P25 message."""
    timestamp: float
    opcode: int
    opcodeName: str
    nac: int | None = None
    summary: str


@router.get("/systems/{system_id}/messages", response_model=list[MessageResponse])
async def get_messages(
    system_id: str,
    request: Request,
    limit: int = 100,
    offset: int = 0,
) -> list[MessageResponse]:
    """Get recent decoded P25 messages from a trunking system.

    Returns decoded TSBK messages in reverse chronological order (newest first).
    Similar to the message log shown in SDRTrunk's UI.

    Args:
        system_id: Trunking system ID
        limit: Maximum number of messages to return (default 100, max 500)
        offset: Number of messages to skip (for pagination)

    Returns:
        List of decoded messages with timestamps and summaries
    """
    manager = get_trunking_manager(request)
    system = manager.get_system(system_id)
    if system is None:
        raise HTTPException(status_code=404, detail=f"System {system_id} not found")

    # Clamp limit to prevent abuse
    limit = min(limit, 500)

    messages = system.get_messages(limit=limit, offset=offset)
    return [
        MessageResponse(
            timestamp=msg.get("timestamp", 0),
            opcode=msg.get("opcode", 0),
            opcodeName=msg.get("opcode_name", ""),
            nac=msg.get("nac"),
            summary=msg.get("summary", ""),
        )
        for msg in messages
    ]


@router.delete("/systems/{system_id}/messages")
async def clear_messages(system_id: str, request: Request) -> dict[str, Any]:
    """Clear the message log for a trunking system.

    Returns:
        Number of messages cleared
    """
    manager = get_trunking_manager(request)
    system = manager.get_system(system_id)
    if system is None:
        raise HTTPException(status_code=404, detail=f"System {system_id} not found")

    count = system.clear_messages()
    return {"status": "ok", "cleared": count}


@router.get("/recipes", response_model=list[TrunkingRecipeResponse])
async def list_trunking_recipes(request: Request) -> list[TrunkingRecipeResponse]:
    """List available trunking system recipes/templates from config.

    Returns pre-defined trunking system configurations that can be used
    as templates when creating new trunking systems via the UI.
    """
    state = getattr(request.app.state, "app_state", None)
    if state is None:
        return []

    config = getattr(state, "config", None)
    if config is None:
        return []

    recipes = []
    for sys_id, sys_data in config.trunking_systems.items():
        if not isinstance(sys_data, dict):
            continue

        # Parse protocol for category
        protocol = sys_data.get("protocol", "p25_phase1")
        category = "P25 Phase II" if protocol == "p25_phase2" else "P25 Phase I"

        # Count talkgroups
        talkgroups = sys_data.get("talkgroups", {})
        tg_count = len(talkgroups) if isinstance(talkgroups, dict) else 0

        recipes.append(TrunkingRecipeResponse(
            id=sys_id,
            name=sys_data.get("name", f"System {sys_id}"),
            description=f"{category} trunking system with {tg_count} talkgroups configured",
            category=category,
            protocol=protocol,
            controlChannels=[float(f) for f in sys_data.get("control_channels", [])],
            centerHz=float(sys_data.get("center_hz", 851_000_000)),
            sampleRate=int(sys_data.get("sample_rate", 8_000_000)),
            gain=sys_data.get("gain"),
            talkgroupCount=tg_count,
        ))

    return recipes


# ============================================================================
# Hunt Mode Control Endpoints
# ============================================================================

class HuntModeRequest(BaseModel):
    """Request to set hunt mode."""
    mode: str = Field(..., description="Hunt mode: auto, manual, or scan_once")
    lockedFrequency: float | None = Field(None, description="Frequency to lock to (for manual mode)")


class ControlChannelResponse(BaseModel):
    """Control channel info."""
    frequencyHz: float
    enabled: bool
    isCurrent: bool
    isLocked: bool
    snrDb: float | None = None
    powerDb: float | None = None
    syncDetected: bool = False
    measurementTime: float | None = None


class ChannelEnabledRequest(BaseModel):
    """Request to enable/disable a channel."""
    enabled: bool = Field(..., description="Whether to enable the channel")


@router.get("/systems/{system_id}/hunt-mode")
async def get_hunt_mode(request: Request, system_id: str) -> dict[str, Any]:
    """Get the current hunt mode for a trunking system."""
    manager = get_trunking_manager(request)
    system = manager.get_system(system_id)
    if system is None:
        raise HTTPException(status_code=404, detail=f"System {system_id} not found")

    return {
        "mode": system.get_hunt_mode().value,
        "lockedFrequencyHz": system.get_locked_frequency(),
    }


@router.patch("/systems/{system_id}/hunt-mode")
async def set_hunt_mode(
    request: Request,
    system_id: str,
    req: HuntModeRequest,
) -> dict[str, Any]:
    """Set the hunt mode for a trunking system.

    Hunt modes:
    - auto: Hunt continuously, roam if better channel found
    - manual: Lock to specified channel, no hunting ever
    - scan_once: Scan all channels once, lock to best, stay there
    """
    manager = get_trunking_manager(request)
    system = manager.get_system(system_id)
    if system is None:
        raise HTTPException(status_code=404, detail=f"System {system_id} not found")

    try:
        mode = HuntMode(req.mode)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid hunt mode: {req.mode}. Use 'auto', 'manual', or 'scan_once'"
        )

    try:
        system.set_hunt_mode(mode, req.lockedFrequency)
        return {
            "status": "ok",
            "mode": mode.value,
            "lockedFrequencyHz": req.lockedFrequency,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/systems/{system_id}/channels", response_model=list[ControlChannelResponse])
async def get_control_channels(request: Request, system_id: str) -> list[ControlChannelResponse]:
    """Get all control channels with their current status."""
    manager = get_trunking_manager(request)
    system = manager.get_system(system_id)
    if system is None:
        raise HTTPException(status_code=404, detail=f"System {system_id} not found")

    channels = system.get_control_channels_info()
    return [
        ControlChannelResponse(
            frequencyHz=ch["frequencyHz"],
            enabled=ch["enabled"],
            isCurrent=ch["isCurrent"],
            isLocked=ch["isLocked"],
            snrDb=ch["snrDb"],
            powerDb=ch["powerDb"],
            syncDetected=ch["syncDetected"],
            measurementTime=ch["measurementTime"],
        )
        for ch in channels
    ]


@router.patch("/systems/{system_id}/channels/{freq_mhz}/enabled")
async def set_channel_enabled(
    request: Request,
    system_id: str,
    freq_mhz: float,
    req: ChannelEnabledRequest,
) -> dict[str, Any]:
    """Enable or disable a control channel.

    Args:
        freq_mhz: Frequency in MHz (e.g., 413.45 for 413.45 MHz)
    """
    manager = get_trunking_manager(request)
    system = manager.get_system(system_id)
    if system is None:
        raise HTTPException(status_code=404, detail=f"System {system_id} not found")

    freq_hz = freq_mhz * 1_000_000

    try:
        system.set_channel_enabled(freq_hz, req.enabled)
        return {
            "status": "ok",
            "frequencyHz": freq_hz,
            "enabled": req.enabled,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/systems/{system_id}/scan")
async def trigger_scan(request: Request, system_id: str) -> dict[str, Any]:
    """Trigger an immediate scan of all control channels.

    Returns the current measurement results for all channels.
    """
    manager = get_trunking_manager(request)
    system = manager.get_system(system_id)
    if system is None:
        raise HTTPException(status_code=404, detail=f"System {system_id} not found")

    measurements = system.trigger_scan()

    # Find the best channel
    best_freq = None
    best_snr = float('-inf')
    for freq, m in measurements.items():
        if m.get("syncDetected") and m.get("snrDb", float('-inf')) > best_snr:
            best_snr = m["snrDb"]
            best_freq = freq

    return {
        "status": "ok",
        "measurements": measurements,
        "bestChannelHz": best_freq,
    }


@router.post("/systems/{system_id}/channels/{freq_mhz}/lock")
async def lock_to_channel(
    request: Request,
    system_id: str,
    freq_mhz: float,
) -> dict[str, Any]:
    """Lock to a specific control channel.

    This sets the hunt mode to MANUAL and locks to the specified frequency.

    Args:
        freq_mhz: Frequency in MHz (e.g., 413.45 for 413.45 MHz)
    """
    manager = get_trunking_manager(request)
    system = manager.get_system(system_id)
    if system is None:
        raise HTTPException(status_code=404, detail=f"System {system_id} not found")

    freq_hz = freq_mhz * 1_000_000

    try:
        system.set_hunt_mode(HuntMode.MANUAL, freq_hz)
        return {
            "status": "ok",
            "mode": "manual",
            "lockedFrequencyHz": freq_hz,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# WebSocket Endpoints
# ============================================================================

@router.websocket("/stream/{system_id}")
async def trunking_stream(websocket: WebSocket, system_id: str) -> None:
    """WebSocket stream for real-time trunking events for a specific system."""
    await websocket.accept()

    # Get manager from app state
    state = getattr(websocket.app.state, "app_state", None)
    if state is None:
        await websocket.close(code=1011, reason="AppState not initialized")
        return

    manager = getattr(state, "trunking_manager", None)
    if manager is None:
        await websocket.close(code=1011, reason="TrunkingManager not initialized")
        return

    # Check system exists
    system = manager.get_system(system_id)
    if system is None:
        await websocket.close(code=1008, reason=f"System '{system_id}' not found")
        return

    # Subscribe to events
    queue = await manager.subscribe_events()

    try:
        while True:
            event = await queue.get()

            # Filter events for this system
            if event.get("systemId") != system_id and event.get("type") != "snapshot":
                continue

            # For snapshot, filter to just this system
            if event.get("type") == "snapshot":
                event = {
                    "type": "snapshot",
                    "systems": [s for s in event.get("systems", []) if s.get("id") == system_id],
                    "activeCalls": [
                        c for c in event.get("activeCalls", [])
                        if c.get("systemId") == system_id or "systemId" not in c
                    ],
                    "messages": [
                        m for m in event.get("messages", [])
                        if m.get("systemId") == system_id
                    ],
                    "callHistory": [
                        c for c in event.get("callHistory", [])
                        if c.get("systemId") == system_id
                    ],
                }

            await websocket.send_json(event)

    except WebSocketDisconnect:
        logger.info(f"Trunking WebSocket disconnected for system {system_id}")
    except Exception as e:
        logger.error(f"Trunking WebSocket error: {e}")
    finally:
        await manager.unsubscribe_events(queue)


@router.websocket("/stream")
async def trunking_stream_all(websocket: WebSocket) -> None:
    """WebSocket stream for real-time trunking events for all systems."""
    await websocket.accept()

    # Get manager from app state
    state = getattr(websocket.app.state, "app_state", None)
    if state is None:
        await websocket.close(code=1011, reason="AppState not initialized")
        return

    manager = getattr(state, "trunking_manager", None)
    if manager is None:
        await websocket.close(code=1011, reason="TrunkingManager not initialized")
        return

    # Subscribe to events
    queue = await manager.subscribe_events()

    try:
        while True:
            event = await queue.get()
            await websocket.send_json(event)

    except WebSocketDisconnect:
        logger.info("Trunking WebSocket disconnected")
    except Exception as e:
        logger.error(f"Trunking WebSocket error: {e}")
    finally:
        await manager.unsubscribe_events(queue)


# ============================================================================
# Voice Stream Endpoints
# ============================================================================

@router.get("/systems/{system_id}/voice-streams", response_model=list[VoiceStreamResponse])
async def list_voice_streams(system_id: str, request: Request) -> list[VoiceStreamResponse]:
    """List active voice streams for a trunking system."""
    manager = get_trunking_manager(request)
    system = manager.get_system(system_id)
    if system is None:
        raise HTTPException(status_code=404, detail=f"System {system_id} not found")

    voice_channels = system.get_voice_channels()
    return [
        VoiceStreamResponse(
            id=vc.id,
            systemId=vc.cfg.system_id,
            callId=vc.cfg.call_id,
            recorderId=vc.cfg.recorder_id,
            state=vc.state,
            talkgroupId=vc.talkgroup_id,
            talkgroupName=vc.talkgroup_name,
            sourceId=vc.source_id,
            encrypted=vc.encrypted,
            startTime=vc.start_time,
            durationSeconds=vc.duration_seconds,
            silenceSeconds=vc.silence_seconds,
            audioFrameCount=vc.audio_frame_count,
            audioBytesSent=vc.audio_bytes_sent,
            subscriberCount=len(vc._audio_sinks),
            sourceLocation=LocationResponse(
                unitId=vc.source_location.unit_id,
                latitude=vc.source_location.latitude,
                longitude=vc.source_location.longitude,
                altitude=vc.source_location.altitude_m,
                speed=vc.source_location.speed_kmh,
                heading=vc.source_location.heading_deg,
                accuracy=vc.source_location.accuracy_m,
                timestamp=vc.source_location.timestamp,
                ageSeconds=vc.source_location.age_seconds(),
                source=vc.source_location.source,
            ) if vc.source_location else None,
        )
        for vc in voice_channels
    ]


@router.websocket("/stream/{system_id}/voice")
async def voice_stream_all(websocket: WebSocket, system_id: str) -> None:
    """WebSocket stream for all voice audio from a trunking system.

    Delivers multiplexed audio from all active voice channels with metadata.
    Each message is JSON with base64-encoded PCM16 audio.
    """
    await websocket.accept()

    state = getattr(websocket.app.state, "app_state", None)
    if state is None:
        await websocket.close(code=1011, reason="AppState not initialized")
        return

    manager = getattr(state, "trunking_manager", None)
    if manager is None:
        await websocket.close(code=1011, reason="TrunkingManager not initialized")
        return

    system = manager.get_system(system_id)
    if system is None:
        await websocket.close(code=1008, reason=f"System {system_id} not found")
        return

    # Subscribe to all voice channels
    subscribed_queues: list[asyncio.Queue[bytes]] = []
    subscribed_channels: list[str] = []

    async def subscribe_to_channel(voice_channel) -> None:
        """Subscribe to a voice channel's audio stream."""
        queue = await voice_channel.subscribe_audio("json")
        subscribed_queues.append(queue)
        subscribed_channels.append(voice_channel.id)
        logger.info(f"Voice stream {system_id}: Subscribed to {voice_channel.id}")

    async def unsubscribe_all() -> None:
        """Unsubscribe from all voice channels."""
        for vc in system.get_voice_channels():
            for q in subscribed_queues:
                vc.unsubscribe(q)
        subscribed_queues.clear()
        subscribed_channels.clear()

    async def poll_for_new_channels() -> None:
        """Periodically check for new voice channels."""
        while True:
            await asyncio.sleep(0.5)
            for vc in system.get_voice_channels():
                if vc.id not in subscribed_channels and vc.state == "active":
                    await subscribe_to_channel(vc)

    async def send_audio_from_queues() -> None:
        """Read from all subscribed queues and send to WebSocket."""
        bytes_sent = 0
        messages_sent = 0
        while True:
            if not subscribed_queues:
                await asyncio.sleep(0.1)
                continue

            # Wait on all queues with timeout
            for queue in list(subscribed_queues):
                try:
                    # Non-blocking check
                    try:
                        data = queue.get_nowait()
                        await websocket.send_bytes(data)
                        bytes_sent += len(data)
                        messages_sent += 1
                        if messages_sent % 10 == 0:
                            logger.info(f"Voice stream {system_id}: Sent {messages_sent} messages, {bytes_sent} bytes")
                    except asyncio.QueueEmpty:
                        pass
                except Exception as e:
                    logger.error(f"Voice stream error: {e}")

            await asyncio.sleep(0.01)  # Prevent busy loop

    try:
        # Initial subscription to existing channels
        for vc in system.get_voice_channels():
            if vc.state == "active":
                await subscribe_to_channel(vc)

        # Run both tasks concurrently
        poll_task = asyncio.create_task(poll_for_new_channels())
        send_task = asyncio.create_task(send_audio_from_queues())

        # Wait for WebSocket disconnect
        try:
            while True:
                # Keep connection alive, handle client messages
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
        except WebSocketDisconnect:
            pass

        poll_task.cancel()
        send_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await poll_task
        with contextlib.suppress(asyncio.CancelledError):
            await send_task

    except WebSocketDisconnect:
        logger.info(f"Voice stream WebSocket disconnected for {system_id}")
    except Exception as e:
        logger.error(f"Voice stream WebSocket error: {e}")
    finally:
        await unsubscribe_all()


@router.websocket("/stream/{system_id}/voice/{stream_id}")
async def voice_stream_single(websocket: WebSocket, system_id: str, stream_id: str) -> None:
    """WebSocket stream for a single voice channel's audio.

    Delivers audio from one voice channel with metadata.
    Each message is JSON with base64-encoded PCM16 audio.
    """
    await websocket.accept()

    state = getattr(websocket.app.state, "app_state", None)
    if state is None:
        await websocket.close(code=1011, reason="AppState not initialized")
        return

    manager = getattr(state, "trunking_manager", None)
    if manager is None:
        await websocket.close(code=1011, reason="TrunkingManager not initialized")
        return

    system = manager.get_system(system_id)
    if system is None:
        await websocket.close(code=1008, reason=f"System {system_id} not found")
        return

    voice_channel = system.get_voice_channel(stream_id)
    if voice_channel is None:
        await websocket.close(code=1008, reason=f"Voice stream {stream_id} not found")
        return

    # Subscribe to audio
    queue = await voice_channel.subscribe_audio("json")

    try:
        while True:
            try:
                data = await asyncio.wait_for(queue.get(), timeout=1.0)
                await websocket.send_bytes(data)
            except asyncio.TimeoutError:
                # Check if channel still exists
                if voice_channel.state == "ended":
                    await websocket.send_json({"type": "ended", "streamId": stream_id})
                    break

    except WebSocketDisconnect:
        logger.info(f"Voice stream WebSocket disconnected for {stream_id}")
    except Exception as e:
        logger.error(f"Voice stream WebSocket error: {e}")
    finally:
        voice_channel.unsubscribe(queue)


@router.get("/stream/{system_id}/voice/{stream_id}.pcm")
async def voice_stream_pcm(request: Request, system_id: str, stream_id: str) -> StreamingResponse:
    """HTTP streaming endpoint for raw PCM audio.

    Returns continuous PCM16 audio at 48kHz mono.
    Designed for piping to Whisper or other audio processing tools.

    Example:
        curl -s "http://localhost:8087/api/v1/trunking/stream/psern/voice/vr0.pcm" | \\
            whisper --model medium --language en -
    """
    state = getattr(request.app.state, "app_state", None)
    if state is None:
        raise HTTPException(status_code=500, detail="AppState not initialized")

    manager = getattr(state, "trunking_manager", None)
    if manager is None:
        raise HTTPException(status_code=500, detail="TrunkingManager not initialized")

    system = manager.get_system(system_id)
    if system is None:
        raise HTTPException(status_code=404, detail=f"System {system_id} not found")

    voice_channel = system.get_voice_channel(stream_id)
    if voice_channel is None:
        raise HTTPException(status_code=404, detail=f"Voice stream {stream_id} not found")

    # Subscribe to raw PCM audio
    queue = await voice_channel.subscribe_audio("pcm16")

    async def generate_pcm():
        """Generator that yields raw PCM bytes."""
        try:
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=5.0)
                    yield data
                except asyncio.TimeoutError:
                    if voice_channel.state == "ended":
                        break
                    # Send silence to keep stream alive
                    yield b"\x00" * 960  # 10ms of silence at 48kHz mono PCM16
        finally:
            voice_channel.unsubscribe(queue)

    return StreamingResponse(
        generate_pcm(),
        media_type="audio/x-pcm",
        headers={
            "X-Audio-Sample-Rate": str(voice_channel.cfg.output_rate),
            "X-Audio-Channels": "1",
            "X-Audio-Format": "pcm16",
            "X-Stream-Id": stream_id,
            "X-Talkgroup-Id": str(voice_channel.talkgroup_id),
            "X-Talkgroup-Name": voice_channel.talkgroup_name,
        },
    )
