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
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Request
from pydantic import BaseModel, Field

from wavecapsdr.trunking import (
    TrunkingManager,
    TrunkingSystem,
    TrunkingSystemConfig,
    TalkgroupConfig,
    TrunkingProtocol,
)
from wavecapsdr.decoders.voice import VoiceDecoder

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trunking", tags=["trunking"])


# ============================================================================
# Pydantic Models for API
# ============================================================================

class TalkgroupRequest(BaseModel):
    """Request to create/update a talkgroup."""
    tgid: int = Field(..., description="Talkgroup ID (decimal)")
    name: str = Field(..., description="Human-readable name")
    alpha_tag: Optional[str] = Field(None, description="Short identifier")
    category: Optional[str] = Field(None, description="Category for grouping")
    priority: int = Field(5, ge=1, le=10, description="Priority (1=highest, 10=lowest)")
    record: bool = Field(True, description="Whether to record calls")
    monitor: bool = Field(True, description="Whether to stream audio live")


class CreateSystemRequest(BaseModel):
    """Request to create a new trunking system."""
    id: str = Field(..., description="Unique system identifier")
    name: str = Field(..., description="Human-readable system name")
    protocol: str = Field("p25_phase1", description="Protocol: p25_phase1 or p25_phase2")
    control_channels: List[float] = Field(..., description="Control channel frequencies (Hz)")
    center_hz: float = Field(..., description="SDR center frequency (Hz)")
    sample_rate: int = Field(8_000_000, description="SDR sample rate (Hz)")
    device_id: Optional[str] = Field(None, description="SoapySDR device string")
    max_voice_recorders: int = Field(4, ge=1, le=16, description="Maximum concurrent recordings")
    recording_path: Optional[str] = Field(None, description="Path for audio file storage")
    record_unknown: bool = Field(False, description="Record unknown talkgroups")
    talkgroups: Optional[Dict[str, TalkgroupRequest]] = Field(
        None, description="Initial talkgroup configuration"
    )


class SystemResponse(BaseModel):
    """Response containing system details."""
    id: str
    name: str
    protocol: str
    deviceId: Optional[str]
    state: str
    controlChannelState: str
    controlChannelFreqHz: Optional[float]
    nac: Optional[int]
    systemId: Optional[int]
    rfssId: Optional[int]
    siteId: Optional[int]
    decodeRate: float
    activeCalls: int
    stats: Dict[str, Any]


class ActiveCallResponse(BaseModel):
    """Response containing active call details."""
    id: str
    talkgroupId: int
    talkgroupName: str
    sourceId: Optional[int]
    frequencyHz: float
    channelId: int
    state: str
    startTime: float
    lastActivityTime: float
    encrypted: bool
    audioFrames: int
    durationSeconds: float


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
    imbe: Dict[str, Any]
    ambe2: Dict[str, Any]
    anyAvailable: bool


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
        nac=d["nac"],
        systemId=d["systemId"],
        rfssId=d["rfssId"],
        siteId=d["siteId"],
        decodeRate=d["decodeRate"],
        activeCalls=d["activeCalls"],
        stats=d["stats"],
    )


# ============================================================================
# REST Endpoints
# ============================================================================

@router.get("/systems", response_model=List[SystemResponse])
async def list_systems(request: Request) -> List[SystemResponse]:
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

    # Build talkgroups dict
    talkgroups: Dict[int, TalkgroupConfig] = {}
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
        control_channels=req.control_channels,
        center_hz=req.center_hz,
        sample_rate=req.sample_rate,
        device_id=req.device_id or "",
        max_voice_recorders=req.max_voice_recorders,
        talkgroups=talkgroups,
        recording_path=req.recording_path or "./recordings",
        record_unknown=req.record_unknown,
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
async def delete_system(request: Request, system_id: str) -> Dict[str, str]:
    """Remove a trunking system."""
    manager = get_trunking_manager(request)

    try:
        await manager.remove_system(system_id)
        return {"status": "ok", "message": f"System '{system_id}' removed"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/systems/{system_id}/start")
async def start_system(request: Request, system_id: str) -> Dict[str, str]:
    """Start a trunking system."""
    manager = get_trunking_manager(request)

    try:
        await manager.start_system(system_id)
        return {"status": "ok", "message": f"System '{system_id}' started"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/systems/{system_id}/stop")
async def stop_system(request: Request, system_id: str) -> Dict[str, str]:
    """Stop a trunking system."""
    manager = get_trunking_manager(request)

    try:
        await manager.stop_system(system_id)
        return {"status": "ok", "message": f"System '{system_id}' stopped"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/systems/{system_id}/talkgroups", response_model=List[TalkgroupResponse])
async def get_talkgroups(request: Request, system_id: str) -> List[TalkgroupResponse]:
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
    talkgroups: List[TalkgroupRequest],
) -> Dict[str, Any]:
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


@router.get("/systems/{system_id}/calls/active", response_model=List[ActiveCallResponse])
async def get_system_active_calls(request: Request, system_id: str) -> List[ActiveCallResponse]:
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
        )
        for c in calls
    ]


@router.get("/calls", response_model=List[ActiveCallResponse])
async def get_all_active_calls(request: Request) -> List[ActiveCallResponse]:
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
                        # Would need system_id in call, for now send all
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
