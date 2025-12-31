"""MCP (Model Context Protocol) server for WaveCap-SDR.

Provides an SSE-based MCP endpoint that exposes SDR control tools to AI assistants.
Secured via API key authentication.

MCP Protocol Reference: https://modelcontextprotocol.io/
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Coroutine, TypeVar, cast

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

if TYPE_CHECKING:
    from wavecapsdr.state import AppState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp", tags=["mcp"])


# -----------------------------------------------------------------------------
# MCP Protocol Types
# -----------------------------------------------------------------------------


class MCPToolInput(BaseModel):
    """Input schema for an MCP tool."""

    type: str = "object"
    properties: dict[str, Any] = {}
    required: list[str] = []


@dataclass
class MCPTool:
    """Definition of an MCP tool."""

    name: str
    description: str
    input_schema: MCPToolInput
    handler: "ToolHandler"


class MCPRequest(BaseModel):
    """JSON-RPC 2.0 request for MCP."""

    jsonrpc: str = "2.0"
    id: int | str | None = None
    method: str
    params: dict[str, Any] = {}


class MCPToolCallParams(BaseModel):
    """Parameters for tools/call method."""

    name: str
    arguments: dict[str, Any] = {}


# -----------------------------------------------------------------------------
# Tool Registry
# -----------------------------------------------------------------------------

TOOLS: dict[str, MCPTool] = {}

ToolHandler = Callable[["AppState", dict[str, Any]], Coroutine[Any, Any, Any]]
F = TypeVar("F", bound=ToolHandler)


def register_tool(
    name: str,
    description: str,
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
) -> Callable[[F], F]:
    """Decorator to register an MCP tool handler."""

    def decorator(func: F) -> F:
        TOOLS[name] = MCPTool(
            name=name,
            description=description,
            input_schema=MCPToolInput(
                properties=properties or {},
                required=required or [],
            ),
            handler=func,
        )
        return func

    return decorator


# -----------------------------------------------------------------------------
# Device Tools
# -----------------------------------------------------------------------------


@register_tool(
    name="list_devices",
    description="List available SDR devices with their capabilities (frequency range, sample rates, gains)",
)
async def list_devices(state: AppState, args: dict[str, Any]) -> dict[str, Any]:
    """List available SDR devices."""
    devices = state.driver.enumerate()
    return {
        "devices": [
            {
                "id": d.id,
                "name": state.config.device_names.get(d.id, d.label),
                "label": d.label,
                "driver": d.driver,
                "freq_range": {"min": d.freq_min_hz, "max": d.freq_max_hz},
                "sample_rates": d.sample_rates,
                "gains": d.gains,
                "antennas": d.antennas,
            }
            for d in devices
        ]
    }


@register_tool(
    name="refresh_devices",
    description="Rescan for connected SDR devices",
)
async def refresh_devices(state: AppState, args: dict[str, Any]) -> dict[str, Any]:
    """Refresh device list."""
    # The driver re-enumerates on each call
    devices = list(state.driver.enumerate())
    return {"message": f"Found {len(devices)} device(s)", "count": len(devices)}


@register_tool(
    name="get_device_health",
    description="Check SDRplay API service health (Linux/macOS only)",
)
async def get_device_health(state: AppState, args: dict[str, Any]) -> dict[str, Any]:
    """Check SDRplay service health."""
    from wavecapsdr.sdrplay_recovery import get_recovery

    recovery = get_recovery()
    stats = recovery.stats
    return {
        "sdrplay": {
            "enabled": recovery.enabled,
            "cooldown_seconds": recovery.cooldown_seconds,
            "max_restarts_per_hour": recovery.max_restarts_per_hour,
            "recovery_count": stats.recovery_count,
            "recovery_failures": stats.recovery_failures,
            "last_recovery_attempt": stats.last_recovery_attempt,
            "last_recovery_success": stats.last_recovery_success,
            "last_error": stats.last_error,
        }
    }


# -----------------------------------------------------------------------------
# Capture Tools
# -----------------------------------------------------------------------------


@register_tool(
    name="list_captures",
    description="List all active RF captures with their state and settings",
)
async def list_captures(state: AppState, args: dict[str, Any]) -> dict[str, Any]:
    """List all captures."""
    captures = state.captures.list_captures()
    return {
        "captures": [
            {
                "id": c.cfg.id,
                "state": c.state,
                "device_id": c.cfg.device_id,
                "center_hz": c.cfg.center_hz,
                "sample_rate": c.cfg.sample_rate,
                "gain": c.cfg.gain,
                "bandwidth": c.cfg.bandwidth,
                "channel_count": len(state.captures.list_channels(c.cfg.id)),
            }
            for c in captures
        ]
    }


@register_tool(
    name="create_capture",
    description="Create a new RF capture on an SDR device",
    properties={
        "device_id": {
            "type": "string",
            "description": "Device ID to use (optional, uses first available if not specified)",
        },
        "center_hz": {
            "type": "number",
            "description": "Center frequency in Hz (e.g., 162475000 for 162.475 MHz)",
        },
        "sample_rate": {
            "type": "integer",
            "description": "Sample rate in Hz (e.g., 2400000 for 2.4 MHz)",
        },
        "gain": {
            "type": "number",
            "description": "RF gain in dB (optional)",
        },
        "bandwidth": {
            "type": "number",
            "description": "Filter bandwidth in Hz (optional)",
        },
        "preset": {
            "type": "string",
            "description": "Name of preset to use (overrides other settings)",
        },
    },
    required=["center_hz", "sample_rate"],
)
async def create_capture(state: AppState, args: dict[str, Any]) -> dict[str, Any]:
    """Create a new capture."""
    device_id = args.get("device_id")
    if device_id is None:
        devices = list(state.driver.enumerate())
        if not devices:
            raise ValueError("No SDR devices available")
        device_id = devices[0].id

    capture = state.captures.create_capture(
        device_id=device_id,
        center_hz=args["center_hz"],
        sample_rate=args["sample_rate"],
        gain=args.get("gain"),
        bandwidth=args.get("bandwidth"),
    )

    return {
        "id": capture.cfg.id,
        "state": capture.state,
        "center_hz": capture.cfg.center_hz,
        "sample_rate": capture.cfg.sample_rate,
    }


@register_tool(
    name="get_capture",
    description="Get details of a specific capture",
    properties={
        "capture_id": {"type": "string", "description": "Capture ID"},
    },
    required=["capture_id"],
)
async def get_capture(state: AppState, args: dict[str, Any]) -> dict[str, Any]:
    """Get capture details."""
    capture = state.captures.get_capture(args["capture_id"])
    if not capture:
        raise ValueError(f"Capture not found: {args['capture_id']}")

    return {
        "id": capture.cfg.id,
        "state": capture.state,
        "device_id": capture.cfg.device_id,
        "center_hz": capture.cfg.center_hz,
        "sample_rate": capture.cfg.sample_rate,
        "gain": capture.cfg.gain,
        "bandwidth": capture.cfg.bandwidth,
        "channels": [
            {
                "id": ch.cfg.id,
                "name": ch.cfg.name,
                "offset_hz": ch.cfg.offset_hz,
                "mode": ch.cfg.mode,
                "state": ch.state,
            }
            for ch in state.captures.list_channels(capture.cfg.id)
        ],
    }


@register_tool(
    name="start_capture",
    description="Start an RF capture (begin receiving samples from the SDR)",
    properties={
        "capture_id": {"type": "string", "description": "Capture ID to start"},
    },
    required=["capture_id"],
)
async def start_capture(state: AppState, args: dict[str, Any]) -> dict[str, Any]:
    """Start a capture."""
    capture = state.captures.get_capture(args["capture_id"])
    if not capture:
        raise ValueError(f"Capture not found: {args['capture_id']}")

    capture.start()
    return {"id": capture.cfg.id, "state": capture.state}


@register_tool(
    name="stop_capture",
    description="Stop an RF capture",
    properties={
        "capture_id": {"type": "string", "description": "Capture ID to stop"},
    },
    required=["capture_id"],
)
async def stop_capture(state: AppState, args: dict[str, Any]) -> dict[str, Any]:
    """Stop a capture."""
    capture = state.captures.get_capture(args["capture_id"])
    if not capture:
        raise ValueError(f"Capture not found: {args['capture_id']}")

    await capture.stop()
    return {"id": capture.cfg.id, "state": capture.state}


@register_tool(
    name="update_capture",
    description="Update capture settings (gain, frequency, bandwidth)",
    properties={
        "capture_id": {"type": "string", "description": "Capture ID"},
        "center_hz": {"type": "number", "description": "New center frequency in Hz"},
        "gain": {"type": "number", "description": "New RF gain in dB"},
        "bandwidth": {"type": "number", "description": "New bandwidth in Hz"},
    },
    required=["capture_id"],
)
async def update_capture(state: AppState, args: dict[str, Any]) -> dict[str, Any]:
    """Update capture settings."""
    capture = state.captures.get_capture(args["capture_id"])
    if not capture:
        raise ValueError(f"Capture not found: {args['capture_id']}")

    await capture.reconfigure(
        center_hz=args.get("center_hz"),
        gain=args.get("gain"),
        bandwidth=args.get("bandwidth"),
    )

    return {
        "id": capture.cfg.id,
        "center_hz": capture.cfg.center_hz,
        "gain": capture.cfg.gain,
        "bandwidth": capture.cfg.bandwidth,
    }


# -----------------------------------------------------------------------------
# Channel Tools
# -----------------------------------------------------------------------------


@register_tool(
    name="list_channels",
    description="List all demodulation channels for a capture",
    properties={
        "capture_id": {"type": "string", "description": "Capture ID"},
    },
    required=["capture_id"],
)
async def list_channels(state: AppState, args: dict[str, Any]) -> dict[str, Any]:
    """List channels for a capture."""
    capture = state.captures.get_capture(args["capture_id"])
    if not capture:
        raise ValueError(f"Capture not found: {args['capture_id']}")

    return {
        "channels": [
            {
                "id": ch.cfg.id,
                "name": ch.cfg.name,
                "offset_hz": ch.cfg.offset_hz,
                "mode": ch.cfg.mode,
                "squelch_db": ch.cfg.squelch_db,
                "state": ch.state,
                "frequency_hz": capture.cfg.center_hz + ch.cfg.offset_hz,
            }
            for ch in state.captures.list_channels(capture.cfg.id)
        ]
    }


@register_tool(
    name="create_channel",
    description="Create a new demodulation channel",
    properties={
        "capture_id": {"type": "string", "description": "Capture ID to add channel to"},
        "offset_hz": {
            "type": "number",
            "description": "Frequency offset from center in Hz",
        },
        "mode": {
            "type": "string",
            "description": "Demodulation mode: nbfm, wbfm, am, usb, lsb, raw, p25, dmr",
        },
        "name": {"type": "string", "description": "Display name for the channel"},
        "squelch_db": {
            "type": "number",
            "description": "Squelch threshold in dB (default: -60)",
        },
    },
    required=["capture_id", "offset_hz", "mode"],
)
async def create_channel(state: AppState, args: dict[str, Any]) -> dict[str, Any]:
    """Create a new channel."""
    capture = state.captures.get_capture(args["capture_id"])
    if not capture:
        raise ValueError(f"Capture not found: {args['capture_id']}")

    channel = state.captures.create_channel(
        cid=capture.cfg.id,
        mode=args["mode"],
        offset_hz=args["offset_hz"],
        squelch_db=args.get("squelch_db", -60),
    )
    if "name" in args:
        channel.cfg.name = args["name"]

    return {
        "id": channel.cfg.id,
        "name": channel.cfg.name,
        "offset_hz": channel.cfg.offset_hz,
        "mode": channel.cfg.mode,
        "frequency_hz": capture.cfg.center_hz + channel.cfg.offset_hz,
    }


@register_tool(
    name="update_channel",
    description="Update channel settings",
    properties={
        "channel_id": {"type": "string", "description": "Channel ID"},
        "squelch_db": {"type": "number", "description": "New squelch threshold in dB"},
        "name": {"type": "string", "description": "New display name"},
        "offset_hz": {"type": "number", "description": "New frequency offset in Hz"},
    },
    required=["channel_id"],
)
async def update_channel(state: AppState, args: dict[str, Any]) -> dict[str, Any]:
    """Update channel settings."""
    channel = state.captures.get_channel(args["channel_id"])
    if not channel:
        raise ValueError(f"Channel not found: {args['channel_id']}")

    if "squelch_db" in args:
        channel.cfg.squelch_db = args["squelch_db"]
    if "name" in args:
        channel.cfg.name = args["name"]
    if "offset_hz" in args:
        channel.cfg.offset_hz = args["offset_hz"]

    return {
        "id": channel.cfg.id,
        "name": channel.cfg.name,
        "squelch_db": channel.cfg.squelch_db,
        "offset_hz": channel.cfg.offset_hz,
    }


@register_tool(
    name="delete_channel",
    description="Delete a demodulation channel",
    properties={
        "channel_id": {"type": "string", "description": "Channel ID to delete"},
    },
    required=["channel_id"],
)
async def delete_channel(state: AppState, args: dict[str, Any]) -> dict[str, Any]:
    """Delete a channel."""
    channel = state.captures.get_channel(args["channel_id"])
    if not channel:
        raise ValueError(f"Channel not found: {args['channel_id']}")

    state.captures.delete_channel(args["channel_id"])

    return {"deleted": args["channel_id"]}


@register_tool(
    name="get_channel_metrics",
    description="Get signal metrics for a channel (RSSI, SNR, S-meter)",
    properties={
        "channel_id": {"type": "string", "description": "Channel ID"},
    },
    required=["channel_id"],
)
async def get_channel_metrics(state: AppState, args: dict[str, Any]) -> dict[str, Any]:
    """Get channel signal metrics."""
    channel = state.captures.get_channel(args["channel_id"])
    if not channel:
        raise ValueError(f"Channel not found: {args['channel_id']}")

    return {
        "channel_id": args["channel_id"],
        "rssi_dbfs": channel.rssi_db,
        "snr_db": channel.snr_db,
        "s_meter": None,
        "noise_floor_dbfs": None,
    }


# -----------------------------------------------------------------------------
# Trunking Tools
# -----------------------------------------------------------------------------


@register_tool(
    name="list_trunking_systems",
    description="List configured P25 trunking systems",
)
async def list_trunking_systems(state: AppState, args: dict[str, Any]) -> dict[str, Any]:
    """List trunking systems."""
    systems = state.trunking_manager.list_systems()
    return {"systems": [s.to_dict() for s in systems]}


@register_tool(
    name="start_trunking",
    description="Start a P25 trunking system",
    properties={
        "system_id": {"type": "string", "description": "Trunking system ID"},
    },
    required=["system_id"],
)
async def start_trunking(state: AppState, args: dict[str, Any]) -> dict[str, Any]:
    """Start a trunking system."""
    await state.trunking_manager.start_system(args["system_id"])
    system = state.trunking_manager.get_system(args["system_id"])
    return {"id": args["system_id"], "state": system.state.value if system else "unknown"}


@register_tool(
    name="stop_trunking",
    description="Stop a P25 trunking system",
    properties={
        "system_id": {"type": "string", "description": "Trunking system ID"},
    },
    required=["system_id"],
)
async def stop_trunking(state: AppState, args: dict[str, Any]) -> dict[str, Any]:
    """Stop a trunking system."""
    await state.trunking_manager.stop_system(args["system_id"])
    system = state.trunking_manager.get_system(args["system_id"])
    return {"id": args["system_id"], "state": system.state.value if system else "unknown"}


@register_tool(
    name="get_active_calls",
    description="Get active voice calls on a trunking system",
    properties={
        "system_id": {"type": "string", "description": "Trunking system ID"},
    },
    required=["system_id"],
)
async def get_active_calls(state: AppState, args: dict[str, Any]) -> dict[str, Any]:
    """Get active calls on a trunking system."""
    system = state.trunking_manager.get_system(args["system_id"])
    if not system:
        raise ValueError(f"Trunking system not found: {args['system_id']}")

    calls = system.get_active_calls()
    return {
        "calls": [
            {
                "id": c.id,
                "talkgroup_id": c.talkgroup_id,
                "talkgroup_name": c.talkgroup_name,
                "frequency_hz": c.frequency_hz,
                "encrypted": c.encrypted,
                "state": c.state,
            }
            for c in calls
        ]
    }


@register_tool(
    name="get_talkgroups",
    description="Get talkgroups for a trunking system with aliases",
    properties={
        "system_id": {"type": "string", "description": "Trunking system ID"},
    },
    required=["system_id"],
)
async def get_talkgroups(state: AppState, args: dict[str, Any]) -> dict[str, Any]:
    """Get talkgroups for a trunking system."""
    system = state.trunking_manager.get_system(args["system_id"])
    if not system:
        raise ValueError(f"Trunking system not found: {args['system_id']}")

    talkgroups = system.cfg.talkgroups
    return {
        "talkgroups": [
            {
                "tgid": tg.tgid,
                "name": tg.name,
                "alpha_tag": tg.alpha_tag,
                "category": tg.category,
                "priority": tg.priority,
                "record": tg.record,
                "monitor": tg.monitor,
            }
            for tg in talkgroups.values()
        ]
    }


# -----------------------------------------------------------------------------
# Utility Tools
# -----------------------------------------------------------------------------


@register_tool(
    name="get_recipes",
    description="List available capture recipes/templates",
)
async def get_recipes(state: AppState, args: dict[str, Any]) -> dict[str, Any]:
    """List available recipes."""
    return {
        "recipes": [
            {
                "id": name,
                "name": recipe.name,
                "description": recipe.description,
                "category": recipe.category,
                "center_hz": recipe.center_hz,
                "sample_rate": recipe.sample_rate,
                "channel_count": len(recipe.channels),
            }
            for name, recipe in state.config.recipes.items()
        ]
    }


@register_tool(
    name="identify_frequency",
    description="Identify a radio service by frequency",
    properties={
        "frequency_hz": {
            "type": "number",
            "description": "Frequency in Hz to identify",
        },
    },
    required=["frequency_hz"],
)
async def identify_frequency(state: AppState, args: dict[str, Any]) -> dict[str, Any]:
    """Identify a frequency."""
    from wavecapsdr.frequency_namer import get_frequency_namer

    freq_hz = args["frequency_hz"]
    result = get_frequency_namer().identify_frequency(freq_hz)

    return {
        "frequency_hz": freq_hz,
        "frequency_mhz": freq_hz / 1_000_000,
        "service": result.service_type if result else None,
        "description": result.description if result else None,
        "band": result.band_name if result else None,
    }


@register_tool(
    name="get_system_health",
    description="Get overall system health and status",
)
async def get_system_health(state: AppState, args: dict[str, Any]) -> dict[str, Any]:
    """Get system health."""
    captures = state.captures.list_captures()
    devices = list(state.driver.enumerate())

    return {
        "status": "ok",
        "devices_available": len(devices),
        "captures_active": len([c for c in captures if c.state == "running"]),
        "captures_total": len(captures),
        "trunking_systems": len(state.trunking_manager.list_systems()),
    }


# -----------------------------------------------------------------------------
# MCP Protocol Handler
# -----------------------------------------------------------------------------


def get_app_state(request: Request) -> AppState:
    """Dependency to get AppState from request."""
    return cast(AppState, request.app.state.app_state)


def check_mcp_auth(
    request: Request,
    x_mcp_api_key: str | None = Header(None, alias="X-MCP-API-Key"),
) -> None:
    """Check MCP API key authentication."""
    state = cast(AppState, request.app.state.app_state)

    if not state.config.mcp.enabled:
        raise HTTPException(status_code=404, detail="MCP endpoint not enabled")

    expected_key = state.config.mcp.api_key
    if expected_key and x_mcp_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


async def handle_mcp_request(state: AppState, request_data: dict[str, Any]) -> dict[str, Any]:
    """Handle a single MCP JSON-RPC request."""
    try:
        req = MCPRequest(**request_data)
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": request_data.get("id"),
            "error": {"code": -32600, "message": f"Invalid request: {e}"},
        }

    try:
        if req.method == "initialize":
            # MCP initialization
            return {
                "jsonrpc": "2.0",
                "id": req.id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "wavecap-sdr",
                        "version": "1.0.0",
                    },
                },
            }

        elif req.method == "tools/list":
            # List available tools
            tools_list = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": {
                        "type": tool.input_schema.type,
                        "properties": tool.input_schema.properties,
                        "required": tool.input_schema.required,
                    },
                }
                for tool in TOOLS.values()
            ]
            return {
                "jsonrpc": "2.0",
                "id": req.id,
                "result": {"tools": tools_list},
            }

        elif req.method == "tools/call":
            # Call a tool
            params = MCPToolCallParams(**req.params)
            tool = TOOLS.get(params.name)
            if not tool:
                return {
                    "jsonrpc": "2.0",
                    "id": req.id,
                    "error": {"code": -32601, "message": f"Unknown tool: {params.name}"},
                }

            try:
                result = await tool.handler(state, params.arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": req.id,
                    "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]},
                }
            except Exception as e:
                logger.exception(f"Tool {params.name} failed")
                return {
                    "jsonrpc": "2.0",
                    "id": req.id,
                    "error": {"code": -32000, "message": str(e)},
                }

        else:
            return {
                "jsonrpc": "2.0",
                "id": req.id,
                "error": {"code": -32601, "message": f"Unknown method: {req.method}"},
            }

    except Exception as e:
        logger.exception("MCP request handling failed")
        return {
            "jsonrpc": "2.0",
            "id": req.id,
            "error": {"code": -32603, "message": f"Internal error: {e}"},
        }


@router.get("")
async def mcp_sse_endpoint(
    request: Request,
    _auth: None = Depends(check_mcp_auth),
    state: AppState = Depends(get_app_state),
) -> StreamingResponse:
    """SSE endpoint for MCP protocol.

    This endpoint establishes a Server-Sent Events connection for MCP communication.
    The client sends requests via the /message endpoint and receives responses here.
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        # Send initial connection event
        yield f"event: open\ndata: {json.dumps({'status': 'connected'})}\n\n"

        # Keep connection alive with periodic pings
        import asyncio

        while True:
            await asyncio.sleep(30)
            yield f"event: ping\ndata: {json.dumps({'time': 'ping'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/message")
async def mcp_message_endpoint(
    request: Request,
    _auth: None = Depends(check_mcp_auth),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    """HTTP endpoint for MCP JSON-RPC messages.

    Alternative to SSE for clients that prefer request/response pattern.
    """
    body = await request.json()
    response = await handle_mcp_request(state, body)
    return response
