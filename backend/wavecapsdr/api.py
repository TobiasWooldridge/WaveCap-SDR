from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger(__name__)

from .models import (
    DeviceModel,
    CaptureModel,
    CreateCaptureRequest,
    UpdateCaptureRequest,
    ChannelModel,
    CreateChannelRequest,
    UpdateChannelRequest,
    RecipeModel,
    RecipeChannelModel,
)
from .state import AppState
from .frequency_namer import get_frequency_namer


router = APIRouter()


@router.get("/health")
def health_check(request: Request):
    """Comprehensive health check endpoint that tests all major API components."""
    state = getattr(request.app.state, "app_state", None)
    if state is None:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "AppState not initialized"}
        )

    health_status = {
        "status": "ok",
        "timestamp": __import__("time").time(),
        "checks": {}
    }

    try:
        # Check devices
        devices = state.captures.list_devices()
        health_status["checks"]["devices"] = {
            "status": "ok",
            "count": len(devices),
            "devices": [{"id": d["id"], "driver": d["driver"], "label": d["label"]} for d in devices]
        }
    except Exception as e:
        health_status["checks"]["devices"] = {"status": "error", "error": str(e)}
        health_status["status"] = "degraded"

    try:
        # Check captures
        captures = state.captures.list_captures()
        health_status["checks"]["captures"] = {
            "status": "ok",
            "count": len(captures),
            "captures": [{
                "id": c.cfg.id,
                "state": c.state,
                "device_id": c.cfg.device_id[:50] + "..." if len(c.cfg.device_id) > 50 else c.cfg.device_id,
                "center_hz": c.cfg.center_hz,
                "sample_rate": c.cfg.sample_rate,
                "antenna": c.antenna
            } for c in captures]
        }
    except Exception as e:
        health_status["checks"]["captures"] = {"status": "error", "error": str(e)}
        health_status["status"] = "degraded"

    try:
        # Check channels for each capture
        all_channels = []
        for cap in state.captures.list_captures():
            channels = state.captures.list_channels(cap.cfg.id)
            for ch in channels:
                all_channels.append({
                    "id": ch.cfg.id,
                    "capture_id": ch.cfg.capture_id,
                    "mode": ch.cfg.mode,
                    "state": ch.state
                })
        health_status["checks"]["channels"] = {
            "status": "ok",
            "count": len(all_channels),
            "channels": all_channels
        }
    except Exception as e:
        health_status["checks"]["channels"] = {"status": "error", "error": str(e)}
        health_status["status"] = "degraded"

    # Overall status
    if any(check.get("status") == "error" for check in health_status["checks"].values()):
        health_status["status"] = "degraded"

    status_code = 200 if health_status["status"] in ["ok", "degraded"] else 500
    return JSONResponse(status_code=status_code, content=health_status)


def get_state(request: Request) -> AppState:
    state = getattr(request.app.state, "app_state", None)
    if state is None:
        raise RuntimeError("AppState not initialized")
    return state


def _to_capture_model(cap) -> CaptureModel:
    """Helper to convert a Capture to CaptureModel consistently."""
    return CaptureModel(
        id=cap.cfg.id,
        deviceId=cap.cfg.device_id,
        state=cap.state,  # type: ignore[arg-type]
        centerHz=cap.cfg.center_hz,
        sampleRate=cap.cfg.sample_rate,
        gain=cap.cfg.gain,
        bandwidth=cap.cfg.bandwidth,
        ppm=cap.cfg.ppm,
        antenna=cap.antenna,
        deviceSettings=cap.cfg.device_settings,
        elementGains=cap.cfg.element_gains,
        streamFormat=cap.cfg.stream_format,
        dcOffsetAuto=cap.cfg.dc_offset_auto,
        iqBalanceAuto=cap.cfg.iq_balance_auto,
        errorMessage=cap.error_message,
    )


def auth_check(request: Request, state: AppState = Depends(get_state)) -> None:
    token = state.config.server.auth_token
    if token is None:
        return
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if auth.split(" ", 1)[1] != token:
        raise HTTPException(status_code=403, detail="Invalid token")


@router.get("/devices", response_model=List[DeviceModel], response_model_by_alias=False)
def list_devices(_: None = Depends(auth_check), state: AppState = Depends(get_state)):
    devices = state.captures.list_devices()
    return [DeviceModel(**d) for d in devices]


@router.get("/recipes", response_model=List[RecipeModel])
def list_recipes(_: None = Depends(auth_check), state: AppState = Depends(get_state)):
    """Get all available capture creation recipes."""
    recipes = []
    for recipe_id, recipe_cfg in state.config.recipes.items():
        channels = [
            RecipeChannelModel(
                offsetHz=ch.offset_hz,
                name=ch.name,
                mode=ch.mode,
                squelchDb=ch.squelch_db,
            )
            for ch in recipe_cfg.channels
        ]
        recipes.append(
            RecipeModel(
                id=recipe_id,
                name=recipe_cfg.name,
                description=recipe_cfg.description,
                category=recipe_cfg.category,
                centerHz=recipe_cfg.center_hz,
                sampleRate=recipe_cfg.sample_rate,
                gain=recipe_cfg.gain,
                bandwidth=recipe_cfg.bandwidth,
                channels=channels,
                allowFrequencyInput=recipe_cfg.allow_frequency_input,
                frequencyLabel=recipe_cfg.frequency_label,
            )
        )
    return recipes


@router.get("/frequency/identify")
def identify_frequency(
    frequency_hz: float,
    _: None = Depends(auth_check),
):
    """Identify a frequency and return its auto-generated name."""
    namer = get_frequency_namer()
    freq_info = namer.identify_frequency(frequency_hz)

    if freq_info:
        return {
            "frequency_hz": freq_info.frequency_hz,
            "name": freq_info.suggested_name,
            "band": freq_info.band_name,
            "description": freq_info.description,
        }

    return None


@router.get("/captures", response_model=List[CaptureModel])
def list_captures(_: None = Depends(auth_check), state: AppState = Depends(get_state)):
    return [_to_capture_model(c) for c in state.captures.list_captures()]


@router.post("/captures", response_model=CaptureModel)
def create_capture(
    req: CreateCaptureRequest,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    cap = state.captures.create_capture(
        device_id=req.deviceId,
        center_hz=req.centerHz,
        sample_rate=req.sampleRate,
        gain=req.gain,
        bandwidth=req.bandwidth,
        ppm=req.ppm,
        antenna=req.antenna,
        device_settings=req.deviceSettings,
        element_gains=req.elementGains,
        stream_format=req.streamFormat,
        dc_offset_auto=req.dcOffsetAuto,
        iq_balance_auto=req.iqBalanceAuto,
    )

    # Automatically create a default channel with offset 0 (unless disabled)
    if req.createDefaultChannel:
        state.captures.create_channel(
            cid=cap.cfg.id,
            mode="wbfm",
            offset_hz=0,
            audio_rate=state.config.stream.default_audio_rate,
            squelch_db=-60,
        )

    return _to_capture_model(cap)


@router.post("/captures/{cid}/start", response_model=CaptureModel)
async def start_capture(
    cid: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    cap = state.captures.get_capture(cid)
    if cap is None:
        raise HTTPException(status_code=404, detail="Capture not found")
    cap.start()
    # Auto-start any existing channels so playback works immediately
    for ch in state.captures.list_channels(cid):
        if ch.state != "running":
            ch.start()
    return _to_capture_model(cap)


@router.post("/captures/{cid}/stop", response_model=CaptureModel)
async def stop_capture(
    cid: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    cap = state.captures.get_capture(cid)
    if cap is None:
        raise HTTPException(status_code=404, detail="Capture not found")
    await cap.stop()
    return _to_capture_model(cap)


@router.get("/captures/{cid}", response_model=CaptureModel)
def get_capture(
    cid: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    cap = state.captures.get_capture(cid)
    if cap is None:
        raise HTTPException(status_code=404, detail="Capture not found")
    return _to_capture_model(cap)


@router.patch("/captures/{cid}", response_model=CaptureModel)
async def update_capture(
    cid: str,
    req: UpdateCaptureRequest,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    cap = state.captures.get_capture(cid)
    if cap is None:
        raise HTTPException(status_code=404, detail="Capture not found")

    # Handle device change if requested
    if req.deviceId is not None and req.deviceId != cap.cfg.device_id:
        if cap.state in ("running", "starting"):
            raise HTTPException(
                status_code=400,
                detail="Cannot change device while capture is running. Stop the capture first."
            )

        # Validate new device exists
        devices = state.captures.list_devices()
        new_device = next((d for d in devices if d["id"] == req.deviceId), None)
        if new_device is None:
            raise HTTPException(
                status_code=404,
                detail=f"Device '{req.deviceId}' not found"
            )

        # Update device in config
        cap.cfg.device_id = req.deviceId
        logger.info(f"Changed capture {cid} device to {req.deviceId}")

    # Validate changes against device constraints before applying
    devices = state.captures.list_devices()
    device_info = next((d for d in devices if d["id"] == cap.cfg.device_id), None)

    if device_info:
        # Validate frequency range
        if req.centerHz is not None:
            freq_min = device_info.get("freq_min_hz", 0)
            freq_max = device_info.get("freq_max_hz", 6e9)
            if not (freq_min <= req.centerHz <= freq_max):
                raise HTTPException(
                    status_code=400,
                    detail=f"Frequency {req.centerHz} Hz is out of range [{freq_min}, {freq_max}] for this device"
                )

        # Validate sample rate
        if req.sampleRate is not None:
            valid_rates = device_info.get("sample_rates", [])
            if valid_rates and req.sampleRate not in valid_rates:
                raise HTTPException(
                    status_code=400,
                    detail=f"Sample rate {req.sampleRate} is not supported. Valid rates: {valid_rates}"
                )

        # Validate gain range
        if req.gain is not None:
            gain_min = device_info.get("gain_min")
            gain_max = device_info.get("gain_max")
            if gain_min is not None and gain_max is not None:
                if not (gain_min <= req.gain <= gain_max):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Gain {req.gain} dB is out of range [{gain_min}, {gain_max}] for this device"
                    )

        # Validate bandwidth range
        if req.bandwidth is not None:
            bw_min = device_info.get("bandwidth_min")
            bw_max = device_info.get("bandwidth_max")
            if bw_min is not None and bw_max is not None:
                if not (bw_min <= req.bandwidth <= bw_max):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Bandwidth {req.bandwidth} Hz is out of range [{bw_min}, {bw_max}] for this device"
                    )

        # Validate PPM range
        if req.ppm is not None:
            ppm_min = device_info.get("ppm_min")
            ppm_max = device_info.get("ppm_max")
            if ppm_min is not None and ppm_max is not None:
                if not (ppm_min <= req.ppm <= ppm_max):
                    raise HTTPException(
                        status_code=400,
                        detail=f"PPM {req.ppm} is out of range [{ppm_min}, {ppm_max}] for this device"
                    )

        # Validate antenna
        if req.antenna is not None:
            valid_antennas = device_info.get("antennas", [])
            if valid_antennas and req.antenna not in valid_antennas:
                raise HTTPException(
                    status_code=400,
                    detail=f"Antenna '{req.antenna}' is not supported. Valid antennas: {valid_antennas}"
                )

    # Use reconfigure method with timeout protection
    try:
        # Add timeout to prevent hanging
        await asyncio.wait_for(
            cap.reconfigure(
                center_hz=req.centerHz,
                sample_rate=req.sampleRate,
                gain=req.gain,
                bandwidth=req.bandwidth,
                ppm=req.ppm,
                antenna=req.antenna,
                device_settings=req.deviceSettings,
                element_gains=req.elementGains,
                stream_format=req.streamFormat,
                dc_offset_auto=req.dcOffsetAuto,
                iq_balance_auto=req.iqBalanceAuto,
            ),
            timeout=30.0  # 30 second timeout for reconfiguration
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Capture reconfiguration timed out. The SDRplay service may be stuck. Try restarting the service."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reconfigure capture: {str(e)}"
        )

    # Persist changes to config file if this capture is associated with a preset
    preset_name = state.capture_presets.get(cid)
    if preset_name and state.config_path:
        preset = state.config.presets.get(preset_name)
        if preset:
            # Update the preset with new values
            if req.centerHz is not None:
                preset.center_hz = req.centerHz
            if req.sampleRate is not None:
                preset.sample_rate = req.sampleRate
            if req.gain is not None:
                preset.gain = req.gain
            if req.bandwidth is not None:
                preset.bandwidth = req.bandwidth
            if req.ppm is not None:
                preset.ppm = req.ppm
            if req.antenna is not None:
                preset.antenna = req.antenna

            # Save the updated config to file
            try:
                from .config import save_config
                save_config(state.config, state.config_path)
            except Exception as e:
                # Log error but don't fail the request - the settings are already applied in memory
                print(f"Warning: Failed to persist config changes to file: {e}")

    return _to_capture_model(cap)


@router.delete("/captures/{cid}")
async def delete_capture(
    cid: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    await state.captures.delete_capture(cid)
    return JSONResponse(status_code=204, content={})


@router.get("/captures/{cid}/channels", response_model=List[ChannelModel])
def list_channels(
    cid: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    chans = state.captures.list_channels(cid)
    return [
        ChannelModel(
            id=ch.cfg.id,
            captureId=ch.cfg.capture_id,
            mode=ch.cfg.mode,  # type: ignore[arg-type]
            state=ch.state,  # type: ignore[arg-type]
            offsetHz=ch.cfg.offset_hz,
            audioRate=ch.cfg.audio_rate,
            squelchDb=ch.cfg.squelch_db,
            name=ch.cfg.name,
            autoName=ch.cfg.auto_name,
            signalPowerDb=ch.signal_power_db,
            rssiDb=ch.rssi_db,
            snrDb=ch.snr_db,
        )
        for ch in chans
    ]


@router.post("/captures/{cid}/channels", response_model=ChannelModel)
def create_channel(
    cid: str,
    req: CreateChannelRequest,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    ch = state.captures.create_channel(
        cid=cid,
        mode=req.mode,
        offset_hz=req.offsetHz or 0.0,
        audio_rate=req.audioRate,
        squelch_db=req.squelchDb,
    )

    # Set user-provided name
    ch.cfg.name = req.name

    # Generate auto_name using frequency recognition
    cap = state.captures.get_capture(cid)
    if cap is not None:
        namer = get_frequency_namer()
        ch.cfg.auto_name = namer.suggest_channel_name(cap.cfg.center_hz, ch.cfg.offset_hz)

    # Auto-start channel if capture is already running so audio begins immediately
    if cap is not None and cap.state == "running" and ch.state != "running":
        ch.start()

    return ChannelModel(
        id=ch.cfg.id,
        captureId=ch.cfg.capture_id,
        mode=ch.cfg.mode,  # type: ignore[arg-type]
        state=ch.state,  # type: ignore[arg-type]
        offsetHz=ch.cfg.offset_hz,
        audioRate=ch.cfg.audio_rate,
        squelchDb=ch.cfg.squelch_db,
        name=ch.cfg.name,
        autoName=ch.cfg.auto_name,
        signalPowerDb=ch.signal_power_db,
        rssiDb=ch.rssi_db,
        snrDb=ch.snr_db,
    )


@router.post("/channels/{chan_id}/start", response_model=ChannelModel)
def start_channel(
    chan_id: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    ch = state.captures.get_channel(chan_id)
    if ch is None:
        raise HTTPException(status_code=404, detail="Channel not found")
    ch.start()
    return ChannelModel(
        id=ch.cfg.id,
        captureId=ch.cfg.capture_id,
        mode=ch.cfg.mode,  # type: ignore[arg-type]
        state=ch.state,  # type: ignore[arg-type]
        offsetHz=ch.cfg.offset_hz,
        audioRate=ch.cfg.audio_rate,
        squelchDb=ch.cfg.squelch_db,
        signalPowerDb=ch.signal_power_db,
        rssiDb=ch.rssi_db,
        snrDb=ch.snr_db,
    )


@router.post("/channels/{chan_id}/stop", response_model=ChannelModel)
def stop_channel(
    chan_id: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    ch = state.captures.get_channel(chan_id)
    if ch is None:
        raise HTTPException(status_code=404, detail="Channel not found")
    ch.stop()
    return ChannelModel(
        id=ch.cfg.id,
        captureId=ch.cfg.capture_id,
        mode=ch.cfg.mode,  # type: ignore[arg-type]
        state=ch.state,  # type: ignore[arg-type]
        offsetHz=ch.cfg.offset_hz,
        audioRate=ch.cfg.audio_rate,
        squelchDb=ch.cfg.squelch_db,
        signalPowerDb=ch.signal_power_db,
        rssiDb=ch.rssi_db,
        snrDb=ch.snr_db,
    )


@router.get("/channels/{chan_id}", response_model=ChannelModel)
def get_channel(
    chan_id: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    ch = state.captures.get_channel(chan_id)
    if ch is None:
        raise HTTPException(status_code=404, detail="Channel not found")
    return ChannelModel(
        id=ch.cfg.id,
        captureId=ch.cfg.capture_id,
        mode=ch.cfg.mode,  # type: ignore[arg-type]
        state=ch.state,  # type: ignore[arg-type]
        offsetHz=ch.cfg.offset_hz,
        audioRate=ch.cfg.audio_rate,
        squelchDb=ch.cfg.squelch_db,
        signalPowerDb=ch.signal_power_db,
        rssiDb=ch.rssi_db,
        snrDb=ch.snr_db,
    )


@router.patch("/channels/{chan_id}", response_model=ChannelModel)
def update_channel(
    chan_id: str,
    req: UpdateChannelRequest,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    ch = state.captures.get_channel(chan_id)
    if ch is None:
        raise HTTPException(status_code=404, detail="Channel not found")

    # Update channel configuration
    if req.mode is not None:
        ch.cfg.mode = req.mode
    if req.offsetHz is not None:
        ch.cfg.offset_hz = req.offsetHz
    if req.audioRate is not None:
        ch.cfg.audio_rate = req.audioRate
    if req.squelchDb is not None:
        ch.cfg.squelch_db = req.squelchDb

    return ChannelModel(
        id=ch.cfg.id,
        captureId=ch.cfg.capture_id,
        mode=ch.cfg.mode,  # type: ignore[arg-type]
        state=ch.state,  # type: ignore[arg-type]
        offsetHz=ch.cfg.offset_hz,
        audioRate=ch.cfg.audio_rate,
        squelchDb=ch.cfg.squelch_db,
        signalPowerDb=ch.signal_power_db,
        rssiDb=ch.rssi_db,
        snrDb=ch.snr_db,
    )


@router.delete("/channels/{chan_id}")
def delete_channel(
    chan_id: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    state.captures.delete_channel(chan_id)
    return JSONResponse(status_code=204, content={})


@router.websocket("/stream/captures/{cid}/iq")
async def stream_capture_iq(websocket: WebSocket, cid: str):
    # Auth: optional token via header or query `token`
    app_state: AppState = getattr(websocket.app.state, "app_state")
    token = app_state.config.server.auth_token
    if token is not None:
        auth = websocket.headers.get("authorization") or websocket.query_params.get("token")
        if not auth:
            await websocket.close(code=4401)
            return
        if auth.startswith("Bearer "):
            auth = auth.split(" ", 1)[1]
        if auth != token:
            await websocket.close(code=4403)
            return

    await websocket.accept()
    cap = app_state.captures.get_capture(cid)
    if cap is None:
        await websocket.close(code=4404)
        return
    q = await cap.subscribe_iq()

    try:
        while True:
            data = await q.get()
            await websocket.send_bytes(data)
    except WebSocketDisconnect:
        pass
    finally:
        cap.unsubscribe(q)


@router.websocket("/stream/captures/{cid}/spectrum")
async def stream_capture_spectrum(websocket: WebSocket, cid: str):
    """Stream FFT/spectrum data for waterfall/spectrum analyzer display.

    Only calculates FFT when there are active subscribers for efficiency.
    """
    app_state: AppState = getattr(websocket.app.state, "app_state")
    token = app_state.config.server.auth_token
    if token is not None:
        auth = websocket.headers.get("authorization") or websocket.query_params.get("token")
        if not auth:
            await websocket.close(code=4401)
            return
        if auth.startswith("Bearer "):
            auth = auth.split(" ", 1)[1]
        if auth != token:
            await websocket.close(code=4403)
            return

    await websocket.accept()
    cap = app_state.captures.get_capture(cid)
    if cap is None:
        await websocket.close(code=4404)
        return
    q = await cap.subscribe_fft()

    logger.info(f"Spectrum WebSocket stream started for capture {cid}, client={websocket.client}")
    try:
        while True:
            data = await q.get()
            # Send as JSON for easy frontend processing
            await websocket.send_json(data)
    except WebSocketDisconnect:
        logger.info(f"Spectrum WebSocket stream disconnected for capture {cid}")
    except asyncio.CancelledError:
        logger.info(f"Spectrum WebSocket stream cancelled for capture {cid}")
        raise
    except Exception as e:
        logger.error(f"Spectrum WebSocket stream error for capture {cid}: {type(e).__name__}: {e}", exc_info=True)
        raise
    finally:
        cap.unsubscribe_fft(q)
        logger.info(f"Spectrum WebSocket stream ended for capture {cid}")


@router.get("/stream/channels/{chan_id}.pcm")
async def stream_channel_http(
    request: Request,
    chan_id: str,
    format: str = "pcm16",
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    """Stream channel audio over HTTP for VLC and other players.

    Query parameters:
        format: Audio format - "pcm16" (16-bit signed PCM) or "f32" (32-bit float). Default: pcm16
    """
    # Validate format parameter
    if format not in ["pcm16", "f32"]:
        raise HTTPException(status_code=400, detail="Invalid format. Use 'pcm16' or 'f32'")

    ch = state.captures.get_channel(chan_id)
    if ch is None:
        raise HTTPException(status_code=404, detail="Channel not found")

    # Get channel config to set proper content-type header
    audio_rate = ch.cfg.audio_rate

    async def audio_generator():
        q = await ch.subscribe_audio(format=format)
        logger.info(f"HTTP stream started for channel {chan_id}, format={format}, client={request.client}")
        packet_count = 0
        try:
            while True:
                # Check disconnect every 10 packets (~200ms) even if queue has data
                if packet_count % 10 == 0:
                    if await request.is_disconnected():
                        logger.info(f"HTTP stream client disconnected for channel {chan_id}")
                        break

                # Get data with shorter timeout to check disconnect more frequently
                try:
                    data = await asyncio.wait_for(q.get(), timeout=0.5)
                    yield data
                    packet_count += 1
                except asyncio.TimeoutError:
                    # No data available, will check disconnect on next iteration
                    continue
        except asyncio.CancelledError:
            logger.info(f"HTTP stream cancelled for channel {chan_id}")
            raise  # Re-raise to properly handle cancellation
        except Exception as e:
            logger.error(f"HTTP stream error for channel {chan_id}: {type(e).__name__}: {e}", exc_info=True)
            raise  # Re-raise to notify client of error
        finally:
            ch.unsubscribe(q)
            logger.info(f"HTTP stream ended for channel {chan_id}, packets sent: {packet_count}")

    # Set appropriate content-type for raw PCM audio
    if format == "f32":
        media_type = "audio/x-raw"
    else:
        media_type = "audio/x-raw"

    return StreamingResponse(
        audio_generator(),
        media_type=media_type,
        headers={
            "Cache-Control": "no-cache",
            "X-Audio-Format": format,
            "X-Audio-Rate": str(audio_rate),
            "X-Audio-Channels": "1",
        },
    )


@router.get("/stream/channels/{chan_id}.mp3")
async def stream_channel_mp3(
    request: Request,
    chan_id: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    """Stream channel audio as MP3."""
    ch = state.captures.get_channel(chan_id)
    if ch is None:
        raise HTTPException(status_code=404, detail="Channel not found")

    async def audio_generator():
        q = await ch.subscribe_audio(format="mp3")
        logger.info(f"MP3 stream started for channel {chan_id}, client={request.client}")
        packet_count = 0
        try:
            while True:
                if packet_count % 10 == 0:
                    if await request.is_disconnected():
                        logger.info(f"MP3 stream client disconnected for channel {chan_id}")
                        break

                try:
                    data = await asyncio.wait_for(q.get(), timeout=0.5)
                    yield data
                    packet_count += 1
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            logger.info(f"MP3 stream cancelled for channel {chan_id}")
            raise
        except Exception as e:
            logger.error(f"MP3 stream error for channel {chan_id}: {type(e).__name__}: {e}", exc_info=True)
            raise
        finally:
            ch.unsubscribe(q)
            logger.info(f"MP3 stream ended for channel {chan_id}, packets sent: {packet_count}")

    return StreamingResponse(
        audio_generator(),
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-cache",
            "X-Audio-Rate": str(ch.cfg.audio_rate),
        },
    )


@router.get("/stream/channels/{chan_id}.opus")
async def stream_channel_opus(
    request: Request,
    chan_id: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    """Stream channel audio as Opus."""
    ch = state.captures.get_channel(chan_id)
    if ch is None:
        raise HTTPException(status_code=404, detail="Channel not found")

    async def audio_generator():
        q = await ch.subscribe_audio(format="opus")
        logger.info(f"Opus stream started for channel {chan_id}, client={request.client}")
        packet_count = 0
        try:
            while True:
                if packet_count % 10 == 0:
                    if await request.is_disconnected():
                        logger.info(f"Opus stream client disconnected for channel {chan_id}")
                        break

                try:
                    data = await asyncio.wait_for(q.get(), timeout=0.5)
                    yield data
                    packet_count += 1
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            logger.info(f"Opus stream cancelled for channel {chan_id}")
            raise
        except Exception as e:
            logger.error(f"Opus stream error for channel {chan_id}: {type(e).__name__}: {e}", exc_info=True)
            raise
        finally:
            ch.unsubscribe(q)
            logger.info(f"Opus stream ended for channel {chan_id}, packets sent: {packet_count}")

    return StreamingResponse(
        audio_generator(),
        media_type="audio/opus",
        headers={
            "Cache-Control": "no-cache",
            "X-Audio-Rate": str(ch.cfg.audio_rate),
        },
    )


@router.get("/stream/channels/{chan_id}.aac")
async def stream_channel_aac(
    request: Request,
    chan_id: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    """Stream channel audio as AAC."""
    ch = state.captures.get_channel(chan_id)
    if ch is None:
        raise HTTPException(status_code=404, detail="Channel not found")

    async def audio_generator():
        q = await ch.subscribe_audio(format="aac")
        logger.info(f"AAC stream started for channel {chan_id}, client={request.client}")
        packet_count = 0
        try:
            while True:
                if packet_count % 10 == 0:
                    if await request.is_disconnected():
                        logger.info(f"AAC stream client disconnected for channel {chan_id}")
                        break

                try:
                    data = await asyncio.wait_for(q.get(), timeout=0.5)
                    yield data
                    packet_count += 1
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            logger.info(f"AAC stream cancelled for channel {chan_id}")
            raise
        except Exception as e:
            logger.error(f"AAC stream error for channel {chan_id}: {type(e).__name__}: {e}", exc_info=True)
            raise
        finally:
            ch.unsubscribe(q)
            logger.info(f"AAC stream ended for channel {chan_id}, packets sent: {packet_count}")

    return StreamingResponse(
        audio_generator(),
        media_type="audio/aac",
        headers={
            "Cache-Control": "no-cache",
            "X-Audio-Rate": str(ch.cfg.audio_rate),
        },
    )


@router.websocket("/stream/channels/{chan_id}")
async def stream_channel_audio(websocket: WebSocket, chan_id: str, format: str = "pcm16"):
    """Stream channel audio with configurable format.

    Query parameters:
        format: Audio format - "pcm16" (16-bit signed PCM) or "f32" (32-bit float). Default: pcm16
    """
    app_state: AppState = getattr(websocket.app.state, "app_state")
    token = app_state.config.server.auth_token
    if token is not None:
        auth = websocket.headers.get("authorization") or websocket.query_params.get("token")
        if not auth:
            await websocket.close(code=4401)
            return
        if auth.startswith("Bearer "):
            auth = auth.split(" ", 1)[1]
        if auth != token:
            await websocket.close(code=4403)
            return

    # Validate format parameter
    format_param = websocket.query_params.get("format", "pcm16")
    if format_param not in ["pcm16", "f32"]:
        await websocket.close(code=4400)  # Bad request
        return

    await websocket.accept()
    ch = app_state.captures.get_channel(chan_id)
    if ch is None:
        await websocket.close(code=4404)
        return
    q = await ch.subscribe_audio(format=format_param)

    logger.info(f"WebSocket stream started for channel {chan_id}, format={format_param}, client={websocket.client}")
    try:
        while True:
            data = await q.get()
            await websocket.send_bytes(data)
    except WebSocketDisconnect:
        logger.info(f"WebSocket stream disconnected for channel {chan_id}")
    except asyncio.CancelledError:
        logger.info(f"WebSocket stream cancelled for channel {chan_id}")
        raise
    except Exception as e:
        logger.error(f"WebSocket stream error for channel {chan_id}: {type(e).__name__}: {e}", exc_info=True)
        raise
    finally:
        ch.unsubscribe(q)
        logger.info(f"WebSocket stream ended for channel {chan_id}")
