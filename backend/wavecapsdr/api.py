from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

from .models import (
    DeviceModel,
    CaptureModel,
    CreateCaptureRequest,
    ChannelModel,
    CreateChannelRequest,
)
from .state import AppState


router = APIRouter()


def get_state(request: Request) -> AppState:
    state = getattr(request.app.state, "app_state", None)
    if state is None:
        raise RuntimeError("AppState not initialized")
    return state


def auth_check(request: Request, state: AppState = Depends(get_state)) -> None:
    token = state.config.server.auth_token
    if token is None:
        return
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if auth.split(" ", 1)[1] != token:
        raise HTTPException(status_code=403, detail="Invalid token")


@router.get("/devices", response_model=List[DeviceModel])
def list_devices(_: None = Depends(auth_check), state: AppState = Depends(get_state)):
    devices = state.captures.list_devices()
    return [DeviceModel(**d) for d in devices]


@router.get("/captures", response_model=List[CaptureModel])
def list_captures(_: None = Depends(auth_check), state: AppState = Depends(get_state)):
    return [
        CaptureModel(
            id=c.cfg.id,
            deviceId=c.cfg.device_id,
            state=c.state,  # type: ignore[arg-type]
            centerHz=c.cfg.center_hz,
            sampleRate=c.cfg.sample_rate,
            antenna=c.antenna,
        )
        for c in state.captures.list_captures()
    ]


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
    )
    return CaptureModel(
        id=cap.cfg.id,
        deviceId=cap.cfg.device_id,
        state=cap.state,  # type: ignore[arg-type]
        centerHz=cap.cfg.center_hz,
        sampleRate=cap.cfg.sample_rate,
        antenna=cap.antenna,
    )


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
    return CaptureModel(
        id=cap.cfg.id,
        deviceId=cap.cfg.device_id,
        state=cap.state,  # type: ignore[arg-type]
        centerHz=cap.cfg.center_hz,
        sampleRate=cap.cfg.sample_rate,
        antenna=cap.antenna,
    )


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
    return CaptureModel(
        id=cap.cfg.id,
        deviceId=cap.cfg.device_id,
        state=cap.state,  # type: ignore[arg-type]
        centerHz=cap.cfg.center_hz,
        sampleRate=cap.cfg.sample_rate,
        antenna=cap.antenna,
    )


@router.get("/captures/{cid}", response_model=CaptureModel)
def get_capture(
    cid: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    cap = state.captures.get_capture(cid)
    if cap is None:
        raise HTTPException(status_code=404, detail="Capture not found")
    return CaptureModel(
        id=cap.cfg.id,
        deviceId=cap.cfg.device_id,
        state=cap.state,  # type: ignore[arg-type]
        centerHz=cap.cfg.center_hz,
        sampleRate=cap.cfg.sample_rate,
        antenna=cap.antenna,
    )


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
    return ChannelModel(
        id=ch.cfg.id,
        captureId=ch.cfg.capture_id,
        mode=ch.cfg.mode,  # type: ignore[arg-type]
        state=ch.state,  # type: ignore[arg-type]
        offsetHz=ch.cfg.offset_hz,
        audioRate=ch.cfg.audio_rate,
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
    )


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
        try:
            while True:
                data = await q.get()
                yield data
        except Exception:
            pass
        finally:
            ch.unsubscribe(q)

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

    try:
        while True:
            data = await q.get()
            await websocket.send_bytes(data)
    except WebSocketDisconnect:
        pass
    finally:
        ch.unsubscribe(q)
