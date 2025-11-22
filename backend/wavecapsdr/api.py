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
    CreateScannerRequest,
    UpdateScannerRequest,
    ScannerModel,
    ScanHitModel,
    SpectrumSnapshotModel,
    ExtendedMetricsModel,
    MetricsHistoryModel,
    MetricsHistoryPoint,
)
from .state import AppState
from .frequency_namer import get_frequency_namer
from .device_namer import (
    generate_capture_name,
    get_device_nickname,
    set_device_nickname,
    get_device_shorthand,
)
from .config import save_config


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

    try:
        # Check streaming/queue stats and cleanup zombies
        total_subscribers = 0
        total_zombies_cleaned = 0
        queue_stats = []
        for cap in state.captures.list_captures():
            channels = state.captures.list_channels(cap.cfg.id)
            for ch in channels:
                # Run zombie cleanup during health check
                zombies_cleaned = ch.cleanup_zombie_subscribers()
                total_zombies_cleaned += zombies_cleaned
                # Get queue stats
                stats = ch.get_queue_stats()
                total_subscribers += stats["total_subscribers"]
                if stats["total_subscribers"] > 0 or stats["drops_since_last_log"] > 0:
                    queue_stats.append({
                        "channel_id": ch.cfg.id,
                        **stats
                    })
        health_status["checks"]["streaming"] = {
            "status": "ok",
            "total_subscribers": total_subscribers,
            "zombies_cleaned": total_zombies_cleaned,
            "channel_stats": queue_stats
        }
    except Exception as e:
        health_status["checks"]["streaming"] = {"status": "error", "error": str(e)}
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
        name=cap.cfg.name,
        autoName=cap.cfg.auto_name,
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


@router.get("/devices/{device_id}/name")
def get_device_name(device_id: str, _: None = Depends(auth_check), state: AppState = Depends(get_state)):
    """Get custom nickname for a device."""
    nickname = get_device_nickname(device_id)
    # Also get device info for shorthand fallback
    devices = state.captures.list_devices()
    device = next((d for d in devices if d["id"] == device_id), None)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    shorthand = get_device_shorthand(device_id, device["label"])
    return {
        "device_id": device_id,
        "nickname": nickname,
        "shorthand": shorthand,
        "label": device["label"]
    }


@router.patch("/devices/{device_id}/name")
def update_device_name(device_id: str, request: dict, _: None = Depends(auth_check), state: AppState = Depends(get_state)):
    """Set custom nickname for a device."""
    nickname = request.get("nickname", "")

    # Validate device exists
    devices = state.captures.list_devices()
    device = next((d for d in devices if d["id"] == device_id), None)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    # Update nickname
    set_device_nickname(device_id, nickname)

    # Save to config
    if nickname:
        state.config.device_names[device_id] = nickname
    elif device_id in state.config.device_names:
        del state.config.device_names[device_id]

    if state.config_path:
        try:
            save_config(state.config, state.config_path)
        except Exception as e:
            logger.error(f"Failed to save device nickname: {e}")

    return {"device_id": device_id, "nickname": nickname}


@router.get("/devices", response_model=List[DeviceModel], response_model_by_alias=False)
def list_devices(_: None = Depends(auth_check), state: AppState = Depends(get_state)):
    result = []
    seen_ids = set()

    # Try to enumerate devices, but don't fail if enumeration errors
    # (e.g., "Broken pipe" when devices are busy)
    try:
        devices = state.captures.list_devices()
        for d in devices:
            device_id = d["id"]
            device_label = d["label"]
            seen_ids.add(device_id)

            # Get nickname and shorthand name
            nickname = get_device_nickname(device_id)
            shorthand = get_device_shorthand(device_id, device_label)

            result.append(DeviceModel(**d, nickname=nickname, shorthand=shorthand))
    except Exception:
        # Enumeration failed (devices busy, driver error, etc.)
        # Continue to add devices from active captures below
        pass

    # Also include devices from active captures that aren't in enumeration
    # (devices in use may not show up in driver enumeration)
    for cap in state.captures.list_captures():
        device_id = cap.cfg.device_id
        if device_id not in seen_ids and cap.device is not None:
            seen_ids.add(device_id)
            # Extract driver and label from device_id string
            driver = "unknown"
            label = device_id
            for part in device_id.split(","):
                if part.startswith("driver="):
                    driver = part.split("=", 1)[1]
                elif part.startswith("label="):
                    label = part.split("=", 1)[1]

            # Try to get device info from the open device
            device_info = cap.device.info if cap.device else None
            nickname = get_device_nickname(device_id)
            shorthand = get_device_shorthand(device_id, label)

            # Create a minimal DeviceModel for the in-use device
            in_use_device = DeviceModel(
                id=device_id,
                driver=driver,
                label=label,
                freqMinHz=device_info.freq_min_hz if device_info else 0,
                freqMaxHz=device_info.freq_max_hz if device_info else 6e9,
                sampleRates=device_info.sample_rates if device_info else [],
                gains=device_info.gains if device_info else [],
                gainMin=device_info.gain_min if device_info else 0,
                gainMax=device_info.gain_max if device_info else 50,
                bandwidthMin=device_info.bandwidth_min if device_info else 0,
                bandwidthMax=device_info.bandwidth_max if device_info else 10e6,
                ppmMin=-100,
                ppmMax=100,
                antennas=device_info.antennas if device_info else [],
                nickname=nickname,
                shorthand=shorthand,
            )
            result.append(in_use_device)

    return result


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

    # Set user-provided name if given
    if req.name:
        cap.cfg.name = req.name

    # Generate auto_name using device namer
    devices = state.captures.list_devices()
    device = next((d for d in devices if d["id"] == cap.cfg.device_id), None)
    if device:
        device_nickname = get_device_nickname(cap.cfg.device_id)
        cap.cfg.auto_name = generate_capture_name(
            center_hz=req.centerHz,
            device_id=cap.cfg.device_id,
            device_label=device["label"],
            recipe_name=None,
            device_nickname=device_nickname,
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
    import traceback
    try:
        return await _update_capture_impl(cid, req, state)
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"[ERROR] update_capture failed: {e}\n{traceback.format_exc()}"
        print(error_msg, flush=True)
        # Also write to temp file for debugging
        with open("/tmp/wavecapsdr_error.log", "a") as f:
            f.write(f"\n--- {__import__('datetime').datetime.now()} ---\n{error_msg}\n")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


async def _update_capture_impl(cid: str, req: UpdateCaptureRequest, state: AppState):
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

        # Update device in config and requested_device_id (used when opening device)
        cap.cfg.device_id = req.deviceId
        cap.requested_device_id = req.deviceId
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

    # Update user-provided name if given
    if req.name is not None:
        cap.cfg.name = req.name

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

    # Regenerate auto_name if frequency or device changed
    if req.centerHz is not None or req.deviceId is not None:
        devices = state.captures.list_devices()
        device = next((d for d in devices if d["id"] == cap.cfg.device_id), None)
        if device:
            device_nickname = get_device_nickname(cap.cfg.device_id)
            cap.cfg.auto_name = generate_capture_name(
                center_hz=cap.cfg.center_hz,  # Now it's updated after reconfigure
                device_id=cap.cfg.device_id,
                device_label=device["label"],
                recipe_name=None,
                device_nickname=device_nickname,
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
            audioRmsDb=ch.audio_rms_db,
            audioPeakDb=ch.audio_peak_db,
            audioClippingCount=ch.audio_clipping_count,
            # Filter configuration
            enableDeemphasis=ch.cfg.enable_deemphasis,
            deemphasisTauUs=ch.cfg.deemphasis_tau_us,
            enableMpxFilter=ch.cfg.enable_mpx_filter,
            mpxCutoffHz=ch.cfg.mpx_cutoff_hz,
            enableFmHighpass=ch.cfg.enable_fm_highpass,
            fmHighpassHz=ch.cfg.fm_highpass_hz,
            enableFmLowpass=ch.cfg.enable_fm_lowpass,
            fmLowpassHz=ch.cfg.fm_lowpass_hz,
            enableAmHighpass=ch.cfg.enable_am_highpass,
            amHighpassHz=ch.cfg.am_highpass_hz,
            enableAmLowpass=ch.cfg.enable_am_lowpass,
            amLowpassHz=ch.cfg.am_lowpass_hz,
            enableSsbBandpass=ch.cfg.enable_ssb_bandpass,
            ssbBandpassLowHz=ch.cfg.ssb_bandpass_low_hz,
            ssbBandpassHighHz=ch.cfg.ssb_bandpass_high_hz,
            ssbMode=ch.cfg.ssb_mode,
            enableAgc=ch.cfg.enable_agc,
            agcTargetDb=ch.cfg.agc_target_db,
            agcAttackMs=ch.cfg.agc_attack_ms,
            agcReleaseMs=ch.cfg.agc_release_ms,
            enableNoiseBlanker=ch.cfg.enable_noise_blanker,
            noiseBlankerThresholdDb=ch.cfg.noise_blanker_threshold_db,
            notchFrequencies=ch.cfg.notch_frequencies,
            enableNoiseReduction=ch.cfg.enable_noise_reduction,
            noiseReductionDb=ch.cfg.noise_reduction_db,
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

    # Set notch frequencies if provided
    if req.notchFrequencies is not None:
        ch.cfg.notch_frequencies = req.notchFrequencies

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
        audioRmsDb=ch.audio_rms_db,
        audioPeakDb=ch.audio_peak_db,
        audioClippingCount=ch.audio_clipping_count,
        # Filter configuration
        enableDeemphasis=ch.cfg.enable_deemphasis,
        deemphasisTauUs=ch.cfg.deemphasis_tau_us,
        enableMpxFilter=ch.cfg.enable_mpx_filter,
        mpxCutoffHz=ch.cfg.mpx_cutoff_hz,
        enableFmHighpass=ch.cfg.enable_fm_highpass,
        fmHighpassHz=ch.cfg.fm_highpass_hz,
        enableFmLowpass=ch.cfg.enable_fm_lowpass,
        fmLowpassHz=ch.cfg.fm_lowpass_hz,
        enableAmHighpass=ch.cfg.enable_am_highpass,
        amHighpassHz=ch.cfg.am_highpass_hz,
        enableAmLowpass=ch.cfg.enable_am_lowpass,
        amLowpassHz=ch.cfg.am_lowpass_hz,
        enableSsbBandpass=ch.cfg.enable_ssb_bandpass,
        ssbBandpassLowHz=ch.cfg.ssb_bandpass_low_hz,
        ssbBandpassHighHz=ch.cfg.ssb_bandpass_high_hz,
        ssbMode=ch.cfg.ssb_mode,
        enableAgc=ch.cfg.enable_agc,
        agcTargetDb=ch.cfg.agc_target_db,
        agcAttackMs=ch.cfg.agc_attack_ms,
        agcReleaseMs=ch.cfg.agc_release_ms,
        enableNoiseBlanker=ch.cfg.enable_noise_blanker,
        noiseBlankerThresholdDb=ch.cfg.noise_blanker_threshold_db,
        notchFrequencies=ch.cfg.notch_frequencies,
        enableNoiseReduction=ch.cfg.enable_noise_reduction,
        noiseReductionDb=ch.cfg.noise_reduction_db,
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
        name=ch.cfg.name,
        autoName=ch.cfg.auto_name,
        signalPowerDb=ch.signal_power_db,
        rssiDb=ch.rssi_db,
        snrDb=ch.snr_db,
        audioRmsDb=ch.audio_rms_db,
        audioPeakDb=ch.audio_peak_db,
        audioClippingCount=ch.audio_clipping_count,
        # Filter configuration
        enableDeemphasis=ch.cfg.enable_deemphasis,
        deemphasisTauUs=ch.cfg.deemphasis_tau_us,
        enableMpxFilter=ch.cfg.enable_mpx_filter,
        mpxCutoffHz=ch.cfg.mpx_cutoff_hz,
        enableFmHighpass=ch.cfg.enable_fm_highpass,
        fmHighpassHz=ch.cfg.fm_highpass_hz,
        enableFmLowpass=ch.cfg.enable_fm_lowpass,
        fmLowpassHz=ch.cfg.fm_lowpass_hz,
        enableAmHighpass=ch.cfg.enable_am_highpass,
        amHighpassHz=ch.cfg.am_highpass_hz,
        enableAmLowpass=ch.cfg.enable_am_lowpass,
        amLowpassHz=ch.cfg.am_lowpass_hz,
        enableSsbBandpass=ch.cfg.enable_ssb_bandpass,
        ssbBandpassLowHz=ch.cfg.ssb_bandpass_low_hz,
        ssbBandpassHighHz=ch.cfg.ssb_bandpass_high_hz,
        ssbMode=ch.cfg.ssb_mode,
        enableAgc=ch.cfg.enable_agc,
        agcTargetDb=ch.cfg.agc_target_db,
        agcAttackMs=ch.cfg.agc_attack_ms,
        agcReleaseMs=ch.cfg.agc_release_ms,
        enableNoiseBlanker=ch.cfg.enable_noise_blanker,
        noiseBlankerThresholdDb=ch.cfg.noise_blanker_threshold_db,
        notchFrequencies=ch.cfg.notch_frequencies,
        enableNoiseReduction=ch.cfg.enable_noise_reduction,
        noiseReductionDb=ch.cfg.noise_reduction_db,
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
        name=ch.cfg.name,
        autoName=ch.cfg.auto_name,
        signalPowerDb=ch.signal_power_db,
        rssiDb=ch.rssi_db,
        snrDb=ch.snr_db,
        audioRmsDb=ch.audio_rms_db,
        audioPeakDb=ch.audio_peak_db,
        audioClippingCount=ch.audio_clipping_count,
        # Filter configuration
        enableDeemphasis=ch.cfg.enable_deemphasis,
        deemphasisTauUs=ch.cfg.deemphasis_tau_us,
        enableMpxFilter=ch.cfg.enable_mpx_filter,
        mpxCutoffHz=ch.cfg.mpx_cutoff_hz,
        enableFmHighpass=ch.cfg.enable_fm_highpass,
        fmHighpassHz=ch.cfg.fm_highpass_hz,
        enableFmLowpass=ch.cfg.enable_fm_lowpass,
        fmLowpassHz=ch.cfg.fm_lowpass_hz,
        enableAmHighpass=ch.cfg.enable_am_highpass,
        amHighpassHz=ch.cfg.am_highpass_hz,
        enableAmLowpass=ch.cfg.enable_am_lowpass,
        amLowpassHz=ch.cfg.am_lowpass_hz,
        enableSsbBandpass=ch.cfg.enable_ssb_bandpass,
        ssbBandpassLowHz=ch.cfg.ssb_bandpass_low_hz,
        ssbBandpassHighHz=ch.cfg.ssb_bandpass_high_hz,
        ssbMode=ch.cfg.ssb_mode,
        enableAgc=ch.cfg.enable_agc,
        agcTargetDb=ch.cfg.agc_target_db,
        agcAttackMs=ch.cfg.agc_attack_ms,
        agcReleaseMs=ch.cfg.agc_release_ms,
        enableNoiseBlanker=ch.cfg.enable_noise_blanker,
        noiseBlankerThresholdDb=ch.cfg.noise_blanker_threshold_db,
        notchFrequencies=ch.cfg.notch_frequencies,
        enableNoiseReduction=ch.cfg.enable_noise_reduction,
        noiseReductionDb=ch.cfg.noise_reduction_db,
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
        name=ch.cfg.name,
        autoName=ch.cfg.auto_name,
        signalPowerDb=ch.signal_power_db,
        rssiDb=ch.rssi_db,
        snrDb=ch.snr_db,
        audioRmsDb=ch.audio_rms_db,
        audioPeakDb=ch.audio_peak_db,
        audioClippingCount=ch.audio_clipping_count,
        # Filter configuration
        enableDeemphasis=ch.cfg.enable_deemphasis,
        deemphasisTauUs=ch.cfg.deemphasis_tau_us,
        enableMpxFilter=ch.cfg.enable_mpx_filter,
        mpxCutoffHz=ch.cfg.mpx_cutoff_hz,
        enableFmHighpass=ch.cfg.enable_fm_highpass,
        fmHighpassHz=ch.cfg.fm_highpass_hz,
        enableFmLowpass=ch.cfg.enable_fm_lowpass,
        fmLowpassHz=ch.cfg.fm_lowpass_hz,
        enableAmHighpass=ch.cfg.enable_am_highpass,
        amHighpassHz=ch.cfg.am_highpass_hz,
        enableAmLowpass=ch.cfg.enable_am_lowpass,
        amLowpassHz=ch.cfg.am_lowpass_hz,
        enableSsbBandpass=ch.cfg.enable_ssb_bandpass,
        ssbBandpassLowHz=ch.cfg.ssb_bandpass_low_hz,
        ssbBandpassHighHz=ch.cfg.ssb_bandpass_high_hz,
        ssbMode=ch.cfg.ssb_mode,
        enableAgc=ch.cfg.enable_agc,
        agcTargetDb=ch.cfg.agc_target_db,
        agcAttackMs=ch.cfg.agc_attack_ms,
        agcReleaseMs=ch.cfg.agc_release_ms,
        enableNoiseBlanker=ch.cfg.enable_noise_blanker,
        noiseBlankerThresholdDb=ch.cfg.noise_blanker_threshold_db,
        notchFrequencies=ch.cfg.notch_frequencies,
        enableNoiseReduction=ch.cfg.enable_noise_reduction,
        noiseReductionDb=ch.cfg.noise_reduction_db,
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
    if req.name is not None:
        ch.cfg.name = req.name
    if req.notchFrequencies is not None:
        ch.cfg.notch_frequencies = req.notchFrequencies

    # Update filter configuration
    if req.enableDeemphasis is not None:
        ch.cfg.enable_deemphasis = req.enableDeemphasis
    if req.deemphasisTauUs is not None:
        ch.cfg.deemphasis_tau_us = req.deemphasisTauUs
    if req.enableMpxFilter is not None:
        ch.cfg.enable_mpx_filter = req.enableMpxFilter
    if req.mpxCutoffHz is not None:
        ch.cfg.mpx_cutoff_hz = req.mpxCutoffHz
    if req.enableFmHighpass is not None:
        ch.cfg.enable_fm_highpass = req.enableFmHighpass
    if req.fmHighpassHz is not None:
        ch.cfg.fm_highpass_hz = req.fmHighpassHz
    if req.enableFmLowpass is not None:
        ch.cfg.enable_fm_lowpass = req.enableFmLowpass
    if req.fmLowpassHz is not None:
        ch.cfg.fm_lowpass_hz = req.fmLowpassHz
    if req.enableAmHighpass is not None:
        ch.cfg.enable_am_highpass = req.enableAmHighpass
    if req.amHighpassHz is not None:
        ch.cfg.am_highpass_hz = req.amHighpassHz
    if req.enableAmLowpass is not None:
        ch.cfg.enable_am_lowpass = req.enableAmLowpass
    if req.amLowpassHz is not None:
        ch.cfg.am_lowpass_hz = req.amLowpassHz
    if req.enableSsbBandpass is not None:
        ch.cfg.enable_ssb_bandpass = req.enableSsbBandpass
    if req.ssbBandpassLowHz is not None:
        ch.cfg.ssb_bandpass_low_hz = req.ssbBandpassLowHz
    if req.ssbBandpassHighHz is not None:
        ch.cfg.ssb_bandpass_high_hz = req.ssbBandpassHighHz
    if req.ssbMode is not None:
        ch.cfg.ssb_mode = req.ssbMode
    if req.enableAgc is not None:
        ch.cfg.enable_agc = req.enableAgc
    if req.agcTargetDb is not None:
        ch.cfg.agc_target_db = req.agcTargetDb
    if req.agcAttackMs is not None:
        ch.cfg.agc_attack_ms = req.agcAttackMs
    if req.agcReleaseMs is not None:
        ch.cfg.agc_release_ms = req.agcReleaseMs
    if req.enableNoiseBlanker is not None:
        ch.cfg.enable_noise_blanker = req.enableNoiseBlanker
    if req.noiseBlankerThresholdDb is not None:
        ch.cfg.noise_blanker_threshold_db = req.noiseBlankerThresholdDb
    if req.enableNoiseReduction is not None:
        ch.cfg.enable_noise_reduction = req.enableNoiseReduction
    if req.noiseReductionDb is not None:
        ch.cfg.noise_reduction_db = req.noiseReductionDb

    # Regenerate auto_name if offset changed
    if req.offsetHz is not None:
        cap = state.captures.get_capture(ch.cfg.capture_id)
        if cap is not None:
            namer = get_frequency_namer()
            ch.cfg.auto_name = namer.suggest_channel_name(cap.cfg.center_hz, ch.cfg.offset_hz)

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
        audioRmsDb=ch.audio_rms_db,
        audioPeakDb=ch.audio_peak_db,
        audioClippingCount=ch.audio_clipping_count,
        # Filter configuration
        enableDeemphasis=ch.cfg.enable_deemphasis,
        deemphasisTauUs=ch.cfg.deemphasis_tau_us,
        enableMpxFilter=ch.cfg.enable_mpx_filter,
        mpxCutoffHz=ch.cfg.mpx_cutoff_hz,
        enableFmHighpass=ch.cfg.enable_fm_highpass,
        fmHighpassHz=ch.cfg.fm_highpass_hz,
        enableFmLowpass=ch.cfg.enable_fm_lowpass,
        fmLowpassHz=ch.cfg.fm_lowpass_hz,
        enableAmHighpass=ch.cfg.enable_am_highpass,
        amHighpassHz=ch.cfg.am_highpass_hz,
        enableAmLowpass=ch.cfg.enable_am_lowpass,
        amLowpassHz=ch.cfg.am_lowpass_hz,
        enableSsbBandpass=ch.cfg.enable_ssb_bandpass,
        ssbBandpassLowHz=ch.cfg.ssb_bandpass_low_hz,
        ssbBandpassHighHz=ch.cfg.ssb_bandpass_high_hz,
        ssbMode=ch.cfg.ssb_mode,
        enableAgc=ch.cfg.enable_agc,
        agcTargetDb=ch.cfg.agc_target_db,
        agcAttackMs=ch.cfg.agc_attack_ms,
        agcReleaseMs=ch.cfg.agc_release_ms,
        enableNoiseBlanker=ch.cfg.enable_noise_blanker,
        noiseBlankerThresholdDb=ch.cfg.noise_blanker_threshold_db,
        notchFrequencies=ch.cfg.notch_frequencies,
        enableNoiseReduction=ch.cfg.enable_noise_reduction,
        noiseReductionDb=ch.cfg.noise_reduction_db,
    )


@router.delete("/channels/{chan_id}")
def delete_channel(
    chan_id: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    state.captures.delete_channel(chan_id)
    return JSONResponse(status_code=204, content={})


# ==============================================================================
# Signal monitoring endpoints (for Claude skills)
# ==============================================================================

def _rssi_to_s_units(rssi_db: Optional[float]) -> Optional[str]:
    """Convert RSSI in dB to S-meter units (S0-S9, S9+10, etc.)."""
    if rssi_db is None:
        return None
    # Standard S-meter: S9 = -73 dBm, each S-unit is 6 dB
    # Our RSSI is relative, so we calibrate based on typical SDR levels
    # Assumption: -100 dB RSSI ≈ S0, -40 dB RSSI ≈ S9+20
    s_value = (rssi_db + 127) / 6.0
    if s_value < 0:
        return "S0"
    elif s_value <= 9:
        return f"S{int(s_value)}"
    else:
        over = (s_value - 9) * 6
        return f"S9+{int(over)}"


@router.get("/captures/{cid}/spectrum/snapshot", response_model=SpectrumSnapshotModel)
def get_spectrum_snapshot(
    cid: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    """Get a single FFT spectrum snapshot (no WebSocket required).

    Returns the most recent FFT calculation for the capture.
    Useful for scripts and Claude skills that need spectrum data.
    """
    import time
    cap = state.captures.get_capture(cid)
    if cap is None:
        raise HTTPException(status_code=404, detail="Capture not found")

    # Check if capture is running
    if cap.state != "running":
        raise HTTPException(status_code=400, detail="Capture is not running")

    # Get cached FFT data
    if cap._fft_power_list is None or cap._fft_freqs_list is None:
        raise HTTPException(status_code=503, detail="No spectrum data available yet")

    return SpectrumSnapshotModel(
        power=cap._fft_power_list,
        freqs=cap._fft_freqs_list,
        centerHz=cap.cfg.center_hz,
        sampleRate=cap.cfg.sample_rate,
        timestamp=time.time(),
    )


@router.get("/channels/{chan_id}/metrics/extended", response_model=ExtendedMetricsModel)
def get_channel_extended_metrics(
    chan_id: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    """Get extended signal metrics for a channel.

    Includes RSSI, SNR, signal power, S-meter reading, squelch state,
    and stream health metrics. Useful for tuning and monitoring.
    """
    import time
    ch = state.captures.get_channel(chan_id)
    if ch is None:
        raise HTTPException(status_code=404, detail="Channel not found")

    # Get parent capture state
    cap = state.captures.get_capture(ch.cfg.capture_id)
    capture_state = cap.state if cap else "unknown"

    # Get queue stats for stream health
    queue_stats = ch.get_queue_stats()

    # Calculate squelch state
    squelch_open = True
    if ch.cfg.squelch_db is not None and ch.signal_power_db is not None:
        squelch_open = ch.signal_power_db >= ch.cfg.squelch_db

    return ExtendedMetricsModel(
        channelId=chan_id,
        rssiDb=ch.rssi_db,
        snrDb=ch.snr_db,
        signalPowerDb=ch.signal_power_db,
        sUnits=_rssi_to_s_units(ch.rssi_db),
        squelchOpen=squelch_open,
        streamSubscribers=queue_stats.get("total_subscribers", 0),
        streamDropsPerSec=queue_stats.get("drops_since_last_log", 0) / 60.0,  # Approximate
        captureState=capture_state,
        timestamp=time.time(),
    )


@router.get("/channels/{chan_id}/metrics/history", response_model=MetricsHistoryModel)
def get_channel_metrics_history(
    chan_id: str,
    seconds: int = 60,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    """Get time-series history of signal metrics.

    Note: Currently returns a single point (current values) since
    historical tracking is not yet implemented. Future versions
    will maintain a rolling buffer of metrics.
    """
    import time
    ch = state.captures.get_channel(chan_id)
    if ch is None:
        raise HTTPException(status_code=404, detail="Channel not found")

    # Currently just return current values as a single point
    # TODO: Implement rolling buffer in Channel class for true history
    current_point = MetricsHistoryPoint(
        timestamp=time.time(),
        rssiDb=ch.rssi_db,
        snrDb=ch.snr_db,
        signalPowerDb=ch.signal_power_db,
    )

    return MetricsHistoryModel(
        channelId=chan_id,
        points=[current_point],
        durationSeconds=float(seconds),
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


# ==============================================================================
# Scanner endpoints
# ==============================================================================

from .scanner import ScannerService, ScanConfig, ScanMode


def _to_scanner_model(scanner_id: str, scanner: ScannerService) -> ScannerModel:
    """Convert ScannerService to ScannerModel."""
    return ScannerModel(
        id=scanner_id,
        captureId=scanner.capture_id,
        state=scanner.status.state,
        currentFrequency=scanner.status.current_frequency,
        currentIndex=scanner.status.current_index,
        scanList=scanner.config.scan_list,
        mode=scanner.config.mode,
        dwellTimeMs=scanner.config.dwell_time_ms,
        priorityFrequencies=scanner.config.priority_frequencies,
        priorityIntervalS=scanner.config.priority_interval_s,
        squelchThresholdDb=scanner.config.squelch_threshold_db,
        lockoutList=scanner.status.lockout_list,
        pauseDurationMs=scanner.config.pause_duration_ms,
        hits=[ScanHitModel(frequencyHz=freq, timestamp=ts) for freq, ts in scanner.status.hits],
    )


@router.post("/scanners", response_model=ScannerModel, status_code=201)
def create_scanner(
    req: CreateScannerRequest,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    """Create a new scanner for a capture."""
    # Validate capture exists
    capture = state.captures.get_capture(req.captureId)
    if capture is None:
        raise HTTPException(status_code=404, detail=f"Capture {req.captureId} not found")

    # Generate scanner ID
    scanner_id = f"scan_{len(state.scanners) + 1}"

    # Create scanner config
    config = ScanConfig(
        scan_list=req.scanList,
        mode=ScanMode(req.mode),
        dwell_time_ms=req.dwellTimeMs,
        priority_frequencies=req.priorityFrequencies,
        priority_interval_s=req.priorityIntervalS,
        squelch_threshold_db=req.squelchThresholdDb,
        lockout_frequencies=req.lockoutFrequencies,
        pause_duration_ms=req.pauseDurationMs,
    )

    # Create scanner service
    scanner = ScannerService(capture_id=req.captureId, config=config)

    # Set up update callback to tune the capture
    async def update_frequency(freq_hz: float):
        try:
            state.captures.update_capture(req.captureId, center_hz=freq_hz)
        except Exception as e:
            logger.error(f"Scanner failed to update frequency: {e}")

    scanner.set_update_callback(update_frequency)

    # Set up RSSI callback to get current RSSI from capture
    def get_rssi() -> Optional[float]:
        # Get first channel's RSSI as a proxy for activity
        channels = list(capture.channels.values())
        if channels:
            return channels[0].rssi_db
        return None

    scanner.set_rssi_callback(get_rssi)

    # Store scanner
    state.scanners[scanner_id] = scanner

    return _to_scanner_model(scanner_id, scanner)


@router.get("/scanners", response_model=List[ScannerModel])
def list_scanners(_: None = Depends(auth_check), state: AppState = Depends(get_state)):
    """List all scanners."""
    return [_to_scanner_model(sid, scanner) for sid, scanner in state.scanners.items()]


@router.get("/scanners/{sid}", response_model=ScannerModel)
def get_scanner(
    sid: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    """Get scanner by ID."""
    scanner = state.scanners.get(sid)
    if scanner is None:
        raise HTTPException(status_code=404, detail=f"Scanner {sid} not found")
    return _to_scanner_model(sid, scanner)


@router.patch("/scanners/{sid}", response_model=ScannerModel)
def update_scanner(
    sid: str,
    req: UpdateScannerRequest,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    """Update scanner configuration."""
    scanner = state.scanners.get(sid)
    if scanner is None:
        raise HTTPException(status_code=404, detail=f"Scanner {sid} not found")

    # Update config
    if req.scanList is not None:
        scanner.config.scan_list = req.scanList
    if req.mode is not None:
        scanner.config.mode = ScanMode(req.mode)
    if req.dwellTimeMs is not None:
        scanner.config.dwell_time_ms = req.dwellTimeMs
    if req.priorityFrequencies is not None:
        scanner.config.priority_frequencies = req.priorityFrequencies
    if req.priorityIntervalS is not None:
        scanner.config.priority_interval_s = req.priorityIntervalS
    if req.squelchThresholdDb is not None:
        scanner.config.squelch_threshold_db = req.squelchThresholdDb
    if req.lockoutFrequencies is not None:
        scanner.status.lockout_list = req.lockoutFrequencies
    if req.pauseDurationMs is not None:
        scanner.config.pause_duration_ms = req.pauseDurationMs

    return _to_scanner_model(sid, scanner)


@router.delete("/scanners/{sid}", status_code=204)
def delete_scanner(
    sid: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    """Delete a scanner."""
    scanner = state.scanners.get(sid)
    if scanner is None:
        raise HTTPException(status_code=404, detail=f"Scanner {sid} not found")

    # Stop scanner if running
    scanner.stop()

    # Remove from state
    del state.scanners[sid]


@router.post("/scanners/{sid}/start", response_model=ScannerModel)
def start_scanner(
    sid: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    """Start scanning."""
    scanner = state.scanners.get(sid)
    if scanner is None:
        raise HTTPException(status_code=404, detail=f"Scanner {sid} not found")

    scanner.start()
    return _to_scanner_model(sid, scanner)


@router.post("/scanners/{sid}/stop", response_model=ScannerModel)
def stop_scanner(
    sid: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    """Stop scanning."""
    scanner = state.scanners.get(sid)
    if scanner is None:
        raise HTTPException(status_code=404, detail=f"Scanner {sid} not found")

    scanner.stop()
    return _to_scanner_model(sid, scanner)


@router.post("/scanners/{sid}/pause", response_model=ScannerModel)
def pause_scanner(
    sid: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    """Pause scanning."""
    scanner = state.scanners.get(sid)
    if scanner is None:
        raise HTTPException(status_code=404, detail=f"Scanner {sid} not found")

    scanner.pause()
    return _to_scanner_model(sid, scanner)


@router.post("/scanners/{sid}/resume", response_model=ScannerModel)
def resume_scanner(
    sid: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    """Resume scanning from pause."""
    scanner = state.scanners.get(sid)
    if scanner is None:
        raise HTTPException(status_code=404, detail=f"Scanner {sid} not found")

    scanner.resume()
    return _to_scanner_model(sid, scanner)


@router.post("/scanners/{sid}/lock", response_model=ScannerModel)
def lock_scanner(
    sid: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    """Lock on current frequency."""
    scanner = state.scanners.get(sid)
    if scanner is None:
        raise HTTPException(status_code=404, detail=f"Scanner {sid} not found")

    scanner.lock()
    return _to_scanner_model(sid, scanner)


@router.post("/scanners/{sid}/unlock", response_model=ScannerModel)
def unlock_scanner(
    sid: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    """Unlock and resume scanning."""
    scanner = state.scanners.get(sid)
    if scanner is None:
        raise HTTPException(status_code=404, detail=f"Scanner {sid} not found")

    scanner.unlock()
    return _to_scanner_model(sid, scanner)


@router.post("/scanners/{sid}/lockout", response_model=ScannerModel)
def lockout_frequency(
    sid: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    """Add current frequency to lockout list."""
    scanner = state.scanners.get(sid)
    if scanner is None:
        raise HTTPException(status_code=404, detail=f"Scanner {sid} not found")

    scanner.lockout_current()
    return _to_scanner_model(sid, scanner)


@router.delete("/scanners/{sid}/lockout/{freq}", response_model=ScannerModel)
def clear_lockout(
    sid: str,
    freq: float,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    """Remove frequency from lockout list."""
    scanner = state.scanners.get(sid)
    if scanner is None:
        raise HTTPException(status_code=404, detail=f"Scanner {sid} not found")

    scanner.clear_lockout(freq)
    return _to_scanner_model(sid, scanner)


@router.delete("/scanners/{sid}/lockouts", response_model=ScannerModel)
def clear_all_lockouts(
    sid: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
):
    """Clear all lockouts."""
    scanner = state.scanners.get(sid)
    if scanner is None:
        raise HTTPException(status_code=404, detail=f"Scanner {sid} not found")

    scanner.clear_all_lockouts()
    return _to_scanner_model(sid, scanner)
