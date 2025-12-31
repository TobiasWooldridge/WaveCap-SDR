from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any, Literal, cast

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from .config import AppConfig, default_config_path, load_config, save_config
from .device_namer import (
    generate_capture_name,
    get_device_nickname,
    get_device_shorthand,
    set_device_nickname,
)
from .frequency_namer import get_frequency_namer
from .models import (
    CaptureModel,
    ChannelModel,
    ClassifiedChannelModel,
    ClassifiedChannelsResponse,
    ConfigWarning,
    CreateCaptureRequest,
    CreateChannelRequest,
    CreateScannerRequest,
    DeviceModel,
    ExtendedMetricsModel,
    MetricsHistoryModel,
    MetricsHistoryPoint,
    POCSAGMessageModel,
    RDSDataModel,
    RecipeChannelModel,
    RecipeModel,
    ScanHitModel,
    ScannerModel,
    SpectrumSnapshotModel,
    UpdateCaptureRequest,
    UpdateChannelRequest,
    UpdateScannerRequest,
)
from .sdrplay_recovery import get_recovery
from .state import AppState
from .state_broadcaster import get_broadcaster

router = APIRouter()


class ReloadConfigRequest(BaseModel):
    """Request body for hot reloading the YAML config."""

    configPath: str | None = None
    startCaptures: bool = True


@router.get("/health")
def health_check(request: Request) -> JSONResponse:
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

    try:
        # Check SDRplay recovery status
        recovery = get_recovery()
        stats = recovery.stats
        health_status["checks"]["recovery"] = {
            "status": "ok",
            "enabled": recovery.enabled,
            "cooldown_seconds": recovery.cooldown_seconds,
            "max_restarts_per_hour": recovery.max_restarts_per_hour,
            "recovery_count": stats.recovery_count,
            "recovery_failures": stats.recovery_failures,
            "last_recovery_attempt": stats.last_recovery_attempt,
            "last_recovery_success": stats.last_recovery_success,
            "last_error": stats.last_error,
        }
    except Exception as e:
        health_status["checks"]["recovery"] = {"status": "error", "error": str(e)}

    # Overall status
    if any(check.get("status") == "error" for check in health_status["checks"].values()):
        health_status["status"] = "degraded"

    status_code = 200 if health_status["status"] in ["ok", "degraded"] else 500
    return JSONResponse(status_code=status_code, content=health_status)


@router.post("/shutdown")
async def shutdown_server(request: Request) -> dict[str, str]:
    """Gracefully shutdown the server.

    This endpoint triggers a graceful shutdown sequence:
    1. Stop all trunking systems (saves state)
    2. Stop all captures
    3. Shutdown SDR devices
    4. Terminate the server process

    The server will acknowledge the request immediately and then shutdown.
    """
    import os
    import signal

    state: AppState | None = getattr(request.app.state, "app_state", None)
    if state is None:
        return {"status": "error", "message": "AppState not initialized"}

    logger.info("Shutdown requested via API - starting graceful shutdown")

    async def do_shutdown() -> None:
        """Perform shutdown in background to allow response to be sent."""
        await asyncio.sleep(0.5)  # Allow response to be sent

        try:
            # Stop trunking systems first
            if hasattr(state, "trunking_manager") and state.trunking_manager:
                logger.info("Stopping trunking systems...")
                await state.trunking_manager.stop()

            # Stop all captures
            logger.info("Stopping captures...")
            for capture in state.captures.list_captures():
                try:
                    await state.captures.stop_capture(capture.cfg.id)
                except Exception as e:
                    logger.error(f"Error stopping capture {capture.cfg.id}: {e}")

            logger.info("Graceful shutdown complete - sending SIGTERM")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            # Send SIGTERM to ourselves to trigger uvicorn shutdown
            os.kill(os.getpid(), signal.SIGTERM)

    # Schedule shutdown in background
    asyncio.create_task(do_shutdown())

    return {"status": "ok", "message": "Shutdown initiated"}


# Frontend log storage
_frontend_logs: list[dict[str, Any]] = []
_FRONTEND_LOG_MAX = 500  # Keep last 500 log entries


@router.post("/frontend-logs")
async def receive_frontend_logs(request: Request) -> dict[str, Any]:
    """Receive console logs from the frontend for debugging."""
    global _frontend_logs
    try:
        body = await request.json()
        logs = body.get("logs", [])
        for log in logs:
            log["received_at"] = time.time()
            _frontend_logs.append(log)
        # Trim to max size
        if len(_frontend_logs) > _FRONTEND_LOG_MAX:
            _frontend_logs = _frontend_logs[-_FRONTEND_LOG_MAX:]
        return {"status": "ok", "received": len(logs)}
    except Exception as e:
        logger.error(f"Error receiving frontend logs: {e}")
        return {"status": "error", "message": str(e)}


@router.get("/frontend-logs")
def get_frontend_logs(
    level: str | None = None,
    prefix: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Get recent frontend logs for debugging.

    Args:
        level: Filter by log level (log, warn, error, debug)
        prefix: Filter by message prefix (e.g., "[Spectrum]")
        limit: Max number of logs to return (default 100)
    """
    logs = _frontend_logs[-limit:]
    if level:
        logs = [log for log in logs if log.get("level") == level]
    if prefix:
        logs = [log for log in logs if any(str(arg).startswith(prefix) for arg in log.get("args", []))]
    return {
        "count": len(logs),
        "total_stored": len(_frontend_logs),
        "logs": logs,
    }


@router.delete("/frontend-logs")
def clear_frontend_logs() -> dict[str, Any]:
    """Clear all stored frontend logs."""
    global _frontend_logs
    count = len(_frontend_logs)
    _frontend_logs = []
    return {"status": "ok", "cleared": count}


@router.get("/debug/perf", response_model=None)
def get_performance_metrics(request: Request) -> JSONResponse | dict[str, Any]:
    """Get detailed performance metrics for all captures.

    Returns timing statistics (loop time, DSP time, FFT time) and queue depths
    for performance profiling and optimization validation.
    """
    state = getattr(request.app.state, "app_state", None)
    if state is None:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "AppState not initialized"}
        )

    import time
    result: dict[str, Any] = {
        "timestamp": time.time(),
        "captures": {},
    }

    try:
        for cap in state.captures.list_captures():
            # Get performance stats from capture
            perf_stats = cap.get_perf_stats() if hasattr(cap, 'get_perf_stats') else {}

            # Get channel queue depths
            channels = state.captures.list_channels(cap.cfg.id)
            channel_stats = {}
            for ch in channels:
                stats = ch.get_queue_stats()
                channel_stats[ch.cfg.id] = {
                    "subscribers": stats.get("total_subscribers", 0),
                    "drops": stats.get("drops_since_last_log", 0),
                    "mode": ch.cfg.mode,
                    "rssi_db": ch.rssi_db,
                }

            result["captures"][cap.cfg.id] = {
                "state": cap.state,
                "device_id": cap.cfg.device_id[:30] + "..." if len(cap.cfg.device_id) > 30 else cap.cfg.device_id,
                "sample_rate": cap.cfg.sample_rate,
                "timing": perf_stats,
                "channels": channel_stats,
            }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

    return result


def get_state(request: Request) -> AppState:
    state: AppState | None = getattr(request.app.state, "app_state", None)
    if state is None:
        raise RuntimeError("AppState not initialized")
    return state


def _to_rds_data_model(rds_data: Any) -> RDSDataModel | None:
    """Convert RDSData from backend to RDSDataModel for API response."""
    if rds_data is None:
        return None
    data_dict = rds_data.to_dict()
    return RDSDataModel(
        piCode=data_dict.get("piCode"),
        psName=data_dict.get("psName"),
        radioText=data_dict.get("radioText"),
        pty=data_dict.get("pty", 0),
        ptyName=data_dict.get("ptyName", "None"),
        ta=data_dict.get("ta", False),
        tp=data_dict.get("tp", False),
        ms=data_dict.get("ms", True),
    )


def _compute_config_warnings(cap: Any) -> list[ConfigWarning]:
    """Compute configuration warnings for a capture."""
    warnings: list[ConfigWarning] = []

    device_id_lower = cap.cfg.device_id.lower() if cap.cfg.device_id else ""
    is_rtl = "rtlsdr" in device_id_lower or "rtl-sdr" in device_id_lower
    is_sdrplay = "sdrplay" in device_id_lower

    # RTL-SDR unstable sample rate warning
    # RTL-SDR devices are known to be unstable at sample rates below ~900kHz
    # The most stable rates are 1.024 MHz, 2.048 MHz, and 2.4 MHz
    if is_rtl and cap.cfg.sample_rate < 900_000:
        warnings.append(ConfigWarning(
            code="rtl_unstable_sample_rate",
            severity="warning",
            message=f"Sample rate {cap.cfg.sample_rate / 1_000_000:.3f} MHz may be unstable on RTL-SDR. "
                    f"Consider using 1.024 MHz or higher to reduce IQ overflows."
        ))

    # Bandwidth exceeds sample rate warning
    # Bandwidth should not exceed sample rate (Nyquist limit)
    if cap.cfg.bandwidth and cap.cfg.bandwidth > cap.cfg.sample_rate:
        warnings.append(ConfigWarning(
            code="bandwidth_exceeds_sample_rate",
            severity="warning",
            message=f"Bandwidth ({cap.cfg.bandwidth / 1_000_000:.3f} MHz) exceeds sample rate "
                    f"({cap.cfg.sample_rate / 1_000_000:.3f} MHz). Reduce bandwidth or increase sample rate."
        ))

    # Bandwidth too close to sample rate (may cause aliasing at edges)
    # Generally bandwidth should be at most 80% of sample rate for clean edges
    if cap.cfg.bandwidth and cap.cfg.bandwidth > cap.cfg.sample_rate * 0.9:
        if cap.cfg.bandwidth <= cap.cfg.sample_rate:  # Don't duplicate the exceeds warning
            warnings.append(ConfigWarning(
                code="bandwidth_near_sample_rate",
                severity="info",
                message=f"Bandwidth is >{90}% of sample rate. Consider reducing bandwidth or "
                        f"increasing sample rate to avoid aliasing at spectrum edges."
            ))

    # Very high sample rate warning for SDRplay
    # SDRplay can do 10 MHz but may have USB bandwidth issues
    if is_sdrplay and cap.cfg.sample_rate > 8_000_000:
        warnings.append(ConfigWarning(
            code="sdrplay_high_sample_rate",
            severity="info",
            message=f"Sample rate {cap.cfg.sample_rate / 1_000_000:.1f} MHz is high for SDRplay. "
                    f"May cause USB bandwidth issues on some systems."
        ))

    # Gain set to 0 warning (might be unintentional)
    if cap.cfg.gain is not None and cap.cfg.gain == 0:
        warnings.append(ConfigWarning(
            code="zero_gain",
            severity="info",
            message="Gain is set to 0 dB. If signals appear weak, try increasing gain."
        ))

    return warnings


def _to_capture_model(cap: Any, trunking_manager: Any = None) -> CaptureModel:
    """Helper to convert a Capture to CaptureModel consistently.

    Args:
        cap: Capture instance to convert
        trunking_manager: Optional TrunkingManager to look up trunking ownership
    """
    from .error_tracker import ErrorStats, get_error_tracker

    # Get overflow rate from error tracker
    tracker = get_error_tracker()
    stats = tracker.get_stats()
    overflow_stats = stats.get("iq_overflow", ErrorStats())

    # Determine retry status
    is_retrying = cap.state == "starting" and cap._retry_count > 0
    retry_attempt = cap._retry_count if is_retrying else None
    retry_max = cap._max_retries if is_retrying else None
    retry_delay = cap._retry_delay if is_retrying else None

    # Compute configuration warnings
    config_warnings = _compute_config_warnings(cap)

    # Check if this capture is owned by a trunking system
    trunking_system_id = None
    if trunking_manager is not None:
        trunking_system_id = trunking_manager.get_system_for_capture(cap.cfg.id)

    return CaptureModel(
        id=cap.cfg.id,
        deviceId=cap.cfg.device_id,
        state=cap.state,
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
        # FFT/Spectrum settings
        fftFps=cap.cfg.fft_fps,
        fftMaxFps=cap.cfg.fft_max_fps,
        fftSize=cap.cfg.fft_size,
        fftAccelerator=cap.cfg.fft_accelerator,
        # Error indicators
        iqOverflowCount=cap._iq_overflow_count,
        iqOverflowRate=overflow_stats.rate_per_second,
        retryAttempt=retry_attempt,
        retryMaxAttempts=retry_max,
        retryDelay=retry_delay,
        # Configuration warnings
        configWarnings=config_warnings,
        # Trunking system ownership
        trunkingSystemId=trunking_system_id,
    )


def _to_channel_model(ch: Any) -> ChannelModel:
    """Helper to convert a Channel to ChannelModel consistently."""
    from .error_tracker import ErrorStats, get_error_tracker

    # Get drop rate from error tracker
    tracker = get_error_tracker()
    stats = tracker.get_stats()
    drop_stats = stats.get("audio_drop", ErrorStats())

    return ChannelModel(
        id=ch.cfg.id,
        captureId=ch.cfg.capture_id,
        mode=cast(Literal["wbfm", "nbfm", "am", "ssb", "raw", "p25", "dmr"], ch.cfg.mode),
        state=cast(Literal["created", "running", "stopped"], ch.state),
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
        # Error indicators
        audioDropCount=ch._drop_count,
        audioDropRate=drop_stats.rate_per_second,
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
        ssbBfoOffsetHz=ch.cfg.ssb_bfo_offset_hz,
        enableAgc=ch.cfg.enable_agc,
        agcTargetDb=ch.cfg.agc_target_db,
        agcAttackMs=ch.cfg.agc_attack_ms,
        agcReleaseMs=ch.cfg.agc_release_ms,
        enableNoiseBlanker=ch.cfg.enable_noise_blanker,
        noiseBlankerThresholdDb=ch.cfg.noise_blanker_threshold_db,
        notchFrequencies=ch.cfg.notch_frequencies,
        enableNoiseReduction=ch.cfg.enable_noise_reduction,
        noiseReductionDb=ch.cfg.noise_reduction_db,
        rdsData=_to_rds_data_model(ch.rds_data),
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


async def _start_captures_from_config(app_state: AppState, config: AppConfig) -> list[str]:
    """Recreate and start captures defined in the config."""
    started: list[str] = []
    created_any = False

    for cap_cfg in config.captures:
        preset_name = cap_cfg.preset
        preset = config.presets.get(preset_name)
        if preset is None:
            logger.warning("Preset '%s' not found; skipping capture", preset_name)
            continue

        try:
            cap = app_state.captures.create_capture(
                device_id=cap_cfg.device_id,
                center_hz=preset.center_hz,
                sample_rate=preset.sample_rate,
                gain=preset.gain,
                bandwidth=preset.bandwidth,
                ppm=preset.ppm,
                antenna=preset.antenna,
                device_settings=preset.device_settings,
                element_gains=preset.element_gains,
                stream_format=preset.stream_format,
                dc_offset_auto=preset.dc_offset_auto,
                iq_balance_auto=preset.iq_balance_auto,
            )

            devices = app_state.captures.list_devices()
            device = next((d for d in devices if d["id"] == cap.cfg.device_id), None)
            if device:
                device_nickname = get_device_nickname(cap.cfg.device_id)
                cap.cfg.auto_name = generate_capture_name(
                    center_hz=preset.center_hz,
                    device_id=cap.cfg.device_id,
                    device_label=device["label"],
                    recipe_name=None,
                    device_nickname=device_nickname,
                )

            for offset_hz in preset.offsets:
                ch = app_state.captures.create_channel(
                    cid=cap.cfg.id,
                    mode="wbfm",
                    offset_hz=offset_hz,
                    audio_rate=config.stream.default_audio_rate,
                    squelch_db=preset.squelch_db,
                )
                ch.start()

            cap.start()
            app_state.capture_presets[cap.cfg.id] = preset_name
            started.append(cap.cfg.id)
            created_any = True
        except Exception as e:
            logger.error("Failed to start capture for preset '%s': %s", preset_name, e)

    if not created_any:
        # Initialize a default (stopped) capture for UI if nothing configured
        try:
            default_preset_name: str = next(iter(config.presets.keys()), "")
            preset = config.presets.get(default_preset_name) if default_preset_name else None

            devices = app_state.captures.list_devices()
            device_id = devices[0]["id"] if devices else None

            center_hz = preset.center_hz if preset else 100_000_000.0
            sample_rate = preset.sample_rate if preset else 1_000_000

            cap = app_state.captures.create_capture(
                device_id=device_id,
                center_hz=center_hz,
                sample_rate=sample_rate,
                gain=(preset.gain if preset else None),
                bandwidth=(preset.bandwidth if preset else None),
                ppm=(preset.ppm if preset else None),
                antenna=(preset.antenna if preset else None),
                device_settings=(preset.device_settings if preset else None),
                element_gains=(preset.element_gains if preset else None),
                stream_format=(preset.stream_format if preset else None),
                dc_offset_auto=(preset.dc_offset_auto if preset else True),
                iq_balance_auto=(preset.iq_balance_auto if preset else True),
            )

            if device_id:
                device = next((d for d in devices if d["id"] == device_id), None)
                if device:
                    device_nickname = get_device_nickname(device_id)
                    cap.cfg.auto_name = generate_capture_name(
                        center_hz=center_hz,
                        device_id=device_id,
                        device_label=device["label"],
                        recipe_name=None,
                        device_nickname=device_nickname,
                    )

            started.append(cap.cfg.id)
        except Exception as e:
            logger.error("Failed to create default capture during reload: %s", e)

    return started


@router.post("/config/reload")
async def reload_config(
    request: Request,
    req: ReloadConfigRequest,
    _: None = Depends(auth_check),
) -> dict[str, Any]:
    """Hot-reload the YAML configuration file with minimal disruption."""
    state: AppState | None = getattr(request.app.state, "app_state", None)
    cfg_path = req.configPath or (state.config_path if state and state.config_path else default_config_path())
    new_config = load_config(cfg_path)

    # Gracefully stop existing managers/captures
    if state is not None:
        with contextlib.suppress(Exception):
            await state.trunking_manager.stop()
        for capture in state.captures.list_captures():
            with contextlib.suppress(Exception):
                await state.captures.stop_capture(capture.cfg.id)

    # Build fresh state from the new config and swap it in
    new_state = AppState.from_config(new_config, cfg_path)
    request.app.state.app_state = new_state

    # Start trunking manager (auto-starts systems based on config)
    await new_state.trunking_manager.start()

    started_captures: list[str] = []
    if req.startCaptures:
        started_captures = await _start_captures_from_config(new_state, new_config)

    systems = [sys.to_dict() for sys in new_state.trunking_manager.list_systems()]
    return {
        "status": "ok",
        "configPath": cfg_path,
        "capturesStarted": started_captures,
        "systems": systems,
        "note": "RF-level changes applied without restarting the process",
    }


@router.get("/devices/{device_id}/name")
def get_device_name(device_id: str, _: None = Depends(auth_check), state: AppState = Depends(get_state)) -> dict[str, Any]:
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
def update_device_name(device_id: str, request: dict[str, Any], _: None = Depends(auth_check), state: AppState = Depends(get_state)) -> dict[str, Any]:
    """Set custom nickname for a device."""
    nickname = request.get("nickname", "")

    # Validate device exists - check enumeration AND active captures
    # (SDRplay devices don't appear in enumeration when busy streaming)
    device_found = False
    try:
        devices = state.captures.list_devices()
        device_found = any(d["id"] == device_id for d in devices)
    except Exception:
        pass  # Enumeration may fail if all devices are busy

    # Also check devices from active captures
    if not device_found:
        for cap in state.captures.list_captures():
            if cap.cfg.device_id == device_id:
                device_found = True
                break

    if not device_found:
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


def _get_stable_device_id(device_id: str) -> str:
    """Extract a stable identifier from a SoapySDR device ID string.

    Device IDs can contain volatile fields like 'tuner' that change based on
    device availability. This function extracts driver + serial (or label) to
    create a stable ID for deduplication.
    """
    driver = ""
    serial = ""
    label = ""
    for part in device_id.split(","):
        if part.startswith("driver="):
            driver = part.split("=", 1)[1]
        elif part.startswith("serial="):
            serial = part.split("=", 1)[1]
        elif part.startswith("label="):
            label = part.split("=", 1)[1]
    # Use serial if available, otherwise fall back to label
    return f"{driver}:{serial}" if serial else f"{driver}:{label}"


@router.get("/devices", response_model=list[DeviceModel], response_model_by_alias=False)
def list_devices(_: None = Depends(auth_check), state: AppState = Depends(get_state)) -> list[DeviceModel]:
    result: list[DeviceModel] = []
    seen_ids = set()  # Full device IDs
    seen_stable_ids = set()  # Stable IDs for deduplication

    # Try to enumerate devices, but don't fail if enumeration errors
    # (e.g., "Broken pipe" when devices are busy)
    try:
        devices = state.captures.list_devices()
        for d in devices:
            device_id = d["id"]
            device_label = d["label"]
            stable_id = _get_stable_device_id(device_id)

            # Skip if we've already seen this physical device
            if stable_id in seen_stable_ids:
                continue

            seen_ids.add(device_id)
            seen_stable_ids.add(stable_id)

            # Get nickname and shorthand name
            nickname = get_device_nickname(device_id)
            shorthand = get_device_shorthand(device_id, device_label)

            result.append(DeviceModel(**d, nickname=nickname, shorthand=shorthand))
    except Exception:
        # Enumeration failed (devices busy, driver error, etc.)
        # Continue to add devices from active captures below
        pass

    # Also include devices from active captures that aren't in enumeration
    # (devices in use may not show up in driver enumeration due to timeouts or being busy)
    for cap in state.captures.list_captures():
        device_id = cap.cfg.device_id

        # Skip placeholder device IDs
        if device_id in ("auto", "fake0", ""):
            continue

        stable_id = _get_stable_device_id(device_id)

        # Skip if we've already seen this physical device (by stable ID)
        if stable_id in seen_stable_ids:
            continue

        if device_id not in seen_ids:
            seen_ids.add(device_id)
            seen_stable_ids.add(stable_id)
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
                freq_min_hz=device_info.freq_min_hz if device_info else 0,
                freq_max_hz=device_info.freq_max_hz if device_info else 6e9,
                sample_rates=list(device_info.sample_rates) if device_info else [],
                gains=list(device_info.gains) if device_info else [],
                gain_min=device_info.gain_min if device_info else 0,
                gain_max=device_info.gain_max if device_info else 50,
                bandwidth_min=device_info.bandwidth_min if device_info else 0,
                bandwidth_max=device_info.bandwidth_max if device_info else 10e6,
                ppm_min=-100,
                ppm_max=100,
                antennas=list(device_info.antennas) if device_info else [],
                nickname=nickname,
                shorthand=shorthand,
            )
            result.append(in_use_device)

    return result


@router.post("/devices/refresh", response_model=list[DeviceModel], response_model_by_alias=False)
def refresh_devices(_: None = Depends(auth_check), state: AppState = Depends(get_state)) -> list[DeviceModel]:
    """Force re-enumeration of all SDR devices.

    Invalidates the device cache and performs a fresh enumeration.
    Use after USB power cycling or when devices aren't appearing.
    """
    from .devices.soapy import invalidate_sdrplay_caches

    # Invalidate all caches to force fresh enumeration
    invalidate_sdrplay_caches()

    # Return fresh device list
    return list_devices(_, state)


@router.get("/devices/sdrplay/health")
def get_sdrplay_health() -> dict[str, Any]:
    """Get SDRplay service health status for monitoring.

    Returns metrics about SDRplay enumeration success/failure history,
    which can be used to detect stuck service states proactively.
    """
    from .devices.soapy import get_sdrplay_health_status
    health = get_sdrplay_health_status()

    # Also include recovery module status
    recovery = get_recovery()
    allowed, reason = recovery.can_restart()

    return {
        "health": health,
        "recovery": {
            "enabled": recovery.enabled,
            "can_restart": allowed,
            "can_restart_reason": reason,
            "cooldown_seconds": recovery.cooldown_seconds,
            "recovery_count": recovery.stats.recovery_count,
            "recovery_failures": recovery.stats.recovery_failures,
            "last_recovery_attempt": recovery.stats.last_recovery_attempt,
            "last_recovery_success": recovery.stats.last_recovery_success,
            "last_error": recovery.stats.last_error,
        }
    }


@router.post("/devices/sdrplay/restart-service")
def restart_sdrplay_service(_: None = Depends(auth_check)) -> dict[str, Any]:
    """Restart the SDRplay API service to recover from stuck states.

    Use this when SDRplay captures are stuck in 'starting' state or
    device enumeration hangs. The service will be killed and restarted
    by the system service manager (launchd on macOS, systemd on Linux).

    Rate limited: max 5 restarts per hour with 60-second cooldown between attempts.
    """
    from .devices.soapy import invalidate_sdrplay_caches, reset_sdrplay_health_counters

    recovery = get_recovery()
    allowed, reason = recovery.can_restart()

    if not allowed:
        raise HTTPException(status_code=429, detail=f"Service restart not allowed: {reason}")

    success = recovery.restart_service(reason="User requested via API")

    if success:
        # Reset health tracking counters and invalidate all caches
        reset_sdrplay_health_counters()
        invalidate_sdrplay_caches()
        return {
            "status": "ok",
            "message": "SDRplay service restarted and caches cleared",
            "stats": {
                "recovery_count": recovery.stats.recovery_count,
                "last_recovery_success": recovery.stats.last_recovery_success,
            },
            "caches_cleared": True,
        }
    else:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to restart SDRplay service: {recovery.stats.last_error or 'Unknown error'}"
        )


@router.get("/devices/usb/hubs")
def get_usb_hubs(_: None = Depends(auth_check)) -> dict[str, Any]:
    """Get USB hub status for power management.

    Returns list of controllable USB hubs and their port status,
    including connected devices with vendor/product IDs and serials.
    Requires uhubctl to be installed.
    """
    from .uhubctl import get_hub_status_dict

    return get_hub_status_dict()


@router.post("/devices/usb/power-cycle-all")
async def power_cycle_all_usb(
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
) -> dict[str, Any]:
    """Power cycle all USB ports with connected devices.

    This performs a hardware reset of all devices on controllable USB hubs.
    All running captures will be stopped first, then all ports are cycled.

    Requires uhubctl to be installed.
    """
    from .uhubctl import is_uhubctl_available, power_cycle_all_ports

    if not is_uhubctl_available():
        raise HTTPException(
            status_code=503,
            detail="uhubctl not installed. Install with: brew install uhubctl"
        )

    # Stop all running captures first
    stopped_captures = []
    for capture in state.captures.list_captures():
        if capture.state == "running":
            try:
                await capture.stop()
                stopped_captures.append(capture.cfg.id)
                logger.info(f"Stopped capture {capture.cfg.id} before USB power cycle")
            except Exception as e:
                logger.warning(f"Failed to stop capture {capture.cfg.id}: {e}")

    # Power cycle all ports
    success, message, ports_cycled = power_cycle_all_ports(delay=2.0)

    if not success and ports_cycled == 0:
        raise HTTPException(status_code=503, detail=message)

    return {
        "status": "ok" if success else "partial",
        "message": message,
        "portsCycled": ports_cycled,
        "stoppedCaptures": stopped_captures,
    }


@router.post("/devices/usb/power-cycle/{capture_id}")
async def power_cycle_capture_device(capture_id: str, _: None = Depends(auth_check), state: AppState = Depends(get_state)) -> dict[str, Any]:
    """Power cycle the USB port for a capture's device.

    This performs a hardware reset by cycling the USB port power,
    which can recover devices from stuck states that software
    restart cannot fix.

    Requires uhubctl to be installed and the device to be connected
    to a controllable USB hub.
    """
    from .uhubctl import is_uhubctl_available, power_cycle_device

    if not is_uhubctl_available():
        raise HTTPException(
            status_code=503,
            detail="uhubctl not installed. Install with: brew install uhubctl"
        )

    # Get the capture to find its device
    capture = state.captures.get_capture(capture_id)
    if not capture:
        raise HTTPException(status_code=404, detail=f"Capture {capture_id} not found")

    device_id = capture.cfg.device_id

    # Stop the capture first if it's running
    was_running = capture.state == "running"
    if was_running:
        try:
            await capture.stop()
            logger.info(f"Stopped capture {capture_id} before USB power cycle")
        except Exception as e:
            logger.warning(f"Failed to stop capture before power cycle: {e}")

    # Power cycle the device
    success, message = power_cycle_device(device_id, delay=2.0)

    if not success:
        raise HTTPException(status_code=503, detail=message)

    return {
        "status": "ok",
        "message": message,
        "captureId": capture_id,
        "deviceId": device_id,
        "wasRunning": was_running,
    }


def _adjust_recipe_for_device(recipe_cfg: Any, device_info: dict[str, Any]) -> dict[str, Any]:
    """Adjust recipe parameters to fit device capabilities.

    Returns a dict with adjusted sampleRate, bandwidth, and gain values.
    """
    adjustments = {
        "sampleRate": recipe_cfg.sample_rate,
        "bandwidth": recipe_cfg.bandwidth,
        "gain": recipe_cfg.gain,
    }

    # Adjust sample rate to closest valid rate
    valid_rates = device_info.get("sample_rates", [])
    if valid_rates and recipe_cfg.sample_rate not in valid_rates:
        adjustments["sampleRate"] = min(valid_rates, key=lambda r: abs(r - recipe_cfg.sample_rate))

    # Adjust bandwidth to fit device range
    bw_min = device_info.get("bandwidth_min")
    bw_max = device_info.get("bandwidth_max")
    if bw_min is not None and bw_max is not None and recipe_cfg.bandwidth is not None:
        adjustments["bandwidth"] = max(bw_min, min(bw_max, recipe_cfg.bandwidth))

    # Adjust gain to fit device range
    gain_min = device_info.get("gain_min")
    gain_max = device_info.get("gain_max")
    if gain_min is not None and gain_max is not None and recipe_cfg.gain is not None:
        adjustments["gain"] = max(gain_min, min(gain_max, recipe_cfg.gain))

    return adjustments


@router.get("/recipes", response_model=list[RecipeModel])
def list_recipes(
    device_id: str | None = None,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
) -> list[RecipeModel]:
    """Get all available capture creation recipes.

    If device_id is provided, recipe parameters are adjusted to fit the device's capabilities.
    """
    # Get device info if device_id provided
    device_info = None
    if device_id:
        devices = state.captures.list_devices()
        device_stable_id = _get_stable_device_id(device_id)
        device_info = next(
            (d for d in devices if d["id"] == device_id or _get_stable_device_id(d["id"]) == device_stable_id),
            None
        )

    recipes = []
    for recipe_id, recipe_cfg in state.config.recipes.items():
        channels = [
            RecipeChannelModel(
                offsetHz=ch.offset_hz,
                name=ch.name,
                mode=ch.mode,
                squelchDb=ch.squelch_db,
                enablePocsag=ch.enable_pocsag,
                pocsagBaud=ch.pocsag_baud,
            )
            for ch in recipe_cfg.channels
        ]

        # Get adjusted values if device provided
        if device_info:
            adjusted = _adjust_recipe_for_device(recipe_cfg, device_info)
            sample_rate = adjusted["sampleRate"]
            bandwidth = adjusted["bandwidth"]
            gain = adjusted["gain"]
        else:
            sample_rate = recipe_cfg.sample_rate
            bandwidth = recipe_cfg.bandwidth
            gain = recipe_cfg.gain

        recipes.append(
            RecipeModel(
                id=recipe_id,
                name=recipe_cfg.name,
                description=recipe_cfg.description,
                category=recipe_cfg.category,
                centerHz=recipe_cfg.center_hz,
                sampleRate=sample_rate,
                gain=gain,
                bandwidth=bandwidth,
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
) -> dict[str, Any] | None:
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


@router.get("/captures", response_model=list[CaptureModel])
def list_captures(_: None = Depends(auth_check), state: AppState = Depends(get_state)) -> list[CaptureModel]:
    tm = getattr(state, "trunking_manager", None)
    return [_to_capture_model(c, tm) for c in state.captures.list_captures()]


@router.post("/captures", response_model=CaptureModel)
def create_capture(
    req: CreateCaptureRequest,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
) -> CaptureModel:
    # Validate sample rate against device capabilities before creating
    if req.deviceId:
        try:
            devices = state.captures.list_devices()
            device_stable_id = _get_stable_device_id(req.deviceId)
            device_info = next(
                (d for d in devices if d["id"] == req.deviceId or _get_stable_device_id(d["id"]) == device_stable_id),
                None
            )
            if device_info:
                valid_rates = device_info.get("sample_rates", [])
                if valid_rates and req.sampleRate not in valid_rates:
                    closest = min(valid_rates, key=lambda r: abs(r - req.sampleRate))
                    raise HTTPException(
                        status_code=400,
                        detail=f"Sample rate {req.sampleRate} is not supported by this device. "
                               f"Valid rates: {[int(r) for r in valid_rates]}. Closest: {int(closest)}"
                    )
        except HTTPException:
            raise  # Re-raise validation errors
        except Exception:
            # Device enumeration failed - continue without validation
            # (device will validate on start)
            pass

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

    # Set FFT settings if provided
    if req.fftFps is not None:
        cap.cfg.fft_fps = req.fftFps
    if req.fftMaxFps is not None:
        cap.cfg.fft_max_fps = req.fftMaxFps
    if req.fftSize is not None:
        cap.cfg.fft_size = req.fftSize
    if req.fftAccelerator is not None:
        cap.cfg.fft_accelerator = req.fftAccelerator

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
    default_channel = None
    if req.createDefaultChannel:
        default_channel = state.captures.create_channel(
            cid=cap.cfg.id,
            mode="wbfm",
            offset_hz=0,
            audio_rate=state.config.stream.default_audio_rate,
            squelch_db=-60,
        )

    # Emit state change for WebSocket subscribers
    tm = getattr(state, "trunking_manager", None)
    capture_model = _to_capture_model(cap, tm)
    get_broadcaster().emit_capture_change("created", cap.cfg.id, capture_model.model_dump())
    if default_channel:
        channel_model = _to_channel_model(default_channel)
        get_broadcaster().emit_channel_change("created", default_channel.cfg.id, channel_model.model_dump())

    return capture_model


@router.post("/captures/{cid}/start", response_model=CaptureModel)
async def start_capture(
    cid: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
) -> CaptureModel:
    cap = state.captures.get_capture(cid)
    if cap is None:
        raise HTTPException(status_code=404, detail="Capture not found")

    # Prevent starting a capture that's owned by a trunking system
    # (trunking system controls its own capture lifecycle)
    if cap.trunking_system_id:
        raise HTTPException(
            status_code=409,
            detail=f"Capture is owned by trunking system '{cap.trunking_system_id}'. "
                   f"Use trunking API to control it."
        )

    # Auto-stop other captures using the same device
    # This ensures only one capture per device is running at a time
    target_device_id = cap.requested_device_id or cap.cfg.device_id
    if target_device_id and target_device_id != "auto":
        # Use stable device ID to compare (ignores volatile fields like 'tuner')
        target_stable_id = _get_stable_device_id(target_device_id)
        for other_cap in state.captures.list_captures():
            if other_cap.cfg.id == cid:
                continue  # Skip the capture we're starting
            # Skip captures owned by trunking systems - they manage their own lifecycle
            if other_cap.trunking_system_id:
                continue
            # Check if other capture is using the same device using stable ID
            other_stable_id = _get_stable_device_id(other_cap.cfg.device_id)
            if other_stable_id == target_stable_id and other_cap.state in ("running", "starting"):
                await other_cap.stop()

    cap.start()
    # Auto-start any existing channels so playback works immediately
    for ch in state.captures.list_channels(cid):
        if ch.state != "running":
            ch.start()
            # Emit channel started
            channel_model = _to_channel_model(ch)
            get_broadcaster().emit_channel_change("started", ch.cfg.id, channel_model.model_dump())

    # Emit capture started
    tm = getattr(state, "trunking_manager", None)
    capture_model = _to_capture_model(cap, tm)
    get_broadcaster().emit_capture_change("started", cid, capture_model.model_dump())
    return capture_model


@router.post("/captures/{cid}/stop", response_model=CaptureModel)
async def stop_capture(
    cid: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
) -> CaptureModel:
    cap = state.captures.get_capture(cid)
    if cap is None:
        raise HTTPException(status_code=404, detail="Capture not found")

    # Prevent stopping a capture that's owned by a trunking system
    if cap.trunking_system_id:
        raise HTTPException(
            status_code=409,
            detail=f"Capture is owned by trunking system '{cap.trunking_system_id}'. "
                   f"Use trunking API to stop it."
        )

    await cap.stop()

    # Emit capture stopped
    tm = getattr(state, "trunking_manager", None)
    capture_model = _to_capture_model(cap, tm)
    get_broadcaster().emit_capture_change("stopped", cid, capture_model.model_dump())
    return capture_model


@router.post("/captures/{cid}/restart", response_model=CaptureModel)
async def restart_capture(
    cid: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
) -> CaptureModel:
    """Restart a capture (stop then start).

    Useful for recovering from error states or refreshing the SDR connection.
    """
    cap = state.captures.get_capture(cid)
    if cap is None:
        raise HTTPException(status_code=404, detail="Capture not found")

    tm = getattr(state, "trunking_manager", None)

    # Stop first (if running)
    if cap.state in ("running", "starting", "failed"):
        await cap.stop()
        get_broadcaster().emit_capture_change("stopped", cid, _to_capture_model(cap, tm).model_dump())

    # Brief pause to let SDR device settle
    await asyncio.sleep(0.5)

    # Start again
    cap.start()

    # Auto-start any existing channels
    for ch in state.captures.list_channels(cid):
        if ch.state != "running":
            ch.start()
            get_broadcaster().emit_channel_change("started", ch.cfg.id, _to_channel_model(ch).model_dump())

    # Emit capture started
    capture_model = _to_capture_model(cap, tm)
    get_broadcaster().emit_capture_change("started", cid, capture_model.model_dump())
    return capture_model


@router.get("/captures/{cid}", response_model=CaptureModel)
def get_capture(
    cid: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
) -> CaptureModel:
    cap = state.captures.get_capture(cid)
    if cap is None:
        raise HTTPException(status_code=404, detail="Capture not found")
    tm = getattr(state, "trunking_manager", None)
    return _to_capture_model(cap, tm)


@router.patch("/captures/{cid}", response_model=CaptureModel)
async def update_capture(
    cid: str,
    req: UpdateCaptureRequest,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
) -> CaptureModel:
    import traceback
    logger.info(f"PATCH /captures/{cid} - request: {req}")
    try:
        result = await _update_capture_impl(cid, req, state)
        logger.info(f"PATCH /captures/{cid} - success")
        return result
    except HTTPException as e:
        logger.warning(f"PATCH /captures/{cid} - HTTPException: {e.status_code} {e.detail}")
        raise
    except Exception as e:
        error_msg = f"update_capture failed: {e}\n{traceback.format_exc()}"
        logger.error(f"PATCH /captures/{cid} - {error_msg}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e!s}")


async def _update_capture_impl(cid: str, req: UpdateCaptureRequest, state: AppState) -> CaptureModel:
    cap = state.captures.get_capture(cid)
    if cap is None:
        raise HTTPException(status_code=404, detail="Capture not found")

    # Handle device change if requested
    device_changed = False
    new_device_info = None
    if req.deviceId is not None and req.deviceId != cap.cfg.device_id:
        if cap.state in ("running", "starting"):
            raise HTTPException(
                status_code=400,
                detail="Cannot change device while capture is running. Stop the capture first."
            )

        # Validate new device exists (use stable ID matching to handle volatile fields like 'tuner')
        # Build device list from enumeration + devices from active captures (same as GET /devices)
        devices = []
        seen_stable_ids = set()
        try:
            for d in state.captures.list_devices():
                stable_id = _get_stable_device_id(d["id"])
                if stable_id not in seen_stable_ids:
                    seen_stable_ids.add(stable_id)
                    devices.append(d)
        except Exception:
            pass  # Continue to add devices from captures below
        # Include devices from captures (may not be in enumeration due to USB errors)
        for cap_iter in state.captures.list_captures():
            stable_id = _get_stable_device_id(cap_iter.cfg.device_id)
            if stable_id not in seen_stable_ids:
                seen_stable_ids.add(stable_id)
                devices.append({"id": cap_iter.cfg.device_id})
        req_stable_id = _get_stable_device_id(req.deviceId)
        new_device = next(
            (d for d in devices if d["id"] == req.deviceId or _get_stable_device_id(d["id"]) == req_stable_id),
            None
        )
        if new_device is None:
            raise HTTPException(
                status_code=404,
                detail=f"Device '{req.deviceId}' not found"
            )

        # Update device in config and requested_device_id (used when opening device)
        cap.cfg.device_id = req.deviceId
        cap.requested_device_id = req.deviceId
        # Release cached device so the new one will be opened on next start
        cap.release_device()
        device_changed = True
        new_device_info = new_device
        logger.info(f"Changed capture {cid} device to {req.deviceId}")

    # Validate changes against device constraints before applying
    devices = state.captures.list_devices()
    cap_stable_id = _get_stable_device_id(cap.cfg.device_id)
    device_info = new_device_info or next(
        (d for d in devices if d["id"] == cap.cfg.device_id or _get_stable_device_id(d["id"]) == cap_stable_id),
        None
    )

    # When device changes, auto-adjust parameters that don't fit the new device
    adjusted_params = []
    if device_changed and device_info:
        # Auto-adjust sample rate to closest valid rate
        valid_rates = device_info.get("sample_rates", [])
        if valid_rates and cap.cfg.sample_rate not in valid_rates:
            # Find closest valid rate
            old_rate = cap.cfg.sample_rate
            closest_rate = min(valid_rates, key=lambda r: abs(r - old_rate))
            cap.cfg.sample_rate = closest_rate
            adjusted_params.append(f"sample_rate: {old_rate} -> {closest_rate}")

        # Auto-adjust bandwidth to fit new device range
        bw_min = device_info.get("bandwidth_min")
        bw_max = device_info.get("bandwidth_max")
        if bw_min is not None and bw_max is not None and cap.cfg.bandwidth is not None:
            old_bw = cap.cfg.bandwidth
            new_bw = max(bw_min, min(bw_max, old_bw))
            if new_bw != old_bw:
                cap.cfg.bandwidth = new_bw
                adjusted_params.append(f"bandwidth: {old_bw} -> {new_bw}")

        # Auto-adjust gain to fit new device range
        gain_min = device_info.get("gain_min")
        gain_max = device_info.get("gain_max")
        if gain_min is not None and gain_max is not None and cap.cfg.gain is not None:
            old_gain = cap.cfg.gain
            new_gain = max(gain_min, min(gain_max, old_gain))
            if new_gain != old_gain:
                cap.cfg.gain = new_gain
                adjusted_params.append(f"gain: {old_gain} -> {new_gain}")

        # Auto-adjust antenna if current one isn't available
        valid_antennas = device_info.get("antennas", [])
        if valid_antennas and cap.cfg.antenna and cap.cfg.antenna not in valid_antennas:
            old_ant = cap.cfg.antenna
            cap.cfg.antenna = valid_antennas[0]  # Use first available
            adjusted_params.append(f"antenna: {old_ant} -> {valid_antennas[0]}")

        # Auto-adjust frequency if out of range
        freq_min = device_info.get("freq_min_hz", 0)
        freq_max = device_info.get("freq_max_hz", 6e9)
        if not (freq_min <= cap.cfg.center_hz <= freq_max):
            old_freq = cap.cfg.center_hz
            new_freq = max(freq_min, min(freq_max, old_freq))
            cap.cfg.center_hz = new_freq
            adjusted_params.append(f"center_hz: {old_freq} -> {new_freq}")

        if adjusted_params:
            logger.info(f"Auto-adjusted capture {cid} params for new device: {', '.join(adjusted_params)}")

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

    # Update FFT settings if provided
    if req.fftFps is not None:
        cap.cfg.fft_fps = req.fftFps
    if req.fftMaxFps is not None:
        cap.cfg.fft_max_fps = req.fftMaxFps
    if req.fftSize is not None:
        cap.cfg.fft_size = req.fftSize
    if req.fftAccelerator is not None:
        cap.cfg.fft_accelerator = req.fftAccelerator
        # Reset the FFT backend so it gets re-initialized with new accelerator
        cap._fft_backend = None

    # Use reconfigure method with timeout protection
    try:
        # Add timeout to prevent hanging
        removed_channel_ids = await asyncio.wait_for(
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
        # Also remove channels from CaptureManager's tracking
        for ch_id in removed_channel_ids:
            state.captures._channels.pop(ch_id, None)
        if removed_channel_ids:
            logger.info(f"Removed {len(removed_channel_ids)} out-of-band channels: {removed_channel_ids}")
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Capture reconfiguration timed out. The SDRplay service may be stuck. Try restarting the service."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reconfigure capture: {e!s}"
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

    # Emit state change for WebSocket subscribers
    tm = getattr(state, "trunking_manager", None)
    capture_model = _to_capture_model(cap, tm)
    get_broadcaster().emit_capture_change("updated", cid, capture_model.model_dump())

    return capture_model


@router.delete("/captures/{cid}")
async def delete_capture(
    cid: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
) -> Response:
    # Check if capture exists and is not owned by a trunking system
    cap = state.captures.get_capture(cid)
    if cap is not None and cap.trunking_system_id:
        raise HTTPException(
            status_code=409,
            detail=f"Capture is owned by trunking system '{cap.trunking_system_id}'. "
                   f"Delete the trunking system instead."
        )

    # Get channel IDs before deletion to emit events
    channel_ids = [ch.cfg.id for ch in state.captures.list_channels(cid)]

    await state.captures.delete_capture(cid)

    # Emit deletion events
    for chan_id in channel_ids:
        get_broadcaster().emit_channel_change("deleted", chan_id, None)
    get_broadcaster().emit_capture_change("deleted", cid, None)

    return Response(status_code=204)


@router.get("/captures/{cid}/channels", response_model=list[ChannelModel])
def list_channels(
    cid: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
) -> list[ChannelModel]:
    chans = state.captures.list_channels(cid)
    return [_to_channel_model(ch) for ch in chans]


@router.post("/captures/{cid}/channels", response_model=ChannelModel)
def create_channel(
    cid: str,
    req: CreateChannelRequest,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
) -> ChannelModel:
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

    # Apply SSB settings if provided
    if req.ssbMode is not None:
        ch.cfg.ssb_mode = req.ssbMode
    if req.ssbBfoOffsetHz is not None:
        ch.cfg.ssb_bfo_offset_hz = req.ssbBfoOffsetHz
    if req.ssbBandpassLowHz is not None:
        ch.cfg.ssb_bandpass_low_hz = req.ssbBandpassLowHz
    if req.ssbBandpassHighHz is not None:
        ch.cfg.ssb_bandpass_high_hz = req.ssbBandpassHighHz
    if req.enableSsbBandpass is not None:
        ch.cfg.enable_ssb_bandpass = req.enableSsbBandpass

    # Generate auto_name using frequency recognition
    cap = state.captures.get_capture(cid)
    if cap is not None:
        namer = get_frequency_namer()
        ch.cfg.auto_name = namer.suggest_channel_name(cap.cfg.center_hz, ch.cfg.offset_hz)

    # Auto-start channel if capture is already running so audio begins immediately
    if cap is not None and cap.state == "running" and ch.state != "running":
        ch.start()

    # Emit state change for WebSocket subscribers
    channel_model = _to_channel_model(ch)
    get_broadcaster().emit_channel_change("created", ch.cfg.id, channel_model.model_dump())

    return channel_model


@router.post("/channels/{chan_id}/start", response_model=ChannelModel)
def start_channel(
    chan_id: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
) -> ChannelModel:
    ch = state.captures.get_channel(chan_id)
    if ch is None:
        raise HTTPException(status_code=404, detail="Channel not found")
    ch.start()

    # Emit state change for WebSocket subscribers
    channel_model = _to_channel_model(ch)
    get_broadcaster().emit_channel_change("started", chan_id, channel_model.model_dump())

    return channel_model


@router.post("/channels/{chan_id}/stop", response_model=ChannelModel)
def stop_channel(
    chan_id: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
) -> ChannelModel:
    ch = state.captures.get_channel(chan_id)
    if ch is None:
        raise HTTPException(status_code=404, detail="Channel not found")
    ch.stop()

    # Emit state change for WebSocket subscribers
    channel_model = _to_channel_model(ch)
    get_broadcaster().emit_channel_change("stopped", chan_id, channel_model.model_dump())

    return channel_model


@router.get("/channels/{chan_id}", response_model=ChannelModel)
def get_channel(
    chan_id: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
) -> ChannelModel:
    ch = state.captures.get_channel(chan_id)
    if ch is None:
        raise HTTPException(status_code=404, detail="Channel not found")
    return ChannelModel(
        id=ch.cfg.id,
        captureId=ch.cfg.capture_id,
        mode=cast(Literal["wbfm", "nbfm", "am", "ssb", "raw", "p25", "dmr"], ch.cfg.mode),
        state=cast(Literal["created", "running", "stopped"], ch.state),
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
        ssbBfoOffsetHz=ch.cfg.ssb_bfo_offset_hz,
        enableAgc=ch.cfg.enable_agc,
        agcTargetDb=ch.cfg.agc_target_db,
        agcAttackMs=ch.cfg.agc_attack_ms,
        agcReleaseMs=ch.cfg.agc_release_ms,
        enableNoiseBlanker=ch.cfg.enable_noise_blanker,
        noiseBlankerThresholdDb=ch.cfg.noise_blanker_threshold_db,
        notchFrequencies=ch.cfg.notch_frequencies,
        enableNoiseReduction=ch.cfg.enable_noise_reduction,
        noiseReductionDb=ch.cfg.noise_reduction_db,
        rdsData=_to_rds_data_model(ch.rds_data),
    )


@router.patch("/channels/{chan_id}", response_model=ChannelModel)
def update_channel(
    chan_id: str,
    req: UpdateChannelRequest,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
) -> ChannelModel:
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
    if req.ssbBfoOffsetHz is not None:
        ch.cfg.ssb_bfo_offset_hz = req.ssbBfoOffsetHz
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

    # Emit state change for WebSocket subscribers
    channel_model = _to_channel_model(ch)
    get_broadcaster().emit_channel_change("updated", chan_id, channel_model.model_dump())

    return channel_model


@router.delete("/channels/{chan_id}")
def delete_channel(
    chan_id: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
) -> Response:
    state.captures.delete_channel(chan_id)

    # Emit state change for WebSocket subscribers
    get_broadcaster().emit_channel_change("deleted", chan_id, None)

    return Response(status_code=204)


# ==============================================================================
# Signal monitoring endpoints (for Claude skills)
# ==============================================================================

def _rssi_to_s_units(rssi_db: float | None) -> str | None:
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
) -> SpectrumSnapshotModel:
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


@router.get("/captures/{cid}/classified-channels", response_model=ClassifiedChannelsResponse)
def get_classified_channels(
    cid: str,
    reset: bool = False,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
) -> ClassifiedChannelsResponse:
    """Get classified channels (control vs voice) based on spectrum analysis.

    The classifier accumulates spectrum data over time and identifies:
    - Control channels: constant power, low variance (always transmitting)
    - Voice channels: intermittent power, high variance (on/off during calls)
    - Variable: moderate variance (could be either)

    Query params:
    - reset: If true, reset the classifier and start fresh collection
    """
    cap = state.captures.get_capture(cid)
    if cap is None:
        raise HTTPException(status_code=404, detail="Capture not found")

    if reset:
        cap._channel_classifier.reset()

    # Get classification results
    classified = cap._channel_classifier.classify()
    status = cap._channel_classifier.get_status()

    return ClassifiedChannelsResponse(
        channels=[
            ClassifiedChannelModel(
                freqHz=ch.freq_hz,
                powerDb=ch.power_db,
                stdDevDb=ch.std_dev_db,
                channelType=ch.channel_type,
            )
            for ch in classified
        ],
        status=status,
    )


@router.get("/channels/{chan_id}/metrics/extended", response_model=ExtendedMetricsModel)
def get_channel_extended_metrics(
    chan_id: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
) -> ExtendedMetricsModel:
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
) -> MetricsHistoryModel:
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


@router.get("/channels/{chan_id}/decode/pocsag", response_model=list[POCSAGMessageModel])
def get_channel_pocsag_messages(
    chan_id: str,
    limit: int = 50,
    since: float | None = None,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
) -> list[POCSAGMessageModel]:
    """Get decoded POCSAG pager messages from an NBFM channel.

    Args:
        chan_id: Channel ID
        limit: Maximum number of messages to return (default 50)
        since: Only return messages after this Unix timestamp (for polling)

    Returns:
        List of decoded POCSAG messages (most recent first)
    """
    ch = state.captures.get_channel(chan_id)
    if ch is None:
        raise HTTPException(status_code=404, detail="Channel not found")

    # Get messages from the channel's POCSAG decoder
    messages = ch.get_pocsag_messages(limit=limit, since_timestamp=since)

    # Look up aliases from config
    pocsag_aliases = state.config.pocsag_aliases

    return [
        POCSAGMessageModel(
            address=msg["address"],
            function=msg["function"],
            messageType=msg["messageType"],
            message=msg["message"],
            timestamp=msg["timestamp"],
            baudRate=msg["baudRate"],
            alias=pocsag_aliases.get(msg["address"]),
        )
        for msg in messages
    ]


@router.websocket("/stream/captures/{cid}/iq")
async def stream_capture_iq(websocket: WebSocket, cid: str) -> None:
    # Auth: optional token via header or query `token`
    app_state: AppState = websocket.app.state.app_state
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
async def stream_capture_spectrum(websocket: WebSocket, cid: str) -> None:
    """Stream FFT/spectrum data for waterfall/spectrum analyzer display.

    Only calculates FFT when there are active subscribers for efficiency.
    """
    app_state: AppState = websocket.app.state.app_state
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


@router.get("/stream/channels/{chan_id}.pcm", response_model=None)
async def stream_channel_http(
    request: Request,
    chan_id: str,
    format: str = "pcm16",
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
) -> StreamingResponse:
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

    async def audio_generator() -> AsyncGenerator[bytes, None]:
        q = await ch.subscribe_audio(format=format)
        logger.info(f"HTTP stream started for channel {chan_id}, format={format}, client={request.client}")
        packet_count = 0
        try:
            while True:
                # Check disconnect every 10 packets (~200ms) even if queue has data
                if packet_count % 10 == 0 and await request.is_disconnected():
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


@router.get("/stream/channels/{chan_id}.mp3", response_model=None)
async def stream_channel_mp3(
    request: Request,
    chan_id: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
) -> StreamingResponse:
    """Stream channel audio as MP3."""
    ch = state.captures.get_channel(chan_id)
    if ch is None:
        raise HTTPException(status_code=404, detail="Channel not found")

    async def audio_generator() -> AsyncGenerator[bytes, None]:
        q = await ch.subscribe_audio(format="mp3")
        logger.info(f"MP3 stream started for channel {chan_id}, client={request.client}")
        packet_count = 0
        try:
            while True:
                if packet_count % 10 == 0 and await request.is_disconnected():
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


@router.get("/stream/channels/{chan_id}.opus", response_model=None)
async def stream_channel_opus(
    request: Request,
    chan_id: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
) -> StreamingResponse:
    """Stream channel audio as Opus."""
    ch = state.captures.get_channel(chan_id)
    if ch is None:
        raise HTTPException(status_code=404, detail="Channel not found")

    async def audio_generator() -> AsyncGenerator[bytes, None]:
        q = await ch.subscribe_audio(format="opus")
        logger.info(f"Opus stream started for channel {chan_id}, client={request.client}")
        packet_count = 0
        try:
            while True:
                if packet_count % 10 == 0 and await request.is_disconnected():
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


@router.get("/stream/channels/{chan_id}.aac", response_model=None)
async def stream_channel_aac(
    request: Request,
    chan_id: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
) -> StreamingResponse:
    """Stream channel audio as AAC."""
    ch = state.captures.get_channel(chan_id)
    if ch is None:
        raise HTTPException(status_code=404, detail="Channel not found")

    async def audio_generator() -> AsyncGenerator[bytes, None]:
        q = await ch.subscribe_audio(format="aac")
        logger.info(f"AAC stream started for channel {chan_id}, client={request.client}")
        packet_count = 0
        try:
            while True:
                if packet_count % 10 == 0 and await request.is_disconnected():
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
async def stream_channel_audio(websocket: WebSocket, chan_id: str, format: str = "pcm16") -> None:
    """Stream channel audio with configurable format.

    Query parameters:
        format: Audio format - "pcm16" (16-bit signed PCM) or "f32" (32-bit float). Default: pcm16
    """
    app_state: AppState = websocket.app.state.app_state
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
# Health/Error Stream
# ==============================================================================

@router.websocket("/stream/health")
async def stream_health(websocket: WebSocket) -> None:
    """Real-time health and error stream.

    Sends JSON messages:
    - {"type": "error", "event": {...}} - When errors occur
    - {"type": "stats", "data": {...}} - Every 2 seconds with aggregate stats

    Error event format:
    {
        "type": "error",
        "event": {
            "type": "iq_overflow" | "audio_drop" | "device_retry",
            "capture_id": "...",
            "channel_id": "..." | null,
            "timestamp": 1234567890.123,
            "count": 1,
            "details": {...}
        }
    }

    Stats format:
    {
        "type": "stats",
        "data": {
            "iq_overflow": {"total": 0, "lastMinute": 0, "rate": 0.0},
            "audio_drop": {"total": 0, "lastMinute": 0, "rate": 0.0},
            "device_retry": {"total": 0, "lastMinute": 0, "rate": 0.0}
        }
    }
    """
    from .error_tracker import get_error_tracker

    # Health stream is diagnostic data - no auth required
    await websocket.accept()

    tracker = get_error_tracker()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)

    def on_error(event: Any) -> None:
        try:
            queue.put_nowait({"type": "error", "event": event.to_dict()})
        except asyncio.QueueFull:
            pass  # Drop if queue is full

    unsubscribe = tracker.subscribe(on_error)

    try:
        while True:
            # Send error events as they come, or stats every 2 seconds
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=2.0)
                await websocket.send_json(msg)
            except asyncio.TimeoutError:
                # Send periodic stats
                stats = tracker.get_stats()
                await websocket.send_json({
                    "type": "stats",
                    "data": {
                        error_type: stat.to_dict()
                        for error_type, stat in stats.items()
                    },
                })
    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:
        raise
    finally:
        unsubscribe()


# ==============================================================================
# State Stream (replaces polling)
# ==============================================================================

@router.websocket("/stream/state")
async def stream_state(websocket: WebSocket) -> None:
    """Real-time state change stream for captures, channels, and scanners.

    Replaces HTTP polling with WebSocket push for state updates.

    Sends JSON messages:
    - {"type": "capture", "action": "created|updated|deleted|started|stopped", "id": "...", "data": {...}}
    - {"type": "channel", "action": "created|updated|deleted|started|stopped", "id": "...", "data": {...}}
    - {"type": "scanner", "action": "created|updated|deleted|started|stopped", "id": "...", "data": {...}}
    - {"type": "snapshot", "captures": [...], "channels": [...], "scanners": [...]}

    On connect, sends a full state snapshot, then pushes incremental changes.
    """
    from .state_broadcaster import get_broadcaster

    # State stream is for UI updates - no auth required (same as health stream)
    await websocket.accept()
    logger.info("WebSocket /api/v1/stream/state connected")

    app_state: AppState = websocket.app.state.app_state
    broadcaster = get_broadcaster()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)

    def on_state_change(change: Any) -> None:
        try:
            queue.put_nowait(change.to_dict())
        except asyncio.QueueFull:
            pass  # Drop if queue is full

    unsubscribe = broadcaster.subscribe(on_state_change)

    try:
        # Send initial full state snapshot
        tm = getattr(app_state, "trunking_manager", None)
        captures = [_to_capture_model(c, tm).model_dump() for c in app_state.captures.list_captures()]
        channels = [_to_channel_model(ch).model_dump() for ch in app_state.captures.list_channels()]
        scanners = [
            _to_scanner_model(sid, s).model_dump()
            for sid, s in app_state.scanners.items()
        ]

        await websocket.send_json({
            "type": "snapshot",
            "captures": captures,
            "channels": channels,
            "scanners": scanners,
        })

        # Then stream incremental changes with periodic keepalive
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_json(msg)
            except asyncio.TimeoutError:
                # Send keepalive ping
                await websocket.send_json({"type": "ping", "timestamp": time.time()})
    except WebSocketDisconnect:
        logger.info("WebSocket /api/v1/stream/state disconnected")
    except asyncio.CancelledError:
        raise
    finally:
        unsubscribe()


# ==============================================================================
# System Metrics and Log Stream
# ==============================================================================


@router.websocket("/stream/system")
async def stream_system(websocket: WebSocket) -> None:
    """Real-time system metrics and log stream.

    Sends JSON messages:
    - {"type": "metrics", "system": {...}, "captures": [...]} - Every 1 second
    - {"type": "log", "entry": {...}} - When log entries occur
    - {"type": "logs_snapshot", "entries": [...]} - Initial snapshot on connect
    - {"type": "error", "event": {...}} - When errors occur

    System metrics format:
    {
        "type": "metrics",
        "system": {
            "timestamp": 1234567890.123,
            "cpuPercent": 45.2,
            "cpuPerCore": [40.0, 50.0, 45.0, 46.0],
            "memoryUsedMb": 4096.5,
            "memoryTotalMb": 16384.0,
            "memoryPercent": 25.0,
            "temperatures": {"CPU": 65.0}
        },
        "captures": [
            {
                "captureId": "c1",
                "deviceId": "...",
                "state": "running",
                "iqOverflowCount": 0,
                "iqOverflowRate": 0.0,
                "channelCount": 3,
                "totalSubscribers": 2,
                "totalDrops": 0,
                "perfLoopMs": 5.2,
                "perfDspMs": 2.1,
                "perfFftMs": 1.5
            }
        ]
    }

    Log entry format:
    {
        "type": "log",
        "entry": {
            "timestamp": 1234567890.123,
            "level": "INFO",
            "loggerName": "wavecapsdr.capture",
            "message": "Capture c1 started"
        }
    }
    """
    from .error_tracker import get_error_tracker
    from .log_streamer import LogEntry, get_log_streamer
    from .system_metrics import get_capture_metrics, get_system_metrics

    # System stream is diagnostic data - no auth required
    await websocket.accept()

    state: AppState = websocket.app.state.app_state
    tracker = get_error_tracker()
    log_streamer = get_log_streamer()

    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=500)

    # Subscribe to errors
    def on_error(event: Any) -> None:
        try:
            queue.put_nowait({"type": "error", "event": event.to_dict()})
        except asyncio.QueueFull:
            pass

    # Subscribe to logs
    def on_log(entry: LogEntry) -> None:
        try:
            queue.put_nowait({"type": "log", "entry": entry.to_dict()})
        except asyncio.QueueFull:
            pass

    unsubscribe_error = tracker.subscribe(on_error)
    unsubscribe_log = log_streamer.subscribe(on_log)

    try:
        # Send initial snapshot of recent logs
        recent_logs = log_streamer.get_recent(100)
        await websocket.send_json({
            "type": "logs_snapshot",
            "entries": [e.to_dict() for e in recent_logs],
        })

        while True:
            # Drain queue with 1 second timeout for metrics
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=1.0)
                await websocket.send_json(msg)
            except asyncio.TimeoutError:
                # Send periodic metrics every second
                system = get_system_metrics()
                captures = get_capture_metrics(state)
                await websocket.send_json({
                    "type": "metrics",
                    "system": system.to_dict(),
                    "captures": [c.to_dict() for c in captures],
                })
    except WebSocketDisconnect:
        logger.info("WebSocket /api/v1/stream/system disconnected")
    except asyncio.CancelledError:
        raise
    finally:
        unsubscribe_error()
        unsubscribe_log()


@router.get("/system/metrics")
def get_system_metrics_endpoint(
    request: Request,
) -> dict[str, Any]:
    """Get current system and capture metrics (one-time fetch)."""
    from .system_metrics import get_capture_metrics, get_system_metrics

    state: AppState = request.app.state.app_state
    system = get_system_metrics()
    captures = get_capture_metrics(state)

    return {
        "system": system.to_dict(),
        "captures": [c.to_dict() for c in captures],
    }


# ==============================================================================
# Scanner endpoints
# ==============================================================================

from .scanner import ScanConfig, ScanMode, ScannerService


def _to_scanner_model(scanner_id: str, scanner: ScannerService) -> ScannerModel:
    """Convert ScannerService to ScannerModel."""
    return ScannerModel(
        id=scanner_id,
        captureId=scanner.capture_id,
        state=scanner.status.state.value,
        currentFrequency=scanner.status.current_frequency,
        currentIndex=scanner.status.current_index,
        scanList=scanner.config.scan_list,
        mode=scanner.config.mode.value,
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
) -> ScannerModel:
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
    async def update_frequency(freq_hz: float) -> None:
        try:
            cap = state.captures.get_capture(req.captureId)
            if cap:
                await cap.reconfigure(center_hz=freq_hz)
        except Exception as e:
            logger.error(f"Scanner failed to update frequency: {e}")

    scanner.set_update_callback(update_frequency)

    # Set up RSSI callback to get current RSSI from capture
    def get_rssi() -> float:
        # Get first channel's RSSI as a proxy for activity
        channels = list(capture._channels.values())
        if channels:
            rssi = channels[0].rssi_db
            return rssi if rssi is not None else -120.0
        return -120.0  # No signal

    scanner.set_rssi_callback(get_rssi)

    # Store scanner
    state.scanners[scanner_id] = scanner

    return _to_scanner_model(scanner_id, scanner)


@router.get("/scanners", response_model=list[ScannerModel])
def list_scanners(_: None = Depends(auth_check), state: AppState = Depends(get_state)) -> list[ScannerModel]:
    """List all scanners."""
    return [_to_scanner_model(sid, scanner) for sid, scanner in state.scanners.items()]


@router.get("/scanners/{sid}", response_model=ScannerModel)
def get_scanner(
    sid: str,
    _: None = Depends(auth_check),
    state: AppState = Depends(get_state),
) -> ScannerModel:
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
) -> ScannerModel:
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
) -> None:
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
) -> ScannerModel:
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
) -> ScannerModel:
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
) -> ScannerModel:
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
) -> ScannerModel:
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
) -> ScannerModel:
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
) -> ScannerModel:
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
) -> ScannerModel:
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
) -> ScannerModel:
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
) -> ScannerModel:
    """Clear all lockouts."""
    scanner = state.scanners.get(sid)
    if scanner is None:
        raise HTTPException(status_code=404, detail=f"Scanner {sid} not found")

    scanner.clear_all_lockouts()
    return _to_scanner_model(sid, scanner)


# ==============================================================================
# Frontend error logging endpoint
# ==============================================================================

import json
from pathlib import Path

from pydantic import BaseModel


class FrontendErrorReport(BaseModel):
    """Error report from the frontend JavaScript application."""
    level: str = "error"  # error, warn, info, debug
    message: str
    stack: str | None = None
    componentStack: str | None = None  # React error boundary stack
    url: str | None = None
    userAgent: str | None = None
    timestamp: float | None = None
    context: dict[str, Any] | None = None  # Additional context (component name, props, etc.)


class FrontendLogEntry(BaseModel):
    """Single log entry from frontend logger."""
    timestamp: str
    level: str
    message: str
    data: dict[str, Any] | None = None
    source: str | None = None
    stack: str | None = None


class FrontendLogBatch(BaseModel):
    """Batch of log entries from frontend."""
    entries: list[FrontendLogEntry]


# In-memory log buffer for recent frontend errors (for the production engineer agent)
_frontend_error_log: list[dict[str, Any]] = []
_MAX_FRONTEND_ERRORS = 500

# File path for persistent frontend logs (accessible to Claude for debugging)
_FRONTEND_LOG_FILE = Path(__file__).parent.parent / "logs" / "frontend.log"


def _ensure_log_dir() -> None:
    """Ensure the logs directory exists."""
    _FRONTEND_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def _write_to_log_file(entry: dict[str, Any]) -> None:
    """Append a log entry to the frontend log file."""
    try:
        _ensure_log_dir()
        with open(_FRONTEND_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
        # Rotate log file if too large (> 1MB)
        if _FRONTEND_LOG_FILE.stat().st_size > 1_000_000:
            _rotate_log_file()
    except Exception as e:
        logger.warning(f"Failed to write frontend log to file: {e}")


def _rotate_log_file() -> None:
    """Rotate the log file, keeping only the last 500 lines."""
    try:
        with open(_FRONTEND_LOG_FILE) as f:
            lines = f.readlines()
        with open(_FRONTEND_LOG_FILE, "w") as f:
            f.writelines(lines[-500:])
    except Exception as e:
        logger.warning(f"Failed to rotate frontend log file: {e}")


def _process_log_entry(entry: dict[str, Any], request: Request) -> dict[str, Any]:
    """Process and store a single log entry."""
    import time

    log_entry = {
        "level": entry.get("level", "info"),
        "message": entry.get("message", ""),
        "stack": entry.get("stack"),
        "componentStack": entry.get("componentStack"),
        "url": entry.get("url") or str(request.url),
        "userAgent": entry.get("userAgent") or request.headers.get("user-agent"),
        "timestamp": entry.get("timestamp") or time.time(),
        "context": entry.get("context") or entry.get("data"),
        "clientIp": request.client.host if request.client else None,
        "source": entry.get("source", "frontend"),
    }

    # Log to Python logger
    level = log_entry["level"]
    log_msg = f"[FRONTEND {level.upper()}] {log_entry['message']}"
    if log_entry.get("stack"):
        log_msg += f"\nStack: {log_entry['stack']}"

    if level == "error":
        logger.error(log_msg)
    elif level == "warn":
        logger.warning(log_msg)
    elif level == "debug":
        logger.debug(log_msg)
    else:
        logger.info(log_msg)

    # Store in memory buffer
    _frontend_error_log.append(log_entry)
    if len(_frontend_error_log) > _MAX_FRONTEND_ERRORS:
        _frontend_error_log.pop(0)

    # Write to file for Claude access
    _write_to_log_file(log_entry)

    return log_entry


@router.post("/logs")
def log_frontend_batch(
    batch: FrontendLogBatch,
    request: Request,
) -> dict[str, Any]:
    """Receive batch of log entries from the frontend logger.

    This endpoint receives multiple log entries at once for efficiency.
    Logs are stored in memory and written to a file for debugging.
    """
    processed = 0
    for entry in batch.entries:
        _process_log_entry(entry.model_dump(), request)
        processed += 1

    return {"status": "logged", "count": processed}


@router.post("/log/frontend")
def log_frontend_error(
    report: FrontendErrorReport,
    request: Request,
) -> dict[str, str]:
    """Receive and log errors from the frontend JavaScript application.

    This endpoint allows the frontend to report JavaScript errors, React
    component crashes, and other issues to the server for centralized logging.
    Errors are stored in memory for retrieval by debugging tools.
    """
    _process_log_entry(report.model_dump(), request)
    return {"status": "logged"}


@router.get("/log/frontend")
def get_frontend_errors(
    limit: int = 50,
    level: str | None = None,
    since: float | None = None,
    _: None = Depends(auth_check),
) -> list[dict[str, Any]]:
    """Retrieve recent frontend errors from the log buffer.

    Args:
        limit: Maximum number of errors to return (default 50)
        level: Filter by level (error, warn, info)
        since: Only return errors after this Unix timestamp

    Returns:
        List of error log entries (most recent first)
    """
    results = _frontend_error_log.copy()

    # Apply filters
    if level:
        results = [e for e in results if e["level"] == level]
    if since:
        results = [e for e in results if (e.get("timestamp") or 0) > since]

    # Sort by timestamp descending and limit
    results.sort(key=lambda e: e.get("timestamp") or 0, reverse=True)
    return results[:limit]


@router.delete("/log/frontend")
def clear_frontend_errors(_: None = Depends(auth_check)) -> dict[str, str]:
    """Clear the frontend error log buffer."""
    _frontend_error_log.clear()
    return {"status": "cleared"}
