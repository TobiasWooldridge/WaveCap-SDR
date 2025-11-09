from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import AppConfig
from .api import router as api_router
from .state import AppState


def create_app(config: AppConfig, config_path: str | None = None) -> FastAPI:
    app = FastAPI(title="WaveCap-SDR", version="0.1.0")

    app.state.app_state = AppState.from_config(config, config_path)

    @app.on_event("startup")
    async def startup_event():
        """Auto-start configured captures on server startup.

        If no captures are configured, initialize a default capture so the UI
        has something to show. The default capture is created but not started.
        """
        app_state: AppState = app.state.app_state

        created_any = False

        # Create and start captures explicitly listed in config
        for cap_cfg in config.captures:
            preset_name = cap_cfg.preset
            preset = config.presets.get(preset_name)
            if preset is None:
                print(f"Warning: Preset '{preset_name}' not found, skipping capture")
                continue

            try:
                # Create the capture
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

                # Create channels for each offset
                for i, offset_hz in enumerate(preset.offsets):
                    ch = app_state.captures.create_channel(
                        cid=cap.cfg.id,
                        mode="wbfm",
                        offset_hz=offset_hz,
                        audio_rate=config.stream.default_audio_rate,
                        squelch_db=preset.squelch_db,
                    )
                    ch.start()

                # Start the capture (only for configured ones)
                cap.start()

                # Track the preset for this capture for persistence
                app_state.capture_presets[cap.cfg.id] = preset_name

                device_info = cap_cfg.device_id or "any device"
                print(f"Auto-started capture '{cap.cfg.id}' with preset '{preset_name}' on {device_info}")
                created_any = True
            except Exception as e:
                print(f"Failed to auto-start capture with preset '{preset_name}': {e}")

        # If no captures were configured/created, initialize a default capture (do not start)
        if not created_any:
            try:
                # Prefer the first preset if available (e.g., 'kexp')
                preset_name = next(iter(config.presets.keys()), None)
                preset = config.presets.get(preset_name) if preset_name else None

                # Choose device if one is available
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

                default_channels = 0
                if preset:
                    for offset_hz in preset.offsets:
                        ch = app_state.captures.create_channel(
                            cid=cap.cfg.id,
                            mode="wbfm",
                            offset_hz=offset_hz,
                            audio_rate=config.stream.default_audio_rate,
                            squelch_db=preset.squelch_db,
                        )
                        default_channels += 1

                if preset_name:
                    # Track preset for potential persistence of later changes
                    app_state.capture_presets[cap.cfg.id] = preset_name

                print(
                    f"Initialized default capture '{cap.cfg.id}'"
                    f" (center={center_hz:.0f} Hz, rate={sample_rate} Hz)"
                    f" using device {device_id or 'auto'}; not started."
                    f" Channels seeded: {default_channels}.",
                    flush=True,
                )
            except Exception as e:
                print(f"Warning: Failed to initialize default capture: {e}", flush=True)

    app.include_router(api_router, prefix="/api/v1")

    # Serve static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        # Mount assets directory at /assets/ for React build
        assets_dir = static_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        # Keep /static mount for backward compatibility
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/favicon.svg")
    def favicon():
        """Serve the favicon."""
        favicon_path = static_dir / "favicon.svg"
        if favicon_path.exists():
            return FileResponse(favicon_path, media_type="image/svg+xml")
        return {"message": "Favicon not found"}

    @app.get("/")
    def root():
        """Serve the React app."""
        index_path = static_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return {"message": "WaveCap-SDR API", "docs": "/docs"}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
