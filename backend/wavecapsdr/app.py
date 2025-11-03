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
        """Auto-start configured captures on server startup."""
        app_state: AppState = app.state.app_state

        # Create and start captures from config
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

                # Start the capture
                cap.start()

                # Track the preset for this capture for persistence
                app_state.capture_presets[cap.cfg.id] = preset_name

                device_info = cap_cfg.device_id or "any device"
                print(f"Auto-started capture '{cap.cfg.id}' with preset '{preset_name}' on {device_info}")
            except Exception as e:
                print(f"Failed to auto-start capture with preset '{preset_name}': {e}")

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

