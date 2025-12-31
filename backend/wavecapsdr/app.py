from __future__ import annotations

import logging
import logging.handlers
import inspect
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, AsyncIterator, Callable, TextIO, cast

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
import slowapi.extension as slowapi_extension
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .api import router as api_router
from .config import AppConfig
from .device_namer import generate_capture_name, get_device_nickname
from .mcp_server import router as mcp_router
from .state import AppState
from .trunking.api import router as trunking_router

# Work around slowapi using deprecated asyncio.iscoroutinefunction on Python 3.14+.
slowapi_asyncio = cast(Any, getattr(slowapi_extension, "asyncio", None))
if slowapi_asyncio is not None:
    slowapi_asyncio.iscoroutinefunction = inspect.iscoroutinefunction


class SafeStreamHandler(logging.StreamHandler[TextIO]):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            super().emit(record)
        except ValueError:
            pass


def cleanup_orphan_sdrplay_workers() -> None:
    """Kill any orphaned SDRplay worker processes from previous crashes.

    These are multiprocessing workers that weren't cleaned up when the parent
    process crashed or was forcefully terminated. They show up as high-CPU
    processes with PPID=1 (inherited by init).
    """
    import os
    import subprocess

    try:
        # Find Python multiprocessing workers that are orphaned (PPID=1)
        result = subprocess.run(
            ["ps", "-eo", "pid,ppid,command"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        orphan_pids = []
        for line in result.stdout.splitlines():
            # Look for multiprocessing spawn workers with PPID=1
            if "multiprocessing.spawn" in line and "spawn_main" in line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        pid = int(parts[0])
                        ppid = int(parts[1])
                        if ppid == 1:  # Orphaned (parent is init)
                            orphan_pids.append(pid)
                    except ValueError:
                        continue

        if orphan_pids:
            logging.warning(f"Found {len(orphan_pids)} orphaned worker processes: {orphan_pids}")
            for pid in orphan_pids:
                try:
                    os.kill(pid, 9)  # SIGKILL
                    logging.info(f"Killed orphaned worker process {pid}")
                except ProcessLookupError:
                    pass  # Already dead
                except PermissionError:
                    logging.warning(f"Permission denied killing process {pid}")
            logging.info(f"Cleaned up {len(orphan_pids)} orphaned worker processes")
    except Exception as e:
        logging.warning(f"Failed to clean up orphaned workers: {e}")


def setup_file_logging() -> None:
    """Configure file-based logging with rotation.

    Logs to backend/logs/wavecapsdr.log with 5MB rotation, 3 backups.
    """
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "wavecapsdr.log"

    # Create rotating file handler (5MB max, 3 backups = 20MB total max)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    # Also add console handler for INFO and above
    console_handler = SafeStreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '[%(levelname)s] %(name)s: %(message)s'
    ))
    root_logger.addHandler(console_handler)

    logging.info("File logging initialized: %s", log_file)

# Global DSP thread pool executor for CPU-intensive audio processing
# This keeps DSP work off the main asyncio event loop to prevent HTTP starvation
_dsp_executor: ThreadPoolExecutor | None = None


def get_dsp_executor() -> ThreadPoolExecutor:
    """Get the global DSP thread pool executor.

    Sized for 3 SDRs with multiple channels each. NumPy/SciPy release the GIL
    during heavy computation, enabling true parallelism in thread pools.
    """
    global _dsp_executor
    if _dsp_executor is None:
        import os
        # Use 8 workers or CPU count + 2, whichever is smaller
        # This handles 3 SDRs with 4+ channels each efficiently
        max_workers = min(8, (os.cpu_count() or 4) + 2)
        _dsp_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="DSP-")
    return _dsp_executor


def shutdown_dsp_executor() -> None:
    """Shutdown the DSP executor on application exit."""
    global _dsp_executor
    if _dsp_executor is not None:
        _dsp_executor.shutdown(wait=True, cancel_futures=True)
        _dsp_executor = None


def create_app(config: AppConfig, config_path: str | None = None) -> FastAPI:
    # Setup file logging before anything else
    setup_file_logging()

    # Install log streamer handler for real-time log streaming
    from .log_streamer import get_log_streamer

    get_log_streamer().install_handler()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Auto-start configured captures on server startup.

        If no captures are configured, initialize a default capture so the UI
        has something to show. The default capture is created but not started.
        """
        # Clean up any orphaned worker processes from previous crashes
        cleanup_orphan_sdrplay_workers()

        app_state: AppState = app.state.app_state

        # Start P25 trunking manager (already created in AppState.from_config)
        await app_state.trunking_manager.start()
        logging.info("P25 TrunkingManager started")

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

                # Generate auto_name using device shorthand
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

                # Create channels for each offset
                for _i, offset_hz in enumerate(preset.offsets):
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
                default_preset_name: str = next(iter(config.presets.keys()), "")
                preset = config.presets.get(default_preset_name) if default_preset_name else None

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

                # Generate auto_name using device shorthand
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

                if default_preset_name:
                    # Track preset for potential persistence of later changes
                    app_state.capture_presets[cap.cfg.id] = default_preset_name

                print(
                    f"Initialized default capture '{cap.cfg.id}'"
                    f" (center={center_hz:.0f} Hz, rate={sample_rate} Hz)"
                    f" using device {device_id or 'auto'}; not started."
                    f" Channels seeded: {default_channels}.",
                    flush=True,
                )
            except Exception as e:
                print(f"Warning: Failed to initialize default capture: {e}", flush=True)

        try:
            yield
        finally:
            try:
                await app_state.trunking_manager.stop()
            except Exception as e:
                logging.warning("Error stopping trunking manager during shutdown: %s", e)

            for capture in app_state.captures.list_captures():
                try:
                    await app_state.captures.stop_capture(capture.cfg.id)
                except Exception as e:
                    logging.warning("Error stopping capture %s: %s", capture.cfg.id, e)

            shutdown_dsp_executor()

    app = FastAPI(title="WaveCap-SDR", version="0.1.0", lifespan=lifespan)

    # Configure CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins for local development
        allow_credentials=True,
        allow_methods=["*"],  # Allow all HTTP methods
        allow_headers=["*"],  # Allow all headers
        expose_headers=["*"],  # Expose all headers to the client
    )

    # Configure rate limiting
    limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])
    app.state.limiter = limiter
    rate_limit_handler = cast(Callable[[Request, Exception], Response], _rate_limit_exceeded_handler)
    app.add_exception_handler(RateLimitExceeded, rate_limit_handler)

    app.state.app_state = AppState.from_config(config, config_path)

    app.include_router(api_router, prefix="/api/v1")
    app.include_router(trunking_router, prefix="/api/v1")
    app.include_router(mcp_router, prefix="/api/v1")

    # Serve static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        # Mount assets directory at /assets/ for React build
        assets_dir = static_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        # Keep /static mount for backward compatibility
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/favicon.svg", response_model=None)
    def favicon() -> FileResponse | dict[str, str]:
        """Serve the favicon."""
        favicon_path = static_dir / "favicon.svg"
        if favicon_path.exists():
            return FileResponse(favicon_path, media_type="image/svg+xml")
        return {"message": "Favicon not found"}

    @app.get("/", response_model=None)
    def root() -> FileResponse | dict[str, str]:
        """Serve the React app."""
        index_path = static_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return {"message": "WaveCap-SDR API", "docs": "/docs"}

    @app.get("/health")
    @limiter.limit("30/minute")
    def health(request: Request) -> dict[str, str]:
        return {"status": "ok"}

    return app
