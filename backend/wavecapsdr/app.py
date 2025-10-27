from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import AppConfig
from .api import router as api_router
from .state import AppState


def create_app(config: AppConfig) -> FastAPI:
    app = FastAPI(title="WaveCap-SDR", version="0.1.0")

    app.state.app_state = AppState.from_config(config)

    app.include_router(api_router, prefix="/api/v1")

    # Serve static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    def root():
        """Serve the index/catalog page."""
        index_path = static_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return {"message": "WaveCap-SDR API", "docs": "/docs"}

    @app.get("/player.html")
    def player():
        """Serve the player page."""
        player_path = static_dir / "player.html"
        if player_path.exists():
            return FileResponse(player_path)
        return {"message": "Player not found"}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app

