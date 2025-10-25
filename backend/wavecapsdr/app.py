from __future__ import annotations

from fastapi import FastAPI

from .config import AppConfig
from .api import router as api_router
from .state import AppState


def create_app(config: AppConfig) -> FastAPI:
    app = FastAPI(title="WaveCap-SDR", version="0.1.0")

    app.state.app_state = AppState.from_config(config)

    app.include_router(api_router, prefix="/api/v1")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app

