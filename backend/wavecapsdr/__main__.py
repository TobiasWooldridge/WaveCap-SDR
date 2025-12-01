from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import uvicorn

from .app import create_app
from .config import load_config, ServerConfig


def _default_config_path() -> str:
    """Get default config path relative to module location."""
    # Look for config in: backend/config/wavecapsdr.yaml (relative to this module)
    module_dir = Path(__file__).resolve().parent
    config_path = module_dir.parent / "config" / "wavecapsdr.yaml"
    if config_path.exists():
        return str(config_path)
    # Fallback to relative path (for backwards compatibility)
    return "config/wavecapsdr.yaml"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WaveCap-SDR server")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=os.environ.get("WAVECAPSDR_CONFIG", _default_config_path()),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--bind",
        type=str,
        default=None,
        help="Override bind address (e.g., 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override port (e.g., 8087)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.bind is not None:
        cfg.server.bind_address = args.bind
    if args.port is not None:
        cfg.server.port = args.port

    app = create_app(cfg, config_path=args.config)

    # Configure uvicorn with appropriate timeouts for streaming
    # timeout_keep_alive: How long to wait for HTTP keep-alive between requests (default: 5)
    # We set this higher to support long-running streaming connections
    uvicorn.run(
        app,
        host=cfg.server.bind_address,
        port=cfg.server.port,
        log_level="info",
        timeout_keep_alive=300,  # 5 minutes for streaming connections
        timeout_graceful_shutdown=10,  # 10 seconds for graceful shutdown
    )


if __name__ == "__main__":
    main(sys.argv[1:])

