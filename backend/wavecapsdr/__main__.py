from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import uvicorn

from .app import create_app
from .config import load_config, ServerConfig


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WaveCap-SDR server")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=os.environ.get("WAVECAPSDR_CONFIG", "config/wavecapsdr.yaml"),
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

    app = create_app(cfg)

    # Keep default of loopback unless explicitly configured otherwise.
    uvicorn.run(
        app,
        host=cfg.server.bind_address,
        port=cfg.server.port,
        log_level="info",
    )


if __name__ == "__main__":
    main(sys.argv[1:])

