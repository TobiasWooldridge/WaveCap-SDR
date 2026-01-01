from __future__ import annotations

import argparse
import atexit
import contextlib
import logging
import os
import signal
import sys
from pathlib import Path

import uvicorn

from .app import create_app
from .config import default_config_path, load_config
from .utils.log_levels import log_level_name


def _get_lock_file_path(port: int) -> Path:
    """Get path to lock file for the given port."""
    # Use system temp directory for reliability during startup
    tmp_dir = Path("/tmp")
    return tmp_dir / f"wavecapsdr-{port}.lock"


def _is_process_running(pid: int) -> bool:
    """Check if a process with the given PID is running."""
    try:
        # On Unix, sending signal 0 checks if process exists without affecting it
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _acquire_lock(port: int) -> bool:
    """Acquire the lock file for the given port.

    Returns True if lock acquired, False if another instance is running.
    Handles stale lock files gracefully (e.g., after system restart).
    """
    lock_path = _get_lock_file_path(port)

    if lock_path.exists():
        try:
            content = lock_path.read_text().strip()
            old_pid = int(content)

            if _is_process_running(old_pid):
                # Another instance is actually running
                print(
                    f"ERROR: Another WaveCap-SDR instance is running (PID {old_pid}) on port {port}",
                    file=sys.stderr,
                )
                return False
            else:
                # Stale lock file (process died without cleanup, or system restarted)
                print(f"Removing stale lock file (old PID {old_pid} not running)")
                lock_path.unlink()
        except (ValueError, OSError) as e:
            # Corrupt lock file - remove it
            print(f"Removing invalid lock file: {e}")
            with contextlib.suppress(OSError):
                lock_path.unlink()

    # Create new lock file with our PID
    try:
        lock_path.write_text(str(os.getpid()))
        return True
    except OSError as e:
        print(f"ERROR: Cannot create lock file: {e}", file=sys.stderr)
        return False


def _release_lock(port: int) -> None:
    """Release the lock file on exit."""
    lock_path = _get_lock_file_path(port)
    try:
        if lock_path.exists():
            # Only remove if it contains our PID (safety check)
            content = lock_path.read_text().strip()
            if content == str(os.getpid()):
                lock_path.unlink()
    except OSError:
        pass  # Best effort cleanup


def _default_config_path() -> str:
    """Get default config path relative to module location."""
    return default_config_path()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.bind is not None:
        cfg.server.bind_address = args.bind
    if args.port is not None:
        cfg.server.port = args.port

    # Acquire lock to prevent multiple instances on the same port
    port = cfg.server.port
    if not _acquire_lock(port):
        sys.exit(1)

    # Register cleanup handlers
    atexit.register(_release_lock, port)

    # Handle signals for clean shutdown
    def signal_handler(signum: int, frame: object) -> None:
        _release_lock(port)
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    app = create_app(cfg, config_path=args.config)

    # Configure uvicorn with appropriate timeouts for streaming
    # timeout_keep_alive: How long to wait for HTTP keep-alive between requests (default: 5)
    # We set this higher to support long-running streaming connections
    uvicorn_level_env = os.environ.get("WAVECAP_UVICORN_LOG_LEVEL")
    if uvicorn_level_env is None:
        uvicorn_level_env = os.environ.get("WAVECAP_LOG_LEVEL")
    uvicorn_log_level = log_level_name(uvicorn_level_env, logging.INFO)

    uvicorn.run(
        app,
        host=cfg.server.bind_address,
        port=cfg.server.port,
        log_level=uvicorn_log_level,
        timeout_keep_alive=300,  # 5 minutes for streaming connections
        timeout_graceful_shutdown=10,  # 10 seconds for graceful shutdown
    )


if __name__ == "__main__":
    main(sys.argv[1:])
