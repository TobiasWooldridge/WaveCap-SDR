from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import threading
import time
from typing import IO, Any, cast

_LOCK_PATH = os.path.join(tempfile.gettempdir(), "wavecapsdr_sdrplay.lock")
_FALLBACK_LOCK_PATH = os.path.join(os.path.expanduser("~"), ".wavecapsdr", "sdrplay.lock")
_LOCK = threading.RLock()
_LOCK_COUNT = 0
_LOCK_FILE: IO[bytes] | None = None
_LOCK_FILE_DISABLED = False

logger = logging.getLogger(__name__)


def _lock_file(file_handle: IO[bytes]) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt_any = cast(Any, msvcrt)
        file_handle.seek(0)
        msvcrt_any.locking(file_handle.fileno(), msvcrt_any.LK_LOCK, 1)
    else:
        import fcntl

        fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX)


def _unlock_file(file_handle: IO[bytes]) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt_any = cast(Any, msvcrt)
        file_handle.seek(0)
        msvcrt_any.locking(file_handle.fileno(), msvcrt_any.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)


def _ensure_lock_file_header(file_handle: IO[bytes]) -> None:
    file_handle.seek(0, os.SEEK_END)
    if file_handle.tell() == 0:
        file_handle.write(b"0")
        file_handle.flush()
    file_handle.seek(0)


def _open_lock_file() -> IO[bytes] | None:
    paths = [_LOCK_PATH]
    if _LOCK_PATH != _FALLBACK_LOCK_PATH:
        paths.append(_FALLBACK_LOCK_PATH)
    last_error: OSError | None = None
    for path in paths:
        try:
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            file_handle = open(path, "a+b")
            _ensure_lock_file_header(file_handle)
            return file_handle
        except OSError as exc:
            last_error = exc
            logger.warning("Failed to open SDRplay lock file %s: %s", path, exc)
    if last_error is not None:
        logger.warning("Disabling SDRplay cross-process lock: %s", last_error)
    return None


def _read_timestamp(file_handle: IO[bytes]) -> float:
    file_handle.seek(0)
    raw = file_handle.read().decode("ascii", errors="ignore").strip()
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _write_timestamp(file_handle: IO[bytes], timestamp: float) -> None:
    file_handle.seek(0)
    file_handle.truncate()
    file_handle.write(f"{timestamp:.6f}".encode("ascii"))
    file_handle.flush()
    with contextlib.suppress(Exception):
        os.fsync(file_handle.fileno())


def acquire_sdrplay_lock(cooldown: float = 1.0) -> None:
    """Acquire cross-process SDRplay lock with cooldown enforcement."""
    global _LOCK_COUNT, _LOCK_FILE, _LOCK_FILE_DISABLED

    thread_id = threading.current_thread().name
    logger.debug("[LOCK] Thread %s attempting to acquire SDRplay global lock...", thread_id)

    _LOCK.acquire()
    try:
        if _LOCK_COUNT == 0:
            if _LOCK_FILE is None and not _LOCK_FILE_DISABLED:
                _LOCK_FILE = _open_lock_file()
                if _LOCK_FILE is None:
                    _LOCK_FILE_DISABLED = True
            if _LOCK_FILE is not None:
                _lock_file(_LOCK_FILE)

                last_operation = _read_timestamp(_LOCK_FILE)
                elapsed = time.time() - last_operation
                if elapsed < cooldown:
                    sleep_time = cooldown - elapsed
                    logger.debug("[LOCK] Thread %s cooldown wait: %.2fs", thread_id, sleep_time)
                    time.sleep(sleep_time)
        _LOCK_COUNT += 1
    except Exception:
        logger.warning("[LOCK] Thread %s exception, releasing lock", thread_id)
        _LOCK.release()
        raise

    logger.debug("[LOCK] Thread %s ACQUIRED SDRplay global lock", thread_id)


def release_sdrplay_lock() -> None:
    """Release cross-process SDRplay lock and update cooldown timestamp."""
    global _LOCK_COUNT, _LOCK_FILE

    thread_id = threading.current_thread().name
    if _LOCK_COUNT == 0:
        logger.warning("[LOCK] Thread %s release called without lock", thread_id)
        return

    try:
        _LOCK_COUNT -= 1
        if _LOCK_COUNT == 0 and _LOCK_FILE is not None:
            _write_timestamp(_LOCK_FILE, time.time())
            _unlock_file(_LOCK_FILE)
            _LOCK_FILE.close()
            _LOCK_FILE = None
    finally:
        _LOCK.release()
        logger.debug("[LOCK] Thread %s RELEASED SDRplay global lock", thread_id)
