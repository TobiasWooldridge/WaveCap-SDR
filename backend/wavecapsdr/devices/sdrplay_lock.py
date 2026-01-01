from __future__ import annotations

import contextlib
import os
import tempfile
import threading
import time
from typing import IO, Any, cast

_LOCK_PATH = os.path.join(tempfile.gettempdir(), "wavecapsdr_sdrplay.lock")
_LOCK = threading.RLock()
_LOCK_COUNT = 0
_LOCK_FILE: IO[bytes] | None = None


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


def _open_lock_file() -> IO[bytes]:
    file_handle = open(_LOCK_PATH, "a+b")
    file_handle.seek(0, os.SEEK_END)
    if file_handle.tell() == 0:
        file_handle.write(b"0")
        file_handle.flush()
    file_handle.seek(0)
    return file_handle


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
    global _LOCK_COUNT, _LOCK_FILE

    thread_id = threading.current_thread().name
    print(f"[LOCK] Thread {thread_id} attempting to acquire SDRplay global lock...", flush=True)

    _LOCK.acquire()
    try:
        if _LOCK_COUNT == 0:
            _LOCK_FILE = _open_lock_file()
            _lock_file(_LOCK_FILE)

            last_operation = _read_timestamp(_LOCK_FILE)
            elapsed = time.time() - last_operation
            if elapsed < cooldown:
                sleep_time = cooldown - elapsed
                print(f"[LOCK] Thread {thread_id} cooldown wait: {sleep_time:.2f}s", flush=True)
                time.sleep(sleep_time)
        _LOCK_COUNT += 1
    except Exception:
        print(f"[LOCK] Thread {thread_id} exception, releasing lock", flush=True)
        _LOCK.release()
        raise

    print(f"[LOCK] Thread {thread_id} ACQUIRED SDRplay global lock", flush=True)


def release_sdrplay_lock() -> None:
    """Release cross-process SDRplay lock and update cooldown timestamp."""
    global _LOCK_COUNT, _LOCK_FILE

    thread_id = threading.current_thread().name
    if _LOCK_COUNT == 0:
        print(f"[LOCK] Thread {thread_id} release called without lock", flush=True)
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
        print(f"[LOCK] Thread {thread_id} RELEASED SDRplay global lock", flush=True)
