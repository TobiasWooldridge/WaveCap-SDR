from __future__ import annotations

import multiprocessing
import queue
import time

from wavecapsdr.devices import sdrplay_lock


def _lock_worker(lock_path: str, output: multiprocessing.Queue[tuple[float, float]]) -> None:
    sdrplay_lock._LOCK_PATH = lock_path
    sdrplay_lock._LOCK_COUNT = 0
    sdrplay_lock._LOCK_FILE = None
    sdrplay_lock._LOCK_FILE_DISABLED = False

    sdrplay_lock.acquire_sdrplay_lock(cooldown=0.0)
    start_time = time.monotonic()
    time.sleep(0.15)
    end_time = time.monotonic()
    sdrplay_lock.release_sdrplay_lock()

    output.put((start_time, end_time))


def test_sdrplay_lock_reentrant(tmp_path, monkeypatch) -> None:
    lock_path = tmp_path / "sdrplay.lock"
    monkeypatch.setattr(sdrplay_lock, "_LOCK_PATH", str(lock_path))
    monkeypatch.setattr(sdrplay_lock, "_LOCK_COUNT", 0)
    monkeypatch.setattr(sdrplay_lock, "_LOCK_FILE", None)
    monkeypatch.setattr(sdrplay_lock, "_LOCK_FILE_DISABLED", False)

    try:
        sdrplay_lock.acquire_sdrplay_lock(cooldown=0.0)
        sdrplay_lock.acquire_sdrplay_lock(cooldown=0.0)
    finally:
        sdrplay_lock.release_sdrplay_lock()
        sdrplay_lock.release_sdrplay_lock()

    assert lock_path.exists()
    value = lock_path.read_text(encoding="ascii").strip()
    assert float(value) >= 0.0


def test_sdrplay_lock_serializes_processes(tmp_path) -> None:
    lock_path = tmp_path / "sdrplay.lock"
    ctx = multiprocessing.get_context("spawn")
    output: multiprocessing.Queue[tuple[float, float]] = ctx.Queue()

    proc1 = ctx.Process(target=_lock_worker, args=(str(lock_path), output))
    proc2 = ctx.Process(target=_lock_worker, args=(str(lock_path), output))

    proc1.start()
    proc2.start()

    proc1.join(timeout=5)
    proc2.join(timeout=5)

    if proc1.is_alive():
        proc1.terminate()
        proc1.join(timeout=1)
    if proc2.is_alive():
        proc2.terminate()
        proc2.join(timeout=1)

    results = []
    try:
        results.append(output.get(timeout=1))
        results.append(output.get(timeout=1))
    except queue.Empty as exc:
        raise AssertionError("Missing lock timing results") from exc

    results.sort(key=lambda item: item[0])
    first_start, first_end = results[0]
    second_start, _second_end = results[1]

    assert second_start >= first_end
