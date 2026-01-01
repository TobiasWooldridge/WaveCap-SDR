from __future__ import annotations

from wavecapsdr.devices import sdrplay_lock


def test_sdrplay_lock_reentrant(tmp_path, monkeypatch) -> None:
    lock_path = tmp_path / "sdrplay.lock"
    monkeypatch.setattr(sdrplay_lock, "_LOCK_PATH", str(lock_path))
    monkeypatch.setattr(sdrplay_lock, "_LOCK_COUNT", 0)
    monkeypatch.setattr(sdrplay_lock, "_LOCK_FILE", None)

    try:
        sdrplay_lock.acquire_sdrplay_lock(cooldown=0.0)
        sdrplay_lock.acquire_sdrplay_lock(cooldown=0.0)
    finally:
        sdrplay_lock.release_sdrplay_lock()
        sdrplay_lock.release_sdrplay_lock()

    assert lock_path.exists()
    value = lock_path.read_text(encoding="ascii").strip()
    assert float(value) >= 0.0
