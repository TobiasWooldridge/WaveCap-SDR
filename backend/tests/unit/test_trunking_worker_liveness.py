from __future__ import annotations

from multiprocessing import Pipe

from wavecapsdr.trunking.process_manager import TrunkingProcessManager, _WorkerHandle


class _DeadProcess:
    def __init__(self) -> None:
        self.join_calls = 0
        self.terminate_calls = 0

    def is_alive(self) -> bool:
        return False

    def join(self, timeout: float | None = None) -> None:
        self.join_calls += 1

    def terminate(self) -> None:
        self.terminate_calls += 1


class _AliveProcess:
    def is_alive(self) -> bool:
        return True

    def join(self, timeout: float | None = None) -> None:
        return None

    def terminate(self) -> None:
        return None


def test_require_worker_restarts_dead_process(tmp_path, monkeypatch) -> None:
    manager = TrunkingProcessManager(config_path=str(tmp_path / "wavecapsdr.yaml"))
    cmd_parent, cmd_child = Pipe()
    event_parent, event_child = Pipe()

    handle = _WorkerHandle(
        device_id="dev0",
        system_ids={"sys0"},
        cmd_conn=cmd_parent,
        event_conn=event_parent,
        process=_DeadProcess(),
    )
    manager._system_to_worker["sys0"] = handle
    manager._workers["dev0"] = handle

    new_cmd_parent, new_cmd_child = Pipe()
    new_event_parent, new_event_child = Pipe()
    new_handle = _WorkerHandle(
        device_id="dev0",
        system_ids={"sys0"},
        cmd_conn=new_cmd_parent,
        event_conn=new_event_parent,
        process=_AliveProcess(),
    )

    def _start_worker(device_id: str, system_ids: list[str]) -> _WorkerHandle:
        assert device_id == "dev0"
        assert set(system_ids) == {"sys0"}
        return new_handle

    monkeypatch.setattr(manager, "_start_worker", _start_worker)

    try:
        returned = manager._require_worker("sys0")
        assert returned is new_handle
        assert manager._system_to_worker["sys0"] is new_handle
        assert manager._workers["dev0"] is new_handle
    finally:
        cmd_child.close()
        event_child.close()
        new_cmd_child.close()
        new_event_child.close()
        new_cmd_parent.close()
        new_event_parent.close()

    assert cmd_parent.closed is True
    assert event_parent.closed is True
