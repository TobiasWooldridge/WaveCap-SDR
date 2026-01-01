from __future__ import annotations

from multiprocessing import Pipe

import pytest

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


def test_require_worker_rejects_dead_process(tmp_path) -> None:
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

    try:
        with pytest.raises(RuntimeError, match="not running"):
            manager._require_worker("sys0")
    finally:
        cmd_child.close()
        event_child.close()

    assert cmd_parent.closed is True
    assert event_parent.closed is True
