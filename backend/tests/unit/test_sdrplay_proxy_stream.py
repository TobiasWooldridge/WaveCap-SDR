"""Unit tests for SDRplay proxy stream behavior."""

from multiprocessing import Pipe
from multiprocessing.shared_memory import SharedMemory

from wavecapsdr.devices.sdrplay_proxy import SDRplayProxyStream
from wavecapsdr.devices.sdrplay_worker import (
    FLAG_DATA_READY,
    FLAG_OVERFLOW,
    FLAG_RUNNING,
    SHM_SIZE,
    _write_header,
)


def test_sdrplay_proxy_overflow_only_on_increment() -> None:
    """Overflow should only be reported when the worker count increases."""
    shm = SharedMemory(create=True, size=SHM_SIZE)
    status_pipe, other_end = Pipe()
    try:
        stream = SDRplayProxyStream(
            shm=shm,
            status_pipe=status_pipe,
            sample_rate=2_000_000,
        )

        flags = FLAG_RUNNING | FLAG_DATA_READY | FLAG_OVERFLOW
        _write_header(
            shm,
            write_idx=10,
            sample_count=10,
            overflow_count=1,
            sample_rate=2_000_000,
            flags=flags,
            timestamp=1.0,
        )
        samples, overflow = stream.read(10)
        assert samples.size == 10
        assert overflow is True

        _write_header(
            shm,
            write_idx=20,
            sample_count=20,
            overflow_count=1,
            sample_rate=2_000_000,
            flags=flags,
            timestamp=2.0,
        )
        samples, overflow = stream.read(10)
        assert samples.size == 10
        assert overflow is False
    finally:
        status_pipe.close()
        other_end.close()
        shm.close()
        shm.unlink()


def test_sdrplay_proxy_read_does_not_consume_status_pipe() -> None:
    shm = SharedMemory(create=True, size=SHM_SIZE)
    status_pipe, other_end = Pipe()
    try:
        stream = SDRplayProxyStream(
            shm=shm,
            status_pipe=status_pipe,
            sample_rate=2_000_000,
        )

        flags = FLAG_RUNNING | FLAG_DATA_READY
        _write_header(
            shm,
            write_idx=4,
            sample_count=4,
            overflow_count=0,
            sample_rate=2_000_000,
            flags=flags,
            timestamp=1.0,
        )

        payload = {"type": "configured"}
        other_end.send(payload)

        samples, overflow = stream.read(2)
        assert samples.size == 2
        assert overflow is False

        assert status_pipe.poll() is True
        assert status_pipe.recv() == payload
    finally:
        status_pipe.close()
        other_end.close()
        shm.close()
        shm.unlink()
