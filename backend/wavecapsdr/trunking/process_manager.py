from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from dataclasses import dataclass, field
from multiprocessing import get_context
from multiprocessing.connection import Connection
from typing import Any, ClassVar, Iterable, cast

from wavecapsdr.decoders.lrrp import RadioLocation
from wavecapsdr.trunking.config import HuntMode, TalkgroupConfig, TrunkingSystemConfig
from wavecapsdr.trunking.manager_types import TrunkingManagerLike, TrunkingSystemLike
from wavecapsdr.trunking.process_worker import run_trunking_worker
from wavecapsdr.trunking.system import ActiveCall, CallState

logger = logging.getLogger(__name__)

_MESSAGE_LOG_MAX_SIZE = 500
_CALL_HISTORY_MAX_SIZE = 100


class _WorkerRpcTransportError(RuntimeError):
    pass


def _normalize_device_id(device_id: str | None) -> str:
    if device_id and device_id.strip():
        return device_id
    return "auto"


def group_trunking_systems_by_device(
    systems: Iterable[TrunkingSystemConfig],
) -> dict[str, list[TrunkingSystemConfig]]:
    grouped: dict[str, list[TrunkingSystemConfig]] = {}
    for system in systems:
        key = _normalize_device_id(system.device_id)
        grouped.setdefault(key, []).append(system)
    return grouped


def _location_from_dict(data: dict[str, Any]) -> RadioLocation:
    return RadioLocation(
        unit_id=int(data.get("unitId", 0)),
        latitude=float(data.get("latitude", 0.0)),
        longitude=float(data.get("longitude", 0.0)),
        altitude_m=data.get("altitude"),
        speed_kmh=data.get("speed"),
        heading_deg=data.get("heading"),
        accuracy_m=data.get("accuracy"),
        timestamp=float(data.get("timestamp", 0.0)),
        source=str(data.get("source", "unknown")),
    )


def _active_call_from_dict(data: dict[str, Any]) -> ActiveCall:
    state_raw = data.get("state", "idle")
    try:
        state = CallState(state_raw)
    except ValueError:
        state = CallState.IDLE
    source_location = None
    if isinstance(data.get("sourceLocation"), dict):
        source_location = _location_from_dict(data["sourceLocation"])
    return ActiveCall(
        id=str(data.get("id", "")),
        talkgroup_id=int(data.get("talkgroupId", 0)),
        talkgroup_name=str(data.get("talkgroupName", "")),
        talkgroup_category=str(data.get("talkgroupCategory", "")),
        source_id=data.get("sourceId"),
        frequency_hz=float(data.get("frequencyHz", 0.0)),
        channel_id=int(data.get("channelId", 0)),
        state=state,
        start_time=float(data.get("startTime", 0.0)),
        last_activity_time=float(data.get("lastActivityTime", data.get("startTime", 0.0))),
        talkgroup_alpha_tag=str(data.get("talkgroupAlphaTag", "")),
        talkgroup_priority=data.get("talkgroupPriority"),
        talkgroup_record=data.get("talkgroupRecord"),
        talkgroup_monitor=data.get("talkgroupMonitor"),
        encrypted=bool(data.get("encrypted", False)),
        recorder_id=data.get("recorderId"),
        source_location=source_location,
        audio_frames=int(data.get("audioFrames", 0)),
        duration_seconds=float(data.get("durationSeconds", 0.0)),
    )


def _normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    if "opcode_name" in message:
        return message
    message["opcode_name"] = message.pop("opcodeName", "")
    message.setdefault("timestamp", 0)
    message.setdefault("opcode", 0)
    message.setdefault("nac", None)
    message.setdefault("summary", "")
    message.setdefault("raw", None)
    return message


@dataclass
class TrunkingSystemProxy(TrunkingSystemLike):
    cfg: TrunkingSystemConfig
    _system_state: dict[str, Any] = field(default_factory=dict)
    _active_calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    _messages: list[dict[str, Any]] = field(default_factory=list)
    _call_history: list[dict[str, Any]] = field(default_factory=list)
    _locations: list[RadioLocation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        if self._system_state:
            return dict(self._system_state)
        return {
            "id": self.cfg.id,
            "name": self.cfg.name,
            "protocol": self.cfg.protocol.value,
            "deviceId": self.cfg.device_id or None,
            "state": "stopped",
            "controlChannelState": "unlocked",
            "controlChannelFreqHz": None,
            "centerHz": self.cfg.center_hz,
            "nac": None,
            "systemId": None,
            "rfssId": None,
            "siteId": None,
            "decodeRate": 0.0,
            "activeCalls": len(self._active_calls),
            "stats": {},
            "channelMeasurements": {},
            "huntMode": self.cfg.default_hunt_mode.value,
            "lockedFrequencyHz": None,
            "controlChannels": self.get_control_channels_info(),
            "captureId": None,
        }

    def get_active_calls(self) -> list[ActiveCall]:
        return [_active_call_from_dict(call) for call in self._active_calls.values()]

    def get_all_locations(self) -> list[RadioLocation]:
        return list(self._locations)

    def get_messages(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        messages = list(reversed(self._messages))
        return messages[offset : offset + limit]

    def clear_messages(self) -> int:
        count = len(self._messages)
        self._messages.clear()
        return count

    def get_hunt_mode(self) -> HuntMode:
        mode = self._system_state.get("huntMode")
        if isinstance(mode, str):
            try:
                return HuntMode(mode)
            except ValueError:
                pass
        return self.cfg.default_hunt_mode

    def set_hunt_mode(self, mode: HuntMode, locked_freq: float | None = None) -> None:
        raise NotImplementedError("set_hunt_mode handled by TrunkingProcessManager")

    def get_locked_frequency(self) -> float | None:
        locked = self._system_state.get("lockedFrequencyHz")
        if isinstance(locked, (int, float)):
            return float(locked)
        return None

    def trigger_scan(self) -> dict[float, dict[str, Any]]:
        raise NotImplementedError("trigger_scan handled by TrunkingProcessManager")

    def get_control_channels_info(self) -> list[dict[str, Any]]:
        control_channels = self._system_state.get("controlChannels")
        if isinstance(control_channels, list):
            return control_channels
        return [
            {
                "frequencyHz": freq,
                "name": self.cfg.get_control_channel_name(freq),
                "enabled": True,
                "isCurrent": False,
                "isLocked": False,
                "snrDb": None,
                "powerDb": None,
                "syncDetected": False,
                "measurementTime": None,
            }
            for freq in self.cfg.control_channel_frequencies
        ]

    def set_channel_enabled(self, freq_hz: float, enabled: bool) -> None:
        raise NotImplementedError("set_channel_enabled handled by TrunkingProcessManager")

    def get_voice_channels(self) -> list[Any]:
        return []

    def get_voice_channel(self, channel_id: str) -> Any | None:
        return None

    def get_voice_recorder(self, recorder_id: str) -> Any | None:
        return None

    def replace_snapshot(
        self,
        system_state: dict[str, Any],
        active_calls: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        call_history: list[dict[str, Any]],
    ) -> None:
        self._system_state = dict(system_state)
        self._active_calls = {call.get("id", ""): call for call in active_calls}
        self._system_state["activeCalls"] = len(self._active_calls)
        self._messages = messages[-_MESSAGE_LOG_MAX_SIZE:]
        self._call_history = call_history[-_CALL_HISTORY_MAX_SIZE:]

    def update_system_state(self, system_state: dict[str, Any]) -> None:
        self._system_state = dict(system_state)

    def upsert_active_call(self, call: dict[str, Any]) -> None:
        call_id = call.get("id")
        if call_id:
            self._active_calls[call_id] = call
            self._system_state["activeCalls"] = len(self._active_calls)

    def remove_active_call(self, call_id: str) -> None:
        self._active_calls.pop(call_id, None)
        self._system_state["activeCalls"] = len(self._active_calls)

    def append_message(self, message: dict[str, Any]) -> None:
        self._messages.append(message)
        if len(self._messages) > _MESSAGE_LOG_MAX_SIZE:
            self._messages = self._messages[-_MESSAGE_LOG_MAX_SIZE:]

    def append_call_history(self, call: dict[str, Any]) -> None:
        self._call_history.append(call)
        if len(self._call_history) > _CALL_HISTORY_MAX_SIZE:
            self._call_history = self._call_history[-_CALL_HISTORY_MAX_SIZE:]

    def set_locations(self, locations: list[RadioLocation]) -> None:
        self._locations = list(locations)

    def capture_id(self) -> str | None:
        capture_id = self._system_state.get("captureId")
        if isinstance(capture_id, str):
            return capture_id
        return None


@dataclass
class _WorkerHandle:
    device_id: str
    system_ids: set[str]
    cmd_conn: Connection
    event_conn: Connection
    process: Any
    rpc_lock: threading.Lock = field(default_factory=threading.Lock)
    event_thread: threading.Thread | None = None


class TrunkingProcessManager(TrunkingManagerLike):
    supports_voice_streams: ClassVar[bool] = False

    def __init__(
        self,
        config_path: str,
        rpc_timeout_s: float = 5.0,
        event_queue_size: int = 200,
    ) -> None:
        self._config_path = config_path
        self._rpc_timeout_s = rpc_timeout_s
        self._event_queue_size = event_queue_size
        self._ctx = get_context("spawn")
        self._workers: dict[str, _WorkerHandle] = {}
        self._systems: dict[str, TrunkingSystemProxy] = {}
        self._system_to_worker: dict[str, _WorkerHandle] = {}
        self._pending_configs: list[TrunkingSystemConfig] = []
        self._event_queues: set[asyncio.Queue[dict[str, Any]]] = set()
        self._running = False
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._request_id = 0
        self._request_lock = threading.Lock()
        self._refresh_task: asyncio.Task[None] | None = None
        # FFT subscriber tracking: device_id -> set of (queue, loop) tuples
        self._fft_subscribers: dict[str, set[tuple[asyncio.Queue[dict[str, Any]], asyncio.AbstractEventLoop]]] = {}
        self._fft_subscribers_lock = threading.Lock()

    def set_config_path(self, config_path: str) -> None:
        self._config_path = config_path

    def set_capture_manager(self, _capture_manager: Any) -> None:
        return None

    def register_config(self, config: TrunkingSystemConfig) -> None:
        self._pending_configs.append(config)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        try:
            self._event_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._event_loop = None

        configs = list(self._pending_configs)
        if not configs and self._systems:
            configs = [proxy.cfg for proxy in self._systems.values()]
        self._pending_configs.clear()

        for config in configs:
            self._systems.setdefault(config.id, TrunkingSystemProxy(cfg=config))

        grouped = group_trunking_systems_by_device(configs)
        for device_id, systems in grouped.items():
            system_ids = [system.id for system in systems]
            handle = self._start_worker(device_id, system_ids)
            self._workers[device_id] = handle
            for system_id in system_ids:
                self._system_to_worker[system_id] = handle

        await self._refresh_snapshots()

        # Start periodic refresh task to keep stats updated
        self._refresh_task = asyncio.create_task(self._periodic_refresh())

    async def _periodic_refresh(self) -> None:
        """Periodically refresh snapshots from workers to keep stats current."""
        while self._running:
            await asyncio.sleep(2.0)  # Refresh every 2 seconds
            if self._running:
                try:
                    await self._refresh_snapshots()
                except Exception as e:
                    logger.debug(f"Periodic snapshot refresh failed: {e}")

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False

        # Cancel periodic refresh task
        if self._refresh_task:
            self._refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._refresh_task
            self._refresh_task = None

        for handle in list(self._workers.values()):
            self._shutdown_worker(handle)
        self._workers.clear()
        self._system_to_worker.clear()
        self._event_queues.clear()

    async def add_system(
        self,
        config: TrunkingSystemConfig,
        talkgroups_csv: str | None = None,
    ) -> TrunkingSystemLike:
        if config.id in self._systems:
            raise ValueError(f"System '{config.id}' already exists")

        self._systems[config.id] = TrunkingSystemProxy(cfg=config)
        device_id = _normalize_device_id(config.device_id)
        handle = self._workers.get(device_id)
        if handle is None:
            handle = self._start_worker(device_id, [])
            self._workers[device_id] = handle

        self._system_to_worker[config.id] = handle
        handle.system_ids.add(config.id)

        args: dict[str, Any] = {"config": config}
        if talkgroups_csv:
            args["talkgroups_csv"] = talkgroups_csv
        await self._rpc_call(handle, "add_system", system_id=config.id, args=args)
        return self._systems[config.id]

    async def remove_system(self, system_id: str) -> None:
        handle = self._system_to_worker.get(system_id)
        if handle is None:
            raise ValueError(f"System '{system_id}' not found")
        await self._rpc_call(handle, "remove_system", system_id=system_id)
        handle.system_ids.discard(system_id)
        self._system_to_worker.pop(system_id, None)
        self._systems.pop(system_id, None)
        if not handle.system_ids:
            self._shutdown_worker(handle)
            self._workers.pop(handle.device_id, None)

    async def start_system(self, system_id: str, persist: bool = True) -> None:
        handle = self._require_worker(system_id)
        await self._rpc_call(handle, "start_system", system_id=system_id, args={"persist": persist})

    async def stop_system(self, system_id: str, persist: bool = True) -> None:
        handle = self._require_worker(system_id)
        await self._rpc_call(handle, "stop_system", system_id=system_id, args={"persist": persist})

    async def update_talkgroups(
        self,
        system_id: str,
        talkgroups: list[TalkgroupConfig],
    ) -> tuple[int, int]:
        proxy = self._systems.get(system_id)
        if proxy is None:
            raise ValueError(f"System '{system_id}' not found")
        added = 0
        updated = 0
        for tg in talkgroups:
            if tg.tgid in proxy.cfg.talkgroups:
                updated += 1
            else:
                added += 1
        handle = self._require_worker(system_id)
        await self._rpc_call(
            handle,
            "update_talkgroups",
            system_id=system_id,
            args={"talkgroups": talkgroups},
        )
        for tg in talkgroups:
            proxy.cfg.talkgroups[tg.tgid] = tg
        return added, updated

    async def clear_messages(self, system_id: str) -> int:
        proxy = self._systems.get(system_id)
        if proxy is None:
            raise ValueError(f"System '{system_id}' not found")
        handle = self._require_worker(system_id)
        result = await self._rpc_call(handle, "clear_messages", system_id=system_id)
        proxy.clear_messages()
        return int(result.get("cleared", 0))

    async def set_hunt_mode(
        self,
        system_id: str,
        mode: HuntMode,
        locked_freq: float | None = None,
    ) -> None:
        handle = self._require_worker(system_id)
        await self._rpc_call(
            handle,
            "set_hunt_mode",
            system_id=system_id,
            args={"mode": mode.value, "locked_freq": locked_freq},
        )

    async def set_channel_enabled(
        self,
        system_id: str,
        freq_hz: float,
        enabled: bool,
    ) -> None:
        handle = self._require_worker(system_id)
        await self._rpc_call(
            handle,
            "set_channel_enabled",
            system_id=system_id,
            args={"freq_hz": freq_hz, "enabled": enabled},
        )

    async def trigger_scan(self, system_id: str) -> dict[float, dict[str, Any]]:
        handle = self._require_worker(system_id)
        result = await self._rpc_call(handle, "trigger_scan", system_id=system_id)
        measurements = result.get("measurements", {})
        if isinstance(measurements, dict):
            return cast(dict[float, dict[str, Any]], measurements)
        return {}

    async def get_all_locations(self, system_id: str) -> list[RadioLocation]:
        proxy = self._systems.get(system_id)
        if proxy is None:
            raise ValueError(f"System '{system_id}' not found")
        handle = self._require_worker(system_id)
        result = await self._rpc_call(handle, "get_all_locations", system_id=system_id)
        locations_raw = result.get("locations", [])
        locations = [_location_from_dict(loc) for loc in locations_raw if isinstance(loc, dict)]
        proxy.set_locations(locations)
        return locations

    def list_systems(self) -> list[TrunkingSystemLike]:
        return list(self._systems.values())

    def get_system(self, system_id: str) -> TrunkingSystemLike | None:
        return self._systems.get(system_id)

    def get_system_for_capture(self, capture_id: str) -> str | None:
        for system in self._systems.values():
            if system.capture_id() == capture_id:
                return system.cfg.id
        return None

    def get_active_calls(self, system_id: str | None = None) -> list[ActiveCall]:
        if system_id:
            system = self._systems.get(system_id)
            if system is None:
                return []
            return system.get_active_calls()
        calls: list[ActiveCall] = []
        for system in self._systems.values():
            calls.extend(system.get_active_calls())
        return calls

    async def subscribe_events(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self._event_queue_size)
        self._event_queues.add(queue)

        await self._refresh_snapshots()
        snapshot = self._build_snapshot()
        try:
            queue.put_nowait(snapshot)
        except asyncio.QueueFull:
            pass
        return queue

    async def unsubscribe_events(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._event_queues.discard(queue)

    # =========================================================================
    # FFT Streaming for Subprocess Captures
    # =========================================================================

    def _parse_trunking_capture_id(self, capture_id: str) -> str | None:
        """Parse 'trunking:{system_id}' format, returning system_id or None."""
        if capture_id.startswith("trunking:"):
            return capture_id[9:]  # len("trunking:") == 9
        return None

    def get_device_for_capture(self, capture_id: str) -> str | None:
        """Get the device ID that owns a capture.

        Accepts either raw capture ID (e.g., 'c1') or trunking format
        (e.g., 'trunking:sa_grn_2').
        """
        # Check if this is a trunking:{system_id} format
        system_id = self._parse_trunking_capture_id(capture_id)
        if system_id is not None:
            system = self._systems.get(system_id)
            if system is not None:
                return system.cfg.device_id or "auto"
            return None

        # Fall back to checking raw capture IDs
        for system in self._systems.values():
            if system.capture_id() == capture_id:
                return system.cfg.device_id or "auto"
        return None

    async def subscribe_fft(self, capture_id: str) -> asyncio.Queue[dict[str, Any]]:
        """Subscribe to FFT data for a subprocess capture.

        Returns a queue that will receive FFT data frames.
        """
        device_id = self.get_device_for_capture(capture_id)
        if device_id is None:
            raise ValueError(f"Capture '{capture_id}' not found in any trunking system")

        # Normalize device ID
        device_id = _normalize_device_id(device_id)

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=4)
        loop = asyncio.get_running_loop()

        with self._fft_subscribers_lock:
            if device_id not in self._fft_subscribers:
                self._fft_subscribers[device_id] = set()
            is_first = len(self._fft_subscribers[device_id]) == 0
            self._fft_subscribers[device_id].add((queue, loop))

        # If first subscriber, tell the worker to start forwarding FFT
        if is_first:
            handle = self._workers.get(device_id)
            if handle is not None:
                try:
                    await self._rpc_call(handle, "subscribe_fft")
                    logger.info(f"Started FFT forwarding from worker {device_id}")
                except Exception as e:
                    logger.error(f"Failed to start FFT forwarding: {e}")

        return queue

    def unsubscribe_fft(self, capture_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Unsubscribe from FFT data for a subprocess capture."""
        device_id = self.get_device_for_capture(capture_id)
        if device_id is None:
            return

        device_id = _normalize_device_id(device_id)

        with self._fft_subscribers_lock:
            subs = self._fft_subscribers.get(device_id)
            if subs is None:
                return
            # Remove this subscriber
            subs_to_remove = [s for s in subs if s[0] is queue]
            for s in subs_to_remove:
                subs.discard(s)
            is_last = len(subs) == 0

        # If last subscriber, tell the worker to stop forwarding FFT
        if is_last:
            handle = self._workers.get(device_id)
            if handle is not None:
                try:
                    # Fire and forget - don't wait for response
                    self._send_command(handle, "unsubscribe_fft")
                    logger.info(f"Stopped FFT forwarding from worker {device_id}")
                except Exception as e:
                    logger.warning(f"Failed to stop FFT forwarding: {e}")

    def _broadcast_fft(self, capture_id: str, fft_data: dict[str, Any]) -> None:
        """Broadcast FFT data to all subscribers for a capture."""
        device_id = self.get_device_for_capture(capture_id)
        if device_id is None:
            return

        device_id = _normalize_device_id(device_id)

        with self._fft_subscribers_lock:
            subs = self._fft_subscribers.get(device_id)
            if not subs:
                return
            subscribers = list(subs)

        for queue, loop in subscribers:
            try:
                if loop.is_running():
                    loop.call_soon_threadsafe(
                        lambda q=queue, d=fft_data: q.put_nowait(d) if not q.full() else None
                    )
            except Exception:
                pass

    def _send_command(self, handle: _WorkerHandle, action: str, **kwargs: Any) -> None:
        """Send a command to a worker without waiting for response."""
        request_id = self._next_request_id()
        msg: dict[str, Any] = {"id": request_id, "action": action}
        msg.update(kwargs)
        try:
            handle.cmd_conn.send(msg)
        except Exception:
            pass

    def has_subprocess_capture(self, capture_id: str) -> bool:
        """Check if a capture ID belongs to a subprocess trunking system."""
        return self.get_device_for_capture(capture_id) is not None

    def _require_worker(self, system_id: str) -> _WorkerHandle:
        handle = self._system_to_worker.get(system_id)
        if handle is None:
            raise ValueError(f"System '{system_id}' not found")
        if not handle.process.is_alive():
            handle = self._restart_worker(handle, f"system '{system_id}' worker not running")
        return handle

    def _next_request_id(self) -> int:
        with self._request_lock:
            self._request_id += 1
            return self._request_id

    async def _rpc_call(
        self,
        handle: _WorkerHandle,
        action: str,
        system_id: str | None = None,
        args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = {
            "id": self._next_request_id(),
            "action": action,
            "system_id": system_id,
            "args": args or {},
        }

        def _send_and_wait() -> dict[str, Any]:
            with handle.rpc_lock:
                if not handle.process.is_alive():
                    raise _WorkerRpcTransportError("Worker process is not running")
                try:
                    handle.cmd_conn.send(request)
                except (BrokenPipeError, EOFError, OSError) as exc:
                    raise _WorkerRpcTransportError("Worker RPC send failed") from exc
                if not handle.cmd_conn.poll(self._rpc_timeout_s):
                    if not handle.process.is_alive():
                        raise _WorkerRpcTransportError(f"Worker process died during RPC '{action}'")
                    raise TimeoutError(f"Worker RPC timeout for {action}")
                try:
                    response = handle.cmd_conn.recv()
                except (EOFError, OSError) as exc:
                    if not handle.process.is_alive():
                        raise _WorkerRpcTransportError(
                            f"Worker process died during RPC '{action}'"
                        ) from exc
                    raise _WorkerRpcTransportError("Worker RPC recv failed") from exc
                if not isinstance(response, dict):
                    raise RuntimeError("Worker RPC returned invalid response")
                return cast(dict[str, Any], response)

        try:
            response = await asyncio.to_thread(_send_and_wait)
        except (TimeoutError, _WorkerRpcTransportError) as exc:
            self._restart_worker(handle, f"RPC '{action}' failed: {exc}")
            raise RuntimeError(f"Worker RPC failed for '{action}', worker restarted") from exc
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "Worker RPC error"))
        result = response.get("result", {})
        if isinstance(result, dict):
            return result
        return {}

    def _start_worker(self, device_id: str, system_ids: list[str]) -> _WorkerHandle:
        cmd_parent, cmd_child = self._ctx.Pipe()
        event_parent, event_child = self._ctx.Pipe()
        process = self._ctx.Process(
            target=run_trunking_worker,
            args=(self._config_path, system_ids, device_id, cmd_child, event_child),
            daemon=False,  # allow SDRplay worker subprocesses
        )
        process.start()
        with contextlib.suppress(Exception):
            cmd_child.close()
        with contextlib.suppress(Exception):
            event_child.close()
        handle = _WorkerHandle(
            device_id=device_id,
            system_ids=set(system_ids),
            cmd_conn=cmd_parent,
            event_conn=event_parent,
            process=process,
        )
        handle.event_thread = self._start_event_thread(handle)
        return handle

    def _restart_worker(self, handle: _WorkerHandle, reason: str) -> _WorkerHandle:
        logger.warning("Restarting trunking worker %s: %s", handle.device_id, reason)
        system_ids = list(handle.system_ids)
        self._shutdown_worker(handle)
        new_handle = self._start_worker(handle.device_id, system_ids)
        self._workers[handle.device_id] = new_handle
        for system_id in system_ids:
            self._system_to_worker[system_id] = new_handle
        return new_handle

    def _shutdown_worker(self, handle: _WorkerHandle) -> None:
        try:
            handle.cmd_conn.send({"action": "shutdown"})
        except Exception:
            pass
        try:
            handle.cmd_conn.close()
        except Exception:
            pass
        try:
            handle.event_conn.close()
        except Exception:
            pass
        if handle.process.is_alive():
            handle.process.join(timeout=5)
        if handle.process.is_alive():
            handle.process.terminate()

    def _start_event_thread(self, handle: _WorkerHandle) -> threading.Thread:
        def _reader() -> None:
            while True:
                try:
                    event = handle.event_conn.recv()
                except EOFError:
                    break
                self._dispatch_event(event)

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        return thread

    def _dispatch_event(self, event: dict[str, Any]) -> None:
        loop = self._event_loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(self._process_event, event)
            return
        self._process_event(event)

    def _process_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "snapshot":
            self._apply_snapshot(event)
        elif event_type in ("system_added", "system_update"):
            system_data = event.get("system")
            if isinstance(system_data, dict):
                self._apply_system_state(system_data)
        elif event_type == "system_removed":
            system_id = event.get("systemId")
            if isinstance(system_id, str):
                self._systems.pop(system_id, None)
                handle = self._system_to_worker.pop(system_id, None)
                if handle is not None:
                    handle.system_ids.discard(system_id)
        elif event_type in ("call_start", "call_update"):
            system_id = event.get("systemId")
            call = event.get("call")
            if isinstance(system_id, str) and isinstance(call, dict):
                proxy = self._systems.get(system_id)
                if proxy:
                    proxy.upsert_active_call(call)
        elif event_type == "call_end":
            system_id = event.get("systemId")
            call = event.get("call")
            call_id = event.get("callId")
            if not isinstance(call_id, str) and isinstance(call, dict):
                call_id = call.get("id")
            if isinstance(system_id, str):
                proxy = self._systems.get(system_id)
                if proxy and isinstance(call_id, str):
                    proxy.remove_active_call(call_id)
                if proxy and isinstance(call, dict):
                    call.setdefault("endTime", time.time())
                    proxy.append_call_history(self._strip_system_id(call))
        elif event_type == "message":
            system_id = event.get("systemId")
            message = event.get("message")
            if isinstance(system_id, str) and isinstance(message, dict):
                proxy = self._systems.get(system_id)
                if proxy:
                    proxy.append_message(_normalize_message(message))
        elif event_type == "fft":
            # Forward FFT data to local subscribers
            capture_id = event.get("captureId")
            fft_data = event.get("data")
            if isinstance(capture_id, str) and fft_data is not None:
                self._broadcast_fft(capture_id, fft_data)
            return  # Don't broadcast FFT to regular event subscribers

        self._schedule_broadcast(event)

    def _apply_system_state(self, system_data: dict[str, Any]) -> None:
        system_id = system_data.get("id")
        if not isinstance(system_id, str):
            return
        proxy = self._systems.get(system_id)
        if proxy is None:
            return
        proxy.update_system_state(system_data)

    def _apply_snapshot(self, snapshot: dict[str, Any]) -> None:
        systems = snapshot.get("systems", [])
        system_ids: set[str] = set()
        system_state_by_id: dict[str, dict[str, Any]] = {}
        if isinstance(systems, list):
            for system_data in systems:
                if not isinstance(system_data, dict):
                    continue
                system_id = system_data.get("id")
                if isinstance(system_id, str):
                    system_ids.add(system_id)
                    system_state_by_id[system_id] = system_data
                    self._apply_system_state(system_data)

        active_calls = snapshot.get("activeCalls", [])
        active_by_system: dict[str, list[dict[str, Any]]] = {sid: [] for sid in system_ids}
        if isinstance(active_calls, list):
            for call in active_calls:
                if not isinstance(call, dict):
                    continue
                system_id = call.get("systemId")
                if isinstance(system_id, str) and system_id in active_by_system:
                    active_by_system[system_id].append(self._strip_system_id(call))

        messages = snapshot.get("messages", [])
        messages_by_system: dict[str, list[dict[str, Any]]] = {sid: [] for sid in system_ids}
        if isinstance(messages, list):
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                system_id = msg.get("systemId")
                if isinstance(system_id, str) and system_id in messages_by_system:
                    messages_by_system[system_id].append(_normalize_message(msg))

        call_history = snapshot.get("callHistory", [])
        history_by_system: dict[str, list[dict[str, Any]]] = {sid: [] for sid in system_ids}
        if isinstance(call_history, list):
            for call in call_history:
                if not isinstance(call, dict):
                    continue
                system_id = call.get("systemId")
                if isinstance(system_id, str) and system_id in history_by_system:
                    history_by_system[system_id].append(self._strip_system_id(call))

        for system_id in system_ids:
            proxy = self._systems.get(system_id)
            if proxy is None:
                continue
            proxy.replace_snapshot(
                system_state=system_state_by_id.get(system_id, proxy.to_dict()),
                active_calls=active_by_system.get(system_id, []),
                messages=messages_by_system.get(system_id, []),
                call_history=history_by_system.get(system_id, []),
            )

    def _strip_system_id(self, call: dict[str, Any]) -> dict[str, Any]:
        call = dict(call)
        call.pop("systemId", None)
        return call

    async def _refresh_snapshots(self) -> None:
        for handle in self._workers.values():
            try:
                snapshot = await self._rpc_call(handle, "snapshot")
                if isinstance(snapshot, dict):
                    # Debug: log captureId from snapshot
                    for sys_data in snapshot.get("systems", []):
                        cap_id = sys_data.get("captureId")
                        if cap_id:
                            logger.debug(f"Snapshot has captureId={cap_id} for {sys_data.get('id')}")
                    self._apply_snapshot(snapshot)
            except Exception as exc:
                logger.warning(
                    "Failed to refresh snapshot from worker %s: %s", handle.device_id, exc
                )

    def _build_snapshot(self) -> dict[str, Any]:
        all_messages: list[dict[str, Any]] = []
        all_call_history: list[dict[str, Any]] = []
        all_active_calls: list[dict[str, Any]] = []
        systems = []

        for system in self._systems.values():
            system_dict = system.to_dict()
            systems.append(system_dict)
            system_id = system.cfg.id
            for call in system._active_calls.values():
                call_with_id = dict(call)
                call_with_id["systemId"] = system_id
                all_active_calls.append(call_with_id)
            for msg in system._messages:
                msg_with_id = dict(msg)
                msg_with_id["systemId"] = system_id
                msg_with_id["opcodeName"] = msg_with_id.pop("opcode_name", "")
                all_messages.append(msg_with_id)
            for call in system._call_history:
                call_with_id = dict(call)
                call_with_id["systemId"] = system_id
                all_call_history.append(call_with_id)

        all_messages.sort(key=lambda m: m.get("timestamp", 0), reverse=True)
        all_call_history.sort(key=lambda c: c.get("endTime", c.get("startTime", 0)), reverse=True)

        return {
            "type": "snapshot",
            "systems": systems,
            "activeCalls": all_active_calls,
            "messages": all_messages[:200],
            "callHistory": all_call_history[:100],
        }

    async def _broadcast_event(self, event: dict[str, Any]) -> None:
        dead_queues: list[asyncio.Queue[dict[str, Any]]] = []
        for queue in self._event_queues:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                    queue.put_nowait(event)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass
            except Exception:
                dead_queues.append(queue)
        for queue in dead_queues:
            self._event_queues.discard(queue)

    def _schedule_broadcast(self, event: dict[str, Any]) -> None:
        loop = self._event_loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(lambda: loop.create_task(self._broadcast_event(event)))
            return
        try:
            running_loop = asyncio.get_running_loop()
            running_loop.create_task(self._broadcast_event(event))
        except RuntimeError:
            logger.debug("Dropping event %s (no event loop)", event.get("type"))
