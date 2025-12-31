from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
from multiprocessing.connection import Connection
from typing import Any

from wavecapsdr.capture import CaptureManager
from wavecapsdr.config import AppConfig, load_config
from wavecapsdr.trunking.config import HuntMode, TalkgroupConfig, TrunkingSystemConfig
from wavecapsdr.trunking.manager import TrunkingManager


logger = logging.getLogger(__name__)


def run_trunking_worker(
    config_path: str,
    system_ids: list[str],
    device_id: str,
    cmd_conn: Connection,
    event_conn: Connection,
) -> None:
    """Entry point for per-device trunking worker process."""
    logging.basicConfig(
        level=logging.INFO,
        format="[TrunkingWorker] %(levelname)s: %(message)s",
    )
    try:
        asyncio.run(
            _worker_main(
                config_path=config_path,
                system_ids=system_ids,
                device_id=device_id,
                cmd_conn=cmd_conn,
                event_conn=event_conn,
            )
        )
    except Exception:
        logger.exception("Worker crashed")


async def _worker_main(
    config_path: str,
    system_ids: list[str],
    device_id: str,
    cmd_conn: Connection,
    event_conn: Connection,
) -> None:
    cfg = load_config(config_path)
    from wavecapsdr.state import create_device_driver
    driver = create_device_driver(cfg)
    captures = CaptureManager(cfg, driver)

    manager = TrunkingManager()
    manager.set_capture_manager(captures)
    manager.set_config_path(config_path)

    config_dir = os.path.dirname(config_path) if config_path else None
    for sys_id in system_ids:
        sys_data = cfg.trunking_systems.get(sys_id)
        if sys_data is None:
            logger.warning("Missing trunking config for %s (device %s)", sys_id, device_id)
            continue
        sys_data_with_id = dict(sys_data)
        if "id" not in sys_data_with_id:
            sys_data_with_id["id"] = sys_id
        trunking_config = TrunkingSystemConfig.from_dict(
            sys_data_with_id,
            config_dir=config_dir,
            rr_config=cfg.radioreference,
        )
        manager.register_config(trunking_config)

    await manager.start()

    event_queue = await manager.subscribe_events()
    event_task = asyncio.create_task(_forward_events(event_queue, event_conn))

    cmd_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _read_commands() -> None:
        while True:
            try:
                msg = cmd_conn.recv()
            except EOFError:
                loop.call_soon_threadsafe(cmd_queue.put_nowait, {"action": "shutdown"})
                break
            loop.call_soon_threadsafe(cmd_queue.put_nowait, msg)

    reader = threading.Thread(target=_read_commands, daemon=True)
    reader.start()

    try:
        while True:
            req = await cmd_queue.get()
            action = req.get("action")
            if action == "shutdown":
                break
            response = await _handle_request(manager, req)
            cmd_conn.send(response)
    finally:
        event_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await event_task
        await manager.stop()
        await _stop_captures(captures)
        cmd_conn.close()
        event_conn.close()


async def _handle_request(manager: TrunkingManager, req: dict[str, Any]) -> dict[str, Any]:
    request_id = req.get("id")
    action = req.get("action")
    system_id = req.get("system_id")
    args = req.get("args", {}) or {}
    result: dict[str, Any] = {}

    try:
        if action == "ping":
            result = {"status": "ok"}
        elif action == "list_systems":
            result = {"systems": [system.to_dict() for system in manager.list_systems()]}
        elif action == "snapshot":
            result = _build_snapshot(manager)
        elif action == "start_system":
            system_id_str = _require_system_id(system_id)
            await manager.start_system(system_id_str, persist=bool(args.get("persist", True)))
            result = {"status": "ok"}
        elif action == "stop_system":
            system_id_str = _require_system_id(system_id)
            await manager.stop_system(system_id_str, persist=bool(args.get("persist", True)))
            result = {"status": "ok"}
        elif action == "add_system":
            config = args.get("config")
            if not isinstance(config, TrunkingSystemConfig):
                raise ValueError("add_system missing config")
            await manager.add_system(config, talkgroups_csv=args.get("talkgroups_csv"))
            result = {"status": "ok"}
        elif action == "remove_system":
            system_id_str = _require_system_id(system_id)
            await manager.remove_system(system_id_str)
            result = {"status": "ok"}
        elif action == "get_active_calls":
            calls = manager.get_active_calls(system_id=system_id)
            result = {"calls": [call.to_dict() for call in calls]}
        elif action == "get_messages":
            system_id_str = _require_system_id(system_id)
            system = manager.get_system(system_id_str)
            if system is None:
                raise ValueError(f"System '{system_id}' not found")
            result = {"messages": system.get_messages(limit=args.get("limit", 100), offset=args.get("offset", 0))}
        elif action == "clear_messages":
            system_id_str = _require_system_id(system_id)
            system = manager.get_system(system_id_str)
            if system is None:
                raise ValueError(f"System '{system_id}' not found")
            result = {"cleared": system.clear_messages()}
        elif action == "update_talkgroups":
            system_id_str = _require_system_id(system_id)
            system = manager.get_system(system_id_str)
            if system is None:
                raise ValueError(f"System '{system_id}' not found")
            talkgroups = args.get("talkgroups", [])
            if not isinstance(talkgroups, list):
                raise ValueError("update_talkgroups requires list of talkgroups")
            updated = 0
            added = 0
            for tg in talkgroups:
                if isinstance(tg, TalkgroupConfig):
                    tg_cfg = tg
                elif isinstance(tg, dict):
                    tg_cfg = TalkgroupConfig(**tg)
                else:
                    raise ValueError("Invalid talkgroup payload")
                if tg_cfg.tgid in system.cfg.talkgroups:
                    updated += 1
                else:
                    added += 1
                system.cfg.talkgroups[tg_cfg.tgid] = tg_cfg
            result = {"status": "ok", "added": added, "updated": updated}
        elif action == "get_all_locations":
            system_id_str = _require_system_id(system_id)
            system = manager.get_system(system_id_str)
            if system is None:
                raise ValueError(f"System '{system_id}' not found")
            result = {"locations": [loc.to_dict() for loc in system.get_all_locations()]}
        elif action == "set_hunt_mode":
            system_id_str = _require_system_id(system_id)
            system = manager.get_system(system_id_str)
            if system is None:
                raise ValueError(f"System '{system_id}' not found")
            mode = args.get("mode")
            if isinstance(mode, str):
                mode = HuntMode(mode)
            if not isinstance(mode, HuntMode):
                raise ValueError("Invalid hunt mode")
            system.set_hunt_mode(mode, args.get("locked_freq"))
            result = {"status": "ok"}
        elif action == "set_channel_enabled":
            system_id_str = _require_system_id(system_id)
            system = manager.get_system(system_id_str)
            if system is None:
                raise ValueError(f"System '{system_id}' not found")
            system.set_channel_enabled(args["freq_hz"], args["enabled"])
            result = {"status": "ok"}
        elif action == "trigger_scan":
            system_id_str = _require_system_id(system_id)
            system = manager.get_system(system_id_str)
            if system is None:
                raise ValueError(f"System '{system_id}' not found")
            result = {"measurements": system.trigger_scan()}
        else:
            raise ValueError(f"Unknown action: {action}")

        return {"id": request_id, "ok": True, "result": result}
    except Exception as exc:
        return {"id": request_id, "ok": False, "error": str(exc)}


def _build_snapshot(manager: TrunkingManager) -> dict[str, Any]:
    all_messages: list[dict[str, Any]] = []
    all_active_calls: list[dict[str, Any]] = []
    all_call_history: list[dict[str, Any]] = []
    systems = manager.list_systems()

    for system in systems:
        for msg in system.get_messages(limit=200):
            all_messages.append({
                "systemId": system.cfg.id,
                "timestamp": msg.get("timestamp", 0),
                "opcode": msg.get("opcode", 0),
                "opcodeName": msg.get("opcode_name", ""),
                "nac": msg.get("nac"),
                "summary": msg.get("summary", ""),
                "raw": msg.get("raw"),
            })
        for active_call in system.get_active_calls():
            call_dict = active_call.to_dict()
            call_dict["systemId"] = system.cfg.id
            all_active_calls.append(call_dict)
        for call_entry in system.get_call_history(limit=50):
            call_entry["systemId"] = system.cfg.id
            all_call_history.append(call_entry)

    all_messages.sort(key=lambda m: m.get("timestamp", 0), reverse=True)
    all_call_history.sort(key=lambda c: c.get("endTime", c.get("startTime", 0)), reverse=True)

    return {
        "systems": [s.to_dict() for s in systems],
        "activeCalls": all_active_calls,
        "messages": all_messages[:200],
        "callHistory": all_call_history[:100],
    }


async def _forward_events(queue: asyncio.Queue[dict[str, Any]], event_conn: Connection) -> None:
    while True:
        event = await queue.get()
        try:
            event_conn.send(event)
        except (BrokenPipeError, EOFError):
            break


async def _stop_captures(captures: CaptureManager) -> None:
    for cap in captures.list_captures():
        await cap.stop()
        cap.release_device()


def _require_system_id(system_id: Any) -> str:
    if not isinstance(system_id, str) or not system_id:
        raise ValueError("Request is missing system_id")
    return system_id
