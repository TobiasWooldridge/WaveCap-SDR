from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import queue as queue_module
import threading
import time
from multiprocessing.connection import Connection
from typing import Any

from wavecapsdr.capture import CaptureManager
from wavecapsdr.config import AppConfig, load_config
from wavecapsdr.trunking.config import HuntMode, TalkgroupConfig, TrunkingSystemConfig
from wavecapsdr.trunking.manager import TrunkingManager
from wavecapsdr.utils.log_levels import parse_log_level


logger = logging.getLogger(__name__)
_EVENT_PIPE_QUEUE_SIZE = 500
_FFT_FORWARD_INTERVAL = 1.0 / 30  # 30 FPS max for forwarded FFT
_AUDIO_FORWARD_INTERVAL = 0.02  # 50 FPS max for audio (20ms chunks)


def _should_use_sdrplay_proxy(device_id: str) -> bool:
    if not device_id or device_id == "auto":
        return True
    return "sdrplay" not in device_id.lower()


def run_trunking_worker(
    config_path: str,
    system_ids: list[str],
    device_id: str,
    cmd_conn: Connection,
    event_conn: Connection,
) -> None:
    """Entry point for per-device trunking worker process."""
    # Set up logging to both console and file
    log_level = parse_log_level(os.getenv("WAVECAP_LOG_LEVEL"), logging.INFO)

    # Create a safe device ID for filename (remove special chars)
    safe_device_id = "".join(c if c.isalnum() else "_" for c in device_id[:30])
    log_dir = os.path.join(os.path.dirname(config_path) if config_path else ".", "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    worker_log_path = os.path.join(log_dir, f"trunking_worker_{safe_device_id}.log")

    # Set up root logger with both handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Console handler (existing behavior)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter("[TrunkingWorker] %(levelname)s: %(message)s"))
    root_logger.addHandler(console_handler)

    # File handler for diagnostics
    file_handler = logging.FileHandler(worker_log_path, mode="w")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [TrunkingWorker] %(levelname)s: %(message)s")
    )
    root_logger.addHandler(file_handler)

    logger.info(f"Worker logging to: {worker_log_path}")
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

    use_sdrplay_proxy = _should_use_sdrplay_proxy(device_id)
    if not use_sdrplay_proxy:
        logger.info(
            "Trunking worker %s: SDRplay proxy disabled (direct Soapy in per-device worker)",
            device_id,
        )
    driver = create_device_driver(cfg, use_sdrplay_proxy=use_sdrplay_proxy)
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
    send_queue: queue_module.Queue[dict[str, Any] | None] = queue_module.Queue(
        maxsize=_EVENT_PIPE_QUEUE_SIZE
    )
    sender_thread = _start_event_sender(send_queue, event_conn)
    event_task = asyncio.create_task(_forward_events(event_queue, send_queue))

    # Create FFT forwarder for spectrum streaming to main process
    global _fft_forwarder
    _fft_forwarder = FFTForwarder(manager, send_queue)

    # Create audio forwarder for voice channel streaming to main process
    global _audio_forwarder
    _audio_forwarder = AudioForwarder(manager, send_queue)

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
        if _fft_forwarder is not None:
            await _fft_forwarder.stop()
        if _audio_forwarder is not None:
            await _audio_forwarder.stop()
        _stop_event_sender(send_queue, sender_thread)
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
            result = {
                "messages": system.get_messages(
                    limit=args.get("limit", 100), offset=args.get("offset", 0)
                )
            }
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
        elif action == "subscribe_fft":
            # Subscribe to FFT forwarding from subprocess
            if _fft_forwarder is not None:
                await _fft_forwarder.subscribe()
                result = {"status": "ok"}
            else:
                raise ValueError("FFT forwarder not initialized")
        elif action == "unsubscribe_fft":
            # Unsubscribe from FFT forwarding
            if _fft_forwarder is not None:
                await _fft_forwarder.unsubscribe()
                result = {"status": "ok"}
            else:
                raise ValueError("FFT forwarder not initialized")
        elif action == "subscribe_audio":
            # Subscribe to audio forwarding from subprocess
            if _audio_forwarder is not None:
                await _audio_forwarder.subscribe()
                result = {"status": "ok"}
            else:
                raise ValueError("Audio forwarder not initialized")
        elif action == "unsubscribe_audio":
            # Unsubscribe from audio forwarding
            if _audio_forwarder is not None:
                await _audio_forwarder.unsubscribe()
                result = {"status": "ok"}
            else:
                raise ValueError("Audio forwarder not initialized")
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
            all_messages.append(
                {
                    "systemId": system.cfg.id,
                    "timestamp": msg.get("timestamp", 0),
                    "opcode": msg.get("opcode", 0),
                    "opcodeName": msg.get("opcode_name", ""),
                    "nac": msg.get("nac"),
                    "summary": msg.get("summary", ""),
                    "raw": msg.get("raw"),
                }
            )
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


async def _forward_events(
    queue: asyncio.Queue[dict[str, Any]],
    send_queue: queue_module.Queue[dict[str, Any] | None],
) -> None:
    while True:
        event = await queue.get()
        try:
            send_queue.put_nowait(event)
        except queue_module.Full:
            try:
                send_queue.get_nowait()
            except queue_module.Empty:
                pass
            with contextlib.suppress(queue_module.Full):
                send_queue.put_nowait(event)


def _start_event_sender(
    send_queue: queue_module.Queue[dict[str, Any] | None],
    event_conn: Connection,
) -> threading.Thread:
    def _sender() -> None:
        while True:
            event = send_queue.get()
            if event is None:
                break
            try:
                event_conn.send(event)
            except (BrokenPipeError, EOFError, OSError):
                break

    thread = threading.Thread(target=_sender, daemon=True)
    thread.start()
    return thread


def _stop_event_sender(
    send_queue: queue_module.Queue[dict[str, Any] | None],
    sender_thread: threading.Thread,
) -> None:
    with contextlib.suppress(queue_module.Full):
        send_queue.put_nowait(None)
    sender_thread.join(timeout=1.0)


async def _stop_captures(captures: CaptureManager) -> None:
    for cap in captures.list_captures():
        await cap.stop()
        cap.release_device()


def _require_system_id(system_id: Any) -> str:
    if not isinstance(system_id, str) or not system_id:
        raise ValueError("Request is missing system_id")
    return system_id


class FFTForwarder:
    """Forwards FFT data from subprocess captures to main process.

    Subscribes to capture FFT when there are remote subscribers,
    and forwards FFT frames via the event pipe at a throttled rate.
    """

    def __init__(
        self,
        manager: TrunkingManager,
        send_queue: queue_module.Queue[dict[str, Any] | None],
    ) -> None:
        self._manager = manager
        self._send_queue = send_queue
        self._subscriber_count = 0
        self._fft_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._last_send_time = 0.0

    async def subscribe(self) -> None:
        """Add a subscriber. Starts FFT forwarding if first subscriber."""
        async with self._lock:
            self._subscriber_count += 1
            if self._subscriber_count == 1:
                self._stop_event.clear()
                self._fft_task = asyncio.create_task(self._forward_fft())
                logger.info("FFT forwarding started (first subscriber)")

    async def unsubscribe(self) -> None:
        """Remove a subscriber. Stops FFT forwarding if last subscriber."""
        async with self._lock:
            self._subscriber_count = max(0, self._subscriber_count - 1)
            if self._subscriber_count == 0 and self._fft_task is not None:
                self._stop_event.set()
                self._fft_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._fft_task
                self._fft_task = None
                logger.info("FFT forwarding stopped (no subscribers)")

    async def stop(self) -> None:
        """Stop all FFT forwarding."""
        async with self._lock:
            self._subscriber_count = 0
            if self._fft_task is not None:
                self._stop_event.set()
                self._fft_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._fft_task
                self._fft_task = None

    async def _forward_fft(self) -> None:
        """Forward FFT data from the trunking capture to main process."""
        # Find the capture from one of the running systems
        capture = None
        for system in self._manager.list_systems():
            cap = getattr(system, "_capture", None)
            if cap is not None:
                capture = cap
                break

        if capture is None:
            logger.warning("FFT forwarding: no capture found")
            return

        capture_id = capture.cfg.id
        logger.info(f"FFT forwarding: subscribing to capture {capture_id}")

        try:
            fft_queue = await capture.subscribe_fft()
            try:
                while not self._stop_event.is_set():
                    try:
                        # Get FFT data with timeout
                        fft_data = await asyncio.wait_for(fft_queue.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue

                    # Throttle to avoid flooding the pipe
                    now = time.monotonic()
                    if now - self._last_send_time < _FFT_FORWARD_INTERVAL:
                        continue
                    self._last_send_time = now

                    # Forward FFT as a special event
                    event = {
                        "type": "fft",
                        "captureId": capture_id,
                        "data": fft_data,
                    }
                    try:
                        self._send_queue.put_nowait(event)
                    except queue_module.Full:
                        # Drop FFT frame if queue is full
                        pass
            finally:
                capture.unsubscribe_fft(fft_queue)
        except Exception as e:
            logger.error(f"FFT forwarding error: {e}")


# Global FFT forwarder for the worker (set in _worker_main)
_fft_forwarder: FFTForwarder | None = None


class AudioForwarder:
    """Forwards voice channel audio from subprocess to main process.

    Subscribes to all active voice channels and forwards audio frames
    via the event pipe. Audio events include call metadata and base64-encoded
    PCM16 audio data.
    """

    def __init__(
        self,
        manager: TrunkingManager,
        send_queue: queue_module.Queue[dict[str, Any] | None],
    ) -> None:
        self._manager = manager
        self._send_queue = send_queue
        self._subscriber_count = 0
        self._audio_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._subscribed_channels: set[str] = set()

    async def subscribe(self) -> None:
        """Add a subscriber. Starts audio forwarding if first subscriber."""
        async with self._lock:
            self._subscriber_count += 1
            if self._subscriber_count == 1:
                self._stop_event.clear()
                self._audio_task = asyncio.create_task(self._forward_audio())
                logger.info("Audio forwarding started (first subscriber)")

    async def unsubscribe(self) -> None:
        """Remove a subscriber. Stops audio forwarding if last subscriber."""
        async with self._lock:
            self._subscriber_count = max(0, self._subscriber_count - 1)
            if self._subscriber_count == 0 and self._audio_task is not None:
                self._stop_event.set()
                self._audio_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._audio_task
                self._audio_task = None
                self._subscribed_channels.clear()
                logger.info("Audio forwarding stopped (no subscribers)")

    async def stop(self) -> None:
        """Stop all audio forwarding."""
        async with self._lock:
            self._subscriber_count = 0
            if self._audio_task is not None:
                self._stop_event.set()
                self._audio_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._audio_task
                self._audio_task = None
                self._subscribed_channels.clear()

    async def _forward_audio(self) -> None:
        """Forward audio from all voice channels to main process."""
        import base64

        logger.info("Audio forwarding: starting channel monitor")

        # Track queues per channel
        channel_queues: dict[str, asyncio.Queue[bytes]] = {}

        try:
            while not self._stop_event.is_set():
                # Check for new voice channels from any system
                for system in self._manager.list_systems():
                    voice_channels = system.get_voice_channels()
                    for vc in voice_channels:
                        if vc.id not in self._subscribed_channels and vc.state == "active":
                            # Subscribe to this voice channel
                            try:
                                queue = await vc.subscribe_audio("json")
                                channel_queues[vc.id] = queue
                                self._subscribed_channels.add(vc.id)
                                logger.info(f"Audio forwarding: subscribed to {vc.id}")
                            except Exception as e:
                                logger.error(f"Audio forwarding: failed to subscribe to {vc.id}: {e}")

                # Read from all subscribed queues (non-blocking)
                for channel_id, queue in list(channel_queues.items()):
                    try:
                        # Non-blocking check for audio data
                        data = queue.get_nowait()
                        # Forward as audio event
                        event = {
                            "type": "voice_audio",
                            "channelId": channel_id,
                            "data": data.decode("utf-8") if isinstance(data, bytes) else data,
                        }
                        try:
                            self._send_queue.put_nowait(event)
                        except queue_module.Full:
                            # Drop audio frame if queue is full
                            pass
                    except asyncio.QueueEmpty:
                        pass
                    except Exception as e:
                        logger.error(f"Audio forwarding error for {channel_id}: {e}")

                # Clean up ended channels
                ended_channels = []
                for system in self._manager.list_systems():
                    for vc in system.get_voice_channels():
                        if vc.id in self._subscribed_channels and vc.state == "ended":
                            ended_channels.append(vc)

                for vc in ended_channels:
                    if vc.id in channel_queues:
                        vc.unsubscribe(channel_queues[vc.id])
                        del channel_queues[vc.id]
                    self._subscribed_channels.discard(vc.id)
                    logger.info(f"Audio forwarding: unsubscribed from {vc.id} (ended)")

                await asyncio.sleep(_AUDIO_FORWARD_INTERVAL)

        except asyncio.CancelledError:
            # Clean up all subscriptions
            for system in self._manager.list_systems():
                for vc in system.get_voice_channels():
                    if vc.id in channel_queues:
                        vc.unsubscribe(channel_queues[vc.id])
            channel_queues.clear()
            self._subscribed_channels.clear()
            raise
        except Exception as e:
            logger.error(f"Audio forwarding error: {e}")


# Global audio forwarder for the worker (set in _worker_main)
_audio_forwarder: AudioForwarder | None = None
