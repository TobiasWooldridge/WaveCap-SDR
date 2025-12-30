"""TrunkingManager - Top-level manager for multiple trunking systems.

This module provides the high-level interface for managing P25 trunking
systems. It handles:
- Creating and configuring trunking systems from YAML config
- Starting/stopping systems
- Routing events to WebSocket subscribers
- Managing shared SDR resources

Usage:
    manager = TrunkingManager()
    manager.set_capture_manager(capture_manager)  # Required for SDR access
    await manager.add_system(config)
    await manager.start_system("psern")
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from wavecapsdr.config import default_config_path, update_trunking_system_state
from wavecapsdr.trunking.config import TrunkingSystemConfig, load_talkgroups_csv
from wavecapsdr.trunking.system import (
    ActiveCall,
    TrunkingSystem,
    TrunkingSystemState,
)

if TYPE_CHECKING:
    from wavecapsdr.capture import CaptureManager

logger = logging.getLogger(__name__)

# Default config file path
DEFAULT_CONFIG_PATH = default_config_path()


@dataclass
class TrunkingManager:
    """Top-level manager for P25 trunking systems.

    Manages multiple trunking systems (e.g., PSERN, SA-GRN) and provides
    a unified interface for control and monitoring.
    """

    # Systems keyed by ID
    _systems: dict[str, TrunkingSystem] = field(default_factory=dict)

    # Event subscribers
    _event_queues: set[asyncio.Queue[dict[str, Any]]] = field(default_factory=set)

    # Background tasks
    _maintenance_task: asyncio.Task[None] | None = None
    _running: bool = False
    _event_loop: asyncio.AbstractEventLoop | None = None

    # Reference to CaptureManager for SDR access
    _capture_manager: CaptureManager | None = None

    # Pending configs to load on start()
    _pending_configs: list[TrunkingSystemConfig] = field(default_factory=list)

    # Config file path for state persistence
    _config_path: str = DEFAULT_CONFIG_PATH

    def __post_init__(self) -> None:
        """Initialize the manager."""
        self._pending_configs = []
        logger.info("TrunkingManager initialized")

    def set_config_path(self, config_path: str) -> None:
        """Set the config file path for state persistence.

        Args:
            config_path: Path to the config file
        """
        self._config_path = config_path
        logger.debug(f"TrunkingManager: Config path set to {config_path}")

    def set_capture_manager(self, capture_manager: CaptureManager) -> None:
        """Set the CaptureManager reference for SDR access.

        Must be called before starting any trunking systems.

        Args:
            capture_manager: CaptureManager instance
        """
        self._capture_manager = capture_manager
        logger.info("TrunkingManager: CaptureManager reference set")

    def register_config(self, config: TrunkingSystemConfig) -> None:
        """Register a system config to be loaded during start().

        This is a synchronous method for use during app initialization.
        Systems are not actually created until start() is called.

        Args:
            config: System configuration to register
        """
        self._pending_configs.append(config)
        logger.info(f"Registered trunking system config: {config.id} ({config.name})")

    async def start(self) -> None:
        """Start the trunking manager.

        Starts background maintenance tasks and loads any registered configs.
        Individual systems must be started separately with start_system().
        """
        if self._running:
            return

        self._running = True
        try:
            self._event_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._event_loop = None
            logger.warning("TrunkingManager: No running event loop; events may be dropped")

        # Load pending system configs
        for config in self._pending_configs:
            try:
                await self.add_system(config)
            except Exception as e:
                logger.error(f"Failed to load trunking system '{config.id}': {e}")
        self._pending_configs.clear()

        # Start maintenance task
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())

        logger.info("TrunkingManager started")

    async def stop(self) -> None:
        """Stop the trunking manager and all systems.

        This is a graceful shutdown - it does NOT persist auto_start=false
        so that systems will automatically restart on the next server start.
        """
        if not self._running:
            return

        self._running = False

        # Stop all systems without persisting (graceful shutdown preserves restart state)
        for system_id in list(self._systems.keys()):
            await self.stop_system(system_id, persist=False)

        # Cancel maintenance task
        if self._maintenance_task:
            self._maintenance_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._maintenance_task
            self._maintenance_task = None

        logger.info("TrunkingManager stopped")

    async def add_system(
        self,
        config: TrunkingSystemConfig,
        talkgroups_csv: str | None = None,
    ) -> TrunkingSystem:
        """Add a trunking system.

        Args:
            config: System configuration
            talkgroups_csv: Optional path to trunk-recorder format CSV for talkgroups

        Returns:
            Created TrunkingSystem instance
        """
        if config.id in self._systems:
            raise ValueError(f"System '{config.id}' already exists")

        # Load talkgroups from CSV if provided
        if talkgroups_csv:
            talkgroups = load_talkgroups_csv(talkgroups_csv)
            config.talkgroups.update(talkgroups)
            logger.info(
                f"Loaded {len(talkgroups)} talkgroups from {talkgroups_csv}"
            )

        # Create system
        system = TrunkingSystem(cfg=config)

        # Wire up event callbacks
        system.on_call_start = lambda call: self._on_call_start(config.id, call)
        system.on_call_update = lambda call: self._on_call_update(config.id, call)
        system.on_call_end = lambda call: self._on_call_end(config.id, call)
        system.on_system_update = lambda sys: self._on_system_update(sys)
        system.on_message = lambda msg: self._on_message(config.id, msg)

        self._systems[config.id] = system

        logger.info(f"Added trunking system: {config.id} ({config.name})")

        # Broadcast event
        await self._broadcast_event({
            "type": "system_added",
            "systemId": config.id,
            "system": system.to_dict(),
        })

        # Auto-start if configured (persist=False since config already has auto_start=true)
        if config.auto_start:
            logger.info(f"Auto-starting trunking system: {config.id}")
            try:
                await self.start_system(config.id, persist=False)
            except Exception as e:
                logger.error(f"Failed to auto-start system '{config.id}': {e}")

        return system

    async def remove_system(self, system_id: str) -> None:
        """Remove a trunking system.

        Args:
            system_id: System ID to remove
        """
        if system_id not in self._systems:
            raise ValueError(f"System '{system_id}' not found")

        system = self._systems[system_id]

        # Stop if running
        if system.state not in (TrunkingSystemState.STOPPED, TrunkingSystemState.FAILED):
            await self.stop_system(system_id)

        del self._systems[system_id]

        logger.info(f"Removed trunking system: {system_id}")

        # Broadcast event
        await self._broadcast_event({
            "type": "system_removed",
            "systemId": system_id,
        })

    async def start_system(self, system_id: str, persist: bool = True) -> None:
        """Start a trunking system.

        Args:
            system_id: System ID to start
            persist: Whether to persist auto_start=true to config file

        Raises:
            ValueError: If system not found
            RuntimeError: If CaptureManager not set
        """
        if system_id not in self._systems:
            raise ValueError(f"System '{system_id}' not found")

        if self._capture_manager is None:
            raise RuntimeError(
                "CaptureManager not set. Call set_capture_manager() before starting systems."
            )

        system = self._systems[system_id]
        await system.start(self._capture_manager)

        # Persist auto_start=true to config
        if persist:
            try:
                update_trunking_system_state(self._config_path, system_id, True)
                logger.info(f"Persisted auto_start=true for system '{system_id}'")
            except Exception as e:
                logger.warning(f"Failed to persist state for '{system_id}': {e}")

    async def stop_system(self, system_id: str, persist: bool = True) -> None:
        """Stop a trunking system.

        Args:
            system_id: System ID to stop
            persist: Whether to persist auto_start=false to config file.
                     Set to False during graceful shutdown to preserve restart state.
        """
        if system_id not in self._systems:
            raise ValueError(f"System '{system_id}' not found")

        system = self._systems[system_id]
        await system.stop()

        # Persist auto_start=false to config (unless during graceful shutdown)
        if persist:
            try:
                update_trunking_system_state(self._config_path, system_id, False)
                logger.info(f"Persisted auto_start=false for system '{system_id}'")
            except Exception as e:
                logger.warning(f"Failed to persist state for '{system_id}': {e}")

    def get_system(self, system_id: str) -> TrunkingSystem | None:
        """Get a trunking system by ID.

        Args:
            system_id: System ID

        Returns:
            TrunkingSystem or None if not found
        """
        return self._systems.get(system_id)

    def get_system_for_capture(self, capture_id: str) -> str | None:
        """Get the trunking system ID that owns a capture.

        Args:
            capture_id: Capture ID to look up

        Returns:
            Trunking system ID if the capture is owned by a trunking system,
            None otherwise
        """
        for system in self._systems.values():
            if system._capture is not None and system._capture.cfg.id == capture_id:
                return system.cfg.id
        return None

    def list_systems(self) -> list[TrunkingSystem]:
        """Get all trunking systems.

        Returns:
            List of TrunkingSystem instances
        """
        return list(self._systems.values())

    def get_active_calls(self, system_id: str | None = None) -> list[ActiveCall]:
        """Get active calls across all systems or a specific system.

        Args:
            system_id: Optional system ID to filter by

        Returns:
            List of ActiveCall instances
        """
        calls: list[ActiveCall] = []

        if system_id:
            system = self._systems.get(system_id)
            if system:
                calls.extend(system.get_active_calls())
        else:
            for system in self._systems.values():
                calls.extend(system.get_active_calls())

        return calls

    async def subscribe_events(self) -> asyncio.Queue[dict[str, Any]]:
        """Subscribe to trunking events.

        Returns:
            Queue that will receive event dictionaries
        """
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        self._event_queues.add(queue)

        # Send initial snapshot with buffered messages and call history
        # Aggregate messages and call history from all systems
        all_messages: list[dict[str, Any]] = []
        all_call_history: list[dict[str, Any]] = []
        for system in self._systems.values():
            # Add system ID to each message/call for frontend routing
            # Transform messages to match WebSocket format (camelCase)
            for msg in system.get_messages(limit=200):
                all_messages.append({
                    "systemId": system.cfg.id,
                    "timestamp": msg.get("timestamp", 0),
                    "opcode": msg.get("opcode", 0),
                    "opcodeName": msg.get("opcode_name", ""),
                    "nac": msg.get("nac"),
                    "summary": msg.get("summary", ""),
                })
            for call in system.get_call_history(limit=50):
                call["systemId"] = system.cfg.id
                all_call_history.append(call)

        # Sort by timestamp (newest first) and limit
        all_messages.sort(key=lambda m: m.get("timestamp", 0), reverse=True)
        all_call_history.sort(key=lambda c: c.get("endTime", c.get("startTime", 0)), reverse=True)

        snapshot = {
            "type": "snapshot",
            "systems": [s.to_dict() for s in self._systems.values()],
            "activeCalls": [c.to_dict() for c in self.get_active_calls()],
            "messages": all_messages[:200],  # Last 200 messages across all systems
            "callHistory": all_call_history[:100],  # Last 100 calls across all systems
        }
        with contextlib.suppress(asyncio.QueueFull):
            queue.put_nowait(snapshot)

        return queue

    async def unsubscribe_events(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Unsubscribe from trunking events.

        Args:
            queue: Queue previously returned by subscribe_events()
        """
        self._event_queues.discard(queue)

    async def _broadcast_event(self, event: dict[str, Any]) -> None:
        """Broadcast an event to all subscribers."""
        dead_queues: list[asyncio.Queue[dict[str, Any]]] = []

        for queue in self._event_queues:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drop oldest and retry
                try:
                    queue.get_nowait()
                    queue.put_nowait(event)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass
            except Exception:
                dead_queues.append(queue)

        # Clean up dead queues
        for queue in dead_queues:
            self._event_queues.discard(queue)

    def _schedule_broadcast(self, event: dict[str, Any]) -> None:
        """Schedule an event broadcast on the manager event loop."""
        loop = self._event_loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(lambda: loop.create_task(self._broadcast_event(event)))
            return

        try:
            running_loop = asyncio.get_running_loop()
            running_loop.create_task(self._broadcast_event(event))
        except RuntimeError:
            logger.warning(
                "TrunkingManager: Dropping event %s (no event loop)",
                event.get("type"),
            )

    def _on_call_start(self, system_id: str, call: ActiveCall) -> None:
        """Handle call start event."""
        self._schedule_broadcast({
            "type": "call_start",
            "systemId": system_id,
            "call": call.to_dict(),
        })

    def _on_call_update(self, system_id: str, call: ActiveCall) -> None:
        """Handle call update event."""
        self._schedule_broadcast({
            "type": "call_update",
            "systemId": system_id,
            "call": call.to_dict(),
        })

    def _on_call_end(self, system_id: str, call: ActiveCall) -> None:
        """Handle call end event."""
        self._schedule_broadcast({
            "type": "call_end",
            "systemId": system_id,
            "callId": call.id,
            "call": call.to_dict(),
        })

    def _on_system_update(self, system: TrunkingSystem) -> None:
        """Handle system state update event."""
        self._schedule_broadcast({
            "type": "system_update",
            "systemId": system.cfg.id,
            "system": system.to_dict(),
        })

    def _on_message(self, system_id: str, message: dict[str, Any]) -> None:
        """Handle decoded message event.

        Broadcasts the message to WebSocket subscribers for real-time display.
        """
        self._schedule_broadcast({
            "type": "message",
            "systemId": system_id,
            "message": {
                "timestamp": message.get("timestamp", 0),
                "opcode": message.get("opcode", 0),
                "opcodeName": message.get("opcode_name", ""),
                "nac": message.get("nac"),
                "summary": message.get("summary", ""),
            },
        })

    async def _maintenance_loop(self) -> None:
        """Background maintenance loop.

        Runs periodically to:
        - Check for timed-out calls
        - Update decode rate statistics
        - Clean up stale resources
        """
        while self._running:
            try:
                # Check call timeouts
                for system in self._systems.values():
                    if system.state == TrunkingSystemState.RUNNING:
                        system.check_call_timeouts()

                # Brief sleep
                await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"TrunkingManager maintenance error: {e}")
                await asyncio.sleep(1.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert manager state to dictionary for API serialization."""
        total_calls = 0
        for system in self._systems.values():
            total_calls += len(system.get_active_calls())

        return {
            "systemCount": len(self._systems),
            "systems": [s.to_dict() for s in self._systems.values()],
            "totalActiveCalls": total_calls,
            "subscriberCount": len(self._event_queues),
        }
