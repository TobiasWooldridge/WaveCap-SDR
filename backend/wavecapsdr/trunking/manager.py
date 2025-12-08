"""TrunkingManager - Top-level manager for multiple trunking systems.

This module provides the high-level interface for managing P25 trunking
systems. It handles:
- Creating and configuring trunking systems from YAML config
- Starting/stopping systems
- Routing events to WebSocket subscribers
- Managing shared SDR resources

Usage:
    manager = TrunkingManager()
    await manager.add_system(config)
    await manager.start_system("psern")
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from wavecapsdr.trunking.config import TrunkingSystemConfig, load_talkgroups_csv
from wavecapsdr.trunking.system import (
    TrunkingSystem,
    TrunkingSystemState,
    ActiveCall,
)

logger = logging.getLogger(__name__)


@dataclass
class TrunkingManager:
    """Top-level manager for P25 trunking systems.

    Manages multiple trunking systems (e.g., PSERN, SA-GRN) and provides
    a unified interface for control and monitoring.
    """

    # Systems keyed by ID
    _systems: Dict[str, TrunkingSystem] = field(default_factory=dict)

    # Event subscribers
    _event_queues: Set[asyncio.Queue[Dict[str, Any]]] = field(default_factory=set)

    # Background tasks
    _maintenance_task: Optional[asyncio.Task[None]] = None
    _running: bool = False

    def __post_init__(self) -> None:
        """Initialize the manager."""
        logger.info("TrunkingManager initialized")

    async def start(self) -> None:
        """Start the trunking manager.

        Starts background maintenance tasks. Individual systems must be
        started separately with start_system().
        """
        if self._running:
            return

        self._running = True

        # Start maintenance task
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())

        logger.info("TrunkingManager started")

    async def stop(self) -> None:
        """Stop the trunking manager and all systems."""
        if not self._running:
            return

        self._running = False

        # Stop all systems
        for system_id in list(self._systems.keys()):
            await self.stop_system(system_id)

        # Cancel maintenance task
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
            self._maintenance_task = None

        logger.info("TrunkingManager stopped")

    async def add_system(
        self,
        config: TrunkingSystemConfig,
        talkgroups_csv: Optional[str] = None,
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

        self._systems[config.id] = system

        logger.info(f"Added trunking system: {config.id} ({config.name})")

        # Broadcast event
        await self._broadcast_event({
            "type": "system_added",
            "systemId": config.id,
            "system": system.to_dict(),
        })

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

    async def start_system(self, system_id: str) -> None:
        """Start a trunking system.

        Args:
            system_id: System ID to start
        """
        if system_id not in self._systems:
            raise ValueError(f"System '{system_id}' not found")

        system = self._systems[system_id]
        await system.start()

    async def stop_system(self, system_id: str) -> None:
        """Stop a trunking system.

        Args:
            system_id: System ID to stop
        """
        if system_id not in self._systems:
            raise ValueError(f"System '{system_id}' not found")

        system = self._systems[system_id]
        await system.stop()

    def get_system(self, system_id: str) -> Optional[TrunkingSystem]:
        """Get a trunking system by ID.

        Args:
            system_id: System ID

        Returns:
            TrunkingSystem or None if not found
        """
        return self._systems.get(system_id)

    def list_systems(self) -> List[TrunkingSystem]:
        """Get all trunking systems.

        Returns:
            List of TrunkingSystem instances
        """
        return list(self._systems.values())

    def get_active_calls(self, system_id: Optional[str] = None) -> List[ActiveCall]:
        """Get active calls across all systems or a specific system.

        Args:
            system_id: Optional system ID to filter by

        Returns:
            List of ActiveCall instances
        """
        calls: List[ActiveCall] = []

        if system_id:
            system = self._systems.get(system_id)
            if system:
                calls.extend(system.get_active_calls())
        else:
            for system in self._systems.values():
                calls.extend(system.get_active_calls())

        return calls

    async def subscribe_events(self) -> asyncio.Queue[Dict[str, Any]]:
        """Subscribe to trunking events.

        Returns:
            Queue that will receive event dictionaries
        """
        queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=100)
        self._event_queues.add(queue)

        # Send initial snapshot
        snapshot = {
            "type": "snapshot",
            "systems": [s.to_dict() for s in self._systems.values()],
            "activeCalls": [c.to_dict() for c in self.get_active_calls()],
        }
        try:
            queue.put_nowait(snapshot)
        except asyncio.QueueFull:
            pass

        return queue

    async def unsubscribe_events(self, queue: asyncio.Queue[Dict[str, Any]]) -> None:
        """Unsubscribe from trunking events.

        Args:
            queue: Queue previously returned by subscribe_events()
        """
        self._event_queues.discard(queue)

    async def _broadcast_event(self, event: Dict[str, Any]) -> None:
        """Broadcast an event to all subscribers."""
        dead_queues: List[asyncio.Queue[Dict[str, Any]]] = []

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

    def _on_call_start(self, system_id: str, call: ActiveCall) -> None:
        """Handle call start event."""
        asyncio.create_task(self._broadcast_event({
            "type": "call_start",
            "systemId": system_id,
            "call": call.to_dict(),
        }))

    def _on_call_update(self, system_id: str, call: ActiveCall) -> None:
        """Handle call update event."""
        asyncio.create_task(self._broadcast_event({
            "type": "call_update",
            "systemId": system_id,
            "call": call.to_dict(),
        }))

    def _on_call_end(self, system_id: str, call: ActiveCall) -> None:
        """Handle call end event."""
        asyncio.create_task(self._broadcast_event({
            "type": "call_end",
            "systemId": system_id,
            "call": call.to_dict(),
        }))

    def _on_system_update(self, system: TrunkingSystem) -> None:
        """Handle system state update event."""
        asyncio.create_task(self._broadcast_event({
            "type": "system_update",
            "systemId": system.cfg.id,
            "system": system.to_dict(),
        }))

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

    def to_dict(self) -> Dict[str, Any]:
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
