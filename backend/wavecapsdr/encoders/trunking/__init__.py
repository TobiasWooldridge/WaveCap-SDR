"""Trunking encoder package with protocol dispatch."""

from __future__ import annotations

from wavecapsdr.encoders.trunking.base import (
    ControlChannelEncoder,
    ControlChannelFrame,
    TrafficChannelEncoder,
    TrafficChannelFrame,
)
from wavecapsdr.encoders.trunking.p25 import (
    P25ControlChannelEncoder,
    P25TrafficChannelEncoder,
)
from wavecapsdr.encoders.trunking.registry import (
    ControlChannelEncoderFactory,
    EncoderRegistration,
    TrafficChannelEncoderFactory,
    TrunkingEncoderRegistry,
)
from wavecapsdr.trunking.config import TrunkingProtocol

__all__ = [
    "ControlChannelEncoder",
    "ControlChannelEncoderFactory",
    "ControlChannelFrame",
    "EncoderRegistration",
    "P25ControlChannelEncoder",
    "P25TrafficChannelEncoder",
    "TrafficChannelEncoder",
    "TrafficChannelEncoderFactory",
    "TrafficChannelFrame",
    "TrunkingEncoderRegistry",
    "get_control_channel_encoder",
    "get_traffic_channel_encoder",
    "register_builtin_encoders",
]


def register_builtin_encoders(registry: TrunkingEncoderRegistry) -> None:
    """Register built-in encoders for supported protocols."""
    registration = EncoderRegistration(
        control_factory=lambda proto: P25ControlChannelEncoder(protocol=proto),
        traffic_factory=lambda proto: P25TrafficChannelEncoder(protocol=proto),
    )
    registry.register(TrunkingProtocol.P25_PHASE1, registration)
    registry.register(TrunkingProtocol.P25_PHASE2, registration)


_DEFAULT_REGISTRY = TrunkingEncoderRegistry()
register_builtin_encoders(_DEFAULT_REGISTRY)


def get_control_channel_encoder(protocol: TrunkingProtocol) -> ControlChannelEncoder:
    """Fetch a control-channel encoder for the specified protocol."""
    return _DEFAULT_REGISTRY.get_control_encoder(protocol)


def get_traffic_channel_encoder(protocol: TrunkingProtocol) -> TrafficChannelEncoder:
    """Fetch a traffic-channel encoder for the specified protocol."""
    return _DEFAULT_REGISTRY.get_traffic_encoder(protocol)
