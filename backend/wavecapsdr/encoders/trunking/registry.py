"""Registry helpers for trunking encoders."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from wavecapsdr.encoders.trunking.base import (
    ControlChannelEncoder,
    TrafficChannelEncoder,
)
from wavecapsdr.trunking.config import TrunkingProtocol

ControlChannelEncoderFactory = Callable[[TrunkingProtocol], ControlChannelEncoder]
TrafficChannelEncoderFactory = Callable[[TrunkingProtocol], TrafficChannelEncoder]


@dataclass
class EncoderRegistration:
    """Factory registration for a trunking protocol."""
    control_factory: ControlChannelEncoderFactory | None = None
    traffic_factory: TrafficChannelEncoderFactory | None = None

    def __post_init__(self) -> None:
        if self.control_factory is None and self.traffic_factory is None:
            raise ValueError("At least one encoder factory must be provided")


class TrunkingEncoderRegistry:
    """Registry for protocol-specific trunking encoders."""

    def __init__(self) -> None:
        self._control_factories: dict[TrunkingProtocol, ControlChannelEncoderFactory] = {}
        self._traffic_factories: dict[TrunkingProtocol, TrafficChannelEncoderFactory] = {}

    def register(self, protocol: TrunkingProtocol, registration: EncoderRegistration) -> None:
        """Register encoder factories for a protocol."""
        if registration.control_factory:
            self._control_factories[protocol] = registration.control_factory
        if registration.traffic_factory:
            self._traffic_factories[protocol] = registration.traffic_factory

    def get_control_encoder(self, protocol: TrunkingProtocol) -> ControlChannelEncoder:
        """Create a control-channel encoder for a protocol."""
        factory = self._control_factories.get(protocol)
        if factory is None:
            raise KeyError(f"No control-channel encoder registered for {protocol}")
        encoder = factory(protocol)
        self._validate_encoder(encoder, protocol)
        return encoder

    def get_traffic_encoder(self, protocol: TrunkingProtocol) -> TrafficChannelEncoder:
        """Create a traffic-channel encoder for a protocol."""
        factory = self._traffic_factories.get(protocol)
        if factory is None:
            raise KeyError(f"No traffic-channel encoder registered for {protocol}")
        encoder = factory(protocol)
        self._validate_encoder(encoder, protocol)
        return encoder

    @staticmethod
    def _validate_encoder(
        encoder: ControlChannelEncoder | TrafficChannelEncoder,
        protocol: TrunkingProtocol,
    ) -> None:
        if encoder.protocol != protocol:
            raise ValueError(
                f"Encoder protocol mismatch: expected {protocol}, got {encoder.protocol}"
            )
        if encoder.symbol_rate <= 0:
            raise ValueError("Encoder symbol_rate must be positive")
