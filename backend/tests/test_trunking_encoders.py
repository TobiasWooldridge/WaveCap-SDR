import numpy as np
import pytest

from wavecapsdr.encoders.trunking import (
    ControlChannelFrame,
    P25ControlChannelEncoder,
    P25TrafficChannelEncoder,
    TrafficChannelFrame,
    TrunkingEncoderRegistry,
    get_control_channel_encoder,
    get_traffic_channel_encoder,
    register_builtin_encoders,
)
from wavecapsdr.trunking.config import TrunkingProtocol


def test_control_channel_frame_validation() -> None:
    with pytest.raises(ValueError):
        ControlChannelFrame(dibits=[], symbol_rate=4800)
    with pytest.raises(ValueError):
        ControlChannelFrame(dibits=[0, 1, 2, 4], symbol_rate=4800)
    with pytest.raises(ValueError):
        ControlChannelFrame(dibits=[0, 1], symbol_rate=0)


def test_p25_control_encoder_maps_dibits_to_symbols() -> None:
    frame = ControlChannelFrame(dibits=[0, 1, 2, 3], symbol_rate=4800)
    encoder = P25ControlChannelEncoder(protocol=TrunkingProtocol.P25_PHASE1)

    encoded = encoder.encode(frame=frame, sample_rate=48_000)

    assert encoded.shape[0] == 4 * 10  # 48000/4800 samples per symbol
    # Validate mapping for 0,1,2,3 -> +1,+3,-1,-3
    assert encoded[:10].tolist() == [1.0] * 10
    assert encoded[10:20].tolist() == [3.0] * 10
    assert encoded[20:30].tolist() == [-1.0] * 10
    assert encoded[30:40].tolist() == [-3.0] * 10


def test_p25_phase2_traffic_encoder_uses_phase2_symbol_rate() -> None:
    frame = TrafficChannelFrame(dibits=[1, 1, 0], symbol_rate=6000)
    encoder = P25TrafficChannelEncoder(protocol=TrunkingProtocol.P25_PHASE2)

    encoded = encoder.encode(frame=frame, sample_rate=48_000)

    # 48000 / 6000 = 8 samples per symbol
    assert encoded.shape[0] == 3 * 8
    assert encoder.symbol_rate == 6000
    assert np.all(encoded[:16] == 3.0)
    assert np.all(encoded[16:] == 1.0)


def test_registry_dispatches_builtins() -> None:
    registry = TrunkingEncoderRegistry()

    with pytest.raises(KeyError):
        registry.get_control_encoder(TrunkingProtocol.P25_PHASE1)

    register_builtin_encoders(registry)

    control = registry.get_control_encoder(TrunkingProtocol.P25_PHASE1)
    traffic = registry.get_traffic_encoder(TrunkingProtocol.P25_PHASE2)

    assert isinstance(control, P25ControlChannelEncoder)
    assert isinstance(traffic, P25TrafficChannelEncoder)


def test_module_level_registry_available() -> None:
    control = get_control_channel_encoder(TrunkingProtocol.P25_PHASE1)
    traffic = get_traffic_channel_encoder(TrunkingProtocol.P25_PHASE1)

    assert isinstance(control, P25ControlChannelEncoder)
    assert isinstance(traffic, P25TrafficChannelEncoder)


def test_symbol_rate_mismatch_is_rejected() -> None:
    encoder = P25ControlChannelEncoder(protocol=TrunkingProtocol.P25_PHASE1)
    mismatched_frame = ControlChannelFrame(dibits=[0, 0, 0], symbol_rate=6000)

    with pytest.raises(ValueError):
        encoder.encode(frame=mismatched_frame, sample_rate=48_000)
