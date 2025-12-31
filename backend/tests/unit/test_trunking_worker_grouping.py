from __future__ import annotations

from wavecapsdr.trunking.config import TrunkingSystemConfig
from wavecapsdr.trunking.process_manager import group_trunking_systems_by_device


def _system(system_id: str, device_id: str) -> TrunkingSystemConfig:
    return TrunkingSystemConfig(
        id=system_id,
        name=f"System {system_id}",
        control_channels=[851_000_000],
        center_hz=855_000_000,
        sample_rate=2_000_000,
        device_id=device_id,
    )


def test_group_trunking_systems_by_device() -> None:
    configs = [
        _system("a", "devA"),
        _system("b", "devA"),
        _system("c", ""),
    ]

    grouped = group_trunking_systems_by_device(configs)

    assert set(grouped.keys()) == {"devA", "auto"}
    assert [cfg.id for cfg in grouped["devA"]] == ["a", "b"]
    assert [cfg.id for cfg in grouped["auto"]] == ["c"]
