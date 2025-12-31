"""Encoders for common P25 Phase 1 TSBK control PDUs.

These helpers build the 8-byte opcode payloads using the same bit layouts
implemented by SDRTrunk and the WaveCap TSBK decoder. They keep the field
ordering consistent so encode→decode tests can validate round trips without
reimplementing the trellis/interleaver layers.
"""

from __future__ import annotations

from wavecapsdr.decoders.tsbk_utils import bits_to_payload, write_field
from wavecapsdr.validation import (
    TGID_MIN,
    TGID_MAX,
    UNIT_ID_MAX,
    UNIT_ID_MIN,
    SYSTEM_ID_MAX,
    WACN_MAX,
    validate_int_range,
)


def _new_payload() -> list[int]:
    return [0] * 64


def encode_status_update(
    unit_status: int,
    user_status: int,
    target_id: int,
    source_id: int,
) -> bytes:
    """Encode STATUS_UPDATE/STATUS_UPDATE_REQUEST payload."""
    _require_byte(unit_status, "unit_status")
    _require_byte(user_status, "user_status")
    _require_unit_id(target_id, "target_id")
    _require_unit_id(source_id, "source_id")

    bits = _new_payload()
    write_field(bits, 0, 8, unit_status)
    write_field(bits, 8, 8, user_status)
    write_field(bits, 16, 24, target_id)
    write_field(bits, 40, 24, source_id)
    return bits_to_payload(bits)


def encode_group_affiliation_response(
    response_code: int,
    announcement_group: int,
    group_address: int,
    target_id: int,
    *,
    global_affiliation: bool = False,
) -> bytes:
    """Encode GROUP_AFFILIATION_RESPONSE payload (opcode 0x28)."""
    _require_response_code(response_code)
    _require_tgid(group_address, "group_address")
    _require_tgid(announcement_group, "announcement_group")
    _require_unit_id(target_id, "target_id")

    bits = _new_payload()
    if global_affiliation:
        write_field(bits, 0, 1, 1)
    write_field(bits, 6, 2, response_code)
    write_field(bits, 8, 16, announcement_group)
    write_field(bits, 24, 16, group_address)
    write_field(bits, 40, 24, target_id)
    return bits_to_payload(bits)


def encode_unit_registration_request(
    wacn: int,
    system_id: int,
    source_id: int,
    capability: int = 0,
    *,
    emergency: bool = False,
) -> bytes:
    """Encode UNIT_REGISTRATION_REQUEST payload (opcode 0x2C ISP)."""
    _require_wacn(wacn)
    _require_system_id(system_id)
    _require_unit_id(source_id, "source_id")
    _require_byte(capability, "capability")

    bits = _new_payload()
    if emergency:
        write_field(bits, 0, 1, 1)
    write_field(bits, 1, 7, capability)
    write_field(bits, 8, 20, wacn)
    write_field(bits, 28, 12, system_id)
    write_field(bits, 40, 24, source_id)
    return bits_to_payload(bits)


def encode_unit_deregistration_request(
    wacn: int,
    system_id: int,
    source_id: int,
) -> bytes:
    """Encode UNIT_DEREGISTRATION_REQUEST payload (opcode 0x2F ISP)."""
    _require_wacn(wacn)
    _require_system_id(system_id)
    _require_unit_id(source_id, "source_id")

    bits = _new_payload()
    write_field(bits, 8, 20, wacn)
    write_field(bits, 28, 12, system_id)
    write_field(bits, 40, 24, source_id)
    return bits_to_payload(bits)


def encode_unit_registration_response(
    response_code: int,
    system_id: int,
    source_id: int,
    registered_address: int | None = None,
) -> bytes:
    """Encode UNIT_REGISTRATION_RESPONSE payload (opcode 0x2C OSP)."""
    _require_response_code(response_code)
    _require_system_id(system_id)
    _require_unit_id(source_id, "source_id")
    address = source_id if registered_address is None else registered_address
    _require_unit_id(address, "registered_address")

    bits = _new_payload()
    write_field(bits, 2, 2, response_code)
    write_field(bits, 4, 12, system_id)
    write_field(bits, 16, 24, source_id)
    write_field(bits, 40, 24, address)
    return bits_to_payload(bits)


def encode_unit_deregistration_ack(
    wacn: int,
    system_id: int,
    target_id: int,
) -> bytes:
    """Encode UNIT_DEREGISTRATION_ACK payload (opcode 0x2F OSP)."""
    _require_wacn(wacn)
    _require_system_id(system_id)
    _require_unit_id(target_id, "target_id")

    bits = _new_payload()
    write_field(bits, 8, 20, wacn)
    write_field(bits, 28, 12, system_id)
    write_field(bits, 40, 24, target_id)
    return bits_to_payload(bits)


def encode_unit_service_request(
    service_options: int,
    target_id: int,
    source_id: int,
) -> bytes:
    """Encode UNIT-TO-UNIT voice service request payload (opcode 0x05 ISP)."""
    _require_byte(service_options, "service_options")
    _require_unit_id(target_id, "target_id")
    _require_unit_id(source_id, "source_id")

    bits = _new_payload()
    write_field(bits, 0, 8, service_options)
    write_field(bits, 16, 24, target_id)
    write_field(bits, 40, 24, source_id)
    return bits_to_payload(bits)


def _require_unit_id(value: int, label: str) -> None:
    ok, reason = validate_int_range(value, UNIT_ID_MIN, UNIT_ID_MAX, label)
    if not ok:
        raise ValueError(reason)


def _require_tgid(value: int, label: str) -> None:
    ok, reason = validate_int_range(value, TGID_MIN, TGID_MAX, label)
    if not ok:
        raise ValueError(reason)


def _require_byte(value: int, label: str) -> None:
    ok, reason = validate_int_range(value, 0, 0xFF, label)
    if not ok:
        raise ValueError(reason)


def _require_response_code(value: int) -> None:
    ok, reason = validate_int_range(value, 0, 3, "response_code")
    if not ok:
        raise ValueError(reason)


def _require_system_id(value: int) -> None:
    ok, reason = validate_int_range(value, 0, SYSTEM_ID_MAX, "system_id")
    if not ok:
        raise ValueError(reason)


def _require_wacn(value: int) -> None:
    ok, reason = validate_int_range(value, 0, WACN_MAX, "wacn")
    if not ok:
        raise ValueError(reason)
