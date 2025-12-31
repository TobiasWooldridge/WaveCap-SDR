from wavecapsdr.dsp.flex import parse_flex_json_message


def test_parse_flex_alphanumeric_message() -> None:
    payload = {
        "demod_name": "flex_alphanumeric",
        "capcode": 123456789,
        "message": "DISPATCH 123",
        "sync_baud": 1600,
        "sync_level": 4,
        "phase_number": "A",
        "cycle_number": 12,
        "frame_number": 345,
    }

    msg = parse_flex_json_message(payload)

    assert msg is not None
    assert msg.capcode == 123456789
    assert msg.message_type == "alpha"
    assert msg.message == "DISPATCH 123"
    assert msg.baud_rate == 1600
    assert msg.levels == 4
    assert msg.phase == "A"
    assert msg.cycle_number == 12
    assert msg.frame_number == 345


def test_parse_flex_numeric_message() -> None:
    payload = {
        "demod_name": "flex_numeric",
        "capcode": 987654321,
        "message": "404",
        "sync_baud": 3200,
        "sync_level": 2,
    }

    msg = parse_flex_json_message(payload)

    assert msg is not None
    assert msg.capcode == 987654321
    assert msg.message_type == "numeric"
    assert msg.message == "404"
    assert msg.baud_rate == 3200
    assert msg.levels == 2


def test_parse_flex_tone_message() -> None:
    payload = {
        "demod_name": "flex_tone_only",
        "capcode": 246801357,
        "tone": "TONE 1",
        "tone_long": "TONE 1 (Long)",
    }

    msg = parse_flex_json_message(payload)

    assert msg is not None
    assert msg.capcode == 246801357
    assert msg.message_type == "tone"
    assert msg.message == "TONE 1 (Long)"


def test_parse_flex_ignores_non_flex_payload() -> None:
    payload = {"demod_name": "pocsag_numeric", "address": 1234}

    msg = parse_flex_json_message(payload)

    assert msg is None
