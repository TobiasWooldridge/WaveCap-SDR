import logging

from wavecapsdr.utils.log_levels import log_level_name, parse_log_level


def test_parse_log_level_named() -> None:
    assert parse_log_level("info", logging.DEBUG) == logging.INFO
    assert parse_log_level("WARN", logging.INFO) == logging.WARNING
    assert parse_log_level("fatal", logging.INFO) == logging.CRITICAL


def test_parse_log_level_numeric() -> None:
    assert parse_log_level("10", logging.INFO) == 10


def test_parse_log_level_unknown_uses_default() -> None:
    assert parse_log_level("nope", logging.WARNING) == logging.WARNING


def test_log_level_name_known() -> None:
    assert log_level_name("WARNING", logging.INFO) == "warning"
    assert log_level_name("10", logging.INFO) == "debug"


def test_log_level_name_unknown_uses_default() -> None:
    assert log_level_name("15", logging.INFO) == "info"
