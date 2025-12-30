import numpy as np

from wavecapsdr.validation import (
    validate_audio_samples,
    validate_discriminator_samples,
    validate_frequency_hz,
    validate_int_range,
)


def test_validate_audio_samples_rejects_non_finite() -> None:
    samples = np.array([0.0, np.nan], dtype=np.float32)
    ok, reason = validate_audio_samples(samples)
    assert not ok
    assert "non-finite" in reason


def test_validate_audio_samples_rejects_out_of_range() -> None:
    samples = np.array([0.0, 1.25], dtype=np.float32)
    ok, reason = validate_audio_samples(samples)
    assert not ok
    assert "exceeds" in reason


def test_validate_discriminator_samples_rejects_out_of_range() -> None:
    samples = np.array([0.0, 15.0], dtype=np.float32)
    ok, reason = validate_discriminator_samples(samples)
    assert not ok
    assert "exceeds" in reason


def test_validate_frequency_hz_rejects_invalid() -> None:
    ok, reason = validate_frequency_hz(-1.0)
    assert not ok
    assert "out of range" in reason


def test_validate_int_range_rejects_invalid() -> None:
    ok, reason = validate_int_range("nope", 0, 1, "value")
    assert not ok
    assert "not an int" in reason
