from __future__ import annotations

import math
from typing import Any

import numpy as np

AUDIO_MAX_ABS = 1.2
DISC_AUDIO_MAX_ABS = 10.0

FREQ_MIN_HZ = 1e6
FREQ_MAX_HZ = 1e10

TGID_MIN = 1
TGID_MAX = 0xFFFF
UNIT_ID_MIN = 0
UNIT_ID_MAX = 0xFFFFFF

SYSTEM_ID_MAX = 0xFFF
WACN_MAX = 0xFFFFF
RFSS_ID_MAX = 0xFF
SITE_ID_MAX = 0xFF

CHANNEL_ID_MIN = 1
CHANNEL_ID_MAX = 0xFFFF
CHANNEL_NUMBER_MAX = 0xFFF
IDENTIFIER_MAX = 0xF

BASE_FREQ_MIN_MHZ = 1.0
BASE_FREQ_MAX_MHZ = 10000.0
CHANNEL_SPACING_MIN_KHZ = 0.125
CHANNEL_SPACING_MAX_KHZ = 1000.0
TX_OFFSET_MAX_MHZ = 1000.0


def validate_finite_array(values: np.ndarray) -> bool:
    return bool(np.isfinite(values).all())


def validate_audio_samples(
    audio: np.ndarray,
    max_abs: float = AUDIO_MAX_ABS,
) -> tuple[bool, str]:
    if audio.size == 0:
        return True, ""
    if not validate_finite_array(audio):
        return False, "non-finite audio samples"
    max_val = float(np.max(np.abs(audio)))
    if max_val > max_abs:
        return False, f"audio max abs {max_val:.3f} exceeds {max_abs:.3f}"
    return True, ""


def validate_discriminator_samples(
    audio: np.ndarray,
    max_abs: float = DISC_AUDIO_MAX_ABS,
) -> tuple[bool, str]:
    if audio.size == 0:
        return True, ""
    if not validate_finite_array(audio):
        return False, "non-finite discriminator samples"
    max_val = float(np.max(np.abs(audio)))
    if max_val > max_abs:
        return False, f"discriminator max abs {max_val:.3f} exceeds {max_abs:.3f}"
    return True, ""


def validate_frequency_hz(
    freq_hz: float,
    min_hz: float = FREQ_MIN_HZ,
    max_hz: float = FREQ_MAX_HZ,
) -> tuple[bool, str]:
    if not math.isfinite(freq_hz):
        return False, "frequency is not finite"
    if freq_hz < min_hz or freq_hz > max_hz:
        return (
            False,
            f"frequency {freq_hz:.1f} out of range {min_hz:.1f}-{max_hz:.1f}",
        )
    return True, ""


def validate_int_range(
    value: Any,
    min_value: int,
    max_value: int,
    label: str,
) -> tuple[bool, str]:
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        return False, f"{label} is not an int"
    if int_value < min_value or int_value > max_value:
        return (
            False,
            f"{label} out of range {min_value}-{max_value} (got {int_value})",
        )
    return True, ""


def validate_float_range(
    value: Any,
    min_value: float,
    max_value: float,
    label: str,
) -> tuple[bool, str]:
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        return False, f"{label} is not a float"
    if not math.isfinite(float_value):
        return False, f"{label} is not finite"
    if float_value < min_value or float_value > max_value:
        return (
            False,
            f"{label} out of range {min_value}-{max_value} (got {float_value})",
        )
    return True, ""
