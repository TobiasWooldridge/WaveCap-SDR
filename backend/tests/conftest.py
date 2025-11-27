"""Shared pytest fixtures for WaveCap-SDR tests."""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_rate() -> int:
    """Default sample rate for tests."""
    return 48_000


@pytest.fixture
def iq_sample_rate() -> int:
    """Default IQ sample rate for tests."""
    return 2_000_000


@pytest.fixture
def generate_fm_signal():
    """Factory to generate synthetic FM-modulated IQ signals."""
    def _generate(
        sample_rate: int,
        duration_s: float,
        audio_freq: float = 1000.0,
        deviation: float = 75_000.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate FM-modulated IQ and expected audio.

        Args:
            sample_rate: Sample rate in Hz
            duration_s: Duration in seconds
            audio_freq: Modulating audio frequency in Hz
            deviation: FM deviation in Hz

        Returns:
            Tuple of (IQ samples, expected audio waveform)
        """
        n_samples = int(sample_rate * duration_s)
        t = np.arange(n_samples, dtype=np.float64) / sample_rate

        # Generate modulating audio (sine wave)
        audio = np.sin(2 * np.pi * audio_freq * t).astype(np.float32)

        # Generate FM modulated signal
        # Instantaneous phase = 2*pi*fc*t + 2*pi*kf*integral(m(t))
        # For sine modulation: integral = -(1/audio_freq)*cos(2*pi*audio_freq*t)
        # We're centering at 0 Hz, so fc=0
        modulation_index = deviation / audio_freq
        phase = modulation_index * np.sin(2 * np.pi * audio_freq * t)
        iq = np.exp(1j * phase).astype(np.complex64)

        return iq, audio

    return _generate


@pytest.fixture
def generate_tone():
    """Factory to generate single tone audio signals."""
    def _generate(
        sample_rate: int,
        duration_s: float,
        frequency: float,
        amplitude: float = 0.5,
    ) -> np.ndarray:
        n_samples = int(sample_rate * duration_s)
        t = np.arange(n_samples, dtype=np.float64) / sample_rate
        return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

    return _generate


@pytest.fixture
def generate_noise():
    """Factory to generate noise signals."""
    def _generate(
        n_samples: int,
        amplitude: float = 0.1,
        seed: int = 42,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return (amplitude * rng.standard_normal(n_samples)).astype(np.float32)

    return _generate


@pytest.fixture
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


@pytest.fixture
def backend_root() -> Path:
    """Get the backend root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def config_dir(backend_root: Path) -> Path:
    """Get the config directory."""
    return backend_root / "config"
