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


# ============================================================================
# Voice/Trunking Fixtures
# ============================================================================

@pytest.fixture
def voice_channel_config():
    """Sample voice channel configuration."""
    from wavecapsdr.trunking.voice_channel import VoiceChannelConfig

    return VoiceChannelConfig(
        id="test_vc0",
        system_id="test_system",
        call_id="call_001",
        recorder_id="vr0",
        audio_rate=8000,
        output_rate=48000,
    )


@pytest.fixture
def sample_radio_location():
    """Sample GPS location for testing."""
    from wavecapsdr.trunking.voice_channel import RadioLocation

    return RadioLocation(
        unit_id=12345678,
        latitude=47.6062,
        longitude=-122.3321,
        altitude_m=100.0,
        speed_kmh=45.0,
        heading_deg=270.0,
        source="elc",
    )


@pytest.fixture
def sample_audio_f32():
    """Sample audio data as float32 array (100ms at 48kHz)."""
    t = np.linspace(0, 0.1, 4800, dtype=np.float32)
    return np.sin(2 * np.pi * 1000 * t).astype(np.float32)


@pytest.fixture
def sample_lrrp_location_packet():
    """Sample LRRP immediate location response packet.

    Contains:
    - Opcode 0x02 (IMMEDIATE_LOC_RESPONSE)
    - Unit ID = 1
    - LOC_2D element with lat=45.0, lon=90.0
    """
    return bytes([
        0x02,  # Version 0, opcode 0x02
        0x00, 0x00, 0x01,  # Unit ID = 1
        0x22, 0x06,  # LOC_2D, length 6
        0x40, 0x00, 0x00,  # lat = 45.0
        0x40, 0x00, 0x00,  # lon = 90.0
    ])


@pytest.fixture
def sample_elc_gps_data():
    """Sample Extended Link Control GPS bytes (LCF 0x09).

    6 bytes encoding lat=45.0, lon=90.0
    """
    return bytes([0x40, 0x00, 0x00, 0x40, 0x00, 0x00])


@pytest.fixture
def trunking_system_config():
    """Sample trunking system configuration."""
    from wavecapsdr.trunking.config import TrunkingSystemConfig, TrunkingProtocol

    return TrunkingSystemConfig(
        id="test-system",
        name="Test System",
        protocol=TrunkingProtocol.P25_PHASE1,
        control_channels=[851_000_000, 851_100_000],
        center_hz=855_000_000,
        sample_rate=4_000_000,
        max_voice_recorders=4,
    )
