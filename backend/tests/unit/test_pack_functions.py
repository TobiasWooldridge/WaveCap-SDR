"""Unit tests for IQ and audio packing functions.

These functions convert numpy arrays to wire format for streaming.
"""

import numpy as np
import pytest

from wavecapsdr.capture import pack_iq16, pack_pcm16, pack_f32, freq_shift


class TestPackIQ16:
    """Tests for pack_iq16 function."""

    def test_pack_iq16_basic(self):
        """Pack simple IQ samples to 16-bit integers."""
        # IQ samples: (0.5+0.5j), (-0.5-0.5j)
        iq = np.array([0.5 + 0.5j, -0.5 - 0.5j], dtype=np.complex64)
        data = pack_iq16(iq)

        # 2 complex samples -> 4 int16 values -> 8 bytes
        assert len(data) == 8

        # Unpack and verify
        unpacked = np.frombuffer(data, dtype=np.int16)
        assert len(unpacked) == 4
        # 0.5 * 32767 ≈ 16383
        assert abs(unpacked[0] - 16383) < 2  # I of first sample
        assert abs(unpacked[1] - 16383) < 2  # Q of first sample
        assert abs(unpacked[2] + 16383) < 2  # I of second sample
        assert abs(unpacked[3] + 16383) < 2  # Q of second sample

    def test_pack_iq16_empty(self):
        """Empty input should return empty bytes."""
        iq = np.array([], dtype=np.complex64)
        data = pack_iq16(iq)
        assert data == b""

    def test_pack_iq16_clips_overflow(self):
        """Values > 1.0 should be clipped."""
        iq = np.array([2.0 + 2.0j], dtype=np.complex64)
        data = pack_iq16(iq)
        unpacked = np.frombuffer(data, dtype=np.int16)
        # Should be clipped to ~32767
        assert abs(unpacked[0] - 32767) < 2
        assert abs(unpacked[1] - 32767) < 2

    def test_pack_iq16_clips_underflow(self):
        """Values < -1.0 should be clipped."""
        iq = np.array([-2.0 - 2.0j], dtype=np.complex64)
        data = pack_iq16(iq)
        unpacked = np.frombuffer(data, dtype=np.int16)
        # Should be clipped to ~-32767
        assert abs(unpacked[0] + 32767) < 2
        assert abs(unpacked[1] + 32767) < 2

    def test_pack_iq16_preserves_zeros(self):
        """Zero values should remain zero."""
        iq = np.array([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex64)
        data = pack_iq16(iq)
        unpacked = np.frombuffer(data, dtype=np.int16)
        np.testing.assert_array_equal(unpacked, [0, 0, 0, 0])


class TestPackPCM16:
    """Tests for pack_pcm16 function."""

    def test_pack_pcm16_basic(self):
        """Pack audio samples to 16-bit PCM."""
        audio = np.array([0.5, -0.5, 0.0, 1.0], dtype=np.float32)
        data = pack_pcm16(audio)

        # 4 samples -> 8 bytes
        assert len(data) == 8

        unpacked = np.frombuffer(data, dtype=np.int16)
        assert abs(unpacked[0] - 16383) < 2  # 0.5
        assert abs(unpacked[1] + 16383) < 2  # -0.5
        assert unpacked[2] == 0  # 0.0
        assert abs(unpacked[3] - 32767) < 2  # 1.0

    def test_pack_pcm16_empty(self):
        """Empty input should return empty bytes."""
        audio = np.array([], dtype=np.float32)
        data = pack_pcm16(audio)
        assert data == b""

    def test_pack_pcm16_clips(self):
        """Values outside [-1, 1] should be clipped."""
        audio = np.array([2.0, -2.0], dtype=np.float32)
        data = pack_pcm16(audio)
        unpacked = np.frombuffer(data, dtype=np.int16)
        assert abs(unpacked[0] - 32767) < 2
        assert abs(unpacked[1] + 32767) < 2


class TestPackF32:
    """Tests for pack_f32 function."""

    def test_pack_f32_basic(self):
        """Pack audio samples as 32-bit float."""
        audio = np.array([0.5, -0.5, 0.0], dtype=np.float32)
        data = pack_f32(audio)

        # 3 samples * 4 bytes = 12 bytes
        assert len(data) == 12

        unpacked = np.frombuffer(data, dtype=np.float32)
        np.testing.assert_array_almost_equal(unpacked, [0.5, -0.5, 0.0])

    def test_pack_f32_empty(self):
        """Empty input should return empty bytes."""
        audio = np.array([], dtype=np.float32)
        data = pack_f32(audio)
        assert data == b""

    def test_pack_f32_clips(self):
        """Values outside [-1, 1] should be clipped."""
        audio = np.array([2.0, -2.0], dtype=np.float32)
        data = pack_f32(audio)
        unpacked = np.frombuffer(data, dtype=np.float32)
        np.testing.assert_array_almost_equal(unpacked, [1.0, -1.0])


class TestFreqShift:
    """Tests for frequency shift function."""

    def test_freq_shift_zero_offset(self):
        """Zero offset should return unchanged signal."""
        iq = np.random.randn(1000).astype(np.complex64)
        shifted = freq_shift(iq, 0.0, 48000)
        np.testing.assert_array_equal(iq, shifted)

    def test_freq_shift_empty(self):
        """Empty input should return empty output."""
        iq = np.array([], dtype=np.complex64)
        shifted = freq_shift(iq, 1000, 48000)
        assert shifted.size == 0

    def test_freq_shift_preserves_magnitude(self):
        """Frequency shift should preserve signal magnitude."""
        iq = (0.5 + 0.3j) * np.ones(1000, dtype=np.complex64)
        shifted = freq_shift(iq, 5000, 48000)
        # Magnitude should be preserved
        np.testing.assert_allclose(np.abs(shifted), np.abs(iq), rtol=1e-5)

    def test_freq_shift_changes_phase(self):
        """Frequency shift should create phase rotation."""
        sample_rate = 48000
        offset_hz = 1000
        n = 1000
        iq = np.ones(n, dtype=np.complex64)
        shifted = freq_shift(iq, offset_hz, sample_rate)

        # After frequency shift, phase should change linearly
        # with a rate of offset_hz/sample_rate radians per sample
        phases = np.angle(shifted)
        phase_diff = np.diff(np.unwrap(phases))
        expected_phase_diff = -2 * np.pi * offset_hz / sample_rate

        np.testing.assert_allclose(
            phase_diff, expected_phase_diff * np.ones(n - 1), rtol=0.01
        )

    def test_freq_shift_output_type(self):
        """Output should be complex64."""
        iq = np.ones(100, dtype=np.complex64)
        shifted = freq_shift(iq, 1000, 48000)
        assert shifted.dtype == np.complex64
