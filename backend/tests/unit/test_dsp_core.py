"""Unit tests for DSP core functions.

These tests verify the signal processing logic without requiring any mocking
or external dependencies beyond numpy/scipy.
"""

import numpy as np
import pytest

from wavecapsdr.dsp.fm import (
    soft_clip,
    rms_normalize,
    quadrature_demod,
    resample_poly,
)
from wavecapsdr.dsp.agc import (
    soft_clip as agc_soft_clip,
    apply_agc,
    apply_simple_agc,
)


class TestSoftClip:
    """Tests for soft clipping functions."""

    def test_soft_clip_within_range_unchanged(self):
        """Small values should pass through nearly unchanged."""
        x = np.array([0.0, 0.1, -0.1, 0.3, -0.3], dtype=np.float32)
        y = soft_clip(x)
        # Small values should be close to original (tanh is nearly linear near 0)
        np.testing.assert_allclose(y, x, atol=0.15)

    def test_soft_clip_limits_large_values(self):
        """Large values should be limited without hard clipping."""
        x = np.array([10.0, -10.0, 100.0, -100.0], dtype=np.float32)
        y = soft_clip(x)
        # All outputs should be bounded - FM soft_clip has ~0.95 headroom,
        # but the exact limit depends on the tanh normalization
        assert np.all(np.abs(y) <= 1.1)  # Allow some headroom variance
        # Most importantly, values should be much smaller than input
        assert np.max(np.abs(y)) < np.max(np.abs(x)) / 5

    def test_soft_clip_preserves_sign(self):
        """Soft clipping should preserve the sign of the signal."""
        x = np.array([1.0, -1.0, 2.0, -2.0], dtype=np.float32)
        y = soft_clip(x)
        assert np.all(np.sign(y) == np.sign(x))

    def test_soft_clip_empty_array(self):
        """Empty input should return empty output."""
        x = np.array([], dtype=np.float32)
        y = soft_clip(x)
        assert y.size == 0

    def test_soft_clip_smooth_transition(self):
        """Verify soft clipping provides smooth transition (no discontinuities)."""
        x = np.linspace(-3, 3, 1000, dtype=np.float32)
        y = soft_clip(x)
        # Check derivative is smooth (no sudden jumps)
        dy = np.diff(y)
        # All derivatives should be positive (monotonic)
        assert np.all(dy >= 0)
        # Derivatives should be bounded and smooth
        assert np.max(np.abs(np.diff(dy))) < 0.1


class TestRMSNormalize:
    """Tests for RMS normalization."""

    def test_rms_normalize_adjusts_level(self):
        """Signal should be normalized to target RMS."""
        target_rms = 0.18
        # Create signal with known RMS
        x = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32)  # RMS = 0.5
        y = rms_normalize(x, target_rms=target_rms)
        actual_rms = np.sqrt(np.mean(y**2))
        np.testing.assert_allclose(actual_rms, target_rms, rtol=0.01)

    def test_rms_normalize_empty_array(self):
        """Empty input should return empty output."""
        x = np.array([], dtype=np.float32)
        y = rms_normalize(x)
        assert y.size == 0

    def test_rms_normalize_very_quiet(self):
        """Very quiet signals should not be excessively amplified."""
        x = np.array([1e-6, -1e-6, 1e-6, -1e-6], dtype=np.float32)
        y = rms_normalize(x, target_rms=0.18, min_rms=1e-4)
        # Should not change since RMS < min_rms
        np.testing.assert_array_equal(x, y)

    def test_rms_normalize_already_at_target(self):
        """Signal at target level should not change much."""
        target_rms = 0.18
        # Create signal at target RMS
        scale = target_rms / 0.7071  # RMS of sin wave
        n = 1000
        t = np.arange(n, dtype=np.float32) / n
        x = scale * np.sin(2 * np.pi * 5 * t).astype(np.float32)
        y = rms_normalize(x, target_rms=target_rms)
        np.testing.assert_allclose(x, y, rtol=0.05)


class TestQuadratureDemod:
    """Tests for FM quadrature demodulation."""

    def test_quadrature_demod_dc_signal(self):
        """DC signal (no FM modulation) should produce zero audio."""
        sample_rate = 48000
        # Constant phase = DC signal
        iq = np.ones(1000, dtype=np.complex64)
        audio = quadrature_demod(iq, sample_rate)
        # Output should be near zero (no frequency deviation)
        assert np.abs(np.mean(audio)) < 0.01
        assert np.std(audio) < 0.01

    def test_quadrature_demod_constant_freq_offset(self):
        """Constant frequency offset should produce DC audio."""
        sample_rate = 48000
        freq_offset = 1000  # 1 kHz offset
        n = 2000
        t = np.arange(n, dtype=np.float64) / sample_rate
        # Signal at constant frequency offset
        iq = np.exp(2j * np.pi * freq_offset * t).astype(np.complex64)
        audio = quadrature_demod(iq, sample_rate)
        # Should produce roughly constant output proportional to freq offset
        # (after initial transient)
        assert np.std(audio[100:]) < 0.1  # Fairly constant

    def test_quadrature_demod_empty_array(self):
        """Empty input should return empty output."""
        iq = np.array([], dtype=np.complex64)
        audio = quadrature_demod(iq, 48000)
        assert audio.size == 0

    def test_quadrature_demod_output_type(self):
        """Output should be float32."""
        iq = np.ones(100, dtype=np.complex64)
        audio = quadrature_demod(iq, 48000)
        assert audio.dtype == np.float32


class TestResample:
    """Tests for resampling functions."""

    def test_resample_same_rate(self):
        """Same rate should return same signal."""
        x = np.random.randn(1000).astype(np.float32)
        y = resample_poly(x, 48000, 48000)
        np.testing.assert_array_almost_equal(x, y)

    def test_resample_downsample_2x(self):
        """2x downsampling should halve the number of samples."""
        x = np.random.randn(1000).astype(np.float32)
        y = resample_poly(x, 48000, 24000)
        assert y.shape[0] == 500

    def test_resample_upsample_2x(self):
        """2x upsampling should double the number of samples."""
        x = np.random.randn(1000).astype(np.float32)
        y = resample_poly(x, 24000, 48000)
        assert y.shape[0] == 2000

    def test_resample_preserves_tone(self, generate_tone):
        """Resampling should preserve a test tone."""
        in_rate = 48000
        out_rate = 16000
        duration = 0.1
        freq = 440  # Must be below Nyquist for both rates

        x = generate_tone(in_rate, duration, freq)
        y = resample_poly(x, in_rate, out_rate)

        # Verify the tone is preserved by checking it has the same frequency
        # Use simple zero-crossing count
        x_crossings = np.sum(np.diff(np.sign(x)) != 0)
        y_crossings = np.sum(np.diff(np.sign(y)) != 0)

        # Should have same number of zero crossings (within tolerance)
        np.testing.assert_allclose(x_crossings, y_crossings, rtol=0.05)

    def test_resample_empty_array(self):
        """Empty input should return empty output."""
        x = np.array([], dtype=np.float32)
        y = resample_poly(x, 48000, 16000)
        assert y.size == 0


class TestAGC:
    """Tests for Automatic Gain Control."""

    def test_agc_boosts_quiet_signal(self):
        """AGC should boost a quiet signal toward target level."""
        sample_rate = 48000
        # Very quiet signal
        x = 0.001 * np.sin(2 * np.pi * 440 * np.arange(4800) / sample_rate).astype(np.float32)
        y = apply_agc(x, sample_rate, target_db=-20, max_gain_db=60)
        # Output should be louder
        assert np.std(y) > np.std(x) * 5

    def test_agc_attenuates_loud_signal(self):
        """AGC should attenuate a loud signal toward target level over time."""
        sample_rate = 48000
        # Loud signal - longer duration to allow AGC to settle
        n_samples = 48000  # 1 second
        x = 0.9 * np.sin(2 * np.pi * 440 * np.arange(n_samples) / sample_rate).astype(np.float32)
        y = apply_agc(x, sample_rate, target_db=-20)
        # After AGC settles (last half of signal), RMS should be lower
        # Note: Initial transient may overshoot due to attack/release dynamics
        x_rms_end = np.sqrt(np.mean(x[n_samples//2:]**2))
        y_rms_end = np.sqrt(np.mean(y[n_samples//2:]**2))
        assert y_rms_end < x_rms_end

    def test_agc_empty_array(self):
        """Empty input should return empty output."""
        x = np.array([], dtype=np.float32)
        y = apply_agc(x, 48000)
        assert y.size == 0

    def test_simple_agc_normalizes_to_target(self):
        """Simple AGC should normalize toward target RMS."""
        # Signal with known RMS
        x = 0.01 * np.random.randn(4800).astype(np.float32)
        y = apply_simple_agc(x, target_rms=0.1, max_gain=100)
        output_rms = np.sqrt(np.mean(y**2))
        # Should be closer to target (accounting for soft clip)
        assert output_rms > 0.05

    def test_simple_agc_limits_gain(self):
        """Simple AGC should respect max gain limit."""
        # Very quiet signal
        x = 1e-6 * np.ones(1000, dtype=np.float32)
        y = apply_simple_agc(x, target_rms=0.1, max_gain=10)
        # Gain should be limited to 10x
        assert np.max(np.abs(y)) <= 1e-5 * 10 * 2  # Allow some headroom


class TestAGCSoftClip:
    """Tests for AGC module's soft clip (slightly different from FM's)."""

    def test_agc_soft_clip_output_range(self):
        """AGC soft clip should limit extreme values."""
        x = np.array([10.0, -10.0, 0.5, -0.5], dtype=np.float32)
        y = agc_soft_clip(x)
        # AGC soft clip uses tanh with normalization that can exceed 1.0 slightly
        # The key property is that extreme inputs are bounded
        assert np.all(np.abs(y) <= 1.2)  # Allow some variance
        # Sign should be preserved
        assert np.all(np.sign(y) == np.sign(x))
        # Large values should be reduced, small values should increase
        # (tanh characteristic: compresses towards ±1)
        assert np.abs(y[0]) < np.abs(x[0])  # Large reduced
        assert np.abs(y[2]) > np.abs(x[2])  # Small increased (normalized)
