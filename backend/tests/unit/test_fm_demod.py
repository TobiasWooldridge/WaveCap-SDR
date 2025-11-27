"""Unit tests for FM demodulation pipeline.

Tests the complete FM demodulation with synthetic signals.
"""

import numpy as np
import pytest

from wavecapsdr.dsp.fm import (
    wbfm_demod,
    nbfm_demod,
    quadrature_demod,
    deemphasis_filter,
    lpf_audio,
)


class TestWBFMDemod:
    """Tests for wideband FM demodulation."""

    def test_wbfm_demod_output_type(self):
        """Output should be float32."""
        iq = np.random.randn(10000).astype(np.complex64)
        audio = wbfm_demod(iq, sample_rate=200_000)
        assert audio.dtype == np.float32

    def test_wbfm_demod_resamples_to_audio_rate(self):
        """Output should be resampled to audio_rate."""
        sample_rate = 200_000
        audio_rate = 48_000
        duration = 0.1
        n_samples = int(sample_rate * duration)
        iq = np.random.randn(n_samples).astype(np.complex64) * 0.1

        audio = wbfm_demod(iq, sample_rate, audio_rate=audio_rate)

        # Expected output samples (approximately)
        expected_samples = int(n_samples * audio_rate / sample_rate)
        # Allow some tolerance due to filter edge effects
        assert abs(len(audio) - expected_samples) < expected_samples * 0.1

    def test_wbfm_demod_output_bounded(self):
        """Output should be bounded by soft clipping."""
        iq = np.random.randn(10000).astype(np.complex64) * 10  # Large signal
        audio = wbfm_demod(iq, sample_rate=200_000)
        # Soft clip should keep values bounded
        assert np.max(np.abs(audio)) <= 1.0

    def test_wbfm_demod_with_modulated_signal(self, generate_fm_signal):
        """Demodulating an FM signal should recover something correlated with audio."""
        sample_rate = 200_000
        duration = 0.1
        audio_freq = 1000

        iq, expected = generate_fm_signal(
            sample_rate, duration, audio_freq=audio_freq, deviation=75_000
        )

        audio = wbfm_demod(
            iq,
            sample_rate,
            audio_rate=48_000,
            enable_deemphasis=False,
            enable_mpx_filter=False,
        )

        # Audio should have content (not silence)
        assert np.std(audio) > 0.01

    def test_wbfm_demod_empty_input(self):
        """Empty input should return empty output."""
        iq = np.array([], dtype=np.complex64)
        audio = wbfm_demod(iq, sample_rate=200_000)
        assert audio.size == 0

    def test_wbfm_demod_with_deemphasis(self):
        """Deemphasis should reduce high-frequency content."""
        sample_rate = 200_000
        n = 20000
        # Create signal with high frequency content
        t = np.arange(n, dtype=np.float64) / sample_rate
        # Simulate FM with high-freq audio
        phase = 0.5 * np.sin(2 * np.pi * 10000 * t)
        iq = np.exp(1j * phase).astype(np.complex64)

        audio_no_deemph = wbfm_demod(iq, sample_rate, enable_deemphasis=False)
        audio_with_deemph = wbfm_demod(iq, sample_rate, enable_deemphasis=True)

        # With deemphasis, high-freq content should be reduced
        # This is a relative comparison
        energy_no = np.sum(np.abs(audio_no_deemph) ** 2)
        energy_with = np.sum(np.abs(audio_with_deemph) ** 2)
        # Deemphasis should reduce energy from high-freq signal
        assert energy_with <= energy_no

    def test_wbfm_demod_with_mpx_filter(self):
        """MPX filter should be applicable."""
        sample_rate = 200_000
        iq = np.random.randn(10000).astype(np.complex64) * 0.1

        # Should not crash with MPX filter enabled
        audio = wbfm_demod(iq, sample_rate, enable_mpx_filter=True)
        assert audio.size > 0


class TestNBFMDemod:
    """Tests for narrowband FM demodulation."""

    def test_nbfm_demod_output_type(self):
        """Output should be float32."""
        iq = np.random.randn(10000).astype(np.complex64)
        audio = nbfm_demod(iq, sample_rate=48_000)
        assert audio.dtype == np.float32

    def test_nbfm_demod_output_bounded(self):
        """Output should be bounded by soft clipping."""
        iq = np.random.randn(10000).astype(np.complex64) * 10
        audio = nbfm_demod(iq, sample_rate=48_000)
        assert np.max(np.abs(audio)) <= 1.0

    def test_nbfm_demod_with_voice_filters(self):
        """Voice filters should work together."""
        sample_rate = 48_000
        iq = np.random.randn(10000).astype(np.complex64) * 0.1

        audio = nbfm_demod(
            iq,
            sample_rate,
            enable_highpass=True,
            highpass_hz=300,
            enable_lowpass=True,
            lowpass_hz=3000,
        )

        assert audio.size > 0
        assert audio.dtype == np.float32

    def test_nbfm_demod_empty_input(self):
        """Empty input should return empty output."""
        iq = np.array([], dtype=np.complex64)
        audio = nbfm_demod(iq, sample_rate=48_000)
        assert audio.size == 0


class TestDeemphasisFilter:
    """Tests for deemphasis filter."""

    def test_deemphasis_reduces_highs(self):
        """Deemphasis should attenuate high frequencies."""
        sample_rate = 48000
        n = 4800

        # Low frequency tone
        t = np.arange(n, dtype=np.float32) / sample_rate
        low_freq = np.sin(2 * np.pi * 100 * t).astype(np.float32)
        high_freq = np.sin(2 * np.pi * 10000 * t).astype(np.float32)

        low_out = deemphasis_filter(low_freq, sample_rate, tau=75e-6)
        high_out = deemphasis_filter(high_freq, sample_rate, tau=75e-6)

        # High frequency should be attenuated more than low frequency
        low_ratio = np.std(low_out) / (np.std(low_freq) + 1e-10)
        high_ratio = np.std(high_out) / (np.std(high_freq) + 1e-10)

        assert high_ratio < low_ratio

    def test_deemphasis_output_type(self):
        """Output should be float32."""
        x = np.random.randn(1000).astype(np.float32)
        y = deemphasis_filter(x, 48000)
        assert y.dtype == np.float32


class TestLPFAudio:
    """Tests for MPX lowpass filter."""

    def test_lpf_audio_removes_high_frequencies(self):
        """LPF should attenuate frequencies above cutoff."""
        sample_rate = 200_000
        cutoff = 15_000
        n = 20000

        t = np.arange(n, dtype=np.float32) / sample_rate

        # Signal below cutoff
        low_tone = np.sin(2 * np.pi * 5000 * t).astype(np.float32)
        # Signal above cutoff
        high_tone = np.sin(2 * np.pi * 30000 * t).astype(np.float32)

        low_out = lpf_audio(low_tone, sample_rate, cutoff)
        high_out = lpf_audio(high_tone, sample_rate, cutoff)

        # Low tone should pass (mostly unchanged)
        low_energy_ratio = np.sum(low_out**2) / np.sum(low_tone**2)
        # High tone should be attenuated
        high_energy_ratio = np.sum(high_out**2) / np.sum(high_tone**2)

        assert low_energy_ratio > 0.5  # Most of low tone passes
        assert high_energy_ratio < 0.2  # Most of high tone blocked

    def test_lpf_audio_empty_input(self):
        """Empty input should return empty output."""
        x = np.array([], dtype=np.float32)
        y = lpf_audio(x, 48000, 15000)
        assert y.size == 0

    def test_lpf_audio_output_type(self):
        """Output should be float32."""
        x = np.random.randn(1000).astype(np.float32)
        y = lpf_audio(x, 48000, 15000)
        assert y.dtype == np.float32
