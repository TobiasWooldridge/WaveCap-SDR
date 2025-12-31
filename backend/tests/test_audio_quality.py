"""Audio quality and validation tests.

Tests audio processing, normalization, format conversion, and quality metrics.
Adapted from SDRTrunk's AudioUtils patterns.

Reference: https://github.com/DSheirer/sdrtrunk
"""

import numpy as np
import pytest

from wavecapsdr.trunking.voice_channel import pack_pcm16


# ============================================================================
# Audio Sample Validation Tests
# ============================================================================

class TestAudioSampleValidation:
    """Test audio sample validation and range checking."""

    def test_valid_audio_range(self):
        """Audio samples should be in [-1.0, 1.0] range."""
        audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)

        assert np.all(audio >= -1.0)
        assert np.all(audio <= 1.0)

    def test_detect_out_of_range_samples(self):
        """Detect samples outside valid range."""
        audio = np.array([0.0, 1.5, -0.5, 2.0, -2.0], dtype=np.float32)

        out_of_range = np.abs(audio) > 1.0
        assert np.sum(out_of_range) == 3

    def test_detect_nan_samples(self):
        """Detect NaN values in audio."""
        audio = np.array([0.0, np.nan, 0.5, np.nan], dtype=np.float32)

        nan_count = np.sum(np.isnan(audio))
        assert nan_count == 2

    def test_detect_inf_samples(self):
        """Detect infinite values in audio."""
        audio = np.array([0.0, np.inf, -np.inf, 0.5], dtype=np.float32)

        inf_count = np.sum(np.isinf(audio))
        assert inf_count == 2

    def test_sanitize_audio_samples(self):
        """Sanitize audio by clipping and replacing invalid values."""
        audio = np.array([0.0, 1.5, np.nan, -2.0, np.inf], dtype=np.float32)

        # Replace NaN/Inf with 0
        sanitized = np.where(np.isfinite(audio), audio, 0.0)
        # Clip to valid range
        sanitized = np.clip(sanitized, -1.0, 1.0)

        assert np.all(np.isfinite(sanitized))
        assert np.all(np.abs(sanitized) <= 1.0)


# ============================================================================
# Audio Normalization Tests
# ============================================================================

class TestAudioNormalization:
    """Test audio normalization (adapted from SDRTrunk AudioUtils)."""

    def test_normalize_quiet_audio(self):
        """Normalize quiet audio to target level."""
        # Quiet audio at -20dB (0.1 amplitude)
        audio = np.array([0.0, 0.1, -0.1, 0.05], dtype=np.float32)
        target_peak = 0.95  # -0.44 dBFS like SDRTrunk

        max_val = np.max(np.abs(audio))
        if max_val > 0:
            gain = target_peak / max_val
            normalized = audio * gain
        else:
            normalized = audio

        assert np.max(np.abs(normalized)) == pytest.approx(target_peak, rel=0.01)

    def test_normalize_loud_audio(self):
        """Loud audio should be attenuated."""
        # Audio already at 0.9 peak
        audio = np.array([0.0, 0.9, -0.9, 0.5], dtype=np.float32)
        target_peak = 0.95

        max_val = np.max(np.abs(audio))
        gain = target_peak / max_val
        normalized = audio * gain

        assert np.max(np.abs(normalized)) == pytest.approx(target_peak, rel=0.01)

    def test_normalize_silent_audio(self):
        """Silent audio should remain silent (avoid division by zero)."""
        audio = np.zeros(100, dtype=np.float32)
        target_peak = 0.95

        max_val = np.max(np.abs(audio))
        if max_val > 0.001:  # Threshold to avoid amplifying noise
            gain = target_peak / max_val
            normalized = audio * gain
        else:
            normalized = audio

        assert np.all(normalized == 0)

    def test_limit_maximum_gain(self):
        """Limit maximum gain to prevent over-amplification."""
        # Very quiet audio
        audio = np.array([0.001, -0.001], dtype=np.float32)
        target_peak = 0.95
        max_gain = 10.0  # Limit gain to 20dB

        max_val = np.max(np.abs(audio))
        gain = min(target_peak / max_val, max_gain)
        normalized = audio * gain

        # Should be limited, not fully normalized
        assert np.max(np.abs(normalized)) < target_peak
        assert gain == max_gain


# ============================================================================
# Silence Detection Tests
# ============================================================================

class TestSilenceDetection:
    """Test silence and noise floor detection."""

    def test_detect_digital_silence(self):
        """Detect complete digital silence (all zeros)."""
        audio = np.zeros(1000, dtype=np.float32)

        rms = np.sqrt(np.mean(audio ** 2))
        is_silent = rms < 0.001

        assert is_silent

    def test_detect_near_silence(self):
        """Detect near-silence (very low level noise)."""
        rng = np.random.default_rng(42)
        # -60dB noise floor
        audio = (rng.random(1000).astype(np.float32) - 0.5) * 0.002

        rms = np.sqrt(np.mean(audio ** 2))
        is_near_silent = rms < 0.01  # -40dB threshold

        assert is_near_silent

    def test_detect_audio_present(self):
        """Detect when actual audio is present."""
        t = np.linspace(0, 0.1, 4800, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 1000 * t)

        rms = np.sqrt(np.mean(audio ** 2))
        has_audio = rms > 0.01

        assert has_audio

    def test_silence_duration_calculation(self):
        """Calculate duration of silence in samples."""
        sample_rate = 48000
        # 100ms of audio, then 200ms silence, then 100ms audio
        audio = np.concatenate([
            np.sin(np.linspace(0, 10, 4800)),  # 100ms tone
            np.zeros(9600),                      # 200ms silence
            np.sin(np.linspace(0, 10, 4800)),  # 100ms tone
        ]).astype(np.float32)

        # Find silence regions (RMS below threshold in 10ms windows)
        window_size = int(sample_rate * 0.01)  # 10ms
        threshold = 0.01

        silence_samples = 0
        for i in range(0, len(audio) - window_size, window_size):
            window = audio[i:i + window_size]
            rms = np.sqrt(np.mean(window ** 2))
            if rms < threshold:
                silence_samples += window_size

        silence_duration = silence_samples / sample_rate
        assert silence_duration == pytest.approx(0.2, abs=0.02)  # ~200ms


# ============================================================================
# Audio Format Conversion Tests
# ============================================================================

class TestAudioFormatConversion:
    """Test audio format conversion (PCM16, F32, etc.)."""

    def test_f32_to_pcm16_silence(self):
        """Convert silent F32 to PCM16."""
        audio = np.zeros(100, dtype=np.float32)
        pcm = pack_pcm16(audio)

        samples = np.frombuffer(pcm, dtype=np.int16)
        assert np.all(samples == 0)

    def test_f32_to_pcm16_full_scale(self):
        """Convert full-scale F32 to PCM16."""
        audio = np.array([1.0, -1.0], dtype=np.float32)
        pcm = pack_pcm16(audio)

        samples = np.frombuffer(pcm, dtype=np.int16)
        assert samples[0] == 32767
        assert samples[1] == -32767

    def test_f32_to_pcm16_clipping(self):
        """Values > 1.0 should be clipped."""
        audio = np.array([2.0, -2.0, 1.5], dtype=np.float32)
        pcm = pack_pcm16(audio)

        samples = np.frombuffer(pcm, dtype=np.int16)
        assert samples[0] == 32767
        assert samples[1] == -32767
        assert samples[2] == 32767

    def test_f32_to_pcm16_precision(self):
        """Check conversion preserves reasonable precision."""
        # Small value that should be representable
        audio = np.array([0.5, -0.5, 0.25], dtype=np.float32)
        pcm = pack_pcm16(audio)

        samples = np.frombuffer(pcm, dtype=np.int16)
        # 0.5 * 32767 ≈ 16383
        assert abs(samples[0] - 16383) <= 1
        assert abs(samples[1] - (-16383)) <= 1
        assert abs(samples[2] - 8191) <= 1

    def test_pcm16_to_f32_roundtrip(self):
        """Test F32 -> PCM16 -> F32 roundtrip."""
        original = np.array([0.0, 0.5, -0.5, 0.9, -0.9], dtype=np.float32)

        # F32 to PCM16
        pcm = pack_pcm16(original)
        pcm_samples = np.frombuffer(pcm, dtype=np.int16)

        # PCM16 back to F32
        recovered = pcm_samples.astype(np.float32) / 32767.0

        # Should be close (quantization error)
        np.testing.assert_allclose(original, recovered, atol=1/32767)


# ============================================================================
# Audio Duration Tests
# ============================================================================

class TestAudioDuration:
    """Test audio duration calculations."""

    def test_duration_from_samples(self):
        """Calculate duration from sample count."""
        sample_rate = 48000
        samples = np.zeros(48000, dtype=np.float32)  # 1 second

        duration = len(samples) / sample_rate
        assert duration == 1.0

    def test_duration_fractional(self):
        """Calculate fractional duration."""
        sample_rate = 48000
        samples = np.zeros(24000, dtype=np.float32)  # 0.5 seconds

        duration = len(samples) / sample_rate
        assert duration == 0.5

    def test_duration_from_pcm_bytes(self):
        """Calculate duration from PCM byte count."""
        sample_rate = 48000
        bytes_per_sample = 2  # PCM16

        byte_count = 96000  # 48000 samples * 2 bytes
        sample_count = byte_count // bytes_per_sample
        duration = sample_count / sample_rate

        assert duration == 1.0


# ============================================================================
# Audio Quality Metrics Tests
# ============================================================================

class TestAudioQualityMetrics:
    """Test audio quality measurement."""

    def test_rms_level(self):
        """Calculate RMS level of audio."""
        # Sine wave at 0.5 amplitude has RMS of 0.5/sqrt(2) ≈ 0.354
        t = np.linspace(0, 1, 48000, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 1000 * t)

        rms = np.sqrt(np.mean(audio ** 2))
        expected_rms = 0.5 / np.sqrt(2)

        assert rms == pytest.approx(expected_rms, rel=0.01)

    def test_peak_level(self):
        """Calculate peak level of audio."""
        audio = np.array([0.0, 0.3, -0.8, 0.5, -0.2], dtype=np.float32)

        peak = np.max(np.abs(audio))
        assert peak == pytest.approx(0.8)

    def test_crest_factor(self):
        """Calculate crest factor (peak/RMS ratio)."""
        # Sine wave has crest factor of sqrt(2) ≈ 1.414
        t = np.linspace(0, 1, 48000, dtype=np.float32)
        audio = np.sin(2 * np.pi * 1000 * t)

        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio ** 2))
        crest_factor = peak / rms

        assert crest_factor == pytest.approx(np.sqrt(2), rel=0.01)

    def test_dc_offset_detection(self):
        """Detect DC offset in audio."""
        # Audio with DC offset
        t = np.linspace(0, 0.1, 4800, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 1000 * t) + 0.1  # 0.1 DC offset

        dc_offset = np.mean(audio)
        has_dc_offset = abs(dc_offset) > 0.01

        assert has_dc_offset
        assert dc_offset == pytest.approx(0.1, abs=0.01)

    def test_dc_offset_removal(self):
        """Remove DC offset from audio."""
        t = np.linspace(0, 0.1, 4800, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 1000 * t) + 0.1

        # Remove DC offset
        corrected = audio - np.mean(audio)

        assert abs(np.mean(corrected)) < 0.001


# ============================================================================
# Audio Buffer Management Tests
# ============================================================================

class TestAudioBufferManagement:
    """Test audio buffer operations."""

    def test_buffer_concatenation(self):
        """Concatenate multiple audio buffers."""
        buffers = [
            np.array([0.1, 0.2], dtype=np.float32),
            np.array([0.3, 0.4], dtype=np.float32),
            np.array([0.5, 0.6], dtype=np.float32),
        ]

        combined = np.concatenate(buffers)

        assert len(combined) == 6
        np.testing.assert_allclose(combined, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], rtol=1e-6)

    def test_buffer_slicing(self):
        """Slice audio buffer for streaming."""
        audio = np.arange(100, dtype=np.float32)
        chunk_size = 25

        chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]

        assert len(chunks) == 4
        assert all(len(c) == chunk_size for c in chunks)

    def test_ring_buffer_behavior(self):
        """Test ring buffer for continuous audio."""
        buffer_size = 10
        ring_buffer = np.zeros(buffer_size, dtype=np.float32)
        write_pos = 0

        # Write 15 samples (wraps around)
        for i in range(15):
            ring_buffer[write_pos % buffer_size] = float(i)
            write_pos += 1

        # Last 10 values should be 5-14
        expected = np.array([10, 11, 12, 13, 14, 5, 6, 7, 8, 9], dtype=np.float32)
        np.testing.assert_array_equal(ring_buffer, expected)


# ============================================================================
# Audio Segment Tests (SDRTrunk AudioSegment pattern)
# ============================================================================

class TestAudioSegment:
    """Test audio segment management (SDRTrunk pattern)."""

    def test_segment_sample_accumulation(self):
        """Accumulate samples in audio segment."""
        buffers = []
        sample_count = 0

        # Add multiple buffers
        for i in range(5):
            buf = np.zeros(100, dtype=np.float32)
            buffers.append(buf)
            sample_count += len(buf)

        assert sample_count == 500
        assert len(buffers) == 5

    def test_segment_duration_tracking(self):
        """Track segment duration as samples accumulate."""
        sample_rate = 8000
        sample_count = 0

        # Add 100ms worth of samples (800 samples at 8kHz)
        sample_count += 800
        duration_ms = (sample_count / sample_rate) * 1000

        assert duration_ms == 100.0

    def test_segment_consumer_counting(self):
        """Track consumer count for cleanup."""
        consumer_count = 0

        # Add consumers
        consumer_count += 1
        consumer_count += 1
        assert consumer_count == 2

        # Remove consumers
        consumer_count -= 1
        assert consumer_count == 1

        # Last consumer - trigger cleanup
        consumer_count -= 1
        should_cleanup = consumer_count <= 0
        assert should_cleanup
