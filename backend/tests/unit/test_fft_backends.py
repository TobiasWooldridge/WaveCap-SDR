"""Unit tests for FFT backend implementations."""

import numpy as np
import pytest

from wavecapsdr.dsp.fft import (
    FFTBackend,
    FFTResult,
    ScipyFFTBackend,
    get_backend,
    available_backends,
)
from wavecapsdr.dsp.fft.base import FFTBackend as FFTBackendBase


class TestFFTResult:
    """Tests for FFTResult dataclass."""

    def test_fft_result_creation(self):
        """Test creating an FFTResult."""
        power_db = np.zeros(1024, dtype=np.float32)
        freqs = np.linspace(-24000, 24000, 1024, dtype=np.float32)

        result = FFTResult(power_db=power_db, freqs=freqs, bin_hz=46.875)

        assert result.power_db.shape == (1024,)
        assert result.freqs.shape == (1024,)
        assert result.bin_hz == 46.875


class TestScipyFFTBackend:
    """Tests for scipy FFT backend."""

    def test_initialization(self):
        """Test backend initialization."""
        backend = ScipyFFTBackend(fft_size=2048)

        assert backend.fft_size == 2048
        assert backend.name in ("scipy", "numpy")

    def test_window_creation(self):
        """Test Hanning window is created correctly."""
        backend = ScipyFFTBackend(fft_size=1024)

        window = backend.window
        assert window.shape == (1024,)
        assert window.dtype == np.float32
        # Hanning window starts and ends at ~0
        assert window[0] < 0.01
        assert window[-1] < 0.01
        # Hanning window peaks at center
        assert window[512] > 0.99

    def test_execute_with_valid_input(self):
        """Test FFT execution with valid IQ samples."""
        backend = ScipyFFTBackend(fft_size=2048)

        # Generate test signal: single tone at 1 kHz
        sample_rate = 48000
        t = np.arange(2048) / sample_rate
        freq = 1000  # 1 kHz tone
        iq = np.exp(2j * np.pi * freq * t).astype(np.complex64)

        result = backend.execute(iq, sample_rate)

        assert isinstance(result, FFTResult)
        assert result.power_db.shape == (2048,)
        assert result.freqs.shape == (2048,)
        assert result.bin_hz == pytest.approx(sample_rate / 2048, rel=0.01)

        # Verify peak is near 1 kHz
        peak_idx = np.argmax(result.power_db)
        peak_freq = result.freqs[peak_idx]
        assert abs(peak_freq - freq) < 50  # Within 50 Hz of expected

    def test_execute_with_insufficient_samples(self):
        """Test FFT with fewer samples than fft_size."""
        backend = ScipyFFTBackend(fft_size=2048)

        # Only 100 samples
        iq = np.zeros(100, dtype=np.complex64)

        result = backend.execute(iq, 48000)

        # Should return zeros
        assert result.power_db.shape == (2048,)
        assert np.all(result.power_db == 0)

    def test_execute_with_noise(self):
        """Test FFT with noise input."""
        backend = ScipyFFTBackend(fft_size=4096)

        # Random noise
        np.random.seed(42)
        iq = (np.random.randn(4096) + 1j * np.random.randn(4096)).astype(np.complex64)

        result = backend.execute(iq, 48000)

        assert result.power_db.shape == (4096,)
        # Noise should have relatively flat spectrum
        std = np.std(result.power_db)
        assert std < 10  # Not too much variation

    def test_different_fft_sizes(self):
        """Test different FFT sizes."""
        sample_rate = 48000
        iq = np.random.randn(8192).astype(np.complex64) + 1j * np.random.randn(8192).astype(np.complex64)

        for fft_size in [512, 1024, 2048, 4096]:
            backend = ScipyFFTBackend(fft_size=fft_size)
            result = backend.execute(iq, sample_rate)

            assert result.power_db.shape == (fft_size,)
            assert result.freqs.shape == (fft_size,)
            assert result.bin_hz == pytest.approx(sample_rate / fft_size, rel=0.01)


class TestBackendRegistry:
    """Tests for FFT backend registry."""

    def test_get_backend_auto(self):
        """Test auto backend selection."""
        backend = get_backend("auto")

        assert isinstance(backend, FFTBackendBase)
        # On most systems, scipy should be available
        assert backend.name in ("scipy", "numpy", "fftw", "mlx", "cuda")

    def test_get_backend_scipy(self):
        """Test explicit scipy backend."""
        backend = get_backend("scipy")

        assert isinstance(backend, ScipyFFTBackend)

    def test_get_backend_with_fft_size(self):
        """Test backend with custom FFT size."""
        backend = get_backend("scipy", fft_size=4096)

        assert backend.fft_size == 4096

    def test_available_backends(self):
        """Test available backends list."""
        backends = available_backends()

        assert isinstance(backends, list)
        assert "scipy" in backends  # scipy should always be available

    def test_get_backend_fallback(self):
        """Test fallback when requested backend unavailable."""
        # Request non-existent backend
        backend = get_backend("nonexistent_backend")

        # Should fall back to scipy
        assert backend.name in ("scipy", "numpy")


class TestBackendEquivalence:
    """Test that all backends produce equivalent results."""

    def test_all_backends_produce_similar_results(self):
        """Verify all available backends produce similar FFT results."""
        sample_rate = 48000
        fft_size = 2048

        # Generate test signal
        t = np.arange(fft_size) / sample_rate
        freq = 5000  # 5 kHz tone
        iq = np.exp(2j * np.pi * freq * t).astype(np.complex64)

        backends = available_backends()
        results = {}

        for name in backends:
            try:
                backend = get_backend(name, fft_size=fft_size)
                results[name] = backend.execute(iq, sample_rate)
            except ImportError:
                continue

        # Compare all results to scipy baseline
        if "scipy" in results:
            baseline = results["scipy"]
            for name, result in results.items():
                if name == "scipy":
                    continue

                # Peak should be at same location (within 1 bin)
                baseline_peak = np.argmax(baseline.power_db)
                result_peak = np.argmax(result.power_db)
                assert abs(baseline_peak - result_peak) <= 1, f"{name} peak mismatch"

                # Power levels should be similar (within 3 dB)
                peak_diff = abs(
                    baseline.power_db[baseline_peak] - result.power_db[result_peak]
                )
                assert peak_diff < 3, f"{name} power mismatch: {peak_diff} dB"


class TestBackendPerformance:
    """Basic performance sanity checks."""

    def test_backend_execution_time(self):
        """Test that FFT execution completes in reasonable time."""
        import time

        backend = get_backend("auto", fft_size=4096)

        # Generate test data
        iq = np.random.randn(4096).astype(np.complex64) + 1j * np.random.randn(4096).astype(np.complex64)

        # Time 100 executions
        start = time.perf_counter()
        for _ in range(100):
            backend.execute(iq, 48000)
        elapsed = time.perf_counter() - start

        # Should complete 100 FFTs in under 1 second on any modern system
        assert elapsed < 1.0, f"FFT too slow: {elapsed:.3f}s for 100 iterations"

        # Report timing
        per_fft_ms = elapsed * 10  # ms per FFT
        print(f"\n{backend.name} backend: {per_fft_ms:.2f} ms per 4096-point FFT")
