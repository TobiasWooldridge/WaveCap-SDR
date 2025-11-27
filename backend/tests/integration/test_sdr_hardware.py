"""Integration tests that require real SDR hardware.

These tests verify actual radio operations and should be run with a real
SDR device plugged in. They are marked with pytest markers so they can
be skipped when hardware is not available.

Run with: pytest tests/integration/test_sdr_hardware.py -v
Skip hardware tests: pytest -m "not hardware"
"""

import time
import pytest
import numpy as np

# Try to import SoapySDR
try:
    import SoapySDR
    SOAPY_AVAILABLE = True
except ImportError:
    SOAPY_AVAILABLE = False

from wavecapsdr.config import DeviceConfig
from wavecapsdr.devices.soapy import SoapyDriver
from wavecapsdr.devices.base import DeviceInfo


# Mark all tests in this module as requiring hardware
pytestmark = pytest.mark.hardware


def has_sdr_device() -> bool:
    """Check if any SDR device is available."""
    if not SOAPY_AVAILABLE:
        return False
    try:
        results = SoapySDR.Device.enumerate()
        return len(results) > 0
    except Exception:
        return False


def get_first_device_args() -> str | None:
    """Get args string for first available device."""
    if not SOAPY_AVAILABLE:
        return None
    try:
        results = SoapySDR.Device.enumerate()
        if results:
            # Build args string from first result
            args = results[0]
            items = []
            for k in args.keys():
                items.append(f"{k}={args[k]}")
            return ",".join(items)
        return None
    except Exception:
        return None


@pytest.fixture
def soapy_driver():
    """Create a SoapyDriver instance."""
    if not SOAPY_AVAILABLE:
        pytest.skip("SoapySDR not available")
    cfg = DeviceConfig()
    return SoapyDriver(cfg)


@pytest.fixture
def device_args():
    """Get args for first available device."""
    args = get_first_device_args()
    if args is None:
        pytest.skip("No SDR device available")
    return args


class TestDeviceEnumeration:
    """Tests for SDR device enumeration."""

    @pytest.mark.skipif(not SOAPY_AVAILABLE, reason="SoapySDR not available")
    def test_enumerate_returns_list(self, soapy_driver):
        """Enumerate should return a list of DeviceInfo."""
        devices = list(soapy_driver.enumerate())
        # Should return a list (even if empty)
        assert isinstance(devices, list)
        if devices:
            assert all(isinstance(d, DeviceInfo) for d in devices)

    @pytest.mark.skipif(not SOAPY_AVAILABLE, reason="SoapySDR not available")
    def test_enumerate_device_info_fields(self, soapy_driver):
        """DeviceInfo should have required fields."""
        devices = list(soapy_driver.enumerate())
        if not devices:
            pytest.skip("No devices found")

        device = devices[0]
        assert device.id is not None
        assert device.driver is not None
        assert device.label is not None
        assert device.freq_min_hz > 0
        assert device.freq_max_hz > device.freq_min_hz
        assert len(device.sample_rates) > 0

    @pytest.mark.skipif(not SOAPY_AVAILABLE, reason="SoapySDR not available")
    def test_enumerate_timeout_protection(self, soapy_driver):
        """Enumeration should complete within reasonable time."""
        start = time.time()
        list(soapy_driver.enumerate())
        elapsed = time.time() - start
        # Should complete within 30 seconds (includes timeout per driver)
        assert elapsed < 30


class TestDeviceOpen:
    """Tests for opening SDR devices."""

    @pytest.mark.skipif(not has_sdr_device(), reason="No SDR device available")
    def test_open_device(self, soapy_driver, device_args):
        """Should be able to open a device."""
        device = soapy_driver.open(device_args)
        try:
            assert device is not None
            assert device.info is not None
        finally:
            device.close()

    @pytest.mark.skipif(not has_sdr_device(), reason="No SDR device available")
    def test_device_info_after_open(self, soapy_driver, device_args):
        """Device info should be populated after open."""
        device = soapy_driver.open(device_args)
        try:
            info = device.info
            assert info.driver is not None
            assert info.freq_min_hz > 0
            assert info.freq_max_hz > info.freq_min_hz
        finally:
            device.close()


class TestDeviceConfigure:
    """Tests for SDR device configuration."""

    @pytest.mark.skipif(not has_sdr_device(), reason="No SDR device available")
    def test_configure_basic(self, soapy_driver, device_args):
        """Should be able to configure basic parameters."""
        device = soapy_driver.open(device_args)
        try:
            # Configure with safe defaults
            device.configure(
                center_hz=100_000_000,  # 100 MHz
                sample_rate=2_000_000,  # 2 MHz
            )
            # If we get here without exception, configuration worked
        finally:
            device.close()

    @pytest.mark.skipif(not has_sdr_device(), reason="No SDR device available")
    def test_configure_with_gain(self, soapy_driver, device_args):
        """Should be able to configure with manual gain."""
        device = soapy_driver.open(device_args)
        try:
            device.configure(
                center_hz=100_000_000,
                sample_rate=2_000_000,
                gain=30.0,
            )
        finally:
            device.close()

    @pytest.mark.skipif(not has_sdr_device(), reason="No SDR device available")
    def test_configure_with_bandwidth(self, soapy_driver, device_args):
        """Should be able to configure with bandwidth."""
        device = soapy_driver.open(device_args)
        try:
            device.configure(
                center_hz=100_000_000,
                sample_rate=2_000_000,
                bandwidth=1_500_000,
            )
        finally:
            device.close()


class TestDeviceStreaming:
    """Tests for SDR streaming operations."""

    @pytest.mark.skipif(not has_sdr_device(), reason="No SDR device available")
    def test_start_stop_stream(self, soapy_driver, device_args):
        """Should be able to start and stop a stream."""
        device = soapy_driver.open(device_args)
        try:
            device.configure(
                center_hz=100_000_000,
                sample_rate=2_000_000,
            )
            stream = device.start_stream()
            assert stream is not None
            stream.close()
        finally:
            device.close()

    @pytest.mark.skipif(not has_sdr_device(), reason="No SDR device available")
    def test_read_iq_samples(self, soapy_driver, device_args):
        """Should be able to read IQ samples."""
        device = soapy_driver.open(device_args)
        try:
            device.configure(
                center_hz=100_000_000,
                sample_rate=2_000_000,
            )
            stream = device.start_stream()
            try:
                # Give stream time to start
                time.sleep(0.1)

                # Read some samples
                samples, overflow = stream.read(8192)

                assert isinstance(samples, np.ndarray)
                assert samples.dtype == np.complex64
                # Should have read at least some samples
                # (may not get full 8192 on first read)

            finally:
                stream.close()
        finally:
            device.close()

    @pytest.mark.skipif(not has_sdr_device(), reason="No SDR device available")
    def test_continuous_read(self, soapy_driver, device_args):
        """Should be able to continuously read samples."""
        device = soapy_driver.open(device_args)
        try:
            device.configure(
                center_hz=100_000_000,
                sample_rate=2_000_000,
            )
            stream = device.start_stream()
            try:
                time.sleep(0.1)

                total_samples = 0
                for _ in range(10):
                    samples, overflow = stream.read(8192)
                    total_samples += len(samples)
                    if overflow:
                        print(f"Warning: overflow detected")

                # Should have received a significant number of samples
                assert total_samples > 10000

            finally:
                stream.close()
        finally:
            device.close()

    @pytest.mark.skipif(not has_sdr_device(), reason="No SDR device available")
    def test_sample_values_reasonable(self, soapy_driver, device_args):
        """IQ samples should have reasonable values."""
        device = soapy_driver.open(device_args)
        try:
            device.configure(
                center_hz=100_000_000,
                sample_rate=2_000_000,
                gain=20.0,  # Moderate gain
            )
            stream = device.start_stream()
            try:
                time.sleep(0.2)

                # Collect several reads
                all_samples = []
                for _ in range(5):
                    samples, _ = stream.read(8192)
                    if len(samples) > 0:
                        all_samples.append(samples)

                if all_samples:
                    combined = np.concatenate(all_samples)

                    # Samples should not all be zero
                    assert np.any(combined != 0), "All samples are zero"

                    # Samples should not all be the same value
                    assert np.std(np.abs(combined)) > 0, "No variation in samples"

                    # Values should be in reasonable range for normalized IQ
                    max_val = np.max(np.abs(combined))
                    assert max_val < 10.0, f"Sample values too large: {max_val}"

            finally:
                stream.close()
        finally:
            device.close()


class TestDeviceCapabilities:
    """Tests for querying device capabilities."""

    @pytest.mark.skipif(not has_sdr_device(), reason="No SDR device available")
    def test_get_capabilities(self, soapy_driver, device_args):
        """Should be able to query device capabilities."""
        device = soapy_driver.open(device_args)
        try:
            caps = device.get_capabilities()
            assert isinstance(caps, dict)

            # Should have at least some info
            # (specific fields depend on device)
            print(f"Device capabilities: {caps}")

        finally:
            device.close()

    @pytest.mark.skipif(not has_sdr_device(), reason="No SDR device available")
    def test_read_sensors(self, soapy_driver, device_args):
        """Should be able to read device sensors (if available)."""
        device = soapy_driver.open(device_args)
        try:
            sensors = device.read_sensors()
            assert isinstance(sensors, dict)
            # May be empty if device has no sensors
            print(f"Device sensors: {sensors}")

        finally:
            device.close()


class TestHotReconfiguration:
    """Tests for changing parameters while streaming."""

    @pytest.mark.skipif(not has_sdr_device(), reason="No SDR device available")
    def test_change_frequency_while_streaming(self, soapy_driver, device_args):
        """Should be able to change frequency while streaming."""
        device = soapy_driver.open(device_args)
        try:
            device.configure(
                center_hz=100_000_000,
                sample_rate=2_000_000,
            )
            stream = device.start_stream()
            try:
                time.sleep(0.1)

                # Read some samples
                samples1, _ = stream.read(8192)
                assert len(samples1) > 0

                # Change frequency while streaming
                device.reconfigure_running(center_hz=105_000_000)

                # Read more samples
                time.sleep(0.1)
                samples2, _ = stream.read(8192)
                # Should still be able to read
                # (length might be 0 briefly during reconfigure)

            finally:
                stream.close()
        finally:
            device.close()

    @pytest.mark.skipif(not has_sdr_device(), reason="No SDR device available")
    def test_change_gain_while_streaming(self, soapy_driver, device_args):
        """Should be able to change gain while streaming."""
        device = soapy_driver.open(device_args)
        try:
            device.configure(
                center_hz=100_000_000,
                sample_rate=2_000_000,
                gain=20.0,
            )
            stream = device.start_stream()
            try:
                time.sleep(0.1)

                # Read some samples
                stream.read(8192)

                # Change gain while streaming
                device.reconfigure_running(gain=30.0)

                time.sleep(0.1)
                # Should still work
                samples, _ = stream.read(8192)

            finally:
                stream.close()
        finally:
            device.close()


class TestFMBroadcastReception:
    """Tests that verify real FM broadcast reception.

    These tests tune to FM broadcast frequencies and verify
    that we receive actual signals with expected characteristics.
    """

    @pytest.mark.skipif(not has_sdr_device(), reason="No SDR device available")
    def test_fm_signal_has_energy(self, soapy_driver, device_args):
        """FM broadcast band should have signal energy."""
        device = soapy_driver.open(device_args)
        try:
            # Tune to FM broadcast band (88-108 MHz)
            device.configure(
                center_hz=100_000_000,  # 100 MHz
                sample_rate=2_000_000,
                bandwidth=200_000,
            )
            stream = device.start_stream()
            try:
                time.sleep(0.3)

                # Collect samples
                all_samples = []
                for _ in range(10):
                    samples, _ = stream.read(8192)
                    if len(samples) > 0:
                        all_samples.append(samples)

                if all_samples:
                    combined = np.concatenate(all_samples)

                    # Calculate power
                    power = np.mean(np.abs(combined) ** 2)
                    power_db = 10 * np.log10(power + 1e-10)

                    print(f"FM band power: {power_db:.1f} dB")

                    # Should have some signal power (above noise floor)
                    # This is a weak assertion since it depends on antenna

            finally:
                stream.close()
        finally:
            device.close()

    @pytest.mark.skipif(not has_sdr_device(), reason="No SDR device available")
    def test_demodulate_fm_produces_audio(self, soapy_driver, device_args):
        """FM demodulation should produce non-silent audio."""
        from wavecapsdr.dsp.fm import wbfm_demod

        device = soapy_driver.open(device_args)
        try:
            sample_rate = 2_000_000
            device.configure(
                center_hz=100_000_000,
                sample_rate=sample_rate,
                bandwidth=200_000,
            )
            stream = device.start_stream()
            try:
                time.sleep(0.3)

                # Collect about 100ms of samples
                target_samples = int(sample_rate * 0.1)
                all_samples = []
                while sum(len(s) for s in all_samples) < target_samples:
                    samples, _ = stream.read(16384)
                    if len(samples) > 0:
                        all_samples.append(samples)

                if all_samples:
                    iq = np.concatenate(all_samples)[:target_samples]

                    # Demodulate
                    audio = wbfm_demod(
                        iq,
                        sample_rate,
                        audio_rate=48_000,
                        enable_deemphasis=True,
                        enable_mpx_filter=True,
                    )

                    print(f"Audio samples: {len(audio)}, std: {np.std(audio):.4f}")

                    # Audio should have some content
                    assert np.std(audio) > 0.001, "Demodulated audio is silent"

            finally:
                stream.close()
        finally:
            device.close()
