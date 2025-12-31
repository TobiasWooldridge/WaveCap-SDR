"""Polyphase Channelizer for WaveCap-SDR.

Port of SDRTrunk's Non-Maximally Decimated Polyphase Filter Bank (NMDPFB) channelizer.
Divides wideband IQ samples into equal bandwidth channels with 2x oversampling.

Reference: Fred Harris, "Multirate Signal Processing for Communications Systems", p230-233
"""

from __future__ import annotations

import numpy as np
from wavecapsdr.typing import NDArrayAny
from scipy import signal
from scipy.fft import ifft
from typing import Optional


# Default channel bandwidth (matches SDRTrunk)
DEFAULT_CHANNEL_BANDWIDTH = 25000

# Default taps per channel (matches SDRTrunk)
DEFAULT_TAPS_PER_CHANNEL = 9

# Perfect reconstruction target: -6.02 dB at band edge
PERFECT_RECONSTRUCTION_GAIN = 0.5


class PolyphaseChannelizer:
    """Non-Maximally Decimated Polyphase Filter Bank channelizer.

    Extracts multiple channels from wideband IQ samples with 2x oversampling.
    Uses efficient polyphase structure with FFT-based channel separation.
    """

    def __init__(
        self,
        sample_rate: float,
        channel_bandwidth: int = DEFAULT_CHANNEL_BANDWIDTH,
        taps_per_channel: int = DEFAULT_TAPS_PER_CHANNEL,
    ):
        """Initialize the polyphase channelizer.

        Args:
            sample_rate: Input sample rate in Hz (e.g., 8_000_000 for 8 MHz)
            channel_bandwidth: Bandwidth per channel in Hz (default 25 kHz)
            taps_per_channel: Filter taps per polyphase arm (default 9)
        """
        self.sample_rate = sample_rate
        self.channel_bandwidth = channel_bandwidth
        self.taps_per_channel = taps_per_channel

        # Calculate channel count (must be even for 2x oversampling)
        self.channel_count = int(sample_rate / channel_bandwidth)
        if self.channel_count % 2 != 0:
            self.channel_count -= 1

        # Output sample rate per channel (2x oversampled)
        self.channel_sample_rate = (sample_rate / self.channel_count) * 2

        # Design the prototype filter and split into polyphase arms
        self._design_filter()

        # Sample history buffer for each arm (for filtering)
        self.arm_history = np.zeros((self.channel_count, self.taps_per_channel), dtype=np.complex64)

        # Block counter for 2x oversampling
        self.block_counter = 0

    def _design_filter(self) -> None:
        """Design the prototype lowpass filter and split into polyphase arms."""
        filter_length = self.channel_count * self.taps_per_channel - 1

        # Normalized cutoff: channel_bandwidth / sample_rate, then * 2 for firwin
        # Use slightly wider than 0.5 channel to get -6dB at channel edge
        cutoff = (self.channel_bandwidth * 0.9) / (self.sample_rate / 2)

        # Design lowpass prototype with Kaiser window (80 dB stopband)
        proto = signal.firwin(
            filter_length,
            cutoff,
            window=("kaiser", 8.0),
        ).astype(np.float64)

        # Split into polyphase arms
        # Arm k gets taps at indices k, k+M, k+2M, ...
        self.arms = np.zeros((self.channel_count, self.taps_per_channel), dtype=np.float64)
        for arm in range(self.channel_count):
            arm_taps = proto[arm :: self.channel_count]
            self.arms[arm, : len(arm_taps)] = arm_taps

    def process(self, samples: NDArrayAny) -> list[NDArrayAny]:
        """Process complex IQ samples through the channelizer.

        OPTIMIZED: Uses vectorized numpy operations instead of per-arm Python loops.
        - numpy.roll for shifting all arm histories at once
        - numpy.einsum for computing all dot products in parallel
        Provides ~2-3x speedup over original implementation.

        Args:
            samples: Complex IQ samples (numpy array of complex64)

        Returns:
            List of channel result arrays, each containing all channels
            for one output time sample. Each array has shape (channel_count,)
            containing complex samples.
        """
        results: list[NDArrayAny] = []

        # Process in blocks of channel_count samples
        # With 2x oversampling, we output 2 samples per channel_count input samples
        # by processing at half-block offsets
        block_size = self.channel_count

        for block_start in range(0, len(samples) - block_size + 1, block_size // 2):
            # Get input block
            block = samples[block_start : block_start + block_size]
            if len(block) < block_size:
                break

            # VECTORIZED: Shift all arm histories at once using numpy.roll
            # Roll along axis=1 shifts each row's elements to the right
            self.arm_history = np.roll(self.arm_history, 1, axis=1)

            # VECTORIZED: Insert new samples into all arms at once
            # arm k gets sample at index k from block
            self.arm_history[:, 0] = block

            # VECTORIZED: Compute all arm dot products in parallel using einsum
            # einsum 'ij,ij->i' computes sum over j of (arm_history[i,j] * arms[i,j])
            # This is equivalent to a per-row dot product
            arm_outputs = np.einsum("ij,ij->i", self.arm_history, self.arms).astype(np.complex64)

            # FFT to separate channels
            channel_outputs = np.fft.fft(arm_outputs).astype(np.complex64)
            results.append(channel_outputs)

        return results

    def reset(self) -> None:
        """Reset the channelizer state."""
        self.arm_history.fill(0)
        self.block_counter = 0

    def extract_channel(
        self,
        channel_results: list[NDArrayAny],
        channel_index: int,
    ) -> NDArrayAny:
        """Extract samples for a specific channel from processed results.

        Args:
            channel_results: List of processed channel arrays from process()
            channel_index: Index of channel to extract (0 to channel_count-1)

        Returns:
            Complex samples for the specified channel
        """
        return np.array([result[channel_index] for result in channel_results], dtype=np.complex64)


class ChannelCalculator:
    """Calculates channel indices for extracting specific frequencies."""

    def __init__(
        self,
        center_frequency: float,
        sample_rate: float,
        channel_bandwidth: int = DEFAULT_CHANNEL_BANDWIDTH,
    ):
        """Initialize the channel calculator.

        Args:
            center_frequency: Tuner center frequency in Hz
            sample_rate: Input sample rate in Hz
            channel_bandwidth: Bandwidth per channel in Hz
        """
        self.center_frequency = center_frequency
        self.sample_rate = sample_rate
        self.channel_bandwidth = channel_bandwidth

        # Calculate channel count (must be even)
        self.channel_count = int(sample_rate / channel_bandwidth)
        if self.channel_count % 2 != 0:
            self.channel_count -= 1

    def get_channel_index(self, target_frequency: float) -> int:
        """Get the channel index for a target frequency.

        Args:
            target_frequency: Frequency to extract in Hz

        Returns:
            Channel index (0 to channel_count-1)

        Note:
            FFT output ordering: Channel 0 = DC, positive frequencies in
            ascending order, negative frequencies wrap to end of array.
            - Channel 0: 0 Hz (DC)
            - Channel 1: +channel_bandwidth
            - Channel N/2: +Nyquist
            - Channel N-1: -channel_bandwidth
        """
        offset = target_frequency - self.center_frequency

        # Convert offset to channel index
        # Positive offsets: index = offset / bandwidth
        # Negative offsets: wrap around to end of array
        channels_offset = int(round(offset / self.channel_bandwidth))

        # Wrap negative indices to end of array
        if channels_offset < 0:
            return self.channel_count + channels_offset
        else:
            return channels_offset % self.channel_count

    def get_channel_center_frequency(self, channel_index: int) -> float:
        """Get the center frequency of a channel.

        Args:
            channel_index: Channel index (0 to channel_count-1)

        Returns:
            Center frequency of the channel in Hz
        """
        # FFT ordering: channels 0 to N/2-1 are positive, N/2 to N-1 are negative
        if channel_index < self.channel_count // 2:
            offset = channel_index * self.channel_bandwidth
        else:
            offset = (channel_index - self.channel_count) * self.channel_bandwidth

        return self.center_frequency + offset


def channelize_samples(
    samples: NDArrayAny,
    sample_rate: float,
    target_frequency: float,
    center_frequency: float,
    channel_bandwidth: int = DEFAULT_CHANNEL_BANDWIDTH,
) -> tuple[NDArrayAny, float]:
    """Convenience function to extract a single channel from wideband samples.

    Args:
        samples: Complex IQ samples at sample_rate
        sample_rate: Input sample rate in Hz
        target_frequency: Frequency to extract in Hz
        center_frequency: Tuner center frequency in Hz
        channel_bandwidth: Channel bandwidth in Hz (default 25 kHz)

    Returns:
        Tuple of (channel_samples, channel_sample_rate)
    """
    # Create channelizer
    channelizer = PolyphaseChannelizer(sample_rate, channel_bandwidth)

    # Create channel calculator
    calculator = ChannelCalculator(center_frequency, sample_rate, channel_bandwidth)

    # Get target channel index
    channel_index = calculator.get_channel_index(target_frequency)

    # Process samples
    results = channelizer.process(samples)

    # Extract target channel
    channel_samples = channelizer.extract_channel(results, channel_index)

    return channel_samples, channelizer.channel_sample_rate
