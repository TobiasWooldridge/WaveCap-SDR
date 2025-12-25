"""Analyze P25 sample to determine modulation type and carrier offset."""

import wave
import numpy as np
import logging

logging.basicConfig(level=logging.WARNING)


def load_iq_wav(filepath):
    with wave.open(filepath, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

        if sample_width == 3:  # 24-bit
            data_bytes = np.frombuffer(raw_data, dtype=np.uint8)
            data_bytes = data_bytes.reshape(-1, n_channels, 3)
            samples = np.zeros((n_frames, n_channels), dtype=np.int32)
            for ch in range(n_channels):
                b0 = data_bytes[:, ch, 0].astype(np.int32)
                b1 = data_bytes[:, ch, 1].astype(np.int32)
                b2 = data_bytes[:, ch, 2].astype(np.int32)
                raw = b0 | (b1 << 8) | (b2 << 16)
                samples[:, ch] = np.where(raw >= 0x800000, raw - 0x1000000, raw)
            samples = samples.astype(np.float32) / 8388608.0
        else:
            samples = np.frombuffer(raw_data, dtype=np.int16).reshape(-1, n_channels)
            samples = samples.astype(np.float32) / 32768.0

        iq = samples[:, 0] + 1j * samples[:, 1]
        return iq, sample_rate


if __name__ == "__main__":
    iq, sample_rate = load_iq_wav('tests/fixtures/p25_samples/P25_CQPSK-CC_IF.wav')
    print(f'Sample rate: {sample_rate} Hz')
    print(f'Duration: {len(iq) / sample_rate:.2f} seconds')
    print(f'Samples: {len(iq)}')

    # Check for carrier offset using FFT
    fft = np.fft.fft(iq[:sample_rate])  # First second
    freqs = np.fft.fftfreq(len(fft), 1/sample_rate)
    mag = np.abs(fft)

    # Find peak frequency
    peak_idx = np.argmax(mag)
    peak_freq = freqs[peak_idx]
    print(f'Peak frequency: {peak_freq:.1f} Hz')

    # Find top 5 peaks
    sorted_idx = np.argsort(mag)[::-1]
    print('Top 5 frequency components:')
    for i in range(5):
        idx = sorted_idx[i]
        print(f'  {freqs[idx]:.1f} Hz: magnitude {mag[idx]:.2f}')

    # Check constellation by computing phase of each sample
    print('\nConstellation analysis:')
    phases = np.angle(iq[:1000]) * 180 / np.pi
    print(f'Phase range: {phases.min():.1f}° to {phases.max():.1f}°')
    print(f'Phase mean: {phases.mean():.1f}°')
    print(f'Phase std: {phases.std():.1f}°')

    # Check magnitude distribution
    mags = np.abs(iq[:1000])
    print(f'Magnitude range: {mags.min():.4f} to {mags.max():.4f}')
    print(f'Magnitude mean: {mags.mean():.4f}')

    # Check if it looks like 4-level (C4FM) or phase-only (CQPSK)
    # C4FM has 4 distinct amplitude levels, CQPSK has constant amplitude
    print('\nModulation type detection:')
    print(f'Amplitude variation (std/mean): {mags.std()/mags.mean():.2%}')

    # For C4FM, measure the frequency deviation at symbol centers
    # Compute instantaneous frequency
    inst_phase = np.unwrap(np.angle(iq))
    inst_freq = np.diff(inst_phase) * sample_rate / (2 * np.pi)
    print(f'Instantaneous frequency: mean={inst_freq.mean():.1f} Hz, std={inst_freq.std():.1f} Hz')

    # Bin the instantaneous frequency to see if it shows 4 levels
    freq_hist, freq_edges = np.histogram(inst_freq, bins=50)
    peak_bins = np.argsort(freq_hist)[::-1][:5]
    print('Top 5 frequency bins:')
    for b in peak_bins:
        center = (freq_edges[b] + freq_edges[b+1]) / 2
        print(f'  {center:.0f} Hz: count {freq_hist[b]}')
