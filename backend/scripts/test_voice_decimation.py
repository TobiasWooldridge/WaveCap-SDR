#!/usr/bin/env python3
"""Test voice channel decimation to understand where signal is lost."""

import numpy as np
from scipy import signal as scipy_signal

def test_decimation_pipeline():
    """Simulate voice channel extraction from wideband IQ."""

    # Parameters matching VoiceRecorder
    sample_rate = 6_000_000  # 6 MHz
    target_rate = 48_000  # Target 48 kHz
    voice_offset_hz = 1_650_000  # 1.65 MHz offset (typical)

    # Generate test signal: voice at +1.65 MHz offset
    duration_s = 0.01  # 10ms
    n_samples = int(sample_rate * duration_s)
    t = np.arange(n_samples) / sample_rate

    # Create P25-like signal: 4800 symbols/sec, ±1.8 kHz deviation
    # C4FM uses 4 levels: -3, -1, +1, +3 mapped to ±1800 Hz outer, ±600 Hz inner
    symbol_rate = 4800
    symbols_per_sec = int(duration_s * symbol_rate)
    symbols = np.random.choice([-3, -1, 1, 3], symbols_per_sec)  # C4FM dibits

    # Upsample symbols to sample rate with proper pulse shaping (simple boxcar here)
    samples_per_symbol = sample_rate // symbol_rate
    fm_signal = np.repeat(symbols, samples_per_symbol)[:n_samples].astype(np.float64)

    # FM modulate at +1.65 MHz offset
    # Deviation: each dibit unit = 600 Hz, so ±3 gives ±1800 Hz
    deviation_per_dibit = 600  # Hz per dibit unit
    freq_deviation = fm_signal * deviation_per_dibit  # ±1800 Hz for ±3 dibits

    # Instantaneous frequency = carrier + deviation
    inst_freq = voice_offset_hz + freq_deviation

    # Phase is integral of frequency: φ(t) = 2π ∫ f(t) dt
    # For discrete samples: φ[n] = 2π Σ f[k] / fs
    phase = 2 * np.pi * np.cumsum(inst_freq) / sample_rate
    voice_iq = np.exp(1j * phase).astype(np.complex64)

    # Add noise
    noise_power = 0.01
    noise = np.sqrt(noise_power/2) * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
    iq = (voice_iq + noise).astype(np.complex64)

    print(f"=== Input IQ ===")
    print(f"Sample rate: {sample_rate/1e6:.1f} MHz")
    print(f"Voice offset: {voice_offset_hz/1e3:.1f} kHz")
    print(f"Samples: {n_samples}")
    print(f"Raw power: {np.mean(np.abs(iq)**2):.6f}")

    # Measure power at different frequencies using FFT
    fft = np.fft.fft(iq)
    fft_power = np.abs(fft) ** 2
    fft_freqs = np.fft.fftfreq(len(fft), 1/sample_rate)

    # Find power at DC (±25 kHz) vs at voice offset (±25 kHz)
    dc_mask = np.abs(fft_freqs) < 25000
    voice_mask = np.abs(fft_freqs - voice_offset_hz) < 25000

    dc_power = np.sum(fft_power[dc_mask]) / len(fft)
    voice_power = np.sum(fft_power[voice_mask]) / len(fft)

    print(f"\n=== Before Frequency Shift ===")
    print(f"Power at DC (±25kHz): {dc_power:.6f}")
    print(f"Power at voice offset (±25kHz): {voice_power:.6f}")

    # Apply frequency shift to center voice at DC
    n = np.arange(len(iq), dtype=np.float64)
    phase_shift = -2.0 * np.pi * voice_offset_hz * n / sample_rate
    shift = np.exp(1j * phase_shift).astype(np.complex64)
    centered_iq = iq * shift

    print(f"\n=== After Frequency Shift ===")
    print(f"Centered power: {np.mean(np.abs(centered_iq)**2):.6f}")

    # Measure power at DC after shift
    fft_centered = np.fft.fft(centered_iq)
    fft_power_centered = np.abs(fft_centered) ** 2

    dc_power_after = np.sum(fft_power_centered[dc_mask]) / len(fft_centered)
    voice_power_after = np.sum(fft_power_centered[voice_mask]) / len(fft_centered)

    print(f"Power at DC (±25kHz): {dc_power_after:.6f}")
    print(f"Power at old voice offset: {voice_power_after:.6f}")

    # Two-stage decimation
    stage1_decim = 25  # 6 MHz → 240 kHz
    stage2_decim = 5   # 240 kHz → 48 kHz

    # Stage 1 filter
    stage1_cutoff = 0.8 / stage1_decim
    stage1_taps = scipy_signal.firwin(157, stage1_cutoff, window=("kaiser", 7.857))

    filtered1 = scipy_signal.lfilter(stage1_taps, 1.0, centered_iq)
    decimated1 = filtered1[::stage1_decim]

    print(f"\n=== Stage 1 Decimation ({stage1_decim}:1) ===")
    print(f"Cutoff: {stage1_cutoff:.4f} normalized = {stage1_cutoff * sample_rate / 2 / 1000:.1f} kHz")
    print(f"Samples after: {len(decimated1)}")
    print(f"Power after stage1: {np.mean(np.abs(decimated1)**2):.6f}")

    # Stage 2 filter
    stage1_rate = sample_rate // stage1_decim
    stage2_cutoff = 0.8 / stage2_decim
    stage2_taps = scipy_signal.firwin(73, stage2_cutoff, window=("kaiser", 7.857))

    filtered2 = scipy_signal.lfilter(stage2_taps, 1.0, decimated1)
    decimated2 = filtered2[::stage2_decim]

    print(f"\n=== Stage 2 Decimation ({stage2_decim}:1) ===")
    print(f"Cutoff: {stage2_cutoff:.4f} normalized = {stage2_cutoff * stage1_rate / 2 / 1000:.1f} kHz")
    print(f"Samples after: {len(decimated2)}")
    print(f"Power after stage2: {np.mean(np.abs(decimated2)**2):.6f}")

    # FM discriminator
    phase = np.angle(decimated2)
    print(f"\n=== FM Discriminator ===")
    print(f"Phase before unwrap range: [{phase.min():.3f}, {phase.max():.3f}]")

    phase_unwrapped = np.unwrap(phase)
    print(f"Phase after unwrap range: [{phase_unwrapped.min():.3f}, {phase_unwrapped.max():.3f}]")

    disc_audio = np.diff(phase_unwrapped)

    print(f"Samples: {len(disc_audio)}")
    print(f"Disc range: [{disc_audio.min():.3f}, {disc_audio.max():.3f}]")
    print(f"Expected for P25: ±0.24 rad")

    # Check where large jumps occur
    large_jumps = np.where(np.abs(disc_audio) > 1.0)[0]
    print(f"Large jumps (>1 rad): {len(large_jumps)} locations")
    if len(large_jumps) > 0 and len(large_jumps) <= 10:
        for idx in large_jumps[:5]:
            mag = np.abs(decimated2[idx])
            print(f"  Jump at {idx}: {disc_audio[idx]:.3f} rad, magnitude={mag:.6f}")

    # Check signal magnitude statistics
    mags = np.abs(decimated2)
    print(f"\nSignal magnitude: min={mags.min():.6f}, max={mags.max():.6f}, mean={mags.mean():.6f}")
    near_zero = np.sum(mags < 0.1)
    print(f"Samples with magnitude < 0.1: {near_zero} ({100*near_zero/len(mags):.1f}%)")

    # Check if discriminator output is valid
    disc_range = disc_audio.max() - disc_audio.min()
    if disc_range > 2.0:
        print(f"WARNING: Discriminator range {disc_range:.2f} > 2.0 rad indicates NOISE")
    else:
        print(f"OK: Discriminator range {disc_range:.2f} rad is valid P25")

    return disc_audio

if __name__ == "__main__":
    test_decimation_pipeline()
