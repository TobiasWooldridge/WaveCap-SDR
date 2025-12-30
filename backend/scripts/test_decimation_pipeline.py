#!/usr/bin/env python3
"""Test decimation pipeline on SDRTrunk recording.

This tests if our 3-stage decimation is causing the TSBK decode failures.
The SDRTrunk recording is at 50 kHz baseband, so we simulate:
1. Upsampling to 6 MHz (to mimic SDR capture)
2. Running through our 3-stage decimation pipeline
3. Decoding with ControlChannelMonitor

If this fails but direct decode works, the decimation is the problem.
"""

import sys
import wave
import numpy as np
from scipy import signal as scipy_signal

sys.path.insert(0, "/Users/thw/Projects/WaveCap-SDR/backend")

from wavecapsdr.trunking.control_channel import ControlChannelMonitor
from wavecapsdr.trunking.config import TrunkingProtocol


def load_iq_from_wav(wav_path: str) -> tuple[np.ndarray, int]:
    """Load IQ samples from a WAV file (stereo = I/Q interleaved)."""
    with wave.open(wav_path, 'rb') as w:
        sample_rate = w.getframerate()
        n_frames = w.getnframes()
        n_channels = w.getnchannels()
        sample_width = w.getsampwidth()

        print(f"WAV file: {wav_path}")
        print(f"  Channels: {n_channels}")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Sample width: {sample_width} bytes")
        print(f"  Frames: {n_frames}")
        print(f"  Duration: {n_frames / sample_rate:.2f} seconds")

        raw_data = w.readframes(n_frames)

        if sample_width == 2:
            samples = np.frombuffer(raw_data, dtype=np.int16)
        elif sample_width == 4:
            samples = np.frombuffer(raw_data, dtype=np.int32)
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        samples = samples.reshape(-1, n_channels)

        if n_channels == 2:
            i = samples[:, 0].astype(np.float64)
            q = samples[:, 1].astype(np.float64)
            max_val = 2 ** (sample_width * 8 - 1)
            i = i / max_val
            q = q / max_val
            iq = i + 1j * q
        else:
            raise ValueError(f"Expected stereo file, got {n_channels} channels")

        return iq, sample_rate


def test_direct_decode(iq: np.ndarray, sample_rate: int) -> int:
    """Test direct decode without decimation pipeline."""
    print(f"\n=== Test 1: Direct decode at {sample_rate} Hz ===")

    # Create control channel monitor at the input sample rate
    monitor = ControlChannelMonitor(
        protocol=TrunkingProtocol.P25_PHASE1,
        sample_rate=sample_rate
    )

    # Process in chunks
    chunk_size = 10000
    total_tsbks = 0

    for i in range(0, len(iq), chunk_size):
        chunk = iq[i:i+chunk_size]
        if len(chunk) < chunk_size:
            break
        results = monitor.process_iq(chunk)
        for tsbk_data in results:
            if tsbk_data:
                total_tsbks += 1

    stats = monitor.get_stats()
    print(f"Frames decoded: {stats.get('frames_decoded', 0)}")
    print(f"TSBK attempts: {stats.get('tsbk_attempts', 0)}")
    print(f"TSBK CRC pass: {stats.get('tsbk_crc_pass', 0)}")
    if stats.get('tsbk_attempts', 0) > 0:
        rate = 100 * stats.get('tsbk_crc_pass', 0) / stats.get('tsbk_attempts', 1)
        print(f"CRC pass rate: {rate:.1f}%")
    print(f"Total TSBKs: {total_tsbks}")

    return total_tsbks


def test_with_decimation_pipeline(iq: np.ndarray, input_rate: int) -> int:
    """Test with 3-stage decimation pipeline like the live system."""
    print(f"\n=== Test 2: With 3-stage decimation pipeline ===")

    # Simulate upsampling to 6 MHz (like SDR capture)
    # Actually, let's just use the same decimation ratios relative to input
    # The key is to test the filter cascade

    # SDRTrunk is at 50 kHz, our live system expects 6 MHz
    # Let's simulate: 50 kHz → run through filters proportionally

    # Actually, let's test with 3-stage decimation using similar relative factors
    # Input: 50 kHz → decimate by 2 to 25 kHz

    # Stage 1: 50 kHz → 25 kHz (2:1)
    stage1_factor = 2
    stage1_rate = input_rate // stage1_factor

    # Design filter
    stage1_cutoff = 0.8 / stage1_factor
    stage1_taps = scipy_signal.firwin(41, stage1_cutoff, window=("kaiser", 7.857))

    # Apply filter and decimate
    iq_stage1 = scipy_signal.lfilter(stage1_taps, 1.0, iq)
    iq_stage1 = iq_stage1[::stage1_factor]

    print(f"Stage 1: {input_rate/1e3:.1f} kHz → {stage1_rate/1e3:.1f} kHz")
    print(f"  Samples: {len(iq)} → {len(iq_stage1)}")
    print(f"  Power before: {np.mean(np.abs(iq)**2):.6f}")
    print(f"  Power after: {np.mean(np.abs(iq_stage1)**2):.6f}")

    # Create control channel monitor at decimated rate
    monitor = ControlChannelMonitor(
        protocol=TrunkingProtocol.P25_PHASE1,
        sample_rate=stage1_rate
    )

    # Process in chunks
    chunk_size = 5000
    total_tsbks = 0

    for i in range(0, len(iq_stage1), chunk_size):
        chunk = iq_stage1[i:i+chunk_size]
        if len(chunk) < chunk_size:
            break
        results = monitor.process_iq(chunk)
        for tsbk_data in results:
            if tsbk_data:
                total_tsbks += 1

    stats = monitor.get_stats()
    print(f"Frames decoded: {stats.get('frames_decoded', 0)}")
    print(f"TSBK attempts: {stats.get('tsbk_attempts', 0)}")
    print(f"TSBK CRC pass: {stats.get('tsbk_crc_pass', 0)}")
    if stats.get('tsbk_attempts', 0) > 0:
        rate = 100 * stats.get('tsbk_crc_pass', 0) / stats.get('tsbk_attempts', 1)
        print(f"CRC pass rate: {rate:.1f}%")
    print(f"Total TSBKs: {total_tsbks}")

    return total_tsbks


def test_with_simulated_wideband(iq: np.ndarray, input_rate: int) -> int:
    """Simulate the full wideband capture → narrowband extraction pipeline.

    This most closely matches what happens in the live system:
    1. Take narrowband signal at 50 kHz
    2. Upsample to simulate wideband (6 MHz) capture
    3. Apply offset to simulate off-center channel
    4. Run through 3-stage decimation with frequency shift
    5. Decode
    """
    print(f"\n=== Test 3: Simulated wideband capture pipeline ===")

    # Parameters matching live system
    wideband_rate = 6000000  # 6 MHz SDR capture
    offset_hz = 500000  # 500 kHz offset (typical control channel offset)

    # Upsample the 50 kHz baseband to 6 MHz
    # This simulates what the SDR would capture
    upsample_factor = wideband_rate // input_rate
    print(f"Upsampling {input_rate/1e3:.1f} kHz → {wideband_rate/1e6:.1f} MHz ({upsample_factor}:1)")

    # Insert zeros for upsampling
    iq_wideband = np.zeros(len(iq) * upsample_factor, dtype=np.complex128)
    iq_wideband[::upsample_factor] = iq

    # Apply interpolation filter (lowpass at original Nyquist)
    # Cutoff = input_rate / wideband_rate = 1/120
    interp_cutoff = 0.8 * input_rate / wideband_rate
    interp_taps = scipy_signal.firwin(501, interp_cutoff, window=("kaiser", 7.857))
    iq_wideband = scipy_signal.lfilter(interp_taps, 1.0, iq_wideband) * upsample_factor

    # Apply frequency offset (simulate off-center channel)
    t = np.arange(len(iq_wideband)) / wideband_rate
    iq_wideband = iq_wideband * np.exp(2j * np.pi * offset_hz * t)

    print(f"Wideband samples: {len(iq_wideband)}")
    print(f"Wideband power: {np.mean(np.abs(iq_wideband)**2):.6f}")

    # Now run through 3-stage decimation (like system.py)
    # Frequency shift to center the channel
    t_shifted = np.arange(len(iq_wideband)) / wideband_rate
    iq_centered = iq_wideband * np.exp(-2j * np.pi * offset_hz * t_shifted)

    # Stage 1: 6 MHz → 200 kHz (30:1)
    stage1_factor = 30
    stage1_rate = wideband_rate // stage1_factor
    stage1_cutoff = 0.8 / stage1_factor
    stage1_taps = scipy_signal.firwin(157, stage1_cutoff, window=("kaiser", 7.857))
    iq_stage1 = scipy_signal.lfilter(stage1_taps, 1.0, iq_centered)
    iq_stage1 = iq_stage1[::stage1_factor]
    print(f"Stage 1: {wideband_rate/1e6:.1f} MHz → {stage1_rate/1e3:.1f} kHz")
    print(f"  Samples: {len(iq_centered)} → {len(iq_stage1)}")

    # Stage 2: 200 kHz → 50 kHz (4:1)
    stage2_factor = 4
    stage2_rate = stage1_rate // stage2_factor
    stage2_cutoff = 0.8 / stage2_factor
    stage2_taps = scipy_signal.firwin(73, stage2_cutoff, window=("kaiser", 7.857))
    iq_stage2 = scipy_signal.lfilter(stage2_taps, 1.0, iq_stage1)
    iq_stage2 = iq_stage2[::stage2_factor]
    print(f"Stage 2: {stage1_rate/1e3:.1f} kHz → {stage2_rate/1e3:.1f} kHz")
    print(f"  Samples: {len(iq_stage1)} → {len(iq_stage2)}")

    # Stage 3: 50 kHz → 25 kHz (2:1)
    stage3_factor = 2
    stage3_rate = stage2_rate // stage3_factor
    stage3_cutoff = 0.8 / stage3_factor
    stage3_taps = scipy_signal.firwin(41, stage3_cutoff, window=("kaiser", 7.857))
    iq_stage3 = scipy_signal.lfilter(stage3_taps, 1.0, iq_stage2)
    iq_stage3 = iq_stage3[::stage3_factor]
    print(f"Stage 3: {stage2_rate/1e3:.1f} kHz → {stage3_rate/1e3:.1f} kHz")
    print(f"  Samples: {len(iq_stage2)} → {len(iq_stage3)}")
    print(f"  Final power: {np.mean(np.abs(iq_stage3)**2):.6f}")

    # Create control channel monitor at final rate
    monitor = ControlChannelMonitor(
        protocol=TrunkingProtocol.P25_PHASE1,
        sample_rate=stage3_rate
    )

    # Process in chunks
    chunk_size = 2500
    total_tsbks = 0

    for i in range(0, len(iq_stage3), chunk_size):
        chunk = iq_stage3[i:i+chunk_size]
        if len(chunk) < chunk_size:
            break
        results = monitor.process_iq(chunk)
        for tsbk_data in results:
            if tsbk_data:
                total_tsbks += 1

    stats = monitor.get_stats()
    print(f"Frames decoded: {stats.get('frames_decoded', 0)}")
    print(f"TSBK attempts: {stats.get('tsbk_attempts', 0)}")
    print(f"TSBK CRC pass: {stats.get('tsbk_crc_pass', 0)}")
    if stats.get('tsbk_attempts', 0) > 0:
        rate = 100 * stats.get('tsbk_crc_pass', 0) / stats.get('tsbk_attempts', 1)
        print(f"CRC pass rate: {rate:.1f}%")
    print(f"Total TSBKs: {total_tsbks}")

    return total_tsbks


if __name__ == "__main__":
    wav_path = sys.argv[1] if len(sys.argv) > 1 else "20251227_224220_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav"

    # Load IQ samples
    iq, sample_rate = load_iq_from_wav(wav_path)

    print(f"\nLoaded {len(iq)} samples")
    print(f"IQ power: {np.mean(np.abs(iq)**2):.6f}")
    print(f"IQ peak: {np.max(np.abs(iq)):.6f}")

    # Test 1: Direct decode (should work)
    tsbks_direct = test_direct_decode(iq, sample_rate)

    # Test 2: With simple decimation
    tsbks_decimated = test_with_decimation_pipeline(iq, sample_rate)

    # Test 3: Simulated full wideband pipeline
    tsbks_wideband = test_with_simulated_wideband(iq, sample_rate)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Test 1 (Direct decode at 50 kHz):     {tsbks_direct:3d} TSBKs")
    print(f"Test 2 (Decimated 50→25 kHz):         {tsbks_decimated:3d} TSBKs")
    print(f"Test 3 (Simulated wideband pipeline): {tsbks_wideband:3d} TSBKs")
