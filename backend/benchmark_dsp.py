#!/usr/bin/env python3
"""Performance benchmark for WaveCap-SDR DSP components.

Tests the performance of:
1. FM Demodulator (vectorized + JIT)
2. 8-tap Polyphase Interpolator (JIT)
3. Polyphase Channelizer (vectorized)
4. Sync Correlation (JIT)

Run with: python benchmark_dsp.py
"""

import time
import numpy as np


def benchmark_fm_demodulator(iterations: int = 10) -> dict:
    """Benchmark FM demodulator performance."""
    from wavecapsdr.dsp.p25.c4fm import _FMDemodulator

    # Simulate 1 second of 50 kHz samples
    sample_rate = 50000
    n_samples = sample_rate
    i = np.random.randn(n_samples).astype(np.float32) * 0.5
    q = np.random.randn(n_samples).astype(np.float32) * 0.5

    demod = _FMDemodulator(symbol_delay=10)

    # Warmup
    demod.demodulate(i[:1000], q[:1000])
    demod.reset()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        demod.reset()
        demod.demodulate(i, q)
    elapsed = time.perf_counter() - start

    samples_per_sec = n_samples * iterations / elapsed
    return {
        "component": "FM Demodulator",
        "samples_per_sec": samples_per_sec,
        "total_samples": n_samples * iterations,
        "elapsed_sec": elapsed,
        "iterations": iterations,
    }


def benchmark_interpolator(iterations: int = 10) -> dict:
    """Benchmark 8-tap interpolator performance."""
    from wavecapsdr.dsp.p25.c4fm import _Interpolator

    # Simulate processing 10000 symbols
    n_symbols = 10000
    samples = np.random.randn(n_symbols + 10).astype(np.float32)
    interp = _Interpolator()

    # Warmup (trigger JIT compilation)
    for i in range(100):
        interp.filter(samples, i % (len(samples) - 8), 0.5)

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        for i in range(n_symbols):
            interp.filter(samples, i % (len(samples) - 8), 0.5)
    elapsed = time.perf_counter() - start

    symbols_per_sec = n_symbols * iterations / elapsed
    return {
        "component": "8-tap Interpolator",
        "symbols_per_sec": symbols_per_sec,
        "total_symbols": n_symbols * iterations,
        "elapsed_sec": elapsed,
        "iterations": iterations,
    }


def benchmark_sync_detector(iterations: int = 10) -> dict:
    """Benchmark sync correlation performance."""
    from wavecapsdr.dsp.p25.c4fm import _SoftSyncDetector

    # Simulate processing 10000 symbols
    n_symbols = 10000
    symbols = np.random.randn(n_symbols).astype(np.float32) * 3.0
    detector = _SoftSyncDetector()

    # Warmup (trigger JIT compilation)
    for i in range(100):
        detector.process(symbols[i])
    detector.reset()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        detector.reset()
        for sym in symbols:
            detector.process(sym)
    elapsed = time.perf_counter() - start

    symbols_per_sec = n_symbols * iterations / elapsed
    return {
        "component": "Sync Detector",
        "symbols_per_sec": symbols_per_sec,
        "total_symbols": n_symbols * iterations,
        "elapsed_sec": elapsed,
        "iterations": iterations,
    }


def benchmark_channelizer(iterations: int = 5) -> dict:
    """Benchmark polyphase channelizer performance."""
    from wavecapsdr.dsp.channelizer import PolyphaseChannelizer

    # Simulate 1 second of 8 MHz samples
    sample_rate = 8_000_000
    n_samples = sample_rate
    samples = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)).astype(np.complex64) * 0.5

    channelizer = PolyphaseChannelizer(sample_rate=sample_rate)

    # Warmup
    channelizer.process(samples[:100000])
    channelizer.reset()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        channelizer.reset()
        channelizer.process(samples)
    elapsed = time.perf_counter() - start

    samples_per_sec = n_samples * iterations / elapsed
    return {
        "component": "Polyphase Channelizer",
        "samples_per_sec": samples_per_sec,
        "total_samples": n_samples * iterations,
        "elapsed_sec": elapsed,
        "iterations": iterations,
    }


def benchmark_full_demodulator(iterations: int = 5) -> dict:
    """Benchmark full C4FM demodulator pipeline."""
    from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator

    # Simulate 1 second of 50 kHz channel samples
    sample_rate = 50000
    n_samples = sample_rate
    iq = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)).astype(np.complex64) * 0.5

    demod = C4FMDemodulator(sample_rate=sample_rate)

    # Warmup (trigger JIT compilation and filter state initialization)
    demod.demodulate(iq[:10000])
    demod.reset()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        demod.reset()
        demod.demodulate(iq)
    elapsed = time.perf_counter() - start

    samples_per_sec = n_samples * iterations / elapsed
    return {
        "component": "Full C4FM Demodulator",
        "samples_per_sec": samples_per_sec,
        "total_samples": n_samples * iterations,
        "elapsed_sec": elapsed,
        "iterations": iterations,
    }


def main():
    print("=" * 60)
    print("WaveCap-SDR DSP Performance Benchmark")
    print("=" * 60)
    print()

    # Check if numba is available
    try:
        from wavecapsdr.dsp.p25.c4fm import NUMBA_AVAILABLE
        print(f"Numba JIT: {'ENABLED' if NUMBA_AVAILABLE else 'DISABLED (falling back to numpy)'}")
    except ImportError:
        print("Numba JIT: UNKNOWN")
    print()

    results = []

    print("Running benchmarks...")
    print("-" * 60)

    # FM Demodulator
    print("Benchmarking FM Demodulator...", flush=True)
    results.append(benchmark_fm_demodulator())

    # Interpolator
    print("Benchmarking 8-tap Interpolator...", flush=True)
    results.append(benchmark_interpolator())

    # Sync Detector
    print("Benchmarking Sync Detector...", flush=True)
    results.append(benchmark_sync_detector())

    # Full C4FM Demodulator
    print("Benchmarking Full C4FM Demodulator...", flush=True)
    results.append(benchmark_full_demodulator())

    # Channelizer (takes longest)
    print("Benchmarking Polyphase Channelizer...", flush=True)
    results.append(benchmark_channelizer())

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print()

    for r in results:
        print(f"{r['component']}:")
        if "samples_per_sec" in r:
            rate = r["samples_per_sec"]
            if rate >= 1_000_000:
                print(f"  Throughput: {rate / 1_000_000:.2f} M samples/sec")
            else:
                print(f"  Throughput: {rate / 1_000:.2f} K samples/sec")
        if "symbols_per_sec" in r:
            rate = r["symbols_per_sec"]
            if rate >= 1_000_000:
                print(f"  Throughput: {rate / 1_000_000:.2f} M symbols/sec")
            else:
                print(f"  Throughput: {rate / 1_000:.2f} K symbols/sec")
        print(f"  Elapsed: {r['elapsed_sec']:.3f}s ({r['iterations']} iterations)")
        print()

    # Performance targets
    print("=" * 60)
    print("Performance Targets")
    print("=" * 60)
    print()
    print("For real-time 8 MHz wideband capture:")
    print("  - Channelizer: 8 M samples/sec (minimum)")
    print("  - C4FM Demod:  50 K samples/sec per channel")
    print("  - Sync Detect: 4800 symbols/sec per channel (1x real-time)")
    print()

    # Check if targets are met
    channelizer_rate = next((r["samples_per_sec"] for r in results if "Channelizer" in r["component"]), 0)
    demod_rate = next((r["samples_per_sec"] for r in results if "Full C4FM" in r["component"]), 0)

    print("Status:")
    if channelizer_rate >= 8_000_000:
        print(f"  ✓ Channelizer: {channelizer_rate / 1_000_000:.1f}M samples/sec (target: 8M)")
    else:
        print(f"  ✗ Channelizer: {channelizer_rate / 1_000_000:.1f}M samples/sec (target: 8M)")

    channels_supported = int(demod_rate / 50_000)
    print(f"  - C4FM Demod: {demod_rate / 1_000:.0f}K samples/sec = ~{channels_supported} channels")


if __name__ == "__main__":
    main()
