#!/usr/bin/env python3
"""Profile P25 processing pipeline to identify bottlenecks."""

import cProfile
import pstats
import io
import time
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from wavecapsdr.decoders.p25 import P25Decoder
from wavecapsdr.decoders.p25_framer import P25P1MessageFramer, P25P1SoftSyncDetector


def generate_test_iq(sample_rate: int = 48000, duration_ms: int = 100) -> np.ndarray:
    """Generate synthetic IQ samples for testing."""
    n_samples = int(sample_rate * duration_ms / 1000)
    # Generate random IQ with some structure
    t = np.arange(n_samples) / sample_rate
    # Add some frequency offset and noise
    freq_offset = 1000  # Hz
    phase = 2 * np.pi * freq_offset * t
    i = np.cos(phase) + np.random.randn(n_samples) * 0.3
    q = np.sin(phase) + np.random.randn(n_samples) * 0.3
    return (i + 1j * q).astype(np.complex64)


def benchmark_sync_detector():
    """Benchmark sync detector implementations."""
    print("\n=== Sync Detector Benchmark ===")

    detector = P25P1SoftSyncDetector()
    samples = np.random.randn(10000).astype(np.float32) * 3

    # Warm up
    for _ in range(3):
        detector.process_batch(samples)

    # Per-symbol benchmark
    detector.reset()
    start = time.perf_counter()
    iterations = 0
    while time.perf_counter() - start < 1.0:
        for s in samples:
            detector.process(s)
        iterations += 1
    per_symbol_rate = iterations * len(samples) / (time.perf_counter() - start)
    print(f"  Per-symbol: {per_symbol_rate/1e6:.2f}M samples/sec")

    # Batch benchmark
    detector.reset()
    start = time.perf_counter()
    iterations = 0
    while time.perf_counter() - start < 1.0:
        detector.process_batch(samples)
        iterations += 1
    batch_rate = iterations * len(samples) / (time.perf_counter() - start)
    print(f"  Batch:      {batch_rate/1e6:.2f}M samples/sec")
    print(f"  Speedup:    {batch_rate/per_symbol_rate:.1f}x")


def benchmark_framer():
    """Benchmark message framer implementations."""
    print("\n=== Message Framer Benchmark ===")

    framer = P25P1MessageFramer()
    framer.start()

    # Generate test data
    n_symbols = 10000
    soft_symbols = np.random.randn(n_symbols).astype(np.float32) * 3
    dibits = np.random.randint(0, 4, n_symbols, dtype=np.int32)

    # Warm up
    for _ in range(3):
        framer.process_batch(soft_symbols, dibits)

    # Per-symbol benchmark
    framer.reset()
    start = time.perf_counter()
    iterations = 0
    while time.perf_counter() - start < 1.0:
        for i in range(n_symbols):
            framer.process_with_soft_sync(soft_symbols[i], int(dibits[i]))
        iterations += 1
    per_symbol_rate = iterations * n_symbols / (time.perf_counter() - start)
    print(f"  Per-symbol: {per_symbol_rate/1e6:.2f}M symbols/sec")

    # Batch benchmark
    framer.reset()
    start = time.perf_counter()
    iterations = 0
    while time.perf_counter() - start < 1.0:
        framer.process_batch(soft_symbols, dibits)
        iterations += 1
    batch_rate = iterations * n_symbols / (time.perf_counter() - start)
    print(f"  Batch:      {batch_rate/1e6:.2f}M symbols/sec")
    print(f"  Speedup:    {batch_rate/per_symbol_rate:.1f}x")

    # Profile the batch method to see what's slow
    print("\n  Profiling batch method:")
    framer.reset()

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(100):
        framer.process_batch(soft_symbols, dibits)

    profiler.disable()

    # Print top functions
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(15)

    for line in s.getvalue().split('\n')[:20]:
        print(f"    {line}")


def benchmark_p25_decoder():
    """Benchmark full P25 decoder pipeline."""
    print("\n=== P25 Decoder Pipeline Benchmark ===")

    decoder = P25Decoder(sample_rate=48000)

    # Generate 100ms of IQ
    iq = generate_test_iq(48000, 100)

    # Warm up
    for _ in range(3):
        decoder.process(iq)

    # Benchmark
    start = time.perf_counter()
    iterations = 0
    while time.perf_counter() - start < 2.0:
        decoder.process(iq)
        iterations += 1

    elapsed = time.perf_counter() - start
    samples_per_sec = iterations * len(iq) / elapsed
    realtime_ratio = samples_per_sec / 48000

    print(f"  {samples_per_sec/1e6:.2f}M samples/sec")
    print(f"  {realtime_ratio:.1f}x realtime")

    # Profile
    print("\n  Profiling decoder pipeline:")

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(50):
        decoder.process(iq)

    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)

    for line in s.getvalue().split('\n')[:25]:
        print(f"    {line}")


def benchmark_state_machine():
    """Benchmark the _process state machine in isolation."""
    print("\n=== State Machine (_process) Benchmark ===")

    framer = P25P1MessageFramer()
    framer.start()

    # Generate test dibits
    n_dibits = 100000
    dibits = np.random.randint(0, 4, n_dibits, dtype=np.int32)

    # Warm up
    framer.reset()
    for d in dibits[:1000]:
        framer.process(int(d))

    # Benchmark pure state machine (no sync detection)
    framer.reset()
    start = time.perf_counter()
    for d in dibits:
        framer.process(int(d))
    elapsed = time.perf_counter() - start

    rate = n_dibits / elapsed
    print(f"  State machine only: {rate/1e6:.2f}M dibits/sec")
    print(f"  Per-dibit time: {elapsed/n_dibits*1e9:.1f} ns")

    # P25 symbol rate is 4800 baud (9600 bits/sec = 4800 dibits/sec)
    # With 3 channels, need 14400 dibits/sec
    required_rate = 14400
    headroom = rate / required_rate
    print(f"  Headroom vs 3-channel P25: {headroom:.0f}x")


if __name__ == '__main__':
    print("=" * 60)
    print("P25 Processing Pipeline Performance Profile")
    print("=" * 60)

    benchmark_sync_detector()
    benchmark_state_machine()
    benchmark_framer()
    benchmark_p25_decoder()

    print("\n" + "=" * 60)
    print("Complete")
