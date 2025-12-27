#!/usr/bin/env python3
"""Test the differential demod fix for P25 C4FM decoding.

This script verifies that switching from _fm_discriminator to _differential_demod
fixes the "zero frames decoded" issue.

Usage:
    # Test with sdrtrunk recording
    python test_differential_demod_fix.py sdrtrunk_baseband.wav

    # Or generate synthetic test signal
    python test_differential_demod_fix.py --synthetic
"""

import argparse
import importlib.util
import numpy as np
import sys
import wave
from pathlib import Path

# Direct file imports to avoid FastAPI dependency in __init__.py
backend_path = Path(__file__).parent.parent

def import_module_from_file(name: str, path: Path):
    """Import a module directly from file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Import C4FM demodulator directly
c4fm_module = import_module_from_file(
    'c4fm',
    backend_path / 'wavecapsdr' / 'dsp' / 'p25' / 'c4fm.py'
)
C4FMDemodulator = c4fm_module.C4FMDemodulator

# Try to import decoder
HAS_DECODER = False
try:
    p25_module = import_module_from_file(
        'p25',
        backend_path / 'wavecapsdr' / 'decoders' / 'p25.py'
    )
    P25Decoder = p25_module.P25Decoder
    HAS_DECODER = True
except Exception as e:
    print(f"Warning: Could not import P25Decoder: {e}")


def load_sdrtrunk_wav(path: str) -> tuple[int, np.ndarray]:
    """Load sdrtrunk baseband WAV (stereo 16-bit: I=left, Q=right)."""
    with wave.open(path, 'rb') as wf:
        rate = wf.getframerate()
        n_frames = wf.getnframes()
        n_channels = wf.getnchannels()

        if n_channels != 2:
            raise ValueError(f"Expected stereo WAV (I/Q), got {n_channels} channels")

        raw = wf.readframes(n_frames)
        samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
        iq = (samples[:, 0] + 1j * samples[:, 1]).astype(np.complex64) / 32768.0

    return rate, iq


def generate_synthetic_p25(sample_rate: int = 48000, duration_sec: float = 1.0) -> np.ndarray:
    """Generate synthetic P25 C4FM signal with frame sync pattern.

    This creates a test signal with known frame sync patterns to verify
    the demodulator can detect them.
    """
    symbol_rate = 4800
    samples_per_symbol = sample_rate / symbol_rate

    # P25 frame sync: 0x5575F5FF77FF (48 bits = 24 dibits)
    # Dibits: 01 01 01 01 01 11 01 01 11 11 01 01 11 11 11 11 01 11 01 11 11 11 11 11
    sync_dibits = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1,
                  1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    # Generate random payload dibits
    n_symbols = int(duration_sec * symbol_rate)
    dibits = np.random.randint(0, 4, n_symbols)

    # Insert frame sync every ~1800 dibits (900 symbols = one P25 frame period)
    frame_period = 900
    for i in range(0, n_symbols, frame_period):
        if i + len(sync_dibits) < n_symbols:
            dibits[i:i+len(sync_dibits)] = sync_dibits[:len(sync_dibits)]

    # Map dibits to C4FM symbols: 0->+1, 1->+3, 2->-1, 3->-3
    dibit_to_symbol = {0: 1.0, 1: 3.0, 2: -1.0, 3: -3.0}
    symbols = np.array([dibit_to_symbol[d] for d in dibits], dtype=np.float32)

    # Upsample with RRC pulse shaping
    n_samples = int(len(symbols) * samples_per_symbol) + 100
    upsampled = np.zeros(n_samples, dtype=np.float32)

    for i, sym in enumerate(symbols):
        idx = int(i * samples_per_symbol)
        if idx < len(upsampled):
            upsampled[idx] = sym

    # RRC filter
    from scipy import signal
    span = 8
    n_taps = span * int(samples_per_symbol) + 1
    rrc = signal.firwin(n_taps, 0.5 / samples_per_symbol, window='hamming')
    filtered = signal.lfilter(rrc, 1.0, upsampled)

    # FM modulate (C4FM: deviation = symbol * deviation_hz)
    deviation_hz = 1800  # ~1800 Hz deviation for symbol ±3
    phase = np.cumsum(filtered * deviation_hz * 2 * np.pi / sample_rate)
    iq = np.exp(1j * phase).astype(np.complex64)

    # Add some noise
    noise = (np.random.randn(len(iq)) + 1j * np.random.randn(len(iq))) * 0.05
    iq = (iq + noise).astype(np.complex64)

    return iq


def analyze_soft_symbols(soft: np.ndarray) -> dict:
    """Analyze soft symbol distribution."""
    if len(soft) == 0:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}

    return {
        'mean': float(np.mean(soft)),
        'std': float(np.std(soft)),
        'min': float(np.min(soft)),
        'max': float(np.max(soft)),
    }


def test_demodulator(iq: np.ndarray, sample_rate: int) -> dict:
    """Test C4FM demodulator on IQ data."""
    demod = C4FMDemodulator(sample_rate=sample_rate)
    dibits, soft = demod.demodulate(iq)

    return {
        'n_dibits': len(dibits),
        'soft_stats': analyze_soft_symbols(soft),
        'dibits': dibits,
        'soft': soft,
    }


def test_decoder(iq: np.ndarray, sample_rate: int) -> dict:
    """Test full P25 decoder on IQ data."""
    if not HAS_DECODER:
        return {
            'n_frames': 0,
            'frame_types': {},
            'frames': [],
            'error': 'P25Decoder not available'
        }

    decoder = P25Decoder(sample_rate=sample_rate)
    frames = decoder.process_iq(iq)

    frame_types = {}
    for f in frames:
        t = f.frame_type.name if hasattr(f.frame_type, 'name') else str(f.frame_type)
        frame_types[t] = frame_types.get(t, 0) + 1

    return {
        'n_frames': len(frames),
        'frame_types': frame_types,
        'frames': frames,
    }


def main():
    parser = argparse.ArgumentParser(description='Test differential demod fix')
    parser.add_argument('input', nargs='?', help='Input WAV file (sdrtrunk baseband)')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic test signal')
    parser.add_argument('--sample-rate', type=int, default=48000, help='Sample rate for synthetic')
    parser.add_argument('--duration', type=float, default=2.0, help='Duration for synthetic (sec)')
    args = parser.parse_args()

    print("=" * 60)
    print("P25 C4FM Differential Demod Fix Test")
    print("=" * 60)
    print()

    # Load or generate IQ data
    if args.synthetic:
        print(f"Generating synthetic P25 signal ({args.duration}s at {args.sample_rate} Hz)...")
        iq = generate_synthetic_p25(args.sample_rate, args.duration)
        sample_rate = args.sample_rate
        print(f"Generated {len(iq)} samples")
    elif args.input:
        print(f"Loading {args.input}...")
        sample_rate, iq = load_sdrtrunk_wav(args.input)
        print(f"Loaded {len(iq)} samples at {sample_rate} Hz")
    else:
        print("Error: Provide input WAV file or use --synthetic")
        return 1

    print()

    # Test demodulator
    print("Testing C4FM demodulator...")
    print("-" * 40)
    demod_result = test_demodulator(iq, sample_rate)
    print(f"  Dibits recovered: {demod_result['n_dibits']}")
    stats = demod_result['soft_stats']
    print(f"  Soft symbol stats:")
    print(f"    mean: {stats['mean']:.3f}")
    print(f"    std:  {stats['std']:.3f}")
    print(f"    min:  {stats['min']:.3f}")
    print(f"    max:  {stats['max']:.3f}")

    # Expected: std should be around 1.4-2.0 for properly normalized ±1,±3 symbols
    if stats['std'] > 0.5:
        print("  [OK] Soft symbols have reasonable variance")
    else:
        print("  [WARN] Soft symbols have low variance - may indicate demod issue")

    print()

    # Test full decoder
    print("Testing full P25 decoder...")
    print("-" * 40)
    decode_result = test_decoder(iq, sample_rate)
    print(f"  Frames decoded: {decode_result['n_frames']}")

    if decode_result['frame_types']:
        print("  Frame types:")
        for ft, count in sorted(decode_result['frame_types'].items()):
            print(f"    {ft}: {count}")

    print()

    # Summary
    print("=" * 60)
    print("RESULT")
    print("=" * 60)

    if decode_result['n_frames'] > 0:
        print("[PASS] P25 decoder successfully decoded frames!")
        print(f"       The differential demod fix is working.")
        return 0
    elif demod_result['n_dibits'] > 0:
        print("[PARTIAL] Demodulator produced dibits but no frames decoded.")
        print("          Frame sync or decoder may need additional fixes.")
        return 1
    else:
        print("[FAIL] No dibits or frames decoded.")
        print("       There may be additional issues.")
        return 2


if __name__ == '__main__':
    sys.exit(main())
