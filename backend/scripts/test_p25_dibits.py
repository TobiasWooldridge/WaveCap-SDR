#!/usr/bin/env python3
"""Test P25 C4FM demodulation and log dibits for comparison with SDRTrunk.

Usage:
    python scripts/test_p25_dibits.py /tmp/sagrn_cc_48k.wav --center-offset 0
"""

from __future__ import annotations

import argparse
import logging
import sys
import wave
from pathlib import Path

import numpy as np
from scipy.signal import decimate, firwin, lfilter

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator
from wavecapsdr.decoders.p25_frames import FRAME_SYNC_DIBITS, decode_nid

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_iq_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load IQ from stereo WAV file (I=left, Q=right)."""
    with wave.open(str(path), 'rb') as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()

        raw = wf.readframes(n_frames)

    if sample_width == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    if n_channels == 2:
        # Stereo: I=left, Q=right
        samples = samples.reshape(-1, 2)
        iq = samples[:, 0] + 1j * samples[:, 1]
    else:
        raise ValueError("Expected stereo WAV with I/Q")

    return iq, sample_rate


def freq_shift(iq: np.ndarray, offset_hz: float, sample_rate: int) -> np.ndarray:
    """Shift frequency by mixing with complex sinusoid."""
    t = np.arange(len(iq)) / sample_rate
    shift = np.exp(-2j * np.pi * offset_hz * t)
    return iq * shift.astype(np.complex64)


def find_sync_positions(dibits: np.ndarray, min_score: int = 20) -> list[int]:
    """Find P25 frame sync positions in dibit stream."""
    sync_pattern = FRAME_SYNC_DIBITS
    sync_len = len(sync_pattern)
    positions = []

    for i in range(len(dibits) - sync_len):
        # Count matching dibits
        matches = np.sum(dibits[i:i+sync_len] == sync_pattern)
        if matches >= min_score:
            positions.append((i, matches))

    return positions


def main():
    parser = argparse.ArgumentParser(description="Test P25 C4FM demodulation")
    parser.add_argument("input", type=Path, help="Input IQ WAV file")
    parser.add_argument("--center-offset", type=float, default=0,
                       help="Frequency offset from center in Hz")
    parser.add_argument("--target-rate", type=int, default=48000,
                       help="Target sample rate for C4FM (default: 48000)")
    parser.add_argument("--log-dibits", type=Path, default=None,
                       help="Output file for dibit log")
    args = parser.parse_args()

    # Load IQ
    logger.info(f"Loading {args.input}")
    iq, sample_rate = load_iq_wav(args.input)
    logger.info(f"Loaded {len(iq)} samples at {sample_rate} Hz ({len(iq)/sample_rate:.2f}s)")

    # Apply frequency shift if needed
    if args.center_offset != 0:
        logger.info(f"Applying frequency shift: {args.center_offset} Hz")
        iq = freq_shift(iq, args.center_offset, sample_rate)

    # Decimate to target rate
    if sample_rate > args.target_rate:
        # Low-pass filter first
        decimation = sample_rate // args.target_rate
        if sample_rate % args.target_rate != 0:
            logger.warning(f"Sample rate {sample_rate} not evenly divisible by {args.target_rate}")
            decimation = round(sample_rate / args.target_rate)

        # Design anti-aliasing filter
        cutoff = args.target_rate * 0.4 / sample_rate  # Nyquist-relative
        taps = firwin(101, cutoff)

        # Apply filter
        iq_filtered = lfilter(taps, 1.0, iq)

        # Decimate
        actual_rate = sample_rate // decimation
        iq_decimated = iq_filtered[::decimation]
        logger.info(f"Decimated {decimation}x: {len(iq_decimated)} samples at {actual_rate} Hz")
    else:
        iq_decimated = iq
        actual_rate = sample_rate

    # Create C4FM demodulator
    logger.info(f"Creating C4FM demodulator at {actual_rate} Hz")
    demod = C4FMDemodulator(sample_rate=actual_rate)

    # Demodulate
    logger.info("Demodulating...")
    dibits, soft = demod.demodulate(iq_decimated)
    logger.info(f"Got {len(dibits)} dibits ({len(dibits)/4800:.2f}s at 4800 baud)")

    # Find sync patterns
    logger.info("Searching for P25 sync patterns...")
    syncs = find_sync_positions(dibits, min_score=20)
    logger.info(f"Found {len(syncs)} potential sync positions")

    # Analyze frames at sync positions
    for i, (pos, score) in enumerate(syncs[:20]):  # First 20 syncs
        if pos + 57 >= len(dibits):
            continue

        # Extract NID (33 dibits after 24-dibit sync)
        nid_dibits = dibits[pos + 24:pos + 57]

        # Try to decode NID
        nid = decode_nid(nid_dibits, skip_status_at_10=True)

        if nid:
            logger.info(f"Sync[{i}] @ {pos}: score={score}/24, NAC=0x{nid.nac:03X}, "
                       f"DUID={nid.duid.name}, errors={nid.errors}")

            # Log first 32 dibits after sync for debugging
            frame_dibits = dibits[pos:pos+56]
            dibit_str = ' '.join(str(d) for d in frame_dibits[:32])
            logger.info(f"  Dibits[0:32]: {dibit_str}")
        else:
            logger.info(f"Sync[{i}] @ {pos}: score={score}/24, NID decode failed")

    # Log all dibits to file if requested
    if args.log_dibits:
        with open(args.log_dibits, 'w') as f:
            for i, d in enumerate(dibits):
                f.write(f"{i},{d}\n")
        logger.info(f"Wrote {len(dibits)} dibits to {args.log_dibits}")

    # Print dibit distribution
    counts = [np.sum(dibits == d) for d in range(4)]
    total = len(dibits)
    logger.info(f"Dibit distribution: 0={counts[0]/total:.1%} 1={counts[1]/total:.1%} "
               f"2={counts[2]/total:.1%} 3={counts[3]/total:.1%}")


if __name__ == "__main__":
    main()
