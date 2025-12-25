#!/usr/bin/env python3
"""Test C4FM vs CQPSK demodulation on live SA-GRN control channel.

Captures IQ from SDR and tests both demodulators to determine which works better.
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce noise
logging.getLogger('urllib3').setLevel(logging.WARNING)


def test_demodulators_on_iq(iq_samples: np.ndarray, sample_rate: int = 48000) -> dict:
    """Test both demodulators on the same IQ samples.

    Args:
        iq_samples: Complex IQ samples at specified sample rate
        sample_rate: Sample rate in Hz

    Returns:
        Dict with results from both demodulators
    """
    from wavecapsdr.decoders.p25 import C4FMDemodulator, CQPSKDemodulator

    results = {
        "sample_rate": sample_rate,
        "num_samples": len(iq_samples),
    }

    # Test C4FM
    logger.info("Testing C4FM demodulator...")
    c4fm = C4FMDemodulator(sample_rate=sample_rate, symbol_rate=4800)
    c4fm_dibits = c4fm.demodulate(iq_samples)
    results["c4fm"] = {
        "num_dibits": len(c4fm_dibits),
        "dibit_counts": [int(np.sum(c4fm_dibits == i)) for i in range(4)],
    }

    # Test CQPSK
    logger.info("Testing CQPSK demodulator...")
    cqpsk = CQPSKDemodulator(sample_rate=sample_rate, symbol_rate=4800)
    cqpsk_dibits = cqpsk.demodulate(iq_samples)
    results["cqpsk"] = {
        "num_dibits": len(cqpsk_dibits),
        "dibit_counts": [int(np.sum(cqpsk_dibits == i)) for i in range(4)],
    }

    # Try to find frame sync in both
    from wavecapsdr.decoders.p25 import P25FrameSync
    sync = P25FrameSync()

    # Check C4FM dibits for sync
    c4fm_syncs = []
    for i in range(len(c4fm_dibits) - 24):
        score = sync._soft_correlation(c4fm_dibits[i:i+24])
        if score >= 60:
            c4fm_syncs.append((i, score))
    results["c4fm"]["sync_positions"] = c4fm_syncs[:5]

    # Check CQPSK dibits for sync
    cqpsk_syncs = []
    for i in range(len(cqpsk_dibits) - 24):
        score = sync._soft_correlation(cqpsk_dibits[i:i+24])
        if score >= 60:
            cqpsk_syncs.append((i, score))
    results["cqpsk"]["sync_positions"] = cqpsk_syncs[:5]

    return results


def capture_iq_from_api(freq_hz: float, duration_sec: float = 2.0) -> np.ndarray | None:
    """Capture IQ samples from WaveCap-SDR API.

    This requires an active capture with the control channel.
    """
    import requests
    import websocket
    import struct

    # Get captures
    try:
        resp = requests.get("http://localhost:8087/api/v1/captures", timeout=5)
        resp.raise_for_status()
        captures = resp.json()
    except Exception as e:
        logger.error(f"Failed to get captures: {e}")
        return None

    if not captures:
        logger.error("No captures running")
        return None

    # Find capture that covers our frequency
    for cap in captures:
        center = cap.get("centerHz", 0)
        sr = cap.get("sampleRate", 0)
        half_bw = sr / 2
        if center - half_bw <= freq_hz <= center + half_bw:
            logger.info(f"Found capture {cap['id']} covering {freq_hz/1e6:.3f} MHz")

            # Connect to IQ stream
            ws_url = f"ws://localhost:8087/api/v1/stream/captures/{cap['id']}/iq"
            try:
                ws = websocket.create_connection(ws_url, timeout=10)
                logger.info(f"Connected to IQ stream: {ws_url}")

                # Collect IQ samples for duration
                iq_data = []
                start_time = time.time()
                while time.time() - start_time < duration_sec:
                    data = ws.recv()
                    if isinstance(data, bytes):
                        # IQ16 format: int16 I, int16 Q interleaved
                        samples = np.frombuffer(data, dtype=np.int16)
                        samples_f = samples.astype(np.float32) / 32768.0
                        iq = samples_f[0::2] + 1j * samples_f[1::2]
                        iq_data.append(iq)

                ws.close()

                if iq_data:
                    all_iq = np.concatenate(iq_data)
                    logger.info(f"Captured {len(all_iq)} IQ samples")
                    return all_iq

            except Exception as e:
                logger.error(f"Failed to capture IQ: {e}")
                return None

    logger.error(f"No capture covers frequency {freq_hz/1e6:.3f} MHz")
    return None


def main():
    """Run live demodulator comparison."""
    import argparse

    parser = argparse.ArgumentParser(description="Test P25 demodulators on live IQ")
    parser.add_argument("--freq", type=float, default=413.45e6, help="Control channel frequency (Hz)")
    parser.add_argument("--duration", type=float, default=2.0, help="Capture duration (seconds)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("P25 Demodulator Comparison Test")
    logger.info("=" * 60)
    logger.info(f"Control channel: {args.freq/1e6:.4f} MHz")

    # Capture IQ
    iq = capture_iq_from_api(args.freq, args.duration)
    if iq is None:
        logger.error("Failed to capture IQ - make sure trunking system is running")
        return 1

    # The IQ is at the capture's sample rate (6MHz), need to decimate
    # The trunking system decimates to 48kHz for control channel
    # Let's apply the same processing
    capture_rate = 6000000
    target_rate = 48000
    decim = capture_rate // target_rate  # 125

    # Frequency shift to control channel offset
    # Assuming capture center is 414.5 MHz, control at 413.45 MHz
    center_hz = 414.5e6
    offset_hz = args.freq - center_hz
    logger.info(f"Frequency offset: {offset_hz/1e3:.1f} kHz")

    # Apply frequency shift
    t = np.arange(len(iq)) / capture_rate
    shift = np.exp(-2j * np.pi * offset_hz * t).astype(np.complex64)
    iq_shifted = iq * shift

    # Decimate
    from scipy.signal import decimate as scipy_decimate
    iq_decim = scipy_decimate(iq_shifted, decim, ftype='fir', zero_phase=False)
    logger.info(f"After decimation: {len(iq_decim)} samples at {target_rate} Hz")

    # Test both demodulators
    results = test_demodulators_on_iq(iq_decim, sample_rate=target_rate)

    # Print results
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)

    for demod in ["c4fm", "cqpsk"]:
        r = results[demod]
        logger.info(f"\n{demod.upper()}:")
        logger.info(f"  Dibits: {r['num_dibits']}")
        logger.info(f"  Dibit distribution: {r['dibit_counts']}")
        logger.info(f"  Frame syncs found: {len(r['sync_positions'])}")
        if r['sync_positions']:
            logger.info(f"  First syncs: {r['sync_positions'][:3]}")

    # Determine which is better
    c4fm_syncs = len(results["c4fm"]["sync_positions"])
    cqpsk_syncs = len(results["cqpsk"]["sync_positions"])

    logger.info("")
    if c4fm_syncs > cqpsk_syncs:
        logger.info(f">>> C4FM found more syncs ({c4fm_syncs} vs {cqpsk_syncs})")
    elif cqpsk_syncs > c4fm_syncs:
        logger.info(f">>> CQPSK found more syncs ({cqpsk_syncs} vs {c4fm_syncs})")
    else:
        logger.info(f">>> Both found same number of syncs ({c4fm_syncs})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
