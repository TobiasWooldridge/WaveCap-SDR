#!/usr/bin/env python3
"""Debug TSBK decoding by capturing dibits and analyzing the decode chain.

This script connects to the WaveCap-SDR trunking system and collects
diagnostic data about TSBK decoding failures.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import requests

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wavecapsdr.decoders.p25_frames import (
    DUID,
    decode_nid,
    decode_tsdu,
    extract_tsbk_blocks,
    deinterleave_data,
    dibits_to_bits,
    remove_status_symbols_with_offset,
)
from wavecapsdr.dsp.fec.trellis import trellis_decode, trellis_encode
from wavecapsdr.dsp.fec.bch import bch_decode_nid

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce noise from other modules
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)


def analyze_tsbk_block(dibits: np.ndarray, block_idx: int = 0) -> dict:
    """Analyze a single TSBK block (98 dibits) and report what's happening."""

    result = {
        "input_dibits": len(dibits),
        "steps": [],
        "success": False,
    }

    if len(dibits) < 98:
        result["error"] = f"Not enough dibits: {len(dibits)} < 98"
        return result

    # Take first 98 dibits
    tsbk_dibits = dibits[:98]
    result["steps"].append(f"Step 1: Input 98 dibits: {list(tsbk_dibits[:20])}...")

    # Convert to bits for deinterleaving
    interleaved_bits = dibits_to_bits(tsbk_dibits)
    result["steps"].append(f"Step 2: Converted to {len(interleaved_bits)} bits")

    # Deinterleave
    deinterleaved_bits = deinterleave_data(interleaved_bits)
    result["steps"].append(f"Step 3: Deinterleaved {len(deinterleaved_bits)} bits")

    # Convert back to dibits for trellis decoder
    trellis_dibits = np.zeros(98, dtype=np.uint8)
    for i in range(98):
        trellis_dibits[i] = (deinterleaved_bits[i*2] << 1) | deinterleaved_bits[i*2 + 1]

    result["steps"].append(f"Step 4: Trellis input dibits: {list(trellis_dibits[:20])}...")

    # Try trellis decode with all 4 phase rotations
    best_error = float('inf')
    best_rotation = 0
    best_decoded = None

    for xor_mask in [0, 1, 2, 3]:
        rotated = (trellis_dibits ^ xor_mask).astype(np.uint8) if xor_mask != 0 else trellis_dibits
        decoded, error = trellis_decode(rotated, debug=False)

        result["steps"].append(f"Step 5.{xor_mask}: XOR mask {xor_mask} -> error_metric={error}")

        if error < best_error:
            best_error = error
            best_rotation = xor_mask
            best_decoded = decoded

    result["best_rotation"] = best_rotation
    result["best_error_metric"] = best_error

    if best_decoded is None or len(best_decoded) < 48:
        result["error"] = f"Trellis decode failed, got {len(best_decoded) if best_decoded is not None else 0} dibits"
        return result

    result["steps"].append(f"Step 6: Best rotation={best_rotation}, error={best_error}, decoded {len(best_decoded)} dibits")

    # Convert to bits for CRC check
    decoded_bits = np.zeros(96, dtype=np.uint8)
    for i in range(48):
        if i < len(best_decoded):
            decoded_bits[i*2] = (best_decoded[i] >> 1) & 1
            decoded_bits[i*2 + 1] = best_decoded[i] & 1

    # Parse TSBK structure
    last_block = bool(decoded_bits[0])
    protect = bool(decoded_bits[1])
    opcode = int(np.packbits(decoded_bits[2:8])[0] >> 2)
    mfid = int(np.packbits(decoded_bits[8:16])[0])

    result["parsed"] = {
        "last_block": last_block,
        "protect": protect,
        "opcode": f"0x{opcode:02X}",
        "mfid": f"0x{mfid:02X}",
    }

    # Calculate CRC
    from wavecapsdr.decoders.p25_frames import crc16_ccitt_p25
    crc_valid, crc_errors = crc16_ccitt_p25(decoded_bits)

    result["crc_valid"] = crc_valid
    result["crc_errors"] = crc_errors
    result["success"] = crc_valid

    result["steps"].append(f"Step 7: CRC valid={crc_valid}, errors={crc_errors}")

    return result


def poll_trunking_stats(base_url: str = "http://localhost:8087") -> dict:
    """Poll trunking system stats."""
    try:
        resp = requests.get(f"{base_url}/api/v1/trunking/systems", timeout=5)
        resp.raise_for_status()
        systems = resp.json()
        if systems:
            return systems[0]
    except Exception as e:
        logger.error(f"Failed to poll stats: {e}")
    return {}


def main():
    """Main debug loop."""
    logger.info("=" * 60)
    logger.info("TSBK Debug Script Starting")
    logger.info("=" * 60)

    # Poll stats every 5 seconds and report
    prev_frames = 0
    prev_tsbk = 0
    prev_attempts = 0
    prev_crc_pass = 0

    for i in range(60):  # Run for 5 minutes
        stats = poll_trunking_stats()

        if not stats:
            logger.warning("No trunking system found")
            time.sleep(5)
            continue

        # Extract relevant stats
        state = stats.get("state", "unknown")
        cc_state = stats.get("controlChannelState", "unknown")
        cc_freq = stats.get("controlChannelFreqHz", 0)

        inner_stats = stats.get("stats", {})
        control_monitor = inner_stats.get("control_monitor", {})

        sync_state = control_monitor.get("sync_state", "unknown")
        frames = control_monitor.get("frames_decoded", 0)
        tsbk = control_monitor.get("tsbk_decoded", 0)
        sync_losses = control_monitor.get("sync_losses", 0)

        # New stats I added
        tsbk_attempts = control_monitor.get("tsbk_attempts", 0)
        tsbk_crc_pass = control_monitor.get("tsbk_crc_pass", 0)
        tsbk_crc_rate = control_monitor.get("tsbk_crc_pass_rate", 0)

        # Calculate deltas
        d_frames = frames - prev_frames
        d_tsbk = tsbk - prev_tsbk
        d_attempts = tsbk_attempts - prev_attempts
        d_crc_pass = tsbk_crc_pass - prev_crc_pass

        prev_frames = frames
        prev_tsbk = tsbk
        prev_attempts = tsbk_attempts
        prev_crc_pass = tsbk_crc_pass

        logger.info("-" * 40)
        logger.info(f"Poll {i+1}: state={state}, cc_state={cc_state}")
        logger.info(f"  CC freq: {cc_freq/1e6:.4f} MHz")
        logger.info(f"  Sync: {sync_state}, losses: {sync_losses}")
        logger.info(f"  Frames: {frames} (+{d_frames}), TSBK: {tsbk} (+{d_tsbk})")
        logger.info(f"  TSBK attempts: {tsbk_attempts} (+{d_attempts})")
        logger.info(f"  TSBK CRC pass: {tsbk_crc_pass} (+{d_crc_pass})")
        logger.info(f"  TSBK CRC rate: {tsbk_crc_rate}%")

        # If we have attempts but no passes, that's the problem
        if tsbk_attempts > 0 and tsbk_crc_pass == 0:
            logger.warning("*** TSBK CRC is failing 100% - need to debug FEC chain ***")

        time.sleep(5)


if __name__ == "__main__":
    main()
