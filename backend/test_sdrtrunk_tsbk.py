#!/usr/bin/env python3
"""Test TSBK decoding from SDRTrunk baseband recordings.

Compare TSBK decode success rate between recording and live system.
"""

import logging
import numpy as np
import scipy.io.wavfile as wav

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator
from wavecapsdr.decoders.p25_frames import decode_nid, decode_tsdu, DUID

# Frame sync pattern
FRAME_SYNC_DIBITS = np.array([
    1, 1, 1, 1, 1, 3, 1, 1, 3, 3, 1, 1, 3, 3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3
], dtype=np.uint8)


def find_frame_sync(dibits: np.ndarray, threshold: int = 3) -> list[int]:
    """Find frame sync positions."""
    positions = []
    sync_len = len(FRAME_SYNC_DIBITS)

    for i in range(len(dibits) - sync_len):
        matches = np.sum(dibits[i:i + sync_len] == FRAME_SYNC_DIBITS)
        if matches >= sync_len - threshold:
            positions.append(i)

    return positions


def fm_demodulate(iq: np.ndarray) -> np.ndarray:
    """FM demodulate IQ to discriminator audio."""
    phase = np.angle(iq)
    phase_unwrapped = np.unwrap(phase)
    disc_audio = np.diff(phase_unwrapped)
    return disc_audio.astype(np.float32)


def test_tsbk_decoding(filename: str) -> None:
    """Test TSBK decoding from baseband recording."""

    print(f"\nLoading {filename}...")
    sample_rate, audio = wav.read(filename)
    print(f"Duration: {len(audio)/sample_rate:.1f}s")

    if audio.ndim != 2:
        print("Error: Expected 2-channel IQ")
        return

    # Convert to complex IQ
    iq = (audio[:, 0] + 1j * audio[:, 1]).astype(np.complex64) / 32768.0

    # FM demodulate
    disc_audio = fm_demodulate(iq)

    # Create C4FM demodulator
    demod = C4FMDemodulator(sample_rate=sample_rate)

    # Process all at once to accumulate dibits
    print("Demodulating...")
    all_dibits, _ = demod.demodulate_discriminator(disc_audio)
    print(f"Total dibits: {len(all_dibits)}")

    # Find all frame syncs
    print("Finding frame syncs...")
    sync_positions = find_frame_sync(all_dibits)
    print(f"Found {len(sync_positions)} potential syncs")

    # Filter to unique syncs (at least 100 dibits apart)
    unique_syncs = []
    last_pos = -100
    for pos in sync_positions:
        if pos > last_pos + 100:
            unique_syncs.append(pos)
            last_pos = pos

    print(f"Unique syncs: {len(unique_syncs)}")

    # Try to decode each frame
    tsbk_attempts = 0
    tsbk_valid = 0
    nid_valid = 0

    for sync_pos in unique_syncs:
        # Need entire frame: sync(24) + NID(33) + TSBK data with status symbols
        # TSDU frame can be ~360 dibits total
        frame_end = sync_pos + 360
        if frame_end > len(all_dibits):
            continue

        # Extract the full frame including sync
        frame_dibits = all_dibits[sync_pos:frame_end]

        # Use decode_tsdu which handles status symbol removal and NID decoding
        try:
            tsdu = decode_tsdu(frame_dibits)
            if tsdu is None:
                continue

            nid_valid += 1

            # Only count if we got TSBK blocks
            if tsdu.tsbk_blocks:
                tsbk_attempts += 1
                for block in tsdu.tsbk_blocks:
                    if block.crc_valid:
                        tsbk_valid += 1
                        print(f"  Valid TSBK: opcode=0x{block.opcode:02x} mfid=0x{block.mfid:02x}")
        except Exception as e:
            logger.debug(f"TSDU decode error: {e}")

    print(f"\n=== TSBK Decoding Results ===")
    print(f"NID valid: {nid_valid}")
    print(f"TSBK frame attempts: {tsbk_attempts}")
    print(f"TSBK blocks with valid CRC: {tsbk_valid}")
    if tsbk_attempts > 0:
        rate = tsbk_valid / (tsbk_attempts * 3) * 100  # 3 blocks per frame
        print(f"TSBK success rate: {rate:.1f}%")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_tsbk_decoding(sys.argv[1])
    else:
        filename = "/Users/thw/SDRTrunk/recordings/20251226_091126_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav"
        test_tsbk_decoding(filename)
