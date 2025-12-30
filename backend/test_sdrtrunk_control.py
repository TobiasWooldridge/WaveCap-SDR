#!/usr/bin/env python3
"""Test P25 control channel decoding with SDRTrunk baseband recordings.

This validates the C4FM demodulation and frame sync detection pipeline.
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
from wavecapsdr.decoders.p25_frames import decode_nid, DUID

# Frame sync as dibit array (24 dibits)
FRAME_SYNC_DIBITS = np.array([
    1, 1, 1, 1, 1, 3, 1, 1, 3, 3, 1, 1, 3, 3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3
], dtype=np.uint8)


def find_frame_sync(dibits: np.ndarray, threshold: int = 3) -> list[int]:
    """Find frame sync pattern positions in dibits.

    Args:
        dibits: Input dibit stream
        threshold: Max allowed differences (for error tolerance)

    Returns:
        List of sync start positions
    """
    positions = []
    sync_len = len(FRAME_SYNC_DIBITS)

    for i in range(len(dibits) - sync_len):
        # Count matching dibits
        matches = np.sum(dibits[i:i + sync_len] == FRAME_SYNC_DIBITS)
        if matches >= sync_len - threshold:
            positions.append(i)

    return positions


def fm_demodulate(iq: np.ndarray) -> np.ndarray:
    """FM demodulate IQ signal to discriminator audio."""
    phase = np.angle(iq)
    phase_unwrapped = np.unwrap(phase)
    disc_audio = np.diff(phase_unwrapped)
    return disc_audio.astype(np.float32)


def test_control_channel(filename: str) -> None:
    """Test P25 frame detection on control channel baseband."""

    print(f"\nLoading {filename}...")
    sample_rate, audio = wav.read(filename)
    print(f"Sample rate: {sample_rate}")
    print(f"Audio shape: {audio.shape}")
    print(f"Duration: {len(audio)/sample_rate:.1f}s")

    if audio.ndim != 2 or audio.shape[1] != 2:
        print("Error: Expected 2-channel IQ audio")
        return

    # Convert to complex IQ
    iq = (audio[:, 0] + 1j * audio[:, 1]).astype(np.complex64) / 32768.0
    print(f"IQ power: {np.mean(np.abs(iq)**2):.6f}")

    # FM demodulate
    disc_audio = fm_demodulate(iq)
    print(f"Discriminator RMS: {np.sqrt(np.mean(disc_audio**2)):.6f}")

    # Create C4FM demodulator at 50kHz (SDRTrunk's rate)
    demod = C4FMDemodulator(sample_rate=sample_rate)

    # Process in chunks
    chunk_size = sample_rate // 10  # 100ms chunks
    num_chunks = len(disc_audio) // chunk_size

    print(f"\nProcessing {num_chunks} chunks...")

    total_dibits = 0
    sync_count = 0
    nid_attempt_count = 0
    valid_nid_count = 0
    nacs_seen = set()
    duids_seen = {}

    dibit_buffer = np.array([], dtype=np.uint8)
    last_sync_pos = -100  # Avoid counting same sync multiple times

    for i in range(num_chunks):
        chunk = disc_audio[i * chunk_size : (i + 1) * chunk_size]

        # C4FM demodulate
        dibits, soft_symbols = demod.demodulate_discriminator(chunk)
        total_dibits += len(dibits)

        # Add to buffer
        dibit_buffer = np.concatenate([dibit_buffer, dibits])

        # Keep buffer manageable
        buffer_start = 0
        if len(dibit_buffer) > 10000:
            buffer_start = len(dibit_buffer) - 5000
            dibit_buffer = dibit_buffer[-5000:]

        # Find frame syncs
        sync_positions = find_frame_sync(dibit_buffer)

        for sync_pos in sync_positions:
            # Check if this is a new sync (not too close to last one)
            global_pos = buffer_start + sync_pos
            if global_pos < last_sync_pos + 24:
                continue
            last_sync_pos = global_pos

            sync_count += 1

            # NID starts after 24-dibit sync, need 33 dibits (includes status)
            nid_start = sync_pos + 24
            if nid_start + 33 > len(dibit_buffer):
                continue

            nid_dibits = dibit_buffer[nid_start:nid_start + 33]
            nid_attempt_count += 1

            nid = decode_nid(nid_dibits, skip_status_at_10=True)
            if nid is not None:
                nacs_seen.add(hex(nid.nac))
                duids_seen[nid.duid] = duids_seen.get(nid.duid, 0) + 1
                if nid.duid in [DUID.TDU, DUID.TSDU, DUID.LDU1, DUID.LDU2, DUID.HDU, DUID.TDULC]:
                    valid_nid_count += 1

        if i > 0 and i % 20 == 0:
            print(f"  Chunk {i}/{num_chunks}: dibits={total_dibits}, "
                  f"syncs={sync_count}, NIDs={nid_attempt_count} (valid={valid_nid_count})")

    print(f"\n=== Results ===")
    print(f"Total dibits: {total_dibits}")
    print(f"Frame syncs found: {sync_count}")
    print(f"NID decode attempts: {nid_attempt_count}")
    print(f"Valid NIDs: {valid_nid_count}")
    print(f"NACs seen: {sorted(nacs_seen)}")
    print(f"DUIDs: {duids_seen}")

    # Summary
    if sync_count > 0:
        print(f"\n✓ Frame sync detection working!")
        print(f"  Found {sync_count} frame syncs")
    else:
        print(f"\n✗ No frame syncs detected")

    if valid_nid_count > 0:
        print(f"\n✓ NID decoding working!")
        print(f"  Found {len(nacs_seen)} unique NACs")
        if '0x3dc' in [n.lower() for n in nacs_seen]:
            print(f"  ✓ SA-GRN NAC 0x3DC detected!")
    else:
        print(f"\n✗ No valid NIDs decoded")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_control_channel(sys.argv[1])
    else:
        # Use a medium-sized recording
        filename = "/Users/thw/SDRTrunk/recordings/20251226_091126_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav"
        test_control_channel(filename)
