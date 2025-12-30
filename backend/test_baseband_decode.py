#!/usr/bin/env python3
"""Test WaveCap-SDR P25 decoder using SDRTrunk baseband recording."""

import sys
import wave
import numpy as np
import logging

# Enable detailed logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Add wavecapsdr to path
sys.path.insert(0, '/Users/thw/Projects/WaveCap-SDR/backend')

from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator
from wavecapsdr.decoders.p25_frames import (
    decode_tsdu, FRAME_SYNC_DIBITS, DUID
)
from wavecapsdr.trunking.control_channel import ControlChannelMonitor

def load_baseband_wav(filepath: str, max_samples: int = 0) -> np.ndarray:
    """Load SDRTrunk baseband recording (complex float32 in WAV)."""
    with wave.open(filepath, 'rb') as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()

        print(f"WAV: {n_channels} ch, {sample_width} bytes/sample, {sample_rate} Hz, {n_frames} frames")

        if max_samples > 0:
            n_frames = min(n_frames, max_samples)

        raw = wf.readframes(n_frames)

        # SDRTrunk baseband: 2 channels (I/Q)
        if n_channels == 2 and sample_width == 4:
            # Interleaved I/Q as float32
            data = np.frombuffer(raw, dtype=np.float32)
            i = data[0::2]
            q = data[1::2]
            return (i + 1j * q).astype(np.complex64), sample_rate
        elif n_channels == 2 and sample_width == 2:
            # Interleaved I/Q as int16 (normalized to float)
            data = np.frombuffer(raw, dtype=np.int16)
            i = data[0::2].astype(np.float32) / 32768.0
            q = data[1::2].astype(np.float32) / 32768.0
            return (i + 1j * q).astype(np.complex64), sample_rate
        else:
            raise ValueError(f"Unexpected format: {n_channels} ch, {sample_width} bytes")


def test_demodulator(recording_path: str, duration_sec: float = 5.0):
    """Test C4FM demodulator on recording."""
    print(f"\n=== Testing demodulator on: {recording_path} ===")

    # Load first N seconds of recording
    iq, sample_rate = load_baseband_wav(recording_path)
    max_samples = int(duration_sec * sample_rate)
    if len(iq) > max_samples:
        iq = iq[:max_samples]

    print(f"Processing {len(iq)} samples ({len(iq)/sample_rate:.2f} sec) at {sample_rate} Hz")

    # Create demodulator
    demod = C4FMDemodulator(sample_rate=sample_rate)

    # Demodulate
    dibits, soft = demod.demodulate(iq)
    print(f"Demodulated {len(dibits)} dibits")

    # Find sync patterns
    sync_count = 0
    sync_positions = []
    for i in range(len(dibits) - len(FRAME_SYNC_DIBITS)):
        if np.array_equal(dibits[i:i+len(FRAME_SYNC_DIBITS)], FRAME_SYNC_DIBITS):
            sync_positions.append(i)
            sync_count += 1
            if sync_count <= 5:
                print(f"  Found sync at dibit {i}")

    print(f"Found {sync_count} sync patterns")

    # Try to decode frames at sync positions
    tsdu_decoded = 0
    crc_passed = 0
    nac_counts = {}
    duid_counts = {}

    from wavecapsdr.decoders.p25_frames import decode_nid

    for pos in sync_positions[:100]:  # Try first 100 syncs
        # Need at least 360 dibits for a TSDU frame
        if pos + 360 > len(dibits):
            continue

        frame_dibits = dibits[pos:pos + 360]
        frame_soft = soft[pos:pos + 360] if soft is not None else None

        # Decode NID to see what frame types we're getting
        nid_dibits = frame_dibits[24:57]  # Skip sync, get NID (33 dibits)
        nid = decode_nid(nid_dibits, skip_status_at_10=True)
        if nid:
            nac_counts[nid.nac] = nac_counts.get(nid.nac, 0) + 1
            duid_counts[nid.duid.name if hasattr(nid.duid, 'name') else str(nid.duid)] = duid_counts.get(nid.duid.name if hasattr(nid.duid, 'name') else str(nid.duid), 0) + 1

            # Only decode TSDU frames (DUID=7)
            if nid.duid.value == 7:  # TSDU
                try:
                    tsdu = decode_tsdu(frame_dibits, frame_soft)
                    if tsdu and tsdu.tsbk_blocks:
                        tsdu_decoded += 1
                        print(f"  Pos {pos}: NAC=0x{tsdu.nid.nac:03x}, DUID={tsdu.nid.duid}, blocks={len(tsdu.tsbk_blocks)}")

                        for i, block in enumerate(tsdu.tsbk_blocks):
                            if block.crc_valid:
                                crc_passed += 1
                                print(f"    Block {i}: opcode=0x{block.opcode:02x} mfid=0x{block.mfid:02x} CRC=PASS")
                            else:
                                print(f"    Block {i}: opcode=0x{block.opcode:02x} mfid=0x{block.mfid:02x} CRC=FAIL")
                except Exception as e:
                    print(f"  Pos {pos}: Error - {e}")

    print(f"\nNAC distribution: {dict(sorted(nac_counts.items(), key=lambda x: -x[1])[:10])}")
    print(f"DUID distribution: {dict(sorted(duid_counts.items(), key=lambda x: -x[1]))}")

    print(f"\nResults: {tsdu_decoded} TSDU frames decoded, {crc_passed} TSBK blocks with valid CRC")
    return crc_passed > 0


if __name__ == "__main__":
    # Use a smaller recording for testing
    recording = "/Users/thw/SDRTrunk/recordings/20251227_121743_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav"

    success = test_demodulator(recording, duration_sec=30.0)
    sys.exit(0 if success else 1)
