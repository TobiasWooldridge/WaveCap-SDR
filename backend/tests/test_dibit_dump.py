"""Dump raw dibits from CQPSK demodulator for debugging."""

import wave
import numpy as np
import logging

logging.basicConfig(level=logging.WARNING)

from wavecapsdr.decoders.p25 import CQPSKDemodulator, P25FrameSync


def load_iq_wav(filepath):
    with wave.open(filepath, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

        if sample_width == 3:  # 24-bit
            data_bytes = np.frombuffer(raw_data, dtype=np.uint8)
            data_bytes = data_bytes.reshape(-1, n_channels, 3)
            samples = np.zeros((n_frames, n_channels), dtype=np.int32)
            for ch in range(n_channels):
                b0 = data_bytes[:, ch, 0].astype(np.int32)
                b1 = data_bytes[:, ch, 1].astype(np.int32)
                b2 = data_bytes[:, ch, 2].astype(np.int32)
                raw = b0 | (b1 << 8) | (b2 << 16)
                samples[:, ch] = np.where(raw >= 0x800000, raw - 0x1000000, raw)
            samples = samples.astype(np.float32) / 8388608.0
        else:
            samples = np.frombuffer(raw_data, dtype=np.int16).reshape(-1, n_channels)
            samples = samples.astype(np.float32) / 32768.0

        iq = samples[:, 0] + 1j * samples[:, 1]
        return iq, sample_rate


if __name__ == "__main__":
    iq, sample_rate = load_iq_wav('tests/fixtures/p25_samples/P25_CQPSK-CC_IF.wav')

    # Demodulate a chunk
    demod = CQPSKDemodulator(sample_rate=sample_rate, symbol_rate=4800)
    frame_sync = P25FrameSync()

    # Process first 5 seconds to get some frames
    chunk = iq[:sample_rate * 5]
    dibits = demod.demodulate(chunk)

    print(f'Demodulated {len(dibits)} dibits from {len(chunk)} samples')
    print(f'Dibit distribution: d0={np.sum(dibits==0)}, d1={np.sum(dibits==1)}, d2={np.sum(dibits==2)}, d3={np.sum(dibits==3)}')

    # Look for frame sync
    sync_pos, frame_type, nac, duid = frame_sync.find_sync(dibits)
    if sync_pos is not None:
        print(f'Found sync at position {sync_pos}, frame type: {frame_type}, NAC={nac:03X}')

        # Show raw dibits around sync
        print(f'Dibits around sync (first 60): {list(dibits[sync_pos:sync_pos+60])}')

        # Check if sync pattern matches
        expected_sync = [1, 1, 1, 1, 1, 3, 1, 1, 3, 3, 1, 1, 3, 3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3]
        actual_sync = list(dibits[sync_pos:sync_pos+24])
        print(f'Expected sync: {expected_sync}')
        print(f'Actual sync:   {actual_sync}')

        # Compare
        errors = sum(1 for a, b in zip(expected_sync, actual_sync) if a != b)
        print(f'Sync errors: {errors} / 24')

        # Try different dibit mappings
        print("\n--- Testing different dibit mappings ---")

        # Mapping 1: invert all bits (XOR 3)
        inverted = dibits ^ 3
        inv_sync = list(inverted[sync_pos:sync_pos+24])
        inv_errors = sum(1 for a, b in zip(expected_sync, inv_sync) if a != b)
        print(f'XOR 3 (full invert): {inv_sync}, errors={inv_errors}')

        # Mapping 2: swap 0/2 (XOR 2)
        swapped = dibits ^ 2
        swap_sync = list(swapped[sync_pos:sync_pos+24])
        swap_errors = sum(1 for a, b in zip(expected_sync, swap_sync) if a != b)
        print(f'XOR 2 (swap 0<->2):  {swap_sync}, errors={swap_errors}')

        # Mapping 3: bit reverse within dibit (0->0, 1->2, 2->1, 3->3)
        bit_rev = np.array([0, 2, 1, 3], dtype=np.uint8)[dibits]
        br_sync = list(bit_rev[sync_pos:sync_pos+24])
        br_errors = sum(1 for a, b in zip(expected_sync, br_sync) if a != b)
        print(f'Bit reverse:         {br_sync}, errors={br_errors}')

        # Mapping 4: XOR 1 (swap 0<->1, 2<->3)
        xor1 = dibits ^ 1
        xor1_sync = list(xor1[sync_pos:sync_pos+24])
        xor1_errors = sum(1 for a, b in zip(expected_sync, xor1_sync) if a != b)
        print(f'XOR 1 (swap 0<->1):  {xor1_sync}, errors={xor1_errors}')

        # Mapping 5: Rotate +1 (0->1, 1->2, 2->3, 3->0) - 90 degree phase offset
        rot1 = (dibits + 1) % 4
        rot1_sync = list(rot1[sync_pos:sync_pos+24])
        rot1_errors = sum(1 for a, b in zip(expected_sync, rot1_sync) if a != b)
        print(f'Rotate +1 (+90°):    {rot1_sync}, errors={rot1_errors}')

        # Mapping 6: Rotate -1 (0->3, 1->0, 2->1, 3->2) - -90 degree phase offset
        rot3 = (dibits + 3) % 4
        rot3_sync = list(rot3[sync_pos:sync_pos+24])
        rot3_errors = sum(1 for a, b in zip(expected_sync, rot3_sync) if a != b)
        print(f'Rotate -1 (-90°):    {rot3_sync}, errors={rot3_errors}')

        # Mapping 7: Rotate +2 (0->2, 1->3, 2->0, 3->1) - 180 degree phase offset
        rot2 = (dibits + 2) % 4
        rot2_sync = list(rot2[sync_pos:sync_pos+24])
        rot2_errors = sum(1 for a, b in zip(expected_sync, rot2_sync) if a != b)
        print(f'Rotate +2 (180°):    {rot2_sync}, errors={rot2_errors}')

        # Mapping 8: Gray code transform (0->0, 1->1, 2->3, 3->2)
        gray = np.array([0, 1, 3, 2], dtype=np.uint8)[dibits]
        gray_sync = list(gray[sync_pos:sync_pos+24])
        gray_errors = sum(1 for a, b in zip(expected_sync, gray_sync) if a != b)
        print(f'Gray code:           {gray_sync}, errors={gray_errors}')

        # Mapping 9: Inverse Gray code transform (0->0, 1->1, 2->3, 3->2) on original
        inv_gray = np.array([0, 1, 3, 2], dtype=np.uint8)[dibits]
        inv_gray_sync = list(inv_gray[sync_pos:sync_pos+24])
        inv_gray_errors = sum(1 for a, b in zip(expected_sync, inv_gray_sync) if a != b)
        print(f'Inverse Gray:        {inv_gray_sync}, errors={inv_gray_errors}')

    else:
        print('No sync found')
