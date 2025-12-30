#!/usr/bin/env python3
"""Debug raw buffer values at NID positions."""

import sys
import wave
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, '/Users/thw/Projects/WaveCap-SDR/backend')


def load_baseband(filepath: str, max_samples: int = None) -> tuple[np.ndarray, int]:
    with wave.open(filepath, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        if max_samples:
            n_frames = min(n_frames, max_samples)
        raw = wf.readframes(n_frames)
        data = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
        iq = (data[:, 0] + 1j * data[:, 1]).astype(np.complex64) / 32768.0
        return iq, sample_rate


def main(filepath: str):
    print("=" * 70)
    print("Buffer Value Debug at NID Positions")
    print("=" * 70)

    iq, sample_rate = load_baseband(filepath, int(2 * 50000))  # 2 seconds
    print(f"Sample rate: {sample_rate} Hz")

    from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator

    # Process with 100ms chunks
    demod = C4FMDemodulator(sample_rate=sample_rate)
    samples_per_symbol = demod.samples_per_symbol
    print(f"Samples per symbol: {samples_per_symbol:.3f}")

    chunk_samples = int(sample_rate * 0.1)  # 100ms

    all_dibits = []
    all_soft = []

    # Hook into the demodulator to capture buffer state
    buffer_snapshots = []

    original_demod = demod.demodulate
    def instrumented_demod(iq_chunk):
        result = original_demod(iq_chunk)
        # Capture buffer state after processing
        buffer_snapshots.append({
            'buffer': demod._buffer.copy(),
            'buffer_pointer': demod._buffer_pointer,
            'sync_count': demod._sync_count,
            'pll': demod._equalizer.pll,
            'gain': demod._equalizer.gain,
        })
        return result

    for start in range(0, len(iq), chunk_samples):
        end = min(start + chunk_samples, len(iq))
        chunk = iq[start:end]
        dibits, soft = instrumented_demod(chunk)
        all_dibits.extend(dibits)
        all_soft.extend(soft)

    print(f"\nTotal syncs: {demod._sync_count}")
    print(f"Total buffer snapshots: {len(buffer_snapshots)}")

    # Find a sync in the output
    dibits = np.array(all_dibits, dtype=np.uint8)
    soft = np.array(all_soft, dtype=np.float32)

    expected_sync = np.array([3, 3, 3, 3, 3, -3, 3, 3, -3, -3, 3, 3, -3, -3, -3, -3, 3, -3, 3, -3, -3, -3, -3, -3], dtype=np.float32)

    # Find first sync
    for i in range(len(soft) - 57):  # 24 sync + 33 NID
        window = soft[i:i+24]
        corr = np.sum(window * expected_sync)
        if corr > 200:
            print(f"\n{'='*70}")
            print(f"Found sync at dibit position {i}, correlation {corr:.1f}")
            print(f"{'='*70}")

            nid_start = i + 24

            # Show NID symbols from output
            print(f"\nNID symbols from demodulator output:")
            expected_nac = [0, 3, 3, 1, 3, 0]
            print(f"{'Pos':>4} {'Dibit':>6} {'Expect':>7} {'Soft':>8} {'ExpSoft':>8}")
            print("-" * 50)
            for j in range(6):
                pos = nid_start + j
                if pos < len(dibits):
                    exp_soft = {0: 1.0, 1: 3.0, 2: -1.0, 3: -3.0}[expected_nac[j]]
                    print(f"{j:>4} {dibits[pos]:>6} {expected_nac[j]:>7} {soft[pos]:>+8.2f} {exp_soft:>+8.1f}")

            # Now look at raw buffer values
            # We need to figure out which buffer snapshot contains this sync
            # This is tricky because dibits accumulate across chunks

            print(f"\nLast buffer snapshot state:")
            snap = buffer_snapshots[-1]
            print(f"  buffer_pointer: {snap['buffer_pointer']}")
            print(f"  pll: {snap['pll']:.4f}")
            print(f"  gain: {snap['gain']:.3f}")

            # Look at the actual raw buffer values
            # First, let's see what range of buffer is populated
            buffer = snap['buffer']
            nonzero = np.nonzero(buffer)[0]
            if len(nonzero) > 0:
                print(f"  buffer non-zero range: {nonzero[0]} to {nonzero[-1]}")

            # The sync was found by correlating dibits. 
            # The NID values should be at positions after the sync.
            # Let's trace what _resample_nid_jit would compute.

            print(f"\n" + "=" * 70)
            print("Manual _resample_nid_jit trace")
            print("=" * 70)

            # Find where sync would be in the BUFFER (not in the output dibits)
            # This requires correlating in the buffer too

            # Let's find a high-correlation region in the buffer
            # Sync pattern in radians (before normalization)
            # soft = rad * 4/pi, so rad = soft * pi/4
            sync_rad = expected_sync * (np.pi / 4)

            best_corr = -1000
            best_buf_pos = 0
            search_start = max(0, snap['buffer_pointer'] - int(30 * samples_per_symbol))
            search_end = min(len(buffer), snap['buffer_pointer'])

            for buf_pos in range(search_start, search_end):
                # Extract 24 symbols worth of samples at symbol centers
                corr_sum = 0.0
                valid = True
                for sym in range(24):
                    sample_idx = int(buf_pos + sym * samples_per_symbol + samples_per_symbol / 2)
                    if sample_idx >= len(buffer):
                        valid = False
                        break
                    corr_sum += buffer[sample_idx] * sync_rad[sym]
                if valid and corr_sum > best_corr:
                    best_corr = corr_sum
                    best_buf_pos = buf_pos

            print(f"Best sync in buffer at position {best_buf_pos}, correlation {best_corr:.1f}")

            # Now trace NID values at this position
            nid_buf_start = best_buf_pos + 24 * samples_per_symbol
            print(f"\nNID starts at buffer position {nid_buf_start:.1f}")
            print(f"\nRaw buffer values at NID symbol centers:")
            print(f"{'Sym':>4} {'SamplePos':>10} {'RawPhase':>10} {'+PLL':>10} {'*Gain':>10} {'SoftNorm':>10} {'Dibit':>6} {'Expect':>6}")
            print("-" * 80)

            pll = snap['pll']
            gain = snap['gain']
            boundary = np.pi / 2

            for sym in range(6):
                sample_pos = nid_buf_start + sym * samples_per_symbol + samples_per_symbol / 2
                idx = int(sample_pos)
                mu = sample_pos - idx
                if idx >= 0 and idx + 1 < len(buffer):
                    x1 = buffer[idx]
                    x2 = buffer[idx + 1]
                    interp = x1 + (x2 - x1) * mu
                else:
                    interp = buffer[max(0, min(idx, len(buffer) - 1))]

                corrected = (interp + pll) * gain
                soft_norm = corrected * (4 / np.pi)

                # Dibit decision (on corrected radians)
                if corrected >= boundary:
                    dibit = 1
                elif corrected >= 0:
                    dibit = 0
                elif corrected >= -boundary:
                    dibit = 2
                else:
                    dibit = 3

                print(f"{sym:>4} {sample_pos:>10.1f} {interp:>+10.4f} {interp+pll:>+10.4f} {corrected:>+10.4f} {soft_norm:>+10.2f} {dibit:>6} {expected_nac[sym]:>6}")

            break  # Only analyze first sync

    print("\n" + "=" * 70)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("/Users/thw/SDRTrunk/recordings/20251227_121743_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav")
