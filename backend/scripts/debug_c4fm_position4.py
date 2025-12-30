#!/usr/bin/env python3
"""Trace exact behavior at NID position 4 to understand the systematic error.

Uses a modified C4FM demodulator that exposes internal state at the error position.
"""

import sys
import wave
import numpy as np
from scipy import signal

sys.path.insert(0, '/Users/thw/Projects/WaveCap-SDR/backend')

# Frame sync as dibit array (24 dibits)
FRAME_SYNC_DIBITS = np.array([
    1, 1, 1, 1, 1, 3, 1, 1, 3, 3, 1, 1, 3, 3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3
], dtype=np.uint8)

EXPECTED_NAC_DIBITS = [0, 3, 3, 1, 3, 0]


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
    print(f"\n{'='*70}")
    print("Position 4 Error Investigation")
    print(f"{'='*70}")

    iq, sample_rate = load_baseband(filepath, int(10 * 50000))
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Samples: {len(iq)}")

    # Import and modify C4FM demodulator
    from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator
    SYNC = FRAME_SYNC_DIBITS

    # Create demodulator
    demod = C4FMDemodulator(sample_rate=sample_rate)
    dibits, soft = demod.demodulate(iq)

    print(f"Total dibits: {len(dibits)}")
    print(f"Soft range: [{soft.min():.2f}, {soft.max():.2f}]")

    # Find syncs
    syncs = []
    for i in range(len(dibits) - 32):
        matches = sum(1 for j in range(24) if dibits[i+j] == SYNC[j])
        if matches == 24:
            syncs.append(i)

    print(f"Perfect syncs: {len(syncs)}")

    # Analyze position 4 for each sync
    print(f"\n{'='*70}")
    print("Position 4 Analysis Across All Syncs")
    print(f"{'='*70}")

    position_4_soft = []
    for sync_idx, sync_pos in enumerate(syncs[:20]):
        nid_4_pos = sync_pos + 24 + 4
        if nid_4_pos < len(soft):
            s = soft[nid_4_pos]
            d = dibits[nid_4_pos]
            position_4_soft.append(s)

            # Also get surrounding symbols for context
            context_soft = [soft[nid_4_pos + j] if nid_4_pos + j < len(soft) else np.nan
                           for j in range(-2, 3)]
            context_dibits = [dibits[nid_4_pos + j] if nid_4_pos + j < len(dibits) else -1
                             for j in range(-2, 3)]

            print(f"Sync {sync_idx:2d} @ {sync_pos:6d}: pos4 soft={s:+6.2f} dibit={d} "
                  f"| context: soft=[{', '.join(f'{x:+.1f}' for x in context_soft)}] "
                  f"dibits={context_dibits}")

    if position_4_soft:
        arr = np.array(position_4_soft)
        print(f"\nPosition 4 Statistics:")
        print(f"  Mean: {arr.mean():+.3f}")
        print(f"  Std:  {arr.std():.3f}")
        print(f"  Min:  {arr.min():+.3f}")
        print(f"  Max:  {arr.max():+.3f}")
        print(f"  Expected: -3.0 (for dibit 3)")
        print(f"  Error: {abs(arr.mean() - (-3.0)):.3f}")

    # Now let's look at the raw buffer values at position 4
    print(f"\n{'='*70}")
    print("Investigating Raw Phase Buffer at Position 4")
    print(f"{'='*70}")

    # Re-run with access to internal buffer
    # The C4FM uses _buffer internally - let's trace it
    demod2 = C4FMDemodulator(sample_rate=sample_rate)

    # Get the internal buffer after demodulation
    dibits2, soft2 = demod2.demodulate(iq)

    # Access internal state
    buffer = demod2._buffer
    equalizer_pll = demod2._equalizer.pll
    equalizer_gain = demod2._equalizer.gain

    print(f"Final equalizer state: PLL={equalizer_pll:+.4f}, Gain={equalizer_gain:.4f}")

    # Check if there's a pattern in the sync->NID transition
    print(f"\n{'='*70}")
    print("Sync->NID Transition Pattern")
    print(f"{'='*70}")

    # The transition is: sync ends with [...,3,3,3,3,3], NID starts with [0,3,3,1,3,0,...]
    # Position 4 is the 5th NID symbol, which should be dibit 3 (-3)
    # The sequence is: sync[23]=3, NID[0]=0, NID[1]=3, NID[2]=3, NID[3]=1, NID[4]=3

    print("\nExpected symbol sequence at transition:")
    print("  SYNC[19:24] = [3, 3, 3, 3, 3]  (all -3)")
    print("  NID[0:6]    = [0, 3, 3, 1, 3, 0] (NAC)")
    print("  NID[6:8]    = [0, 1] (DUID)")
    print("\nSymbol pattern: ...-3,-3,-3,-3,-3,+1,-3,-3,+3,-3,+1,+1,+3...")
    print("                                    ^           ^")
    print("                                   NID[0]      NID[4]=position 4")

    print("\nNote: Position 4 comes after: -3, -3, +1, -3, -3, +3")
    print("The transition +3 -> -3 at position 4 may be affected by ISI")

    # Let's compute ISI contribution
    print(f"\n{'='*70}")
    print("ISI Analysis at Position 4")
    print(f"{'='*70}")

    # RRC filter impulse response
    sps = sample_rate / 4800.0
    rrc = demod2._rrc_filter
    print(f"RRC filter length: {len(rrc)} taps")

    # The ISI at symbol k is influenced by symbols k-N to k+N
    # where N depends on filter length and samples per symbol
    filter_span = len(rrc) / sps
    print(f"Filter spans {filter_span:.1f} symbols")

    # Position 4's neighbors and their expected values:
    neighbors = [
        (-3, 1, +3),   # NID[1] = 3 -> +3
        (-2, 3, +3),   # NID[2] = 3 -> +3  (wait, expected is -3!)
        (-1, 1, +3),   # NID[3] = 1 -> +3
        (0, 3, -3),    # NID[4] = 3 -> -3 (POSITION 4!)
        (+1, 0, +1),   # NID[5] = 0 -> +1
    ]

    # Wait, I had the expected values wrong! Let me recheck:
    print("\nRe-checking expected values:")
    for i, exp_dibit in enumerate(EXPECTED_NAC_DIBITS):
        exp_soft = {0: 1.0, 1: 3.0, 2: -1.0, 3: -3.0}[exp_dibit]
        print(f"  NID[{i}] = dibit {exp_dibit} -> soft {exp_soft:+.0f}")

    # So the sequence of expected soft values for NID[0:6] is:
    # [+1, -3, -3, +3, -3, +1]
    # Position 4 (expected -3) comes after position 3 (expected +3)
    # This is a +3 -> -3 transition (full swing from outer+ to outer-)

    print("\nActual soft values at NID positions:")
    if syncs:
        sync_pos = syncs[0]
        for i in range(8):
            pos = sync_pos + 24 + i
            if pos < len(soft):
                exp_dibit = (EXPECTED_NAC_DIBITS + [0, 1])[i]
                exp_soft = {0: 1.0, 1: 3.0, 2: -1.0, 3: -3.0}[exp_dibit]
                print(f"  NID[{i}] actual={soft[pos]:+6.2f}, expected={exp_soft:+.0f}, "
                      f"dibit={dibits[pos]} (exp {exp_dibit})")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("/Users/thw/SDRTrunk/recordings/20251227_121743_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav")
