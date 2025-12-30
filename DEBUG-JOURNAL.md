# P25 SA-GRN Debug Journal

## Overview
This journal documents the step-by-step comparison of WaveCap-SDR against SDRTrunk for P25 trunked audio decoding.

**System:** SA-GRN (South Australia Government Radio Network)
**Reference:** SDRTrunk (known working with same radio+antenna)
**Goal:** Identify where WaveCap-SDR diverges from SDRTrunk

---

## Session: 2024-12-30

### Round 1: Control Channel Verification

#### Step 1.1: Check Available Recordings

**Control Channel Recordings:** 30+ files available
- Most recent: `20251227_150621_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav` (464 MB)
- Medium size for testing: `20251227_121743_413075000_SA-GRN_Adelaide-Metro_Control-Channel_0_baseband.wav` (23 MB)

**Voice/Traffic Recordings:** NONE AVAILABLE
- Need to configure SDRTrunk to record traffic channel basebands
- Will proceed with control channel testing first

#### Step 2.1: Test TSBK Decoding on CC Recording

**Test:** `test_sdrtrunk_control.py` on 116-second CC recording

**Results:**
```
Sample rate: 50000
Duration: 116.1s
IQ power: 0.000013
Discriminator RMS: 0.980634

Total dibits: 556800
Frame syncs found: 10
NID decode attempts: 10
Valid NIDs: 10
NACs seen: ['0x3d0']
DUIDs: {TSDU: 10}
```

**Issues Found:**
1. **Frame sync count too low:** Only 10 syncs in 116s (expected ~300+ at 2.78/sec)
2. **NAC mismatch:** Detected 0x3d0, expected 0x3dc for SA-GRN

**Analysis:**
- DUID=7 (TSDU) is correct for control channel
- Low sync count suggests many frames are being missed
- NAC being 0x3d0 vs 0x3dc could be BCH decode error (1 bit difference)

#### Step 2.2: Test ControlChannelMonitor (Full Pipeline)

**Test:** `test_control_channel_monitor.py` on same 116-second CC recording

**Results:**
```
=== Summary ===
Batch mode:         FAIL (0% CRC pass)
Chunked  4000 smp: FAIL (0% CRC pass, 72 frames, 201 TSBK attempts)
Chunked 10000 smp: FAIL (0% CRC pass, 17 frames, 41 TSBK attempts)
Chunked 25000 smp: FAIL (0% CRC pass, 39 frames, 101 TSBK attempts)
Chunked 50000 smp: FAIL (0% CRC pass, 15 frames, 41 TSBK attempts)
```

**Critical Issues:**
1. **TSBK CRC pass rate = 0%** - All TSBK blocks fail CRC validation
2. **NACs scattered:** 0x2c4, 0xac4, 0xae4, 0x92c, 0x3d0 (should all be 0x3dc)
3. **Trellis error metrics:** 23-29 (should be much lower for clean decode)
4. **NID valid rate:** Only 47.5% (should be >90%)

**FINDING: Control channel is NOT decoding correctly!**

This means the problem is NOT just in the voice path - the control channel itself has issues:
- Sync detection works (state=synced)
- Frame extraction works (frames decoded)
- BUT TSBK parsing fails (0% CRC)

**Hypothesis:** The C4FM demodulation is producing incorrect dibits, causing:
- BCH decode errors (wrong NAC)
- Trellis decode errors (high error metrics)
- CRC failures (corrupted TSBK data)

#### Step 2.3: Compare C4FM Demodulation

**Test:** `debug_c4fm_comparison.py` on same recording

**Key Results:**
```
Discriminator: min=-3.1484, max=3.1484, std=1.2705
Expected for P25 C4FM: ~±0.23 radians

WaveCap C4FM (IQ-based):
  Soft symbol std: 2.5445 (expected ~0.78)
  Found 43 sync patterns (24/24 matches)
  NAC=0x3d0 detected (expected 0x3dc)
  DUIDs: LDU1, LDU2, TSDU, TDULC - all correct

Discriminator-based C4FM:
  Soft symbol std: 10.1301 (way too high)
  Dibit distribution: very uneven {0: 17002, 1: 47627, 2: 17317, 3: 62054}
```

**Critical Observations:**
1. **Discriminator range is ~14x too large** (±3.14 vs expected ±0.23)
2. **Soft symbol std is 3x too high** (2.54 vs 0.78)
3. **NAC consistently 0x3d0 instead of 0x3dc** (systematic 4-bit error)
4. **Frame sync works perfectly** (24/24 matches)
5. **DUIDs decode correctly** (LDU1, LDU2, TSDU, TDULC)

**Root Cause Analysis:**
- Sync detection works because it uses correlation (tolerant of scaling)
- NID decoding mostly works (BCH corrects some errors)
- TSBK fails because trellis decoder is sensitive to exact symbol values
- The 3x gain error in soft symbols corrupts trellis decoding

**Hypothesis:** The C4FM symbol decision thresholds or gain are miscalibrated,
causing trellis decoder to produce garbage (0% CRC pass rate).

#### Step 2.4: Detailed Dibit Comparison

**Test:** `debug_dibit_comparison.py` comparing SDRTrunk reference vs WaveCap

**Critical Findings:**
```
SDRTrunk Reference:
  Soft std: 0.5267 (low)
  Distribution: {0: 12959, 1: 118, 2: 10822, 3: 101}
  ALMOST NO 1s or 3s - BROKEN!
  Sync patterns found: 0

WaveCap C4FM:
  Soft std: 2.4813 (high but more spread)
  Distribution: {0: 5734, 1: 5249, 2: 6140, 3: 6877}
  Even distribution - GOOD
  Sync patterns found: 4 (with 24/24 matches)

Match rate between them: 24.7% (random!)
```

**Analysis:**
1. The SDRTrunk reference implementation I wrote is BROKEN
   - Produces mostly 0s and 2s, almost no 1s and 3s
   - Cannot find sync patterns
   - Cannot be used as a comparison baseline

2. WaveCap C4FM produces reasonable dibits
   - Even distribution of all 4 dibit values
   - Successfully finds sync patterns with perfect match
   - NAC extraction shows 0x3d0 (close but not quite 0x3dc)

3. Despite good sync detection, TSBK still fails CRC
   - The issue is NOT in basic C4FM demodulation
   - The issue is likely in TSBK-specific processing

**Hypothesis Updated:**
The C4FM demodulator is working (finds syncs, even dibit distribution).
The problem is in the TSBK extraction/decoding pipeline:
- Status symbol stripping?
- Deinterleaving?
- Bit/dibit ordering?

---

### Round 2: TSBK Pipeline Deep Dive (2024-12-30)

#### Step 2.5: End-to-End TSBK Trace

**Test:** Traced complete TSBK extraction pipeline on a single TSDU frame

**Pipeline Steps:**
1. Sync detection (24 dibits) → **WORKS** (24/24 matches)
2. NID extraction (33 dibits with status at pos 11) → **WORKS** (NAC=0x3d0, DUID=TSDU)
3. Status symbol removal (positions 71, 107, 143...) → Verified correct
4. TSBK block extraction (98 dibits) → Data extracted
5. Deinterleave (196 bits) → Applied correctly
6. Trellis 1/2 decode → **FAILS** (error_metric=25, expected <5)
7. CRC-16 CCITT → **FAILS** (0% pass rate)

**Key Finding:** Trellis error metric is consistently 17-29 for ALL TSBK blocks.
- Error metric of 25 in 196 bits = ~12.5% bit error rate
- This is WAY too high for valid trellis-encoded data
- Trellis encoder/decoder verified working in isolation (round-trip test)

#### Step 2.6: Soft Symbol Analysis

**Finding:** C4FM soft symbols are scaled ~1.4x higher than expected:
```
                    Actual    Expected   Ratio
Dibit 0 (+1):       +0.962    +0.785     1.23x
Dibit 1 (+3):       +3.307    +2.356     1.40x
Dibit 2 (-1):       -0.995    -0.785     1.27x
Dibit 3 (-3):       -3.245    -2.356     1.38x
```

**BUT:** This is from the SDRTrunk baseband recording - which SDRTrunk itself decodes correctly!
This means the issue is in how WaveCap processes the data, not the recording.

#### Step 2.7: Polarity and Deinterleave Tests

**Tests Performed:**
- Normal processing: error_metric=25, CRC=FAIL
- XOR 2 (polarity flip): error_metric=25, CRC=FAIL
- Without deinterleave: error_metric=25, CRC=FAIL
- Dibit-level deinterleave: error_metric=25, CRC=FAIL

**Conclusion:** The error is NOT in:
- Signal polarity
- Deinterleave pattern
- Bit/dibit ordering in deinterleave

The error IS in: The dibits going into trellis are already wrong.

---

## Gap Identified: Radio Configuration Not Tested

**IMPORTANT:** All testing so far uses SDRTrunk baseband recordings.
We have NOT verified that WaveCap's live radio capture works correctly.

**Missing Tests:**
1. SDR device detection and enumeration
2. Sample rate verification (should be 6 MHz for SA-GRN)
3. Center frequency accuracy
4. Gain settings (LNA, IF, baseband)
5. Frequency offset between WaveCap and SDRTrunk
6. AGC behavior with weak signals
7. Control channel frequency calculation from IDEN_UP

**Next Step:** Before continuing TSBK debugging, verify radio config matches SDRTrunk.

---

### Round 3: Deep Dive into Symbol Errors (2024-12-30 continued)

#### Step 3.1: NID Bit-Level Trace

**Test:** Created `debug_nid_trace.py` to trace exact bit/dibit values in NID

**Critical Finding - NAC Position 4 Error:**
```
Expected NAC: 0x3DC = dibits [0, 3, 3, 1, 3, 0]
Actual NAC:   0x3D0 = dibits [0, 3, 3, 1, 0, 0]
                                       ↑ ERROR HERE

Position 4: Expected dibit 3 (-3 symbol), got dibit 0 (+1 symbol)
```

**BCH Correction Behavior:**
- BCH "corrects" raw dibits to NAC=0x3D0 with 0 errors reported
- This means the parity bits are ALSO consistent with 0x3D0
- The entire codeword is corrupted, not just NAC

#### Step 3.2: Soft Symbol Investigation

**Soft Symbol Values at Error Position:**
```
Sync region (all correct):
  [23] dibit=3 soft=-3.116 expected=-3.0 ✓

NID region:
  NAC[0] dibit=0 soft=+0.990 expected=+1.0 ✓
  NAC[1] dibit=3 soft=-2.289 expected=-3.0 ✓
  NAC[2] dibit=3 soft=-2.341 expected=-3.0 ✓
  NAC[3] dibit=1 soft=+3.077 expected=+3.0 ✓
  NAC[4] dibit=0 soft=+1.289 expected=-3.0 ✗ ← HUGE ERROR!
  NAC[5] dibit=0 soft=+0.881 expected=+1.0 ✓
  NAC[6] dibit=2 soft=-0.556 expected=+3.0 ✗
  NAC[7] dibit=2 soft=-1.112 expected=-3.0 ✗
```

**Key Observation:**
- Position 4 soft symbol is +1.289 (correctly decoded as dibit 0)
- But EXPECTED value should be ~-2.5 (dibit 3, -3 symbol)
- The FM demodulator outputs +1.3 instead of -2.5 at this position!
- This is a ~3.8 unit error - completely wrong polarity

#### Step 3.3: IQ Polarity and Frequency Tests

**Test:** Tried all IQ transformations and frequency offsets

**Results:**
| Configuration | Syncs | NAC Match |
|---------------|-------|-----------|
| Original (I+jQ) | 4 | 75% [0,3,3,1,0,0] |
| Conjugate (I-jQ) | 0 | N/A |
| Swapped (Q+jI) | 0 | N/A |
| Negate both (-I-jQ) | 4 | 75% [0,3,3,1,0,0] |
| +92 Hz offset | 4 | 79% |

**Conclusion:** Polarity and frequency offset are NOT the cause.
The error is in the C4FM demodulator's symbol timing/extraction.

#### Step 3.4: Baseband Signal Analysis

**IQ Characteristics:**
```
Signal power: 0.000008 (very weak - 1% of full scale)
RMS: 0.002765
Frequency offset: +92 Hz
IQ balance: 1.008 (good)
```

**Instantaneous Frequency Distribution:**
- Does NOT show expected 4-level C4FM peaks at ±600, ±1800 Hz
- Shows broad noise distribution ±3000 Hz
- Likely due to weak signal buried in noise

**Phase per Symbol Distribution:**
- Range: ±3.14 radians (full rotation)
- No clear 4-level structure in histogram
- Yet sync detection works perfectly (24/24)!

#### Step 3.5: ROOT CAUSE IDENTIFIED

**The Problem:**
The C4FM demodulator produces correct soft symbols in the sync region
but INCORRECT soft symbols starting at NID position 4.

**Evidence:**
1. All 24 sync symbols decode correctly with reasonable soft values
2. NID positions 0-3 decode correctly
3. NID position 4 has soft=+1.289 when expected ~-2.5
4. This error is consistent across all frames and recordings
5. Polarity and frequency corrections don't help

**Root Cause Hypothesis:**
The error occurs AFTER sync detection, suggesting the equalizer or
timing recovery is updating incorrectly after finding sync. The
C4FM demodulator's timing or PLL adjustment after sync causes
subsequent symbols to be sampled at wrong phase.

**Specific Issue:**
Looking at the symbol sequence: sync ends with `...,3,3,3,3,3]`
NID starts with `[0,3,3,1,3,0,...]`

Position 4 (3→0→1→3 transition region) shows the largest error.
The equalizer may be overcorrecting after the long run of identical
symbols at the end of sync.

---

### Next Steps

1. **Investigate equalizer behavior after sync:**
   - Check PLL and gain updates after sync detection
   - Compare timing recovery in sync vs NID region

2. **Try disabling equalizer updates:**
   - Process with fixed PLL=0 and gain=1
   - See if symbol errors persist

3. **Compare with SDRTrunk's DifferentialDemodulator:**
   - SDRTrunk uses 8-tap polyphase interpolation
   - May have different timing recovery behavior

---

### Round 4: Deep Dive into Position 4 Error (2024-12-30 continued)

#### Step 4.1: SDRTrunk Source Code Comparison

**Key Finding from SDRTrunk (P25P1DemodulatorC4FM.java):**
SDRTrunk has a `validateNID()` method that **RESAMPLES the NID after timing optimization**:
```java
double pointer = bufferOffset + correction.getTimingCorrection() + mSamplesPerSymbol;
```

**WaveCap's Bug:**
WaveCap extracts ALL symbols upfront, then runs sync detection and timing optimization.
The optimized timing is applied to FUTURE symbols, but the NID symbols that were
already extracted are NOT resampled with the corrected timing.

#### Step 4.2: Position 4 Error Analysis

**Evidence:**
```
NID[4] actual= +1.29, expected=-3, dibit=0 (exp 3)

Position 4 Statistics (across 4 syncs):
  Mean: +1.313
  Std:  0.092  (very consistent - not noise!)
  Error: 4.313 (from expected -3.0)
```

**Pattern:**
- Error is 100% consistent
- Soft value is POSITIVE when should be strongly NEGATIVE
- This is NOT random noise - it's a systematic bug

**Context at position 4:**
- NID[2] = -2.3 (correct, dibit 3)
- NID[3] = +3.1 (correct, dibit 1)
- NID[4] = +1.3 (WRONG, should be ~-3)
- NID[5] = +0.9 (correct, dibit 0)

**Equalizer State:** PLL=0.0, Gain=1.219 (INITIAL values, not updated!)
This confirms the timing optimizer either didn't run or didn't persist corrections.

#### Step 4.3: Root Cause Confirmed

The C4FM demodulator extracts symbols in a streaming loop. When sync is detected:
1. Timing optimization computes correction
2. Correction is applied to `sample_point` for FUTURE symbols
3. But the NID symbols (positions 24-56) were already extracted with OLD timing
4. Those symbols are returned as-is, with the timing error baked in

**Fix Required:**
After sync detection and timing optimization, resample the NID region
(next 33 symbols after sync) with the corrected timing, PLL, and gain.
This matches SDRTrunk's `validateNID()` approach.

---

### Round 5: TSBK Deinterleave Fix (2024-12-30 Session 2)

#### Step 5.1: Timing/Resampling Fixes (Previous Session)

**Summary of Previous Fixes:**
1. Fixed half-symbol timing offset in NID resampling
2. Extended message resampling from 33 to 340 dibits
3. Verified trellis encode/decode round-trip works (error_metric=0)
4. NAC now correctly decoded as 0x3D0

**BUT:** TSBK CRC still failed at 100% with error_metric=25

#### Step 5.2: Comparing with SDRTrunk Trellis Implementation

**Investigation:** Fetched SDRTrunk's ViterbiDecoder_1_2_P25.java to compare
with WaveCap's trellis decoder.

**Key Finding:** SDRTrunk reads 4-bit symbols (nibbles), not 2-bit dibits!
- SDRTrunk: 196 bits → 49 nibbles (4 bits each) → Viterbi decode
- WaveCap: 196 bits → 98 dibits (2 bits each) → Viterbi decode

Both approaches should produce same result since we group dibits into pairs.
The trellis encoder table was verified to match SDRTrunk's TRANSITION_MATRIX.

#### Step 5.3: Deinterleave Logic Bug FOUND!

**Critical Discovery:** SDRTrunk's deinterleaveChunk() uses:
```java
for(int i = ...) {
    deinterleaved.set(pattern[i]);  // output[pattern[i]] = input[i]
}
```

But WaveCap's deinterleave_data() used:
```python
for i in range(196):
    deinterleaved[i] = bits[DATA_DEINTERLEAVE[i]]  # output[i] = input[pattern[i]]
```

**These are INVERSES of each other!**

The comment in the code incorrectly claimed it matched SDRTrunk, but the
actual implementation was reversed.

#### Step 5.4: The Fix

**Changed from:**
```python
# WRONG: output[i] = input[pattern[i]]
deinterleaved[i] = bits[DATA_DEINTERLEAVE[i]]
```

**To:**
```python
# CORRECT: output[pattern[i]] = input[i]
deinterleaved[DATA_DEINTERLEAVE[i]] = bits[i]
```

#### Step 5.5: Results After Fix

**Before Fix:**
- TSBK CRC pass rate: 0%
- Error metric: consistently 25

**After Fix:**
- TSBK CRC pass rate: 74-83% (depending on chunk size)
- Valid TSBK messages being decoded!

**Decoded TSBK Types:**
| Opcode | Name | Description |
|--------|------|-------------|
| 0x00 | GRP_V_CH_GRANT | Group Voice Channel Grant |
| 0x02 | GRP_V_CH_GRANT_UPDT | Group Voice Channel Grant Update |
| 0x03 | GRP_V_CH_GRANT_UPDT_EXP | Extended Grant Update |
| 0x14 | SNDCP_CH_GNT | SNDCP Channel Grant |
| 0x16 | SNDCP_CH_ANN_EXP | SNDCP Channel Announce |
| 0x30 | TDMA_SYNC | TDMA Sync |
| 0x34 | IDEN_UP_VU | Identifier Update (frequency bands) |
| 0x3A | RFSS_STS_BCAST | RFSS Status Broadcast |
| 0x3C | ADJ_STS_BCAST | Adjacent Status Broadcast |

**Sample IDEN_UP_VU Decode:**
```
identifier: 2
bandwidth_khz: 12.5
channel_spacing_khz: 6.25
tx_offset_mhz: 5.2
base_freq_mhz: 420.0125
```

**Sample Voice Grant:**
```
opcode: GRP_V_CH_GRANT
channel: 4282 (band 1, ch 186)
tgid: 2222
source_id: 5064113
encrypted: True
```

---

## Summary of All Fixes

1. **Half-symbol timing offset in NID resampling**
   - Cause: Symbol centers were offset by 0.5 symbols
   - Fix: Corrected sample point calculation

2. **Message resampling too short**
   - Cause: Only 33 dibits resampled after sync, but TSBK needs 340+
   - Fix: Extended resampling to 340 dibits

3. **Deinterleave direction inverted** ← CRITICAL FIX
   - Cause: Used `output[i] = input[pattern[i]]` instead of `output[pattern[i]] = input[i]`
   - Fix: Corrected to match SDRTrunk's logic
   - Impact: 0% → 74-83% TSBK CRC pass rate

---

## Next Steps

1. **Investigate remaining 17-26% CRC failures**
   - Could be signal quality issues (weak signal, fading)
   - Could be frame alignment issues at chunk boundaries

2. **Test voice grant following**
   - Now that control channel decodes correctly, test voice tracking

3. **Capture voice channel basebands from SDRTrunk**
   - Enable traffic channel recording for voice pipeline testing

