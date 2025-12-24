# P25 NID and BCH Error Correction Implementation Comparison

## Summary

This document compares the P25 NID (Network ID) and BCH error correction implementations between WaveCap-SDR and SDRTrunk, and documents the fixes applied to WaveCap-SDR to match SDRTrunk's proven implementation.

**Date:** 2025-12-22
**Compared Versions:**
- WaveCap-SDR: `main` branch (commit 7dc7b04)
- SDRTrunk: Latest main branch

## Key Differences Found

### 1. Status Symbol Handling ✅ VERIFIED CORRECT

**Location:**
- SDRTrunk: `P25P1MessageFramer.java`, line 824
- WaveCap-SDR: `p25_frames.py`, line 347

**SDRTrunk Implementation:**
```java
for(int i = 0; i < DIBIT_LENGTH_NID; i++)
{
    if(i != 11)  // Skip status symbol at position 11
    {
        dibit = mNIDBuffer[i];
        nid.add(dibit.getBit1(), dibit.getBit2());
    }
}
```

**WaveCap-SDR Implementation (CORRECT):**
```python
for i in range(required_len):
    if skip_status_at_11 and i == 11:
        continue  # Skip status symbol (CRITICAL - matches SDRTrunk line 824)
    clean_dibits.append(int(dibits[i]))
```

**Status:** ✅ **ALREADY CORRECT** - WaveCap-SDR properly skips position 11

**Explanation:**
P25 inserts a status symbol every 35 dibits from frame start. Since the sync pattern is 24 dibits, the first status symbol appears at dibit position 35 (sync 24 + NID 11 = 35). This status symbol **must** be skipped when collecting NID dibits.

---

### 2. BCH(63,16,23) Error Correction ✅ IMPLEMENTED

**Location:**
- SDRTrunk: `BCH_63_16_23_P25.java`, `BCH_63.java`, `BCH.java`
- WaveCap-SDR: Previously **MISSING**, now implemented in `dsp/fec/bch.py`

**SDRTrunk Code Parameters:**
```java
public static final int K = 16;  // Data bits (NAC + DUID)
private static final int T = 11; // Error-correcting capability
public static final int M = 6;   // GF(2^6)
public static final int N = 63;  // Codeword length
public static final int PRIMITIVE_POLYNOMIAL_GF_63 = 0x43;
```

**WaveCap-SDR Implementation (NEW):**
```python
class BCH_63_16_23:
    M = 6  # GF(2^6)
    N = 63  # Codeword length (2^M - 1)
    K = 16  # Message length (12-bit NAC + 4-bit DUID)
    T = 11  # Error correction capacity
    PRIMITIVE_POLYNOMIAL = 0x43  # x^6 + x + 1
```

**Status:** ✅ **NEWLY IMPLEMENTED**

**What Was Added:**
1. Complete BCH decoder implementation (`backend/wavecapsdr/dsp/fec/bch.py`)
   - Galois Field GF(2^6) arithmetic
   - Syndrome calculation
   - Berlekamp-Massey algorithm for error locator polynomial
   - Chien search for root finding
   - Can correct up to 11 bit errors in 63-bit codeword

2. Integration into `decode_nid()`:
   - Converts dibits to bits
   - Calls BCH decoder
   - Extracts corrected NAC and DUID

**Algorithm:**
- **Input:** 63-bit codeword (16 data bits + 47 parity bits)
- **Output:** 16-bit corrected data (12-bit NAC + 4-bit DUID) + error count
- **Error Correction Capacity:** Up to 11 bit errors

---

### 3. NAC Tracking for BCH Assistance ✅ IMPLEMENTED

**Location:**
- SDRTrunk: `NACTracker.java`
- WaveCap-SDR: Previously **MISSING**, now implemented in `decoders/nac_tracker.py`

**SDRTrunk Implementation:**
```java
public class NACTracker {
    private static final int MAX_TRACKER_COUNT = 3;
    private static final int MIN_OBSERVATION_THRESHOLD = 3;

    public void track(int nac) { ... }
    public int getTrackedNAC() { ... }
}
```

**WaveCap-SDR Implementation (NEW):**
```python
class NACTracker:
    MAX_TRACKER_COUNT = 3
    MIN_OBSERVATION_THRESHOLD = 3

    def track(self, nac: int): ...
    def get_tracked_nac(self) -> int: ...
```

**Status:** ✅ **NEWLY IMPLEMENTED**

**Purpose:**
The NAC tracker maintains a count of recently observed NAC values. When BCH decode fails on the first pass, the decoder can use the tracked NAC to overwrite potentially corrupted NAC bits and retry decoding. This two-pass approach significantly improves decode success rate.

**Features:**
- Tracks up to 3 distinct NAC values
- Requires 3 observations before NAC becomes "dominant"
- Auto-prunes oldest tracker when limit exceeded
- Provides statistics for debugging

---

### 4. Two-Pass BCH Decoding ✅ IMPLEMENTED

**Location:**
- SDRTrunk: `BCH_63_16_23_P25.java`, lines 55-67
- WaveCap-SDR: Now implemented in `dsp/fec/bch.py`

**SDRTrunk Logic:**
```java
public void decode(CorrectedBinaryMessage message, int observedNAC) {
    decode(message);  // First pass

    if(message.getCorrectedBitCount() == BCH.MESSAGE_NOT_CORRECTED
       && observedNAC > 0) {
        // Check if NAC differs, overwrite and try again
        if(message.getInt(NAC_FIELD) != observedNAC) {
            message.setInt(observedNAC, NAC_FIELD);
            decode(message);  // Second pass
        }
    }
}
```

**WaveCap-SDR Implementation (NEW):**
```python
def decode(self, codeword: np.ndarray, tracked_nac: Optional[int] = None):
    # First pass: try to decode as-is
    data, errors = self._decode_internal(codeword)

    if errors != self.MESSAGE_NOT_CORRECTED:
        return data, errors

    # Second pass: if we have a tracked NAC and first pass failed
    if tracked_nac is not None and tracked_nac > 0:
        current_nac = extract_nac(codeword)
        if current_nac != tracked_nac:
            codeword_copy = codeword.copy()
            overwrite_nac(codeword_copy, tracked_nac)
            data, errors = self._decode_internal(codeword_copy)
            return data, errors

    return data, errors
```

**Status:** ✅ **IMPLEMENTED**

**Benefits:**
- Significantly improves decode success rate when NAC bits are corrupted
- No performance penalty when first pass succeeds
- Maintains statistical dominance (only uses well-established NAC values)

---

## Integration Points

### decode_nid() Function

**Old Signature:**
```python
def decode_nid(dibits: np.ndarray, skip_status_at_11: bool = True) -> Optional[NID]:
```

**New Signature:**
```python
def decode_nid(
    dibits: np.ndarray,
    skip_status_at_11: bool = True,
    nac_tracker: Optional[NACTracker] = None,
) -> Optional[NID]:
```

**Changes:**
1. Added optional `nac_tracker` parameter (backward compatible)
2. Replaced manual NAC/DUID extraction with BCH decode
3. Tracks successful NAC values for future decodes
4. Returns error count in NID object

### NID Dataclass

**Old:**
```python
@dataclass
class NID:
    nac: int
    duid: DUID
    errors: int = 0  # Was always 0
```

**New:**
```python
@dataclass
class NID:
    nac: int
    duid: DUID
    errors: int = 0  # Now contains actual BCH error count
```

The `errors` field now contains the actual number of bit errors corrected by BCH (0-11), or -1 if decode failed.

---

## Files Created/Modified

### New Files Created

1. **`backend/wavecapsdr/dsp/fec/bch.py`** (467 lines)
   - Complete BCH(63,16,23) decoder implementation
   - Galois Field arithmetic
   - Berlekamp-Massey algorithm
   - Chien search
   - Two-pass decoding with NAC tracking

2. **`backend/wavecapsdr/decoders/nac_tracker.py`** (122 lines)
   - NAC observation tracker
   - Dominant NAC selection
   - Automatic pruning
   - Statistics reporting

3. **`backend/tests/test_p25_bch.py`** (169 lines)
   - Unit tests for BCH decoder
   - Unit tests for NAC tracker
   - Integration tests for decode_nid()
   - All tests pass ✅

### Modified Files

1. **`backend/wavecapsdr/decoders/p25_frames.py`**
   - Added BCH and NAC tracker imports
   - Updated `decode_nid()` function with BCH integration
   - Added optional nac_tracker parameter
   - Enhanced debug logging

---

## Test Results

```bash
$ pytest tests/test_p25_bch.py -v
============================= test session starts ==============================
tests/test_p25_bch.py::TestBCH::test_initialization PASSED
tests/test_p25_bch.py::TestBCH::test_galois_tables PASSED
tests/test_p25_bch.py::TestBCH::test_decode_no_errors PASSED
tests/test_p25_bch.py::TestBCH::test_decode_with_nac_tracking PASSED
tests/test_p25_bch.py::TestNACTracker::test_initialization PASSED
tests/test_p25_bch.py::TestNACTracker::test_single_nac_tracking PASSED
tests/test_p25_bch.py::TestNACTracker::test_multiple_nac_values PASSED
tests/test_p25_bch.py::TestNACTracker::test_max_tracker_count PASSED
tests/test_p25_bch.py::TestNACTracker::test_reset PASSED
tests/test_p25_bch.py::TestNIDDecode::test_decode_with_tracker PASSED
tests/test_p25_bch.py::TestNIDDecode::test_decode_without_tracker PASSED
tests/test_p25_bch.py::TestNIDDecode::test_status_symbol_skipping PASSED
tests/test_p25_bch.py::TestNIDDecode::test_decode_too_short PASSED

============================== 13 passed in 0.48s ===============================
```

**All existing P25 tests continue to pass:**
```bash
$ pytest tests/test_p25_dsp.py -v
tests/test_p25_dsp.py::TestGolay::test_encode_decode_no_errors PASSED
tests/test_p25_dsp.py::TestGolay::test_single_bit_error_correction PASSED
tests/test_p25_dsp.py::TestGolay::test_double_bit_error_correction PASSED
tests/test_p25_dsp.py::TestGolay::test_triple_bit_error_correction PASSED
... (all 38 tests pass)
```

---

## NID Structure Reference

### P25 NID Format

```
Bit Position:  0         12       16                            64
              ┌──────────┬────────┬──────────────────────────────┐
              │   NAC    │  DUID  │         BCH Parity           │
              │ (12 bits)│(4 bits)│         (48 bits)            │
              └──────────┴────────┴──────────────────────────────┘
                                  │                              │
                                  └──────── BCH(63,16,23) ───────┘
                                   (uses first 63 bits)
```

**Fields:**
- **NAC (Network Access Code):** 12 bits - Identifies the network
- **DUID (Data Unit ID):** 4 bits - Frame type (HDU, LDU1, LDU2, TDU, TSDU, etc.)
- **BCH Parity:** 47 bits - Error correction (BCH(63,16,23) uses 63 bits total)
- **Reserved:** 1 bit (bit 63) - Not used by BCH

### Status Symbol Positions

```
Frame Layout:
┌────────────────┬───────────────┬─────────────────┐
│  Sync (24 dib) │  NID (33 dib) │  Payload ...    │
└────────────────┴───────────────┴─────────────────┘
                  ↑
                  Position 11 (relative to NID start)
                  = Position 35 (absolute from frame start)
                  = Status Symbol (MUST BE SKIPPED)
```

**Status symbols appear every 35 dibits** starting from frame sync.

---

## Expected Performance Improvements

1. **Error Correction:**
   - **Before:** No error correction on NID (accept/reject only)
   - **After:** Can correct up to 11 bit errors in NID
   - **Impact:** Dramatically improved NID decode success in noisy conditions

2. **NAC Tracking:**
   - **Before:** Each NID decode independent
   - **After:** Two-pass decode with NAC hint
   - **Impact:** Improved decode when NAC bits corrupted but rest of NID intact

3. **Statistics:**
   - **Before:** No visibility into decode quality
   - **After:** Error count available in NID.errors field
   - **Impact:** Can monitor link quality and tune receiver

---

## Usage Example

```python
from wavecapsdr.decoders.p25_frames import decode_nid
from wavecapsdr.decoders.nac_tracker import NACTracker
import numpy as np

# Create NAC tracker (one per channel/receiver)
nac_tracker = NACTracker()

# Decode NID dibits
dibits = np.array([...])  # 33 dibits from demodulator
nid = decode_nid(dibits, skip_status_at_11=True, nac_tracker=nac_tracker)

if nid:
    print(f"NAC: 0x{nid.nac:03x}")
    print(f"DUID: {nid.duid}")
    print(f"Errors corrected: {nid.errors}")

    # NAC is automatically tracked for future decodes
else:
    print("NID decode failed (>11 bit errors)")
```

---

## References

1. **SDRTrunk Source Code:**
   - `src/main/java/io/github/dsheirer/module/decode/p25/phase1/P25P1MessageFramer.java`
   - `src/main/java/io/github/dsheirer/edac/bch/BCH_63_16_23_P25.java`
   - `src/main/java/io/github/dsheirer/edac/bch/BCH_63.java`
   - `src/main/java/io/github/dsheirer/edac/bch/BCH.java`
   - `src/main/java/io/github/dsheirer/module/decode/p25/phase1/NACTracker.java`

2. **TIA-102.BAAA-A:** P25 Phase 1 Physical Layer Specification
   - Network ID (NID) structure
   - BCH(63,16,23) error correction
   - Status symbol placement

3. **Linux Kernel BCH Implementation:**
   - https://github.com/Parrot-Developers/bch/blob/master/include/linux/bch.h
   - Original C implementation that SDRTrunk ported to Java

---

## Conclusion

WaveCap-SDR now has **full parity** with SDRTrunk's P25 NID decoding implementation:

✅ Status symbol handling (was already correct)
✅ BCH(63,16,23) error correction (newly implemented)
✅ NAC tracking for two-pass decode (newly implemented)
✅ All tests passing
✅ Backward compatible API

The implementation is based on the proven SDRTrunk design and the Linux kernel BCH decoder, ensuring correctness and reliability.
