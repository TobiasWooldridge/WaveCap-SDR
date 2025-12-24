# P25 Frame Sync Detection: WaveCap-SDR vs SDRTrunk Comparison

## Executive Summary

This document compares P25 Phase 1 frame synchronization implementations between WaveCap-SDR and SDRTrunk, identifying critical bugs and implementation differences that were preventing reliable frame decoding.

## Critical Bugs Found and Fixed

### 1. **CRITICAL: Inverted C4FM Constellation Mapping**

**Bug**: WaveCap-SDR's C4FM demodulator had the dibit constellation mapping inverted.

**Location**: `/Users/thw/Projects/WaveCap-SDR/backend/wavecapsdr/decoders/p25.py` (lines 217-226)

**Original (INCORRECT)**:
```python
# Levels: -1 (dibit 0), -0.33 (dibit 1), +0.33 (dibit 2), +1 (dibit 3)
if y_k < self.thresholds[0]:
    dibit = 0  # -3 symbol -> WRONG!
elif y_k < self.thresholds[1]:
    dibit = 1  # -1 symbol -> WRONG!
elif y_k < self.thresholds[2]:
    dibit = 2  # +1 symbol -> WRONG!
else:
    dibit = 3  # +3 symbol -> WRONG!
```

**Fixed (CORRECT)**:
```python
# Per TIA-102.BAAA constellation:
# Symbol level (normalized) -> C4FM Symbol -> Dibit value
# < -0.67 (-1.0)  -> -3 symbol -> dibit 3
# -0.67 to 0      -> -1 symbol -> dibit 2
# 0 to +0.67      -> +1 symbol -> dibit 0
# > +0.67 (+1.0)  -> +3 symbol -> dibit 1
if y_k < self.thresholds[0]:  # < -0.67
    dibit = 3  # -3 symbol
elif y_k < self.thresholds[1]:  # -0.67 to 0
    dibit = 2  # -1 symbol
elif y_k < self.thresholds[2]:  # 0 to +0.67
    dibit = 0  # +1 symbol
else:  # > +0.67
    dibit = 1  # +3 symbol
```

**Impact**: This bug caused ALL demodulated dibits to be incorrect, making frame sync detection impossible.

**Root Cause**: Incorrect assumption about C4FM constellation. Per TIA-102.BAAA:
- **+3 symbol** (max positive deviation, +1800 Hz) = dibit **1** (binary 01)
- **+1 symbol** (small positive, +600 Hz) = dibit **0** (binary 00)
- **-1 symbol** (small negative, -600 Hz) = dibit **2** (binary 10)
- **-3 symbol** (max negative deviation, -1800 Hz) = dibit **3** (binary 11)

---

### 2. **CRITICAL: Incorrect Frame Sync Pattern**

**Bug**: WaveCap-SDR used an incorrect 48-bit sync pattern.

**Location**: Multiple files
- `/Users/thw/Projects/WaveCap-SDR/backend/wavecapsdr/decoders/p25.py`
- `/Users/thw/Projects/WaveCap-SDR/backend/wavecapsdr/decoders/p25_frames.py`
- `/Users/thw/Projects/WaveCap-SDR/backend/wavecapsdr/trunking/control_channel.py`

**Original (INCORRECT)**:
```python
# WaveCap used incorrect symbol sequence and wrong dibit encoding:
FRAME_SYNC_DIBITS = np.array([3, 3, 3, 3, 3, 0, 3, 3, 0, 0, 3, 3,
                               3, 3, 0, 3, 0, 3, 0, 0, 0, 3, 0, 0], dtype=np.uint8)
# Hex: 0xFFCF0FF33030
```

**Fixed (CORRECT)**:
```python
# C4FM symbols: +3 +3 +3 +3 +3 -3 +3 +3 -3 -3 +3 +3 -3 -3 -3 -3 +3 -3 +3 -3 -3 -3 -3 -3
# Dibit encoding: +3 symbol = dibit 1, -3 symbol = dibit 3
FRAME_SYNC_DIBITS = np.array([1, 1, 1, 1, 1, 3, 1, 1, 3, 3, 1, 1,
                               3, 3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3], dtype=np.uint8)
# Hex: 0x5575F5FF77FF (matches SDRTrunk)
```

**Impact**: Even with correct demodulation, sync would never be detected due to pattern mismatch.

---

### 3. **Architectural Issue: Multiple Sync Patterns**

**Bug**: `p25_frames.py` incorrectly defined multiple sync patterns for different frame types.

**Original (INCORRECT)**:
```python
FRAME_SYNC_PATTERNS = {
    0x5575F5FF77FF: DUID.HDU,
    0x5575F5FF77FD: DUID.TDU,
    0x5575F5FF77F7: DUID.LDU1,
    0x5575F5FF775F: DUID.TSDU,
    # ...
}
```

**Fixed (CORRECT)**:
```python
# Per TIA-102.BAAA, P25 Phase 1 uses ONE sync pattern for ALL frame types.
# The frame type is determined by the DUID field in the NID that follows sync.
FRAME_SYNC_PATTERN = 0x5575F5FF77FF
FRAME_SYNC_DIBITS = np.array([1, 1, 1, 1, 1, 3, 1, 1, 3, 3, 1, 1,
                               3, 3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3], dtype=np.uint8)
```

**Impact**: Incorrect understanding of P25 framing structure.

---

## Implementation Comparison

### Sync Pattern

| Aspect | WaveCap-SDR (Before Fix) | WaveCap-SDR (After Fix) | SDRTrunk |
|--------|--------------------------|-------------------------|----------|
| **Sync Hex** | 0xFFCF0FF33030 | **0x5575F5FF77FF** | **0x5575F5FF77FF** |
| **Pattern Type** | Multiple patterns | **Single pattern** | **Single pattern** |
| **+3 Symbol Encoding** | dibit 3 (WRONG) | **dibit 1** | **dibit 1** |
| **-3 Symbol Encoding** | dibit 0 (WRONG) | **dibit 3** | **dibit 3** |

### Sync Detection Method

| Feature | WaveCap-SDR | SDRTrunk |
|---------|-------------|----------|
| **Hard Sync** | ✓ (correlation search) | ✓ (rolling bit comparison) |
| **Soft Sync** | ✗ Missing | **✓ (dot product correlation)** |
| **Error Tolerance** | 4 dibit errors | 4 **bit** errors (= 2 dibit errors) |
| **Sync Threshold** | Hard-coded 4 dibits | 60.0 correlation score for soft |

**Key Difference**: SDRTrunk's sync detection allows **4 bit errors**, which equals **2 dibit errors** (since each dibit = 2 bits). WaveCap-SDR's tolerance of 4 **dibit** errors is MORE lenient (8 bit errors), which could lead to false positives.

### Hard Sync Detection

**SDRTrunk** (`P25P1HardSyncDetector.java`):
```java
public boolean process(Dibit dibit)
{
    mValue = (Long.rotateLeft(mValue, 2) & SYNC_MASK) + dibit.getValue();
    return Long.bitCount(mValue ^ SYNC_PATTERN) <= MAXIMUM_BIT_ERROR;  // 4 BIT errors
}
```

- Uses rolling 48-bit buffer
- Compares via XOR and counts differing **bits**
- Threshold: **4 bit errors** (hamming distance)

**WaveCap-SDR** (`control_channel.py` `_find_sync_in_buffer()`):
```python
for i in range(len(self._dibit_buffer) - self.FRAME_SYNC_DIBITS + 1):
    errors = 0
    for j in range(self.FRAME_SYNC_DIBITS):
        if self._dibit_buffer[i + j] != sync_dibits[j]:
            errors += 1
            if errors > max_errors:  # 4 DIBIT errors
                break
    if errors <= max_errors:
        return i
```

- Searches dibit buffer linearly
- Compares dibits directly
- Threshold: **4 dibit errors** (= 8 bit errors)

**Recommendation**: WaveCap-SDR should reduce tolerance to **2 dibit errors** to match SDRTrunk's 4-bit-error threshold and reduce false positives.

---

### Soft Sync Detection

**SDRTrunk** (`P25P1SoftSyncDetectorScalar.java`):
```java
public float calculate()
{
    float symbol;
    float score = 0;

    for(int x = 0; x < 24; x++)
    {
        symbol = mSymbols[mSymbolPointer + x];
        score += SYNC_PATTERN_SYMBOLS[x] * symbol;  // Dot product
    }

    return score;
}
```

- Uses **soft symbol values** (demodulated phases)
- Computes **dot product** correlation against ideal sync pattern
- Threshold: correlation score > 60.0
- Processes every symbol (no search, just rolling correlation)

**SYNC_PATTERN_SYMBOLS** is the sync pattern converted to ideal phase values:
- +3 symbol → `Dibit.D01_PLUS_3.getIdealPhase()` = `3π/4` radians
- -3 symbol → `Dibit.D11_MINUS_3.getIdealPhase()` = `-3π/4` radians

**WaveCap-SDR**: No soft sync detection implemented.

**Recommendation**: Add soft sync detection to WaveCap-SDR for improved sensitivity and earlier sync detection.

---

### Sync Loss Detection

| Feature | WaveCap-SDR | SDRTrunk |
|---------|-------------|----------|
| **Method** | Timeout (2.0 seconds) + sync verification on each frame | Continuous per-symbol sync tracking |
| **Timeout** | 2.0 seconds | 4800 symbols (1 second at 4800 baud) |
| **Re-sync** | Automatic on loss | Automatic via continuous detection |

**SDRTrunk** tracks sync continuously by running the sync detector on **every symbol**, so sync loss is detected immediately when correlation drops.

**WaveCap-SDR** only checks sync at frame boundaries and declares loss after 2 seconds of no frames.

**Recommendation**: Consider reducing timeout to 1 second to match SDRTrunk.

---

## Algorithm Differences

### 1. Sync Detection Algorithm

**SDRTrunk**:
- **Dual-mode**: Both hard and soft sync detection run in parallel
- **Rolling correlation**: Correlation calculated on every symbol
- **Threshold-based**: Soft sync uses correlation score threshold (60.0)
- **Immediate detection**: Sync detected within 24 symbols (5ms at 4800 baud)

**WaveCap-SDR**:
- **Single-mode**: Only hard sync (dibit correlation search)
- **Buffer search**: Searches entire buffer when enough dibits accumulated
- **Count-based**: Counts dibit mismatches, allows 4 errors
- **Delayed detection**: Waits until MIN_FRAME_DIBITS (150) accumulated

**Impact**: SDRTrunk's approach is faster and more sensitive, especially with soft symbols.

---

### 2. Error Tolerance

**SDRTrunk**:
```java
MAXIMUM_BIT_ERROR = 4;  // 4 bits out of 48 = 8.3% bit error rate
```

**WaveCap-SDR**:
```python
sync_threshold = 4  # 4 dibits out of 24 = 16.7% dibit error rate = 33.3% bit error rate
```

**Recommendation**: WaveCap should use **2 dibit errors** maximum (= 4 bit errors) to match SDRTrunk and reduce false positives.

---

### 3. Status Symbol Handling

Both implementations correctly handle P25's status symbols (every 35 dibits from frame start).

**SDRTrunk** (`P25P1MessageFramer.java`):
```java
mStatusSymbolDibitCounter++;

if(mStatusSymbolDibitCounter == 36)
{
    // Send status dibit to channel status processor
    mChannelStatusProcessor.receive(symbol);

    mStatusSymbolDibitCounter = 0;
    mDibitCounter++;
    return false;  // Don't process this symbol as data
}
```

**WaveCap-SDR** (`p25_frames.py` `decode_nid()`):
```python
if skip_status_at_11 and i == 11:
    continue  # Skip status symbol at position 11 within NID
```

Both approaches are correct but differ in implementation detail.

---

## Recommended Enhancements for WaveCap-SDR

### 1. Add Soft Symbol Sync Detection (HIGH PRIORITY)

Implement correlation-based soft sync similar to SDRTrunk:

```python
class SoftSyncDetector:
    """Soft symbol correlation-based sync detector."""

    # Ideal phase values for sync pattern symbols
    # +3 symbol = 3π/4, -3 symbol = -3π/4
    SYNC_SYMBOLS = np.array([
        3*np.pi/4, 3*np.pi/4, 3*np.pi/4, 3*np.pi/4, 3*np.pi/4,
        -3*np.pi/4, 3*np.pi/4, 3*np.pi/4, -3*np.pi/4, -3*np.pi/4,
        3*np.pi/4, 3*np.pi/4, -3*np.pi/4, -3*np.pi/4, -3*np.pi/4,
        -3*np.pi/4, 3*np.pi/4, -3*np.pi/4, 3*np.pi/4, -3*np.pi/4,
        -3*np.pi/4, -3*np.pi/4, -3*np.pi/4, -3*np.pi/4
    ], dtype=np.float32)

    def __init__(self):
        self.symbol_buffer = np.zeros(48, dtype=np.float32)  # 2x pattern length
        self.pointer = 0
        self.threshold = 60.0  # SDRTrunk's threshold

    def process(self, soft_symbol: float) -> float:
        """Process soft symbol and return correlation score.

        Args:
            soft_symbol: Demodulated phase value (radians)

        Returns:
            Correlation score (>60.0 indicates sync)
        """
        # Store symbol in circular buffer
        self.symbol_buffer[self.pointer] = soft_symbol
        self.symbol_buffer[self.pointer + 24] = soft_symbol
        self.pointer = (self.pointer + 1) % 24

        # Compute dot product correlation
        score = np.dot(self.symbol_buffer[self.pointer:self.pointer + 24],
                       self.SYNC_SYMBOLS)

        return score
```

**Benefits**:
- Earlier sync detection (before full dibit decision)
- Better performance in noisy conditions
- Matches SDRTrunk's proven approach

---

### 2. Reduce Error Tolerance (MEDIUM PRIORITY)

Change sync threshold from 4 dibit errors to **2 dibit errors**:

```python
# In P25FrameSync
self.sync_threshold = 2  # 2 dibit errors = 4 bit errors (matches SDRTrunk)

# In ControlChannelMonitor._find_sync_in_buffer()
max_errors = 2  # Reduced from 4
```

**Benefits**:
- Reduces false positive sync detections
- Matches SDRTrunk's 4-bit error threshold
- Improves decoder reliability

---

### 3. Continuous Sync Tracking (LOW PRIORITY)

Instead of only checking sync at frame boundaries, run sync detector continuously:

```python
def process_iq(self, iq: np.ndarray):
    """Process IQ and detect sync continuously."""
    dibits = self.demodulator.demodulate(iq)

    for dibit in dibits:
        # Update rolling sync detector
        if self.soft_sync.process(soft_symbol) > self.threshold:
            self.on_sync_detected()

        # Also process frame data if synced
        if self.sync_state == SyncState.SYNCED:
            self.process_frame_dibit(dibit)
```

**Benefits**:
- Faster sync reacquisition after loss
- Continuous sync quality monitoring
- Earlier detection of sync degradation

---

## Verification

All fixes have been verified against SDRTrunk's implementation:

```python
# Verified sync pattern
assert FRAME_SYNC_PATTERN == 0x5575F5FF77FF  # ✓ Matches SDRTrunk
assert len(FRAME_SYNC_DIBITS) == 24  # ✓ 48 bits = 24 dibits

# Verified constellation
# +3 symbol (max positive) -> dibit 1
# -3 symbol (max negative) -> dibit 3
# Matches SDRTrunk's Dibit.D01_PLUS_3 and Dibit.D11_MINUS_3
```

---

## Files Modified

1. `/Users/thw/Projects/WaveCap-SDR/backend/wavecapsdr/decoders/p25.py`
   - Fixed C4FM constellation mapping (lines 228-241)
   - Fixed FRAME_SYNC_DIBITS pattern (lines 395-396)

2. `/Users/thw/Projects/WaveCap-SDR/backend/wavecapsdr/decoders/p25_frames.py`
   - Replaced FRAME_SYNC_PATTERNS dict with single FRAME_SYNC_PATTERN (lines 71-81)
   - Added correct FRAME_SYNC_DIBITS array

3. `/Users/thw/Projects/WaveCap-SDR/backend/wavecapsdr/trunking/control_channel.py`
   - Fixed _get_sync_dibits() to return correct pattern (lines 357-374)
   - Updated imports to use FRAME_SYNC_PATTERN

---

## References

1. **TIA-102.BAAA-A**: Project 25 Physical Layer Specification (C4FM constellation)
2. **SDRTrunk Source**: https://github.com/DSheirer/sdrtrunk
   - `P25P1SyncDetector.java`: Base sync pattern definition
   - `P25P1HardSyncDetector.java`: Hard symbol sync
   - `P25P1SoftSyncDetector.java`: Soft symbol correlation sync
   - `P25P1MessageFramer.java`: Frame assembly and sync loss detection
3. **P25 Frame Structure**: Sync (48 bits) + NID (64 bits) + Frame Data

---

## Test Results

After fixes:
- ✅ Sync pattern matches SDRTrunk (0x5575F5FF77FF)
- ✅ C4FM constellation matches TIA-102.BAAA
- ✅ Single sync pattern used for all frame types
- ⚠️ Error tolerance still higher than SDRTrunk (pending reduction)
- ❌ Soft sync detection not yet implemented (enhancement pending)

---

## Conclusion

The critical bugs in WaveCap-SDR's P25 decoder have been identified and fixed:

1. **Inverted C4FM constellation** - Now matches TIA-102.BAAA spec
2. **Incorrect sync pattern** - Now matches SDRTrunk (0x5575F5FF77FF)
3. **Multiple sync patterns** - Now correctly uses single pattern

These fixes enable proper P25 Phase 1 frame synchronization. Additional enhancements (soft sync, reduced error tolerance) will further improve reliability and match SDRTrunk's proven implementation.
