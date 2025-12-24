# C4FM Demodulation Comparison: WaveCap-SDR vs SDRTrunk

## Executive Summary

This document compares the C4FM (4-FSK) demodulation implementations between WaveCap-SDR and SDRTrunk, identifying key differences that could impact P25 Phase 1 decoding performance.

## Architecture Overview

### SDRTrunk Approach (P25P1DemodulatorC4FM.java)
- **Input**: Pre-demodulated phase samples (PI/4 DQPSK output) in radians
- **Symbol timing**: Optimized at each sync detection with fine/coarse modes
- **Equalizer**: PLL + gain correction applied to phase samples
- **Interpolation**: Linear interpolation for fractional sample timing
- **Constellation gain**: Initialized at 1.219, adapted 1.0-1.25 range

### WaveCap-SDR Approach (c4fm.py)
- **Input**: Complex IQ samples
- **FM Discriminator**: Quadrature demodulation to extract frequency
- **RRC Filter**: Root-raised cosine matched filter
- **Symbol timing**: Gardner TED with PI controller
- **Interpolation**: Cubic (Farrow structure)
- **Constellation gain**: Initialized at 1.0, adapted 1.0-1.25 range

## Key Differences

### 1. Symbol Decision Thresholds ⚠️ CRITICAL

**SDRTrunk (Dibit.java lines 24-27)**
```java
D01_PLUS_3:  phase = +3π/4  (dibit value 1)
D00_PLUS_1:  phase = +π/4   (dibit value 0)
D10_MINUS_1: phase = -π/4   (dibit value 2)
D11_MINUS_3: phase = -3π/4  (dibit value 3)
```
Decision boundary: π/2 (line 56)
```java
return sample > π/2 ? D01_PLUS_3 : D00_PLUS_1;  // Positive samples
return sample < -π/2 ? D11_MINUS_3 : D10_MINUS_1; // Negative samples
```

**WaveCap-SDR (c4fm.py lines 98-103)**
```python
SYMBOL_LEVELS = [3.0, 1.0, -1.0, -3.0]  # Arbitrary units
DIBIT_MAP = [3, 2, 1, 0]  # Maps symbol index to dibit
```
Thresholds: [-0.67, 0.0, +0.67] - calculated for ±3, ±1 levels

**Issue**: WaveCap-SDR maps frequency deviation to symbol levels, but the dibit mapping doesn't align with SDRTrunk's phase-based mapping.

**Impact**: MODERATE - Different dibit assignments could cause bit errors in frame sync and data decoding.

### 2. Interpolation Method

**SDRTrunk (LinearInterpolator.java line 45)**
```java
return x1 + ((x2 - x1) * mu);  // Simple linear
```

**WaveCap-SDR (c4fm.py lines 224-246)**
```python
# Cubic Lagrange interpolation
c0 = s1
c1 = (s2 - s0) / 2
c2 = s0 - 5*s1/2 + 2*s2 - s3/2
c3 = (s3 - s0)/2 + 3*(s1 - s2)/2
return c0 + mu * (c1 + mu * (c2 + mu * c3))
```

**Impact**: LOW - Cubic is theoretically better, but linear is simpler and proven to work.

### 3. Constellation Gain Correction ⚠️ CRITICAL

**SDRTrunk (P25P1DemodulatorC4FM.java lines 52-54, 778)**
```java
EQUALIZER_LOOP_GAIN = 0.15f
EQUALIZER_MAXIMUM_GAIN = 1.25f
mGain = 1.219f  // Initial gain
```
Applied as: `(sample + mPll) * mGain` (line 881)

Updated on sync (lines 1052-1053):
```java
mPll += (correction.getPllAdjustment() * EQUALIZER_LOOP_GAIN);
mGain += (correction.getGainAdjustment() * EQUALIZER_LOOP_GAIN);
```

**WaveCap-SDR (c4fm.py lines 105-113, 330-332)**
```python
GAIN_INITIAL = 1.0  # Start at unity
GAIN_LOOP_ALPHA = 0.15  # Same as SDRTrunk
# Applied as:
current = (raw_symbol + self._dc_offset) * self._constellation_gain
```

**Issue**: WaveCap initializes at 1.0 instead of 1.219, and updates every 24 symbols instead of at sync detection.

**Impact**: MODERATE - Lower initial gain could cause early frame sync failures until gain adapts.

### 4. Timing Recovery Loop Filter ⚠️ CRITICAL

**SDRTrunk (P25P1DemodulatorC4FM.java lines 144-150)**
Uses proportional-integral (PI) loop filter with damping:
```java
damping = 1.0
theta = loop_bw / (damping + 1 / (4 * damping))
d = 1 + 2*damping*theta + theta^2
mKp = 4*damping*theta / d  // Proportional gain
mKi = 4*theta^2 / d         // Integral gain
```

Loop update (lines 342-345):
```java
self._loop_integrator += self._ki * error
phase_adjustment = self._kp * error
self._ted_phase += phase_adjustment
```

**WaveCap-SDR (c4fm.py lines 78-80, 202-205)**
Simple first-order loop:
```python
self._gain_mu = 0.05  # Fixed gain
ted = (y_k - self._last_symbol) * y_mid
self._mu = self._mu + self._gain_mu * ted
```

**Issue**: WaveCap uses simple Gardner with fixed gain, SDRTrunk uses 2nd-order PI controller with calculated coefficients.

**Impact**: HIGH - Poorer timing recovery could cause symbol slips and loss of sync.

### 5. Phase Unwrapping

**SDRTrunk (P25P1DemodulatorC4FM.java lines 145-156)**
```java
// Unwrap phases when reloading buffer
if(mBuffer[x - 1] > 1.5f && mBuffer[x] < -1.5f)
    mBuffer[x] += TWO_PI;
else if(mBuffer[x - 1] < -1.5f && mBuffer[x] > 1.5f)
    mBuffer[x] -= TWO_PI;
```

**WaveCap-SDR**: No phase unwrapping (operates on frequency, not phase)

**Impact**: N/A - WaveCap works in frequency domain after FM discriminator

### 6. DC Offset Correction

**SDRTrunk (P25P1DemodulatorC4FM.java)**: PLL value acts as DC offset correction (line 881)

**WaveCap-SDR (c4fm.py lines 147-149)**
```python
for i in range(len(inst_freq)):
    self._dc_estimate = self._dc_estimate * (1 - self._dc_alpha) + inst_freq[i] * self._dc_alpha
    inst_freq[i] = inst_freq[i] - self._dc_estimate
```

**Impact**: LOW - Both handle DC offset, different approaches

## Recommendations

### Critical Fixes (High Priority)

1. **Fix Dibit Mapping** (c4fm.py)
   - Current mapping doesn't align with P25 standard
   - Should use phase-based decision like SDRTrunk
   - Fix threshold calculation for proper quadrant mapping

2. **Improve Timing Recovery Loop** (c4fm.py)
   - Implement 2nd-order PI controller like SDRTrunk
   - Calculate Kp and Ki from loop bandwidth
   - Add integrator with anti-windup

3. **Adjust Initial Constellation Gain** (c4fm.py)
   - Change from 1.0 to 1.219 (SDRTrunk proven value)
   - Update gain at sync detection events, not periodic intervals

### Medium Priority

4. **Simplify Interpolation** (c4fm.py)
   - Consider switching from cubic to linear interpolation
   - Linear is proven to work in SDRTrunk with excellent results
   - Reduces computational complexity

### Low Priority

5. **Add Sync-Based Gain Updates**
   - Trigger constellation gain updates on frame sync detection
   - More aligned with signal structure than periodic updates

## Implementation Notes

### Dibit Mapping Fix

The P25 standard defines C4FM as:
```
Symbol Level   Frequency Deviation   Dibit (binary)
+3             +1800 Hz              01 (decimal 1)
+1             +600 Hz               00 (decimal 0)
-1             -600 Hz               10 (decimal 2)
-3             -1800 Hz              11 (decimal 3)
```

WaveCap-SDR currently uses frequency-based slicing with arbitrary thresholds. This needs to be corrected to match the standard mapping.

### Timing Recovery Fix

SDRTrunk's PI controller provides better tracking and stability:
- **Proportional term** (Kp): Fast response to timing errors
- **Integral term** (Ki): Eliminates steady-state error
- **Damping factor**: Critical damping (1.0) prevents oscillation

WaveCap-SDR's simple first-order loop with fixed gain is more susceptible to:
- Timing drift over long captures
- Symbol slips during weak signals
- Slow convergence after sync loss

## References

- **SDRTrunk**: https://github.com/DSheirer/sdrtrunk
  - `P25P1DemodulatorC4FM.java` - Main demodulator
  - `Dibit.java` - Symbol definitions and phase mapping
  - `LinearInterpolator.java` - Fractional sample interpolation

- **P25 Standard**: TIA-102.BAAA-A (Common Air Interface)
  - Section 4.2.2: C4FM modulation
  - Section 4.2.3: Symbol mapping

## Test Cases Needed

1. **Dibit Mapping Validation**
   - Generate known C4FM signal with +3, +1, -1, -3 symbols
   - Verify correct dibit output matches P25 standard
   - Test at various signal levels and SNR

2. **Timing Recovery Stress Test**
   - Long capture (>60 seconds) to test drift
   - Variable sample rate to test lock acquisition
   - Weak signal (-100 dBm) to test tracking

3. **Frame Sync Reliability**
   - Count successful frame sync detections
   - Measure time to first sync after tuning
   - Test sync retention during fading

## Changelog

### 2025-12-22
- Initial comparison analysis
- Identified 6 key differences
- Prioritized fixes based on impact
