"""C4FM (Continuous 4-level FM) demodulator for P25 Phase I.

C4FM is the modulation used by P25 Phase I systems. It transmits 4 frequency
deviation levels at 4800 baud, encoding 2 bits (dibits) per symbol:

    Dibit   Symbol   Phase (radians)   Deviation
    ------  ------   ---------------   ---------
    01      +3       +3π/4             +1800 Hz
    00      +1       +π/4              +600 Hz
    10      -1       -π/4              -600 Hz
    11      -3       -3π/4             -1800 Hz

This implementation is a direct port of SDRTrunk's P25P1DemodulatorC4FM.java
for maximum compatibility with the proven Java implementation.

Reference: TIA-102.BAAA-A (P25 Common Air Interface)
Based on: SDRTrunk P25P1DemodulatorC4FM.java, ScalarFMDemodulator.java
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import signal

# Try to import numba for JIT compilation - fall back gracefully if not available
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create a no-op decorator when numba is not available
    def jit(*args, **kwargs):  # type: ignore
        def decorator(func):  # type: ignore
            return func
        return decorator

logger = logging.getLogger(__name__)

# Import profiler for performance analysis
from wavecapsdr.utils.profiler import get_profiler
_c4fm_profiler = get_profiler("C4FM", enabled=True)


# Constants from SDRTrunk P25P1DemodulatorC4FM.java
SYMBOL_RATE = 4800

# Equalizer constants (from SDRTrunk)
EQUALIZER_LOOP_GAIN = 0.15  # 15% per update
EQUALIZER_MAXIMUM_PLL = np.pi / 3.0  # ±800 Hz max frequency offset
EQUALIZER_MAXIMUM_GAIN = 1.25
EQUALIZER_INITIAL_GAIN = 1.219  # SDRTrunk's empirically determined initial gain


def design_baseband_lpf(
    sample_rate: float,
    passband_hz: float = 5200.0,  # SDRTrunk P25P1DecoderC4FM.getBasebandFilter()
    stopband_hz: float = 6500.0,  # SDRTrunk: tighter stopband for better rejection
    num_taps: int = 63,
) -> np.ndarray:
    """Design baseband low-pass filter for P25 C4FM.

    This filter removes out-of-band noise before FM demodulation.
    P25 C4FM signal bandwidth: ~5.8kHz (4800 baud * 1.2 excess BW).

    Uses SDRTrunk's exact parameters: 5200 Hz passband, 6500 Hz stopband.

    Args:
        sample_rate: Sample rate in Hz
        passband_hz: Passband edge frequency
        stopband_hz: Stopband edge frequency
        num_taps: Filter length (odd number recommended)

    Returns:
        FIR filter coefficients
    """
    # Use Parks-McClellan optimal equiripple FIR filter
    from scipy.signal import remez

    # Normalize frequencies to Nyquist
    nyquist = sample_rate / 2.0
    bands = [0, passband_hz, stopband_hz, nyquist]
    desired = [1, 0]  # passband gain = 1, stopband gain = 0

    try:
        # Design equiripple filter
        h = remez(num_taps, bands, desired, Hz=sample_rate)
    except Exception:
        # Fall back to windowed sinc if remez fails
        h = signal.firwin(num_taps, passband_hz, fs=sample_rate, window='hamming')

    return h.astype(np.float32)


def design_rrc_filter(
    samples_per_symbol: float,
    num_taps: int = 101,
    alpha: float = 0.2,
) -> np.ndarray:
    """Design a Root-Raised Cosine (RRC) matched filter.

    The RRC filter is the standard matched filter for P25 C4FM. Both transmitter
    and receiver use RRC filters, and together they form a Raised Cosine response
    with zero ISI at symbol centers.

    Args:
        samples_per_symbol: Number of samples per symbol
        num_taps: Filter length (odd number for symmetry)
        alpha: Roll-off factor (0.2 for P25 Phase I)

    Returns:
        RRC filter coefficients normalized to unit energy
    """
    # Ensure odd number of taps
    if num_taps % 2 == 0:
        num_taps += 1

    # Time vector centered at zero
    n = np.arange(num_taps) - (num_taps - 1) / 2
    t = n / samples_per_symbol

    # RRC impulse response
    h = np.zeros(num_taps, dtype=np.float64)

    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = (1 - alpha + 4 * alpha / np.pi)
        elif abs(ti) == 1 / (4 * alpha):
            h[i] = (alpha / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
            )
        else:
            num = np.sin(np.pi * ti * (1 - alpha)) + 4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha))
            den = np.pi * ti * (1 - (4 * alpha * ti) ** 2)
            h[i] = num / den

    # Normalize to DC gain of 1.0 to preserve symbol amplitude
    h = h / np.sum(h)

    return h.astype(np.float32)


@dataclass
class Dibit:
    """P25 dibit with ideal phase value."""
    value: int  # 0, 1, 2, or 3
    symbol: float  # +1, +3, -1, or -3
    ideal_phase: float  # Phase in radians

    @staticmethod
    def from_soft_symbol(soft: float) -> 'Dibit':
        """Decide dibit from soft symbol (phase in radians).

        Uses π/2 as decision boundary (SDRTrunk compatible).
        """
        boundary = np.pi / 2.0

        if soft >= boundary:
            return Dibit(1, 3.0, 3.0 * np.pi / 4.0)  # +3
        elif soft >= 0:
            return Dibit(0, 1.0, np.pi / 4.0)        # +1
        elif soft >= -boundary:
            return Dibit(2, -1.0, -np.pi / 4.0)      # -1
        else:
            return Dibit(3, -3.0, -3.0 * np.pi / 4.0) # -3


class _Equalizer:
    """Port of SDRTrunk's P25P1DemodulatorC4FM.Equalizer.

    Applies PLL (frequency offset correction) and gain adjustment to
    FM-demodulated phase samples.
    """

    def __init__(self):
        self.pll = 0.0
        self.gain = EQUALIZER_INITIAL_GAIN
        self.initialized = False

    def reset(self):
        """Reset equalizer to initial state."""
        self.pll = 0.0
        self.gain = EQUALIZER_INITIAL_GAIN
        self.initialized = False

    def equalize(self, sample: float) -> float:
        """Apply equalization: (sample + pll) * gain"""
        return (sample + self.pll) * self.gain

    def get_equalized_symbol(
        self,
        buffer: np.ndarray,
        offset: int,
        mu: float
    ) -> float:
        """Equalize samples and interpolate at mu using 8-tap polyphase filter.

        The interpolator uses 8 samples centered around the interpolation point.
        The interpolated value falls between buffer[offset+3] and buffer[offset+4].

        Args:
            buffer: Sample buffer
            offset: Starting offset for 8-sample window
            mu: Fractional position (0.0 to 1.0)

        Returns:
            Equalized, interpolated sample value
        """
        # Use polyphase interpolator
        interpolated = _interpolator.filter(buffer, offset, mu)
        return self.equalize(interpolated)

    def apply_correction(self, pll_adj: float, gain_adj: float):
        """Apply correction with loop gain."""
        if self.initialized:
            self.pll += pll_adj * EQUALIZER_LOOP_GAIN
            self.gain += gain_adj * EQUALIZER_LOOP_GAIN
        else:
            self.pll += pll_adj
            self.gain += gain_adj
            self.initialized = True

        # Constrain values
        self.pll = float(np.clip(self.pll, -EQUALIZER_MAXIMUM_PLL, EQUALIZER_MAXIMUM_PLL))
        self.gain = float(np.clip(self.gain, 1.0, EQUALIZER_MAXIMUM_GAIN))


class _FMDemodulator:
    """Symbol-spaced differential demodulator for C4FM.

    Unlike sample-by-sample FM demodulation, this compares samples that are
    approximately one symbol period apart. This gives full symbol phase values
    (±π/4, ±3π/4 for C4FM) instead of tiny per-sample phase increments.

    Reference: SDRTrunk DifferentialDemodulatorFloatScalar.java
    """

    def __init__(self, symbol_delay: int = 10):
        """Initialize demodulator.

        Args:
            symbol_delay: Number of samples between compared I/Q pairs.
                         Should be approximately samples_per_symbol.
        """
        self.symbol_delay = max(1, symbol_delay)
        # Buffer to hold previous samples for delay line
        self._i_buffer = np.zeros(self.symbol_delay, dtype=np.float32)
        self._q_buffer = np.zeros(self.symbol_delay, dtype=np.float32)
        self._buffer_pos = 0
        self._buffer_filled = False

    def reset(self):
        self._i_buffer.fill(0)
        self._q_buffer.fill(0)
        self._buffer_pos = 0
        self._buffer_filled = False

    def demodulate(self, i: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Demodulate I/Q samples to phase values using symbol-spaced differential.

        Computes phase = atan2(Im(s[n] * conj(s[n-delay])), Re(s[n] * conj(s[n-delay])))
        where delay ≈ samples_per_symbol.

        This gives symbol phase values in the range [-π, π], with C4FM symbols
        appearing at ±π/4 and ±3π/4.

        OPTIMIZED: Vectorized implementation using numpy operations instead of
        per-sample Python loop. Provides ~10-20x speedup.
        """
        n = len(i)
        if n == 0:
            return np.array([], dtype=np.float32)

        delay = self.symbol_delay

        # Build delayed signal by concatenating history buffer with current samples
        # The delayed signal is the previous 'delay' samples shifted
        i_extended = np.concatenate([self._i_buffer, i])
        q_extended = np.concatenate([self._q_buffer, q])

        # Delayed samples: samples from 0 to n (exclusive), which are delay samples behind
        i_delayed = i_extended[:n]
        q_delayed = q_extended[:n]

        # Update circular buffer with last 'delay' samples for next call
        # We need to preserve the circular buffer state properly
        if n >= delay:
            # Take the last 'delay' samples from current input
            self._i_buffer[:] = i[-delay:]
            self._q_buffer[:] = q[-delay:]
            self._buffer_pos = 0
            self._buffer_filled = True
        else:
            # n < delay: need to shift buffer and add new samples
            # Current buffer has samples at positions [buffer_pos, buffer_pos+1, ..., buffer_pos+delay-1] (mod delay)
            # After processing n samples, we need buffer to contain the last 'delay' samples
            # which are: buffer[n:delay] + i[0:n]
            new_buf_i = np.concatenate([self._i_buffer[n:], i])
            new_buf_q = np.concatenate([self._q_buffer[n:], q])
            self._i_buffer[:len(new_buf_i)] = new_buf_i
            self._q_buffer[:len(new_buf_q)] = new_buf_q
            self._buffer_pos = 0

        # Vectorized differential demodulation: s[n] * conj(s[n-delay])
        # Real part: i_curr * i_prev + q_curr * q_prev
        # Imag part: q_curr * i_prev - i_curr * q_prev
        demod_i = i * i_delayed + q * q_delayed
        demod_q = q * i_delayed - i * q_delayed

        # Vectorized arctan2 - single call for entire array
        demodulated = np.arctan2(demod_q, demod_i).astype(np.float32)

        return demodulated


# JIT-compiled interpolator for maximum performance
# This is called for every symbol extraction
@jit(nopython=True, cache=True)
def _interpolate_8tap_jit(samples: np.ndarray, offset: int, taps: np.ndarray) -> float:
    """JIT-compiled 8-tap interpolation.

    Explicit loop allows numba to optimize to SIMD instructions.
    """
    result = 0.0
    for i in range(8):
        result += samples[offset + i] * taps[i]
    return result


# JIT-compiled timing optimizer functions
# These replace the Python _TimingOptimizer methods for ~100x speedup

@jit(nopython=True, cache=True)
def _timing_score_jit(
    buffer: np.ndarray,
    offset: float,
    pll: float,
    gain: float,
    samples_per_symbol: float,
    sync_symbols: np.ndarray,
    interpolator_taps: np.ndarray,
) -> float:
    """JIT-compiled sync correlation score calculation.

    Walks back through 24 symbols from offset and computes correlation
    against sync pattern using 8-tap polyphase interpolation.
    """
    score = 0.0
    max_offset = len(buffer) - 8

    # Walk back through 24 symbols
    ptr = offset - (23.0 * samples_per_symbol)

    for i in range(24):
        buf_idx = int(ptr)
        interp_offset = buf_idx - 3

        if 0 <= interp_offset <= max_offset:
            mu = ptr - buf_idx
            # Inline interpolation and equalization
            mu_inverted = 1.0 - mu
            tap_idx = int(mu_inverted * 128.0 + 0.5)
            if tap_idx < 0:
                tap_idx = 0
            if tap_idx > 128:
                tap_idx = 128

            # 8-tap interpolation
            taps = interpolator_taps[tap_idx]
            interpolated = 0.0
            for j in range(8):
                interpolated += buffer[interp_offset + j] * taps[j]

            # Equalization
            soft = (interpolated + pll) * gain
            score += soft * sync_symbols[i]

        ptr += samples_per_symbol

    return score


@jit(nopython=True, cache=True)
def _timing_correction_jit(
    buffer: np.ndarray,
    offset: float,
    pll: float,
    gain: float,
    samples_per_symbol: float,
    sync_symbols: np.ndarray,
    interpolator_taps: np.ndarray,
) -> tuple[float, float]:
    """JIT-compiled PLL and gain correction calculation.

    Compares resampled sync symbols against ideal values to compute
    correction factors for the equalizer.
    """
    max_offset = len(buffer) - 8
    balance_plus = 0.0
    balance_minus = 0.0
    gain_accum = 0.0
    plus_count = 0
    minus_count = 0

    ptr = offset - (23.0 * samples_per_symbol)

    for i in range(24):
        buf_idx = int(ptr)
        interp_offset = buf_idx - 3

        if 0 <= interp_offset <= max_offset:
            mu = ptr - buf_idx
            # Inline interpolation and equalization
            mu_inverted = 1.0 - mu
            tap_idx = int(mu_inverted * 128.0 + 0.5)
            if tap_idx < 0:
                tap_idx = 0
            if tap_idx > 128:
                tap_idx = 128

            taps = interpolator_taps[tap_idx]
            interpolated = 0.0
            for j in range(8):
                interpolated += buffer[interp_offset + j] * taps[j]

            soft = (interpolated + pll) * gain
            ideal = sync_symbols[i]

            if ideal > 0:
                balance_plus += (soft - ideal)
                plus_count += 1
            else:
                balance_minus += (soft - ideal)
                minus_count += 1

            gain_accum += abs(ideal) - abs(soft)

        ptr += samples_per_symbol

    # Average the corrections
    if plus_count > 0:
        balance_plus /= -plus_count
    if minus_count > 0:
        balance_minus /= -minus_count

    pll_correction = (balance_plus + balance_minus) / 2.0
    # Clip to ±π/2
    half_pi = 1.5707963267948966
    if pll_correction < -half_pi:
        pll_correction = -half_pi
    elif pll_correction > half_pi:
        pll_correction = half_pi

    # Gain correction normalized to ideal +3 symbol value (3π/4)
    gain_correction = gain_accum / (24.0 * 2.356194490192345)  # 3π/4

    return pll_correction, gain_correction


@jit(nopython=True, cache=True)
def _timing_optimize_jit(
    buffer: np.ndarray,
    buffer_offset: float,
    pll: float,
    gain: float,
    samples_per_symbol: float,
    sync_symbols: np.ndarray,
    interpolator_taps: np.ndarray,
    fine_sync: bool,
) -> tuple[float, float, float, float]:
    """JIT-compiled hill climbing optimization for sync timing.

    Searches ±½ symbol period (or full symbol in fine mode) to find
    the timing offset that maximizes sync correlation score.

    Returns: (timing_adjustment, optimized_score, pll_adj, gain_adj)
    """
    sps = samples_per_symbol

    # Search parameters
    if fine_sync:
        step_size = sps / 16.0
        step_size_min = sps / 200.0
        max_adjustment = sps
    else:
        step_size = sps / 8.0
        step_size_min = sps / 200.0
        max_adjustment = sps / 2.0

    adjustment = 0.0
    offset = buffer_offset

    # Score at center
    score_center = _timing_score_jit(
        buffer, offset, pll, gain, sps, sync_symbols, interpolator_taps
    )

    # Score left and right
    score_left = _timing_score_jit(
        buffer, offset - step_size, pll, gain, sps, sync_symbols, interpolator_taps
    )
    score_right = _timing_score_jit(
        buffer, offset + step_size, pll, gain, sps, sync_symbols, interpolator_taps
    )

    # Hill climbing search
    while step_size > step_size_min and abs(adjustment) <= max_adjustment:
        if score_left > score_right and score_left > score_center:
            adjustment -= step_size
            score_right = score_center
            score_center = score_left
            score_left = _timing_score_jit(
                buffer, offset + adjustment - step_size, pll, gain, sps,
                sync_symbols, interpolator_taps
            )
        elif score_right > score_left and score_right > score_center:
            adjustment += step_size
            score_left = score_center
            score_center = score_right
            score_right = _timing_score_jit(
                buffer, offset + adjustment + step_size, pll, gain, sps,
                sync_symbols, interpolator_taps
            )
        else:
            step_size *= 0.5
            if step_size > step_size_min:
                score_left = _timing_score_jit(
                    buffer, offset + adjustment - step_size, pll, gain, sps,
                    sync_symbols, interpolator_taps
                )
                score_right = _timing_score_jit(
                    buffer, offset + adjustment + step_size, pll, gain, sps,
                    sync_symbols, interpolator_taps
                )

    # Calculate equalizer correction from optimized position
    pll_adj, gain_adj = _timing_correction_jit(
        buffer, offset + adjustment, pll, gain, sps, sync_symbols, interpolator_taps
    )

    return adjustment, score_center, pll_adj, gain_adj


# JIT-compiled symbol recovery - the main performance bottleneck
# This processes all phases and returns dibits/soft_symbols in one fast pass
@jit(nopython=True, cache=True)
def _symbol_recovery_jit(
    phases: np.ndarray,
    buffer: np.ndarray,
    buffer_pointer: int,
    sample_point: float,
    samples_per_symbol: float,
    pll: float,
    gain: float,
    interpolator_taps: np.ndarray,  # Shape: (129, 8)
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """JIT-compiled symbol recovery loop.

    Processes all phase samples through:
    1. Buffer management (circular buffer with shift)
    2. Symbol timing (sample_point countdown)
    3. 8-tap polyphase interpolation
    4. Equalization (PLL + gain)
    5. Dibit decision (π/2 boundaries)

    Args:
        phases: FM-demodulated phase values
        buffer: Sample buffer (will be modified in place)
        buffer_pointer: Current buffer write position
        sample_point: Countdown to next symbol decision
        samples_per_symbol: Samples per symbol (e.g., 10.4)
        pll: Current PLL offset
        gain: Current gain value
        interpolator_taps: Polyphase filter coefficients (129 x 8)

    Returns:
        Tuple of:
        - dibits: Hard decision dibits (0-3)
        - soft_symbols: Soft symbol values (normalized ±1, ±3)
        - symbol_indices: Buffer indices where symbols were extracted
        - buffer_pointer: Updated buffer pointer
        - sample_point: Updated sample point
    """
    n_phases = len(phases)
    buffer_len = len(buffer)
    half_buffer = buffer_len // 2

    # Pre-allocate output arrays (max possible symbols)
    max_symbols = n_phases // 4 + 10  # Conservative upper bound
    dibits = np.empty(max_symbols, dtype=np.uint8)
    soft_symbols = np.empty(max_symbols, dtype=np.float32)
    symbol_indices = np.empty(max_symbols, dtype=np.int32)
    symbol_count = 0

    # Decision boundary
    boundary = 1.5707963267948966  # np.pi / 2.0

    for phase in phases:
        buffer_pointer += 1
        sample_point -= 1.0

        # Buffer management - shift when near end
        if buffer_pointer >= buffer_len - 1:
            # Shift buffer left by half
            for i in range(half_buffer):
                buffer[i] = buffer[i + half_buffer]
            for i in range(half_buffer, buffer_len):
                buffer[i] = 0.0
            buffer_pointer -= half_buffer

        # Store sample
        buffer[buffer_pointer] = phase

        # Symbol decision point
        if sample_point < 1.0:
            idx = buffer_pointer
            mu = 1.0 - sample_point

            # Compute interpolator offset
            interp_offset = idx - 4

            # Ensure we have enough samples for interpolation
            if interp_offset >= 0 and idx < buffer_len:
                # Convert mu to tap index (inverted for TAPS convention)
                mu_inverted = 1.0 - mu
                tap_idx = int(mu_inverted * 128.0 + 0.5)
                if tap_idx < 0:
                    tap_idx = 0
                if tap_idx > 128:
                    tap_idx = 128

                # 8-tap interpolation
                taps = interpolator_taps[tap_idx]
                interpolated = 0.0
                for i in range(8):
                    interpolated += buffer[interp_offset + i] * taps[i]

                # Equalization: (sample + pll) * gain
                soft_symbol_rad = (interpolated + pll) * gain

                # Dibit decision using π/2 boundaries
                if soft_symbol_rad >= boundary:
                    dibit = 1  # +3
                elif soft_symbol_rad >= 0:
                    dibit = 0  # +1
                elif soft_symbol_rad >= -boundary:
                    dibit = 2  # -1
                else:
                    dibit = 3  # -3

                # Normalize to ±1, ±3 scale
                soft_symbol_norm = soft_symbol_rad * 1.2732395447351628  # 4.0 / np.pi

                # Store results
                dibits[symbol_count] = dibit
                soft_symbols[symbol_count] = soft_symbol_norm
                symbol_indices[symbol_count] = idx
                symbol_count += 1

            # Advance to next symbol
            sample_point += samples_per_symbol

    # Return only the filled portions
    return (
        dibits[:symbol_count],
        soft_symbols[:symbol_count],
        symbol_indices[:symbol_count],
        buffer_pointer,
        sample_point,
    )


class _Interpolator:
    """8-tap polyphase interpolator for precise sub-sample timing.

    This is a direct port of SDRTrunk's Interpolator.java. The interpolator
    uses 8 samples centered around the interpolation point with 128 precomputed
    coefficient sets for sub-sample positions.

    Reference: SDRTrunk Interpolator.java
    """

    NTAPS = 8
    NSTEPS = 128

    # Precomputed filter coefficients for 128 fractional positions
    # Each row is 8 coefficients for taps at positions -4, -3, -2, -1, 0, 1, 2, 3
    # mu=0 uses row 0 (centered on sample -1), mu=1 uses row 128 (centered on sample 0)
    TAPS = np.array([
        #    -4            -3            -2            -1             0             1             2             3        mu
        [  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00 ], #   0/128
        [ -1.54700e-04,  8.53777e-04, -2.76968e-03,  7.89295e-03,  9.98534e-01, -5.41054e-03,  1.24642e-03, -1.98993e-04 ], #   1/128
        [ -3.09412e-04,  1.70888e-03, -5.55134e-03,  1.58840e-02,  9.96891e-01, -1.07209e-02,  2.47942e-03, -3.96391e-04 ], #   2/128
        [ -4.64053e-04,  2.56486e-03, -8.34364e-03,  2.39714e-02,  9.95074e-01, -1.59305e-02,  3.69852e-03, -5.92100e-04 ], #   3/128
        [ -6.18544e-04,  3.42130e-03, -1.11453e-02,  3.21531e-02,  9.93082e-01, -2.10389e-02,  4.90322e-03, -7.86031e-04 ], #   4/128
        [ -7.72802e-04,  4.27773e-03, -1.39548e-02,  4.04274e-02,  9.90917e-01, -2.60456e-02,  6.09305e-03, -9.78093e-04 ], #   5/128
        [ -9.26747e-04,  5.13372e-03, -1.67710e-02,  4.87921e-02,  9.88580e-01, -3.09503e-02,  7.26755e-03, -1.16820e-03 ], #   6/128
        [ -1.08030e-03,  5.98883e-03, -1.95925e-02,  5.72454e-02,  9.86071e-01, -3.57525e-02,  8.42626e-03, -1.35627e-03 ], #   7/128
        [ -1.23337e-03,  6.84261e-03, -2.24178e-02,  6.57852e-02,  9.83392e-01, -4.04519e-02,  9.56876e-03, -1.54221e-03 ], #   8/128
        [ -1.38589e-03,  7.69462e-03, -2.52457e-02,  7.44095e-02,  9.80543e-01, -4.50483e-02,  1.06946e-02, -1.72594e-03 ], #   9/128
        [ -1.53777e-03,  8.54441e-03, -2.80746e-02,  8.31162e-02,  9.77526e-01, -4.95412e-02,  1.18034e-02, -1.90738e-03 ], #  10/128
        [ -1.68894e-03,  9.39154e-03, -3.09033e-02,  9.19033e-02,  9.74342e-01, -5.39305e-02,  1.28947e-02, -2.08645e-03 ], #  11/128
        [ -1.83931e-03,  1.02356e-02, -3.37303e-02,  1.00769e-01,  9.70992e-01, -5.82159e-02,  1.39681e-02, -2.26307e-03 ], #  12/128
        [ -1.98880e-03,  1.10760e-02, -3.65541e-02,  1.09710e-01,  9.67477e-01, -6.23972e-02,  1.50233e-02, -2.43718e-03 ], #  13/128
        [ -2.13733e-03,  1.19125e-02, -3.93735e-02,  1.18725e-01,  9.63798e-01, -6.64743e-02,  1.60599e-02, -2.60868e-03 ], #  14/128
        [ -2.28483e-03,  1.27445e-02, -4.21869e-02,  1.27812e-01,  9.59958e-01, -7.04471e-02,  1.70776e-02, -2.77751e-03 ], #  15/128
        [ -2.43121e-03,  1.35716e-02, -4.49929e-02,  1.36968e-01,  9.55956e-01, -7.43154e-02,  1.80759e-02, -2.94361e-03 ], #  16/128
        [ -2.57640e-03,  1.43934e-02, -4.77900e-02,  1.46192e-01,  9.51795e-01, -7.80792e-02,  1.90545e-02, -3.10689e-03 ], #  17/128
        [ -2.72032e-03,  1.52095e-02, -5.05770e-02,  1.55480e-01,  9.47477e-01, -8.17385e-02,  2.00132e-02, -3.26730e-03 ], #  18/128
        [ -2.86289e-03,  1.60193e-02, -5.33522e-02,  1.64831e-01,  9.43001e-01, -8.52933e-02,  2.09516e-02, -3.42477e-03 ], #  19/128
        [ -3.00403e-03,  1.68225e-02, -5.61142e-02,  1.74242e-01,  9.38371e-01, -8.87435e-02,  2.18695e-02, -3.57923e-03 ], #  20/128
        [ -3.14367e-03,  1.76185e-02, -5.88617e-02,  1.83711e-01,  9.33586e-01, -9.20893e-02,  2.27664e-02, -3.73062e-03 ], #  21/128
        [ -3.28174e-03,  1.84071e-02, -6.15931e-02,  1.93236e-01,  9.28650e-01, -9.53307e-02,  2.36423e-02, -3.87888e-03 ], #  22/128
        [ -3.41815e-03,  1.91877e-02, -6.43069e-02,  2.02814e-01,  9.23564e-01, -9.84679e-02,  2.44967e-02, -4.02397e-03 ], #  23/128
        [ -3.55283e-03,  1.99599e-02, -6.70018e-02,  2.12443e-01,  9.18329e-01, -1.01501e-01,  2.53295e-02, -4.16581e-03 ], #  24/128
        [ -3.68570e-03,  2.07233e-02, -6.96762e-02,  2.22120e-01,  9.12947e-01, -1.04430e-01,  2.61404e-02, -4.30435e-03 ], #  25/128
        [ -3.81671e-03,  2.14774e-02, -7.23286e-02,  2.31843e-01,  9.07420e-01, -1.07256e-01,  2.69293e-02, -4.43955e-03 ], #  26/128
        [ -3.94576e-03,  2.22218e-02, -7.49577e-02,  2.41609e-01,  9.01749e-01, -1.09978e-01,  2.76957e-02, -4.57135e-03 ], #  27/128
        [ -4.07279e-03,  2.29562e-02, -7.75620e-02,  2.51417e-01,  8.95936e-01, -1.12597e-01,  2.84397e-02, -4.69970e-03 ], #  28/128
        [ -4.19774e-03,  2.36801e-02, -8.01399e-02,  2.61263e-01,  8.89984e-01, -1.15113e-01,  2.91609e-02, -4.82456e-03 ], #  29/128
        [ -4.32052e-03,  2.43930e-02, -8.26900e-02,  2.71144e-01,  8.83893e-01, -1.17526e-01,  2.98593e-02, -4.94589e-03 ], #  30/128
        [ -4.44107e-03,  2.50946e-02, -8.52109e-02,  2.81060e-01,  8.77666e-01, -1.19837e-01,  3.05345e-02, -5.06363e-03 ], #  31/128
        [ -4.55932e-03,  2.57844e-02, -8.77011e-02,  2.91006e-01,  8.71305e-01, -1.22047e-01,  3.11866e-02, -5.17776e-03 ], #  32/128
        [ -4.67520e-03,  2.64621e-02, -9.01591e-02,  3.00980e-01,  8.64812e-01, -1.24154e-01,  3.18153e-02, -5.28823e-03 ], #  33/128
        [ -4.78866e-03,  2.71272e-02, -9.25834e-02,  3.10980e-01,  8.58189e-01, -1.26161e-01,  3.24205e-02, -5.39500e-03 ], #  34/128
        [ -4.89961e-03,  2.77794e-02, -9.49727e-02,  3.21004e-01,  8.51437e-01, -1.28068e-01,  3.30021e-02, -5.49804e-03 ], #  35/128
        [ -5.00800e-03,  2.84182e-02, -9.73254e-02,  3.31048e-01,  8.44559e-01, -1.29874e-01,  3.35600e-02, -5.59731e-03 ], #  36/128
        [ -5.11376e-03,  2.90433e-02, -9.96402e-02,  3.41109e-01,  8.37557e-01, -1.31581e-01,  3.40940e-02, -5.69280e-03 ], #  37/128
        [ -5.21683e-03,  2.96543e-02, -1.01915e-01,  3.51186e-01,  8.30432e-01, -1.33189e-01,  3.46042e-02, -5.78446e-03 ], #  38/128
        [ -5.31716e-03,  3.02507e-02, -1.04150e-01,  3.61276e-01,  8.23188e-01, -1.34699e-01,  3.50903e-02, -5.87227e-03 ], #  39/128
        [ -5.41467e-03,  3.08323e-02, -1.06342e-01,  3.71376e-01,  8.15826e-01, -1.36111e-01,  3.55525e-02, -5.95620e-03 ], #  40/128
        [ -5.50931e-03,  3.13987e-02, -1.08490e-01,  3.81484e-01,  8.08348e-01, -1.37426e-01,  3.59905e-02, -6.03624e-03 ], #  41/128
        [ -5.60103e-03,  3.19495e-02, -1.10593e-01,  3.91596e-01,  8.00757e-01, -1.38644e-01,  3.64044e-02, -6.11236e-03 ], #  42/128
        [ -5.68976e-03,  3.24843e-02, -1.12650e-01,  4.01710e-01,  7.93055e-01, -1.39767e-01,  3.67941e-02, -6.18454e-03 ], #  43/128
        [ -5.77544e-03,  3.30027e-02, -1.14659e-01,  4.11823e-01,  7.85244e-01, -1.40794e-01,  3.71596e-02, -6.25277e-03 ], #  44/128
        [ -5.85804e-03,  3.35046e-02, -1.16618e-01,  4.21934e-01,  7.77327e-01, -1.41727e-01,  3.75010e-02, -6.31703e-03 ], #  45/128
        [ -5.93749e-03,  3.39894e-02, -1.18526e-01,  4.32038e-01,  7.69305e-01, -1.42566e-01,  3.78182e-02, -6.37730e-03 ], #  46/128
        [ -6.01374e-03,  3.44568e-02, -1.20382e-01,  4.42134e-01,  7.61181e-01, -1.43313e-01,  3.81111e-02, -6.43358e-03 ], #  47/128
        [ -6.08674e-03,  3.49066e-02, -1.22185e-01,  4.52218e-01,  7.52958e-01, -1.43968e-01,  3.83800e-02, -6.48585e-03 ], #  48/128
        [ -6.15644e-03,  3.53384e-02, -1.23933e-01,  4.62289e-01,  7.44637e-01, -1.44531e-01,  3.86247e-02, -6.53412e-03 ], #  49/128
        [ -6.22280e-03,  3.57519e-02, -1.25624e-01,  4.72342e-01,  7.36222e-01, -1.45004e-01,  3.88454e-02, -6.57836e-03 ], #  50/128
        [ -6.28577e-03,  3.61468e-02, -1.27258e-01,  4.82377e-01,  7.27714e-01, -1.45387e-01,  3.90420e-02, -6.61859e-03 ], #  51/128
        [ -6.34530e-03,  3.65227e-02, -1.28832e-01,  4.92389e-01,  7.19116e-01, -1.45682e-01,  3.92147e-02, -6.65479e-03 ], #  52/128
        [ -6.40135e-03,  3.68795e-02, -1.30347e-01,  5.02377e-01,  7.10431e-01, -1.45889e-01,  3.93636e-02, -6.68698e-03 ], #  53/128
        [ -6.45388e-03,  3.72167e-02, -1.31800e-01,  5.12337e-01,  7.01661e-01, -1.46009e-01,  3.94886e-02, -6.71514e-03 ], #  54/128
        [ -6.50285e-03,  3.75341e-02, -1.33190e-01,  5.22267e-01,  6.92808e-01, -1.46043e-01,  3.95900e-02, -6.73929e-03 ], #  55/128
        [ -6.54823e-03,  3.78315e-02, -1.34515e-01,  5.32164e-01,  6.83875e-01, -1.45993e-01,  3.96678e-02, -6.75943e-03 ], #  56/128
        [ -6.58996e-03,  3.81085e-02, -1.35775e-01,  5.42025e-01,  6.74865e-01, -1.45859e-01,  3.97222e-02, -6.77557e-03 ], #  57/128
        [ -6.62802e-03,  3.83650e-02, -1.36969e-01,  5.51849e-01,  6.65779e-01, -1.45641e-01,  3.97532e-02, -6.78771e-03 ], #  58/128
        [ -6.66238e-03,  3.86006e-02, -1.38094e-01,  5.61631e-01,  6.56621e-01, -1.45343e-01,  3.97610e-02, -6.79588e-03 ], #  59/128
        [ -6.69300e-03,  3.88151e-02, -1.39150e-01,  5.71370e-01,  6.47394e-01, -1.44963e-01,  3.97458e-02, -6.80007e-03 ], #  60/128
        [ -6.71985e-03,  3.90083e-02, -1.40136e-01,  5.81063e-01,  6.38099e-01, -1.44503e-01,  3.97077e-02, -6.80032e-03 ], #  61/128
        [ -6.74291e-03,  3.91800e-02, -1.41050e-01,  5.90706e-01,  6.28739e-01, -1.43965e-01,  3.96469e-02, -6.79662e-03 ], #  62/128
        [ -6.76214e-03,  3.93299e-02, -1.41891e-01,  6.00298e-01,  6.19318e-01, -1.43350e-01,  3.95635e-02, -6.78902e-03 ], #  63/128
        [ -6.77751e-03,  3.94578e-02, -1.42658e-01,  6.09836e-01,  6.09836e-01, -1.42658e-01,  3.94578e-02, -6.77751e-03 ], #  64/128
        [ -6.78902e-03,  3.95635e-02, -1.43350e-01,  6.19318e-01,  6.00298e-01, -1.41891e-01,  3.93299e-02, -6.76214e-03 ], #  65/128
        [ -6.79662e-03,  3.96469e-02, -1.43965e-01,  6.28739e-01,  5.90706e-01, -1.41050e-01,  3.91800e-02, -6.74291e-03 ], #  66/128
        [ -6.80032e-03,  3.97077e-02, -1.44503e-01,  6.38099e-01,  5.81063e-01, -1.40136e-01,  3.90083e-02, -6.71985e-03 ], #  67/128
        [ -6.80007e-03,  3.97458e-02, -1.44963e-01,  6.47394e-01,  5.71370e-01, -1.39150e-01,  3.88151e-02, -6.69300e-03 ], #  68/128
        [ -6.79588e-03,  3.97610e-02, -1.45343e-01,  6.56621e-01,  5.61631e-01, -1.38094e-01,  3.86006e-02, -6.66238e-03 ], #  69/128
        [ -6.78771e-03,  3.97532e-02, -1.45641e-01,  6.65779e-01,  5.51849e-01, -1.36969e-01,  3.83650e-02, -6.62802e-03 ], #  70/128
        [ -6.77557e-03,  3.97222e-02, -1.45859e-01,  6.74865e-01,  5.42025e-01, -1.35775e-01,  3.81085e-02, -6.58996e-03 ], #  71/128
        [ -6.75943e-03,  3.96678e-02, -1.45993e-01,  6.83875e-01,  5.32164e-01, -1.34515e-01,  3.78315e-02, -6.54823e-03 ], #  72/128
        [ -6.73929e-03,  3.95900e-02, -1.46043e-01,  6.92808e-01,  5.22267e-01, -1.33190e-01,  3.75341e-02, -6.50285e-03 ], #  73/128
        [ -6.71514e-03,  3.94886e-02, -1.46009e-01,  7.01661e-01,  5.12337e-01, -1.31800e-01,  3.72167e-02, -6.45388e-03 ], #  74/128
        [ -6.68698e-03,  3.93636e-02, -1.45889e-01,  7.10431e-01,  5.02377e-01, -1.30347e-01,  3.68795e-02, -6.40135e-03 ], #  75/128
        [ -6.65479e-03,  3.92147e-02, -1.45682e-01,  7.19116e-01,  4.92389e-01, -1.28832e-01,  3.65227e-02, -6.34530e-03 ], #  76/128
        [ -6.61859e-03,  3.90420e-02, -1.45387e-01,  7.27714e-01,  4.82377e-01, -1.27258e-01,  3.61468e-02, -6.28577e-03 ], #  77/128
        [ -6.57836e-03,  3.88454e-02, -1.45004e-01,  7.36222e-01,  4.72342e-01, -1.25624e-01,  3.57519e-02, -6.22280e-03 ], #  78/128
        [ -6.53412e-03,  3.86247e-02, -1.44531e-01,  7.44637e-01,  4.62289e-01, -1.23933e-01,  3.53384e-02, -6.15644e-03 ], #  79/128
        [ -6.48585e-03,  3.83800e-02, -1.43968e-01,  7.52958e-01,  4.52218e-01, -1.22185e-01,  3.49066e-02, -6.08674e-03 ], #  80/128
        [ -6.43358e-03,  3.81111e-02, -1.43313e-01,  7.61181e-01,  4.42134e-01, -1.20382e-01,  3.44568e-02, -6.01374e-03 ], #  81/128
        [ -6.37730e-03,  3.78182e-02, -1.42566e-01,  7.69305e-01,  4.32038e-01, -1.18526e-01,  3.39894e-02, -5.93749e-03 ], #  82/128
        [ -6.31703e-03,  3.75010e-02, -1.41727e-01,  7.77327e-01,  4.21934e-01, -1.16618e-01,  3.35046e-02, -5.85804e-03 ], #  83/128
        [ -6.25277e-03,  3.71596e-02, -1.40794e-01,  7.85244e-01,  4.11823e-01, -1.14659e-01,  3.30027e-02, -5.77544e-03 ], #  84/128
        [ -6.18454e-03,  3.67941e-02, -1.39767e-01,  7.93055e-01,  4.01710e-01, -1.12650e-01,  3.24843e-02, -5.68976e-03 ], #  85/128
        [ -6.11236e-03,  3.64044e-02, -1.38644e-01,  8.00757e-01,  3.91596e-01, -1.10593e-01,  3.19495e-02, -5.60103e-03 ], #  86/128
        [ -6.03624e-03,  3.59905e-02, -1.37426e-01,  8.08348e-01,  3.81484e-01, -1.08490e-01,  3.13987e-02, -5.50931e-03 ], #  87/128
        [ -5.95620e-03,  3.55525e-02, -1.36111e-01,  8.15826e-01,  3.71376e-01, -1.06342e-01,  3.08323e-02, -5.41467e-03 ], #  88/128
        [ -5.87227e-03,  3.50903e-02, -1.34699e-01,  8.23188e-01,  3.61276e-01, -1.04150e-01,  3.02507e-02, -5.31716e-03 ], #  89/128
        [ -5.78446e-03,  3.46042e-02, -1.33189e-01,  8.30432e-01,  3.51186e-01, -1.01915e-01,  2.96543e-02, -5.21683e-03 ], #  90/128
        [ -5.69280e-03,  3.40940e-02, -1.31581e-01,  8.37557e-01,  3.41109e-01, -9.96402e-02,  2.90433e-02, -5.11376e-03 ], #  91/128
        [ -5.59731e-03,  3.35600e-02, -1.29874e-01,  8.44559e-01,  3.31048e-01, -9.73254e-02,  2.84182e-02, -5.00800e-03 ], #  92/128
        [ -5.49804e-03,  3.30021e-02, -1.28068e-01,  8.51437e-01,  3.21004e-01, -9.49727e-02,  2.77794e-02, -4.89961e-03 ], #  93/128
        [ -5.39500e-03,  3.24205e-02, -1.26161e-01,  8.58189e-01,  3.10980e-01, -9.25834e-02,  2.71272e-02, -4.78866e-03 ], #  94/128
        [ -5.28823e-03,  3.18153e-02, -1.24154e-01,  8.64812e-01,  3.00980e-01, -9.01591e-02,  2.64621e-02, -4.67520e-03 ], #  95/128
        [ -5.17776e-03,  3.11866e-02, -1.22047e-01,  8.71305e-01,  2.91006e-01, -8.77011e-02,  2.57844e-02, -4.55932e-03 ], #  96/128
        [ -5.06363e-03,  3.05345e-02, -1.19837e-01,  8.77666e-01,  2.81060e-01, -8.52109e-02,  2.50946e-02, -4.44107e-03 ], #  97/128
        [ -4.94589e-03,  2.98593e-02, -1.17526e-01,  8.83893e-01,  2.71144e-01, -8.26900e-02,  2.43930e-02, -4.32052e-03 ], #  98/128
        [ -4.82456e-03,  2.91609e-02, -1.15113e-01,  8.89984e-01,  2.61263e-01, -8.01399e-02,  2.36801e-02, -4.19774e-03 ], #  99/128
        [ -4.69970e-03,  2.84397e-02, -1.12597e-01,  8.95936e-01,  2.51417e-01, -7.75620e-02,  2.29562e-02, -4.07279e-03 ], # 100/128
        [ -4.57135e-03,  2.76957e-02, -1.09978e-01,  9.01749e-01,  2.41609e-01, -7.49577e-02,  2.22218e-02, -3.94576e-03 ], # 101/128
        [ -4.43955e-03,  2.69293e-02, -1.07256e-01,  9.07420e-01,  2.31843e-01, -7.23286e-02,  2.14774e-02, -3.81671e-03 ], # 102/128
        [ -4.30435e-03,  2.61404e-02, -1.04430e-01,  9.12947e-01,  2.22120e-01, -6.96762e-02,  2.07233e-02, -3.68570e-03 ], # 103/128
        [ -4.16581e-03,  2.53295e-02, -1.01501e-01,  9.18329e-01,  2.12443e-01, -6.70018e-02,  1.99599e-02, -3.55283e-03 ], # 104/128
        [ -4.02397e-03,  2.44967e-02, -9.84679e-02,  9.23564e-01,  2.02814e-01, -6.43069e-02,  1.91877e-02, -3.41815e-03 ], # 105/128
        [ -3.87888e-03,  2.36423e-02, -9.53307e-02,  9.28650e-01,  1.93236e-01, -6.15931e-02,  1.84071e-02, -3.28174e-03 ], # 106/128
        [ -3.73062e-03,  2.27664e-02, -9.20893e-02,  9.33586e-01,  1.83711e-01, -5.88617e-02,  1.76185e-02, -3.14367e-03 ], # 107/128
        [ -3.57923e-03,  2.18695e-02, -8.87435e-02,  9.38371e-01,  1.74242e-01, -5.61142e-02,  1.68225e-02, -3.00403e-03 ], # 108/128
        [ -3.42477e-03,  2.09516e-02, -8.52933e-02,  9.43001e-01,  1.64831e-01, -5.33522e-02,  1.60193e-02, -2.86289e-03 ], # 109/128
        [ -3.26730e-03,  2.00132e-02, -8.17385e-02,  9.47477e-01,  1.55480e-01, -5.05770e-02,  1.52095e-02, -2.72032e-03 ], # 110/128
        [ -3.10689e-03,  1.90545e-02, -7.80792e-02,  9.51795e-01,  1.46192e-01, -4.77900e-02,  1.43934e-02, -2.57640e-03 ], # 111/128
        [ -2.94361e-03,  1.80759e-02, -7.43154e-02,  9.55956e-01,  1.36968e-01, -4.49929e-02,  1.35716e-02, -2.43121e-03 ], # 112/128
        [ -2.77751e-03,  1.70776e-02, -7.04471e-02,  9.59958e-01,  1.27812e-01, -4.21869e-02,  1.27445e-02, -2.28483e-03 ], # 113/128
        [ -2.60868e-03,  1.60599e-02, -6.64743e-02,  9.63798e-01,  1.18725e-01, -3.93735e-02,  1.19125e-02, -2.13733e-03 ], # 114/128
        [ -2.43718e-03,  1.50233e-02, -6.23972e-02,  9.67477e-01,  1.09710e-01, -3.65541e-02,  1.10760e-02, -1.98880e-03 ], # 115/128
        [ -2.26307e-03,  1.39681e-02, -5.82159e-02,  9.70992e-01,  1.00769e-01, -3.37303e-02,  1.02356e-02, -1.83931e-03 ], # 116/128
        [ -2.08645e-03,  1.28947e-02, -5.39305e-02,  9.74342e-01,  9.19033e-02, -3.09033e-02,  9.39154e-03, -1.68894e-03 ], # 117/128
        [ -1.90738e-03,  1.18034e-02, -4.95412e-02,  9.77526e-01,  8.31162e-02, -2.80746e-02,  8.54441e-03, -1.53777e-03 ], # 118/128
        [ -1.72594e-03,  1.06946e-02, -4.50483e-02,  9.80543e-01,  7.44095e-02, -2.52457e-02,  7.69462e-03, -1.38589e-03 ], # 119/128
        [ -1.54221e-03,  9.56876e-03, -4.04519e-02,  9.83392e-01,  6.57852e-02, -2.24178e-02,  6.84261e-03, -1.23337e-03 ], # 120/128
        [ -1.35627e-03,  8.42626e-03, -3.57525e-02,  9.86071e-01,  5.72454e-02, -1.95925e-02,  5.98883e-03, -1.08030e-03 ], # 121/128
        [ -1.16820e-03,  7.26755e-03, -3.09503e-02,  9.88580e-01,  4.87921e-02, -1.67710e-02,  5.13372e-03, -9.26747e-04 ], # 122/128
        [ -9.78093e-04,  6.09305e-03, -2.60456e-02,  9.90917e-01,  4.04274e-02, -1.39548e-02,  4.27773e-03, -7.72802e-04 ], # 123/128
        [ -7.86031e-04,  4.90322e-03, -2.10389e-02,  9.93082e-01,  3.21531e-02, -1.11453e-02,  3.42130e-03, -6.18544e-04 ], # 124/128
        [ -5.92100e-04,  3.69852e-03, -1.59305e-02,  9.95074e-01,  2.39714e-02, -8.34364e-03,  2.56486e-03, -4.64053e-04 ], # 125/128
        [ -3.96391e-04,  2.47942e-03, -1.07209e-02,  9.96891e-01,  1.58840e-02, -5.55134e-03,  1.70888e-03, -3.09412e-04 ], # 126/128
        [ -1.98993e-04,  1.24642e-03, -5.41054e-03,  9.98534e-01,  7.89295e-03, -2.76968e-03,  8.53777e-04, -1.54700e-04 ], # 127/128
        [  0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00 ], # 128/128
    ], dtype=np.float32)

    def filter(self, samples: np.ndarray, offset: int, mu: float) -> float:
        """Compute interpolated sample value at fractional position.

        Uses 8 samples starting at offset. The interpolated value falls between
        samples[offset+3] and samples[offset+4]. A mu of 0.0 returns samples[offset+3],
        and mu of 1.0 returns samples[offset+4].

        Note: The TAPS array is designed with inverted mu convention (mu=0 gives
        offset+4, mu=1 gives offset+3), so we invert mu to match the documented
        behavior where mu=0 gives offset+3.

        OPTIMIZED: Uses numpy.dot instead of per-tap Python loop for ~5-10x speedup.

        Args:
            samples: Sample array (length >= offset + 8)
            offset: Starting index for the 8-sample window
            mu: Fractional position (0.0 to 1.0)

        Returns:
            Interpolated sample value
        """
        # Invert mu to match TAPS convention: mu=0 should use row 128, mu=1 should use row 0
        mu_inverted = 1.0 - mu

        # Convert inverted mu to tap index (0-128)
        tap_idx = int(mu_inverted * self.NSTEPS + 0.5)
        tap_idx = min(max(tap_idx, 0), self.NSTEPS)

        # Get coefficients for this fractional position
        taps = self.TAPS[tap_idx]

        # Check bounds: need 8 samples from offset to offset+7
        end_idx = offset + self.NTAPS
        if offset >= 0 and end_idx <= len(samples):
            # Fast path: JIT-compiled interpolation for maximum speed
            return _interpolate_8tap_jit(samples, offset, taps)
        else:
            # Slow path: handle edge cases with bounds checking
            result = 0.0
            for i in range(self.NTAPS):
                if 0 <= offset + i < len(samples):
                    result += samples[offset + i] * taps[i]
            return float(result)


# Singleton interpolator instance
_interpolator = _Interpolator()


# JIT-compiled sync correlation function for maximum performance
# This is called for every symbol, so performance is critical
@jit(nopython=True, cache=True)
def _sync_correlate_jit(sync_symbols: np.ndarray, buffer: np.ndarray, pointer: int) -> float:
    """JIT-compiled sync correlation.

    Computes dot product between sync pattern and circular buffer.
    Using explicit loop allows numba to optimize to SIMD instructions.
    """
    score = 0.0
    for i in range(24):
        score += sync_symbols[i] * buffer[pointer + i]
    return score


class _SoftSyncDetector:
    """Soft sync pattern detector for P25 Phase 1.

    Maintains a sliding window of 24 symbols and computes correlation
    against the P25 sync pattern. Returns score at each symbol.

    Reference: SDRTrunk P25P1SoftSyncDetectorScalar.java
    """

    # P25 sync pattern: 0x5575F5FF77FF (48 bits = 24 dibits)
    # Ideal symbol values: +3 for dibit 1, -3 for dibit 3
    SYNC_PATTERN = 0x5575F5FF77FF
    SYNC_THRESHOLD = 60.0  # SDRTrunk P25P1MessageFramer threshold

    def __init__(self):
        # Pre-compute ideal symbol values for sync pattern
        self._sync_symbols = self._pattern_to_symbols()
        # Circular buffer for 24 symbols (doubled for easy wraparound)
        self._buffer = np.zeros(48, dtype=np.float32)
        self._pointer = 0

    def _pattern_to_symbols(self) -> np.ndarray:
        """Convert sync pattern to ideal symbol values."""
        symbols = np.zeros(24, dtype=np.float32)
        pattern = self.SYNC_PATTERN
        for i in range(24):
            dibit = (pattern >> ((23 - i) * 2)) & 0x3
            # Sync pattern only contains dibits 1 (+3) and 3 (-3)
            symbols[i] = 3.0 if dibit == 1 else -3.0
        return symbols

    def reset(self):
        """Reset detector state."""
        self._buffer.fill(0.0)
        self._pointer = 0

    def process(self, soft_symbol: float) -> float:
        """Process one soft symbol and return correlation score.

        Args:
            soft_symbol: Soft symbol value (normalized to ±1, ±3)

        Returns:
            Correlation score (max ~216 for perfect match)
        """
        # Store in circular buffer
        self._buffer[self._pointer] = soft_symbol
        self._buffer[self._pointer + 24] = soft_symbol
        self._pointer = (self._pointer + 1) % 24

        # Compute correlation
        return self._correlate()

    def _correlate(self) -> float:
        """Compute correlation against sync pattern.

        OPTIMIZED: Uses JIT-compiled function for ~10-20x speedup.
        """
        return _sync_correlate_jit(self._sync_symbols, self._buffer, self._pointer)


class _TimingOptimizer:
    """Optimizes symbol timing at sync detection points.

    When sync is detected, searches ±½ symbol period to find the
    timing offset that maximizes sync correlation score.

    OPTIMIZED: Uses JIT-compiled functions for ~100x speedup.

    Reference: SDRTrunk P25P1DemodulatorC4FM.Equalizer.optimize()
    """

    def __init__(self, samples_per_symbol: float):
        self.samples_per_symbol = samples_per_symbol
        # Pre-compute sync pattern symbols (needed for JIT functions)
        self._sync_symbols = _SoftSyncDetector()._pattern_to_symbols()

    def optimize(
        self,
        buffer: np.ndarray,
        buffer_offset: float,
        equalizer: '_Equalizer',
        fine_sync: bool = False
    ) -> tuple[float, float, float, float]:
        """Find optimal timing adjustment that maximizes sync correlation.

        OPTIMIZED: Delegates to JIT-compiled _timing_optimize_jit for ~100x speedup.

        Args:
            buffer: Phase sample buffer
            buffer_offset: Current sample position in buffer
            equalizer: Equalizer for symbol extraction
            fine_sync: If True, use smaller search range

        Returns:
            Tuple of (timing_adjustment, optimized_score, pll_adj, gain_adj)
        """
        return _timing_optimize_jit(
            buffer,
            buffer_offset,
            equalizer.pll,
            equalizer.gain,
            self.samples_per_symbol,
            self._sync_symbols,
            _interpolator.TAPS,
            fine_sync,
        )


class C4FMDemodulator:
    """C4FM (4-level FSK) demodulator for P25 Phase I.

    This is a direct port of SDRTrunk's P25P1DemodulatorC4FM.java for
    maximum compatibility with the proven Java implementation.

    Features:
    - Per-sample FM demodulation with atan
    - Equalizer with PLL (frequency offset tracking) and Gain AGC
    - Soft sync detection with timing optimization
    - Fixed-rate symbol timing with linear interpolation
    - π/2 decision boundaries for dibit slicing

    C4FM symbol mapping (per TIA-102.BAAA-A):
        Symbol   Phase      Dibit
        +3       +3π/4      01 (1)
        +1       +π/4       00 (0)
        -1       -π/4       10 (2)
        -3       -3π/4      11 (3)

    Example usage:
        demod = C4FMDemodulator(sample_rate=19200)  # 4 SPS like SDRTrunk
        dibits, soft = demod.demodulate(iq_samples)
    """

    # Sync detection thresholds (from SDRTrunk)
    SYNC_THRESHOLD_DETECTION = 60.0  # SDRTrunk P25P1MessageFramer SYNC_DETECTION_THRESHOLD
    SYNC_THRESHOLD_OPTIMIZED = 60.0  # Match detection threshold

    def __init__(
        self,
        sample_rate: int = 19200,  # SDRTrunk uses ~19.2 kHz (4 SPS)
        symbol_rate: int = 4800,
        **kwargs  # Accept but ignore other params for API compatibility
    ):
        """Initialize C4FM demodulator.

        Args:
            sample_rate: Input sample rate in Hz (~19200 for 4 SPS like SDRTrunk)
            symbol_rate: Symbol rate (4800 baud for P25)
        """
        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.samples_per_symbol = sample_rate / symbol_rate

        # Baseband LPF (5.2kHz passband, 6.5kHz stopband - matches SDRTrunk)
        # Applied to I and Q before RRC for noise rejection
        self._baseband_lpf = design_baseband_lpf(sample_rate)
        self._lpf_state_i = np.zeros(len(self._baseband_lpf) - 1, dtype=np.float32)
        self._lpf_state_q = np.zeros(len(self._baseband_lpf) - 1, dtype=np.float32)

        # RRC pulse shaping filter (alpha=0.2, 16 symbols span)
        # Applied after baseband LPF, before FM demodulation (matches SDRTrunk)
        self._rrc_filter = design_rrc_filter(self.samples_per_symbol, num_taps=int(16 * self.samples_per_symbol) + 1, alpha=0.2)
        self._rrc_state_i = np.zeros(len(self._rrc_filter) - 1, dtype=np.float32)
        self._rrc_state_q = np.zeros(len(self._rrc_filter) - 1, dtype=np.float32)

        # FM demodulator (symbol-spaced differential)
        # Delay should be approximately 1 symbol period.
        # Use ceiling to avoid undersampling the phase change
        import math
        symbol_delay = int(math.ceil(self.samples_per_symbol))
        self._fm_demod = _FMDemodulator(symbol_delay=symbol_delay)

        # Equalizer (PLL + gain AGC)
        self._equalizer = _Equalizer()

        # Soft sync detection - dual detectors for improved acquisition
        # Primary detector at normal timing
        self._sync_detector = _SoftSyncDetector()
        # Lagging detector at +0.5 symbol offset (SDRTrunk-compatible)
        self._sync_detector_lagging = _SoftSyncDetector()
        self._lagging_offset = self.samples_per_symbol / 2.0

        self._timing_optimizer = _TimingOptimizer(self.samples_per_symbol)

        # Sync state
        self._fine_sync = False
        self._symbols_since_sync = 0
        self._max_fine_sync_adjustment = self.samples_per_symbol * 0.2  # 1/5th symbol

        # Symbol timing state
        self._sample_point = self.samples_per_symbol
        self._buffer = np.zeros(2048, dtype=np.float32)
        self._buffer_pointer = 0

        # Statistics
        self._symbols_processed = 0
        self._sync_count = 0

        logger.debug(
            f"C4FM demod initialized (SDRTrunk-style): {sample_rate} Hz, "
            f"{symbol_rate} baud, {self.samples_per_symbol:.2f} sps"
        )

    def reset(self) -> None:
        """Reset demodulator state for a new signal."""
        self._fm_demod.reset()
        self._equalizer.reset()
        self._sync_detector.reset()
        self._sync_detector_lagging.reset()
        self._lpf_state_i.fill(0)
        self._lpf_state_q.fill(0)
        self._rrc_state_i.fill(0)
        self._rrc_state_q.fill(0)
        self._sample_point = self.samples_per_symbol
        self._buffer.fill(0)
        self._buffer_pointer = 0
        self._symbols_processed = 0
        self._sync_count = 0
        self._fine_sync = False
        self._symbols_since_sync = 0

    @property
    def _ted_phase(self) -> float:
        """API compatibility - SDRTrunk uses fixed timing, not Gardner TED."""
        return 0.0

    def demodulate(self, iq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Demodulate C4FM signal to dibits.

        Processes IQ samples through SDRTrunk-compatible pipeline:
        1. Per-sample FM demodulation (atan)
        2. Circular buffer with symbol-rate sampling
        3. Equalizer (PLL + gain) on each symbol
        4. Linear interpolation for fractional timing
        5. π/2 decision boundaries for dibit slicing

        Args:
            iq: Complex IQ samples (np.complex64 or np.complex128)

        Returns:
            Tuple of (dibits, soft_symbols):
            - dibits: Hard decision dibits (0-3), shape (N,)
            - soft_symbols: Soft symbol values (phase in radians), shape (N,)
        """
        if len(iq) == 0:
            return (
                np.array([], dtype=np.uint8),
                np.array([], dtype=np.float32),
            )

        # [DIAG-STAGE4] C4FM demodulator input diagnostics
        if not hasattr(self, '_diag_demod_calls'):
            self._diag_demod_calls = 0
            logger.info(f"[DIAG-STAGE4] FIRST CALL: C4FM demodulate() invoked, iq.shape={iq.shape}")
        self._diag_demod_calls += 1

        if self._diag_demod_calls % 50 == 0:
            iq_mean = np.mean(iq)
            iq_power = float(np.mean(np.abs(iq)**2))
            iq_peak = float(np.max(np.abs(iq)))
            dc_offset = float(np.abs(iq_mean) / iq_peak) if iq_peak > 0 else 0.0
            logger.info(
                f"[DIAG-STAGE4] C4FM input: calls={self._diag_demod_calls}, "
                f"power={iq_power:.6f}, peak={iq_peak:.4f}, "
                f"dc_offset={dc_offset:.4f}, samples={len(iq)}, rate={self.sample_rate}"
            )

        # Extract I/Q components
        i = iq.real.astype(np.float32)
        q = iq.imag.astype(np.float32)

        # Apply baseband LPF to I and Q for noise rejection
        with _c4fm_profiler.measure("baseband_lpf"):
            i_lpf, self._lpf_state_i = signal.lfilter(
                self._baseband_lpf, 1.0, i, zi=self._lpf_state_i
            )
            q_lpf, self._lpf_state_q = signal.lfilter(
                self._baseband_lpf, 1.0, q, zi=self._lpf_state_q
            )

        # Apply RRC pulse shaping filter (matches SDRTrunk pipeline)
        with _c4fm_profiler.measure("rrc_filter"):
            i_rrc, self._rrc_state_i = signal.lfilter(
                self._rrc_filter, 1.0, i_lpf, zi=self._rrc_state_i
            )
            q_rrc, self._rrc_state_q = signal.lfilter(
                self._rrc_filter, 1.0, q_lpf, zi=self._rrc_state_q
            )

        # FM demodulate to phase values (per-sample)
        with _c4fm_profiler.measure("fm_demod"):
            phases = self._fm_demod.demodulate(
                i_rrc.astype(np.float32),
                q_rrc.astype(np.float32)
            )

        # JIT-compiled symbol recovery for maximum performance
        # This replaces the slow Python for-loop with a numba-compiled version
        _c4fm_profiler.start("symbol_recovery")

        # Run JIT-compiled symbol recovery
        dibits_arr, soft_symbols_arr, symbol_indices, self._buffer_pointer, self._sample_point = \
            _symbol_recovery_jit(
                phases.astype(np.float32),
                self._buffer,
                self._buffer_pointer,
                self._sample_point,
                self.samples_per_symbol,
                self._equalizer.pll,
                self._equalizer.gain,
                _interpolator.TAPS,
            )

        _c4fm_profiler.stop("symbol_recovery")

        # Update symbols processed count
        self._symbols_processed += len(dibits_arr)

        # Process sync detection on extracted symbols (much faster than per-sample)
        # This runs in Python but only for ~960 symbols instead of ~10,000 samples
        _c4fm_profiler.start("sync_detection")
        for i, soft_symbol_norm in enumerate(soft_symbols_arr):
            self._symbols_since_sync += 1
            sync_score_primary = self._sync_detector.process(soft_symbol_norm)

            # Determine which detector to use
            use_lagging = False
            additional_offset = 0.0

            if self._fine_sync:
                sync_score = sync_score_primary
            else:
                # In coarse mode, check lagging detector too
                lag_pos = symbol_indices[i] - int(self._lagging_offset)
                if lag_pos >= 4:
                    lag_mu = 1.0 - (self._lagging_offset - int(self._lagging_offset))
                    lag_offset = lag_pos - 4
                    if lag_offset >= 0 and lag_pos < len(self._buffer):
                        soft_symbol_lag = self._equalizer.get_equalized_symbol(
                            self._buffer, lag_offset, lag_mu
                        )
                        soft_symbol_lag_norm = soft_symbol_lag * (4.0 / np.pi)
                        sync_score_lag = self._sync_detector_lagging.process(soft_symbol_lag_norm)
                    else:
                        sync_score_lag = 0.0
                else:
                    sync_score_lag = 0.0

                if sync_score_lag > sync_score_primary and sync_score_lag >= self.SYNC_THRESHOLD_DETECTION:
                    sync_score = sync_score_lag
                    use_lagging = True
                    additional_offset = -self._lagging_offset
                else:
                    sync_score = sync_score_primary

            if sync_score >= self.SYNC_THRESHOLD_DETECTION:
                # Run timing optimization
                mu = 0.5  # Approximate - we don't have exact mu from JIT
                timing_adj, opt_score, pll_adj, gain_adj = \
                    self._timing_optimizer.optimize(
                        self._buffer,
                        symbol_indices[i] + mu + additional_offset,
                        self._equalizer,
                        fine_sync=self._fine_sync
                    )

                if opt_score >= self.SYNC_THRESHOLD_OPTIMIZED:
                    if self._fine_sync:
                        timing_adj = np.clip(
                            timing_adj,
                            -self._max_fine_sync_adjustment,
                            self._max_fine_sync_adjustment
                        )

                    self._sample_point += timing_adj + additional_offset
                    self._equalizer.apply_correction(pll_adj, gain_adj)

                    self._sync_count += 1
                    self._fine_sync = True
                    self._symbols_since_sync = 0

                    if self._sync_count <= 5 or self._sync_count % 100 == 0:
                        detector_type = "LAG" if use_lagging else "PRI"
                        logger.debug(
                            f"P25 sync #{self._sync_count} [{detector_type}]: score={opt_score:.1f}, "
                            f"timing_adj={timing_adj:.2f}, pll={self._equalizer.pll:.4f}, "
                            f"gain={self._equalizer.gain:.3f}"
                        )

            # Lose fine sync if no sync detected for too long
            if self._symbols_since_sync > 3600:
                self._fine_sync = False
                self._symbols_since_sync = 0

        _c4fm_profiler.stop("sync_detection")

        # Convert to lists for return (maintaining API compatibility)
        dibits = dibits_arr.tolist()
        soft_symbols = soft_symbols_arr.tolist()

        # [DIAG-STAGE5] Symbol output statistics
        if len(soft_symbols) > 0 and self._diag_demod_calls % 100 == 0:
            soft_arr = np.array(soft_symbols, dtype=np.float32)
            # Symbol histogram - should show 4 peaks for good C4FM at ±1, ±3 (normalized)
            # Random noise would show uniform distribution
            hist, _ = np.histogram(soft_arr, bins=16, range=(-4, 4))
            hist_str = ",".join([str(int(h)) for h in hist])
            symbol_mean = float(np.mean(soft_arr))
            symbol_std = float(np.std(soft_arr))
            # Count symbols near ideal positions (±1, ±3)
            near_p3 = np.sum(np.abs(soft_arr - 3.0) < 1.0)
            near_p1 = np.sum(np.abs(soft_arr - 1.0) < 1.0)
            near_m1 = np.sum(np.abs(soft_arr + 1.0) < 1.0)
            near_m3 = np.sum(np.abs(soft_arr + 3.0) < 1.0)
            in_constellation = near_p3 + near_p1 + near_m1 + near_m3
            constellation_pct = 100.0 * in_constellation / len(soft_arr)
            logger.info(
                f"[DIAG-STAGE5] Symbols: count={len(soft_arr)}, "
                f"mean={symbol_mean:.3f}, std={symbol_std:.3f}, "
                f"constellation_pct={constellation_pct:.1f}%, "
                f"histogram=[{hist_str}]"
            )

        # Report profiling periodically
        _c4fm_profiler.report()

        return (
            np.array(dibits, dtype=np.uint8),
            np.array(soft_symbols, dtype=np.float32),
        )

    def get_timing_offset(self) -> float:
        """Get current timing offset (PLL value).

        Returns:
            PLL offset in radians
        """
        return self._equalizer.pll


def c4fm_demod_simple(
    iq: np.ndarray,
    sample_rate: int = 19200,  # SDRTrunk uses ~19.2 kHz (4 SPS)
    symbol_rate: int = 4800,
) -> np.ndarray:
    """Simplified C4FM demodulator (no state, single-shot).

    This is a stateless version for processing complete frames.
    For streaming operation, use C4FMDemodulator class.

    Args:
        iq: Complex IQ samples
        sample_rate: Sample rate in Hz (~19200 for 4 SPS)
        symbol_rate: Symbol rate in baud

    Returns:
        Dibits (0-3)
    """
    demod = C4FMDemodulator(sample_rate=sample_rate, symbol_rate=symbol_rate)
    dibits, _ = demod.demodulate(iq)
    return dibits
