"""
P25 Phase 1 and Phase 2 decoder with full trunking support.

Implements:
- C4FM (4-FSK) demodulation for Phase 1
- TDMA demodulation for Phase 2
- Frame synchronization
- Error correction (Trellis, Reed-Solomon, Golay)
- Control channel (TSBK) decoding
- Voice channel following
- IMBE voice codec support (if available)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, cast

import numpy as np
from scipy.signal import resample_poly

from wavecapsdr.decoders.p25_tsbk import TSBKParser
from wavecapsdr.decoders.p25_frames import extract_link_control
from wavecapsdr.decoders.p25_framer import (
    P25P1MessageFramer,
    P25P1Message,
    P25P1DataUnitID,
)
from wavecapsdr.decoders.p25_phase2 import (
    P25P2Decoder as _P25P2StreamingDecoder,
    P25P2Timeslot,
    P25P2TimeslotType,
)
from wavecapsdr.dsp.fec.bch import bch_decode
from wavecapsdr.dsp.fec.trellis import TrellisDecoder
from wavecapsdr.dsp.p25.c4fm import C4FMDemodulator as _WorkingC4FMDemodulator

logger = logging.getLogger(__name__)


class _C4FMDemodulatorWrapper:
    """Wrapper around the working C4FM demodulator from dsp/p25/c4fm.py.

    The wrapped demodulator returns (dibits, soft_symbols) tuple.
    This wrapper provides both outputs for streaming framer support.
    """

    def __init__(self, sample_rate: int) -> None:
        self._demod = _WorkingC4FMDemodulator(sample_rate=sample_rate)

    def demodulate(self, iq: np.ndarray) -> np.ndarray:
        """Demodulate IQ to dibits only (legacy API)."""
        dibits, _ = self._demod.demodulate(iq)
        return dibits

    def demodulate_soft(self, iq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Demodulate IQ to both dibits and soft symbols.

        Returns:
            Tuple of (dibits, soft_symbols)
        """
        return self._demod.demodulate(iq)

    def reset(self) -> None:
        """Reset demodulator state."""
        self._demod.reset()


class DibitRingBuffer:
    """
    Pre-allocated ring buffer for dibit accumulation.

    Avoids repeated np.concatenate() calls which create new arrays each time.
    Uses a simple circular buffer with head/tail pointers.
    """

    def __init__(self, capacity: int = 8000) -> None:
        self._buffer = np.zeros(capacity, dtype=np.uint8)
        self._capacity = capacity
        self._head = 0  # Write position
        self._tail = 0  # Read position
        self._size = 0  # Current number of elements

    def append(self, dibits: np.ndarray) -> None:
        """Append dibits to the buffer, discarding oldest if full."""
        n = len(dibits)
        if n == 0:
            return

        # If incoming data is larger than capacity, only keep last 'capacity' elements
        if n >= self._capacity:
            dibits = dibits[-self._capacity:]
            n = len(dibits)
            self._buffer[:] = dibits
            self._head = 0
            self._tail = 0
            self._size = n
            return

        # Make room if needed by discarding oldest
        if self._size + n > self._capacity:
            discard = self._size + n - self._capacity
            self._tail = (self._tail + discard) % self._capacity
            self._size -= discard

        # Write new data, handling wrap-around
        space_to_end = self._capacity - self._head
        if n <= space_to_end:
            self._buffer[self._head:self._head + n] = dibits
        else:
            # Split write across wrap-around
            self._buffer[self._head:] = dibits[:space_to_end]
            self._buffer[:n - space_to_end] = dibits[space_to_end:]

        self._head = (self._head + n) % self._capacity
        self._size += n

    def consume(self, n: int) -> None:
        """Remove n elements from the front of the buffer."""
        if n >= self._size:
            self._tail = self._head
            self._size = 0
        else:
            self._tail = (self._tail + n) % self._capacity
            self._size -= n

    def get_contiguous(self, max_len: int | None = None) -> np.ndarray:
        """
        Get buffer contents as a contiguous array.

        This creates a copy but is only called when we need to process frames.
        """
        if self._size == 0:
            return np.array([], dtype=np.uint8)

        length = self._size if max_len is None else min(self._size, max_len)

        if self._tail + length <= self._capacity:
            # No wrap-around, return view (or copy for safety)
            return np.asarray(self._buffer[self._tail:self._tail + length].copy(), dtype=np.uint8)
        else:
            # Handle wrap-around
            result = np.empty(length, dtype=np.uint8)
            first_part = self._capacity - self._tail
            result[:first_part] = self._buffer[self._tail:]
            result[first_part:length] = self._buffer[:length - first_part]
            return result

    def __len__(self) -> int:
        return self._size

    @property
    def size(self) -> int:
        return self._size


class P25FrameType(Enum):
    """P25 frame types"""
    HDU = "Header Data Unit"
    LDU1 = "Logical Link Data Unit 1"
    LDU2 = "Logical Link Data Unit 2"
    TDU = "Terminator Data Unit"
    TSDU = "Trunking Signaling Data Unit"
    PDU = "Packet Data Unit"
    UNKNOWN = "Unknown"


@dataclass
class P25Frame:
    """Decoded P25 frame"""
    frame_type: P25FrameType
    nac: int  # Network Access Code
    duid: int  # Data Unit ID
    algid: int | None = None  # Algorithm ID (encryption)
    kid: int | None = None  # Key ID
    tgid: int | None = None  # Talkgroup ID
    source: int | None = None  # Source radio ID
    voice_data: bytes | None = None  # IMBE voice frames
    tsbk_opcode: int | None = None  # TSBK opcode
    tsbk_data: dict[str, Any] | None = None  # TSBK decoded data
    errors: int = 0  # Error count


class CQPSKDemodulator:
    """
    CQPSK (Compatible QPSK) / LSM (Linear Simulcast Modulation) demodulator for P25 Phase 1.

    Used for P25 simulcast systems that use phase modulation instead of C4FM.
    NOTE: Most P25 systems (including SA-GRN) use C4FM. Only use this for true LSM systems.

    P25 CQPSK is π/4-DQPSK where the transmitted symbol is encoded as a phase CHANGE:
    - dibit 00 → +π/4 (+45°)   → symbol +1
    - dibit 01 → +3π/4 (+135°) → symbol +3
    - dibit 10 → -3π/4 (-135°) → symbol -3
    - dibit 11 → -π/4 (-45°)   → symbol -1

    Demodulates 4800 baud CQPSK signal to dibits using:
    - Carrier frequency offset estimation and correction
    - Differential demodulation (phase transitions)
    - Gardner Timing Error Detector (TED) for symbol timing recovery
    - π/4-DQPSK slicing with rotated decision boundaries

    Based on SDRTrunk's P25P1DecoderLSM.
    """

    # LSM baseband filter cutoff (Hz) - matches SDRTrunk (wider than C4FM)
    BASEBAND_CUTOFF_HZ = 7250

    # SDRTrunk equalizer gain for LSM (slightly different from C4FM)
    EQUALIZER_GAIN = 1.0  # LSM uses adaptive gain, start at 1.0

    # MMSE interpolation parameters
    # Increased from 8 to 32 taps to support Gardner TED at 20 sps
    # Mid-point sample is 10 samples back, so we need at least 20 in buffer
    MMSE_NTAPS = 32
    MMSE_NSTEPS = 128

    def __init__(self, sample_rate: int = 19200, symbol_rate: int = 4800) -> None:
        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.samples_per_symbol = sample_rate / symbol_rate  # Float for fractional

        # π/4-DQPSK decision boundaries
        # Phase changes are at ±π/4 (±45°) and ±3π/4 (±135°)
        # Decision boundaries are at 0, ±π/2, ±π
        self.quarter_pi = np.pi / 4
        self.half_pi = np.pi / 2
        self.three_quarter_pi = 3 * np.pi / 4

        # Carrier frequency offset estimation (based on OP25 gardner_costas_cc)
        # Using magnitude-weighted tracking like OP25
        self._freq_offset = 0.0  # Estimated frequency offset in radians/sample
        self._costas_alpha = 0.125  # Not used
        self._costas_beta = 0.0005  # Moderate gain with magnitude weighting
        self._freq_min = -0.02  # ~-150 Hz at 48kHz
        self._freq_max = 0.02   # ~+150 Hz at 48kHz
        self._phase_acc = 0.0  # Phase accumulator for NCO
        self._carrier_phase = 0.0  # Carrier phase tracking (Costas)

        # AGC state
        self._agc_gain = 1.0
        self._agc_alpha = 0.005
        self._agc_target = 1.0  # Target magnitude for normalized IQ

        # Gardner TED state
        # Using lower gains for stability - original OP25 values caused drift
        self._mu = 0.0  # Fractional symbol timing offset (0 to 1)
        self._gain_mu = 0.015  # Reduced timing error gain
        self._gain_omega = 0.0  # DISABLED - lock omega at nominal to test
        self._omega = self.samples_per_symbol  # Symbol period estimate
        self._prev_symbol = 0.0 + 0.0j
        self._prev_diff = 0.0 + 0.0j

        # Symbol clock for sample-by-sample processing with MMSE
        self._symbol_clock = 0.0  # Fraction of symbol period elapsed (0 to 1)
        self._symbol_time = 1.0 / self.samples_per_symbol  # Time increment per sample

        # Baseband low-pass filter (7250 Hz for LSM, matches SDRTrunk)
        self._baseband_taps = self._design_baseband_filter()

        # RRC filter for matched filtering
        self._rrc_taps = self._design_rrc_filter(alpha=0.2, num_taps=65)

        # Equalizer gain (adaptive for LSM)
        self._equalizer_gain = self.EQUALIZER_GAIN

        # Sample history for MMSE interpolation (complex samples)
        self._history = np.zeros(self.MMSE_NTAPS, dtype=np.complex64)
        self._history_idx = 0

        # Generate full MMSE interpolation table
        self._mmse_taps = self._generate_mmse_taps()

        # Diagnostic tracking
        self._symbol_values: list[float] = []
        self._symbol_count = 0
        self._diag_interval = 50000  # Reduced logging frequency
        self._raw_phases: list[float] = []
        # Track raw symbol magnitudes and phases (before differential)
        self._symbol_mags: list[float] = []
        self._symbol_phases: list[float] = []

    def _generate_mmse_taps(self) -> np.ndarray:
        """
        Generate MMSE interpolation table for 8-tap sinc interpolation.

        Uses windowed sinc for optimal fractional sample reconstruction.
        Only uses 8 taps centered around the interpolation point for efficiency,
        even though the history buffer is larger (to support Gardner TED).
        """
        # We only use 8 taps for interpolation (like OP25), centered on sample point
        interp_ntaps = 8
        taps = np.zeros((self.MMSE_NSTEPS + 1, interp_ntaps), dtype=np.float32)

        for step in range(self.MMSE_NSTEPS + 1):
            mu = step / self.MMSE_NSTEPS  # Fractional offset 0.0 to 1.0

            for tap in range(interp_ntaps):
                # Tap positions: -3, -2, -1, 0, 1, 2, 3, 4 relative to sample point
                t = tap - 3 - mu

                if abs(t) < 1e-6:
                    # At sample point
                    taps[step, tap] = 1.0
                else:
                    # Sinc interpolation with Hann window
                    sinc_val = np.sin(np.pi * t) / (np.pi * t)
                    # Hann window over 8 taps
                    window = 0.5 * (1 + np.cos(np.pi * t / 4)) if abs(t) < 4 else 0
                    taps[step, tap] = sinc_val * window

            # Normalize taps so they sum to 1
            tap_sum = np.sum(taps[step])
            if abs(tap_sum) > 1e-6:
                taps[step] /= tap_sum

        return taps

    def _mmse_interpolate_at_offset(self, sample_offset: int, mu: float) -> complex:
        """
        MMSE FIR interpolation at a specific sample offset from the newest sample.

        Args:
            sample_offset: Integer samples back from newest (0 = most recent)
            mu: Fractional offset 0.0 to 1.0 between integer samples

        Returns:
            Interpolated complex sample value
        """
        # The history buffer is circular with _history_idx pointing to next write position
        # So the most recent sample is at (_history_idx - 1) % NTAPS
        # And sample_offset samples back is at (_history_idx - 1 - sample_offset) % NTAPS

        # Select tap coefficients for this fractional offset
        imu = round(mu * self.MMSE_NSTEPS)
        imu = min(imu, self.MMSE_NSTEPS)

        # Use 8 taps centered around the interpolation point
        # Taps range from -3 to +4 relative to sample_offset
        result = 0.0 + 0.0j
        for tap in range(8):
            # tap 0,1,2,3,4,5,6,7 correspond to offsets -3,-2,-1,0,1,2,3,4 from sample_offset
            offset = sample_offset + (tap - 3)
            if 0 <= offset < self.MMSE_NTAPS:
                # Index into circular buffer: newest is at _history_idx - 1
                hist_idx = (self._history_idx - 1 - offset) % self.MMSE_NTAPS
                result += self._mmse_taps[imu, tap] * self._history[hist_idx]

        return result

    def _mmse_interpolate_complex(self, mu: float) -> complex:
        """
        MMSE FIR interpolation at fractional offset mu from most recent sample.

        Args:
            mu: Fractional offset 0.0 to 1.0

        Returns:
            Interpolated complex sample value
        """
        # Interpolate at the most recent sample position
        return self._mmse_interpolate_at_offset(0, mu)

    def _design_baseband_filter(self, num_taps: int = 63) -> np.ndarray:
        """
        Design baseband low-pass filter for LSM/CQPSK.

        SDRTrunk uses 7250 Hz cutoff for LSM (wider than 5200 Hz for C4FM).
        """
        from scipy import signal as scipy_signal

        nyquist = self.sample_rate / 2
        normalized_cutoff = self.BASEBAND_CUTOFF_HZ / nyquist
        normalized_cutoff = min(0.99, max(0.01, normalized_cutoff))

        taps = scipy_signal.firwin(num_taps, normalized_cutoff, window='hamming')
        return np.asarray(taps, dtype=np.float32)

    def _design_rrc_filter(self, alpha: float = 0.2, num_taps: int = 65) -> np.ndarray:
        """Design Root-Raised Cosine filter for P25."""
        sps = round(self.samples_per_symbol)
        t = np.arange(-(num_taps-1)//2, (num_taps-1)//2 + 1) / sps

        h = np.zeros(num_taps)
        for i, ti in enumerate(t):
            if ti == 0:
                h[i] = 1.0 - alpha + 4*alpha/np.pi
            elif abs(ti) == 1/(4*alpha) if alpha > 0 else False:
                h[i] = (alpha/np.sqrt(2)) * ((1+2/np.pi)*np.sin(np.pi/(4*alpha)) +
                                              (1-2/np.pi)*np.cos(np.pi/(4*alpha)))
            else:
                num = np.sin(np.pi*ti*(1-alpha)) + 4*alpha*ti*np.cos(np.pi*ti*(1+alpha))
                den = np.pi*ti*(1-(4*alpha*ti)**2)
                if abs(den) > 1e-10:
                    h[i] = num / den
                else:
                    h[i] = 0.0

        # Normalize to DC gain of 1.0 to preserve symbol amplitude
        h = h / np.sum(h)
        return h.astype(np.float32)

    def demodulate(self, iq: np.ndarray) -> np.ndarray:
        """
        Demodulate CQPSK signal to dibits using differential phase detection.

        Args:
            iq: Complex IQ samples

        Returns:
            Array of dibits (0-3) as uint8
        """
        if iq.size == 0:
            return np.array([], dtype=np.uint8)

        # Ensure complex input
        if not np.iscomplexobj(iq):
            logger.warning(f"CQPSK demodulate: expected complex IQ, got {iq.dtype}")
            if len(iq) % 2 == 0:
                iq = iq[::2] + 1j * iq[1::2]
            else:
                return np.array([], dtype=np.uint8)

        x: np.ndarray = iq.astype(np.complex64, copy=False)

        # AGC: normalize IQ magnitude
        magnitudes = np.abs(x)
        mean_mag = np.mean(magnitudes) if len(magnitudes) > 0 else 1.0
        max_mag = np.max(magnitudes) if len(magnitudes) > 0 else 0.0
        if mean_mag > 1e-8:
            target_gain = self._agc_target / mean_mag
            self._agc_gain = self._agc_gain * (1 - self._agc_alpha) + target_gain * self._agc_alpha
            self._agc_gain = np.clip(self._agc_gain, 0.01, 500.0)

        # Log raw IQ signal strength periodically
        if not hasattr(self, '_iq_diag_count'):
            self._iq_diag_count = 0
        self._iq_diag_count += 1
        if self._iq_diag_count % 20 == 1:
            logger.info(
                f"CQPSK raw IQ: samples={len(x)}, mean_mag={mean_mag:.4f}, max_mag={max_mag:.4f}, "
                f"agc_gain={self._agc_gain:.2f}, dtype={x.dtype}"
            )

        x = x * self._agc_gain

        # Apply frequency offset correction (NCO) - helps with carrier frequency error
        # Even for DQPSK, frequency offset causes inter-symbol phase drift that
        # can affect timing recovery. Using very slow tracking for stability.
        if abs(self._freq_offset) > 1e-7:
            n = np.arange(len(x))
            nco = np.exp(-1j * (self._phase_acc + self._freq_offset * n))
            x = x * nco
            self._phase_acc += self._freq_offset * len(x)
            self._phase_acc = np.angle(np.exp(1j * self._phase_acc))

        # Baseband low-pass filter (7250 Hz for LSM, matches SDRTrunk)
        if len(x) >= len(self._baseband_taps):
            x_i = np.convolve(x.real, self._baseband_taps, mode='same')
            x_q = np.convolve(x.imag, self._baseband_taps, mode='same')
            x = (x_i + 1j * x_q).astype(np.complex64)

        # DISABLED: RRC pulse shaping filter - testing if it causes ISI
        # if len(x) >= len(self._rrc_taps):
        #     x_i = np.convolve(x.real, self._rrc_taps, mode='same')
        #     x_q = np.convolve(x.imag, self._rrc_taps, mode='same')
        #     x = (x_i + 1j * x_q).astype(np.complex64)

        # Symbol timing recovery with differential demodulation
        symbols = self._cqpsk_timing_recovery(x)

        return symbols

    def _cqpsk_timing_recovery(self, samples: np.ndarray) -> np.ndarray:
        """
        CQPSK timing recovery with MMSE interpolation and differential phase detection.

        Uses π/4-DQPSK differential detection where:
        - diff = curr * conj(prev) gives phase change from prev to curr
        - Phase changes are ±π/4, ±3π/4 for the 4 dibits

        This implementation uses MMSE (8-tap sinc) interpolation like OP25's
        gardner_costas_cc for better fractional sample accuracy.
        """
        sps = self.samples_per_symbol
        symbols = []

        for sample in samples:
            # Add sample to MMSE history buffer
            self._history[self._history_idx] = sample
            self._history_idx = (self._history_idx + 1) % self.MMSE_NTAPS

            # Advance symbol clock
            self._symbol_clock += self._symbol_time

            # Output symbol when clock wraps past 1.0
            if self._symbol_clock >= 1.0:
                self._symbol_clock -= 1.0

                # Get fractional timing offset for MMSE interpolation
                # symbol_clock is how far past 1.0 we went (overshoot)
                # mu increases toward OLDER samples in our MMSE implementation
                # So mu = overshoot_samples gives the symbol point
                mu = self._symbol_clock / self._symbol_time
                mu = np.clip(mu, 0.0, 1.0 - 1e-6)

                # MMSE interpolate current symbol sample
                curr = self._mmse_interpolate_complex(mu)

                # Track raw symbol for constellation analysis
                self._symbol_mags.append(abs(curr))
                self._symbol_phases.append(float(np.angle(curr)))
                if len(self._symbol_mags) > 1000:
                    self._symbol_mags.pop(0)
                    self._symbol_phases.pop(0)

                # Differential demodulation: curr * conj(prev)
                # This gives the phase change FROM prev TO curr
                # Note: carrier phase cancels in differential detection
                # Use normalized symbols for phase-only detection
                curr_mag = abs(curr)
                prev_mag = abs(self._prev_symbol)
                if curr_mag > 1e-6 and prev_mag > 1e-6:
                    diff = (curr / curr_mag) * np.conj(self._prev_symbol / prev_mag)
                else:
                    diff = curr * np.conj(self._prev_symbol)
                phase = np.angle(diff)

                # Track raw phase for diagnostics
                self._raw_phases.append(phase)
                if len(self._raw_phases) > 100:
                    self._raw_phases.pop(0)

                # π/4-DQPSK slicing (OP25 reference: gardner_costas_cc_impl.cc)
                # Phase changes are at ±π/4 (±45°) and ±3π/4 (±135°)
                # Decision boundaries at 0, ±π/2, π
                #
                # OP25 slicer mapping (differential phase → dibit):
                # [0, π/2)     → dibit 0 (around +π/4)
                # [π/2, π)     → dibit 1 (around +3π/4)
                # [-π/2, 0)    → dibit 2 (around -π/4)
                # [-π, -π/2)   → dibit 3 (around -3π/4)
                if phase >= self.half_pi:
                    dibit = 1  # +3π/4 quadrant: +90° to +180°
                elif phase >= 0:
                    dibit = 0  # +π/4 quadrant: 0° to +90°
                elif phase >= -self.half_pi:
                    dibit = 2  # -π/4 quadrant: -90° to 0°
                else:
                    dibit = 3  # -3π/4 quadrant: -180° to -90°

                symbols.append(dibit)

                # Frequency offset estimation using phase error from each symbol
                # Expected phases match the dibit mapping above
                if phase >= self.half_pi:
                    expected = self.three_quarter_pi  # +135° for dibit 1
                elif phase >= 0:
                    expected = self.quarter_pi  # +45° for dibit 0
                elif phase >= -self.half_pi:
                    expected = -self.quarter_pi  # -45° for dibit 2
                else:
                    expected = -self.three_quarter_pi  # -135° for dibit 3

                # Phase error wrapped to [-π, π]
                phase_error = phase - expected
                if phase_error > np.pi:
                    phase_error -= 2 * np.pi
                elif phase_error < -np.pi:
                    phase_error += 2 * np.pi

                # Slow frequency tracking - use very low gain to avoid oscillation
                # Weight by symbol magnitude (like OP25): weak symbols have unreliable phase
                self._freq_offset += self._costas_beta * phase_error * curr_mag
                self._freq_offset = np.clip(self._freq_offset, self._freq_min, self._freq_max)

                # Gardner TED for PSK: ted = real[(curr - prev) * conj(mid)]
                # Uses mid-point sample between current and previous symbols
                # With 32-sample buffer and 20 sps, we can access all needed samples
                half_sps = int(round(sps / 2))  # ~10 samples for mid-point
                full_sps = int(round(sps))      # ~20 samples for previous symbol

                # Get mid-point sample (half symbol period back)
                # and previous symbol (full symbol period back)
                if full_sps + 4 < self.MMSE_NTAPS:  # Ensure we have enough history
                    mid = self._mmse_interpolate_at_offset(half_sps, mu)
                    prev_sym = self._mmse_interpolate_at_offset(full_sps, mu)

                    # Gardner TED error signal for PSK
                    ted = np.real((curr - prev_sym) * np.conj(mid))

                    # Apply timing correction
                    self._symbol_clock += self._gain_mu * ted

                    # Omega adaptation (symbol rate tracking)
                    self._omega += self._gain_omega * ted
                    self._omega = np.clip(self._omega, sps * 0.95, sps * 1.05)
                    self._symbol_time = 1.0 / self._omega

                # Keep clock in valid range
                while self._symbol_clock >= 1.0:
                    self._symbol_clock -= 1.0
                while self._symbol_clock < 0.0:
                    self._symbol_clock += 1.0

                # Save state for next symbol
                self._prev_symbol = curr
                self._prev_diff = diff

                # Diagnostic tracking
                self._symbol_values.append(phase)
                self._symbol_count += 1
                if self._symbol_count % self._diag_interval == 0:
                    vals = np.array(self._symbol_values[-self._diag_interval:])
                    # Count symbols in each quadrant (TIA-102.BAAB Table 2.3 mapping)
                    # d0 (00) at +135°: phase >= +90°
                    # d1 (01) at +45°:  0° <= phase < +90°
                    # d2 (10) at -135°: phase < -90°
                    # d3 (11) at -45°:  -90° <= phase < 0°
                    q0 = np.sum(vals >= self.half_pi)  # dibit 00: +3π/4 quadrant
                    q1 = np.sum((vals >= 0) & (vals < self.half_pi))  # dibit 01: +π/4 quadrant
                    q2 = np.sum(vals < -self.half_pi)  # dibit 10: -3π/4 quadrant
                    q3 = np.sum((vals >= -self.half_pi) & (vals < 0))  # dibit 11: -π/4 quadrant

                    # Phase histogram around expected constellation points
                    pi_4 = np.pi / 4
                    pi_8 = np.pi / 8
                    near_p45 = np.sum(np.abs(vals - pi_4) < pi_8)  # +45°
                    near_p135 = np.sum(np.abs(vals - 3*pi_4) < pi_8)  # +135°
                    near_m45 = np.sum(np.abs(vals + pi_4) < pi_8)  # -45°
                    near_m135 = np.sum(np.abs(vals + 3*pi_4) < pi_8)  # -135°
                    near_wrap = np.sum((vals > np.pi - pi_8) | (vals < -np.pi + pi_8))

                    # Calculate clustering quality (% of phases near expected values)
                    total_clustered = near_p45 + near_p135 + near_m45 + near_m135
                    cluster_pct = 100 * total_clustered / len(vals) if len(vals) > 0 else 0

                    logger.debug(
                        f"CQPSK MMSE: count={self._symbol_count}, "
                        f"dist=[d0:{q0}, d1:{q1}, d2:{q2}, d3:{q3}], "
                        f"freq_off={self._freq_offset*self.sample_rate/(2*np.pi):.1f}Hz, "
                        f"agc={self._agc_gain:.2f}, "
                        f"phase mean={vals.mean():.3f}, std={vals.std():.3f}"
                    )
                    logger.debug(
                        f"CQPSK phase clusters: +45°={near_p45}, +135°={near_p135}, "
                        f"-45°={near_m45}, -135°={near_m135}, wrap={near_wrap}, "
                        f"quality={cluster_pct:.1f}%, omega={self._omega:.2f}"
                    )
                    # Log raw constellation stats
                    if self._symbol_mags:
                        mags = np.array(self._symbol_mags[-500:])
                        phases = np.array(self._symbol_phases[-500:])
                        logger.debug(
                            f"CQPSK constellation: mag_mean={np.mean(mags):.3f}, "
                            f"mag_std={np.std(mags):.3f}, phase_std={np.std(phases):.3f}"
                        )

        return np.array(symbols, dtype=np.uint8)


class C4FMDemodulator:
    """
    C4FM (4-level FSK) demodulator for P25 Phase 1.

    Demodulates 4800 baud C4FM signal to dibits using:
    - FM discriminator for frequency demodulation
    - Baseband low-pass filter (5200 Hz cutoff, matching SDRTrunk)
    - Root-Raised Cosine (RRC) pulse shaping filter
    - 1.219x equalizer gain (from SDRTrunk, compensates for RRC imbalance)
    - MMSE (Minimum Mean Square Error) interpolation for symbol timing
    - Symbol spread tracking for automatic deviation adaptation
    - Fine frequency correction for DC offset removal

    Based on OP25's fsk4_demod_ff and SDRTrunk's P25P1DecoderC4FM.

    NOTE: Most P25 systems use C4FM including SA-GRN. Only use CQPSKDemodulator
    for systems explicitly using LSM (Linear Simulcast Modulation) with phase modulation.
    """

    # SDRTrunk equalizer gain - compensates for RRC filter constellation compression
    # Original SDRTrunk value is 1.219
    EQUALIZER_GAIN = 1.219

    # C4FM baseband filter cutoff (Hz) - matches SDRTrunk
    BASEBAND_CUTOFF_HZ = 5200

    # MMSE interpolation taps from OP25 (8 taps, 128 fractional steps)
    # This provides much more accurate interpolation than linear
    MMSE_NTAPS = 8
    MMSE_NSTEPS = 128
    MMSE_TAPS = np.array([
        # Each row is tap coefficients for a fractional offset (0/128 to 128/128)
        # Subset of key positions - full table generated at init
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # 0/128 (integer)
        [-6.77751e-03, 3.94578e-02, -1.42658e-01, 6.09836e-01, 6.09836e-01, -1.42658e-01, 3.94578e-02, -6.77751e-03],  # 64/128 (0.5)
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # 128/128 (1.0)
    ], dtype=np.float32)

    def __init__(self, sample_rate: int = 19200, symbol_rate: int = 4800) -> None:
        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.samples_per_symbol = sample_rate / symbol_rate  # Float for fractional

        # C4FM deviation levels (normalized to ±1)
        # P25 TIA-102.BAAA constellation mapping:
        # Symbol | Frequency | Dibit Binary | Dibit Value
        # +3     | +1800 Hz  | 01          | 1
        # +1     | +600 Hz   | 00          | 0
        # -1     | -600 Hz   | 10          | 2
        # -3     | -1800 Hz  | 11          | 3
        self.deviation_hz = 600.0  # Base deviation (±600 Hz steps)
        self.max_deviation = 1800.0  # ±1800 Hz max

        # Thresholds for 4-level slicing (fixed, based on symbol spread)
        self._threshold_low = -2.0   # Between -3 and -1
        self._threshold_mid = 0.0    # Between -1 and +1
        self._threshold_high = 2.0   # Between +1 and +3
        self._use_4level = True

        # Symbol timing recovery state (OP25-style)
        self._symbol_clock = 0.0  # Fractional position within symbol
        self._symbol_time = symbol_rate / sample_rate  # Time increment per sample

        # Symbol spread tracking (OP25 adaptation)
        # Nominal spread of 2.0 gives outputs at -3, -1, +1, +3
        self._symbol_spread = 2.0
        self._K_SYMBOL_SPREAD = 0.0100  # Spread tracking gain (from OP25)

        # Timing loop gain (from OP25)
        self._K_SYMBOL_TIMING = 0.025  # Symbol clock tracking gain

        # Fine frequency correction (OP25's DC offset tracking)
        self._fine_freq_correction = 0.0
        self._K_FINE_FREQUENCY = 0.125  # Fast fine loop gain

        # Coarse frequency correction for external tuning requests
        self._coarse_freq_correction = 0.0
        self._K_COARSE_FREQUENCY = 0.00125

        # Sample history for MMSE interpolation
        self._history = np.zeros(self.MMSE_NTAPS, dtype=np.float32)
        self._history_idx = 0

        # Generate full MMSE interpolation table
        self._mmse_taps = self._generate_mmse_taps()

        # Baseband low-pass filter (5200 Hz cutoff, matches SDRTrunk)
        self._baseband_taps = self._design_baseband_filter()

        # RRC filter coefficients (alpha=0.2 for P25)
        self._rrc_taps = self._design_rrc_filter(alpha=0.2, num_taps=65)

        # DC removal state (alpha=0.05 gives 20-symbol time constant for faster convergence)
        self._dc_alpha = 0.05
        self._dc_estimate = 0.0

        # SDRTrunk-style equalizer: PLL + gain
        # PLL tracks frequency offset, gain compensates RRC filter imbalance
        self._equalizer_pll = 0.0  # Phase-locked loop correction
        self._equalizer_gain = self.EQUALIZER_GAIN  # 1.219x gain from SDRTrunk

        # Constellation gain normalization (legacy, kept for compatibility)
        self._constellation_gain = 1.0
        self._target_std = 2.5

        # Diagnostic: symbol value tracking
        self._symbol_values: list[float] = []
        self._symbol_count = 0
        self._diag_interval = 50000  # Reduced logging frequency

    def _generate_mmse_taps(self) -> np.ndarray:
        """
        Generate full MMSE interpolation table.

        Uses sinc-based interpolation with windowing for optimal
        fractional sample reconstruction.
        """
        taps = np.zeros((self.MMSE_NSTEPS + 1, self.MMSE_NTAPS), dtype=np.float32)

        for step in range(self.MMSE_NSTEPS + 1):
            mu = step / self.MMSE_NSTEPS  # Fractional offset 0.0 to 1.0

            for tap in range(self.MMSE_NTAPS):
                # Tap positions: -3, -2, -1, 0, 1, 2, 3, 4 relative to sample point
                t = tap - 3 - mu

                if abs(t) < 1e-6:
                    # At sample point
                    taps[step, tap] = 1.0
                else:
                    # Sinc interpolation with Hann window
                    sinc_val = np.sin(np.pi * t) / (np.pi * t)
                    # Hann window over 8 taps
                    window = 0.5 * (1 + np.cos(np.pi * t / 4)) if abs(t) < 4 else 0
                    taps[step, tap] = sinc_val * window

            # Normalize taps so they sum to 1
            tap_sum = np.sum(taps[step])
            if abs(tap_sum) > 1e-6:
                taps[step] /= tap_sum

        return taps

    def _design_baseband_filter(self, num_taps: int = 63) -> np.ndarray:
        """
        Design baseband low-pass filter for C4FM.

        SDRTrunk uses 5200 Hz cutoff for C4FM (vs 7250 Hz for LSM).
        This filters the FM discriminator output before pulse shaping.
        """
        from scipy import signal as scipy_signal

        # Normalized cutoff: cutoff_hz / (sample_rate / 2)
        nyquist = self.sample_rate / 2
        normalized_cutoff = self.BASEBAND_CUTOFF_HZ / nyquist

        # Clamp to valid range
        normalized_cutoff = min(0.99, max(0.01, normalized_cutoff))

        # Design low-pass FIR filter with Hamming window
        taps = scipy_signal.firwin(num_taps, normalized_cutoff, window='hamming')
        return np.asarray(taps, dtype=np.float32)

    def _design_rrc_filter(self, alpha: float = 0.2, num_taps: int = 65) -> np.ndarray:
        """Design Root-Raised Cosine filter for P25 C4FM."""
        sps = round(self.samples_per_symbol)
        t = np.arange(-(num_taps-1)//2, (num_taps-1)//2 + 1) / sps

        h = np.zeros(num_taps)
        for i, ti in enumerate(t):
            if ti == 0:
                h[i] = 1.0 - alpha + 4*alpha/np.pi
            elif abs(ti) == 1/(4*alpha) if alpha > 0 else False:
                h[i] = (alpha/np.sqrt(2)) * ((1+2/np.pi)*np.sin(np.pi/(4*alpha)) +
                                              (1-2/np.pi)*np.cos(np.pi/(4*alpha)))
            else:
                num = np.sin(np.pi*ti*(1-alpha)) + 4*alpha*ti*np.cos(np.pi*ti*(1+alpha))
                den = np.pi*ti*(1-(4*alpha*ti)**2)
                if abs(den) > 1e-10:
                    h[i] = num / den
                else:
                    h[i] = 0.0

        # Normalize to DC gain of 1.0 to preserve symbol amplitude
        # (Energy normalization amplifies symbols by ~3-4x and breaks slicing)
        h = h / np.sum(h)
        return h.astype(np.float32)

    def demodulate(self, iq: np.ndarray) -> np.ndarray:
        """
        Demodulate C4FM signal to dibits (2-bit symbols).

        Args:
            iq: Complex IQ samples

        Returns:
            Array of dibits (0-3) as uint8
        """
        if iq.size == 0:
            return np.array([], dtype=np.uint8)

        # Validate input - must be complex
        if not np.iscomplexobj(iq):
            logger.warning(f"C4FM demodulate: expected complex IQ, got {iq.dtype}")
            if len(iq) % 2 == 0:
                iq = iq[::2] + 1j * iq[1::2]
            else:
                return np.array([], dtype=np.uint8)

        # FM discriminator (quadrature demodulation)
        x: np.ndarray = iq.astype(np.complex64, copy=False)
        prod = x[1:] * np.conj(x[:-1])
        # Scale to ±3 symbol range for ±1800 Hz deviation
        # Using deviation_hz (600) as base: ±1800/600 = ±3, ±600/600 = ±1
        inst_freq = cast(np.ndarray, np.angle(prod)) * self.sample_rate / (2 * np.pi * self.deviation_hz)

        if len(inst_freq) < len(self._rrc_taps):
            return np.array([], dtype=np.uint8)

        # Debug: Log raw FM discriminator output
        if self._symbol_count % 10000 == 0 and len(inst_freq) > 10:
            logger.info(
                f"C4FM FM disc raw: mean={np.mean(inst_freq):.2f}, std={np.std(inst_freq):.2f}, "
                f"min={np.min(inst_freq):.2f}, max={np.max(inst_freq):.2f}"
            )

        # Skip DC removal for now - it's too aggressive and causes symbol level drift
        # DC offset is better handled by the fine frequency correction loop

        # Apply RRC matched filter using lfilter (causal, streaming-friendly)
        # Skip baseband filter - RRC already provides adequate pulse shaping
        from scipy import signal as scipy_signal
        try:
            filtered = scipy_signal.lfilter(self._rrc_taps, 1.0, inst_freq).astype(np.float32)
        except Exception:
            filtered = inst_freq.astype(np.float32)

        # Apply SDRTrunk equalizer gain (1.219x)
        # This compensates for RRC filter constellation compression
        filtered = filtered * self._equalizer_gain

        # Legacy constellation gain normalization (adaptive, kept for compatibility)
        if len(filtered) > 100:
            current_std = np.std(filtered)
            if current_std > 0.1:
                ideal_gain = self._target_std / current_std
                self._constellation_gain = self._constellation_gain * 0.95 + ideal_gain * 0.05
                self._constellation_gain = max(0.1, min(2.0, self._constellation_gain))

        filtered = filtered * self._constellation_gain

        # Symbol timing recovery using MMSE interpolation (OP25-style)
        symbols = self._mmse_timing_recovery(filtered)

        return symbols

    def _mmse_interpolate(self, mu: float) -> float:
        """
        MMSE FIR interpolation at fractional offset mu.

        Args:
            mu: Fractional offset 0.0 to 1.0

        Returns:
            Interpolated sample value
        """
        # Select tap coefficients for this fractional offset
        imu = round(mu * self.MMSE_NSTEPS)
        if imu > self.MMSE_NSTEPS:
            imu = self.MMSE_NSTEPS

        # Apply FIR filter with history buffer
        result = 0.0
        for i in range(self.MMSE_NTAPS):
            hist_idx = (self._history_idx + i) % self.MMSE_NTAPS
            result += self._mmse_taps[imu, i] * self._history[hist_idx]

        return result

    def _mmse_timing_recovery(self, samples: np.ndarray) -> np.ndarray:
        """
        Symbol timing recovery using MMSE interpolation (OP25-style).

        This implements OP25's fsk4_demod_ff tracking loop which includes:
        - MMSE FIR interpolation for accurate fractional sample recovery
        - Symbol spread tracking (adapts to actual signal deviation)
        - Fine frequency correction (tracks DC offset)
        - Gradient-based timing adjustment
        """
        symbols = []

        for sample in samples:
            # Add sample to history buffer
            self._history[self._history_idx] = sample
            self._history_idx = (self._history_idx + 1) % self.MMSE_NTAPS

            # Advance symbol clock
            self._symbol_clock += self._symbol_time

            # Output symbol when clock wraps
            if self._symbol_clock > 1.0:
                self._symbol_clock -= 1.0

                # Get fractional timing offset for interpolation
                mu = self._symbol_clock / self._symbol_time
                if mu > 1.0:
                    mu = 1.0

                # MMSE interpolate at current position and one step ahead
                interp = self._mmse_interpolate(mu)

                # Also get interpolation at next fractional step (for gradient)
                mu_p1 = min(mu + 1.0 / self.MMSE_NSTEPS, 1.0)
                interp_p1 = self._mmse_interpolate(mu_p1)

                # Apply fine frequency correction (DC offset tracking)
                interp -= self._fine_freq_correction
                interp_p1 -= self._fine_freq_correction

                # Output normalized by symbol spread
                output = 2.0 * interp / self._symbol_spread

                # Compute symbol error for tracking loops
                symbol_error = self._compute_symbol_error(interp)

                # Update symbol spread (OP25-style adaptation)
                self._update_symbol_spread(interp, symbol_error)

                # Update timing using gradient
                if interp_p1 < interp:
                    self._symbol_clock += symbol_error * self._K_SYMBOL_TIMING
                else:
                    self._symbol_clock -= symbol_error * self._K_SYMBOL_TIMING

                # Update frequency correction loops
                self._coarse_freq_correction += (
                    (self._fine_freq_correction - self._coarse_freq_correction)
                    * self._K_COARSE_FREQUENCY
                )
                self._fine_freq_correction += symbol_error * self._K_FINE_FREQUENCY

                # 4-level symbol slicing (using normalized output)
                if output < -2.0:
                    dibit = 3  # -3 symbol
                elif output < 0.0:
                    dibit = 2  # -1 symbol
                elif output < 2.0:
                    dibit = 0  # +1 symbol
                else:
                    dibit = 1  # +3 symbol

                symbols.append(dibit)

                # Diagnostic tracking
                self._symbol_values.append(output)
                self._symbol_count += 1
                if self._symbol_count % self._diag_interval == 0:
                    vals = np.array(self._symbol_values[-self._diag_interval:])
                    d3 = np.sum(vals < -2.0)
                    d2 = np.sum((vals >= -2.0) & (vals < 0.0))
                    d0 = np.sum((vals >= 0.0) & (vals < 2.0))
                    d1 = np.sum(vals >= 2.0)
                    logger.debug(
                        f"C4FM MMSE: count={self._symbol_count}, "
                        f"dist=[d0:{d0}, d1:{d1}, d2:{d2}, d3:{d3}], "
                        f"spread={self._symbol_spread:.3f}, "
                        f"fine_freq={self._fine_freq_correction:.3f}, "
                        f"mean={vals.mean():.3f}, std={vals.std():.3f}"
                    )

        return np.array(symbols, dtype=np.uint8)

    def _compute_symbol_error(self, interp: float) -> float:
        """
        Compute symbol error for tracking loops.

        Determines which symbol level was detected and computes
        the error from the expected position.
        """
        if interp < -self._symbol_spread:
            # Symbol is -3: Expected at -1.5 * symbol_spread
            return interp + (1.5 * self._symbol_spread)
        elif interp < 0.0:
            # Symbol is -1: Expected at -0.5 * symbol_spread
            return interp + (0.5 * self._symbol_spread)
        elif interp < self._symbol_spread:
            # Symbol is +1: Expected at +0.5 * symbol_spread
            return interp - (0.5 * self._symbol_spread)
        else:
            # Symbol is +3: Expected at +1.5 * symbol_spread
            return interp - (1.5 * self._symbol_spread)

    def _update_symbol_spread(self, interp: float, symbol_error: float) -> None:
        """
        Update symbol spread (deviation) estimate.

        Tracks the actual signal deviation level adaptively.
        """
        # Outer symbols contribute half as much to spread adaptation
        if interp < -self._symbol_spread or interp >= self._symbol_spread:
            # Outer symbol (±3)
            self._symbol_spread -= symbol_error * 0.5 * self._K_SYMBOL_SPREAD
        else:
            # Inner symbol (±1)
            if interp < 0.0:
                self._symbol_spread -= symbol_error * self._K_SYMBOL_SPREAD
            else:
                self._symbol_spread += symbol_error * self._K_SYMBOL_SPREAD

        # Constrain spread to ±20% of nominal 2.0
        SYMBOL_SPREAD_MAX = 2.4
        SYMBOL_SPREAD_MIN = 1.6
        self._symbol_spread = max(SYMBOL_SPREAD_MIN, min(SYMBOL_SPREAD_MAX, self._symbol_spread))


class DiscriminatorDemodulator:
    """
    Demodulator for FM discriminator audio input (mono).

    Used when input is already FM-demodulated audio (e.g., from discriminator tap,
    DSD-FME compatible .dis files, or recordings from scanner discriminator output).

    Skips the FM demodulation step and goes directly to:
    - DC offset removal
    - Baseband filtering
    - Symbol timing recovery

    Expected input: Mono audio at 48kHz (standard discriminator rate) with
    symbol levels at approximately ±1 and ±3 (relative units).
    """

    # Same parameters as C4FMDemodulator for consistency
    BASEBAND_CUTOFF_HZ = 5200
    MMSE_NTAPS = 8
    MMSE_NSTEPS = 128

    def __init__(self, sample_rate: int = 48000, symbol_rate: int = 4800) -> None:
        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.samples_per_symbol = sample_rate / symbol_rate

        # DC offset tracking
        self._dc_estimate = 0.0
        self._dc_alpha = 0.001  # Slow DC tracking

        # Symbol timing recovery state
        self._symbol_clock = 0.0
        self._symbol_time = symbol_rate / sample_rate

        # Symbol spread tracking
        self._symbol_spread = 2.0
        self._K_SYMBOL_SPREAD = 0.0100

        # Timing loop gain
        self._K_SYMBOL_TIMING = 0.025

        # Frequency correction
        self._fine_freq_correction = 0.0
        self._K_FINE_FREQUENCY = 0.125
        self._coarse_freq_correction = 0.0
        self._K_COARSE_FREQUENCY = 0.00125

        # Input scaling - discriminator audio may need normalization
        self._input_gain = 1.0
        self._auto_gain = True

        # Sample history for MMSE
        self._history = np.zeros(self.MMSE_NTAPS, dtype=np.float32)
        self._history_idx = 0

        # Generate MMSE interpolation taps
        self._mmse_taps = self._generate_mmse_taps()

        # Design baseband filter
        self._baseband_taps = self._design_baseband_filter()

        # Diagnostics
        self._symbol_count = 0
        self._symbol_values: list[float] = []
        self._diag_interval = 50000  # Reduced logging frequency

    def _generate_mmse_taps(self) -> np.ndarray:
        """Generate MMSE interpolation filter coefficients."""
        taps = np.zeros((self.MMSE_NSTEPS + 1, self.MMSE_NTAPS), dtype=np.float32)

        for step in range(self.MMSE_NSTEPS + 1):
            mu = step / self.MMSE_NSTEPS
            for tap in range(self.MMSE_NTAPS):
                t = tap - 3 - mu
                if abs(t) < 1e-6:
                    taps[step, tap] = 1.0
                else:
                    sinc_val = np.sin(np.pi * t) / (np.pi * t)
                    window = 0.5 * (1 + np.cos(np.pi * t / 4)) if abs(t) < 4 else 0
                    taps[step, tap] = sinc_val * window

            # Normalize
            tap_sum = np.sum(taps[step])
            if tap_sum > 0:
                taps[step] /= tap_sum

        return taps

    def _design_baseband_filter(self) -> np.ndarray:
        """Design baseband low-pass filter."""
        from scipy.signal import firwin
        ntaps = 65
        cutoff = self.BASEBAND_CUTOFF_HZ / (self.sample_rate / 2)
        cutoff = min(cutoff, 0.99)
        return np.asarray(firwin(ntaps, cutoff, window='hamming'), dtype=np.float32)

    def demodulate(self, audio: np.ndarray) -> np.ndarray:
        """
        Demodulate discriminator audio to dibits.

        Args:
            audio: Mono audio samples (float32, normalized to ±1 range)

        Returns:
            Array of dibits (0-3) as uint8
        """
        if audio.size == 0:
            return np.array([], dtype=np.uint8)

        # Ensure float32
        samples = audio.astype(np.float32, copy=False)

        # Auto-gain normalization
        if self._auto_gain and len(samples) > 100:
            max_val = np.max(np.abs(samples))
            if max_val > 0.01:
                # Target ±3 symbol range (so max should be ~3)
                target_max = 3.0
                new_gain = target_max / max_val
                self._input_gain = self._input_gain * 0.9 + new_gain * 0.1

        samples = samples * self._input_gain

        # DC offset removal
        for i in range(len(samples)):
            self._dc_estimate = self._dc_estimate * (1 - self._dc_alpha) + samples[i] * self._dc_alpha
            samples[i] = samples[i] - self._dc_estimate

        # Baseband low-pass filter
        if len(samples) >= len(self._baseband_taps):
            samples = np.convolve(samples, self._baseband_taps, mode='same').astype(np.float32)

        # Symbol timing recovery
        return self._mmse_timing_recovery(samples)

    def _mmse_interpolate(self, mu: float) -> float:
        """MMSE FIR interpolation."""
        imu = round(mu * self.MMSE_NSTEPS)
        imu = min(imu, self.MMSE_NSTEPS)

        result = 0.0
        for i in range(self.MMSE_NTAPS):
            hist_idx = (self._history_idx + i) % self.MMSE_NTAPS
            result += self._mmse_taps[imu, i] * self._history[hist_idx]

        return result

    def _mmse_timing_recovery(self, samples: np.ndarray) -> np.ndarray:
        """Symbol timing recovery using MMSE interpolation."""
        symbols = []

        for sample in samples:
            self._history[self._history_idx] = sample
            self._history_idx = (self._history_idx + 1) % self.MMSE_NTAPS

            self._symbol_clock += self._symbol_time

            if self._symbol_clock > 1.0:
                self._symbol_clock -= 1.0

                mu = self._symbol_clock / self._symbol_time
                mu = min(mu, 1.0)

                interp = self._mmse_interpolate(mu)
                mu_p1 = min(mu + 1.0 / self.MMSE_NSTEPS, 1.0)
                interp_p1 = self._mmse_interpolate(mu_p1)

                interp -= self._fine_freq_correction
                interp_p1 -= self._fine_freq_correction

                output = 2.0 * interp / self._symbol_spread

                # Symbol error
                if interp < -self._symbol_spread:
                    symbol_error = interp + (1.5 * self._symbol_spread)
                elif interp < 0.0:
                    symbol_error = interp + (0.5 * self._symbol_spread)
                elif interp < self._symbol_spread:
                    symbol_error = interp - (0.5 * self._symbol_spread)
                else:
                    symbol_error = interp - (1.5 * self._symbol_spread)

                # Update spread
                if interp < -self._symbol_spread or interp >= self._symbol_spread:
                    self._symbol_spread -= symbol_error * 0.5 * self._K_SYMBOL_SPREAD
                else:
                    if interp < 0.0:
                        self._symbol_spread -= symbol_error * self._K_SYMBOL_SPREAD
                    else:
                        self._symbol_spread += symbol_error * self._K_SYMBOL_SPREAD
                self._symbol_spread = max(1.6, min(2.4, self._symbol_spread))

                # Update timing
                if interp_p1 < interp:
                    self._symbol_clock += symbol_error * self._K_SYMBOL_TIMING
                else:
                    self._symbol_clock -= symbol_error * self._K_SYMBOL_TIMING

                # Update frequency correction
                self._coarse_freq_correction += (
                    (self._fine_freq_correction - self._coarse_freq_correction)
                    * self._K_COARSE_FREQUENCY
                )
                self._fine_freq_correction += symbol_error * self._K_FINE_FREQUENCY

                # 4-level slicing
                if output < -2.0:
                    dibit = 3
                elif output < 0.0:
                    dibit = 2
                elif output < 2.0:
                    dibit = 0
                else:
                    dibit = 1

                symbols.append(dibit)
                self._symbol_count += 1

                self._symbol_values.append(output)
                if self._symbol_count % self._diag_interval == 0:
                    vals = np.array(self._symbol_values[-self._diag_interval:])
                    logger.debug(
                        f"Discriminator: count={self._symbol_count}, "
                        f"spread={self._symbol_spread:.3f}, mean={vals.mean():.3f}"
                    )

        return np.array(symbols, dtype=np.uint8)

    def reset(self) -> None:
        """Reset demodulator state."""
        self._dc_estimate = 0.0
        self._symbol_clock = 0.0
        self._symbol_spread = 2.0
        self._fine_freq_correction = 0.0
        self._coarse_freq_correction = 0.0
        self._history.fill(0)
        self._history_idx = 0
        self._symbol_count = 0
        self._symbol_values = []


class P25TrellisDecoder:
    """
    P25 1/2 rate trellis decoder (Viterbi algorithm).

    This is a wrapper around the optimized TrellisDecoder in dsp/fec/trellis.py.

    P25 uses a 4-state trellis code where:
    - Input: 2 bits (dibit) per time instant
    - Output: 4 bits (nibble/symbol) per time instant
    - Each 4-bit symbol is encoded as two consecutive C4FM dibits

    For TSBK (96 data bits = 48 input dibits):
    - 48 input dibits + 1 flushing = 49 symbols
    - 49 * 4 bits = 196 transmitted bits = 98 dibits
    """

    def __init__(self) -> None:
        self._decoder = TrellisDecoder()

    def decode(self, dibits: np.ndarray) -> tuple[np.ndarray | None, int]:
        """
        Decode trellis-encoded dibits using Viterbi algorithm.

        Args:
            dibits: Array of received dibits (0-3), length should be 98 for TSBK

        Returns:
            (decoded_dibits, error_count) or (None, -1) if decode failed
        """
        if len(dibits) < 4:
            return None, -1

        try:
            decoded, error_metric = self._decoder.decode(dibits)

            if len(decoded) == 0:
                return None, -1

            # Truncate to 48 dibits for TSBK (remove flushing dibit)
            if len(decoded) > 48:
                decoded = decoded[:48]

            return decoded, int(error_metric)
        except Exception as e:
            logger.debug(f"Trellis decode error: {e}")
            return None, -1


class P25FrameSync:
    """Frame synchronization for P25 with soft correlation (SDRTrunk-style)"""

    # P25 Frame Sync is 48 bits (24 dibits) representing the pattern:
    # +3 +3 +3 +3 +3 -3 +3 +3 -3 -3 +3 +3 +3 +3 -3 +3 -3 +3 -3 -3 -3 +3 -3 -3
    # These map to dibits: 3 3 3 3 3 0 3 3 0 0 3 3 3 3 0 3 0 3 0 0 0 3 0 0
    #
    # After frame sync comes the NID (Network ID):
    # - NAC (12 bits, 6 dibits) - Network Access Code
    # - DUID (4 bits, 2 dibits) - Data Unit ID
    #
    # The DUID determines frame type:
    DUID_HDU = 0x0   # Header Data Unit
    DUID_TDU = 0x3   # Terminator Data Unit (without LC)
    DUID_LDU1 = 0x5  # Logical Link Data Unit 1
    DUID_LDU2 = 0xA  # Logical Link Data Unit 2
    DUID_TSDU = 0x7  # Trunking Signaling Data Unit
    DUID_PDU = 0xC   # Packet Data Unit
    DUID_TDULC = 0xF # Terminator with LC

    # Frame sync pattern as dibits (24 dibits = 48 bits)
    # Per TIA-102.BAAA, P25 uses the same sync pattern for all frame types:
    # C4FM symbols: +3 +3 +3 +3 +3 -3 +3 +3 -3 -3 +3 +3 -3 -3 -3 -3 +3 -3 +3 -3 -3 -3 -3 -3
    #
    # Correct dibit encoding per constellation mapping:
    # +3 symbol -> dibit 1 (binary 01)
    # -3 symbol -> dibit 3 (binary 11)
    #
    # This matches SDRTrunk's pattern: 0x5575F5FF77FF
    FRAME_SYNC_DIBITS = np.array([1, 1, 1, 1, 1, 3, 1, 1, 3, 3, 1, 1,
                                   3, 3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3], dtype=np.uint8)

    # Sync pattern as soft symbols (for soft correlation)
    # dibit 1 -> +3, dibit 3 -> -3
    SYNC_PATTERN_SYMBOLS = np.array([+3, +3, +3, +3, +3, -3, +3, +3, -3, -3, +3, +3,
                                      -3, -3, -3, -3, +3, -3, +3, -3, -3, -3, -3, -3], dtype=np.float32)

    # Dibit to symbol conversion (for soft correlation)
    # dibit 0 -> +1, dibit 1 -> +3, dibit 2 -> -1, dibit 3 -> -3
    DIBIT_TO_SYMBOL = np.array([+1.0, +3.0, -1.0, -3.0], dtype=np.float32)

    # SDRTrunk-style soft correlation thresholds
    # Max score is 24 * 9 = 216 (all symbols at ±3 perfectly matched)
    SOFT_SYNC_THRESHOLD = 60  # SDRTrunk uses 60 for coarse detection
    SOFT_SYNC_THRESHOLD_OPTIMIZE = 80  # SDRTrunk uses 80 for optimization

    # Reversed polarity sync pattern (each dibit XOR 2)
    # Per OP25: P25_FRAME_SYNC_REV_P = P25_FRAME_SYNC_MAGIC ^ 0xAAAAAAAAAAAALL
    # This swaps dibits 0<->2 and 1<->3
    FRAME_SYNC_DIBITS_REV = FRAME_SYNC_DIBITS ^ 2  # [3,3,3,3,3,1,3,3,1,1,3,3,...]

    # Reversed sync symbols (for soft correlation)
    SYNC_PATTERN_SYMBOLS_REV = np.array([-3, -3, -3, -3, -3, +3, -3, -3, +3, +3, -3, -3,
                                          +3, +3, +3, +3, -3, +3, -3, +3, +3, +3, +3, +3], dtype=np.float32)

    def __init__(self) -> None:
        self.duid_to_frame_type = {
            self.DUID_HDU: P25FrameType.HDU,
            self.DUID_TDU: P25FrameType.TDU,
            self.DUID_LDU1: P25FrameType.LDU1,
            self.DUID_LDU2: P25FrameType.LDU2,
            self.DUID_TSDU: P25FrameType.TSDU,
            self.DUID_PDU: P25FrameType.PDU,
            self.DUID_TDULC: P25FrameType.TDU,
        }
        # Hard sync threshold (allow 4 dibit errors)
        self.sync_threshold = 4
        # Soft correlation mode
        self.use_soft_sync = True
        # Polarity reversal tracking (OP25-style)
        self.reverse_p: int = 0  # 0 = normal, 2 = reversed (XOR mask)
        # NAC tracking for BCH decode assistance
        self._tracked_nac: int | None = None

    def _decode_nid_with_bch(self, nid_dibits: np.ndarray) -> tuple[int, int, int]:
        """
        Decode NID (Network ID) using BCH(63,16,23) error correction.

        Args:
            nid_dibits: 32 dibits containing the NID (64 bits)

        Returns:
            Tuple of (nac, duid, errors) where:
            - nac: 12-bit Network Access Code
            - duid: 4-bit Data Unit ID
            - errors: Number of errors corrected (-1 if BCH failed)
        """
        # Debug: log first 8 dibits (NAC + DUID) periodically
        if not hasattr(self, '_nid_debug_count'):
            self._nid_debug_count = 0
        self._nid_debug_count += 1
        if self._nid_debug_count <= 10 or self._nid_debug_count % 100 == 0:
            dibit_str = ' '.join(str(int(d)) for d in nid_dibits[:8])
            logger.info(f"NID decode #{self._nid_debug_count}: dibits[0:8]={dibit_str}")

        # Convert 32 dibits to 64 bits
        bits = np.zeros(64, dtype=np.uint8)
        for i, d in enumerate(nid_dibits):
            bits[i * 2] = (int(d) >> 1) & 1
            bits[i * 2 + 1] = int(d) & 1

        # BCH(63,16,23) uses 63 bits - first 63 of 64
        bch_codeword = bits[:63]

        # BCH decode with tracked NAC for assistance
        decoded_data, errors = bch_decode(bch_codeword, self._tracked_nac)

        if errors < 0:
            # BCH decode failed - fall back to simple extraction
            nac = 0
            for i in range(6):
                nac = (nac << 2) | int(nid_dibits[i])
            duid = int((nid_dibits[6] << 2) | nid_dibits[7])
            return nac, duid, 99  # 99 = BCH failed

        # Extract NAC (12 bits) and DUID (4 bits) from decoded data
        nac = (decoded_data >> 4) & 0xFFF
        duid = decoded_data & 0xF

        return nac, duid, errors

    def _soft_correlation(self, dibits: np.ndarray, check_reversed: bool = False) -> tuple[float, bool]:
        """
        Compute soft correlation score between dibits and sync pattern.

        SDRTrunk-style: dot product of received symbols with ideal sync symbols.
        Max score is 24 * 9 = 216 when all symbols match perfectly at ±3.

        Args:
            dibits: 24 dibits to correlate against sync pattern
            check_reversed: Also check for reversed polarity sync

        Returns:
            Tuple of (score, is_reversed) where is_reversed indicates polarity reversal detected
        """
        # Convert dibits to symbols: 0->+1, 1->+3, 2->-1, 3->-3
        symbols = self.DIBIT_TO_SYMBOL[np.clip(dibits[:24], 0, 3)]

        # Dot product with normal sync pattern
        normal_score = float(np.dot(symbols, self.SYNC_PATTERN_SYMBOLS))

        if not check_reversed:
            return normal_score, False

        # Also check reversed polarity sync pattern
        # OP25 is strict about polarity flips: only flip if reversed is clearly better
        # (normal allows some errors, reversed requires near-perfect match)
        rev_score = float(np.dot(symbols, self.SYNC_PATTERN_SYMBOLS_REV))

        # Polarity detection disabled for now - causes flip-flopping with weak signals.
        # The sample file test shows normal polarity works correctly.
        # TODO: Re-enable when we have better sync threshold management or
        # implement OP25-style "detect once and latch" polarity handling.
        #
        # Original condition was: if rev_score > 100 and normal_score < -50:
        #     return rev_score, True

        # Return absolute best score to help find sync even with questionable polarity
        if normal_score < 0:
            return max(abs(normal_score), abs(rev_score)), False

        return normal_score, False

    def find_sync(self, dibits: np.ndarray) -> tuple[int | None, P25FrameType | None, int, int]:
        """
        Search for P25 frame sync pattern in dibit stream.

        Uses SDRTrunk-style soft correlation for more robust detection.
        Falls back to hard matching if soft detection fails.
        Automatically detects and corrects polarity reversal (OP25-style).

        The P25 frame structure is:
        - 48-bit frame sync (24 dibits)
        - 64-bit NID (32 dibits): NAC (12 bits) + DUID (4 bits) + parity

        Returns:
            (sync_position, frame_type, nac, duid) or (None, None, 0, 0) if not found
        """
        # Need at least sync (24 dibits) + NID (8 dibits for NAC+DUID minimum)
        if len(dibits) < 32:
            return None, None, 0, 0

        # Ensure dibits are uint8 with values 0-3
        if dibits.dtype != np.uint8:
            dibits = dibits.astype(np.uint8)

        # Apply current polarity correction (OP25-style: dibit ^= reverse_p)
        if self.reverse_p:
            logger.debug(f"P25 find_sync: applying polarity XOR {self.reverse_p}")
            dibits = dibits ^ self.reverse_p
        else:
            logger.debug(f"P25 find_sync: no polarity XOR (reverse_p={self.reverse_p})")

        # Clip to valid dibit range (0-3)
        if np.any(dibits > 3):
            logger.warning(f"P25 find_sync: dibits out of range (max={dibits.max()}), clipping")
            dibits = np.clip(dibits, 0, 3).astype(np.uint8)

        sync_len = len(self.FRAME_SYNC_DIBITS)
        best_pos = None
        best_score = 0.0
        polarity_flip_detected = False

        # Search for frame sync pattern
        for start_pos in range(len(dibits) - sync_len - 8):  # Need sync + some NID
            window = dibits[start_pos:start_pos + sync_len]

            if self.use_soft_sync:
                # SDRTrunk-style soft correlation, also check for reversed polarity
                score, is_reversed = self._soft_correlation(window, check_reversed=True)

                if score > best_score:
                    best_score = score
                    best_pos = start_pos
                    polarity_flip_detected = is_reversed

                # Early exit if we find a strong match
                if score >= self.SOFT_SYNC_THRESHOLD_OPTIMIZE:
                    break
            else:
                # Hard dibit matching (legacy) - check both polarities
                errors_normal = int(np.sum(window != self.FRAME_SYNC_DIBITS))
                errors_rev = int(np.sum(window != self.FRAME_SYNC_DIBITS_REV))

                if errors_rev < errors_normal and errors_rev <= self.sync_threshold:
                    best_pos = start_pos
                    best_score = 216 - errors_rev * 9
                    polarity_flip_detected = True
                    break
                elif errors_normal <= self.sync_threshold:
                    best_pos = start_pos
                    best_score = 216 - errors_normal * 9
                    break

        # Check if we found sync above threshold
        if best_pos is None:
            return None, None, 0, 0

        if self.use_soft_sync and best_score < self.SOFT_SYNC_THRESHOLD:
            # Log near-misses for debugging
            if best_score > 30:
                logger.debug(f"P25 soft sync near-miss: score={best_score:.1f} < threshold={self.SOFT_SYNC_THRESHOLD}")
            return None, None, 0, 0

        # Auto-flip polarity if reversed sync detected (OP25-style)
        if polarity_flip_detected:
            old_p = self.reverse_p
            self.reverse_p ^= 0x02  # Toggle between 0 and 2
            logger.info(f"P25: Reversed FS polarity detected - autocorrecting (reverse_p={old_p}->{self.reverse_p}, id={id(self):#x}, best_score={best_score:.1f}, best_pos={best_pos})")
            # Re-apply polarity correction to the dibits we'll use for NID
            dibits = dibits ^ 0x02

        # Extract NID with BCH error correction
        # Note: NID contains a status symbol at position 11 (position 35 from frame start)
        # Per OP25/TIA-102.BAAA, we need 33 raw dibits to get 32 clean NID dibits
        nid_start = best_pos + sync_len
        if nid_start + 33 > len(dibits):  # Need 33 raw dibits (32 NID + 1 status)
            # Fallback to simple extraction if not enough data
            if nid_start + 8 > len(dibits):
                return None, None, 0, 0
            # Simple extraction without BCH (no status stripping for short reads)
            nac_dibits = dibits[nid_start:nid_start + 6]
            nac = 0
            for d in nac_dibits:
                nac = (nac << 2) | int(d)
            duid_dibits = dibits[nid_start + 6:nid_start + 8]
            duid = int((duid_dibits[0] << 2) | duid_dibits[1])
            bch_errors = 99  # Mark as uncorrected
        else:
            # Extract 33 raw dibits and remove status symbol at position 11
            raw_nid = dibits[nid_start:nid_start + 33]
            # Skip status symbol: positions 0-10 + positions 12-32 = 32 clean dibits
            nid_dibits = np.concatenate([raw_nid[:11], raw_nid[12:33]])
            nac, duid, bch_errors = self._decode_nid_with_bch(nid_dibits)

        frame_type = self.duid_to_frame_type.get(duid, P25FrameType.UNKNOWN)

        # Track NAC for future decodes
        if hasattr(self, '_tracked_nac') and 0x001 <= nac <= 0xFFE and bch_errors < 10:
            self._tracked_nac = nac

        # Debug: log NAC and DUID for first few syncs
        if not hasattr(self, '_sync_debug_count'):
            self._sync_debug_count = 0
        self._sync_debug_count += 1
        if self._sync_debug_count <= 20:
            sync_method = "soft" if self.use_soft_sync else "hard"
            logger.info(
                f"P25FrameSync ({sync_method}): pos={best_pos}, score={best_score:.1f}, "
                f"NAC={nac:03x}, DUID={duid:x} -> {frame_type}, BCH_err={bch_errors}"
            )

        logger.debug(f"P25 sync found at {best_pos}, score={best_score:.1f}, DUID={duid:x} -> {frame_type}")
        return best_pos, frame_type, nac, duid


class P25Modulation(str, Enum):
    """P25 modulation types."""
    C4FM = "c4fm"    # Phase 1: Standard 4-level FSK (non-simulcast)
    LSM = "lsm"      # Phase 1: Linear Simulcast Modulation (CQPSK/differential QPSK)
    PHASE2 = "phase2"  # Phase 2: CQPSK TDMA with 2 timeslots at 6000 symbols/second


class P25Decoder:
    """
    Complete P25 Phase 1 and Phase 2 decoder with trunking support.

    Supports multiple modulation types:
    - C4FM: Phase 1 standard 4-level FSK (most P25 systems)
    - LSM: Phase 1 CQPSK for simulcast systems using phase modulation
    - PHASE2: Phase 2 CQPSK TDMA with 2 timeslots at 6000 symbols/second

    Phase 2 uses TDMA with SuperFrame structure:
    - 720 dibits per SuperFrame fragment
    - 4 timeslots per fragment (2 logical channels)
    - Sync patterns at positions 360 and 540 dibits

    Also supports discriminator audio input via process_discriminator() for:
    - Pre-recorded discriminator tap audio
    - DSD-FME compatible .dis files
    - Scanner discriminator output
    """

    # P25 frame sizes in dibits
    # Frame sync is 24 dibits (48 bits) + NID is 32 dibits (64 bits)
    MIN_SYNC_DIBITS = 32  # Sync (24) + minimum NID (8) for DUID extraction
    MIN_FRAME_DIBITS = 150  # Minimum to attempt frame decode (sync + NID + some data)
    MAX_BUFFER_DIBITS = 4000  # ~2 frames worth, prevent unbounded growth

    # Minimum sample rate for C4FM demodulation (4 samples/symbol at 4800 baud)
    MIN_SAMPLE_RATE = 19200  # 4 sps

    def __init__(
        self,
        sample_rate: int = 48000,
        modulation: P25Modulation = P25Modulation.C4FM,
    ) -> None:
        self.sample_rate = sample_rate
        self.modulation = modulation
        self.demodulator: CQPSKDemodulator | _C4FMDemodulatorWrapper

        # Only resample if input rate is below minimum
        # Resampling degrades signal quality, so avoid when possible
        if sample_rate >= self.MIN_SAMPLE_RATE:
            # Use input rate directly - no resampling
            self._upsample_up = 1
            self._upsample_down = 1
            self._demod_sample_rate = sample_rate
            sps = sample_rate / 4800
            logger.info(f"P25 decoder: using input rate {sample_rate}Hz directly ({sps:.1f} sps)")
        else:
            # Upsample to minimum rate
            from math import gcd
            target_rate = self.MIN_SAMPLE_RATE
            g = gcd(target_rate, sample_rate)
            self._upsample_up = target_rate // g
            self._upsample_down = sample_rate // g
            self._demod_sample_rate = target_rate
            logger.info(f"P25 decoder: input {sample_rate}Hz -> upsampling {self._upsample_up}/{self._upsample_down} -> {target_rate}Hz")

        # Select demodulator based on modulation type - always use demod sample rate
        if modulation == P25Modulation.PHASE2:
            # Phase 2 uses CQPSK at 6000 symbols/second with TDMA
            self.demodulator = CQPSKDemodulator(self._demod_sample_rate, symbol_rate=6000)
            logger.info(f"P25 decoder initialized with Phase 2 CQPSK/TDMA demodulator (demod_rate={self._demod_sample_rate})")
        elif modulation == P25Modulation.LSM:
            self.demodulator = CQPSKDemodulator(self._demod_sample_rate)
            logger.info(f"P25 decoder initialized with CQPSK/LSM demodulator (demod_rate={self._demod_sample_rate})")
        else:
            # Use working C4FM demodulator from dsp/p25/c4fm.py (with Gardner timing)
            self.demodulator = _C4FMDemodulatorWrapper(self._demod_sample_rate)
            logger.info(f"P25 decoder initialized with C4FM demodulator (demod_rate={self._demod_sample_rate})")

        # Discriminator demodulator (created on demand)
        self._discriminator_demod: DiscriminatorDemodulator | None = None

        # Legacy frame sync (for backward compatibility)
        self.frame_sync = P25FrameSync()
        self.trellis = P25TrellisDecoder()

        # SDRTrunk-compatible streaming message framer (Phase 1)
        self._message_framer = P25P1MessageFramer()
        self._message_framer.set_listener(self._on_framer_message)
        self._message_framer.start()
        self._framed_messages: list[P25P1Message] = []

        # Phase 2 TDMA decoder (only initialized if using Phase 2)
        self._phase2_decoder: _P25P2StreamingDecoder | None = None
        self._phase2_timeslots: list[P25P2Timeslot] = []
        if modulation == P25Modulation.PHASE2:
            self._phase2_decoder = _P25P2StreamingDecoder()
            self._phase2_decoder.on_timeslot = self._on_phase2_timeslot
            logger.info("P25 Phase 2 TDMA decoder initialized")

        # TSBK parser for full parsing of control channel messages
        self.tsbk_parser = TSBKParser()

        # Pre-allocated ring buffer for dibit accumulation (legacy, for batch mode)
        self._dibit_buffer = DibitRingBuffer(capacity=self.MAX_BUFFER_DIBITS)

        # Use streaming mode by default (SDRTrunk-compatible)
        self._use_streaming_framer = True

        # Trunking state
        self.control_channel = True  # Are we on control channel?
        self.current_tgid: int | None = None
        self.voice_channel_freq: float | None = None

        # Callbacks
        self.on_voice_frame: Callable[[bytes], None] | None = None
        self.on_tsbk_message: Callable[[dict[str, Any]], None] | None = None
        self.on_grant: Callable[[int, float], None] | None = None  # (tgid, freq)
        self.on_location: Callable[[dict[str, Any]], None] | None = None  # GPS location from ELC

        # Debug counters
        self._process_count = 0
        self._no_sync_count = 0
        self._sync_count = 0
        self._tsbk_decode_count = 0

    def _on_framer_message(self, message: P25P1Message) -> None:
        """Callback from streaming framer when a message is assembled."""
        self._framed_messages.append(message)

    def _on_phase2_timeslot(self, timeslot: P25P2Timeslot) -> None:
        """Callback from Phase 2 decoder when a timeslot is received."""
        self._phase2_timeslots.append(timeslot)

    def process_iq(self, iq: np.ndarray) -> list[P25Frame]:
        """
        Process IQ samples and decode P25 frames.

        Uses SDRTrunk-compatible streaming architecture by default,
        processing symbols one at a time through the message framer.

        For Phase 2, uses TDMA SuperFrame detection and timeslot demultiplexing.

        Args:
            iq: Complex IQ samples

        Returns:
            List of decoded P25 frames
        """
        self._process_count += 1

        # Upsample IQ if input sample rate is below minimum for accurate demodulation
        if self._upsample_up != 1 or self._upsample_down != 1:
            # resample_poly handles complex arrays correctly (processes real/imag separately)
            iq = resample_poly(iq, self._upsample_up, self._upsample_down)

        # Route Phase 2 to dedicated TDMA processor
        if self.modulation == P25Modulation.PHASE2:
            return self._process_iq_phase2(iq)

        # Use streaming framer (SDRTrunk-compatible)
        if self._use_streaming_framer:
            return self._process_iq_streaming(iq)

        # Legacy batch processing
        return self._process_iq_batch(iq)

    def _process_iq_streaming(self, iq: np.ndarray) -> list[P25Frame]:
        """Process IQ using SDRTrunk-compatible streaming framer."""
        # Clear message buffer
        self._framed_messages.clear()

        # Demodulate to both dibits and soft symbols
        if hasattr(self.demodulator, 'demodulate_soft'):
            dibits, soft_symbols = self.demodulator.demodulate_soft(iq)
        else:
            # Fallback for demodulators without soft output
            dibits = self.demodulator.demodulate(iq)
            # Approximate soft symbols from dibits: 0->+1, 1->+3, 2->-1, 3->-3
            dibit_to_soft = np.array([1.0, 3.0, -1.0, -3.0], dtype=np.float32)
            soft_symbols = dibit_to_soft[np.clip(dibits, 0, 3)]

        if len(dibits) == 0:
            return []

        # Validate dibits
        if np.any(dibits > 3):
            logger.warning(f"P25: dibits out of range (max={dibits.max()}), clipping")
            dibits = np.clip(dibits, 0, 3).astype(np.uint8)

        # Process symbols through streaming framer using BATCH method
        # This is ~50-100x faster than the per-symbol loop because
        # sync correlation is done vectorized with numpy
        self._message_framer.process_batch(soft_symbols, dibits)

        # Convert framer messages to P25Frame objects
        frames = []
        for msg in self._framed_messages:
            frame = self._convert_framer_message(msg)
            if frame is not None:
                frames.append(frame)
                self._sync_count += 1

                # Handle trunking logic
                if frame.tsbk_opcode is not None and frame.tsbk_data:
                    self._handle_tsbk(frame)

        # Log status periodically
        if self._process_count % 100 == 0:
            logger.info(
                f"P25 decoder (streaming): processed={self._process_count}, "
                f"syncs={self._sync_count}, frames={len(frames)}"
            )

        return frames

    def _convert_framer_message(self, msg: P25P1Message) -> P25Frame | None:
        """Convert a P25P1Message from the streaming framer to a P25Frame."""
        # Map DUID to frame type
        duid_to_frame_type = {
            P25P1DataUnitID.HEADER_DATA_UNIT: P25FrameType.HDU,
            P25P1DataUnitID.TERMINATOR_DATA_UNIT: P25FrameType.TDU,
            P25P1DataUnitID.LOGICAL_LINK_DATA_UNIT_1: P25FrameType.LDU1,
            P25P1DataUnitID.LOGICAL_LINK_DATA_UNIT_2: P25FrameType.LDU2,
            P25P1DataUnitID.TRUNKING_SIGNALING_BLOCK_1: P25FrameType.TSDU,
            P25P1DataUnitID.TRUNKING_SIGNALING_BLOCK_2: P25FrameType.TSDU,
            P25P1DataUnitID.TRUNKING_SIGNALING_BLOCK_3: P25FrameType.TSDU,
            P25P1DataUnitID.PACKET_DATA_UNIT: P25FrameType.PDU,
            P25P1DataUnitID.TERMINATOR_DATA_UNIT_LINK_CONTROL: P25FrameType.TDU,
        }

        frame_type = duid_to_frame_type.get(msg.duid, P25FrameType.UNKNOWN)

        # Create base frame
        frame = P25Frame(
            frame_type=frame_type,
            nac=msg.nac,
            duid=int(msg.duid),
            errors=msg.corrected_bit_count,
        )

        # Decode frame content based on type
        if frame_type == P25FrameType.TSDU:
            # Decode TSBK from message bits
            try:
                tsbk_result = self._decode_tsbk_from_bits(msg.bits)
                if tsbk_result:
                    frame.tsbk_opcode = tsbk_result.get('opcode')
                    frame.tsbk_data = tsbk_result
            except Exception as e:
                logger.debug(f"TSBK decode error: {e}")

        elif frame_type in (P25FrameType.LDU1, P25FrameType.LDU2):
            # Extract voice and link control data
            try:
                if frame_type == P25FrameType.LDU1:
                    lc_info = self._decode_ldu1_from_bits(msg.bits)
                else:
                    lc_info = self._decode_ldu2_from_bits(msg.bits)
                if lc_info:
                    frame.tgid = lc_info.get('tgid')
                    frame.source = lc_info.get('source')
                    frame.algid = lc_info.get('algid')
                    frame.kid = lc_info.get('kid')
            except Exception as e:
                logger.debug(f"LDU decode error: {e}")

        return frame

    def _decode_tsbk_from_bits(self, bits: np.ndarray) -> dict[str, Any] | None:
        """Decode TSBK message from raw bits."""
        if len(bits) < 196:
            return None

        # Convert bits back to dibits for existing decoder
        dibits = np.zeros(98, dtype=np.uint8)
        for i in range(98):
            if i * 2 + 1 < len(bits):
                dibits[i] = (int(bits[i * 2]) << 1) | int(bits[i * 2 + 1])

        # Use existing TSBK parser
        return self._decode_tsdu_dibits(dibits)

    def _decode_ldu1_from_bits(self, bits: np.ndarray) -> dict[str, Any] | None:
        """Decode LDU1 link control from bits."""
        # LDU1 contains Link Control in specific positions
        # For now, return minimal info
        return {'type': 'LDU1'}

    def _decode_ldu2_from_bits(self, bits: np.ndarray) -> dict[str, Any] | None:
        """Decode LDU2 encryption info from bits."""
        # LDU2 contains Encryption Sync Parameters
        return {'type': 'LDU2'}

    def _decode_tsdu_dibits(self, dibits: np.ndarray) -> dict[str, Any] | None:
        """Decode TSDU from dibits using existing _decode_tsdu logic.

        This is a wrapper for the streaming framer that takes clean dibits
        and returns the TSBK data dictionary.
        """
        if len(dibits) < 98:
            return None

        # Try trellis decoding with deinterleave
        try:
            deint = self._deinterleave_data(dibits[:98])
            decoded_dibits, errors = self.trellis.decode(deint)

            if errors >= 0 and decoded_dibits is not None and len(decoded_dibits) >= 48:
                # Convert 48 decoded dibits to 96 bits
                # Each dibit contains 2 bits: bit0 = (dibit >> 1), bit1 = (dibit & 1)
                decoded_bits = np.zeros(96, dtype=np.uint8)
                for i in range(48):
                    decoded_bits[i * 2] = (decoded_dibits[i] >> 1) & 1
                    decoded_bits[i * 2 + 1] = decoded_dibits[i] & 1
                return self._parse_tsbk_bits(decoded_bits)
        except Exception as e:
            logger.debug(f"TSDU decode failed: {e}")

        return None

    def _parse_tsbk_bits(self, decoded_bits: np.ndarray) -> dict[str, Any] | None:
        """Parse TSBK message from trellis-decoded bits."""
        if len(decoded_bits) < 96:
            return None

        # TSBK structure (96 bits = 12 bytes):
        # LB (1), Protect (1), Opcode (6), MFID (8), Data (64), CRC (16)
        lb = int(decoded_bits[0])
        protect = int(decoded_bits[1])
        opcode = 0
        for i in range(6):
            opcode = (opcode << 1) | int(decoded_bits[2 + i])
        mfid = 0
        for i in range(8):
            mfid = (mfid << 1) | int(decoded_bits[8 + i])

        # Extract 64-bit data payload
        data_bits = decoded_bits[16:80]
        data_bytes = bytes(
            int(''.join(str(int(b)) for b in data_bits[i:i+8]), 2)
            for i in range(0, 64, 8)
        )

        # Use existing TSBK parser for full decoding
        tsbk_data: dict[str, Any] | None = self.tsbk_parser.parse(opcode, mfid, data_bytes)

        if tsbk_data:
            tsbk_data['opcode'] = opcode
            tsbk_data['mfid'] = mfid
            tsbk_data['last_block'] = lb
            tsbk_data['protected'] = protect
            return tsbk_data

        return {
            'opcode': opcode,
            'mfid': mfid,
            'last_block': lb,
            'protected': protect,
            'raw_data': data_bytes.hex(),
        }

    def _process_iq_phase2(self, iq: np.ndarray) -> list[P25Frame]:
        """Process IQ using Phase 2 TDMA decoder.

        Phase 2 uses SuperFrame fragments (720 dibits) with 4 timeslots.
        Each fragment contains 2 sync patterns at positions 360 and 540.
        """
        if self._phase2_decoder is None:
            logger.warning("Phase 2 decoder not initialized")
            return []

        # Clear timeslot buffer
        self._phase2_timeslots.clear()

        # Demodulate to dibits
        dibits = self.demodulator.demodulate(iq)

        if len(dibits) == 0:
            return []

        # Process dibits through Phase 2 TDMA decoder
        timeslots = self._phase2_decoder.process_dibits(dibits)

        # Convert timeslots to P25Frame objects
        frames: list[P25Frame] = []
        for ts in timeslots:
            # Create a P25Frame for each timeslot
            # Phase 2 timeslots contain voice or SACCH/FACCH data
            frame_type = (
                P25FrameType.LDU1
                if ts.slot_type == P25P2TimeslotType.VOICE
                else P25FrameType.TSDU
            )

            frame = P25Frame(
                frame_type=frame_type,
                nac=0,  # NAC is in ISCH, not decoded yet
                duid=0,
                voice_data=ts.dibits.tobytes() if ts.slot_type == P25P2TimeslotType.VOICE else None,
                tsbk_data={
                    "phase": 2,
                    "timeslot": ts.timeslot_number,
                    "slot_type": ts.slot_type.name,
                    "isch": ts.isch_dibits.tobytes().hex(),
                    "timestamp": ts.timestamp,
                },
            )
            frames.append(frame)

        if frames:
            logger.debug(f"P25 Phase 2: decoded {len(frames)} timeslots")

        return frames

    def _process_iq_batch(self, iq: np.ndarray) -> list[P25Frame]:
        """Legacy batch processing (pre-streaming framer)."""
        # Demodulate to dibits
        new_dibits = self.demodulator.demodulate(iq)

        if len(new_dibits) == 0:
            return []

        # Validate and clip dibits to valid range (0-3) before buffering
        if np.any(new_dibits > 3):
            logger.warning(f"P25: new dibits out of range (max={new_dibits.max()}), clipping")
            new_dibits = np.clip(new_dibits, 0, 3).astype(np.uint8)

        # Append to ring buffer (auto-discards oldest when full)
        self._dibit_buffer.append(new_dibits)

        # Log status periodically
        if self._process_count % 100 == 0:
            logger.info(f"P25 decoder: processed={self._process_count}, syncs={self._sync_count}, no_sync={self._no_sync_count}, buffer={len(self._dibit_buffer)}")

        # Need at least MIN_FRAME_DIBITS for meaningful frame decode
        if len(self._dibit_buffer) < self.MIN_FRAME_DIBITS:
            return []

        # Get contiguous view of buffer for sync detection
        buffer_data = self._dibit_buffer.get_contiguous()

        # Find frame sync in buffer
        sync_pos, frame_type, nac, duid = self.frame_sync.find_sync(buffer_data)
        if frame_type is None:
            frame_type = P25FrameType.UNKNOWN

        if sync_pos is None:
            self._no_sync_count += 1
            return []

        self._sync_count += 1
        logger.info(f"Found P25 frame sync at position {sync_pos}: {frame_type} NAC={nac:03X} (buffer={len(self._dibit_buffer)})")

        # Work with the buffer_data array from here on

        # P25 frame structure (per TIA-102.BAAA and SDRTrunk):
        # - 24 dibits: Frame sync (FS)
        # - 32 dibits: NID (NAC + DUID + BCH parity) - but includes 1 status at position 36
        # - 1 status dibit at position 36 (within NID)
        # - Frame data starts after position 57 (24 + 32 + 1 status = 57 raw dibits)
        #
        # Status symbols occur every 36 dibits from frame start:
        # - Position 36: Status 1 (in NID)
        # - Position 72: Status 2 (in data)
        # - Position 108: Status 3
        # - etc.
        # We use 8 for minimal extraction, but for proper framing need to account for status

        # For TSDU, we need to extract starting at position 57 (after sync + NID + 1 status)
        # and then strip subsequent status symbols every 36 dibits
        if frame_type == P25FrameType.TSDU:
            # TSDU starts after sync + NID + 1 status = 57 dibits
            header_raw_dibits = 57  # 24 + 32 + 1
        else:
            # For other frame types, use minimal header (we'll fix these later)
            header_raw_dibits = 32  # Just sync + minimal NID

        # Minimum raw frame data dibits required per frame type
        # For TSDU: 98 clean dibits requires ~101 raw dibits (with status symbols)
        MIN_FRAME_DATA: dict[P25FrameType, int] = {
            P25FrameType.HDU: 100,    # Header data unit
            P25FrameType.LDU1: 900,   # Voice frame 1
            P25FrameType.LDU2: 900,   # Voice frame 2
            P25FrameType.TDU: 10,     # Terminator (short)
            P25FrameType.TSDU: 104,   # 98 clean + ~3 status symbols + margin
            P25FrameType.PDU: 100,    # Packet data unit
            P25FrameType.UNKNOWN: 32, # Minimum for any unknown frame
        }

        # Calculate available frame data
        available_data = len(buffer_data) - sync_pos - header_raw_dibits
        min_required = MIN_FRAME_DATA.get(frame_type, MIN_FRAME_DATA[P25FrameType.UNKNOWN])

        if available_data < min_required:
            # Not enough data for this frame type - keep sync position and wait for more
            logger.debug(f"P25: Need more data for {frame_type}: have {available_data}, need {min_required}")
            # Trim buffer to start at sync position (discard data before sync)
            self._dibit_buffer.consume(sync_pos)
            return []

        # Extract frame data after header (from the buffer_data array we already have)
        frame_dibits = buffer_data[sync_pos + header_raw_dibits:]

        # Consume the entire frame from buffer
        # For TSDU, need raw dibits including status symbols
        if frame_type == P25FrameType.TSDU:
            # TSDU: 98 clean dibits + 3 status symbols = 101 raw dibits
            # But assembler may need up to 104 raw to get 98 clean
            consume_len = sync_pos + header_raw_dibits + 104
        elif frame_type in (P25FrameType.LDU1, P25FrameType.LDU2):
            # LDU frames are ~1800 bits = 900 dibits
            consume_len = sync_pos + header_raw_dibits + 900
        elif frame_type == P25FrameType.HDU:
            consume_len = sync_pos + header_raw_dibits + 500  # HDU is ~648 bits
        else:
            # For TDU and others, consume header plus minimal frame
            consume_len = sync_pos + header_raw_dibits + min_required

        # Consume from ring buffer (don't consume more than available)
        consume_len = min(consume_len, len(buffer_data))
        self._dibit_buffer.consume(consume_len)

        # Decode frame based on type
        frames = []
        if frame_type == P25FrameType.HDU:
            frame = self._decode_hdu(frame_dibits)
        elif frame_type == P25FrameType.LDU1:
            frame = self._decode_ldu1(frame_dibits)
        elif frame_type == P25FrameType.LDU2:
            frame = self._decode_ldu2(frame_dibits)
        elif frame_type == P25FrameType.TDU:
            frame = self._decode_tdu(frame_dibits)
        elif frame_type == P25FrameType.TSDU:
            frame = self._decode_tsdu(frame_dibits)
        else:
            frame = P25Frame(frame_type=P25FrameType.UNKNOWN, nac=nac, duid=duid)

        if frame:
            if frame_type is not None:
                frame.frame_type = frame_type
            # Use NAC from find_sync (BCH-decoded) instead of frame decoder's extraction
            frame.nac = nac
            frame.duid = duid
            frames.append(frame)

            # Handle trunking logic
            if frame.tsbk_opcode is not None and frame.tsbk_data:
                self._handle_tsbk(frame)

        return frames

    def process_discriminator(
        self, audio: np.ndarray, sample_rate: int = 48000
    ) -> list[P25Frame]:
        """
        Process FM discriminator audio and decode P25 frames.

        Use this method when input is pre-demodulated FM audio (mono),
        such as discriminator tap recordings or .dis files.

        Args:
            audio: Mono audio samples (float32 or int16)
            sample_rate: Audio sample rate (default 48000)

        Returns:
            List of decoded P25 frames
        """
        # Create discriminator demodulator on demand with correct sample rate
        if self._discriminator_demod is None or self._discriminator_demod.sample_rate != sample_rate:
            self._discriminator_demod = DiscriminatorDemodulator(sample_rate=sample_rate)
            logger.info(f"P25 decoder: created discriminator demodulator (sample_rate={sample_rate})")

        self._process_count += 1

        # Normalize int16 to float
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        # Demodulate to dibits
        new_dibits = self._discriminator_demod.demodulate(audio)

        if len(new_dibits) == 0:
            return []

        # Validate and clip dibits
        if np.any(new_dibits > 3):
            logger.warning(f"P25: dibits out of range (max={new_dibits.max()}), clipping")
            new_dibits = np.clip(new_dibits, 0, 3).astype(np.uint8)

        # Append to ring buffer
        self._dibit_buffer.append(new_dibits)

        # Log status periodically
        if self._process_count % 100 == 0:
            logger.info(
                f"P25 discriminator: processed={self._process_count}, "
                f"syncs={self._sync_count}, buffer={len(self._dibit_buffer)}"
            )

        # Need enough dibits for frame decode
        if len(self._dibit_buffer) < self.MIN_FRAME_DIBITS:
            return []

        # From here, use the same frame decoding logic as process_iq
        buffer_data = self._dibit_buffer.get_contiguous()
        sync_pos, frame_type, nac, duid = self.frame_sync.find_sync(buffer_data)
        if frame_type is None:
            frame_type = P25FrameType.UNKNOWN

        if sync_pos is None:
            self._no_sync_count += 1
            return []

        self._sync_count += 1
        logger.info(f"Found P25 frame sync at position {sync_pos}: {frame_type} NAC={nac:03X}")

        # Calculate frame boundaries (same as process_iq)
        if frame_type == P25FrameType.TSDU:
            header_raw_dibits = 57
        else:
            header_raw_dibits = 32

        MIN_FRAME_DATA: dict[P25FrameType, int] = {
            P25FrameType.HDU: 100,
            P25FrameType.LDU1: 900,
            P25FrameType.LDU2: 900,
            P25FrameType.TDU: 10,
            P25FrameType.TSDU: 104,
            P25FrameType.PDU: 100,
            P25FrameType.UNKNOWN: 32,
        }

        available_data = len(buffer_data) - sync_pos - header_raw_dibits
        min_required = MIN_FRAME_DATA.get(frame_type, MIN_FRAME_DATA[P25FrameType.UNKNOWN])

        if available_data < min_required:
            self._dibit_buffer.consume(sync_pos)
            return []

        # Extract frame dibits
        if frame_type == P25FrameType.TSDU:
            frame_dibits = buffer_data[sync_pos + 57:sync_pos + 57 + 104]
        else:
            frame_len = min(900, len(buffer_data) - sync_pos - header_raw_dibits)
            frame_dibits = buffer_data[sync_pos + header_raw_dibits:sync_pos + header_raw_dibits + frame_len]

        consume_len = sync_pos + header_raw_dibits + len(frame_dibits)
        self._dibit_buffer.consume(consume_len)

        # Decode frame
        frames = []
        if frame_type == P25FrameType.HDU:
            frame = self._decode_hdu(frame_dibits)
        elif frame_type == P25FrameType.LDU1:
            frame = self._decode_ldu1(frame_dibits)
        elif frame_type == P25FrameType.LDU2:
            frame = self._decode_ldu2(frame_dibits)
        elif frame_type == P25FrameType.TDU:
            frame = self._decode_tdu(frame_dibits)
        elif frame_type == P25FrameType.TSDU:
            frame = self._decode_tsdu(frame_dibits)
        else:
            frame = P25Frame(frame_type=P25FrameType.UNKNOWN, nac=nac, duid=duid)

        if frame:
            if frame_type is not None:
                frame.frame_type = frame_type
            frame.nac = nac
            frame.duid = duid
            frames.append(frame)

            if frame.tsbk_opcode is not None and frame.tsbk_data:
                self._handle_tsbk(frame)

        return frames

    def _decode_hdu(self, dibits: np.ndarray) -> P25Frame | None:
        """Decode Header Data Unit"""
        # HDU contains:
        # - NAC (12 bits)
        # - DUID (4 bits)
        # - MI (Message Indicator, 72 bits)
        # - ALGID (8 bits) - Encryption algorithm
        # - KID (16 bits) - Key ID

        if len(dibits) < 100:
            return None

        # Extract NAC (12 bits = 6 dibits)
        nac = 0
        for i in range(6):
            nac = (nac << 2) | dibits[i]

        # Extract DUID (4 bits = 2 dibits)
        duid = 0
        for i in range(2):
            duid = (duid << 2) | dibits[6 + i]

        # Extract ALGID and KID (simplified)
        algid = 0
        kid = 0
        for i in range(4):
            algid = (algid << 2) | dibits[40 + i]
        for i in range(8):
            kid = (kid << 2) | dibits[44 + i]

        logger.info(f"HDU: NAC={nac:03x} ALGID={algid:02x} KID={kid:04x}")

        return P25Frame(
            frame_type=P25FrameType.HDU,
            nac=nac,
            duid=duid,
            algid=algid,
            kid=kid
        )

    def _decode_ldu1(self, dibits: np.ndarray) -> P25Frame | None:
        """Decode Logical Link Data Unit 1 (voice frame)"""
        if len(dibits) < 900:  # LDU1 is ~1800 bits
            return None

        # Extract link control data (contains TGID, source ID, and possibly GPS)
        link_control = extract_link_control(dibits)

        # If GPS data found, notify via callback
        if link_control.has_gps and self.on_location:
            location_data = {
                "source_id": link_control.source_id,
                "latitude": link_control.gps_latitude,
                "longitude": link_control.gps_longitude,
                "altitude_m": link_control.gps_altitude_m,
                "speed_kmh": link_control.gps_speed_kmh,
                "heading_deg": link_control.gps_heading_deg,
                "lcf": link_control.lcf,
            }
            logger.info(
                f"LDU1 GPS: unit={link_control.source_id} "
                f"lat={link_control.gps_latitude:.6f} lon={link_control.gps_longitude:.6f}"
            )
            self.on_location(location_data)

        # Extract voice IMBE frames (9 frames per LDU)
        voice_data = self._extract_imbe_frames(dibits)

        if self.on_voice_frame and voice_data:
            self.on_voice_frame(voice_data)

        return P25Frame(
            frame_type=P25FrameType.LDU1,
            nac=0,  # Would extract from frame
            duid=5,
            voice_data=voice_data,
            tgid=link_control.tgid if link_control.tgid else None,
            source=link_control.source_id if link_control.source_id else None,
        )

    def _decode_ldu2(self, dibits: np.ndarray) -> P25Frame | None:
        """Decode Logical Link Data Unit 2 (voice frame)"""
        if len(dibits) < 900:
            return None

        # Similar to LDU1 but with encryption sync
        voice_data = self._extract_imbe_frames(dibits)

        if self.on_voice_frame and voice_data:
            self.on_voice_frame(voice_data)

        return P25Frame(
            frame_type=P25FrameType.LDU2,
            nac=0,
            duid=10,
            voice_data=voice_data
        )

    def _decode_tdu(self, dibits: np.ndarray) -> P25Frame | None:
        """Decode Terminator Data Unit (end of transmission)"""
        logger.info("TDU: End of transmission")
        return P25Frame(frame_type=P25FrameType.TDU, nac=0, duid=3)

    # P25 Data Deinterleave pattern (98 dibits)
    # From p25.rs DeinterleaveRedirector - operates on dibit positions directly
    # deinterleaved_dibits[i] = input_dibits[DEINTERLEAVE[i]]
    DATA_DEINTERLEAVE = np.array([
        0, 1, 26, 27, 50, 51, 74, 75, 2, 3, 28, 29, 52, 53, 76, 77,
        4, 5, 30, 31, 54, 55, 78, 79, 6, 7, 32, 33, 56, 57, 80, 81,
        8, 9, 34, 35, 58, 59, 82, 83, 10, 11, 36, 37, 60, 61, 84, 85,
        12, 13, 38, 39, 62, 63, 86, 87, 14, 15, 40, 41, 64, 65, 88, 89,
        16, 17, 42, 43, 66, 67, 90, 91, 18, 19, 44, 45, 68, 69, 92, 93,
        20, 21, 46, 47, 70, 71, 94, 95, 22, 23, 48, 49, 72, 73, 96, 97, 24, 25
    ], dtype=np.int16)

    # P25 Data Interleave pattern (98 dibits) - inverse of deinterleave
    DATA_INTERLEAVE = np.array([
        0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57,
        64, 65, 72, 73, 80, 81, 88, 89, 96, 97, 2, 3, 10, 11, 18, 19,
        26, 27, 34, 35, 42, 43, 50, 51, 58, 59, 66, 67, 74, 75, 82, 83,
        90, 91, 4, 5, 12, 13, 20, 21, 28, 29, 36, 37, 44, 45, 52, 53,
        60, 61, 68, 69, 76, 77, 84, 85, 92, 93, 6, 7, 14, 15, 22, 23,
        30, 31, 38, 39, 46, 47, 54, 55, 62, 63, 70, 71, 78, 79, 86, 87, 94, 95
    ], dtype=np.int16)

    def _deinterleave_data(self, dibits: np.ndarray) -> np.ndarray:
        """
        Deinterleave P25 data block (TSBK).

        Uses 98-dibit deinterleave pattern directly on dibit positions.
        deinterleaved_dibits[i] = input_dibits[DEINTERLEAVE[i]]
        """
        if len(dibits) < 98:
            return dibits

        # Apply deinterleave pattern using advanced indexing (gather)
        return dibits[self.DATA_DEINTERLEAVE].astype(np.uint8)

    def _interleave_data(self, dibits: np.ndarray) -> np.ndarray:
        """
        Interleave P25 data block (reverse of deinterleave).
        Used for testing if data is already deinterleaved.
        """
        if len(dibits) < 98:
            return dibits

        # Apply interleave pattern using advanced indexing (gather)
        return dibits[self.DATA_INTERLEAVE].astype(np.uint8)

    # Pre-computed status symbol positions for common initial counters
    # Status symbols occur every 36 dibits from frame start (at 1-indexed positions 36, 72, 108, ...)
    # In 0-indexed frame positions: 35, 71, 107, ...
    # For TSDU data starting at frame position 57, first status is at 71 (relative position 14)
    # These arrays contain indices to KEEP (non-status positions) for up to 120 raw dibits
    _STATUS_KEEP_INDICES: dict[tuple[int, int], np.ndarray] = {}

    @classmethod
    def _get_status_keep_indices(cls, initial_counter: int, max_len: int = 120) -> np.ndarray:
        """Get pre-computed indices of non-status dibits for given initial counter."""
        cache_key = (initial_counter, max_len)
        if cache_key not in cls._STATUS_KEEP_INDICES:
            # Compute which indices to keep (not status symbols)
            # Counter increments each dibit; skip when counter reaches 36 (every 36 dibits)
            keep = []
            counter = initial_counter
            for i in range(max_len):
                counter += 1
                if counter == 36:  # Status symbol every 36 dibits
                    counter = 0
                    continue  # Skip this index
                keep.append(i)
            cls._STATUS_KEEP_INDICES[cache_key] = np.array(keep, dtype=np.int16)
        return cls._STATUS_KEEP_INDICES[cache_key]

    # Track if we've logged status stripping info (once per session)
    _status_strip_logged = False

    def _strip_status_symbols(self, dibits: np.ndarray, initial_counter: int = 21) -> np.ndarray:
        """
        Strip P25 status symbols from raw dibit stream.

        P25 inserts a status symbol every 36 dibits from frame start.
        Status symbols are at 0-indexed frame positions 35, 71, 107, 143, ...

        For TSDU data starting at frame position 57:
        - First status at position 71 = TSDU relative position 14
        - 57 % 36 = 21, so initial_counter = 21

        When counter reaches 36, that dibit is a status symbol and is skipped.

        Args:
            dibits: Raw dibit stream with embedded status symbols
            initial_counter: Starting value of status symbol counter (21 for TSDU at position 57)

        Returns:
            Clean dibit stream with status symbols removed
        """
        n = len(dibits)
        if n == 0:
            return np.array([], dtype=np.uint8)

        # Use pre-computed keep indices for vectorized extraction
        keep_indices = self._get_status_keep_indices(initial_counter, max_len=n)

        # Only use indices that are within bounds
        valid_indices = keep_indices[keep_indices < n]

        # Log status stripping details once per session for debugging
        if not P25Decoder._status_strip_logged and n >= 101:
            P25Decoder._status_strip_logged = True
            # Find which indices were skipped (status symbols)
            all_indices = set(range(n))
            kept = set(valid_indices)
            skipped = sorted(all_indices - kept)
            logger.info(
                f"Status strip: initial_counter={initial_counter}, input={n} dibits, "
                f"output={len(valid_indices)} dibits, skipped indices={skipped[:5]}..."
            )
            # Log actual dibit values at skipped positions
            if skipped:
                status_vals = [dibits[i] for i in skipped[:3]]
                logger.info(f"Status symbol values at skipped positions: {status_vals}")

        return np.asarray(dibits[valid_indices], dtype=np.uint8)

    def _decode_tsdu(self, dibits: np.ndarray) -> P25Frame | None:
        """
        Decode Trunking Signaling Data Unit (TSBK messages).

        A TSDU contains 1-3 TSBK (Trunking Signaling Block) messages.
        Each TSBK is:
        - 98 encoded dibits (which form 49 4-bit symbols)
        - Deinterleaved, then trellis decoded to 48 dibits (96 bits = 12 bytes)
        - Decoded content:
          - LB (1 bit): Last Block flag
          - Protect (1 bit): Protected flag
          - Opcode (6 bits): Message type
          - MFID (8 bits): Manufacturer ID (0=standard, 0x90=Motorola)
          - Data (64 bits = 8 bytes): Opcode-specific payload
          - CRC-16 (16 bits): Error check

        TSDU structure allows up to 3 TSBKs.
        """
        # One TSBK block = 98 encoded dibits (196 bits including trellis flush)
        # After trellis 1/2 rate decode: 49 dibits, but only first 48 are data
        TSBK_ENCODED_DIBITS = 98
        TSBK_DECODED_DIBITS = 48  # After 1/2 rate trellis decode (use first 48, discard flush)

        logger.debug(f"TSDU decode: received {len(dibits)} raw dibits")

        # Try multiple approaches to find the best decoding

        if len(dibits) < 98:
            logger.debug(f"TSDU too short: {len(dibits)} dibits (need 98+)")
            return None

        # Approach 1: Raw data without status stripping
        raw_dibits = dibits[:TSBK_ENCODED_DIBITS] if len(dibits) >= 98 else None

        # Approach 2: Strip status symbols (initial counter=21 for TSDU data at frame position 57)
        # Status symbols at frame positions 35, 71, 107... (0-indexed)
        # TSDU data starts at 57, first status at 71 = relative position 14
        clean_dibits = None
        if len(dibits) >= 101:
            clean_dibits = self._strip_status_symbols(dibits, initial_counter=21)
            clean_dibits = clean_dibits[:TSBK_ENCODED_DIBITS] if len(clean_dibits) >= 98 else None

        # Approach 3: Strip status with counter=0 (in case our counter is wrong)
        clean_dibits_c0 = None
        if len(dibits) >= 101:
            clean_dibits_c0 = self._strip_status_symbols(dibits, initial_counter=0)
            if len(clean_dibits_c0) >= 98:
                clean_dibits_c0 = clean_dibits_c0[:TSBK_ENCODED_DIBITS]
            else:
                clean_dibits_c0 = None

        # Collect all approaches with deinterleave variations
        results = []

        # Debug: dump first 20 dibits of raw data
        if raw_dibits is not None:
            dibit_str = ' '.join(str(d) for d in raw_dibits[:20])
            logger.debug(f"TSBK raw dibits[0:20]: {dibit_str}")

        for name, data in [("raw", raw_dibits), ("strip22", clean_dibits), ("strip0", clean_dibits_c0)]:
            if data is None:
                continue

            # Try all 4 phase rotations (QPSK ambiguity)
            # XOR 0: identity (no change)
            # XOR 1: swap 0↔1 and 2↔3 (90° rotation in QPSK space)
            # XOR 2: swap 0↔2 and 1↔3 (180° polarity flip)
            # XOR 3: swap 0↔3 and 1↔2 (270° rotation)
            for xor_mask in [0, 1, 2, 3]:
                rotated = (data ^ xor_mask).astype(np.uint8) if xor_mask != 0 else data

                # Try with deinterleave
                deint = self._deinterleave_data(rotated)
                _, err_deint = self.trellis.decode(deint)
                results.append((f"{name}_xor{xor_mask}_deint", deint, err_deint))

                # Try without deinterleave
                _, err_raw = self.trellis.decode(rotated)
                results.append((f"{name}_xor{xor_mask}_raw", rotated, err_raw))

        if not results:
            return None

        # Find best result
        best = min(results, key=lambda x: x[2])
        logger.info(f"TSBK decode attempts: {[(n, e) for n, _, e in results]}, best={best[0]} with {best[2]} errors")

        block_dibits = best[1]
        decoded, errors = self.trellis.decode(block_dibits)

        if decoded is None:
            logger.debug(f"TSBK trellis decode failed: errors={errors}")
            return None

        if len(decoded) < TSBK_DECODED_DIBITS:
            logger.debug(f"TSBK trellis output too short: {len(decoded)} dibits (need {TSBK_DECODED_DIBITS})")
            return None

        # Convert dibits to bits for CRC validation
        # P25 TSBK uses MSB-first bit ordering within each dibit
        decoded_bits = np.zeros(96, dtype=np.uint8)
        for i in range(min(48, len(decoded))):
            decoded_bits[i*2] = (decoded[i] >> 1) & 1
            decoded_bits[i*2 + 1] = decoded[i] & 1

        # Use syndrome-based CRC check from p25_frames (matches SDRTrunk)
        from wavecapsdr.decoders.p25_frames import crc16_ccitt_p25
        crc_valid, crc_errors = crc16_ccitt_p25(decoded_bits)
        # For logging, calculate the received CRC
        received_crc = 0
        for i in range(16):
            received_crc = (received_crc << 1) | int(decoded_bits[80 + i])
        crc = received_crc if crc_valid else 0  # Placeholder for logging

        # Log all decode attempts for debugging
        # Show first 12 decoded dibits (24 bits = 3 bytes: LB + opcode + MFID start)
        decoded_hex = ''.join(f'{d}' for d in decoded[:12])
        first_bytes = bytes([
            (decoded[0] << 6) | (decoded[1] << 4) | (decoded[2] << 2) | decoded[3],
            (decoded[4] << 6) | (decoded[5] << 4) | (decoded[6] << 2) | decoded[7],
            (decoded[8] << 6) | (decoded[9] << 4) | (decoded[10] << 2) | decoded[11]
        ])
        logger.info(f"TSBK CRC check: errors={errors}, crc_valid={crc_valid}, calc_crc=0x{crc:04x}, recv_crc=0x{received_crc:04x}, first_dibits={decoded_hex}, first_bytes={first_bytes.hex()}")

        # Check error threshold - temporarily relaxed for debugging
        if errors > 30 or (errors > 8 and not crc_valid):
            logger.debug(f"TSBK rejected: errors={errors}, crc_valid={crc_valid}")
            return None

        self._tsbk_decode_count += 1

        # Extract TSBK fields from decoded dibits
        # TSBK structure (96 decoded bits = 48 dibits):
        # - Bits 0-1: Last Block (LB) and Protect flags
        # - Bits 2-7: Opcode (6 bits)
        # - Bits 8-15: Manufacturer ID (8 bits)
        # - Bits 16-79: Data (64 bits = 8 bytes)
        # - Bits 80-95: CRC (16 bits)

        # Extract header: first 8 bits (4 dibits) = LB + Protect + Opcode
        header_bits = 0
        for i in range(4):
            header_bits = (header_bits << 2) | (decoded[i] & 0x3)

        lb = (header_bits >> 7) & 0x1
        protect = (header_bits >> 6) & 0x1
        opcode = header_bits & 0x3F

        # Extract MFID: next 8 bits (4 dibits)
        mfid = 0
        for i in range(4, 8):
            if i < len(decoded):
                mfid = (mfid << 2) | (decoded[i] & 0x3)

        # Extract data payload: 64 bits (32 dibits) starting at dibit 8
        data_bytes = bytearray()
        for byte_idx in range(8):  # 8 bytes of data
            byte_val = 0
            for dibit_idx in range(4):  # 4 dibits per byte
                dibit_pos = 8 + byte_idx * 4 + dibit_idx
                if dibit_pos < len(decoded):
                    byte_val = (byte_val << 2) | (decoded[dibit_pos] & 0x3)
            data_bytes.append(byte_val)

        # Use TSBKParser for full parsing
        tsbk_data = self.tsbk_parser.parse(opcode, mfid, bytes(data_bytes))
        tsbk_data['lb'] = lb
        tsbk_data['protect'] = protect
        tsbk_data['raw_opcode'] = opcode
        tsbk_data['trellis_errors'] = errors

        logger.info(f"TSBK: LB={lb} Opcode=0x{opcode:02X} MFID={mfid} Errors={errors} -> {tsbk_data.get('type', 'UNKNOWN')}")

        if self.on_tsbk_message:
            self.on_tsbk_message(tsbk_data)

        return P25Frame(
            frame_type=P25FrameType.TSDU,
            nac=0,
            duid=7,
            tsbk_opcode=opcode,
            tsbk_data=tsbk_data
        )

    def _decode_tsbk_opcode(self, opcode: int, dibits: np.ndarray) -> dict[str, Any]:
        """Decode TSBK opcode and extract trunking information.

        Uses correct P25 TIA-102.AABB opcode values matching SDRTrunk.
        """
        data: dict[str, Any] = {}

        # Voice grants (OSP) - 0x00-0x06
        if opcode == 0x00:  # Group Voice Channel Grant
            data['type'] = 'GRP_V_CH_GRANT'
            data['opcode_name'] = 'GRP_V_CH_GRANT'

        elif opcode == 0x02:  # Group Voice Channel Grant Update
            data['type'] = 'GRP_V_CH_GRANT_UPDT'
            data['opcode_name'] = 'GRP_V_CH_GRANT_UPDT'

        elif opcode == 0x03:  # Group Voice Channel Grant Update Explicit
            data['type'] = 'GRP_V_CH_GRANT_UPDT_EXP'
            data['opcode_name'] = 'GRP_V_CH_GRANT_UPDT_EXP'

        elif opcode == 0x04:  # Unit to Unit Voice Channel Grant
            data['type'] = 'UU_V_CH_GRANT'
            data['opcode_name'] = 'UU_V_CH_GRANT'

        elif opcode == 0x05:  # Unit to Unit Answer Request
            data['type'] = 'UU_ANS_REQ'
            data['opcode_name'] = 'UU_ANS_REQ'

        elif opcode == 0x06:  # Unit to Unit Voice Channel Grant Update
            data['type'] = 'UU_V_CH_GRANT_UPDT'
            data['opcode_name'] = 'UU_V_CH_GRANT_UPDT'

        # Telephone interconnect - 0x08-0x0A
        elif opcode == 0x08:  # Telephone Interconnect Voice Channel Grant
            data['type'] = 'TEL_INT_CH_GRANT'
            data['opcode_name'] = 'TEL_INT_CH_GRANT'

        # Control responses - 0x20-0x27
        elif opcode == 0x20:  # Acknowledge Response
            data['type'] = 'ACK_RSP'
            data['opcode_name'] = 'ACK_RSP'

        elif opcode == 0x21:  # Queued Response
            data['type'] = 'QUE_RSP'
            data['opcode_name'] = 'QUE_RSP'

        elif opcode == 0x24:  # Extended Function Command
            data['type'] = 'EXT_FNCT_CMD'
            data['opcode_name'] = 'EXT_FNCT_CMD'

        elif opcode == 0x27:  # Deny Response
            data['type'] = 'DENY_RSP'
            data['opcode_name'] = 'DENY_RSP'

        # Affiliation/Registration - 0x28-0x2F
        elif opcode == 0x28:  # Group Affiliation Response
            data['type'] = 'GRP_AFF_RSP'
            data['opcode_name'] = 'GRP_AFF_RSP'

        elif opcode == 0x2E:  # Authentication Command
            data['type'] = 'AUTH_CMD'
            data['opcode_name'] = 'AUTH_CMD'

        # Channel identification - 0x33-0x35
        elif opcode == 0x33:  # Identifier Update TDMA
            data['type'] = 'IDEN_UP_TDMA'
            data['opcode_name'] = 'IDEN_UP_TDMA'

        elif opcode == 0x34:  # Identifier Update VHF/UHF
            data['type'] = 'IDEN_UP_VU'
            data['opcode_name'] = 'IDEN_UP_VU'

        elif opcode == 0x35:  # Time and Date Announcement
            data['type'] = 'TIME_DATE_ANN'
            data['opcode_name'] = 'TIME_DATE_ANN'

        # System status broadcasts - 0x38-0x3D
        elif opcode == 0x38:  # System Service Broadcast
            data['type'] = 'SYS_SRV_BCAST'
            data['opcode_name'] = 'SYS_SRV_BCAST'

        elif opcode == 0x39:  # Secondary Control Channel Broadcast
            data['type'] = 'SCCB'
            data['opcode_name'] = 'SCCB'

        elif opcode == 0x3A:  # RFSS Status Broadcast
            data['type'] = 'RFSS_STS_BCAST'
            data['opcode_name'] = 'RFSS_STS_BCAST'

        elif opcode == 0x3B:  # Network Status Broadcast
            data['type'] = 'NET_STS_BCAST'
            data['opcode_name'] = 'NET_STS_BCAST'

        elif opcode == 0x3C:  # Adjacent Status Broadcast
            data['type'] = 'ADJ_STS_BCAST'
            data['opcode_name'] = 'ADJ_STS_BCAST'

        elif opcode == 0x3D:  # Identifier Update
            data['type'] = 'IDEN_UP'
            data['opcode_name'] = 'IDEN_UP'

        # Reserved opcodes that appear on SA-GRN
        elif opcode == 0x0C:  # Reserved
            data['type'] = 'OSP_RESERVED_0C'
            data['opcode_name'] = 'OSP_RESERVED_0C'

        elif opcode == 0x0E:  # Reserved
            data['type'] = 'OSP_RESERVED_0E'
            data['opcode_name'] = 'OSP_RESERVED_0E'

        elif opcode == 0x11:  # Group Data Channel Grant (obsolete)
            data['type'] = 'GRP_DATA_CH_GRANT'
            data['opcode_name'] = 'GRP_DATA_CH_GRANT'

        elif opcode == 0x13:  # Group Data Channel Announcement Explicit (obsolete)
            data['type'] = 'GRP_DATA_CH_ANN_EXP'
            data['opcode_name'] = 'GRP_DATA_CH_ANN_EXP'

        elif opcode == 0x1B:  # Reserved
            data['type'] = 'OSP_RESERVED_1B'
            data['opcode_name'] = 'OSP_RESERVED_1B'

        else:
            data['type'] = 'UNKNOWN'
            data['opcode'] = opcode
            data['opcode_name'] = f'UNKNOWN_0x{opcode:02X}'

        return data

    def _extract_imbe_frames(self, dibits: np.ndarray) -> bytes | None:
        """Extract IMBE voice frames from LDU"""
        # IMBE codec: 88 bits per 20ms frame, 9 frames per LDU
        # This is simplified - real decoder needs to extract and de-interleave

        if len(dibits) < 900:
            return None

        # Convert dibits to bytes (simplified)
        imbe_data = bytearray()
        for i in range(min(396, len(dibits))):  # 9 frames * 88 bits / 2
            imbe_data.append(dibits[i])

        return bytes(imbe_data)

    def _handle_tsbk(self, frame: P25Frame) -> None:
        """Handle trunking signaling (TSBK) messages"""
        if not frame.tsbk_data:
            return

        msg_type = frame.tsbk_data.get('type')

        if msg_type == 'GROUP_VOICE_GRANT':
            tgid = frame.tsbk_data.get('tgid')
            freq_hz = frame.tsbk_data.get('frequency_mhz', 0) * 1e6
            logger.info(f"Voice grant: TGID={tgid} Freq={freq_hz/1e6:.4f} MHz")

            # If we're monitoring this talkgroup, tune to voice channel
            if tgid == self.current_tgid:
                self.voice_channel_freq = freq_hz
                logger.info(f"Following TGID {tgid} to {freq_hz/1e6:.4f} MHz")
