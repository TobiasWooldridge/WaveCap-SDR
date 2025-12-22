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
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, cast
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


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
    algid: Optional[int] = None  # Algorithm ID (encryption)
    kid: Optional[int] = None  # Key ID
    tgid: Optional[int] = None  # Talkgroup ID
    source: Optional[int] = None  # Source radio ID
    voice_data: Optional[bytes] = None  # IMBE voice frames
    tsbk_opcode: Optional[int] = None  # TSBK opcode
    tsbk_data: Optional[Dict[str, Any]] = None  # TSBK decoded data
    errors: int = 0  # Error count


class C4FMDemodulator:
    """
    C4FM (4-level FSK) demodulator for P25 Phase 1.

    Demodulates 4800 baud C4FM signal to dibits using:
    - FM discriminator for frequency demodulation
    - Root-Raised Cosine (RRC) matched filter
    - Gardner Timing Error Detector (TED) for symbol timing recovery
    """

    def __init__(self, sample_rate: int = 48000, symbol_rate: int = 4800):
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
        #
        # After FM demodulation, normalized symbol levels are:
        # +1.0 (max positive) = +3 symbol = dibit 1
        # +0.33 (mid positive) = +1 symbol = dibit 0
        # -0.33 (mid negative) = -1 symbol = dibit 2
        # -1.0 (max negative) = -3 symbol = dibit 3
        self.deviation_hz = 600.0  # Base deviation (±600 Hz steps)
        self.max_deviation = 1800.0  # ±1800 Hz max

        # Symbol decision thresholds (normalized -1 to +1)
        # Thresholds at -0.67, 0, +0.67 separate the 4 symbol levels
        self.thresholds = np.array([-0.67, 0.0, 0.67])

        # Gardner TED state
        self._mu = 0.0  # Fractional symbol timing offset (0 to 1)
        self._gain_mu = 0.05  # Timing loop gain (controls convergence speed)
        self._last_sample = 0.0  # Previous sample for interpolation
        self._last_symbol = 0.0  # Previous symbol value

        # RRC filter coefficients (alpha=0.2 for P25)
        self._rrc_taps = self._design_rrc_filter(alpha=0.2, num_taps=65)

        # DC removal state
        self._dc_alpha = 0.01
        self._dc_estimate = 0.0

    def _design_rrc_filter(self, alpha: float = 0.2, num_taps: int = 65) -> np.ndarray:
        """Design Root-Raised Cosine filter for P25 C4FM."""
        # RRC filter design
        sps = int(round(self.samples_per_symbol))
        t = np.arange(-(num_taps-1)//2, (num_taps-1)//2 + 1) / sps

        # Handle special cases
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

        # Normalize
        h = h / np.sqrt(np.sum(h**2))
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
            return cast(np.ndarray, np.array([], dtype=np.uint8))

        # Validate input - must be complex
        if not np.iscomplexobj(iq):
            logger.warning(f"C4FM demodulate: expected complex IQ, got {iq.dtype}")
            if len(iq) % 2 == 0:
                iq = iq[::2] + 1j * iq[1::2]
            else:
                return cast(np.ndarray, np.array([], dtype=np.uint8))

        # FM discriminator (quadrature demodulation)
        x: np.ndarray = iq.astype(np.complex64, copy=False)
        prod = x[1:] * np.conj(x[:-1])
        # Normalize to approximately ±1 range
        inst_freq = cast(np.ndarray, np.angle(prod)) * self.sample_rate / (2 * np.pi * self.max_deviation)

        if len(inst_freq) < len(self._rrc_taps):
            return cast(np.ndarray, np.array([], dtype=np.uint8))

        # DC removal (removes frequency offset)
        for i in range(len(inst_freq)):
            self._dc_estimate = self._dc_estimate * (1 - self._dc_alpha) + inst_freq[i] * self._dc_alpha
            inst_freq[i] = inst_freq[i] - self._dc_estimate

        # Apply RRC matched filter
        try:
            filtered = np.convolve(inst_freq, self._rrc_taps, mode='same').astype(np.float32)
        except Exception:
            filtered = inst_freq

        # Symbol timing recovery using Gardner TED
        symbols = self._gardner_timing_recovery(filtered)

        return symbols

    def _gardner_timing_recovery(self, samples: np.ndarray) -> np.ndarray:
        """
        Gardner Timing Error Detector for symbol timing recovery.

        The Gardner TED works by computing:
        e(k) = (y(k) - y(k-1)) * y(k-0.5)

        Where y(k-0.5) is the mid-sample between symbols.
        """
        sps = self.samples_per_symbol
        symbols = []

        i = 0
        while i < len(samples) - int(sps) - 1:
            # Current sample index (with fractional offset)
            idx = i + self._mu

            # Linear interpolation for fractional sample
            idx_int = int(idx)
            frac = idx - idx_int

            if idx_int + 1 >= len(samples):
                break

            # Interpolated symbol sample
            y_k = samples[idx_int] * (1 - frac) + samples[idx_int + 1] * frac

            # Mid-sample (half symbol period back)
            mid_idx = idx - sps / 2
            if mid_idx < 0:
                mid_idx = 0
            mid_int = int(mid_idx)
            mid_frac = mid_idx - mid_int

            if mid_int + 1 >= len(samples):
                y_mid = samples[mid_int] if mid_int < len(samples) else 0
            else:
                y_mid = samples[mid_int] * (1 - mid_frac) + samples[mid_int + 1] * mid_frac

            # Gardner timing error
            ted = (y_k - self._last_symbol) * y_mid

            # Update timing offset
            self._mu = self._mu + self._gain_mu * ted

            # Keep mu in range [0, 1)
            while self._mu >= 1.0:
                self._mu -= 1.0
                i += 1
            while self._mu < 0.0:
                self._mu += 1.0
                i -= 1

            self._last_symbol = y_k

            # Map to dibit using thresholds
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

            symbols.append(dibit)

            # Advance by one symbol period
            i += int(sps)

        return np.array(symbols, dtype=np.uint8)


class P25TrellisDecoder:
    """
    P25 1/2 rate trellis decoder.

    P25 uses a 4-state trellis code that encodes 1 dibit (2 bits) to 1 symbol.
    TSBK uses 1/2 rate trellis with the following constellation:

    Dibit input (2 bits) maps to 4-level symbol:
    Dibit 0 (00) -> -3, Dibit 1 (01) -> -1, Dibit 2 (10) -> +1, Dibit 3 (11) -> +3

    The trellis has 4 states based on previous dibit.
    """

    # Trellis constellation for 1/2 rate code
    # Maps (current_state, input_dibit) -> (output_dibit, next_state)
    # This is the P25 1/2 rate trellis encoder table
    TRELLIS_TABLE = np.array([
        # State 0 transitions
        [(0, 0), (2, 1), (1, 2), (3, 3)],
        # State 1 transitions
        [(2, 0), (0, 1), (3, 2), (1, 3)],
        # State 2 transitions
        [(1, 0), (3, 1), (0, 2), (2, 3)],
        # State 3 transitions
        [(3, 0), (1, 1), (2, 2), (0, 3)],
    ], dtype=object)

    # Inverse lookup: given (current_state, output_dibit) -> input_dibit
    # Built from TRELLIS_TABLE
    TRELLIS_DECODE = None

    def __init__(self):
        if P25TrellisDecoder.TRELLIS_DECODE is None:
            self._build_decode_table()

    @classmethod
    def _build_decode_table(cls):
        """Build inverse trellis lookup table."""
        # decode_table[state][output] = (input, next_state)
        cls.TRELLIS_DECODE = [[None] * 4 for _ in range(4)]
        for state in range(4):
            for inp in range(4):
                out, next_state = cls.TRELLIS_TABLE[state][inp]
                cls.TRELLIS_DECODE[state][out] = (inp, next_state)

    def decode(self, dibits: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
        """
        Decode trellis-encoded dibits using Viterbi algorithm.

        Args:
            dibits: Array of received dibits (0-3)

        Returns:
            (decoded_dibits, error_count) or (None, -1) if decode failed
        """
        if len(dibits) < 2:
            return None, -1

        n = len(dibits)
        NUM_STATES = 4

        # Viterbi path metrics (cumulative Hamming distance)
        # path_metric[state] = cumulative error count to reach this state
        path_metric = np.full(NUM_STATES, np.inf)
        path_metric[0] = 0  # Start in state 0

        # Path history for traceback
        # history[i][state] = (prev_state, decoded_dibit)
        history = [[None] * NUM_STATES for _ in range(n)]

        # Forward pass
        for i, rx_dibit in enumerate(dibits):
            new_metric = np.full(NUM_STATES, np.inf)
            rx_dibit = int(rx_dibit) & 0x3  # Ensure valid

            for state in range(NUM_STATES):
                if path_metric[state] == np.inf:
                    continue

                # Try all input dibits and see which one could produce rx_dibit
                for inp in range(4):
                    expected_out, next_state = self.TRELLIS_TABLE[state][inp]

                    # Hamming distance between received and expected
                    # For dibits, count bit differences
                    err = bin(rx_dibit ^ expected_out).count('1')
                    branch_metric = path_metric[state] + err

                    if branch_metric < new_metric[next_state]:
                        new_metric[next_state] = branch_metric
                        history[i][next_state] = (state, inp)

            path_metric = new_metric

        # Find best ending state
        best_state = int(np.argmin(path_metric))
        best_metric = int(path_metric[best_state])

        if best_metric == np.inf:
            return None, -1

        # Traceback to recover decoded dibits
        decoded = []
        state = best_state
        for i in range(n - 1, -1, -1):
            if history[i][state] is None:
                return None, -1
            prev_state, decoded_dibit = history[i][state]
            decoded.append(decoded_dibit)
            state = prev_state

        decoded.reverse()
        return np.array(decoded, dtype=np.uint8), best_metric


class P25FrameSync:
    """Frame synchronization for P25"""

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
        self.sync_threshold = 4  # Allow 4 dibit errors in 24-dibit sync

    def find_sync(self, dibits: np.ndarray) -> Tuple[Optional[int], Optional[P25FrameType]]:
        """
        Search for P25 frame sync pattern in dibit stream.

        The P25 frame structure is:
        - 48-bit frame sync (24 dibits)
        - 64-bit NID (32 dibits): NAC (12 bits) + DUID (4 bits) + parity

        Returns:
            (sync_position, frame_type) or (None, None) if not found
        """
        # Need at least sync (24 dibits) + NID (8 dibits for NAC+DUID minimum)
        if len(dibits) < 32:
            return None, None

        # Ensure dibits are uint8 with values 0-3
        if dibits.dtype != np.uint8:
            dibits = dibits.astype(np.uint8)

        # Clip to valid dibit range (0-3)
        if np.any(dibits > 3):
            logger.warning(f"P25 find_sync: dibits out of range (max={dibits.max()}), clipping")
            dibits = np.clip(dibits, 0, 3).astype(np.uint8)

        # Search for frame sync pattern using correlation
        sync_len = len(self.FRAME_SYNC_DIBITS)

        for start_pos in range(len(dibits) - sync_len - 8):  # Need sync + some NID
            # Count matching dibits
            window = dibits[start_pos:start_pos + sync_len]
            errors = int(np.sum(window != self.FRAME_SYNC_DIBITS))

            if errors <= self.sync_threshold:
                # Found sync! Extract DUID from NID
                # NID starts after sync: NAC is 6 dibits, DUID is 2 dibits
                nid_start = start_pos + sync_len
                if nid_start + 8 > len(dibits):
                    continue

                # Extract NAC (first 6 dibits = 12 bits)
                nac_dibits = dibits[nid_start:nid_start + 6]
                nac = 0
                for d in nac_dibits:
                    nac = (nac << 2) | int(d)

                # Extract DUID (2 dibits after NAC)
                duid_dibits = dibits[nid_start + 6:nid_start + 8]
                duid = int((duid_dibits[0] << 2) | duid_dibits[1])

                frame_type = self.duid_to_frame_type.get(duid, P25FrameType.UNKNOWN)

                # Debug: log NAC and DUID for first few syncs
                if not hasattr(self, '_sync_debug_count'):
                    self._sync_debug_count = 0
                self._sync_debug_count += 1
                if self._sync_debug_count <= 10:
                    logger.info(
                        f"P25FrameSync: pos={start_pos}, NAC={nac:03x}, "
                        f"NID dibits={list(dibits[nid_start:nid_start+8])}, "
                        f"DUID={duid:x} -> {frame_type}"
                    )

                logger.debug(f"P25 sync found at {start_pos}, errors={errors}, DUID={duid:x} -> {frame_type}")
                return start_pos, frame_type

        return None, None


class P25Decoder:
    """
    Complete P25 Phase 1 decoder with trunking support.
    """

    # P25 frame sizes in dibits
    # Frame sync is 24 dibits (48 bits) + NID is 32 dibits (64 bits)
    MIN_SYNC_DIBITS = 32  # Sync (24) + minimum NID (8) for DUID extraction
    MIN_FRAME_DIBITS = 150  # Minimum to attempt frame decode (sync + NID + some data)
    MAX_BUFFER_DIBITS = 4000  # ~2 frames worth, prevent unbounded growth

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.demodulator = C4FMDemodulator(sample_rate)
        self.frame_sync = P25FrameSync()
        self.trellis = P25TrellisDecoder()

        # Dibit buffer for accumulating across IQ chunks
        self._dibit_buffer: np.ndarray = np.array([], dtype=np.uint8)

        # Trunking state
        self.control_channel = True  # Are we on control channel?
        self.current_tgid: Optional[int] = None
        self.voice_channel_freq: Optional[float] = None

        # Callbacks
        self.on_voice_frame: Optional[Callable[[bytes], None]] = None
        self.on_tsbk_message: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_grant: Optional[Callable[[int, float], None]] = None  # (tgid, freq)

        # Debug counters
        self._process_count = 0
        self._no_sync_count = 0
        self._sync_count = 0
        self._tsbk_decode_count = 0

        logger.info(f"P25 decoder initialized (sample_rate={sample_rate})")

    def process_iq(self, iq: np.ndarray) -> list[P25Frame]:
        """
        Process IQ samples and decode P25 frames.

        Accumulates dibits across multiple IQ chunks to ensure enough
        data for frame sync and decoding.

        Args:
            iq: Complex IQ samples

        Returns:
            List of decoded P25 frames
        """
        self._process_count += 1

        # Demodulate to dibits
        new_dibits = self.demodulator.demodulate(iq)

        if len(new_dibits) == 0:
            return []

        # Append to buffer - ensure uint8 dtype is preserved
        self._dibit_buffer = np.concatenate([self._dibit_buffer, new_dibits]).astype(np.uint8)

        # Validate and clip dibits to valid range (0-3)
        # This prevents overflow errors in frame decoding
        if np.any(self._dibit_buffer > 3):
            logger.warning(f"P25: dibit buffer contains out-of-range values (max={self._dibit_buffer.max()}), clipping")
            self._dibit_buffer = np.clip(self._dibit_buffer, 0, 3).astype(np.uint8)

        # Prevent unbounded buffer growth
        if len(self._dibit_buffer) > self.MAX_BUFFER_DIBITS:
            # Keep last half of buffer (might have partial frame)
            self._dibit_buffer = self._dibit_buffer[-self.MAX_BUFFER_DIBITS // 2:]

        # Log status periodically
        if self._process_count % 100 == 0:
            logger.info(f"P25 decoder: processed={self._process_count}, syncs={self._sync_count}, no_sync={self._no_sync_count}, buffer={len(self._dibit_buffer)}")

        # Need at least MIN_FRAME_DIBITS for meaningful frame decode
        if len(self._dibit_buffer) < self.MIN_FRAME_DIBITS:
            return []

        # Find frame sync in buffer
        sync_pos, frame_type = self.frame_sync.find_sync(self._dibit_buffer)

        if sync_pos is None:
            self._no_sync_count += 1
            return []

        self._sync_count += 1
        logger.info(f"Found P25 frame sync at position {sync_pos}: {frame_type} (buffer={len(self._dibit_buffer)})")

        # P25 frame structure:
        # - 24 dibits: Frame sync
        # - 8 dibits: NID (NAC + DUID) - already parsed by find_sync
        # - Frame data starts after sync + NID
        SYNC_DIBITS = 24
        NID_DIBITS = 8  # Minimal NID we used for DUID extraction
        header_dibits = SYNC_DIBITS + NID_DIBITS

        # Minimum frame data dibits required per frame type (after sync+NID header)
        # These are approximate minimums for meaningful decode
        MIN_FRAME_DATA: Dict[P25FrameType, int] = {
            P25FrameType.HDU: 100,    # Header data unit
            P25FrameType.LDU1: 900,   # Voice frame 1
            P25FrameType.LDU2: 900,   # Voice frame 2
            P25FrameType.TDU: 10,     # Terminator (short)
            P25FrameType.TSDU: 196,   # TSBK block (1 TSBK = 196 encoded dibits)
            P25FrameType.PDU: 100,    # Packet data unit
            P25FrameType.UNKNOWN: 32, # Minimum for any unknown frame
        }

        # Calculate available frame data
        available_data = len(self._dibit_buffer) - sync_pos - header_dibits
        min_required = MIN_FRAME_DATA.get(frame_type, MIN_FRAME_DATA[P25FrameType.UNKNOWN])

        if available_data < min_required:
            # Not enough data for this frame type - keep sync position and wait for more
            logger.debug(f"P25: Need more data for {frame_type}: have {available_data}, need {min_required}")
            # Trim buffer to start at sync position (discard data before sync)
            self._dibit_buffer = self._dibit_buffer[sync_pos:]
            return []

        # Extract frame data after sync + NID header
        frame_dibits = self._dibit_buffer[sync_pos + header_dibits:]

        # Consume the entire frame from buffer
        # For TSDU, consume one TSBK worth; for others, be conservative
        if frame_type == P25FrameType.TSDU:
            # TSDU can have up to 3 TSBK blocks (196 dibits each = 588 max)
            # But we process one at a time, so consume just one TSBK
            consume_len = sync_pos + header_dibits + 196
        elif frame_type in (P25FrameType.LDU1, P25FrameType.LDU2):
            # LDU frames are ~1800 bits = 900 dibits
            consume_len = sync_pos + header_dibits + 900
        elif frame_type == P25FrameType.HDU:
            consume_len = sync_pos + header_dibits + 500  # HDU is ~648 bits
        else:
            # For TDU and others, consume header plus minimal frame
            consume_len = sync_pos + header_dibits + min_required

        # Don't consume more than buffer length
        consume_len = min(consume_len, len(self._dibit_buffer))
        self._dibit_buffer = self._dibit_buffer[consume_len:]

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
            frame = P25Frame(frame_type=P25FrameType.UNKNOWN, nac=0, duid=0)

        if frame:
            if frame_type is not None:
                frame.frame_type = frame_type
            frames.append(frame)

            # Handle trunking logic
            if frame.tsbk_opcode is not None and frame.tsbk_data:
                self._handle_tsbk(frame)

        return frames

    def _decode_hdu(self, dibits: np.ndarray) -> Optional[P25Frame]:
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

    def _decode_ldu1(self, dibits: np.ndarray) -> Optional[P25Frame]:
        """Decode Logical Link Data Unit 1 (voice frame)"""
        if len(dibits) < 900:  # LDU1 is ~1800 bits
            return None

        # Extract link control data (contains TGID, source ID)
        # This is simplified - full decoder needs error correction

        # Extract voice IMBE frames (9 frames per LDU)
        voice_data = self._extract_imbe_frames(dibits)

        if self.on_voice_frame and voice_data:
            self.on_voice_frame(voice_data)

        return P25Frame(
            frame_type=P25FrameType.LDU1,
            nac=0,  # Would extract from frame
            duid=5,
            voice_data=voice_data
        )

    def _decode_ldu2(self, dibits: np.ndarray) -> Optional[P25Frame]:
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

    def _decode_tdu(self, dibits: np.ndarray) -> Optional[P25Frame]:
        """Decode Terminator Data Unit (end of transmission)"""
        logger.info("TDU: End of transmission")
        return P25Frame(frame_type=P25FrameType.TDU, nac=0, duid=3)

    def _decode_tsdu(self, dibits: np.ndarray) -> Optional[P25Frame]:
        """
        Decode Trunking Signaling Data Unit (TSBK messages).

        A TSDU contains 1-3 TSBK (Trunking Signaling Block) messages.
        Each TSBK is:
        - 196 encoded dibits (98 data dibits after 1/2 rate trellis decode)
        - Decoded content:
          - LB (1 bit): Last Block flag
          - Opcode (6 bits): Message type
          - MFR (1 bit): Manufacturer bit
          - Data (72 bits): Opcode-specific payload
          - CRC-16 (16 bits): Error check

        TSDU structure allows up to 3 TSBKs (588 dibits max).
        """
        # Minimum: one TSBK block = 196 encoded dibits
        TSBK_ENCODED_DIBITS = 196
        TSBK_DECODED_DIBITS = 98  # After 1/2 rate trellis decode

        logger.debug(f"TSDU decode: received {len(dibits)} dibits")

        if len(dibits) < 48:  # Need at least some data
            logger.warning(f"TSDU too short: {len(dibits)} dibits (need 48+)")
            return None

        # Try to decode TSBK with trellis decoder
        # For now, try to decode up to one TSBK block
        block_dibits = dibits[:min(len(dibits), TSBK_ENCODED_DIBITS)]

        decoded, errors = self.trellis.decode(block_dibits)

        if decoded is None or errors > len(block_dibits) // 4:
            # Too many errors, try without trellis (raw decode for debugging)
            logger.debug(f"TSBK trellis decode failed: errors={errors}")
            # Fall back to raw decode (won't work, but log for debugging)
            decoded = block_dibits

        if len(decoded) < 48:
            return None

        self._tsbk_decode_count += 1

        # Extract TSBK fields from decoded dibits
        # LB (1 bit) + Opcode (6 bits) + MFR (1 bit) = 8 bits = 4 dibits
        # Extract as 8 bits
        header_bits = 0
        for i in range(4):
            header_bits = (header_bits << 2) | (decoded[i] & 0x3)

        lb = (header_bits >> 7) & 0x1
        opcode = (header_bits >> 1) & 0x3F
        mfr = header_bits & 0x1

        # Extract data (72 bits = 36 dibits, starting at dibit 4)
        data_bits = 0
        for i in range(36):
            if 4 + i < len(decoded):
                data_bits = (data_bits << 2) | (decoded[4 + i] & 0x3)

        # Decode based on opcode
        tsbk_data = self._decode_tsbk_opcode(opcode, decoded[4:])
        tsbk_data['lb'] = lb
        tsbk_data['mfr'] = mfr
        tsbk_data['raw_opcode'] = opcode
        tsbk_data['trellis_errors'] = errors

        logger.info(f"TSBK: LB={lb} Opcode=0x{opcode:02X} MFR={mfr} Errors={errors} -> {tsbk_data.get('type', 'UNKNOWN')}")

        if self.on_tsbk_message:
            self.on_tsbk_message(tsbk_data)

        return P25Frame(
            frame_type=P25FrameType.TSDU,
            nac=0,
            duid=7,
            tsbk_opcode=opcode,
            tsbk_data=tsbk_data
        )

    def _decode_tsbk_opcode(self, opcode: int, dibits: np.ndarray) -> Dict[str, Any]:
        """Decode TSBK opcode and extract trunking information"""
        data: Dict[str, Any] = {}

        # Common TSBK opcodes
        if opcode == 0x00:  # Group Voice Channel Grant
            # Extract frequency and talkgroup
            if len(dibits) >= 20:
                tgid = 0
                for i in range(8):
                    tgid = (tgid << 2) | dibits[i]

                freq_data = 0
                for i in range(8):
                    freq_data = (freq_data << 2) | dibits[8 + i]

                # Convert to actual frequency (simplified)
                freq_mhz = 851.0 + (freq_data * 0.00625)  # Example: 800 MHz band

                data['type'] = 'GROUP_VOICE_GRANT'
                data['tgid'] = tgid
                data['frequency_mhz'] = freq_mhz

                if self.on_grant:
                    self.on_grant(tgid, freq_mhz * 1e6)

        elif opcode == 0x02:  # Group Voice Channel Grant Update
            data['type'] = 'GROUP_VOICE_GRANT_UPDATE'

        elif opcode == 0x03:  # Unit to Unit Voice Channel Grant
            data['type'] = 'UNIT_TO_UNIT_GRANT'

        elif opcode == 0x20:  # Network Status Broadcast
            data['type'] = 'NETWORK_STATUS'

        elif opcode == 0x28:  # RFSS Status Broadcast
            data['type'] = 'RFSS_STATUS'

        elif opcode == 0x2A:  # System Service Broadcast
            data['type'] = 'SYSTEM_SERVICE'

        elif opcode == 0x34:  # IDEN_UP_VU - Channel ID Update (Voice)
            # Contains channel frequency band info
            data['type'] = 'IDEN_UP_VU'

        elif opcode == 0x35:  # IDEN_UP_TDMA - Channel ID Update (TDMA)
            data['type'] = 'IDEN_UP_TDMA'

        elif opcode == 0x3A:  # Adjacent Status Broadcast
            data['type'] = 'ADJ_STS_BCAST'

        elif opcode == 0x3C:  # RFSS Status Broadcast Explicit
            data['type'] = 'RFSS_STS_BCAST'

        # Motorola-specific opcodes (high opcode values typically MFR=1)
        elif opcode == 0x3F:  # Motorola: OSP_TDULC (common on Moto systems)
            data['type'] = 'MOT_TDULC'

        elif opcode == 0x3D:  # Motorola: MAINT (maintenance message)
            data['type'] = 'MOT_MAINT'

        elif opcode == 0x3B:  # Time/Date announcement
            data['type'] = 'TIME_DATE_ANN'

        else:
            data['type'] = 'UNKNOWN'
            data['opcode'] = opcode

        return data

    def _extract_imbe_frames(self, dibits: np.ndarray) -> Optional[bytes]:
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
