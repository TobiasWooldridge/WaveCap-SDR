"""RDS (Radio Data System) decoder for FM broadcast.

RDS is transmitted as a BPSK-modulated subcarrier at 57 kHz (3x the 19 kHz pilot).
Data rate: 1187.5 bps, grouped into 104-bit blocks.

Key data types decoded:
- PI (Program Identification): Station identifier
- PS (Program Service): 8-character station name
- RT (Radio Text): Up to 64-character scrolling text
- PTY (Program Type): Genre code (news, music, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
from wavecapsdr.typing import NDArrayAny

# RDS Constants
RDS_SUBCARRIER_HZ = 57000  # 57 kHz subcarrier (3x 19 kHz pilot)
RDS_BAUD_RATE = 1187.5  # bits per second
RDS_SAMPLES_PER_SYMBOL = 48  # For internal processing at ~57 kHz sample rate

# Syndrome words for block identification
# Block A, B, C, C', D with their offset words
SYNDROMES = {
    0x3D8: "A",
    0x3D4: "B",
    0x25C: "C",
    0x3CC: "Cp",  # C' for type B groups
    0x258: "D",
}

# Generator polynomial for RDS CRC: x^10 + x^8 + x^7 + x^5 + x^4 + x^3 + 1
RDS_POLY = 0x5B9

# PTY (Program Type) codes - US RBDS
PTY_CODES = {
    0: "None",
    1: "News",
    2: "Information",
    3: "Sports",
    4: "Talk",
    5: "Rock",
    6: "Classic Rock",
    7: "Adult Hits",
    8: "Soft Rock",
    9: "Top 40",
    10: "Country",
    11: "Oldies",
    12: "Soft",
    13: "Nostalgia",
    14: "Jazz",
    15: "Classical",
    16: "R&B",
    17: "Soft R&B",
    18: "Language",
    19: "Religious Music",
    20: "Religious Talk",
    21: "Personality",
    22: "Public",
    23: "College",
    24: "Spanish Talk",
    25: "Spanish Music",
    26: "Hip Hop",
    27: "Unassigned",
    28: "Unassigned",
    29: "Weather",
    30: "Emergency Test",
    31: "Emergency",
}


@dataclass
class RDSData:
    """Decoded RDS data for a station."""

    pi_code: str | None = None  # Program Identification (hex string like "A1B2")
    ps_name: str = "        "  # Program Service name (8 chars)
    radio_text: str = ""  # Radio Text (up to 64 chars)
    pty: int = 0  # Program Type code
    pty_name: str = "None"  # Program Type name
    ta: bool = False  # Traffic Announcement flag
    tp: bool = False  # Traffic Program flag
    ms: bool = True  # Music/Speech switch (True = Music)

    # Internal state for building strings
    _ps_segments: dict[int, str] = field(
        default_factory=lambda: {0: "  ", 1: "  ", 2: "  ", 3: "  "}
    )
    _rt_segments: dict[int, str] = field(default_factory=dict)
    _rt_ab_flag: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "piCode": self.pi_code,
            "psName": self.ps_name.strip() or None,
            "radioText": self.radio_text.strip() or None,
            "pty": self.pty,
            "ptyName": self.pty_name,
            "ta": self.ta,
            "tp": self.tp,
            "ms": self.ms,
        }


@lru_cache(maxsize=8)
def _get_bpf_coeffs(
    sample_rate: int, center_freq: int, bandwidth: int
) -> tuple[NDArrayAny, NDArrayAny] | None:
    """Get bandpass filter coefficients for RDS subcarrier extraction."""
    from scipy import signal

    nyquist = sample_rate / 2
    low = (center_freq - bandwidth / 2) / nyquist
    high = (center_freq + bandwidth / 2) / nyquist

    # Clamp to valid range
    low = max(0.001, min(0.999, low))
    high = max(0.001, min(0.999, high))

    if low >= high:
        return None

    b, a = signal.butter(4, [low, high], btype="band")
    return b, a


def _crc_check(block: int) -> tuple[bool, str]:
    """Check CRC and identify block type.

    Args:
        block: 26-bit block (16 data + 10 checkword)

    Returns:
        Tuple of (valid, block_type) where block_type is 'A', 'B', 'C', 'Cp', or 'D'
    """
    # Calculate syndrome
    reg = 0
    for i in range(26):
        bit = (block >> (25 - i)) & 1
        msb = (reg >> 9) & 1
        reg = ((reg << 1) | bit) & 0x3FF
        if msb:
            reg ^= RDS_POLY

    # Check against known syndromes
    if reg in SYNDROMES:
        return True, SYNDROMES[reg]

    return False, ""


class RDSDecoder:
    """Real-time RDS decoder.

    Takes demodulated FM audio and extracts RDS data.
    The FM audio must be sampled at a rate high enough to contain the 57 kHz subcarrier.
    Typically this means processing the FM signal BEFORE the MPX lowpass filter.
    """

    def __init__(self, sample_rate: int = 250000):
        """Initialize RDS decoder.

        Args:
            sample_rate: Sample rate of input FM audio in Hz.
                        Must be > 114 kHz to capture 57 kHz subcarrier.
        """
        self.sample_rate = sample_rate
        self.data = RDSData()

        # Bit recovery state
        self._bit_buffer: list[int] = []
        self._symbol_phase = 0.0
        self._last_sample = 0.0
        self._pll_phase = 0.0
        self._pll_freq = RDS_SUBCARRIER_HZ

        # Block sync state
        self._block_buffer: list[int] = []
        self._synced = False
        self._block_count = 0
        self._group_blocks: dict[str, int] = {}

        # Downsampled buffer for processing
        self._resample_buffer = np.array([], dtype=np.float32)

    def process(self, fm_baseband: NDArrayAny) -> RDSData | None:
        """Process FM baseband and extract RDS data.

        Args:
            fm_baseband: Demodulated FM audio at original sample rate.
                        This should be the output of quadrature_demod BEFORE
                        any lowpass filtering (MPX filter).

        Returns:
            Updated RDSData object if any data was decoded, None otherwise.
        """
        if fm_baseband.size == 0:
            return None

        if self.sample_rate < 114000:
            # Sample rate too low for 57 kHz subcarrier
            return None

        try:
            from scipy import signal

            # Extract 57 kHz RDS subcarrier using bandpass filter
            coeffs = _get_bpf_coeffs(self.sample_rate, RDS_SUBCARRIER_HZ, 4000)
            if coeffs is None:
                return None
            b, a = coeffs

            rds_signal = signal.lfilter(b, a, fm_baseband).astype(np.float32)

            # Mix down to baseband using 57 kHz carrier
            t = np.arange(len(rds_signal)) / self.sample_rate
            carrier = np.cos(2 * np.pi * RDS_SUBCARRIER_HZ * t + self._pll_phase).astype(np.float32)
            baseband = rds_signal * carrier * 2  # *2 to compensate for mixing loss

            # Update PLL phase for next block
            self._pll_phase = (
                self._pll_phase + 2 * np.pi * RDS_SUBCARRIER_HZ * len(rds_signal) / self.sample_rate
            ) % (2 * np.pi)

            # Lowpass filter to get BPSK baseband (cutoff ~2.4 kHz for 1187.5 baud)
            nyquist = self.sample_rate / 2
            cutoff = 2400 / nyquist
            if cutoff < 1.0:
                b_lpf, a_lpf = signal.butter(4, cutoff, btype="low")
                baseband = signal.lfilter(b_lpf, a_lpf, baseband).astype(np.float32)

            # Decimate to ~10x symbol rate for bit recovery
            target_rate = int(RDS_BAUD_RATE * 10)  # ~11875 Hz
            decimate_factor = max(1, self.sample_rate // target_rate)
            baseband = baseband[::decimate_factor]
            effective_rate = self.sample_rate / decimate_factor

            # Bit recovery using zero-crossing detection
            decoded_any = False
            samples_per_bit = effective_rate / RDS_BAUD_RATE

            for sample in baseband:
                # Zero-crossing detector for bit timing
                if self._last_sample < 0 and sample >= 0:
                    # Rising edge - reset phase
                    self._symbol_phase = samples_per_bit / 2
                elif self._last_sample >= 0 and sample < 0:
                    # Falling edge - reset phase
                    self._symbol_phase = samples_per_bit / 2

                self._last_sample = sample
                self._symbol_phase += 1

                # Sample at middle of bit period
                if self._symbol_phase >= samples_per_bit:
                    self._symbol_phase -= samples_per_bit

                    # Differential decode: bit = 1 if sign change
                    bit = 1 if sample < 0 else 0
                    self._bit_buffer.append(bit)

                    # Try to decode when we have enough bits
                    if len(self._bit_buffer) >= 26:
                        if self._try_sync_and_decode():
                            decoded_any = True

                        # Keep buffer manageable
                        if len(self._bit_buffer) > 104:
                            self._bit_buffer = self._bit_buffer[-104:]

            return self.data if decoded_any else None

        except ImportError:
            return None
        except Exception:
            return None

    def _try_sync_and_decode(self) -> bool:
        """Try to sync and decode RDS blocks from bit buffer.

        Returns:
            True if any valid data was decoded.
        """
        if len(self._bit_buffer) < 26:
            return False

        decoded_any = False

        # Build 26-bit block from buffer
        block = 0
        for i in range(26):
            block = (block << 1) | self._bit_buffer[i]

        # Check CRC
        valid, block_type = _crc_check(block)

        if valid:
            data = block >> 10  # Extract 16-bit data

            if block_type == "A":
                # Block A: PI code
                self._group_blocks = {"A": data}
                self._synced = True
                self._bit_buffer = self._bit_buffer[26:]

            elif self._synced and block_type == "B" and "A" in self._group_blocks:
                self._group_blocks["B"] = data
                self._bit_buffer = self._bit_buffer[26:]

            elif self._synced and block_type in ("C", "Cp") and "B" in self._group_blocks:
                self._group_blocks["C"] = data
                self._bit_buffer = self._bit_buffer[26:]

            elif self._synced and block_type == "D" and "C" in self._group_blocks:
                self._group_blocks["D"] = data
                self._bit_buffer = self._bit_buffer[26:]

                # Complete group - decode it
                if self._decode_group():
                    decoded_any = True

                self._group_blocks = {}
            else:
                # Out of sync, shift by 1 bit
                self._bit_buffer = self._bit_buffer[1:]
                self._synced = False
        else:
            # No valid block, shift by 1 bit
            self._bit_buffer = self._bit_buffer[1:]
            if len(self._bit_buffer) < 26:
                self._synced = False

        return decoded_any

    def _decode_group(self) -> bool:
        """Decode a complete RDS group (4 blocks).

        Returns:
            True if group was successfully decoded.
        """
        if not all(k in self._group_blocks for k in ("A", "B", "C", "D")):
            return False

        block_a = self._group_blocks["A"]
        block_b = self._group_blocks["B"]
        block_c = self._group_blocks["C"]
        block_d = self._group_blocks["D"]

        # Block A: PI code
        self.data.pi_code = f"{block_a:04X}"

        # Block B: Group type and flags
        group_type = (block_b >> 12) & 0xF  # 0-15
        group_version = (block_b >> 11) & 0x1  # 0=A, 1=B
        tp = (block_b >> 10) & 0x1
        pty = (block_b >> 5) & 0x1F

        self.data.tp = bool(tp)
        self.data.pty = pty
        self.data.pty_name = PTY_CODES.get(pty, "Unknown")

        # Decode based on group type
        if group_type == 0:
            # Group 0A/0B: Basic tuning and PS name
            ta = (block_b >> 4) & 0x1
            ms = (block_b >> 3) & 0x1
            segment = block_b & 0x3  # 0-3, each segment is 2 chars

            self.data.ta = bool(ta)
            self.data.ms = bool(ms)

            # PS name in block D (2 characters)
            char1 = (block_d >> 8) & 0xFF
            char2 = block_d & 0xFF

            # Filter to printable ASCII
            if 32 <= char1 <= 126 and 32 <= char2 <= 126:
                self.data._ps_segments[segment] = chr(char1) + chr(char2)

                # Rebuild PS name
                self.data.ps_name = "".join(self.data._ps_segments.get(i, "  ") for i in range(4))

            return True

        elif group_type == 2:
            # Group 2A/2B: Radio Text
            ab_flag = (block_b >> 4) & 0x1
            segment = block_b & 0xF  # 0-15 for 2A, 0-15 for 2B

            # Clear RT if A/B flag changed (new message)
            if ab_flag != self.data._rt_ab_flag:
                self.data._rt_segments = {}
                self.data._rt_ab_flag = bool(ab_flag)

            if group_version == 0:
                # 2A: 4 characters per segment (blocks C and D)
                chars = []
                for val in (block_c >> 8, block_c & 0xFF, block_d >> 8, block_d & 0xFF):
                    if 32 <= val <= 126:
                        chars.append(chr(val))
                    elif val == 0x0D:  # Carriage return = end of message
                        break
                    else:
                        chars.append(" ")

                self.data._rt_segments[segment] = "".join(chars)
            else:
                # 2B: 2 characters per segment (block D only)
                char1 = (block_d >> 8) & 0xFF
                char2 = block_d & 0xFF
                chars = []
                for val in (char1, char2):
                    if 32 <= val <= 126:
                        chars.append(chr(val))
                    elif val == 0x0D:
                        break
                    else:
                        chars.append(" ")

                self.data._rt_segments[segment] = "".join(chars)

            # Rebuild radio text
            max_segment = max(self.data._rt_segments.keys()) if self.data._rt_segments else 0
            self.data.radio_text = "".join(
                self.data._rt_segments.get(i, "    " if group_version == 0 else "  ")
                for i in range(max_segment + 1)
            )

            return True

        return False

    def reset(self) -> None:
        """Reset decoder state."""
        self.data = RDSData()
        self._bit_buffer = []
        self._symbol_phase = 0.0
        self._last_sample = 0.0
        self._pll_phase = 0.0
        self._synced = False
        self._group_blocks = {}
