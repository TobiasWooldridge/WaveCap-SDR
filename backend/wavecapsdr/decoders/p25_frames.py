"""P25 Frame Structure Definitions and Parsers.

This module defines the P25 frame structures and provides parsers for each frame type:
- NID (Network ID) - Common header with NAC and DUID
- HDU (Header Data Unit) - Start of voice transmission
- LDU1/LDU2 (Logical Data Units) - Voice frames with IMBE data
- TDU (Terminator Data Unit) - End of transmission
- TSDU (Trunking Signaling Data Unit) - Control channel messages

Frame structure follows TIA-102.BAAA-A specification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from wavecapsdr.decoders.nac_tracker import NACTracker
from wavecapsdr.dsp.fec.bch import bch_decode
from wavecapsdr.dsp.fec.golay import golay_decode
from wavecapsdr.dsp.fec.trellis import trellis_decode

logger = logging.getLogger(__name__)


# P25 Phase 1 data deinterleave pattern (196 bits)
# Adapted from OP25 gr-op25_repeater/lib/p25p1_fdma.cc and Wireshark p25cai
# For position i in deinterleaved output, read from DATA_DEINTERLEAVE[i] in input
DATA_DEINTERLEAVE = [
    0, 1, 2, 3, 52, 53, 54, 55, 100, 101, 102, 103, 148, 149, 150, 151,
    4, 5, 6, 7, 56, 57, 58, 59, 104, 105, 106, 107, 152, 153, 154, 155,
    8, 9, 10, 11, 60, 61, 62, 63, 108, 109, 110, 111, 156, 157, 158, 159,
    12, 13, 14, 15, 64, 65, 66, 67, 112, 113, 114, 115, 160, 161, 162, 163,
    16, 17, 18, 19, 68, 69, 70, 71, 116, 117, 118, 119, 164, 165, 166, 167,
    20, 21, 22, 23, 72, 73, 74, 75, 120, 121, 122, 123, 168, 169, 170, 171,
    24, 25, 26, 27, 76, 77, 78, 79, 124, 125, 126, 127, 172, 173, 174, 175,
    28, 29, 30, 31, 80, 81, 82, 83, 128, 129, 130, 131, 176, 177, 178, 179,
    32, 33, 34, 35, 84, 85, 86, 87, 132, 133, 134, 135, 180, 181, 182, 183,
    36, 37, 38, 39, 88, 89, 90, 91, 136, 137, 138, 139, 184, 185, 186, 187,
    40, 41, 42, 43, 92, 93, 94, 95, 140, 141, 142, 143, 188, 189, 190, 191,
    44, 45, 46, 47, 96, 97, 98, 99, 144, 145, 146, 147, 192, 193, 194, 195,
    48, 49, 50, 51,
]

# Dibit-level deinterleave (98 dibits) derived from the bit-level mapping.
# Each dibit consumes two consecutive bit positions in the deinterleaved stream.
DATA_DEINTERLEAVE_DIBITS = np.array(
    [DATA_DEINTERLEAVE[i * 2] // 2 for i in range(98)],
    dtype=np.int16,
)


# CRC-16 CCITT lookup table for 80-bit P25 messages (64 data + 16 CRC bits)
# Pre-computed XOR values for each bit position when that bit is set
# Polynomial: x^16 + x^12 + x^5 + 1 (0x1021)
# Generator = 0x1021, computed by shifting 0x8000 through 80 bit positions
def _generate_ccitt_checksums(num_bits: int = 80) -> list:
    """Generate CRC-16 CCITT checksums for bit positions."""
    checksums = []
    poly = 0x1021  # CRC-16 CCITT polynomial

    for bit_pos in range(num_bits):
        # Start with 1 in the current bit position
        crc = 0
        7 - (bit_pos % 8)
        bit_pos // 8

        # Process as if only this bit is set
        # Shift through 16 iterations to get CRC contribution
        0x8000 >> (bit_pos % 16) if bit_pos < 16 else 0

        # For simplicity, compute directly
        # Each bit contributes: shift the CRC and XOR if MSB was set
        crc = 0
        for i in range(16):
            msb = (crc >> 15) & 1
            crc = (crc << 1) & 0xFFFF
            if bit_pos < 16:
                crc |= 1 if (bit_pos == i) else 0
            if msb:
                crc ^= poly

        # For bits beyond first 16, continue shifting
        for i in range(max(0, bit_pos - 15)):
            msb = (crc >> 15) & 1
            crc = (crc << 1) & 0xFFFF
            if msb:
                crc ^= poly

        checksums.append(crc)

    return checksums

# Pre-generate the table at module load time
CCITT_80_CHECKSUMS = _generate_ccitt_checksums(80)


class DUID(IntEnum):
    """Data Unit ID values for P25 frames."""
    HDU = 0x0    # Header Data Unit
    TDU = 0x3    # Terminator Data Unit (no link control)
    LDU1 = 0x5   # Logical Link Data Unit 1
    TSDU = 0x7   # Trunking Signaling Data Unit
    LDU2 = 0xA   # Logical Link Data Unit 2
    PDU = 0xC    # Packet Data Unit
    TDULC = 0xF  # Terminator Data Unit with Link Control


# Frame sync pattern (48 bits = 24 dibits)
# Per TIA-102.BAAA, P25 Phase 1 uses ONE sync pattern for ALL frame types.
# The frame type is determined by the DUID field in the NID that follows sync.
#
# C4FM symbol sequence: +3 +3 +3 +3 +3 -3 +3 +3 -3 -3 +3 +3 -3 -3 -3 -3 +3 -3 +3 -3 -3 -3 -3 -3
# Dibit encoding (per constellation): +3 symbol = dibit 1, -3 symbol = dibit 3
# Dibits: 1 1 1 1 1 3 1 1 3 3 1 1 3 3 3 3 1 3 1 3 3 3 3 3
# Hex value: 0x5575F5FF77FF (verified against SDRTrunk)
FRAME_SYNC_PATTERN = 0x5575F5FF77FF
FRAME_SYNC_DIBITS = np.array([1, 1, 1, 1, 1, 3, 1, 1, 3, 3, 1, 1,
                               3, 3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3], dtype=np.uint8)

# Status symbols positions in frames (for de-interleaving)
STATUS_SYMBOL_INTERVAL = 36  # Status symbol every 36 dibits (positions 36, 72, 108, ...)
STATUS_SYMBOL_INDEX_IN_NID = 11  # 0-based index within 33-dibit NID (frame pos 35)


@dataclass
class NID:
    """Network ID - common header for all P25 frames.

    Structure (64 bits total):
    - NAC: Network Access Code (12 bits) - identifies the network
    - DUID: Data Unit ID (4 bits) - frame type
    - BCH parity (48 bits) - error correction
    """
    nac: int  # Network Access Code (12 bits)
    duid: DUID  # Data Unit ID (4 bits)
    errors: int = 0  # BCH correction count


@dataclass
class LinkControl:
    """Link Control information from LDU frames.

    Contains call metadata:
    - LCF: Link Control Format (8 bits)
    - MFID: Manufacturer ID (8 bits)
    - Talkgroup or Unit ID
    - Source ID
    - GPS data (for LCF 0x09, 0x0A, 0x0B)
    """
    lcf: int = 0  # Link Control Format
    mfid: int = 0  # Manufacturer ID (0 = standard)
    tgid: int = 0  # Talkgroup ID
    source_id: int = 0  # Source radio ID
    emergency: bool = False
    encrypted: bool = False
    # GPS data (from Extended Link Control)
    has_gps: bool = False
    gps_latitude: float = 0.0
    gps_longitude: float = 0.0
    gps_altitude_m: float | None = None
    gps_speed_kmh: float | None = None
    gps_heading_deg: float | None = None


@dataclass
class EncryptionSync:
    """Encryption Sync from LDU2 frames.

    Contains encryption parameters for encrypted calls:
    - ALGID: Algorithm ID
    - KID: Key ID
    - MI: Message Indicator (IV)
    """
    algid: int = 0  # Algorithm ID (0x80 = unencrypted)
    kid: int = 0  # Key ID
    mi: bytes = field(default_factory=lambda: bytes(9))  # Message Indicator


@dataclass
class HDUFrame:
    """Header Data Unit - marks start of voice transmission.

    Contains:
    - NID with NAC
    - Message Indicator (MI) for encryption
    - ALGID and KID
    - TGID
    """
    nid: NID
    mi: bytes  # Message Indicator (72 bits = 9 bytes)
    algid: int  # Algorithm ID
    kid: int  # Key ID
    tgid: int  # Talkgroup ID
    errors: int = 0


@dataclass
class LDUFrame:
    """Logical Data Unit - voice frame.

    Contains 9 IMBE voice frames (20ms each = 180ms total)
    plus link control (LDU1) or encryption sync (LDU2).
    """
    nid: NID
    imbe_frames: list[bytes]  # 9 IMBE frames, 88 bits each
    link_control: LinkControl | None = None  # LDU1 only
    encryption_sync: EncryptionSync | None = None  # LDU2 only
    errors: int = 0


@dataclass
class TDUFrame:
    """Terminator Data Unit - marks end of transmission."""
    nid: NID
    link_control: LinkControl | None = None  # TDULC only


@dataclass
class TSDUFrame:
    """Trunking Signaling Data Unit - control channel message.

    Contains one or more TSBK (Trunking Signaling Block) messages.
    """
    nid: NID
    tsbk_blocks: list[TSBKBlock]
    errors: int = 0


@dataclass
class TSBKBlock:
    """Single TSBK message block.

    Structure (96 bits per TIA-102.AABB-A):
    - LB: Last Block flag (1 bit)
    - Protect: Protection flag (1 bit)
    - Opcode (6 bits)
    - MFID: Manufacturer ID (8 bits)
    - Data (64 bits)
    - CRC (16 bits)
    """
    last_block: bool
    opcode: int
    mfid: int
    data: bytes  # 8 bytes of data
    crc_valid: bool = True


def dibits_to_bits(dibits: np.ndarray) -> np.ndarray:
    """Convert dibit array to bit array."""
    bits = np.zeros(len(dibits) * 2, dtype=np.uint8)
    for i, d in enumerate(dibits):
        bits[i * 2] = (d >> 1) & 1
        bits[i * 2 + 1] = d & 1
    return bits


def bits_to_int(bits: np.ndarray, start: int, length: int) -> int:
    """Extract integer from bit array."""
    value = 0
    for i in range(length):
        value = (value << 1) | int(bits[start + i])
    return value


def deinterleave_data(bits: np.ndarray) -> np.ndarray:
    """Deinterleave 196-bit block using P25 data deinterleave pattern.

    Args:
        bits: Interleaved bit array (must be 196 bits)

    Returns:
        Deinterleaved bit array (196 bits)
    """
    if len(bits) != 196:
        logger.warning(f"deinterleave_data: expected 196 bits, got {len(bits)}")
        return bits

    # Read FROM the deinterleave table positions (matches OP25/Wireshark)
    # For each output position i, read from input position DATA_DEINTERLEAVE[i]
    deinterleaved = np.array([bits[DATA_DEINTERLEAVE[i]] for i in range(196)], dtype=np.uint8)

    return deinterleaved


def crc16_ccitt_p25(bits: np.ndarray) -> tuple[bool, int]:
    """Validate CRC-16 CCITT for P25 TSBK message.

    P25 TSBK uses 96-bit messages: 80 bits data + 16 CRC bits.
    The CRC is computed over the first 80 bits.

    CRC-16 CCITT parameters:
    - Polynomial: 0x1021 (x^16 + x^12 + x^5 + 1)
    - Initial value: 0x0000
    - Final XOR: 0xFFFF

    Args:
        bits: 96-bit array (80 bits data + 16 bits CRC)

    Returns:
        Tuple of (crc_valid, corrected_bit_count)
        - crc_valid: True if CRC passed
        - corrected_bit_count: 0=passed, -1=failed (no correction attempted)
    """
    if len(bits) < 96:
        return (False, -1)

    # Calculate CRC over first 80 bits using standard CCITT algorithm
    poly = 0x1021
    crc = 0x0000  # Initial value

    for i in range(80):
        bit = int(bits[i])
        msb = (crc >> 15) & 1
        crc = ((crc << 1) | bit) & 0xFFFF
        if msb:
            crc ^= poly

    # Process 16 more zero bits to flush (standard CRC finalization)
    for _ in range(16):
        msb = (crc >> 15) & 1
        crc = (crc << 1) & 0xFFFF
        if msb:
            crc ^= poly

    # Apply final XOR (per P25 TSBK CRC parameters)
    crc ^= 0xFFFF

    # Extract received CRC (bits 80-95)
    received_crc = bits_to_int(bits, 80, 16)

    if crc == received_crc:
        return (True, 0)

    # CRC failed
    return (False, -1)


def remove_status_symbols(dibits: np.ndarray) -> np.ndarray:
    """Remove status symbols from dibit stream.

    P25 inserts status symbols at regular intervals for
    monitoring signal quality. These must be removed before
    frame decoding.
    """
    if len(dibits) == 0:
        return dibits

    result = []
    for i, d in enumerate(dibits):
        # Skip status symbol positions
        if (i + 1) % STATUS_SYMBOL_INTERVAL != 0:
            result.append(d)

    return np.array(result, dtype=np.uint8)


def decode_nid(
    dibits: np.ndarray,
    skip_status_at_10: bool = True,
    nac_tracker: NACTracker | None = None,
) -> NID | None:
    """Decode Network ID from NID dibits with BCH error correction.

    NID structure (64 bits = 32 dibits of data):
    - NAC (12 bits) + DUID (4 bits) = 8 dibits = 16 bits
    - BCH(63,16,23) parity = 24 dibits = 47 bits (63 - 16)

    IMPORTANT: P25 inserts a status symbol every 36 dibits from frame start.
    Status symbols are at 0-indexed frame positions 35, 71, 107, etc.
    (where (pos+1) % 36 == 0).

    Since sync is 24 dibits (positions 0-23), the NID starts at position 24.
    The first status symbol is at frame position 35, which is NID position 11
    (35 - 24 = 11). This status symbol must be skipped.

    BCH Error Correction:
    - First pass: decode codeword as-is
    - Second pass (if first fails): use tracked NAC to overwrite NAC field and retry

    Args:
        dibits: NID dibits (33 if status included, 32 if already stripped)
        skip_status_at_10: If True, expects 33 dibits and skips the status symbol
        nac_tracker: Optional NAC tracker for BCH second-pass correction

    Returns:
        Decoded NID or None if decoding fails
    """
    required_len = 33 if skip_status_at_10 else 32
    if len(dibits) < required_len:
        logger.debug(f"decode_nid: too short, len={len(dibits)}, required={required_len}")
        return None

    # Build clean dibit array, skipping status symbol when requested
    # Status symbol is at frame position 35 = NID position 11 (since NID starts at 24)
    clean_dibits = []
    for i in range(required_len):
        if skip_status_at_10 and i == STATUS_SYMBOL_INDEX_IN_NID:
            continue  # Skip status symbol at frame position 35
        clean_dibits.append(int(dibits[i]))  # Ensure Python int

    if len(clean_dibits) < 32:
        logger.debug(f"decode_nid: clean_dibits too short after skip: {len(clean_dibits)}")
        return None

    # Convert dibits to bits for BCH decoder (32 dibits = 64 bits)
    bits = np.zeros(64, dtype=np.uint8)
    for i, d in enumerate(clean_dibits):
        bits[i * 2] = (d >> 1) & 1
        bits[i * 2 + 1] = d & 1

    # Note: BCH(63,16,23) expects 63 bits, but NID is 64 bits (16 data + 48 parity)
    # We use the first 63 bits for BCH decoding
    bch_codeword = bits[:63]

    # Get tracked NAC if available
    tracked_nac = nac_tracker.get_tracked_nac() if nac_tracker else None

    # BCH decode with optional NAC tracking assistance
    decoded_data, errors = bch_decode(bch_codeword, tracked_nac)

    if errors < 0:
        # BCH decode failed - use simple extraction as fallback
        # This works because NAC/DUID are in the first 8 dibits before status symbol
        logger.debug("decode_nid: BCH decode failed, using simple extraction")

        # Extract NAC from first 6 dibits (12 bits)
        nac = 0
        for i in range(min(6, len(clean_dibits))):
            nac = (nac << 2) | clean_dibits[i]

        # Extract DUID from dibits 6-7 (4 bits)
        if len(clean_dibits) >= 8:
            duid_val = (clean_dibits[6] << 2) | clean_dibits[7]
        else:
            return None

        # Validate DUID
        try:
            duid = DUID(duid_val)
        except ValueError:
            logger.warning(f"Invalid DUID from simple extraction: 0x{duid_val:x}")
            return None

        # Track NAC even from simple extraction
        if nac_tracker is not None and 0x001 <= nac <= 0xFFE:
            nac_tracker.track(nac)

        # Return with high error count to indicate BCH failed
        return NID(nac=nac, duid=duid, errors=99)

    # Extract NAC (12 bits) and DUID (4 bits) from decoded data
    nac = (decoded_data >> 4) & 0xFFF
    duid_val = decoded_data & 0xF

    # Validate DUID
    try:
        duid = DUID(duid_val)
    except ValueError:
        logger.warning(f"Invalid DUID after BCH: 0x{duid_val:x}")
        return None

    # Track the NAC for future BCH decodes
    if nac_tracker is not None:
        nac_tracker.track(nac)

    # Debug logging (first 10 decodes only)
    if not hasattr(decode_nid, '_decode_count'):
        decode_nid._decode_count = 0
    decode_nid._decode_count += 1
    if decode_nid._decode_count <= 10:
        tracked_nac_str = f"0x{tracked_nac:03x}" if tracked_nac else "0x000"
        logger.info(
            f"decode_nid: NAC=0x{nac:03x}, DUID=0x{duid_val:x}, "
            f"errors={errors}, tracked_nac={tracked_nac_str}"
        )

    return NID(nac=nac, duid=duid, errors=errors)


def decode_hdu(dibits: np.ndarray) -> HDUFrame | None:
    """Decode Header Data Unit.

    HDU structure (total 792 bits = 396 dibits):
    - NID (64 bits)
    - MI - Message Indicator (72 bits)
    - MFID (8 bits)
    - ALGID (8 bits)
    - KID (16 bits)
    - TGID (16 bits)
    - Golay parity blocks
    """
    if len(dibits) < 200:
        logger.debug(f"HDU too short: {len(dibits)} dibits")
        return None

    # Decode NID
    nid = decode_nid(dibits[:32])
    if nid is None:
        return None

    # Remove status symbols and decode rest
    clean_dibits = remove_status_symbols(dibits[32:])
    bits = dibits_to_bits(clean_dibits)

    if len(bits) < 120:
        return None

    # Extract MI (72 bits = 9 bytes) - Golay protected
    mi_bytes = bytearray(9)
    for i in range(9):
        # Each byte is Golay(24,12) encoded
        if (i + 1) * 24 <= len(bits):
            codeword = bits_to_int(bits, i * 24, 24)
            decoded, _errors = golay_decode(codeword)
            if decoded >= 0:
                mi_bytes[i] = decoded & 0xFF

    # Extract ALGID (8 bits)
    algid = bits_to_int(bits, 72, 8) if len(bits) > 80 else 0

    # Extract KID (16 bits)
    kid = bits_to_int(bits, 80, 16) if len(bits) > 96 else 0

    # Extract TGID (16 bits)
    tgid = bits_to_int(bits, 96, 16) if len(bits) > 112 else 0

    logger.info(f"HDU: NAC={nid.nac:03X} ALGID={algid:02X} KID={kid:04X} TGID={tgid}")

    return HDUFrame(
        nid=nid,
        mi=bytes(mi_bytes),
        algid=algid,
        kid=kid,
        tgid=tgid
    )


def decode_ldu1(dibits: np.ndarray) -> LDUFrame | None:
    """Decode Logical Data Unit 1 (voice + link control).

    LDU1 structure (1800 bits = 900 dibits):
    - NID (64 bits)
    - 9 IMBE voice frames (88 bits each = 792 bits)
    - Link Control (72 bits)
    - Low Speed Data (16 bits)
    - Status symbols interspersed
    """
    if len(dibits) < 800:
        logger.debug(f"LDU1 too short: {len(dibits)} dibits")
        return None

    # Decode NID
    nid = decode_nid(dibits[:32])
    if nid is None:
        return None

    # Extract IMBE frames
    # IMBE frame positions in LDU (after removing status symbols)
    imbe_frames = extract_imbe_frames(dibits)

    # Extract Link Control (Hamming protected)
    lc = extract_link_control(dibits)

    return LDUFrame(
        nid=nid,
        imbe_frames=imbe_frames,
        link_control=lc
    )


def decode_ldu2(dibits: np.ndarray) -> LDUFrame | None:
    """Decode Logical Data Unit 2 (voice + encryption sync).

    LDU2 structure similar to LDU1 but contains encryption
    sync instead of link control.
    """
    if len(dibits) < 800:
        logger.debug(f"LDU2 too short: {len(dibits)} dibits")
        return None

    # Decode NID
    nid = decode_nid(dibits[:32])
    if nid is None:
        return None

    # Extract IMBE frames
    imbe_frames = extract_imbe_frames(dibits)

    # Extract Encryption Sync
    es = extract_encryption_sync(dibits)

    return LDUFrame(
        nid=nid,
        imbe_frames=imbe_frames,
        encryption_sync=es
    )


def decode_tdu(dibits: np.ndarray) -> TDUFrame | None:
    """Decode Terminator Data Unit."""
    if len(dibits) < 32:
        return None

    nid = decode_nid(dibits[:32])
    if nid is None:
        return None

    # Check if this is TDULC (with link control)
    lc = None
    if nid.duid == DUID.TDULC and len(dibits) > 100:
        lc = extract_link_control(dibits)

    logger.info(f"TDU: NAC={nid.nac:03X} (end of transmission)")

    return TDUFrame(nid=nid, link_control=lc)


def remove_status_symbols_with_offset(dibits: np.ndarray, frame_offset: int) -> np.ndarray:
    """Remove status symbols from dibit stream with frame position offset.

    Status symbols appear every 36 dibits from frame start. When extracting
    a portion of the frame, we need to know the frame offset to correctly
    identify status symbol positions.

    Args:
        dibits: Dibit array to process
        frame_offset: Starting position within the frame (for status symbol calculation)

    Returns:
        Array with status symbols removed (dtype preserved)
    """
    if len(dibits) == 0:
        return dibits

    result = []
    for i, d in enumerate(dibits):
        frame_pos = frame_offset + i
        # Status symbol at every 36th position from frame start (positions 36, 72, 108, ...)
        # Frame position 36 is the first status symbol (in NID)
        if (frame_pos + 1) % STATUS_SYMBOL_INTERVAL != 0:
            result.append(d)

    return np.array(result, dtype=dibits.dtype)


def decode_tsdu(dibits: np.ndarray, soft: np.ndarray | None = None) -> TSDUFrame | None:
    """Decode Trunking Signaling Data Unit.

    TSDU structure (360 dibits typical):
    - Sync: 24 dibits (48 bits)
    - NID: 33 dibits (includes status symbol at position 12)
    - TSBK blocks: up to 3 blocks, each 196 dibits (with status symbols)

    TSDU contains 1-3 TSBK blocks, each 96 bits after decoding.
    """
    # Frame sync is 24 dibits
    SYNC_DIBITS = 24
    # NID is 33 dibits (32 data + status symbol at position 12)
    NID_DIBITS = 33

    min_frame_dibits = SYNC_DIBITS + NID_DIBITS + 100  # At least some TSBK data
    if len(dibits) < min_frame_dibits:
        logger.debug(f"TSDU too short: {len(dibits)} dibits, need at least {min_frame_dibits}")
        return None

    # Skip sync (24 dibits), decode NID (33 dibits with status symbol)
    nid_start = SYNC_DIBITS
    nid_end = nid_start + NID_DIBITS
    nid_dibits = dibits[nid_start:nid_end]

    # Debug: log first 15 NID dibits
    if not hasattr(decode_tsdu, '_debug_count'):
        decode_tsdu._debug_count = 0
    decode_tsdu._debug_count += 1
    if decode_tsdu._debug_count <= 10:
        logger.info(f"decode_tsdu: nid_dibits[0:15]={list(nid_dibits[:15])}, len={len(nid_dibits)}")

    nid = decode_nid(nid_dibits, skip_status_at_10=True)
    if nid is None:
        logger.debug("TSDU: NID decode failed")
        return None

    if decode_tsdu._debug_count <= 10:
        logger.info(f"decode_tsdu: NID decoded: nac=0x{nid.nac:03x}, duid={nid.duid}")

    # Extract TSBK data and remove status symbols
    # TSBK data starts at frame position 57 (after sync + NID)
    tsbk_data_start = nid_end
    tsbk_raw = dibits[tsbk_data_start:]
    tsbk_soft = soft[tsbk_data_start:] if soft is not None else None

    # Remove status symbols from TSBK data
    # Status symbols are at frame positions 72, 108, 144, etc.
    # which are positions 15, 51, 87, etc. relative to TSBK data start
    tsbk_clean = remove_status_symbols_with_offset(tsbk_raw, frame_offset=tsbk_data_start)
    tsbk_soft_clean = (
        remove_status_symbols_with_offset(tsbk_soft, frame_offset=tsbk_data_start)
        if tsbk_soft is not None
        else None
    )

    logger.debug(f"TSDU: tsbk_raw={len(tsbk_raw)}, tsbk_clean={len(tsbk_clean)} dibits")

    # Extract TSBK blocks from cleaned data
    tsbk_blocks = extract_tsbk_blocks(tsbk_clean, tsbk_soft_clean)

    return TSDUFrame(
        nid=nid,
        tsbk_blocks=tsbk_blocks
    )


def extract_imbe_frames(dibits: np.ndarray) -> list[bytes]:
    """Extract 9 IMBE voice frames from LDU.

    Each IMBE frame is 88 bits (44 dibits), protected by
    various error correction codes.

    Returns list of 9 IMBE frames as bytes.
    """
    frames = []

    # IMBE frame positions (simplified - actual positions are interleaved)
    # Real decoder needs to de-interleave based on P25 spec
    clean_dibits = remove_status_symbols(dibits[32:])

    frame_size = 44  # dibits per IMBE frame

    for i in range(9):
        start = i * frame_size
        if start + frame_size <= len(clean_dibits):
            frame_dibits = clean_dibits[start:start + frame_size]
            # Convert dibits to bytes (11 bytes per frame)
            frame_bytes = bytearray(11)
            for j in range(min(44, len(frame_dibits))):
                byte_idx = j // 4
                bit_offset = (j % 4) * 2
                if byte_idx < 11:
                    frame_bytes[byte_idx] |= (frame_dibits[j] & 0x3) << (6 - bit_offset)
            frames.append(bytes(frame_bytes))
        else:
            # Pad with silence frame
            frames.append(bytes(11))

    return frames


def extract_link_control(dibits: np.ndarray) -> LinkControl:
    """Extract Link Control from LDU1/TDULC.

    LC is 72 bits (36 dibits) protected by Reed-Solomon.
    Handles standard LC and Extended LC (GPS data).
    """
    # Simplified extraction - real decoder needs RS correction
    clean_dibits = remove_status_symbols(dibits)
    bits = dibits_to_bits(clean_dibits)

    # LC offset in frame (after NID and before voice)
    lc_offset = 64  # bits

    if len(bits) < lc_offset + 72:
        return LinkControl()

    # Extract LC fields
    lcf = bits_to_int(bits, lc_offset, 8)
    mfid = bits_to_int(bits, lc_offset + 8, 8)

    # Default values
    tgid = 0
    source_id = 0
    has_gps = False
    gps_lat = 0.0
    gps_lon = 0.0
    gps_alt: float | None = None
    gps_speed: float | None = None
    gps_heading: float | None = None

    # Field interpretation depends on LCF
    if lcf == 0x00:  # Group Voice Channel User
        tgid = bits_to_int(bits, lc_offset + 24, 16)
        source_id = bits_to_int(bits, lc_offset + 40, 24)

    elif lcf == 0x09:  # GPS Position (Extended Link Control)
        # Standard GPS: 6 bytes (48 bits) of lat/lon
        source_id = bits_to_int(bits, lc_offset + 16, 24)
        gps_lat, gps_lon = _decode_lc_gps_coords(bits, lc_offset + 40)
        has_gps = True
        logger.debug(f"ELC GPS: unit={source_id} lat={gps_lat:.6f} lon={gps_lon:.6f}")

    elif lcf == 0x0A:  # GPS Position Extended (with altitude)
        source_id = bits_to_int(bits, lc_offset + 16, 24)
        gps_lat, gps_lon = _decode_lc_gps_coords(bits, lc_offset + 40)
        # Altitude in remaining bits (if present)
        if len(bits) >= lc_offset + 88:
            alt_raw = bits_to_int(bits, lc_offset + 88, 16)
            gps_alt = float(alt_raw) - 500.0  # 500m offset
        has_gps = True

    elif lcf == 0x0B:  # GPS Position with Velocity
        source_id = bits_to_int(bits, lc_offset + 16, 24)
        gps_lat, gps_lon = _decode_lc_gps_coords(bits, lc_offset + 40)
        # Velocity in remaining bits
        if len(bits) >= lc_offset + 96:
            speed_raw = bits_to_int(bits, lc_offset + 88, 8)
            heading_raw = bits_to_int(bits, lc_offset + 96, 9)
            gps_speed = speed_raw * 2.0  # km/h
            gps_heading = heading_raw * 360.0 / 512.0  # degrees
        has_gps = True

    return LinkControl(
        lcf=lcf,
        mfid=mfid,
        tgid=tgid,
        source_id=source_id,
        has_gps=has_gps,
        gps_latitude=gps_lat,
        gps_longitude=gps_lon,
        gps_altitude_m=gps_alt,
        gps_speed_kmh=gps_speed,
        gps_heading_deg=gps_heading,
    )


def _decode_lc_gps_coords(bits: np.ndarray, offset: int) -> tuple[float, float]:
    """Decode GPS coordinates from Link Control bits.

    GPS in Extended LC uses 24-bit signed values:
    - Latitude: -90 to +90 degrees
    - Longitude: -180 to +180 degrees

    Args:
        bits: Bit array
        offset: Starting bit position

    Returns:
        Tuple of (latitude, longitude)
    """
    if len(bits) < offset + 48:
        return (0.0, 0.0)

    # Latitude: 24-bit signed
    lat_raw = bits_to_int(bits, offset, 24)
    if lat_raw & 0x800000:  # Sign extend
        lat_raw -= 0x1000000
    latitude = lat_raw * 90.0 / (1 << 23)

    # Longitude: 24-bit signed
    lon_raw = bits_to_int(bits, offset + 24, 24)
    if lon_raw & 0x800000:  # Sign extend
        lon_raw -= 0x1000000
    longitude = lon_raw * 180.0 / (1 << 23)

    return (latitude, longitude)


def extract_encryption_sync(dibits: np.ndarray) -> EncryptionSync:
    """Extract Encryption Sync from LDU2.

    ES contains ALGID, KID, and Message Indicator (MI).
    """
    clean_dibits = remove_status_symbols(dibits)
    bits = dibits_to_bits(clean_dibits)

    es_offset = 64  # bits

    if len(bits) < es_offset + 96:
        return EncryptionSync()

    algid = bits_to_int(bits, es_offset, 8)
    kid = bits_to_int(bits, es_offset + 8, 16)

    # MI is 72 bits (9 bytes)
    mi = bytearray(9)
    for i in range(9):
        mi[i] = bits_to_int(bits, es_offset + 24 + i * 8, 8)

    return EncryptionSync(
        algid=algid,
        kid=kid,
        mi=bytes(mi)
    )


def extract_tsbk_blocks(dibits: np.ndarray, soft: np.ndarray | None = None) -> list[TSBKBlock]:
    """Extract TSBK blocks from TSDU.

    Each TSBK block in a TSDU is:
    - 98 dibits (trellis-encoded, interleaved) = 196 bits
    - After deinterleaving: 196 bits (reordered)
    - After trellis 1/2 rate decode: 49 dibits = 98 bits
    - Final structure: 1 LB + 1 Protect + 6 Opcode + 8 MFID + 64 Data + 16 CRC = 96 bits
      (2 bits are flushing bits from trellis and discarded)

    TSDU can contain 1-3 TSBK blocks.

    Processing steps:
    1. Extract 98 dibits per TSBK block
    2. Convert 98 dibits → 196 bits
    3. Deinterleave 196 bits using DATA_DEINTERLEAVE pattern
    4. Convert 196 bits → 98 dibits for trellis decoder
    5. Trellis decode 98 dibits → 49 dibits (98 bits, use first 96)
    6. Validate CRC-16 CCITT and parse TSBK structure

    Args:
        dibits: TSDU dibits after NID (with status symbols removed)
        soft: Optional soft symbol values aligned with dibits

    Returns:
        List of decoded TSBK blocks (1-3 blocks)
    """
    blocks = []

    # Each TSBK block is 98 dibits (196 bits after conversion)
    # TSDU can hold up to 3 TSBK blocks: 98 * 3 = 294 dibits
    TSBK_ENCODED_SIZE = 98  # dibits per TSBK

    # DEBUG: Log input size
    logger.info(f"extract_tsbk_blocks: input dibits={len(dibits)}")
    if soft is not None and len(soft) != len(dibits):
        logger.warning(
            f"extract_tsbk_blocks: soft length mismatch (soft={len(soft)}, dibits={len(dibits)})"
        )
        soft = None

    def _rotate_soft(vals: np.ndarray, xor_mask: int) -> np.ndarray:
        if xor_mask == 0:
            return vals
        if xor_mask == 2:
            return -vals
        # XOR 1/3 swaps inner/outer levels; approximate by swapping |1| and |3|
        abs_vals = np.clip(np.abs(vals), 0.0, 3.0)
        swapped = (4.0 - abs_vals) * np.sign(vals)
        if xor_mask == 1:
            return swapped
        # xor_mask == 3
        return -swapped

    offset = 0
    for block_idx in range(3):  # Max 3 TSBKs per TSDU
        # Check if we have enough dibits for another TSBK
        if offset + TSBK_ENCODED_SIZE > len(dibits):
            logger.info(f"TSBK block {block_idx}: not enough dibits (have {len(dibits)-offset}, need {TSBK_ENCODED_SIZE})")
            break

        # Extract 98 dibits for this TSBK
        tsbk_dibits = dibits[offset:offset + TSBK_ENCODED_SIZE]
        tsbk_soft = soft[offset:offset + TSBK_ENCODED_SIZE] if soft is not None else None

        # Try all 4 phase rotations (QPSK ambiguity) and find best
        # XOR 0: identity (no change)
        # XOR 1: swap 0↔1 and 2↔3 (90° rotation)
        # XOR 2: swap 0↔2 and 1↔3 (180° polarity flip)
        # XOR 3: swap 0↔3 and 1↔2 (270° rotation)
        best_decoded = None
        best_error = float('inf')
        best_xor = 0

        for xor_mask in [0, 1, 2, 3]:
            rotated = (tsbk_dibits ^ xor_mask).astype(np.uint8) if xor_mask else tsbk_dibits
            rot_soft = None
            if tsbk_soft is not None:
                rot_soft = _rotate_soft(tsbk_soft, xor_mask)

            # Convert 98 dibits to 196 bits for deinterleaving
            interleaved_bits = dibits_to_bits(rotated)

            if len(interleaved_bits) < 196:
                continue

            # Step 1: Deinterleave using P25 data pattern (196 bits)
            deinterleaved_bits = deinterleave_data(interleaved_bits)

            # Step 2: Trellis decode (1/2 rate: 196 bits → 98 bits)
            # Convert bits back to dibits for trellis decoder
            trellis_dibits_rot = np.zeros(98, dtype=np.uint8)
            for i in range(98):
                trellis_dibits_rot[i] = (deinterleaved_bits[i*2] << 1) | deinterleaved_bits[i*2 + 1]

            # Try with deinterleave
            if rot_soft is not None and len(rot_soft) >= 98:
                deint_soft = rot_soft[DATA_DEINTERLEAVE_DIBITS]
                decoded, error = trellis_decode(trellis_dibits_rot, soft_values=deint_soft)
            else:
                decoded, error = trellis_decode(trellis_dibits_rot)

            if error < best_error:
                best_error = error
                best_decoded = decoded
                best_xor = xor_mask

            # Try without deinterleave (some captures may already be deinterleaved)
            if rot_soft is not None and len(rot_soft) >= 98:
                decoded_raw, error_raw = trellis_decode(rotated, soft_values=rot_soft)
            else:
                decoded_raw, error_raw = trellis_decode(rotated)

            if error_raw < best_error:
                best_error = error_raw
                best_decoded = decoded_raw
                best_xor = xor_mask

        if best_decoded is None:
            logger.warning(f"TSBK block {block_idx}: all XOR rotations failed")
            break

        decoded_dibits = best_decoded
        error_metric = best_error

        if block_idx == 0:
            logger.info(f"TSBK block 0: best XOR mask={best_xor}, error_metric={best_error}")

        # Convert decoded dibits to bits (49 dibits → 98 bits, but we only need 96)
        if len(decoded_dibits) < 48:
            logger.warning(f"TSBK block {block_idx}: trellis decode failed, got {len(decoded_dibits)} dibits")
            break

        # Take first 48 decoded dibits and convert to 96 bits
        decoded_bits = np.zeros(96, dtype=np.uint8)
        for i in range(48):
            if i < len(decoded_dibits):
                decoded_bits[i*2] = (decoded_dibits[i] >> 1) & 1
                decoded_bits[i*2 + 1] = decoded_dibits[i] & 1

        # Step 3: Validate CRC-16 CCITT (first 80 bits + 16-bit CRC = 96 bits total)
        crc_valid, crc_errors = crc16_ccitt_p25(decoded_bits)

        # Step 4: Parse TSBK structure
        # Bit layout: LB(1) + Protect(1) + Opcode(6) + MFID(8) + Data(64) + CRC(16)
        last_block = bool(decoded_bits[0])
        bool(decoded_bits[1])
        opcode = bits_to_int(decoded_bits, 2, 6)
        mfid = bits_to_int(decoded_bits, 8, 8)

        # Extract data (64 bits = 8 bytes, starting at bit 16)
        data = bytearray(8)
        for i in range(8):
            data[i] = bits_to_int(decoded_bits, 16 + i * 8, 8)

        logger.info(
            f"TSBK {block_idx}: LB={last_block} opcode=0x{opcode:02x} "
            f"mfid=0x{mfid:02x} crc_valid={crc_valid} errors={crc_errors} error_metric={error_metric}"
        )

        blocks.append(TSBKBlock(
            last_block=last_block,
            opcode=opcode,
            mfid=mfid,
            data=bytes(data),
            crc_valid=crc_valid
        ))

        # Stop if this was the last block
        if last_block:
            break

        # Move to next TSBK block
        offset += TSBK_ENCODED_SIZE

    return blocks
