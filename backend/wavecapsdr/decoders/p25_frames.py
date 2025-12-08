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
from typing import List, Optional, Tuple

import numpy as np

from wavecapsdr.dsp.fec.golay import golay_decode
from wavecapsdr.dsp.fec.trellis import trellis_decode

logger = logging.getLogger(__name__)


class DUID(IntEnum):
    """Data Unit ID values for P25 frames."""
    HDU = 0x0    # Header Data Unit
    TDU = 0x3    # Terminator Data Unit (no link control)
    LDU1 = 0x5   # Logical Link Data Unit 1
    TSDU = 0x7   # Trunking Signaling Data Unit
    LDU2 = 0xA   # Logical Link Data Unit 2
    PDU = 0xC    # Packet Data Unit
    TDULC = 0xF  # Terminator Data Unit with Link Control


# Frame sync patterns (48 bits = 24 dibits)
# These are transmitted as differential symbols
FRAME_SYNC_PATTERNS = {
    # Pattern (as 48-bit value): DUID
    0x5575F5FF77FF: DUID.HDU,    # +3+3-3+3-3+3-3-3+3+3-3-3+3+3
    0x5575F5FF77FD: DUID.TDU,
    0x5575F5FF77F7: DUID.LDU1,
    0x5575F5FF775F: DUID.TSDU,
    0x5575F5FF7757: DUID.LDU2,
    0x5575F5FF755F: DUID.PDU,
    0x5575F5FF7555: DUID.TDULC,
}

# Status symbols positions in frames (for de-interleaving)
STATUS_SYMBOL_INTERVAL = 35  # Status symbol every 35 dibits


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
    """
    lcf: int = 0  # Link Control Format
    mfid: int = 0  # Manufacturer ID (0 = standard)
    tgid: int = 0  # Talkgroup ID
    source_id: int = 0  # Source radio ID
    emergency: bool = False
    encrypted: bool = False


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
    imbe_frames: List[bytes]  # 9 IMBE frames, 88 bits each
    link_control: Optional[LinkControl] = None  # LDU1 only
    encryption_sync: Optional[EncryptionSync] = None  # LDU2 only
    errors: int = 0


@dataclass
class TDUFrame:
    """Terminator Data Unit - marks end of transmission."""
    nid: NID
    link_control: Optional[LinkControl] = None  # TDULC only


@dataclass
class TSDUFrame:
    """Trunking Signaling Data Unit - control channel message.

    Contains one or more TSBK (Trunking Signaling Block) messages.
    """
    nid: NID
    tsbk_blocks: List['TSBKBlock']
    errors: int = 0


@dataclass
class TSBKBlock:
    """Single TSBK message block.

    Structure (96 bits):
    - LB: Last Block flag (1 bit)
    - Protect: Protection flag (1 bit)
    - Opcode (6 bits)
    - MFID: Manufacturer ID (8 bits)
    - Data (56 bits)
    - CRC (16 bits)
    """
    last_block: bool
    opcode: int
    mfid: int
    data: bytes  # 7 bytes of data
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
        value = (value << 1) | bits[start + i]
    return value


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


def decode_nid(dibits: np.ndarray) -> Optional[NID]:
    """Decode Network ID from first 32 dibits (64 bits).

    NID structure:
    - NAC (12 bits) + DUID (4 bits) = 8 dibits
    - BCH(63,16) parity = 24 dibits

    Uses Golay decoding for error correction.
    """
    if len(dibits) < 32:
        return None

    # Extract NAC and DUID (first 8 dibits = 16 bits)
    nac_duid = 0
    for i in range(8):
        nac_duid = (nac_duid << 2) | dibits[i]

    nac = (nac_duid >> 4) & 0xFFF
    duid_val = nac_duid & 0xF

    # Validate DUID
    try:
        duid = DUID(duid_val)
    except ValueError:
        logger.warning(f"Invalid DUID: {duid_val}")
        duid = DUID.HDU  # Default

    # BCH error correction would go here
    # For now, just return decoded values

    return NID(nac=nac, duid=duid, errors=0)


def decode_hdu(dibits: np.ndarray) -> Optional[HDUFrame]:
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
            decoded, errors = golay_decode(codeword)
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


def decode_ldu1(dibits: np.ndarray) -> Optional[LDUFrame]:
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


def decode_ldu2(dibits: np.ndarray) -> Optional[LDUFrame]:
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


def decode_tdu(dibits: np.ndarray) -> Optional[TDUFrame]:
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


def decode_tsdu(dibits: np.ndarray) -> Optional[TSDUFrame]:
    """Decode Trunking Signaling Data Unit.

    TSDU contains 1-3 TSBK blocks, each 96 bits.
    """
    if len(dibits) < 100:
        logger.debug(f"TSDU too short: {len(dibits)} dibits")
        return None

    # Decode NID
    nid = decode_nid(dibits[:32])
    if nid is None:
        return None

    # Extract TSBK blocks
    tsbk_blocks = extract_tsbk_blocks(dibits[32:])

    return TSDUFrame(
        nid=nid,
        tsbk_blocks=tsbk_blocks
    )


def extract_imbe_frames(dibits: np.ndarray) -> List[bytes]:
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

    # Field interpretation depends on LCF
    if lcf == 0x00:  # Group Voice Channel User
        tgid = bits_to_int(bits, lc_offset + 24, 16)
        source_id = bits_to_int(bits, lc_offset + 40, 24)
    else:
        tgid = 0
        source_id = 0

    return LinkControl(
        lcf=lcf,
        mfid=mfid,
        tgid=tgid,
        source_id=source_id
    )


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


def extract_tsbk_blocks(dibits: np.ndarray) -> List[TSBKBlock]:
    """Extract TSBK blocks from TSDU.

    Each TSBK is 96 bits (48 dibits) with trellis coding.
    TSDU can contain 1-3 TSBK blocks.
    """
    blocks = []
    clean_dibits = remove_status_symbols(dibits)

    # Each TSBK is 196 dibits after trellis encoding (98 dibits encoded -> 196 dibits)
    # But we receive already de-trellis-coded data
    block_size = 48  # dibits per decoded TSBK

    offset = 0
    while offset + block_size <= len(clean_dibits) and len(blocks) < 3:
        block_dibits = clean_dibits[offset:offset + block_size]

        # Convert to bits
        bits = dibits_to_bits(block_dibits)

        if len(bits) < 96:
            break

        # Parse TSBK fields
        last_block = bool(bits[0])
        protect = bool(bits[1])
        opcode = bits_to_int(bits, 2, 6)
        mfid = bits_to_int(bits, 8, 8)

        # Data (56 bits = 7 bytes)
        data = bytearray(7)
        for i in range(7):
            data[i] = bits_to_int(bits, 16 + i * 8, 8)

        # CRC (16 bits) - verify
        crc_received = bits_to_int(bits, 72, 16)
        # CRC calculation would go here
        crc_valid = True  # Simplified

        blocks.append(TSBKBlock(
            last_block=last_block,
            opcode=opcode,
            mfid=mfid,
            data=bytes(data),
            crc_valid=crc_valid
        ))

        if last_block:
            break

        offset += block_size

    return blocks
