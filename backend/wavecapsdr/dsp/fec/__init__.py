"""Forward Error Correction (FEC) module for P25 decoding.

This module provides error correction decoders used in P25:
- Golay(24,12) - Corrects up to 3 bit errors, used for header fields
- Trellis (1/2 rate) - Viterbi decoder, used for voice frame data
- Reed-Solomon - For data packet correction (future)

These FEC codes protect critical control information and voice data
from channel errors in the P25 standard.
"""

from wavecapsdr.dsp.fec.golay import golay_decode, golay_encode, golay_syndrome
from wavecapsdr.dsp.fec.trellis import TrellisDecoder, trellis_decode

__all__ = [
    "TrellisDecoder",
    "golay_decode",
    "golay_encode",
    "golay_syndrome",
    "trellis_decode",
]
