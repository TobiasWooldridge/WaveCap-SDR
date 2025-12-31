from __future__ import annotations

import pytest

from wavecapsdr.decoders.p25_framer import (
    P25P1DataUnitID,
    P25P1MessageAssembler,
    P25P1MessageFramer,
)


def _noop_listener(_: object) -> None:
    return None


def test_message_assembler_rejects_invalid_dibit() -> None:
    assembler = P25P1MessageAssembler(nac=0x123, duid=P25P1DataUnitID.TRUNKING_SIGNALING_BLOCK_1)

    with pytest.raises(AssertionError):
        assembler.receive(5)


def test_dispatch_tsbk_requires_complete_blocks() -> None:
    framer = P25P1MessageFramer()
    framer.set_listener(_noop_listener)
    framer.start()

    framer._message_assembler = P25P1MessageAssembler(
        nac=0x123,
        duid=P25P1DataUnitID.TRUNKING_SIGNALING_BLOCK_1,
    )
    framer._message_assembler._bits = [0] * 150

    with pytest.raises(AssertionError):
        framer._dispatch_tsbk()


def test_dispatch_pdu_requires_block_alignment() -> None:
    framer = P25P1MessageFramer()
    framer.set_listener(_noop_listener)
    framer.start()

    framer._message_assembler = P25P1MessageAssembler(
        nac=0x321,
        duid=P25P1DataUnitID.PACKET_DATA_UNIT,
    )
    framer._message_assembler._bits = [0] * 200

    with pytest.raises(AssertionError):
        framer._dispatch_pdu()
