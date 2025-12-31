from wavecapsdr.decoders.p25_tsbk import TSBKOpcode, TSBKParser


def test_tsbk_iden_up_vu_rejects_out_of_range_base_freq() -> None:
    parser = TSBKParser()

    data = bytearray(8)
    ident = 1
    bw = 0x5  # 12.5 kHz
    data[0] = (ident << 4) | bw

    # tx_offset_sign=1, tx_offset=1
    data[1] = 0x80
    data[2] = 0x04  # tx_offset upper bits, spacing top bits = 0
    data[3] = 0x01  # spacing = 1 (0.125 kHz)

    base_freq_mhz = 11000.0
    base_freq_raw = int(base_freq_mhz / 0.000005)
    data[4:8] = base_freq_raw.to_bytes(4, "big")

    result = parser.parse(TSBKOpcode.IDEN_UP_VU, 0, bytes(data))
    assert result.message_type == "PARSE_ERROR"
    assert "base_freq_mhz" in result.error
