#!/usr/bin/env python3
"""TSBK diagnostic: trace deinterleave/trellis/CRC on control-channel captures."""

from __future__ import annotations

import argparse
import logging
import re
import wave
from pathlib import Path

import numpy as np

from wavecapsdr.decoders.p25 import (
    C4FMDemodulator,
    CQPSKDemodulator,
    P25FrameSync,
    P25FrameType,
)
from wavecapsdr.decoders.p25_frames import (
    bits_to_int,
    decode_nid,
    deinterleave_data,
    dibits_to_bits,
)
from wavecapsdr.decoders.p25 import P25Decoder
from wavecapsdr.dsp.fec.trellis import trellis_decode

logger = logging.getLogger(__name__)


def load_iq_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

    if n_channels != 2:
        raise ValueError(f"Expected stereo IQ WAV, got {n_channels} channels")

    if sample_width == 3:
        data_bytes = np.frombuffer(raw_data, dtype=np.uint8).reshape(-1, n_channels, 3)
        samples = np.zeros((n_frames, n_channels), dtype=np.int32)
        for ch in range(n_channels):
            b0 = data_bytes[:, ch, 0].astype(np.int32)
            b1 = data_bytes[:, ch, 1].astype(np.int32)
            b2 = data_bytes[:, ch, 2].astype(np.int32)
            raw24 = b0 | (b1 << 8) | (b2 << 16)
            samples[:, ch] = np.where(raw24 >= 0x800000, raw24 - 0x1000000, raw24)
        samples_f = samples.astype(np.float32) / 8388608.0
    elif sample_width == 2:
        samples = np.frombuffer(raw_data, dtype=np.int16).reshape(-1, n_channels)
        samples_f = samples.astype(np.float32) / 32768.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    iq = samples_f[:, 0] + 1j * samples_f[:, 1]
    return iq, sample_rate


def _infer_rate_from_name(path: Path) -> int | None:
    match = re.search(r"(\\d{5,})", path.stem)
    if not match:
        return None
    return int(match.group(1))


def load_iq_rawiq(path: Path, sample_rate: int | None) -> tuple[np.ndarray, int]:
    if sample_rate is None:
        sample_rate = _infer_rate_from_name(path)
    if sample_rate is None:
        raise ValueError("rawiq requires --sample-rate or a rate in the filename")

    raw = path.read_bytes()
    data = np.frombuffer(raw, dtype=np.int16)
    if len(data) % 2 != 0:
        raise ValueError("rawiq length is not an even I/Q sample count")

    i = data[0::2].astype(np.float32) / 32768.0
    q = data[1::2].astype(np.float32) / 32768.0
    iq = i + 1j * q
    return iq, sample_rate


def _dibits_to_bits_lsb(dibits: np.ndarray) -> np.ndarray:
    bits = np.zeros(len(dibits) * 2, dtype=np.uint8)
    for i, d in enumerate(dibits):
        bits[i * 2] = d & 1
        bits[i * 2 + 1] = (d >> 1) & 1
    return bits


def _bits_to_dibits(bits: np.ndarray) -> np.ndarray:
    dibits = np.zeros(len(bits) // 2, dtype=np.uint8)
    for i in range(len(dibits)):
        dibits[i] = (bits[i * 2] << 1) | bits[i * 2 + 1]
    return dibits


def _strip_status_symbols(
    dibits: np.ndarray,
    frame_offset: int,
    interval: int,
) -> np.ndarray:
    if len(dibits) == 0 or interval <= 0:
        return dibits

    result = []
    for i, d in enumerate(dibits):
        frame_pos = frame_offset + i
        if (frame_pos + 1) % interval != 0:
            result.append(d)

    return np.array(result, dtype=dibits.dtype)


def _crc_check(
    bits: np.ndarray,
    init: int,
    xorout: int,
    flush: bool,
) -> tuple[bool, int, int]:
    if len(bits) < 96:
        return False, 0, 0
    poly = 0x1021
    crc = init
    for i in range(80):
        bit = int(bits[i])
        msb = (crc >> 15) & 1
        crc = ((crc << 1) | bit) & 0xFFFF
        if msb:
            crc ^= poly
    if flush:
        for _ in range(16):
            msb = (crc >> 15) & 1
            crc = (crc << 1) & 0xFFFF
            if msb:
                crc ^= poly
    crc ^= xorout
    recv = bits_to_int(bits, 80, 16)
    return crc == recv, crc, recv


def analyze_tsdu(
    frame_dibits: np.ndarray,
    frame_idx: int,
    decode_offsets: list[int],
    status_intervals: list[int],
    crc_variants: list[tuple[str, int, int, bool]],
) -> list[tuple[tuple[str, str, int, int, str, str], int, bool]]:
    nid = decode_nid(frame_dibits[24:57], skip_status_at_10=True)
    if nid is None:
        print(f"frame {frame_idx}: NID decode failed")
        return []

    print(f"frame {frame_idx}: nac=0x{nid.nac:03x} duid={nid.duid} bch_err={nid.errors}")

    tsbk_raw = frame_dibits[57:]

    results = []

    sources = [("raw", tsbk_raw)]
    for interval in status_intervals:
        cleaned = _strip_status_symbols(tsbk_raw, frame_offset=57, interval=interval)
        sources.append((f"clean{interval}", cleaned))
    deint_modes = ["bits", "dibits", "none"]
    bit_orders = ["msb", "lsb"]

    for block_idx in range(3):
        start = block_idx * 98
        if start + 98 > len(tsbk_raw):
            break

        for source_name, source in sources:
            if start + 98 > len(source):
                continue
            tsbk_dibits = source[start:start + 98]

            for xor_mask in (0, 1, 2, 3):
                rotated = (tsbk_dibits ^ xor_mask).astype(np.uint8) if xor_mask else tsbk_dibits

                for deint_mode in deint_modes:
                    if deint_mode == "bits":
                        interleaved_bits = dibits_to_bits(rotated)
                        if len(interleaved_bits) < 196:
                            continue
                        deinterleaved_bits = deinterleave_data(interleaved_bits)
                        trellis_dibits = _bits_to_dibits(deinterleaved_bits)
                    elif deint_mode == "dibits":
                        trellis_dibits = rotated[P25Decoder.DATA_DEINTERLEAVE]
                    else:
                        trellis_dibits = rotated

                    decoded, trellis_err = trellis_decode(trellis_dibits)
                    if decoded is None or len(decoded) < 48:
                        continue

                    for decode_offset in decode_offsets:
                        if decode_offset + 48 > len(decoded):
                            continue
                        payload = decoded[decode_offset:decode_offset + 48]

                        for bit_order in bit_orders:
                            bits = (
                                dibits_to_bits(payload)
                                if bit_order == "msb"
                                else _dibits_to_bits_lsb(payload)
                            )

                            for variant_name, init, xorout, flush in crc_variants:
                                crc_ok, _calc, _recv = _crc_check(bits, init, xorout, flush)
                                key = (
                                    source_name,
                                    deint_mode,
                                    xor_mask,
                                    decode_offset,
                                    bit_order,
                                    variant_name,
                                )
                                results.append((key, trellis_err, crc_ok))

    return results


def analyze_capture(
    iq: np.ndarray,
    sample_rate: int,
    modulation: str,
    max_frames: int,
    chunk_ms: float,
    decode_offsets: list[int],
    status_intervals: list[int],
    sync_threshold: int,
    crc_variants: list[tuple[str, int, int, bool]],
) -> None:
    if modulation == "lsm":
        demod = CQPSKDemodulator(sample_rate=sample_rate, symbol_rate=4800)
    else:
        demod = C4FMDemodulator(sample_rate=sample_rate, symbol_rate=4800)

    frame_sync = P25FrameSync()
    frame_sync.SOFT_SYNC_THRESHOLD = sync_threshold
    frame_sync.SOFT_SYNC_THRESHOLD_OPTIMIZE = max(
        sync_threshold,
        frame_sync.SOFT_SYNC_THRESHOLD_OPTIMIZE,
    )
    buffer: list[int] = []

    chunk_len = max(100, int(sample_rate * (chunk_ms / 1000.0)))
    frame_count = 0
    stats: dict[tuple[str, str, int, int, str, str], dict[str, float]] = {}

    for start in range(0, len(iq), chunk_len):
        chunk = iq[start:start + chunk_len]
        if len(chunk) < 100:
            break

        dibits = demod.demodulate(chunk.astype(np.complex64, copy=False))
        if len(dibits) == 0:
            continue
        buffer.extend(dibits.tolist())

        while True:
            if len(buffer) < 360:
                break
            buf_arr = np.array(buffer, dtype=np.uint8)
            sync_pos, frame_type, _nac, _duid = frame_sync.find_sync(buf_arr)
            if sync_pos is None:
                if len(buffer) > 720:
                    buffer = buffer[-360:]
                break
            if sync_pos + 360 > len(buffer):
                break

            frame = buf_arr[sync_pos:sync_pos + 360]
            buffer = buffer[sync_pos + 360:]

            if frame_type == P25FrameType.TSDU:
                frame_count += 1
                results = analyze_tsdu(
                    frame,
                    frame_count,
                    decode_offsets,
                    status_intervals,
                    crc_variants,
                )
                for key, trellis_err, crc_ok in results:
                    entry = stats.setdefault(key, {"total": 0, "pass": 0, "err_sum": 0.0})
                    entry["total"] += 1
                    if crc_ok:
                        entry["pass"] += 1
                    entry["err_sum"] += trellis_err
                if frame_count >= max_frames:
                    break
            else:
                buffer = buffer[-360:]
                break

        if frame_count >= max_frames:
            break

    ranked = []
    for key, entry in stats.items():
        total = int(entry["total"])
        passed = int(entry["pass"])
        if total == 0:
            continue
        pass_rate = passed / total
        err_mean = entry["err_sum"] / total
        ranked.append((pass_rate, err_mean, total, passed, key))

    ranked.sort(key=lambda r: (-r[0], r[1], -r[2]))
    interval_label = ",".join(str(v) for v in status_intervals)
    print(
        "summary: frames=%d, candidates=%d, sync_threshold=%d, status_intervals=%s"
        % (frame_count, len(ranked), sync_threshold, interval_label)
    )
    for pass_rate, err_mean, total, passed, key in ranked[:10]:
        source_name, deint_mode, xor_mask, decode_offset, bit_order, variant = key
        print(
            f"  pass_rate={pass_rate*100:.1f}% pass={passed}/{total} err_mean={err_mean:.2f} "
            f"src={source_name} deint={deint_mode} xor={xor_mask} offset={decode_offset} "
            f"bits={bit_order} crc={variant}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="TSBK diagnostic for control-channel captures")
    parser.add_argument("input", type=Path, help="Input IQ capture (.wav or .rawiq)")
    parser.add_argument("--modulation", choices=["c4fm", "lsm"], default="c4fm", help="P25 modulation")
    parser.add_argument("--sample-rate", type=int, default=None, help="Sample rate for rawiq inputs (Hz)")
    parser.add_argument("--start-seconds", type=float, default=0.0, help="Seconds to skip from start")
    parser.add_argument("--limit-seconds", type=float, default=0.0, help="Limit analysis to N seconds")
    parser.add_argument("--max-frames", type=int, default=5, help="Max TSDU frames to analyze")
    parser.add_argument("--chunk-ms", type=float, default=100.0, help="Chunk size in milliseconds")
    parser.add_argument("--decode-offsets", type=str, default="0,1", help="Comma-separated decode offsets")
    parser.add_argument(
        "--status-intervals",
        type=str,
        default="35",
        help="Comma-separated status symbol intervals to try",
    )
    parser.add_argument(
        "--sync-thresholds",
        type=str,
        default="60",
        help="Comma-separated soft sync thresholds to try",
    )
    parser.add_argument("--crc-sweep", action="store_true", help="Sweep CRC init/xor/flush variants")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    path = args.input
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    if path.suffix.lower() == ".wav":
        iq, sample_rate = load_iq_wav(path)
    elif path.suffix.lower() == ".rawiq":
        iq, sample_rate = load_iq_rawiq(path, args.sample_rate)
    else:
        raise SystemExit(f"Unsupported input format: {path.suffix}")

    if args.start_seconds > 0:
        start_idx = int(args.start_seconds * sample_rate)
        iq = iq[start_idx:]
    if args.limit_seconds > 0:
        end_idx = int(args.limit_seconds * sample_rate)
        iq = iq[:end_idx]

    logger.info(
        "Loaded %s: samples=%d, sample_rate=%d Hz, mean_mag=%.4f",
        path.name,
        len(iq),
        sample_rate,
        float(np.mean(np.abs(iq))) if len(iq) else 0.0,
    )

    decode_offsets = [int(v.strip()) for v in args.decode_offsets.split(",") if v.strip()]
    status_intervals = [int(v.strip()) for v in args.status_intervals.split(",") if v.strip()]
    if not status_intervals:
        status_intervals = [35]
    sync_thresholds = [int(v.strip()) for v in args.sync_thresholds.split(",") if v.strip()]
    if not sync_thresholds:
        sync_thresholds = [60]
    crc_variants = [
        ("ccitt_init_ffff", 0xFFFF, 0x0000, True),
    ]
    if args.crc_sweep:
        crc_variants = [
            ("ccitt_init_ffff", 0xFFFF, 0x0000, True),
            ("ccitt_init_0000", 0x0000, 0x0000, True),
            ("ccitt_init_ffff_noflush", 0xFFFF, 0x0000, False),
            ("ccitt_init_0000_noflush", 0x0000, 0x0000, False),
            ("ccitt_init_ffff_xorffff", 0xFFFF, 0xFFFF, True),
            ("ccitt_init_0000_xorffff", 0x0000, 0xFFFF, True),
        ]

    for sync_threshold in sync_thresholds:
        print(
            "=== sync_threshold=%d status_intervals=%s ==="
            % (sync_threshold, ",".join(str(v) for v in status_intervals))
        )
        analyze_capture(
            iq=iq,
            sample_rate=sample_rate,
            modulation=args.modulation,
            max_frames=args.max_frames,
            chunk_ms=args.chunk_ms,
            decode_offsets=decode_offsets,
            status_intervals=status_intervals,
            sync_threshold=sync_threshold,
            crc_variants=crc_variants,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
