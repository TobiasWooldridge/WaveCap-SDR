#!/usr/bin/env python3
"""Modular P25 decoding pipeline - run individual stages for comparison.

This script allows running each P25 decoding stage independently, writing
intermediate outputs to files for comparison with SDRTrunk or other decoders.

Stages:
1. fm_demod - FM discriminator to soft symbols
2. slice - Soft symbols to hard dibits
3. sync - Find frame sync, output aligned frames
4. nid - Decode NID (BCH error correction)
5. tsbk - Decode TSBK (deinterleave, trellis, CRC)

Usage:
    # Run full pipeline with stage output
    python scripts/p25_pipeline_stages.py --input capture.wav --output-dir /tmp/p25_stages/

    # Run single stage
    python scripts/p25_pipeline_stages.py --input capture.wav --stage fm_demod --output soft_symbols.npy

    # Compare two pipelines
    python scripts/p25_pipeline_stages.py --compare-dirs /tmp/wavecap/ /tmp/sdrtrunk/
"""

from __future__ import annotations

import argparse
import json
import logging
import struct
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from wavecapsdr.decoders.p25 import (
    C4FMDemodulator,
    P25FrameSync,
    P25FrameType,
)
from wavecapsdr.decoders.p25_frames import (
    decode_nid,
    dibits_to_bits,
    deinterleave_data,
)
from wavecapsdr.dsp.fec.trellis import trellis_decode

logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    """Result from a pipeline stage."""
    stage: str
    data: Any
    metadata: dict


def load_iq_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load IQ data from stereo WAV file."""
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

    if n_channels != 2:
        raise ValueError(f"Expected stereo IQ WAV, got {n_channels} channels")

    if sample_width == 2:
        samples = np.frombuffer(raw_data, dtype=np.int16).reshape(-1, n_channels)
        samples_f = samples.astype(np.float32) / 32768.0
    elif sample_width == 3:
        # 24-bit samples
        data_bytes = np.frombuffer(raw_data, dtype=np.uint8).reshape(-1, n_channels, 3)
        samples = np.zeros((n_frames, n_channels), dtype=np.int32)
        for ch in range(n_channels):
            b0 = data_bytes[:, ch, 0].astype(np.int32)
            b1 = data_bytes[:, ch, 1].astype(np.int32)
            b2 = data_bytes[:, ch, 2].astype(np.int32)
            raw24 = b0 | (b1 << 8) | (b2 << 16)
            samples[:, ch] = np.where(raw24 >= 0x800000, raw24 - 0x1000000, raw24)
        samples_f = samples.astype(np.float32) / 8388608.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    iq = samples_f[:, 0] + 1j * samples_f[:, 1]
    return iq, sample_rate


def load_iq_rawiq(path: Path, sample_rate: int | None) -> tuple[np.ndarray, int]:
    """Load IQ data from raw int16 interleaved file."""
    if sample_rate is None:
        # Try to read from metadata file
        meta_path = path.with_suffix(".meta")
        if meta_path.exists():
            for line in meta_path.read_text().split("\n"):
                if line.startswith("sample_rate="):
                    sample_rate = int(line.split("=")[1])
                    break
        if sample_rate is None:
            raise ValueError("rawiq requires --sample-rate or a .meta file")

    raw = path.read_bytes()
    data = np.frombuffer(raw, dtype=np.int16)
    if len(data) % 2 != 0:
        data = data[:-1]

    i = data[0::2].astype(np.float32) / 32768.0
    q = data[1::2].astype(np.float32) / 32768.0
    iq = i + 1j * q
    return iq, sample_rate


class P25Pipeline:
    """Modular P25 decoding pipeline with stage outputs."""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.symbol_rate = 4800
        self.demod = C4FMDemodulator(sample_rate=sample_rate, symbol_rate=self.symbol_rate)
        self.frame_sync = P25FrameSync()

    def stage_fm_demod(self, iq: np.ndarray) -> StageResult:
        """Stage 1: FM demodulation to soft symbols.

        Returns soft symbol values (float) in range approximately -3 to +3.
        """
        # Run in chunks to simulate real-time
        chunk_size = self.sample_rate // 10  # 100ms chunks
        all_soft_symbols = []

        for start in range(0, len(iq), chunk_size):
            chunk = iq[start:start + chunk_size]
            if len(chunk) < 100:
                break

            # Get dibits from demodulator
            dibits = self.demod.demodulate(chunk.astype(np.complex64, copy=False))

            # Convert dibits back to approximate soft symbols for analysis
            # This is an approximation - ideally we'd capture the actual soft symbols
            dibit_to_sym = {0: 1.0, 1: 3.0, 2: -1.0, 3: -3.0}
            soft = np.array([dibit_to_sym[d] for d in dibits], dtype=np.float32)
            all_soft_symbols.extend(soft.tolist())

        soft_symbols = np.array(all_soft_symbols, dtype=np.float32)

        # Calculate statistics
        stats = {
            "count": len(soft_symbols),
            "mean": float(np.mean(soft_symbols)),
            "std": float(np.std(soft_symbols)),
            "min": float(np.min(soft_symbols)) if len(soft_symbols) else 0,
            "max": float(np.max(soft_symbols)) if len(soft_symbols) else 0,
        }

        # Count occurrences
        unique, counts = np.unique(soft_symbols, return_counts=True)
        stats["distribution"] = {float(k): int(v) for k, v in zip(unique, counts)}

        return StageResult(stage="fm_demod", data=soft_symbols, metadata=stats)

    def stage_slice(self, soft_symbols: np.ndarray) -> StageResult:
        """Stage 2: Slice soft symbols to hard dibits.

        Symbol levels: +3 -> 1, +1 -> 0, -1 -> 2, -3 -> 3
        """
        # Slice at thresholds: -2, 0, +2
        dibits = np.zeros(len(soft_symbols), dtype=np.uint8)
        dibits[soft_symbols >= 2.0] = 1   # +3 -> 01
        dibits[(soft_symbols >= 0.0) & (soft_symbols < 2.0)] = 0   # +1 -> 00
        dibits[(soft_symbols >= -2.0) & (soft_symbols < 0.0)] = 2  # -1 -> 10
        dibits[soft_symbols < -2.0] = 3   # -3 -> 11

        stats = {
            "count": len(dibits),
            "dibit_0": int(np.sum(dibits == 0)),
            "dibit_1": int(np.sum(dibits == 1)),
            "dibit_2": int(np.sum(dibits == 2)),
            "dibit_3": int(np.sum(dibits == 3)),
        }

        return StageResult(stage="slice", data=dibits, metadata=stats)

    def stage_sync(self, dibits: np.ndarray, max_frames: int = 100) -> StageResult:
        """Stage 3: Find frame sync and extract aligned frames."""
        frames = []
        buffer = dibits.tolist()
        frame_sync = P25FrameSync()

        while len(frames) < max_frames:
            if len(buffer) < 360:
                break

            buf_arr = np.array(buffer, dtype=np.uint8)
            sync_pos, frame_type, nac, duid = frame_sync.find_sync(buf_arr)

            if sync_pos is None:
                if len(buffer) > 720:
                    buffer = buffer[-360:]
                break

            if sync_pos + 360 > len(buffer):
                break

            frame_dibits = buf_arr[sync_pos:sync_pos + 360]
            frames.append({
                "index": len(frames),
                "sync_pos": sync_pos,
                "frame_type": frame_type.name if frame_type else "UNKNOWN",
                "nac": nac,
                "duid": duid,
                "dibits": frame_dibits.tolist(),
            })
            buffer = buffer[sync_pos + 360:]

        stats = {
            "frames_found": len(frames),
            "frame_types": {},
        }
        for f in frames:
            ft = f["frame_type"]
            stats["frame_types"][ft] = stats["frame_types"].get(ft, 0) + 1

        return StageResult(stage="sync", data=frames, metadata=stats)

    def stage_nid(self, frames: list[dict]) -> StageResult:
        """Stage 4: Decode NID from each frame."""
        results = []

        for frame in frames:
            dibits = np.array(frame["dibits"], dtype=np.uint8)
            nid_dibits = dibits[24:57]  # NID location in P25 frame

            nid_result = decode_nid(nid_dibits, skip_status_at_10=True)

            results.append({
                "frame_index": frame["index"],
                "frame_type": frame["frame_type"],
                "nid_success": nid_result is not None,
                "nac": nid_result.nac if nid_result else None,
                "duid": nid_result.duid if nid_result else None,
                "bch_errors": nid_result.errors if nid_result else None,
            })

        # Statistics
        success_count = sum(1 for r in results if r["nid_success"])
        nac_values = [r["nac"] for r in results if r["nac"] is not None]

        stats = {
            "total_frames": len(results),
            "nid_success": success_count,
            "nid_fail": len(results) - success_count,
            "nac_values": list(set(nac_values)),
            "most_common_nac": max(set(nac_values), key=nac_values.count) if nac_values else None,
        }

        return StageResult(stage="nid", data=results, metadata=stats)

    def stage_tsbk(self, frames: list[dict], nid_results: list[dict]) -> StageResult:
        """Stage 5: Decode TSBK from TSDU frames."""
        results = []

        for frame, nid in zip(frames, nid_results):
            if frame["frame_type"] != "TSDU":
                continue

            dibits = np.array(frame["dibits"], dtype=np.uint8)
            tsbk_raw = dibits[57:]  # TSBK starts after NID

            # Process 3 TSBK blocks per TSDU
            for block_idx in range(3):
                start = block_idx * 98
                if start + 98 > len(tsbk_raw):
                    break

                block_dibits = tsbk_raw[start:start + 98]

                # Strip status symbols (every 36th dibit)
                cleaned = []
                for i, d in enumerate(block_dibits):
                    frame_pos = 57 + start + i
                    if (frame_pos + 1) % 36 != 0:
                        cleaned.append(d)
                cleaned = np.array(cleaned, dtype=np.uint8)

                if len(cleaned) < 96:
                    continue

                # Convert to bits and deinterleave
                bits = dibits_to_bits(cleaned[:98] if len(cleaned) >= 98 else cleaned)
                if len(bits) < 196:
                    continue
                deint_bits = deinterleave_data(bits)

                # Convert back to dibits for trellis
                trellis_dibits = np.zeros(len(deint_bits) // 2, dtype=np.uint8)
                for i in range(len(trellis_dibits)):
                    trellis_dibits[i] = (deint_bits[i * 2] << 1) | deint_bits[i * 2 + 1]

                # Trellis decode
                decoded, trellis_err = trellis_decode(trellis_dibits)

                if decoded is None or len(decoded) < 48:
                    results.append({
                        "frame_index": frame["index"],
                        "block_index": block_idx,
                        "trellis_success": False,
                        "trellis_errors": trellis_err,
                    })
                    continue

                # CRC check
                payload_bits = dibits_to_bits(decoded[:48])
                crc_ok, calc_crc, recv_crc = self._crc16_check(payload_bits)

                results.append({
                    "frame_index": frame["index"],
                    "block_index": block_idx,
                    "trellis_success": True,
                    "trellis_errors": trellis_err,
                    "crc_ok": crc_ok,
                    "crc_calculated": calc_crc,
                    "crc_received": recv_crc,
                    "opcode": int(payload_bits[2]) << 5 | int(payload_bits[3]) << 4 | int(payload_bits[4]) << 3 | int(payload_bits[5]) << 2 | int(payload_bits[6]) << 1 | int(payload_bits[7]) if len(payload_bits) >= 8 else None,
                })

        # Statistics
        crc_pass = sum(1 for r in results if r.get("crc_ok"))
        trellis_pass = sum(1 for r in results if r.get("trellis_success"))

        stats = {
            "total_blocks": len(results),
            "trellis_success": trellis_pass,
            "crc_pass": crc_pass,
            "crc_fail": trellis_pass - crc_pass,
        }

        return StageResult(stage="tsbk", data=results, metadata=stats)

    def _crc16_check(self, bits: np.ndarray) -> tuple[bool, int, int]:
        """CRC-16 CCITT check."""
        if len(bits) < 96:
            return False, 0, 0

        poly = 0x1021
        crc = 0xFFFF  # Init value

        for i in range(80):
            bit = int(bits[i])
            msb = (crc >> 15) & 1
            crc = ((crc << 1) | bit) & 0xFFFF
            if msb:
                crc ^= poly

        # Flush 16 zeros
        for _ in range(16):
            msb = (crc >> 15) & 1
            crc = (crc << 1) & 0xFFFF
            if msb:
                crc ^= poly

        # Extract received CRC
        recv_crc = 0
        for i in range(80, 96):
            recv_crc = (recv_crc << 1) | int(bits[i])

        return crc == recv_crc, crc, recv_crc


def run_pipeline(
    iq: np.ndarray,
    sample_rate: int,
    output_dir: Path,
    max_frames: int = 100,
) -> dict[str, StageResult]:
    """Run full pipeline with stage outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = P25Pipeline(sample_rate=sample_rate)
    results = {}

    # Stage 1: FM Demodulation
    logger.info("Stage 1: FM demodulation...")
    result = pipeline.stage_fm_demod(iq)
    results["fm_demod"] = result
    np.save(output_dir / "stage1_soft_symbols.npy", result.data)
    (output_dir / "stage1_soft_symbols.json").write_text(json.dumps(result.metadata, indent=2))
    logger.info(f"  Symbols: {result.metadata['count']}, mean={result.metadata['mean']:.3f}, std={result.metadata['std']:.3f}")

    # Stage 2: Symbol slicing
    logger.info("Stage 2: Symbol slicing...")
    result = pipeline.stage_slice(result.data)
    results["slice"] = result
    np.save(output_dir / "stage2_dibits.npy", result.data)
    (output_dir / "stage2_dibits.json").write_text(json.dumps(result.metadata, indent=2))
    logger.info(f"  Dibits: {result.metadata}")

    # Stage 3: Frame sync
    logger.info("Stage 3: Frame sync...")
    result = pipeline.stage_sync(result.data, max_frames=max_frames)
    results["sync"] = result
    (output_dir / "stage3_frames.json").write_text(json.dumps(result.data, indent=2))
    (output_dir / "stage3_frames_meta.json").write_text(json.dumps(result.metadata, indent=2))
    logger.info(f"  Frames: {result.metadata}")

    # Stage 4: NID decode
    logger.info("Stage 4: NID decode...")
    result = pipeline.stage_nid(result.data)
    results["nid"] = result
    (output_dir / "stage4_nid.json").write_text(json.dumps(result.data, indent=2))
    (output_dir / "stage4_nid_meta.json").write_text(json.dumps(result.metadata, indent=2))
    logger.info(f"  NID: {result.metadata}")

    # Stage 5: TSBK decode
    logger.info("Stage 5: TSBK decode...")
    result = pipeline.stage_tsbk(results["sync"].data, result.data)
    results["tsbk"] = result
    (output_dir / "stage5_tsbk.json").write_text(json.dumps(result.data, indent=2))
    (output_dir / "stage5_tsbk_meta.json").write_text(json.dumps(result.metadata, indent=2))
    logger.info(f"  TSBK: {result.metadata}")

    # Summary
    summary = {
        "stages": {name: r.metadata for name, r in results.items()},
    }
    (output_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2))

    return results


def compare_directories(dir1: Path, dir2: Path) -> None:
    """Compare pipeline outputs from two directories."""
    print(f"\nComparing pipeline outputs:")
    print(f"  Dir 1: {dir1}")
    print(f"  Dir 2: {dir2}")
    print()

    # Compare each stage
    stages = ["stage1_soft_symbols", "stage2_dibits", "stage3_frames_meta", "stage4_nid_meta", "stage5_tsbk_meta"]

    for stage in stages:
        json_file = f"{stage}.json"
        path1 = dir1 / json_file
        path2 = dir2 / json_file

        if not path1.exists() or not path2.exists():
            print(f"  {stage}: MISSING")
            continue

        data1 = json.loads(path1.read_text())
        data2 = json.loads(path2.read_text())

        print(f"  {stage}:")
        print(f"    Dir 1: {data1}")
        print(f"    Dir 2: {data2}")

        if data1 == data2:
            print(f"    Status: MATCH")
        else:
            print(f"    Status: DIFFER")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="P25 pipeline stages for comparison")
    parser.add_argument("--input", type=Path, help="Input IQ file (.wav or .rawiq)")
    parser.add_argument("--sample-rate", type=int, default=None, help="Sample rate for rawiq")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for stage files")
    parser.add_argument("--max-frames", type=int, default=100, help="Max frames to process")
    parser.add_argument("--compare-dirs", nargs=2, type=Path, help="Compare two output directories")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.compare_dirs:
        compare_directories(args.compare_dirs[0], args.compare_dirs[1])
        return 0

    if not args.input:
        parser.error("--input is required")

    # Load IQ data
    if args.input.suffix.lower() == ".wav":
        iq, sample_rate = load_iq_wav(args.input)
    elif args.input.suffix.lower() == ".rawiq":
        iq, sample_rate = load_iq_rawiq(args.input, args.sample_rate)
    else:
        parser.error(f"Unsupported input format: {args.input.suffix}")

    logger.info(f"Loaded {args.input}: {len(iq)} samples at {sample_rate} Hz")

    # Set output directory
    output_dir = args.output_dir or Path(f"/tmp/p25_stages/{args.input.stem}")

    # Run pipeline
    results = run_pipeline(iq, sample_rate, output_dir, max_frames=args.max_frames)

    print(f"\nPipeline complete. Output in: {output_dir}")
    print("\nSummary:")
    for stage, result in results.items():
        print(f"  {stage}: {result.metadata}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
