#!/usr/bin/env python3
"""Import WaveCap webstream transcripts + recordings for offline analysis."""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path
from datetime import datetime
import shutil

DEFAULT_DB = Path(__file__).resolve().parents[2] / "../WaveCap/state/runtime.sqlite"
DEFAULT_RECORDINGS = Path(__file__).resolve().parents[2] / "../WaveCap/state/recordings"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "recordings/voice_captures/wavecap_import"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import WaveCap transcripts and WAVs")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to WaveCap runtime.sqlite")
    parser.add_argument("--recordings", type=Path, default=DEFAULT_RECORDINGS, help="WaveCap recordings directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--limit", type=int, default=50, help="Max transcripts to import")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.db.exists():
        raise SystemExit(f"DB not found: {args.db}")
    if not args.recordings.exists():
        raise SystemExit(f"Recordings dir not found: {args.recordings}")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.csv"

    conn = sqlite3.connect(str(args.db))
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, streamId, timestamp, recordingUrl, duration, confidence, text "
        "FROM transcriptions WHERE recordingUrl IS NOT NULL "
        "ORDER BY timestamp DESC LIMIT ?",
        (args.limit,),
    )
    rows = cursor.fetchall()
    conn.close()

    imported = 0
    with manifest_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "id",
            "stream_id",
            "timestamp",
            "duration",
            "confidence",
            "recording",
            "text",
        ])
        for row in rows:
            rec_id, stream_id, ts, recording_url, duration, confidence, text = row
            if not recording_url:
                continue
            recording_name = recording_url.lstrip("/").replace("recordings/", "")
            src = args.recordings / recording_name
            if not src.exists():
                continue
            timestamp = ts if isinstance(ts, str) else datetime.fromisoformat(ts).isoformat()
            dest_name = f"{timestamp.replace(':', '').replace(' ', '_')}_{recording_name}"
            dest = args.output / dest_name
            if not dest.exists():
                shutil.copy2(src, dest)
            writer.writerow([rec_id, stream_id, timestamp, duration, confidence, dest.name, text])
            imported += 1

    print(f"Imported {imported} recordings to {args.output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
