from __future__ import annotations

import argparse
import sys
from pathlib import Path

from wavecapsdr.config import default_config_path, load_config
from wavecapsdr.trunking.config import (
    RadioReferenceTalkgroupsConfig,
    load_talkgroups_radioreference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync talkgroups from RadioReference and write to a YAML cache file",
    )
    parser.add_argument(
        "--config",
        default=default_config_path(),
        help="Path to wavecapsdr.yaml (default: auto-detect)",
    )
    parser.add_argument(
        "--system",
        required=True,
        help="Trunking system id from config (e.g., sa_grn_2)",
    )
    parser.add_argument(
        "--rr-system-id",
        type=int,
        default=None,
        help="RadioReference system ID (sid) if not configured in talkgroups_rr",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override cache file path for the YAML talkgroups output",
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Skip fetching from RadioReference and read from cache file if present",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return 1

    cfg = load_config(str(config_path))
    sys_data = cfg.trunking_systems.get(args.system)
    if not isinstance(sys_data, dict):
        print(f"System '{args.system}' not found in trunking.systems", file=sys.stderr)
        return 1

    rr_raw = sys_data.get("talkgroups_rr") if isinstance(sys_data.get("talkgroups_rr"), dict) else None
    rr_system_id = None
    if rr_raw:
        rr_system_id = rr_raw.get("system_id") or rr_raw.get("sid")
    if rr_system_id is None:
        rr_system_id = args.rr_system_id
    if rr_system_id is None:
        print("RadioReference system_id is required (talkgroups_rr.system_id or --rr-system-id)", file=sys.stderr)
        return 1

    cache_file = args.output
    if cache_file is None and rr_raw:
        cache_file = rr_raw.get("cache_file")
    if cache_file is None:
        cache_file = f"talkgroups/rr_{int(rr_system_id)}.yaml"

    settings = RadioReferenceTalkgroupsConfig(
        system_id=int(rr_system_id),
        category_id=rr_raw.get("category_id") if rr_raw else None,
        tag_id=rr_raw.get("tag_id") if rr_raw else None,
        tgid=rr_raw.get("tgid") if rr_raw else None,
        cache_file=cache_file,
        enabled=True,
        refresh=not args.no_refresh,
    )

    talkgroups = load_talkgroups_radioreference(
        settings,
        cfg.radioreference,
        config_dir=str(config_path.parent),
    )

    if not talkgroups:
        print("No talkgroups loaded. Check credentials or filters.")
        return 2

    resolved_cache = Path(cache_file)
    if not resolved_cache.is_absolute():
        resolved_cache = config_path.parent / resolved_cache

    print(f"Fetched {len(talkgroups)} talkgroups from RadioReference")
    print(f"Wrote cache: {resolved_cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
