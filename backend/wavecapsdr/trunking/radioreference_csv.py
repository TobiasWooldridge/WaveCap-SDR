"""RadioReference CSV file parsers for control channels and sites.

Parses RadioReference export files (trs_sites_*.csv) to extract control channel
frequencies. Only frequencies ending with 'c' suffix are control channels.
"""

from __future__ import annotations

import csv
import logging
import os
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SiteInfo:
    """Information about a radio site from RadioReference."""

    rfss: int
    site_dec: int
    site_hex: str
    site_nac: str
    description: str
    county: str
    lat: float | None
    lon: float | None
    range_miles: float | None
    frequencies: list[float]  # All frequencies in Hz
    control_channels: list[float]  # Only control channel frequencies in Hz


def parse_frequency_with_control_marker(freq_str: str) -> tuple[float, bool]:
    """Parse a frequency string, returning (frequency_hz, is_control_channel).

    Args:
        freq_str: Frequency string like "414.762500c" or "412.612500"

    Returns:
        Tuple of (frequency in Hz, whether it's a control channel)
    """
    freq_str = freq_str.strip()
    is_control = freq_str.endswith("c")
    if is_control:
        freq_str = freq_str[:-1]

    try:
        freq_mhz = float(freq_str)
        return freq_mhz * 1_000_000, is_control
    except ValueError:
        raise ValueError(f"Invalid frequency format: {freq_str}")


def parse_frequencies_column(freqs_str: str) -> tuple[list[float], list[float]]:
    """Parse the frequencies column from RadioReference CSV.

    Args:
        freqs_str: Comma-separated frequencies like "414.762500c,412.612500,413.312500"

    Returns:
        Tuple of (all_frequencies_hz, control_channel_frequencies_hz)
    """
    all_freqs: list[float] = []
    control_freqs: list[float] = []

    for part in freqs_str.split(","):
        part = part.strip()
        if not part or not re.match(r"^\d", part):
            continue

        try:
            freq_hz, is_control = parse_frequency_with_control_marker(part)
            all_freqs.append(freq_hz)
            if is_control:
                control_freqs.append(freq_hz)
        except ValueError:
            continue

    return all_freqs, control_freqs


def load_sites_csv(csv_path: str, config_dir: str | None = None) -> list[SiteInfo]:
    """Load site information from RadioReference trs_sites CSV export.

    RadioReference exports have each frequency as a separate column after the
    "Frequencies" header. This parser handles both formats:
    1. Single "Frequencies" column with comma-separated values
    2. Multiple columns after "Frequencies" (one per frequency)

    Args:
        csv_path: Path to CSV file (absolute or relative to config_dir, supports ~)
        config_dir: Base directory for resolving relative paths

    Returns:
        List of SiteInfo objects
    """
    # Expand ~ to home directory
    csv_path = os.path.expanduser(csv_path)

    # Resolve relative path
    if config_dir and not os.path.isabs(csv_path):
        csv_path = os.path.join(config_dir, csv_path)

    sites: list[SiteInfo] = []

    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)

            # Find column indices
            col_map = {h.strip(): i for i, h in enumerate(headers)}
            freq_start_idx = col_map.get("Frequencies", len(headers))

            for row in reader:
                try:
                    # Get frequencies from all columns starting at freq_start_idx
                    freq_parts = row[freq_start_idx:] if freq_start_idx < len(row) else []

                    all_freqs: list[float] = []
                    control_freqs: list[float] = []

                    for freq_str in freq_parts:
                        freq_str = freq_str.strip()
                        if not freq_str or not re.match(r"^\d", freq_str):
                            continue
                        try:
                            freq_hz, is_control = parse_frequency_with_control_marker(freq_str)
                            all_freqs.append(freq_hz)
                            if is_control:
                                control_freqs.append(freq_hz)
                        except ValueError:
                            continue

                    # Parse optional numeric fields
                    lat = None
                    lon = None
                    range_miles = None

                    def get_col(name: str, default: str = "") -> str:
                        idx = col_map.get(name)
                        if idx is not None and idx < len(row):
                            return row[idx].strip()
                        return default

                    lat_str = get_col("Lat")
                    if lat_str:
                        try:
                            lat = float(lat_str)
                        except ValueError:
                            pass

                    lon_str = get_col("Lon")
                    if lon_str:
                        try:
                            lon = float(lon_str)
                        except ValueError:
                            pass

                    range_str = get_col("Range")
                    if range_str:
                        try:
                            range_miles = float(range_str)
                        except ValueError:
                            pass

                    site_dec_str = get_col("Site Dec", "0")
                    try:
                        site_dec = int(site_dec_str)
                    except ValueError:
                        site_dec = 0

                    rfss_str = get_col("RFSS", "1")
                    try:
                        rfss = int(rfss_str)
                    except ValueError:
                        rfss = 1

                    sites.append(
                        SiteInfo(
                            rfss=rfss,
                            site_dec=site_dec,
                            site_hex=get_col("Site Hex"),
                            site_nac=get_col("Site NAC"),
                            description=get_col("Description"),
                            county=get_col("County Name"),
                            lat=lat,
                            lon=lon,
                            range_miles=range_miles,
                            frequencies=all_freqs,
                            control_channels=control_freqs,
                        )
                    )
                except (ValueError, KeyError) as e:
                    logger.debug(f"Skipping invalid row: {e}")
                    continue

    except FileNotFoundError:
        logger.warning(f"Sites CSV file not found: {csv_path}")
    except Exception as e:
        logger.error(f"Error loading sites from {csv_path}: {e}")

    return sites


def load_control_channels_csv(
    csv_path: str, config_dir: str | None = None
) -> list[tuple[float, str]]:
    """Load control channel frequencies from RadioReference trs_sites CSV.

    Extracts unique control channel frequencies (those ending with 'c' suffix)
    from all sites. Returns frequencies with site name for reference.

    Args:
        csv_path: Path to CSV file (absolute or relative to config_dir, supports ~)
        config_dir: Base directory for resolving relative paths

    Returns:
        List of (frequency_hz, site_name) tuples, deduplicated by frequency
    """
    sites = load_sites_csv(csv_path, config_dir)

    # Collect control channels with their site names
    # Use dict to deduplicate by frequency (keep first site name)
    freq_to_site: dict[float, str] = {}

    for site in sites:
        for freq_hz in site.control_channels:
            if freq_hz not in freq_to_site:
                freq_to_site[freq_hz] = site.description

    # Sort by frequency
    result = sorted(freq_to_site.items(), key=lambda x: x[0])

    logger.info(
        f"Loaded {len(result)} unique control channels from {len(sites)} sites in {csv_path}"
    )

    return result
