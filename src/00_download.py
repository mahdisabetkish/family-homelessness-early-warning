"""Stage 00 - Download the raw open data.

Files are cached: a rerun that finds a non-empty file leaves it alone, so the
pipeline can be rerun without going back to gov.uk. The registry of sources
lives in `fhew.sources` so it can be checked against `fhew.config.YEARS`
without a network call.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import requests

from fhew import config, sources

USER_AGENT = "family-homelessness-early-warning/1.0"


def fetch(url: str, dest: Path) -> None:
    """Download `url` to `dest`, skipping if already present."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached  {dest.name} ({dest.stat().st_size:,} bytes)")
        return
    print(f"  GET     {dest.name} ...", end="", flush=True)
    response = requests.get(url, timeout=180, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    dest.write_bytes(response.content)
    digest = hashlib.sha256(response.content).hexdigest()[:12]
    print(f" ok ({len(response.content):,} bytes, sha256:{digest})")


def main() -> int:
    config.HCLIC_RAW.mkdir(parents=True, exist_ok=True)
    config.IOD_RAW.mkdir(parents=True, exist_ok=True)

    print("H-CLIC statutory homelessness, detailed LA tables")
    for year in config.YEARS:
        url = sources.HCLIC[year]
        fetch(url, config.HCLIC_RAW / f"hclic_{year}{Path(url).suffix}")

    print("\nEnglish Indices of Deprivation 2019")
    for name, url in sources.IOD.items():
        fetch(url, config.IOD_RAW / name)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
