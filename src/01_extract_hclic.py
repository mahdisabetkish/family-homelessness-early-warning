"""Stage 01 - Extract every H-CLIC count column into one tidy file per year.

Parsing these workbooks is the slow part of the pipeline: each is around
1.3 MB of zipped XML that the ODF reader loads in full. Doing it once and
caching the result means the measure-selection logic downstream can be
changed and rerun in seconds rather than minutes, which matters because the
published column wording drifts between releases.

Output: data/interim/hclic_<year>.parquet, one row per local authority, one
column per published measure, with the original header text preserved as the
column name so selection stays auditable.
"""
from __future__ import annotations

import sys

import pandas as pd

from fhew import config, hclic


def extract_year(year: str) -> pd.DataFrame:
    """Read every configured table for one year into a single wide frame."""
    path = config.HCLIC_RAW / f"hclic_{year}.ods"
    merged: pd.DataFrame | None = None

    for table in config.TABLES_READ:
        frame = hclic.counts_only(hclic.read_table(path, table))
        # Prefix with the table id so identically worded measures in different
        # tables (for example 'Total owed a relief duty') stay distinct.
        frame = frame.rename(columns={
            c: f"{table}::{c}" for c in frame.columns if c not in {"la_code", "la_name"}
        })
        merged = frame if merged is None else merged.merge(
            frame.drop(columns="la_name"), on="la_code", how="outer")

    assert merged is not None
    hclic.release()
    # Rebuild in one pass: the repeated merges leave a heavily fragmented frame.
    merged = merged.copy()
    merged.insert(2, "financial_year", year)
    return merged


def main() -> int:
    config.INTERIM.mkdir(parents=True, exist_ok=True)
    for year in config.YEARS:
        out = config.INTERIM / f"hclic_{year}.parquet"
        if out.exists():
            print(f"  cached  {out.name}")
            continue
        frame = extract_year(year)
        frame.to_parquet(out, index=False)
        print(f"  wrote   {out.name}  {frame.shape[0]} authorities, "
              f"{frame.shape[1] - 3} measures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
