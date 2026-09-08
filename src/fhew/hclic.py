"""
Utilities for parsing MHCLG H-CLIC 'detailed local authority' spreadsheets.

These are publication spreadsheets, not data files: two to six stacked header
rows with merged cells, footnote markers glued onto labels, region subtotals
interleaved with local authorities, and sheet names that change between
releases.  Everything needed to turn them into a tidy frame lives here so the
pipeline stages stay readable.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

# ONS codes for the geographies we keep. E06 unitary, E07 non-metropolitan
# district, E08 metropolitan district, E09 London borough.
LA_CODE = re.compile(r"^E0[6789]\d{6}$")

# Codes we explicitly drop: country (E92) and region (E12) subtotals.
AGG_CODE = re.compile(r"^E(92|12)\d{6}$")


@lru_cache(maxsize=1)
def _book(path: Path) -> pd.ExcelFile:
    """Open a workbook once and reuse it.

    Each of these files is ~1.3 MB of zipped XML with 30-odd sheets; the ODF
    reader parses the whole document per open, so re-opening it for every
    table dominates the runtime of the pipeline.
    """
    return pd.ExcelFile(path, engine="odf")


def release() -> None:
    """Drop the cached workbook.

    A parsed workbook holds the whole document tree in memory; without this
    the pipeline would carry every year's tree at once.
    """
    _book.cache_clear()


def resolve_sheet(path: Path, table: str) -> str:
    """Find the sheet holding `table` (e.g. 'A5R').

    Older releases suffix the period ('A5R_Apr_19-Mar_20'); newer ones do not.
    Matching on the token before the first underscore avoids 'A1' colliding
    with 'A10'.
    """
    sheets = _book(path).sheet_names
    for name in sheets:
        if name.split("_")[0].upper() == table.upper():
            return name
    raise KeyError(f"table {table!r} not found in {path.name}: {sheets}")


def _clean(text: object) -> str:
    """Normalise a header cell: strip footnote digits, collapse whitespace."""
    s = str(text)
    if s in {"nan", "NaT", "None"}:
        return ""
    s = s.replace("–", "-").replace("’", "'").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    # Trailing footnote markers, e.g. 'Total owed a relief duty1,2'
    s = re.sub(r"[\d,]+$", "", s).strip() if re.search(r"[a-z)]\s*[\d,]+$", s) else s
    return s


def _first_data_row(raw: pd.DataFrame) -> int:
    """Index of the first row that carries a geography code in column 0."""
    col0 = raw[0].astype(str)
    hits = col0[col0.str.match(r"^E\d{8}$", na=False)]
    if hits.empty:
        raise ValueError("no geography codes found in column 0")
    return int(hits.index[0])


def _drop_title_rows(header: pd.DataFrame) -> pd.DataFrame:
    """Remove the sheet title and any banner rows above the real header.

    A title occupies a single cell in column 0 ("Table A5R - Number of
    households owed a relief duty..."); left as-is, forward-filling would
    smear it across every column label.
    """
    keep = []
    for idx in header.index:
        populated = [c for c in header.columns if _clean(header.at[idx, c])]
        # A genuine header row spans several measure columns; a title does not.
        if len(populated) >= 2 and not (len(populated) == 1 and populated[0] == 0):
            keep.append(idx)
    return header.loc[keep] if keep else header


def _build_columns(header: pd.DataFrame) -> list[str]:
    """Collapse stacked, merged header rows into one label per column.

    Merged cells arrive as a value in the leftmost cell and blanks to its
    right, so each header row is forward-filled before the rows are joined.
    Repeated tokens are dropped so 'Domestic abuse > Domestic abuse' collapses
    to 'Domestic abuse'.
    """
    filled = header.map(_clean).replace("", pd.NA).ffill(axis=1)
    names: list[str] = []
    for col in filled.columns:
        parts: list[str] = []
        for value in filled[col]:
            token = "" if pd.isna(value) else str(value)
            if token and (not parts or parts[-1] != token):
                parts.append(token)
        names.append(" > ".join(parts))

    # Header rows are often blank above the geography columns and can repeat
    # across sub-breakdowns, so disambiguate before they become an index.
    seen: dict[str, int] = {}
    unique: list[str] = []
    for i, name in enumerate(names):
        label = name or f"col{i}"
        seen[label] = seen.get(label, 0) + 1
        unique.append(label if seen[label] == 1 else f"{label} #{seen[label]}")
    return unique


def read_table(path: Path, table: str) -> pd.DataFrame:
    """Read one H-CLIC table into a tidy local-authority frame.

    Returns columns `la_code`, `la_name`, then one column per measure, with
    region and England subtotal rows removed.
    """
    sheet = resolve_sheet(path, table)
    raw = _book(path).parse(sheet_name=sheet, header=None)

    start = _first_data_row(raw)
    columns = _build_columns(_drop_title_rows(raw.iloc[:start]))

    body = raw.iloc[start:].copy()
    body.columns = columns
    # Columns 0 and 1 are always the geography code and name, whatever the
    # header rows above them happen to contain.
    body.columns = ["la_code", "la_name", *columns[2:]]

    code = body["la_code"].astype(str)
    body = body[code.str.match(LA_CODE, na=False) & ~code.str.match(AGG_CODE, na=False)]

    # Measure columns are numeric; the publication uses '..', '-' and ':' for
    # suppressed or not-applicable values, which must become NaN not zero.
    for col in body.columns[2:]:
        body[col] = pd.to_numeric(body[col], errors="coerce")

    body["la_name"] = body["la_name"].astype(str).str.strip()
    return body.reset_index(drop=True)


def pick(frame: pd.DataFrame, *needles: str) -> str:
    """Return the single column whose label contains all `needles`.

    Raises if the match is ambiguous or empty, so a layout change between
    releases fails loudly instead of silently selecting the wrong measure.
    """
    lowered = [n.lower() for n in needles]
    hits = [c for c in frame.columns if all(n in c.lower() for n in lowered)]
    if not hits:
        raise KeyError(f"no column matching {needles}")
    if len(hits) > 1:
        # Prefer the shortest label: deeper headers are more specific
        # sub-breakdowns of the same measure.
        hits.sort(key=len)
        if len(hits[0]) == len(hits[1]):
            raise KeyError(f"ambiguous match for {needles}: {hits[:4]}")
    return hits[0]


def counts_only(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the count column of each measure and drop the percentage twin.

    Every measure in these tables is published as a count immediately followed
    by its share of the row total (and sometimes a trailing blank), all sharing
    one merged header.  `_build_columns` gives those siblings the same base
    label with a ' #n' suffix, so the first member of each group is the count.
    """
    keep = ["la_code", "la_name"]
    seen: set[str] = set()
    for col in frame.columns[2:]:
        base = col.split(" #")[0]
        if base not in seen:
            seen.add(base)
            keep.append(col)
    return frame[keep]


def sum_matching(frame: pd.DataFrame, include: str, exclude: str | None = None) -> pd.Series:
    """Row-sum every count column whose label contains `include`.

    Used for roll-ups such as 'all household types with dependent children',
    where the published breakdown splits by sex of the main applicant.
    """
    cols = [
        c for c in frame.columns[2:]
        if include.lower() in c.lower()
        and (exclude is None or exclude.lower() not in c.lower())
    ]
    if not cols:
        raise KeyError(f"no columns contain {include!r}")
    return frame[cols].sum(axis=1, min_count=1)
