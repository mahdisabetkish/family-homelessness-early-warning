"""The spreadsheet parser.

These publications are the most fragile input in the project, so the parser
is tested against a synthetic sheet that reproduces each awkward feature of
the real ones: a banner title, stacked merged headers, a percentage column
sharing a header with its count, and region subtotals above the authorities.
"""
from __future__ import annotations

import pandas as pd
import pytest

from fhew import hclic


@pytest.fixture
def raw_sheet() -> pd.DataFrame:
    """A miniature of the real layout, with the same traps."""
    return pd.DataFrame([
        # r0: banner title, one populated cell
        ["Table A5R - Number of households owed a relief duty", None, None, None, None],
        # r1: merged group header, blanks to the right of each group
        [None, None, "Total owed a relief duty", "Single parent with children", None],
        # r2: sub-header
        [None, None, None, "Male", "Female"],
        # data: England and a region above the authorities
        ["E92000001", "ENGLAND", 1000, 200, 300],
        ["E12000006", "East of England", 400, 80, 120],
        ["E07000071", "Colchester", 100, 20, 30],
        ["E09000001", "City of London", 5, 1, 2],
    ])


def test_drops_banner_and_aggregate_rows(raw_sheet, monkeypatch):
    monkeypatch.setattr(hclic, "_book", lambda path: _FakeBook(raw_sheet))
    monkeypatch.setattr(hclic, "resolve_sheet", lambda path, table: "A5R")

    frame = hclic.read_table("ignored.ods", "A5R")

    assert list(frame["la_code"]) == ["E07000071", "E09000001"]
    assert "ENGLAND" not in set(frame["la_name"])
    assert "East of England" not in set(frame["la_name"])


def test_column_labels_collapse_merged_headers(raw_sheet, monkeypatch):
    monkeypatch.setattr(hclic, "_book", lambda path: _FakeBook(raw_sheet))
    monkeypatch.setattr(hclic, "resolve_sheet", lambda path, table: "A5R")

    frame = hclic.read_table("ignored.ods", "A5R")
    labels = list(frame.columns)

    assert labels[0] == "la_code" and labels[1] == "la_name"
    # The banner must not appear in any measure label.
    assert not any("Table A5R" in label for label in labels)
    # The merged parent is joined to its child, not lost.
    assert any("Single parent with children" in label and "Male" in label
               for label in labels)


def test_pick_is_unambiguous(raw_sheet, monkeypatch):
    monkeypatch.setattr(hclic, "_book", lambda path: _FakeBook(raw_sheet))
    monkeypatch.setattr(hclic, "resolve_sheet", lambda path, table: "A5R")
    frame = hclic.read_table("ignored.ods", "A5R")

    column = hclic.pick(frame, "total owed a relief duty")
    assert frame.loc[frame["la_code"] == "E07000071", column].iloc[0] == 100

    with pytest.raises(KeyError):
        hclic.pick(frame, "a measure that does not exist")


def test_sum_matching_rolls_up_split_categories(raw_sheet, monkeypatch):
    monkeypatch.setattr(hclic, "_book", lambda path: _FakeBook(raw_sheet))
    monkeypatch.setattr(hclic, "resolve_sheet", lambda path, table: "A5R")
    frame = hclic.read_table("ignored.ods", "A5R")

    totals = hclic.sum_matching(frame, "single parent with children")
    # Colchester: 20 male + 30 female.
    assert totals.iloc[0] == 50


def test_counts_only_drops_the_percentage_twin():
    frame = pd.DataFrame({
        "la_code": ["E07000071"],
        "la_name": ["Colchester"],
        "Total owed a relief duty": [100],
        "Total owed a relief duty #2": [1.0],
        "Single parent": [40],
        "Single parent #2": [0.4],
    })
    counts = hclic.counts_only(frame)
    assert list(counts.columns) == [
        "la_code", "la_name", "Total owed a relief duty", "Single parent"]


class _FakeBook:
    """Stands in for a pandas ExcelFile without touching the filesystem."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame
        self.sheet_names = ["A5R"]

    def parse(self, sheet_name: str, header=None) -> pd.DataFrame:
        return self._frame.copy()
