"""Measure selection, boundary maps and rate conversion.

The panel tests elsewhere check the artefact once it is built. These check
the rules that build it, so a change to a header pattern or a boundary map is
caught without a two-minute parse of seven spreadsheets.
"""
from __future__ import annotations

import pandas as pd
import pytest

from fhew import panel


@pytest.fixture
def headers() -> list[str]:
    return [
        "la_code", "la_name",
        "A1::Total initial assessments",
        "A1::Total owed a prevention or relief duty",
        "A1::Total owed a prevention or relief duty > Of which: single",
        "A5R::Total owed a relief duty",
        "A5R::Single parent with dependent children > Male",
        "A5R::Single parent with dependent children > Female",
        "A5R::Single parent without dependent children > Male",
    ]


def test_a_measure_matches_on_header_text_not_position(headers):
    assert panel.find_column(headers, "A1", "total initial assessments") == \
        "A1::Total initial assessments"


def test_the_shortest_matching_header_wins(headers):
    """A deeper header is a sub-breakdown of the total, never the total."""
    assert panel.find_column(headers, "A1", "total owed a prevention or relief duty") == \
        "A1::Total owed a prevention or relief duty"


def test_a_measure_is_confined_to_its_own_table(headers):
    """'Total owed a relief duty' exists in more than one table."""
    assert panel.find_column(headers, "A1", "total owed a relief duty") is None
    assert panel.find_column(headers, "A5R", "total owed a relief duty") is not None


def test_a_missing_measure_returns_nothing_rather_than_guessing(headers):
    assert panel.find_column(headers, "A1", "households in temporary accommodation") is None


def test_child_rollups_sum_the_split_columns_and_exclude_the_negation(headers):
    frame = pd.DataFrame([dict.fromkeys(headers, 0) | {
        "la_code": "E07000071", "la_name": "Colchester", "financial_year": "2024-25",
        "A5R::Single parent with dependent children > Male": 10,
        "A5R::Single parent with dependent children > Female": 25,
        "A5R::Single parent without dependent children > Male": 99,
    }])
    frame["financial_year"] = "2024-25"
    selected, resolved = panel.select_measures(frame)
    assert selected["relief_with_children"].iloc[0] == 35
    assert "2 columns summed" in resolved["relief_with_children"]


def test_an_unresolved_measure_is_null_not_zero(headers):
    frame = pd.DataFrame([dict.fromkeys(headers, 5) | {
        "la_code": "E07000071", "la_name": "Colchester", "financial_year": "2018-19"}])
    selected, resolved = panel.select_measures(frame)
    assert resolved["s21_notice"] is None
    assert pd.isna(selected["s21_notice"].iloc[0])


def test_rates_divide_by_the_published_denominator():
    frame = pd.DataFrame({
        "households_in_area_000": [50.0, 100.0],
        "relief_with_children": [100.0, 100.0],
        **{m: [1.0, 1.0] for m in panel.MEASURES if m != "households_in_area_000"},
        **{m: [1.0, 1.0] for m in panel.CHILD_ROLLUPS if m != "relief_with_children"},
    })
    rated = panel.add_rates(frame)
    # 100 families in 50,000 households is 2 per 1,000
    assert list(rated["rate_relief_with_children"]) == [2.0, 1.0]
    assert list(rated["households_in_area"]) == [50_000.0, 100_000.0]


def test_a_missing_count_stays_missing_after_rate_conversion():
    frame = pd.DataFrame({
        "households_in_area_000": [50.0],
        "relief_with_children": [None],
        **{m: [1.0] for m in panel.MEASURES if m != "households_in_area_000"},
        **{m: [1.0] for m in panel.CHILD_ROLLUPS if m != "relief_with_children"},
    })
    assert pd.isna(panel.add_rates(frame)["rate_relief_with_children"].iloc[0])


def test_no_district_is_its_own_successor():
    for successor, predecessors in panel.SUCCESSORS.items():
        assert successor not in predecessors


def test_a_district_belongs_to_at_most_one_successor():
    seen: set[str] = set()
    for predecessors in panel.SUCCESSORS.values():
        overlap = seen & set(predecessors)
        assert not overlap, f"{overlap} claimed by two successors"
        seen |= set(predecessors)


def test_the_two_boundary_maps_do_not_contradict_each_other():
    """One map creates authorities, the other retires them. A code in both
    would be reconciled twice and end up with whichever score ran last."""
    created = set(panel.SUCCESSORS)
    retired = set(panel.PREDECESSOR_OF_2019)
    assert not created & retired


def test_every_boundary_code_is_a_well_formed_district_code():
    codes = (set(panel.SUCCESSORS)
             | {c for v in panel.SUCCESSORS.values() for c in v}
             | set(panel.PREDECESSOR_OF_2019)
             | set(panel.PREDECESSOR_OF_2019.values()))
    malformed = [c for c in codes if not (len(c) == 9 and c.startswith("E0"))]
    assert not malformed


def test_the_coverage_grid_counts_the_gaps_it_prints():
    years = ["2018-19", "2019-20"]
    coverage = {y: dict.fromkeys(panel.all_measures(), "found") for y in years}
    coverage["2018-19"]["s21_notice"] = None

    lines, gaps = panel.coverage_grid(coverage, years)
    assert gaps == 1
    assert "1 measure-years missing" in lines[-1]
    assert any("s21_notice" in line and "X" in line for line in lines)
