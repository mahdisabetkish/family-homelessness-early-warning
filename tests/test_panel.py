"""The assembled panel.

These run against the built artefacts, so they check the properties that
matter for the analysis rather than re-testing pandas.
"""
from __future__ import annotations

import pandas as pd
import pytest

COLCHESTER = "E07000071"


def test_every_authority_year_has_a_deprivation_score(panel):
    """Boundary reconciliation must leave no authority-year unmatched.

    Seven successor authorities are rebuilt from their predecessors' LSOAs and
    fourteen abolished districts inherit their successor's score. If either
    mapping falls out of date, later years silently lose their covariates.
    """
    missing = panel[panel["imd_score"].isna()]
    assert missing.empty, (
        f"{missing['la_code'].nunique()} authorities without a deprivation "
        f"score, e.g. {sorted(missing['la_name'].unique())[:5]}")


def test_rates_are_counts_over_the_published_denominator(panel):
    rows = panel[panel["relief_with_children"].notna()
                 & panel["households_in_area_000"].notna()]
    expected = rows["relief_with_children"] / rows["households_in_area_000"]
    pd.testing.assert_series_equal(
        rows["rate_relief_with_children"], expected, check_names=False)


def test_non_reporting_is_null_not_zero(panel):
    """A missing return and a genuine zero must stay distinguishable.

    Colchester filed nothing in 2023-24. If that arrives as 0 rather than NaN,
    every downstream mean and trend is wrong in a way nothing will flag.
    """
    row = panel[(panel["la_code"] == COLCHESTER)
                & (panel["financial_year"] == "2023-24")]
    assert len(row) == 1
    assert row["relief_total"].isna().all()
    assert row["relief_with_children"].isna().all()


def test_one_row_per_authority_per_year(panel):
    duplicated = panel.duplicated(subset=["la_code", "financial_year"])
    assert not duplicated.any()


def test_children_never_exceed_total_households_owed_a_duty(panel):
    rows = panel[panel["relief_with_children"].notna() & panel["relief_total"].notna()]
    assert (rows["relief_with_children"] <= rows["relief_total"]).all()


@pytest.mark.parametrize("year", ["2018-19", "2021-22", "2024-25"])
def test_authority_counts_match_the_published_releases(panel, year):
    """Guards against a parser change silently dropping authorities."""
    counts = {"2018-19": 326, "2021-22": 309, "2024-25": 296}
    assert len(panel[panel["financial_year"] == year]) == counts[year]
