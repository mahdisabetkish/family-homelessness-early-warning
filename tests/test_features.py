"""Feature construction and labelling.

The single most damaging error available in this project is leakage between
the feature year and the outcome year, so these run the real functions over a
small panel where the right answer can be worked out by hand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fhew import config, features


def test_labels_come_from_the_following_year(toy_panel):
    design = features.build_design(toy_panel)
    for _, row in design.iterrows():
        assert int(row["outcome_year"][:4]) == int(row["financial_year"][:4]) + 1


def test_next_year_never_crosses_an_authority_boundary(toy_panel):
    """The shift that builds `next_year` is grouped, so the last year of one
    authority must not borrow the first year of the next."""
    design = features.build_design(toy_panel)
    lookup = toy_panel.set_index(["la_code", "financial_year"])[config.OUTCOME]
    for _, row in design.iterrows():
        assert np.isclose(row["next_year"],
                          lookup[(row["la_code"], row["outcome_year"])])


def test_the_final_year_of_each_authority_is_dropped(toy_panel):
    """It has no following year, so it cannot carry a label."""
    design = features.build_design(toy_panel)
    assert (design["financial_year"] != config.YEARS[-1]).all()
    # three authorities, one usable row for every year but the last
    assert len(design) == 3 * (len(config.YEARS) - 1)


def test_escalation_fires_only_on_a_quarter_or_more(toy_panel):
    design = features.add_labels(features.build_design(toy_panel))
    labelled = design[design["y_escalation"].notna()]
    ratio = labelled["next_year"] / labelled[config.OUTCOME]
    expected = (ratio >= config.ESCALATION_THRESHOLD).astype(int)
    pd.testing.assert_series_equal(labelled["y_escalation"].astype(int), expected,
                                   check_names=False)


def test_escalation_is_undefined_where_the_base_is_zero(toy_panel):
    """A relative rise from zero is not a large rise, it is undefined."""
    design = features.add_labels(features.build_design(toy_panel))
    zero_base = design[design[config.OUTCOME] == 0]
    assert len(zero_base) == 1
    assert zero_base["y_escalation"].isna().all()


def test_the_level_cutoff_ignores_the_test_year(toy_panel):
    """Doubling every test-year value must not move the threshold.

    If it did, the label definition would depend on the year being scored and
    the reported prevalence would be fixed at the nominal quintile by
    construction rather than measured.
    """
    baseline = features.add_labels(features.build_design(toy_panel))

    inflated = toy_panel.copy()
    test_rows = inflated["financial_year"] == config.TEST_OUTCOME_YEAR
    inflated.loc[test_rows, config.OUTCOME] *= 2
    shifted = features.add_labels(features.build_design(inflated))

    assert shifted.attrs["level_cutoff"] == baseline.attrs["level_cutoff"]


def test_no_label_leaks_into_the_feature_list(toy_panel):
    design = features.add_labels(features.build_design(toy_panel))
    columns = features.feature_columns(design)
    assert "next_year" not in columns
    assert not any(c.startswith("y_") for c in columns)


def test_the_feature_list_survives_a_round_trip_through_parquet(toy_panel, tmp_path):
    """Parquet drops `attrs`, so the list has to be rebuildable from names."""
    design = features.add_labels(features.build_design(toy_panel))
    expected = features.feature_columns(design)

    path = tmp_path / "design.parquet"
    design.to_parquet(path, index=False)
    reloaded = pd.read_parquet(path)

    assert features.feature_columns(reloaded) == expected


def test_one_year_changes_are_within_authority_differences(toy_panel):
    design = features.build_design(toy_panel)
    rising = design[(design["la_code"] == "E07000001")
                    & (design["financial_year"] == config.YEARS[2])]
    # 1.0 -> 2.0 between the second and third years
    assert np.isclose(rising[f"d1_{config.OUTCOME}"].iloc[0], 1.0)
