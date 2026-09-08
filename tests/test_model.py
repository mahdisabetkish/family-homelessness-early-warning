"""The design matrix and its labels.

The single most damaging error available in this project is leakage between
the feature year and the outcome year, so that is what these check.
"""
from __future__ import annotations

import numpy as np

OUTCOME = "rate_relief_with_children"


def test_outcome_year_always_follows_the_feature_year(design):
    """Features come from year t, labels from t+1, never the reverse."""
    for _, row in design.iterrows():
        feature_start = int(row["financial_year"][:4])
        outcome_start = int(row["outcome_year"][:4])
        assert outcome_start == feature_start + 1, (
            f"{row['la_name']}: features {row['financial_year']}, "
            f"outcome {row['outcome_year']}")


def test_next_year_belongs_to_the_same_authority(design, panel):
    """The shift that builds `next_year` must not cross an authority boundary."""
    lookup = panel.set_index(["la_code", "financial_year"])[OUTCOME]
    sample = design.sample(min(200, len(design)), random_state=0)
    for _, row in sample.iterrows():
        expected = lookup.get((row["la_code"], row["outcome_year"]))
        if expected is None or np.isnan(expected):
            continue
        assert np.isclose(row["next_year"], expected)


def test_escalation_label_matches_its_definition(design):
    labelled = design[design["y_escalation"].notna()]
    ratio = labelled["next_year"] / labelled[OUTCOME]
    assert ((ratio >= 1.25) == (labelled["y_escalation"] == 1)).all()


def test_escalation_label_is_null_where_the_base_is_zero(design):
    zero_base = design[design[OUTCOME] == 0]
    assert zero_base["y_escalation"].isna().all()


def test_level_cutoff_comes_from_training_years_only(design):
    """The label definition must not depend on the year being scored.

    A cutoff taken from the test year's own distribution would fix its
    prevalence at exactly 20% by construction. Because it is fixed from the
    training years, prevalence in later years is free to drift, and it does.
    """
    prevalence = design.groupby("outcome_year")["y_level"].mean()
    assert prevalence.loc["2024-25"] > 0.25, (
        "test-year prevalence sitting at the nominal quintile suggests the "
        "cutoff was recomputed on the test year")


def test_no_feature_column_is_the_label(design):
    features = [c for c in design.columns
                if c.startswith(("rate_", "d1_", "d2_", "share_"))]
    assert "next_year" not in features
    assert not any(c.startswith("y_") for c in features)
