"""Turn the panel into a design matrix with next-year labels.

One row per authority-year, features measured in year t, labels taken from
the same authority in year t+1. Two properties matter more than any modelling
choice made later, so they live here where they can be tested:

* No feature is measured after the year it predicts. Every column is built
  from the panel row for year t or earlier, and the label is a shift of -1.
* The level threshold is fixed from the training years alone. Defining it on
  the outcome year's own distribution would let the test year's cross-section
  decide what counts as a positive in the test year.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def build_design(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per authority-year, with year-t features and year-(t+1) labels.

    Deltas are included because the level of demand is largely structural
    while the *change* carries the early signal a service can act on.
    """
    outcome = config.OUTCOME
    panel = panel.sort_values(["la_code", "financial_year"]).copy()

    rate_cols = [c for c in panel.columns if c.startswith("rate_")]
    deprivation_cols = [c for c in panel.columns if c.endswith("_score")]

    # Composition: what share of homeless households contain children. A small
    # authority with a high share is a different problem from a large one.
    with np.errstate(invalid="ignore", divide="ignore"):
        panel["share_relief_children"] = (
            panel["relief_with_children"] / panel["relief_total"])
        panel["share_prevention_children"] = (
            panel["prevention_with_children"] / panel["prevention_total"])
        panel["prevention_to_relief"] = (
            panel["prevention_total"] / panel["relief_total"])

    grouped = panel.groupby("la_code", sort=False)
    for column in rate_cols:
        panel[f"d1_{column}"] = grouped[column].diff()
    # Two-year trend, to separate a one-off spike from a sustained rise.
    panel[f"d2_{outcome}"] = grouped[outcome].diff(2)

    # Labels come from the same authority one year later.
    panel["next_year"] = grouped[outcome].shift(-1)
    panel["outcome_year"] = grouped["financial_year"].shift(-1)

    feature_cols = (
        rate_cols
        + [f"d1_{c}" for c in rate_cols]
        + [f"d2_{outcome}", "share_relief_children", "share_prevention_children",
           "prevention_to_relief"]
        + deprivation_cols
    )
    keep = ["la_code", "la_name", "financial_year", "outcome_year",
            outcome, "next_year", "households_in_area", "relief_with_children"]
    # OUTCOME is both an identifier column and a feature (this year's level),
    # so select the union once rather than duplicating the column.
    ordered = keep + [c for c in feature_cols if c not in keep]
    design = panel[ordered].copy()
    design.attrs["feature_cols"] = feature_cols
    return design[design["outcome_year"].notna() & design["next_year"].notna()]


def add_labels(design: pd.DataFrame) -> pd.DataFrame:
    """Attach both task labels.

    Level asks whether the authority will be in the worst national quintile
    next year; escalation whether family homelessness will rise by a quarter
    or more. The first is close to a lookup, the second is not, and showing
    both is what makes the comparison informative.
    """
    design = design.copy()
    train_mask = design["outcome_year"].isin(config.TRAIN_OUTCOME_YEARS)
    cutoff = design.loc[train_mask, "next_year"].quantile(config.LEVEL_QUINTILE)
    design.attrs["level_cutoff"] = float(cutoff)

    design["y_level"] = (design["next_year"] >= cutoff).astype(int)
    # Escalation is a within-authority relative change, so it needs no
    # cross-sectional information at all.
    ratio = design["next_year"] / design[config.OUTCOME].replace(0, np.nan)
    design["y_escalation"] = (ratio >= config.ESCALATION_THRESHOLD).astype(int)
    design.loc[ratio.isna(), "y_escalation"] = np.nan
    return design


def feature_columns(design: pd.DataFrame) -> list[str]:
    """The feature list, from the frame's attrs if present or rebuilt if not.

    Parquet does not carry `attrs`, so a design matrix read back from disk has
    lost the list that built it. Rebuilding it from the column names keeps a
    stage that only reads the artefact working.
    """
    if design.attrs.get("feature_cols"):
        return list(design.attrs["feature_cols"])
    identifiers = {"la_code", "la_name", "financial_year", "outcome_year",
                   "next_year", "households_in_area", "relief_with_children",
                   "y_level", "y_escalation"}
    return [c for c in design.columns if c not in identifiers]


def prevalence_by_year(design: pd.DataFrame, task: str) -> pd.Series:
    """Share of positives in each outcome year, for reporting the split."""
    return design.groupby("outcome_year")[task].mean()
