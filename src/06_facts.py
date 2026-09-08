"""
Stage 06 - Emit the numbers the slides quote, as LaTeX macros.

Every figure in the deck that is not a plot comes from here. Hard-coding them
into the .tex would let the slides drift from the pipeline the moment a stage
is rerun; defining them as macros means a stale number becomes a compile-time
mismatch rather than a claim made out loud in an interview.
"""
from __future__ import annotations

import re
import sys

import pandas as pd

from fhew import config, export, features


def count_tests() -> int:
    """Test functions defined under tests/.

    Counted from the source rather than from a pytest run: the slide should
    not need the suite to have been executed, and a count that comes from the
    files cannot quietly describe a different tree.
    """
    return sum(len(re.findall(r"^def test_", path.read_text(), re.MULTILINE))
               for path in sorted((config.ROOT / "tests").glob("test_*.py")))


def main() -> int:
    panel = pd.read_parquet(config.PANEL_FILE)
    design = pd.read_parquet(config.DESIGN_FILE)
    lsoa = pd.read_parquet(config.COLCHESTER_LSOA_FILE)

    colchester = panel[panel["la_code"] == config.COLCHESTER].set_index("financial_year")
    first, last = config.YEARS[0], config.YEARS[-1]

    def missing_returns(year: str) -> int:
        """Authorities that filed no H-CLIC return that year."""
        rows = panel[panel["financial_year"] == year]
        return int(rows["relief_total"].isna().sum())

    def england_rate(year: str) -> float:
        part = panel[(panel["financial_year"] == year)
                     & panel["relief_with_children"].notna()]
        return part["relief_with_children"].sum() / (part["households_in_area"].sum() / 1_000)

    children_first = colchester.loc[first, "relief_with_children"]
    children_last = colchester.loc[last, "relief_with_children"]

    escalation = pd.read_csv(
        config.TABLES / "performance_y_escalation.csv").set_index("model")
    level = pd.read_csv(config.TABLES / "performance_y_level.csv").set_index("model")
    predictions = pd.read_csv(config.TABLES / "predictions_y_escalation.csv")

    flagged = lsoa[lsoa["is_anomaly"]]
    hidden = flagged[flagged["imd_decile_in_england"] >= 5]

    facts = {
        # scale of the exercise
        "NumAuthorities": f"{panel['la_code'].nunique()}",
        "NumAuthorityYears": f"{len(panel):,}",
        "NumFeatures": f"{len(features.feature_columns(design))}",
        "NumYears": f"{panel['financial_year'].nunique()}",
        "FirstYear": first.replace("-", "--"),
        "LastYear": last.replace("-", "--"),
        "NumDesignRows": f"{len(design):,}",

        # the trend
        "ColchesterReliefFirst": f"{colchester.loc[first, 'relief_total']:.0f}",
        "ColchesterReliefLast": f"{colchester.loc[last, 'relief_total']:.0f}",
        "ColchesterChildrenFirst": f"{children_first:.0f}",
        "ColchesterChildrenLast": f"{children_last:.0f}",
        "ColchesterChildrenGrowth":
            f"{children_last / children_first - 1:.0%}".replace("%", r"\%"),
        "EnglandRateFirst": f"{england_rate(first):.2f}",
        "EnglandRateLast": f"{england_rate(last):.2f}",

        # data quality
        "MissingReturnsLast": f"{missing_returns(last)}",
        "MissingReturnsPrev": f"{missing_returns(config.VALID_OUTCOME_YEAR)}",

        # model results
        "LevelPersistenceAUC": f"{level.loc['Persistence baseline', 'roc_auc']:.2f}",
        "LevelModelAUC": f"{level.loc[config.HEADLINE_MODEL, 'roc_auc']:.2f}",
        "EscPersistenceAUC": f"{escalation.loc['Persistence baseline', 'roc_auc']:.2f}",
        "EscPersistenceLo": f"{escalation.loc['Persistence baseline', 'roc_auc_lo']:.2f}",
        "EscPersistenceHi": f"{escalation.loc['Persistence baseline', 'roc_auc_hi']:.2f}",
        "EscLogisticAUC": f"{escalation.loc['Logistic regression', 'roc_auc']:.2f}",
        "EscLogisticLo": f"{escalation.loc['Logistic regression', 'roc_auc_lo']:.2f}",
        "EscLogisticHi": f"{escalation.loc['Logistic regression', 'roc_auc_hi']:.2f}",
        "EscModelPrecision":
            f"{escalation.loc[config.HEADLINE_MODEL, 'precision_at_30']:.0%}"
            .replace("%", r"\%"),
        "EscBaseRate":
            f"{predictions['y_escalation'].mean():.0%}".replace("%", r"\%"),
        "EscTestN": f"{len(predictions)}",
        "BrierBefore": f"{level.loc['Logistic regression', 'brier']:.2f}",
        "BrierAfter": f"{level.loc[config.HEADLINE_MODEL, 'brier']:.3f}",

        # how the code is built
        "NumModules": f"{len(list((config.ROOT / 'src' / 'fhew').glob('*.py'))) - 1}",
        "NumStages": f"{len(list((config.ROOT / 'src').glob('0*.py')))}",
        "NumTests": f"{count_tests()}",

        # neighbourhood layer
        "NumLSOA": f"{len(lsoa)}",
        "NumChildren": f"{lsoa['children_0_15'].sum():,.0f}",
        "NumSegments": f"{lsoa['segment'].nunique()}",
        "NumFlagged": f"{len(flagged)}",
        "NumHiddenFlagged": f"{len(hidden)}",
        "HiddenChildren": f"{hidden['children_0_15'].sum():,.0f}",
    }

    export.write_latex_macros(facts)

    print(f"Wrote slides/facts.tex with {len(facts)} macros")
    for key, value in facts.items():
        print(f"  \\{key:<26} {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
