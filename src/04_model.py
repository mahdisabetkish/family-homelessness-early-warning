"""Stage 04 - Predictive modelling with a prospective backtest.

The operational question a housing service actually asks is not "where is
homelessness high" - they know that - but "where is it about to get worse,
and where should this month's limited capacity go". This stage therefore
fits two tasks and reports them side by side, because the contrast is the
finding:

  Task A - LEVEL:      will this authority be in the worst quintile for family
                       homelessness next year?
  Task B - ESCALATION: will family homelessness rise by 25% or more here next
                       year?

Task A is close to a lookup: last year's value answers it. Task B is the one
worth modelling. Presenting both is the point - it shows where machine
learning earns its place and where it does not.

The feature construction is in `fhew.features`, the models in `fhew.models`
and the backtest in `fhew.evaluation`. This stage wires them together and
writes the tables the slides and the dashboard read.
"""
from __future__ import annotations

import json
import sys
import warnings

import pandas as pd

from fhew import config, evaluation, features, models

warnings.filterwarnings("ignore", category=UserWarning)


def main() -> int:
    config.TABLES.mkdir(parents=True, exist_ok=True)
    config.MODELS.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(config.PANEL_FILE)
    design = features.add_labels(features.build_design(panel))
    feature_cols = features.feature_columns(design)

    print(f"Design matrix: {len(design)} authority-years, {len(feature_cols)} features")
    print(f"  outcome: {config.OUTCOME} (families with children owed a relief duty "
          f"per 1,000 households)")
    print(f"  level cutoff fixed from training years: "
          f"{design.attrs['level_cutoff']:.2f} per 1,000")
    for task, description in evaluation.TASKS.items():
        counts = features.prevalence_by_year(design, task)
        print(f"  {task} ({description}) prevalence by outcome year: "
              + ", ".join(f"{y[2:]}={v:.0%}" for y, v in counts.items()))

    design.to_parquet(config.DESIGN_FILE, index=False)

    summary = {}
    for task, title in [("y_level", "TASK A - LEVEL"),
                        ("y_escalation", "TASK B - ESCALATION")]:
        result = evaluation.evaluate(task, design, feature_cols)
        print(f"\n{title}: held-out year {config.TEST_OUTCOME_YEAR} "
              f"(n={result.attrs['n_test']}, prevalence "
              f"{result.attrs['prevalence_test']:.0%}, "
              f"trained on {result.attrs['n_train']} rows)")
        print("\n".join(evaluation.format_table(result)))

        result.to_csv(config.TABLES / f"performance_{task}.csv", index=False)
        result.attrs["predictions"].to_csv(
            config.TABLES / f"predictions_{task}.csv", index=False)
        summary[task] = {
            "n_test": result.attrs["n_test"],
            "prevalence": result.attrs["prevalence_test"],
            "results": result.to_dict("records"),
        }

    (config.MODELS / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float))
    (config.MODELS / "specifications.json").write_text(
        json.dumps(models.specifications(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
