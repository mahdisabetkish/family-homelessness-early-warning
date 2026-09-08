"""The prospective backtest, and the metrics that go with it.

Validation is strictly temporal: a model never sees a year at or after the
one it is scored on. The metrics are chosen to match how the output would be
used rather than to flatter it - a fixed-length worklist means the hit rate
at the top matters more than the ranking of the whole country, and an
interval around every AUC keeps small differences from being read as real.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import config, models
from .mlp import MCDropoutMLP

TASKS = {
    "y_level": "worst national quintile next year",
    "y_escalation": "25%+ rise next year",
}


def bootstrap_ci(metric: Callable[[np.ndarray, np.ndarray], float],
                 y_true: np.ndarray, scores: np.ndarray,
                 draws: int = config.BOOTSTRAP_DRAWS,
                 seed: int = config.RANDOM_SEED) -> tuple[float, float]:
    """Percentile bootstrap interval for a ranking metric.

    The held-out year has only around 270 authorities and the positive class
    is small, so differences of a few points between models are well inside
    sampling noise. Reporting the interval keeps the comparison honest.
    Resamples that lose a class are skipped, because the metric is undefined
    there rather than zero.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(metric(y_true[idx], scores[idx]))
    if not values:
        return (np.nan, np.nan)
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Share of the top-k ranked authorities that really did have the outcome.

    This is the metric that matches how the output would be used: a team works
    a fixed-size list, so what matters is the hit rate at the top, not the
    ranking of the whole country.
    """
    k = min(k, len(scores))
    order = np.argsort(-scores)[:k]
    return float(y_true[order].mean())


def split(design: pd.DataFrame, task: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train / validation / test rows for one task, split by outcome year."""
    labelled = design[design[task].notna()]
    return (
        labelled[labelled["outcome_year"].isin(config.TRAIN_OUTCOME_YEARS)],
        labelled[labelled["outcome_year"] == config.VALID_OUTCOME_YEAR],
        labelled[labelled["outcome_year"] == config.TEST_OUTCOME_YEAR],
    )


def evaluate(task: str, design: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Fit on the training years, tune nothing on test, score the held-out year.

    The returned frame carries the per-authority predictions and the split
    sizes in `attrs`, so a caller can report them without recomputing the
    split it was just given.
    """
    train, valid, test = split(design, task)

    # The validation year is folded into training for the final fit; it is
    # reported separately first so the choice can be seen, not assumed.
    fit = pd.concat([train, valid])
    x_fit, y_fit = fit[features], fit[task].astype(int).to_numpy()
    x_test, y_test = test[features], test[task].astype(int).to_numpy()

    def score_row(name: str, scores: np.ndarray, brier: float) -> dict:
        low, high = bootstrap_ci(roc_auc_score, y_test, scores)
        return {
            "model": name,
            "roc_auc": roc_auc_score(y_test, scores),
            "roc_auc_lo": low,
            "roc_auc_hi": high,
            "pr_auc": average_precision_score(y_test, scores),
            "precision_at_30": precision_at_k(y_test, scores, config.CAPACITY),
            "brier": brier,
        }

    baseline = models.persistence_scores(test, task)
    prior = DummyClassifier(strategy="prior").fit(x_fit, y_fit).predict_proba(x_test)[:, 1]
    rows = [
        score_row(models.PERSISTENCE, baseline, np.nan),
        {"model": models.PREVALENCE, "roc_auc": 0.5,
         "roc_auc_lo": np.nan, "roc_auc_hi": np.nan,
         "pr_auc": float(y_test.mean()), "precision_at_30": float(y_test.mean()),
         "brier": brier_score_loss(y_test, prior)},
    ]

    predictions = test[["la_code", "la_name", "outcome_year", config.OUTCOME,
                        "next_year", task]].copy()
    predictions["persistence"] = baseline

    for name, model in models.make_models().items():
        model.fit(x_fit, y_fit)
        probability = model.predict_proba(x_test)[:, 1]
        rows.append(score_row(name, probability, brier_score_loss(y_test, probability)))
        predictions[name] = probability
        # The network is the only model here that can say how sure it is. A
        # high score the model is confident about and a high score it is
        # guessing at deserve different responses from a service, and a ranked
        # list on its own cannot express the difference.
        estimator = model.named_steps.get("clf")
        if isinstance(estimator, MCDropoutMLP):
            transformed = x_test
            for _, step in model.steps[:-1]:
                transformed = step.transform(transformed)
            predictions["uncertainty"] = estimator.predict_uncertainty(transformed)

    result = pd.DataFrame(rows)
    result.attrs["predictions"] = predictions
    result.attrs["n_train"] = len(fit)
    result.attrs["n_test"] = len(test)
    result.attrs["prevalence_test"] = float(y_test.mean())
    return result


def format_table(result: pd.DataFrame) -> list[str]:
    """Render one task's results as fixed-width lines for a terminal."""
    lines = [f"  {'model':<32}{'ROC-AUC (95% CI)':>22}{'PR-AUC':>9}"
             f"{'P@30':>8}{'Brier':>9}"]
    for _, row in result.iterrows():
        brier = "    -" if pd.isna(row["brier"]) else f"{row['brier']:.3f}"
        if pd.isna(row["roc_auc_lo"]):
            auc = f"{row['roc_auc']:.3f}"
        else:
            auc = (f"{row['roc_auc']:.3f} "
                   f"[{row['roc_auc_lo']:.2f}-{row['roc_auc_hi']:.2f}]")
        lines.append(f"  {row['model']:<32}{auc:>22}"
                     f"{row['pr_auc']:>9.3f}{row['precision_at_30']:>8.0%}{brier:>9}")
    return lines
