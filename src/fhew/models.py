"""The model zoo, and what each member is for.

Four fitted models spanning the interpretability / capacity trade-off, plus
two baselines that are not fitted at all. The rationale text lives beside the
constructors on purpose: the dashboard and the slides both read it from here,
so a hyperparameter cannot change without the sentence describing it sitting
one screen away from the change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .mlp import MCDropoutMLP

PERSISTENCE = "Persistence baseline"
PREVALENCE = "Prevalence (no model)"
BASELINES = (PERSISTENCE, PREVALENCE)


def make_models(seed: int = config.RANDOM_SEED) -> dict[str, Pipeline]:
    """The fitted models, spanning the interpretability / capacity trade-off."""
    return {
        "Logistic regression": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, C=0.3, random_state=seed)),
        ]),
        "Gradient boosting": Pipeline([
            # HistGradientBoosting handles missing values natively, which
            # matters because several measures only exist from 2022-23.
            ("clf", HistGradientBoostingClassifier(
                max_depth=3, max_iter=300, learning_rate=0.06,
                l2_regularization=1.0, min_samples_leaf=15, random_state=seed)),
        ]),
        "Gradient boosting (calibrated)": Pipeline([
            ("clf", CalibratedClassifierCV(
                HistGradientBoostingClassifier(
                    max_depth=3, max_iter=300, learning_rate=0.06,
                    l2_regularization=1.0, min_samples_leaf=15, random_state=seed),
                method="isotonic", cv=3)),
        ]),
        # A neural network is not expected to win on 1,686 rows. It is here
        # for the uncertainty: dropout left on at inference gives a spread per
        # authority, which a ranked list on its own cannot express.
        "Neural network (MC dropout)": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", MCDropoutMLP(hidden=(64, 32), dropout=0.3, random_state=seed)),
        ]),
    }


# What each model is for, in the terms a reader of the dashboard needs.
RATIONALE = {
    "Logistic regression": (
        "The interpretable reference. Strong L2 regularisation, because 83 "
        "features against roughly 1,100 training rows will otherwise overfit. "
        "Its standardised coefficients are what the driver chart reports, and "
        "on the harder task it is the best of the four."),
    "Gradient boosting": (
        "Deliberately shallow: depth 3, and at least 15 samples per leaf. "
        "Chosen partly because it handles missing values natively, and 14% of "
        "the design matrix is missing in a way that is informative rather than "
        "random."),
    "Gradient boosting (calibrated)": (
        "The same model with isotonic calibration. Whoever acts on a triage "
        "score reads it as a probability, so it has to be one. This moves the "
        "Brier score on the level task from 0.19 to 0.089 while barely "
        "changing the ranking."),
    "Neural network (MC dropout)": (
        "Not expected to win on this sample size, and it does not. It is here "
        "for the uncertainty: dropout left switched on at inference gives an "
        "approximate posterior, so each authority carries a spread rather than "
        "a point estimate. Early stopping on a held-out slice of the training "
        "years, and a reweighted loss for the small positive class."),
    PERSISTENCE: (
        "Not a fitted model. Last year's value for the level task, last year's "
        "change for escalation. This is what a service already has without any "
        "analytics, and two of the three findings come from it."),
    PREVALENCE: (
        "Predicts the base rate for everyone. It fixes the floor, so that a "
        "hit rate in a fixed-length list has a meaning to compare against."),
}


def persistence_scores(frame: pd.DataFrame, task: str) -> np.ndarray:
    """The baseline a service already has without any model.

    For the level task, this year's rate. For escalation, this year's
    one-year change - the obvious "it is already going up" heuristic.
    """
    if task == "y_level":
        return frame[config.OUTCOME].fillna(0).to_numpy()
    return frame[f"d1_{config.OUTCOME}"].fillna(0).to_numpy()


def specifications() -> list[dict]:
    """Introspect the pipelines so the published config is the real one.

    Hand-written model tables go stale the first time someone tunes a
    parameter. Reading the parameters off the constructed estimator means the
    dashboard's model page cannot describe a model that is not being fitted.
    """
    specs = []
    for name, pipeline in make_models().items():
        steps = []
        for step_name, step in pipeline.steps:
            params = step.get_params(deep=False)
            interesting = {
                k: (list(v) if isinstance(v, tuple) else v)
                for k, v in params.items()
                if k not in {"random_state", "verbose", "n_jobs", "estimator"}
                and not k.startswith("_") and (v is not None)
                and isinstance(v, int | float | str | bool | tuple)
            }
            steps.append({"step": step_name,
                          "class": type(step).__name__,
                          "params": interesting})
        specs.append({"model": name, "kind": "predictive",
                      "steps": steps, "why": RATIONALE.get(name, "")})
    for name in BASELINES:
        specs.append({"model": name, "kind": "baseline", "steps": [],
                      "why": RATIONALE[name]})
    return specs
