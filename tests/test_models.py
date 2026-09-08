"""The model zoo.

These check the parts that the dashboard and the slides read back out, so a
model added or renamed cannot leave a page describing something that is no
longer being fitted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fhew import config, models


def test_every_model_has_a_rationale():
    """The dashboard prints one per model, so a missing entry is a blank cell."""
    missing = [name for name in models.make_models() if name not in models.RATIONALE]
    assert not missing, f"no rationale written for {missing}"


def test_every_rationale_belongs_to_something_that_runs():
    known = set(models.make_models()) | set(models.BASELINES)
    orphaned = [name for name in models.RATIONALE if name not in known]
    assert not orphaned, f"{orphaned} described but never fitted"


def test_the_headline_model_is_one_of_the_fitted_models():
    assert config.HEADLINE_MODEL in models.make_models()


def test_specifications_cover_the_models_and_the_baselines():
    named = {spec["model"] for spec in models.specifications()}
    assert named == set(models.make_models()) | set(models.BASELINES)


def test_specifications_report_the_parameters_that_are_set():
    """The published table is introspected, not typed, so it must find them."""
    spec = next(s for s in models.specifications()
                if s["model"] == "Gradient boosting")
    params = spec["steps"][-1]["params"]
    assert params["max_depth"] == 3
    assert params["min_samples_leaf"] == 15


def test_models_are_reproducible_from_the_seed():
    """Two constructions with the same seed must give the same predictions."""
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(120, 4)), columns=list("abcd"))
    y = (x["a"] + rng.normal(scale=0.5, size=120) > 0).astype(int)

    first = models.make_models(seed=7)["Logistic regression"].fit(x, y)
    second = models.make_models(seed=7)["Logistic regression"].fit(x, y)
    np.testing.assert_allclose(first.predict_proba(x), second.predict_proba(x))


@pytest.mark.parametrize("task,column", [
    ("y_level", config.OUTCOME),
    ("y_escalation", f"d1_{config.OUTCOME}"),
])
def test_the_persistence_baseline_uses_the_right_column(task, column):
    """Level compares against this year's rate, escalation against its change."""
    frame = pd.DataFrame({config.OUTCOME: [1.0, 2.0, np.nan],
                          f"d1_{config.OUTCOME}": [0.1, np.nan, 0.3]})
    scores = models.persistence_scores(frame, task)
    expected = frame[column].fillna(0).to_numpy()
    np.testing.assert_allclose(scores, expected)
