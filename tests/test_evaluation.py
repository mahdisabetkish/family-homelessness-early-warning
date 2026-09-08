"""Metrics and the backtest split.

Every headline claim in the write-up is one of these numbers, so they are
checked against cases whose answer is known by hand rather than against a
previous run of the same code.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from fhew import config, evaluation, features


def test_precision_at_k_counts_only_the_top_of_the_list():
    truth = np.array([1, 1, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    assert evaluation.precision_at_k(truth, scores, 2) == 1.0
    assert evaluation.precision_at_k(truth, scores, 4) == 0.5


def test_precision_at_k_reads_the_scores_not_the_row_order():
    truth = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.8])
    assert evaluation.precision_at_k(truth, scores, 2) == 1.0


def test_precision_at_k_caps_at_the_number_of_rows():
    truth = np.array([1, 0])
    scores = np.array([0.9, 0.1])
    assert evaluation.precision_at_k(truth, scores, 30) == 0.5


def test_a_perfect_ranking_has_an_interval_that_excludes_chance():
    truth = np.concatenate([np.ones(40), np.zeros(60)])
    scores = np.concatenate([np.linspace(1.0, 0.6, 40), np.linspace(0.4, 0.0, 60)])
    low, high = evaluation.bootstrap_ci(roc_auc_score, truth, scores, draws=300)
    assert low > 0.5 and high <= 1.0


def test_a_random_ranking_has_an_interval_that_contains_chance():
    rng = np.random.default_rng(1)
    truth = rng.integers(0, 2, 200)
    scores = rng.random(200)
    low, high = evaluation.bootstrap_ci(roc_auc_score, truth, scores, draws=300)
    assert low < 0.5 < high


def test_the_interval_is_reproducible():
    """The published intervals have to be the same on a rerun."""
    rng = np.random.default_rng(2)
    truth = rng.integers(0, 2, 120)
    scores = rng.random(120)
    first = evaluation.bootstrap_ci(roc_auc_score, truth, scores, draws=200)
    second = evaluation.bootstrap_ci(roc_auc_score, truth, scores, draws=200)
    assert first == second


def test_a_single_class_gives_no_interval_rather_than_a_wrong_one():
    truth = np.ones(20, dtype=int)
    low, high = evaluation.bootstrap_ci(roc_auc_score, truth, np.random.random(20), draws=50)
    assert np.isnan(low) and np.isnan(high)


def test_the_split_partitions_by_outcome_year(toy_panel):
    design = features.add_labels(features.build_design(toy_panel))
    train, valid, test = evaluation.split(design, "y_level")

    assert set(train["outcome_year"]) <= set(config.TRAIN_OUTCOME_YEARS)
    assert set(valid["outcome_year"]) == {config.VALID_OUTCOME_YEAR}
    assert set(test["outcome_year"]) == {config.TEST_OUTCOME_YEAR}


def test_no_authority_year_appears_in_two_splits(toy_panel):
    design = features.add_labels(features.build_design(toy_panel))
    train, valid, test = evaluation.split(design, "y_level")

    def keys(frame):
        return set(zip(frame["la_code"], frame["outcome_year"], strict=True))

    assert not keys(train) & keys(test)
    assert not keys(valid) & keys(test)
    assert not keys(train) & keys(valid)


def test_no_training_row_is_scored_after_the_test_year(toy_panel):
    design = features.add_labels(features.build_design(toy_panel))
    train, valid, _ = evaluation.split(design, "y_level")
    for frame in (train, valid):
        assert (frame["outcome_year"] < config.TEST_OUTCOME_YEAR).all()


@pytest.mark.parametrize("task", ["y_level", "y_escalation"])
def test_both_tasks_are_described(task):
    assert task in evaluation.TASKS
    assert evaluation.TASKS[task]
