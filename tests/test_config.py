"""The configuration is the contract between the stages.

Nothing here reads data. These check that the constants the whole pipeline
shares are internally consistent, because a split that overlaps or a year with
no source would otherwise surface as a confusing failure several stages later.
"""
from __future__ import annotations

from fhew import config, sources


def test_every_financial_year_has_a_source():
    missing = [y for y in config.YEARS if y not in sources.HCLIC]
    assert not missing, f"no download URL for {missing}"


def test_no_source_is_configured_for_a_year_the_pipeline_ignores():
    extra = [y for y in sources.HCLIC if y not in config.YEARS]
    assert not extra, f"{extra} would be downloaded but never read"


def test_the_splits_do_not_overlap():
    train = set(config.TRAIN_OUTCOME_YEARS)
    assert config.VALID_OUTCOME_YEAR not in train
    assert config.TEST_OUTCOME_YEAR not in train
    assert config.VALID_OUTCOME_YEAR != config.TEST_OUTCOME_YEAR


def test_the_test_year_is_the_most_recent_year():
    """A test year that is not last would leave later data unused and unaudited."""
    assert config.YEARS[-1] == config.TEST_OUTCOME_YEAR
    assert config.YEARS[-2] == config.VALID_OUTCOME_YEAR
    assert max(config.TRAIN_OUTCOME_YEARS) < config.VALID_OUTCOME_YEAR


def test_years_are_ordered_and_contiguous():
    starts = [int(y[:4]) for y in config.YEARS]
    assert starts == sorted(starts)
    assert starts == list(range(starts[0], starts[0] + len(starts)))


def test_colchester_is_one_of_the_essex_districts():
    assert config.COLCHESTER in config.ESSEX


def test_thresholds_are_the_ones_the_write_up_quotes():
    assert config.ESCALATION_THRESHOLD == 1.25   # a 25% rise
    assert config.LEVEL_QUINTILE == 0.80         # worst fifth
    assert config.CAPACITY == 30
