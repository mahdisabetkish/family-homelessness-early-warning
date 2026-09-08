"""Shared fixtures.

Two kinds of test live here. Most of them run against `fhew` directly on
small synthetic inputs and need nothing built, so `pytest` is useful on a
fresh clone. The rest check properties of the real artefacts and skip rather
than fail when the pipeline has not been run.
"""
from __future__ import annotations

import pandas as pd
import pytest

from fhew import config


def _load(path):
    if not path.exists():
        pytest.skip(f"{path.relative_to(config.ROOT)} not built yet; run `make all`")
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def panel():
    return _load(config.PANEL_FILE)


@pytest.fixture(scope="session")
def design():
    return _load(config.DESIGN_FILE)


@pytest.fixture(scope="session")
def lsoa():
    return _load(config.COLCHESTER_LSOA_FILE)


@pytest.fixture
def toy_panel() -> pd.DataFrame:
    """Three authorities over the full run of years, with the design columns.

    The series are chosen so that each label rule has at least one case that
    exercises it: one authority jumps, one never moves, and one starts at zero
    so its relative change is undefined. Spanning every configured year means
    the real train / validation / test split applies unchanged.
    """
    rows = []
    series = {
        # a doubling in the third year, then a gentle rise
        "E07000001": [1.0, 1.0, 2.0, 2.1, 2.2, 2.3, 2.4],
        # flat: never escalates
        "E07000002": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        # zero base: the relative change out of the first year is undefined
        "E07000003": [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    }
    for code, values in series.items():
        for year, value in zip(config.YEARS, values, strict=True):
            rows.append({
                "la_code": code,
                "la_name": f"Authority {code[-1]}",
                "financial_year": year,
                "households_in_area": 50_000.0,
                "relief_with_children": value * 50.0,
                "relief_total": 200.0,
                "prevention_with_children": 40.0,
                "prevention_total": 150.0,
                "rate_relief_with_children": value,
                "rate_relief_total": 4.0,
                "imd_score": 20.0,
            })
    return pd.DataFrame(rows)
