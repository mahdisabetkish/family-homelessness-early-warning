"""The neighbourhood layer.

Segment names end up on a slide and in the dashboard legend, so the naming
rule is tested directly: two segments sharing a label would make the chart
unreadable without anything failing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fhew import neighbourhoods


def _profile(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    return frame.set_index(pd.RangeIndex(len(frame), name="segment"))


def test_segment_names_are_unique():
    """Two segments can peak on the same domain; the labels still have to differ."""
    rng = np.random.default_rng(0)
    profile = _profile([
        dict(zip(neighbourhoods.FEATURES, rng.normal(size=len(neighbourhoods.FEATURES)),
                 strict=True))
        for _ in range(5)
    ])
    names = neighbourhoods.name_segments(profile)
    assert len(set(names.values())) == len(names)


def test_identical_segments_still_get_distinct_names():
    """The degenerate case: no domain stands out anywhere."""
    profile = _profile([dict.fromkeys(neighbourhoods.FEATURES, 1.0) for _ in range(4)])
    names = neighbourhoods.name_segments(profile)
    assert len(set(names.values())) == len(names)


def test_the_most_child_deprived_segment_is_named_first():
    profile = _profile([
        dict.fromkeys(neighbourhoods.FEATURES, 0.1) | {"idaci": 0.05},
        dict.fromkeys(neighbourhoods.FEATURES, 0.1) | {"idaci": 0.40},
        dict.fromkeys(neighbourhoods.FEATURES, 0.1) | {"idaci": 0.20},
    ])
    names = neighbourhoods.name_segments(profile)
    assert names[1].startswith("Highest child deprivation")
    assert names[0].startswith("Least deprived")


def test_the_segment_count_respects_the_operational_floor():
    """k = 2 usually wins on silhouette but only recovers deprived / not."""
    rng = np.random.default_rng(0)
    matrix = np.vstack([rng.normal(loc, 0.3, size=(25, len(neighbourhoods.FEATURES)))
                        for loc in (-3, 3)])
    k, scores = neighbourhoods.choose_k(matrix, range(2, 7))

    assert k >= neighbourhoods.MIN_SEGMENTS
    # the unconstrained winner is reported rather than hidden
    assert int(scores.loc[scores["silhouette"].idxmax(), "k"]) == 2


def test_the_index_is_excluded_from_the_features_it_summarises():
    """IMD is a weighted average of the domains, so it would count twice."""
    assert "imd" in neighbourhoods.DOMAINS.values()
    assert "imd" not in neighbourhoods.FEATURES


def test_every_feature_has_a_readable_name():
    missing = [f for f in neighbourhoods.FEATURES if f not in neighbourhoods.PRETTY_DOMAIN]
    assert not missing


def test_anomaly_scores_rise_with_how_unusual_a_row_is():
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(60, len(neighbourhoods.FEATURES)))
    matrix[0] = 12.0   # one obvious outlier
    frame = pd.DataFrame(matrix, columns=neighbourhoods.FEATURES)

    flagged = neighbourhoods.flag_anomalies(frame, matrix)
    assert flagged["anomaly_score"].idxmax() == 0
    assert bool(flagged.loc[0, "is_anomaly"])


def test_the_dominant_driver_is_the_largest_standardised_deviation():
    # Every domain rises steadily across the four rows except crime, which is
    # flat and then spikes. The spike is the larger deviation once each domain
    # is put on its own scale, which is the whole point of standardising.
    frame = pd.DataFrame({f: [0.0, 1.0, 2.0, 3.0] for f in neighbourhoods.FEATURES})
    frame["crime"] = [0.0, 0.0, 0.0, 10.0]
    driver, z = neighbourhoods.dominant_driver(frame.loc[3], frame)
    assert driver == "crime"
    assert z > 0
