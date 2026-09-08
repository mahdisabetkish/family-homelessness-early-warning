"""The neighbourhood layer: segmentation and anomaly detection for Colchester.

The local-authority panel answers 'which areas'; a service needs 'which
neighbourhoods'. This module works on the LSOAs that make up Colchester and
does two things a housing team would use directly.

Segmentation groups neighbourhoods into service profiles by k-means over the
deprivation domains, so outreach can be designed per type rather than per
postcode. Anomaly detection surfaces neighbourhoods whose profile is unusual
*for Colchester* - a high overall deprivation score is not news, but low
overall deprivation with high child income deprivation is, because it is
invisible to a headline triage.

Everything runs on published open data at LSOA level. Nothing here identifies
a household; the equivalent inside a council would run on linked
administrative data in a governed environment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config

DOMAINS = {
    "Index of Multiple Deprivation (IMD) Score": "imd",
    "Income Score (rate)": "income",
    "Employment Score (rate)": "employment",
    "Health Deprivation and Disability Score": "health",
    "Crime Score": "crime",
    "Barriers to Housing and Services Score": "barriers",
    "Living Environment Score": "living_env",
    "Income Deprivation Affecting Children Index (IDACI) Score (rate)": "idaci",
    "Children and Young People Sub-domain Score": "children_young_people",
    "Wider Barriers Sub-domain Score": "wider_barriers",
}

# Features the models see. IMD itself is excluded: it is a weighted average of
# the others, so including it would let one composite dominate the geometry.
FEATURES = ["income", "employment", "health", "crime", "barriers",
            "living_env", "idaci", "children_young_people", "wider_barriers"]

# Two segments always wins on silhouette here, but it only recovers
# "deprived / not deprived", which a service already knows. Segments are for
# designing differentiated outreach, so the selection is constrained to a
# range that is operationally useful and the trade-off is reported rather
# than buried.
MIN_SEGMENTS = 4

CONTAMINATION = 0.10
N_TREES = 500

PRETTY_DOMAIN = {
    "income": "income pressure", "employment": "worklessness",
    "health": "poor health", "crime": "crime", "barriers": "access barriers",
    "living_env": "poor living environment", "idaci": "child poverty",
    "children_young_people": "youth disadvantage",
    "wider_barriers": "housing affordability and overcrowding",
}


def load_lsoa() -> pd.DataFrame:
    """Read IoD2019 LSOA scores, keeping Essex districts."""
    frame = pd.read_csv(config.IOD_RAW / "iod2019_lsoa_all.csv")
    # Header cells contain embedded newlines ("Education\n Skills and ...").
    frame.columns = [" ".join(str(c).split()) for c in frame.columns]

    rename = {" ".join(k.split()): v for k, v in DOMAINS.items()}
    keep = {
        "LSOA code (2011)": "lsoa_code",
        "LSOA name (2011)": "lsoa_name",
        "Local Authority District code (2019)": "la_code",
        "Local Authority District name (2019)": "la_name",
        "Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)":
            "imd_decile_in_england",
        "Total population: mid 2015 (excluding prisoners)": "population",
        "Dependent Children aged 0-15: mid 2015 (excluding prisoners)": "children_0_15",
        **rename,
    }
    missing = [c for c in keep if c not in frame.columns]
    assert not missing, f"IoD2019 columns not found: {missing}"

    frame = frame[list(keep)].rename(columns=keep)
    return frame[frame["la_code"].isin(config.ESSEX)].reset_index(drop=True)


def choose_k(matrix: np.ndarray, candidates: range) -> tuple[int, pd.DataFrame]:
    """Score each candidate segment count, then pick the best usable one.

    With around a hundred neighbourhoods and no ground truth, silhouette is
    the honest criterion: it rewards separation without assuming a label. The
    full score table comes back with the choice so the constraint is visible
    rather than implied.
    """
    rows = []
    for k in candidates:
        model = KMeans(n_clusters=k, n_init=25, random_state=config.RANDOM_SEED).fit(matrix)
        rows.append({"k": k,
                     "silhouette": silhouette_score(matrix, model.labels_),
                     "inertia": model.inertia_})
    scores = pd.DataFrame(rows)
    usable = scores[scores["k"] >= MIN_SEGMENTS]
    return int(usable.loc[usable["silhouette"].idxmax(), "k"]), scores


def name_segments(profile: pd.DataFrame) -> dict[int, str]:
    """Give each segment a label a housing officer can read.

    Segments are walked in order of child income deprivation, and each is
    named for the domain on which it deviates most from the Colchester
    average. A domain already used is skipped, so no two segments end up with
    the same name and the numbering stays stable across reruns.
    """
    centre = profile[FEATURES].mean()
    spread = profile[FEATURES].std().replace(0, 1)
    order = profile["idaci"].sort_values(ascending=False).index

    names: dict[int, str] = {}
    used: set[str] = set()
    for rank, segment in enumerate(order, start=1):
        deviation = ((profile.loc[segment, FEATURES] - centre) / spread)
        for domain in deviation.abs().sort_values(ascending=False).index:
            if domain not in used:
                used.add(domain)
                standout = domain
                break
        else:
            standout = deviation.abs().idxmax()

        label = PRETTY_DOMAIN[standout]
        if rank == 1:
            names[segment] = f"Highest child deprivation ({label})"
        elif rank == len(order):
            names[segment] = f"Least deprived ({label})"
        else:
            names[segment] = f"Mixed, {label}"
    return names


def segment(frame: pd.DataFrame, matrix: np.ndarray, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit k-means, attach segment ids and readable names, return the profile."""
    kmeans = KMeans(n_clusters=k, n_init=50, random_state=config.RANDOM_SEED)
    frame = frame.copy()
    frame["segment"] = kmeans.fit_predict(matrix)
    profile = frame.groupby("segment")[FEATURES].mean()
    labels = name_segments(profile)
    frame["segment_name"] = frame["segment"].map(labels)
    return frame, profile.assign(segment_name=pd.Series(labels))


def flag_anomalies(frame: pd.DataFrame, matrix: np.ndarray) -> pd.DataFrame:
    """Isolation Forest over the same features: unusual for Colchester."""
    forest = IsolationForest(n_estimators=N_TREES, contamination=CONTAMINATION,
                             random_state=config.RANDOM_SEED)
    forest.fit(matrix)
    frame = frame.copy()
    # Negate so that larger = more anomalous, which reads more naturally.
    frame["anomaly_score"] = -forest.score_samples(matrix)
    frame["is_anomaly"] = forest.predict(matrix) == -1
    return frame


def standardise(frame: pd.DataFrame) -> np.ndarray:
    """Feature matrix for both unsupervised models, on a common scale."""
    return StandardScaler().fit_transform(frame[FEATURES])


def dominant_driver(row: pd.Series, frame: pd.DataFrame) -> tuple[str, float]:
    """Which domain makes this neighbourhood unusual, and by how many SDs."""
    z = (row[FEATURES] - frame[FEATURES].mean()) / frame[FEATURES].std()
    driver = z.abs().idxmax()
    return str(driver), float(z[driver])
