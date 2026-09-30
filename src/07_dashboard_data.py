"""
Stage 07 - Export the JSON the dashboard reads.

The dashboard is a static page: no server, no build step, no database. Every
number it shows is written here from the pipeline artefacts, so the
page cannot drift from the analysis. Files are kept small enough to load on a
phone connection.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from fhew import config, features, neighbourhoods
from fhew.export import write_json as write

MODEL = config.HEADLINE_MODEL


def national_ranking(panel: pd.DataFrame) -> list[dict]:
    """Every authority scored for 2024-25, with the outcome that followed."""
    predictions = pd.read_csv(config.TABLES / "predictions_y_escalation.csv")
    level = pd.read_csv(config.TABLES / "predictions_y_level.csv")[
        ["la_code", MODEL, "y_level"]].rename(
        columns={MODEL: "level_score", "y_level": "level_actual"})

    frame = predictions.merge(level, on="la_code", how="left")
    frame = frame.rename(columns={
        "uncertainty": "escalation_uncertainty",
        MODEL: "escalation_score",
        "y_escalation": "escalation_actual",
        config.OUTCOME: "rate_now",
        "next_year": "rate_next",
    })

    # Carry the raw counts so the page can show what a rate means in families.
    counts = panel[panel["financial_year"] == config.TEST_OUTCOME_YEAR][
        ["la_code", "relief_with_children", "relief_total", "households_in_area"]]
    frame = frame.merge(counts, on="la_code", how="left")
    frame["region"] = np.where(frame["la_code"].isin(config.ESSEX), "Essex", "England")

    keep = ["la_code", "la_name", "region", "escalation_score",
            "escalation_uncertainty", "escalation_actual",
            "level_score", "level_actual", "rate_now", "rate_next",
            "relief_with_children", "relief_total", "households_in_area"]
    keep = [c for c in keep if c in frame.columns]
    frame = frame[keep].sort_values("escalation_score", ascending=False)
    return frame.to_dict("records")


def trend(panel: pd.DataFrame) -> dict:
    """Pooled rate per group, with missing years preserved as null."""
    years = sorted(panel["financial_year"].unique())

    def rate(mask: pd.Series) -> list[float | None]:
        part = panel[mask]
        out = []
        for year in years:
            rows = part[(part["financial_year"] == year)
                        & part["relief_with_children"].notna()]
            if rows.empty:
                out.append(None)
                continue
            households = rows["households_in_area"].sum() / 1_000
            out.append(float(rows["relief_with_children"].sum() / households))
        return out

    everywhere = pd.Series(True, index=panel.index)
    return {
        "years": years,
        "series": [
            {"name": "England", "values": rate(everywhere)},
            {"name": "Essex districts", "values": rate(panel["la_code"].isin(config.ESSEX))},
            {"name": "Colchester", "values": rate(panel["la_code"] == config.COLCHESTER)},
        ],
    }


def precision_curve() -> dict:
    """Hit rate at every list length, for the capacity slider."""
    predictions = pd.read_csv(config.TABLES / "predictions_y_escalation.csv")
    truth = predictions["y_escalation"].to_numpy()
    ks = list(range(1, len(truth) + 1))

    def curve(column: str) -> list[float]:
        order = np.argsort(-predictions[column].to_numpy())
        ranked = truth[order]
        return [float(ranked[:k].mean()) for k in ks]

    return {
        "k": ks,
        "base_rate": float(truth.mean()),
        "model": curve(MODEL),
        "persistence": curve("persistence"),
        "n": len(truth),
    }


def performance() -> list[dict]:
    """Both tasks, every model, as the dashboard's comparison table."""
    rows = []
    for task, label in [("y_level", "Level"), ("y_escalation", "Escalation")]:
        frame = pd.read_csv(config.TABLES / f"performance_{task}.csv")
        for _, row in frame.iterrows():
            rows.append({
                "task": label, "model": row["model"],
                "roc_auc": row["roc_auc"],
                "lo": row.get("roc_auc_lo"), "hi": row.get("roc_auc_hi"),
                "pr_auc": row["pr_auc"],
                "precision_at_30": row["precision_at_30"],
                "brier": row["brier"],
            })
    return rows


def models() -> list[dict]:
    """Model specifications joined to what each one scored.

    The specifications are introspected from the fitted pipelines by stage 04,
    so what the dashboard prints is the configuration that actually ran.
    """
    specs = json.loads((config.MODELS / "specifications.json").read_text())
    scores = {}
    for task, label in [("y_level", "Level"), ("y_escalation", "Escalation")]:
        frame = pd.read_csv(config.TABLES / f"performance_{task}.csv")
        for _, row in frame.iterrows():
            scores.setdefault(row["model"], {})[label] = {
                "roc_auc": row["roc_auc"], "lo": row.get("roc_auc_lo"),
                "hi": row.get("roc_auc_hi"), "pr_auc": row["pr_auc"],
                "precision_at_30": row["precision_at_30"], "brier": row["brier"],
            }
    for spec in specs:
        spec["results"] = scores.get(spec["model"], {})
    return specs


def unsupervised() -> list[dict]:
    """The two models behind the neighbourhood layer."""
    lsoa = pd.read_parquet(config.COLCHESTER_LSOA_FILE)
    return [
        {"model": "k-means", "kind": "unsupervised",
         "config": {"n_clusters": int(lsoa["segment"].nunique()), "n_init": 50,
                    "features": len(neighbourhoods.FEATURES),
                    "selection": "silhouette, restricted to "
                                 f"k >= {neighbourhoods.MIN_SEGMENTS}"},
         "why": ("Groups neighbourhoods into service profiles. k = 2 scores "
                 "highest on silhouette but only recovers deprived and not "
                 "deprived, which the service already knows, so the search is "
                 "restricted to a range that is operationally useful and the "
                 "trade-off is reported rather than buried."),
         "outcome": {"segments": int(lsoa["segment"].nunique()),
                     "sizes": lsoa.groupby("segment_name").size()
                     .sort_values(ascending=False).to_dict()}},
        {"model": "Isolation Forest", "kind": "unsupervised",
         "config": {"n_estimators": neighbourhoods.N_TREES,
                    "contamination": neighbourhoods.CONTAMINATION,
                    "features": len(neighbourhoods.FEATURES)},
         "why": ("Finds neighbourhoods that are unusual for Colchester, which "
                 "is not the same as deprived in absolute terms. Contamination "
                 "is fixed rather than tuned: with 105 points and no ground "
                 "truth there is nothing honest to tune it against, so the "
                 "ranked scores are published alongside the binary flag."),
         "outcome": {"flagged": int(lsoa["is_anomaly"].sum()),
                     "of": int(len(lsoa)),
                     "in_less_deprived_half": int(
                         ((lsoa["is_anomaly"]) & (lsoa["imd_decile_in_england"] >= 5)).sum())}},
    ]


def colchester_lsoa() -> list[dict]:
    lsoa = pd.read_parquet(config.COLCHESTER_LSOA_FILE)
    order = lsoa.groupby("segment")["idaci"].mean().sort_values().index.tolist()
    lsoa = lsoa.copy()
    lsoa["segment_rank"] = lsoa["segment"].map({s: i + 1 for i, s in enumerate(order)})

    domains = neighbourhoods.FEATURES
    means, stds = lsoa[domains].mean(), lsoa[domains].std()
    # What makes each neighbourhood unusual, in plain terms.
    z = (lsoa[domains] - means) / stds
    lsoa["driver"] = z.abs().idxmax(axis=1)
    lsoa["driver_z"] = z.max(axis=1).where(
        z.max(axis=1).abs() >= z.min(axis=1).abs(), z.min(axis=1))

    keep = ["lsoa_code", "lsoa_name", "segment_rank", "segment_name",
            "imd_decile_in_england", "imd", "population", "children_0_15",
            "anomaly_score", "is_anomaly", "driver", "driver_z", *domains]
    # `idaci` is both a headline column and a model feature; take it once.
    keep = list(dict.fromkeys(keep))
    return lsoa[keep].sort_values("anomaly_score", ascending=False).to_dict("records")


def data_quality(panel: pd.DataFrame) -> dict:
    years = sorted(panel["financial_year"].unique())
    rows = []
    for year in years:
        part = panel[panel["financial_year"] == year]
        missing = part[part["relief_total"].isna()]
        rows.append({
            "year": year,
            "authorities": int(len(part)),
            "missing": int(len(missing)),
            "colchester_missing": bool((missing["la_code"] == config.COLCHESTER).any()),
        })
    return {"by_year": rows}


def main() -> int:
    panel = pd.read_parquet(config.PANEL_FILE)

    write("ranking", national_ranking(panel))
    write("trend", trend(panel))
    write("precision", precision_curve())
    write("performance", performance())
    write("colchester", colchester_lsoa())
    write("models", models())
    write("unsupervised", unsupervised())
    write("quality", data_quality(panel))

    # Colchester is missing from the scored set: its 2023-24 return is absent,
    # so the 2023-24 to 2024-25 pair has no baseline and the row is dropped.
    # The headline figures therefore come from the panel directly, and the page
    # is told the authority could not be scored rather than left to discover a
    # missing record at render time.
    colchester = panel[panel["la_code"] == config.COLCHESTER].set_index("financial_year")
    scored = {r["la_code"] for r in national_ranking(panel)}
    design = pd.read_parquet(config.DESIGN_FILE)

    meta = {
        "focus": {
            "la_code": config.COLCHESTER,
            "name": "Colchester",
            "scoreable": config.COLCHESTER in scored,
            "children_first": int(
                colchester.loc[config.YEARS[0], "relief_with_children"]),
            "children_last": int(
                colchester.loc[config.YEARS[-1], "relief_with_children"]),
            "relief_last": int(colchester.loc[config.YEARS[-1], "relief_total"]),
            "missing_years": [
                year for year, row in colchester.iterrows()
                if pd.isna(row["relief_total"])
            ],
        },
        "authorities": int(panel["la_code"].nunique()),
        "authority_years": int(len(panel)),
        "years": sorted(panel["financial_year"].unique()),
        "test_year": config.TEST_OUTCOME_YEAR,
        "features": len(features.feature_columns(design)),
        "sources": [
            {"name": "MHCLG statutory homelessness (H-CLIC), detailed LA tables",
             "url": "https://www.gov.uk/government/statistical-data-sets/live-tables-on-homelessness"},
            {"name": "English Indices of Deprivation 2019",
             "url": "https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019"},
        ],
    }
    write("meta", meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
