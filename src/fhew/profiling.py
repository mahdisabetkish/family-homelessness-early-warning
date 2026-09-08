"""Measure the built artefacts, so the dataset card cannot drift from the data.

Every count in DATASET.md and in the dashboard's Dataset tab is computed here
from the parquet files the pipeline wrote, rather than typed in. The things a
reader needs before trusting a result: how many rows, how many are usable, how
the features break down, how much is missing and where, how the classes
balance, and how the split was made.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .neighbourhoods import FEATURES as LSOA_FEATURES

YEARS = config.YEARS
TRAIN = config.TRAIN_OUTCOME_YEARS
VALID = config.VALID_OUTCOME_YEAR
TEST = config.TEST_OUTCOME_YEAR
OUTCOME = config.OUTCOME

FEATURE_GROUPS = {
    "Current-year rates": lambda c: c.startswith("rate_"),
    "One-year changes": lambda c: c.startswith("d1_"),
    "Two-year change in the outcome": lambda c: c.startswith("d2_"),
    "Composition shares and ratios": lambda c: c.startswith("share_") or c == "prevention_to_relief",
    "Deprivation domains (IoD2019)": lambda c: c.endswith("_score"),
}

MEASURE_THEMES = {
    "Scale of demand (A1)": ("assessments_total", "owed_any_duty", "owed_prevention",
                             "owed_relief", "s21_notice"),
    "Household composition (A5P, A5R)": ("relief_total", "prevention_total",
                                         "relief_with_children", "prevention_with_children"),
    "Reason for loss of home (A2R)": ("loss_",),
    "Support needs (A3)": ("need_",),
    "Accommodation at application (A4R)": ("from_",),
}


def theme_of(measure: str) -> str:
    for theme, keys in MEASURE_THEMES.items():
        if any(measure == k or measure.startswith(k) for k in keys):
            return theme
    return "Other"


def profile() -> dict:
    panel = pd.read_parquet(config.PANEL_FILE)
    design = pd.read_parquet(config.DESIGN_FILE)
    lsoa = pd.read_parquet(config.COLCHESTER_LSOA_FILE)

    measures = [c for c in panel.columns
                if not c.startswith("rate_")
                and not c.endswith("_score")
                and c not in {"la_code", "la_name", "financial_year",
                              "households_in_area", "households_in_area_000"}]
    features = [c for c in design.columns
                if any(test(c) for test in FEATURE_GROUPS.values())]

    # --- rows per year, and how many are usable -----------------------------
    by_year = []
    for year in YEARS:
        rows = panel[panel["financial_year"] == year]
        reporting = rows["relief_total"].notna().sum()
        by_year.append({
            "year": year,
            "authorities": int(len(rows)),
            "reporting": int(reporting),
            "missing": int(len(rows) - reporting),
            "households": float(rows["households_in_area"].sum()),
            "families_homeless": float(rows["relief_with_children"].sum(min_count=1)),
        })

    # --- design matrix, split and labels ------------------------------------
    def split_of(year: str) -> str:
        if year in TRAIN:
            return "train"
        if year == VALID:
            return "validation"
        if year == TEST:
            return "test (held out)"
        return "unused (no lag available)"

    splits = []
    for year in sorted(design["outcome_year"].unique()):
        rows = design[design["outcome_year"] == year]
        level = rows["y_level"].dropna()
        esc = rows["y_escalation"].dropna()
        splits.append({
            "outcome_year": year,
            "feature_year": rows["financial_year"].iloc[0],
            "split": split_of(year),
            "rows": int(len(rows)),
            "level_positive": int(level.sum()),
            "level_prevalence": float(level.mean()),
            "escalation_rows": int(len(esc)),
            "escalation_positive": int(esc.sum()),
            "escalation_prevalence": float(esc.mean()),
        })

    # --- missingness --------------------------------------------------------
    missing = (design[features].isna().mean().sort_values(ascending=False))
    groups = []
    for name, test in FEATURE_GROUPS.items():
        cols = [c for c in features if test(c)]
        groups.append({
            "group": name,
            "features": len(cols),
            "missing_rate": float(design[cols].isna().to_numpy().mean()) if cols else 0.0,
        })

    # --- outcome distribution ----------------------------------------------
    outcome = panel[OUTCOME].dropna()
    quantiles = {f"p{int(q * 100)}": float(outcome.quantile(q))
                 for q in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)}

    raw_files = sorted(config.HCLIC_RAW.glob("*.ods")) + sorted(config.IOD_RAW.glob("*"))

    return {
        "sources": {
            "files": len(raw_files),
            "bytes": int(sum(f.stat().st_size for f in raw_files)),
            "hclic_years": len(YEARS),
            "tables_per_year": list(config.TABLES_READ),
            "columns_extracted_per_year": {
                y: int(pd.read_parquet(config.INTERIM / f"hclic_{y}.parquet").shape[1] - 3)
                for y in YEARS if (config.INTERIM / f"hclic_{y}.parquet").exists()
            },
        },
        "panel": {
            "rows": int(len(panel)),
            "authority_codes": int(panel["la_code"].nunique()),
            "years": len(YEARS),
            "measures": len(measures),
            "measures_by_theme": {
                theme: sorted(m for m in measures if theme_of(m) == theme)
                for theme in list(MEASURE_THEMES) + ["Other"]
            },
            "by_year": by_year,
        },
        "design": {
            "rows": int(len(design)),
            "features": len(features),
            "feature_groups": groups,
            "unit": "one English local authority in one financial year",
            "outcome": OUTCOME,
            "outcome_quantiles": quantiles,
            "outcome_mean": float(outcome.mean()),
            "outcome_n": int(len(outcome)),
            "splits": splits,
            "most_missing": [{"feature": f, "missing_rate": float(v)}
                             for f, v in missing.head(8).items() if v > 0],
            "overall_missing_rate": float(design[features].isna().to_numpy().mean()),
        },
        "neighbourhoods": {
            "lsoas": int(len(lsoa)),
            "population": int(lsoa["population"].sum()),
            "children_0_15": int(lsoa["children_0_15"].sum()),
            "features": len(LSOA_FEATURES),
            "segments": int(lsoa["segment"].nunique()),
            "flagged": int(lsoa["is_anomaly"].sum()),
            "segment_sizes": (lsoa.groupby("segment_name").size()
                              .sort_values(ascending=False).to_dict()),
        },
    }


def write_markdown(p: dict) -> None:
    """A dataset card, in the order a sceptical reader asks the questions."""
    d, panel, src, nb = p["design"], p["panel"], p["sources"], p["neighbourhoods"]
    L = []
    add = L.append

    add("# Dataset")
    add("")
    add("Generated by `src/08_data_profile.py`. Every count below is measured from")
    add("the built artefacts, so it cannot drift from the data it describes.")
    add("")
    add("## At a glance")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| Unit of observation | {d['unit']} |")
    add(f"| Panel rows | **{panel['rows']:,}** authority-years |")
    add(f"| Distinct authority codes | {panel['authority_codes']} |")
    add(f"| Financial years | {panel['years']} ({', '.join(YEARS[0:1])} to {YEARS[-1]}) |")
    add(f"| Measures extracted | {panel['measures']} per authority-year |")
    add(f"| Modelling rows (after lagging) | **{d['rows']:,}** |")
    add(f"| Features | **{d['features']}** |")
    add(f"| Overall missing rate | {d['overall_missing_rate']:.1%} |")
    add(f"| Source files | {src['files']} ({src['bytes'] / 1e6:.0f} MB) |")
    add("")
    add("The modelling table is smaller than the panel because every row needs a")
    add("following year to supply its label, and the earliest year has no")
    add("preceding year to difference against.")
    add("")

    add("## Rows per year")
    add("")
    add("| Financial year | Authorities | Filed a return | Missing | Households in scope | Families with children made homeless |")
    add("|---|---:|---:|---:|---:|---:|")
    for r in panel["by_year"]:
        fam = "n/a" if r["families_homeless"] is None or np.isnan(r["families_homeless"]) \
            else f"{r['families_homeless']:,.0f}"
        add(f"| {r['year']} | {r['authorities']} | {r['reporting']} | {r['missing']} "
            f"| {r['households']:,.0f} | {fam} |")
    add("")
    add("Missing means the authority filed no H-CLIC return that year. It is never")
    add("recoded to zero.")
    add("")

    add("## Features")
    add("")
    add(f"{d['features']} features, all measured in the feature year, grouped as follows.")
    add("")
    add("| Group | Features | Missing |")
    add("|---|---:|---:|")
    for g in d["feature_groups"]:
        add(f"| {g['group']} | {g['features']} | {g['missing_rate']:.1%} |")
    add("")
    add("The underlying measures, before differencing and rate conversion:")
    add("")
    for theme, items in panel["measures_by_theme"].items():
        if items:
            add(f"- **{theme}** ({len(items)}): `" + "`, `".join(items) + "`")
    add("")

    if d["most_missing"]:
        add("### Where the missing values are")
        add("")
        add("| Feature | Missing |")
        add("|---|---:|")
        for m in d["most_missing"]:
            add(f"| `{m['feature']}` | {m['missing_rate']:.1%} |")
        add("")
        add("Two causes, and they are different. An authority that filed no return is")
        add("missing everything for that year. A measure MHCLG had not yet introduced")
        add("is missing for every authority in the early years, which is why the")
        add("one-year changes are missing more often than the levels they derive from.")
        add("")

    add("## Outcome")
    add("")
    add(f"`{d['outcome']}`: families with dependent children newly owed a relief duty,")
    add("meaning already homeless rather than threatened with homelessness, per 1,000")
    add("resident households.")
    add("")
    add("| Statistic | Value |")
    add("|---|---:|")
    add(f"| Observations | {d['outcome_n']:,} |")
    add(f"| Mean | {d['outcome_mean']:.2f} |")
    for k, v in d["outcome_quantiles"].items():
        label = {"p0": "Minimum", "p50": "Median", "p100": "Maximum"}.get(k, k.replace("p", "") + "th percentile")
        add(f"| {label} | {v:.2f} |")
    add("")

    add("## Labels and the split")
    add("")
    add("Features come from year *t*, labels from year *t+1* for the same authority.")
    add("")
    add("| Feature year | Outcome year | Split | Rows | Level positives | Escalation rows | Escalation positives |")
    add("|---|---|---|---:|---:|---:|---:|")
    for s in d["splits"]:
        add(f"| {s['feature_year']} | {s['outcome_year']} | {s['split']} | {s['rows']} "
            f"| {s['level_positive']} ({s['level_prevalence']:.0%}) "
            f"| {s['escalation_rows']} "
            f"| {s['escalation_positive']} ({s['escalation_prevalence']:.0%}) |")
    add("")
    add("Escalation has fewer usable rows than level because a relative change is")
    add("undefined where the current rate is zero.")
    add("")
    add("Level prevalence drifts upward across the years rather than sitting at 20%.")
    add("That is the intended behaviour: the cutoff is a quantile of the training")
    add("years, not of the year being scored, so a genuinely worsening national")
    add("picture shows up as rising prevalence instead of being normalised away.")
    add("")

    add("## Neighbourhood layer")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| LSOAs (Colchester) | {nb['lsoas']} |")
    add(f"| Population | {nb['population']:,} |")
    add(f"| Children aged 0-15 | {nb['children_0_15']:,} |")
    add(f"| Features | {nb['features']} deprivation domain and sub-domain scores |")
    add(f"| Segments (k-means) | {nb['segments']} |")
    add(f"| Flagged by Isolation Forest | {nb['flagged']} |")
    add("")
    add("| Segment | LSOAs |")
    add("|---|---:|")
    for name, n in nb["segment_sizes"].items():
        add(f"| {name} | {n} |")
    add("")

    add("## What is not in the data")
    add("")
    add("No personal or household-level records. These are published counts, so the")
    add("smallest unit anywhere in this project is a local authority in a year, or a")
    add("neighbourhood of roughly 1,500 people. Nothing here identifies a household,")
    add("and nothing here could.")
    add("")
    add("See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for the definition of every field.")

    (config.ROOT / "DATASET.md").write_text("\n".join(L) + "\n")

