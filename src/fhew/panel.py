"""Turn the published H-CLIC extracts into one modelling panel.

Two things here are the real work rather than plumbing.

**Column wording drifts between releases.** MHCLG reworded and added measures
over the seven years: support needs gained 'Difficulties budgeting' in
2022-23, and 'No fixed abode' replaced a rough-sleeping split. Each measure
therefore carries a list of alternative header patterns tried in order, and
the coverage of every measure in every year is reported, so a gap shows up as
a data-quality fact rather than a silent column of zeros.

**Local authorities merged during the period.** IoD2019 is published on 2019
boundaries; by 2024-25 seven successor authorities had replaced twenty-five
predecessor districts, and 2018-19 H-CLIC still reports fourteen districts
abolished that April. Both directions are reconciled here so the later years
are not simply dropped.
"""
from __future__ import annotations

import pandas as pd

from . import config

# measure -> list of alternative (table, *header substrings) patterns, tried in
# order. Alternatives exist because the published wording changed.
MEASURES: dict[str, list[tuple[str, ...]]] = {
    "assessments_total":      [("A1", "total initial assessments"),
                               ("A1", "total number of households assessed")],
    "owed_any_duty":          [("A1", "total owed a prevention or relief duty"),
                               ("A1", "assessed as owed a duty"),
                               ("A1", "total households assessed as owed a duty")],
    "owed_prevention":        [("A1", "prevention duty owed", "assessed as"),
                               ("A1", "threatened with homelessness", "prevention duty owed")],
    "owed_relief":            [("A1", "already homeless", "relief duty owed"),
                               ("A1", "homeless - relief duty owed")],
    "s21_notice":             [("A1", "section 21")],
    "households_in_area_000": [("A1", "number of households in area")],

    "relief_total":           [("A5R", "total owed a relief duty")],
    "prevention_total":       [("A5P", "total owed a prevention duty")],

    "loss_end_ast":           [("A2R", "total end of ast"),
                               ("A2R", "end of assured shorthold")],
    "loss_rent_arrears":      [("A2R", "rent arrears, due to"), ("A2R", "rent arrears")],
    "loss_landlord_selling":  [("A2R", "landlord wishing to sell or re-let")],
    "loss_family_friends":    [("A2R", "family or friends no longer willing")],
    "loss_domestic_abuse":    [("A2R", "total domestic abuse"), ("A2R", "domestic abuse")],
    "loss_social_tenancy":    [("A2R", "total end of social rented tenancy"),
                               ("A2R", "end of social rented tenancy")],
    "loss_supported_housing": [("A2R", "total evicted from supported housing"),
                               ("A2R", "eviction from supported housing")],
    "loss_institution":       [("A2R", "total departure from institution"),
                               ("A2R", "left institution with no accommodation"),
                               ("A2R", "departure from institution")],

    "need_any":               [("A3", "total households with support needs"),
                               ("A3", "households with one or more support needs")],
    "need_mental_health":     [("A3", "history of mental health problems")],
    "need_physical_health":   [("A3", "physical ill health and disability")],
    "need_domestic_abuse":    [("A3", "has experienced domestic abuse")],
    "need_drug":              [("A3", "drug dependency")],
    "need_alcohol":           [("A3", "alcohol dependency")],
    "need_offending":         [("A3", "offending history")],
    "need_repeat_homeless":   [("A3", "history of repeat homelessness")],
    "need_rough_sleeping":    [("A3", "history of rough sleeping")],
    "need_care_leaver_18_20": [("A3", "care leaver aged 18-20")],
    "need_young_parent":      [("A3", "young parent requiring support")],
    "need_learning_disab":    [("A3", "learning disability")],

    "from_private_rented":    [("A4R", "total prs"), ("A4R", "private rented sector")],
    "from_social_rented":     [("A4R", "total srs"), ("A4R", "social rented sector")],
    "from_family":            [("A4R", "living with family")],
    "from_friends":           [("A4R", "living with friends")],
    "from_temp_accom":        [("A4R", "temporary accommodation")],
    "from_rough_sleeping":    [("A4R", "rough sleeping")],
}

# Household types containing dependent children are published split by sex of
# the main applicant, so they are summed rather than picked.
CHILD_ROLLUPS = {
    "relief_with_children":     ("A5R", "with dependent children", "without dependent children"),
    "prevention_with_children": ("A5P", "with dependent children", "without dependent children"),
}

# Predecessor districts -> successor authority, for reorganisations after the
# 2019 boundaries on which IoD2019 is published.
# Source: ONS register of local government reorganisations, 2020-2023.
SUCCESSORS: dict[str, list[str]] = {
    # 2020: Buckinghamshire
    "E06000060": ["E07000004", "E07000005", "E07000006", "E07000007"],
    # 2021: Northamptonshire split into two unitaries
    "E06000061": ["E07000150", "E07000152", "E07000153", "E07000156"],
    "E06000062": ["E07000151", "E07000154", "E07000155"],
    # 2023: Cumbria split into two unitaries
    "E06000063": ["E07000026", "E07000028", "E07000029"],
    "E06000064": ["E07000027", "E07000030", "E07000031"],
    # 2023: North Yorkshire
    "E06000065": ["E07000163", "E07000164", "E07000165", "E07000166",
                  "E07000167", "E07000168", "E07000169"],
    # 2023: Somerset (E07000246 was itself a 2019 merger, so present in IoD2019)
    "E06000066": ["E07000187", "E07000188", "E07000189", "E07000246"],
}

# The reverse problem: 2018-19 H-CLIC predates the April 2019 mergers, so it
# reports districts that IoD2019 (published on post-merger boundaries) does
# not contain. Each is given its successor's deprivation score - the scores
# are a static contextual covariate, so this is a boundary reconciliation
# rather than an estimate.
PREDECESSOR_OF_2019: dict[str, str] = {
    # Bournemouth, Christchurch and Poole
    "E06000028": "E06000058", "E07000048": "E06000058", "E06000029": "E06000058",
    # Dorset
    "E07000049": "E06000059", "E07000050": "E06000059", "E07000051": "E06000059",
    "E07000052": "E06000059", "E07000053": "E06000059",
    # East Suffolk
    "E07000205": "E07000244", "E07000206": "E07000244",
    # West Suffolk
    "E07000201": "E07000245", "E07000204": "E07000245",
    # Somerset West and Taunton
    "E07000190": "E07000246", "E07000191": "E07000246",
}

# Sheet in the IoD2019 district summaries workbook -> panel column.
DOMAIN_SHEETS = {
    "IMD": "imd_score", "Income": "income_score", "Employment": "employment_score",
    "Education": "education_score", "Health": "health_score", "Crime": "crime_score",
    "Barriers": "barriers_score", "Living": "living_env_score", "IDACI": "idaci_score",
}

# LSOA-level column in the IoD2019 scores file -> panel column, used when a
# successor authority's score has to be rebuilt from its predecessors.
LSOA_SCORE_COLUMNS = {
    "Index of Multiple Deprivation (IMD) Score": "imd_score",
    "Income Score (rate)": "income_score",
    "Employment Score (rate)": "employment_score",
    "Education, Skills and Training Score": "education_score",
    "Health Deprivation and Disability Score": "health_score",
    "Crime Score": "crime_score",
    "Barriers to Housing and Services Score": "barriers_score",
    "Living Environment Score": "living_env_score",
    "Income Deprivation Affecting Children Index (IDACI) Score (rate)": "idaci_score",
}

LSOA_DISTRICT_CODE = "Local Authority District code (2019)"
LSOA_POPULATION = "Total population: mid 2015 (excluding prisoners)"


def find_column(columns: list[str], table: str, *needles: str) -> str | None:
    """First column from `table` whose header contains every needle."""
    prefix = f"{table}::"
    hits = [c for c in columns
            if c.startswith(prefix) and all(n in c.lower() for n in needles)]
    if not hits:
        return None
    # Shortest label wins: deeper headers are more specific sub-breakdowns.
    return min(hits, key=len)


def select_measures(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    """Map published columns onto model measure names for one year.

    Returns the selected frame and, alongside it, which published header each
    measure resolved to. The second return value is what makes the selection
    auditable: a measure that silently failed to match is a `None` here, not a
    column of zeros downstream.
    """
    columns = list(frame.columns)
    out = frame[["la_code", "la_name", "financial_year"]].copy()
    resolved: dict[str, str | None] = {}

    for measure, patterns in MEASURES.items():
        column = None
        for table, *needles in patterns:
            column = find_column(columns, table, *[n.lower() for n in needles])
            if column is not None:
                break
        resolved[measure] = column
        out[measure] = frame[column] if column is not None else pd.NA

    for measure, (table, include, exclude) in CHILD_ROLLUPS.items():
        prefix = f"{table}::"
        cols = [c for c in columns
                if c.startswith(prefix) and include in c.lower() and exclude not in c.lower()]
        resolved[measure] = f"{len(cols)} columns summed" if cols else None
        out[measure] = frame[cols].sum(axis=1, min_count=1) if cols else pd.NA

    return out, resolved


def all_measures() -> list[str]:
    """Every measure name the panel carries, in a stable order."""
    return list(MEASURES) + list(CHILD_ROLLUPS)


def coverage_grid(coverage: dict[str, dict[str, str | None]],
                  years: list[str] | None = None) -> tuple[list[str], int]:
    """Render the measure-by-year coverage grid, and count the gaps.

    A dot means the measure resolved to a published column that year, an X
    that the release does not carry it. The count is returned so a stage can
    fail loudly if coverage ever collapses.
    """
    years = years or config.YEARS
    measures = all_measures()
    lines = [f"  {'measure':<24}" + "".join(f"{y[2:]:>7}" for y in years)]
    gaps = 0
    for measure in measures:
        marks = ""
        for year in years:
            found = coverage[year][measure] is not None
            gaps += 0 if found else 1
            marks += f"{'.' if found else 'X':>7}"
        lines.append(f"  {measure:<24}{marks}")
    lines.append(f"  {gaps} measure-years missing out of {len(measures) * len(years)}")
    return lines, gaps


def deprivation_on_2019_boundaries() -> pd.DataFrame:
    """Local-authority deprivation scores as published (2019 boundaries)."""
    book = pd.ExcelFile(config.IOD_RAW / "iod2019_lad_summaries.xlsx")
    out: pd.DataFrame | None = None
    for sheet, name in DOMAIN_SHEETS.items():
        frame = book.parse(sheet)
        frame.columns = [" ".join(str(c).split()) for c in frame.columns]
        code = next(c for c in frame.columns if "code" in c.lower())
        score = next(c for c in frame.columns if "average score" in c.lower())
        part = frame[[code, score]].rename(columns={code: "la_code", score: name})
        out = part if out is None else out.merge(part, on="la_code", how="outer")
    assert out is not None
    return out


def rebuild_successors(lad: pd.DataFrame,
                       log: list[str] | None = None) -> pd.DataFrame:
    """Add deprivation rows for authorities created after 2019.

    Successor scores are population-weighted means of their predecessors'
    LSOAs, which reproduces how the published district summaries are
    constructed. Districts abolished in April 2019 then inherit their
    successor's score, so 2018-19 is not lost.
    """
    log = log if log is not None else []
    lsoa = pd.read_csv(config.IOD_RAW / "iod2019_lsoa_all.csv")
    lsoa.columns = [" ".join(str(c).split()) for c in lsoa.columns]

    missing = [c for c in LSOA_SCORE_COLUMNS if c not in lsoa.columns]
    assert not missing, f"IoD2019 LSOA columns not found: {missing}"

    rows = []
    for successor, predecessors in SUCCESSORS.items():
        part = lsoa[lsoa[LSOA_DISTRICT_CODE].isin(predecessors)]
        if part.empty:
            log.append(f"    ! no LSOAs found for successor {successor}")
            continue
        weights = part[LSOA_POPULATION]
        row = {"la_code": successor}
        for source, name in LSOA_SCORE_COLUMNS.items():
            row[name] = float((part[source] * weights).sum() / weights.sum())
        rows.append(row)

    rebuilt = pd.DataFrame(rows)
    log.append(f"  rebuilt deprivation for {len(rebuilt)} successor authorities "
               f"from {sum(len(v) for v in SUCCESSORS.values())} predecessor districts")
    lad = pd.concat([lad, rebuilt], ignore_index=True)

    inherited = (
        pd.DataFrame({"la_code": list(PREDECESSOR_OF_2019),
                      "_successor": list(PREDECESSOR_OF_2019.values())})
        .merge(lad.rename(columns={"la_code": "_successor"}), on="_successor", how="left")
        .drop(columns="_successor")
    )
    log.append(f"  mapped {len(inherited)} pre-2019 districts onto their successors")
    return pd.concat([lad, inherited], ignore_index=True)


def add_rates(panel: pd.DataFrame) -> pd.DataFrame:
    """Express counts per 1,000 resident households.

    Raw counts are dominated by authority size; the denominator MHCLG
    publishes in table A1 is the number of households in the area, in
    thousands, so dividing by it directly gives a rate per 1,000.
    """
    households = pd.to_numeric(panel["households_in_area_000"], errors="coerce")
    panel = panel.copy()
    panel["households_in_area"] = households * 1_000

    count_cols = [c for c in MEASURES if c != "households_in_area_000"]
    count_cols += list(CHILD_ROLLUPS)
    for column in count_cols:
        values = pd.to_numeric(panel[column], errors="coerce")
        panel[f"rate_{column}"] = values / households  # per 1,000 households
    return panel


def build(years: list[str] | None = None) -> tuple[pd.DataFrame, dict, list[str]]:
    """Assemble the full panel from the cached per-year extracts.

    Returns the panel, the per-year measure resolution, and a log of what the
    boundary reconciliation did, so the caller decides how to report it.
    """
    years = years or config.YEARS
    frames, coverage = [], {}
    for year in years:
        raw = pd.read_parquet(config.INTERIM / f"hclic_{year}.parquet")
        selected, resolved = select_measures(raw)
        frames.append(selected)
        coverage[year] = resolved
    panel = pd.concat(frames, ignore_index=True)

    log: list[str] = []
    lad = deprivation_on_2019_boundaries()
    log.append(f"  IoD2019 district summaries: {len(lad)} authorities (2019 boundaries)")
    lad = rebuild_successors(lad, log)

    panel = panel.merge(lad, on="la_code", how="left")
    unmatched = panel[panel["imd_score"].isna()]
    if len(unmatched):
        log.append(f"  ! {unmatched['la_code'].nunique()} authorities still unmatched: "
                   f"{sorted(unmatched['la_name'].unique())[:6]}")
    else:
        log.append("  all authority-years matched to a deprivation score")

    return add_rates(panel), coverage, log
