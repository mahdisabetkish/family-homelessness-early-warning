"""Stage 05 - Figures for the README and dashboard.

Seven figures, one per claim the analysis makes. The house style, the palette and
the save helper live in `fhew.plotting`, so the dashboard and the README
cannot end up on different colours; what stays here is the composition of
each individual chart, which is the part that is genuinely one-off.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.calibration import calibration_curve

from fhew import config, features
from fhew.models import make_models
from fhew.plotting import (
    CRITICAL,
    GRID,
    INK,
    INK_2,
    INK_MUTED,
    NEUTRAL,
    SERIES,
    SURFACE,
    save,
    style,
    use_house_style,
)

warnings.filterwarnings("ignore")
use_house_style()

# matplotlib is configured by fhew.plotting before pyplot is imported here.
import matplotlib.pyplot as plt  # noqa: E402


# --- figure 1: the trend the project exists to change -----------------------
def figure_trend(panel: pd.DataFrame) -> None:
    """Families with children owed a relief duty, per 1,000 households."""
    essex_codes = [f"E070000{n}" for n in range(66, 78)]
    years = sorted(panel["financial_year"].unique())

    def series(mask: pd.Series) -> np.ndarray:
        """Pooled rate for a group of authorities.

        The numerator uses min_count=1 so a year in which every authority in
        the group is missing stays missing. Summing it to zero would draw a
        collapse in family homelessness where there is really a reporting gap:
        Colchester made no H-CLIC return in 2023-24.
        """
        part = panel[mask]
        grouped = part.groupby("financial_year")
        total = grouped["relief_with_children"].sum(min_count=1)
        households = (grouped
                      .apply(lambda g: g.loc[g["relief_with_children"].notna(),
                                             "households_in_area"].sum())
                      / 1_000)
        return (total / households.replace(0, np.nan)).reindex(years).to_numpy()

    lines = [
        ("England", series(pd.Series(True, index=panel.index)), SERIES[0]),
        ("Essex districts", series(panel["la_code"].isin(essex_codes)), SERIES[1]),
        ("Colchester", series(panel["la_code"] == "E07000071"), SERIES[2]),
    ]

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    x = np.arange(len(years))
    offsets = {"England": 0, "Essex districts": 7, "Colchester": -7}
    for label, values, colour in lines:
        ax.plot(x, values, color=colour, linewidth=2.0, marker="o",
                markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.2,
                zorder=3, label=label)
        gaps = np.isnan(values)
        if gaps.any():
            # Bridge the gap with a dotted segment so the reader sees that the
            # series continues and that the year is unobserved, not zero. The
            # hollow marker sits where the value would have been interpolated,
            # which is precisely the number nobody should quote.
            observed = ~gaps
            ax.plot(x[observed], values[observed], color=colour, linewidth=1.1,
                    linestyle=(0, (2, 2)), zorder=2)
            bridged = np.interp(x[gaps], x[observed], values[observed])
            ax.scatter(x[gaps], bridged, s=48, facecolor=SURFACE,
                       edgecolor=colour, linewidth=1.4, zorder=4)
        last = values[~np.isnan(values)][-1]
        # Direct label at the line end: identity never rests on colour alone.
        ax.annotate(f"{label}  {last:.2f}", (x[-1], last),
                    xytext=(8, offsets[label]), textcoords="offset points",
                    color=INK_2, fontsize=8.5, va="center")

    colchester = lines[2][1]
    gap = np.isnan(colchester)
    if gap.any():
        observed = ~gap
        height = float(np.interp(x[gap], x[observed], colchester[observed])[0])
        ax.annotate("no H-CLIC return\nfrom Colchester", (float(x[gap][0]), height),
                    xytext=(-6, -38), textcoords="offset points", ha="center",
                    fontsize=7.8, color=CRITICAL,
                    arrowprops=dict(arrowstyle="-", color=CRITICAL, linewidth=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels([y.replace("-", "–") for y in years])
    ax.set_xlim(-0.3, len(years) + 1.9)
    ax.set_ylim(0.55, None)
    ax.set_ylabel("per 1,000 resident households")
    ax.set_title("Families with dependent children newly assessed as homeless",
                 loc="left", pad=10)
    # Lower left is the only region no series passes through.
    ax.legend(loc="lower left", ncol=1, handlelength=1.6, borderpad=0.2,
              labelspacing=0.35)
    style(ax)
    save(fig, "fig_trend")


# --- figure 2: where prediction earns its place -----------------------------
def figure_tasks() -> None:
    """Task A is a lookup; Task B is the one worth modelling."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.08})
    panels = [
        ("y_level", "Task A · LEVEL", "worst quintile next year", axes[0]),
        ("y_escalation", "Task B · ESCALATION", "25%+ rise next year", axes[1]),
    ]
    keep = ["Persistence baseline", "Logistic regression",
            "Gradient boosting (calibrated)"]
    short = {"Persistence baseline": "Last year's value",
             "Logistic regression": "Logistic regression",
             "Gradient boosting (calibrated)": "Gradient boosting"}

    for task, title, subtitle, ax in panels:
        result = pd.read_csv(config.TABLES / f"performance_{task}.csv")
        result = result[result["model"].isin(keep)].set_index("model").loc[keep]
        y = np.arange(len(result))[::-1]
        # The baseline is a reference, not a competitor: neutral, not a hue.
        colours = [NEUTRAL] + [SERIES[0]] * (len(result) - 1)

        ax.barh(y, result["roc_auc"], height=0.46, color=colours, zorder=3)
        errors = np.vstack([result["roc_auc"] - result["roc_auc_lo"],
                            result["roc_auc_hi"] - result["roc_auc"]])
        ax.errorbar(result["roc_auc"], y, xerr=errors, fmt="none",
                    ecolor=INK_2, elinewidth=1.1, capsize=2.5, zorder=4)
        for yi, (value, high) in enumerate(
                zip(result["roc_auc"], result["roc_auc_hi"], strict=True)):
            ax.annotate(f"{value:.2f}", (high, y[yi]), xytext=(6, 0),
                        textcoords="offset points", va="center",
                        color=INK, fontsize=8.5, fontweight="bold")

        ax.axvline(0.5, color=CRITICAL, linewidth=1.1, linestyle=(0, (4, 3)), zorder=2)
        ax.set_yticks(y)
        ax.set_xlim(0.2, 1.12)
        ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("ROC-AUC, held-out 2024–25")
        ax.set_title(title, loc="left", pad=16, fontsize=10)
        ax.annotate(subtitle, (0, 1.0), xycoords="axes fraction",
                    xytext=(0, 6), textcoords="offset points",
                    fontsize=8.5, color=INK_2)
        ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)

    axes[0].set_yticklabels([short[m] for m in keep], fontsize=8.5)
    fig.text(0.5, -0.10,
             "Dashed line marks chance (0.5); bars carry 95% bootstrap intervals. "
             "Last year's value already answers Task A; on Task B it is "
             "significantly worse than chance.",
             fontsize=7.8, color=INK_MUTED, ha="center")
    save(fig, "fig_tasks")


# --- figure 3: the metric that matches how a list is worked -----------------
def figure_precision_at_k() -> None:
    """Precision as a function of how many authorities a team can take on."""
    predictions = pd.read_csv(config.TABLES / "predictions_y_escalation.csv")
    truth = predictions["y_escalation"].to_numpy()
    base_rate = truth.mean()
    ks = np.arange(5, 121, 5)

    def curve(scores: np.ndarray) -> list[float]:
        order = np.argsort(-scores)
        return [truth[order[:k]].mean() for k in ks]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for label, column, colour in [
        ("Model ranking", "Gradient boosting (calibrated)", SERIES[0]),
        ("Last year's change", "persistence", SERIES[1]),
    ]:
        values = curve(predictions[column].to_numpy())
        ax.plot(ks, values, color=colour, linewidth=2.0, zorder=3, label=label)
        ax.annotate(label, (ks[-1], values[-1]), xytext=(8, 0),
                    textcoords="offset points", color=INK_2,
                    fontsize=8.5, va="center")

    ax.axhline(base_rate, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(f"pick at random: {base_rate:.0%}", (ks[0], base_rate),
                xytext=(0, 9), textcoords="offset points",
                color=INK_MUTED, fontsize=8)

    # Mark the capacity the reported P@k uses, so the table and the curve agree.
    at_30 = curve(predictions["Gradient boosting (calibrated)"].to_numpy())[
        list(ks).index(30)]
    ax.scatter([30], [at_30], s=70, facecolor=SURFACE, edgecolor=SERIES[0],
               linewidth=2.0, zorder=5)
    ax.annotate(f"k = 30: {at_30:.0%} vs {base_rate:.0%}", (30, at_30),
                xytext=(10, 12), textcoords="offset points",
                color=INK, fontsize=8.5, fontweight="bold")

    ax.set_xlabel("authorities the team can take on (k)")
    ax.set_ylabel("share that really did escalate")
    ax.set_ylim(0, 0.62)
    ax.set_xlim(0, 158)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Hit rate at the top of the list, held-out 2024–25",
                 loc="left", pad=10)
    ax.legend(loc="upper right", handlelength=1.6)
    style(ax)
    save(fig, "fig_precision_at_k")


# --- figure 4: are the probabilities believable? ----------------------------
def figure_calibration() -> None:
    """A score used to triage must mean what it says."""
    predictions = pd.read_csv(config.TABLES / "predictions_y_escalation.csv")
    truth = predictions["y_escalation"].to_numpy()

    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.plot([0, 0.6], [0, 0.6], color=INK_MUTED, linewidth=1.1,
            linestyle=(0, (4, 3)), zorder=2, label="perfect calibration")
    for label, column, colour in [
        ("Calibrated", "Gradient boosting (calibrated)", SERIES[0]),
        ("Uncalibrated", "Gradient boosting", SERIES[1]),
    ]:
        observed, predicted = calibration_curve(
            truth, predictions[column], n_bins=5, strategy="quantile")
        ax.plot(predicted, observed, color=colour, linewidth=2.0, marker="o",
                markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2,
                zorder=3, label=label)

    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Reliability", loc="left", pad=10)
    ax.legend(loc="upper left", handlelength=1.6)
    style(ax)
    save(fig, "fig_calibration")


# --- figure 5: what the model keys on ---------------------------------------
def figure_drivers(design: pd.DataFrame) -> None:
    """Which features the escalation model leans on, and in which direction.

    Standardised logistic coefficients: every feature enters the pipeline on a
    common scale, so the coefficients are directly comparable in size. The
    logistic model is the one used for this chart because it is also the best
    of the four on this task, so the attribution describes a model worth
    attributing rather than only the most convenient one.
    """
    # The same feature list and the same estimator the backtest fits, taken
    # from the model zoo rather than rebuilt here, so the chart cannot end up
    # explaining a model nobody ran.
    columns = features.feature_columns(design)
    labelled = design[design["y_escalation"].notna()]
    fit = labelled[labelled["outcome_year"] <= config.VALID_OUTCOME_YEAR]

    pipeline = make_models()["Logistic regression"].fit(
        fit[columns], fit["y_escalation"].astype(int))

    coefficients = pd.Series(
        pipeline.named_steps["clf"].coef_[0], index=columns)
    top = coefficients.reindex(coefficients.abs().sort_values(ascending=False).index)[:12]
    top = top[::-1]

    # Published measure names, keyed on the base measure. Prefixes for
    # differences and the deprivation suffix are handled generically so a new
    # feature never renders as a raw column name.
    BASE = {
        "relief_with_children": "families with children homeless",
        "prevention_with_children": "families with children in prevention",
        "relief_total": "households owed a relief duty",
        "prevention_total": "households owed a prevention duty",
        "owed_prevention": "assessed as owed a prevention duty",
        "owed_relief": "assessed as owed a relief duty",
        "owed_any_duty": "assessed as owed any duty",
        "assessments_total": "homelessness assessments",
        "s21_notice": "Section 21 no-fault evictions",
        "loss_end_ast": "private tenancy ended",
        "loss_rent_arrears": "home lost through rent arrears",
        "loss_landlord_selling": "landlord selling or re-letting",
        "loss_family_friends": "family or friends could not accommodate",
        "loss_domestic_abuse": "home lost through domestic abuse",
        "loss_social_tenancy": "social tenancy ended",
        "loss_supported_housing": "evicted from supported housing",
        "loss_institution": "left an institution with nowhere to go",
        "need_any": "households with a support need",
        "need_mental_health": "support need: mental health",
        "need_physical_health": "support need: physical health",
        "need_domestic_abuse": "support need: domestic abuse",
        "need_drug": "support need: drug dependency",
        "need_alcohol": "support need: alcohol dependency",
        "need_offending": "support need: offending history",
        "need_repeat_homeless": "support need: repeat homelessness",
        "need_rough_sleeping": "support need: rough sleeping",
        "need_care_leaver_18_20": "support need: care leaver 18-20",
        "need_young_parent": "support need: young parent",
        "need_learning_disab": "support need: learning disability",
        "from_private_rented": "applied from private rented",
        "from_social_rented": "applied from social rented",
        "from_family": "applied from living with family",
        "from_friends": "applied from living with friends",
        "from_temp_accom": "applied from temporary accommodation",
        "from_rough_sleeping": "applied from rough sleeping",
        "share_relief_children": "share of homeless households with children",
        "share_prevention_children": "share in prevention with children",
        "prevention_to_relief": "prevention-to-relief ratio",
        "imd": "deprivation (IMD)", "idaci": "child income deprivation (IDACI)",
        "income": "income deprivation", "employment": "employment deprivation",
        "education": "education deprivation", "health": "health deprivation",
        "crime": "crime deprivation", "barriers": "barriers to housing",
        "living_env": "living environment",
    }

    def label_of(feature: str) -> str:
        """Turn a column name into something a non-technical reader can read."""
        name, prefix = feature, ""
        if name.startswith("d1_"):
            name, prefix = name[3:], "1yr change in "
        elif name.startswith("d2_"):
            name, prefix = name[3:], "2yr change in "
        name = name.removeprefix("rate_").removesuffix("_score")
        text = prefix + BASE.get(name, name.replace("_", " "))
        return text[0].upper() + text[1:]

    labels = [label_of(f) for f in top.index]

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    y = np.arange(len(top))
    # Direction is the meaning here, so a diverging pair with a neutral zero.
    colours = [SERIES[0] if v > 0 else SERIES[1] for v in top]
    ax.barh(y, top.to_numpy(), height=0.62, color=colours, zorder=3)
    ax.axvline(0, color=INK_MUTED, linewidth=0.9, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("standardised log-odds contribution")
    ax.set_title("What moves the escalation score", loc="left", pad=10)
    handles = [Line2D([0], [0], color=SERIES[0], linewidth=6),
               Line2D([0], [0], color=SERIES[1], linewidth=6)]
    ax.legend(handles, ["raises risk", "lowers risk"], loc="lower right",
              handlelength=1.0)
    ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    save(fig, "fig_drivers")


# --- figure 6: the neighbourhood layer --------------------------------------
def figure_colchester() -> None:
    """Segments carry order, so they use a sequential ramp, not eight hues."""
    lsoa = pd.read_parquet(config.COLCHESTER_LSOA_FILE)
    order = lsoa.groupby("segment")["idaci"].mean().sort_values().index
    ramp = plt.get_cmap("Blues")(np.linspace(0.30, 0.92, len(order)))
    colour_of = {segment: ramp[i] for i, segment in enumerate(order)}

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for segment in order:
        part = lsoa[lsoa["segment"] == segment]
        ax.scatter(part["imd_decile_in_england"], part["idaci"],
                   s=part["children_0_15"] / 3.2, color=colour_of[segment],
                   edgecolor=SURFACE, linewidth=0.9, zorder=3,
                   label=f"segment {list(order).index(segment) + 1}")

    flagged = lsoa[lsoa["is_anomaly"]]
    ax.scatter(flagged["imd_decile_in_england"], flagged["idaci"],
               s=flagged["children_0_15"] / 3.2, facecolor="none",
               edgecolor=CRITICAL, linewidth=1.6, zorder=4)
    for _, row in flagged.nlargest(4, "anomaly_score").iterrows():
        ax.annotate(row["lsoa_name"].replace("Colchester ", ""),
                    (row["imd_decile_in_england"], row["idaci"]),
                    xytext=(9, 5), textcoords="offset points",
                    fontsize=7.6, color=INK_2)

    ax.set_xlabel("national deprivation decile  (1 = most deprived)")
    ax.set_ylabel("child income deprivation (IDACI)")
    ax.set_xticks(range(1, 11))
    ax.set_title("Colchester neighbourhoods: 105 LSOAs, segment and anomaly",
                 loc="left", pad=10)
    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=7,
                      markerfacecolor=colour_of[s], markeredgecolor=SURFACE)
               for s in order]
    handles.append(Line2D([0], [0], marker="o", linestyle="none", markersize=8,
                          markerfacecolor="none", markeredgecolor=CRITICAL,
                          markeredgewidth=1.6))
    labels = [f"segment {i + 1}" for i in range(len(order))] + ["flagged as unusual"]
    ax.legend(handles, labels, loc="upper right", ncol=2, handlelength=1.0,
              columnspacing=1.0)
    fig.text(0.0, -0.04,
             "Marker area is the number of children aged 0–15. Segments are "
             "ordered by child income deprivation.", fontsize=7.8, color=INK_MUTED)
    style(ax)
    save(fig, "fig_colchester")


# --- figure 7: what the source data does not contain ------------------------
def figure_data_quality(panel: pd.DataFrame) -> None:
    """Non-reporting is a first-class result, not a footnote.

    Any framework built on these returns has to state which authorities are
    missing in which year, because a silent zero looks exactly like success.
    """
    years = sorted(panel["financial_year"].unique())
    counts, totals = [], []
    for year in years:
        part = panel[panel["financial_year"] == year]
        totals.append(len(part))
        counts.append(int(part["relief_total"].isna().sum()))

    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    x = np.arange(len(years))
    ax.bar(x, counts, width=0.6, color=SERIES[1], zorder=3)
    for xi, count in enumerate(counts):
        ax.annotate(f"{count}", (xi, count), xytext=(0, 4),
                    textcoords="offset points", ha="center",
                    color=INK, fontsize=8.5, fontweight="bold")

    colchester = panel[(panel["la_code"] == "E07000071")
                       & panel["relief_total"].isna()]["financial_year"].tolist()
    for year in colchester:
        ax.annotate("includes\nColchester", (years.index(year), counts[years.index(year)]),
                    xytext=(26, 20), textcoords="offset points", ha="center",
                    fontsize=7.6, color=CRITICAL,
                    arrowprops=dict(arrowstyle="-", color=CRITICAL, linewidth=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels([y.replace("-", "–") for y in years])
    ax.set_ylabel("authorities with no return")
    ax.set_ylim(0, max(counts) * 1.45)
    ax.set_title("Missing H-CLIC returns, by year "
                 f"(out of ~{int(np.mean(totals))} authorities)", loc="left", pad=10)
    style(ax)
    save(fig, "fig_data_quality")


def main() -> int:
    panel = pd.read_parquet(config.PANEL_FILE)
    design = pd.read_parquet(config.DESIGN_FILE)
    figure_trend(panel)
    figure_tasks()
    figure_precision_at_k()
    figure_calibration()
    figure_drivers(design)
    figure_colchester()
    figure_data_quality(panel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
