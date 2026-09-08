"""Stage 03 - Neighbourhood layer for Colchester.

Segments the Colchester LSOAs by deprivation profile and flags the ones whose
profile is unusual for the borough. The model choices live in
`fhew.neighbourhoods`; this stage runs them and prints what a housing team
would want to read: how the segment count was chosen, who is in each segment,
and what makes each flagged neighbourhood unusual.
"""
from __future__ import annotations

import sys

from fhew import config, neighbourhoods


def main() -> int:
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    config.TABLES.mkdir(parents=True, exist_ok=True)

    essex = neighbourhoods.load_lsoa()
    colchester = essex[essex["la_code"] == config.COLCHESTER].reset_index(drop=True)
    print(f"Essex LSOAs: {len(essex)}   Colchester LSOAs: {len(colchester)}")
    print(f"Colchester population {colchester['population'].sum():,.0f}, "
          f"children 0-15 {colchester['children_0_15'].sum():,.0f}")

    matrix = neighbourhoods.standardise(colchester)

    k, scores = neighbourhoods.choose_k(matrix, range(2, 9))
    best_overall = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    print("\nSegment count selection (silhouette):")
    for _, row in scores.iterrows():
        mark = "  <-- selected" if int(row["k"]) == k else ""
        if int(row["k"]) == best_overall and best_overall != k:
            mark = "  (best overall, but only separates deprived / not deprived)"
        print(f"   k={int(row['k'])}  silhouette={row['silhouette']:.3f}{mark}")

    colchester, profile = neighbourhoods.segment(colchester, matrix, k)
    colchester = neighbourhoods.flag_anomalies(colchester, matrix)

    print(f"\nSegments (k={k}):")
    for segment_id, row in profile.iterrows():
        members = colchester[colchester["segment"] == segment_id]
        print(f"   [{segment_id}] {row['segment_name']:<52} "
              f"n={len(members):>3}  children={members['children_0_15'].sum():>6,.0f}  "
              f"IDACI={row['idaci']:.3f}")

    flagged = colchester[colchester["is_anomaly"]].sort_values(
        "anomaly_score", ascending=False)
    print(f"\nIsolation Forest flagged {len(flagged)} of {len(colchester)} neighbourhoods:")
    for _, row in flagged.iterrows():
        driver, z = neighbourhoods.dominant_driver(row, colchester)
        print(f"   {row['lsoa_name']:<20} score={row['anomaly_score']:.3f}  "
              f"IMD decile={row['imd_decile_in_england']:.0f}  "
              f"driver={driver} (z={z:+.1f})  "
              f"children={row['children_0_15']:.0f}")

    colchester.to_parquet(config.COLCHESTER_LSOA_FILE, index=False)
    essex.to_parquet(config.ESSEX_LSOA_FILE, index=False)
    profile.to_csv(config.TABLES / "colchester_segments.csv")
    print(f"\nWrote {len(colchester)} rows to "
          f"{config.COLCHESTER_LSOA_FILE.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
