"""Stage 02 - Assemble the modelling panel.

Reads the cached per-year H-CLIC extracts, selects the measures the model
needs, converts them to rates, and joins the 2019 Indices of Deprivation. The
selection rules, the boundary reconciliation and the rate conversion all live
in `fhew.panel`; this stage runs them and reports what happened.

The coverage grid is printed rather than summarised because a measure that
quietly stopped resolving is the failure mode that would matter most here.
"""
from __future__ import annotations

import sys

from fhew import config
from fhew import panel as panel_module


def main() -> int:
    config.PROCESSED.mkdir(parents=True, exist_ok=True)

    panel, coverage, log = panel_module.build()

    print("Measure coverage by year (. = found, X = absent from that release)")
    lines, gaps = panel_module.coverage_grid(coverage)
    print("\n".join(lines))
    # A handful of gaps is expected: MHCLG introduced measures mid-period. A
    # sudden collapse would mean the header patterns had stopped matching, so
    # it fails here rather than becoming a panel full of nulls.
    ceiling = len(panel_module.all_measures()) * len(config.YEARS) // 4
    assert gaps < ceiling, f"{gaps} measure-years unresolved; header patterns need review"

    print("\nDeprivation")
    print("\n".join(log))

    panel.to_parquet(config.PANEL_FILE, index=False)
    print(f"\nWrote {config.PANEL_FILE.relative_to(config.ROOT)}  shape={panel.shape}")
    print(f"  {panel['la_code'].nunique()} distinct authorities "
          f"across {len(config.YEARS)} years")
    return 0


if __name__ == "__main__":
    sys.exit(main())
