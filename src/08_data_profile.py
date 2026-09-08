"""Stage 08 - Profile the dataset.

Writes the dataset card (DATASET.md) and the JSON the dashboard's Dataset tab
reads. The measurement lives in `fhew.profiling`, and everything it reports is
computed from the built artefacts rather than typed in, so the counts cannot
drift from the data they describe.
"""
from __future__ import annotations

import json
import sys

from fhew import config, profiling


def main() -> int:
    measured = profiling.profile()
    config.DASHBOARD_DATA.mkdir(parents=True, exist_ok=True)
    (config.DASHBOARD_DATA / "profile.json").write_text(
        json.dumps(measured, separators=(",", ":"), default=float))
    profiling.write_markdown(measured)

    design = measured["design"]
    print(f"Panel        {measured['panel']['rows']:,} authority-years, "
          f"{measured['panel']['measures']} measures")
    print(f"Design       {design['rows']:,} rows, {design['features']} features, "
          f"{design['overall_missing_rate']:.1%} missing")
    print(f"Neighbourhood {measured['neighbourhoods']['lsoas']} LSOAs, "
          f"{measured['neighbourhoods']['features']} features")
    print("Wrote DATASET.md and docs/data/profile.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
