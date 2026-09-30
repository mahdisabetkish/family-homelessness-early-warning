"""Writing the artefacts the dashboard reads.

The dashboard is a static page: no server, no build step, no database. Every
number it shows is written from the pipeline artefacts, so the page cannot
drift from the analysis.
"""
from __future__ import annotations

import json

import numpy as np

from . import config


def round_floats(obj: object, places: int = 4) -> object:
    """Trim float precision before serialising - it halves the payload.

    NaN is written as null rather than the bare `NaN` token, which is valid
    Python but not valid JSON and would fail in a browser.
    """
    if isinstance(obj, float):
        return None if np.isnan(obj) else round(obj, places)
    if isinstance(obj, dict):
        return {k: round_floats(v, places) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, places) for v in obj]
    return obj


def write_json(name: str, payload: object) -> None:
    """Write one dashboard payload to docs/data/<name>.json."""
    config.DASHBOARD_DATA.mkdir(parents=True, exist_ok=True)
    path = config.DASHBOARD_DATA / f"{name}.json"
    path.write_text(json.dumps(round_floats(payload), separators=(",", ":")))
    print(f"  wrote docs/data/{name}.json  ({path.stat().st_size / 1024:.0f} kB)")

