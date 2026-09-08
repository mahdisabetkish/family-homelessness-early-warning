"""Family homelessness early warning: a reproducible analysis package.

The pipeline is split in two. This package holds the logic - parsing,
measure selection, boundary reconciliation, feature construction, the models
and their evaluation - and is importable and unit-tested. The numbered
scripts in ``src/`` are thin stage runners: they read one artefact, call in
here, print what happened and write the next artefact.

The split exists so that the parts where a bug would quietly change a
published number can be tested directly, rather than inferred from whether
the output files look plausible.
"""
from __future__ import annotations

__version__ = "1.0.0"

__all__ = [
    "config",
    "evaluation",
    "export",
    "features",
    "hclic",
    "mlp",
    "models",
    "neighbourhoods",
    "panel",
    "plotting",
    "profiling",
]
