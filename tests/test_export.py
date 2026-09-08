"""Serialisation for the dashboard and the slides.

The page is static and reads JSON directly, so anything the browser cannot
parse fails silently as a blank panel rather than as an error.
"""
from __future__ import annotations

import json

import numpy as np

from fhew import export


def test_missing_values_become_null_rather_than_the_nan_token():
    """`NaN` is valid Python and invalid JSON; a browser would refuse the file."""
    payload = export.round_floats({"score": float("nan")})
    assert payload == {"score": None}
    assert json.loads(json.dumps(payload)) == {"score": None}


def test_rounding_reaches_inside_nested_structures():
    payload = export.round_floats(
        {"rows": [{"score": 0.123456789}], "meta": {"rate": 1.987654321}}, places=3)
    assert payload["rows"][0]["score"] == 0.123
    assert payload["meta"]["rate"] == 1.988


def test_values_that_are_not_floats_are_left_alone():
    payload = export.round_floats({"name": "Colchester", "n": 105, "flag": True})
    assert payload == {"name": "Colchester", "n": 105, "flag": True}


def test_infinity_is_not_silently_turned_into_null():
    """An infinite value is a bug upstream; it should stay visible, not vanish."""
    assert export.round_floats(float("inf")) == float("inf")


def test_the_whole_payload_survives_a_json_round_trip():
    payload = export.round_floats(
        {"a": [1.0, float("nan"), 3.5], "b": {"c": np.float64(2.25)}})
    assert json.loads(json.dumps(payload)) == {"a": [1.0, None, 3.5], "b": {"c": 2.25}}


def test_latex_macros_are_written_one_per_fact(tmp_path, monkeypatch):
    monkeypatch.setattr(export.config, "SLIDES", tmp_path)
    export.write_latex_macros({"NumAuthorities": "338", "EscTestN": "262"})

    written = (tmp_path / "facts.tex").read_text()
    assert r"\newcommand{\NumAuthorities}{338}" in written
    assert r"\newcommand{\EscTestN}{262}" in written
