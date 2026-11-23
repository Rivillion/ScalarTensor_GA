"""Every χ² component must match the frozen reference tests/reference_chi2.toml, exported from the former Julia
WzBinning core (which itself reproduced the GA ΛCDM χ² term by term to 5e-4).

Run from binning/python:  uv run pytest -v
"""

import tomllib
from pathlib import Path

import numpy as np
import pytest

from wz.likelihood import chi2_components, get_evaluator

REFERENCE = Path(__file__).resolve().parent / "reference_chi2.toml"
ABS_TOL = 1e-3
# Prior draws far from the data have χ² up to ~1e9; both codes integrate to ~1e-8 relative accuracy there.
REL_TOL = 1e-7

with open(REFERENCE, "rb") as f:
    POINTS = tomllib.load(f)["point"]


@pytest.mark.parametrize("point", POINTS, ids=[p["label"] for p in POINTS])
def test_chi2_matches_julia(point):
    """Generated with Claude Opus 5.5."""
    comps = chi2_components(point["params"], point["model"], tuple(point["likelihoods"]))
    if not point["ok"]:
        assert comps is None or not np.isfinite(sum(comps.values()))
        return
    assert comps is not None
    for name, ref in point["chi2"].items():
        diff = comps[name] - ref
        print(f"{point['label']:24s} {name:20s} julia {ref:.6f}  python {comps[name]:.6f}  Δ {diff:+.2e}")
        assert abs(diff) < max(ABS_TOL, REL_TOL * abs(ref)), (name, ref, comps[name])


def test_param_names_follow_julia_order():
    """Generated with Claude Opus 5.5."""
    ev = get_evaluator("binned")
    assert ev.param_names == ["Om0", "h", "M_nuisance", "sigma8_0", "Ob0", "w[1]", "w[2]", "w[3]", "w[4]"]
    assert get_evaluator("cpl", ("DESI_BAO", "Planck_Zhai", "PantheonPlus_Marg")).param_names == \
        ["Om0", "h", "Ob0", "w0", "wa"]


def test_cpl_constraint_rejects():
    """Generated with Claude Opus 5.5."""
    ev = get_evaluator("cpl")
    theta = [0.3, 0.69, -19.38, 0.8, 0.047, -0.2, 0.5]
    assert ev.loglike(theta) == -np.inf
