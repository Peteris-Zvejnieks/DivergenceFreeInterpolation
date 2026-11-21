"""Regression test against outputs recorded from a known-good version (see make_golden.py)."""

from pathlib import Path

import numpy as np
import pytest

import divergence_free_interpolant as dfi

GOLDEN = np.load(Path(__file__).parent / "data" / "golden.npz")
CASES = sorted({key.split("/")[0] for key in GOLDEN.files})


@pytest.mark.parametrize("case", CASES)
def test_matches_golden_values(case):
    dim, nu, k, support = GOLDEN[f"{case}/config"]
    interp = dfi.interpolant(nu=int(nu), k=int(k), dim=int(dim))
    interp.condition(GOLDEN[f"{case}/nodes"], GOLDEN[f"{case}/values"], support_radii=support)
    actual = interp(*GOLDEN[f"{case}/queries"].T)
    expected = GOLDEN[f"{case}/expected"]
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10 * np.abs(expected).max())
