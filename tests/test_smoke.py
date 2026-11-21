"""Minimal smoke tests so CI has something to run; the real suite comes in phase 2."""

import numpy as np
import pytest

import Divergence_Free_Interpolant as dfi


@pytest.mark.parametrize("dim", [2, 3])
def test_condition_and_evaluate_at_nodes(dim):
    rng = np.random.default_rng(0)
    points = rng.random((15, dim))
    values = rng.standard_normal((15, dim))

    interp = dfi.interpolant(nu=5, k=3, dim=dim)
    interp.condition(points, values, support_radii=1.5)

    result = np.array([interp(*p) for p in points])
    np.testing.assert_allclose(result, values, atol=1e-8)
