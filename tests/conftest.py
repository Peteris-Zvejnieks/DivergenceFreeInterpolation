import numpy as np
import pytest

import divergence_free_interpolant as dfi

# Kernel parameters (nu, k) that are positive definite in both 2D and 3D (nu >= k + 2).
VALID_PARAMS = [(3, 1), (4, 2), (5, 3)]


def div_free_field(points):
    """Analytic divergence-free fields, rows are points: shape (N, dim) -> (N, dim)."""
    points = np.asarray(points, dtype=float)
    if points.shape[1] == 2:
        x, y = points.T
        return np.stack([-2 * x**3 * y, 3 * x**2 * y**2], axis=1)
    x, y, z = points.T
    return np.stack([np.sin(y) * z, np.cos(z) - x, 0.3 * np.exp(x) - y], axis=1)


def conditioned(dim, nu=5, k=3, n=30, support=1.0, seed=0, method="linsolve"):
    """An interpolant conditioned on a random sample of `div_free_field` in the unit cube."""
    rng = np.random.default_rng(seed)
    nodes = rng.random((n, dim))
    values = div_free_field(nodes)
    interp = dfi.interpolant(nu=nu, k=k, dim=dim)
    interp.condition(nodes, values, support_radii=support, method=method)
    return interp, nodes, values


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
