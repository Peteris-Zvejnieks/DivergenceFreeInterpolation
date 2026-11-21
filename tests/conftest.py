import numpy as np
import pytest

import divergence_free_interpolant as dfi

DIVERGENCE_FREE = "divergence_free"
CURL_FREE = "curl_free"
KINDS = [DIVERGENCE_FREE, CURL_FREE]
INTERPOLANTS = {
    DIVERGENCE_FREE: dfi.DivergenceFreeInterpolant,
    CURL_FREE: dfi.CurlFreeInterpolant,
}

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


def potential(points):
    """Smooth scalar potentials whose gradients are the curl-free test fields."""
    points = np.asarray(points, dtype=float)
    if points.shape[1] == 2:
        x, y = points.T
        return np.exp(x) * np.sin(y) + x * y**2
    x, y, z = points.T
    return np.sin(x) * y + z**2 * x + np.exp(y * z)


def curl_free_field(points):
    """Gradients of `potential`: shape (N, dim) -> (N, dim)."""
    points = np.asarray(points, dtype=float)
    if points.shape[1] == 2:
        x, y = points.T
        return np.stack(
            [np.exp(x) * np.sin(y) + y**2, np.exp(x) * np.cos(y) + 2 * x * y], axis=1
        )
    x, y, z = points.T
    return np.stack(
        [
            np.cos(x) * y + z**2,
            np.sin(x) + z * np.exp(y * z),
            2 * z * x + y * np.exp(y * z),
        ],
        axis=1,
    )


FIELDS = {DIVERGENCE_FREE: div_free_field, CURL_FREE: curl_free_field}


def fitted(kind, dim, n=30, support=1.0, seed=0, **options):
    """An interpolant of the matching analytic field, fitted to random points in the unit cube."""
    rng = np.random.default_rng(seed)
    points = rng.random((n, dim))
    values = FIELDS[kind](points) if dim in (2, 3) else rng.normal(size=(n, dim))
    interp = INTERPOLANTS[kind](support, **options).fit(points, values)
    return interp, points, values


def jacobian(function, points, h=1e-4):
    """Central-difference Jacobian J[m, a, b] = d f_a / d x_b of a vector field at points (M, d)."""
    columns = [
        (function(points + h * e) - function(points - h * e)) / (2 * h)
        for e in np.eye(points.shape[1])
    ]
    return np.stack(columns, axis=-1)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture(params=["rust", "numpy"])
def each_backend(request, monkeypatch):
    """Runs a test with the Rust core and with the numpy reference implementation."""
    from divergence_free_interpolant import _backend

    monkeypatch.setattr(_backend, "BACKEND", request.param)
    return request.param
