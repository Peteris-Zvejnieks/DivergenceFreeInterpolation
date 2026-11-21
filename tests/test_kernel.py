"""The matrix-valued divergence-free kernel Phi = (grad grad^T - Laplacian I) psi."""

import numpy as np
import pytest
import sympy as sp

from divergence_free_interpolant._kernel import _RBF_kernel

from conftest import VALID_PARAMS

# Points inside the unit ball (the kernel's support), as exact rationals.
SAMPLE_POINTS = {
    2: [(sp.Rational(1, 3), sp.Rational(1, 5)), (sp.Rational(-1, 2), sp.Rational(1, 7))],
    3: [
        (sp.Rational(1, 3), sp.Rational(1, 5), sp.Rational(-1, 4)),
        (sp.Rational(-1, 6), sp.Rational(2, 7), sp.Rational(1, 9)),
    ],
}


def evaluate(kernel, points):
    """Evaluate the numpy kernel at points (M, dim) -> (M, dim, dim)."""
    points = np.asarray(points, dtype=float)
    radius = np.linalg.norm(points, axis=1)
    out = np.array(kernel.kernel_numpy(*points.T, radius), dtype=float)
    return np.moveaxis(out, -1, 0)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("nu, k", VALID_PARAMS)
def test_columns_are_divergence_free_symbolically(dim, nu, k):
    kernel = _RBF_kernel(nu, k, dim)
    xs = sp.symbols(f"x0:{dim}")
    r = kernel.rbf.free_symbols.pop()
    matrix = kernel.kernel.subs(r, sp.sqrt(sum(x**2 for x in xs)))
    for column in range(dim):
        divergence = sum(sp.diff(matrix[row, column], xs[row]) for row in range(dim))
        for point in SAMPLE_POINTS[dim]:
            value = divergence.subs(dict(zip(xs, point))).evalf(50)
            assert abs(value) < 1e-40, f"column {column} has divergence {value} at {point}"


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("nu, k", VALID_PARAMS)
def test_symmetric_and_even(dim, nu, k, rng):
    kernel = _RBF_kernel(nu, k, dim)
    points = rng.uniform(-0.6, 0.6, size=(50, dim))
    values = evaluate(kernel, points)
    np.testing.assert_allclose(values, np.swapaxes(values, 1, 2), atol=1e-14)
    np.testing.assert_allclose(values, evaluate(kernel, -points), atol=1e-14)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("nu, k", VALID_PARAMS)
def test_finite_and_positive_definite_at_origin(dim, nu, k):
    kernel = _RBF_kernel(nu, k, dim)
    at_origin = evaluate(kernel, np.zeros((1, dim)))[0]
    assert np.isfinite(at_origin).all()
    assert np.linalg.eigvalsh(at_origin).min() > 0


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("nu, k", VALID_PARAMS)
def test_continuous_at_origin(dim, nu, k, rng):
    """The value used at r = 0 must be the limit of the kernel, not a special case."""
    kernel = _RBF_kernel(nu, k, dim)
    at_origin = evaluate(kernel, np.zeros((1, dim)))[0]
    directions = rng.normal(size=(10, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    nearby = evaluate(kernel, 1e-7 * directions)
    np.testing.assert_allclose(nearby, np.broadcast_to(at_origin, nearby.shape), atol=1e-6)


@pytest.mark.parametrize("dim", [2, 3])
def test_vanishes_outside_support_boundary(dim):
    """At |x| = 1 every entry is zero, so the compact cutoff at r > 1 is continuous."""
    kernel = _RBF_kernel(5, 3, dim)
    on_sphere = np.eye(dim)
    np.testing.assert_allclose(evaluate(kernel, on_sphere), 0, atol=1e-14)
