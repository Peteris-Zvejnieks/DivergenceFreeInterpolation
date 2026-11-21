"""The matrix-valued kernels evaluated through their radial form."""

import numbers

import numpy as np
import pytest

from divergence_free_interpolant._kernel import (
    MatrixKernel,
    radial_coefficients,
    resolve_parameters,
)

from conftest import CURL_FREE, DIVERGENCE_FREE, KINDS, VALID_PARAMS, jacobian
from reference_kernel import reference_kernel


def sample(rng, dim, n=40, low=0.05, high=0.95):
    """Random points with norms in [low, high] (inside the unit support, away from the origin)."""
    directions = rng.normal(size=(n, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions * rng.uniform(low, high, size=(n, 1))


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_radial_parts_are_polynomials_with_h_zero_at_origin(k):
    a, h = radial_coefficients(k + 2, k)
    # Exact rationals (fractions.Fraction from the shipped table)
    assert all(isinstance(c, numbers.Rational) for c in a + h)
    # H(0) = psi''(0) - lim psi'(r)/r = 0
    assert h[0] == 0
    # psi decreases from its maximum at the origin
    assert a[0] < 0


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("nu, k", VALID_PARAMS)
def test_matches_symbolic_reference(kind, dim, nu, k, rng):
    nu = max(nu, resolve_parameters(k, None, dim))
    points = sample(rng, dim)
    expected = reference_kernel(kind, nu, k, dim)(points)
    actual = MatrixKernel(kind, dim, nu, k).blocks(points, 1.0)
    # absolute tolerance relative to the kernel's scale: entries near |x| = 1 are tiny and the
    # symbolic reference loses digits to cancellation there
    np.testing.assert_allclose(
        actual, expected, rtol=1e-10, atol=1e-11 * np.abs(expected).max()
    )


@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("nu, k", [(4, 2), (5, 3)])
def test_divergence_free_columns(dim, nu, k, rng):
    kernel = MatrixKernel(
        DIVERGENCE_FREE, dim, max(nu, resolve_parameters(k, None, dim)), k
    )
    points = sample(rng, dim, low=0.1, high=0.9)
    for column in range(dim):
        jac = jacobian(lambda x, c=column: kernel.blocks(x, 1.0)[..., c], points)
        divergence = np.trace(jac, axis1=1, axis2=2)
        assert (
            np.abs(divergence).max() < 1e-6 * np.abs(kernel.blocks(points, 1.0)).max()
        )


@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("nu, k", [(4, 2), (5, 3)])
def test_curl_free_columns(dim, nu, k, rng):
    """Curl-free means a symmetric Jacobian (in any dimension)."""
    kernel = MatrixKernel(CURL_FREE, dim, max(nu, resolve_parameters(k, None, dim)), k)
    points = sample(rng, dim, low=0.1, high=0.9)
    for column in range(dim):
        jac = jacobian(lambda x, c=column: kernel.blocks(x, 1.0)[..., c], points)
        assert (
            np.abs(jac - np.swapaxes(jac, 1, 2)).max()
            < 1e-6 * np.abs(kernel.blocks(points, 1.0)).max()
        )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dim", [2, 3])
def test_symmetric_and_even(kind, dim, rng):
    kernel = MatrixKernel(kind, dim, None, 3)
    points = sample(rng, dim)
    values = kernel.blocks(points, 1.0)
    np.testing.assert_allclose(values, np.swapaxes(values, 1, 2), atol=1e-15)
    np.testing.assert_allclose(values, kernel.blocks(-points, 1.0), atol=1e-15)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("nu, k", VALID_PARAMS)
def test_positive_definite_and_continuous_at_origin(kind, dim, nu, k, rng):
    kernel = MatrixKernel(kind, dim, nu, k)
    at_origin = kernel.blocks(np.zeros(dim), 1.0)
    assert np.isfinite(at_origin).all()
    assert np.linalg.eigvalsh(at_origin).min() > 0
    nearby = kernel.blocks(sample(rng, dim, low=1e-7, high=1e-7), 1.0)
    np.testing.assert_allclose(
        nearby,
        np.broadcast_to(at_origin, nearby.shape),
        atol=1e-6 * np.abs(at_origin).max(),
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dim", [2, 3])
def test_compact_support(kind, dim, rng):
    kernel = MatrixKernel(kind, dim, None, 3)
    on_boundary = sample(rng, dim, low=1.0, high=1.0)
    outside = sample(rng, dim, low=1.0001, high=5.0)
    np.testing.assert_allclose(kernel.blocks(on_boundary, 1.0), 0, atol=1e-15)
    np.testing.assert_array_equal(kernel.blocks(outside, 1.0), 0)


@pytest.mark.parametrize("kind", KINDS)
def test_support_radius_scaling(kind, rng):
    """Phi_delta(x) = delta^-d Phi(x / delta)."""
    kernel = MatrixKernel(kind, 3, None, 2)
    points = sample(rng, 3)
    np.testing.assert_allclose(
        kernel.blocks(2.5 * points, 2.5),
        2.5**-3 * kernel.blocks(points, 1.0),
        rtol=1e-12,
    )


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="kind"):
        MatrixKernel("harmonic", 2, None, 3)
