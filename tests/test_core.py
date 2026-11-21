"""The Rust core against the numpy reference implementation."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from divergence_free_interpolant import _assembly, _backend, _core, _thinning
from divergence_free_interpolant._kernel import get_kernel

from conftest import CURL_FREE, DIVERGENCE_FREE, KINDS


@pytest.fixture(autouse=True)
def rust_backend(monkeypatch):
    """These tests compare the Rust core with the numpy reference explicitly."""
    monkeypatch.setattr(_backend, "BACKEND", "rust")


def random_problem(dim, n, seed=0):
    rng = np.random.default_rng(seed)
    nodes = rng.random((n, dim))
    coefficients = rng.normal(size=(n, dim))
    queries = rng.uniform(-0.1, 1.1, size=(3 * n, dim))
    return nodes, coefficients, queries


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_gram_matrix_matches_numpy(kind, dim):
    kernel = get_kernel(kind, dim, None, 3)
    points, _, _ = random_problem(dim, 400)
    support = 0.25 if dim < 4 else 0.45
    rust = _assembly.gram_matrix(kernel, points, support)
    reference = _assembly._gram_matrix_numpy(kernel, points, support)
    assert rust.shape == reference.shape
    assert rust.nnz == reference.nnz
    scale = np.abs(reference.data).max()
    assert np.abs((rust - reference).toarray()).max() < 1e-13 * scale


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_evaluation_matches_numpy(kind, dim):
    kernel = get_kernel(kind, dim, None, 3)
    nodes, coefficients, queries = random_problem(dim, 300)
    support = 0.3 if dim < 4 else 0.5
    rust = _assembly.evaluate(kernel, None, nodes, coefficients, queries, support)
    reference = _assembly._evaluate_numpy(
        kernel, cKDTree(nodes), nodes, coefficients, queries, support
    )
    np.testing.assert_allclose(
        rust, reference, rtol=1e-12, atol=1e-12 * np.abs(reference).max()
    )


@pytest.mark.parametrize(
    "kind, dim", [(CURL_FREE, 2), (CURL_FREE, 3), (DIVERGENCE_FREE, 2)]
)
def test_potential_matches_numpy(kind, dim):
    kernel = get_kernel(kind, dim, None, 3)
    nodes, coefficients, queries = random_problem(dim, 300)
    rust = _assembly.evaluate_potential(kernel, None, nodes, coefficients, queries, 0.3)
    reference = _assembly._evaluate_potential_numpy(
        kernel, cKDTree(nodes), nodes, coefficients, queries, 0.3
    )
    np.testing.assert_allclose(
        rust, reference, rtol=1e-12, atol=1e-12 * np.abs(reference).max()
    )


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("stop", [0.0, 0.05])
def test_farthest_point_ordering_matches_numpy(dim, stop, monkeypatch):
    points = np.random.default_rng(dim).random((2000, dim))
    order, radii = _thinning.farthest_point_ordering(points, stop)
    monkeypatch.setattr(_backend, "BACKEND", "numpy")
    expected_order, expected_radii = _thinning.farthest_point_ordering(points, stop)
    np.testing.assert_array_equal(order, expected_order)
    np.testing.assert_allclose(radii, expected_radii, rtol=1e-14)


def test_core_validates_its_inputs():
    kernel = get_kernel(DIVERGENCE_FREE, 3, None, 3).core
    points = np.random.default_rng(0).random((10, 3))
    with pytest.raises(ValueError, match="3 columns"):
        _core.gram_matrix(kernel, points[:, :2].copy(), 0.5)
    with pytest.raises(ValueError, match="same shape"):
        _core.evaluate(kernel, points, points[:5].copy(), points, 0.5)
    with pytest.raises(ValueError, match="2D"):
        _core.evaluate_potential(kernel, points, points, points, 0.5)
    with pytest.raises(ValueError, match="kind"):
        _core.RadialKernel("harmonic", 2, [1.0], [1.0])
    with pytest.raises(ValueError, match="dimension"):
        _core.RadialKernel(CURL_FREE, 1, [1.0], [1.0])
    with pytest.raises(ValueError, match="start"):
        _core.farthest_point_ordering(points, 10, 0.0)


def test_unknown_backend_raises(monkeypatch):
    monkeypatch.setattr(_backend, "BACKEND", "fortran")
    kernel = get_kernel(CURL_FREE, 2, None, 3)
    with pytest.raises(ValueError, match="backend"):
        _assembly.gram_matrix(kernel, np.zeros((3, 2)), 0.5)
