"""Mathematical properties of fitted interpolants."""

from itertools import pairwise

import numpy as np
import pytest
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import qmc

import divergence_free_interpolant as dfi
from divergence_free_interpolant import _assembly, _backend

from conftest import (
    CURL_FREE,
    DIVERGENCE_FREE,
    FIELDS,
    INTERPOLANTS,
    KINDS,
    VALID_PARAMS,
    fitted,
    jacobian,
    potential,
)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("nu, k", VALID_PARAMS)
def test_reproduces_data_at_nodes(kind, dim, nu, k):
    interp, points, values = fitted(kind, dim, n=25, support=0.8, nu=nu, k=k)
    np.testing.assert_allclose(interp(points), values, atol=1e-9)


@pytest.mark.parametrize("kind", KINDS)
def test_works_in_four_dimensions(kind):
    interp, points, values = fitted(kind, 4, n=40, support=1.2)
    assert interp.dim == 4
    np.testing.assert_allclose(interp(points), values, atol=1e-9)


@pytest.mark.parametrize("solver", ["direct", "cg", "lstsq"])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dim", [2, 3])
def test_all_solvers_agree(kind, dim, solver):
    reference, points, _ = fitted(kind, dim, n=40, support=0.6, solver="direct")
    interp, _, values = fitted(kind, dim, n=40, support=0.6, solver=solver, tol=1e-12)
    queries = np.random.default_rng(1).random((50, dim))
    np.testing.assert_allclose(interp(points), values, atol=1e-8)
    np.testing.assert_allclose(interp(queries), reference(queries), atol=1e-8)


@pytest.mark.parametrize("dim", [2, 3, 4])
def test_divergence_free_interpolant_has_zero_divergence(dim, rng):
    interp, _, values = fitted(DIVERGENCE_FREE, dim, n=40, support=0.8)
    points = rng.uniform(0.2, 0.8, size=(30, dim))
    divergence = np.trace(jacobian(interp, points), axis1=1, axis2=2)
    assert np.abs(divergence).max() < 1e-6 * np.abs(values).max()


@pytest.mark.parametrize("dim", [2, 3, 4])
def test_curl_free_interpolant_has_symmetric_jacobian(dim, rng):
    interp, _, values = fitted(CURL_FREE, dim, n=40, support=0.8)
    points = rng.uniform(0.2, 0.8, size=(30, dim))
    jac = jacobian(interp, points)
    assert np.abs(jac - np.swapaxes(jac, 1, 2)).max() < 1e-6 * np.abs(values).max()


def _square_loop(n=400, center=0.5, half=0.1):
    """Midpoints, outward normals, tangents (counter-clockwise) and length element of a square."""
    t = (np.arange(n) + 0.5) / n * 2 * half - half
    ones = np.ones_like(t)
    sides = [
        (np.stack([center + half * ones, center + t], 1), [1.0, 0.0], [0.0, 1.0]),
        (np.stack([center - half * ones, center - t], 1), [-1.0, 0.0], [0.0, -1.0]),
        (np.stack([center - t, center + half * ones], 1), [0.0, 1.0], [-1.0, 0.0]),
        (np.stack([center + t, center - half * ones], 1), [0.0, -1.0], [1.0, 0.0]),
    ]
    return sides, 2 * half / n


def test_zero_net_flux_through_closed_curve():
    interp, _, _ = fitted(DIVERGENCE_FREE, 2, n=40, support=0.8)
    sides, ds = _square_loop()
    flux = sum((interp(points) @ normal).sum() * ds for points, normal, _ in sides)
    assert abs(flux) < 1e-6


def test_zero_circulation_around_closed_curve():
    interp, _, _ = fitted(CURL_FREE, 2, n=40, support=0.8)
    sides, ds = _square_loop()
    circulation = sum(
        (interp(points) @ tangent).sum() * ds for points, _, tangent in sides
    )
    assert abs(circulation) < 1e-6


@pytest.mark.parametrize("dim", [2, 3])
def test_potential_gradient_is_the_interpolant(dim, rng):
    interp, _, values = fitted(CURL_FREE, dim, n=60, support=0.7)
    points = rng.uniform(0.2, 0.8, size=(30, dim))
    gradient = jacobian(lambda x: interp.potential(x)[..., np.newaxis], points)[:, 0, :]
    np.testing.assert_allclose(
        gradient, interp(points), atol=1e-6 * np.abs(values).max()
    )


def test_stream_function_perpendicular_gradient_is_the_interpolant(rng):
    interp, _, values = fitted(DIVERGENCE_FREE, 2, n=60, support=0.7)
    points = rng.uniform(0.2, 0.8, size=(30, 2))
    gradient = jacobian(lambda x: interp.stream_function(x)[..., np.newaxis], points)[
        :, 0, :
    ]
    np.testing.assert_allclose(
        np.stack([gradient[:, 1], -gradient[:, 0]], 1),
        interp(points),
        atol=1e-6 * np.abs(values).max(),
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_recovers_potential_up_to_a_constant(dim):
    nodes = qmc.Halton(dim, seed=0).random(400 if dim == 2 else 1000)
    interp = dfi.CurlFreeInterpolant(1.0).fit(nodes, FIELDS[CURL_FREE](nodes))
    queries = np.random.default_rng(3).uniform(0.15, 0.85, size=(200, dim))
    difference = interp.potential(queries) - potential(queries)
    assert np.ptp(difference) < 1e-2 * np.ptp(potential(queries))


def _relative_max_error(kind, dim, n, support, queries):
    nodes = qmc.Halton(dim, seed=0).random(n)
    interp = INTERPOLANTS[kind](support).fit(nodes, FIELDS[kind](nodes))
    exact = FIELDS[kind](queries)
    return np.abs(interp(queries) - exact).max() / np.abs(exact).max()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "dim, sizes, max_final_relative_error",
    [
        (2, [16, 64, 256], 5e-3),
        (3, [32, 128, 512], 6e-2),
    ],
)
def test_error_decreases_with_more_nodes(
    kind, dim, sizes, max_final_relative_error, rng
):
    """Approximation error vs. an analytic field shrinks as the node set is refined."""
    queries = rng.uniform(0.1, 0.9, size=(200, dim))
    errors = [_relative_max_error(kind, dim, n, 1.0, queries) for n in sizes]
    assert all(later < 0.5 * earlier for earlier, later in pairwise(errors)), errors
    assert errors[-1] < max_final_relative_error, errors


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("nu, k", VALID_PARAMS)
def test_gram_matrix_is_symmetric_positive_definite(kind, dim, nu, k, rng):
    points = rng.random((20, dim))
    gram = INTERPOLANTS[kind](0.7, nu=nu, k=k)._gram_matrix(points).toarray()
    assert gram.shape == (dim * len(points), dim * len(points))
    np.testing.assert_allclose(gram, gram.T, atol=1e-13)
    np.linalg.cholesky(gram)  # raises LinAlgError if not positive definite


@pytest.mark.parametrize("kind", KINDS)
def test_gram_matrix_is_sparse_with_compact_support(kind):
    points = np.array([[0.0, 0.0], [0.2, 0.0], [5.0, 5.0]])
    gram = INTERPOLANTS[kind](0.5)._gram_matrix(points)
    assert sparse.issparse(gram)
    dense = gram.toarray()
    np.testing.assert_array_equal(dense[:4, 4:], 0)
    assert np.abs(dense[:2, 2:4]).max() > 0
    # 2x2 blocks are stored densely: 2 diagonal + 2 coupling + 1 isolated diagonal block
    assert gram.nnz == 5 * 4


@pytest.mark.parametrize("kind", KINDS)
def test_chunked_evaluation_matches_single_pass(kind, monkeypatch):
    """The numpy reference implementation evaluates in chunks bounded by PAIR_BUDGET."""
    monkeypatch.setattr(_backend, "BACKEND", "numpy")
    interp, _, _ = fitted(kind, 2, n=50, support=0.5)
    queries = np.random.default_rng(4).random((1000, 2))
    expected = interp(queries)
    expected_potential = interp._scalar_potential(queries)
    monkeypatch.setattr(_assembly, "PAIR_BUDGET", 7)
    np.testing.assert_array_equal(interp(queries), expected)
    np.testing.assert_array_equal(interp._scalar_potential(queries), expected_potential)


def test_evaluation_chunks_respect_the_pair_budget(monkeypatch):
    """Chunks cover all queries in order, each within the budget unless it is a single query."""
    monkeypatch.setattr(_assembly, "PAIR_BUDGET", 100)
    rng = np.random.default_rng(5)
    nodes = rng.random((300, 3))
    # Some of the queries lie far away from all nodes
    queries = np.vstack([rng.random((500, 3)), rng.random((50, 3)) + 10])
    tree = cKDTree(nodes)
    counts = tree.query_ball_point(queries, 0.4, return_length=True)
    chunks = list(_assembly._chunks(tree, queries, 0.4))
    assert chunks[0].start == 0 and chunks[-1].stop == len(queries)
    assert all(a.stop == b.start for a, b in pairwise(chunks))
    assert all(counts[c].sum() <= 100 or c.stop - c.start == 1 for c in chunks)
    assert len(chunks) > counts.sum() // 100


def test_far_from_data_the_field_and_potentials_vanish():
    interp, _, _ = fitted(CURL_FREE, 2, n=30, support=0.3)
    far = np.array([[5.0, 5.0], [-3.0, 0.5]])
    np.testing.assert_array_equal(interp(far), 0)
    np.testing.assert_array_equal(interp.potential(far), 0)


@pytest.mark.parametrize("kind", KINDS)
def test_sparse_path_handles_thousands_of_points(kind):
    """Small support relative to the domain: a large but very sparse system."""
    points = qmc.Halton(2, seed=1).random(5000)
    values = FIELDS[kind](points)
    interp = INTERPOLANTS[kind](0.05).fit(points, values)
    np.testing.assert_allclose(interp(points[::50]), values[::50], atol=1e-8)


# --- general linear functionals (weighted point evaluations) --------------------------------


def _directional_weights(normals):
    """Weights for one functional n_i . u(x_i) per point: sparse (N d, N)."""
    n, d = normals.shape
    rows = (d * np.arange(n)[:, None] + np.arange(d)).ravel()
    cols = np.repeat(np.arange(n), d)
    return sparse.csr_array((normals.ravel(), (rows, cols)), shape=(n * d, n))


@pytest.mark.parametrize("kind", KINDS)
def test_directional_data_is_reproduced(kind, rng):
    """Only the component along a given direction is known at each point."""
    points = rng.random((40, 2))
    normals = rng.normal(size=(40, 2))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    data = np.einsum("nd,nd->n", normals, FIELDS[kind](points))
    interp = INTERPOLANTS[kind](0.6)
    interp._fit_functionals(points, _directional_weights(normals), data)
    np.testing.assert_allclose(
        np.einsum("nd,nd->n", normals, interp(points)), data, atol=1e-9
    )


@pytest.mark.parametrize("kind", KINDS)
def test_identity_weights_match_plain_fit(kind):
    plain, points, values = fitted(kind, 2, n=30, support=0.6)
    general = INTERPOLANTS[kind](0.6)
    general._fit_functionals(
        points, sparse.eye_array(points.size, format="csr"), values.ravel()
    )
    queries = np.random.default_rng(5).random((40, 2))
    np.testing.assert_allclose(general(queries), plain(queries), rtol=1e-10, atol=1e-12)
