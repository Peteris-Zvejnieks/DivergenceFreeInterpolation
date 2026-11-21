"""Mathematical properties of the conditioned interpolant."""

from itertools import pairwise

import numpy as np
import pytest
from scipy.stats import qmc

import Divergence_Free_Interpolant as dfi

from conftest import VALID_PARAMS, conditioned, div_free_field

METHODS = ["linsolve", "lstsq", "penrose", "SVD"]


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("nu, k", VALID_PARAMS)
def test_reproduces_data_at_nodes(dim, nu, k):
    interp, nodes, values = conditioned(dim, nu, k, n=25, support=0.8)
    np.testing.assert_allclose(interp(*nodes.T), values, atol=1e-9)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dim", [2, 3])
def test_all_solver_methods_agree(dim, method):
    reference, nodes, _ = conditioned(dim, n=25, support=0.8, method="linsolve")
    interp, _, values = conditioned(dim, n=25, support=0.8, method=method)
    queries = np.random.default_rng(1).random((40, dim))
    np.testing.assert_allclose(interp(*nodes.T), values, atol=1e-8)
    np.testing.assert_allclose(interp(*queries.T), reference(*queries.T), atol=1e-8)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("nu, k", [(4, 2), (5, 3)])
def test_interpolated_field_is_divergence_free(dim, nu, k, rng):
    """Central differences of the interpolant; any physical divergence would be O(|u|)."""
    interp, _, values = conditioned(dim, nu, k, n=30, support=0.8)
    points = rng.uniform(0.1, 0.9, size=(30, dim))
    h = 1e-5
    divergence = np.zeros(len(points))
    for axis in range(dim):
        step = np.zeros(dim)
        step[axis] = h
        forward = interp(*(points + step).T)[:, axis]
        backward = interp(*(points - step).T)[:, axis]
        divergence += (forward - backward) / (2 * h)
    scale = np.abs(values).max()
    assert np.abs(divergence).max() < 1e-6 * scale


def test_zero_net_flux_through_closed_curve():
    """Divergence-free => zero net flux through a closed curve (here: a small square)."""
    interp, _, _ = conditioned(2, n=40, support=0.8)
    center, half = np.array([0.5, 0.5]), 0.1
    t = (np.arange(400) + 0.5) / 400 * (2 * half) - half  # midpoint rule on each side
    ds = 2 * half / 400
    sides = [
        (np.stack([center[0] + half + 0 * t, center[1] + t], 1), np.array([1.0, 0.0])),
        (np.stack([center[0] - half + 0 * t, center[1] + t], 1), np.array([-1.0, 0.0])),
        (np.stack([center[0] + t, center[1] + half + 0 * t], 1), np.array([0.0, 1.0])),
        (np.stack([center[0] + t, center[1] - half + 0 * t], 1), np.array([0.0, -1.0])),
    ]
    flux = sum((interp(*pts.T) @ normal).sum() * ds for pts, normal in sides)
    assert abs(flux) < 1e-6


def _max_error(dim, nu, k, n, support, queries):
    nodes = qmc.Halton(dim, seed=0).random(n)
    interp = dfi.interpolant(nu=nu, k=k, dim=dim)
    interp.condition(nodes, div_free_field(nodes), support_radii=support)
    return np.abs(interp(*queries.T) - div_free_field(queries)).max()


@pytest.mark.parametrize(
    "dim, sizes, max_final_error",
    [
        (2, [16, 64, 256], 1e-2),
        (3, [32, 128, 512], 5e-2),
    ],
)
def test_error_decreases_with_more_nodes(dim, sizes, max_final_error, rng):
    """Approximation error vs. an analytic field shrinks as the node set is refined."""
    queries = rng.uniform(0.1, 0.9, size=(200, dim))
    errors = [_max_error(dim, 5, 3, n, 1.0, queries) for n in sizes]
    assert all(later < 0.5 * earlier for earlier, later in pairwise(errors)), errors
    assert errors[-1] < max_final_error, errors


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("nu, k", VALID_PARAMS)
def test_gram_matrix_is_symmetric_positive_definite(dim, nu, k, rng):
    nodes = rng.random((20, dim))
    interp = dfi.interpolant(nu=nu, k=k, dim=dim)
    gram = interp._gram_matrix(nodes, support_radii=0.7)
    assert gram.shape == (dim * len(nodes), dim * len(nodes))
    np.testing.assert_allclose(gram, gram.T, atol=1e-13)
    np.linalg.cholesky(gram)  # raises LinAlgError if not positive definite


def test_support_radius_is_compact(rng):
    """Nodes farther apart than the support radius do not interact."""
    nodes = np.array([[0.0, 0.0], [0.2, 0.0], [5.0, 5.0]])
    interp = dfi.interpolant(dim=2)
    gram = interp._gram_matrix(nodes, support_radii=0.5)
    np.testing.assert_array_equal(gram[:4, 4:], 0)
    assert np.abs(gram[:2, 2:4]).max() > 0
