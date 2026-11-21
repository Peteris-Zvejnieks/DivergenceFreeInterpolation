"""Cluster block-Jacobi preconditioner for conjugate gradients."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

import divergence_free_interpolant as dfi
from divergence_free_interpolant._kernel import get_kernel
from divergence_free_interpolant._preconditioner import cluster_preconditioner, clusters
from divergence_free_interpolant._solvers import solve


def test_isolated_points_are_singletons_and_close_pairs_merge():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.001]])
    labels = clusters(points, cKDTree(points))
    assert labels[3] == labels[4]
    assert len(set(labels[:4])) == 4


def test_large_groups_fall_back_to_single_points():
    """A chain of close points would make one huge dense block."""
    chain = np.column_stack([np.linspace(0, 1, 40) * 0.01, np.zeros(40)])
    spread = np.random.default_rng(0).random((200, 2)) + 2
    points = np.vstack([chain, spread])
    labels = clusters(points, cKDTree(points), max_size=10)
    assert np.bincount(labels).max() <= 10


@pytest.mark.parametrize("kind", ["divergence_free", "curl_free"])
def test_inverts_the_matrix_restricted_to_the_clusters(kind, rng):
    points = rng.random((300, 2))
    tree = cKDTree(points)
    support = 6 * float(np.median(tree.query(points, k=2)[0][:, 1]))
    kernel = get_kernel(kind, 2, None, 3)
    interp = dfi.DivergenceFreeInterpolant(support)
    interp._kind = kind
    gram = interp._gram_matrix(points).toarray()
    inverse = cluster_preconditioner(kernel, points, support, tree).toarray()
    labels = np.repeat(clusters(points, tree), 2)
    same_cluster = labels[:, None] == labels[None, :]
    assert np.sum(np.bincount(clusters(points, tree)) > 1) > 0
    np.testing.assert_allclose(
        inverse @ np.where(same_cluster, gram, 0), np.eye(600), atol=1e-10
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_reduces_cg_iterations_on_random_points(dim, rng):
    points = rng.random((3000, dim))
    tree = cKDTree(points)
    support = 6 * float(np.median(tree.query(points, k=2)[0][:, 1]))
    kernel = get_kernel("divergence_free", dim, None, 3)
    gram = dfi.DivergenceFreeInterpolant(support)._gram_matrix(points)
    rhs = rng.normal(size=gram.shape[0])
    plain, plain_info = solve(gram, rhs, "cg")
    preconditioner = cluster_preconditioner(kernel, points, support, tree)
    clustered, clustered_info = solve(gram, rhs, "cg", preconditioner=preconditioner)
    assert clustered_info.iterations * 3 < plain_info.iterations
    np.testing.assert_allclose(clustered, plain, rtol=1e-6, atol=1e-8)
