"""Sparse assembly and evaluation of matrix-kernel interpolants.

The Rust core (`_core`) does the work, in parallel. The numpy versions (`*_numpy`, using scipy's
KD-tree) are the reference implementation it is tested against; see `_backend`.
"""

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from . import _backend, _core

# Queries are evaluated in chunks holding at most this many query-node pairs, which bounds the
# memory of the (pairs, d, d) kernel blocks and their temporaries (a few hundred MB in 3D).
PAIR_BUDGET = 500_000


def _contiguous(array):
    return np.ascontiguousarray(array, dtype=float)


def gram_matrix(kernel, points, support_radius, tree=None):
    """Sparse symmetric interpolation matrix with d x d blocks Phi(x_i - x_j), shape (N d, N d).

    `tree` (a scipy KD-tree of the points) is only used by the numpy implementation.
    """
    if _backend.use_numpy():
        return _gram_matrix_numpy(kernel, points, support_radius, tree)
    indptr, indices, data = _core.gram_matrix(
        kernel.core, _contiguous(points), float(support_radius)
    )
    return sparse.csr_array((data, indices, indptr), shape=(points.size, points.size))


def evaluate(kernel, tree, nodes, coefficients, queries, support_radius):
    """sum_j Phi(q - x_j) c_j for queries (M, d), with coefficients (N, d); returns (M, d)."""
    if _backend.use_numpy():
        return _evaluate_numpy(
            kernel, tree, nodes, coefficients, queries, support_radius
        )
    values = _core.evaluate(
        kernel.core,
        _contiguous(nodes),
        _contiguous(coefficients),
        _contiguous(queries),
        float(support_radius),
    )
    return values.reshape(queries.shape)


def evaluate_potential(kernel, tree, nodes, coefficients, queries, support_radius):
    """Scalar potential (see MatrixKernel.potential_terms) at queries (M, d); returns (M,)."""
    if _backend.use_numpy():
        return _evaluate_potential_numpy(
            kernel, tree, nodes, coefficients, queries, support_radius
        )
    return _core.evaluate_potential(
        kernel.core,
        _contiguous(nodes),
        _contiguous(coefficients),
        _contiguous(queries),
        float(support_radius),
    )


# --- numpy reference implementation ----------------------------------------------------------


def _gram_matrix_numpy(kernel, points, support_radius, tree=None):
    n, d = points.shape
    tree = cKDTree(points) if tree is None else tree
    pairs = tree.query_pairs(support_radius, output_type="ndarray")
    i, j = pairs[:, 0], pairs[:, 1]
    off_diagonal = kernel.blocks(points[i] - points[j], support_radius)
    diagonal = kernel.blocks(np.zeros(d), support_radius)
    nodes = np.arange(n)

    # Phi is even and symmetric, so block (j, i) equals block (i, j)
    row_nodes = np.concatenate([i, j, nodes])
    col_nodes = np.concatenate([j, i, nodes])
    data = np.concatenate(
        [off_diagonal, off_diagonal, np.broadcast_to(diagonal, (n, d, d))]
    )

    component = np.arange(d)
    rows = d * row_nodes[:, None, None] + component[None, :, None]
    cols = d * col_nodes[:, None, None] + component[None, None, :]
    rows, cols = np.broadcast_arrays(rows, cols)
    return sparse.csr_array(
        (data.ravel(), (rows.ravel(), cols.ravel())), shape=(n * d, n * d)
    )


def _chunks(tree, queries, support_radius):
    """Consecutive query slices with at most PAIR_BUDGET neighbour pairs each (or one query)."""
    counts = tree.query_ball_point(queries, support_radius, return_length=True)
    cumulative = np.concatenate([[0], np.cumsum(counts)])
    start = 0
    while start < len(queries):
        stop = (
            np.searchsorted(cumulative, cumulative[start] + PAIR_BUDGET, side="right")
            - 1
        )
        stop = max(int(stop), start + 1)
        yield slice(start, stop)
        start = stop


def _pairs_in_chunks(tree, queries, support_radius):
    """Yield (chunk slice, query indices, node indices) for all query-node pairs within the support."""
    for chunk in _chunks(tree, queries, support_radius):
        pairs = cKDTree(queries[chunk]).sparse_distance_matrix(
            tree, support_radius, output_type="ndarray"
        )
        yield chunk, pairs["i"], pairs["j"]


def _evaluate_numpy(kernel, tree, nodes, coefficients, queries, support_radius):
    out = np.zeros(queries.shape)
    for chunk, i, j in _pairs_in_chunks(tree, queries, support_radius):
        blocks = kernel.blocks(queries[chunk][i] - nodes[j], support_radius)
        contributions = np.einsum("pab,pb->pa", blocks, coefficients[j])
        size = chunk.stop - chunk.start
        for axis in range(queries.shape[1]):
            out[chunk, axis] = np.bincount(
                i, weights=contributions[:, axis], minlength=size
            )
    return out


def _evaluate_potential_numpy(
    kernel, tree, nodes, coefficients, queries, support_radius
):
    out = np.zeros(len(queries))
    for chunk, i, j in _pairs_in_chunks(tree, queries, support_radius):
        terms = kernel.potential_terms(
            queries[chunk][i] - nodes[j], coefficients[j], support_radius
        )
        out[chunk] = np.bincount(i, weights=terms, minlength=chunk.stop - chunk.start)
    return out
