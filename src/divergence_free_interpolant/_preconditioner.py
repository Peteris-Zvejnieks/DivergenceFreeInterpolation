"""Block-Jacobi preconditioner over clusters of unusually close points.

Randomly placed points contain pairs and small groups that are much closer together than the
typical spacing. Their kernel rows are nearly identical, which makes the interpolation matrix
ill-conditioned, and conjugate gradients then needs thousands of iterations. Inverting the small
dense block of each such cluster exactly removes those near-singular directions: on random
data this cuts the iteration count by an order of magnitude or more.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

# Points closer than this fraction of the median nearest-neighbour distance form a cluster
CLUSTER_FRACTION = 0.7
# Larger connected groups (e.g. chains of close points) fall back to per-point blocks
MAX_CLUSTER_SIZE = 32


def clusters(points, tree, fraction=CLUSTER_FRACTION, max_size=MAX_CLUSTER_SIZE):
    """Cluster label per point: connected groups of points closer than fraction * spacing."""
    n = len(points)
    if n < 2:
        return np.zeros(n, dtype=np.intp)
    spacing = float(np.median(tree.query(points, k=2)[0][:, 1]))
    pairs = tree.query_pairs(fraction * spacing, output_type="ndarray")
    if not len(pairs):
        return np.arange(n)
    graph = sparse.coo_array(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)
    )
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    too_big = sizes[labels] > max_size
    labels[too_big] = labels.max() + 1 + np.arange(too_big.sum())
    return labels


def cluster_preconditioner(kernel, points, support_radius, tree):
    """Sparse block-diagonal inverse of the interpolation matrix restricted to the clusters."""
    n, d = points.shape
    labels = clusters(points, tree)
    _, labels = np.unique(labels, return_inverse=True)
    sizes = np.bincount(labels)
    order = np.argsort(labels, kind="stable")
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])

    rows, cols, values = [], [], []
    for size in np.unique(sizes):
        groups = np.flatnonzero(sizes == size)
        # Point indices of each cluster of this size: (clusters, size)
        members = order[starts[groups][:, None] + np.arange(size)]
        located = points[members]
        differences = located[:, :, None, :] - located[:, None, :, :]
        # (clusters, size, size, d, d) -> (clusters, size * d, size * d)
        blocks = kernel.blocks(differences, support_radius)
        blocks = blocks.transpose(0, 1, 3, 2, 4).reshape(
            len(groups), size * d, size * d
        )
        inverses = np.linalg.inv(blocks)
        index = (d * members[:, :, None] + np.arange(d)).reshape(len(groups), size * d)
        rows.append(np.repeat(index, size * d, axis=1).ravel())
        cols.append(np.tile(index, (1, size * d)).ravel())
        values.append(inverses.ravel())
    return sparse.csr_array(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n * d, n * d),
    )
