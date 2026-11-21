"""Greedy farthest-point ordering, used to build the nested levels of multilevel interpolation."""

import heapq

import numpy as np
from scipy.spatial import cKDTree

from . import _backend, _core


def farthest_point_ordering(points, stop_radius=0.0, tree=None):
    """Greedy farthest-point ordering of (N, d) points.

    Starting from the point closest to the centroid, repeatedly picks the point farthest from all
    points picked so far. Returns (order, radii): point indices in insertion order and each
    point's distance to the previously picked ones when it was inserted (inf for the first).
    The radii never increase, and every prefix order[:m] is quasi-uniform: its points are at
    least radii[m - 1] apart, and every input point lies within radii[m] of it.

    The ordering stops before the first point whose radius would be below `stop_radius`, so the
    result can be a prefix of a full permutation.

    Only points within the current radius of a newly picked point can move closer to the picked
    set, so with a KD-tree the cost is about O(N log N) rather than O(N^2).
    """
    points = np.ascontiguousarray(points, dtype=float)
    start = int(np.argmin(np.linalg.norm(points - points.mean(axis=0), axis=1)))
    if not _backend.use_numpy():
        order, radii = _core.farthest_point_ordering(points, start, float(stop_radius))
        return order.astype(np.intp), radii
    return _farthest_point_ordering_numpy(points, start, stop_radius, tree)


def _farthest_point_ordering_numpy(points, start, stop_radius, tree=None):
    """The numpy/heapq reference implementation (`tree`: a scipy KD-tree of the points)."""
    n = len(points)
    tree = cKDTree(points) if tree is None else tree

    distance = np.linalg.norm(points - points[start], axis=1)
    distance[start] = 0.0
    picked = np.zeros(n, dtype=bool)
    picked[start] = True
    order, radii = [start], [np.inf]

    heap = list(zip((-distance).tolist(), range(n), strict=True))
    heapq.heapify(heap)
    while heap:
        negative, i = heapq.heappop(heap)
        radius = -negative
        # Skip points picked already and stale heap entries (superseded by a smaller distance)
        if picked[i] or radius != distance[i]:
            continue
        if radius < stop_radius:
            break
        picked[i] = True
        distance[i] = 0.0
        order.append(i)
        radii.append(radius)

        neighbours = np.asarray(tree.query_ball_point(points[i], radius), dtype=np.intp)
        if neighbours.size:
            new = np.linalg.norm(points[neighbours] - points[i], axis=1)
            closer = new < distance[neighbours]
            neighbours, new = neighbours[closer], new[closer]
            distance[neighbours] = new
            for value, j in zip((-new).tolist(), neighbours.tolist(), strict=True):
                heapq.heappush(heap, (value, j))

    return np.array(order, dtype=np.intp), np.array(radii)
