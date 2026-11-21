"""Greedy farthest-point ordering."""

import numpy as np
import pytest
from scipy.spatial.distance import cdist, pdist

from divergence_free_interpolant._thinning import farthest_point_ordering


def naive_ordering(points):
    """O(N^2) reference implementation with the same start point and tie-breaking."""
    start = int(np.argmin(np.linalg.norm(points - points.mean(axis=0), axis=1)))
    distance = np.linalg.norm(points - points[start], axis=1)
    distance[start] = -1.0
    order, radii = [start], [np.inf]
    for _ in range(len(points) - 1):
        i = int(np.argmax(distance))
        order.append(i)
        radii.append(distance[i])
        distance = np.minimum(distance, np.linalg.norm(points - points[i], axis=1))
        distance[order] = -1.0
    return np.array(order), np.array(radii)


@pytest.mark.parametrize("dim", [2, 3])
def test_matches_naive_implementation(dim, rng):
    points = rng.random((400, dim))
    order, radii = farthest_point_ordering(points)
    expected_order, expected_radii = naive_ordering(points)
    np.testing.assert_array_equal(order, expected_order)
    np.testing.assert_allclose(radii, expected_radii)


@pytest.mark.parametrize("dim", [2, 3])
def test_prefixes_are_separated_and_covering(dim, rng):
    points = rng.random((300, dim))
    order, radii = farthest_point_ordering(points)
    assert sorted(order) == list(range(len(points)))
    assert radii[0] == np.inf
    assert np.all(np.diff(radii[1:]) <= 0)
    for m in [2, 5, 20, 100, 299]:
        prefix = points[order[:m]]
        assert pdist(prefix).min() >= radii[m - 1] - 1e-12
        assert cdist(points, prefix).min(axis=1).max() <= radii[m] + 1e-12


def test_stop_radius_returns_the_prefix(rng):
    points = rng.random((500, 2))
    order, radii = farthest_point_ordering(points)
    stop = radii[100]
    short_order, short_radii = farthest_point_ordering(points, stop_radius=stop)
    count = np.sum(radii >= stop)
    np.testing.assert_array_equal(short_order, order[:count])
    np.testing.assert_array_equal(short_radii, radii[:count])


def test_duplicate_points_come_last(rng):
    points = rng.random((50, 2))
    points = np.vstack([points, points[:5]])
    _, radii = farthest_point_ordering(points)
    assert np.all(radii[-5:] == 0)
    _, positive = farthest_point_ordering(points, stop_radius=1e-12)
    assert len(positive) == 50
