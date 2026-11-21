"""Multilevel interpolation."""

from itertools import pairwise

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.stats import qmc

import divergence_free_interpolant as dfi
from divergence_free_interpolant._multilevel import level_hierarchy

from conftest import CURL_FREE, DIVERGENCE_FREE, FIELDS, KINDS, jacobian

MULTILEVEL = {
    DIVERGENCE_FREE: dfi.MultilevelDivergenceFreeInterpolant,
    CURL_FREE: dfi.MultilevelCurlFreeInterpolant,
}
SINGLE = {
    DIVERGENCE_FREE: dfi.DivergenceFreeInterpolant,
    CURL_FREE: dfi.CurlFreeInterpolant,
}


def fitted(kind, dim, n=2000, seed=0, **options):
    points = np.random.default_rng(seed).random((n, dim))
    values = FIELDS[kind](points)
    return MULTILEVEL[kind](**options).fit(points, values), points, values


# --- level hierarchy -------------------------------------------------------------------------


def test_levels_are_nested_with_geometric_spacings(rng):
    points = rng.random((5000, 2))
    levels = level_hierarchy(points, support_factor=6.0, refinement=0.5)
    assert len(levels) >= 4
    for (coarse, coarse_spacing), (fine, fine_spacing) in pairwise(levels):
        assert set(coarse) < set(fine)
        assert fine_spacing / coarse_spacing == pytest.approx(0.5)
    nodes, spacing = levels[-1]
    np.testing.assert_array_equal(nodes, np.arange(len(points)))
    assert spacing == pytest.approx(
        np.median(cKDTree(points).query(points, k=2)[0][:, 1])
    )


def test_levels_are_quasi_uniform(rng):
    """Each non-final level is separated by and covers the data within its spacing."""
    points = rng.random((3000, 2))
    tree = cKDTree(points)
    for nodes, spacing in level_hierarchy(points, 6.0, 0.5)[:-1]:
        level = points[nodes]
        assert cKDTree(level).query(level, k=2)[0][:, 1].min() >= spacing
        assert cKDTree(level).query(points)[0].max() < spacing
        assert len(tree.query_ball_point(level[0], 1e-9)) == 1


def test_coarsest_support_stays_within_the_data_extent(rng):
    points = rng.random((3000, 3))
    levels = level_hierarchy(points, 6.0, 0.5)
    extent = np.linalg.norm(points - points.mean(axis=0), axis=1).max() * 2
    assert 6.0 * levels[0][1] <= extent


# --- interpolation properties ----------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dim", [2, 3])
def test_reproduces_data_at_nodes(kind, dim):
    interp, points, values = fitted(kind, dim, n=1500)
    assert len(interp.levels) >= 2
    np.testing.assert_allclose(interp(points), values, atol=1e-8)


@pytest.mark.parametrize("dim", [2, 3])
def test_divergence_free_result_has_zero_divergence(dim, rng):
    interp, _, values = fitted(DIVERGENCE_FREE, dim, n=1500)
    points = rng.uniform(0.2, 0.8, size=(30, dim))
    divergence = np.trace(jacobian(interp, points, h=1e-5), axis1=1, axis2=2)
    assert np.abs(divergence).max() < 1e-5 * np.abs(values).max()


@pytest.mark.parametrize("dim", [2, 3])
def test_curl_free_result_has_symmetric_jacobian(dim, rng):
    interp, _, values = fitted(CURL_FREE, dim, n=1500)
    points = rng.uniform(0.2, 0.8, size=(30, dim))
    jac = jacobian(interp, points, h=1e-5)
    assert np.abs(jac - np.swapaxes(jac, 1, 2)).max() < 1e-5 * np.abs(values).max()


@pytest.mark.parametrize("kind", KINDS)
def test_converges_where_a_single_level_saturates(kind, rng):
    """With support proportional to the spacing, one level stops improving; multilevel does not."""
    queries = rng.uniform(0.15, 0.85, size=(2000, 2))
    exact = FIELDS[kind](queries)
    multi_errors, single_errors = [], []
    for n in [250, 1000, 4000]:
        points = qmc.Halton(2, seed=0).random(n)
        values = FIELDS[kind](points)
        multi = MULTILEVEL[kind](support_factor=6.0).fit(points, values)
        spacing = multi.levels[-1].spacing
        single = SINGLE[kind](6.0 * spacing).fit(points, values)
        multi_errors.append(np.abs(multi(queries) - exact).max())
        single_errors.append(np.abs(single(queries) - exact).max())
    assert all(fine < 0.4 * coarse for coarse, fine in pairwise(multi_errors)), (
        multi_errors
    )
    assert multi_errors[-1] < 0.01 * single_errors[-1], (multi_errors, single_errors)


@pytest.mark.parametrize("kind", KINDS)
def test_a_single_level_hierarchy_equals_the_single_level_interpolant(kind, rng):
    points = rng.random((30, 2))
    values = FIELDS[kind](points)
    multi = MULTILEVEL[kind](support_factor=6.0).fit(points, values)
    assert len(multi.levels) == 1
    single = SINGLE[kind](multi.levels[0].support_radius).fit(points, values)
    queries = rng.random((50, 2))
    np.testing.assert_allclose(multi(queries), single(queries), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("dim", [2, 3])
def test_potential_gradient_is_the_interpolant(dim, rng):
    interp, _, values = fitted(CURL_FREE, dim, n=1500)
    points = rng.uniform(0.2, 0.8, size=(20, dim))
    gradient = jacobian(lambda x: interp.potential(x)[..., None], points, h=1e-5)[:, 0]
    np.testing.assert_allclose(
        gradient, interp(points), atol=1e-5 * np.abs(values).max()
    )


def test_stream_function_perpendicular_gradient_is_the_interpolant(rng):
    interp, _, values = fitted(DIVERGENCE_FREE, 2, n=1500)
    points = rng.uniform(0.2, 0.8, size=(20, 2))
    gradient = jacobian(lambda x: interp.stream_function(x)[..., None], points, h=1e-5)[
        :, 0
    ]
    np.testing.assert_allclose(
        np.stack([gradient[:, 1], -gradient[:, 0]], 1),
        interp(points),
        atol=1e-5 * np.abs(values).max(),
    )


# --- levels summary and solvers --------------------------------------------------------------


def test_levels_summary(rng):
    interp, points, _ = fitted(DIVERGENCE_FREE, 2, n=4000, support_factor=5.0)
    levels = interp.levels
    assert levels[-1].n_points == len(points)
    assert [level.n_points for level in levels] == sorted(
        level.n_points for level in levels
    )
    for level in levels:
        assert level.support_radius == pytest.approx(5.0 * level.spacing)
        assert level.solve_info.unknowns == 2 * level.n_points
    # Large intermediate (quasi-uniform) levels use CG with a bounded number of iterations
    intermediate = [level for level in levels[:-1] if level.n_points * 2 > 3000]
    assert intermediate and all(lv.solve_info.solver == "cg" for lv in intermediate)
    assert all(lv.solve_info.iterations < 1000 for lv in intermediate)


def test_partial_sums_over_the_coarsest_levels(rng):
    interp, points, values = fitted(CURL_FREE, 2, n=2000)
    queries = rng.random((50, 2))
    count = len(interp.levels)
    partial = [interp(queries, levels=j) for j in range(1, count + 1)]
    np.testing.assert_allclose(partial[-1], interp(queries))
    np.testing.assert_allclose(
        partial[1] - partial[0], interp._fitted[1](queries), atol=1e-12
    )
    np.testing.assert_allclose(
        interp.potential(queries, levels=1), interp._fitted[0].potential(queries)
    )
    # Coarse levels alone do not interpolate the data; all of them together do
    assert np.abs(interp(points, levels=1) - values).max() > 1e-3
    for bad in [0, count + 1, 1.5]:
        with pytest.raises(ValueError, match="levels"):
            interp(queries, levels=bad)


def test_explicit_solver_applies_to_every_level():
    interp, _, _ = fitted(CURL_FREE, 2, n=1500, solver="direct")
    assert {level.solve_info.solver for level in interp.levels} == {"direct"}


# --- API -------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_output_shapes(kind):
    interp, _, _ = fitted(kind, 2, n=300)
    assert interp(np.full(2, 0.5)).shape == (2,)
    assert interp(np.full((4, 5, 2), 0.5)).shape == (4, 5, 2)
    assert interp.dim == 2


def test_before_fit():
    interp = dfi.MultilevelCurlFreeInterpolant()
    assert interp.levels == () and interp.dim is None
    with pytest.raises(RuntimeError, match="fit"):
        interp(np.zeros(2))


@pytest.mark.parametrize(
    "option, message",
    [
        ({"support_factor": 0}, "support_factor"),
        ({"support_factor": np.nan}, "support_factor"),
        ({"refinement": 1.0}, "refinement"),
        ({"refinement": 0}, "refinement"),
        ({"k": 0}, "k = 0"),
        ({"nu": 2.5}, "integer"),
        ({"solver": "cholesky"}, "solver"),
    ],
)
def test_invalid_options_raise(option, message):
    with pytest.raises(ValueError, match=message):
        dfi.MultilevelDivergenceFreeInterpolant(**option)


def test_needs_two_points():
    with pytest.raises(ValueError, match="at least 2"):
        dfi.MultilevelDivergenceFreeInterpolant().fit(
            np.zeros((1, 2)), np.zeros((1, 2))
        )


def test_duplicates_raise_before_any_work_and_keep_the_previous_fit():
    interp, points, values = fitted(DIVERGENCE_FREE, 2, n=300)
    with pytest.raises(ValueError, match="coincide"):
        interp.fit(np.vstack([points, points[:1]]), np.vstack([values, values[:1]]))
    np.testing.assert_allclose(interp(points), values, atol=1e-8)


def test_lstsq_tolerates_duplicates(rng):
    points = rng.random((60, 2))
    values = FIELDS[CURL_FREE](points)
    interp = dfi.MultilevelCurlFreeInterpolant(solver="lstsq")
    interp.fit(np.vstack([points, points[:1]]), np.vstack([values, values[:1]]))
    np.testing.assert_allclose(interp(points), values, atol=1e-6)


def test_level_spacings_do_not_change_with_later_attribute_changes():
    interp, _, _ = fitted(CURL_FREE, 2, n=1500)
    before = [level.spacing for level in interp.levels]
    interp.support_factor = 12.0
    assert [level.spacing for level in interp.levels] == before


def test_coarse_levels_with_fewer_points_than_dimensions(rng):
    """An outlier can leave the coarsest 4D level with 2 points: not a transposed input."""
    points = np.vstack([rng.random((400, 4)), np.full((1, 4), 100.0)])
    values = rng.normal(size=points.shape)
    # A small support factor keeps 4D neighbourhoods cheap; accuracy is not tested here
    interp = dfi.MultilevelCurlFreeInterpolant(support_factor=3.0).fit(points, values)
    assert interp.levels[0].n_points < 4
    np.testing.assert_allclose(interp(points), values, atol=1e-8)


def test_invalid_dimension_is_rejected_before_building_levels(monkeypatch):
    from divergence_free_interpolant import _multilevel

    def unreachable(*args):
        raise AssertionError("the level hierarchy should not be built")

    monkeypatch.setattr(_multilevel, "level_hierarchy", unreachable)
    points = np.random.default_rng(0).random((100, 1))
    with pytest.raises(ValueError, match="dimension"):
        dfi.MultilevelCurlFreeInterpolant().fit(points, points)
    points = np.random.default_rng(0).random((100, 2))
    with pytest.raises(ValueError, match="nu = 3"):
        dfi.MultilevelCurlFreeInterpolant(k=3, nu=3).fit(points, points)


def test_lstsq_accepts_all_points_coinciding():
    interp = dfi.MultilevelDivergenceFreeInterpolant(solver="lstsq")
    interp.fit(np.zeros((5, 2)), np.ones((5, 2)))
    assert len(interp.levels) == 1
    np.testing.assert_allclose(interp(np.zeros(2)), [1.0, 1.0], atol=1e-8)


def test_levels_argument_rejects_booleans():
    interp, _, _ = fitted(CURL_FREE, 2, n=300)
    with pytest.raises(ValueError, match="levels"):
        interp(np.zeros(2), levels=True)


def test_stream_function_only_in_2d():
    interp, _, _ = fitted(DIVERGENCE_FREE, 3, n=300)
    with pytest.raises(ValueError, match="2D"):
        interp.stream_function(np.zeros(3))
