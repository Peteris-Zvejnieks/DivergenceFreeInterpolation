"""Public API: shapes, validation, state handling and the deprecated 0.1 interface."""

import numpy as np
import pytest

import divergence_free_interpolant as dfi

from conftest import INTERPOLANTS, KINDS, fitted
from reference_kernel import reference_kernel

# --- shapes ----------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dim", [2, 3])
def test_output_shapes(kind, dim):
    interp, _, _ = fitted(kind, dim, n=15)
    assert interp(np.full(dim, 0.5)).shape == (dim,)
    assert interp(np.full((7, dim), 0.5)).shape == (7, dim)
    grid = np.stack(np.meshgrid(*[np.linspace(0, 1, 4)] * dim, indexing="ij"), axis=-1)
    assert interp(grid).shape == (4,) * dim + (dim,)


def test_scalar_potential_shapes():
    curl_free, _, _ = fitted("curl_free", 3, n=15)
    assert curl_free.potential(np.full(3, 0.5)).shape == ()
    assert curl_free.potential(np.full((5, 6, 3), 0.5)).shape == (5, 6)
    div_free, _, _ = fitted("divergence_free", 2, n=15)
    assert div_free.stream_function(np.full((7, 2), 0.5)).shape == (7,)


@pytest.mark.parametrize("kind", KINDS)
def test_batched_evaluation_matches_pointwise(kind, rng):
    interp, _, _ = fitted(kind, 3, n=15)
    points = rng.random((10, 3))
    np.testing.assert_allclose(
        interp(points), np.array([interp(p) for p in points]), rtol=1e-13, atol=1e-15
    )


def test_fit_returns_self_and_infers_dimension():
    interp = dfi.CurlFreeInterpolant(0.5)
    assert interp.dim is None
    points = np.random.default_rng(0).random((10, 3))
    assert interp.fit(points, points) is interp
    assert interp.dim == 3


# --- validation ------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_query_with_wrong_dimension_raises(kind):
    interp, _, _ = fitted(kind, 2, n=10)
    with pytest.raises(ValueError, match=r"\(\.\.\., 2\)"):
        interp(np.zeros((5, 3)))
    with pytest.raises(ValueError, match="shape"):
        interp(0.5)


@pytest.mark.parametrize("kind", KINDS)
def test_non_finite_input_raises(kind):
    points = np.random.default_rng(0).random((10, 2))
    bad = points.copy()
    bad[3, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        INTERPOLANTS[kind](0.5).fit(bad, points)
    with pytest.raises(ValueError, match="finite"):
        INTERPOLANTS[kind](0.5).fit(points, bad)
    interp = INTERPOLANTS[kind](0.5).fit(points, points)
    with pytest.raises(ValueError, match="finite"):
        interp(bad)


@pytest.mark.parametrize(
    "points_shape, values_shape",
    [
        ((10, 3), (3, 10)),  # values transposed, the old (dim, N) layout
        ((10, 3), (9, 3)),  # mismatched number of points
        ((30,), (30,)),  # flat arrays
        ((0, 3), (0, 3)),  # no points
    ],
)
def test_wrong_input_shapes_raise(points_shape, values_shape):
    with pytest.raises(ValueError, match="shape"):
        dfi.DivergenceFreeInterpolant(0.5).fit(
            np.ones(points_shape), np.ones(values_shape)
        )


def test_one_dimensional_data_raises():
    with pytest.raises(ValueError, match="dimension"):
        dfi.CurlFreeInterpolant(0.5).fit(
            np.random.default_rng(0).random((5, 1)), np.ones((5, 1))
        )


@pytest.mark.parametrize("support", [0, -1.0, np.nan, np.inf])
def test_invalid_support_radius_raises(support):
    with pytest.raises(ValueError, match="support_radius"):
        dfi.DivergenceFreeInterpolant(support)


def test_invalid_k_raises():
    with pytest.raises(ValueError, match="k = 0"):
        dfi.DivergenceFreeInterpolant(0.5, k=0)


@pytest.mark.parametrize("nu, k, dim", [(2, 1, 2), (3, 2, 3), (4, 3, 2), (3, 1, 4)])
def test_non_positive_definite_nu_raises_on_fit(nu, k, dim):
    """nu < floor(dim/2) + k + 1 does not give a positive definite kernel in that dimension."""
    interp = dfi.DivergenceFreeInterpolant(0.5, nu=nu, k=k)
    points = np.random.default_rng(0).random((5, dim))
    with pytest.raises(ValueError, match=f"nu = {nu}"):
        interp.fit(points, points)


def test_integral_float_parameters_are_accepted():
    """e.g. values read from a config file or a numpy float array."""
    points = np.random.default_rng(0).random((10, 2))
    as_float = dfi.DivergenceFreeInterpolant(0.8, k=3.0, nu=6.0).fit(points, points)
    as_int = dfi.DivergenceFreeInterpolant(0.8, k=3, nu=6).fit(points, points)
    assert (as_float.k, as_float.nu) == (3, 6)
    np.testing.assert_array_equal(as_float(points), as_int(points))


@pytest.mark.parametrize("option", [{"k": 2.5}, {"nu": 5.5}, {"k": "3"}, {"k": True}])
def test_non_integral_parameters_raise(option):
    with pytest.raises(ValueError, match="must be an integer"):
        dfi.DivergenceFreeInterpolant(0.5, **option)


def test_transposed_input_is_detected():
    points = np.random.default_rng(0).random((10, 3))
    with pytest.raises(ValueError, match="transposed"):
        dfi.DivergenceFreeInterpolant(0.5).fit(points.T, points.T)


def test_fewer_points_than_dimensions_is_fine_in_2d_and_3d():
    points = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    interp = dfi.CurlFreeInterpolant(1.0).fit(points, points)
    np.testing.assert_allclose(interp(points), points, atol=1e-12)


@pytest.mark.parametrize("tol", [0, -1e-8, 1.0, np.inf, np.nan])
def test_invalid_tolerance_raises(tol):
    with pytest.raises(ValueError, match="tol"):
        dfi.DivergenceFreeInterpolant(0.5, tol=tol)
    with pytest.raises(ValueError, match="tol"):
        dfi.MultilevelDivergenceFreeInterpolant(tol=tol)


def test_unknown_solver_raises():
    with pytest.raises(ValueError, match="solver"):
        dfi.DivergenceFreeInterpolant(0.5, solver="cholesky")


def test_calling_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        dfi.CurlFreeInterpolant(0.5)(np.zeros(2))
    with pytest.raises(RuntimeError, match="fit"):
        dfi.CurlFreeInterpolant(0.5).potential(np.zeros(2))


def test_stream_function_only_in_2d():
    interp, _, _ = fitted("divergence_free", 3, n=10)
    with pytest.raises(ValueError, match="2D"):
        interp.stream_function(np.zeros(3))


# --- state handling --------------------------------------------------------------------------


def test_instances_are_independent():
    rng = np.random.default_rng(0)
    first, second = (
        dfi.DivergenceFreeInterpolant(1.0),
        dfi.DivergenceFreeInterpolant(1.0),
    )
    points_first, points_second = rng.random((10, 2)), rng.random((10, 2)) + 5
    first.fit(points_first, np.tile([1.0, 0.0], (10, 1)))
    second.fit(points_second, np.tile([0.0, 1.0], (10, 1)))
    np.testing.assert_allclose(first(points_first[0]), [1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(second(points_second[0]), [0.0, 1.0], atol=1e-9)


def test_refitting_replaces_previous_data():
    interp, points, _ = fitted("curl_free", 2, n=10)
    interp.fit(points, np.zeros_like(points))
    np.testing.assert_allclose(interp(points), 0, atol=1e-12)


@pytest.mark.parametrize("solver", ["auto", "direct", "cg"])
def test_duplicate_points_raise_and_keep_previous_state(solver):
    interp, points, values = fitted("divergence_free", 2, n=10, solver=solver)
    duplicated = np.vstack([points, points[:1]])
    with pytest.raises(ValueError, match="coincide"):
        interp.fit(duplicated, np.vstack([values, values[:1] + 1]))
    np.testing.assert_allclose(interp(points), values, atol=1e-9)


def test_lstsq_tolerates_duplicate_points():
    rng = np.random.default_rng(0)
    points = rng.random((10, 2))
    values = rng.normal(size=(10, 2))
    interp = dfi.DivergenceFreeInterpolant(1.0, solver="lstsq")
    interp.fit(np.vstack([points, points[:1]]), np.vstack([values, values[:1]]))
    np.testing.assert_allclose(interp(points), values, atol=1e-6)


def test_failed_solve_keeps_previous_state(monkeypatch):
    """A solver failure leaves the old fit in place."""
    from divergence_free_interpolant import _interpolant

    interp, points, values = fitted("curl_free", 2, n=10)

    def failing_solve(*args, **kwargs):
        raise np.linalg.LinAlgError("did not converge")

    monkeypatch.setattr(_interpolant, "solve", failing_solve)
    with pytest.raises(np.linalg.LinAlgError, match="did not converge"):
        interp.fit(points + 0.01, values)
    np.testing.assert_allclose(interp(points), values, atol=1e-9)


def test_auto_uses_direct_for_mid_size_3d_systems():
    """CG is not reliably faster on arbitrary 3D data, so mid-size systems are factorised."""
    from divergence_free_interpolant._solvers import resolve_solver

    assert resolve_solver("auto", 3 * 4000, 3) == "direct"
    assert resolve_solver("auto", 3 * 10_000, 3) == "cg"
    assert resolve_solver("auto", 2 * 100_000, 2) == "direct"


def test_auto_falls_back_to_direct_when_cg_fails(monkeypatch):
    from divergence_free_interpolant import _interpolant, _solvers

    monkeypatch.setattr(_interpolant, "resolve_solver", lambda *args: "cg")
    monkeypatch.setattr(_solvers.splinalg, "cg", lambda *args, **kwargs: (args[1], 17))
    interp, points, values = fitted("divergence_free", 3, n=50)
    assert interp.solve_info.solver == "direct"
    np.testing.assert_allclose(interp(points), values, atol=1e-9)
    # An explicitly requested CG solve still fails loudly
    with pytest.raises(np.linalg.LinAlgError, match="did not reach rtol"):
        fitted("divergence_free", 3, n=50, solver="cg")


def test_cg_iterations_are_capped(monkeypatch):
    from divergence_free_interpolant import _solvers

    calls = []
    original = _solvers.splinalg.cg

    def spy(*args, **kwargs):
        calls.append(kwargs["maxiter"])
        return original(*args, **kwargs)

    monkeypatch.setattr(_solvers.splinalg, "cg", spy)
    # 400 unknowns would allow 4000 iterations; the cap lowers that
    monkeypatch.setattr(_solvers, "CG_MAX_ITERATIONS", 3000)
    fitted("curl_free", 2, n=200, support=0.2, solver="cg")
    assert calls == [3000]


def test_cg_reports_unreachable_tolerance(monkeypatch):
    from divergence_free_interpolant import _solvers

    monkeypatch.setattr(_solvers.splinalg, "cg", lambda *args, **kwargs: (args[1], 17))
    with pytest.raises(np.linalg.LinAlgError, match="did not reach rtol"):
        fitted("curl_free", 2, n=10, solver="cg")


def test_changing_parameters_after_fit_only_affects_the_next_fit():
    interp, points, values = fitted("divergence_free", 2, n=20, support=0.5)
    before = interp(points)
    interp.support_radius = 0.8
    np.testing.assert_array_equal(interp(points), before)
    # A failed refit with the new radius keeps the old fit, evaluated with its own radius
    with pytest.raises(ValueError, match="coincide"):
        interp.fit(np.vstack([points, points[:1]]), np.vstack([values, values[:1]]))
    np.testing.assert_array_equal(interp(points), before)
    interp.fit(points, values)
    assert interp._fit.support_radius == 0.8


@pytest.mark.parametrize("tol", [1e-4, 1e-6, 1e-8])
def test_cg_tolerance_is_respected(tol):
    """A loose CG tolerance is a valid choice, not a singular matrix."""
    interp, points, values = fitted(
        "curl_free", 2, n=40, support=0.6, solver="cg", tol=tol
    )
    error = np.linalg.norm(interp(points) - values) / np.linalg.norm(values)
    assert error < 10 * tol


def test_cg_breakdown_message(monkeypatch):
    from divergence_free_interpolant import _solvers

    monkeypatch.setattr(_solvers.splinalg, "cg", lambda *args, **kwargs: (args[1], -1))
    with pytest.raises(np.linalg.LinAlgError, match="broke down"):
        fitted("curl_free", 2, n=10, solver="cg")


def test_modifying_input_arrays_after_fit_has_no_effect():
    interp, points, values = fitted("divergence_free", 2, n=10)
    probe = points.copy()
    before = interp(probe)
    points += 1.0
    values *= -1.0
    np.testing.assert_array_equal(interp(probe), before)


# --- deprecated 0.1 interface ----------------------------------------------------------------


def test_legacy_interpolant_warns_and_matches_new_api():
    rng = np.random.default_rng(0)
    points, values = rng.random((20, 2)), rng.normal(size=(20, 2))
    with pytest.deprecated_call():
        legacy = dfi.interpolant(nu=5, k=3, dim=2)
    legacy.condition(points, values, support_radii=0.7)
    new = dfi.DivergenceFreeInterpolant(0.7, nu=5, k=3).fit(points, values)
    grid = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 6))
    np.testing.assert_allclose(legacy(*grid), new(np.stack(grid, axis=-1)), rtol=1e-12)
    assert legacy(0.3, 0.4).shape == (2,)


@pytest.mark.parametrize("method", ["linsolve", "lstsq", "SVD", "penrose"])
def test_legacy_methods_still_accepted(method):
    rng = np.random.default_rng(0)
    points, values = rng.random((15, 3)), rng.normal(size=(15, 3))
    with pytest.deprecated_call():
        legacy = dfi.interpolant(dim=3)
    legacy.condition(points, values, support_radii=0.8, method=method)
    np.testing.assert_allclose(legacy(*points.T), values, atol=1e-8)


@pytest.mark.parametrize("dim", [2, 3])
def test_legacy_call_accepts_keyword_coordinates(dim):
    rng = np.random.default_rng(0)
    points, values = rng.random((15, dim)), rng.normal(size=(15, dim))
    with pytest.deprecated_call():
        legacy = dfi.interpolant(dim=dim)
    legacy.condition(points, values, support_radii=0.8)
    names = "xyz"[:dim]
    query = rng.random((4, dim))
    expected = legacy(*query.T)
    np.testing.assert_array_equal(
        legacy(**dict(zip(names, query.T, strict=True))), expected
    )
    np.testing.assert_array_equal(
        legacy(query[:, 0], **dict(zip(names[1:], query.T[1:], strict=True))), expected
    )


def test_legacy_attributes():
    rng = np.random.default_rng(0)
    points, values = rng.random((12, 2)), rng.normal(size=(12, 2))
    with pytest.deprecated_call():
        legacy = dfi.interpolant(nu=5, k=3, dim=2)
    np.testing.assert_array_equal(legacy.zero_kernel, np.zeros((2, 2)))
    for name in ["XY", "sol", "N", "support_radii", "method"]:
        assert not hasattr(legacy, name)  # as in 0.1: only set by condition()
    legacy.condition(points, values, support_radii=0.6, method="SVD")
    np.testing.assert_array_equal(legacy.XY, points)
    assert legacy.sol.shape == (12, 2, 1)
    assert (legacy.N, legacy.support_radii, legacy.method) == (12, 0.6, "SVD")
    with pytest.raises(ValueError, match="read-only"):
        legacy.XY[0, 0] = 1.0
    # The coefficients and the unscaled kernel reproduce the data like in 0.1:
    # u(x_i) = sum_j delta^-d K((x_i - x_j) / delta) sol_j
    differences = (points[:, None, :] - points[None, :, :]) / 0.6
    radius = np.linalg.norm(differences, axis=-1)
    blocks = legacy.kernel(differences[..., 0], differences[..., 1], radius)
    reproduced = 0.6**-2 * np.einsum("abij,jb->ia", blocks, legacy.sol[:, :, 0])
    np.testing.assert_allclose(reproduced, values, atol=1e-9)


@pytest.mark.parametrize("dim", [2, 3])
def test_legacy_kernel_matches_symbolic_reference(dim, rng):
    with pytest.deprecated_call():
        legacy = dfi.interpolant(nu=5, k=3, dim=dim)
    points = rng.uniform(-0.55, 0.55, size=(20, dim))
    blocks = legacy.kernel(*points.T, np.linalg.norm(points, axis=1))
    expected = reference_kernel("divergence_free", 5, 3, dim)(points)
    np.testing.assert_allclose(
        np.moveaxis(blocks, -1, 0), expected, rtol=1e-10, atol=1e-14
    )


def test_legacy_errors():
    with pytest.deprecated_call(), pytest.raises(ValueError, match="nu"):
        dfi.interpolant(nu=3, k=3, dim=2)
    with pytest.deprecated_call():
        legacy = dfi.interpolant(dim=2)
    with pytest.raises(RuntimeError, match="condition"):
        legacy(0.1, 0.2)
    points = np.random.default_rng(0).random((10, 2))
    with pytest.raises(KeyError, match="not found"):
        legacy.condition(points, points, method="cholesky")
    with pytest.raises(ValueError, match="shape"):
        legacy.condition(points.T, points.T)
    legacy.condition(points, points)
    with pytest.raises(TypeError, match="2 coordinate"):
        legacy(0.1, 0.2, 0.3)
    with pytest.raises(TypeError, match="2 coordinate"):
        legacy(x=0.1, y=0.2, z=0.3)
    with pytest.deprecated_call():
        legacy_3d = dfi.interpolant(dim=3)
    legacy_3d.condition(np.random.default_rng(0).random((10, 3)), np.ones((10, 3)))
    with pytest.raises(TypeError, match="3 coordinate"):
        legacy_3d(x=0.1, y=0.2)
    with pytest.deprecated_call(), pytest.raises(ValueError, match="dimension"):
        dfi.interpolant(dim=4)
