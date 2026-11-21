"""Public API: shapes, validation and instance independence."""

import numpy as np
import pytest

import Divergence_Free_Interpolant as dfi

from conftest import conditioned


@pytest.mark.parametrize("dim", [2, 3])
def test_output_shapes(dim):
    interp, _, _ = conditioned(dim, n=15)
    assert interp(*[0.5] * dim).shape == (dim,)
    assert interp(*[np.linspace(0, 1, 7)] * dim).shape == (7, dim)
    grids = np.meshgrid(*[np.linspace(0, 1, 4)] * dim, indexing="ij")
    assert interp(*grids).shape == (4,) * dim + (dim,)


@pytest.mark.parametrize("dim", [2, 3])
def test_vectorised_call_matches_pointwise_calls(dim, rng):
    interp, _, _ = conditioned(dim, n=15)
    points = rng.random((10, dim))
    vectorised = interp(*points.T)
    pointwise = np.array([interp(*p) for p in points])
    np.testing.assert_allclose(vectorised, pointwise, rtol=1e-12, atol=1e-14)


def test_two_dimensional_interpolant_rejects_z():
    interp, _, _ = conditioned(2, n=10)
    with pytest.raises(TypeError, match="2 coordinate"):
        interp(0.1, 0.2, 0.3)


def test_three_dimensional_interpolant_requires_z():
    interp, _, _ = conditioned(3, n=10)
    with pytest.raises(TypeError, match="3 coordinate"):
        interp(0.1, 0.2)


def test_calling_before_condition_raises():
    with pytest.raises(RuntimeError, match="condition"):
        dfi.interpolant(dim=2)(0.1, 0.2)


def test_instances_are_independent():
    rng = np.random.default_rng(0)
    first, second = dfi.interpolant(dim=2), dfi.interpolant(dim=2)
    nodes_first, nodes_second = rng.random((10, 2)), rng.random((10, 2)) + 5
    first.condition(nodes_first, np.tile([1.0, 0.0], (10, 1)))
    second.condition(nodes_second, np.tile([0.0, 1.0], (10, 1)))
    np.testing.assert_allclose(first(*nodes_first[0]), [1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(second(*nodes_second[0]), [0.0, 1.0], atol=1e-9)


def test_reconditioning_replaces_previous_data():
    interp, nodes, _ = conditioned(2, n=10)
    interp.condition(nodes, np.zeros_like(nodes))
    np.testing.assert_allclose(interp(*nodes.T), 0, atol=1e-12)


@pytest.mark.parametrize("dim", [1, 4])
def test_unsupported_dimension_raises(dim):
    with pytest.raises(ValueError, match="dimension"):
        dfi.interpolant(dim=dim)


@pytest.mark.parametrize("nu, k, dim", [(2, 1, 2), (3, 2, 3), (4, 3, 2)])
def test_non_positive_definite_nu_raises(nu, k, dim):
    """nu < floor(dim/2) + k + 1 does not give a positive definite kernel."""
    with pytest.raises(ValueError, match=f"nu = {nu}"):
        dfi.interpolant(nu=nu, k=k, dim=dim)


def test_k_zero_raises():
    """k = 0: psi is not C^2, so the kernel is singular at the origin."""
    with pytest.raises(ValueError, match="k = 0"):
        dfi.interpolant(nu=5, k=0, dim=2)


@pytest.mark.parametrize("method", ["cholesky", None])
def test_unknown_method_raises(method):
    interp = dfi.interpolant(dim=2)
    nodes = np.random.default_rng(0).random((5, 2))
    with pytest.raises(KeyError, match="not found"):
        interp.condition(nodes, nodes, method=method)


@pytest.mark.parametrize(
    "positions_shape, values_shape",
    [
        ((10, 3), (3, 10)),  # values in the old (dim, N) layout
        ((3, 10), (3, 10)),  # both in the old (dim, N) layout
        ((10, 2), (10, 2)),  # 2D data for a 3D interpolant
        ((10, 3), (9, 3)),  # mismatched number of points
        ((30,), (30,)),  # flat arrays
    ],
)
def test_wrong_input_shapes_raise(positions_shape, values_shape):
    interp = dfi.interpolant(dim=3)
    with pytest.raises(ValueError, match="shape"):
        interp.condition(np.ones(positions_shape), np.ones(values_shape))


def test_failed_condition_keeps_previous_state():
    interp, nodes, values = conditioned(2, n=10)
    duplicated = np.vstack([nodes, nodes[:1]])  # repeated node -> singular matrix
    with pytest.raises(np.linalg.LinAlgError):
        interp.condition(duplicated, np.vstack([values, values[:1] + 1]))
    np.testing.assert_allclose(interp(*nodes.T), values, atol=1e-9)


def test_modifying_input_arrays_after_condition_has_no_effect():
    interp, nodes, values = conditioned(2, n=10)
    probe = nodes.copy()
    before = interp(*probe.T)
    nodes += 1.0
    values *= -1.0
    np.testing.assert_array_equal(interp(*probe.T), before)
