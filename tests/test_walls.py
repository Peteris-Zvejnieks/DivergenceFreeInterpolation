"""Walls built into the divergence-free kernel ("tier 2"): exact, everywhere along the wall."""

import pickle

import numpy as np
import pytest

import divergence_free_interpolant as dfi
from divergence_free_interpolant import _core
from divergence_free_interpolant._kernel import DIVERGENCE_FREE, get_kernel
from divergence_free_interpolant.walls import (
    Circle,
    Cylinder,
    Line,
    Plane,
    Sphere,
    restrict_functionals,
    wall_function,
)

from conftest import div_free_field, jacobian

# Each case: walls, and a sampler of (points on the walls, their unit normals)


def _circle_points(center, radius, n, rng):
    theta = rng.uniform(0, 2 * np.pi, n)
    normals = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return np.asarray(center) + radius * normals, normals


def _sphere_points(center, radius, n, rng):
    normals = rng.normal(size=(n, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return np.asarray(center) + radius * normals, normals


def _plane_points(point, normal, n, rng):
    normal = np.asarray(normal, float) / np.linalg.norm(normal)
    offsets = rng.uniform(-1, 1, (n, len(normal)))
    offsets -= np.outer(offsets @ normal, normal)
    return np.asarray(point) + offsets, np.tile(normal, (n, 1))


def _cylinder_points(point, axis, radius, n, rng):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    directions = rng.normal(size=(n, 3))
    directions -= np.outer(directions @ axis, axis)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    along = rng.uniform(-1, 1, (n, 1)) * axis
    return np.asarray(point) + along + radius * directions, directions


CASES = {
    "circle": (
        2,
        lambda c: [Circle((0.1, 0.0), 0.3, band=0.4, condition=c)],
        lambda rng: _circle_points((0.1, 0.0), 0.3, 200, rng),
    ),
    "circle inside": (
        2,
        lambda c: [Circle((0.0, 0.0), 1.2, band=0.5, condition=c, inside=True)],
        lambda rng: _circle_points((0.0, 0.0), 1.2, 200, rng),
    ),
    "line": (
        2,
        lambda c: [Line((0.0, -0.9), (0.2, 1.0), band=0.4, condition=c)],
        lambda rng: _plane_points((0.0, -0.9), (0.2, 1.0), 200, rng),
    ),
    "sphere": (
        3,
        lambda c: [Sphere((0.0, 0.1, 0.0), 0.35, band=0.4, condition=c)],
        lambda rng: _sphere_points((0.0, 0.1, 0.0), 0.35, 200, rng),
    ),
    "plane": (
        3,
        lambda c: [Plane((0.0, 0.0, -0.9), (0.0, 0.3, 1.0), band=0.4, condition=c)],
        lambda rng: _plane_points((0.0, 0.0, -0.9), (0.0, 0.3, 1.0), 200, rng),
    ),
    "cylinder": (
        3,
        lambda c: [
            Cylinder((0.0, 0.0, 0.0), (1.0, 0.5, 0.2), 0.3, band=0.4, condition=c)
        ],
        lambda rng: _cylinder_points((0.0, 0.0, 0.0), (1.0, 0.5, 0.2), 0.3, 200, rng),
    ),
}


def _flow_points(walls, dim, n, rng, gap=0.15):
    """Random points in [-1, 1]^dim on the flow side of every wall, at least `gap` away."""
    candidates = rng.uniform(-1, 1, (20 * n, dim))
    keep = np.ones(len(candidates), dtype=bool)
    for wall in walls:
        keep &= wall.signed_distance(candidates)[0] > gap
    return candidates[keep][:n]


def _fit(case, condition, model=None, n=None, **options):
    dim, make_walls, sample_wall = CASES[case]
    walls = make_walls(condition)
    rng = np.random.default_rng(0)
    points = _flow_points(walls, dim, n or (120 if dim == 2 else 300), rng)
    values = div_free_field(points)
    if model is None:
        model = dfi.DivergenceFreeInterpolant(0.8, walls=walls, **options)
    model.fit(points, values)
    return model, points, values, sample_wall(rng)


# --- the walls ---------------------------------------------------------------------------------


def test_wall_validation_and_repr():
    with pytest.raises(ValueError, match="band"):
        Circle((0, 0), 1.0, band=0.0)
    with pytest.raises(ValueError, match="condition"):
        Circle((0, 0), 1.0, band=0.5, condition="free_slip")
    with pytest.raises(ValueError, match="center"):
        Circle((0, 0, 0), 1.0, band=0.5)
    with pytest.raises(ValueError, match="radius"):
        Sphere((0, 0, 0), -1.0, band=0.5)
    with pytest.raises(ValueError, match="normal"):
        Line((0, 0), (0, 0), band=0.5)
    with pytest.raises(ValueError, match="smaller than the radius"):
        Circle((0, 0), 1.0, band=1.0, inside=True)
    with pytest.raises(ValueError, match="smaller than the radius"):
        Cylinder((0, 0, 0), (0, 0, 1), 0.5, band=0.6, inside=True)
    with pytest.raises(TypeError):
        Circle((0, 0), 1.0, 0.5)

    wall = Line((0, 1), (0, 2), band=0.5, condition="no_slip")
    assert repr(wall) == (
        "Line(point=(0, 1), normal=(0, 1), band=0.5, condition='no_slip')"
    )
    assert wall == Line((0.0, 1.0), (0, 5), band=0.5, condition="no_slip")
    assert wall != Line((0.0, 1.0), (0, 5), band=0.5)
    assert len({wall, Line((0.0, 1.0), (0, 5), band=0.5, condition="no_slip")}) == 1
    assert pickle.loads(pickle.dumps(wall)) == wall


def test_walls_argument_validation():
    circle = Circle((0, 0), 0.3, band=0.4)
    assert dfi.DivergenceFreeInterpolant(1.0, walls=circle).walls == (circle,)
    assert dfi.DivergenceFreeInterpolant(1.0, walls=None).walls == ()
    with pytest.raises(TypeError, match="walls"):
        dfi.DivergenceFreeInterpolant(1.0, walls=3)
    with pytest.raises(TypeError, match="walls"):
        dfi.DivergenceFreeInterpolant(1.0, walls=[object()])
    # Walls are a divergence-free construction
    with pytest.raises(TypeError):
        dfi.CurlFreeInterpolant(1.0, walls=circle)
    with pytest.raises(TypeError):
        dfi.MultilevelCurlFreeInterpolant(walls=circle)
    points = np.random.default_rng(0).uniform(-1, 1, (20, 3))
    with pytest.raises(ValueError, match="2D wall cannot bound 3D"):
        dfi.DivergenceFreeInterpolant(1.0, walls=circle).fit(points, points)
    with pytest.raises(ValueError, match="2D wall cannot bound 3D"):
        dfi.MultilevelDivergenceFreeInterpolant(walls=circle).fit(points, points)


@pytest.mark.parametrize("case", CASES)
def test_wall_function_gradient(case):
    dim, make_walls, _ = CASES[case]
    walls = make_walls("no_penetration") + make_walls("no_slip")
    points = np.random.default_rng(1).uniform(-1, 1, (50, dim))
    g, gradient = wall_function(walls, points)
    numeric = np.stack(
        [
            (wall_function(walls, points + h)[0] - wall_function(walls, points - h)[0])
            / 2e-6
            for h in np.eye(dim) * 1e-6
        ],
        axis=-1,
    )
    np.testing.assert_allclose(gradient, numeric, atol=1e-6)
    assert g.shape == (50,)
    on_wall, _ = CASES[case][2](np.random.default_rng(2))
    np.testing.assert_allclose(wall_function(walls, on_wall)[0], 0, atol=1e-12)


# --- the kernel --------------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_rust_kernel_matches_numpy(case):
    dim, make_walls, _ = CASES[case]
    walls = tuple(make_walls("no_penetration"))
    kernel = get_kernel(DIVERGENCE_FREE, dim, None, 3, walls)
    rng = np.random.default_rng(3)
    points = _flow_points(walls, dim, 200, rng, gap=0.0)
    delta = 0.5
    indptr, indices, data = _core.gram_matrix(kernel.core, points, delta)
    from scipy import sparse

    rust = sparse.csr_array((data, indices, indptr), shape=(points.size,) * 2).toarray()
    blocks = kernel.pair_blocks(points[:, None], points[None, :], delta)
    numpy = blocks.transpose(0, 2, 1, 3).reshape(points.size, points.size)
    np.testing.assert_allclose(rust, numpy, atol=1e-12 * np.abs(numpy).max())
    # The matrix is symmetric although its blocks are not
    np.testing.assert_allclose(numpy, numpy.T, atol=1e-12 * np.abs(numpy).max())

    coefficients = rng.normal(size=points.shape)
    queries = rng.uniform(-1, 1, (100, dim))
    values = _core.evaluate(kernel.core, points, coefficients, queries, delta)
    expected = np.einsum(
        "qpab,pb->qa",
        kernel.pair_blocks(queries[:, None], points[None], delta),
        coefficients,
    )
    np.testing.assert_allclose(values.reshape(queries.shape), expected, atol=1e-10)
    if dim == 2:
        potential = _core.evaluate_potential(
            kernel.core, points, coefficients, queries, delta
        )
        terms = kernel.pair_potential_terms(
            queries[:, None], points[None], coefficients[None], delta
        )
        np.testing.assert_allclose(potential, terms.sum(axis=1), atol=1e-10)


def test_kernel_is_unchanged_away_from_walls():
    walls = (Circle((0, 0), 0.3, band=0.2),)
    plain = get_kernel(DIVERGENCE_FREE, 2, None, 3)
    walled = get_kernel(DIVERGENCE_FREE, 2, None, 3, walls)
    rng = np.random.default_rng(4)
    x, y = rng.uniform(0.6, 1.0, (30, 2)), rng.uniform(0.6, 1.0, (30, 2))
    np.testing.assert_allclose(
        walled.pair_blocks(x, y, 0.7), plain.pair_blocks(x, y, 0.7), atol=1e-12
    )


# --- fits ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("condition", ["no_penetration", "no_slip"])
def test_fits_satisfy_the_wall_exactly(case, condition, each_backend):
    model, points, values, (on_wall, normals) = _fit(case, condition)
    np.testing.assert_allclose(model(points), values, atol=1e-8)
    u = model(on_wall)
    scale = np.abs(values).max()
    assert np.abs(np.sum(u * normals, axis=1)).max() < 1e-10 * scale
    if condition == "no_slip":
        assert np.abs(u).max() < 1e-10 * scale
    else:
        # The tangential flow along the wall is not constrained
        assert np.abs(u).max() > 1e-3 * scale
    # Exactly divergence-free, also near the wall
    probes = points[:20] + 0.01
    divergence = np.trace(jacobian(model, probes), axis1=1, axis2=2)
    assert np.abs(divergence).max() < 1e-5 * scale


def test_stream_function_with_walls():
    model, points, _, (on_wall, _) = _fit("circle", "no_penetration")
    queries = points[:30] + 0.01
    step = 1e-6
    d_dx = (
        model.stream_function(queries + [step, 0])
        - model.stream_function(queries - [step, 0])
    ) / (2 * step)
    d_dy = (
        model.stream_function(queries + [0, step])
        - model.stream_function(queries - [0, step])
    ) / (2 * step)
    np.testing.assert_allclose(
        model(queries), np.stack([d_dy, -d_dx], axis=1), atol=1e-6
    )
    # The wall is a streamline: the stream function vanishes along it
    assert np.abs(model.stream_function(on_wall)).max() < 1e-12


def test_cg_matches_direct():
    direct, _, _, _ = _fit("sphere", "no_penetration", solver="direct")
    cg, _, _, _ = _fit("sphere", "no_penetration", solver="cg", tol=1e-12)
    queries = np.random.default_rng(5).uniform(-1, 1, (200, 3))
    np.testing.assert_allclose(cg(queries), direct(queries), atol=1e-7)
    assert cg.solve_info.solver == "cg"


def test_multilevel_with_walls(each_backend):
    walls = [Circle((0.1, 0.0), 0.3, band=0.3)]
    model = dfi.MultilevelDivergenceFreeInterpolant(walls=walls)
    model, points, values, (on_wall, normals) = _fit(
        "circle", "no_penetration", model=model, n=1500
    )
    assert len(model.levels) > 1
    np.testing.assert_allclose(model(points), values, atol=1e-7)
    u = model(on_wall)
    assert np.abs(np.sum(u * normals, axis=1)).max() < 1e-10
    for levels in range(1, len(model.levels) + 1):
        assert (
            np.abs(np.sum(model(on_wall, levels=levels) * normals, axis=1)).max()
            < 1e-10
        )
    assert np.abs(model.stream_function(on_wall)).max() < 1e-12


def test_walled_interpolants_can_be_pickled():
    model, points, _, _ = _fit("circle", "no_slip")
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored(points), model(points))
    assert restored.walls == model.walls


# --- data on and behind walls ----------------------------------------------------------------


def test_data_on_a_wall_keeps_its_free_components(each_backend):
    circle = Circle((0.0, 0.0), 0.3, band=0.4)
    rng = np.random.default_rng(6)
    points = _flow_points([circle], 2, 100, rng)
    on_wall, normals = _circle_points((0.0, 0.0), 0.3, 10, rng)
    tangents = np.stack([-normals[:, 1], normals[:, 0]], axis=1)
    # Data on the wall with a normal component, which the wall overrides
    wall_values = 0.7 * tangents + 0.2 * normals
    model = dfi.DivergenceFreeInterpolant(0.8, walls=circle, solver="direct").fit(
        np.vstack([points, on_wall]), np.vstack([div_free_field(points), wall_values])
    )
    np.testing.assert_allclose(model(points), div_free_field(points), atol=1e-8)
    fitted = model(on_wall)
    np.testing.assert_allclose(np.sum(fitted * tangents, axis=1), 0.7, atol=1e-8)
    assert np.abs(np.sum(fitted * normals, axis=1)).max() < 1e-12

    # No slip leaves nothing to fit on the wall
    no_slip = Circle((0.0, 0.0), 0.3, band=0.4, condition="no_slip")
    model = dfi.DivergenceFreeInterpolant(0.8, walls=no_slip).fit(
        np.vstack([points, on_wall]), np.vstack([div_free_field(points), wall_values])
    )
    np.testing.assert_allclose(model(points), div_free_field(points), atol=1e-8)
    assert np.abs(model(on_wall)).max() < 1e-12
    with pytest.raises(ValueError, match="nothing is left"):
        dfi.DivergenceFreeInterpolant(0.8, walls=no_slip).fit(on_wall, wall_values)


def test_restrict_functionals_at_a_corner():
    """A data point on two no-penetration walls at once keeps only the shared tangent."""
    walls = (
        Plane((0, 0, 0), (1, 0, 0), band=0.5),
        Plane((0, 0, 0), (0, 1, 0), band=0.5),
    )
    atoms = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 0.3], [0.0, 0.4, 0.1]])
    targets = np.arange(9.0)
    atoms, weights, targets, identity_points = restrict_functionals(
        walls, atoms, None, targets, 3
    )
    assert identity_points == 1
    np.testing.assert_array_equal(atoms[0], [0.5, 0.5, 0.0])
    # 3 point values, then the z component at the corner, then y and z on the x = 0 wall
    assert weights.shape == (9, 3 + 1 + 2)
    np.testing.assert_allclose(np.abs(targets[3]), 5.0)
    np.testing.assert_allclose(np.sort(np.abs(targets[4:])), [7.0, 8.0])


def test_data_behind_a_wall_is_rejected():
    circle = Circle((0.0, 0.0), 0.3, band=0.4)
    points = np.array([[0.5, 0.5], [0.1, 0.0], [-0.6, 0.2]])
    with pytest.raises(ValueError, match="point 1 lies on the far side of Circle"):
        dfi.DivergenceFreeInterpolant(0.8, walls=circle).fit(points, points)
    with pytest.raises(ValueError, match="far side"):
        dfi.MultilevelDivergenceFreeInterpolant(walls=circle).fit(points, points)


def test_duplicates_on_a_wall_are_rejected():
    circle = Circle((0.0, 0.0), 0.3, band=0.4)
    points = np.array([[0.5, 0.5], [0.3, 0.0], [0.3, 0.0], [-0.6, 0.2]])
    with pytest.raises(ValueError, match="points 1 and 2 coincide"):
        dfi.DivergenceFreeInterpolant(0.8, walls=circle).fit(points, points)


def test_walls_with_boundary_conditions():
    circle = Circle((0.0, 0.0), 0.3, band=0.3)
    inflow = dfi.VectorValue(
        dfi.Curve(lambda t: np.stack([-np.ones_like(t), t], -1), (-1, 1)), (1.0, 0.0)
    )
    rng = np.random.default_rng(7)
    points = _flow_points([circle], 2, 150, rng)
    points = points[points[:, 0] > -0.9]
    values = div_free_field(points)
    model = dfi.DivergenceFreeInterpolant(0.6, walls=circle).fit(
        points, values, conditions=inflow
    )
    on_wall, normals = _circle_points((0.0, 0.0), 0.3, 100, rng)
    assert np.abs(np.sum(model(on_wall) * normals, axis=1)).max() < 1e-10
    np.testing.assert_allclose(model(points), values, atol=1e-8)
    t = np.linspace(-0.9, 0.9, 7)
    inlet = model(np.stack([-np.ones_like(t), t], -1))
    np.testing.assert_allclose(inlet, np.tile([1.0, 0.0], (7, 1)), atol=0.05)

    # A condition on the wall itself is redundant
    on_circle = dfi.NormalComponent(
        dfi.Curve(lambda t: 0.3 * np.stack([np.cos(t), np.sin(t)], -1), (0, 2 * np.pi))
    )
    with pytest.raises(ValueError, match="boundary condition lies on"):
        dfi.DivergenceFreeInterpolant(0.6, walls=circle).fit(
            points, values, conditions=on_circle
        )


def test_walls_beat_tier_1_near_the_wall():
    """The spike's cylinder problem: uniform potential flow past a circle of radius 0.3."""
    radius = 0.3

    def potential_flow(p):
        x, y = p[:, 0], p[:, 1]
        r2 = x**2 + y**2
        return np.stack(
            [1 - radius**2 * (x**2 - y**2) / r2**2, -2 * radius**2 * x * y / r2**2], -1
        )

    rng = np.random.default_rng(0)
    candidates = rng.uniform([-1.5, -1.0], [1.5, 1.0], (3000, 2))
    samples = candidates[np.linalg.norm(candidates, axis=1) > radius + 0.25][:150]
    probes = rng.uniform([-1.5, -1.0], [1.5, 1.0], (20000, 2))
    distance = np.linalg.norm(probes, axis=1)
    near = probes[(distance > radius) & (distance < radius + 0.25)]

    def near_error(model):
        model.fit(samples, potential_flow(samples))
        error = np.linalg.norm(model(near) - potential_flow(near), axis=1)
        return np.sqrt(np.mean(error**2))

    wall = Circle((0, 0), radius, band=0.5)
    tier_2 = near_error(dfi.DivergenceFreeInterpolant(1.5, walls=wall))
    plain = near_error(dfi.DivergenceFreeInterpolant(1.5))
    assert tier_2 < 0.12
    assert tier_2 < 0.5 * plain


def test_walls_are_immutable():
    wall = Circle((0.0, 0.0), 0.3, band=0.4)
    with pytest.raises(AttributeError, match="immutable"):
        wall.band = 0.2
    with pytest.raises(ValueError, match="read-only"):
        wall.center[0] = 1.0
    assert pickle.loads(pickle.dumps(wall)) == wall


def test_core_validates_walls():
    kernel = get_kernel(DIVERGENCE_FREE, 2, None, 3)
    a, h, psi = kernel._a.tolist(), kernel._h.tolist(), kernel._psi.tolist()
    for spec in [
        ("half_space", [0.0, 0.0], [1.0], 0.0, False, 0.5, 1),
        ("half_space", [0.0, 0.0], [0.0, 0.0], 0.0, False, 0.5, 1),
        ("ball", [0.0, 0.0], [], -1.0, False, 0.5, 1),
        ("ball", [0.0, 0.0], [], 1.0, False, 0.0, 1),
        ("cylinder", [0.0, 0.0], [1.0, 0.0], 1.0, False, 0.5, 1),
        ("torus", [0.0, 0.0], [], 1.0, False, 0.5, 1),
    ]:
        with pytest.raises(ValueError):
            _core.Kernel(DIVERGENCE_FREE, 2, a, h, psi, [spec])
    with pytest.raises(ValueError, match="divergence-free"):
        _core.Kernel("curl_free", 2, a, h, psi, [])


def test_multilevel_skips_levels_with_nothing_to_fit(each_backend):
    """Coarse levels may hold only points on a no-slip wall; they contribute nothing."""
    wall = Line((0, 0), (0, 1), band=0.5, condition="no_slip")
    rng = np.random.default_rng(8)
    on_wall = np.column_stack([np.linspace(0, 10, 300), np.zeros(300)])
    near = np.column_stack([rng.uniform(4.9, 5.1, 30), rng.uniform(0.05, 0.1, 30)])
    points = np.vstack([on_wall, near])
    values = np.vstack([np.zeros_like(on_wall), np.tile([1.0, 0.0], (30, 1))])
    model = dfi.MultilevelDivergenceFreeInterpolant(walls=wall).fit(points, values)
    assert 1 <= len(model.levels) and model.levels[-1].n_points == len(points)
    np.testing.assert_allclose(model(near), values[300:], atol=1e-7)
    assert np.abs(model(on_wall)).max() < 1e-12


def test_multilevel_reports_the_data_index_of_points_behind_walls():
    container = Circle((0, 0), 1.0, band=0.4, inside=True)
    rng = np.random.default_rng(9)
    inner = rng.uniform(-0.6, 0.6, (40, 2))
    points = np.vstack([inner, [[1.5, 0.0]]])
    with pytest.raises(ValueError, match="point 40 lies on the far side"):
        dfi.MultilevelDivergenceFreeInterpolant(walls=container).fit(points, points)
