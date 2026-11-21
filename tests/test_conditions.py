"""Boundary conditions on parametric curves and surfaces."""

import numpy as np
import pytest
import sympy as sp

import divergence_free_interpolant as dfi
from divergence_free_interpolant._conditions import assemble

from conftest import CURL_FREE, DIVERGENCE_FREE, INTERPOLANTS, KINDS

RADIUS = 0.2
MULTILEVEL = {
    DIVERGENCE_FREE: dfi.MultilevelDivergenceFreeInterpolant,
    CURL_FREE: dfi.MultilevelCurlFreeInterpolant,
}


def circle(radius=RADIUS):
    return dfi.Curve(
        lambda t: np.stack([radius * np.cos(t), radius * np.sin(t)], -1),
        (0, 2 * np.pi),
    )


def sphere(radius=RADIUS):
    def surface(s, t):
        return radius * np.stack(
            [np.sin(s) * np.cos(t), np.sin(s) * np.sin(t), np.cos(s)], -1
        )

    return dfi.Surface(surface, (0, np.pi), (0, 2 * np.pi))


def potential_flow(points):
    """Uniform flow past a cylinder (2D) or sphere (3D) of radius RADIUS at the origin.

    Both divergence-free and curl-free, and tangent to the obstacle's surface.
    """
    points = np.asarray(points, dtype=float)
    r2 = np.sum(points**2, axis=1)
    if points.shape[1] == 2:
        x, y = points.T
        u = 1 - RADIUS**2 * (x**2 - y**2) / r2**2
        v = -2 * RADIUS**2 * x * y / r2**2
        return np.stack([u, v], 1)
    r = np.sqrt(r2)
    flow = np.zeros_like(points)
    flow[:, 0] = 1 + RADIUS**3 / (2 * r**3)
    return flow - 1.5 * RADIUS**3 * points[:, :1] * points / r[:, None] ** 5


def samples_outside(dim, n, seed=0, gap=0.05):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1, 1, size=(8 * n, dim))
    return points[np.linalg.norm(points, axis=1) > RADIUS + gap][:n]


def wall_points(dim, n=1500):
    rng = np.random.default_rng(42)
    directions = rng.normal(size=(n, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return RADIUS * directions, directions


# --- geometry --------------------------------------------------------------------------------


def test_curve_discretization():
    discretization = circle().discretize(0.05)
    segments, _, dim = discretization.points.shape
    assert segments == np.ceil(2 * np.pi * RADIUS / 0.05) and dim == 2
    np.testing.assert_allclose(np.linalg.norm(discretization.points, axis=-1), RADIUS)
    np.testing.assert_allclose(discretization.weights.sum(axis=1), 1)
    # Counter-clockwise circle: the normal to the right of travel points outwards
    np.testing.assert_allclose(
        discretization.normals, discretization.points / RADIUS, atol=1e-8
    )


def test_curve_from_sympy_has_exact_derivatives():
    t = sp.symbols("t")
    symbolic = dfi.Curve([sp.cos(t), 2 * sp.sin(t)], (0, 3))
    numeric = dfi.Curve(lambda s: np.stack([np.cos(s), 2 * np.sin(s)], -1), (0, 3))
    samples = np.linspace(0, 3, 7)
    np.testing.assert_allclose(symbolic(samples), numeric(samples))
    np.testing.assert_allclose(
        symbolic.derivative(samples),
        np.stack([-np.sin(samples), 2 * np.cos(samples)], -1),
    )
    np.testing.assert_allclose(
        numeric.derivative(samples), symbolic.derivative(samples), atol=1e-6
    )
    line = dfi.Curve([t, 2 * t], (0, 1))
    np.testing.assert_allclose(line.derivative(np.array([0.3, 0.7])), [[1, 2], [1, 2]])


def test_curves_in_3d_have_no_normal():
    helix = dfi.Curve(lambda t: np.stack([np.cos(t), np.sin(t), t / 5], -1), (0, 6))
    assert helix.discretize(0.2).normals is None


def test_sphere_discretization_adapts_near_the_poles():
    discretization = sphere(1.0).discretize(0.2)
    points, normals = discretization.points, discretization.normals
    np.testing.assert_allclose(np.linalg.norm(points, axis=-1), 1.0)
    np.testing.assert_allclose(
        np.abs(np.sum(normals * points, axis=-1)), 1.0, atol=1e-6
    )
    np.testing.assert_allclose(discretization.weights.sum(axis=1), 1)
    # About area / segment_length^2 patches, not piled up at the poles
    expected = 4 * np.pi / 0.2**2
    assert 0.7 * expected < len(points) < 1.6 * expected
    near_pole = np.abs(points[:, :, 2]).mean(axis=1) > 0.98
    assert near_pole.sum() < 10


def test_surface_from_sympy():
    s, t = sp.symbols("s t")
    surface = dfi.Surface(
        [sp.sin(s) * sp.cos(t), sp.sin(s) * sp.sin(t), sp.cos(s)],
        (0, sp.pi),
        (0, 2 * sp.pi),
    )
    d_s, d_t = surface.partial_derivatives(np.array([1.0]), np.array([0.5]))
    np.testing.assert_allclose(
        d_s[0], [np.cos(1) * np.cos(0.5), np.cos(1) * np.sin(0.5), -np.sin(1)]
    )
    np.testing.assert_allclose(
        d_t[0], [-np.sin(1) * np.sin(0.5), np.sin(1) * np.cos(0.5), 0]
    )


def test_default_segmentation_resolves_small_boundaries():
    small = dfi.NormalComponent(circle(0.05))
    assert small.default_segment_length(1.0) == pytest.approx(
        2 * np.pi * 0.05 / 32, rel=1e-3
    )
    assert small.default_segment_length(0.001) == 0.001
    assert sphere(1.0).area() == pytest.approx(4 * np.pi, rel=1e-6)
    patches = dfi.NormalComponent(sphere(1.0)).default_segment_length(10.0)
    assert patches == pytest.approx(np.sqrt(4 * np.pi / 128), rel=1e-6)
    explicit = dfi.NormalComponent(circle(), segment_length=0.3)
    assert explicit.default_segment_length(0.01) == 0.3


def test_geometry_validation():
    with pytest.raises(ValueError, match="interval"):
        dfi.Curve(lambda t: np.stack([t, t], -1), (1, 0))
    with pytest.raises(ValueError, match="3D"):
        dfi.Surface(lambda s, t: np.stack([s, t], -1), (0, 1), (0, 1))
    t, u = sp.symbols("t u")
    with pytest.raises(ValueError, match="free symbol"):
        dfi.Curve([t, u], (0, 1))


# --- conditions ------------------------------------------------------------------------------


def test_normal_component_needs_a_codimension_one_boundary():
    helix = dfi.Curve(lambda t: np.stack([np.cos(t), np.sin(t), t / 5], -1), (0, 6))
    points = np.random.default_rng(0).random((20, 3))
    with pytest.raises(ValueError, match="normal component"):
        dfi.DivergenceFreeInterpolant(0.5).fit(
            points, points, conditions=dfi.NormalComponent(helix)
        )


def test_condition_validation():
    points = np.random.default_rng(0).random((20, 2))
    with pytest.raises(TypeError, match="Curve or Surface"):
        dfi.NormalComponent("wall")
    with pytest.raises(ValueError, match="segment_length"):
        dfi.NormalComponent(circle(), segment_length=0)
    with pytest.raises(TypeError, match="NormalComponent or VectorValue"):
        dfi.DivergenceFreeInterpolant(0.5).fit(points, points, conditions=["wall"])
    with pytest.raises(ValueError, match="3D boundary"):
        dfi.DivergenceFreeInterpolant(0.5).fit(
            points, points, conditions=dfi.NormalComponent(sphere())
        )
    with pytest.raises(ValueError, match="broadcast"):
        dfi.DivergenceFreeInterpolant(0.5).fit(
            points, points, conditions=dfi.VectorValue(circle(), value=[1, 2, 3])
        )


@pytest.mark.parametrize(
    "kind, condition, value",
    [
        # A divergence-free field has zero net flux through the closed circle
        (DIVERGENCE_FREE, dfi.NormalComponent, 0.0),
        (CURL_FREE, dfi.NormalComponent, 0.3),
        # A constant vector has zero net flux through any closed curve
        (DIVERGENCE_FREE, dfi.VectorValue, (0.3, -0.1)),
        (CURL_FREE, dfi.VectorValue, (0.3, -0.1)),
    ],
)
def test_segment_means_are_satisfied_exactly(kind, condition, value):
    points = samples_outside(2, 150)
    values = potential_flow(points)
    imposed = condition(circle(), value=value, segment_length=0.04)
    interp = INTERPOLANTS[kind](0.6).fit(points, values, conditions=imposed)
    # All segment means. For a divergence-free field the last segment's flux is dropped as
    # dependent: it then holds only up to the quadrature error of the net flux
    atoms, weights, targets = assemble(points, values, [imposed], [0.04])
    applied = weights.T @ interp(atoms).ravel()
    dropped = kind == DIVERGENCE_FREE and condition is dfi.NormalComponent
    imposed_count = len(targets) - 1 if dropped else len(targets)
    np.testing.assert_allclose(
        applied[:imposed_count], targets[:imposed_count], atol=1e-8
    )
    np.testing.assert_allclose(applied, targets, atol=1e-5)
    # ... and the field itself on the wall, between the segment centres too
    wall, normal = wall_points(2)
    on_wall = interp(wall)
    if condition is dfi.NormalComponent:
        np.testing.assert_allclose(np.sum(on_wall * normal, axis=1), value, atol=0.02)
    else:
        np.testing.assert_allclose(
            on_wall, np.broadcast_to(value, on_wall.shape), atol=0.05
        )
    assert np.abs(interp._fit.coefficients).max() < 1e6


def test_closed_boundaries_are_detected():
    t = sp.symbols("t")
    assert circle().closed
    assert not dfi.Curve(
        lambda s: np.stack([np.cos(s), np.sin(s)], -1), (0, np.pi)
    ).closed
    assert dfi.Curve([sp.cos(t), sp.sin(t)], (0, 2 * sp.pi)).closed
    assert not dfi.Curve(lambda s: np.stack([s, s**2], -1), (0, 1)).closed
    assert dfi.Curve(lambda s: np.stack([s, s**2], -1), (0, 1), closed=True).closed
    assert sphere().closed

    def hemisphere(s, u):
        return np.stack([np.sin(s) * np.cos(u), np.sin(s) * np.sin(u), np.cos(s)], -1)

    def torus(s, u):
        ring = 1 + 0.3 * np.cos(s)
        return np.stack([ring * np.cos(u), ring * np.sin(u), 0.3 * np.sin(s)], -1)

    def tube(s, u):
        return np.stack([np.cos(u), np.sin(u), s], -1)

    assert not dfi.Surface(hemisphere, (0, np.pi / 2), (0, 2 * np.pi)).closed
    assert dfi.Surface(torus, (0, 2 * np.pi), (0, 2 * np.pi)).closed
    assert not dfi.Surface(tube, (0, 1), (0, 2 * np.pi)).closed


def test_impossible_net_flux_through_a_closed_boundary_raises():
    points = samples_outside(2, 100)
    values = potential_flow(points)
    with pytest.raises(ValueError, match="net flux"):
        dfi.DivergenceFreeInterpolant(0.6).fit(
            points, values, conditions=dfi.NormalComponent(circle(), value=0.3)
        )
    # A radial wall velocity also has a net outflow
    with pytest.raises(ValueError, match="net flux"):
        dfi.DivergenceFreeInterpolant(0.6).fit(
            points, values, conditions=dfi.VectorValue(circle(), value=lambda p: p)
        )
    # Both are fine for curl-free fields, and for open boundaries
    dfi.CurlFreeInterpolant(0.6).fit(
        points, values, conditions=dfi.NormalComponent(circle(), value=0.3)
    )
    half = dfi.Curve(
        lambda t: np.stack([RADIUS * np.cos(t), RADIUS * np.sin(t)], -1), (0, np.pi)
    )
    dfi.DivergenceFreeInterpolant(0.6).fit(
        points, values, conditions=dfi.NormalComponent(half, value=0.3)
    )


@pytest.mark.parametrize("geometry", [circle, sphere])
def test_closed_no_penetration_keeps_the_system_well_conditioned(geometry):
    """The dependent segment flux is dropped (divergence theorem): no near-singular mode."""
    dim = 2 if geometry is circle else 3
    points = samples_outside(dim, 150 if dim == 2 else 300)
    values = potential_flow(points)
    interp = dfi.DivergenceFreeInterpolant(0.6)
    condition = dfi.NormalComponent(geometry(), segment_length=0.06)
    smallest = {}
    for divergence_free in (False, True):
        atoms, weights, _ = assemble(
            points, values, [condition], [0.06], divergence_free
        )
        system = (weights.T @ interp._gram_matrix(atoms) @ weights).toarray()
        smallest[divergence_free] = np.linalg.eigvalsh(system).min()
    # Measured improvement: about 500x for the circle and 70x for the sphere
    assert smallest[True] > 10 * smallest[False]


def test_non_finite_values_raise():
    points = samples_outside(2, 50)
    with pytest.raises(ValueError, match="finite"):
        dfi.DivergenceFreeInterpolant(0.6).fit(
            points,
            potential_flow(points),
            conditions=dfi.VectorValue(circle(), value=np.nan),
        )


def test_degenerate_boundaries_raise():
    with pytest.raises(ValueError, match="length"):
        dfi.Curve(lambda t: np.zeros((len(t), 2)) + 0.1)
    with pytest.raises(ValueError, match="area"):
        dfi.Surface(lambda s, t: np.stack([s, s, s], -1), (0, 1), (0, 1))


def test_surface_patches_follow_arc_length():
    """A sphere parametrised by height z has fast-moving poles; patches stay evenly sized."""

    def by_height(z, t):
        ring = np.sqrt(np.clip(1 - z**2, 0, None))
        return np.stack([ring * np.cos(t), ring * np.sin(t), z], -1)

    for surface in (dfi.Surface(by_height, (-1, 1), (0, 2 * np.pi)), sphere(1.0)):
        points = surface.discretize(0.2).points
        spread = np.linalg.norm(points - points.mean(axis=1, keepdims=True), axis=-1)
        assert spread.max() < 0.2


def test_conditions_argument_validation():
    points = samples_outside(2, 30)
    values = potential_flow(points)
    interp = dfi.DivergenceFreeInterpolant(0.6).fit(points, values, conditions=None)
    assert interp._fit.nodes.shape == points.shape
    with pytest.raises(TypeError, match="bare boundary"):
        dfi.DivergenceFreeInterpolant(0.6).fit(points, values, conditions=circle())
    with pytest.raises(TypeError, match="iterable"):
        dfi.DivergenceFreeInterpolant(0.6).fit(points, values, conditions=3)


def test_multilevel_rejects_a_normal_on_a_3d_curve_before_building_levels(monkeypatch):
    from divergence_free_interpolant import _multilevel

    def unreachable(*args):
        raise AssertionError("the level hierarchy should not be built")

    monkeypatch.setattr(_multilevel, "level_hierarchy", unreachable)
    helix = dfi.Curve(lambda t: np.stack([np.cos(t), np.sin(t), t / 5], -1), (0, 6))
    points = np.random.default_rng(0).random((100, 3))
    with pytest.raises(ValueError, match="normal component"):
        dfi.MultilevelDivergenceFreeInterpolant().fit(
            points, points, conditions=dfi.NormalComponent(helix)
        )


def test_cg_handles_short_segments_on_a_small_obstacle():
    """Short segments are strongly coupled; their block preconditioner keeps CG converging."""
    points = samples_outside(2, 300)
    small = dfi.Curve(
        lambda t: np.stack([0.05 * np.cos(t), 0.05 * np.sin(t)], -1), (0, 2 * np.pi)
    )
    for condition in (dfi.NormalComponent(small), dfi.VectorValue(small)):
        interp = dfi.DivergenceFreeInterpolant(0.6, solver="cg").fit(
            points, potential_flow(points), conditions=condition
        )
        assert interp.solve_info.iterations < 3000


@pytest.mark.parametrize("kind", KINDS)
def test_no_penetration_holds_along_the_whole_wall(kind):
    points = samples_outside(2, 300)
    values = potential_flow(points)
    wall, normal = wall_points(2)
    free = INTERPOLANTS[kind](0.6).fit(points, values)
    walled = INTERPOLANTS[kind](0.6).fit(
        points, values, conditions=dfi.NormalComponent(circle())
    )
    free_flux = np.abs(np.sum(free(wall) * normal, axis=1)).max()
    walled_flux = np.abs(np.sum(walled(wall) * normal, axis=1)).max()
    assert walled_flux < 0.02 < free_flux
    np.testing.assert_allclose(walled(points), values, atol=1e-8)


def test_no_slip_stops_the_flow_along_the_wall():
    points = samples_outside(2, 300)
    wall, _ = wall_points(2)
    interp = dfi.DivergenceFreeInterpolant(0.6).fit(
        points, potential_flow(points), conditions=dfi.VectorValue(circle())
    )
    assert np.abs(interp(wall)).max() < 0.05


def test_prescribed_inflow_profile():
    """A non-zero, position-dependent normal component (e.g. an inlet)."""
    rng = np.random.default_rng(1)
    points = rng.uniform(0.1, 1, size=(200, 2))
    values = np.column_stack([np.ones(200), np.zeros(200)])
    inlet = dfi.Curve(lambda t: np.stack([np.zeros_like(t), t], -1), (0, 1))

    # The inlet runs upwards along x = 0, so its normal (to the right of travel) is +x
    def profile(where):
        return 4 * where[:, 1] * (1 - where[:, 1])

    interp = dfi.DivergenceFreeInterpolant(0.5).fit(
        points, values, conditions=dfi.NormalComponent(inlet, value=profile)
    )
    heights = np.linspace(0.05, 0.95, 50)
    on_inlet = np.column_stack([np.zeros(50), heights])
    np.testing.assert_allclose(
        interp(on_inlet)[:, 0], 4 * heights * (1 - heights), atol=0.05
    )


@pytest.mark.parametrize("kind", KINDS)
def test_no_penetration_on_a_sphere(kind):
    points = samples_outside(3, 800, gap=0.05)
    values = potential_flow(points)
    wall, normal = wall_points(3)
    free = INTERPOLANTS[kind](0.8).fit(points, values)
    walled = INTERPOLANTS[kind](0.8).fit(
        points, values, conditions=dfi.NormalComponent(sphere())
    )
    free_flux = np.abs(np.sum(free(wall) * normal, axis=1)).max()
    walled_flux = np.abs(np.sum(walled(wall) * normal, axis=1)).max()
    assert walled_flux < 0.1 * free_flux


def test_conditions_keep_the_field_divergence_free(rng):
    from conftest import jacobian

    points = samples_outside(2, 200)
    interp = dfi.DivergenceFreeInterpolant(0.6).fit(
        points, potential_flow(points), conditions=dfi.VectorValue(circle())
    )
    probe = rng.uniform(-0.6, 0.6, size=(30, 2))
    probe = probe[np.linalg.norm(probe, axis=1) > RADIUS + 0.05]
    divergence = np.trace(jacobian(interp, probe), axis1=1, axis2=2)
    assert np.abs(divergence).max() < 1e-5


def test_cg_with_conditions_matches_direct():
    points = samples_outside(2, 300)
    values = potential_flow(points)
    condition = dfi.NormalComponent(circle())
    direct = dfi.DivergenceFreeInterpolant(0.4, solver="direct").fit(
        points, values, conditions=condition
    )
    cg = dfi.DivergenceFreeInterpolant(0.4, solver="cg", tol=1e-12).fit(
        points, values, conditions=condition
    )
    assert cg.solve_info.solver == "cg"
    probe = samples_outside(2, 100, seed=3)
    np.testing.assert_allclose(cg(probe), direct(probe), atol=1e-8)


def test_duplicate_data_points_still_raise():
    points = samples_outside(2, 50)
    with pytest.raises(ValueError, match="coincide"):
        dfi.DivergenceFreeInterpolant(0.5).fit(
            np.vstack([points, points[:1]]),
            potential_flow(np.vstack([points, points[:1]])),
            conditions=dfi.NormalComponent(circle()),
        )


# --- multilevel ------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_multilevel_no_penetration(kind):
    points = samples_outside(2, 3000, gap=0.02)
    values = potential_flow(points)
    wall, normal = wall_points(2)
    free = MULTILEVEL[kind]().fit(points, values)
    walled = MULTILEVEL[kind]().fit(
        points, values, conditions=[dfi.NormalComponent(circle())]
    )
    assert len(walled.levels) >= 3
    assert walled.levels[-1].n_points == len(points)
    walled_flux = np.abs(np.sum(walled(wall) * normal, axis=1)).max()
    free_flux = np.abs(np.sum(free(wall) * normal, axis=1)).max()
    assert walled_flux < 0.05 * free_flux
    np.testing.assert_allclose(walled(points), values, atol=1e-8)


def test_multilevel_conditions_hold_at_the_finest_segmentation():
    points = samples_outside(2, 2000, gap=0.02)
    values = potential_flow(points)
    condition = dfi.VectorValue(circle(), segment_length=0.01)
    interp = dfi.MultilevelDivergenceFreeInterpolant().fit(
        points, values, conditions=condition
    )
    atoms, weights, targets = assemble(points, values, [condition], [0.01])
    applied = weights.T @ interp(atoms).ravel()
    np.testing.assert_allclose(applied, targets, atol=1e-8)
