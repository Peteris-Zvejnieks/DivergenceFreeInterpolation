"""Walls that divergence-free interpolants satisfy exactly ("tier 2" boundary conditions).

Pass them as `walls=` to DivergenceFreeInterpolant or MultilevelDivergenceFreeInterpolant:

    flow = dfi.DivergenceFreeInterpolant(0.8, walls=[dfi.walls.Circle((0, 0), 0.3, band=0.5)])

Instead of adding conditions to the system (like NormalComponent), a wall is built into the
kernel: the vector potential is multiplied by a wall function g that vanishes on the walls, so
every field the interpolant can produce is tangent to them (u . n = 0) at every point of every
wall, and with condition='no_slip' vanishes there (u = 0). The field stays exactly
divergence-free, and the system keeps its size. See docs/tier2_boundary_conditions.md.

Each wall contributes the factor q(dist / band) to g, where dist is the signed distance to the
wall (positive on the flow side) and q(t) = 1 - (1 - t)^3 below 1, 1 beyond. Only within `band`
of a wall does the kernel change; the band sets how far the wall's influence reaches into the
flow (a few data spacings up to the size of the obstacle is a reasonable range). Several walls
multiply. The interpolant has no meaning on the far side of a wall (inside an obstacle).

Data points on a wall (closer than ON_WALL_TOLERANCE * band) keep only the components the wall
leaves free: the tangential ones for no penetration, none for no slip. Data points on the far
side of a wall are an error.
"""

import numpy as np
from scipy import sparse

NO_PENETRATION = "no_penetration"
NO_SLIP = "no_slip"
CONDITIONS = (NO_PENETRATION, NO_SLIP)

# Data points closer to a wall than this fraction of its band count as lying on it
ON_WALL_TOLERANCE = 1e-3


def _vector(value, dim, name):
    vector = np.array(value, dtype=float)
    if vector.shape != (dim,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be {dim} finite numbers, got {value!r}")
    # Walls are immutable: they key the kernel cache and are built into fitted kernels
    vector.flags.writeable = False
    return vector


def _positive(value, name):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    return number


def _unit(value, dim, name):
    vector = _vector(value, dim, name)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError(f"{name} must not be zero")
    unit = vector / norm
    unit.flags.writeable = False
    return unit


def _radial_distance(offset, radius, inside):
    """Signed distance and its gradient from offsets to a centre (ball) or axis (cylinder)."""
    rho = np.linalg.norm(offset, axis=-1)
    sign = -1.0 if inside else 1.0
    gradient = sign * offset / np.where(rho > 0, rho, 1.0)[..., np.newaxis]
    return sign * (rho - radius), gradient


class NothingToFit(ValueError):
    """Every data component is fixed by the walls (all points lie on no-slip walls)."""


class _Immutable(type):
    """Freezes instances once their constructor has run."""

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        object.__setattr__(instance, "_frozen", True)
        return instance


class Wall(metaclass=_Immutable):
    """Base class of the walls: a shape with a band and a condition. Walls are immutable."""

    dim: int
    _shape: str

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError(
                f"walls are immutable; create a new {type(self).__name__} instead"
            )
        super().__setattr__(name, value)

    def __init__(self, band, condition):
        self.band = _positive(band, "band")
        if condition not in CONDITIONS:
            raise ValueError(
                f"unknown wall condition {condition!r}, expected one of {CONDITIONS}"
            )
        self.condition = condition

    @property
    def power(self) -> int:
        """Power of the wall's factor of g: 2 makes the whole field vanish on the wall."""
        return 2 if self.condition == NO_SLIP else 1

    def signed_distance(self, points):
        """Signed distance to the wall (positive on the flow side) at points (..., dim), and its
        gradient (..., dim)."""
        raise NotImplementedError

    def _parameters(self):
        raise NotImplementedError

    def _core_spec(self):
        """The wall as the Rust core takes it: (shape, point, vector, radius, inside, band,
        power)."""
        raise NotImplementedError

    def _key(self):
        return (type(self), *self._parameters().values(), self.band, self.condition)

    def __eq__(self, other):
        return isinstance(other, Wall) and self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    def __repr__(self):
        def show(value):
            if isinstance(value, tuple):
                return "(" + ", ".join(f"{v:g}" for v in value) + ")"
            return repr(value)

        fields = {**self._parameters(), "band": self.band, "condition": self.condition}
        listed = ", ".join(f"{name}={show(value)}" for name, value in fields.items())
        return f"{type(self).__name__}({listed})"


class _Ball(Wall):
    _shape = "ball"

    def __init__(self, center, radius, *, band, condition=NO_PENETRATION, inside=False):
        super().__init__(band, condition)
        self.center = _vector(center, self.dim, "center")
        self.radius = _positive(radius, "radius")
        self.inside = bool(inside)
        if self.inside and self.band >= self.radius:
            # The distance to the wall has a kink at the centre, which must stay outside the band
            raise ValueError(
                f"with the flow inside, the band ({self.band:g}) must be smaller than the "
                f"radius ({self.radius:g})"
            )

    def signed_distance(self, points):
        return _radial_distance(points - self.center, self.radius, self.inside)

    def _parameters(self):
        return {
            "center": tuple(self.center.tolist()),
            "radius": self.radius,
            "inside": self.inside,
        }

    def _core_spec(self):
        return (
            "ball",
            self.center.tolist(),
            [],
            self.radius,
            self.inside,
            self.band,
            self.power,
        )


class Circle(_Ball):
    """A circular wall in 2D.

    Parameters
    ----------
    center : (2,) array_like
    radius : float
    band : float
        Width of the layer along the wall in which the kernel is modified.
    condition : {'no_penetration', 'no_slip'}, default 'no_penetration'
        'no_penetration': the field is tangent to the wall (u . n = 0); 'no_slip': it vanishes
        on the wall (u = 0).
    inside : bool, default False
        False for flow around a circular obstacle, True for flow inside a circular container
        (the band must then be smaller than the radius).
    """

    dim = 2


class Sphere(_Ball):
    """A spherical wall in 3D. Parameters as for Circle."""

    dim = 3


class _HalfSpace(Wall):
    _shape = "half_space"

    def __init__(self, point, normal, *, band, condition=NO_PENETRATION):
        super().__init__(band, condition)
        self.point = _vector(point, self.dim, "point")
        self.normal = _unit(normal, self.dim, "normal")

    def signed_distance(self, points):
        distance = (points - self.point) @ self.normal
        return distance, np.broadcast_to(self.normal, np.shape(points)).copy()

    def _parameters(self):
        return {
            "point": tuple(self.point.tolist()),
            "normal": tuple(self.normal.tolist()),
        }

    def _core_spec(self):
        return (
            "half_space",
            self.point.tolist(),
            self.normal.tolist(),
            0.0,
            False,
            self.band,
            self.power,
        )


class Line(_HalfSpace):
    """A straight wall in 2D, through `point`, with the flow on the side `normal` points to.

    Parameters
    ----------
    point, normal : (2,) array_like
        A point on the wall and a (not necessarily unit) normal pointing into the flow.
    band, condition
        As for Circle.
    """

    dim = 2


class Plane(_HalfSpace):
    """A plane wall in 3D, through `point`, with the flow on the side `normal` points to.
    Parameters as for Line."""

    dim = 3


class Cylinder(Wall):
    """An infinite circular cylinder in 3D, around the line through `point` along `axis`.

    Parameters
    ----------
    point, axis : (3,) array_like
        A point on the axis and its (not necessarily unit) direction.
    radius : float
    band, condition, inside
        As for Circle: inside=True is flow in a pipe.
    """

    dim = 3
    _shape = "cylinder"

    def __init__(
        self, point, axis, radius, *, band, condition=NO_PENETRATION, inside=False
    ):
        super().__init__(band, condition)
        self.point = _vector(point, 3, "point")
        self.axis = _unit(axis, 3, "axis")
        self.radius = _positive(radius, "radius")
        self.inside = bool(inside)
        if self.inside and self.band >= self.radius:
            raise ValueError(
                f"with the flow inside, the band ({self.band:g}) must be smaller than the "
                f"radius ({self.radius:g})"
            )

    def signed_distance(self, points):
        offset = points - self.point
        offset = offset - (offset @ self.axis)[..., np.newaxis] * self.axis
        return _radial_distance(offset, self.radius, self.inside)

    def _parameters(self):
        return {
            "point": tuple(self.point.tolist()),
            "axis": tuple(self.axis.tolist()),
            "radius": self.radius,
            "inside": self.inside,
        }

    def _core_spec(self):
        return (
            "cylinder",
            self.point.tolist(),
            self.axis.tolist(),
            self.radius,
            self.inside,
            self.band,
            self.power,
        )


def as_walls(walls):
    """Validate walls (None, a single one or an iterable) and return them as a tuple."""
    if walls is None:
        return ()
    if isinstance(walls, Wall):
        return (walls,)
    try:
        walls = tuple(walls)
    except TypeError:
        raise TypeError(
            f"walls must be a wall such as walls.Circle(...) or an iterable of them, "
            f"got {type(walls).__name__}"
        ) from None
    for wall in walls:
        if not isinstance(wall, Wall):
            raise TypeError(
                f"walls must be walls such as walls.Circle(...), got {type(wall).__name__}"
            )
    return walls


def wall_function(walls, points):
    """g and grad g at points (..., dim): the product of the walls' factors q(dist/band)^power."""
    points = np.asarray(points, dtype=float)
    g = np.ones(points.shape[:-1])
    gradient = np.zeros(points.shape)
    for wall in walls:
        distance, normal = wall.signed_distance(points)
        t = distance / wall.band
        near = t < 1
        q = np.where(near, 1 - (1 - t) ** 3, 1.0)
        slope = np.where(near, 3 * (1 - t) ** 2 / wall.band, 0.0)
        if wall.power != 1:
            slope = wall.power * q ** (wall.power - 1) * slope
            q = q**wall.power
        # Product rule: grad(g f) = f grad g + g grad f
        gradient = q[..., np.newaxis] * gradient + (g * slope)[..., np.newaxis] * normal
        g = g * q
    return g, gradient


def check_data(walls, points):
    """Raise if a data point lies on the far side of a wall (inside an obstacle)."""
    for wall in walls:
        relative = wall.signed_distance(points)[0] / wall.band
        behind = np.flatnonzero(relative < -ON_WALL_TOLERANCE)
        if len(behind):
            raise ValueError(
                f"point {behind[0]} lies on the far side of {wall!r}, outside the flow; "
                f"remove the points inside obstacles"
            )


def restrict_functionals(walls, atoms, weights, targets, n_data):
    """Drop the data components that walls fix, for fitting with a wall kernel.

    `atoms` (P, dim) start with the n_data data points, whose point values are the first
    n_data * dim functionals; `weights` (P dim, F) maps coefficients to functionals (None: the
    identity, P = n_data). Data points on a wall keep only the components it leaves free, and
    move behind the other data points. Returns (atoms, weights, targets, identity_points), with
    the identity_points first atoms still carrying plain point values. Raises NothingToFit if
    no functional is left.
    """
    if not walls:
        return atoms, weights, targets, n_data
    dim = atoms.shape[1]
    check_data(walls, atoms[:n_data])
    relative = np.stack([wall.signed_distance(atoms)[0] / wall.band for wall in walls])
    if (relative[:, n_data:] <= ON_WALL_TOLERANCE).any():
        raise ValueError(
            "a boundary condition lies on (or behind) a wall, which already imposes the wall "
            "condition there exactly; drop the condition or the wall"
        )
    touching = np.abs(relative[:, :n_data]) <= ON_WALL_TOLERANCE
    on_wall = np.flatnonzero(touching.any(axis=0))
    if not len(on_wall):
        return atoms, weights, targets, n_data
    off_wall = np.flatnonzero(~touching.any(axis=0))

    # Per data point on a wall: an orthonormal basis of the components no wall fixes there
    bases = []
    for i in on_wall:
        touched = [wall for wall, t in zip(walls, touching[:, i], strict=True) if t]
        if any(wall.condition == NO_SLIP for wall in touched):
            bases.append(np.zeros((dim, 0)))
            continue
        normals = np.array([wall.signed_distance(atoms[i])[1] for wall in touched])
        _, singular, vt = np.linalg.svd(normals)
        rank = int(np.sum(singular > 1e-10 * singular.max()))
        bases.append(vt[rank:].T)
    free = np.array([basis.shape[1] for basis in bases])
    if not len(off_wall) and not free.sum() and len(targets) == n_data * dim:
        raise NothingToFit(
            "all data points lie on no-slip walls, where the field is fixed to zero: "
            "nothing is left to fit"
        )

    data_targets = targets[: n_data * dim].reshape(n_data, dim)
    columns = np.concatenate([[0], np.cumsum(free)])
    tangent = np.zeros((len(on_wall) * dim, columns[-1]))
    tangent_targets = np.zeros(columns[-1])
    for j, (i, basis) in enumerate(zip(on_wall, bases, strict=True)):
        tangent[j * dim : (j + 1) * dim, columns[j] : columns[j + 1]] = basis
        tangent_targets[columns[j] : columns[j + 1]] = basis.T @ data_targets[i]
    blocks = [sparse.identity(len(off_wall) * dim), sparse.csr_array(tangent)]
    if weights is not None:
        # Condition functionals only weight the boundary atoms after the data points
        blocks.append(sparse.csr_array(weights)[n_data * dim :, n_data * dim :])
    order = np.concatenate([off_wall, on_wall, np.arange(n_data, len(atoms))])
    return (
        atoms[order],
        sparse.block_diag(blocks, format="csr"),
        np.concatenate(
            [
                data_targets[off_wall].ravel(),
                tangent_targets,
                targets[n_data * dim :],
            ]
        ),
        len(off_wall),
    )
