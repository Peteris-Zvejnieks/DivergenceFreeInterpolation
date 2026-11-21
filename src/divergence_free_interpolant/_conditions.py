"""Boundary conditions on parametric curves and surfaces, imposed in the mean over segments.

A condition such as "the normal component of the field vanishes on this curve" cannot be imposed
at every point of the curve with finitely many unknowns. Instead the boundary is cut into
segments (patches, for surfaces) of about one point spacing, and the condition holds for the
mean over each segment. Each mean is a weighted sum of point evaluations (Gauss-Legendre
quadrature), i.e. a linear functional the interpolant can be fitted to like a data value.

A divergence-free field has zero net flux through a closed boundary (divergence theorem), so on
a closed boundary the segment means of the normal component are linearly dependent: one of them
is dropped, and a prescribed non-zero net flux is rejected as impossible.
"""

import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse

# Gauss-Legendre points per segment (per direction on surface patches)
QUADRATURE_ORDER = 3
# Relative step of the central differences used when no derivative is given
DIFFERENCE_STEP = 1e-6
# Samples used to measure lengths for the segmentation
LENGTH_SAMPLES = 2001
# Default segmentations resolve every boundary with at least this many segments (curves) or
# patches (surfaces), so small obstacles are not cut into a handful of coarse pieces
MIN_CURVE_SEGMENTS = 32
MIN_SURFACE_PATCHES = 128
# Relative tolerance for detecting closed boundaries (coinciding ends / edges)
CLOSED_TOLERANCE = 1e-8
# Relative tolerance for the net flux through a closed boundary to count as zero
NET_FLUX_TOLERANCE = 1e-6


def _sympy():
    """sympy, imported only for boundaries given as sympy expressions (the "symbolic" extra)."""
    try:
        import sympy
    except ImportError as error:
        raise ImportError(
            "boundaries given as sympy expressions need sympy: "
            "pip install 'Divergence-Free-Interpolant[symbolic]'"
        ) from error
    return sympy


def _is_symbolic(function) -> bool:
    """Whether `function` is a sympy expression, or a sequence of them, rather than a callable.

    sympy expressions are callable themselves, so they are recognised by their type's module,
    without importing sympy.
    """
    if isinstance(function, (list, tuple)):
        return True
    return type(function).__module__.partition(".")[0] == "sympy"


def _lambdify(expressions, symbols):
    """Numpy function of the parameters returning (..., components) from sympy expressions."""
    lambdified = _sympy().lambdify(symbols, list(expressions), "numpy")

    def evaluate(*arguments):
        shape = np.broadcast(*arguments).shape
        # Constant components come back as scalars, so broadcast every component
        columns = [np.broadcast_to(c, shape) for c in lambdified(*arguments)]
        return np.stack(columns, axis=-1).astype(float)

    return evaluate


def _as_function(function, parameters):
    """(function, exact derivatives or None) from a callable or a sequence of sympy expressions.

    The sympy symbols are taken in alphabetical order as the parameters.
    """
    if callable(function) and not _is_symbolic(function):
        return function, None
    expressions = _sympy().Matrix(list(function))
    symbols = sorted(expressions.free_symbols, key=lambda symbol: symbol.name)
    if len(symbols) != parameters:
        raise ValueError(
            f"a sympy boundary needs exactly {parameters} free symbol(s), got {symbols}"
        )
    derivatives = [_lambdify(expressions.diff(symbol), symbols) for symbol in symbols]
    return _lambdify(expressions, symbols), derivatives


def _central_difference(function, arguments, index, scale):
    step = DIFFERENCE_STEP * scale
    forward = list(arguments)
    backward = list(arguments)
    forward[index] = arguments[index] + step
    backward[index] = arguments[index] - step
    return (function(*forward) - function(*backward)) / (2 * step)


def _arc_length_edges(points, parameters, count):
    """Parameter values cutting a sampled polyline into `count` pieces of equal arc length."""
    arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    )
    return np.interp(np.linspace(0, arc[-1], count + 1), arc, parameters)


def _polyline_length(points):
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


@dataclass(frozen=True)
class Discretization:
    """Quadrature on a boundary: points (S, q, d), weights (S, q) summing to 1 per segment,
    segment sizes (S,) (lengths or areas), and unit normals (S, q, d) where defined (curves in
    2D, surfaces in 3D), else None."""

    points: np.ndarray
    weights: np.ndarray
    sizes: np.ndarray
    normals: np.ndarray | None


class Curve:
    """A parametric curve gamma(t) for t in `interval`, in any dimension.

    `function` maps an array of parameters t (shape (n,)) to points (n, dim); it can also be a
    sympy expression (a sequence of expressions in one symbol), which gives exact derivatives.
    `derivative` optionally gives gamma'(t) for a callable; otherwise central differences are
    used. In 2D the normal points to the right of the direction of travel, i.e. outwards for a
    counter-clockwise closed curve. `closed` is detected from the end points unless given.
    """

    def __init__(self, function, interval=(0.0, 1.0), derivative=None, closed=None):
        self._function, derivatives = _as_function(function, 1)
        if derivative is None and derivatives is not None:
            derivative = derivatives[0]
        self._derivative = derivative
        start, stop = (float(v) for v in interval)
        if not (np.isfinite(start) and np.isfinite(stop) and start < stop):
            raise ValueError(f"interval must be finite and increasing, got {interval}")
        self.interval = (start, stop)
        self.dim = self(np.array([start])).shape[-1]
        length = self.length()
        if not (np.isfinite(length) and length > 0):
            raise ValueError(
                f"the curve must have a finite, positive length, got {length}"
            )
        if closed is None:
            ends = self(np.array([start, stop]))
            closed = bool(
                np.linalg.norm(ends[1] - ends[0]) <= CLOSED_TOLERANCE * length
            )
        self.closed = bool(closed)

    def __call__(self, t):
        return np.asarray(self._function(np.asarray(t, dtype=float)), dtype=float)

    def derivative(self, t):
        t = np.asarray(t, dtype=float)
        if self._derivative is not None:
            return np.asarray(self._derivative(t), dtype=float)
        scale = self.interval[1] - self.interval[0]
        return _central_difference(self._function, [t], 0, scale)

    def length(self):
        return _polyline_length(self(np.linspace(*self.interval, LENGTH_SAMPLES)))

    def resolution_length(self):
        """Longest default segment: the curve is cut into at least MIN_CURVE_SEGMENTS."""
        return self.length() / MIN_CURVE_SEGMENTS

    def discretize(self, segment_length, order=QUADRATURE_ORDER):
        """Segments of about `segment_length` (equal arc length) with Gauss-Legendre points."""
        t = np.linspace(*self.interval, LENGTH_SAMPLES)
        samples = self(t)
        count = max(1, math.ceil(_polyline_length(samples) / segment_length))
        edges = _arc_length_edges(samples, t, count)
        nodes, weights = leggauss(order)
        middle = (edges[1:] + edges[:-1]) / 2
        half = (edges[1:] - edges[:-1]) / 2
        flat = (middle[:, None] + half[:, None] * nodes).ravel()
        points = self(flat).reshape(count, order, self.dim)
        tangent = self.derivative(flat).reshape(count, order, self.dim)
        speed = np.linalg.norm(tangent, axis=-1)
        quadrature = weights * half[:, None] * speed
        sizes = quadrature.sum(axis=1)
        normals = None
        if self.dim == 2:
            unit = tangent / np.where(speed > 0, speed, 1.0)[..., None]
            normals = np.stack([unit[..., 1], -unit[..., 0]], axis=-1)
        return Discretization(points, quadrature / sizes[:, None], sizes, normals)


class Surface:
    """A parametric surface sigma(s, t) in 3D, for s in `s_interval` and t in `t_interval`.

    `function` maps parameter arrays s, t (same shape) to points (..., 3); it can also be a sympy
    expression in two symbols (s is the first in alphabetical order). `derivatives` optionally
    gives (d sigma/ds, d sigma/dt) as a pair of callables. The normal is the normalised
    d sigma/ds x d sigma/dt. `closed` (no boundary edge, like a sphere or torus) is detected
    from the parameter edges unless given: each pair of opposite edges must coincide or
    collapse to points.
    """

    def __init__(self, function, s_interval, t_interval, derivatives=None, closed=None):
        self._function, symbolic = _as_function(function, 2)
        self._derivatives = derivatives if derivatives is not None else symbolic
        self.s_interval = tuple(float(v) for v in s_interval)
        self.t_interval = tuple(float(v) for v in t_interval)
        for name, (start, stop) in [
            ("s_interval", self.s_interval),
            ("t_interval", self.t_interval),
        ]:
            if not (np.isfinite(start) and np.isfinite(stop) and start < stop):
                raise ValueError(f"{name} must be finite and increasing")
        self.dim = self(
            np.array([self.s_interval[0]]), np.array([self.t_interval[0]])
        ).shape[-1]
        if self.dim != 3:
            raise ValueError(
                f"surfaces must be in 3D, got points of dimension {self.dim}"
            )
        area = self.area()
        if not (np.isfinite(area) and area > 0):
            raise ValueError(
                f"the surface must have a finite, positive area, got {area}"
            )
        self.closed = self._detect_closed(area) if closed is None else bool(closed)

    def __call__(self, s, t):
        s, t = np.broadcast_arrays(
            np.asarray(s, dtype=float), np.asarray(t, dtype=float)
        )
        return np.asarray(self._function(s, t), dtype=float)

    def partial_derivatives(self, s, t):
        s, t = np.broadcast_arrays(
            np.asarray(s, dtype=float), np.asarray(t, dtype=float)
        )
        if self._derivatives is not None:
            return tuple(np.asarray(d(s, t), dtype=float) for d in self._derivatives)
        scales = (np.ptp(self.s_interval), np.ptp(self.t_interval))
        return tuple(
            _central_difference(self._function, [s, t], index, scales[index])
            for index in range(2)
        )

    def _detect_closed(self, area):
        tolerance = CLOSED_TOLERANCE * math.sqrt(area)
        s = np.linspace(*self.s_interval, 33)
        t = np.linspace(*self.t_interval, 33)
        pairs = [
            (self(self.s_interval[0], t), self(self.s_interval[1], t)),
            (self(s, self.t_interval[0]), self(s, self.t_interval[1])),
        ]
        for first, second in pairs:
            coincide = np.linalg.norm(first - second, axis=-1).max() <= tolerance
            collapse = all(
                np.linalg.norm(edge - edge[0], axis=-1).max() <= tolerance
                for edge in (first, second)
            )
            if not (coincide or collapse):
                return False
        return True

    def area(self):
        nodes, weights = leggauss(8)
        edges_s = np.linspace(*self.s_interval, 33)
        edges_t = np.linspace(*self.t_interval, 33)
        s = (
            (edges_s[1:] + edges_s[:-1])[:, None] + np.diff(edges_s)[:, None] * nodes
        ) / 2
        t = (
            (edges_t[1:] + edges_t[:-1])[:, None] + np.diff(edges_t)[:, None] * nodes
        ) / 2
        s_grid, t_grid = np.meshgrid(s.ravel(), t.ravel(), indexing="ij")
        d_s, d_t = self.partial_derivatives(s_grid, t_grid)
        jacobian = np.linalg.norm(np.cross(d_s, d_t), axis=-1)
        w_s = np.outer(np.diff(edges_s) / 2, weights).ravel()
        w_t = np.outer(np.diff(edges_t) / 2, weights).ravel()
        return float(np.sum(jacobian * w_s[:, None] * w_t[None, :]))

    def resolution_length(self):
        """Longest default patch size: about MIN_SURFACE_PATCHES patches over the surface."""
        return math.sqrt(self.area() / MIN_SURFACE_PATCHES)

    def discretize(self, segment_length, order=QUADRATURE_ORDER):
        """Patches of about `segment_length` across, with tensor Gauss-Legendre points.

        The surface is cut into strips along s, and each strip into patches along t, all of
        equal arc length (as measured through the middle of the parameter range / of the strip),
        so non-uniform parametrisations still get evenly sized patches. The number of patches per
        strip follows that strip's own length, so degenerate edges (like the poles of a sphere)
        do not pile up tiny patches.
        """
        s_line = np.linspace(*self.s_interval, LENGTH_SAMPLES)
        t_line = np.linspace(*self.t_interval, LENGTH_SAMPLES)
        t_probe = np.linspace(*self.t_interval, 9)
        s_lengths = [_polyline_length(self(s_line, t)) for t in t_probe]
        strips = max(1, math.ceil(np.mean(s_lengths) / segment_length))
        t_middle = np.mean(self.t_interval)
        s_edges = _arc_length_edges(self(s_line, t_middle), s_line, strips)
        nodes, weights = leggauss(order)

        points, quadrature, normals = [], [], []
        for s_start, s_stop in pairwise(s_edges):
            s_middle = (s_start + s_stop) / 2
            strip_line = self(s_middle, t_line)
            patches = max(1, math.ceil(_polyline_length(strip_line) / segment_length))
            t_edges = _arc_length_edges(strip_line, t_line, patches)
            s_half = (s_stop - s_start) / 2
            t_half = (t_edges[1:] - t_edges[:-1]) / 2
            s_nodes = s_middle + s_half * nodes
            t_centres = (t_edges[1:] + t_edges[:-1]) / 2
            t_nodes = t_centres[:, None] + t_half[:, None] * nodes
            # (patches, order_s, order_t) parameter grids
            s_grid = np.broadcast_to(s_nodes[None, :, None], (patches, order, order))
            t_grid = np.broadcast_to(t_nodes[:, None, :], (patches, order, order))
            location = self(s_grid, t_grid)
            d_s, d_t = self.partial_derivatives(s_grid, t_grid)
            cross = np.cross(d_s, d_t)
            area = np.linalg.norm(cross, axis=-1)
            weight = weights[None, :, None] * weights[None, None, :] * area
            weight *= s_half * t_half[:, None, None]
            points.append(location.reshape(patches, order * order, 3))
            quadrature.append(weight.reshape(patches, order * order))
            unit = cross / np.where(area > 0, area, 1.0)[..., None]
            normals.append(unit.reshape(patches, order * order, 3))

        quadrature = np.concatenate(quadrature)
        sizes = quadrature.sum(axis=1)
        quadrature = quadrature / np.where(sizes > 0, sizes, 1.0)[:, None]
        return Discretization(
            np.concatenate(points), quadrature, sizes, np.concatenate(normals)
        )


def _net_flux_check(sizes, flux_means, magnitudes, name):
    """Raise unless the prescribed net flux through a closed boundary is (nearly) zero.

    The flux is compared with the size of the prescribed values (`magnitudes`, per-segment means
    of |value|), not with the fluxes themselves: a purely tangential wall velocity has fluxes
    that are only round-off.
    """
    net = float(np.sum(sizes * flux_means))
    scale = float(np.sum(sizes * magnitudes))
    if abs(net) > NET_FLUX_TOLERANCE * scale:
        raise ValueError(
            f"{name} prescribes a net flux of {net:.3g} through a closed boundary, but a "
            f"divergence-free field has zero net flux through any closed boundary "
            f"(divergence theorem); adjust the prescribed values"
        )


class _Condition:
    """A condition on a boundary, imposed in the mean over segments of about one spacing."""

    def __init__(self, boundary, value=0.0, segment_length=None):
        if not isinstance(boundary, (Curve, Surface)):
            raise TypeError(
                f"boundary must be a Curve or Surface, got {type(boundary).__name__}"
            )
        if segment_length is not None:
            segment_length = float(segment_length)
            if not (np.isfinite(segment_length) and segment_length > 0):
                raise ValueError(
                    f"segment_length must be positive, got {segment_length}"
                )
        self.boundary = boundary
        self.value = value
        self.segment_length = segment_length

    def _values(self, points, components):
        """The prescribed value at quadrature points (Q, d): shape (Q,) or (Q, d)."""
        value = self.value(points) if callable(self.value) else self.value
        shape = (len(points),) if components == 1 else (len(points), components)
        value = np.asarray(value, dtype=float)
        try:
            value = np.broadcast_to(value, shape)
        except ValueError:
            raise ValueError(
                f"{type(self).__name__} value must broadcast to {shape}, got shape {value.shape}"
            ) from None
        if not np.isfinite(value).all():
            raise ValueError(f"{type(self).__name__} values must be finite")
        return value

    def default_segment_length(self, spacing):
        """The given segment_length, else `spacing` capped by the boundary's resolution."""
        if self.segment_length is not None:
            return self.segment_length
        return min(spacing, self.boundary.resolution_length())

    def functionals(self, dim, segment_length, divergence_free=False):
        """(atom points (Q, dim), weight entries (rows, columns, values) for an atom-local
        (Q dim, F) matrix, targets (F,)) for this condition at the given segment length.

        `divergence_free` enables the closed-boundary handling for divergence-free fields."""
        if self.boundary.dim != dim:
            raise ValueError(
                f"the boundary is {self.boundary.dim}D but the data is {dim}D"
            )
        discretization = self.boundary.discretize(segment_length)
        closed = divergence_free and self.boundary.closed
        return self._functionals(discretization, dim, closed)


class NormalComponent(_Condition):
    """Prescribe the normal component n . u on a boundary (by default 0).

    For a flow this is a no-penetration (slip) wall, or an inflow/outflow rate if non-zero; for
    a curl-free field it prescribes the normal derivative of the potential (0 means an
    insulating wall). Needs a codimension-1 boundary: a curve in 2D or a surface in 3D. `value`
    is a number or a function of points (Q, dim) returning (Q,). The condition holds in the mean
    over segments of length `segment_length` (default: about the point spacing). For a
    divergence-free field and a closed boundary, the prescribed values must have zero net flux.
    """

    def _functionals(self, discretization, dim, closed):
        if discretization.normals is None:
            raise ValueError(
                "a normal component needs a curve in 2D or a surface in 3D"
            )
        segments, order, _ = discretization.points.shape
        points = discretization.points.reshape(-1, dim)
        atom = np.arange(segments * order)
        rows = (dim * atom[:, None] + np.arange(dim)).ravel()
        columns = np.repeat(np.repeat(np.arange(segments), order), dim)
        values = (discretization.weights[..., None] * discretization.normals).ravel()
        prescribed = self._values(points, 1).reshape(segments, order)
        targets = np.sum(discretization.weights * prescribed, axis=1)
        if closed:
            magnitudes = np.sum(discretization.weights * np.abs(prescribed), axis=1)
            _net_flux_check(
                discretization.sizes, targets, magnitudes, type(self).__name__
            )
            # The segment fluxes of a divergence-free field sum to zero, so the last segment's
            # mean follows from the others: drop it to keep the system non-singular
            keep = columns < segments - 1
            rows, columns, values = rows[keep], columns[keep], values[keep]
            targets = targets[:-1]
        return points, (rows, columns, values), targets


class VectorValue(_Condition):
    """Prescribe the whole vector u on a boundary (by default 0, a no-slip wall).

    Works for curves in any dimension and for surfaces. `value` is a vector or a function of
    points (Q, dim) returning (Q, dim), e.g. the velocity of a moving wall. The condition holds
    in the mean over segments of length `segment_length` (default: about the point spacing).
    For a divergence-free field and a closed boundary, the prescribed values must have zero net
    flux.
    """

    def _functionals(self, discretization, dim, closed):
        segments, order, _ = discretization.points.shape
        points = discretization.points.reshape(-1, dim)
        atom = np.arange(segments * order)
        rows = (dim * atom[:, None] + np.arange(dim)).ravel()
        segment_of_atom = np.repeat(np.arange(segments), order)
        columns = (dim * segment_of_atom[:, None] + np.arange(dim)).ravel()
        values = np.repeat(discretization.weights.ravel(), dim)
        prescribed = self._values(points, dim).reshape(segments, order, dim)
        targets = np.sum(discretization.weights[..., None] * prescribed, axis=1).ravel()
        if closed and discretization.normals is not None:
            normal_flux = np.sum(
                discretization.weights
                * np.sum(prescribed * discretization.normals, axis=-1),
                axis=1,
            )
            magnitudes = np.sum(
                discretization.weights * np.linalg.norm(prescribed, axis=-1), axis=1
            )
            _net_flux_check(
                discretization.sizes, normal_flux, magnitudes, type(self).__name__
            )
        return points, (rows, columns, values), targets


def assemble(points, values, conditions, segment_lengths, divergence_free=False):
    """Atoms, weights and targets for data values plus boundary conditions.

    The first len(points) atoms are the data points with one functional per point and component
    (identity weights, in the first N dim columns); the condition functionals follow.
    `segment_lengths` gives the segment length for each condition.
    """
    n, dim = points.shape
    atoms = [points]
    rows = [np.arange(n * dim)]
    columns = [np.arange(n * dim)]
    weights = [np.ones(n * dim)]
    targets = [values.ravel()]
    atom_offset, functional_offset = n, n * dim
    for condition, length in zip(conditions, segment_lengths, strict=True):
        condition_points, (r, c, w), t = condition.functionals(
            dim, length, divergence_free
        )
        atoms.append(condition_points)
        rows.append(r + dim * atom_offset)
        columns.append(c + functional_offset)
        weights.append(w)
        targets.append(t)
        atom_offset += len(condition_points)
        functional_offset += len(t)
    matrix = sparse.csr_array(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(columns))),
        shape=(dim * atom_offset, functional_offset),
    )
    return np.concatenate(atoms), matrix, np.concatenate(targets)
