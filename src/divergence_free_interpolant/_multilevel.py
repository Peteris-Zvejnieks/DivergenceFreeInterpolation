"""Multilevel interpolation (Farrell, Gillow & Wendland 2017) with nested farthest-point levels.

Level j interpolates the residual left by levels 1..j-1 on a subset X_j of the data with spacing
h_j and support radius support_factor * h_j. The spacings shrink by `refinement` from level to
level and the last level uses all data points, so the sum of the levels interpolates the data
exactly. Because the support is proportional to the spacing, every level has a bounded number
of neighbours per point and a well-conditioned system, unlike a single level whose support must
be large compared to the spacing to be accurate.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ._interpolant import CurlFreeInterpolant, DivergenceFreeInterpolant, _as_data
from ._kernel import as_integer, get_kernel
from ._solvers import SOLVERS, SolveInfo, validate_tolerance
from ._thinning import farthest_point_ordering

# Under solver='auto', intermediate (quasi-uniform) levels larger than this many unknowns use CG,
# which converges in a bounded number of iterations on well-spaced points
QUASI_UNIFORM_DIRECT_LIMIT = 3_000


@dataclass(frozen=True)
class Level:
    """One level of a fitted multilevel interpolant."""

    n_points: int
    spacing: float
    support_radius: float
    solve_info: SolveInfo


def level_hierarchy(points, support_factor, refinement):
    """Nested node sets and spacings, coarse to fine; the last level contains every point.

    The finest spacing is the median nearest-neighbour distance of the data. Coarser spacings grow
    by 1 / refinement until the support of the coarsest level, support_factor * spacing, would
    exceed the extent of the data (the largest distance from the most central point). Level j
    holds the farthest-point prefix whose points are at least spacing_j apart and cover all data
    within spacing_j. If all points coincide (only solvable with 'lstsq'), there is one level.
    """
    n = len(points)
    tree = cKDTree(points)
    nearest = tree.query(points, k=2)[0][:, 1]
    finest = float(np.median(nearest))
    if finest == 0.0:
        positive = nearest[nearest > 0]
        if not positive.size:
            return [(np.arange(n), 1.0)]
        finest = float(np.median(positive))

    order, radii = farthest_point_ordering(points, finest / refinement, tree)
    extent = radii[1] if len(radii) > 1 else finest
    spacings = [finest]
    while support_factor * spacings[-1] / refinement <= extent:
        spacings.append(spacings[-1] / refinement)
    spacings.reverse()

    levels, previous = [], 0
    for spacing in spacings[:-1]:
        # Number of points with insertion radius >= spacing (radii are non-increasing)
        count = int(np.searchsorted(-radii, -spacing, side="right"))
        if count > previous:
            levels.append((order[:count], spacing))
            previous = count
    levels.append((np.arange(n), finest))
    return levels


class _MultilevelInterpolant:
    _single_level: type

    def __init__(
        self,
        *,
        support_factor: float = 6.0,
        refinement: float = 0.5,
        k: int = 3,
        nu: int | None = None,
        solver: str = "auto",
        tol: float = 1e-10,
    ):
        support_factor = float(support_factor)
        if not np.isfinite(support_factor) or support_factor <= 0:
            raise ValueError(
                f"support_factor must be a positive number, got {support_factor}"
            )
        refinement = float(refinement)
        if not 0 < refinement < 1:
            raise ValueError(f"refinement must be between 0 and 1, got {refinement}")
        k = as_integer(k, "k")
        nu = None if nu is None else as_integer(nu, "nu")
        if k < 1:
            raise ValueError(
                f"k = {k} is not supported: the matrix kernels need a twice "
                f"differentiable RBF, so k >= 1"
            )
        if solver not in SOLVERS:
            raise ValueError(f"unknown solver {solver!r}, expected one of {SOLVERS}")
        tol = validate_tolerance(tol)
        self.support_factor = support_factor
        self.refinement = refinement
        self.k = k
        self.nu = nu
        self.solver = solver
        self.tol = tol
        self._fitted = None
        self._spacings = None

    @property
    def dim(self) -> int | None:
        """Dimension of the fitted data, None before fit()."""
        return None if self._fitted is None else self._fitted[0].dim

    @property
    def levels(self) -> tuple[Level, ...]:
        """Per-level summary of the fit, coarse to fine (empty before fit())."""
        if self._fitted is None:
            return ()
        return tuple(
            Level(
                len(level._fit.nodes),
                spacing,
                level._fit.support_radius,
                level.solve_info,
            )
            for level, spacing in zip(self._fitted, self._spacings, strict=True)
        )

    def fit(self, points, values):
        """Interpolate vector `values` (N, dim) given at `points` (N, dim). Returns self."""
        points, values = _as_data(points, values)
        if len(points) < 2:
            raise ValueError("multilevel interpolation needs at least 2 points")
        # Reject unsupported dimensions and parameters before the (costly) hierarchy is built
        get_kernel(self._single_level._kind, points.shape[1], self.nu, self.k)
        if self.solver != "lstsq":
            duplicates = cKDTree(points).query_pairs(0.0, output_type="ndarray")
            if len(duplicates):
                i, j = duplicates[0]
                raise ValueError(
                    f"points {i} and {j} coincide, which makes the system singular; "
                    f"merge duplicate points or use solver='lstsq'"
                )
        residual = values.copy()
        fitted = []
        hierarchy = level_hierarchy(points, self.support_factor, self.refinement)
        for number, (nodes, spacing) in enumerate(hierarchy, start=1):
            solver = self.solver
            last = number == len(hierarchy)
            if solver == "auto" and not last:
                # Farthest-point levels are quasi-uniform, where CG converges in a bounded number
                # of iterations; only the final level holds arbitrarily close data points
                large = len(nodes) * points.shape[1] > QUASI_UNIFORM_DIRECT_LIMIT
                solver = "cg" if large else "direct"
            level = self._single_level(
                self.support_factor * spacing,
                k=self.k,
                nu=self.nu,
                solver=solver,
                tol=self.tol,
            )
            # The data was validated as a whole; a coarse level may legitimately have fewer
            # points than dimensions, which the public fit() would reject as transposed input
            level._fit_functionals(points[nodes], None, residual[nodes].ravel())
            fitted.append(level)
            if not last:
                # Update the residual everywhere, including this level's own nodes, so that
                # inexact (e.g. CG) solves are corrected by the finer levels, not ignored
                residual -= level(points)
        self._fitted = fitted
        self._spacings = tuple(spacing for _, spacing in hierarchy)
        return self

    def __call__(self, points, *, levels: int | None = None):
        """Evaluate the interpolant at `points` of shape (..., dim); returns shape (..., dim).

        With `levels=j`, only the j coarsest levels are summed (to inspect the coarse-to-fine
        refinement); the full interpolant uses all of them.
        """
        return sum(level(points) for level in self._require_fit(levels))

    def _scalar_potential(self, points, levels=None):
        return sum(
            level._scalar_potential(points) for level in self._require_fit(levels)
        )

    def _require_fit(self, levels=None):
        if self._fitted is None:
            raise RuntimeError(
                f"this {type(self).__name__} has no data yet, call fit() first"
            )
        if levels is None:
            return self._fitted
        levels = as_integer(levels, "levels")
        if not 1 <= levels <= len(self._fitted):
            raise ValueError(
                f"levels must be between 1 and {len(self._fitted)}, got {levels}"
            )
        return self._fitted[:levels]


class MultilevelDivergenceFreeInterpolant(_MultilevelInterpolant):
    """Multilevel version of DivergenceFreeInterpolant, for large and multi-scale data sets.

    Instead of one support radius, the data is split into nested levels from coarse to fine (by
    farthest-point sampling); each level interpolates what the coarser ones left over, with a
    support radius proportional to its point spacing. The result is exactly divergence-free and
    interpolates the data, while every level's system stays sparse and well-conditioned.

    Parameters
    ----------
    support_factor : float, default 6.0
        Support radius of each level divided by its point spacing. The levels only reduce the
        error reliably if this is large enough (about 6 with the default k = 3; below 5 the
        refinement can stall). Larger values are more accurate (8 is a good choice in 2D) but
        the number of neighbours per point grows like support_factor^dim.
    refinement : float, default 0.5
        Ratio between the spacings of consecutive levels.
    k, nu, solver, tol
        As for DivergenceFreeInterpolant, applied to every level.
    """

    _single_level = DivergenceFreeInterpolant

    def stream_function(self, points, *, levels: int | None = None):
        """2D only: stream function psi with interpolant (d psi/dy, -d psi/dx); shape (...,).

        `levels` restricts the sum to the coarsest levels, as for calling the interpolant.
        """
        if self._require_fit()[0].dim != 2:
            raise ValueError("a scalar stream function only exists for 2D fields")
        return self._scalar_potential(points, levels)


class MultilevelCurlFreeInterpolant(_MultilevelInterpolant):
    """Multilevel version of CurlFreeInterpolant, for large and multi-scale data sets.

    The interpolant is the gradient of a scalar potential, available through potential().
    Parameters are the same as for MultilevelDivergenceFreeInterpolant.
    """

    _single_level = CurlFreeInterpolant

    def potential(self, points, *, levels: int | None = None):
        """Scalar potential P with grad P equal to the interpolant; shape (...,).

        `levels` restricts the sum to the coarsest levels, as for calling the interpolant.
        """
        return self._scalar_potential(points, levels)
