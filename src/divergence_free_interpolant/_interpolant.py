"""Divergence-free and curl-free RBF interpolants (Fuselier 2008) with compactly supported kernels."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from ._assembly import evaluate, evaluate_potential, gram_matrix
from ._conditions import Curve, NormalComponent, Surface, VectorValue, assemble
from ._kernel import CURL_FREE, DIVERGENCE_FREE, MatrixKernel, as_integer, get_kernel
from ._preconditioner import cluster_preconditioner
from ._solvers import (
    AUTO_FALLBACK_DIRECT_LIMIT,
    SOLVERS,
    SolveInfo,
    resolve_solver,
    solve,
    validate_tolerance,
)

# Boundary functionals are preconditioned with exact inverses of blocks of this many
# consecutive functionals (segments are numbered along the boundary, so blocks are local)
BOUNDARY_BLOCK = 256


@dataclass(frozen=True)
class _Fit:
    kernel: MatrixKernel
    support_radius: float
    nodes: np.ndarray  # (P, d) points the kernels are centred on
    coefficients: np.ndarray  # (P, d)
    tree: cKDTree
    solve_info: SolveInfo


def _as_data(points, values):
    """Copy and validate (N, dim) points and values for fitting."""
    points = _as_points(points)
    values = np.array(values, dtype=float)
    if values.shape != points.shape:
        raise ValueError(
            f"values must have the same shape as points, {points.shape}, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("values must be finite")
    n, dim = points.shape
    if dim > max(n, 3):
        raise ValueError(
            f"got {n} points in {dim} dimensions; more dimensions than points usually "
            f"means the arrays are transposed, pass them with shape (N, dim)"
        )
    return points, values


def as_conditions(conditions, dim):
    """Validate boundary conditions (None, a single one or an iterable) for `dim`D data."""
    if conditions is None:
        return []
    if isinstance(conditions, (NormalComponent, VectorValue)):
        conditions = [conditions]
    if isinstance(conditions, (Curve, Surface)):
        raise TypeError(
            "conditions must be conditions such as NormalComponent(boundary) or "
            "VectorValue(boundary), not a bare boundary"
        )
    try:
        conditions = list(conditions)
    except TypeError:
        raise TypeError(
            f"conditions must be a NormalComponent, a VectorValue or an iterable of them, "
            f"got {type(conditions).__name__}"
        ) from None
    for condition in conditions:
        if not isinstance(condition, (NormalComponent, VectorValue)):
            raise TypeError(
                f"conditions must be NormalComponent or VectorValue, "
                f"got {type(condition).__name__}"
            )
        if condition.boundary.dim != dim:
            raise ValueError(
                f"a {condition.boundary.dim}D boundary cannot constrain {dim}D data"
            )
        if isinstance(condition, NormalComponent) and not (
            isinstance(condition.boundary, Surface) or condition.boundary.dim == 2
        ):
            raise ValueError(
                "a normal component needs a curve in 2D or a surface in 3D"
            )
    return conditions


def median_spacing(points):
    """Median distance from each point to its nearest neighbour (None for a single point)."""
    if len(points) < 2:
        return None
    return float(np.median(cKDTree(points).query(points, k=2)[0][:, 1]))


def _as_points(points):
    """Copy and validate an (N, d) array of points."""
    points = np.array(points, dtype=float)
    if points.ndim != 2 or len(points) == 0:
        raise ValueError(
            f"points must have shape (N, dim) with N >= 1, got {points.shape}"
        )
    if not np.isfinite(points).all():
        raise ValueError("points must be finite")
    return points


class _MatrixKernelInterpolant:
    _kind: str

    def __init__(
        self,
        support_radius: float,
        *,
        k: int = 3,
        nu: int | None = None,
        solver: str = "auto",
        tol: float = 1e-10,
    ):
        support_radius = float(support_radius)
        if not np.isfinite(support_radius) or support_radius <= 0:
            raise ValueError(
                f"support_radius must be a positive number, got {support_radius}"
            )
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
        self.support_radius = support_radius
        self.k = k
        self.nu = nu
        self.solver = solver
        self.tol = tol
        self._fit = None

    @property
    def dim(self) -> int | None:
        """Dimension of the fitted data, None before fit()."""
        return None if self._fit is None else self._fit.kernel.dim

    def fit(self, points, values, conditions=()):
        """Interpolate vector `values` (N, dim) given at `points` (N, dim). Returns self.

        `conditions` are boundary conditions (NormalComponent, VectorValue) that the
        interpolant satisfies in addition, in the mean over boundary segments. Their default
        segment length is the median point spacing, at most half the support radius.
        """
        points, values = _as_data(points, values)
        conditions = as_conditions(conditions, points.shape[1])
        if not conditions:
            self._fit_functionals(points, None, values.ravel())
            return self
        spacing = median_spacing(points)
        default = self.support_radius / 2
        if spacing is not None and spacing > 0:
            default = min(default, spacing)
        lengths = [c.default_segment_length(default) for c in conditions]
        atoms, weights, targets = assemble(
            points, values, conditions, lengths, self._kind == DIVERGENCE_FREE
        )
        self._fit_functionals(atoms, weights, targets, identity_points=len(points))
        return self

    @property
    def solve_info(self) -> SolveInfo | None:
        """How the last fit's system was solved (solver, unknowns, CG iterations)."""
        return None if self._fit is None else self._fit.solve_info

    def __call__(self, points):
        """Evaluate the interpolant at `points` of shape (..., dim); returns shape (..., dim)."""
        fit = self._require_fit()
        queries, shape = self._as_queries(points, fit.kernel.dim)
        values = evaluate(
            fit.kernel,
            fit.tree,
            fit.nodes,
            fit.coefficients,
            queries,
            fit.support_radius,
        )
        return values.reshape(shape + (fit.kernel.dim,))

    def _fit_functionals(self, points, weights, data, identity_points=None):
        """Fit to linear functionals that are weighted sums of point evaluations.

        Functional i is lambda_i(u) = sum_p w_pi . u(x_p), with the weights given as a sparse
        (P dim, n_functionals) matrix (None means one functional per point and component).
        The first `identity_points` points carry plain point values in the first
        identity_points * dim functionals (all points when weights is None); duplicates are
        only an error among those, and they get the cluster preconditioner for CG.
        The interpolation matrix is W^T K W and the kernel coefficients at the points are W c.
        All state is replaced only after the solve succeeds; the parameters used are stored with
        the fit, so changing attributes such as support_radius only affects the next fit.
        """
        support_radius = self.support_radius
        dim = points.shape[1]
        kernel = get_kernel(self._kind, dim, self.nu, self.k)
        tree = cKDTree(points)
        if weights is None:
            identity_points = len(points)
        identity_points = identity_points or 0
        if identity_points and self.solver != "lstsq":
            duplicates = tree.query_pairs(0.0, output_type="ndarray")
            duplicates = duplicates[np.all(duplicates < identity_points, axis=1)]
            if len(duplicates):
                i, j = duplicates[0]
                raise ValueError(
                    f"points {i} and {j} coincide, which makes the system singular; "
                    f"merge duplicate points or use solver='lstsq'"
                )
        matrix = gram_matrix(kernel, points, support_radius, tree)
        if weights is not None:
            matrix = (weights.T @ matrix @ weights).tocsr()
        solver = resolve_solver(self.solver, matrix.shape[0], dim)
        preconditioner = None
        if solver == "cg" and identity_points:
            preconditioner = self._preconditioner(
                kernel, points, support_radius, tree, identity_points, matrix
            )
        try:
            solution, info = solve(
                matrix, data, solver, self.tol, preconditioner, dim=dim
            )
        except np.linalg.LinAlgError:
            # Under 'auto', fall back to the direct factorisation, which copes with nearly
            # coincident points that stall CG, as long as the system is small enough for it
            fallback = (
                self.solver == "auto"
                and solver == "cg"
                and matrix.shape[0] <= AUTO_FALLBACK_DIRECT_LIMIT
            )
            if not fallback:
                raise
            solution, info = solve(matrix, data, "direct", self.tol, dim=dim)
        if weights is not None:
            solution = weights @ solution
        self._fit = _Fit(
            kernel, support_radius, points, solution.reshape(-1, dim), tree, info
        )

    @staticmethod
    def _preconditioner(kernel, points, support_radius, tree, identity_points, matrix):
        """Block-Jacobi: clusters of close points for the point values, and exact inverses of
        consecutive blocks of the boundary functionals (neighbouring segments are strongly
        coupled, which plain Jacobi cannot capture)."""
        if identity_points == len(points) and matrix.shape[0] == points.size:
            return cluster_preconditioner(kernel, points, support_radius, tree)
        data = points[:identity_points]
        blocks = [cluster_preconditioner(kernel, data, support_radius, cKDTree(data))]
        rest = sparse.csr_array(matrix)[data.size :, data.size :]
        for start in range(0, rest.shape[0], BOUNDARY_BLOCK):
            stop = min(start + BOUNDARY_BLOCK, rest.shape[0])
            blocks.append(np.linalg.inv(rest[start:stop, start:stop].toarray()))
        return sparse.block_diag(blocks, format="csr")

    def _gram_matrix(self, points):
        points = _as_points(points)
        kernel = get_kernel(self._kind, points.shape[1], self.nu, self.k)
        return gram_matrix(kernel, points, self.support_radius)

    def _scalar_potential(self, points):
        fit = self._require_fit()
        queries, shape = self._as_queries(points, fit.kernel.dim)
        values = evaluate_potential(
            fit.kernel,
            fit.tree,
            fit.nodes,
            fit.coefficients,
            queries,
            fit.support_radius,
        )
        return values.reshape(shape)

    def _require_fit(self) -> _Fit:
        if self._fit is None:
            raise RuntimeError(
                f"this {type(self).__name__} has no data yet, call fit() first"
            )
        return self._fit

    @staticmethod
    def _as_queries(points, dim):
        queries = np.asarray(points, dtype=float)
        if queries.ndim == 0 or queries.shape[-1] != dim:
            raise ValueError(
                f"query points must have shape (..., {dim}), got {queries.shape}"
            )
        if not np.isfinite(queries).all():
            raise ValueError("query points must be finite")
        return queries.reshape(-1, dim), queries.shape[:-1]


class DivergenceFreeInterpolant(_MatrixKernelInterpolant):
    """Interpolates scattered vector data with an exactly divergence-free vector field.

    Parameters
    ----------
    support_radius : float
        Radius of the compactly supported kernel. Each point interacts with the points within this
        distance: larger values give smoother, more accurate fields but denser, worse-conditioned
        systems. A few times the typical point spacing is a good start.
    k : int, default 3
        Smoothness of the Wendland function psi_{nu,k} (it is C^2k); k >= 1.
    nu : int, optional
        Wendland parameter; defaults to the smallest positive definite choice, dim // 2 + k + 1.
    solver : {'auto', 'direct', 'cg', 'lstsq'}, default 'auto'
        'direct' sparse factorisation (Cholesky for large 3D systems, LU otherwise), 'cg'
        conjugate gradients (preconditioned by exact inverses over clusters of unusually close
        points), 'lstsq' dense least squares (small problems, tolerates duplicate points), 'auto'
        picks 'direct' or 'cg' based on the system size and dimension, and retries a failed 'cg'
        with 'direct' when feasible.
    tol : float, default 1e-10
        Relative residual tolerance of the 'cg' solver.
    """

    _kind = DIVERGENCE_FREE

    def stream_function(self, points):
        """2D only: stream function psi with interpolant (d psi/dy, -d psi/dx); shape (...,).

        Streamlines of the field are the contour lines of psi. psi is determined up to an additive
        constant; this one vanishes farther than support_radius from all data points and
        boundary conditions.
        """
        if self._require_fit().kernel.dim != 2:
            raise ValueError("a scalar stream function only exists for 2D fields")
        return self._scalar_potential(points)


class CurlFreeInterpolant(_MatrixKernelInterpolant):
    """Interpolates scattered vector data with an exactly curl-free (gradient) vector field.

    The interpolant is the gradient of a scalar potential, available through potential().
    Parameters are the same as for DivergenceFreeInterpolant.
    """

    _kind = CURL_FREE

    def potential(self, points):
        """Scalar potential P with grad P equal to the interpolant; shape (...,).

        P is determined up to an additive constant; this one vanishes farther than
        support_radius from all data points and boundary conditions.
        """
        return self._scalar_potential(points)
