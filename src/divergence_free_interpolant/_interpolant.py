"""Divergence-free and curl-free RBF interpolants (Fuselier 2008) with compactly supported kernels."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ._assembly import evaluate, evaluate_potential, gram_matrix
from ._kernel import CURL_FREE, DIVERGENCE_FREE, MatrixKernel, as_integer, get_kernel
from ._solvers import SOLVERS, resolve_solver, solve


@dataclass(frozen=True)
class _Fit:
    kernel: MatrixKernel
    support_radius: float
    nodes: np.ndarray  # (P, d) points the kernels are centred on
    coefficients: np.ndarray  # (P, d)
    tree: cKDTree


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
        if not tol > 0:
            raise ValueError(f"tol must be positive, got {tol}")
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

    def fit(self, points, values):
        """Interpolate vector `values` (N, dim) given at `points` (N, dim). Returns self."""
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
        self._fit_functionals(points, None, values.ravel())
        return self

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

    def _fit_functionals(self, points, weights, data):
        """Fit to linear functionals that are weighted sums of point evaluations.

        Functional i is lambda_i(u) = sum_p w_pi . u(x_p), with the weights given as a sparse
        (P dim, n_functionals) matrix (None means one functional per point and component).
        The interpolation matrix is W^T K W and the kernel coefficients at the points are W c.
        All state is replaced only after the solve succeeds; the parameters used are stored with
        the fit, so changing attributes such as support_radius only affects the next fit.
        """
        support_radius = self.support_radius
        dim = points.shape[1]
        kernel = get_kernel(self._kind, dim, self.nu, self.k)
        tree = cKDTree(points)
        if weights is None and self.solver != "lstsq":
            duplicates = tree.query_pairs(0.0, output_type="ndarray")
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
        solution = solve(matrix, data, solver, self.tol)
        if weights is not None:
            solution = weights @ solution
        self._fit = _Fit(
            kernel, support_radius, points, solution.reshape(-1, dim), tree
        )

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
        'direct' sparse LU factorisation, 'cg' Jacobi-preconditioned conjugate gradients,
        'lstsq' dense least squares (small problems, tolerates duplicate points), 'auto' picks
        'direct' or 'cg' based on the system size.
    tol : float, default 1e-10
        Relative residual tolerance of the 'cg' solver.
    """

    _kind = DIVERGENCE_FREE

    def stream_function(self, points):
        """2D only: stream function psi with interpolant (d psi/dy, -d psi/dx); shape (...,).

        Streamlines of the field are the contour lines of psi. psi is determined up to an additive
        constant; this one vanishes farther than support_radius from all data points.
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
        support_radius from all data points.
        """
        return self._scalar_potential(points)
