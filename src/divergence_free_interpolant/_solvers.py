"""Linear solvers for the sparse symmetric positive definite interpolation systems."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

from . import _backend, _core

SOLVERS = ("auto", "direct", "cg", "lstsq")

# 'auto' factorises systems up to this many unknowns and uses conjugate gradients above it.
# The sparse factorisation is robust to the ill-conditioning of randomly spaced points; in 2D it
# stays cheap for large systems, from 3D on fill-in makes it expensive (see bench_scaling.py).
AUTO_DIRECT_LIMIT = {2: 250_000}
AUTO_DIRECT_LIMIT_DEFAULT = 15_000

# If CG fails under 'auto' (e.g. on nearly coincident points), systems up to this many unknowns
# are retried with the direct solver
AUTO_FALLBACK_DIRECT_LIMIT = 50_000

# CG gives up after this many iterations (or 10 per unknown, if fewer); with the cluster
# preconditioner it typically needs a few hundred
CG_MAX_ITERATIONS = 10_000

# Direct solves of systems this large use the Rust core's sparse Cholesky (faer) from this
# dimension on, scipy's SuperLU otherwise. Measured on random points with about 40 neighbours
# each: 3D 15k unknowns 1.0 s against 2.1 s for SuperLU, 30k unknowns 3.4 s against 12.8 s,
# but SuperLU is faster below about 8000 unknowns, and in 2D its ordering leaves less fill-in
# (100k points: 7.5 s against 14.5 s)
DIRECT_CHOLESKY_FROM_DIM = 3
DIRECT_CHOLESKY_MIN_UNKNOWNS = 8000

# A solution whose relative residual exceeds this is rejected as numerically singular
MAX_RELATIVE_RESIDUAL = 1e-6


@dataclass(frozen=True)
class SolveInfo:
    """How a system was solved: the solver used and, for 'cg', the number of iterations."""

    solver: str
    unknowns: int
    iterations: int | None = None


def validate_tolerance(tol) -> float:
    """CG tolerance: a relative residual, so a finite number between 0 and 1."""
    tol = float(tol)
    if not (np.isfinite(tol) and 0 < tol < 1):
        raise ValueError(f"tol must be a number between 0 and 1, got {tol}")
    return tol


def resolve_solver(solver: str, unknowns: int, dim: int) -> str:
    if solver != "auto":
        return solver
    return (
        "direct"
        if unknowns <= AUTO_DIRECT_LIMIT.get(dim, AUTO_DIRECT_LIMIT_DEFAULT)
        else "cg"
    )


def _csr_arrays(matrix):
    """(indptr, indices, data) of a sparse matrix in canonical CSR form (column indices sorted and
    unique within each row), as the Rust core expects."""
    matrix = sparse.csr_array(matrix)
    if not matrix.has_canonical_format:
        matrix = matrix.copy()
        matrix.sum_duplicates()
    return (
        np.ascontiguousarray(matrix.indptr, dtype=np.int64),
        np.ascontiguousarray(matrix.indices, dtype=np.int64),
        np.ascontiguousarray(matrix.data, dtype=float),
    )


def _scipy_cg(matrix, rhs, tol, preconditioner, maxiter):
    """scipy's conjugate gradients (the reference implementation): (solution, info, iterations)."""
    counter = [0]

    def count(_):
        counter[0] += 1

    solution, info = splinalg.cg(
        matrix,
        rhs,
        rtol=tol,
        atol=0.0,
        M=preconditioner,
        maxiter=maxiter,
        callback=count,
    )
    return solution, info, counter[0]


def _cholesky(matrix, rhs):
    """Sparse Cholesky solve in the Rust core (faer); raises MemoryError if the factor does not
    fit in memory."""
    solution, reason = _core.cholesky_solve(
        _csr_arrays(matrix), np.ascontiguousarray(rhs, dtype=float)
    )
    if solution is None:
        hint = (
            "; are there duplicate or nearly coincident points?"
            if "positive definite" in reason
            else ""
        )
        raise np.linalg.LinAlgError(
            f"the interpolation matrix could not be factorised ({reason}){hint}"
        )
    return solution


def solve(matrix, rhs, solver: str, tol: float = 1e-10, preconditioner=None, dim=None):
    """Solve matrix @ x = rhs with a resolved solver, returning (solution, SolveInfo).

    `preconditioner` approximates the inverse of `matrix` for 'cg' (Jacobi if None). `dim` (the
    dimension of the points) picks the direct solver: a sparse Cholesky factorisation in the
    Rust core for large systems from 3D on, scipy's SuperLU otherwise (see
    DIRECT_CHOLESKY_FROM_DIM). Raises numpy.linalg.LinAlgError if the system cannot be solved
    accurately.
    """
    iterations = None
    use_cholesky = (
        dim is not None
        and dim >= DIRECT_CHOLESKY_FROM_DIM
        and matrix.shape[0] >= DIRECT_CHOLESKY_MIN_UNKNOWNS
        and not _backend.use_numpy()
    )
    if solver == "direct" and use_cholesky:
        solution = _cholesky(matrix, rhs)
    elif solver == "direct":
        # The matrix is symmetric positive definite, so no pivoting is needed; partial pivoting
        # would destroy the symmetric fill-reducing ordering (tens of times more fill-in).
        try:
            factor = splinalg.splu(
                sparse.csc_matrix(matrix),
                permc_spec="MMD_AT_PLUS_A",
                diag_pivot_thresh=0.0,
                options={"SymmetricMode": True},
            )
        # SuperLU reports exact singularity as a RuntimeError
        except RuntimeError as error:
            raise np.linalg.LinAlgError(
                f"the interpolation matrix is singular ({error})"
            ) from error
        solution = factor.solve(rhs)
    elif solver == "cg":
        if preconditioner is None:
            preconditioner = sparse.diags_array(1 / matrix.diagonal())
        maxiter = min(10 * len(rhs), CG_MAX_ITERATIONS)
        if _backend.use_numpy():
            solution, info, iterations = _scipy_cg(
                matrix, rhs, tol, preconditioner, maxiter
            )
        else:
            solution, iterations, info = _core.conjugate_gradients(
                _csr_arrays(matrix),
                _csr_arrays(preconditioner),
                np.ascontiguousarray(rhs, dtype=float),
                float(tol),
                maxiter,
            )
        if info > 0:
            raise np.linalg.LinAlgError(
                f"conjugate gradients did not reach rtol = {tol:g} in {info} iterations; the "
                f"system is probably ill-conditioned (nearly coincident points, or a large "
                f"support radius)"
            )
        if info < 0:
            raise np.linalg.LinAlgError(
                f"conjugate gradients broke down (info = {info}); the system is not "
                f"positive definite"
            )
    elif solver == "lstsq":
        solution = np.linalg.lstsq(matrix.toarray(), rhs, rcond=None)[0]
        return solution, SolveInfo(solver, len(rhs))
    else:
        raise ValueError(f"unknown solver {solver!r}, expected one of {SOLVERS}")

    # CG only guarantees its own tolerance (on a recursively updated residual, hence the margin)
    limit = (
        max(MAX_RELATIVE_RESIDUAL, 10 * tol)
        if solver == "cg"
        else MAX_RELATIVE_RESIDUAL
    )
    scale = max(np.linalg.norm(rhs), np.finfo(float).tiny)
    if (
        not np.isfinite(solution).all()
        or np.linalg.norm(matrix @ solution - rhs) > limit * scale
    ):
        raise np.linalg.LinAlgError(
            "the interpolation matrix is numerically singular; are there "
            "duplicate or nearly coincident points?"
        )
    return solution, SolveInfo(solver, len(rhs), iterations)
