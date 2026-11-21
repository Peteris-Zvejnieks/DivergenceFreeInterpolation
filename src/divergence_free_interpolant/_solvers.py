"""Linear solvers for the sparse symmetric positive definite interpolation systems."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

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


def solve(matrix, rhs, solver: str, tol: float = 1e-10, preconditioner=None):
    """Solve matrix @ x = rhs with a resolved solver, returning (solution, SolveInfo).

    `preconditioner` approximates the inverse of `matrix` for 'cg' (Jacobi if None).
    Raises numpy.linalg.LinAlgError if the system cannot be solved accurately.
    """
    iterations = None
    if solver == "direct":
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
        counter = [0]

        def count(_):
            counter[0] += 1

        solution, info = splinalg.cg(
            matrix,
            rhs,
            rtol=tol,
            atol=0.0,
            M=preconditioner,
            maxiter=min(10 * len(rhs), CG_MAX_ITERATIONS),
            callback=count,
        )
        iterations = counter[0]
        if info > 0:
            raise np.linalg.LinAlgError(
                f"conjugate gradients did not reach rtol = {tol:g} in {info} iterations; the "
                f"system is probably ill-conditioned (nearly coincident points, or a large "
                f"support radius)"
            )
        if info < 0:
            raise np.linalg.LinAlgError(
                f"conjugate gradients broke down (scipy info = {info}); the system is not "
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
