"""Linear solvers for the sparse symmetric positive definite interpolation systems."""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

SOLVERS = ("auto", "direct", "cg", "lstsq")

# 'auto' factorises systems up to this many unknowns and uses conjugate gradients above it.
# Fill-in grows much faster in 3D, so the limit is lower there (see benchmarks/bench_scaling.py).
AUTO_DIRECT_LIMIT = {2: 250_000}
AUTO_DIRECT_LIMIT_DEFAULT = 15_000

# A solution whose relative residual exceeds this is rejected as numerically singular
MAX_RELATIVE_RESIDUAL = 1e-6


def resolve_solver(solver: str, unknowns: int, dim: int) -> str:
    if solver != "auto":
        return solver
    return (
        "direct"
        if unknowns <= AUTO_DIRECT_LIMIT.get(dim, AUTO_DIRECT_LIMIT_DEFAULT)
        else "cg"
    )


def solve(matrix, rhs, solver: str, tol: float = 1e-10):
    """Solve matrix @ x = rhs with a resolved solver; raises numpy.linalg.LinAlgError on failure."""
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
        except RuntimeError as error:  # SuperLU reports exact singularity this way
            raise np.linalg.LinAlgError(
                f"the interpolation matrix is singular ({error})"
            ) from error
        solution = factor.solve(rhs)
    elif solver == "cg":
        jacobi = sparse.diags_array(1 / matrix.diagonal())
        solution, info = splinalg.cg(
            matrix, rhs, rtol=tol, atol=0.0, M=jacobi, maxiter=10 * len(rhs)
        )
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
        return np.linalg.lstsq(matrix.toarray(), rhs, rcond=None)[0]
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
    return solution
