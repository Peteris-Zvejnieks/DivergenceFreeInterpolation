"""Independent symbolic reference for the matrix kernels, by direct differentiation with sympy.

The package evaluates the kernels through the radial form F(r) I + H(r) xhat xhat^T; this module
derives them the straightforward way (Hessian of psi(|x|)) so the two can be compared.
"""

from functools import lru_cache

import numpy as np
import sympy as sp

from divergence_free_interpolant._wendland import _RBF


@lru_cache
def reference_kernel(kind, nu, k, dim):
    """Numpy function of points (M, dim) with 0 < |x| <= 1, returning kernels (M, dim, dim)."""
    xs = sp.symbols(f"x0:{dim}", real=True)
    psi = _RBF(nu, k).eq.subs(_RBF.r, sp.sqrt(sum(x**2 for x in xs)))
    hessian = sp.hessian(psi, xs)
    if kind == "divergence_free":
        # (grad grad^T - Laplacian I) psi
        matrix = hessian - hessian.trace() * sp.eye(dim)
    else:
        # -grad grad^T psi
        matrix = -hessian
    function = sp.lambdify(xs, matrix, "numpy")

    def evaluate(points):
        points = np.asarray(points, dtype=float)
        values = np.array(function(*points.T), dtype=float)
        return np.moveaxis(np.broadcast_to(values, (dim, dim, len(points))), -1, 0)

    return evaluate
