"""Wendland's compactly supported radial functions psi_{nu,k}: exact polynomial coefficients.

The functions used in practice ship with the package as a table of exact coefficients
(_wendland_table.py, generated with sympy by tools/gen_kernels.py), so sympy is not needed at
runtime. Other (nu, k) are derived on demand with sympy, the optional "symbolic" extra.
"""

from fractions import Fraction
from functools import lru_cache

from ._wendland_table import WENDLAND_COEFFICIENTS


def minimal_nu(k: int, dim: int) -> int:
    """Smallest nu for which psi_{nu,k} is positive definite in R^dim."""
    return dim // 2 + k + 1


def _shipped_description() -> str:
    by_k = {}
    for nu, k in WENDLAND_COEFFICIENTS:
        by_k.setdefault(k, []).append(nu)
    return ", ".join(
        f"k = {k} with nu = {min(n)}..{max(n)}" for k, n in sorted(by_k.items())
    )


@lru_cache
def wendland_coefficients(nu: int, k: int) -> tuple[Fraction, ...]:
    """Exact coefficients c_j of psi_{nu,k}(r) = sum_j c_j r^j on 0 <= r <= 1 (ascending powers)."""
    shipped = WENDLAND_COEFFICIENTS.get((nu, k))
    if shipped is not None:
        return shipped
    try:
        from ._wendland_sympy import derive_coefficients
    except ImportError as error:
        raise ImportError(
            f"psi_{nu},{k} is not among the Wendland functions shipped with the package "
            f"({_shipped_description()}); deriving it needs sympy: "
            f"pip install 'Divergence-Free-Interpolant[symbolic]'"
        ) from error
    return derive_coefficients(nu, k)
