"""Wendland's compactly supported radial basis functions psi_{nu,k}, derived symbolically."""

from functools import lru_cache

from sympy import Poly, Rational, expand, product, symbols


def minimal_nu(k: int, dim: int) -> int:
    """Smallest nu for which psi_{nu,k} is positive definite in R^dim."""
    return dim // 2 + k + 1


@lru_cache
def wendland_coefficients(nu: int, k: int) -> tuple[Rational, ...]:
    """Exact coefficients c_j of psi_{nu,k}(r) = sum_j c_j r^j on 0 <= r <= 1 (ascending powers)."""
    poly = Poly(expand(_RBF(nu, k).eq), _RBF.r)
    return tuple(reversed(poly.all_coeffs()))


class _RBF:
    r, i, f = symbols("r i f")

    def __init__(self, nu: int, k: int):
        self.nu = nu
        self.k = k
        eq = 0
        for incr in range(k + 1):
            eq += self._beta(incr, k) * _RBF.r**incr * self.p(nu + 2 * k - incr)
        self.eq = eq

    def _beta(self, j: int, k: int):
        if j == 0 and k == 0:
            return Rational(1)
        else:
            coefficient = 0
            for incr in range(max([0, j - 1]), k):
                coefficient += (
                    self._beta(incr, k - 1)
                    * self._square_bracket(incr - j + 1).subs(_RBF.f, incr + 1)
                    / self._round_bracket(incr - j + 2).subs(
                        _RBF.f, self.nu + 2 * (k - 1) - incr + 1
                    )
                )
            return coefficient

    @staticmethod
    def p(nu: int):
        return (1 - _RBF.r) ** nu

    @staticmethod
    def _square_bracket(l: int):
        if l == -1:
            return 1 / (_RBF.f + 1)
        elif l == 0:
            return Rational(1)
        else:
            return product(_RBF.f - _RBF.i, (_RBF.i, 0, l - 1))

    @staticmethod
    def _round_bracket(l: int):
        if l == 0:
            return Rational(0)
        else:
            return product(_RBF.f + _RBF.i, (_RBF.i, 0, l - 1))
