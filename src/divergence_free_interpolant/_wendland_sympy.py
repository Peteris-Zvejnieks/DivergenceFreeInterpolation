"""Symbolic derivation of Wendland's compactly supported functions psi_{nu,k} with sympy.

This generates the coefficient table shipped with the package (tools/gen_kernels.py), and at
runtime derives functions outside that table. It is the only part of the package that needs
sympy (the optional "symbolic" extra), and it is imported only when needed.
"""

from fractions import Fraction

from sympy import Poly, Rational, expand, product, symbols


def derive_coefficients(nu: int, k: int) -> tuple[Fraction, ...]:
    """Exact coefficients c_j of psi_{nu,k}(r) = sum_j c_j r^j on 0 <= r <= 1 (ascending powers)."""
    poly = Poly(expand(_RBF(nu, k).eq), _RBF.r)
    return tuple(Fraction(int(c.p), int(c.q)) for c in reversed(poly.all_coeffs()))


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
