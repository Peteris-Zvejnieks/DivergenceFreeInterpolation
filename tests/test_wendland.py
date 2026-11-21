"""Wendland radial basis functions psi_{nu,k}."""

import pytest
import sympy as sp

from divergence_free_interpolant._wendland import (
    _RBF,
    minimal_nu,
    wendland_coefficients,
)

r = _RBF.r


def wendland_closed_form(nu, k):
    """Wendland (1995) closed forms, up to a positive constant factor."""
    if k == 0:
        return (1 - r) ** nu
    if k == 1:
        return (1 - r) ** (nu + 1) * ((nu + 1) * r + 1)
    if k == 2:
        return (1 - r) ** (nu + 2) * (
            (nu**2 + 4 * nu + 3) * r**2 + (3 * nu + 6) * r + 3
        )
    if k == 3:
        return (1 - r) ** (nu + 3) * (
            (nu**3 + 9 * nu**2 + 23 * nu + 15) * r**3
            + (6 * nu**2 + 36 * nu + 45) * r**2
            + (15 * nu + 45) * r
            + 15
        )
    raise NotImplementedError(k)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
@pytest.mark.parametrize("nu", [2, 3, 4, 5, 6])
def test_matches_closed_form_up_to_positive_constant(nu, k):
    ratio = sp.cancel(_RBF(nu, k).eq / wendland_closed_form(nu, k))
    assert r not in ratio.free_symbols, (
        f"psi_{nu},{k} is not a constant multiple of the closed form"
    )
    assert ratio > 0


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_smoothness_at_origin(k):
    """psi_{nu,k} is C^{2k}: the odd powers r^1, r^3, ..., r^(2k-1) must vanish."""
    poly = sp.Poly(sp.expand(_RBF(k + 2, k).eq), r)
    for power in range(1, 2 * k, 2):
        assert poly.coeff_monomial(r**power) == 0, f"r^{power} coefficient is nonzero"


@pytest.mark.parametrize("nu, k", [(3, 1), (4, 2), (5, 3)])
def test_compact_support_and_positivity(nu, k):
    psi = _RBF(nu, k).eq
    assert psi.subs(r, 1) == 0
    for value in [0, sp.Rational(1, 4), sp.Rational(1, 2), sp.Rational(9, 10)]:
        assert psi.subs(r, value) > 0


@pytest.mark.parametrize("nu, k", [(3, 1), (4, 2), (5, 3), (6, 4)])
def test_coefficients_reproduce_the_polynomial(nu, k):
    coefficients = wendland_coefficients(nu, k)
    assert (
        sp.expand(sum(c * r**j for j, c in enumerate(coefficients)) - _RBF(nu, k).eq)
        == 0
    )


@pytest.mark.parametrize(
    "k, dim, expected", [(1, 2, 3), (3, 2, 5), (3, 3, 5), (2, 4, 5), (3, 5, 6)]
)
def test_minimal_nu(k, dim, expected):
    assert minimal_nu(k, dim) == expected
