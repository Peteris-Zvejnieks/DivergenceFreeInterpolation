"""Wendland radial basis functions psi_{nu,k}."""

import subprocess
import sys
from pathlib import Path

import pytest
import sympy as sp

from divergence_free_interpolant._wendland import minimal_nu, wendland_coefficients
from divergence_free_interpolant._wendland_sympy import _RBF, derive_coefficients
from divergence_free_interpolant._wendland_table import WENDLAND_COEFFICIENTS

ROOT = Path(__file__).resolve().parents[1]

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


# --- the shipped table -----------------------------------------------------------------------


def test_shipped_table_matches_the_symbolic_derivation():
    assert len(WENDLAND_COEFFICIENTS) == 16
    for (nu, k), coefficients in WENDLAND_COEFFICIENTS.items():
        assert coefficients == derive_coefficients(nu, k)


def test_shipped_table_covers_the_minimal_nu_up_to_nine_dimensions():
    for k in (1, 2, 3, 4):
        for dim in range(2, 10):
            assert (minimal_nu(k, dim), k) in WENDLAND_COEFFICIENTS


def test_parameters_outside_the_table_are_derived_with_sympy():
    assert (12, 2) not in WENDLAND_COEFFICIENTS
    assert wendland_coefficients(12, 2) == derive_coefficients(12, 2)


def test_generated_table_is_up_to_date():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "gen_kernels.py"), "--check"],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr


# --- without sympy -----------------------------------------------------------------------------

WITHOUT_SYMPY = """
import sys

# Make `import sympy` fail, as if it were not installed
sys.modules["sympy"] = None

import numpy as np

import divergence_free_interpolant as dfi

rng = np.random.default_rng(0)
points = rng.random((60, 2))
values = np.column_stack([-points[:, 1], points[:, 0]])
dfi.DivergenceFreeInterpolant(0.5).fit(points, values)
dfi.MultilevelCurlFreeInterpolant().fit(points, values)
circle = dfi.Curve(lambda t: np.stack([0.1 * np.cos(t), 0.1 * np.sin(t)], -1), (0, 6.3))
dfi.DivergenceFreeInterpolant(0.5).fit(
    points, values, conditions=dfi.NormalComponent(circle)
)
try:
    dfi.DivergenceFreeInterpolant(0.5, k=2, nu=12).fit(points, values)
except ImportError as error:
    assert "symbolic" in str(error), error
else:
    raise AssertionError("psi_12,2 should need sympy")
print("ok")
"""


def test_the_package_works_without_sympy():
    """Shipped kernels and callable boundaries need no sympy; other kernels explain why."""
    result = subprocess.run(
        [sys.executable, "-c", WITHOUT_SYMPY],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_default_use_does_not_import_sympy():
    code = (
        "import sys, numpy as np\n"
        "import divergence_free_interpolant as dfi\n"
        "p = np.random.default_rng(0).random((50, 3))\n"
        "dfi.DivergenceFreeInterpolant(0.5).fit(p, p)\n"
        "print('sympy' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
