"""Matrix-valued divergence-free and curl-free kernels built from Wendland functions.

For a radial function psi(r), with A(r) = psi'(r)/r and H(r) = psi''(r) - psi'(r)/r,

    Hess psi(x) = A(r) I + H(r) xhat xhat^T,        xhat = x / r.

The two kernels of Fuselier (2008) are then

    divergence-free:  (grad grad^T - Laplacian I) psi = (-(d-1) A - H) I + H xhat xhat^T
    curl-free:        -grad grad^T psi                = -A I - H xhat xhat^T

For Wendland functions with k >= 1, A and H are polynomials in r and H(0) = 0, so the kernels
are evaluated in any dimension without a special case at r = 0 (where xhat is set to 0).

With walls (see the walls module) the divergence-free kernel is weighted by a wall function g:
K(x, y) = tr(M) I - M^T with M = grad_x grad_y^T [delta^(2-d) g(x) g(y) psi(|x - y| / delta)],
which is no longer a function of x - y alone (docs/tier2_boundary_conditions.md).
"""

import numbers
from functools import lru_cache

import numpy as np
from numpy.polynomial import polynomial

from . import _core
from ._wendland import minimal_nu, wendland_coefficients
from .walls import wall_function

DIVERGENCE_FREE = "divergence_free"
CURL_FREE = "curl_free"
KINDS = (DIVERGENCE_FREE, CURL_FREE)


def as_integer(value, name: str) -> int:
    """Accept integers and integral floats such as 6.0 (e.g. read from a config file)."""
    # bool is an Integral, but True as a count is almost certainly a mistake
    if not isinstance(value, bool):
        if isinstance(value, numbers.Integral):
            return int(value)
        if isinstance(value, numbers.Real) and float(value).is_integer():
            return int(value)
    raise ValueError(f"{name} must be an integer, got {value!r}")


def resolve_parameters(k: int, nu: int | None, dim: int) -> int:
    """Validate the kernel parameters for `dim` and return nu (defaulting to the minimal one)."""
    k = as_integer(k, "k")
    if dim < 2:
        raise ValueError(
            f"dimension {dim} is not supported, the data must be at least 2D"
        )
    if k < 1:
        raise ValueError(
            f"k = {k} is not supported: the matrix kernels need a twice "
            f"differentiable RBF, so k >= 1"
        )
    lowest = minimal_nu(k, dim)
    if nu is None:
        return lowest
    nu = as_integer(nu, "nu")
    if nu < lowest:
        raise ValueError(
            f"nu = {nu}, k = {k} does not give a positive definite kernel in {dim}D, "
            f"nu must be at least {lowest}"
        )
    return nu


def radial_coefficients(nu: int, k: int):
    """Exact ascending coefficients of A(r) = psi'(r)/r and H(r) = psi''(r) - psi'(r)/r."""
    c = wendland_coefficients(nu, k)
    if c[1] != 0:
        raise ValueError(f"psi_{nu},{k} is not smooth at the origin")
    a = [j * c[j] for j in range(2, len(c))]
    h = [j * (j - 2) * c[j] for j in range(2, len(c))]
    return a, h


class MatrixKernel:
    """Divergence-free or curl-free matrix kernel Phi_delta(x) = delta^-d Phi(x / delta), or with
    `walls` the boundary-weighted divergence-free kernel."""

    def __init__(self, kind: str, dim: int, nu: int | None, k: int, walls=()):
        if kind not in KINDS:
            raise ValueError(f"unknown kernel kind {kind!r}, expected one of {KINDS}")
        self.kind = kind
        self.dim = dim
        self.k = as_integer(k, "k")
        self.nu = resolve_parameters(self.k, nu, dim)
        self.walls = tuple(walls)
        if self.walls and kind != DIVERGENCE_FREE:
            raise ValueError("walls are only supported for divergence-free kernels")
        for wall in self.walls:
            if wall.dim != dim:
                raise ValueError(
                    f"a {wall.dim}D wall cannot bound {dim}D data: {wall!r}"
                )
        a, h = radial_coefficients(self.nu, self.k)
        self._a = np.array(a, dtype=float)
        self._h = np.array(h, dtype=float)
        self._psi = np.array(wendland_coefficients(self.nu, self.k), dtype=float)
        # The same kernel in the Rust core, which assembles and evaluates with it
        if self.walls:
            self.core = _core.Kernel(
                kind,
                dim,
                self._a.tolist(),
                self._h.tolist(),
                self._psi.tolist(),
                [wall._core_spec() for wall in self.walls],
            )
        else:
            self.core = _core.Kernel(kind, dim, self._a.tolist(), self._h.tolist())

    def __reduce__(self):
        # The Rust kernel cannot be pickled: rebuild everything from the parameters, so that
        # fitted interpolants can be saved and sent to worker processes
        return (MatrixKernel, (self.kind, self.dim, self.nu, self.k, self.walls))

    def _radial_parts(self, u):
        """A(r), H(r) and xhat for scaled differences u (..., d); zero outside the support."""
        r = np.linalg.norm(u, axis=-1)
        inside = r <= 1
        a = np.where(inside, polynomial.polyval(r, self._a), 0.0)
        h = np.where(inside, polynomial.polyval(r, self._h), 0.0)
        unit = u / np.where(r > 0, r, 1.0)[..., np.newaxis]
        return a, h, unit

    def blocks(self, differences, support_radius: float):
        """Kernel matrices for coordinate differences of shape (..., d), returned as (..., d, d)."""
        a, h, unit = self._radial_parts(
            np.asarray(differences, dtype=float) / support_radius
        )
        outer = unit[..., :, np.newaxis] * unit[..., np.newaxis, :]
        eye = np.eye(self.dim)
        h = h[..., np.newaxis, np.newaxis]
        if self.kind == DIVERGENCE_FREE:
            identity_part = (-(self.dim - 1) * a)[..., np.newaxis, np.newaxis] - h
            out = identity_part * eye + h * outer
        else:
            out = -(a[..., np.newaxis, np.newaxis] * eye + h * outer)
        return support_radius ** (-self.dim) * out

    def pair_blocks(self, x, y, support_radius: float):
        """Kernel matrices K(x, y) for points x and y of shape (..., d), returned as (..., d, d).

        Without walls these are blocks(x - y); block (y, x) is always the transpose of (x, y).
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if not self.walls:
            return self.blocks(x - y, support_radius)
        delta = support_radius
        u = (x - y) / delta
        a, h, unit = self._radial_parts(u)
        r = np.linalg.norm(u, axis=-1)
        psi = np.where(r <= 1, polynomial.polyval(r, self._psi), 0.0)
        gradient = a[..., np.newaxis] * u / delta
        eye = np.eye(self.dim)
        outer = unit[..., :, np.newaxis] * unit[..., np.newaxis, :]
        hessian = (a[..., None, None] * eye + h[..., None, None] * outer) / delta**2
        gx, dgx = wall_function(self.walls, x)
        gy, dgy = wall_function(self.walls, y)
        m = delta ** (2 - self.dim) * (
            dgx[..., :, None] * dgy[..., None, :] * psi[..., None, None]
            - gy[..., None, None] * dgx[..., :, None] * gradient[..., None, :]
            + gx[..., None, None] * gradient[..., :, None] * dgy[..., None, :]
            - (gx * gy)[..., None, None] * hessian
        )
        trace = np.trace(m, axis1=-2, axis2=-1)
        return trace[..., None, None] * eye - np.swapaxes(m, -1, -2)

    def pair_potential_terms(self, x, y, coefficients, support_radius: float):
        """Per-pair terms of the scalar potential (see potential_terms) for points x and nodes y.

        With walls (2D): c . perp(grad_y k), where k = delta^(2-d) g(x) g(y) psi(|x - y| / delta)
        is the scalar kernel of the potential.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if not self.walls:
            return self.potential_terms(x - y, coefficients, support_radius)
        if self.dim != 2:
            raise ValueError("a scalar stream function only exists in 2D")
        delta = support_radius
        u = (x - y) / delta
        a, _, _ = self._radial_parts(u)
        r = np.linalg.norm(u, axis=-1)
        psi = np.where(r <= 1, polynomial.polyval(r, self._psi), 0.0)
        gx, _ = wall_function(self.walls, x)
        gy, dgy = wall_function(self.walls, y)

        def perp_dot(v):
            return v[..., 1] * coefficients[..., 0] - v[..., 0] * coefficients[..., 1]

        return (
            delta ** (2 - self.dim)
            * gx
            * (psi * perp_dot(dgy) - gy * a / delta * perp_dot(u))
        )

    def potential_terms(self, differences, coefficients, support_radius: float):
        """Per-pair terms of the scalar potential of sum_j Phi(x - x_j) c_j.

        With u_j = (x - x_j) / delta:
        curl-free: the interpolant is grad P, P(x) = -delta^(1-d) sum_j A(|u_j|) u_j . c_j;
        2D divergence-free: the interpolant is (d/dy, -d/dx) psi with the stream function
        psi(x) = -delta^(1-d) sum_j A(|u_j|) perp(u_j) . c_j, where perp(u) = (u_y, -u_x).
        """
        u = np.asarray(differences, dtype=float) / support_radius
        a, _, _ = self._radial_parts(u)
        if self.kind == DIVERGENCE_FREE:
            if self.dim != 2:
                raise ValueError("a scalar stream function only exists in 2D")
            u = np.stack([u[..., 1], -u[..., 0]], axis=-1)
        return (
            -(support_radius ** (1 - self.dim)) * a * np.sum(u * coefficients, axis=-1)
        )


@lru_cache
def get_kernel(kind: str, dim: int, nu: int | None, k: int, walls=()) -> MatrixKernel:
    return MatrixKernel(kind, dim, nu, k, tuple(walls))
