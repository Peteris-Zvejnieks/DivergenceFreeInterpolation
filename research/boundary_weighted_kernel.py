"""Research spike: exact wall conditions through a boundary-weighted divergence-free kernel.

Tier 1 (NormalComponent / VectorValue) imposes wall conditions in the mean over segments. This
spike tests tier 2 from the roadmap: build the condition into the kernel so it holds exactly,
everywhere on the wall.

Idea. Divergence-free fields are curls of potentials, u = curl(g A), with A a vector potential
modelled by the scalar kernel k(x, y) = g(x) g(y) psi(x - y) (times the identity), where g
vanishes on the wall. On the wall u = grad(g) x A, which is tangent: u . n = 0 exactly. With g^2
instead of g, u = 0 there (no slip). The matrix kernel is

    K(x, y) = curl_x curl_y^T [k(x, y) I] = tr(M) I - M^T,    M = grad_x grad_y^T k,

which holds in 2D (curl = perpendicular gradient) and 3D alike, and

    M = grad g(x) grad g(y)^T psi - g(y) grad g(x) grad psi^T + g(x) grad psi grad g(y)^T
        - g(x) g(y) Hess psi                                   (psi and its derivatives at x-y)

With g = 1 this is the usual divergence-free kernel (grad grad^T - Laplacian I) psi. K is
symmetric positive semi-definite (L_x L_y^T applied to a positive definite scalar kernel), and
its columns are divergence-free.

The wall function here is g = q((|x| - R) / l) for a cylinder / sphere of radius R, with
q(s) = 1 - (1 - s)^3 for s < 1 and 1 beyond: zero on the wall, slope 3 / l there, and C^2-flat
to 1 at distance l, so the kernel is unchanged away from the wall.

    uv run python research/boundary_weighted_kernel.py

Findings (cylinder / sphere of radius 0.3 in potential flow, no samples within 0.25 of the wall):
- with g = 1 the weighted kernel equals the package's kernel to round-off (3e-16);
- tier 2 makes u . n = 0 on the whole wall to round-off (1e-15), and u = 0 with g^2, against
  8.4e-6 (2D) and 2.5e-3 (3D) between segment centres for tier 1;
- the fields stay divergence-free (finite differences equal for all methods);
- all interpolation matrices were positive definite, in 3D too. Tier 1 initially worsened
  conditioning badly (smallest eigenvalue 6.8e-9 -> 3.2e-12 in 2D, 1.8e-6 -> 1.6e-9 in 3D):
  on a closed wall the segment fluxes of a divergence-free field sum to zero (divergence
  theorem), so one segment functional was redundant. Dropping it (as the package now does)
  gives 2.9e-9 in 2D and 1.0e-7 in 3D, close to the plain interpolant; tier 2 leaves the
  conditioning unchanged;
- 2D accuracy near the wall: RMS error 0.10 (tier 2, band 0.5), 0.17 (tier 2, band 0.25),
  0.22 (tier 1), 0.38 (plain). In 3D all methods are around 0.6 near the wall: the single-level
  reconstruction of the near-uniform far field dominates, so the comparison is inconclusive;
- when no sample lies within the band, tier 2 is the plain interpolant's potential multiplied by
  g near the wall, so the band width l sets how far the wall's influence reaches.

Open questions before productising: a smooth wall function for general parametric boundaries
(the distance to a curve is only smooth within its reach, and open curves have kinked ends),
combining several walls (R-functions / smooth minimum), choosing l automatically, and the
multilevel scheme (g would be shared by all levels).
"""

import numpy as np
from numpy.polynomial import polynomial

import divergence_free_interpolant as dfi
from divergence_free_interpolant._conditions import assemble
from divergence_free_interpolant._interpolant import median_spacing
from divergence_free_interpolant._kernel import get_kernel
from divergence_free_interpolant._wendland import wendland_coefficients

RADIUS = 0.3


# --- the weighted kernel ---------------------------------------------------------------------


class WallFunction:
    """g = q((|x| - R) / band) (or its square for no slip) around an obstacle at the origin."""

    def __init__(self, radius, band, power=1):
        self.radius, self.band, self.power = radius, band, power

    def __call__(self, points):
        r = np.linalg.norm(points, axis=-1)
        s = (r - self.radius) / self.band
        near = s < 1
        q = np.where(near, 1 - (1 - s) ** 3, 1.0)
        slope = np.where(near, 3 * (1 - s) ** 2 / self.band, 0.0)
        gradient = slope[..., None] * points / r[..., None]
        if self.power == 2:
            return q**2, 2 * q[..., None] * gradient
        return q, gradient


class WeightedKernel:
    """K(x, y) = tr(M) I - M^T for k = g(x) g(y) psi(|x - y| / delta)."""

    def __init__(self, dim, wall, support_radius, k=3):
        self.matrix_kernel = get_kernel("divergence_free", dim, None, k)
        self.psi = np.array(wendland_coefficients(self.matrix_kernel.nu, k), float)
        self.dim, self.wall, self.delta = dim, wall, support_radius

    def radial(self, z):
        """psi, grad psi and Hess psi of psi(|z| / delta), for differences z (..., d)."""
        u = z / self.delta
        a, h, unit = self.matrix_kernel._radial_parts(u)
        r = np.linalg.norm(u, axis=-1)
        psi = np.where(r <= 1, polynomial.polyval(r, self.psi), 0.0)
        gradient = a[..., None] * u / self.delta
        eye = np.eye(self.dim)
        outer = unit[..., :, None] * unit[..., None, :]
        hessian = (
            a[..., None, None] * eye + h[..., None, None] * outer
        ) / self.delta**2
        return psi, gradient, hessian

    def __call__(self, x, y):
        """Kernel blocks for points x (n, d) and y (m, d): shape (n, m, d, d)."""
        psi, gradient, hessian = self.radial(x[:, None, :] - y[None, :, :])
        gx, dgx = self.wall(x)
        gy, dgy = self.wall(y)
        m = (
            dgx[:, None, :, None] * dgy[None, :, None, :] * psi[..., None, None]
            - gy[None, :, None, None] * dgx[:, None, :, None] * gradient[..., None, :]
            + gx[:, None, None, None] * gradient[..., :, None] * dgy[None, :, None, :]
            - (gx[:, None] * gy[None, :])[..., None, None] * hessian
        )
        trace = np.trace(m, axis1=-2, axis2=-1)
        return trace[..., None, None] * np.eye(self.dim) - np.swapaxes(m, -1, -2)


class WeightedInterpolant:
    """Dense interpolation with the weighted kernel (fine for a few hundred points)."""

    def __init__(self, kernel):
        self.kernel = kernel

    def gram(self, points):
        blocks = self.kernel(points, points)
        n, d = points.shape
        return blocks.transpose(0, 2, 1, 3).reshape(n * d, n * d)

    def fit(self, points, values):
        self.points = points
        self.matrix = self.gram(points)
        self.coefficients = np.linalg.solve(self.matrix, values.ravel()).reshape(
            points.shape
        )
        return self

    def __call__(self, queries):
        # Bound the (queries, points, d, d) blocks to a few million entries per chunk
        chunk = max(1, 2_000_000 // (len(self.points) * self.kernel.dim**2))
        out = []
        for start in range(0, len(queries), chunk):
            blocks = self.kernel(queries[start : start + chunk], self.points)
            out.append(np.einsum("qpab,pb->qa", blocks, self.coefficients))
        return np.concatenate(out)


# --- test problems ---------------------------------------------------------------------------


def potential_flow(points):
    """Uniform flow past a cylinder (2D) or sphere (3D) of RADIUS: tangent to the wall."""
    r2 = np.sum(points**2, axis=-1)
    if points.shape[-1] == 2:
        x, y = points[..., 0], points[..., 1]
        return np.stack(
            [1 - RADIUS**2 * (x**2 - y**2) / r2**2, -2 * RADIUS**2 * x * y / r2**2], -1
        )
    r = np.sqrt(r2)
    flow = np.zeros_like(points)
    flow[..., 0] = 1 + RADIUS**3 / (2 * r**3)
    return flow - 1.5 * RADIUS**3 * points[..., :1] * points / r[..., None] ** 5


def wall_geometry(dim):
    if dim == 2:
        return dfi.Curve(
            lambda t: np.stack([RADIUS * np.cos(t), RADIUS * np.sin(t)], -1),
            (0, 2 * np.pi),
        )

    def sphere(s, t):
        return RADIUS * np.stack(
            [np.sin(s) * np.cos(t), np.sin(s) * np.sin(t), np.cos(s)], -1
        )

    return dfi.Surface(sphere, (0, np.pi), (0, 2 * np.pi))


def divergence(field, points, h=1e-5):
    total = np.zeros(len(points))
    for axis, step in enumerate(np.eye(points.shape[1]) * h):
        total += (field(points + step)[:, axis] - field(points - step)[:, axis]) / (
            2 * h
        )
    return total


def experiment(dim, n_samples, support, band, seed=0):
    rng = np.random.default_rng(seed)
    low = np.array([-1.5, -1.0, -1.0][:dim])
    high = -low
    candidates = rng.uniform(low, high, size=(20 * n_samples, dim))
    radius = np.linalg.norm(candidates, axis=1)
    # No samples close to the wall, where the methods differ most
    samples = candidates[radius > RADIUS + 0.25][:n_samples]
    values = potential_flow(samples)

    directions = rng.normal(size=(2000, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    on_wall = RADIUS * directions
    probes = rng.uniform(low, high, size=(20000, dim))
    r = np.linalg.norm(probes, axis=1)
    near = probes[(r > RADIUS) & (r < RADIUS + 0.25)]
    outside = probes[r > RADIUS]

    condition = dfi.NormalComponent(wall_geometry(dim))
    models = {
        "plain": dfi.DivergenceFreeInterpolant(support, solver="direct").fit(
            samples, values
        ),
        "tier 1": dfi.DivergenceFreeInterpolant(support, solver="direct").fit(
            samples, values, conditions=condition
        ),
        "tier 2": WeightedInterpolant(
            WeightedKernel(dim, WallFunction(RADIUS, band), support)
        ).fit(samples, values),
        "tier 2, g^2": WeightedInterpolant(
            WeightedKernel(dim, WallFunction(RADIUS, band, power=2), support)
        ).fit(samples, values),
    }

    print(f"\n{dim}D, {len(samples)} samples, support {support}, band l = {band}")
    print(
        f"  {'method':12s} {'max|u.n| wall':>14s} {'max|u| wall':>12s} "
        f"{'RMS near wall':>14s} {'RMS outside':>12s} {'max|div u|':>11s} {'lambda_min':>11s}"
    )
    for name, model in models.items():
        wall_values = model(on_wall)
        flux = np.abs(np.sum(wall_values * directions, axis=1)).max()
        speed = np.linalg.norm(wall_values, axis=1).max()
        near_error = np.linalg.norm(model(near) - potential_flow(near), axis=1)
        all_error = np.linalg.norm(model(outside) - potential_flow(outside), axis=1)
        probe = outside[np.linalg.norm(outside, axis=1) > RADIUS + 0.02][:200]
        div = np.abs(divergence(model, probe)).max()
        eigenvalue = np.linalg.eigvalsh(
            system_matrix(model, samples, values, condition)
        ).min()
        print(
            f"  {name:12s} {flux:14.1e} {speed:12.2f} "
            f"{np.sqrt(np.mean(near_error**2)):14.2e} {np.sqrt(np.mean(all_error**2)):12.2e} "
            f"{div:11.1e} {eigenvalue:11.1e}"
        )


def system_matrix(model, samples, values, condition):
    """The interpolation matrix each method actually solves (data + boundary functionals)."""
    if isinstance(model, WeightedInterpolant):
        return model.matrix
    if model._fit.nodes.shape[0] == len(samples):
        return model._gram_matrix(samples).toarray()
    # Tier 1: W^T K W over the data points and the boundary quadrature points, using the same
    # default segment length as fit()
    length = condition.default_segment_length(
        min(median_spacing(samples), model.support_radius / 2)
    )
    atoms, weights, _ = assemble(
        samples, values, [condition], [length], divergence_free=True
    )
    return (weights.T @ model._gram_matrix(atoms) @ weights).toarray()


def check_reduces_to_plain_kernel():
    """With g = 1 the weighted kernel is delta^(d-2) times the package's kernel."""

    class One:
        def __call__(self, points):
            return np.ones(points.shape[:-1]), np.zeros(points.shape)

    rng = np.random.default_rng(1)
    for dim in (2, 3):
        x, y = rng.uniform(-0.4, 0.4, (30, dim)), rng.uniform(-0.4, 0.4, (20, dim))
        weighted = WeightedKernel(dim, One(), 0.7)(x, y)
        plain = get_kernel("divergence_free", dim, None, 3).blocks(
            x[:, None] - y[None], 0.7
        )
        error = np.abs(weighted - 0.7 ** (dim - 2) * plain).max() / np.abs(plain).max()
        print(
            f"{dim}D: weighted kernel with g = 1 vs plain kernel, relative error {error:.1e}"
        )


def main():
    check_reduces_to_plain_kernel()
    experiment(2, n_samples=150, support=1.5, band=0.5)
    experiment(2, n_samples=150, support=1.5, band=0.25)
    experiment(3, n_samples=1500, support=0.9, band=0.5)
    print(
        "\nSummary: tier 2 imposes the wall condition exactly (to round-off) everywhere on the"
        "\nwall, keeps the field divergence-free and the matrix as well-conditioned as without a"
        "\nwall (tier 1 slightly less so). Near the wall it was the most accurate in 2D; in 3D the"
        "\nbase reconstruction error dominated. It needs a smooth wall function g with known"
        "\ngradient: easy for simple shapes, open for general parametric boundaries."
    )


if __name__ == "__main__":
    main()
