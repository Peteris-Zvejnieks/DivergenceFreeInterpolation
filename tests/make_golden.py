"""Regenerate tests/data/golden.npz.

The golden file pins the interpolant's *output values* (not its internal coefficients) for a
few fixed configurations, so refactors can be checked against known-good behaviour.
Inputs are stored alongside the outputs, so the test does not depend on RNG reproducibility.

Only regenerate on purpose, when a behaviour change is intended and understood:

    uv run python tests/make_golden.py
"""

from pathlib import Path

import numpy as np

import divergence_free_interpolant as dfi

OUT = Path(__file__).parent / "data" / "golden.npz"

# name: (dim, nu, k, support_radius, n_nodes, domain_low, domain_high)
CASES = {
    "2d_nu5_k3": (2, 5, 3, 0.5, 60, 0.0, 1.0),
    "2d_nu4_k2": (2, 4, 2, 1.0, 40, 0.0, 1.0),
    "3d_nu5_k3": (3, 5, 3, 1.0, 60, -1.0, 1.0),
    "3d_nu4_k2": (3, 4, 2, 0.8, 50, -1.0, 1.0),
}


def field(points):
    """Divergence-free reference fields (rows are points)."""
    if points.shape[1] == 2:
        x, y = points.T
        return np.stack([-2 * x**3 * y, 3 * x**2 * y**2], axis=1)
    x, y, z = points.T
    return np.stack([np.sin(y) * z, np.cos(z) - x, 0.3 * np.exp(x) - y], axis=1)


def main():
    rng = np.random.default_rng(20261121)
    arrays = {}
    for name, (dim, nu, k, delta, n, lo, hi) in CASES.items():
        nodes = lo + (hi - lo) * rng.random((n, dim))
        values = field(nodes)
        queries = lo + (hi - lo) * rng.random((100, dim))

        interp = dfi.interpolant(nu=nu, k=k, dim=dim)
        interp.condition(nodes, values, support_radii=delta, method="linsolve")
        expected = interp(*queries.T)

        arrays[f"{name}/config"] = np.array([dim, nu, k, delta])
        arrays[f"{name}/nodes"] = nodes
        arrays[f"{name}/values"] = values
        arrays[f"{name}/queries"] = queries
        arrays[f"{name}/expected"] = expected

    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    print(f"wrote {OUT} ({len(CASES)} cases)")


if __name__ == "__main__":
    main()
