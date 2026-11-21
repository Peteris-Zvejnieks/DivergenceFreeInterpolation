"""Single-level vs multilevel interpolation of an analytic divergence-free field.

Prints fit time and RMS error for both, plus the per-level breakdown of the multilevel fit
(points, support radius, solver and CG iterations), on random points in the unit cube.

    uv run python benchmarks/bench_multilevel.py [--dims 2 3] [--sizes 10000 100000]
"""

import argparse
import math
import time

import numpy as np

import divergence_free_interpolant as dfi

# Single-level support radius: about this many neighbours per point (as in bench_scaling.py)
NEIGHBOURS = {2: 30, 3: 50}


def field(points):
    """A smooth divergence-free field (curl of a vector potential) in 2D or 3D."""
    if points.shape[1] == 2:
        x, y = points.T
        return np.stack([np.cos(3 * y) * x, -np.sin(3 * y) / 3 + np.sin(2 * x)], 1)
    x, y, z = points.T
    return np.stack([np.sin(y) * z, np.cos(z) - x, 0.3 * np.exp(x) - y], 1)


def rms_error(interpolant, queries):
    return float(
        np.sqrt(np.mean(np.sum((interpolant(queries) - field(queries)) ** 2, 1)))
    )


def timed_fit(interpolant, points, values):
    start = time.perf_counter()
    interpolant.fit(points, values)
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 3], choices=[2, 3])
    parser.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000])
    parser.add_argument("--support-factor", type=float, default=6.0)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    for dim in args.dims:
        for n in args.sizes:
            points = rng.random((n, dim))
            values = field(points)
            queries = rng.uniform(0.1, 0.9, size=(20_000, dim))
            ball = math.pi ** (dim / 2) / math.gamma(dim / 2 + 1)
            support = (NEIGHBOURS[dim] / (n * ball)) ** (1 / dim)

            single = dfi.DivergenceFreeInterpolant(support)
            single_time = timed_fit(single, points, values)
            multi = dfi.MultilevelDivergenceFreeInterpolant(
                support_factor=args.support_factor
            )
            multi_time = timed_fit(multi, points, values)

            print(f"\n{dim}D, N = {n}")
            print(
                f"  single level (support {support:.4f}): fit {single_time:7.2f} s, "
                f"RMS error {rms_error(single, queries):.1e}"
            )
            print(
                f"  multilevel (support factor {args.support_factor:g}): fit {multi_time:7.2f} s, "
                f"RMS error {rms_error(multi, queries):.1e}"
            )
            for number, level in enumerate(multi.levels, start=1):
                info = level.solve_info
                iterations = (
                    "" if info.iterations is None else f", {info.iterations} it"
                )
                print(
                    f"    level {number}: {level.n_points:7d} points, support "
                    f"{level.support_radius:.4f}, {info.solver}{iterations}"
                )


if __name__ == "__main__":
    main()
