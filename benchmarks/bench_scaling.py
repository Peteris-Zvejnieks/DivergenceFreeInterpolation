"""Fit/evaluate timings vs. number of points, with a support radius that keeps the number of
neighbours per point fixed (the regime multilevel interpolation will use).

    uv run python benchmarks/bench_scaling.py [--dims 2 3] [--sizes 1000 10000 100000]
"""

import argparse
import math
import time

import numpy as np

import divergence_free_interpolant as dfi

NEIGHBOURS = {2: 30, 3: 50}


def support_for(n, dim, neighbours):
    """Support radius giving about `neighbours` points per ball for n uniform points in [0, 1]^dim."""
    ball = math.pi ** (dim / 2) / math.gamma(dim / 2 + 1)
    return (neighbours / (n * ball)) ** (1 / dim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 3])
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[1_000, 10_000, 100_000]
    )
    parser.add_argument("--solvers", nargs="+", default=["direct", "cg"])
    parser.add_argument("--queries", type=int, default=100_000)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    print(
        f"{'dim':>3} {'N':>8} {'solver':>7} {'support':>8} {'fit [s]':>8} {'eval/pt [us]':>12} {'node err':>9}"
    )
    for dim in args.dims:
        for n in args.sizes:
            points = rng.random((n, dim))
            values = rng.normal(size=(n, dim))
            queries = rng.random((args.queries, dim))
            support = support_for(n, dim, NEIGHBOURS.get(dim, 50))
            for solver in args.solvers:
                interp = dfi.DivergenceFreeInterpolant(support, solver=solver)
                start = time.perf_counter()
                interp.fit(points, values)
                fit_time = time.perf_counter() - start
                start = time.perf_counter()
                interp(queries)
                eval_time = (time.perf_counter() - start) / len(queries)
                error = np.abs(interp(points[:1000]) - values[:1000]).max()
                print(
                    f"{dim:>3} {n:>8} {solver:>7} {support:>8.4f} {fit_time:>8.2f} "
                    f"{eval_time * 1e6:>12.2f} {error:>9.1e}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
