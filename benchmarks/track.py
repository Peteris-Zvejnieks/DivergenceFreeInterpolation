"""A fixed benchmark suite whose timings are tracked over time in benchmarks/history.jsonl.

    uv run python benchmarks/track.py

runs the suite, compares it with the last run and records it; with --no-record it only runs and
compares.

Each case is timed as the best of `--repeat` runs. Every record holds the commit (marked dirty
for uncommitted changes), the date, the package version and the machine; runs are compared with
the previous record from the same machine only, since timings from different machines do not
compare. Commit the history together with performance-relevant changes.
"""

import argparse
import datetime
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

import divergence_free_interpolant as dfi
from divergence_free_interpolant import walls

HISTORY = Path(__file__).with_name("history.jsonl")
# Changes beyond this fraction are flagged
NOTABLE = 0.15


def support_for(n, dim, neighbours):
    """Support radius giving about `neighbours` points per ball for n uniform points."""
    ball = math.pi ** (dim / 2) / math.gamma(dim / 2 + 1)
    return (neighbours / (n * ball)) ** (1 / dim)


def data(n, dim, seed=0, wall=None):
    rng = np.random.default_rng(seed)
    points = rng.random((n, dim))
    if wall is not None:
        points = points[wall.signed_distance(points)[0] > 0.01]
    return points, rng.normal(size=points.shape)


def cases():
    """name -> (setup, timed): setup() builds the inputs, timed(inputs) is measured."""

    def fit(model, n, dim, wall=None):
        def setup():
            return data(n, dim, wall=wall)

        return setup, lambda inputs: model().fit(*inputs)

    def evaluate(model, n, dim, queries):
        def setup():
            fitted = model().fit(*data(n, dim))
            return fitted, np.random.default_rng(1).random((queries, dim))

        return setup, lambda inputs: inputs[0](inputs[1])

    circle = walls.Circle((0.5, 0.5), 0.2, band=0.1)
    return {
        "2D single level fit, 30k points": fit(
            lambda: dfi.DivergenceFreeInterpolant(support_for(30_000, 2, 30)),
            30_000,
            2,
        ),
        "3D single level fit (CG), 30k points": fit(
            lambda: dfi.DivergenceFreeInterpolant(support_for(30_000, 3, 50)),
            30_000,
            3,
        ),
        "3D single level fit (Cholesky), 5k points": fit(
            lambda: dfi.DivergenceFreeInterpolant(
                support_for(5_000, 3, 50), solver="direct"
            ),
            5_000,
            3,
        ),
        "2D multilevel fit, 50k points": fit(
            dfi.MultilevelDivergenceFreeInterpolant, 50_000, 2
        ),
        "3D multilevel fit, 30k points": fit(
            dfi.MultilevelDivergenceFreeInterpolant, 30_000, 3
        ),
        "2D multilevel fit with a wall, 50k points": fit(
            lambda: dfi.MultilevelDivergenceFreeInterpolant(walls=circle),
            50_000,
            2,
            wall=circle,
        ),
        "2D multilevel evaluation, 200k queries": evaluate(
            dfi.MultilevelDivergenceFreeInterpolant, 50_000, 2, 200_000
        ),
        "3D multilevel evaluation, 200k queries": evaluate(
            dfi.MultilevelDivergenceFreeInterpolant, 30_000, 3, 200_000
        ),
    }


def git(*arguments):
    result = subprocess.run(
        ["git", *arguments],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).parent,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def machine():
    """A description of the machine, to compare runs from the same one only."""
    cpu = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return f"{cpu}, {os.cpu_count()} threads, {platform.system()}"


def previous_record(this_machine):
    if not HISTORY.exists():
        return None
    records = [json.loads(line) for line in HISTORY.read_text().splitlines() if line]
    same = [record for record in records if record["machine"] == this_machine]
    return same[-1] if same else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args()

    this_machine = machine()
    previous = previous_record(this_machine)
    if previous:
        print(f"Comparing with {previous['commit']} from {previous['date']}")
    results = {}
    for name, (setup, timed) in cases().items():
        inputs = setup()
        best = math.inf
        for _ in range(args.repeat):
            start = time.perf_counter()
            timed(inputs)
            best = min(best, time.perf_counter() - start)
        results[name] = round(best, 4)
        line = f"{name:<45} {best:8.3f} s"
        if previous and name in previous["results"]:
            change = best / previous["results"][name] - 1
            flag = "  <-- slower" if change > NOTABLE else ""
            flag = "  <-- faster" if change < -NOTABLE else flag
            line += f"  {change:+6.1%}{flag}"
        print(line, flush=True)

    commit = git("rev-parse", "--short", "HEAD") or "unknown"
    if git("status", "--porcelain", "--untracked-files=no"):
        commit += "-dirty"
    record = {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "commit": commit,
        "version": dfi.__version__,
        "python": platform.python_version(),
        "machine": this_machine,
        "results": results,
    }
    if not args.no_record:
        with HISTORY.open("a") as file:
            file.write(json.dumps(record) + "\n")
        print(f"Recorded in {HISTORY.name}")


if __name__ == "__main__":
    main()
