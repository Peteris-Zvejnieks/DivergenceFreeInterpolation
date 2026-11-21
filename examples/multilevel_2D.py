"""Multilevel interpolation of a multi-scale flow from 20 000 scattered velocity measurements.

The flow has large, medium and small vortices. A single support radius has to choose between
them: small enough to resolve the small eddies, it cannot bridge the gaps between random
samples for the large ones. Multilevel interpolation goes from coarse to fine: each level
interpolates what the coarser levels left over, with a support radius proportional to its own
point spacing. The panels show the reconstructed flow speed after the first level, after
the first three, and after all of them.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

import divergence_free_interpolant as dfi

rng = np.random.default_rng(11)

## Gaussian vortices on three scales: (center x, center y, strength, width).
## The strength is proportional to the width, so all vortices spin about equally fast.
VORTICES = []
for count, width in [(3, 0.2), (10, 0.07), (30, 0.025)]:
    for _ in range(count):
        center = rng.uniform(0.1, 0.9, size=2)
        VORTICES.append((*center, rng.choice([-1.0, 1.0]) * width, width))


def stream_function(x, y):
    return sum(
        s * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * w**2))
        for cx, cy, s, w in VORTICES
    )


def velocity(x, y):
    """Analytic velocity (d psi/dy, -d psi/dx), shape (..., 2)."""
    u = sum(
        -s * (y - cy) / w**2 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * w**2))
        for cx, cy, s, w in VORTICES
    )
    v = sum(
        s * (x - cx) / w**2 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * w**2))
        for cx, cy, s, w in VORTICES
    )
    return np.stack([u, v], axis=-1)


## Velocity measurements at scattered points
N = 20_000
samples = rng.random((N, 2))
measured = velocity(samples[:, 0], samples[:, 1])

interpolant = dfi.MultilevelDivergenceFreeInterpolant(support_factor=6.0)
t1 = time.perf_counter()
interpolant.fit(samples, measured)
print(f"Multilevel fit: {time.perf_counter() - t1:.2f} s")
for number, level in enumerate(interpolant.levels, start=1):
    print(
        f"  level {number}: {level.n_points:6d} points, support radius "
        f"{level.support_radius:.4f} ({level.solve_info.solver})"
    )

## Accuracy against the true velocity, compared with a single level using the finest support
queries = rng.uniform(0.05, 0.95, size=(20_000, 2))
exact = velocity(queries[:, 0], queries[:, 1])
single = dfi.DivergenceFreeInterpolant(interpolant.levels[-1].support_radius)
t1 = time.perf_counter()
single.fit(samples, measured)
single_time = time.perf_counter() - t1
for name, model in [("multilevel", interpolant), ("single level", single)]:
    error = np.linalg.norm(model(queries) - exact, axis=1)
    print(f"{name:>12}: RMS velocity error {np.sqrt(np.mean(error**2)):.2e}")
print(f"(single-level fit: {single_time:.2f} s)")

## Flow speed after 1, 3 and all levels, next to the truth. The speed shows every scale
## equally (the stream function of a small eddy is tiny), so the refinement is easy to see.
XX, YY = np.mgrid[0:1:250j, 0:1:250j]
grid = np.stack([XX, YY], axis=-1)
true = np.linalg.norm(velocity(XX, YY), axis=-1)
count = len(interpolant.levels)
stages = [1, min(3, count), count]
panels = [np.linalg.norm(interpolant(grid, levels=j), axis=-1) for j in stages]

levels_contour = np.linspace(0, true.max(), 25)
fig, axes = plt.subplots(
    1, 4, figsize=(20, 5.2), sharex=True, sharey=True, layout="constrained"
)
for ax, speed, levels in zip(axes[:-1], panels, stages, strict=True):
    filled = ax.contourf(
        XX, YY, speed, levels=levels_contour, cmap="magma", extend="max"
    )
    support = interpolant.levels[levels - 1].support_radius
    label = "level 1" if levels == 1 else f"levels 1-{levels}"
    ax.set_title(f"After {label} (finest support {support:.3f})")
axes[-1].contourf(XX, YY, true, levels=levels_contour, cmap="magma", extend="max")
axes[-1].set_title("True flow speed")
for ax in axes:
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
fig.colorbar(filled, ax=list(axes), label="speed |u|", shrink=0.9)

## Compute the layout once, then freeze it so panning/zooming stays responsive
fig.canvas.draw()
fig.set_layout_engine("none")
plt.show()
