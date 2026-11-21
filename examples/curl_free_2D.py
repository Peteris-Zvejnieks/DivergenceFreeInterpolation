"""Reconstruct a landscape from scattered slope measurements.

The slope (gradient) of a height map is a curl-free vector field. Interpolating slope samples with
a curl-free interpolant gives a field that is exactly a gradient, and its potential is the
reconstructed height map, recovered up to a constant offset.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

import divergence_free_interpolant as dfi

rng = np.random.default_rng(7)

## A landscape made of a few Gaussian hills and valleys: (center x, center y, height, width)
HILLS = [
    (0.30, 0.35, 1.0, 0.16),
    (0.70, 0.65, 0.8, 0.12),
    (0.75, 0.25, -0.6, 0.14),
    (0.25, 0.80, 0.5, 0.10),
]


def height(x, y):
    return sum(
        h * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * w**2))
        for cx, cy, h, w in HILLS
    )


def slope(x, y):
    """Analytic gradient of the height map, shape (..., 2)."""
    gx = sum(
        -h * (x - cx) / w**2 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * w**2))
        for cx, cy, h, w in HILLS
    )
    gy = sum(
        -h * (y - cy) / w**2 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * w**2))
        for cx, cy, h, w in HILLS
    )
    return np.stack([gx, gy], axis=-1)


## Slope measurements at scattered points
N = 200
samples = rng.random((N, 2))
sample_slopes = slope(samples[:, 0], samples[:, 1])

## Fit: positions and gradient vectors are both arrays of shape (N, 2)
interpolant = dfi.CurlFreeInterpolant(support_radius=0.35)
t1 = time.perf_counter()
interpolant.fit(samples, sample_slopes)
print(f"Fit time: {time.perf_counter() - t1:.3f} s")

## Evaluate the gradient field and its potential (the reconstructed height) on a grid
XX, YY = np.mgrid[0:1:150j, 0:1:150j]
grid = np.stack([XX, YY], axis=-1)
field = interpolant(grid)
reconstructed = interpolant.potential(grid)
true = height(XX, YY)

## The potential is only defined up to a constant: align it with the true heights before comparing
reconstructed += np.mean(true - reconstructed)
print(
    f"RMS height error: {np.sqrt(np.mean((reconstructed - true) ** 2)):.2e} (height range {np.ptp(true):.2f})"
)

## Plot: samples | reconstruction with gradient streamlines | ground truth
levels = np.linspace(
    min(true.min(), reconstructed.min()), max(true.max(), reconstructed.max()), 21
)
fig, (ax_samples, ax_fit, ax_true) = plt.subplots(
    1, 3, figsize=(16, 5.2), sharex=True, sharey=True, layout="constrained"
)

speed = np.linalg.norm(sample_slopes, axis=1)
ax_samples.quiver(
    *samples.T, *(sample_slopes / speed[:, None]).T, speed, cmap="viridis"
)
ax_samples.set_title(f"Slope measurements ({N} points)")

filled = ax_fit.contourf(XX, YY, reconstructed, levels=levels, cmap="terrain")
ax_fit.streamplot(
    XX.T,
    YY.T,
    field[..., 0].T,
    field[..., 1].T,
    color="k",
    linewidth=0.6,
    density=1.2,
    arrowsize=0.7,
)
ax_fit.plot(*samples.T, ".", color="crimson", markersize=2.5)
ax_fit.set_title("Reconstructed height + gradient streamlines")

ax_true.contourf(XX, YY, true, levels=levels, cmap="terrain")
ax_true.set_title("True height map")

for ax in (ax_samples, ax_fit, ax_true):
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
fig.colorbar(filled, ax=[ax_samples, ax_fit, ax_true], label="height", shrink=0.9)

## Compute the layout once, then freeze it so panning/zooming stays responsive
fig.canvas.draw()
fig.set_layout_engine("none")
plt.show()
