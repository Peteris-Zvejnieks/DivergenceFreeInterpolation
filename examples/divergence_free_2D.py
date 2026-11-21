"""Reconstruct an incompressible flow from scattered velocity measurements.

The velocity of a 2D incompressible flow is divergence-free: u = (d psi/dy, -d psi/dx) for a
stream function psi, whose contour lines are the streamlines. Interpolating velocity samples
with a divergence-free interpolant gives a field that is exactly incompressible, and its stream
function is recovered as well, up to a constant offset.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

import divergence_free_interpolant as dfi

rng = np.random.default_rng(7)

## A flow made of a few Gaussian vortices: (center x, center y, strength, width).
## Positive strength turns counter-clockwise, negative clockwise.
VORTICES = [
    (0.30, 0.30, 1.0, 0.15),
    (0.68, 0.38, -0.9, 0.13),
    (0.45, 0.75, 0.7, 0.14),
    (0.85, 0.80, -0.5, 0.10),
]


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
N = 200
samples = rng.random((N, 2))
sample_velocities = velocity(samples[:, 0], samples[:, 1])

## Fit: positions and velocity vectors are both arrays of shape (N, 2)
interpolant = dfi.DivergenceFreeInterpolant(support_radius=0.35)
t1 = time.perf_counter()
interpolant.fit(samples, sample_velocities)
print(f"Fit time: {time.perf_counter() - t1:.3f} s")

## Evaluate the velocity field and its stream function on a grid
XX, YY = np.mgrid[0:1:150j, 0:1:150j]
grid = np.stack([XX, YY], axis=-1)
field = interpolant(grid)
reconstructed = interpolant.stream_function(grid)
true = stream_function(XX, YY)

## The stream function is only defined up to a constant: align it before comparing
reconstructed += np.mean(true - reconstructed)
rms = np.sqrt(np.mean((reconstructed - true) ** 2))
print(f"RMS stream function error: {rms:.2e} (range {np.ptp(true):.2f})")

## Plot: samples | reconstruction with streamlines | ground truth
limit = max(np.abs(true).max(), np.abs(reconstructed).max())
levels = np.linspace(-limit, limit, 21)
fig, (ax_samples, ax_fit, ax_true) = plt.subplots(
    1, 3, figsize=(16, 5.2), sharex=True, sharey=True, layout="constrained"
)

speed = np.linalg.norm(sample_velocities, axis=1)
ax_samples.quiver(
    *samples.T, *(sample_velocities / speed[:, None]).T, speed, cmap="viridis"
)
ax_samples.set_title(f"Velocity measurements ({N} points)")

filled = ax_fit.contourf(XX, YY, reconstructed, levels=levels, cmap="RdBu_r")
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
ax_fit.plot(*samples.T, ".", color="gold", markersize=2.5)
ax_fit.set_title("Reconstructed stream function + streamlines")

ax_true.contourf(XX, YY, true, levels=levels, cmap="RdBu_r")
ax_true.set_title("True stream function")

for ax in (ax_samples, ax_fit, ax_true):
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
fig.colorbar(
    filled, ax=[ax_samples, ax_fit, ax_true], label="stream function", shrink=0.9
)

## Compute the layout once, then freeze it so panning/zooming stays responsive
fig.canvas.draw()
fig.set_layout_engine("none")
plt.show()
