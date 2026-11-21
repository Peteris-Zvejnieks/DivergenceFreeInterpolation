"""Reconstruct a 3D incompressible flow with swirling vortices from scattered measurements.

The flow is the curl of a vector potential made of Gaussian blobs, u = curl(sum_i g_i(x) omega_i),
so it is divergence-free: each blob swirls around its own axis omega_i. Interpolating velocity
samples with a divergence-free interpolant gives a field that is exactly incompressible.
"""

import time

import numpy as np
import pyvista as pv

import divergence_free_interpolant as dfi

rng = np.random.default_rng(3)

## Vortex blobs: (center, axis (direction and strength), width)
BLOBS = [
    ((-0.4, -0.3, 0.0), (0.0, 0.0, 1.0), 0.35),
    ((0.45, 0.2, 0.3), (1.0, 0.0, 0.0), 0.30),
    ((0.0, 0.45, -0.45), (0.7, 0.7, 0.0), 0.30),
]


def velocity(points):
    """u = sum_i grad(g_i) x omega_i, with Gaussian g_i; shape (..., 3)."""
    field = np.zeros_like(points)
    for center, axis, width in BLOBS:
        offset = points - np.asarray(center)
        gaussian = np.exp(-np.sum(offset**2, axis=-1) / (2 * width**2))
        gradient = -offset * (gaussian / width**2)[..., np.newaxis]
        field += np.cross(gradient, np.asarray(axis))
    return field


## Velocity measurements at scattered points; sampling a slightly larger region than the plotted
## cube [-1, 1]^3 avoids the less accurate edges of the sampled region
N = 1000
samples = 1.2 * (2 * rng.random((N, 3)) - 1)
measured = velocity(samples)

interpolant = dfi.DivergenceFreeInterpolant(support_radius=1.3)
t1 = time.perf_counter()
interpolant.fit(samples, measured)
print(f"Fit time: {time.perf_counter() - t1:.3f} s")

## Evaluate the velocity field on a grid
n = 40
box = pv.ImageData(
    dimensions=(n, n, n), spacing=(2 / (n - 1),) * 3, origin=(-1, -1, -1)
)
t1 = time.perf_counter()
box["u"] = interpolant(box.points)
print(f"Time per evaluation: {(time.perf_counter() - t1) / box.n_points * 1e6:.1f} us")
box["|u|"] = np.linalg.norm(box["u"], axis=1)

## Compare with the exact velocity
error = np.linalg.norm(box["u"] - velocity(box.points), axis=1)
print(
    f"RMS velocity error: {np.sqrt(np.mean(error**2)):.2e} (max speed {box['|u|'].max():.2f})"
)

## Left: measurements. Right: streamlines of the interpolated field, coloured by speed.
plotter = pv.Plotter(shape=(1, 2), window_size=(1600, 800))

plotter.subplot(0, 0)
plotter.add_text("Velocity measurements", font_size=12)
inside = np.all(np.abs(samples) <= 1, axis=1)
cloud = pv.PolyData(samples[inside])
cloud["u"] = measured[inside]
cloud["|u|"] = np.linalg.norm(measured[inside], axis=1)
plotter.add_mesh(
    cloud.glyph(orient="u", scale=False, factor=0.12), scalars="|u|", cmap="plasma"
)
plotter.add_mesh(box.outline(), color="grey")

plotter.subplot(0, 1)
plotter.add_text("Interpolated streamlines + vortex axes", font_size=12)
seeds = pv.PolyData(1.6 * rng.random((60, 3)) - 0.8)
lines = box.streamlines_from_source(
    seeds, vectors="u", integration_direction="both", max_length=2.5
)
plotter.add_mesh(
    lines.tube(radius=0.007),
    scalars="|u|",
    cmap="plasma",
    scalar_bar_args={"title": "|u|"},
)
plotter.add_mesh(box.outline(), color="grey")

for center, axis, _ in BLOBS:
    direction = np.asarray(axis) / np.linalg.norm(axis)
    plotter.add_mesh(
        pv.Arrow(
            start=np.asarray(center) - 0.45 * direction,
            direction=direction,
            scale=0.9,
        ),
        color="black",
    )

plotter.link_views()
plotter.camera_position = [(1.2, -6.0, 2.2), (0.0, 0.0, 0.2), (0.0, 0.0, 1.0)]
plotter.show()
