"""Electrostatics: interpolate an electric field and recover its potential.

An electrostatic field E = -grad V is curl-free. Interpolating scattered field measurements with
a curl-free interpolant gives a field that is exactly a gradient; the interpolant's potential P
satisfies grad P = E, so the electric potential is V = -P (up to a constant).
The charges sit outside the measured region, where the field is smooth.
"""

import time

import numpy as np
import pyvista as pv

import divergence_free_interpolant as dfi

rng = np.random.default_rng(3)

## Point charges outside the cube [-1, 1]^3: (position, charge)
CHARGES = [((1.6, 0.0, 0.0), 1.0), ((-1.6, 0.3, 0.0), -1.0), ((0.0, 0.2, 1.7), 0.6)]


def electric_field(points):
    field = np.zeros_like(points)
    for position, charge in CHARGES:
        offset = points - np.asarray(position)
        field += charge * offset / np.linalg.norm(offset, axis=-1, keepdims=True) ** 3
    return field


def electric_potential(points):
    return sum(
        charge / np.linalg.norm(points - np.asarray(position), axis=-1)
        for position, charge in CHARGES
    )


## Field measurements at scattered points; sampling a slightly larger region than the plotted cube
## [-1, 1]^3 avoids the less accurate edges of the sampled region
N = 1000
samples = 1.2 * (2 * rng.random((N, 3)) - 1)
measured = electric_field(samples)

interpolant = dfi.CurlFreeInterpolant(support_radius=1.4)
t1 = time.perf_counter()
interpolant.fit(samples, measured)
print(f"Fit time: {time.perf_counter() - t1:.3f} s")

## Evaluate the field and the potential on a grid
n = 40
box = pv.ImageData(
    dimensions=(n, n, n), spacing=(2 / (n - 1),) * 3, origin=(-1, -1, -1)
)
t1 = time.perf_counter()
box["E"] = interpolant(box.points)
box["V"] = -interpolant.potential(box.points)
print(
    f"Time per evaluation (field + potential): {(time.perf_counter() - t1) / box.n_points * 1e6:.1f} us"
)

## Compare with the exact potential, after removing the free constant
exact = electric_potential(box.points)
offset = np.mean(exact - box["V"])
error = np.abs(box["V"] + offset - exact).max()
print(f"Max potential error: {error:.2e} (potential range {np.ptp(exact):.2f})")

## Left: measurements. Right: field lines of the interpolated field and equipotential surfaces.
plotter = pv.Plotter(shape=(1, 2), window_size=(1600, 800))

plotter.subplot(0, 0)
plotter.add_text("Field measurements", font_size=12)
inside = np.all(np.abs(samples) <= 1, axis=1)
cloud = pv.PolyData(samples[inside])
cloud["E"] = measured[inside]
cloud["|E|"] = np.linalg.norm(measured[inside], axis=1)
plotter.add_mesh(
    cloud.glyph(orient="E", scale=False, factor=0.12), scalars="|E|", cmap="plasma"
)
plotter.add_mesh(box.outline(), color="grey")

plotter.subplot(0, 1)
plotter.add_text("Interpolated field lines + equipotentials", font_size=12)
seeds = pv.Plane(
    center=(0, 0, 0),
    direction=(1, 0, 0),
    i_size=1.8,
    j_size=1.8,
    i_resolution=7,
    j_resolution=7,
)
lines = box.streamlines_from_source(
    seeds, vectors="E", integration_direction="both", max_length=6.0
)
plotter.add_mesh(lines.tube(radius=0.008), color="black")
surfaces = box.contour(
    isosurfaces=np.linspace(*np.percentile(box["V"], [5, 95]), 9), scalars="V"
)
plotter.add_mesh(
    surfaces, cmap="coolwarm", opacity=0.45, scalar_bar_args={"title": "V"}
)
plotter.add_mesh(box.outline(), color="grey")

for position, charge in CHARGES:
    plotter.add_mesh(
        pv.Sphere(radius=0.08, center=position), color="red" if charge > 0 else "blue"
    )

plotter.link_views()
plotter.camera_position = [(1.2, -6.0, 2.2), (0.0, 0.0, 0.2), (0.0, 0.0, 1.0)]
plotter.show()
