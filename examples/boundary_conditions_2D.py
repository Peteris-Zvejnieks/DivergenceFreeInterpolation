"""Flow past a cylinder: imposing a no-penetration wall as a boundary condition.

Velocity measurements of a uniform flow past a cylinder are scattered around the obstacle, but
none close to it. The interpolant alone knows nothing about the wall, so its streamlines cut
straight through the cylinder. Adding NormalComponent(circle) (zero normal velocity on the wall, in the mean over
short segments) makes the flow go around it, and improves the reconstruction near the wall.
"""

import matplotlib.pyplot as plt
import numpy as np

import divergence_free_interpolant as dfi

RADIUS = 0.3


def velocity(points):
    """Potential flow past a cylinder of RADIUS at the origin, uniform speed 1 far away."""
    x, y = points[..., 0], points[..., 1]
    r2 = x**2 + y**2
    u = 1 - RADIUS**2 * (x**2 - y**2) / r2**2
    v = -2 * RADIUS**2 * x * y / r2**2
    return np.stack([u, v], axis=-1)


def stream_function(points):
    x, y = points[..., 0], points[..., 1]
    return y * (1 - RADIUS**2 / (x**2 + y**2))


## Measurements: scattered points, none close to the cylinder (where measuring is impractical)
rng = np.random.default_rng(5)
candidates = rng.uniform([-1.5, -1.0], [1.5, 1.0], size=(2000, 2))
samples = candidates[np.linalg.norm(candidates, axis=1) > RADIUS + 0.25][:150]
measured = velocity(samples)

## The cylinder wall as a parametric curve
wall = dfi.Curve(
    lambda t: np.stack([RADIUS * np.cos(t), RADIUS * np.sin(t)], axis=-1),
    interval=(0, 2 * np.pi),
)

free = dfi.DivergenceFreeInterpolant(support_radius=1.5).fit(samples, measured)
walled = dfi.DivergenceFreeInterpolant(support_radius=1.5).fit(
    samples, measured, conditions=dfi.NormalComponent(wall)
)

## How much flow crosses the wall, and the error outside the cylinder
angles = np.linspace(0, 2 * np.pi, 720, endpoint=False)
normals = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
on_wall = RADIUS * normals
XX, YY = np.mgrid[-1.5:1.5:300j, -1.0:1.0:200j]
grid = np.stack([XX, YY], axis=-1)
outside = np.linalg.norm(grid, axis=-1) > RADIUS
for name, model in [("no wall condition", free), ("no-penetration wall", walled)]:
    flux = np.abs(np.sum(model(on_wall) * normals, axis=1)).max()
    error = np.linalg.norm(model(grid) - velocity(grid), axis=-1)[outside]
    print(
        f"{name:>20}: max |u.n| on the wall {flux:.1e}, RMS error outside {np.sqrt(np.mean(error**2)):.2e}"
    )

## Plot: measurements | without the wall | with the wall | true flow
fig, axes = plt.subplots(
    1, 4, figsize=(20, 4.6), sharex=True, sharey=True, layout="constrained"
)
speed = np.linalg.norm(measured, axis=1)
axes[0].quiver(*samples.T, *(measured / speed[:, None]).T, speed, cmap="viridis")
axes[0].set_title(f"Velocity measurements ({len(samples)} points)")

panels = [
    (axes[1], free.stream_function(grid), free(grid), "Without the wall condition"),
    (axes[2], walled.stream_function(grid), walled(grid), "With NormalComponent(wall)"),
    (axes[3], stream_function(grid), velocity(grid), "True flow"),
]
# The true stream function is singular inside the cylinder: scale the colours by the outside
limit = np.abs(stream_function(grid)[outside]).max()
levels = np.linspace(-limit, limit, 21)
for ax, psi, field, title in panels:
    # The stream functions are defined up to a constant: align them outside the cylinder
    if ax is not axes[3]:
        psi = psi + np.mean((stream_function(grid) - psi)[outside])
    filled = ax.contourf(XX, YY, psi, levels=levels, cmap="RdBu_r", extend="both")
    ax.streamplot(
        XX.T,
        YY.T,
        field[..., 0].T,
        field[..., 1].T,
        color="k",
        linewidth=0.6,
        density=1.3,
        arrowsize=0.7,
    )
    ax.set_title(title)
# The analytic potential flow is not physical inside the cylinder, so cover it there
axes[3].add_patch(plt.Circle((0, 0), RADIUS, color="lightgrey", zorder=3))
for ax in axes:
    ax.add_patch(
        plt.Circle((0, 0), RADIUS, fill=False, color="gold", linewidth=2.5, zorder=4)
    )
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.0, 1.0)
fig.colorbar(filled, ax=list(axes), label="stream function", shrink=0.9)

## Compute the layout once, then freeze it so panning/zooming stays responsive
fig.canvas.draw()
fig.set_layout_engine("none")
plt.show()
