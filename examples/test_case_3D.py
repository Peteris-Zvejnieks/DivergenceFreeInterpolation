import pyvista as pv
import numpy as np
import time

import Divergence_Free_Interpolant as dfi

np.random.seed(69)
div = lambda n, d: np.divide(n, d, out = np.zeros_like(d), where=d!=0)

## Analytic vectorfield definition; has to be divergence free
vector_field = lambda x, y, z: np.array([z, -x, -y])

## Number of sample points
N = 50

## Random sample points
X, Y, Z = 2*np.random.rand(N) - 1, 2*np.random.rand(N) - 1, 2*np.random.rand(N) - 1

## Get vectorfield sample values
sample_UVW = vector_field(X, Y, Z)

## Initialize the interpolant, nu = 5, k = 3 will suffice almost always, dim is the dimensionality
initialized_interpolant = dfi.interpolant(nu = 5, k = 3, dim = 3)

## Condition the vectorfield 
## initialized_interpolant.condition(positions, vectors, support_radius)
## positions: np.ndarray, (dim, N)
## vectors: np.ndarray, (dim, N)
## support_radius: positive float
t1 = time.perf_counter()
initialized_interpolant.condition(np.array([X, Y, Z]).T, sample_UVW.T, 2)
print('Conditioning time: ', time.perf_counter() - t1)

## Create resampling points
_n, _m, _l = 10, 10, 10
box = pv.RectilinearGrid(np.linspace(-1, 1, _n), np.linspace(-1, 1, _m), np.linspace(-1, 1, _l))
XYZ = box.points
## Call the interpolant passing resampling coordinates
## initialized_interpolant(X, Y)
## X: np.ndarray : any_shape
## Y: np.ndarray : shape like X
## returns np.ndarray: X.shape + (dim,)
t1 = time.perf_counter()
UVW = initialized_interpolant(XYZ[:,0], XYZ[:,1], XYZ[:,2])
print('Time per interpolation: ', (time.perf_counter() - t1)/(XYZ.shape[0]))

## Visualize the sample and resampled fields

p = pv.Plotter(shape = (1, 2))
p.subplot(0, 0)

points = pv.PolyData(np.array([X, Y, Z]).T)
points['vel'] = sample_UVW.T

arrows = points.glyph(orient='vel', scale=True, factor=0.5)

p.add_mesh(points, color='maroon', point_size=5., render_points_as_spheres=True)
p.add_mesh(arrows)

p.subplot(0, 1)

box['vel'] = UVW

stream = box.streamlines('vel',
                         terminal_speed=0.01,
                         max_time=100.0, 
                         n_points=100,
                         source_radius=0.9,
                         source_center=(0, 0, 0))

p.add_mesh(box.extract_feature_edges(45))
p.add_mesh(stream.tube(radius=0.01))


p.link_views()
p.show()
