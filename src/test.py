import sys
sys.path.append('./build')
import Divergence_Free_Interpolant_alt as dfi

import pyvista as pv
import numpy as np
import time

np.random.seed(69)
div = lambda n, d: np.divide(n, d, out = np.zeros_like(d), where=d!=0)

vector_field = lambda x, y, z: np.array([z, -x, -y]).T

N = 1500

Positions = 2*np.random.rand(N, 3) - 1

sample_UVW = vector_field(Positions[:,0], Positions[:,1], Positions[:,2])

interpolant = dfi.Interpolant3D()
interpolant.setCoordinates(Positions)
t1 = time.perf_counter()
interpolant.generateArray()
print('array time: ', time.perf_counter() - t1)


array = interpolant.getArray()

import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams['figure.dpi'] = 500

lim = np.max(np.abs(array))

norm = mpl.colors.Normalize(vmin=-lim, vmax=lim)


plt.imshow(array, norm = norm, cmap = 'coolwarm')
plt.colorbar(norm = norm)
plt.show()
plt.close()


t1 = time.perf_counter()
interpolant.condition(sample_UVW)
print('Solving time: ', time.perf_counter() - t1)


_n, _m, _l = 10, 10, 10
box = pv.RectilinearGrid(np.linspace(-1, 1, _n), np.linspace(-1, 1, _m), np.linspace(-1, 1, _l))
XYZ = box.points

t1 = time.perf_counter()
UVW = interpolant.interpolate(XYZ)
print('Time per interpolation: ', (time.perf_counter() - t1)/(XYZ.shape[0]))

p = pv.Plotter(shape = (1, 2))
p.subplot(0, 0)

points = pv.PolyData(Positions)
points['vel'] = sample_UVW

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
