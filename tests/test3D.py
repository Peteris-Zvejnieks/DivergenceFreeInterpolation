import pyvista as pv
import numpy as np
import time
# import matplotlib.pyplot as plt
# plt.rcParams['figure.dpi'] = 500
import sys
sys.path.append('/home/peteris/Documents/GitHub/DivergenceFreeInterpolation/Divergence_Free_Interpolant/build/')
import Divergence_Free_Interpolant as dfi
print(dir(dfi))

interpolant = dfi.Interpolant3D(support_radii = 3)

np.random.seed(69)
div = lambda n, d: np.divide(n, d, out = np.zeros_like(d), where=d!=0)

vector_field = lambda x, y, z: np.array([z, -x, -y])

N = 1500

X, Y, Z = 2*np.random.rand(N) - 1, 2*np.random.rand(N) - 1, 2*np.random.rand(N) - 1

sample_UVW = vector_field(X, Y, Z)
U, V, W = sample_UVW[0], sample_UVW[1], sample_UVW[2]


# t1 = time.perf_counter()
interpolant.setSampleCoordinates(X, Y, Z)
interpolant.condition(U, V, W)
# print('Conditioning time: ', time.perf_counter() - t1)

## Create resampling points
_n, _m, _l = 10, 10, 10
box = pv.RectilinearGrid(np.linspace(-1, 1, _n), np.linspace(-1, 1, _m), np.linspace(-1, 1, _l))
XYZ = box.points
t1 = time.perf_counter()
UVW = interpolant.interpolate(XYZ[:,0], XYZ[:,1], XYZ[:,2])
print('Time per interpolation: ', (time.perf_counter() - t1)/(XYZ.shape[0])*1e6, ' us' )

# Visualize the sample and resampled fields

p = pv.Plotter(shape = (1, 2))
p.subplot(0, 0)

points = pv.PolyData(np.array([X, Y, Z]).T)
points['vel'] = sample_UVW.T

arrows = points.glyph(orient='vel', scale=True, factor=0.5)

p.add_mesh(points, color='maroon', point_size=5., render_points_as_spheres=True)
p.add_mesh(arrows)

p.subplot(0, 1)

box['vel'] = UVW.T

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