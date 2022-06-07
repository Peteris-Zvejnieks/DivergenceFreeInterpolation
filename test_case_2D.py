import matplotlib.pyplot as plt
import numpy as np
import time
plt.rcParams['figure.dpi'] = 500
from DivergenceFreeInterpolant import interpolant

np.random.seed(69)
div = lambda n, d: np.divide(n, d, out = np.zeros_like(d), where=d!=0)

## Analytic vectorfield definition; has to be divergence free
vector_field = lambda x, y: np.array([-2*x**3 * y, 3*x**2 * y**2])

## Number of sample points
N = 25

## Random sample points
X, Y = np.random.rand(N), np.random.rand(N)

## Get vectorfield sample values
UV = vector_field(X, Y)
U, V = UV[0], UV[1]

S = (U**2 + V**2)**0.5

## Visualize vectorfield
fig, ax = plt.subplots(1, 1)
quiver = ax.quiver(X, Y, div(U, S), div(V, S), S)
ax.set_aspect('equal')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
fig.colorbar(quiver)
plt.show()
plt.close()

## Initialize the interpolant, nu = 5, k = 3 will suffice almost always, dim is the dimensionality
## default 
initialized_interpolant = interpolant(nu = 5, k = 3, dim = 2)

## Condition the vectorfield 
## initialized_interpolant.condition(positions, vectors, support_radius, method)
## positions: np.ndarray, (dim, N)
## vectors: np.ndarray, (dim, N)
## support_radius: positive float
## method: string : default = linsolve: options = [SVD, penrose, linsolve, lstsq]
t1 = time.perf_counter()
initialized_interpolant.condition(np.array([X, Y]).T, UV.T, 1)
print('Conditioning time: ', time.perf_counter() - t1)

## Create resampling points
_n, _m = 100, 100
XX, YY = np.mgrid[0:1:_n*1j, 0:1:_m*1j]

## Call the interpolant passing resampling coordinates
## initialized_interpolant(X, Y)
## X: np.ndarray : any_shape
## Y: np.ndarray : shape like X
## returns np.ndarray: X.shape + (dim,)
t1 = time.perf_counter()
UV = initialized_interpolant(XX, YY)
print('Time per interpolation: ', (time.perf_counter() - t1)/(_n*_m))
UU = UV[:,:,0]
VV = UV[:,:,1]
SS = (UU**2 + VV**2)**0.5

## Visualize interpolated field

fig, ax = plt.subplots(1,1)
stream = ax.streamplot(XX.T, YY.T, div(UU, SS).T, div(VV, SS).T, color = SS.T, density = 1, cmap ='autumn')
fig.colorbar(stream.lines)
ax.set_aspect('equal')
plt.show()
plt.close()