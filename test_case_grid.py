import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import time
import os
mpl.rcParams['figure.dpi'] = 500

from DivergenceFreeInterpolant import interpolant
#%%
v_f = lambda x, y: np.array([-2*x**3 * y, 3*x**2 * y**2])
xmin, xmax = 0, 1
ymin, ymax = 0, 1
l1, l2 = 25, 25

XX, YY = np.mgrid[xmin:xmax:l1*1j, ymin:ymax:l2*1j]

UV = v_f(XX.T, YY.T).T
S = (UV[:, :,0]**2 + UV[:,:,1]**2)**0.5

plt.quiver(XX, YY, UV[:, :,0], UV[:,:,1], S)

ax = plt.gca()
plt.colorbar()
ax.set_aspect('equal')
plt.show()
plt.close()
#%%

interp = interpolant(5, 3)
#%%
t1 = time.perf_counter()

interp.condition(np.array([XX.flatten(), YY.flatten()]).T, UV.reshape(l1*l2,2))

print('Conditioning: ', time.perf_counter() - t1)
#%%
ll1, ll2 = 100, 100
crdsX, crdsY = np.mgrid[xmin:xmax:ll1*1j, ymin:ymax:ll2*1j]

t1 = time.perf_counter()
uv = interp(crdsX, crdsY)
print('Time per interpolation: ', (time.perf_counter() - t1)/(ll1*ll2))

SS = (uv[:,:,0]**2 + uv[:,:,1]**2)**0.5
#%%

fig = plt.figure()
ax = fig.add_subplot(111)

stream = ax.streamplot(crdsX.T, crdsY.T, uv[:,:,0].T, uv[:,:,1].T, color = SS.T, density = 1, cmap ='autumn')
fig.colorbar(stream.lines)
ax.set_aspect('equal')
plt.show()
plt.close()
#%%

fig = plt.figure()
ax = fig.add_subplot(111)

arrows = ax.quiver(crdsX, crdsY, uv[:,:,0]/SS, uv[:,:,1]/SS, SS)
fig.colorbar(arrows)
ax.set_aspect('equal')
plt.show()
plt.close()