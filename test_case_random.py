import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import tqdm
import time
import os
mpl.rcParams['figure.dpi'] = 500

from DivergenceFreeInterpolant import interpolant
#%%
from datetime import datetime
import shutil

now = datetime.now()
string = now.strftime("%d:%m:%Y  %H:%M:%S")
path = './.test_plots/'+string

#%%
v_f = lambda x, y: np.array([-2*x**3 * y, 3*x**2 * y**2])
N = 100
np.random.seed(69)
X, Y = np.random.rand(N), np.random.rand(N)

UV = v_f(X, Y).T
S = (UV[:,0]**2 + UV[:,1]**2)**0.5

plt.quiver(X, Y, UV[:,0], UV[:,1], S)

ax = plt.gca()
plt.colorbar()
ax.set_aspect('equal')

interp = interpolant(5, 3, 2.5)

t1 = time.perf_counter()
interp.condition(np.array([X, Y]).T, UV, 50)
print('Conditioning: ', time.perf_counter() - t1)
os.mkdir(path)
plt.savefig(path + '/0_sample_points.png')
plt.close()

xmin, xmax = 0, 1
ymin, ymax = 0, 1
ll1, ll2 = 50, 50
crdsX, crdsY = np.mgrid[xmin:xmax:ll1*1j, ymin:ymax:ll2*1j]

t1 = time.perf_counter()
try:
    uv = interp(crdsX, crdsY)
except:
    shutil.rmtree(path)
    raise BrokenPipeError("Interpolating")
print('Time per interpolation: ', (time.perf_counter() - t1)/(ll1*ll2))

SS = (uv[:,:,0]**2 + uv[:,:,1]**2)**0.5

fig = plt.figure()
ax = fig.add_subplot(111)

stream = ax.streamplot(crdsX.T, crdsY.T, uv[:,:,0].T, uv[:,:,1].T, color = SS, density = 1, cmap ='autumn')
fig.colorbar(stream.lines)
ax.set_aspect('equal')

plt.savefig(path + '/1_interpolation.png')
plt.close()
#%%
plt.plot(interp.support_radii_history, '-o')
plt.title('support radii')
plt.savefig(path + '/2_support_radii.png')
plt.close()

plt.plot(np.abs(interp.mistako_history))
plt.title('|mistakos|')
plt.yscale('log')
plt.ylabel('log|mistakos|')

plt.savefig(path + '/3_mistakos.png')
plt.close()


plt.plot(interp.condition_numbers)
plt.title('condition_number')
plt.yscale('log')
plt.ylabel('log(Cond(A))')

plt.savefig(path + '/4_condition_number.png')
plt.close()

norm = (interp.residuals[:,0]**2 + interp.residuals[:,1]**2)**0.5
arrows = plt.quiver(interp.XY[:,0], interp.XY[:,1], interp.residuals[:,0]/norm, interp.residuals[:,1]/norm, norm)
plt.title('Residuals')
plt.colorbar(arrows)
ax = plt.gca()
ax.set_aspect('equal')
plt.savefig(path + '/5_residuals.png')
plt.close()