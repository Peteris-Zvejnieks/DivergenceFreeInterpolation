import numpy as np
from time import perf_counter as pc
import matplotlib.pyplot as plt
from Thinning import smart_thinner

np.random.seed(69)
points = np.random.rand(100, 2)   
# plt.scatter(points[:,0], points[:,1])
# ax = plt.gca()
# ax.set_aspect(1)
# plt.show()
# plt.close()

t1 = pc()
subsets, radii = smart_thinner(points, 15)
print(pc() - t1)

plt.plot(radii)

plt.show()
plt.close()
# for boole in subsets:
#     Y = points[boole]
#     plt.scatter(Y[:,0], Y[:,1])
#     ax = plt.gca()
#     ax.set_aspect(1)
#     plt.show()
#     plt.close()