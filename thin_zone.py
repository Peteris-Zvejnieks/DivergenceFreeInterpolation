import numpy as np

def d(x, Y):
    return np.min(np.linalg.norm(Y - x, axis = 1))

#%%
from time import perf_counter as pc
import matplotlib.pyplot as plt


points = np.random.rand(750, 2)   
plt.scatter(points[:,0], points[:,1])
ax = plt.gca()
ax.set_aspect(1)
plt.show()
plt.close()

t1 = pc()
subsets = greedy_thinner(points, random_method)
print(pc() - t1)
for Y in subsets:
    plt.scatter(Y[:,0], Y[:,1])
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()
    plt.close()