import numpy as np
from scipy.spatial import distance_matrix as dm

def random_method(points):
    N = points.shape[0]
    return np.random.randint(0, N)

def random_thinner(points: np.ndarray, n: int = 2) -> list:
    boolean = np.ones(points.shape[0]) == 1
    subsets = [boolean.copy()]
    N = points.shape[0]
    for k in range(N - n):
        removable_point_index = random_method(points[subsets[-1]])
        boolean[removable_point_index] = False
        subsets.append(boolean.copy())
    return subsets
      



def d_Y(x, Y):
    return np.min(np.linalg.norm(Y - x, axis = 1))

def r_YX(Y, X):
    return np.max([d_Y(x, Y) for x in X])

def rd_YX(Y, X):
    distance_matrix = dm(Y, X)
    mins = np.min(distance_matrix, axis = 1)
    return np.max(mins)

def smart_thinner(X, min_points = 2):
    boolean = np.ones(X.shape[0]) == 1
    subsets = [boolean.copy()]
    N = X.shape[0]
    
    smallest_covering_radii = 1e6
    
    for i in range(N):
        if (covering_radii := r_YX(np.delete(X, i, axis = 0), X)) < smallest_covering_radii:
            smallest_covering_radii = covering_radii
            indx = i
    
    radii_history = [smallest_covering_radii]
    boolean[indx] = False
    subsets.append(boolean.copy())
    
    
    
    for k in range(N - min_points):
        Y = X[subsets[-1]]
        n = Y.shape[0]
        indexes = np.cumsum(subsets[-1])
        smallest_covering_radii = 1e6
        for i in range(n):
            if (covering_radii := r_YX(np.delete(Y, i, axis = 0), X)) < smallest_covering_radii:
                smallest_covering_radii = covering_radii
                indx = np.argwhere(indexes == i + 1)[0]
        radii_history.append(smallest_covering_radii)
        boolean[indx] = False
        subsets.append(boolean.copy())
    return subsets, radii_history

