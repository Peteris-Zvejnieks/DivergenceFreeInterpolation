import numpy as np

def greedy_thinner(points: np.ndarray, method, n: int = 2) -> list:
    boolean = np.ones(points.shape[0]) == 1
    subsets = [boolean.copy()]
    N = points.shape[0]
    for k in range(N - n):
        removable_point_index = method(points[subsets[-1]])
        boolean[removable_point_index] = False
        subsets.append(boolean.copy())
    return subsets
      
def random_method(points):
    N = points.shape[0]
    return np.random.randint(0, N)
