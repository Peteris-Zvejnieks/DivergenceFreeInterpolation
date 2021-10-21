import numpy as np
from scipy.linalg import lu_factor, lu_solve

def residual_correction_scheme(A, b, eps = 1e-6):
    LU = lu_factor(A)
    x = lu_solve(LU, b)
    r = b - A @ x
    i = 0
    sum = np.abs(np.sum(r))
    while sum > eps and i < 1e5:
        sum = np.abs(np.sum(r))
        e = lu_solve(LU, r)
        x += e
        r = b - A @ x
        i += 1

    return x