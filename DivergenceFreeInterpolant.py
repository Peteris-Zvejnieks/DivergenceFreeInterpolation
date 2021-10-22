from sympy import var, lambdify, sqrt, Matrix, diff, Piecewise
from RadialBasisFunctions import RBF
from scipy.linalg import svd
import numpy as np
 
class interpolant():
    def __init__(self, nu: int, k: int):   
        
        self.dim = 2
        
        rbf = RBF(nu, k).eq
        
        x0, x1 = var('x0'), var('x1')
        r = rbf.free_symbols.pop()
        r_x = sqrt(x0**2 + x1**2)
        
        f = rbf.subs(r, r_x)
        
        d00, d01, d11 = diff(f, x0, x0), diff(f, x0, x1), diff(f, x1, x1)
        kernel = Matrix([[-d11,  d01],
                         [ d01, -d00]]).subs(r_x, r)

        self.comp_kernel = lambdify((x0, x1, r), kernel, ['numpy'])

    def condition(self, XY, UV):
        self.XY = XY
        
        N = XY.shape[0]
        self.N = N
        
        tmp = np.repeat(XY[:,:,np.newaxis], N, axis=2)
        coordinate_differences = np.swapaxes(tmp.T - tmp, 1, 2)
        coordinate_difference_norms = np.linalg.norm(coordinate_differences, axis = -1)

        tensor = self.comp_kernel(coordinate_differences[:,:,0],
                                  coordinate_differences[:,:,1],
                                  coordinate_difference_norms)

        tensor = tensor.swapaxes(0, 2).swapaxes(1, 3)
        tensor[coordinate_difference_norms > 1] = np.zeros((2,2))
        tensor = tensor.swapaxes(3,1).swapaxes(2,0)

        array = tensor.swapaxes(1, 2).reshape(self.dim * N , self.dim * N, order = 'F')
        
        U, s, V = svd(array)
        self.sol = np.array(np.split(V.T @ np.diag(1/s) @ U.T @ UV.flatten(), N))[:,:,np.newaxis]
        
        interpolant.__call__ = np.vectorize(self.interpolate, signature = '(),()->(%i)'%self.dim)
        
    def interpolate(self, x, y):
        X, Y = x - self.XY[:,0], y - self.XY[:,1]
        R = np.linalg.norm(np.array([X,Y]).T, axis = 1)

        kernel_applied = self.comp_kernel(X, Y, R).T
        kernel_applied[R > 1] = np.zeros((2,2))

        return np.einsum('ijk,ijn', kernel_applied, self.sol).flatten()