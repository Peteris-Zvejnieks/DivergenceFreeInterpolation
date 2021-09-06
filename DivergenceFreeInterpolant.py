from sympy import var, lambdify, sqrt, Matrix, diff
from RadialBasisFunctions import RBF
from scipy.linalg import svd
import numpy as np

class interpolant():
    def __init__(self, nu: int, k: int):   
        
        rbf = RBF(nu, k).eq
        
        x0, x1 = var('x0'), var('x1')
        f = rbf.subs(rbf.free_symbols.pop(), sqrt(x0**2 + x1**2))
        d00, d01, d11 = diff(f, x0, x0), diff(f, x0, x1), diff(f, x1, x1)
        kernel = Matrix([[-d11,  d01],
                         [ d01, -d00]])
        self.comp_kernel = lambdify((x0, x1), kernel, ['numpy'])
    
    def condition(self, XY, UV):    
        self.XY = XY
        
        N = XY.shape[0]
        
        tmp = np.repeat(XY[:,:,np.newaxis], N, axis=2)
        coordinate_differences = np.swapaxes(tmp.T - tmp, 1, 2)
        
        tensor = self.comp_kernel(coordinate_differences[:,:,0], coordinate_differences[:,:,1])
        array = tensor.swapaxes(1, 2).reshape(2 * N , 2 * N, order = 'F')
        
        U, s, V = svd(array)
        self.sol = np.array(np.split(V.T @ np.diag(1/s) @ U.T @ UV.flatten(), N))[:,:,np.newaxis]
        
        interpolant.__call__ = np.vectorize(self.interpolate, signature = '(),()->(2)')
        
    def interpolate(self, x, y):
        kernel_applied = self.comp_kernel(x - self.XY[:,0], y - self.XY[:,1]).T
        return np.einsum('ijk,ijn', kernel_applied, self.sol).flatten()  
