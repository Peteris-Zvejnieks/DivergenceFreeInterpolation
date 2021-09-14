from sympy import var, lambdify, sqrt, Matrix, diff
from RadialBasisFunctions import RBF
from scipy.linalg import svd
import numpy as np
 
class interpolant():
    def __init__(self, nu: int, k: int):   
        
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
        
        tmp = np.repeat(XY[:,:,np.newaxis], N, axis=2)
        coordinate_differences = np.swapaxes(tmp.T - tmp, 1, 2)
        
        tensor = self.comp_kernel(coordinate_differences[:,:,0], 
                                  coordinate_differences[:,:,1], 
                                  np.sqrt(coordinate_differences[:,:,0]**2 + coordinate_differences[:,:,1]**2))
        
        array = tensor.swapaxes(1, 2).reshape(2 * N , 2 * N, order = 'F')
        
        U, s, V = svd(array)
        self.sol = np.array(np.split(V.T @ np.diag(1/s) @ U.T @ UV.flatten(), N))[:,:,np.newaxis]
        
        interpolant.__call__ = np.vectorize(self.interpolate, signature = '(),()->(2)')
        
    def interpolate(self, x, y):
        X, Y = x - self.XY[:,0], y - self.XY[:,1]
        kernel_applied = self.comp_kernel(X, Y, np.sqrt(X**2 + Y**2)).T
        return np.einsum('ijk,ijn', kernel_applied, self.sol).flatten()  