from sympy import var, lambdify, sqrt, Matrix, diff
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
        
        comp_kernel_r = lambdify((x0, x1, r), kernel, ['numpy'])

        def comp_kernel(x,y):
            # if (r := np.sqrt(x**2 + y**2)) > 1: return np.zeros((2, 2))
            # else: return comp_kernel_r(x, y, r)
            return comp_kernel_r(x, y, np.sqrt(x**2 + y**2))

        self.comp_kernel = np.vectorize(comp_kernel, signature='(),()->(2,2)')

    def condition(self, XY, UV):    
        self.XY = XY
        
        N = XY.shape[0]
        
        tmp = np.repeat(XY[:,:,np.newaxis], N, axis=2)
        coordinate_differences = np.swapaxes(tmp.T - tmp, 1, 2)
        
        tensor = self.comp_kernel(coordinate_differences[:,:,0], 
                                  coordinate_differences[:,:,1])
        
        array = tensor.swapaxes(1, 2).reshape(self.dim * N , self.dim * N, order = 'F')
        
        U, s, V = svd(array)
        self.sol = np.array(np.split(V.T @ np.diag(1/s) @ U.T @ UV.flatten(), N))[:,:,np.newaxis]
        
        interpolant.__call__ = np.vectorize(self.interpolate, signature = '(),()->(%i)'%self.dim)
        
    def interpolate(self, x, y):
        X, Y = x - self.XY[:,0], y - self.XY[:,1]
        kernel_applied = self.comp_kernel(X, Y)
        return np.einsum('ijk,ijn', kernel_applied, self.sol).flatten()