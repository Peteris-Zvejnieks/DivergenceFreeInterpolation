from sympy import var, lambdify, sqrt, Matrix, diff, Piecewise
from RadialBasisFunctions import RBF
from scipy.linalg import svd
import numpy as np
 
class interpolant():
    def __init__(self, nu: int = 5, k: int = 3 , dim: int = 3):   
        
        if dim not in [2, 3]: raise ValueError(f'dimensionalit {dim} is not supported, on 2D or 3D are available')
        self.dim = dim
        
        rbf = RBF(nu, k).eq
        
        if dim == 3:
            x0, x1, x2 = var('x0'), var('x1'), var('x2')
            r = rbf.free_symbols.pop()
            r_x = sqrt(x0**2 + x1**2 + x2**2)
            
            f = rbf.subs(r, r_x)
            
            d00, d11, d22 = diff(f, x0, x0), diff(f, x1, x1), diff(f, x2, x2)
            d01, d02, d12 = diff(f, x0, x1), diff(f, x0, x2), diff(f, x1, x2)
            
            dd0, dd1, dd2 = -d11 - d22, -d00-d22, -d00 - d11
            
            kernel = Matrix([[dd0, d01, d02],
                             [d01, dd1, d12],
                             [d02, d12, dd2]]).subs(r_x, r)
            
            self.comp_kernel = lambdify((x0, x1, x2, r), kernel, ['numpy'])
            
        elif dim == 2:
            x0, x1 = var('x0'), var('x1')
            r = rbf.free_symbols.pop()
            r_x = sqrt(x0**2 + x1**2)
            
            f = rbf.subs(r, r_x)
            
            d00, d11 = diff(f, x0, x0), diff(f, x1, x1)
            d01 = diff(f, x0, x1)
            
            dd0, dd1 = - d11, -d00
            
            kernel = Matrix([[dd0, d01],
                             [d01, dd1]]).subs(r_x, r)
            
            self.comp_kernel = lambdify((x0, x1, r), kernel, ['numpy'])
            
        
        self.zero_kernel = np.zeros((self.dim, self.dim))

    def condition(self, XY, UV, support_radii = 50, method = 'linsolve'):
        self.XY = XY
        self.support_radii = support_radii
        N = XY.shape[0]
        self.N = N
        self.method = method
        
        tmp = np.repeat(XY[:, :, np.newaxis], N, axis=2)
        coordinate_differences = np.swapaxes(tmp.T - tmp, 1, 2)/support_radii
        coordinate_difference_norms = np.linalg.norm(coordinate_differences, axis = -1)
        
        if self.dim == 3:
            tensor = support_radii**(-self.dim)*self.comp_kernel(coordinate_differences[:, :, 0],
                                                                 coordinate_differences[:, :, 1],
                                                                 coordinate_differences[:, :, 2],
                                                                 coordinate_difference_norms)
        elif self.dim == 2:
            tensor = support_radii**(-self.dim)*self.comp_kernel(coordinate_differences[:, :, 0],
                                                                 coordinate_differences[:, :, 1],
                                                                 coordinate_difference_norms)
        tensor = tensor.swapaxes(0, 2).swapaxes(1, 3)
        tensor[coordinate_difference_norms > 1] = self.zero_kernel
        tensor = tensor.swapaxes(3, 1).swapaxes(2,0)

        array = tensor.swapaxes(1, 2).reshape(self.dim * N , self.dim * N, order='F')
        
        if method == 'SVD': 
            U, s, V = svd(array)
            sol = V.T @ np.diag(1/s) @ U.T @ UV.flatten()
        elif method == 'penrose':
            sol = np.linalg.pinv(array) @ UV.flatten()
        elif method == 'linsolve':
            sol = np.linalg.solve(array, UV.flatten())
        elif method == 'lstsq':
            sol = np.linalg.lstsq(array, UV.flatten())[0]
        else:
            raise KeyError('Method - ' + method + ' not found, supported methods: SVD, penrose, linsolve, lstsq.')
            
        self.sol = np.array(np.split(sol, N))[:, :, np.newaxis]
        self._set_call()
        
    def _set_call(self):
        signature = ('(),'*self.dim)[:-1] + '->(%i)'%self.dim
        if self.dim == 3:
            interpolant.__call__ = np.vectorize(self._interpolate3D, signature=signature)
        if self.dim == 2:
            interpolant.__call__ = np.vectorize(self._interpolate2D, signature=signature)
        
    def _interpolate3D(self, x, y, z):
        X, Y, Z = (x - self.XY[:, 0])/self.support_radii, (y - self.XY[:, 1])/self.support_radii, (z - self.XY[:, 2])/self.support_radii
        R = np.linalg.norm(np.array([X,Y,Z]).T, axis=1)

        kernel_applied = self.support_radii**(-self.dim)*self.comp_kernel(X, Y, Z, R).T
        kernel_applied[R > 1] = self.zero_kernel

        return np.einsum('ijk,ijn', kernel_applied, self.sol).flatten()
    
    def _interpolate2D(self, x, y):
        X, Y = (x - self.XY[:, 0])/self.support_radii, (y - self.XY[:, 1])/self.support_radii
        R = np.linalg.norm(np.array([X,Y]).T, axis=1)

        kernel_applied = self.support_radii**(-self.dim)*self.comp_kernel(X, Y, R).T
        kernel_applied[R > 1] = self.zero_kernel

        return np.einsum('ijk,ijn', kernel_applied, self.sol).flatten()