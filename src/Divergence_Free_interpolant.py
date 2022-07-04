from scipy.linalg import svd
import numpy as np
from .Radial_Basis_Functions import RBF_kernel
 
class interpolant():
    def __init__(self, nu: int = 5, k: int = 3, dim: int = 3):   
        self.dim = dim
        self.kernel = RBF_kernel(nu, k, dim).kernel_numpy
        self.zero_kernel = np.zeros((dim, dim))

    def condition(self, XY, UV, support_radii = 1, method = 'linsolve'):
        self.XY = XY
        self.support_radii = support_radii
        N = XY.shape[0]
        self.N = N
        self.method = method
        
        tmp = np.repeat(XY[:, :, np.newaxis], N, axis=2)
        coordinate_differences = np.swapaxes(tmp.T - tmp, 1, 2)/support_radii
        coordinate_difference_norms = np.linalg.norm(coordinate_differences, axis = -1)
        
        if self.dim == 3:
            tensor = support_radii**(-self.dim)*self.kernel(coordinate_differences[:, :, 0],
                                                            coordinate_differences[:, :, 1],
                                                            coordinate_differences[:, :, 2],
                                                            coordinate_difference_norms)
        elif self.dim == 2:
            tensor = support_radii**(-self.dim)*self.kernel(coordinate_differences[:, :, 0],
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

        kernel_applied = self.support_radii**(-self.dim)*self.kernel(X, Y, Z, R).T
        kernel_applied[R > 1] = self.zero_kernel

        return np.einsum('ijk,ijn', kernel_applied, self.sol).flatten()
    
    def _interpolate2D(self, x, y):
        X, Y = (x - self.XY[:, 0])/self.support_radii, (y - self.XY[:, 1])/self.support_radii
        R = np.linalg.norm(np.array([X,Y]).T, axis=1)

        kernel_applied = self.support_radii**(-self.dim)*self.kernel(X, Y, R).T
        kernel_applied[R > 1] = self.zero_kernel

        return np.einsum('ijk,ijn', kernel_applied, self.sol).flatten()