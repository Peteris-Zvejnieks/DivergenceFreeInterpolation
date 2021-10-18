from sympy import var, lambdify, sqrt, Matrix, diff
from RadialBasisFunctions import RBF
from scipy.linalg import lu
import numpy as np
from Thinning import smart_thinner, random_method


def mesh_norm(Y):
    for i in range(Y.shape[0]):
        x = Y[i]
        mins = []
        Y_prim = np.delete(Y, i, axis = 0)
        mins.append(np.min(np.linalg.norm(x - Y_prim, axis = 1)))
    return max(mins)           
            

class interpolant():
    def __init__(self, d: int, k: int, nu: float) -> None:   
        
        self.dim = 2
        self.d = d
        self.k = k
        self.nu = nu
        
        rbf = RBF(d, k).eq
        
        x0, x1 = var('x0'), var('x1')
        r = rbf.free_symbols.pop()
        r_x = sqrt(x0**2 + x1**2)
        
        f = rbf.subs(r, r_x)
        
        d00, d01, d11 = diff(f, x0, x0), diff(f, x0, x1), diff(f, x1, x1)
        kernel = Matrix([[-d11,  d01],
                         [ d01, -d00]]).subs(r_x, r)
        
        self.comp_kernel = lambdify((x0, x1, r), kernel, ['numpy'])
        
        self.sigma = (d + 2*k - 1)/2
    
    def condition(self, XY, UV):
        self.XY = XY
        
        subsets = smart_thinner(XY, int(XY.shape[0]*0.1))
        
        def create_interpolant(XY, sol, support_radii):
            
            def interpolant(x, y):
                X, Y = (x - XY[:,0])/support_radii, (y - XY[:,1])/support_radii
                kernel_applied = support_radii**(-self.dim) * self.comp_kernel(X, Y, np.sqrt(X**2 + Y**2)).T
                return np.einsum('ijk,ijn', kernel_applied, sol).flatten()
            
            return np.vectorize(interpolant, signature = '(),()->(%i)'%self.dim)
        
        self.interpolants = []
        self.residuals = UV
        
        steps = list(range(len(subsets) - 1, 0, -5))
        if steps[-1] != 0: steps.append(0)
        for i in steps:
            Y = XY[subsets[i]]
            error = self.residuals[subsets[i]].flatten()
                
            support_radii = self.nu * mesh_norm(Y)**(self.sigma/(self.sigma + 1))
            Y /= support_radii
            
            N = Y.shape[0]
        
            tmp = np.repeat(Y[:,:,np.newaxis], N, axis=2)
            coordinate_differences = np.swapaxes(tmp.T - tmp, 1, 2)
            
            tensor = support_radii**(-self.dim) *self.comp_kernel(coordinate_differences[:,:,0],
                                                                  coordinate_differences[:,:,1],
                                                                  np.sqrt(coordinate_differences[:,:,0]**2 + coordinate_differences[:,:,1]**2))
            
            array = tensor.swapaxes(1, 2).reshape(self.dim * N , self.dim * N, order = 'F')
            
            sol = np.array(np.split(np.linalg.solve(array, error), N))[:,:,np.newaxis]
            # p, l, u = lu(array)
            
            # y = np.linalg.solve(l, np.dot(p, error))
            # sol = np.array(np.split(np.linalg.solve(u, y), N))[:,:,np.newaxis]
            
            self.interpolants.append(create_interpolant(Y, sol, support_radii))
            self.residuals -= self.interpolants[-1](XY[:,0], XY[:,1])
            
        interpolant.__call__ = np.vectorize(self.interpolate, signature = '(),()->(%i)'%self.dim)
        
    def interpolate(self, x, y):
        interpolation = 0
        for interpolant in self.interpolants:
            interpolation += interpolant(x, y)
        return interpolation 