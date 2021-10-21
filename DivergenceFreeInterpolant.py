from sympy import var, lambdify, sqrt, Matrix, diff
from RadialBasisFunctions import RBF
from scipy.linalg import lu_factor, lu_solve
import numpy as np
from Thinning import smart_thinner

def mesh_norm(Y):
    mins = []
    for i in range(Y.shape[0]):
        x = Y[i]
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
        
        comp_kernel_r = lambdify((x0, x1, r), kernel, ['numpy'])

        # def comp_kernel(x,y):
        #     if (r := np.sqrt(x**2 + y**2)) > 1: return np.zeros((2,2))
        #     else: return comp_kernel_r(x, y, r)

        def comp_kernel(x,y):
            r = np.sqrt(x ** 2 + y ** 2)
            return comp_kernel_r(x, y, r) * (r < 1)

        self.comp_kernel = np.vectorize(comp_kernel, signature = '(),()->(%i,%i)' % (self.dim, self.dim))

        self.sigma = (d + 2*k - 1)/2

    def create_interpolant(self, XY, sol, r):

        interpolant = lambda x, y: np.einsum('ijk,ijn',
                                             r ** (-self.dim) * self.comp_kernel((x - XY[:, 0])/r, (y - XY[:, 1])/r),
                                             sol).flatten()
        return np.vectorize(interpolant, signature = '(),()->(%i)' % self.dim)

    def condition(self, XY, UV):
        XY = XY.astype(np.double)
        UV = UV.astype(np.double)

        self.XY = XY.copy()
        self.UV = UV.copy()
        self.interpolants = []

        subset_bool_arrays, self.covering_radii_history = smart_thinner(XY, int(XY.shape[0] * 0.1))
        steps = list(range(len(subset_bool_arrays) - 1, 0, -30))
        if steps[-1] != 0:
            steps.append(0)

        self.support_radii_history = []
        self.mistako_history = []
        self.condition_numbers = []

        error = UV.copy()
        for i in steps:
            Y = XY[subset_bool_arrays[i]]
            support_radii = self.nu * mesh_norm(Y)**(self.sigma/(self.sigma + 1))
            Y /= support_radii
            N = Y.shape[0]
        
            coordinates_extruded = np.repeat(Y[:, :, np.newaxis], N, axis=2)
            coordinate_differences = np.swapaxes(coordinates_extruded.T - coordinates_extruded, 1, 2)
            tensor = support_radii**(-self.dim) * self.comp_kernel(coordinate_differences[:, :, 0],
                                                                   coordinate_differences[:, :, 1])
            
            array = tensor.swapaxes(1, 2).reshape(self.dim * N, self.dim * N, order='F')
            error_on_subset = error[subset_bool_arrays[i]].flatten()

            # q, r = np.linalg.qr(array)
            # p = q.T @ error_on_subset
            # sol = np.linalg.inv(r) @ p
            # sol = np.array(
            #     np.split(sol,
            #              N))[:, :, np.newaxis]

            sol, R, rank, s = np.linalg.lstsq(array, error_on_subset, rcond = None)
            sol = np.array(
                np.split(sol,
                         N))[:, :, np.newaxis]

            # sol = np.array(
            #     np.split(
            #         lu_solve(lu_factor(array), error_on_subset),
            #              N))[:, :, np.newaxis]

            # sol = np.array(
            #     np.split(
            #         rcs(array, error_on_subset),
            #              N))[:, :, np.newaxis]

            local_correction = self.create_interpolant(Y, sol, support_radii)
            self.interpolants.append(local_correction)
            error -= local_correction(XY[:, 0], XY[:, 1])

            self.support_radii_history.append(support_radii)
            self.mistako_history.append(np.sum(error_on_subset - array @ sol.flatten()))
            self.condition_numbers.append(np.linalg.cond(array))

        interpolant.__call__ = np.vectorize(self.interpolate, signature = '(),()->(%i)'%self.dim)
        self.residuals = UV - interpolant.__call__(XY[:, 0], XY[:, 1])
        
    def interpolate(self, x, y):
        interpolation = 0
        for interpolant in self.interpolants:
            interpolation += interpolant(x, y)
        return interpolation 