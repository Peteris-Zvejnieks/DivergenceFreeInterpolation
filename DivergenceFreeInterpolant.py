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
        
        self.comp_kernel = lambdify((x0, x1, r), kernel, ['numpy'])

        self.sigma = (d + 2*k - 1)/2

    def create_interpolant(self, XY, sol, r):

        def interpolant(x, y):
            X, Y = (x - XY[:, 0])/r, (y - XY[:, 1])/r
            R = np.linalg.norm(np.array([X, Y]).T, axis=1)
            kernel_applied = r ** (-self.dim) * self.comp_kernel(X, Y, R).T
            kernel_applied[R > 1] = np.zeros((2, 2))
            return np.einsum('ijk,ijn', kernel_applied, sol).flatten()

        return np.vectorize(interpolant, signature = '(),()->(%i)' % self.dim)

    def condition(self, XY, UV, number_of_steps):
        XY = XY.astype(np.double)
        UV = UV.astype(np.double)

        self.XY = XY.copy()
        self.UV = UV.copy()
        self.interpolants = []

        subset_bool_arrays, self.covering_radii_history = smart_thinner(XY, int(XY.shape[0] * 0.1))
        steps = np.linspace(90, 0, number_of_steps).astype(int)

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
            coordinate_difference_norms = np.linalg.norm(coordinate_differences, axis=-1)


            tensor = support_radii**(-self.dim) * self.comp_kernel(coordinate_differences[:, :, 0],
                                                                   coordinate_differences[:, :, 1],
                                                                   coordinate_difference_norms)

            tensor = tensor.swapaxes(0, 2).swapaxes(1, 3)
            tensor[coordinate_difference_norms > 1] = np.zeros((2, 2))
            tensor = tensor.swapaxes(3, 1).swapaxes(2, 0)
            
            array = tensor.swapaxes(1, 2).reshape(self.dim * N, self.dim * N, order='F')
            error_on_subset = error[subset_bool_arrays[i]].flatten()

            # sol = np.array(
            #     np.split(np.linalg.solve(array, error_on_subset),
            #              N))[:, :, np.newaxis]

            # q, r = np.linalg.qr(array)
            # p = q.T @ error_on_subset
            # sol = np.linalg.inv(r) @ p
            # sol = np.array(
            #     np.split(sol,
            #              N))[:, :, np.newaxis]

            # sol, R, rank, s = np.linalg.lstsq(array, error_on_subset, rcond = None)
            # sol = np.array(
            #     np.split(sol,
            #              N))[:, :, np.newaxis]

            sol = np.array(
                np.split(
                    lu_solve(lu_factor(array), error_on_subset),
                         N))[:, :, np.newaxis]

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