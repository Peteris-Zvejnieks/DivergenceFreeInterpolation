from scipy.linalg import svd as _svd
import numpy as _np
from ._kernel import _RBF_kernel

_METHODS = ('SVD', 'penrose', 'linsolve', 'lstsq')


class interpolant():
    def __init__(self, nu: int = 5, k: int = 3, dim: int = 3):
        self.dim = dim
        self.kernel = _RBF_kernel(nu, k, dim).kernel_numpy
        self.zero_kernel = _np.zeros((dim, dim))
        self._evaluate = None

    def __call__(self, *coordinates):
        if self._evaluate is None:
            raise RuntimeError('the interpolant has no data yet, call condition() first')
        if len(coordinates) != self.dim:
            names = 'x, y' if self.dim == 2 else 'x, y, z'
            raise TypeError(f'expected {self.dim} coordinate arrays ({names}), got {len(coordinates)}')
        return self._evaluate(*coordinates)

    def condition(self, XY, UV, support_radii = 1, method = 'linsolve'):
        if method not in _METHODS:
            raise KeyError(f'Method - {method!r} not found, supported methods: {", ".join(_METHODS)}.')
        # Copy the inputs, so later changes to the caller's arrays cannot desync nodes and coefficients
        XY = _np.array(XY, dtype=float)
        UV = _np.array(UV, dtype=float)
        if XY.ndim != 2 or XY.shape[1] != self.dim:
            raise ValueError(f'positions must have shape (N, {self.dim}), got {XY.shape}')
        if UV.shape != XY.shape:
            raise ValueError(f'vector field values must have the same shape as the positions, '
                             f'(N, {self.dim}) = {XY.shape}, got {UV.shape}')
        N = XY.shape[0]

        # Solve before touching any attributes: if the solver fails, the previous state stays intact
        array = self._gram_matrix(XY, support_radii)
        values = UV.flatten()

        if method == 'SVD':
            U, s, Vh = _svd(array)
            # Drop numerically zero singular values (same cutoff as numpy.linalg.pinv)
            keep = s > s[0] * max(array.shape) * _np.finfo(float).eps
            s_inv = _np.zeros_like(s)
            s_inv[keep] = 1 / s[keep]
            sol = Vh.T @ (s_inv * (U.T @ values))
        elif method == 'penrose':
            sol = _np.linalg.pinv(array) @ values
        elif method == 'linsolve':
            sol = _np.linalg.solve(array, values)
        elif method == 'lstsq':
            sol = _np.linalg.lstsq(array, values, rcond=None)[0]

        self.XY = XY
        self.support_radii = support_radii
        self.N = N
        self.method = method
        self.sol = _np.array(_np.split(sol, N))[:, :, _np.newaxis]
        self._set_call()

    def _gram_matrix(self, XY, support_radii = 1):
        """Interpolation matrix of shape (dim*N, dim*N) for nodes XY of shape (N, dim)."""
        N = XY.shape[0]
        tmp = _np.repeat(XY[:, :, _np.newaxis], N, axis=2)
        coordinate_differences = _np.swapaxes(tmp.T - tmp, 1, 2) / support_radii
        coordinate_difference_norms = _np.linalg.norm(coordinate_differences, axis = -1)

        tensor = support_radii**(-self.dim)*self.kernel(*_np.moveaxis(coordinate_differences, -1, 0),
                                                        coordinate_difference_norms)

        tensor = tensor.swapaxes(0, 2).swapaxes(1, 3)
        tensor[coordinate_difference_norms > 1] = self.zero_kernel
        tensor = tensor.swapaxes(3, 1).swapaxes(2,0)

        return tensor.swapaxes(1, 2).reshape(self.dim * N , self.dim * N, order='F')

    def _set_call(self):
        signature = ('(),'*self.dim)[:-1] + '->(%i)'%self.dim
        self._evaluate = _np.vectorize(self._interpolate, signature=signature)

    def _interpolate(self, *point):
        differences = (_np.array(point) - self.XY) / self.support_radii
        R = _np.linalg.norm(differences, axis=1)

        kernel_applied = self.support_radii**(-self.dim)*self.kernel(*differences.T, R).T
        kernel_applied[R > 1] = self.zero_kernel

        return _np.einsum('ijk,ijn', kernel_applied, self.sol).flatten()
