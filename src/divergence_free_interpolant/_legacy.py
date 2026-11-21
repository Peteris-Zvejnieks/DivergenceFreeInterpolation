"""Deprecated 0.1 API, kept as a thin wrapper around DivergenceFreeInterpolant."""

import warnings

import numpy as np

from ._interpolant import DivergenceFreeInterpolant
from ._kernel import DIVERGENCE_FREE, get_kernel, resolve_parameters

_METHODS = {"linsolve": "direct", "lstsq": "lstsq", "SVD": "lstsq", "penrose": "lstsq"}


class interpolant:
    """Deprecated: use DivergenceFreeInterpolant(support_radius, k=..., nu=...).fit(points, values).

    Behaves like the 0.1 class, including its attributes (XY, sol, N, support_radii, method,
    kernel, zero_kernel), except for two deliberate correctness changes: duplicate points raise
    a ValueError instead of producing a singular solve (method='lstsq' still tolerates them),
    and non-finite coordinates raise a ValueError instead of evaluating to NaN.
    """

    def __init__(self, nu: int = 5, k: int = 3, dim: int = 3):
        warnings.warn(
            "interpolant is deprecated and will be removed, use DivergenceFreeInterpolant",
            DeprecationWarning,
            stacklevel=2,
        )
        if dim not in (2, 3):
            raise ValueError(
                f"dimension {dim} is not supported by the legacy interpolant, only 2D and 3D"
            )
        resolve_parameters(k, nu, dim)
        self.dim, self.nu, self.k = dim, nu, k
        self.zero_kernel = np.zeros((dim, dim))
        self._interpolant = None
        self._method = None

    def condition(self, XY, UV, support_radii=1, method="linsolve"):
        if method not in _METHODS:
            raise KeyError(
                f"Method - {method!r} not found, supported methods: {', '.join(_METHODS)}."
            )
        XY = np.asarray(XY)
        if XY.ndim != 2 or XY.shape[1] != self.dim:
            raise ValueError(
                f"positions must have shape (N, {self.dim}), got {XY.shape}"
            )
        new = DivergenceFreeInterpolant(
            support_radii, k=self.k, nu=self.nu, solver=_METHODS[method]
        )
        self._interpolant = new.fit(XY, UV)
        self._method = method

    def __call__(self, x, y, z=None):
        """Evaluate at coordinate arrays x, y (and z in 3D), positionally or by keyword."""
        if self._interpolant is None:
            raise RuntimeError(
                "the interpolant has no data yet, call condition() first"
            )
        coordinates = (x, y) if z is None else (x, y, z)
        if len(coordinates) != self.dim:
            names = "x, y" if self.dim == 2 else "x, y, z"
            raise TypeError(
                f"expected {self.dim} coordinate arrays ({names}), got {len(coordinates)}"
            )
        points = np.stack(
            np.broadcast_arrays(*(np.asarray(c, dtype=float) for c in coordinates)),
            axis=-1,
        )
        return self._interpolant(points)

    # --- 0.1 attributes, read-only views of the current fit -------------------------------

    def _fit_for(self, name):
        if self._interpolant is None:
            raise AttributeError(
                f"'interpolant' object has no attribute {name!r} (call condition() first)"
            )
        return self._interpolant._fit

    @staticmethod
    def _read_only(array):
        view = array.view()
        view.flags.writeable = False
        return view

    @property
    def XY(self):
        """Conditioning positions, shape (N, dim)."""
        return self._read_only(self._fit_for("XY").nodes)

    @property
    def sol(self):
        """Kernel coefficients, shape (N, dim, 1)."""
        return self._read_only(self._fit_for("sol").coefficients[:, :, np.newaxis])

    @property
    def N(self):
        """Number of conditioning points."""
        return len(self._fit_for("N").nodes)

    @property
    def support_radii(self):
        return self._fit_for("support_radii").support_radius

    @property
    def method(self):
        self._fit_for("method")
        return self._method

    @property
    def kernel(self):
        """The unscaled matrix kernel as a function kernel(x0, x1[, x2], r) -> (dim, dim, ...).

        As in 0.1 the arguments are coordinate differences divided by the support radius and
        their norm r; r is accepted for compatibility but recomputed from the coordinates.
        """
        matrix_kernel = get_kernel(DIVERGENCE_FREE, self.dim, self.nu, self.k)

        def kernel(*arguments):
            *coordinates, _ = arguments
            arrays = np.broadcast_arrays(
                *(np.asarray(c, dtype=float) for c in coordinates)
            )
            blocks = matrix_kernel.blocks(np.stack(arrays, axis=-1), 1.0)
            return np.moveaxis(blocks, (-2, -1), (0, 1))

        return kernel
