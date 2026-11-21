"""Divergence-free and curl-free interpolation of scattered vector data with matrix-valued RBFs."""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

from . import walls
from ._conditions import Curve, NormalComponent, Surface, VectorValue
from ._interpolant import CurlFreeInterpolant, DivergenceFreeInterpolant
from ._legacy import interpolant
from ._multilevel import (
    MultilevelCurlFreeInterpolant,
    MultilevelDivergenceFreeInterpolant,
)

try:
    __version__ = _version("Divergence_Free_Interpolant")
except _PackageNotFoundError:
    # A copy without installed metadata, e.g. vendored or frozen
    __version__ = "unknown"

__all__ = [
    "CurlFreeInterpolant",
    "Curve",
    "DivergenceFreeInterpolant",
    "MultilevelCurlFreeInterpolant",
    "MultilevelDivergenceFreeInterpolant",
    "NormalComponent",
    "Surface",
    "VectorValue",
    "interpolant",
    "walls",
]
