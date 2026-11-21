"""Divergence-free and curl-free interpolation of scattered vector data with matrix-valued RBFs."""

from ._conditions import Curve, NormalComponent, Surface, VectorValue
from ._interpolant import CurlFreeInterpolant, DivergenceFreeInterpolant
from ._legacy import interpolant
from ._multilevel import (
    MultilevelCurlFreeInterpolant,
    MultilevelDivergenceFreeInterpolant,
)

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
]
