"""Divergence-free and curl-free interpolation of scattered vector data with matrix-valued RBFs."""

from ._interpolant import CurlFreeInterpolant, DivergenceFreeInterpolant
from ._legacy import interpolant
from ._multilevel import (
    MultilevelCurlFreeInterpolant,
    MultilevelDivergenceFreeInterpolant,
)

__all__ = [
    "CurlFreeInterpolant",
    "DivergenceFreeInterpolant",
    "MultilevelCurlFreeInterpolant",
    "MultilevelDivergenceFreeInterpolant",
    "interpolant",
]
