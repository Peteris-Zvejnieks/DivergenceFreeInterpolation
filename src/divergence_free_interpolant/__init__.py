"""Divergence-free and curl-free interpolation of scattered vector data with matrix-valued RBFs."""

from ._interpolant import CurlFreeInterpolant, DivergenceFreeInterpolant
from ._legacy import interpolant

__all__ = ["CurlFreeInterpolant", "DivergenceFreeInterpolant", "interpolant"]
