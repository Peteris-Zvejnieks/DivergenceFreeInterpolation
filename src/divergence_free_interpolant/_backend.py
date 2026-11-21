"""Which implementation does the heavy lifting: the Rust core (default) or numpy.

The numpy implementations are the reference the Rust core is tested against, and stay
available for comparisons and debugging. This switch is private: set the environment variable
DFI_BACKEND=numpy before importing the package, or assign BACKEND (tests do).
"""

import os

BACKENDS = ("rust", "numpy")
BACKEND = os.environ.get("DFI_BACKEND", "rust")


def use_numpy() -> bool:
    if BACKEND not in BACKENDS:
        raise ValueError(f"unknown backend {BACKEND!r}, expected one of {BACKENDS}")
    return BACKEND == "numpy"
