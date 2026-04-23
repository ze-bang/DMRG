"""Low-level tensor primitives used by MPS / MPO / DMRG.

This subpackage hides numpy/scipy details from the rest of the library so that
backends (CPU/GPU/symmetric) can be swapped later without touching algorithms.
"""

from __future__ import annotations

from dmrg_pp.tensors.linalg import (
    TruncationInfo,
    left_qr,
    right_qr,
    truncated_svd,
)

__all__ = [
    "TruncationInfo",
    "left_qr",
    "right_qr",
    "truncated_svd",
]
